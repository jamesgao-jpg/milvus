// Focused lifecycle tests exercise cancellation during Send, expiry, and metadata fallback.
package queryutil

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/anypb"

	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
)

// TestReusableQueryExpiryAndShutdown checks RQ-U12 without waiting a full production idle minute.
func TestReusableQueryExpiryAndShutdown(t *testing.T) {
	pool := NewReusableQueryStreamPool(context.Background())
	pool.idleTimeout = 10 * time.Millisecond
	defer pool.Close()
	client := &reusableRPCStubClient{}
	first, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
	require.NoError(t, err)
	_, err = first.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, first.Close())
	require.Eventually(t, func() bool { return pool.Stats().Evicted["idle_expired"] == 1 }, time.Second, time.Millisecond)
	require.Zero(t, pool.Stats().Idle)

	second, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
	require.NoError(t, err)
	pool.Close()
	_, err = second.Recv()
	require.Error(t, err)
	require.NoError(t, second.Close())
	require.Zero(t, pool.Stats().Leased)
	_, err = pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
	require.ErrorIs(t, err, context.Canceled)
}

// TestReusableQueryCancelDuringSend verifies request and component cancellation unblock opening.
func TestReusableQueryCancelDuringSend(t *testing.T) {
	for _, componentShutdown := range []bool{false, true} {
		t.Run(map[bool]string{false: "request", true: "component"}[componentShutdown], func(t *testing.T) {
			pool := NewReusableQueryStreamPool(context.Background())
			defer pool.Close()
			started := make(chan struct{})
			client := &reusableRPCStubClient{open: func(ctx context.Context) *reusableRPCStubStream {
				return &reusableRPCStubStream{ctx: ctx, send: func(*viewpb.QueryOnViewStreamReusableRequest) error {
					close(started)
					<-ctx.Done()
					return ctx.Err()
				}}
			}}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() { _, err := pool.NewStream(ctx, client, 11, reusableRPCStubRequest()); done <- err }()
			select {
			case <-started:
			case <-time.After(time.Second):
				t.Fatal("Send did not begin")
			}
			if componentShutdown {
				pool.Close()
			} else {
				cancel()
			}
			select {
			case err := <-done:
				require.ErrorIs(t, err, context.Canceled)
			case <-time.After(time.Second):
				t.Fatal("Send did not stop")
			}
			require.Zero(t, pool.Stats().Leased)
		})
	}
}

// TestReusableQueryConcurrentOpenFallsBack reserves the QN entry before a slow first Send completes.
func TestReusableQueryConcurrentOpenFallsBack(t *testing.T) {
	pool := NewReusableQueryStreamPool(context.Background())
	defer pool.Close()
	started, release := make(chan struct{}), make(chan struct{})
	client := &reusableRPCStubClient{open: func(ctx context.Context) *reusableRPCStubStream {
		rpc := newReusableRPCStubStream(ctx)
		rpc.send = func(request *viewpb.QueryOnViewStreamReusableRequest) error {
			close(started)
			select {
			case <-release:
			case <-ctx.Done():
				return ctx.Err()
			}
			rpc.send = nil
			return rpc.Send(request)
		}
		return rpc
	}}
	ready := make(chan ReduceStream, 1)
	failure := make(chan error, 1)
	go func() {
		stream, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
		ready <- stream
		failure <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first Send did not begin")
	}
	stream, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
	require.NoError(t, err)
	require.Nil(t, stream)
	close(release)
	stream = <-ready
	require.NoError(t, <-failure)
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, stream.Close())
	require.Equal(t, uint64(1), pool.Stats().Opened)
	require.Equal(t, uint64(1), pool.Stats().BusyFallbacks)
}

// TestReusableQueryScopeAndFallback never drops unknown metadata or routes outside the selected QN.
func TestReusableQueryScopeAndFallback(t *testing.T) {
	pool := NewReusableQueryStreamPool(context.Background())
	defer pool.Close()
	var rpcContext context.Context
	client := &reusableRPCStubClient{open: func(ctx context.Context) *reusableRPCStubStream {
		rpcContext = ctx
		return newReusableRPCStubStream(ctx)
	}}
	request := reusableRPCStubRequest()
	ctx := metadata.NewOutgoingContext(context.Background(), metadata.Pairs("authorization", "secret"))
	stream, err := pool.NewStream(ctx, client, 11, request)
	require.NoError(t, err)
	require.Nil(t, stream)
	request.LegacyReq.IsCount = true
	stream, err = pool.NewStream(context.Background(), client, 11, request)
	require.NoError(t, err)
	require.Nil(t, stream)
	request.LegacyReq.IsCount = false
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	stream, err = pool.NewStream(ctx, client, 11, request)
	require.NoError(t, err)
	nodeID, ok := contextutil.GetPickServerID(rpcContext)
	require.True(t, ok)
	require.Equal(t, int64(11), nodeID)
	_, deadline := rpcContext.Deadline()
	require.False(t, deadline, "persistent RPC must not inherit the first request's deadline")
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, stream.Close())
	cancel()
	require.NoError(t, rpcContext.Err(), "completed request's cancellation must not kill idle transport")
}

// reusableRPCStubClient exposes the new generated method without mocking ordinary Query RPCs.
type reusableRPCStubClient struct {
	viewpb.ViewQueryServiceClient
	open func(context.Context) *reusableRPCStubStream
}

// QueryOnViewStreamReusable creates one controllable transport for unit tests.
func (c *reusableRPCStubClient) QueryOnViewStreamReusable(ctx context.Context, _ ...grpc.CallOption) (viewpb.ViewQueryService_QueryOnViewStreamReusableClient, error) {
	if c.open != nil {
		return c.open(ctx), nil
	}
	return newReusableRPCStubStream(ctx), nil
}

// reusableRPCStubStream models completion-only results and a cancelable Send.
type reusableRPCStubStream struct {
	grpc.ClientStream
	ctx      context.Context
	send     func(*viewpb.QueryOnViewStreamReusableRequest) error
	response *viewpb.QueryOnViewStreamReusableResponse
}

// Context exposes the native transport lifetime, independently from each request.
func (s *reusableRPCStubStream) Context() context.Context { return s.ctx }

// newReusableRPCStubStream uses the actual generated response and canonical status representation.
func newReusableRPCStubStream(ctx context.Context) *reusableRPCStubStream {
	return &reusableRPCStubStream{ctx: ctx}
}

// Send prepares one successful logical completion unless the test overrides it.
func (s *reusableRPCStubStream) Send(request *viewpb.QueryOnViewStreamReusableRequest) error {
	if s.send != nil {
		return s.send(request)
	}
	terminal, err := anypb.New(status.New(codes.OK, "").Proto())
	if err != nil {
		return err
	}
	s.response = &viewpb.QueryOnViewStreamReusableResponse{RequestId: request.GetRequestId(), CompletionStatus: terminal}
	return nil
}

// Recv checks transport cancellation before exposing its prepared response.
func (s *reusableRPCStubStream) Recv() (*viewpb.QueryOnViewStreamReusableResponse, error) {
	if err := s.ctx.Err(); err != nil {
		return nil, err
	}
	response := s.response
	s.response = nil
	return response, nil
}

// reusableRPCStubRequest selects bounded Plain Query without needing a real QueryPlan.
func reusableRPCStubRequest() *viewpb.QueryOnViewRequest {
	return &viewpb.QueryOnViewRequest{LegacyReq: &internalpb.RetrieveRequest{Limit: 2}, StreamChunkBytes: 262144}
}
