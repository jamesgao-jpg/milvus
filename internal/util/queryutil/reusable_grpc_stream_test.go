// Focused lifecycle tests exercise cancellation during Send, expiry, and metadata fallback.
package queryutil

import (
	"context"
	"io"
	"strconv"
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
	pool := NewReusableQueryStreamPool(context.Background(), 1)
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
			pool := NewReusableQueryStreamPool(context.Background(), 1)
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
	pool := NewReusableQueryStreamPool(context.Background(), 1)
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
	pool := NewReusableQueryStreamPool(context.Background(), 1)
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

func TestReusableQueryPoolCapacityAndReuse(t *testing.T) {
	for _, size := range []int{0, 1, 4} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			capacity := max(1, size)
			pool := NewReusableQueryStreamPool(context.Background(), size)
			defer pool.Close()
			started, release := make(chan struct{}, capacity), make(chan struct{}, capacity)
			client := &reusableRPCStubClient{open: func(ctx context.Context) *reusableRPCStubStream {
				started <- struct{}{}
				select {
				case <-release:
				case <-ctx.Done():
				}
				return newReusableRPCStubStream(ctx)
			}}
			type result struct {
				stream ReduceStream
				err    error
			}
			ready := make(chan result, capacity)
			for i := 0; i < capacity; i++ {
				go func() {
					stream, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
					ready <- result{stream, err}
				}()
			}
			for i := 0; i < capacity; i++ {
				select {
				case <-started:
				case <-time.After(time.Second):
					t.Fatal("reserved RPC did not start opening")
				}
			}
			require.Equal(t, capacity, pool.Stats().Leased)
			require.Zero(t, pool.Stats().Opened)
			overflow, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
			require.NoError(t, err)
			require.Nil(t, overflow, "opening RPCs must count toward the cap")
			for i := 0; i < capacity; i++ {
				release <- struct{}{}
			}
			for i := 0; i < capacity; i++ {
				result := <-ready
				require.NoError(t, result.err)
				require.NotNil(t, result.stream)
				_, err := result.stream.Recv()
				require.ErrorIs(t, err, io.EOF)
				require.NoError(t, result.stream.Close())
			}
			require.Equal(t, capacity, pool.Stats().Idle)

			// Hold all leases simultaneously so the second wave must reuse every RPC.
			streams := make([]ReduceStream, capacity)
			for i := range streams {
				streams[i], err = pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
				require.NoError(t, err)
				require.NotNil(t, streams[i])
			}
			for _, stream := range streams {
				_, err := stream.Recv()
				require.ErrorIs(t, err, io.EOF)
				require.NoError(t, stream.Close())
			}
			stats := pool.Stats()
			require.Equal(t, uint64(capacity), stats.Opened)
			require.Equal(t, uint64(capacity), stats.Reused)
			require.Equal(t, uint64(1), stats.BusyFallbacks)
			require.Equal(t, uint64(2*capacity), stats.Completed)
			require.Zero(t, stats.Leased)
		})
	}
}

func TestReusableQueryPoolEvictsOnlyFailedRPC(t *testing.T) {
	for _, reason := range []string{"early_close", "canceled", "transport_error", "logical_error", "protocol_error"} {
		t.Run(reason, func(t *testing.T) {
			pool := NewReusableQueryStreamPool(context.Background(), 2)
			defer pool.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			client := &reusableRPCStubClient{}
			first, err := pool.NewStream(ctx, client, 11, reusableRPCStubRequest())
			require.NoError(t, err)
			second, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
			require.NoError(t, err)
			failed := first.(*reusableQueryStream)
			switch reason {
			case "canceled":
				cancel()
			case "transport_error":
				failed.rpc.cancel()
			case "logical_error":
				terminal, err := anypb.New(status.New(codes.Unavailable, "failed request").Proto())
				require.NoError(t, err)
				failed.rpc.client.(*reusableRPCStubStream).response.CompletionStatus = terminal
			case "protocol_error":
				failed.rpc.client.(*reusableRPCStubStream).response.RequestId++
			}
			if reason != "early_close" {
				_, err = first.Recv()
				require.Error(t, err)
			}
			require.NoError(t, first.Close())
			require.Equal(t, uint64(1), pool.Stats().Evicted[reason])
			require.Equal(t, 1, pool.Stats().Leased)
			_, err = second.Recv()
			require.ErrorIs(t, err, io.EOF)
			require.NoError(t, second.Close())
			third, err := pool.NewStream(context.Background(), client, 11, reusableRPCStubRequest())
			require.NoError(t, err)
			require.Same(t, second.(*reusableQueryStream).rpc, third.(*reusableQueryStream).rpc)
			_, err = third.Recv()
			require.ErrorIs(t, err, io.EOF)
			require.NoError(t, third.Close())
			require.NoError(t, first.Close(), "late Close must not evict a sibling's new lease")
			require.Equal(t, uint64(2), pool.Stats().Opened)
		})
	}
}

func TestReusableQueryPoolRemovesAllNodeRPCs(t *testing.T) {
	pool := NewReusableQueryStreamPool(context.Background(), 3)
	defer pool.Close()
	client := &reusableRPCStubClient{}
	streams := make([]ReduceStream, 0, 6)
	for _, node := range []int64{11, 22} {
		for i := 0; i < 3; i++ {
			stream, err := pool.NewStream(context.Background(), client, node, reusableRPCStubRequest())
			require.NoError(t, err)
			require.NotNil(t, stream)
			streams = append(streams, stream)
		}
	}
	for _, stream := range []ReduceStream{streams[0], streams[3]} {
		_, err := stream.Recv()
		require.ErrorIs(t, err, io.EOF)
		require.NoError(t, stream.Close())
	}
	pool.RetainNodes(map[int64]struct{}{22: {}})
	stats := pool.Stats()
	require.Equal(t, uint64(3), stats.Evicted["node_removed"])
	require.Equal(t, 1, stats.Idle)
	require.Equal(t, 2, stats.Leased)
	pool.Close()
	for _, stream := range streams {
		require.ErrorIs(t, stream.(*reusableQueryStream).rpc.ctx.Err(), context.Canceled)
		require.NoError(t, stream.Close())
	}
	stats = pool.Stats()
	require.Equal(t, uint64(3), stats.Evicted["shutdown"])
	require.Zero(t, stats.Idle)
	require.Zero(t, stats.Leased)
}

func TestReusableQueryPoolExpiryPreservesLeasedRPC(t *testing.T) {
	pool := NewReusableQueryStreamPool(context.Background(), 3)
	pool.idleTimeout = 20 * time.Millisecond
	defer pool.Close()
	streams := make([]ReduceStream, 3)
	for i := range streams {
		var err error
		streams[i], err = pool.NewStream(context.Background(), &reusableRPCStubClient{}, 11, reusableRPCStubRequest())
		require.NoError(t, err)
	}
	for _, stream := range streams[:2] {
		_, err := stream.Recv()
		require.ErrorIs(t, err, io.EOF)
		require.NoError(t, stream.Close())
	}
	require.Eventually(t, func() bool { return pool.Stats().Evicted["idle_expired"] == 2 }, time.Second, time.Millisecond)
	require.Zero(t, pool.Stats().Idle)
	require.Equal(t, 1, pool.Stats().Leased)
	require.NoError(t, streams[2].(*reusableQueryStream).rpc.ctx.Err())
	_, err := streams[2].Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, streams[2].Close())
	require.NoError(t, streams[0].Close())
	require.NoError(t, streams[1].Close())
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
