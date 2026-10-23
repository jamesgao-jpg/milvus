package viewquery

import (
	"context"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/anypb"

	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
)

func TestReusableQueryInterruptsExecutionAndReusesRPC(t *testing.T) {
	started := make(chan struct{})
	provider := &reusableQueryProvider{}
	scheduler := &reusableQueryScheduler{query: func(ctx context.Context, tasks QuerySegmentTasks) (*internalpb.RetrieveResults, error) {
		id := tasks.Tasks()[0].(int64)
		if id == 1 {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return reusableQueryResult(id), nil
	}}
	client, opens := startReusableQueryServer(t, NewServer(provider, scheduler))
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	first, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(1))
	require.NoError(t, err)
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal("Query did not start")
	}
	done := make(chan error, 1)
	go func() { _, err := first.Recv(); done <- err }()
	require.NoError(t, first.Close())
	require.ErrorIs(t, <-done, io.ErrClosedPipe)
	require.Equal(t, int64(1), provider.released.Load())
	stats := pool.Stats()
	require.Equal(t, uint64(1), stats.Interrupted)
	require.Equal(t, 1, stats.Idle)
	require.Empty(t, stats.Evicted)
	second, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(2))
	require.NoError(t, err)
	chunk, err := second.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{2}, chunk.GetIds().GetIntId().GetData())
	require.Equal(t, int64(2), chunk.GetReqID())
	require.NoError(t, second.Close())
	require.Equal(t, int64(1), opens.Load())
	require.Equal(t, uint64(1), pool.Stats().Reused)
	require.Equal(t, int64(2), provider.released.Load())
}

func TestReusableQueryInterruptDiscardsOnlyPreviousRequestChunks(t *testing.T) {
	for _, completedFirst := range []bool{false, true} {
		t.Run(map[bool]string{false: "interrupted", true: "normal_completion_wins"}[completedFirst], func(t *testing.T) {
			service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
				first, err := stream.Recv()
				if err != nil {
					return err
				}
				if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: first.GetRequestId(), Chunk: reusableQueryResult(1)}); err != nil {
					return err
				}
				terminal, _ := anypb.New(status.New(codes.OK, "").Proto())
				old := &viewpb.QueryOnViewStreamReusableResponse{RequestId: first.GetRequestId(), Chunk: reusableQueryResult(999)}
				if completedFirst {
					old.CompletionStatus = terminal
				}
				if err := stream.Send(old); err != nil {
					return err
				}
				interrupt, err := stream.Recv()
				if err != nil {
					return err
				}
				if !interrupt.GetInterrupt() || interrupt.GetRequestId() != first.GetRequestId() {
					return status.Error(codes.Internal, "missing matching interruption")
				}
				if !completedFirst {
					if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: first.GetRequestId(), CompletionStatus: terminal, Interrupted: true}); err != nil {
						return err
					}
				}
				second, err := stream.Recv()
				if err != nil {
					return err
				}
				return sendReusableQuerySuccess(stream, second.GetRequestId(), reusableQueryResult(2))
			}}
			client, opens := startReusableQueryServer(t, service)
			pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
			defer pool.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			first, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(1))
			require.NoError(t, err)
			_, err = first.Recv()
			require.NoError(t, err)
			require.NoError(t, first.Close())
			require.Equal(t, uint64(1), pool.Stats().DiscardedChunks)
			require.Equal(t, 1, pool.Stats().Idle)
			second, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(2))
			require.NoError(t, err)
			chunk, err := second.Recv()
			require.NoError(t, err)
			require.Equal(t, []int64{2}, chunk.GetIds().GetIntId().GetData())
			require.Equal(t, int64(2), chunk.GetReqID())
			require.NoError(t, second.Close())
			require.Equal(t, int64(1), opens.Load())
			require.Empty(t, pool.Stats().Evicted)
		})
	}
}

func TestReusableQueryLateInterruptDoesNotLeakCompletion(t *testing.T) {
	client, _ := startReusableQueryServer(t, NewServer(&reusableQueryProvider{}, &reusableQueryScheduler{}))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := client.QueryOnViewStreamReusable(ctx)
	require.NoError(t, err)
	require.NoError(t, stream.Send(&viewpb.QueryOnViewStreamReusableRequest{RequestId: 1, Request: reusableQueryRequest(1)}))
	first, err := stream.Recv()
	require.NoError(t, err)
	require.NotNil(t, first.GetCompletionStatus())
	require.NoError(t, stream.Send(&viewpb.QueryOnViewStreamReusableRequest{RequestId: 1, Interrupt: true}))
	require.NoError(t, stream.Send(&viewpb.QueryOnViewStreamReusableRequest{RequestId: 2, Request: reusableQueryRequest(2)}))
	second, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, uint64(2), second.GetRequestId())
	require.Equal(t, int64(2), second.GetChunk().GetReqID())
	require.NoError(t, stream.CloseSend())
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
}

func TestReusableQueryConcurrentEarlyCloseReusesPool(t *testing.T) {
	provider := &reusableQueryProvider{}
	client, opens := startReusableQueryServer(t, NewServer(provider, &reusableQueryScheduler{rows: 5}))
	const size, waves = 4, 5
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), size)
	defer pool.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for wave := 0; wave < waves; wave++ {
		streams := make([]queryutil.ReduceStream, size)
		for i := range streams {
			request := reusableQueryRequest(int64(wave*size + i + 1))
			request.StreamChunkBytes = 1
			var err error
			streams[i], err = pool.NewStream(ctx, client, 11, request)
			require.NoError(t, err)
			require.NotNil(t, streams[i])
		}
		var pending sync.WaitGroup
		for i, stream := range streams {
			pending.Add(1)
			go func() {
				defer pending.Done()
				chunk, err := stream.Recv()
				if assert.NoError(t, err) && assert.NotNil(t, chunk) {
					assert.Equal(t, int64(wave*size+i+1), chunk.GetReqID())
				}
				assert.NoError(t, stream.Close())
			}()
		}
		pending.Wait()
		stats := pool.Stats()
		require.Equal(t, uint64(size), stats.Opened)
		require.Equal(t, uint64(wave*size), stats.Reused)
		require.Equal(t, uint64((wave+1)*size), stats.Completed+stats.Interrupted)
		require.Equal(t, size, stats.Idle)
		require.Zero(t, stats.Leased)
		require.Zero(t, stats.BusyFallbacks)
		require.Empty(t, stats.Evicted)
	}
	require.Equal(t, int64(size), opens.Load())
	require.Equal(t, int64(size*waves), provider.released.Load())
}

func TestReusableQueryInterruptWhileSendIsBlocked(t *testing.T) {
	provider := &reusableQueryProvider{observed: make(chan reusableQueryObservation, 1)}
	server := NewServer(provider, &reusableQueryScheduler{rows: 5})
	blocked, release := make(chan struct{}), make(chan struct{})
	service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
		return server.QueryOnViewStreamReusable(&reusableQueryBlockedSendServer{
			ViewQueryService_QueryOnViewStreamReusableServer: stream, blocked: blocked, release: release,
		})
	}}
	client, _ := startReusableQueryServer(t, service)
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	request := reusableQueryRequest(1)
	request.StreamChunkBytes = 1
	stream, err := pool.NewStream(ctx, client, 11, request)
	require.NoError(t, err)
	_, err = stream.Recv()
	require.NoError(t, err)
	select {
	case <-blocked:
	case <-ctx.Done():
		t.Fatal("Send did not block")
	}
	observation := <-provider.observed
	go func() { <-observation.ctx.Done(); close(release) }()
	require.NoError(t, stream.Close())
	require.Equal(t, uint64(1), pool.Stats().Interrupted)
	require.Equal(t, uint64(1), pool.Stats().DiscardedChunks)
	require.Equal(t, 1, pool.Stats().Idle)
	require.Empty(t, pool.Stats().Evicted)
	require.Equal(t, int64(1), provider.released.Load())
}

// Block only the second data Send; interruption must still reach the request context.
type reusableQueryBlockedSendServer struct {
	viewpb.ViewQueryService_QueryOnViewStreamReusableServer
	blocked chan struct{}
	release chan struct{}
	sent    int
}

func (s *reusableQueryBlockedSendServer) Send(response *viewpb.QueryOnViewStreamReusableResponse) error {
	if response.GetChunk() != nil {
		s.sent++
		if s.sent == 2 {
			close(s.blocked)
			select {
			case <-s.release:
			case <-s.Context().Done():
				return s.Context().Err()
			}
		}
	}
	return s.ViewQueryService_QueryOnViewStreamReusableServer.Send(response)
}

func TestReusableQueryRejectsInvalidInterruptCompletion(t *testing.T) {
	ok, _ := anypb.New(status.New(codes.OK, "").Proto())
	failed, _ := anypb.New(status.New(codes.Internal, "bad").Proto())
	for name, response := range map[string]*viewpb.QueryOnViewStreamReusableResponse{
		"wrong_request": {RequestId: 99, CompletionStatus: ok, Interrupted: true},
		"rows":          {RequestId: 1, Chunk: reusableQueryResult(999), CompletionStatus: ok, Interrupted: true},
		"failed":        {RequestId: 1, CompletionStatus: failed, Interrupted: true},
		"no_status":     {RequestId: 1, Interrupted: true},
	} {
		t.Run(name, func(t *testing.T) {
			service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
				request, err := stream.Recv()
				if err != nil {
					return err
				}
				if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: request.GetRequestId(), Chunk: reusableQueryResult(1)}); err != nil {
					return err
				}
				if _, err := stream.Recv(); err != nil {
					return err
				}
				return stream.Send(response)
			}}
			client, _ := startReusableQueryServer(t, service)
			pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
			defer pool.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(1))
			require.NoError(t, err)
			_, err = stream.Recv()
			require.NoError(t, err)
			require.NoError(t, stream.Close())
			require.Equal(t, uint64(1), pool.Stats().Evicted["protocol_error"])
			require.Zero(t, pool.Stats().Idle)
			require.Zero(t, pool.Stats().Leased)
		})
	}
}

func TestReusableQueryRemainsLeasedUntilInterruptCompletion(t *testing.T) {
	interrupted, complete := make(chan struct{}), make(chan struct{})
	service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
		first, err := stream.Recv()
		if err != nil {
			return err
		}
		if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: first.GetRequestId(), Chunk: reusableQueryResult(1)}); err != nil {
			return err
		}
		if _, err := stream.Recv(); err != nil {
			return err
		}
		close(interrupted)
		select {
		case <-complete:
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
		terminal, _ := anypb.New(status.New(codes.OK, "").Proto())
		if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: first.GetRequestId(), CompletionStatus: terminal, Interrupted: true}); err != nil {
			return err
		}
		second, err := stream.Recv()
		if err != nil {
			return err
		}
		return sendReusableQuerySuccess(stream, second.GetRequestId(), reusableQueryResult(2))
	}}
	client, opens := startReusableQueryServer(t, service)
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	first, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(1))
	require.NoError(t, err)
	_, err = first.Recv()
	require.NoError(t, err)
	closed := make(chan error, 1)
	go func() { closed <- first.Close() }()
	select {
	case <-interrupted:
	case <-ctx.Done():
		t.Fatal("interruption did not arrive")
	}
	require.Equal(t, 1, pool.Stats().Leased)
	require.Zero(t, pool.Stats().Idle)
	overflow, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(3))
	require.NoError(t, err)
	require.Nil(t, overflow)
	close(complete)
	require.NoError(t, <-closed)
	second, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(2))
	require.NoError(t, err)
	chunk, err := second.Recv()
	require.NoError(t, err)
	require.Equal(t, int64(2), chunk.GetReqID())
	require.NoError(t, second.Close())
	require.Equal(t, int64(1), opens.Load())
}
