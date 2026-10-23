// Real-TCP tests qualify logical completion, isolation, errors, and cleanup on reusable Query RPCs.
package viewquery

import (
	"context"
	"errors"
	"io"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/wrapperspb"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/internal/views/viewerror"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// TestReusableQueryRequestsAreIsolated covers RQ-U01/U02/U13 and RQ-G01 over real TCP.
func TestReusableQueryRequestsAreIsolated(t *testing.T) {
	previous := otel.GetTextMapPropagator()
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() { otel.SetTextMapPropagator(previous) })
	provider := &reusableQueryProvider{observed: make(chan reusableQueryObservation, 100)}
	client, opens := startReusableQueryServer(t, NewServer(provider, &reusableQueryScheduler{}))
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()

	// Every request gets new tasks, context, MVCC, result metadata, and a fresh adapter.
	for i := int64(1); i <= 100; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		ctx = mlog.WithFields(ctx, mlog.FieldCollectionID(i, mlog.OptPropagated()))
		spanContext := trace.NewSpanContext(trace.SpanContextConfig{
			TraceID: trace.TraceID{byte(i)}, SpanID: trace.SpanID{byte(i)}, TraceFlags: trace.FlagsSampled,
		})
		ctx = trace.ContextWithSpanContext(ctx, spanContext)
		request := reusableQueryRequest(i)
		request.Mvcc.GrowingTimetick = uint64(i)
		request.Version.QueryVersion = i
		request.ShardId.Vchannel = "channel-" + strconv.FormatInt(i, 10)
		stream, err := pool.NewStream(ctx, client, 11, request)
		require.NoError(t, err)
		require.NotNil(t, stream)
		chunk, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, []int64{i}, chunk.GetIds().GetIntId().GetData())
		require.Equal(t, int64(1), chunk.GetAllRetrieveCount())
		require.Equal(t, i, chunk.GetReqID())
		require.Equal(t, i, provider.released.Load(), "tasks must be released before completion")
		observation := <-provider.observed
		require.Equal(t, uint64(i), observation.mvcc)
		require.Equal(t, i, observation.version)
		require.Equal(t, request.GetShardId().GetVchannel(), observation.vchannel)
		require.Equal(t, spanContext.TraceID(), trace.SpanContextFromContext(observation.ctx).TraceID())
		require.Equal(t, strconv.FormatInt(i, 10), observation.collectionLog)
		_, hasDeadline := observation.ctx.Deadline()
		require.True(t, hasDeadline)
		// Do not read EOF: the final Chunk must already carry completion.
		require.NoError(t, stream.Close())
		cancel()
		require.Eventually(t, func() bool { return observation.ctx.Err() != nil }, time.Second, time.Millisecond)
	}
	require.Equal(t, int64(1), opens.Load())
	snapshot := pool.Stats()
	require.Equal(t, uint64(100), snapshot.Completed)
	require.Equal(t, uint64(99), snapshot.Reused)
	require.Equal(t, 1, snapshot.Idle)
	require.Zero(t, snapshot.Leased)
	pool.Close()
	require.Zero(t, pool.Stats().Idle)
}

func TestReusableQueryConcurrentPoolOverTCP(t *testing.T) {
	for _, size := range []int{1, 4, 16} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			provider := &reusableQueryProvider{}
			client, opens := startReusableQueryServer(t, NewServer(provider, &reusableQueryScheduler{}))
			pool := queryutil.NewReusableQueryStreamPool(context.Background(), size)
			defer pool.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			const waves = 10
			for wave := 0; wave < waves; wave++ {
				type lease struct {
					stream queryutil.ReduceStream
					id     int64
					err    error
				}
				ready := make(chan lease, size)
				start := make(chan struct{})
				for i := 0; i < size; i++ {
					id := int64(wave*size + i + 1)
					go func() {
						<-start
						stream, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(id))
						ready <- lease{stream, id, err}
					}()
				}
				close(start)
				leases := make([]lease, size)
				for i := range leases {
					leases[i] = <-ready
					require.NoError(t, leases[i].err)
					require.NotNil(t, leases[i].stream, "pool must not fall back within its capacity")
				}
				require.Equal(t, size, pool.Stats().Leased)
				var consumers sync.WaitGroup
				for _, lease := range leases {
					consumers.Add(1)
					go func() {
						defer consumers.Done()
						defer lease.stream.Close()
						chunk, err := lease.stream.Recv()
						if !assert.NoError(t, err) || !assert.NotNil(t, chunk) {
							return
						}
						assert.Equal(t, []int64{lease.id}, chunk.GetIds().GetIntId().GetData())
						assert.Equal(t, lease.id, chunk.GetReqID())
						if assert.Len(t, chunk.GetFieldsData(), 1) {
							assert.Equal(t, []int64{lease.id * 10}, chunk.GetFieldsData()[0].GetScalars().GetLongData().GetData())
						}
						_, err = lease.stream.Recv()
						assert.ErrorIs(t, err, io.EOF)
						assert.NoError(t, lease.stream.Close())
					}()
				}
				consumers.Wait()
				stats := pool.Stats()
				require.Equal(t, uint64(size), stats.Opened, "no new RPCs after the first concurrent wave")
				require.Zero(t, stats.BusyFallbacks)
				require.Equal(t, uint64(wave*size), stats.Reused)
				require.Equal(t, size, stats.Idle)
				require.Zero(t, stats.Leased)
				require.Empty(t, stats.Evicted)
			}
			require.Equal(t, int64(size), opens.Load())
			require.Equal(t, int64(waves*size), provider.released.Load())
			require.Equal(t, uint64(waves*size), pool.Stats().Completed)
		})
	}
}

// TestReusableQueryLogicalErrors preserves every ViewCode and non-retryable status before/after output.
func TestReusableQueryLogicalErrors(t *testing.T) {
	failures := []error{
		viewerror.NewGRPCStatusFromViewError(viewerror.NewViewInvalidated("stale")).Err(),
		viewerror.NewGRPCStatusFromViewError(viewerror.NewViewNotFound("missing")).Err(),
		viewerror.NewGRPCStatusFromViewError(viewerror.NewOnShutdownError("closing")).Err(),
		viewerror.NewGRPCStatusFromViewError(viewerror.NewNotPrimaryError("moved")).Err(),
		status.Error(codes.Unavailable, "transport unavailable"),
		status.Error(codes.InvalidArgument, "invalid"),
		status.Error(codes.Internal, "internal"),
		status.Error(codes.ResourceExhausted, "full"),
		status.Error(codes.Unknown, "unknown"),
		status.Error(codes.Canceled, "canceled"),
		status.Error(codes.DeadlineExceeded, "expired"),
	}
	for index, failure := range failures {
		for _, afterChunk := range []bool{false, true} {
			t.Run(strconv.Itoa(index)+"/after="+strconv.FormatBool(afterChunk), func(t *testing.T) {
				service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
					request, err := stream.Recv()
					if err != nil {
						return err
					}
					if afterChunk {
						if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: request.GetRequestId(), Chunk: reusableQueryResult(1)}); err != nil {
							return err
						}
					}
					terminal, err := anypb.New(status.Convert(failure).Proto())
					if err != nil {
						return err
					}
					return stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: request.GetRequestId(), CompletionStatus: terminal})
				}}
				client, opens := startReusableQueryServer(t, service)
				pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
				defer pool.Close()
				stream, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
				require.NoError(t, err)
				if afterChunk {
					_, err = stream.Recv()
					require.NoError(t, err)
				}
				chunk, err := stream.Recv()
				require.Nil(t, chunk)
				require.Equal(t, status.Code(failure), status.Code(err))
				require.Equal(t, viewerror.AsViewError(failure).Code, viewerror.AsViewError(err).Code)
				require.Equal(t, index < 4, viewerror.AsViewError(err).IsRetryable())
				require.NoError(t, stream.Close())
				require.Equal(t, int64(1), opens.Load(), "adapter must never replay a request")
				require.Equal(t, uint64(1), pool.Stats().Evicted["logical_error"])
				require.Zero(t, pool.Stats().Idle)
			})
		}
	}
}

// TestReusableQueryAbortsWithoutDraining covers early Close, cancellation, and deadlines.
func TestReusableQueryAbortsWithoutDraining(t *testing.T) {
	for _, mode := range []string{"early_close", "cancel", "deadline"} {
		t.Run(mode, func(t *testing.T) {
			started, stopped := make(chan struct{}), make(chan struct{})
			var attempts atomic.Int64
			service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
				request, err := stream.Recv()
				if err != nil {
					return err
				}
				if attempts.Add(1) > 1 {
					return sendReusableQuerySuccess(stream, request.GetRequestId(), reusableQueryResult(2))
				}
				if mode == "early_close" {
					if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: request.GetRequestId(), Chunk: reusableQueryResult(1)}); err != nil {
						return err
					}
				}
				close(started)
				<-stream.Context().Done()
				close(stopped)
				return stream.Context().Err()
			}}
			client, _ := startReusableQueryServer(t, service)
			pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
			defer pool.Close()
			ctx, cancel := context.WithCancel(context.Background())
			if mode == "deadline" {
				cancel()
				ctx, cancel = context.WithTimeout(context.Background(), 200*time.Millisecond)
			}
			defer cancel()
			stream, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(1))
			require.NoError(t, err)
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("server did not start")
			}
			if mode == "early_close" {
				_, err = stream.Recv()
				require.NoError(t, err)
			} else {
				if mode == "cancel" {
					cancel()
				}
				_, err = stream.Recv()
				require.ErrorIs(t, err, ctx.Err())
			}
			require.NoError(t, stream.Close())
			select {
			case <-stopped:
			case <-time.After(time.Second):
				t.Fatal("server not canceled")
			}
			require.Zero(t, pool.Stats().Idle)
			replacement, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(2))
			require.NoError(t, err)
			chunk, err := replacement.Recv()
			require.NoError(t, err)
			require.Equal(t, []int64{2}, chunk.GetIds().GetIntId().GetData())
			require.NoError(t, replacement.Close())
			require.Equal(t, uint64(2), pool.Stats().Opened)
		})
	}
}

// TestReusableQueryProtocolValidation rejects incomplete or malformed logical responses.
func TestReusableQueryProtocolValidation(t *testing.T) {
	wrongStatus, err := anypb.New(wrapperspb.String("not a status"))
	require.NoError(t, err)
	failedStatus, err := anypb.New(status.New(codes.Internal, "failed").Proto())
	require.NoError(t, err)
	invalidStatus, err := anypb.New(status.New(codes.Code(99), "invalid").Proto())
	require.NoError(t, err)
	cases := map[string]*viewpb.QueryOnViewStreamReusableResponse{
		"unsolicited interruption": {RequestId: 1, CompletionStatus: invalidStatus, Interrupted: true},
		"wrong ID":                 {RequestId: 2, Chunk: reusableQueryResult(1)},
		"missing payload":          {RequestId: 1},
		"wrong status type":        {RequestId: 1, CompletionStatus: wrongStatus},
		"invalid status code":      {RequestId: 1, CompletionStatus: invalidStatus},
		"error with rows":          {RequestId: 1, Chunk: reusableQueryResult(1), CompletionStatus: failedStatus},
		"missing Chunk status":     {RequestId: 1, Chunk: &internalpb.RetrieveResults{}},
		"EOF after Chunk":          {RequestId: 1, Chunk: reusableQueryResult(1)},
		"failed Chunk":             {RequestId: 1, Chunk: &internalpb.RetrieveResults{Status: merr.Status(merr.WrapErrServiceInternalMsg("bad"))}},
		"clean EOF":                nil,
	}
	for name, response := range cases {
		t.Run(name, func(t *testing.T) {
			service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
				if _, err := stream.Recv(); err != nil {
					return err
				}
				if response != nil {
					return stream.Send(response)
				}
				return nil
			}}
			client, _ := startReusableQueryServer(t, service)
			pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
			defer pool.Close()
			stream, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
			require.NoError(t, err)
			if name == "EOF after Chunk" {
				_, err = stream.Recv()
				require.NoError(t, err)
			}
			chunk, err := stream.Recv()
			require.Error(t, err)
			require.False(t, errors.Is(err, io.EOF))
			require.Nil(t, chunk)
			require.NoError(t, stream.Close())
			require.Zero(t, pool.Stats().Idle)
		})
	}
}

// TestReusableQueryEmptyAndBusy verifies empty EOF, separate node entries, and no waiting queue.
func TestReusableQueryEmptyAndBusy(t *testing.T) {
	provider := &reusableQueryProvider{}
	client, _ := startReusableQueryServer(t, NewServer(provider, &reusableQueryScheduler{}))
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	first, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
	require.NoError(t, err)
	busy, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(2))
	require.NoError(t, err)
	require.Nil(t, busy)
	other, err := pool.NewStream(context.Background(), client, 12, reusableQueryRequest(-1))
	require.NoError(t, err)
	chunk, err := other.Recv()
	require.NoError(t, err)
	require.Empty(t, chunk.GetIds().GetIntId().GetData())
	chunk, err = other.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, other.Close())
	_, err = first.Recv()
	require.NoError(t, err)
	require.NoError(t, first.Close())
	require.Equal(t, uint64(1), pool.Stats().BusyFallbacks)
	require.Equal(t, uint64(2), pool.Stats().Opened)
	pool.RetainNodes(map[int64]struct{}{12: {}})
	require.Equal(t, uint64(1), pool.Stats().Evicted["node_removed"])
	require.Equal(t, 1, pool.Stats().Idle)
}

// TestReusableQueryLateCancellationCannotKillNextRequest races old cleanup against a new owner.
func TestReusableQueryLateCancellationCannotKillNextRequest(t *testing.T) {
	client, opens := startReusableQueryServer(t, NewServer(&reusableQueryProvider{}, &reusableQueryScheduler{}))
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	for i := int64(1); i <= 50; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		first, err := pool.NewStream(ctx, client, 11, reusableQueryRequest(i))
		require.NoError(t, err)
		_, err = first.Recv()
		require.NoError(t, err)
		require.NoError(t, first.Close())
		next, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(i+100))
		require.NoError(t, err)
		cancel()
		require.NoError(t, first.Close())
		chunk, err := next.Recv()
		require.NoError(t, err)
		require.Equal(t, []int64{i + 100}, chunk.GetIds().GetIntId().GetData())
		require.NoError(t, next.Close())
	}
	require.Equal(t, int64(1), opens.Load())
}

// TestReusableQueryRejectsLateFrames never delivers request A's late data to request B.
func TestReusableQueryRejectsLateFrames(t *testing.T) {
	service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
		request, err := stream.Recv()
		if err != nil {
			return err
		}
		if err := sendReusableQuerySuccess(stream, request.GetRequestId(), nil); err != nil {
			return err
		}
		if err := stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: request.GetRequestId(), Chunk: reusableQueryResult(99)}); err != nil {
			return err
		}
		_, err = stream.Recv()
		if err != nil {
			return err
		}
		<-stream.Context().Done()
		return stream.Context().Err()
	}}
	client, _ := startReusableQueryServer(t, service)
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	first, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
	require.NoError(t, err)
	_, err = first.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, first.Close())
	next, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(2))
	require.NoError(t, err)
	chunk, err := next.Recv()
	require.Error(t, err)
	require.Nil(t, chunk)
	require.NoError(t, next.Close())
	require.Equal(t, uint64(1), pool.Stats().Evicted["protocol_error"])
}

// TestReusableQueryServerPreservesViewErrors exercises actual provider-to-completion mapping.
func TestReusableQueryServerPreservesViewErrors(t *testing.T) {
	for _, failure := range []*viewerror.ViewError{
		viewerror.NewViewInvalidated("stale"), viewerror.NewViewNotFound("missing"),
		viewerror.NewOnShutdownError("closing"), viewerror.NewNotPrimaryError("moved"),
	} {
		t.Run(failure.Code.String(), func(t *testing.T) {
			client, _ := startReusableQueryServer(t, NewServer(&reusableQueryProvider{failure: failure}, &reusableQueryScheduler{}))
			pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
			defer pool.Close()
			stream, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
			require.NoError(t, err)
			chunk, err := stream.Recv()
			require.Nil(t, chunk)
			require.Equal(t, failure.Code, viewerror.AsViewError(err).Code)
			require.Equal(t, failure.Cause, viewerror.AsViewError(err).Cause)
			require.True(t, viewerror.AsViewError(err).IsRetryable())
			require.NoError(t, stream.Close())
		})
	}
}

// TestReusableQueryMultiChunkCompletion reuses the RPC only after the final complete Chunk.
func TestReusableQueryMultiChunkCompletion(t *testing.T) {
	client, opens := startReusableQueryServer(t, NewServer(&reusableQueryProvider{}, &reusableQueryScheduler{rows: 5}))
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	for i := int64(1); i <= 2; i++ {
		request := reusableQueryRequest(i)
		request.LegacyReq.Limit = 20
		stream, err := pool.NewStream(context.Background(), client, 11, request)
		require.NoError(t, err)
		var ids []int64
		var chunks int
		var metadata int64
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			require.NoError(t, err)
			chunks++
			ids = append(ids, chunk.GetIds().GetIntId().GetData()...)
			metadata += chunk.GetAllRetrieveCount()
		}
		require.Equal(t, []int64{1, 2, 3, 4, 5}, ids)
		require.Greater(t, chunks, 1)
		require.Equal(t, int64(1), metadata)
		require.NoError(t, stream.Close())
	}
	require.Equal(t, int64(1), opens.Load())
}

// TestReusableQueryLegacyStatusSurvivesCompletion preserves Milvus status codes, not just gRPC codes.
func TestReusableQueryLegacyStatusSurvivesCompletion(t *testing.T) {
	original := merr.WrapErrServiceInternalMsg("scheduler returned a failed result")
	client, _ := startReusableQueryServer(t, NewServer(&reusableQueryProvider{}, &reusableQueryScheduler{
		result: &internalpb.RetrieveResults{Status: merr.Status(original)},
	}))
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	stream, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
	require.NoError(t, err)
	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.Equal(t, merr.Code(original), merr.Code(err))
	require.False(t, viewerror.AsViewError(err).IsRetryable())
	require.NoError(t, stream.Close())
}

// TestReusableQueryIdleTransportClosure fails the next exchange, not the previously completed request.
func TestReusableQueryIdleTransportClosure(t *testing.T) {
	service := &reusableQueryTestService{run: func(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
		request, err := stream.Recv()
		if err != nil {
			return err
		}
		return sendReusableQuerySuccess(stream, request.GetRequestId(), nil)
	}}
	client, _ := startReusableQueryServer(t, service)
	pool := queryutil.NewReusableQueryStreamPool(context.Background(), 1)
	defer pool.Close()
	stream, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(1))
	require.NoError(t, err)
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, stream.Close())
	next, err := pool.NewStream(context.Background(), client, 11, reusableQueryRequest(2))
	if err == nil {
		var chunk *internalpb.RetrieveResults
		chunk, err = next.Recv()
		require.Nil(t, chunk)
		require.NoError(t, next.Close())
	}
	require.Error(t, err)
	require.False(t, errors.Is(err, io.EOF), "native EOF must not become logical success")
	require.Equal(t, uint64(1), pool.Stats().Evicted["transport_error"])
	require.Zero(t, pool.Stats().Idle)
}

// reusableQueryObservation records the per-request inputs at the execution boundary.
type reusableQueryObservation struct {
	ctx           context.Context
	mvcc          uint64
	version       int64
	vchannel      string
	collectionLog string
}

// reusableQueryProvider creates fresh tasks instead of retaining a preceding Query result.
type reusableQueryProvider struct {
	released atomic.Int64
	observed chan reusableQueryObservation
	failure  error
}

// AcquireQuerySegmentTasks records the current request and owns a fresh task lease.
func (p *reusableQueryProvider) AcquireQuerySegmentTasks(ctx context.Context, shard qviews.ShardID, version qviews.QueryViewVersion, mvcc *viewpb.QueryPlanMVCC, request *internalpb.RetrieveRequest) (QuerySegmentTasks, error) {
	if p.failure != nil {
		return nil, p.failure
	}
	if p.observed != nil {
		observation := reusableQueryObservation{ctx: ctx, mvcc: mvcc.GetGrowingTimetick(), version: version.QueryVersion, vchannel: shard.VChannel}
		for _, field := range mlog.FieldsFromContext(ctx) {
			if field.Key == "collectionID" {
				observation.collectionLog = field.String
			}
		}
		p.observed <- observation
	}
	tasks := &reusableQueryTasks{owner: p}
	if request.GetReqID() >= 0 {
		tasks.units = []QuerySegmentTask{request.GetReqID()}
	}
	return tasks, nil
}

// AcquireSearchSegmentTasks is unused because the prototype is Query-only.
func (*reusableQueryProvider) AcquireSearchSegmentTasks(context.Context, qviews.ShardID, qviews.QueryViewVersion, *viewpb.QueryPlanMVCC, *internalpb.SearchRequest) (SearchSegmentTasks, error) {
	return nil, status.Error(codes.Unimplemented, "Query-only test provider")
}

// reusableQueryTasks tracks release independently for every request.
type reusableQueryTasks struct {
	owner *reusableQueryProvider
	units []QuerySegmentTask
}

// Tasks exposes only this request's segment tasks.
func (t *reusableQueryTasks) Tasks() []QuerySegmentTask { return t.units }

// Release records that the current request no longer owns segment resources.
func (t *reusableQueryTasks) Release() { t.owner.released.Add(1) }

// reusableQueryScheduler materializes a deterministic result for the current task.
type reusableQueryScheduler struct {
	rows   int
	result *internalpb.RetrieveResults
	query  func(context.Context, QuerySegmentTasks) (*internalpb.RetrieveResults, error)
}

// Query returns a distinct row and metadata for each logical request.
func (s *reusableQueryScheduler) Query(ctx context.Context, tasks QuerySegmentTasks) (*internalpb.RetrieveResults, error) {
	if s.query != nil {
		return s.query(ctx, tasks)
	}
	if s.result != nil {
		return s.result, nil
	}
	result := reusableQueryResult(tasks.Tasks()[0].(int64))
	if s.rows > 0 {
		result.Ids.GetIntId().Data = []int64{1, 2, 3, 4, 5}[:s.rows]
		result.FieldsData[0].GetScalars().GetLongData().Data = []int64{10, 20, 30, 40, 50}[:s.rows]
	}
	return result, nil
}

// Search is outside this prototype's test scope.
func (*reusableQueryScheduler) Search(context.Context, SearchSegmentTasks) (*internalpb.SearchResults, error) {
	return nil, status.Error(codes.Unimplemented, "Query-only test scheduler")
}

// reusableQueryRequest uses a real, valid QueryView request with a distinct logical input.
func reusableQueryRequest(id int64) *viewpb.QueryOnViewRequest {
	request := streamTestQueryRequest()
	request.LegacyReq.ReqID = id
	return request
}

// reusableQueryResult contains row data and request-scoped metadata for isolation checks.
func reusableQueryResult(id int64) *internalpb.RetrieveResults {
	return &internalpb.RetrieveResults{
		Status: merr.Success(), ReqID: id, AllRetrieveCount: 1,
		Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{id}}}},
		FieldsData: []*schemapb.FieldData{{Type: schemapb.DataType_Int64, FieldId: 101,
			Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{id * 10}}}}}}},
	}
}

// reusableQueryTestService allows transport failures without replacing the client adapter.
type reusableQueryTestService struct {
	viewpb.UnimplementedViewQueryServiceServer
	run func(viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error
}

// QueryOnViewStreamReusable delegates to the test's controlled real-gRPC handler.
func (s *reusableQueryTestService) QueryOnViewStreamReusable(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
	return s.run(stream)
}

// sendReusableQuerySuccess attaches authoritative completion to the final response.
func sendReusableQuerySuccess(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer, id uint64, chunk *internalpb.RetrieveResults) error {
	terminal, err := anypb.New(status.New(codes.OK, "").Proto())
	if err != nil {
		return err
	}
	return stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: id, Chunk: chunk, CompletionStatus: terminal})
}

// startReusableQueryServer runs the actual generated service over loopback TCP.
func startReusableQueryServer(t *testing.T, service viewpb.ViewQueryServiceServer) (viewpb.ViewQueryServiceClient, *atomic.Int64) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	opened := &atomic.Int64{}
	server := grpc.NewServer(grpc.StreamInterceptor(func(srv any, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if info.FullMethod == viewpb.ViewQueryService_QueryOnViewStreamReusable_FullMethodName {
			opened.Add(1)
		}
		return handler(srv, stream)
	}))
	viewpb.RegisterViewQueryServiceServer(server, service)
	go func() { _ = server.Serve(listener) }()
	connection, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = connection.Close(); server.Stop(); _ = listener.Close() })
	return viewpb.NewViewQueryServiceClient(connection), opened
}
