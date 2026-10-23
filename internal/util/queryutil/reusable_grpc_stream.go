// Reusable Query RPCs retain transport only; each exclusive lease is a fresh ReduceStream.
package queryutil

import (
	"context"
	"errors"
	"io"
	"slices"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	statuspb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const reusableQueryIdleTimeout = time.Minute

// ReusableQueryStreamPool bounds opening, idle, and leased RPCs per QueryNode.
type ReusableQueryStreamPool struct {
	mu               sync.Mutex
	ctx              context.Context
	cancel           context.CancelFunc
	entries          map[int64][]*reusableQueryRPC
	size             int
	idleTimeout      time.Duration
	interruptTimeout time.Duration
	closed           bool
	nextRPCID        uint64
	stats            ReusableQueryStreamStats
}

// ReusableQueryStreamStats separates logical requests from transport lifetimes.
type ReusableQueryStreamStats struct {
	Opened, Reused, BusyFallbacks, Completed uint64
	Interrupted, DiscardedChunks             uint64
	Idle, Leased                             int
	Evicted                                  map[string]uint64
}

// reusableQueryRPC is exclusively owned until its current adapter closes.
type reusableQueryRPC struct {
	id             uint64
	client         viewpb.ViewQueryService_QueryOnViewStreamReusableClient
	ctx            context.Context
	cancel         context.CancelFunc
	request        uint64
	busy           bool
	timer          *time.Timer
	idleGeneration uint64
}

// NewReusableQueryStreamPool binds cached RPCs to the client's lifetime, not a Query.
func NewReusableQueryStreamPool(ctx context.Context, size int) *ReusableQueryStreamPool {
	ctx, cancel := context.WithCancel(ctx)
	return &ReusableQueryStreamPool{
		ctx: ctx, cancel: cancel, entries: make(map[int64][]*reusableQueryRPC), size: max(1, size),
		idleTimeout: reusableQueryIdleTimeout, interruptTimeout: time.Second,
		stats: ReusableQueryStreamStats{Evicted: make(map[string]uint64)},
	}
}

// SupportsReusableQueryStream excludes every reduction policy outside Plain Query.
func SupportsReusableQueryStream(request *viewpb.QueryOnViewRequest) bool {
	req := request.GetLegacyReq()
	return req != nil && req.GetLimit() > 0 && request.GetStreamChunkBytes() > 0 &&
		!req.GetIsCount() && len(req.GetGroupByFieldIds()) == 0 &&
		len(req.GetAggregates()) == 0 && len(req.GetOrderByFields()) == 0
}

// NewStream returns nil without error when this request must use ordinary streaming.
func (p *ReusableQueryStreamPool) NewStream(ctx context.Context, client viewpb.ViewQueryServiceClient, nodeID int64, request *viewpb.QueryOnViewRequest) (ReduceStream, error) {
	if !SupportsReusableQueryStream(request) {
		return nil, nil
	}
	traceContext := propagation.MapCarrier{}
	if md, ok := metadata.FromOutgoingContext(ctx); ok {
		// Unknown metadata may carry credentials or routing scope. Never silently drop it.
		for key, values := range md {
			if key != "traceparent" && key != "tracestate" && key != "baggage" {
				return nil, nil
			}
			if len(values) != 1 {
				return nil, nil
			}
			traceContext[key] = values[0]
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if client == nil {
		return nil, merr.WrapErrServiceInternalMsg("reusable Query stream requires a client")
	}
	otel.GetTextMapPropagator().Inject(ctx, traceContext)

	// Reserve capacity before opening outside the lock; overflow requests never wait.
	p.mu.Lock()
	if p.closed || p.ctx.Err() != nil {
		p.mu.Unlock()
		return nil, context.Canceled
	}
	var rpc *reusableQueryRPC
	for i := len(p.entries[nodeID]) - 1; i >= 0; i-- {
		candidate := p.entries[nodeID][i]
		if candidate.ctx.Err() != nil {
			p.evictLocked(nodeID, candidate, "transport_error")
		} else if !candidate.busy {
			rpc = candidate
		}
	}
	if rpc == nil && len(p.entries[nodeID]) >= p.size {
		p.stats.BusyFallbacks++
		metrics.ProxyReusableQueryStreamEvents.WithLabelValues("busy_fallback").Inc()
		p.mu.Unlock()
		return nil, nil
	}
	opening := rpc == nil
	if opening {
		rpcCtx, cancel := context.WithCancel(contextutil.WithPickServerID(p.ctx, nodeID))
		p.nextRPCID++
		rpc = &reusableQueryRPC{id: p.nextRPCID, ctx: rpcCtx, cancel: cancel}
		p.entries[nodeID] = append(p.entries[nodeID], rpc)
	} else {
		rpc.timer.Stop()
		rpc.idleGeneration++
		p.stats.Reused++
		metrics.ProxyReusableQueryStreamEvents.WithLabelValues("reused").Inc()
		metrics.ProxyReusableQueryStreams.WithLabelValues("idle").Dec()
	}
	rpc.busy = true
	rpc.request++
	metrics.ProxyReusableQueryStreams.WithLabelValues("leased").Inc()
	s := &reusableQueryStream{pool: p, rpc: rpc, nodeID: nodeID, requestID: rpc.request, ctx: ctx}
	p.mu.Unlock()

	// Cancellation kills this RPC only. Close joins the callback before caching it.
	canceled := make(chan struct{})
	stop := context.AfterFunc(ctx, func() {
		rpc.cancel()
		close(canceled)
	})
	s.stopCancellation = func() {
		if !stop() {
			<-canceled
		}
	}
	var err error
	if opening {
		rpc.client, err = client.QueryOnViewStreamReusable(rpc.ctx)
		if err == nil {
			p.mu.Lock()
			p.stats.Opened++
			metrics.ProxyReusableQueryStreamEvents.WithLabelValues("opened").Inc()
			p.mu.Unlock()
		}
	}
	if err == nil {
		message := &viewpb.QueryOnViewStreamReusableRequest{
			RequestId: s.requestID, Request: request,
			TraceContext: traceContext, LogFields: mlog.GetPropagated(ctx),
		}
		if deadline, ok := ctx.Deadline(); ok {
			message.TimeoutNanos = max(1, time.Until(deadline).Nanoseconds())
		}
		err = rpc.client.Send(message)
	}
	if err != nil {
		if errors.Is(err, io.EOF) {
			// Send's EOF is not logical completion. Recv exposes the actual RPC status.
			_, terminalErr := rpc.client.Recv()
			if terminalErr != nil && !errors.Is(terminalErr, io.EOF) {
				err = terminalErr
			} else {
				err = merr.WrapErrServiceInternalMsg("reusable Query RPC ended before request completion")
			}
		}
		err = s.fail(err, "transport_error")
		_ = s.Close()
		return nil, err
	}
	if mlog.LevelEnabled(mlog.DebugLevel) {
		mlog.Debug(ctx, "reusable Query request sent", mlog.FieldNodeID(nodeID),
			mlog.Uint64("rpcID", rpc.id), mlog.Uint64("logicalRequestID", s.requestID),
			mlog.Int64("requestID", request.GetLegacyReq().GetReqID()), mlog.Bool("reused", !opening))
	}
	return s, nil
}

// Stats returns a copy of counters and the current idle/leased entry counts.
func (p *ReusableQueryStreamPool) Stats() ReusableQueryStreamStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	result := p.stats
	result.Evicted = make(map[string]uint64, len(p.stats.Evicted))
	for reason, count := range p.stats.Evicted {
		result.Evicted[reason] = count
	}
	for _, rpcs := range p.entries {
		for _, rpc := range rpcs {
			if rpc.busy {
				result.Leased++
			} else {
				result.Idle++
			}
		}
	}
	return result
}

// RetainNodes evicts entries whose QueryNodes disappeared from discovery.
func (p *ReusableQueryStreamPool) RetainNodes(nodes map[int64]struct{}) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for nodeID, rpcs := range p.entries {
		if _, exists := nodes[nodeID]; !exists {
			for i := len(rpcs) - 1; i >= 0; i-- {
				p.evictLocked(nodeID, rpcs[i], "node_removed")
			}
		}
	}
}

// Close cancels all leased and idle RPCs before the underlying client closes.
func (p *ReusableQueryStreamPool) Close() {
	p.cancel()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	for nodeID, rpcs := range p.entries {
		for i := len(rpcs) - 1; i >= 0; i-- {
			p.evictLocked(nodeID, rpcs[i], "shutdown")
		}
	}
}

// evictLocked removes only this entry, never a replacement opened by a later request.
func (p *ReusableQueryStreamPool) evictLocked(nodeID int64, rpc *reusableQueryRPC, reason string) {
	rpc.cancel()
	rpcs := p.entries[nodeID]
	index := slices.Index(rpcs, rpc)
	if index < 0 {
		return
	}
	rpcs = slices.Delete(rpcs, index, index+1)
	if len(rpcs) == 0 {
		delete(p.entries, nodeID)
	} else {
		p.entries[nodeID] = rpcs
	}
	if rpc.timer != nil {
		rpc.timer.Stop()
	}
	state := "idle"
	if rpc.busy {
		state = "leased"
	}
	metrics.ProxyReusableQueryStreams.WithLabelValues(state).Dec()
	p.stats.Evicted[reason]++
	metrics.ProxyReusableQueryStreamEvents.WithLabelValues(reason).Inc()
	if mlog.LevelEnabled(mlog.DebugLevel) {
		mlog.Debug(p.ctx, "reusable Query RPC evicted", mlog.FieldNodeID(nodeID),
			mlog.Uint64("rpcID", rpc.id), mlog.String("reason", reason))
	}
}

// reusableQueryStream holds only one logical request and its completion state.
type reusableQueryStream struct {
	pool             *ReusableQueryStreamPool
	rpc              *reusableQueryRPC
	nodeID           int64
	requestID        uint64
	ctx              context.Context
	stopCancellation func()
	ioMu             sync.Mutex
	mu               sync.Mutex
	completed        bool
	closed           bool
	err              error
	closeOnce        sync.Once
}

// Recv returns logical EOF after explicit completion, without ending the RPC.
func (s *reusableQueryStream) Recv() (*internalpb.RetrieveResults, error) {
	s.ioMu.Lock()
	defer s.ioMu.Unlock()
	s.mu.Lock()
	closed, completed, previousErr := s.closed, s.completed, s.err
	s.mu.Unlock()
	if closed {
		return nil, io.ErrClosedPipe
	}
	if previousErr != nil {
		return nil, previousErr
	}
	if err := s.ctx.Err(); err != nil {
		return nil, s.fail(err, "canceled")
	}
	if completed {
		return nil, io.EOF
	}

	chunk, err := s.recv()
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	s.mu.Lock()
	closed = s.closed
	s.mu.Unlock()
	if closed {
		return nil, io.ErrClosedPipe
	}
	return chunk, err
}

// recv is serialized by ioMu, including completion cleanup after Close begins.
func (s *reusableQueryStream) recv() (*internalpb.RetrieveResults, error) {
	response, err := s.rpc.client.Recv()
	if err != nil {
		if errors.Is(err, io.EOF) {
			err = merr.WrapErrServiceInternalMsg("reusable Query RPC ended before request completion")
		}
		return nil, s.fail(err, "transport_error")
	}
	if err := s.ctx.Err(); err != nil {
		return nil, s.fail(err, "canceled")
	}
	if response == nil || response.GetRequestId() != s.requestID ||
		(response.GetChunk() == nil && response.GetCompletionStatus() == nil) {
		return nil, s.fail(merr.WrapErrServiceInternalMsg("invalid reusable Query response for request %d", s.requestID), "protocol_error")
	}
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if response.GetInterrupted() && (!closed || response.GetChunk() != nil || response.GetCompletionStatus() == nil) {
		return nil, s.fail(merr.WrapErrServiceInternalMsg("invalid reusable Query interruption completion"), "protocol_error")
	}
	if chunk := response.GetChunk(); chunk != nil && chunk.GetStatus() == nil {
		return nil, s.fail(merr.WrapErrServiceInternalMsg("reusable Query Chunk is missing its status"), "protocol_error")
	}
	if chunk := response.GetChunk(); chunk != nil && !merr.Ok(chunk.GetStatus()) {
		return nil, s.fail(merr.Error(chunk.GetStatus()), "logical_error")
	}
	if terminal := response.GetCompletionStatus(); terminal != nil {
		completion := &statuspb.Status{}
		if err := terminal.UnmarshalTo(completion); err != nil {
			return nil, s.fail(merr.Wrap(err, "invalid reusable Query completion"), "protocol_error")
		}
		if completion.GetCode() < int32(codes.OK) || completion.GetCode() > int32(codes.Unauthenticated) {
			return nil, s.fail(merr.WrapErrServiceInternalMsg("invalid reusable Query status code"), "protocol_error")
		}
		if response.GetInterrupted() && completion.GetCode() != int32(codes.OK) {
			return nil, s.fail(merr.WrapErrServiceInternalMsg("interrupted reusable Query completion is not OK"), "protocol_error")
		}
		if err := status.FromProto(completion).Err(); err != nil {
			if response.GetChunk() != nil {
				return nil, s.fail(merr.WrapErrServiceInternalMsg("failed reusable Query completion contains a Chunk"), "protocol_error")
			}
			for _, detail := range status.FromProto(completion).Details() {
				if legacyStatus, ok := detail.(*commonpb.Status); ok && !merr.Ok(legacyStatus) {
					return nil, s.fail(merr.Error(legacyStatus), "logical_error")
				}
			}
			return nil, s.fail(err, "logical_error")
		}
		s.mu.Lock()
		s.completed = true
		s.mu.Unlock()
		s.pool.mu.Lock()
		if response.GetInterrupted() {
			s.pool.stats.Interrupted++
			metrics.ProxyReusableQueryStreamEvents.WithLabelValues("interrupted").Inc()
		} else {
			s.pool.stats.Completed++
			metrics.ProxyReusableQueryStreamEvents.WithLabelValues("completed").Inc()
		}
		s.pool.mu.Unlock()
	}
	if chunk := response.GetChunk(); chunk != nil {
		if closed {
			s.pool.mu.Lock()
			s.pool.stats.DiscardedChunks++
			metrics.ProxyReusableQueryStreamEvents.WithLabelValues("discarded_chunk").Inc()
			s.pool.mu.Unlock()
		}
		return chunk, nil
	}
	return nil, io.EOF
}

// fail preserves the original error and immediately makes the RPC ineligible for reuse.
func (s *reusableQueryStream) fail(err error, reason string) error {
	if contextErr := s.ctx.Err(); contextErr != nil {
		err, reason = contextErr, "canceled"
	}
	s.mu.Lock()
	// Close cancels a pending receive without making it a transport failure.
	if s.closed && reason == "transport_error" && (errors.Is(err, context.Canceled) || status.Code(err) == codes.Canceled) {
		reason = "early_close"
	}
	s.err = err
	s.mu.Unlock()
	s.pool.mu.Lock()
	s.pool.evictLocked(s.nodeID, s.rpc, reason)
	s.pool.mu.Unlock()
	return err
}

// Close interrupts unfinished requests and returns only verified clean RPCs to idle.
func (s *reusableQueryStream) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		s.closed = true
		healthy := s.completed && s.err == nil
		canInterrupt := !s.completed && s.err == nil
		s.mu.Unlock()
		s.stopCancellation()
		if canInterrupt && s.ctx.Err() == nil && s.rpc.ctx.Err() == nil {
			ctx, cancel := context.WithTimeout(s.ctx, s.pool.interruptTimeout)
			stopped := make(chan struct{})
			stop := context.AfterFunc(ctx, func() {
				s.pool.mu.Lock()
				reason := "interrupt_timeout"
				if s.ctx.Err() != nil {
					reason = "canceled"
				}
				s.pool.evictLocked(s.nodeID, s.rpc, reason)
				s.pool.mu.Unlock()
				close(stopped)
			})
			err := s.rpc.client.Send(&viewpb.QueryOnViewStreamReusableRequest{RequestId: s.requestID, Interrupt: true})
			if err != nil {
				s.fail(err, "transport_error")
			} else {
				s.ioMu.Lock()
				for {
					s.mu.Lock()
					finished := s.completed || s.err != nil
					s.mu.Unlock()
					if finished {
						break
					}
					if _, err := s.recv(); err != nil {
						break
					}
				}
				s.ioMu.Unlock()
			}
			if !stop() {
				<-stopped
			}
			cancel()
			s.mu.Lock()
			healthy = s.completed && s.err == nil
			s.mu.Unlock()
		}
		if !healthy || s.ctx.Err() != nil {
			s.rpc.cancel()
		}
		// Cancel before joining Recv, and join the old callback before releasing ownership.
		s.ioMu.Lock()
		defer s.ioMu.Unlock()
		s.pool.mu.Lock()
		defer s.pool.mu.Unlock()
		if !slices.Contains(s.pool.entries[s.nodeID], s.rpc) {
			return
		}
		if !healthy || s.rpc.ctx.Err() != nil || s.pool.closed {
			reason := "early_close"
			if s.ctx.Err() != nil {
				reason = "canceled"
			}
			s.pool.evictLocked(s.nodeID, s.rpc, reason)
			return
		}
		s.rpc.busy = false
		metrics.ProxyReusableQueryStreams.WithLabelValues("leased").Dec()
		metrics.ProxyReusableQueryStreams.WithLabelValues("idle").Inc()
		pool, rpc, nodeID := s.pool, s.rpc, s.nodeID
		rpc.idleGeneration++
		generation := rpc.idleGeneration
		rpc.timer = time.AfterFunc(pool.idleTimeout, func() {
			pool.mu.Lock()
			defer pool.mu.Unlock()
			if slices.Contains(pool.entries[nodeID], rpc) && !rpc.busy && rpc.idleGeneration == generation {
				pool.evictLocked(nodeID, rpc, "idle_expired")
			}
		})
	})
	return nil
}

// Interrupt remains unimplemented, as on the existing Query transport.
func (s *reusableQueryStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, merr.WrapErrServiceUnimplemented(status.Error(codes.Unimplemented, "reusable Query Interrupt is not implemented"))
}
