// Reusable Query transport executes each logical request with fresh resources and status.
package viewquery

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/anypb"

	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

var errReusableQueryInterrupted = errors.New("reusable Query interrupted")

// QueryOnViewStreamReusable processes requests sequentially, never retaining Query state.
func (s *Server) QueryOnViewStreamReusable(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
	type request struct {
		message *viewpb.QueryOnViewStreamReusableRequest
		ctx     context.Context
		cancel  context.CancelCauseFunc
		done    chan struct{}
	}
	ctx, cancel := context.WithCancel(stream.Context())
	defer cancel()
	requests, received := make(chan *request), make(chan error, 1)
	var mu sync.Mutex
	var active *request
	// Keep one reader active even when Query execution or a flow-controlled Send blocks.
	go func() {
		var previousID uint64
		defer close(requests)
		for {
			message, err := stream.Recv()
			if err != nil {
				if !errors.Is(err, io.EOF) {
					mu.Lock()
					if active != nil {
						active.cancel(err)
					}
					mu.Unlock()
				}
				received <- err
				return
			}
			if message.GetInterrupt() {
				if message.GetRequestId() != previousID || previousID == 0 || message.GetRequest() != nil ||
					message.GetTimeoutNanos() != 0 || len(message.GetTraceContext()) != 0 || len(message.GetLogFields()) != 0 {
					received <- status.Error(codes.Internal, "invalid reusable Query interruption")
					cancel()
					return
				}
				mu.Lock()
				if active != nil {
					select {
					case <-active.done:
					default:
						active.cancel(errReusableQueryInterrupted)
					}
				}
				mu.Unlock()
				continue
			}
			if message.GetRequestId() <= previousID {
				received <- status.Error(codes.Internal, "reusable Query request IDs must increase")
				cancel()
				return
			}
			mu.Lock()
			previous := active
			mu.Unlock()
			if previous != nil {
				select {
				case <-previous.done:
				case <-ctx.Done():
					received <- ctx.Err()
					return
				}
			}
			requestCtx, requestCancel := context.WithCancelCause(ctx)
			next := &request{message: message, ctx: requestCtx, cancel: requestCancel, done: make(chan struct{})}
			previousID = message.GetRequestId()
			mu.Lock()
			active = next
			mu.Unlock()
			select {
			case requests <- next:
			case <-ctx.Done():
				received <- ctx.Err()
				return
			}
		}
	}()
	for request := range requests {
		err := s.sendReusableQueryResult(stream, request.message, request.ctx)
		mu.Lock()
		request.cancel(nil)
		request.message, request.ctx, request.cancel = nil, nil, nil
		close(request.done)
		if active == request {
			active = nil
		}
		mu.Unlock()
		if err != nil {
			return err
		}
	}
	if err := <-received; !errors.Is(err, io.EOF) {
		return err
	}
	return nil
}

// sendReusableQueryResult releases execution references before the next request is read.
func (s *Server) sendReusableQueryResult(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer, message *viewpb.QueryOnViewStreamReusableRequest, requestCtx context.Context) error {
	ctx := otel.GetTextMapPropagator().Extract(requestCtx, propagation.MapCarrier(message.GetTraceContext()))
	for key, value := range message.GetLogFields() {
		ctx = mlog.WithFields(ctx, mlog.String(key, value))
	}
	var cancel context.CancelFunc
	if message.GetTimeoutNanos() > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(message.GetTimeoutNanos()))
	} else {
		ctx, cancel = context.WithCancel(ctx)
	}
	defer cancel()
	sendCompletion := func(completion *status.Status, interrupted bool) error {
		terminal, err := anypb.New(completion.Proto())
		if err != nil {
			return err
		}
		return stream.Send(&viewpb.QueryOnViewStreamReusableResponse{
			RequestId: message.GetRequestId(), CompletionStatus: terminal, Interrupted: interrupted,
		})
	}

	// Reuse ordinary Query execution, including its segment-task release and error mapping.
	request := message.GetRequest()
	var executionErr error
	if !queryutil.SupportsReusableQueryStream(request) || message.GetTimeoutNanos() < 0 {
		executionErr = status.Error(codes.InvalidArgument, "reusable Query supports bounded Plain Query only")
	}
	var response *viewpb.QueryOnViewResponse
	if executionErr == nil {
		response, executionErr = s.queryOnView(ctx, request)
	}
	if errors.Is(context.Cause(ctx), errReusableQueryInterrupted) {
		response = nil
		cancel()
		return sendCompletion(status.New(codes.OK, ""), true)
	}
	if executionErr == nil {
		if response.GetLegacyResults() == nil || response.GetLegacyResults().GetStatus() == nil {
			executionErr = merr.WrapErrServiceInternalMsg("reusable Query execution returned no result")
		} else if !merr.Ok(response.GetLegacyResults().GetStatus()) {
			executionErr = merr.Error(response.GetLegacyResults().GetStatus())
		}
	}
	completion := status.Convert(executionErr)
	if response.GetLegacyResults().GetStatus() != nil && !merr.Ok(response.GetLegacyResults().GetStatus()) {
		withDetails, err := completion.WithDetails(response.GetLegacyResults().GetStatus())
		if err != nil {
			return err
		}
		completion = withDetails
	}
	if errors.Is(executionErr, context.Canceled) || errors.Is(executionErr, context.DeadlineExceeded) {
		completion = status.FromContextError(executionErr)
	}
	if executionErr != nil {
		cancel()
		return sendCompletion(completion, false)
	}

	chunks, err := queryutil.SplitRetrieveResult(response.GetLegacyResults(), int(request.GetStreamChunkBytes()))
	if err != nil {
		cancel()
		return sendCompletion(status.Convert(err), false)
	}
	terminal, err := anypb.New(status.New(codes.OK, "").Proto())
	if err != nil {
		return err
	}
	if len(chunks) == 0 {
		cancel()
		return sendCompletion(status.New(codes.OK, ""), false)
	}
	for index, chunk := range chunks {
		message := &viewpb.QueryOnViewStreamReusableResponse{RequestId: message.GetRequestId(), Chunk: chunk}
		if index == len(chunks)-1 {
			message.CompletionStatus = terminal
		}
		if err := ctx.Err(); err != nil {
			if errors.Is(context.Cause(ctx), errReusableQueryInterrupted) {
				chunks, response = nil, nil
				cancel()
				return sendCompletion(status.New(codes.OK, ""), true)
			}
			return status.FromContextError(err).Err()
		}
		if message.GetCompletionStatus() != nil {
			cancel()
		}
		if err := stream.Send(message); err != nil {
			return err
		}
	}
	return nil
}
