// Reusable Query transport executes each logical request with fresh resources and status.
package viewquery

import (
	"context"
	"errors"
	"io"
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

// QueryOnViewStreamReusable processes requests sequentially, never retaining Query state.
func (s *Server) QueryOnViewStreamReusable(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer) error {
	var previousID uint64
	for {
		request, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if request.GetRequestId() <= previousID {
			return status.Error(codes.Internal, "reusable Query request IDs must increase")
		}
		previousID = request.GetRequestId()
		if err := s.sendReusableQueryResult(stream, request); err != nil {
			return err
		}
	}
}

// sendReusableQueryResult releases execution references before the next request is read.
func (s *Server) sendReusableQueryResult(stream viewpb.ViewQueryService_QueryOnViewStreamReusableServer, message *viewpb.QueryOnViewStreamReusableRequest) error {
	ctx := otel.GetTextMapPropagator().Extract(stream.Context(), propagation.MapCarrier(message.GetTraceContext()))
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
		terminal, err := anypb.New(completion.Proto())
		if err != nil {
			return err
		}
		return stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: message.GetRequestId(), CompletionStatus: terminal})
	}

	chunks, err := queryutil.SplitRetrieveResult(response.GetLegacyResults(), int(request.GetStreamChunkBytes()))
	if err != nil {
		cancel()
		terminal, packErr := anypb.New(status.Convert(err).Proto())
		if packErr != nil {
			return packErr
		}
		return stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: message.GetRequestId(), CompletionStatus: terminal})
	}
	terminal, err := anypb.New(status.New(codes.OK, "").Proto())
	if err != nil {
		return err
	}
	if len(chunks) == 0 {
		cancel()
		return stream.Send(&viewpb.QueryOnViewStreamReusableResponse{RequestId: message.GetRequestId(), CompletionStatus: terminal})
	}
	for index, chunk := range chunks {
		message := &viewpb.QueryOnViewStreamReusableResponse{RequestId: message.GetRequestId(), Chunk: chunk}
		if index == len(chunks)-1 {
			message.CompletionStatus = terminal
		}
		if err := ctx.Err(); err != nil {
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
