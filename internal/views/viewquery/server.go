package viewquery

import (
	"context"
	"errors"
	"io"
	"time"

	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const defaultStreamChunkBytes = 256 * 1024

// Server implements ViewQueryService as a thin provider+scheduler adapter.
type Server struct {
	viewpb.UnimplementedViewQueryServiceServer
	provider  TaskProvider
	scheduler Scheduler
}

func NewServer(provider TaskProvider, scheduler Scheduler) *Server {
	return &Server{
		provider:  provider,
		scheduler: scheduler,
	}
}

func (s *Server) SearchOnView(ctx context.Context, req *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error) {
	metrics := searchutil.SearchBenchmarkMetricsFromContext(ctx)
	metrics.SetRequest(req)
	response, err := s.searchOnView(ctx, req)
	if err == nil {
		metrics.RecordSendAttempt(response)
		metrics.RecordSendComplete(response)
	}
	return response, err
}

func (s *Server) searchOnView(ctx context.Context, req *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error) {
	if err := validateSearchRequest(req); err != nil {
		return nil, err
	}
	var (
		result *internalpb.SearchResults
		err    error
	)
	if req.GetLegacyReq().GetIsAdvanced() {
		result, err = s.executeAdvancedSearch(ctx, req)
	} else {
		result, err = s.executeSearch(ctx, req, req.GetLegacyReq())
	}
	if err != nil {
		return nil, toRPCError(err)
	}
	searchutil.SearchBenchmarkMetricsFromContext(ctx).RecordGeneratedResult(result)
	return &viewpb.SearchOnViewResponse{LegacyResults: result}, nil
}

func (s *Server) executeAdvancedSearch(ctx context.Context, req *viewpb.SearchOnViewRequest) (*internalpb.SearchResults, error) {
	legacyReq := req.GetLegacyReq()
	if len(legacyReq.GetSubReqs()) == 0 {
		return nil, merr.WrapErrServiceInternalMsg("advanced search request has no sub-requests")
	}

	subRequests := make([]*internalpb.SearchRequest, len(legacyReq.GetSubReqs()))
	parent := proto.Clone(legacyReq).(*internalpb.SearchRequest)
	parent.SubReqs = nil
	for index, subReq := range legacyReq.GetSubReqs() {
		searchReq, err := BuildSubSearchRequest(parent, subReq)
		if err != nil {
			return nil, err
		}
		subRequests[index] = searchReq
	}

	results := make([]*internalpb.SearchResults, len(subRequests))
	group, groupCtx := errgroup.WithContext(ctx)
	for index := range subRequests {
		index := index
		group.Go(func() error {
			if legacyReq.GetSubReqs()[index].GetSkip() {
				results[index] = emptySearchResults(subRequests[index])
				return nil
			}
			result, err := s.executeSearch(groupCtx, req, subRequests[index])
			if err != nil {
				return err
			}
			if result == nil {
				return merr.WrapErrServiceInternalMsg("hybrid sub-search %d returned a nil result", index)
			}
			if !merr.Ok(result.GetStatus()) {
				return merr.Error(result.GetStatus())
			}
			results[index] = result
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		return nil, err
	}
	return assembleAdvancedSearchResults(results), nil
}

func (s *Server) executeSearch(ctx context.Context, req *viewpb.SearchOnViewRequest, searchReq *internalpb.SearchRequest) (*internalpb.SearchResults, error) {
	tasks, err := s.provider.AcquireSearchSegmentTasks(
		ctx,
		qviews.FromProtoShardID(req.GetShardId()),
		qviews.FromProtoQueryViewVersion(req.GetVersion()),
		req.GetMvcc(),
		searchReq,
	)
	if err != nil {
		return nil, err
	}
	defer tasks.Release()
	if len(tasks.Tasks()) == 0 {
		return emptySearchResults(searchReq), nil
	}

	startedAt := time.Now()
	result, err := s.scheduler.Search(ctx, tasks)
	searchutil.SearchBenchmarkMetricsFromContext(ctx).AddANNDuration(time.Since(startedAt))
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (s *Server) SearchOnViewStream(stream viewpb.ViewQueryService_SearchOnViewStreamServer) error {
	initial, err := stream.Recv()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return status.Error(codes.InvalidArgument, "SearchOnViewStream requires an initial request")
		}
		return err
	}
	request := initial.GetRequest()
	if request == nil {
		if initial.GetInterrupt() != nil {
			return status.Error(codes.Unimplemented, "SearchOnViewStream interrupt is not implemented")
		}
		return status.Error(codes.InvalidArgument, "SearchOnViewStream first message must contain a request")
	}
	legacyRequest := request.GetLegacyReq()
	if legacyRequest == nil ||
		legacyRequest.GetIsAdvanced() ||
		len(legacyRequest.GetSubReqs()) > 0 ||
		legacyRequest.GetGroupByFieldId() > 0 ||
		len(legacyRequest.GetGroupByFieldIds()) > 0 {
		return status.Error(codes.InvalidArgument, "SearchOnViewStream supports Plain ANN Search only")
	}

	metrics := searchutil.SearchBenchmarkMetricsFromContext(stream.Context())
	metrics.SetRequest(request)
	response, err := s.searchOnView(stream.Context(), request)
	if err != nil {
		return err
	}
	chunkBytes := int(request.GetStreamChunkBytes())
	if chunkBytes <= 0 {
		chunkBytes = defaultStreamChunkBytes
	}
	splitStartedAt := time.Now()
	chunks, err := searchutil.SplitSearchResult(response.GetLegacyResults(), chunkBytes)
	metrics.AddSplitDuration(time.Since(splitStartedAt))
	if err != nil {
		return status.Errorf(codes.Internal, "split SearchOnView result: %v", err)
	}
	for _, chunk := range chunks {
		message := &viewpb.SearchOnViewStreamResponse{
			Payload: &viewpb.SearchOnViewStreamResponse_Chunk{Chunk: chunk},
		}
		metrics.RecordSendAttempt(message)
		if err := stream.Send(message); err != nil {
			return err
		}
		metrics.RecordSendComplete(message)
	}
	return nil
}

func (s *Server) QueryOnView(ctx context.Context, req *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error) {
	return s.queryOnView(ctx, req)
}

func (s *Server) queryOnView(ctx context.Context, req *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error) {
	if err := validateQueryRequest(req); err != nil {
		return nil, err
	}
	tasks, err := s.provider.AcquireQuerySegmentTasks(
		ctx,
		qviews.FromProtoShardID(req.GetShardId()),
		qviews.FromProtoQueryViewVersion(req.GetVersion()),
		req.GetMvcc(),
		req.GetLegacyReq(),
	)
	if err != nil {
		return nil, toRPCError(err)
	}
	defer tasks.Release()
	if len(tasks.Tasks()) == 0 {
		return &viewpb.QueryOnViewResponse{LegacyResults: emptyQueryResults()}, nil
	}

	result, err := s.scheduler.Query(ctx, tasks)
	if err != nil {
		return nil, toRPCError(err)
	}
	return &viewpb.QueryOnViewResponse{LegacyResults: result}, nil
}

func (s *Server) QueryOnViewStream(stream viewpb.ViewQueryService_QueryOnViewStreamServer) error {
	initial, err := stream.Recv()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return status.Error(codes.InvalidArgument, "QueryOnViewStream requires an initial request")
		}
		return err
	}
	request := initial.GetRequest()
	if request == nil {
		if initial.GetInterrupt() != nil {
			return status.Error(codes.Unimplemented, "QueryOnViewStream interrupt is not implemented")
		}
		return status.Error(codes.InvalidArgument, "QueryOnViewStream first message must contain a request")
	}
	legacyRequest := request.GetLegacyReq()
	if legacyRequest == nil || legacyRequest.GetLimit() <= 0 || legacyRequest.GetIsCount() ||
		len(legacyRequest.GetGroupByFieldIds()) > 0 || len(legacyRequest.GetAggregates()) > 0 ||
		len(legacyRequest.GetOrderByFields()) > 0 {
		return status.Error(codes.InvalidArgument, "QueryOnViewStream supports bounded Plain Query only")
	}

	response, err := s.queryOnView(stream.Context(), request)
	if err != nil {
		return err
	}
	chunkBytes := int(request.GetStreamChunkBytes())
	if chunkBytes <= 0 {
		chunkBytes = defaultStreamChunkBytes
	}
	chunks, err := queryutil.SplitRetrieveResult(response.GetLegacyResults(), chunkBytes)
	if err != nil {
		return status.Errorf(codes.Internal, "split QueryOnView result: %v", err)
	}
	for _, chunk := range chunks {
		if err := stream.Send(&viewpb.QueryOnViewStreamResponse{
			Payload: &viewpb.QueryOnViewStreamResponse_Chunk{Chunk: chunk},
		}); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) RequeryOnView(context.Context, *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error) {
	return nil, status.Error(codes.Unimplemented, "RequeryOnView is not implemented")
}
