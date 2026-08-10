package viewquery

import (
	"context"
	"errors"
	"io"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
)

const defaultSearchStreamChunkSize = 1024

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
	if err := validateSearchRequest(req); err != nil {
		return nil, err
	}
	tasks, err := s.provider.AcquireSearchSegmentTasks(
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
		return &viewpb.SearchOnViewResponse{LegacyResults: emptySearchResults(req.GetLegacyReq())}, nil
	}

	result, err := s.scheduler.Search(ctx, tasks)
	if err != nil {
		return nil, toRPCError(err)
	}
	return &viewpb.SearchOnViewResponse{LegacyResults: result}, nil
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

	response, err := s.SearchOnView(stream.Context(), request)
	if err != nil {
		return err
	}
	chunkSize := int(request.GetStreamChunkSize())
	if chunkSize <= 0 {
		chunkSize = defaultSearchStreamChunkSize
	}
	chunks, err := searchutil.SplitSearchResult(response.GetLegacyResults(), chunkSize)
	if err != nil {
		return status.Errorf(codes.Internal, "split SearchOnView result: %v", err)
	}
	for _, chunk := range chunks {
		if err := stream.Send(&viewpb.SearchOnViewStreamResponse{
			Payload: &viewpb.SearchOnViewStreamResponse_Chunk{Chunk: chunk},
		}); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) QueryOnView(ctx context.Context, req *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error) {
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

func (s *Server) RequeryOnView(context.Context, *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error) {
	return nil, status.Error(codes.Unimplemented, "RequeryOnView is not implemented")
}
