package handler

import (
	"context"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/internal/views/viewerror"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestClientSearchOnViewRoutesByQueryNodeID(t *testing.T) {
	searchReq := &viewpb.SearchOnViewRequest{}
	searchResp := &viewpb.SearchOnViewResponse{}
	service := &fakeViewQueryServiceClient{
		searchOnView: func(ctx context.Context, req *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error) {
			serverID, ok := contextutil.GetPickServerID(ctx)
			require.True(t, ok)
			require.Equal(t, int64(11), serverID)
			require.Same(t, searchReq, req)
			return searchResp, nil
		},
	}
	client := &clientImpl{
		lifetime: typeutil.NewLifetime(),
	}
	client.queryViewClient = &queryViewClient{
		owner:   client,
		service: fakeLazyService[viewpb.ViewQueryServiceClient]{service: service},
	}

	resp, err := client.QueryViewClient().SearchOnView(context.Background(), 11, searchReq)

	require.NoError(t, err)
	require.Same(t, searchResp, resp)
}

func TestClientSearchOnViewStreamRoutesByQueryNodeID(t *testing.T) {
	searchReq := &viewpb.SearchOnViewRequest{}
	clientStream := &fakeSearchOnViewClientStream{}
	service := &fakeViewQueryServiceClient{
		searchOnViewStream: func(ctx context.Context) (viewpb.ViewQueryService_SearchOnViewStreamClient, error) {
			serverID, ok := contextutil.GetPickServerID(ctx)
			require.True(t, ok)
			require.Equal(t, int64(11), serverID)
			clientStream.ctx = ctx
			return clientStream, nil
		},
	}
	client := &clientImpl{
		lifetime: typeutil.NewLifetime(),
	}
	client.queryViewClient = &queryViewClient{
		owner:   client,
		service: fakeLazyService[viewpb.ViewQueryServiceClient]{service: service},
	}

	stream, err := client.QueryViewClient().SearchOnViewStream(context.Background(), 11, searchReq)

	require.NoError(t, err)
	require.Same(t, searchReq, clientStream.sent.GetRequest())
	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, stream.Close())
	require.Equal(t, 1, clientStream.closeCalls)
}

func TestClientConvertsViewQueryRPCError(t *testing.T) {
	viewErr := viewerror.NewViewInvalidated("stale view")
	service := &fakeViewQueryServiceClient{
		queryOnView: func(context.Context, *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error) {
			return nil, viewerror.NewGRPCStatusFromViewError(viewErr).Err()
		},
	}
	client := &clientImpl{
		lifetime: typeutil.NewLifetime(),
	}
	client.queryViewClient = &queryViewClient{
		owner:   client,
		service: fakeLazyService[viewpb.ViewQueryServiceClient]{service: service},
	}

	_, err := client.QueryViewClient().QueryOnView(context.Background(), 11, &viewpb.QueryOnViewRequest{})

	require.Error(t, err)
	require.True(t, viewerror.AsViewError(err).IsViewInvalidated())
}

type fakeLazyService[T any] struct {
	service T
}

func (s fakeLazyService[T]) GetConn(context.Context) (*grpc.ClientConn, error) {
	return nil, nil
}

func (s fakeLazyService[T]) GetService(context.Context) (T, error) {
	return s.service, nil
}

func (s fakeLazyService[T]) Close() {}

type fakeViewQueryServiceClient struct {
	searchOnView       func(context.Context, *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error)
	searchOnViewStream func(context.Context) (viewpb.ViewQueryService_SearchOnViewStreamClient, error)
	queryOnView        func(context.Context, *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error)
	requeryOnView      func(context.Context, *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error)
}

func (c *fakeViewQueryServiceClient) SearchOnView(ctx context.Context, req *viewpb.SearchOnViewRequest, _ ...grpc.CallOption) (*viewpb.SearchOnViewResponse, error) {
	return c.searchOnView(ctx, req)
}

func (c *fakeViewQueryServiceClient) SearchOnViewStream(ctx context.Context, _ ...grpc.CallOption) (viewpb.ViewQueryService_SearchOnViewStreamClient, error) {
	return c.searchOnViewStream(ctx)
}

func (c *fakeViewQueryServiceClient) QueryOnView(ctx context.Context, req *viewpb.QueryOnViewRequest, _ ...grpc.CallOption) (*viewpb.QueryOnViewResponse, error) {
	return c.queryOnView(ctx, req)
}

func (c *fakeViewQueryServiceClient) RequeryOnView(ctx context.Context, req *viewpb.RequeryOnViewRequest, _ ...grpc.CallOption) (*viewpb.RequeryOnViewResponse, error) {
	return c.requeryOnView(ctx, req)
}

type fakeSearchOnViewClientStream struct {
	ctx        context.Context
	sent       *viewpb.SearchOnViewStreamRequest
	closeCalls int
}

func (s *fakeSearchOnViewClientStream) Send(request *viewpb.SearchOnViewStreamRequest) error {
	s.sent = request
	return nil
}

func (*fakeSearchOnViewClientStream) Recv() (*viewpb.SearchOnViewStreamResponse, error) {
	return nil, io.EOF
}

func (*fakeSearchOnViewClientStream) Header() (metadata.MD, error) {
	return nil, nil
}

func (*fakeSearchOnViewClientStream) Trailer() metadata.MD {
	return nil
}

func (s *fakeSearchOnViewClientStream) CloseSend() error {
	s.closeCalls++
	return nil
}

func (s *fakeSearchOnViewClientStream) Context() context.Context {
	return s.ctx
}

func (*fakeSearchOnViewClientStream) SendMsg(interface{}) error {
	return nil
}

func (*fakeSearchOnViewClientStream) RecvMsg(interface{}) error {
	return io.EOF
}
