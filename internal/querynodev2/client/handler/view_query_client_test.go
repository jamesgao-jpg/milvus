package handler

import (
	"context"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/anypb"

	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/internal/views/viewerror"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
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

func TestClientQueryOnViewStreamRoutesByQueryNodeID(t *testing.T) {
	queryReq := &viewpb.QueryOnViewRequest{}
	clientStream := &fakeQueryOnViewClientStream{}
	service := &fakeViewQueryServiceClient{
		queryOnViewStream: func(ctx context.Context) (viewpb.ViewQueryService_QueryOnViewStreamClient, error) {
			serverID, ok := contextutil.GetPickServerID(ctx)
			require.True(t, ok)
			require.Equal(t, int64(11), serverID)
			clientStream.ctx = ctx
			return clientStream, nil
		},
	}
	client := &clientImpl{lifetime: typeutil.NewLifetime()}
	client.queryViewClient = &queryViewClient{
		owner:   client,
		service: fakeLazyService[viewpb.ViewQueryServiceClient]{service: service},
	}

	stream, err := client.QueryViewClient().QueryOnViewStream(context.Background(), 11, queryReq)

	require.NoError(t, err)
	require.Same(t, queryReq, clientStream.sent.GetRequest())
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
	viewpb.ViewQueryServiceClient
	searchOnView       func(context.Context, *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error)
	searchOnViewStream func(context.Context) (viewpb.ViewQueryService_SearchOnViewStreamClient, error)
	queryOnView        func(context.Context, *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error)
	queryOnViewStream  func(context.Context) (viewpb.ViewQueryService_QueryOnViewStreamClient, error)
	requeryOnView      func(context.Context, *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error)
	queryReusable      func(context.Context) (viewpb.ViewQueryService_QueryOnViewStreamReusableClient, error)
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

func (c *fakeViewQueryServiceClient) QueryOnViewStream(ctx context.Context, _ ...grpc.CallOption) (viewpb.ViewQueryService_QueryOnViewStreamClient, error) {
	return c.queryOnViewStream(ctx)
}

// QueryOnViewStreamReusable exposes the prototype only when a test explicitly supplies it.
func (c *fakeViewQueryServiceClient) QueryOnViewStreamReusable(ctx context.Context, _ ...grpc.CallOption) (viewpb.ViewQueryService_QueryOnViewStreamReusableClient, error) {
	return c.queryReusable(ctx)
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

type fakeQueryOnViewClientStream struct {
	ctx        context.Context
	sent       *viewpb.QueryOnViewStreamRequest
	closeCalls int
}

func (s *fakeQueryOnViewClientStream) Send(request *viewpb.QueryOnViewStreamRequest) error {
	s.sent = request
	return nil
}

func (*fakeQueryOnViewClientStream) Recv() (*viewpb.QueryOnViewStreamResponse, error) {
	return nil, io.EOF
}

func (*fakeQueryOnViewClientStream) Header() (metadata.MD, error) {
	return nil, nil
}

func (*fakeQueryOnViewClientStream) Trailer() metadata.MD {
	return nil
}

func (s *fakeQueryOnViewClientStream) CloseSend() error {
	s.closeCalls++
	return nil
}

func (s *fakeQueryOnViewClientStream) Context() context.Context {
	return s.ctx
}

func (*fakeQueryOnViewClientStream) SendMsg(interface{}) error {
	return nil
}

func (*fakeQueryOnViewClientStream) RecvMsg(interface{}) error {
	return io.EOF
}

// TestReusableQueryClientSelection verifies both gates without changing ordinary client APIs.
func TestReusableQueryClientSelection(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	for _, key := range []string{params.ProxyCfg.EnableReduceStream.Key, params.ProxyCfg.EnableReusableQueryStream.Key} {
		value := params.GetWithDefault(key, "false")
		t.Cleanup(func() { _ = params.Save(key, value) })
	}
	for _, enabled := range []bool{false, true} {
		for _, reuse := range []bool{false, true} {
			require.NoError(t, params.Save(params.ProxyCfg.EnableReduceStream.Key, map[bool]string{false: "false", true: "true"}[enabled]))
			require.NoError(t, params.Save(params.ProxyCfg.EnableReusableQueryStream.Key, map[bool]string{false: "false", true: "true"}[reuse]))
			owner := &clientImpl{lifetime: typeutil.NewLifetime()}
			owner.queryViewClient = newQueryViewClient(owner, fakeLazyService[viewpb.ViewQueryServiceClient]{})
			require.Equal(t, enabled && reuse, owner.queryViewClient.reusableQueries != nil)
			owner.Close()
		}
	}
}

// TestReusableQueryClientBusyFallback proves the actual QN client uses legacy transport when leased.
func TestReusableQueryClientBusyFallback(t *testing.T) {
	var reusableOpens, ordinaryOpens int
	service := &fakeViewQueryServiceClient{
		queryReusable: func(ctx context.Context) (viewpb.ViewQueryService_QueryOnViewStreamReusableClient, error) {
			reusableOpens++
			node, ok := contextutil.GetPickServerID(ctx)
			require.True(t, ok)
			require.Equal(t, int64(11), node)
			return &fakeReusableQueryClientStream{ctx: ctx}, nil
		},
		queryOnViewStream: func(ctx context.Context) (viewpb.ViewQueryService_QueryOnViewStreamClient, error) {
			ordinaryOpens++
			return &fakeQueryOnViewClientStream{ctx: ctx}, nil
		},
	}
	owner := &clientImpl{lifetime: typeutil.NewLifetime()}
	owner.queryViewClient = &queryViewClient{owner: owner, service: fakeLazyService[viewpb.ViewQueryServiceClient]{service: service},
		reusableQueries: queryutil.NewReusableQueryStreamPool(context.Background())}
	defer owner.Close()
	request := &viewpb.QueryOnViewRequest{LegacyReq: &internalpb.RetrieveRequest{Limit: 2}, StreamChunkBytes: 262144}
	first, err := owner.QueryViewClient().QueryOnViewStream(context.Background(), 11, request)
	require.NoError(t, err)
	second, err := owner.QueryViewClient().QueryOnViewStream(context.Background(), 11, request)
	require.NoError(t, err)
	require.Equal(t, 1, ordinaryOpens)
	_, err = second.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, second.Close())
	_, err = first.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, first.Close())
	third, err := owner.QueryViewClient().QueryOnViewStream(context.Background(), 11, request)
	require.NoError(t, err)
	_, err = third.Recv()
	require.ErrorIs(t, err, io.EOF)
	require.NoError(t, third.Close())
	require.Equal(t, 1, reusableOpens)
}

// fakeReusableQueryClientStream produces an explicit completion for each request ID.
type fakeReusableQueryClientStream struct {
	grpc.ClientStream
	ctx       context.Context
	requestID uint64
}

// Context exposes the transport context rather than the current logical request.
func (s *fakeReusableQueryClientStream) Context() context.Context { return s.ctx }

// Send records the logical request independently of the native RPC.
func (s *fakeReusableQueryClientStream) Send(request *viewpb.QueryOnViewStreamReusableRequest) error {
	s.requestID = request.GetRequestId()
	return nil
}

// Recv returns completion without relying on native gRPC EOF.
func (s *fakeReusableQueryClientStream) Recv() (*viewpb.QueryOnViewStreamReusableResponse, error) {
	terminal, err := anypb.New(status.New(codes.OK, "").Proto())
	return &viewpb.QueryOnViewStreamReusableResponse{RequestId: s.requestID, CompletionStatus: terminal}, err
}
