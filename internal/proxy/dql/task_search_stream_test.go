package dql

import (
	"context"
	"io"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proxy/shardclient"
	searchreducestream "github.com/milvus-io/milvus/internal/util/searchutil/reduce_stream"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type testLeaderSearchStream struct {
	grpc.ClientStream
	ctx    context.Context
	send   func(*querypb.SearchStreamRequest)
	chunks []*internalpb.SearchResults
	err    error
}

func (s *testLeaderSearchStream) Send(request *querypb.SearchStreamRequest) error {
	if s.send != nil {
		s.send(request)
	}
	return nil
}

func (s *testLeaderSearchStream) CloseSend() error { return nil }

func (s *testLeaderSearchStream) Recv() (*internalpb.SearchResults, error) {
	if err := s.ctx.Err(); err != nil {
		return nil, err
	}
	if len(s.chunks) == 0 {
		if s.err != nil {
			return nil, s.err
		}
		return nil, io.EOF
	}
	chunk := s.chunks[0]
	s.chunks = s.chunks[1:]
	return chunk, nil
}

func TestSearchTaskReduceStreamBatchParity(t *testing.T) {
	paramtable.Init()
	cfg := &paramtable.Get().ProxyCfg
	paramtable.Get().Save(cfg.EnableReduceStream.Key, "true")
	paramtable.Get().Save(cfg.ReduceStreamChunkBytes.Key, "1")
	t.Cleanup(func() {
		paramtable.Get().Reset(cfg.EnableReduceStream.Key)
		paramtable.Get().Reset(cfg.ReduceStreamChunkBytes.Key)
	})
	ctx := context.Background()
	req := &internalpb.SearchRequest{
		Base: &commonpb.MsgBase{}, Nq: 2, Topk: 2,
		SearchType: internalpb.SearchType_PURE_ANN_SEARCH_NO_FILTER,
	}

	lb := shardclient.NewMockLBPolicy(t)
	task := &SearchTask{
		ctx: ctx, SearchRequest: req, request: &milvuspb.SearchRequest{},
		schema: mustNewSchemaInfo(constructCollectionSchema(testInt64Field, testFloatVecField, testVecDim, "test")),
		lb:     lb, resultBuf: typeutil.NewConcurrentSet[*internalpb.SearchResults](),
	}

	batch := []*internalpb.SearchResults{
		{Status: merr.Success(), MetricType: "L2", ResultData: genSearchResultData(2, 2, []int64{1, 5, 6, 8}, []float32{-1, -4, -2, -3})},
		{Status: merr.Success(), MetricType: "L2", ResultData: genSearchResultData(2, 2, []int64{1, 2, 3}, []float32{-1, -1, -2})},
	}
	batch[0].ResultData.Topks = []int64{2, 2}
	batch[1].ResultData.Topks = []int64{2, 1}
	for i, result := range batch {
		result.ChannelsMvcc = map[string]uint64{[]string{"a", "b"}[i]: uint64(i + 1)}
	}
	batchResult, err := reduceResults(ctx, batch, 2, 2, 0, "L2", schemapb.DataType_Int64, &planpb.QueryInfo{}, false, false, 0, nil, false)
	require.NoError(t, err)

	a, b := mocks.NewMockQueryNodeClient(t), mocks.NewMockQueryNodeClient(t)
	var opened []*testLeaderSearchStream
	for i, qn := range []*mocks.MockQueryNodeClient{a, b} {
		result := proto.CloneOf(batch[i])
		channel := []string{"a", "b"}[i]
		var chunks []*internalpb.SearchResults
		require.NoError(t, searchreducestream.SplitSearchResult(result, 2, 2, 1, func(chunk *internalpb.SearchResults) error {
			chunks = append(chunks, chunk)
			return nil
		}))
		qn.EXPECT().SearchStream(mock.Anything).RunAndReturn(func(rpcCtx context.Context, _ ...grpc.CallOption) (querypb.QueryNode_SearchStreamClient, error) {
			stream := &testLeaderSearchStream{ctx: rpcCtx, chunks: chunks, send: func(request *querypb.SearchStreamRequest) {
				require.EqualValues(t, 1, request.GetChunkBytes())
				require.EqualValues(t, 1, request.GetRequest().GetTotalChannelNum())
				require.Equal(t, []string{channel}, request.GetRequest().GetDmlChannels())
			}}
			opened = append(opened, stream)
			return stream, nil
		}).Once()
	}
	lb.EXPECT().UpdateCostMetrics(mock.Anything, mock.Anything).Twice()
	lb.EXPECT().Execute(mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, workload shardclient.CollectionWorkLoad) error {
		if err := workload.Exec(ctx, 1, a, "a"); err != nil {
			return err
		}
		return workload.Exec(ctx, 2, b, "b")
	}).Once()

	require.NoError(t, task.Execute(ctx))
	chunks, err := task.collectSearchResults(ctx)
	require.NoError(t, err)
	require.Len(t, chunks, 1, "ordinary post-processing sees one already-merged result")
	require.Equal(t, map[string]uint64{"a": 1, "b": 2}, chunks[0].GetChannelsMvcc())
	streamResult, err := reduceResults(ctx, chunks, 2, 2, 0, "L2", schemapb.DataType_Int64, &planpb.QueryInfo{}, false, false, 0, nil, false)
	require.NoError(t, err)
	require.True(t, proto.Equal(batchResult, streamResult), "batch: %v; stream: %v", batchResult, streamResult)
	for _, stream := range opened {
		require.ErrorIs(t, stream.ctx.Err(), context.Canceled)
	}
}

func TestSearchStreamFirstStatusIsAvailableForLeaderRetry(t *testing.T) {
	qn := mocks.NewMockQueryNodeClient(t)
	mgr := shardclient.NewMockShardClientManager(t)
	task := &SearchTask{SearchRequest: &internalpb.SearchRequest{Base: &commonpb.MsgBase{}, CollectionID: 42}, shardClientMgr: mgr}
	var rpcCtx context.Context
	qn.EXPECT().SearchStream(mock.Anything).RunAndReturn(func(ctx context.Context, _ ...grpc.CallOption) (querypb.QueryNode_SearchStreamClient, error) {
		rpcCtx = ctx
		return &testLeaderSearchStream{ctx: ctx, chunks: []*internalpb.SearchResults{
			{Status: merr.Status(merr.WrapErrCollectionNotLoaded(42))},
		}}, nil
	}).Once()
	mgr.EXPECT().InvalidateShardLeaderCache([]int64{42}).Once()

	stream, err := task.openSearchStream(context.Background(), qn, 1, "a", 1)
	require.Nil(t, stream)
	require.ErrorIs(t, err, merr.ErrCollectionNotLoaded)
	require.ErrorIs(t, rpcCtx.Err(), context.Canceled)
}

func TestSearchStreamFallsBackToUnaryBeforeFirstChunk(t *testing.T) {
	qn := mocks.NewMockQueryNodeClient(t)
	lb := shardclient.NewMockLBPolicy(t)
	task := &SearchTask{
		SearchRequest: &internalpb.SearchRequest{Base: &commonpb.MsgBase{}},
		lb:            lb, queryChannelsNode: typeutil.NewConcurrentMap[string, int64](),
	}
	var rpcCtx context.Context
	qn.EXPECT().SearchStream(mock.Anything).RunAndReturn(func(ctx context.Context, _ ...grpc.CallOption) (querypb.QueryNode_SearchStreamClient, error) {
		rpcCtx = ctx
		return &testLeaderSearchStream{ctx: ctx, err: status.Error(codes.Unimplemented, "old leader")}, nil
	}).Once()
	batch := &internalpb.SearchResults{Status: merr.Success(), NumQueries: 1}
	qn.EXPECT().Search(mock.Anything, mock.Anything).Return(batch, nil).Once()
	lb.EXPECT().UpdateCostMetrics(int64(1), mock.Anything).Once()

	stream, err := task.openSearchStream(context.Background(), qn, 1, "a", 1)
	require.NoError(t, err)
	defer stream.Close()
	require.ErrorIs(t, rpcCtx.Err(), context.Canceled)
	result, err := stream.Recv()
	require.NoError(t, err)
	require.Same(t, batch, result)
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
}
