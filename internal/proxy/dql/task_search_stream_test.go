package dql

import (
	"context"
	"fmt"
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

func TestSearchReduceStreamPartialResultUsesBatch(t *testing.T) {
	paramtable.Init()
	tables := paramtable.Get()
	key := tables.QueryNodeCfg.PartialResultRequiredDataRatio.Key
	t.Cleanup(func() { tables.Reset(key) })
	task := &SearchTask{SearchRequest: &internalpb.SearchRequest{SearchType: internalpb.SearchType_PURE_ANN_SEARCH_NO_FILTER}}
	tables.Save(key, "1")
	require.True(t, task.supportsStreamingReduce())
	tables.Save(key, "0.9")
	require.False(t, task.supportsStreamingReduce())
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
	t.Run("without offset", func(t *testing.T) { testSearchTaskReduceStreamBatchParity(t, 0, false) })
	t.Run("with offset", func(t *testing.T) { testSearchTaskReduceStreamBatchParity(t, 1, false) })
	t.Run("grouped with offset", func(t *testing.T) { testSearchTaskReduceStreamBatchParity(t, 1, true) })
}

func TestConsumeSearchStreamSingleVChannelQuota(t *testing.T) {
	quota := &paramtable.Get().QuotaConfig.MaxOutputSize
	for _, limit := range []int64{16, 15} {
		t.Run(fmt.Sprintf("quota%d", limit), func(t *testing.T) {
			defer quota.SwapTempValue(quota.SwapTempValue(fmt.Sprint(limit)))
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			leader := &testLeaderSearchStream{ctx: ctx}
			for _, id := range []int64{1, 2} {
				data := genSearchResultData(1, 2, []int64{id}, []float32{-float32(id)})
				data.Topks[0] = 1
				data.FieldsData = []*schemapb.FieldData{getFieldData("value", 100, schemapb.DataType_Int64, []int64{id}, 1)}
				leader.chunks = append(leader.chunks, &internalpb.SearchResults{Status: merr.Success(), MetricType: "L2", ResultData: data})
			}
			inputs := []searchreducestream.ReduceStream{&grpcSearchStream{rpc: leader, cancel: cancel}}
			stream := searchreducestream.NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 2}, inputs, schemapb.DataType_Int64, quota.GetAsInt64(), 1)
			result, err := consumeSearchResultStream(ctx, stream)
			if err == nil {
				final, reduceErr := reduceResults(t.Context(), []*internalpb.SearchResults{result}, 1, 2, 0, "L2", schemapb.DataType_Int64, &planpb.QueryInfo{}, false, false, 0, nil, false)
				require.NoError(t, reduceErr)
				require.Equal(t, []int64{1, 2}, final.GetResults().GetFieldsData()[0].GetScalars().GetLongData().GetData())
			}
			if limit < 16 {
				require.ErrorIs(t, err, merr.ErrParameterInvalid)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func testSearchTaskReduceStreamBatchParity(t *testing.T, offset int64, grouped bool) {
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
		Base: &commonpb.MsgBase{}, Nq: 2, Topk: 2, Offset: offset,
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
	info := &planpb.QueryInfo{}
	if grouped {
		req.GroupByFieldId, req.GroupSize = 101, 2
		info.GroupByFieldId, info.GroupSize = req.GroupByFieldId, req.GroupSize
		// The first vector has three distinct groups, the second reuses NULL.
		// Both vchannels contribute to the same group across Chunk boundaries.
		for i, values := range [][]any{{int64(7), nil, nil, int64(8)}, {int64(7), int64(8), nil}} {
			b, err := typeutil.NewFieldDataBuilder(schemapb.DataType_Int64, false, len(values))
			require.NoError(t, err)
			for _, v := range values {
				b.Add(v)
			}
			batch[i].ResultData.GroupByFieldValue = b.Build()
		}
	}
	for i, result := range batch {
		result.ChannelsMvcc = map[string]uint64{[]string{"a", "b"}[i]: uint64(i + 1)}
	}
	batchResult, err := reduceResults(ctx, batch, 2, 2, offset, "L2", schemapb.DataType_Int64, info, false, false, 0, nil, false)
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
	if grouped {
		require.Equal(t, []int64{3, 3}, chunks[0].GetResultData().GetTopks(), "keep the offset group but discard the third group")
	}
	streamResult, err := reduceResults(ctx, chunks, 2, 2, offset, "L2", schemapb.DataType_Int64, info, false, false, 0, nil, false)
	require.NoError(t, err)
	require.True(t, proto.Equal(batchResult, streamResult), "batch: %v; stream: %v", batchResult, streamResult)
	for _, stream := range opened {
		require.ErrorIs(t, stream.ctx.Err(), context.Canceled)
	}
}

func TestHybridSearchStreamReportsCostFromLeaderSummary(t *testing.T) {
	qn := mocks.NewMockQueryNodeClient(t)
	lb := shardclient.NewMockLBPolicy(t)
	req := &internalpb.SearchRequest{Base: &commonpb.MsgBase{}, IsAdvanced: true, SubReqs: []*internalpb.SubSearchRequest{{Nq: 1, Topk: 2}}}
	task := &SearchTask{SearchRequest: req, lb: lb, queryChannelsNode: typeutil.NewConcurrentMap[string, int64]()}
	cost := &internalpb.CostAggregation{TotalRelatedDataSize: 37}
	qn.EXPECT().SearchStream(mock.Anything).RunAndReturn(func(ctx context.Context, _ ...grpc.CallOption) (querypb.QueryNode_SearchStreamClient, error) {
		return &testLeaderSearchStream{ctx: ctx, chunks: []*internalpb.SearchResults{
			{IsAdvanced: true, SubResults: []*internalpb.SubSearchResults{{StreamEnd: proto.Bool(true)}}}, {IsAdvanced: true, CostAggregation: cost},
		}, send: func(open *querypb.SearchStreamRequest) {
			require.Equal(t, open.ChunkBytes, open.Request.StreamChunkBytes)
		}}, nil
	}).Once()
	lb.EXPECT().UpdateCostMetrics(int64(1), cost).Once()
	stream, err := task.openSearchStream(t.Context(), qn, 1, "a", 1)
	require.NoError(t, err)
	defer stream.Close()
	for range 2 {
		_, err = stream.Recv()
		require.NoError(t, err)
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

func TestHybridGroupReduceStreamBatchParity(t *testing.T) {
	paramtable.Init()
	ctx := t.Context()
	const groupFieldID = 101
	op := hybridSearchReduceOperator{
		traceCtx: ctx, primaryFieldSchema: &schemapb.FieldSchema{DataType: schemapb.DataType_Int64},
		subReqs: []*internalpb.SubSearchRequest{
			{Nq: 2, Topk: 2, Offset: 1}, {Nq: 2, Topk: 2, Offset: 1},
		},
		queryInfos: []*planpb.QueryInfo{{GroupByFieldId: groupFieldID, GroupSize: 2}, {}},
	}
	batch := make([]*internalpb.SearchResults, 2)
	streams := make([]searchreducestream.ReduceStream, 2)
	for i := range batch {
		offset := int64(i * 10)
		grouped := genSearchResultData(2, 2, []int64{1 + offset, 2 + offset, 3 + offset, 4 + offset}, []float32{-1, -1, -3, -2})
		grouped.Topks = []int64{3, 1}
		grouped.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_Int64, ValidData: []bool{true, false, true, true},
			Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
				Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{7, 7, 8}}},
			}},
		}
		plain := genSearchResultData(2, 2, []int64{1 + offset, 2 + offset, 3 + offset, 4 + offset}, []float32{1, 0.5, 1, 0.5})
		plain.Topks = []int64{2, 2}
		batch[i] = &internalpb.SearchResults{
			Status: merr.Success(), IsAdvanced: true, NumQueries: 2,
			ChannelsMvcc: map[string]uint64{"a": 1},
			SubResults: []*internalpb.SubSearchResults{
				{ReqIndex: 0, NumQueries: 2, TopK: 2, MetricType: "L2", ResultData: grouped},
				{ReqIndex: 1, NumQueries: 2, TopK: 2, MetricType: "IP", ResultData: plain},
			},
		}
		cancelCtx, cancel := context.WithCancel(ctx)
		rpc := &testLeaderSearchStream{ctx: cancelCtx}
		require.NoError(t, searchreducestream.SplitHybridSearchResult(proto.CloneOf(batch[i]), 1, func(chunk *internalpb.SearchResults) error {
			rpc.chunks = append(rpc.chunks, chunk)
			return nil
		}))
		require.Len(t, rpc.chunks, 8)
		streams[i] = &grpcSearchStream{rpc: rpc, cancel: cancel}
	}
	drained, err := searchreducestream.DrainSearchStreams(ctx, streams)
	require.NoError(t, err)
	require.Len(t, drained, 2)
	require.Equal(t, map[string]uint64{"a": 1}, drained[0].GetChannelsMvcc())
	want, err := op.run(ctx, nil, batch)
	require.NoError(t, err)
	got, err := op.run(ctx, nil, drained)
	require.NoError(t, err)
	wantResults, gotResults := want[0].([]*milvuspb.SearchResults), got[0].([]*milvuspb.SearchResults)
	require.Len(t, gotResults, len(wantResults))
	for i, expected := range wantResults {
		require.True(t, proto.Equal(expected, gotResults[i]), "sub-search %d: Batch %v; stream %v", i, expected, gotResults[i])
	}
	require.Equal(t, []int64{6, 2}, gotResults[0].GetResults().GetTopks(), "grouped candidates must not be clipped before fusion")
}
