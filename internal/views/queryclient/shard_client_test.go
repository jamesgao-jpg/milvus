package queryclient

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	commonpb "github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/views/queryclient/resolver"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestShardSearchReturnsQueryPlanMVCCForRequery(t *testing.T) {
	shardID := qviews.ShardID{ReplicaID: 1, VChannel: "by-dev-rootcoord-dml_0_100v0"}
	mvcc := &viewpb.QueryPlanMVCC{
		GrowingTimetick:      100,
		TransformingTimetick: 90,
	}
	version := &viewpb.QueryViewVersion{
		QueryVersion: 10,
	}
	queryNode := qviews.NewQueryNode(11)
	planClient := &fakeQueryPlanClient{
		plan: &viewpb.QueryPlan{
			ShardId: shardID.IntoProto(),
			Version: version,
			Mvcc:    mvcc,
			Request: &viewpb.QueryPlan_LegacySearchRequest{
				LegacySearchRequest: &internalpb.SearchRequest{},
			},
			WorkNodes: []*viewpb.QueryPlanWorkNode{
				{
					Node: &viewpb.QueryPlanWorkNode_QueryNode{
						QueryNode: &viewpb.QueryWorkNode{NodeId: queryNode.ID},
					},
				},
			},
		},
	}
	queryService := &fakeViewQueryServiceClient{}
	client := newShardViewQueryClient(
		1,
		planClient,
		queryService,
		&fakeShardResolver{replicas: &resolver.ShardReplicas{
			VChannel:       shardID.VChannel,
			PrimaryShardID: shardID,
			ShardIDs:       []qviews.ShardID{shardID},
		}},
		fixedReplicaPicker{shardID: shardID},
	)

	shardPlan, err := client.Search(context.Background(), &ShardSearchRequest{
		VChannel: shardID.VChannel,
		Req: &internalpb.SearchRequest{
			CollectionID:     100,
			ConsistencyLevel: commonpb.ConsistencyLevel_Bounded,
			Nq:               1,
			Topk:             1,
			MetricType:       "IP",
		},
		Reducer: fakeSearchResultReducer{},
	})
	require.NoError(t, err)

	require.True(t, proto.Equal(mvcc, shardPlan.Mvcc))
	require.True(t, proto.Equal(mvcc, queryService.searchReq.GetMvcc()))
	require.Equal(t, mvcc.GetTransformingTimetick(), queryService.searchReq.GetLegacyReq().GetMvccTimestamp())
}

func TestSessionSearchOnPrimaryLetsSNGenerateQueryPlanMVCC(t *testing.T) {
	shardID := qviews.ShardID{ReplicaID: 1, VChannel: "by-dev-rootcoord-dml_0_100v0"}
	planClient := &fakeQueryPlanClient{
		plan: newTestSearchQueryPlan(shardID, &viewpb.QueryPlanMVCC{
			GrowingTimetick:      100,
			TransformingTimetick: 90,
		}),
	}
	queryService := &fakeViewQueryServiceClient{}
	client := newShardViewQueryClient(
		1,
		planClient,
		queryService,
		&fakeShardResolver{replicas: &resolver.ShardReplicas{
			VChannel:       shardID.VChannel,
			PrimaryShardID: shardID,
			ShardIDs:       []qviews.ShardID{shardID},
		}},
		fixedReplicaPicker{shardID: shardID},
	)

	_, err := client.Search(context.Background(), &ShardSearchRequest{
		VChannel: shardID.VChannel,
		Req: &internalpb.SearchRequest{
			CollectionID:       100,
			ConsistencyLevel:   commonpb.ConsistencyLevel_Session,
			GuaranteeTimestamp: 999,
			Nq:                 1,
			Topk:               1,
			MetricType:         "IP",
		},
		Reducer: fakeSearchResultReducer{},
	})
	require.NoError(t, err)

	require.Equal(t, commonpb.ConsistencyLevel_Strong, planClient.planReq.GetConsistencyLevel())
	require.Nil(t, planClient.planReq.GetQueryPlanMvcc())
	require.Equal(t, 0, planClient.mvccReqCount)
}

func TestSessionSearchOnSecondaryUsesPrimaryWALMVCC(t *testing.T) {
	vchannel := "by-dev-rootcoord-dml_0_100v0"
	primaryShardID := qviews.ShardID{ReplicaID: 1, VChannel: vchannel}
	secondaryShardID := qviews.ShardID{ReplicaID: 2, VChannel: vchannel}
	mvcc := &viewpb.QueryPlanMVCC{
		GrowingTimetick:      100,
		TransformingTimetick: 90,
	}
	planClient := &fakeQueryPlanClient{plan: newTestSearchQueryPlan(secondaryShardID, mvcc)}
	queryService := &fakeViewQueryServiceClient{}
	client := newShardViewQueryClient(
		1,
		planClient,
		queryService,
		&fakeShardResolver{replicas: &resolver.ShardReplicas{
			VChannel:       vchannel,
			PrimaryShardID: primaryShardID,
			ShardIDs:       []qviews.ShardID{primaryShardID, secondaryShardID},
		}},
		fixedReplicaPicker{shardID: secondaryShardID},
	)

	_, err := client.Search(context.Background(), &ShardSearchRequest{
		VChannel: vchannel,
		Req: &internalpb.SearchRequest{
			CollectionID:       100,
			ConsistencyLevel:   commonpb.ConsistencyLevel_Session,
			GuaranteeTimestamp: 999,
			Nq:                 1,
			Topk:               1,
			MetricType:         "IP",
		},
		Reducer: fakeSearchResultReducer{},
	})
	require.NoError(t, err)

	require.Equal(t, 1, planClient.mvccReqCount)
	require.Equal(t, primaryShardID, planClient.mvccShardID)
	require.Equal(t, vchannel, planClient.mvccReq.GetVchannel())
	require.True(t, proto.Equal(mvcc, planClient.planReq.GetQueryPlanMvcc()))
	require.Equal(t, commonpb.ConsistencyLevel(0), planClient.planReq.GetConsistencyLevel())
}

func TestShardSearchStreamReturnsPerVChannelReduceStream(t *testing.T) {
	shardID := qviews.ShardID{ReplicaID: 1, VChannel: "by-dev-rootcoord-dml_0_100v0"}
	leftNode := qviews.NewQueryNode(11)
	rightNode := qviews.NewQueryNode(12)
	request := newTestSearchRequest(3)
	plan := newTestSearchQueryPlan(shardID, &viewpb.QueryPlanMVCC{})
	plan.Request = &viewpb.QueryPlan_LegacySearchRequest{LegacySearchRequest: proto.Clone(request).(*internalpb.SearchRequest)}
	plan.WorkNodes = []*viewpb.QueryPlanWorkNode{
		{Node: &viewpb.QueryPlanWorkNode_QueryNode{QueryNode: &viewpb.QueryWorkNode{NodeId: leftNode.ID}}},
		{Node: &viewpb.QueryPlanWorkNode_QueryNode{QueryNode: &viewpb.QueryWorkNode{NodeId: rightNode.ID}}},
	}

	leftStream := &fakeSearchStream{recv: []fakeSearchStreamRecv{{chunk: newTestSearchChunk(3, []int64{1, 4}, []float32{0.9, 0.6})}}}
	rightStream := &fakeSearchStream{recv: []fakeSearchStreamRecv{{chunk: newTestSearchChunk(3, []int64{2, 3}, []float32{0.8, 0.7})}}}
	queryService := &fakeViewQueryServiceClient{
		searchOnViewStream: func(_ context.Context, node qviews.WorkNode, _ *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error) {
			switch node.Key() {
			case leftNode.Key():
				return leftStream, nil
			case rightNode.Key():
				return rightStream, nil
			default:
				return nil, errors.New("unexpected work node")
			}
		},
	}
	client := newTestShardClient(1, shardID, plan, queryService)

	stream, shardPlan, err := client.SearchStream(context.Background(), shardID.VChannel, request, 2, nil)

	require.NoError(t, err)
	require.Equal(t, shardID, shardPlan.ShardID)
	require.Equal(t, int64(2), queryService.searchReq.GetStreamChunkSize())
	chunk, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1, 2}, chunk.GetResultData().GetIds().GetIntId().GetData())
	require.Equal(t, []float32{0.9, 0.8}, chunk.GetResultData().GetScores())
	chunk, err = stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{3}, chunk.GetResultData().GetIds().GetIntId().GetData())
	require.Equal(t, []float32{0.7}, chunk.GetResultData().GetScores())
	chunk, err = stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)
	require.Equal(t, 1, leftStream.closeCount())
	require.Equal(t, 1, rightStream.closeCount())
}

func TestShardSearchStreamCloseCompletesReplicaPicker(t *testing.T) {
	shardID := qviews.ShardID{ReplicaID: 1, VChannel: "by-dev-rootcoord-dml_0_100v0"}
	request := newTestSearchRequest(1)
	plan := newTestSearchQueryPlan(shardID, &viewpb.QueryPlanMVCC{})
	plan.Request = &viewpb.QueryPlan_LegacySearchRequest{LegacySearchRequest: proto.Clone(request).(*internalpb.SearchRequest)}
	childStream := &fakeSearchStream{}
	done := make(chan ReplicaDoneInfo, 1)
	client := newShardViewQueryClient(
		1,
		&fakeQueryPlanClient{plan: plan},
		&fakeViewQueryServiceClient{
			searchOnViewStream: func(context.Context, qviews.WorkNode, *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error) {
				return childStream, nil
			},
		},
		&fakeShardResolver{replicas: &resolver.ShardReplicas{
			VChannel:       shardID.VChannel,
			PrimaryShardID: shardID,
			ShardIDs:       []qviews.ShardID{shardID},
		}},
		fixedReplicaPicker{
			shardID: shardID,
			done: func(info ReplicaDoneInfo) {
				done <- info
			},
		},
	)

	stream, _, err := client.SearchStream(context.Background(), shardID.VChannel, request, defaultSearchStreamChunkSize, nil)
	require.NoError(t, err)
	require.NoError(t, stream.Close())

	info := <-done
	require.NoError(t, info.Err)
	require.Equal(t, 1, childStream.closeCount())
}

func TestShardSearchClosesOpenedStreamsOnSetupFailure(t *testing.T) {
	shardID := qviews.ShardID{ReplicaID: 1, VChannel: "by-dev-rootcoord-dml_0_100v0"}
	leftNode := qviews.NewQueryNode(11)
	rightNode := qviews.NewQueryNode(12)
	request := newTestSearchRequest(1)
	plan := newTestSearchQueryPlan(shardID, &viewpb.QueryPlanMVCC{})
	plan.Request = &viewpb.QueryPlan_LegacySearchRequest{LegacySearchRequest: proto.Clone(request).(*internalpb.SearchRequest)}
	plan.WorkNodes = []*viewpb.QueryPlanWorkNode{
		{Node: &viewpb.QueryPlanWorkNode_QueryNode{QueryNode: &viewpb.QueryWorkNode{NodeId: leftNode.ID}}},
		{Node: &viewpb.QueryPlanWorkNode_QueryNode{QueryNode: &viewpb.QueryWorkNode{NodeId: rightNode.ID}}},
	}

	openedStream := &fakeSearchStream{}
	queryService := &fakeViewQueryServiceClient{
		searchOnViewStream: func(_ context.Context, node qviews.WorkNode, _ *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error) {
			if node.Key() == leftNode.Key() {
				return openedStream, nil
			}
			return nil, errors.New("open failed")
		},
	}
	client := newTestShardClient(1, shardID, plan, queryService)

	_, _, err := client.SearchStream(context.Background(), shardID.VChannel, request, defaultSearchStreamChunkSize, nil)

	require.ErrorContains(t, err, "open failed")
	require.Equal(t, 1, openedStream.closeCount())
}

func TestShardSearchUsesBatchPathForUnsupportedSearch(t *testing.T) {
	shardID := qviews.ShardID{ReplicaID: 1, VChannel: "by-dev-rootcoord-dml_0_100v0"}
	request := newTestSearchRequest(1)
	request.GroupByFieldId = 100
	plan := newTestSearchQueryPlan(shardID, &viewpb.QueryPlanMVCC{})
	plan.Request = &viewpb.QueryPlan_LegacySearchRequest{LegacySearchRequest: proto.Clone(request).(*internalpb.SearchRequest)}
	batchResult := newTestSearchChunk(1, []int64{10}, []float32{0.9})
	queryService := &fakeViewQueryServiceClient{
		searchResponse: &viewpb.SearchOnViewResponse{LegacyResults: batchResult},
		searchOnViewStream: func(context.Context, qviews.WorkNode, *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error) {
			return nil, errors.New("stream path should not be used")
		},
	}
	reducer := &recordingSearchResultReducer{}
	client := newTestShardClient(1, shardID, plan, queryService)

	_, err := client.Search(context.Background(), &ShardSearchRequest{
		VChannel: shardID.VChannel,
		Req:      request,
		Reducer:  reducer,
	})

	require.NoError(t, err)
	require.Len(t, reducer.results, 1)
	require.Same(t, batchResult, reducer.results[0])
	require.Equal(t, 1, queryService.searchCalls)
	require.Zero(t, queryService.searchStreamCalls)
}

type fakeShardResolver struct {
	replicas *resolver.ShardReplicas
}

func (f *fakeShardResolver) ResolveVChannels(context.Context, int64) ([]string, error) {
	return []string{f.replicas.VChannel}, nil
}

func (f *fakeShardResolver) ResolveShard(context.Context, int64, string) (*resolver.ShardReplicas, error) {
	return f.replicas, nil
}

type fixedReplicaPicker struct {
	shardID qviews.ShardID
	done    func(ReplicaDoneInfo)
}

func (p fixedReplicaPicker) Pick(context.Context, ReplicaPickInfo) (ReplicaPickResult, error) {
	return ReplicaPickResult{ShardID: p.shardID, Done: p.done}, nil
}

type fakeQueryPlanClient struct {
	plan         *viewpb.QueryPlan
	planReq      *viewpb.GetQueryPlanRequest
	mvccReq      *viewpb.GetMVCCTimestampRequest
	mvccShardID  qviews.ShardID
	mvccReqCount int
}

func (f *fakeQueryPlanClient) GetQueryPlan(_ context.Context, _ qviews.ShardID, req *viewpb.GetQueryPlanRequest) (*viewpb.GetQueryPlanResponse, error) {
	f.planReq = proto.Clone(req).(*viewpb.GetQueryPlanRequest)
	return &viewpb.GetQueryPlanResponse{Plan: f.plan}, nil
}

func (f *fakeQueryPlanClient) GetMVCCTimestamp(_ context.Context, shardID qviews.ShardID, req *viewpb.GetMVCCTimestampRequest) (*viewpb.GetMVCCTimestampResponse, error) {
	f.mvccReq = proto.Clone(req).(*viewpb.GetMVCCTimestampRequest)
	f.mvccShardID = shardID
	f.mvccReqCount++
	return &viewpb.GetMVCCTimestampResponse{Mvcc: f.plan.GetMvcc()}, nil
}

type fakeViewQueryServiceClient struct {
	searchReq          *viewpb.SearchOnViewRequest
	searchResponse     *viewpb.SearchOnViewResponse
	searchOnViewStream func(context.Context, qviews.WorkNode, *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error)
	searchCalls        int
	searchStreamCalls  int
}

func (f *fakeViewQueryServiceClient) SearchOnView(_ context.Context, _ qviews.WorkNode, req *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error) {
	f.searchReq = req
	f.searchCalls++
	if f.searchResponse != nil {
		return f.searchResponse, nil
	}
	return &viewpb.SearchOnViewResponse{}, nil

}

func (f *fakeViewQueryServiceClient) SearchOnViewStream(ctx context.Context, node qviews.WorkNode, req *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error) {
	f.searchReq = req
	f.searchStreamCalls++
	if f.searchOnViewStream != nil {
		return f.searchOnViewStream(ctx, node, req)
	}
	return &fakeSearchStream{}, nil
}

func (f *fakeViewQueryServiceClient) QueryOnView(context.Context, qviews.WorkNode, *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error) {
	return &viewpb.QueryOnViewResponse{}, nil
}

func (f *fakeViewQueryServiceClient) RequeryOnView(context.Context, qviews.WorkNode, *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error) {
	return &viewpb.RequeryOnViewResponse{}, nil
}

type fakeSearchResultReducer struct{}

func (fakeSearchResultReducer) Add(qviews.ShardID, *viewpb.SearchOnViewResponse) error {
	return nil
}

func (fakeSearchResultReducer) ResetShard(qviews.ShardID) {}

func (fakeSearchResultReducer) Finish() (*internalpb.SearchResults, error) {
	return &internalpb.SearchResults{}, nil
}

type recordingSearchResultReducer struct {
	results []*internalpb.SearchResults
}

func (r *recordingSearchResultReducer) Add(_ qviews.ShardID, response *viewpb.SearchOnViewResponse) error {
	r.results = append(r.results, response.GetLegacyResults())
	return nil
}

func (*recordingSearchResultReducer) ResetShard(qviews.ShardID) {}

func (*recordingSearchResultReducer) Finish() (*internalpb.SearchResults, error) {
	return &internalpb.SearchResults{}, nil
}

type fakeSearchStreamRecv struct {
	chunk *internalpb.SearchResults
	err   error
}

type fakeSearchStream struct {
	mu         sync.Mutex
	ctx        context.Context
	recv       []fakeSearchStreamRecv
	closeCalls int
}

func (s *fakeSearchStream) Recv() (*internalpb.SearchResults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx != nil {
		select {
		case <-s.ctx.Done():
			return nil, s.ctx.Err()
		default:
		}
	}
	if len(s.recv) == 0 {
		return nil, io.EOF
	}
	next := s.recv[0]
	s.recv = s.recv[1:]
	return next.chunk, next.err
}

func (s *fakeSearchStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeCalls++
	return nil
}

func (*fakeSearchStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, errors.New("not implemented")
}

func (s *fakeSearchStream) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeCalls
}

func newTestShardClient(maxRetries int, shardID qviews.ShardID, plan *viewpb.QueryPlan, queryService ViewQueryServiceClient) *shardViewQueryClient {
	return newShardViewQueryClient(
		maxRetries,
		&fakeQueryPlanClient{plan: plan},
		queryService,
		&fakeShardResolver{replicas: &resolver.ShardReplicas{
			VChannel:       shardID.VChannel,
			PrimaryShardID: shardID,
			ShardIDs:       []qviews.ShardID{shardID},
		}},
		fixedReplicaPicker{shardID: shardID},
	)
}

func newTestSearchRequest(topK int) *internalpb.SearchRequest {
	return &internalpb.SearchRequest{
		CollectionID:     100,
		ConsistencyLevel: commonpb.ConsistencyLevel_Bounded,
		Nq:               1,
		Topk:             int64(topK),
		MetricType:       "IP",
		IsIterator:       true,
	}
}

func newTestSearchChunk(topK int64, ids []int64, scores []float32) *internalpb.SearchResults {
	return &internalpb.SearchResults{
		Status:     merr.Success(),
		MetricType: "IP",
		NumQueries: 1,
		TopK:       topK,
		ResultData: &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       topK,
			Topks:      []int64{int64(len(ids))},
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}},
			},
			Scores: scores,
		},
	}
}

func newTestSearchQueryPlan(shardID qviews.ShardID, mvcc *viewpb.QueryPlanMVCC) *viewpb.QueryPlan {
	queryNode := qviews.NewQueryNode(11)
	return &viewpb.QueryPlan{
		ShardId: shardID.IntoProto(),
		Version: &viewpb.QueryViewVersion{QueryVersion: 10},
		Mvcc:    mvcc,
		Request: &viewpb.QueryPlan_LegacySearchRequest{
			LegacySearchRequest: &internalpb.SearchRequest{},
		},
		WorkNodes: []*viewpb.QueryPlanWorkNode{
			{
				Node: &viewpb.QueryPlanWorkNode_QueryNode{
					QueryNode: &viewpb.QueryWorkNode{NodeId: queryNode.ID},
				},
			},
		},
	}
}
