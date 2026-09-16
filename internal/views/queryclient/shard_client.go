package queryclient

import (
	"context"
	"errors"
	"io"
	"time"

	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"

	commonpb "github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/views/queryclient/reducer"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/internal/views/viewerror"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const defaultSearchStreamChunkSize = 1024

// shardViewQueryClient executes two-phase queries at the shard granularity.
// It owns replica resolution, consistency routing, Phase 1 (GetQueryPlan),
// Phase 2 (SearchOnView/QueryOnView) dispatch, and shard-level retry.
//
// Search and Query use the existing batch executeShard path. SearchStream opens
// the request-scoped per-vchannel ReduceStream used by iterator Search.
type shardViewQueryClient struct {
	maxRetries         int
	queryPlanClient    QueryPlanClient
	queryServiceClient ViewQueryServiceClient
}

func newShardViewQueryClient(
	maxRetries int,
	queryPlanClient QueryPlanClient,
	queryServiceClient ViewQueryServiceClient,
) *shardViewQueryClient {
	return &shardViewQueryClient{
		maxRetries:         maxRetries,
		queryPlanClient:    queryPlanClient,
		queryServiceClient: queryServiceClient,
	}
}

// ShardSearchRequest contains the parameters for a shard-level search execution.
type ShardSearchRequest struct {
	VChannel string
	Req      *internalpb.SearchRequest
	Reducer  reducer.SearchResultReducer
}

// ShardQueryRequest contains the parameters for a shard-level query (retrieve) execution.
type ShardQueryRequest struct {
	VChannel string
	Req      *internalpb.RetrieveRequest
	Reducer  reducer.RetrieveResultReducer
}

// Search executes Phase 1 + Phase 2 for a single shard's search.
// Replica resolution is handled internally. Results are fed into the provided reducer.
// Returns the ShardPlan for potential requery.
func (s *shardViewQueryClient) Search(ctx context.Context, req *ShardSearchRequest) (*ShardPlan, error) {
	return s.executeShard(ctx, req.VChannel, &shardExecParams{
		consistencyLevel: req.Req.ConsistencyLevel,
		buildPlanReq: func(targetShardID qviews.ShardID) *viewpb.GetQueryPlanRequest {
			return &viewpb.GetQueryPlanRequest{
				CollectionId: req.Req.CollectionID,
				ShardId:      targetShardID.IntoProto(),
				PartitionIds: req.Req.PartitionIDs,
				Request: &viewpb.GetQueryPlanRequest_LegacySearchRequest{
					LegacySearchRequest: req.Req,
				},
			}
		},
		dispatchNode: func(ctx context.Context, node qviews.WorkNode, plan *viewpb.QueryPlan, shardID qviews.ShardID) error {
			rpcCtx, metrics := withSearchBenchmarkChild(ctx, node, shardID.VChannel, "batch")
			resp, err := s.queryServiceClient.SearchOnView(rpcCtx, node, &viewpb.SearchOnViewRequest{
				LegacyReq: legacySearchRequestForNode(plan, node),
				ShardId:   shardID.IntoProto(),
				Version:   plan.Version,
				Mvcc:      plan.GetMvcc(),
			})
			if err != nil {
				return err
			}
			metrics.RecordApplicationReceive(resp, resp.GetLegacyResults())
			return req.Reducer.Add(shardID, resp)
		},
		resetShard: req.Reducer.ResetShard,
	})
}

type vchannelReduceStream struct {
	stream  searchutil.ReduceStream
	metrics *searchutil.SearchBenchmarkMetrics
}

func (s *vchannelReduceStream) Recv() (*internalpb.SearchResults, error) {
	startedAt := time.Now()
	chunk, err := s.stream.Recv()
	s.metrics.AddPerVchannelReduceDuration(time.Since(startedAt))
	if err == nil {
		return chunk, nil
	}
	if errors.Is(err, io.EOF) {
		if closeErr := s.stream.Close(); closeErr != nil {
			return nil, closeErr
		}
		return nil, io.EOF
	}

	return nil, errors.Join(err, s.stream.Close())
}

func (s *vchannelReduceStream) Close() error {
	return s.stream.Close()
}

func (s *vchannelReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	metadata, err := s.stream.Interrupt()
	if err != nil {
		return nil, errors.Join(err, s.stream.Close())
	}
	return metadata, nil
}

// SearchStream opens the SN/QN child streams for one vchannel and returns the
// request-scoped ReduceStream without consuming its output.
func (s *shardViewQueryClient) SearchStream(
	ctx context.Context,
	vchannel string,
	req *internalpb.SearchRequest,
	chunkSize int,
) (searchutil.ReduceStream, *ShardPlan, error) {
	if req == nil {
		return nil, nil, errors.New("SearchStream requires a Search request")
	}

	var lastErr error
	for attempt := 0; attempt < s.maxRetries; attempt++ {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}

		targetShardID := qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: vchannel}

		planReq := &viewpb.GetQueryPlanRequest{
			CollectionId: req.GetCollectionID(),
			ShardId:      targetShardID.IntoProto(),
			PartitionIds: req.GetPartitionIDs(),
			Request: &viewpb.GetQueryPlanRequest_LegacySearchRequest{
				LegacySearchRequest: req,
			},
		}
		plan, err := s.executeGetQueryPlan(ctx, targetShardID, planReq, &shardExecParams{
			consistencyLevel: req.GetConsistencyLevel(),
		})
		if err != nil {
			if ve := viewerror.AsViewError(err); ve != nil && ve.IsRetryable() {
				lastErr = err
				continue
			}
			return nil, nil, err
		}

		shardID := qviews.FromProtoShardID(plan.ShardId)
		workNodes := workNodesFromPlan(plan)
		if mlog.LevelEnabled(mlog.DebugLevel) {
			queryNodeIDs := make([]int64, 0, len(workNodes))
			streamingNodePresent := false
			for _, workNode := range workNodes {
				switch node := workNode.(type) {
				case qviews.QueryNode:
					queryNodeIDs = append(queryNodeIDs, node.ID)
				case qviews.StreamingNode:
					streamingNodePresent = true
				}
			}
			mlog.Debug(ctx, "query view work nodes selected",
				mlog.FieldVChannel(vchannel),
				mlog.Int64("replicaID", shardID.ReplicaID),
				mlog.Int64s("queryNodeIDs", queryNodeIDs),
				mlog.Bool("streamingNodePresent", streamingNodePresent),
				mlog.Int("workNodeCount", len(workNodes)),
			)
		}
		childStreams := make([]searchutil.ReduceStream, 0, len(workNodes))
		for _, node := range workNodes {
			rpcCtx, _ := withSearchBenchmarkChild(ctx, node, vchannel, "streaming")
			childStream, openErr := s.queryServiceClient.SearchOnViewStream(rpcCtx, node, &viewpb.SearchOnViewRequest{
				LegacyReq:       legacySearchRequestForNode(plan, node),
				ShardId:         shardID.IntoProto(),
				Version:         plan.Version,
				Mvcc:            plan.GetMvcc(),
				StreamChunkSize: int64(chunkSize),
			})
			if openErr != nil {
				err = openErr
				break
			}
			if childStream == nil {
				err = merr.WrapErrServiceInternalMsg("SearchOnViewStream returned a nil stream for work node %s", node.String())
				break
			}
			childStreams = append(childStreams, childStream)
		}
		if err != nil {
			for _, childStream := range childStreams {
				err = errors.Join(err, childStream.Close())
			}
			if ctx.Err() != nil {
				return nil, nil, ctx.Err()
			}
			if ve := viewerror.AsViewError(err); ve != nil && ve.IsRetryable() {
				lastErr = err
				continue
			}
			return nil, nil, err
		}

		reducedStream, err := searchutil.NewReduceStream(req, childStreams, chunkSize)
		if err != nil {
			for _, childStream := range childStreams {
				err = errors.Join(err, childStream.Close())
			}
			return nil, nil, err
		}

		return &vchannelReduceStream{
				stream:  reducedStream,
				metrics: searchutil.SearchBenchmarkMetricsFromContext(ctx),
			}, &ShardPlan{
				ShardID:   shardID,
				Version:   plan.Version,
				Mvcc:      plan.GetMvcc(),
				WorkNodes: workNodes,
			}, nil
	}
	return nil, nil, lastErr
}

func withSearchBenchmarkChild(ctx context.Context, node qviews.WorkNode, vchannel, mode string) (context.Context, *searchutil.SearchBenchmarkMetrics) {
	parent := searchutil.SearchBenchmarkMetricsFromContext(ctx)
	if parent == nil {
		return ctx, nil
	}
	nodeID := int64(0)
	if queryNode, ok := node.(qviews.QueryNode); ok {
		nodeID = queryNode.ID
	}
	child := parent.Child(mode, node.String(), vchannel, nodeID)
	return searchutil.WithSearchBenchmarkMetrics(ctx, child), child
}

// Query executes Phase 1 + Phase 2 for a single shard's query (retrieve).
// Replica resolution is handled internally. Results are fed into the provided reducer.
// Returns the ShardPlan for potential requery.
func (s *shardViewQueryClient) Query(ctx context.Context, req *ShardQueryRequest) (*ShardPlan, error) {
	return s.executeShard(ctx, req.VChannel, &shardExecParams{
		consistencyLevel: req.Req.ConsistencyLevel,
		buildPlanReq: func(targetShardID qviews.ShardID) *viewpb.GetQueryPlanRequest {
			return &viewpb.GetQueryPlanRequest{
				CollectionId: req.Req.CollectionID,
				ShardId:      targetShardID.IntoProto(),
				PartitionIds: req.Req.PartitionIDs,
				Request: &viewpb.GetQueryPlanRequest_LegacyRetrieveRequest{
					LegacyRetrieveRequest: req.Req,
				},
			}
		},
		dispatchNode: func(ctx context.Context, node qviews.WorkNode, plan *viewpb.QueryPlan, shardID qviews.ShardID) error {
			resp, err := s.queryServiceClient.QueryOnView(ctx, node, &viewpb.QueryOnViewRequest{
				LegacyReq: legacyRetrieveRequestForNode(plan, node),
				ShardId:   shardID.IntoProto(),
				Version:   plan.Version,
				Mvcc:      plan.GetMvcc(),
			})
			if err != nil {
				return err
			}
			if isEmptySuccessfulQueryResponse(resp) {
				return nil
			}
			return req.Reducer.Add(shardID, resp)
		},
		resetShard: req.Reducer.ResetShard,
	})
}

// QueryStream opens the SN/QN child streams for one vchannel and returns its
// request-scoped Plain Query ReduceStream without consuming output.
func (s *shardViewQueryClient) QueryStream(
	ctx context.Context,
	vchannel string,
	req *internalpb.RetrieveRequest,
	chunkSize int,
) (queryutil.ReduceStream, *ShardPlan, error) {
	if req == nil {
		return nil, nil, merr.WrapErrServiceInternalMsg("QueryStream requires a Query request")
	}

	var lastErr error
	for attempt := 0; attempt < s.maxRetries; attempt++ {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}
		targetShardID := qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: vchannel}
		planReq := &viewpb.GetQueryPlanRequest{
			CollectionId: req.GetCollectionID(),
			ShardId:      targetShardID.IntoProto(),
			PartitionIds: req.GetPartitionIDs(),
			Request: &viewpb.GetQueryPlanRequest_LegacyRetrieveRequest{
				LegacyRetrieveRequest: req,
			},
		}
		plan, err := s.executeGetQueryPlan(ctx, targetShardID, planReq, &shardExecParams{
			consistencyLevel: req.GetConsistencyLevel(),
		})
		if err != nil {
			if ve := viewerror.AsViewError(err); ve != nil && ve.IsRetryable() {
				lastErr = err
				continue
			}
			return nil, nil, err
		}

		shardID := qviews.FromProtoShardID(plan.ShardId)
		workNodes := workNodesFromPlan(plan)
		if mlog.LevelEnabled(mlog.DebugLevel) {
			queryNodeIDs := make([]int64, 0, len(workNodes))
			streamingNodePresent := false
			for _, workNode := range workNodes {
				switch node := workNode.(type) {
				case qviews.QueryNode:
					queryNodeIDs = append(queryNodeIDs, node.ID)
				case qviews.StreamingNode:
					streamingNodePresent = true
				}
			}
			mlog.Debug(ctx, "query view work nodes selected",
				mlog.FieldVChannel(vchannel),
				mlog.Int64("replicaID", shardID.ReplicaID),
				mlog.Int64s("queryNodeIDs", queryNodeIDs),
				mlog.Bool("streamingNodePresent", streamingNodePresent),
				mlog.Int("workNodeCount", len(workNodes)),
			)
		}
		childStreams := make([]queryutil.ReduceStream, 0, len(workNodes))
		for _, node := range workNodes {
			childStream, openErr := s.queryServiceClient.QueryOnViewStream(ctx, node, &viewpb.QueryOnViewRequest{
				LegacyReq:       legacyRetrieveRequestForNode(plan, node),
				ShardId:         shardID.IntoProto(),
				Version:         plan.Version,
				Mvcc:            plan.GetMvcc(),
				StreamChunkSize: int64(chunkSize),
			})
			if openErr != nil {
				err = openErr
				break
			}
			if childStream == nil {
				err = merr.WrapErrServiceInternalMsg("QueryOnViewStream returned a nil stream for work node %s", node.String())
				break
			}
			childStreams = append(childStreams, childStream)
		}
		if err != nil {
			for _, childStream := range childStreams {
				err = errors.Join(err, childStream.Close())
			}
			if ctx.Err() != nil {
				return nil, nil, ctx.Err()
			}
			if ve := viewerror.AsViewError(err); ve != nil && ve.IsRetryable() {
				lastErr = err
				continue
			}
			return nil, nil, err
		}

		reducedStream, err := queryutil.NewReduceStream(req, childStreams, chunkSize)
		if err != nil {
			for _, childStream := range childStreams {
				err = errors.Join(err, childStream.Close())
			}
			return nil, nil, err
		}
		return reducedStream, &ShardPlan{
			ShardID:   shardID,
			Version:   plan.Version,
			Mvcc:      plan.GetMvcc(),
			WorkNodes: workNodes,
		}, nil
	}
	return nil, nil, lastErr
}

// ============================================================================
// Shared shard execution framework
// ============================================================================

// shardExecParams parameterizes the per-shard Phase 1 + Phase 2 batch execution.
// Search and Query use executeShard with these callbacks.
type shardExecParams struct {
	consistencyLevel commonpb.ConsistencyLevel
	// buildPlanReq creates the GetQueryPlanRequest for a target shard.
	buildPlanReq func(targetShardID qviews.ShardID) *viewpb.GetQueryPlanRequest
	// dispatchNode executes a Phase 2 RPC on a single work node.
	// Called concurrently for each work node in the plan, with per-node retry
	// for non-ViewError transient failures.
	dispatchNode func(ctx context.Context, node qviews.WorkNode, plan *viewpb.QueryPlan, shardID qviews.ShardID) error
	// resetShard resets the reducer state for the given shard on retry.
	resetShard func(shardID qviews.ShardID)
}

// executeShard runs Phase 1 + Phase 2 for a single shard with retry.
// The client always targets the primary replica; the real replica ID is
// learned from the Phase 1 plan and used for Phase 2.
//
// Shard-level retry handles ViewErrors (view invalidated, not found, etc.).
// Per-node retry within fanOutToWorkNodes handles transient non-view errors.
func (s *shardViewQueryClient) executeShard(
	ctx context.Context,
	vchannel string,
	params *shardExecParams,
) (*ShardPlan, error) {
	var lastErr error

	for attempt := 0; attempt < s.maxRetries; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		// Before Phase 1 the client only knows the vchannel; the replica ID is
		// unknown and is echoed back by the plan for Phase 2.
		targetShardID := qviews.ShardID{ReplicaID: qviews.UnknownReplicaID, VChannel: vchannel}

		// Phase 1: GetQueryPlan with consistency routing.
		planReq := params.buildPlanReq(targetShardID)
		plan, err := s.executeGetQueryPlan(ctx, targetShardID, planReq, params)
		if err != nil {
			if ve := viewerror.AsViewError(err); ve != nil && ve.IsRetryable() {
				lastErr = err
				continue
			}
			return nil, err
		}

		shardID := qviews.FromProtoShardID(plan.ShardId)
		workNodes := workNodesFromPlan(plan)

		// Phase 2: Fan out to all work nodes concurrently.
		err = s.fanOutToWorkNodes(ctx, workNodes, plan, shardID, params.dispatchNode)
		if err != nil {
			if ve := viewerror.AsViewError(err); ve != nil && ve.IsRetryable() {
				lastErr = err
				params.resetShard(shardID)
				continue
			}
			return nil, err
		}

		return &ShardPlan{
			ShardID:   shardID,
			Version:   plan.Version,
			Mvcc:      plan.GetMvcc(),
			WorkNodes: workNodes,
		}, nil
	}
	return nil, lastErr
}

// executeGetQueryPlan handles consistency-level routing and dispatches Phase 1.
//
// The client always targets the primary replica, so strong consistency is
// satisfied directly by the primary's WAL:
//   - Strong/Session: GetQueryPlan(consistency_level=Strong)
//   - Bounded/Eventually: GetQueryPlan(consistency_level=...) — SN generates MVCC from WAL
func (s *shardViewQueryClient) executeGetQueryPlan(
	ctx context.Context,
	targetShardID qviews.ShardID,
	planReq *viewpb.GetQueryPlanRequest,
	params *shardExecParams,
) (*viewpb.QueryPlan, error) {
	switch params.consistencyLevel {
	case commonpb.ConsistencyLevel_Strong, commonpb.ConsistencyLevel_Session:
		planReq.Mvcc = &viewpb.GetQueryPlanRequest_ConsistencyLevel{
			ConsistencyLevel: commonpb.ConsistencyLevel_Strong,
		}
	default:
		planReq.Mvcc = &viewpb.GetQueryPlanRequest_ConsistencyLevel{
			ConsistencyLevel: params.consistencyLevel,
		}
	}

	resp, err := s.queryPlanClient.GetQueryPlan(ctx, targetShardID, planReq)
	if err != nil {
		return nil, err
	}
	return resp.Plan, nil
}

// fanOutToWorkNodes dispatches Phase 2 to all work nodes concurrently.
// Uses errgroup.WithContext for fast-fail: if any node fails, gCtx is canceled
// and in-flight RPCs on other nodes are aborted. The caller (executeShard) then
// retries the entire shard.
//
// Per-node transient retry (network timeout, etc.) is handled by the
// ViewQueryServiceClient implementation, not here.
func (s *shardViewQueryClient) fanOutToWorkNodes(
	ctx context.Context,
	workNodes []qviews.WorkNode,
	plan *viewpb.QueryPlan,
	shardID qviews.ShardID,
	dispatchNode func(ctx context.Context, node qviews.WorkNode, plan *viewpb.QueryPlan, shardID qviews.ShardID) error,
) error {
	g, gCtx := errgroup.WithContext(ctx)
	for _, node := range workNodes {
		node := node
		g.Go(func() error {
			return dispatchNode(gCtx, node, plan, shardID)
		})
	}
	return g.Wait()
}

func legacySearchRequestForNode(plan *viewpb.QueryPlan, node qviews.WorkNode) *internalpb.SearchRequest {
	req := proto.Clone(plan.GetLegacySearchRequest()).(*internalpb.SearchRequest)
	req.MvccTimestamp = legacyMVCCForNode(plan.GetMvcc(), node)
	return req
}

func legacyRetrieveRequestForNode(plan *viewpb.QueryPlan, node qviews.WorkNode) *internalpb.RetrieveRequest {
	req := proto.Clone(plan.GetLegacyRetrieveRequest()).(*internalpb.RetrieveRequest)
	req.MvccTimestamp = legacyMVCCForNode(plan.GetMvcc(), node)
	return req
}

func legacyMVCCForNode(mvcc *viewpb.QueryPlanMVCC, node qviews.WorkNode) uint64 {
	if mvcc == nil {
		return 0
	}
	if node.NodeType() == qviews.NodeTypeStreamingNode {
		return mvcc.GetGrowingTimetick()
	}
	return mvcc.GetTransformingTimetick()
}

func isEmptySuccessfulQueryResponse(resp *viewpb.QueryOnViewResponse) bool {
	result := resp.GetLegacyResults()
	if result == nil || !merr.Ok(result.GetStatus()) {
		return false
	}
	return result.GetAllRetrieveCount() == 0 &&
		typeutil.GetSizeOfIDs(result.GetIds()) == 0 &&
		len(result.GetFieldsData()) == 0
}

// workNodesFromPlan converts proto QueryPlanWorkNode list to domain WorkNode types.
func workNodesFromPlan(plan *viewpb.QueryPlan) []qviews.WorkNode {
	nodes := make([]qviews.WorkNode, 0, len(plan.WorkNodes))
	for _, n := range plan.WorkNodes {
		switch v := n.Node.(type) {
		case *viewpb.QueryPlanWorkNode_QueryNode:
			nodes = append(nodes, qviews.NewQueryNode(v.QueryNode.NodeId))
		case *viewpb.QueryPlanWorkNode_StreamingNode:
			nodes = append(nodes, qviews.StreamingNode{PChannel: v.StreamingNode.Pchannel})
		}
	}
	return nodes
}
