package dql

import (
	"context"
	"io"
	"math"
	"sync"

	"github.com/cockroachdb/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/proxy/shardclient"
	"github.com/milvus-io/milvus/internal/types"
	searchreducestream "github.com/milvus-io/milvus/internal/util/searchutil/reduce_stream"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
)

func (t *SearchTask) supportsStreamingReduce() bool {
	if paramtable.Get().QueryNodeCfg.PartialResultRequiredDataRatio.GetAsFloat() < 1 {
		return false
	}
	if t.SearchRequest == nil {
		return false
	}
	// Request validation and the selected fusion/rerank pipeline are the
	// same as Batch. The stream only changes how candidates reach them.
	return true
}

func (t *SearchTask) executeSearchStream(ctx context.Context, workload shardclient.CollectionWorkLoad, chunkBytes int64) error {
	pk, err := t.schema.GetPkField()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	var mu sync.Mutex
	var streams []searchreducestream.ReduceStream
	workload.CancelOnFailure = cancel
	workload.Exec = func(ctx context.Context, nodeID int64, qn types.QueryNodeClient, channel string) error {
		stream, err := t.openSearchStream(ctx, qn, nodeID, channel, int32(min(chunkBytes, math.MaxInt32)))
		if err != nil {
			return err
		}
		mu.Lock()
		streams = append(streams, stream)
		mu.Unlock()
		return nil
	}
	if err := t.executeSearch(ctx, workload); err != nil {
		if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, ctx.Err()) {
			err = cause
		}
		for _, stream := range streams {
			_ = stream.Close()
		}
		return err
	}

	maxOutput := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()
	if searchreducestream.IsUngroupedHybrid(t.SearchRequest) {
		result, err := searchreducestream.ReduceHybridSearchStreams(ctx, t.SearchRequest, streams, func(index int, inputs []searchreducestream.ReduceStream) (*internalpb.SubSearchResults, error) {
			sub := t.GetSubReqs()[index]
			req := &internalpb.SearchRequest{Nq: sub.GetNq(), Topk: sub.GetTopk(), Offset: sub.GetOffset()}
			chunk, err := consumeSearchResultStream(ctx, searchreducestream.NewReduceStream(req, inputs, pk.GetDataType(), maxOutput, int(chunkBytes)))
			if err != nil {
				return nil, err
			}
			// Keep raw scores and the prefix before OFFSET. The existing
			// Hybrid pipeline applies pagination and fusion once, after all indexes finish.
			return &internalpb.SubSearchResults{
				ReqIndex: int64(index), NumQueries: req.Nq, TopK: req.Topk,
				MetricType: chunk.GetMetricType(), ResultData: chunk.GetResultData(),
			}, nil
		})
		if err == nil {
			t.resultBuf.Insert(result)
		}
		return err
	}
	if t.GetIsAdvanced() {
		results, err := searchreducestream.DrainSearchStreams(ctx, streams)
		if err != nil {
			return err
		}
		for _, result := range results {
			t.resultBuf.Insert(result)
		}
		return nil
	}
	result, err := consumeSearchResultStream(ctx, searchreducestream.NewReduceStream(t.SearchRequest, streams, pk.GetDataType(), maxOutput, int(chunkBytes)))
	if err == nil && t.resultBuf != nil {
		// The pipeline receives the bounded candidates in merge order. For
		// grouped Search it applies the group offset, output order and aggregation.
		t.resultBuf.Insert(result)
	}
	return err
}

func consumeSearchResultStream(ctx context.Context, reducer *searchreducestream.OrderedReduceStream) (result *internalpb.SearchResults, err error) {
	for {
		chunk, recvErr := reducer.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			err = recvErr
			break
		}
		if result == nil {
			result = chunk
		} else {
			result.ResultData = searchreducestream.AppendSearchData(result.GetResultData(), chunk.GetResultData())
		}
		counts := result.GetResultData().GetTopks()
		if len(counts) > 0 && counts[len(counts)-1] >= reducer.HitLimit() {
			break
		}
	}
	closeErr := reducer.Close()
	if err != nil {
		if closeErr != nil {
			mlog.Warn(ctx, "failed to close Search ReduceStream", mlog.Err(closeErr))
		}
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	return result, nil
}

func (t *SearchTask) openSearchStream(ctx context.Context, qn types.QueryNodeClient, nodeID int64, channel string, chunkBytes int32) (*grpcSearchStream, error) {
	req := t.shardSearchRequest(nodeID, channel)
	// Opt in to per-index end markers. Older leaders ignore the inner
	// threshold and return the legacy format; the reader accepts either.
	if searchreducestream.IsUngroupedHybrid(req.GetReq()) {
		req.StreamChunkBytes = chunkBytes
	}
	ctx = retry.WithMaxAttemptsContext(ctx, 1)
	rpcCtx, cancel := context.WithCancel(ctx)
	rpc, err := qn.SearchStream(rpcCtx)
	if err == nil {
		err = rpc.Send(&querypb.SearchStreamRequest{Request: req, ChunkBytes: chunkBytes})
	}
	if err == nil {
		err = rpc.CloseSend()
	}
	var first *internalpb.SearchResults
	if err == nil || (rpc != nil && errors.Is(err, io.EOF)) {
		// Include the first Chunk in leader selection: a successful gRPC open
		// does not tell us whether the delegator accepted and ran the Search.
		first, err = rpc.Recv()
	}
	if status.Code(err) == codes.Unimplemented {
		// During a rolling upgrade, an older leader can serve its unary result.
		cancel()
		rpc = nil
		req.StreamChunkBytes = 0 // Unary fallback must dispatch as Batch.
		first, err = qn.Search(ctx, req)
	}
	if err != nil {
		cancel()
		if ctx.Err() == nil {
			t.shardClientMgr.InvalidateShardLeaderCache([]int64{t.GetCollectionID()})
		}
		if errors.Is(err, io.EOF) {
			return nil, merr.WrapErrServiceInternalMsg("leader SearchStream ended without a result")
		}
		return nil, err
	}
	if first == nil {
		cancel()
		return nil, merr.WrapErrServiceInternalMsg("leader returned no Search result")
	}
	if !merr.Ok(first.GetStatus()) {
		cancel()
		err := merr.Error(first.GetStatus())
		if first.GetStatus().GetErrorCode() == commonpb.ErrorCode_NotShardLeader || errors.Is(err, merr.ErrCollectionNotLoaded) {
			t.shardClientMgr.InvalidateShardLeaderCache([]int64{t.GetCollectionID()})
		}
		return nil, err
	}
	t.queryChannelsNode.Insert(channel, nodeID)
	stream := &grpcSearchStream{rpc: rpc, first: first, cancel: cancel}
	if subs := first.GetSubResults(); len(subs) > 0 && subs[0] != nil && subs[0].StreamEnd != nil {
		// The live Hybrid leader can send B before A finishes. Its cost
		// covers all sub-searches, so it arrives once, in the trailing summary.
		stream.reportCost = func(cost *internalpb.CostAggregation) { t.lb.UpdateCostMetrics(nodeID, cost) }
	} else {
		t.lb.UpdateCostMetrics(nodeID, first.GetCostAggregation())
	}
	return stream, nil
}

// grpcSearchStream retains the first Chunk read inside the leader retry
// callback. Once returned to the reducer, it never retries an individual RPC.
type grpcSearchStream struct {
	rpc        querypb.QueryNode_SearchStreamClient
	first      *internalpb.SearchResults
	cancel     context.CancelFunc
	reportCost func(*internalpb.CostAggregation) // Only set for a live Hybrid leader until its summary arrives.
}

func (s *grpcSearchStream) Recv() (*internalpb.SearchResults, error) {
	if s.first != nil {
		first := s.first
		s.first = nil
		return first, nil
	}
	if s.rpc == nil {
		return nil, io.EOF
	}
	chunk, err := s.rpc.Recv()
	if cost := chunk.GetCostAggregation(); err == nil && s.reportCost != nil && cost != nil {
		s.reportCost(cost)
		s.reportCost = nil
	}
	return chunk, err
}

func (s *grpcSearchStream) Close() error {
	s.cancel()
	return nil
}

func (s *grpcSearchStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}
