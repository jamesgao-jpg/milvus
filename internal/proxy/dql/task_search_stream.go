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
	switch t.GetSearchType() {
	case internalpb.SearchType_PURE_ANN_SEARCH_NO_FILTER, internalpb.SearchType_PURE_ANN_SEARCH_WITH_FILTER:
	default:
		// Range Search, Group By, iterators, and iterative filtering are not
		// part of the first port.
		return false
	}
	// Keep Hybrid, reranking, ORDER BY and offset on Batch in this slice.
	return !t.GetIsAdvanced() && t.aggCtx == nil && t.rerankMeta == nil &&
		len(t.orderByFields) == 0 && t.GetOffset() == 0
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

	reducer := searchreducestream.NewReduceStream(t.SearchRequest, streams, pk.GetDataType(), paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64())
	result, err := reducer.Recv()
	closeErr := reducer.Close()
	if err != nil {
		if closeErr != nil {
			mlog.Warn(ctx, "failed to close Search ReduceStream", mlog.Err(closeErr))
		}
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if t.resultBuf != nil {
		// The ordinary pipeline receives one already-merged result. Its
		// single-result fast path still restores score signs and fills fields.
		t.resultBuf.Insert(result)
	}
	return nil
}

func (t *SearchTask) openSearchStream(ctx context.Context, qn types.QueryNodeClient, nodeID int64, channel string, chunkBytes int32) (*grpcSearchStream, error) {
	req := t.shardSearchRequest(nodeID, channel)
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
	t.lb.UpdateCostMetrics(nodeID, first.GetCostAggregation())
	return &grpcSearchStream{rpc: rpc, first: first, cancel: cancel}, nil
}

// grpcSearchStream retains the first Chunk read inside the leader retry
// callback. Once returned to the reducer, it never retries an individual RPC.
type grpcSearchStream struct {
	rpc    querypb.QueryNode_SearchStreamClient
	first  *internalpb.SearchResults
	cancel context.CancelFunc
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
	return s.rpc.Recv()
}

func (s *grpcSearchStream) Close() error {
	s.cancel()
	return nil
}

func (s *grpcSearchStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}
