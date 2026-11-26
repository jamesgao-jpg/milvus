// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package delegator

import (
	"context"
	"io"
	"math"

	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/grpcclient"
	"github.com/milvus-io/milvus/internal/util/reduce"
	searchreducestream "github.com/milvus-io/milvus/internal/util/searchutil/reduce_stream"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// SearchReduceStream sends ordinary ANN Chunks inside the normal shard read.
func (sd *shardDelegator) SearchReduceStream(ctx context.Context, req *querypb.SearchRequest, send func(*internalpb.SearchResults) error) error {
	results, err := sd.searchWithSnapshot(ctx, req, send)
	return sendRemainingSearchResults(ctx, req, results, err, send)
}

func sendRemainingSearchResults(ctx context.Context, req *querypb.SearchRequest, results []*internalpb.SearchResults, err error, send func(*internalpb.SearchResults) error) error {
	if err != nil || results == nil {
		return err
	}
	// An empty BM25 corpus can skip dispatch entirely. Preserve Batch's
	// successful empty response rather than returning EOF without metadata.
	info := reduce.NewReduceSearchResultInfo(req.GetReq().GetNq(), req.GetReq().GetTopk()).WithMetricType(req.GetReq().GetMetricType())
	result, err := segments.ReduceSearchOnQueryNode(ctx, results, info)
	if err != nil {
		return err
	}
	return searchreducestream.SplitSearchResult(result, req.GetReq().GetNq(), req.GetReq().GetTopk(), int(req.GetStreamChunkBytes()), send)
}

// streamHybridSubSearch runs under the caller's vchannel pin. Each invocation
// owns its worker streams, not the shared leader RPC. End only this ReqIndex,
// even for an empty result; other sub-searches can still be reading workers.
func (sd *shardDelegator) streamHybridSubSearch(ctx context.Context, index int, req *querypb.SearchRequest, sealed []SnapshotItem, growing []SegmentEntry, rowCounts map[int64]int64, send func(*internalpb.SearchResults) error) (*internalpb.SearchResults, error) {
	var summary *internalpb.SearchResults
	write := func(chunk *internalpb.SearchResults) error {
		if err := merr.Error(chunk.GetStatus()); err != nil {
			return err
		}
		if summary == nil {
			// The first Chunk carries worker metadata. Do not retain its hits
			// until the slower sub-searches finish.
			summary = &internalpb.SearchResults{
				NumQueries: req.GetReq().GetNq(), CostAggregation: chunk.GetCostAggregation(),
				ChannelsMvcc: chunk.GetChannelsMvcc(), IsTopkReduce: chunk.GetIsTopkReduce(),
				ScannedRemoteBytes: chunk.GetScannedRemoteBytes(), ScannedTotalBytes: chunk.GetScannedTotalBytes(),
			}
		}
		return send(&internalpb.SearchResults{Status: merr.Success(), IsAdvanced: true, SubResults: []*internalpb.SubSearchResults{{
			ReqIndex: int64(index), StreamEnd: proto.Bool(false), NumQueries: req.GetReq().GetNq(), TopK: req.GetReq().GetTopk(),
			MetricType: chunk.GetMetricType(), ResultData: chunk.GetResultData(), SlicedBlob: chunk.GetSlicedBlob(),
		}}})
	}
	results, err := sd.search(ctx, req, sealed, growing, rowCounts, write)
	if err = sendRemainingSearchResults(ctx, req, results, err, write); err != nil {
		return nil, err
	}
	return summary, send(&internalpb.SearchResults{Status: merr.Success(), IsAdvanced: true, SubResults: []*internalpb.SubSearchResults{{
		ReqIndex: int64(index), StreamEnd: proto.Bool(true),
	}}})
}

func (sd *shardDelegator) reduceSearchStreams(ctx context.Context, req *querypb.SearchRequest, tasks []subTask[*querypb.SearchRequest], send func(*internalpb.SearchResults) error) (_ []*internalpb.SearchResults, err error) {
	pk, err := typeutil.GetPrimaryFieldSchema(sd.collection.Schema())
	if err != nil {
		return nil, err
	}
	// The open group and the reducer share this lifetime. A successful Wait
	// must not cancel the worker RPCs that the reducer is about to read.
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	streams := make([]searchreducestream.ReduceStream, len(tasks))
	var setup errgroup.Group
	for i, task := range tasks {
		setup.Go(func() error {
			if task.targetID == -1 || task.worker == nil {
				err := merr.WrapErrServiceInternalMsg("Search segments have no readable worker")
				cancel(err)
				return err
			}
			stream, err := task.worker.SearchSegmentsStream(workerReadContext(ctx), task.req, req.GetStreamChunkBytes())
			if err != nil {
				if grpcclient.IsServerIDMismatchErr(err) {
					sd.markSegmentOffline(task.req.GetSegmentIDs()...)
				}
				cancel(err)
				return err
			}
			streams[i] = stream
			return nil
		})
	}
	if err := setup.Wait(); err != nil {
		for _, stream := range streams {
			if stream != nil {
				_ = stream.Close()
			}
		}
		return nil, context.Cause(ctx)
	}

	if send == nil && (req.GetReq().GetGroupByFieldId() > 0 || len(req.GetReq().GetGroupByFieldIds()) > 0) {
		// A legacy Proxy did not opt in to per-sub-search Chunks.
		return searchreducestream.DrainSearchStreams(ctx, streams)
	}
	// Buffered callers (including Hybrid sub-searches) still consume a single result.
	chunkBytes := int(req.GetStreamChunkBytes())
	if send == nil {
		chunkBytes = math.MaxInt
	}
	reducer := searchreducestream.NewShardReduceStream(req.GetReq(), streams, pk.GetDataType(), paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64(), chunkBytes)
	defer func() {
		if closeErr := reducer.Close(); closeErr != nil {
			if err == nil {
				err = closeErr
			} else {
				mlog.Warn(ctx, "failed to close worker streams", mlog.Err(closeErr))
			}
		}
	}()
	for {
		chunk, err := reducer.Recv()
		if err == io.EOF {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}

		if send == nil {
			return []*internalpb.SearchResults{chunk}, nil
		}
		if err := send(chunk); err != nil {
			return nil, err
		}
	}
}
