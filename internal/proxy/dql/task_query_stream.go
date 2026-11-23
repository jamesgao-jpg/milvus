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

package dql

import (
	"context"
	"io"
	"math"
	"sync"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/proxy/shardclient"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func (t *QueryTask) executeQueryStream(ctx context.Context, workload shardclient.CollectionWorkLoad, chunkBytes int64) error {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	var mu sync.Mutex
	var streams []queryutil.ReduceStream
	workload.CancelOnFailure = cancel
	workload.Exec = func(ctx context.Context, nodeID int64, qn types.QueryNodeClient, channel string) error {
		req := t.shardQueryRequest(nodeID, channel)
		if req == nil {
			return nil
		}
		ctx = retry.WithMaxAttemptsContext(ctx, 1)
		first, stream, err := queryutil.OpenQueryReduceStream(ctx, qn, req, int32(min(chunkBytes, math.MaxInt32)), false)
		if err != nil {
			if ctx.Err() == nil {
				t.shardclientMgr.InvalidateShardLeaderCache([]int64{t.GetCollectionID()})
			}
			return err
		}
		if err := merr.Error(first.GetStatus()); err != nil {
			if first.GetStatus().GetErrorCode() == commonpb.ErrorCode_NotShardLeader || errors.Is(err, merr.ErrCollectionNotLoaded) {
				t.shardclientMgr.InvalidateShardLeaderCache([]int64{t.GetCollectionID()})
			}
			return err
		}
		if t.actualChannelsMvcc != nil {
			t.actualChannelsMvcc.Insert(channel, first.GetMvccTimestamp())
		}
		t.lb.UpdateCostMetrics(nodeID, first.GetCostAggregation())
		mu.Lock()
		streams = append(streams, stream)
		mu.Unlock()
		return nil
	}
	if err := t.executeQuery(ctx, workload); err != nil {
		for _, stream := range streams {
			_ = stream.Close()
		}
		if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, ctx.Err()) {
			return cause
		}
		return err
	}

	if !queryutil.IsPKOrderedQuery(t.RetrieveRequest) {
		results, err := queryutil.DrainRetrieveStreams(ctx, streams)
		if err != nil {
			return err
		}
		for _, result := range results {
			t.resultBuf.Insert(result)
		}
		return nil
	}
	stream := queryutil.NewOrderedReduceStream(t.RetrieveRequest, t.schema.CollectionSchema, streams, false, chunkBytes)
	result, err := consumeQueryResultStream(ctx, stream, t.GetLimit())
	if err != nil {
		return err
	}
	// Keep output field organization, geometry conversion and offset in PostExecute.
	t.resultBuf.Insert(result)
	return nil
}

// The final SDK result is unary. Append selected rows immediately so consumed
// source Chunks are not kept alive by the final merge.
func consumeQueryResultStream(ctx context.Context, stream queryutil.ReduceStream, limit int64) (_ *internalpb.RetrieveResults, err error) {
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			if err == nil {
				err = closeErr
			} else {
				mlog.Warn(ctx, "failed to close Query stream", mlog.Err(closeErr))
			}
		}
	}()
	var result *internalpb.RetrieveResults
	var units int64
	for limit <= 0 || units < limit {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		rows := typeutil.GetSizeOfIDs(chunk.GetIds())
		if result == nil {
			result = chunk
			if !chunk.GetElementLevel() {
				units += int64(rows)
			} else {
				for _, indices := range chunk.GetElementIndices() {
					units += int64(len(indices.GetIndices()))
				}
			}
			continue
		}
		fields := typeutil.NewFieldDataIdxComputer(chunk.GetFieldsData())
		for row := 0; row < rows && (limit <= 0 || units < limit); row++ {
			typeutil.AppendIDs(result.Ids, chunk.GetIds(), row)
			typeutil.AppendFieldData(result.FieldsData, chunk.GetFieldsData(), int64(row), fields.Compute(int64(row))...)
			if chunk.GetElementLevel() {
				indices := chunk.GetElementIndices()[row]
				result.ElementIndices = append(result.ElementIndices, indices)
				units += int64(len(indices.GetIndices()))
			} else {
				units++
			}
		}
	}
	if result == nil {
		return &internalpb.RetrieveResults{Status: merr.Success()}, nil
	}
	return result, nil
}
