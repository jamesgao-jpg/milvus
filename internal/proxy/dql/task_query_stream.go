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
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy/shardclient"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
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

	if !queryutil.CanMergeQueryStream(t.RetrieveRequest) {
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
	result, err := consumeQueryResultStream(ctx, stream, t.RetrieveRequest)
	if err != nil {
		return err
	}
	// Keep output field organization, geometry conversion and offset in PostExecute.
	t.resultBuf.Insert(result)
	return nil
}

// The final SDK result is unary. Append selected rows immediately so consumed
// source Chunks are not kept alive by the final merge.
func consumeQueryResultStream(ctx context.Context, stream queryutil.ReduceStream, req *internalpb.RetrieveRequest) (_ *internalpb.RetrieveResults, err error) {
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
	var units, outputBytes int64
	maxOutputBytes := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()
	positional := len(req.GetGroupByFieldIds()) != 0 || len(req.GetAggregates()) != 0
	limit := req.GetLimit()
	// ORDER BY keys need not follow PK order. A later candidate can still
	// reveal a cross-vchannel duplicate after the client window is full.
	drain := len(req.GetOrderByFields()) != 0 && len(req.GetGroupByFieldIds()) == 0
	for drain || limit <= 0 || units < limit {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if limit > 0 && units >= limit {
			continue
		}
		rows := queryutil.GetRowCount(chunk)
		ids, columns, elements := chunk.GetIds(), chunk.GetFieldsData(), chunk.GetElementIndices()
		if result == nil {
			// Keep the first Chunk's metadata, but charge its rows through the same append path.
			result = chunk
			if rows > 0 {
				result.Ids, result.FieldsData, result.ElementIndices = nil, nil, nil
			}
		}
		if rows == 0 {
			continue
		}
		if len(result.FieldsData) == 0 {
			result.FieldsData = typeutil.PrepareResultFieldData(columns, 0)
		}
		if len(result.FieldsData) != len(columns) {
			return nil, merr.WrapErrServiceInternalMsg("final Query Chunks have different field counts")
		}
		if ids != nil && result.Ids == nil {
			result.Ids = &schemapb.IDs{}
		}
		fields := typeutil.NewFieldDataIdxComputer(columns)
		for row := 0; row < rows && (limit <= 0 || units < limit); row++ {
			if ids != nil {
				typeutil.AppendIDs(result.Ids, ids, row)
			}
			idx := int64(row)
			if positional {
				// Aggregates on the same field can have the same ID but distinct columns.
				for i := range columns {
					outputBytes += typeutil.AppendFieldData(result.FieldsData[i:i+1], columns[i:i+1], idx)
				}
			} else {
				outputBytes += typeutil.AppendFieldData(result.FieldsData, columns, idx, fields.Compute(idx)...)
			}
			if chunk.GetElementLevel() {
				indices := elements[row]
				result.ElementIndices = append(result.ElementIndices, indices)
				outputBytes += int64(proto.Size(indices))
				units += int64(len(indices.GetIndices()))
			} else {
				units++
			}
			if outputBytes > maxOutputBytes {
				return nil, merr.WrapErrParameterInvalidMsg("query results exceed the maxOutputSize Limit %d", maxOutputBytes)
			}
		}
	}
	if result == nil {
		return &internalpb.RetrieveResults{Status: merr.Success()}, nil
	}
	return result, nil
}
