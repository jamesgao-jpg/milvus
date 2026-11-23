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

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/util/grpcclient"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	qtype "github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// QueryReduceStream sends PK-ordered Chunks while queryWithSnapshot owns the pin.
func (sd *shardDelegator) QueryReduceStream(ctx context.Context, req *querypb.QueryRequest, send func(*internalpb.RetrieveResults) error) error {
	if req.GetStreamChunkBytes() <= 0 || !queryutil.IsPKOrderedQuery(req.GetReq()) {
		return merr.WrapErrServiceInternalMsg("ordered Query stream requires a PK-ordered request and positive Chunk size")
	}
	_, err := sd.queryWithSnapshot(ctx, req, send)
	return err
}

func (sd *shardDelegator) reduceQueryStreams(ctx context.Context, req *querypb.QueryRequest, tasks []subTask[*querypb.QueryRequest], send func(*internalpb.RetrieveResults) error) (_ []*internalpb.RetrieveResults, err error) {
	// Unlike errgroup.WithContext, this context survives successful setup.
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	streams := make([]queryutil.ReduceStream, len(tasks))
	var setup errgroup.Group
	for i, task := range tasks {
		setup.Go(func() error {
			if task.targetID == -1 || task.worker == nil {
				err := merr.WrapErrServiceInternalMsg("Query segments have no readable worker")
				cancel(err)
				return err
			}
			stream, err := task.worker.QuerySegmentsReduceStream(workerReadContext(ctx), task.req, req.GetStreamChunkBytes())
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
	if !queryutil.IsPKOrderedQuery(req.GetReq()) {
		return queryutil.DrainRetrieveStreams(ctx, streams)
	}
	// The unary caller still consumes a single result.
	chunkBytes := int(req.GetStreamChunkBytes())
	if send == nil {
		chunkBytes = math.MaxInt
	}
	reducer := queryutil.NewOrderedReduceStream(req.GetReq(), sd.collection.Schema(), streams, true, int64(chunkBytes))
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
		chunk.Base = &commonpb.MsgBase{SourceID: paramtable.GetNodeID()}
		if chunk.CostAggregation != nil {
			chunk.MvccTimestamp = req.GetReq().GetMvccTimestamp()
		}
		if len(chunk.GetFieldsData()) == 0 {
			if err := qtype.FillRetrieveResultIfEmpty(qtype.NewInternalResult(chunk), req.GetReq().GetOutputFieldsId(), sd.collection.Schema()); err != nil {
				return nil, err
			}
		}
		if send == nil {
			return []*internalpb.RetrieveResults{chunk}, nil
		}
		if err := send(chunk); err != nil {
			return nil, err
		}
	}
}
