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

package reducestream

import (
	"context"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// SplitHybridSearchResult streams each leader sub-search with its existing
// request index. The shared vchannel metadata travels once.
func SplitHybridSearchResult(result *internalpb.SearchResults, chunkBytes int, send func(*internalpb.SearchResults) error) error {
	subs := result.GetSubResults()
	if len(subs) == 0 {
		return send(result)
	}
	first := true
	for _, sub := range subs {
		candidate := &internalpb.SearchResults{
			MetricType: sub.GetMetricType(), NumQueries: sub.GetNumQueries(), TopK: sub.GetTopK(),
			SlicedBlob: sub.GetSlicedBlob(), ResultData: sub.GetResultData(),
		}
		err := SplitSearchResult(candidate, sub.GetNumQueries(), sub.GetTopK(), chunkBytes, func(chunk *internalpb.SearchResults) error {
			out := &internalpb.SearchResults{Status: result.GetStatus(), IsAdvanced: true}
			if first {
				first = false
				out.NumQueries = result.GetNumQueries()
				mergeSearchMetadata(out, result)
			}
			out.SubResults = []*internalpb.SubSearchResults{{
				ReqIndex: sub.GetReqIndex(), MetricType: chunk.GetMetricType(),
				NumQueries: sub.GetNumQueries(), TopK: sub.GetTopK(),
				ResultData: chunk.GetResultData(), SlicedBlob: chunk.GetSlicedBlob(),
			}}
			return send(out)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// DrainSearchStreams preserves the candidate list of each child, including
// group keys and all Hybrid sub-searches, before the existing reduction or
// fusion. Do not apply a hit cap here: topK may count groups, not hits.
func DrainSearchStreams(ctx context.Context, streams []ReduceStream) (_ []*internalpb.SearchResults, err error) {
	children := reduce.NewChildStreams(streams)
	defer func() {
		if closeErr := children.Close(); closeErr != nil {
			if err == nil {
				err = closeErr
			} else {
				mlog.Warn(ctx, "failed to close Search ReduceStream children", mlog.Err(closeErr))
			}
		}
	}()
	results := make([]*internalpb.SearchResults, len(streams))
	err = children.ReceiveUntilReady(func(int) bool { return false }, func(i int, chunk *internalpb.SearchResults) error {
		if chunk == nil {
			return merr.WrapErrServiceInternalMsg("nil Search Chunk")
		}
		if err := merr.Error(chunk.GetStatus()); err != nil {
			return err
		}
		if results[i] == nil {
			results[i] = chunk
			return nil
		}
		result := results[i]
		if !result.GetIsAdvanced() {
			src, err := decodeChunk(chunk, result.GetNumQueries(), result.GetTopK())
			if err != nil {
				return err
			}
			result.ResultData = AppendSearchData(result.GetResultData(), src)
			return nil
		}
		for _, src := range chunk.GetSubResults() {
			var dst *internalpb.SubSearchResults
			for _, sub := range result.GetSubResults() {
				if src.GetReqIndex() == sub.GetReqIndex() {
					dst = sub
					break
				}
			}
			if dst == nil {
				result.SubResults = append(result.SubResults, src)
			} else {
				dst.ResultData = AppendSearchData(dst.GetResultData(), src.GetResultData())
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return results, nil
}

func AppendSearchData(dst, src *schemapb.SearchResultData) *schemapb.SearchResultData {
	if dst == nil {
		return src
	}
	if src == nil {
		return dst
	}
	fields := typeutil.NewFieldDataIdxComputer(src.GetFieldsData())
	var singularDst, singularSrc []*schemapb.FieldData
	if src.GetGroupByFieldValue() != nil {
		singularDst = []*schemapb.FieldData{dst.GetGroupByFieldValue()}
		singularSrc = []*schemapb.FieldData{src.GetGroupByFieldValue()}
	}
	row := int64(0)
	for q, size := range src.GetTopks() {
		for range size {
			appendSearchHit(dst, src, q, row, fields.Compute(row))
			if len(src.GetGroupByFieldValues()) > 0 {
				if len(dst.GetGroupByFieldValues()) == 0 {
					dst.GroupByFieldValues = typeutil.PrepareResultFieldData(src.GetGroupByFieldValues(), 0)
				}
				typeutil.AppendFieldData(dst.GroupByFieldValues, src.GetGroupByFieldValues(), row)
			}
			if len(singularDst) != 0 {
				typeutil.AppendFieldData(singularDst, singularSrc, row)
			}
			row++
		}
	}
	if len(singularDst) != 0 {
		dst.GroupByFieldValue = singularDst[0]
	}
	dst.AllSearchCount += src.GetAllSearchCount()
	return dst
}
