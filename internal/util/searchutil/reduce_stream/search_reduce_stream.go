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
	"io"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/fastpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type ReduceStream = reduce.ReduceStream[*internalpb.SearchResults]

// orderedChildBuffer owns one Chunk and the cursor into its query-major hits.
// The same row numbering is used for scoring, PKs, and nullable field data.
type orderedChildBuffer struct {
	data     *schemapb.SearchResultData     // Data for the current Chunk.
	fields   *typeutil.FieldDataIdxComputer // Nullable field offsets for data.
	query    int                            // Current query vector.
	row      int64                          // Current row in data.
	queryEnd int64                          // Exclusive end of the current query in data.
	first    bool                           // Whether the leader's metadata was received.
}

func (b *orderedChildBuffer) advanceQuery() {
	for b.query < len(b.data.GetTopks()) && b.row == b.queryEnd {
		b.query++
		if b.query < len(b.data.GetTopks()) {
			b.queryEnd += b.data.Topks[b.query]
		}
	}
	if b.query == len(b.data.GetTopks()) {
		b.data, b.fields = nil, nil
	}
}

func (b *orderedChildBuffer) accept(data *schemapb.SearchResultData) {
	b.data = data
	b.fields = typeutil.NewFieldDataIdxComputer(data.GetFieldsData())
	b.query, b.row, b.queryEnd = 0, 0, 0
	if len(data.GetTopks()) > 0 {
		b.queryEnd = data.Topks[0]
	}
	b.advanceQuery()
}

func decodeChunk(chunk *internalpb.SearchResults, nq, topK int64) (*schemapb.SearchResultData, error) {
	data := chunk.GetResultData()
	if data == nil && len(chunk.GetSlicedBlob()) > 0 {
		data = &schemapb.SearchResultData{}
		if err := fastpb.UnmarshalSearchResultData(chunk.GetSlicedBlob(), data); err != nil {
			return nil, merr.Wrap(err, "decode Search Chunk")
		}
	}
	if data == nil {
		return nil, nil
	}
	if data.GetNumQueries() != nq || data.GetTopK() != topK || len(data.GetTopks()) != int(nq) {
		return nil, merr.WrapErrServiceInternalMsg("Search Chunk does not match request nq=%d topK=%d", nq, topK)
	}
	hits := int64(typeutil.GetSizeOfIDs(data.GetIds()))
	remaining := hits
	for _, count := range data.GetTopks() {
		if count < 0 || count > remaining {
			return nil, merr.WrapErrServiceInternalMsg("invalid Search Chunk hit count %d", count)
		}
		remaining -= count
	}
	if remaining != 0 || int64(len(data.GetScores())) != hits {
		return nil, merr.WrapErrServiceInternalMsg("Search Chunk IDs, scores and Topks do not align")
	}
	if indices := data.GetElementIndices(); indices != nil && int64(len(indices.GetData())) != hits {
		return nil, merr.WrapErrServiceInternalMsg("Search Chunk element indices do not align with hits")
	}
	return data, nil
}

func searchUnitBytes(data *schemapb.SearchResultData, row int64, fieldIndexes []int64) (int, error) {
	idsSize, err := reduce.IDsRowSize(data.GetIds(), row)
	if err != nil {
		return 0, err
	}
	size := protowire.SizeTag(reduce.SearchScoresFieldNumber) + protowire.SizeBytes(protowire.SizeFixed32()) +
		protowire.SizeTag(reduce.SearchIDsFieldNumber) + protowire.SizeBytes(idsSize)
	for i, field := range data.GetFieldsData() {
		dataRow := row
		if fieldIndexes != nil {
			dataRow = fieldIndexes[i]
		}
		fieldSize, err := reduce.FieldDataRowSize(field, row, dataRow)
		if err != nil {
			return 0, err
		}
		size += protowire.SizeTag(reduce.SearchFieldsDataFieldNumber) + protowire.SizeBytes(fieldSize)
	}
	if indices := data.GetElementIndices(); indices != nil {
		payload := protowire.SizeTag(reduce.LongArrayDataFieldNumber) + protowire.SizeBytes(protowire.SizeVarint(uint64(indices.Data[row])))
		size += protowire.SizeTag(reduce.SearchElementIndicesFieldNumber) + protowire.SizeBytes(payload)
	}
	return size, nil
}

func newSearchData(nq, topK int64) *schemapb.SearchResultData {
	return &schemapb.SearchResultData{NumQueries: nq, TopK: topK, Topks: make([]int64, nq), Ids: &schemapb.IDs{}}
}

func appendSearchHit(dst, src *schemapb.SearchResultData, query int, row int64, fieldIndexes []int64) int64 {
	if len(dst.FieldsData) == 0 && len(src.FieldsData) != 0 {
		dst.FieldsData = typeutil.PrepareResultFieldData(src.FieldsData, 0)
	}
	var size int64
	if len(dst.FieldsData) != 0 {
		size = typeutil.AppendFieldData(dst.FieldsData, src.FieldsData, row, fieldIndexes...)
	}
	typeutil.AppendPKs(dst.Ids, typeutil.GetPK(src.Ids, row))
	dst.Scores = append(dst.Scores, src.Scores[row])
	dst.Topks[query]++
	if indices := src.GetElementIndices(); indices != nil {
		if dst.ElementIndices == nil {
			dst.ElementIndices = &schemapb.LongArray{}
		}
		dst.ElementIndices.Data = append(dst.ElementIndices.Data, indices.Data[row])
	}
	return size
}

// SplitSearchResult emits one Chunk at a time from the completed leader result.
// Leader metadata and AllSearchCount travel only on the first Chunk.
func SplitSearchResult(result *internalpb.SearchResults, nq, topK int64, chunkBytes int, send func(*internalpb.SearchResults) error) error {
	if chunkBytes <= 0 {
		return merr.WrapErrServiceInternalMsg("Search Chunk bytes must be positive")
	}
	data, err := decodeChunk(result, nq, topK)
	if err != nil {
		return err
	}
	if len(data.GetScores()) == 0 {
		return send(result)
	}

	first := true
	chunk := newSearchData(nq, topK)
	size := 0
	emit := func() error {
		out := &internalpb.SearchResults{Status: result.Status, MetricType: result.MetricType, NumQueries: nq, TopK: topK}
		if first {
			first = false
			out = result
			out.SlicedBlob, out.SlicedNumCount, out.SlicedOffset = nil, 0, 0
			chunk.AllSearchCount = data.AllSearchCount
		}
		out.ResultData = chunk
		return send(out)
	}

	buffer := orderedChildBuffer{}
	buffer.accept(data)
	for buffer.data != nil {
		indexes := buffer.fields.Compute(buffer.row)
		bytes, err := searchUnitBytes(data, buffer.row, indexes)
		if err != nil {
			return err
		}
		if bytes > chunkBytes && size > 0 {
			if err := emit(); err != nil {
				return err
			}
			chunk, size = newSearchData(nq, topK), 0
		}
		appendSearchHit(chunk, data, buffer.query, buffer.row, indexes)
		buffer.row++
		buffer.advanceQuery()
		size += bytes
		if size >= chunkBytes {
			if err := emit(); err != nil {
				return err
			}
			chunk, size = newSearchData(nq, topK), 0
		}
	}
	if size > 0 {
		return emit()
	}
	return nil
}

// OrderedReduceStream merges query-major leader results using 3.0's Batch
// score/PK order. Its single caller is the ordinary Search pipeline, which
// already needs the complete final window, so Recv returns that window once.
type OrderedReduceStream struct {
	recvMu         sync.Mutex                                      // Serializes Recv and Close cleanup.
	childStreams   *reduce.ChildStreams[*internalpb.SearchResults] // Receives and closes leaders.
	childBuffers   []orderedChildBuffer                            // One current Chunk per leader.
	nq             int64                                           // Number of query vectors.
	topK           int64                                           // Maximum hits emitted per query vector.
	pkType         schemapb.DataType                               // Preserves typed IDs for empty Batch results.
	maxOutputBytes int64                                           // Batch quota for a multi-leader merge.
	finished       bool                                            // Whether final output has already been read.
}

func NewReduceStream(request *internalpb.SearchRequest, streams []ReduceStream, pkType schemapb.DataType, maxOutputBytes int64) *OrderedReduceStream {
	return &OrderedReduceStream{
		childStreams: reduce.NewChildStreams(streams), childBuffers: make([]orderedChildBuffer, len(streams)),
		nq: request.GetNq(), topK: request.GetTopk(), pkType: pkType, maxOutputBytes: maxOutputBytes,
	}
}

func (s *OrderedReduceStream) Recv() (*internalpb.SearchResults, error) {
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	if s.finished {
		return nil, io.EOF
	}
	s.finished = true

	out := &internalpb.SearchResults{Status: merr.Success(), NumQueries: s.nq, TopK: s.topK}
	data := newSearchData(s.nq, s.topK)
	switch s.pkType {
	case schemapb.DataType_Int64:
		data.Ids.IdField = &schemapb.IDs_IntId{IntId: &schemapb.LongArray{}}
	case schemapb.DataType_VarChar:
		data.Ids.IdField = &schemapb.IDs_StrId{StrId: &schemapb.StringArray{}}
	}
	var outputBytes int64
	var emptyResult *schemapb.SearchResultData
	numResults := 0
	accept := func(i int, chunk *internalpb.SearchResults) error {
		if !merr.Ok(chunk.GetStatus()) {
			return merr.Error(chunk.GetStatus())
		}
		decoded, err := decodeChunk(chunk, s.nq, s.topK)
		if err != nil {
			return err
		}
		b := &s.childBuffers[i]
		if !b.first {
			b.first = true
			mergeSearchMetadata(out, chunk)
			if decoded != nil {
				numResults++
				if len(decoded.GetScores()) == 0 {
					emptyResult = decoded
				}
				data.AllSearchCount += decoded.AllSearchCount
				if len(data.FieldsData) == 0 && len(decoded.FieldsData) != 0 {
					data.FieldsData = typeutil.PrepareResultFieldData(decoded.FieldsData, 0)
				}
			}
		}
		b.accept(decoded)
		return nil
	}
	for q := 0; q < int(s.nq); q++ {
		hasUnit := func(i int) bool {
			b := &s.childBuffers[i]
			// Skip surplus hits for earlier query vectors before looking for
			// the next head. If this exhausts a Chunk, receive another one.
			for b.data != nil && b.query < q {
				b.row = b.queryEnd
				b.advanceQuery()
			}
			return b.data != nil
		}
		for data.Topks[q] < s.topK {
			if err := s.childStreams.ReceiveUntilReady(hasUnit, accept); err != nil {
				return nil, err
			}
			best := -1
			for i := range s.childBuffers {
				b := &s.childBuffers[i]
				if b.data != nil && b.query == q && (best == -1 || b.precedes(&s.childBuffers[best])) {
					best = i
				}
			}
			if best == -1 {
				break
			}
			b := &s.childBuffers[best]
			outputBytes += appendSearchHit(data, b.data, q, b.row, b.fields.Compute(b.row))
			b.row++
			b.advanceQuery()
		}
		// Batch skips the quota check for a single data-bearing leader with
		// no offset. Metadata-only responses do not count as data-bearing.
		if numResults > 1 && outputBytes > s.maxOutputBytes {
			return nil, merr.WrapErrParameterInvalidMsg("search results exceed the maxOutputSize Limit %d", s.maxOutputBytes)
		}
	}
	if numResults != 0 {
		out.ResultData = data
		if numResults == 1 && emptyResult != nil {
			// Batch passes a single data-bearing leader result through, even
			// if it contains no hits and no typed ID list.
			out.ResultData = emptyResult
		}
	}
	return out, nil
}

func (b *orderedChildBuffer) precedes(other *orderedChildBuffer) bool {
	score, otherScore := b.data.Scores[b.row], other.data.Scores[other.row]
	if score != otherScore {
		return score > otherScore
	}
	return typeutil.ComparePK(typeutil.GetPK(b.data.Ids, b.row), typeutil.GetPK(other.data.Ids, other.row))
}

func mergeSearchMetadata(out, chunk *internalpb.SearchResults) {
	if out.MetricType == "" {
		out.MetricType = chunk.GetMetricType()
	}
	out.IsTopkReduce = out.IsTopkReduce || chunk.GetIsTopkReduce()
	out.IsRecallEvaluation = out.IsRecallEvaluation || chunk.GetIsRecallEvaluation()
	out.ScannedRemoteBytes += chunk.GetScannedRemoteBytes()
	out.ScannedTotalBytes += chunk.GetScannedTotalBytes()
	if cost := chunk.GetCostAggregation(); cost != nil {
		if out.CostAggregation == nil {
			out.CostAggregation = &internalpb.CostAggregation{}
		}
		out.CostAggregation.TotalRelatedDataSize += cost.GetTotalRelatedDataSize()
	}
	if len(chunk.GetChannelsMvcc()) > 0 && out.ChannelsMvcc == nil {
		out.ChannelsMvcc = make(map[string]uint64)
	}
	for ch, ts := range chunk.GetChannelsMvcc() {
		out.ChannelsMvcc[ch] = ts
	}
}

func (s *OrderedReduceStream) Close() error {
	s.childStreams.Cancel()
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	s.finished = true
	clear(s.childBuffers)
	return s.childStreams.Close()
}

func (s *OrderedReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}
