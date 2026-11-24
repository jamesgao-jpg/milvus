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
	"slices"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/resource"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type ReduceStream = reduce.ReduceStream[*internalpb.SearchResults]

// orderedChildBuffer owns one Chunk and the cursor into its query-major hits.
// The same row numbering is used for scoring, PKs, and nullable field data.
type orderedChildBuffer struct {
	data        *schemapb.SearchResultData     // Data for the current Chunk.
	fields      *typeutil.FieldDataIdxComputer // Nullable field offsets for data.
	query       int                            // Current query vector.
	row         int64                          // Current row in data.
	queryEnd    int64                          // Exclusive end of the current query in data.
	first       bool                           // Whether the leader's metadata was received.
	groupValues []func(int) any                // Group columns in request order for the current Chunk.
}

func (b *orderedChildBuffer) advanceQuery() {
	for b.query < len(b.data.GetTopks()) && b.row == b.queryEnd {
		b.query++
		if b.query < len(b.data.GetTopks()) {
			b.queryEnd += b.data.Topks[b.query]
		}
	}
	if b.query == len(b.data.GetTopks()) {
		b.data, b.fields, b.groupValues = nil, nil, nil
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
		if err := resource.UnmarshalSearchResultData(chunk.GetSlicedBlob(), data); err != nil {
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

func groupValueBytes(value any) int {
	if v, ok := value.(string); ok {
		return len(v) + 1
	}
	return 10
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
	groupFields := append([]*schemapb.FieldData{}, data.GetGroupByFieldValues()...)
	singular := data.GetGroupByFieldValue() != nil
	if singular {
		groupFields = append(groupFields, data.GetGroupByFieldValue())
	}
	groupValues := make([]func(int) any, len(groupFields))
	for i, field := range groupFields {
		groupValues[i] = typeutil.GetDataIterator(field)
	}
	start, end := 0, 0
	emit := func() error {
		for i, field := range groupFields {
			builder, err := typeutil.NewFieldDataBuilder(field.GetType(), true, end-start)
			if err != nil {
				return err
			}
			for row := start; row < end; row++ {
				builder.Add(groupValues[i](row))
			}
			group := builder.Build()
			group.FieldId, group.FieldName = field.GetFieldId(), field.GetFieldName()
			// Retain a validity entry even for an all-valid slice of a
			// nullable column so neighboring chunks append consistently.
			if valid := typeutil.GetFieldDataValidData(field); len(valid) != 0 {
				typeutil.SetFieldDataValidData(group, valid[start:end:end])
			}
			if singular && i == len(groupFields)-1 {
				chunk.GroupByFieldValue = group
			} else {
				chunk.GroupByFieldValues = append(chunk.GroupByFieldValues, group)
			}
		}
		start = end
		out := &internalpb.SearchResults{Status: result.Status, MetricType: result.MetricType, NumQueries: nq, TopK: topK}
		if first {
			first = false
			out = result
			out.SlicedBlob, out.SlicedNumCount, out.SlicedOffset = nil, 0, 0
			chunk.AllSearchCount = data.AllSearchCount
		}
		out.ResultData = chunk
		out.NumQueries, out.TopK = nq, topK
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
		for _, valueAt := range groupValues {
			// Group keys only support scalar data. Count their payload without
			// walking a nullable compact vector or rescanning the bitmap.
			bytes += groupValueBytes(valueAt(end))
		}
		if bytes > chunkBytes && size > 0 {
			if err := emit(); err != nil {
				return err
			}
			chunk, size = newSearchData(nq, topK), 0
		}
		appendSearchHit(chunk, data, buffer.query, buffer.row, indexes)
		end++
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

// OrderedReduceStream merges query-major Search results using 3.0's Batch
// score/PK order. Each Recv resumes the merge and returns one output Chunk.
type OrderedReduceStream struct {
	recvMu                sync.Mutex                                      // Serializes Recv and Close cleanup.
	childStreams          *reduce.ChildStreams[*internalpb.SearchResults] // Receives and closes children.
	childBuffers          []orderedChildBuffer                            // One current Chunk per child.
	nq                    int64                                           // Number of query vectors.
	topK                  int64                                           // Candidate window per vector, in hits or distinct groups.
	groupByFieldIDs       []int64                                         // For grouped Search, topK counts distinct keys instead of hits.
	groupSize             int64                                           // Maximum accepted hits per group; one on the plain ANN path.
	groups                map[uint64][]*searchGroup                       // Per-vector group counts across Chunks; check key equality on hash collisions.
	groupCount            int64                                           // Number of distinct groups accepted for this vector.
	offset                int64                                           // Proxy pagination; leading hits do not count against the response quota.
	pkType                schemapb.DataType                               // Preserves typed IDs for empty Batch results.
	maxOutputBytes        int64                                           // Batch quota for a multi-leader merge.
	dedupByPK             bool                                            // Vchannel merge only: deduplicate PKs or (PK, element index).
	chunkBytes            int                                             // Output Chunk payload threshold.
	query                 int                                             // Vector currently being merged.
	emitted               int64                                           // Selected hits for this vector, including offset.
	seen                  map[hitKey]struct{}                             // Vchannel PK deduplication for this vector.
	numResults            int                                             // Number of data-bearing children, for Batch quota semantics.
	outputBytes           int64                                           // Cumulative returned field bytes across Chunks.
	elementLevel, haveHit bool                                            // Prevent mixing element and document hits.
	metadataSent          bool                                            // Even an empty stream reports metadata once.
	finished              bool                                            // Whether final output has already been read.
}

type hitKey struct {
	pk      any
	element int64
}

type searchGroup struct {
	values []any
	count  int64
}

// Look up the group without changing its count. An oversized hit can belong
// to the next output Chunk, so only count it after the sizing check.
func (s *OrderedReduceStream) groupForHit(b *orderedChildBuffer, values []any) (*searchGroup, uint64, int) {
	var size int
	for i, read := range b.groupValues {
		value := read(int(b.row))
		values[i] = reduce.NormalizeScalar(value)
		size += groupValueBytes(value)
	}
	hash := reduce.HashGroupValues(values)
	for _, g := range s.groups[hash] {
		if reduce.EqualGroupValues(g.values, values) {
			if g.count == s.groupSize {
				return nil, hash, size
			}
			return g, hash, size
		}
	}
	if s.groupCount == s.topK {
		return nil, hash, size
	}
	return &searchGroup{values: slices.Clone(values)}, hash, size
}

// HitLimit includes the groups before the offset. The Proxy pipeline applies the offset once.
func (s *OrderedReduceStream) HitLimit() int64 { return s.topK * s.groupSize }

func NewReduceStream(request *internalpb.SearchRequest, streams []ReduceStream, pkType schemapb.DataType, maxOutputBytes int64, chunkBytes int) *OrderedReduceStream {
	s := &OrderedReduceStream{
		childStreams: reduce.NewChildStreams(streams), childBuffers: make([]orderedChildBuffer, len(streams)),
		nq: request.GetNq(), topK: request.GetTopk(), offset: request.GetOffset(), chunkBytes: chunkBytes, pkType: pkType, maxOutputBytes: maxOutputBytes, groupSize: 1,
	}
	info := reduce.NewReduceSearchResultInfo(s.nq, s.topK).WithGroupByFieldIdsFromProto(request.GetGroupByFieldId(), request.GetGroupByFieldIds())
	if s.groupByFieldIDs = info.GetGroupByFieldIds(); len(s.groupByFieldIDs) > 0 {
		s.groupSize = max(1, request.GetGroupSize())
		s.groups = make(map[uint64][]*searchGroup)
	}
	return s
}

func NewShardReduceStream(request *internalpb.SearchRequest, streams []ReduceStream, pkType schemapb.DataType, maxOutputBytes int64, chunkBytes int) *OrderedReduceStream {
	s := NewReduceStream(request, streams, pkType, maxOutputBytes, chunkBytes)
	s.dedupByPK = true
	s.offset = 0
	return s
}

func (s *OrderedReduceStream) Recv() (*internalpb.SearchResults, error) {
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	if s.finished {
		return nil, io.EOF
	}

	out := &internalpb.SearchResults{Status: merr.Success(), NumQueries: s.nq, TopK: s.topK}
	data := newSearchData(s.nq, s.topK)
	switch s.pkType {
	case schemapb.DataType_Int64:
		data.Ids.IdField = &schemapb.IDs_IntId{IntId: &schemapb.LongArray{}}
	case schemapb.DataType_VarChar:
		data.Ids.IdField = &schemapb.IDs_StrId{StrId: &schemapb.StringArray{}}
	}
	var chunkSize int
	var groupBuilders []*typeutil.FieldDataBuilder
	var groupNames []string
	groupKey := make([]any, len(s.groupByFieldIDs))
	var emptyResult *schemapb.SearchResultData
	accept := func(i int, chunk *internalpb.SearchResults) error {
		if !merr.Ok(chunk.GetStatus()) {
			return merr.Error(chunk.GetStatus())
		}
		decoded, err := decodeChunk(chunk, s.nq, s.topK)
		if err != nil {
			return err
		}
		if len(decoded.GetScores()) > 0 {
			isElement := decoded.GetElementIndices() != nil
			if s.haveHit && s.elementLevel != isElement {
				return merr.WrapErrServiceInternalMsg("Search streams mix row and element hits")
			}
			s.elementLevel, s.haveHit = isElement, true
		}
		b := &s.childBuffers[i]
		if !b.first {
			b.first = true
			// The delegator's Batch reducer discards worker responses without
			// a data payload; the Proxy reducer still needs leader metadata.
			if !s.dedupByPK || decoded != nil {
				mergeSearchMetadata(out, chunk)
			}
			if decoded != nil {
				s.numResults++
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
		if b.data != nil && s.groups != nil {
			b.groupValues = make([]func(int) any, len(s.groupByFieldIDs))
			for i, id := range s.groupByFieldIDs {
				field := reduce.FindGroupByFieldData(decoded, id, len(s.groupByFieldIDs) == 1)
				if field == nil {
					return merr.WrapErrParameterInvalidMsg("group-by field %d missing from Search Chunk", id)
				}
				b.groupValues[i] = typeutil.GetDataIterator(field)
			}
		}
		return nil
	}
merge:
	for ; s.query < int(s.nq); s.query++ {
		q := s.query
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
		// Skip surplus hits to reach the next vector. For the final vector,
		// Proxy owns the stop decision; shard streams still bound their candidates.
		finalVector := !s.dedupByPK && q == int(s.nq)-1
		for s.emitted < s.HitLimit() || (finalVector && s.groups == nil) {
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
			var key hitKey
			duplicate := false
			if s.dedupByPK && s.numResults > 1 {
				key = hitKey{pk: typeutil.GetPK(b.data.GetIds(), b.row)}
				if s.elementLevel {
					key.element = b.data.GetElementIndices().GetData()[b.row]
				}
				if s.seen == nil {
					s.seen = make(map[hitKey]struct{})
				}
				_, duplicate = s.seen[key]
			}
			if !duplicate {
				indexes := b.fields.Compute(b.row)
				bytes, err := searchUnitBytes(b.data, b.row, indexes)
				if err != nil {
					return nil, err
				}
				var group *searchGroup
				var hash uint64
				if s.groups != nil {
					var groupBytes int
					group, hash, groupBytes = s.groupForHit(b, groupKey)
					if group == nil {
						b.row++
						b.advanceQuery()
						continue
					}
					bytes += groupBytes
				}
				if chunkSize > 0 && bytes > s.chunkBytes {
					break merge
				}
				if group != nil {
					if group.count == 0 {
						s.groups[hash] = append(s.groups[hash], group)
						s.groupCount++
					}
					group.count++
					if groupBuilders == nil {
						for _, id := range s.groupByFieldIDs {
							field := reduce.FindGroupByFieldData(b.data, id, len(s.groupByFieldIDs) == 1)
							builder, err := typeutil.NewFieldDataBuilder(field.GetType(), true, 0)
							if err != nil {
								return nil, err
							}
							groupBuilders = append(groupBuilders, builder)
							groupNames = append(groupNames, field.GetFieldName())
						}
					}
					for i, builder := range groupBuilders {
						builder.Add(b.groupValues[i](int(b.row)))
					}
				}
				if s.seen != nil {
					s.seen[key] = struct{}{}
				}
				size := appendSearchHit(data, b.data, q, b.row, indexes)
				chunkSize += bytes
				s.emitted++
				// Keep the candidate window intact for the existing Proxy
				// pipeline, which applies the offset once.
				if s.emitted > s.offset {
					s.outputBytes += size
				}
			}
			b.row++
			b.advanceQuery()
			// For grouped Search the final Proxy pipeline charges only the groups after OFFSET.
			if (s.groups == nil || s.dedupByPK) && (s.numResults > 1 || s.offset > 0) && s.outputBytes > s.maxOutputBytes {
				return nil, merr.WrapErrParameterInvalidMsg("search results exceed the maxOutputSize Limit %d", s.maxOutputBytes)
			}
			if chunkSize >= s.chunkBytes || (finalVector && s.emitted == s.HitLimit()) {
				break merge
			}
		}
		s.emitted, s.seen = 0, nil
		clear(s.groups)
		s.groupCount = 0
	}
	s.finished = s.query == int(s.nq)
	if s.metadataSent && len(data.Scores) == 0 {
		return nil, io.EOF
	}
	s.metadataSent = true
	for i, builder := range groupBuilders {
		field := builder.Build()
		field.FieldId, field.FieldName = s.groupByFieldIDs[i], groupNames[i]
		// A later Chunk can contain NULL even if this Chunk has only valid keys.
		if len(typeutil.GetFieldDataValidData(field)) == 0 {
			typeutil.SetFieldDataValidData(field, slices.Repeat([]bool{true}, len(data.Scores)))
		}
		data.GroupByFieldValues = append(data.GroupByFieldValues, field)
	}
	if s.numResults != 0 {
		out.ResultData = data
		if s.numResults == 1 && emptyResult != nil {
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
			out.CostAggregation = &internalpb.CostAggregation{
				ResponseTime: cost.GetResponseTime(),
				ServiceTime:  cost.GetServiceTime(),
				TotalNQ:      cost.GetTotalNQ(),
			}
		}
		out.CostAggregation.TotalRelatedDataSize += cost.GetTotalRelatedDataSize()
		if out.CostAggregation.GetResponseTime() < cost.GetResponseTime() {
			out.CostAggregation.ResponseTime = cost.GetResponseTime()
			out.CostAggregation.ServiceTime = cost.GetServiceTime()
			out.CostAggregation.TotalNQ = cost.GetTotalNQ()
		}
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
	s.groups, s.seen = nil, nil
	return s.childStreams.Close()
}

func (s *OrderedReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}
