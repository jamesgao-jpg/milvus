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

package queryutil

import (
	"context"
	"io"
	"sync"

	"github.com/cockroachdb/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type ReduceStream = reduce.ReduceStream[*internalpb.RetrieveResults]

func SupportsQueryReduceStream(req *internalpb.RetrieveRequest) bool {
	// Proxy normalizes COUNT(*) to an aggregate before routing.
	return req != nil && (!req.GetIsCount() || len(req.GetAggregates()) != 0)
}

func IsPKOrderedQuery(req *internalpb.RetrieveRequest) bool {
	return len(req.GetAggregates()) == 0 && len(req.GetGroupByFieldIds()) == 0 && len(req.GetOrderByFields()) == 0
}

// SplitRetrieveResult sends a worker or delegator result in row-aligned chunks.
// Request metadata belongs to the first chunk; HasMoreResult is not a transport EOF.
func SplitRetrieveResult(result *internalpb.RetrieveResults, chunkBytes int, send func(*internalpb.RetrieveResults) error) error {
	if !merr.Ok(result.GetStatus()) {
		return send(result)
	}
	rows := typeutil.GetSizeOfIDs(result.GetIds())
	grouped := rows == 0 && len(result.GetFieldsData()) > 0
	if grouped {
		first := result.GetFieldsData()[0]
		field, err := agg.NewFieldAccessor(first.GetType())
		if err == nil {
			field.SetVals(first)
			rows = field.RowCount()
		}
	}
	if rows == 0 {
		return send(result)
	}
	sizes := newRowSizeCalculator(result)
	for start := 0; start < rows; {
		end, bytes := start, int64(0)
		for end < rows && (end == start || bytes < int64(chunkBytes)) {
			bytes += sizes.rowSize(int64(end))
			end++
		}
		chunk, err := rangeSliceRetrieveResults(result, start, end)
		if err != nil {
			return err
		}
		if grouped {
			chunk.Ids = nil
		}
		chunk.Status = merr.Success()
		if start == 0 {
			chunk.Base = result.GetBase()
			chunk.CostAggregation = result.GetCostAggregation()
			chunk.AllRetrieveCount = result.GetAllRetrieveCount()
			chunk.ScannedRemoteBytes = result.GetScannedRemoteBytes()
			chunk.ScannedTotalBytes = result.GetScannedTotalBytes()
			chunk.MvccTimestamp = result.GetMvccTimestamp()
			chunk.HasMoreResult = result.GetHasMoreResult()
		}
		if err := send(chunk); err != nil {
			return err
		}
		start = end
	}
	return nil
}

type queryRPCStream struct {
	rpc    querypb.QueryNode_QueryReduceStreamClient
	first  *internalpb.RetrieveResults
	cancel context.CancelFunc
}

// OpenQueryReduceStream includes the first chunk in setup so an older peer or
// a failed leader can be handled before the parent reducer sees any rows.
func OpenQueryReduceStream(ctx context.Context, client querypb.QueryNodeClient, req *querypb.QueryRequest, chunkBytes int32, worker bool) (*internalpb.RetrieveResults, ReduceStream, error) {
	rpcCtx, cancel := context.WithCancel(ctx)
	rpc, err := client.QueryReduceStream(rpcCtx)
	if err == nil {
		err = rpc.Send(&querypb.QueryReduceStreamRequest{Request: req, ChunkBytes: chunkBytes, Worker: worker})
	}
	if err == nil {
		err = rpc.CloseSend()
	}
	var first *internalpb.RetrieveResults
	if err == nil || (rpc != nil && errors.Is(err, io.EOF)) {
		first, err = rpc.Recv()
	}
	if status.Code(err) == codes.Unimplemented {
		cancel()
		rpc = nil
		if worker {
			first, err = client.QuerySegments(ctx, req)
		} else {
			first, err = client.Query(ctx, req)
		}
	}
	if err == nil && first == nil {
		err = merr.WrapErrServiceInternalMsg("peer returned no Query result")
	} else if errors.Is(err, io.EOF) {
		err = merr.WrapErrServiceInternalMsg("QueryReduceStream ended without a result")
	}
	if err != nil || !merr.Ok(first.GetStatus()) {
		cancel()
		return first, nil, err
	}
	return first, &queryRPCStream{rpc: rpc, first: first, cancel: cancel}, nil
}

func (s *queryRPCStream) Recv() (*internalpb.RetrieveResults, error) {
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

func (s *queryRPCStream) Close() error {
	s.cancel()
	return nil
}

func (s *queryRPCStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, merr.ErrServiceUnimplemented
}

type queryChildBuffer struct {
	*timestampedResult
	sizes   *rowSizeCalculator
	pos     int64
	hasMore bool // Worker pagination metadata, not the availability of another Chunk.
	hadRows bool
}

type queryReduceUnit struct {
	source     *internalpb.RetrieveResults
	row, bytes int64
}

// OrderedReduceStream retains the child heads between output Chunks. A shard
// resolves versions of one PK before exposing it; Proxy rejects duplicate PKs.
type OrderedReduceStream struct {
	mu                    sync.Mutex                                        // Serializes Recv with cleanup.
	children              *reduce.ChildStreams[*internalpb.RetrieveResults] // Owns worker or vchannel streams.
	buffers               []queryChildBuffer                                // One cursor per input.
	schema                *schemapb.CollectionSchema                        // Materialization schema for nullable fields.
	shard                 bool                                              // Enables timestamp deduplication and the shard quota.
	stopWhenDrained       bool                                              // Iterator boundary for an incomplete exhausted child.
	chunkBytes            int64                                             // Target output Chunk size.
	metadata              *internalpb.RetrieveResults                       // Worker metadata, carried once.
	metadataSent          bool                                              // Empty results also require a successful first response.
	hasRows, elementLevel bool                                              // Check row and element modes across Chunk boundaries.
	pending               queryReduceUnit                                   // A resolved oversized row must start its own Chunk.
	lastPK                any                                               // Detect duplicate or regressing final input across Recv calls.
	remaining             int64                                             // Positive shard candidate window, measured in rows or elements.
	outputBytes           int64                                             // Shard output quota across all emitted Chunks.
	maxOutputBytes        int64                                             // Captured output quota.
	finished              bool                                              // No further rows can be selected.
}

func NewOrderedReduceStream(req *internalpb.RetrieveRequest, schema *schemapb.CollectionSchema, streams []ReduceStream, shard bool, chunkBytes int64) *OrderedReduceStream {
	return &OrderedReduceStream{children: reduce.NewChildStreams(streams), buffers: make([]queryChildBuffer, len(streams)),
		schema: schema, shard: shard, remaining: req.GetLimit(), chunkBytes: chunkBytes, metadata: &internalpb.RetrieveResults{CostAggregation: &internalpb.CostAggregation{}},
		stopWhenDrained: reduce.ShouldStopWhenDrained(reduce.ToReduceType(req.GetReduceType())),
		maxOutputBytes:  paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()}
}

func (s *OrderedReduceStream) accept(i int, chunk *internalpb.RetrieveResults) error {
	meta, buffer := s.metadata, s.buffers
	if chunk == nil {
		return merr.WrapErrServiceInternalMsg("nil Query chunk")
	}
	if err := merr.Error(chunk.GetStatus()); err != nil {
		return err
	}
	cost := meta.CostAggregation
	if incoming := chunk.GetCostAggregation(); incoming != nil {
		cost.TotalRelatedDataSize += incoming.GetTotalRelatedDataSize()
		if !s.shard || paramtable.Get().QueryNodeCfg.EnableWorkerSQCostMetrics.GetAsBool() || chunk.GetBase().GetSourceID() == paramtable.GetNodeID() {
			if incoming.GetResponseTime() > cost.GetResponseTime() {
				total := cost.GetTotalRelatedDataSize()
				meta.CostAggregation = &internalpb.CostAggregation{
					ResponseTime: incoming.GetResponseTime(), ServiceTime: incoming.GetServiceTime(),
					TotalNQ: incoming.GetTotalNQ(), TotalRelatedDataSize: total,
				}
			}
		}
	}
	meta.AllRetrieveCount += chunk.GetAllRetrieveCount()
	meta.ScannedRemoteBytes += chunk.GetScannedRemoteBytes()
	meta.ScannedTotalBytes += chunk.GetScannedTotalBytes()
	meta.HasMoreResult = meta.HasMoreResult || chunk.GetHasMoreResult()
	buffer[i].hasMore = buffer[i].hasMore || chunk.GetHasMoreResult()
	rows := typeutil.GetSizeOfIDs(chunk.GetIds())
	if chunk.GetElementLevel() && len(chunk.GetElementIndices()) != rows {
		return merr.WrapErrServiceInternalMsg("Query chunk element indices do not match the PK count")
	}
	if rows != 0 {
		if len(chunk.GetFieldsData()) == 0 {
			return merr.WrapErrServiceInternalMsg("Query chunk has IDs without fields")
		}
		if s.hasRows && s.elementLevel != chunk.GetElementLevel() {
			return merr.WrapErrServiceInternalMsg("Query streams mix row and element results")
		}
		s.hasRows, s.elementLevel = true, chunk.GetElementLevel()
		ts, _ := newTimestampedResult(chunk)
		if ts == nil {
			ts = &timestampedResult{result: chunk}
		}
		buffer[i].timestampedResult, buffer[i].pos, buffer[i].hadRows = ts, 0, true
		buffer[i].sizes = newRowSizeCalculator(chunk)
	}
	return nil
}

func (s *OrderedReduceStream) head() (int, error) {
	if err := s.children.ReceiveUntilReady(func(i int) bool { return s.buffers[i].timestampedResult != nil }, s.accept); err != nil {
		return -1, err
	}
	best := -1
	var pk any
	var ts int64
	for i, b := range s.buffers {
		if b.timestampedResult == nil {
			if s.stopWhenDrained && b.hasMore && b.hadRows {
				return -1, nil
			}
			continue
		}
		key := typeutil.GetPK(b.result.GetIds(), b.pos)
		stamp := b.getTimestamp(b.pos)
		if best < 0 || typeutil.ComparePK(key, pk) || (s.shard && key == pk && stamp > ts) {
			best, pk, ts = i, key, stamp
		}
	}
	return best, nil
}

func (b *queryChildBuffer) pop() {
	b.pos++
	if b.pos == int64(typeutil.GetSizeOfIDs(b.result.GetIds())) {
		b.timestampedResult = nil
		b.sizes = nil
	}
}

// produceNextUnit looks past a shard PK's last occurrence before selecting its final row.
func (s *OrderedReduceStream) produceNextUnit() (queryReduceUnit, error) {
	i, err := s.head()
	if err != nil {
		return queryReduceUnit{}, err
	}
	if i < 0 {
		s.finished = true
		return queryReduceUnit{}, nil
	}
	b := &s.buffers[i]
	src, row, ts := b.result, b.pos, b.getTimestamp(b.pos)
	pk := typeutil.GetPK(src.GetIds(), row)
	if s.lastPK != nil && !typeutil.ComparePK(s.lastPK, pk) {
		return queryReduceUnit{}, merr.WrapErrDataIntegrityMsg("duplicate or unordered PK %v in Query stream", pk)
	}
	fieldBytes := b.sizes.rowSize(row)
	b.pop()
	// A newer version of the trailing PK can be in the next worker Chunk.
	// Do not send the row, even at a Chunk boundary, until its PK is final.
	if s.shard {
		for {
			j, err := s.head()
			if err != nil {
				return queryReduceUnit{}, err
			}
			if j < 0 {
				s.finished = true
				break
			}
			next := &s.buffers[j]
			if typeutil.GetPK(next.result.GetIds(), next.pos) != pk {
				break
			}
			if newer := next.getTimestamp(next.pos); newer > ts {
				src, row, ts = next.result, next.pos, newer
				fieldBytes = next.sizes.rowSize(row)
			}
			next.pop()
		}
		s.outputBytes += fieldBytes
		if s.maxOutputBytes > 0 && s.outputBytes > s.maxOutputBytes {
			return queryReduceUnit{}, merr.WrapErrParameterInvalidMsg("query results exceed the maxOutputSize Limit %d", s.maxOutputBytes)
		}
	}
	s.lastPK = pk
	idBytes, err := reduce.IDsRowSize(src.GetIds(), row)
	if err != nil {
		return queryReduceUnit{}, err
	}
	bytes := fieldBytes + int64(idBytes)
	if s.elementLevel {
		bytes += 4 * int64(len(src.GetElementIndices()[row].GetIndices()))
	}
	if s.shard && s.remaining > 0 {
		n := int64(1)
		if s.elementLevel {
			n = int64(len(src.GetElementIndices()[row].GetIndices()))
		}
		s.remaining -= n
		if s.remaining <= 0 {
			s.finished = true
		}
	}
	return queryReduceUnit{source: src, row: row, bytes: bytes}, nil
}

func (s *OrderedReduceStream) Recv() (*internalpb.RetrieveResults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished && s.pending.source == nil {
		return nil, io.EOF
	}
	var sources []*internalpb.RetrieveResults
	indices := make(map[*internalpb.RetrieveResults]int)
	var rows []rowRef
	var bytes int64
	for bytes < s.chunkBytes || len(rows) == 0 {
		unit := s.pending
		if unit.source == nil {
			if s.finished {
				break
			}
			var err error
			unit, err = s.produceNextUnit()
			if err != nil {
				return nil, err
			}
			if unit.source == nil {
				break
			}
			if len(rows) > 0 && unit.bytes > s.chunkBytes {
				s.pending = unit
				break
			}
		} else {
			s.pending = queryReduceUnit{}
		}
		src, row := unit.source, unit.row
		index, ok := indices[src]
		if !ok {
			index = len(sources)
			sources = append(sources, src)
			indices[src] = index
		}
		rows = append(rows, rowRef{resultIdx: index, rowIdx: row})
		bytes += unit.bytes
	}
	if len(rows) == 0 && s.metadataSent {
		return nil, io.EOF
	}
	result, err := buildMergedRetrieveResults(sources, rows, s.schema)
	if err != nil {
		return nil, err
	}
	result.Status = merr.Success()
	if !s.metadataSent {
		result.CostAggregation = s.metadata.CostAggregation
		result.AllRetrieveCount = s.metadata.AllRetrieveCount
		result.ScannedRemoteBytes = s.metadata.ScannedRemoteBytes
		result.ScannedTotalBytes = s.metadata.ScannedTotalBytes
		result.HasMoreResult = s.metadata.HasMoreResult
		s.metadataSent = true
	}
	return result, nil
}

func (s *OrderedReduceStream) Close() error {
	s.children.Cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finished = true
	clear(s.buffers)
	s.pending = queryReduceUnit{}
	return s.children.Close()
}

func (s *OrderedReduceStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, merr.ErrServiceUnimplemented
}

// DrainRetrieveStreams retains full inputs for the existing ORDER BY and
// aggregate reducers. A result cannot be finalized before all children drain.
func DrainRetrieveStreams(ctx context.Context, streams []ReduceStream) (_ []*internalpb.RetrieveResults, err error) {
	children := reduce.NewChildStreams(streams)
	defer func() {
		if closeErr := children.Close(); closeErr != nil {
			if err == nil {
				err = closeErr
			} else {
				mlog.Warn(ctx, "failed to close Query ReduceStream children", mlog.Err(closeErr))
			}
		}
	}()
	inputs := make([][]*internalpb.RetrieveResults, len(streams))
	err = children.ReceiveUntilReady(func(int) bool { return false }, func(i int, chunk *internalpb.RetrieveResults) error {
		if chunk == nil {
			return merr.WrapErrServiceInternalMsg("nil Query chunk")
		}
		if err := merr.Error(chunk.GetStatus()); err != nil {
			return err
		}
		inputs[i] = append(inputs[i], chunk)
		return nil
	})
	if err != nil {
		return nil, err
	}
	var results []*internalpb.RetrieveResults
	for _, input := range inputs {
		results = append(results, input...)
	}
	return results, nil
}
