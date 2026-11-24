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
	"slices"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/util/reduce/orderby"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type orderedGroups struct {
	running   *agg.AggregationResult  // Global aggregate only: one partial for all consumed Chunks.
	reducer   *agg.GroupAggReducer    // Validates partial columns and materializes typed empty results.
	order     []*orderby.OrderByField // Requested order followed by all remaining group keys.
	positions []int                   // Key positions in the positional partial aggregate layout.
	ops       []agg.AggregateBase     // Partial aggregate operations; AVG is already split into SUM and COUNT.
	produced  int64                   // Complete groups, including those skipped by the client offset.
	limit     int64                   // Internal offset + limit; nonpositive means unbounded.
	maxGroups int64                   // Enforced for groups visited, without scanning an unread tail.
}

func newOrderedGroups(req *internalpb.RetrieveRequest, schema *schemapb.CollectionSchema) (*orderedGroups, error) {
	if schema == nil {
		return nil, merr.WrapErrServiceInternalMsg("aggregate Query requires a schema")
	}
	keys := req.GetGroupByFieldIds()
	explicit, err := orderby.ConvertFromPlanOrderByFields(req.GetOrderByFields(), schema)
	if err != nil {
		return nil, err
	}
	// Like Batch, GROUP BY without ORDER BY compares the group keys in declaration
	// order (ascending, NULLs last). The k-way merge must match the workers' top K.
	order, err := orderby.GroupKeyOrder(explicit, keys, schema)
	if err != nil {
		return nil, err
	}
	g := &orderedGroups{
		reducer: agg.NewGroupAggReducer(keys, req.GetAggregates(), typeutil.Unlimited, schema),
		order:   order, limit: req.GetLimit(),
		maxGroups: paramtable.Get().CommonCfg.GroupByMaxGroups.GetAsInt64(),
	}
	for _, f := range g.order {
		pos := slices.Index(keys, f.FieldID)
		if pos < 0 {
			return nil, merr.WrapErrParameterInvalidMsg("ordered GROUP BY requires a grouping key, got %d", f.FieldID)
		}
		g.positions = append(g.positions, pos)
	}
	for _, def := range req.GetAggregates() {
		op, err := agg.FromPB(def)
		if err != nil {
			return nil, err
		}
		g.ops = append(g.ops, op)
	}
	return g, nil
}

func (g *orderedGroups) compare(a, b queryReduceUnit) int {
	for i, pos := range g.positions {
		left, right := a.source.FieldsData[pos], b.source.FieldsData[pos]
		order := g.order[i]
		if c := compareOrderByValues(left, int(a.row), right, int(b.row), left.GetType(), order.Ascending, order.NullsFirst); c != 0 {
			return c
		}
	}
	return 0
}

func (g *orderedGroups) accept(b *queryChildBuffer, chunk *internalpb.RetrieveResults) error {
	if len(chunk.FieldsData) == 0 { // Metadata-only Chunk.
		return nil
	}
	partial := agg.NewAggregationResult(chunk.FieldsData, 0)
	columns := make([]agg.FieldAccessor, len(chunk.FieldsData))
	for i, fd := range chunk.FieldsData {
		col, err := agg.NewFieldAccessor(fd.GetType())
		if err != nil {
			return err
		}
		col.SetVals(fd)
		if n := len(typeutil.GetFieldDataValidData(fd)); n != 0 && n != col.RowCount() {
			return merr.WrapErrServiceInternalMsg("aggregate column %d has %d validity entries for %d rows", i, n, col.RowCount())
		}
		columns[i] = col
		if col.RowCount() != columns[0].RowCount() {
			return merr.WrapErrServiceInternalMsg("aggregate partial has inconsistent column lengths")
		}
	}
	if len(g.order) == 0 {
		if n := columns[0].RowCount(); n == 0 {
			return nil
		} else if n != 1 {
			return merr.WrapErrServiceInternalMsg("global aggregate partial has %d rows, expected one", n)
		}
		inputs := []*agg.AggregationResult{partial}
		if g.running != nil {
			inputs = []*agg.AggregationResult{g.running, partial}
		}
		var err error
		g.running, err = g.reducer.Reduce(context.Background(), inputs)
		return err
	}
	// The Batch reducer validates the positional layout without copying a single partial.
	if _, err := g.reducer.Reduce(context.Background(), []*agg.AggregationResult{partial}); err != nil {
		return err
	}
	if columns[0].RowCount() != 0 {
		b.timestampedResult = &timestampedResult{result: chunk}
		b.groupColumns, b.pos = columns, 0
	}
	return nil
}

func (b *queryChildBuffer) groupHead() queryReduceUnit {
	return queryReduceUnit{source: b.result, row: b.pos}
}

// Each child has already merged its own equal keys. Once all live heads
// are known, all contributions to the least key are at those heads, even
// when the next Chunk is unavailable. No lookahead past the last group is needed.
func (g *orderedGroups) appendNext(ready []*queryChildBuffer, out *internalpb.RetrieveResults) error {
	first := ready[0].groupHead()
	for _, b := range ready[1:] {
		if candidate := b.groupHead(); g.compare(candidate, first) < 0 {
			first = candidate
		}
	}
	var merged *agg.Row
	for _, b := range ready {
		if g.compare(b.groupHead(), first) != 0 {
			continue
		}
		value := func(column int) *agg.FieldValue {
			a := b.groupColumns[column]
			if a.IsNullAt(int(b.pos)) {
				return agg.NewNullFieldValue()
			}
			return agg.NewFieldValue(a.ValAt(int(b.pos)))
		}
		if merged == nil {
			values := make([]*agg.FieldValue, len(b.groupColumns))
			for i := range values {
				values[i] = value(i)
			}
			merged = agg.NewRow(values)
		} else {
			for i, op := range g.ops {
				col := len(b.groupColumns) - len(g.ops) + i
				op.Update(merged.FieldValueAt(col), value(col))
			}
		}
		b.pos++
		if int(b.pos) == b.groupColumns[0].RowCount() {
			b.timestampedResult, b.groupColumns = nil, nil
		}
	}
	if out.FieldsData == nil {
		out.FieldsData = typeutil.PrepareResultFieldData(first.source.FieldsData, 0)
	}
	return agg.AssembleSingleRow(len(out.FieldsData), merged, out.FieldsData)
}

func (s *OrderedReduceStream) recvGroups() (*internalpb.RetrieveResults, error) {
	g := s.groups
	if len(g.order) == 0 {
		// Unlike a group key, a global value is not final until every child reaches EOF.
		if err := s.children.ReceiveUntilReady(func(int) bool { return false }, s.accept); err != nil {
			return nil, err
		}
		if g.running == nil {
			empty, err := g.reducer.EmptyAggResult()
			if err != nil {
				return nil, err
			}
			g.running = empty
		}
		out := agg.AggResult2internalResult(g.running)
		g.running, s.finished = nil, true
		s.attachMetadata(out)
		return out, nil
	}

	out := &internalpb.RetrieveResults{}
	var size int64
	var sizes *rowSizeCalculator
	ready := make([]*queryChildBuffer, 0, len(s.buffers))
	if len(s.orderByChunks) > 0 {
		out, s.orderByChunks = s.orderByChunks[0], nil
		size = s.chunkBytes // Emit the pending oversized group alone.
	}
	for (out.FieldsData == nil || size < s.chunkBytes) && (g.limit <= 0 || g.produced < g.limit) {
		err := s.children.ReceiveUntilReady(func(i int) bool { return s.buffers[i].groupColumns != nil }, s.accept)
		if err != nil {
			return nil, err
		}
		ready = ready[:0]
		for i := range s.buffers {
			if s.buffers[i].groupColumns != nil {
				ready = append(ready, &s.buffers[i])
			}
		}
		if len(ready) == 0 {
			s.finished = true
			break
		}
		g.produced++
		if g.produced > g.maxGroups {
			return nil, merr.WrapErrParameterInvalidMsg("GROUP BY produced too many groups (%d). Add filters or increase common.groupBy.maxGroups (current: %d)", g.produced, g.maxGroups)
		}
		if err := g.appendNext(ready, out); err != nil {
			return nil, err
		}
		if sizes == nil {
			sizes = newRowSizeCalculator(out)
		}
		rows := getRowCount(out)
		n := sizes.rowSize(int64(rows - 1))
		if n > s.chunkBytes && rows > 1 {
			last, err := rangeSliceRetrieveResults(out, rows-1, rows)
			if err != nil {
				return nil, err
			}
			out, err = rangeSliceRetrieveResults(out, 0, rows-1)
			if err != nil {
				return nil, err
			}
			s.orderByChunks = []*internalpb.RetrieveResults{last}
			break
		}
		size += n
	}
	if g.limit > 0 && g.produced == g.limit && len(s.orderByChunks) == 0 {
		s.finished = true
	}
	if out.FieldsData == nil {
		if s.metadataSent {
			return nil, io.EOF
		}
		empty, err := g.reducer.EmptyAggResult()
		if err != nil {
			return nil, err
		}
		out = agg.AggResult2internalResult(empty)
	}
	s.attachMetadata(out)
	return out, nil
}
