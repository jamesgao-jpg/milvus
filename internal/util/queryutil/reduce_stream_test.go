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
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/internal/util/reduce/orderby"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type testQueryChunks struct {
	chunks []*internalpb.RetrieveResults
	closed bool
}

func (s *testQueryChunks) Recv() (*internalpb.RetrieveResults, error) {
	if len(s.chunks) == 0 {
		return nil, io.EOF
	}
	chunk := s.chunks[0]
	s.chunks = s.chunks[1:]
	return chunk, nil
}

func (s *testQueryChunks) Close() error {
	s.closed = true
	return nil
}

func (s *testQueryChunks) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, merr.ErrServiceUnimplemented
}

func splitTestQuery(t *testing.T, result *internalpb.RetrieveResults) *testQueryChunks {
	t.Helper()
	stream := &testQueryChunks{}
	require.NoError(t, SplitRetrieveResult(result, 1, func(chunk *internalpb.RetrieveResults) error {
		require.LessOrEqual(t, typeutil.GetSizeOfIDs(chunk.GetIds()), 1)
		stream.chunks = append(stream.chunks, chunk)
		return nil
	}))
	return stream
}

func queryStreamResult(ids, timestamps []int64) *internalpb.RetrieveResults {
	return &internalpb.RetrieveResults{
		Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}}},
		FieldsData: []*schemapb.FieldData{
			{FieldId: 1, Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: timestamps}}},
			}},
		},
		AllRetrieveCount:   100,
		ScannedRemoteBytes: 10,
	}
}

func TestReduceRetrieveStreamsBatchParity(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()
	for _, tc := range []struct {
		name      string
		limit     int64
		shard     bool
		duplicate bool
		iterator  bool
	}{
		{name: "shard_limit", limit: 2, shard: true, duplicate: true},
		{name: "shard_unlimited", limit: -1, shard: true, duplicate: true},
		{name: "shard_element_count", limit: 3, shard: true, duplicate: true},
		{name: "proxy_unlimited", limit: -1},
		{name: "proxy_duplicate_error", limit: -1, duplicate: true},
		{name: "proxy_iterator_stops_at_exhausted_child", limit: -1, iterator: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			left := queryStreamResult([]int64{1, 4, 6}, []int64{10, 11, 12})
			right := queryStreamResult([]int64{2, 3, 5}, []int64{13, 14, 15})
			if tc.duplicate {
				right.Ids.GetIntId().Data[1] = 4
			}
			if tc.name == "shard_element_count" {
				for _, r := range []*internalpb.RetrieveResults{left, right} {
					r.ElementLevel = true
					r.ElementIndices = []*internalpb.ElementIndices{
						{Indices: []int32{0, 1}}, {Indices: []int32{2}}, {Indices: []int32{3}},
					}
				}
			}
			results := []*internalpb.RetrieveResults{left, right}
			req := &internalpb.RetrieveRequest{Limit: tc.limit}
			if tc.iterator {
				req.ReduceType = int32(reduce.IReduceInOrder)
				right.HasMoreResult = true
			}
			streams := []ReduceStream{splitTestQuery(t, left), splitTestQuery(t, right)}
			reducer := NewOrderedReduceStream(req, nil, streams, tc.shard, 1<<30)
			got, err := reducer.Recv()
			require.NoError(t, reducer.Close())
			for _, stream := range streams {
				require.True(t, stream.(*testQueryChunks).closed)
			}
			reduceType := reduce.ToReduceType(req.GetReduceType())
			var op Operator = NewSortAndCheckPKOperator(reduceType, nil)
			if tc.shard {
				op = NewReduceByPKWithTimestampOperator(reduceType, 0, tc.limit, nil)
			}
			want, wantErr := op.Run(ctx, nil, results)
			if wantErr != nil {
				require.ErrorIs(t, err, merr.ErrDataIntegrity)
				return
			}
			require.NoError(t, err)
			require.EqualValues(t, 200, got.GetAllRetrieveCount())
			require.EqualValues(t, 20, got.GetScannedRemoteBytes())
			wantResult := want[0].(*internalpb.RetrieveResults)
			require.True(t, proto.Equal(wantResult.GetIds(), got.GetIds()))
			require.Equal(t, wantResult.GetHasMoreResult(), got.GetHasMoreResult())
			require.True(t, proto.Equal(wantResult.GetFieldsData()[0], got.GetFieldsData()[0]))
			require.Equal(t, wantResult.GetElementLevel(), got.GetElementLevel())
			require.Len(t, got.GetElementIndices(), len(wantResult.GetElementIndices()))
			for i, indices := range wantResult.GetElementIndices() {
				require.True(t, proto.Equal(indices, got.GetElementIndices()[i]))
			}
		})
	}
}

func TestQueryGroupsAndOrderingAfterStreamDrain(t *testing.T) {
	paramtable.Init()
	ctx := t.Context()
	const groupID = 100
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: groupID, DataType: schemapb.DataType_Int64}}}
	column := func(id int64, values ...int64) *schemapb.FieldData {
		fd := queryStreamResult(nil, values).GetFieldsData()[0]
		fd.FieldId = id
		return fd
	}
	batch := []*internalpb.RetrieveResults{
		{FieldsData: []*schemapb.FieldData{column(groupID, 7, 8), column(0, 2, 1)}},
		{FieldsData: []*schemapb.FieldData{column(groupID, 7, 9), column(0, 5, 4)}},
	}
	streams := []ReduceStream{splitTestQuery(t, batch[0]), splitTestQuery(t, batch[1])}
	parts, err := DrainRetrieveStreams(ctx, streams)
	require.NoError(t, err)
	require.Len(t, parts, 4)
	run := func(inputs []*internalpb.RetrieveResults) *internalpb.RetrieveResults {
		t.Helper()
		op := NewDeduplicateByGroupsOperator(schema, []int64{groupID}, []*planpb.Aggregate{{Op: planpb.AggregateOp_count}}, -1)
		out, err := op.Run(ctx, nil, inputs)
		require.NoError(t, err)
		sort := NewOrderByLimitOperatorWithPositions([]*orderby.OrderByField{{FieldID: groupID, DataType: schemapb.DataType_Int64, Ascending: true}}, []int{0}, 2)
		result, err := sort.Run(ctx, nil, out...)
		require.NoError(t, err)
		return result[0].(*internalpb.RetrieveResults)
	}
	expected, actual := run(batch), run(parts)
	require.True(t, proto.Equal(expected, actual), "Batch %v; stream %v", expected, actual)
	require.Equal(t, []int64{7, 1}, actual.GetFieldsData()[1].GetScalars().GetLongData().GetData())
}

func TestOrderedQueryResolvesTrailingVersionBeforeEmitting(t *testing.T) {
	paramtable.Init()
	for _, limit := range []int64{1, -1} {
		child := splitTestQuery(t, queryStreamResult([]int64{7, 7, 9}, []int64{10, 30, 5}))
		stream := NewOrderedReduceStream(&internalpb.RetrieveRequest{Limit: limit}, nil, []ReduceStream{child}, true, 1)
		first, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, []int64{7}, first.GetIds().GetIntId().GetData())
		require.Equal(t, []int64{30}, first.GetFieldsData()[0].GetScalars().GetLongData().GetData())
		if limit < 0 {
			second, err := stream.Recv()
			require.NoError(t, err)
			require.Equal(t, []int64{9}, second.GetIds().GetIntId().GetData())
			require.Zero(t, second.GetAllRetrieveCount())
		}
		_, err = stream.Recv()
		require.ErrorIs(t, err, io.EOF)
		require.NoError(t, stream.Close())
	}
}

func TestOrderedQueryOversizedRowIsEmittedAlone(t *testing.T) {
	paramtable.Init()
	ids := []string{"a", strings.Repeat("z", 256)}
	input := queryStreamResult(nil, []int64{3, 4})
	input.Ids = &schemapb.IDs{IdField: &schemapb.IDs_StrId{StrId: &schemapb.StringArray{Data: ids}}}
	stream := NewOrderedReduceStream(&internalpb.RetrieveRequest{Limit: 2}, nil,
		[]ReduceStream{&testQueryChunks{chunks: []*internalpb.RetrieveResults{input}}}, true, 96)
	defer stream.Close()
	for _, id := range ids {
		chunk, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, []string{id}, chunk.GetIds().GetStrId().GetData())
	}
	_, err := stream.Recv()
	require.ErrorIs(t, err, io.EOF)
}
