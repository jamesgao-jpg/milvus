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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/internal/util/reduce/orderby"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
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

// The first Recv returns immediately. The second Recv cannot return until the test releases it.
type blockedQueryTail struct {
	testQueryChunks
	waiting chan struct{}
	release chan struct{}
	calls   int
}

func (b *blockedQueryTail) Recv() (*internalpb.RetrieveResults, error) {
	b.calls++
	if b.calls == 2 {
		close(b.waiting)
		<-b.release
	}
	return b.testQueryChunks.Recv()
}

func TestAggregateStreamsBlockedWorker(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "key", DataType: schemapb.DataType_Int64},
	}}
	for _, grouped := range []bool{true, false} {
		name := "global_waits_for_tail"
		if grouped {
			name = "group_limit_leaves_tail_unread"
		}
		t.Run(name, func(t *testing.T) {
			req := &internalpb.RetrieveRequest{Limit: 1, Aggregates: []*planpb.Aggregate{{Op: planpb.AggregateOp_count}}}
			if grouped {
				req.GroupByFieldIds = []int64{100}
			}
			row := func(key, count int64) *internalpb.RetrieveResults {
				fields := []*schemapb.FieldData{makeInt64Field(0, "count", []int64{count})}
				if grouped {
					fields = append([]*schemapb.FieldData{makeInt64Field(100, "key", []int64{key})}, fields...)
				}
				return &internalpb.RetrieveResults{FieldsData: fields}
			}
			child := func(rows ...*internalpb.RetrieveResults) *testQueryChunks { return &testQueryChunks{chunks: rows} }
			blocked := &blockedQueryTail{testQueryChunks: *child(row(1, 2), row(2, 11)), waiting: make(chan struct{}), release: make(chan struct{})}
			shard := func(inputs ...ReduceStream) ReduceStream {
				return NewOrderedReduceStream(req, schema, inputs, true, 1<<20)
			}
			root := NewOrderedReduceStream(req, schema, []ReduceStream{shard(blocked, child(row(1, 3))), shard(child(row(1, 5)))}, false, 1<<20)
			defer root.Close()
			release := sync.OnceFunc(func() { close(blocked.release) })
			defer release()
			done := make(chan struct{})
			var out *internalpb.RetrieveResults
			var err error
			go func() { defer close(done); out, err = root.Recv() }()
			if grouped {
				select {
				case <-done:
				case <-blocked.waiting:
					t.Fatal("read beyond the complete group window")
				case <-time.After(5 * time.Second):
					t.Fatal("group did not arrive")
				}
			} else {
				select {
				case <-blocked.waiting:
				case <-done:
					t.Fatal("global count returned before reading the last worker partial")
				case <-time.After(5 * time.Second):
					t.Fatal("did not request last worker partial")
				}
				release()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatal("global count did not finish")
				}
			}
			require.NoError(t, err)
			want := int64(21)
			if grouped {
				want = 10
				require.Equal(t, []int64{1}, out.FieldsData[0].GetScalars().GetLongData().Data)
			}
			require.Equal(t, []int64{want}, out.FieldsData[len(out.FieldsData)-1].GetScalars().GetLongData().Data)
			_, err = root.Recv()
			require.ErrorIs(t, err, io.EOF)
		})
	}
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

func TestQueryChunkBoundaries(t *testing.T) {
	paramtable.Init()
	for _, tc := range []struct {
		name      string
		stringPKs []string
		elements  []*internalpb.ElementIndices
		budget    int
		want      []int
	}{
		{"exact_int64", nil, nil, 16, []int{1, 1, 1}},                       // 8 PK bytes + 8 timestamp bytes per row.
		{"raw_string_bytes", []string{"a", "bb", "ccc"}, nil, 22, []int{3}}, // 9 + 10 < 22.
		{"normal_crossing", []string{strings.Repeat("a", 72), strings.Repeat("b", 22), "c"}, nil, 100, []int{2, 1}},
		{"oversized_middle", []string{"a", strings.Repeat("b", 112), "c"}, nil, 100, []int{1, 1, 1}},
		{"element_indices", nil, []*internalpb.ElementIndices{{Indices: []int32{0, 1, 2, 3}}, {Indices: []int32{0}}, {Indices: []int32{1}}}, 51, []int{2, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := queryStreamResult([]int64{1, 2, 3}, []int64{4, 5, 6})
			if tc.stringPKs != nil {
				src.Ids = &schemapb.IDs{IdField: &schemapb.IDs_StrId{StrId: &schemapb.StringArray{Data: tc.stringPKs}}}
			}
			src.ElementLevel, src.ElementIndices = tc.elements != nil, tc.elements
			check := func(t *testing.T, chunks []*internalpb.RetrieveResults) {
				var counts []int
				row := 0
				for _, chunk := range chunks {
					n := typeutil.GetSizeOfIDs(chunk.GetIds())
					counts = append(counts, n)
					for i := range n {
						require.Equal(t, typeutil.GetPK(src.Ids, int64(row)), typeutil.GetPK(chunk.Ids, int64(i)))
						if src.ElementLevel {
							require.True(t, proto.Equal(tc.elements[row], chunk.ElementIndices[i]))
						}
						row++
					}
				}
				require.Equal(t, tc.want, counts)
			}
			t.Run("worker", func(t *testing.T) {
				var chunks []*internalpb.RetrieveResults
				require.NoError(t, SplitRetrieveResult(src, tc.budget, func(chunk *internalpb.RetrieveResults) error {
					chunks = append(chunks, chunk)
					return nil
				}))
				check(t, chunks)
			})
			for _, shard := range []bool{true, false} {
				name := "proxy"
				if shard {
					name = "delegator"
				}
				t.Run(name, func(t *testing.T) {
					// One input row per Chunk: output boundaries must not depend on input boundaries.
					child := splitTestQuery(t, src)
					s := NewOrderedReduceStream(&internalpb.RetrieveRequest{Limit: -1}, nil, []ReduceStream{child}, shard, int64(tc.budget))
					defer s.Close()
					var chunks []*internalpb.RetrieveResults
					for {
						chunk, err := s.Recv()
						if err == io.EOF {
							break
						}
						require.NoError(t, err)
						chunks = append(chunks, chunk)
					}
					check(t, chunks)
				})
			}
		})
	}
}

func TestOrderedGroupLimitAndQuota(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, Name: "key", DataType: schemapb.DataType_Int64}}}
	for _, limit := range []int64{2, -1} {
		req := &internalpb.RetrieveRequest{GroupByFieldIds: []int64{100}, Limit: limit}
		input := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{makeInt64Field(100, "key", []int64{1, 2, 3})}}
		s := NewOrderedReduceStream(req, schema, []ReduceStream{splitTestQuery(t, input)}, true, 128)
		s.groups.maxGroups = 2
		out, err := s.Recv()
		if limit < 0 {
			require.ErrorContains(t, err, "too many groups (3)")
			require.Nil(t, out)
		} else {
			require.NoError(t, err)
			require.Equal(t, []int64{1, 2}, out.FieldsData[0].GetScalars().GetLongData().Data)
		}
		require.NoError(t, s.Close())
	}
}

func TestQueryChunkBoundariesWithoutPK(t *testing.T) {
	paramtable.Init()
	// The empty key occupies a row but no payload bytes. It must not share a Chunk with the oversized key.
	keys := []string{"", strings.Repeat("b", 20), "c", "d"}
	src := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{makeStringField(100, "key", keys)}}
	child := &testQueryChunks{}
	require.NoError(t, SplitRetrieveResult(src, 10, func(chunk *internalpb.RetrieveResults) error {
		child.chunks = append(child.chunks, chunk)
		return nil
	}))
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, Name: "key", DataType: schemapb.DataType_VarChar}}}
	s := NewOrderedReduceStream(&internalpb.RetrieveRequest{GroupByFieldIds: []int64{100}, Limit: -1}, schema, []ReduceStream{child}, true, 10)
	defer s.Close()
	want := [][]string{keys[:1], keys[1:2], keys[2:]}
	require.Len(t, child.chunks, len(want))
	for i, values := range want {
		require.Equal(t, values, child.chunks[i].GetFieldsData()[0].GetScalars().GetStringData().GetData())
	}
	for _, values := range want {
		chunk, err := s.Recv()
		require.NoError(t, err)
		require.Equal(t, values, chunk.GetFieldsData()[0].GetScalars().GetStringData().GetData())
	}
	_, err := s.Recv()
	require.ErrorIs(t, err, io.EOF)
}

func TestGlobalAggregateEmptyAndLateError(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: 100, Name: "val", DataType: schemapb.DataType_Int64}}}
	req := &internalpb.RetrieveRequest{Aggregates: []*planpb.Aggregate{{Op: planpb.AggregateOp_count}, {Op: planpb.AggregateOp_sum, FieldId: 100}}}
	root := NewOrderedReduceStream(req, schema, nil, false, 128)
	empty, err := root.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{0}, empty.FieldsData[0].GetScalars().GetLongData().Data)
	require.Equal(t, []bool{false}, typeutil.GetFieldDataValidData(empty.FieldsData[1]))
	require.NoError(t, root.Close())
	valid := &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{makeInt64Field(0, "count", []int64{2}), makeInt64Field(100, "sum", []int64{8})}}
	failure := merr.WrapErrParameterInvalidMsg("late worker error")
	root = NewOrderedReduceStream(req, schema, []ReduceStream{&testQueryChunks{chunks: []*internalpb.RetrieveResults{valid, {Status: merr.Status(failure)}}}}, true, 128)
	out, err := root.Recv()
	require.ErrorContains(t, err, "late worker error")
	require.Equal(t, merr.Code(failure), merr.Code(err))
	require.Nil(t, out)
	require.NoError(t, root.Close())
}

func TestQueryOrderByStream(t *testing.T) {
	paramtable.Init()
	const priceID = 100
	req := &internalpb.RetrieveRequest{Limit: 2, OrderByFields: []*planpb.OrderByField{{FieldId: priceID, Ascending: true}}}
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{FieldID: priceID, Name: "price", DataType: schemapb.DataType_Int64}}}
	input := func(ids, ts, prices []int64) *internalpb.RetrieveResults {
		result := queryStreamResult(ids, ts)
		price := queryStreamResult(nil, prices).FieldsData[0]
		price.FieldId = priceID
		result.FieldsData = append(result.FieldsData, price)
		return result
	}

	t.Run("shard_late_version_changes_top_k", func(t *testing.T) {
		// PK 1 initially ranks first at price 10. A later worker Chunk changes
		// it to 30; it must not displace either PK 2 or PK 3 in the answer.
		a := input([]int64{1, 2}, []int64{100, 100}, []int64{10, 20})
		b := input([]int64{3, 1}, []int64{100, 200}, []int64{25, 30})
		children := []ReduceStream{splitTestQuery(t, a), splitTestQuery(t, b)}
		s := NewOrderedReduceStream(req, schema, children, true, 1)
		defer s.Close()
		fields, err := orderby.ConvertFromPlanOrderByFields(req.GetOrderByFields(), schema)
		require.NoError(t, err)
		dedup, err := NewDeduplicatePKOperator(0, schema).Run(t.Context(), nil, []*internalpb.RetrieveResults{a, b})
		require.NoError(t, err)
		batch, err := NewOrderByLimitOperator(fields, req.Limit).Run(t.Context(), nil, dedup...)
		require.NoError(t, err)
		want := batch[0].(*internalpb.RetrieveResults)
		require.Equal(t, []int64{2, 3}, want.GetIds().GetIntId().GetData())
		for i := range 2 {
			chunk, err := s.Recv()
			require.NoError(t, err)
			for _, child := range children {
				require.Empty(t, child.(*testQueryChunks).chunks)
			}
			require.Equal(t, want.GetIds().GetIntId().GetData()[i:i+1], chunk.GetIds().GetIntId().GetData())
			require.Equal(t, want.FieldsData[1].GetScalars().GetLongData().Data[i:i+1], chunk.FieldsData[1].GetScalars().GetLongData().Data)
			if i == 0 {
				require.EqualValues(t, 200, chunk.AllRetrieveCount)
			} else {
				require.Zero(t, chunk.AllRetrieveCount)
			}
		}
		_, err = s.Recv()
		require.ErrorIs(t, err, io.EOF)
	})
	t.Run("shard_equal_timestamp_keeps_first_child", func(t *testing.T) {
		a := input([]int64{3, 1}, []int64{20, 20}, []int64{5, 10})
		b := input([]int64{1}, []int64{20}, []int64{1})
		s := NewOrderedReduceStream(req, schema, []ReduceStream{splitTestQuery(t, a), splitTestQuery(t, b)}, true, 1<<20)
		defer s.Close()
		chunk, err := s.Recv()
		require.NoError(t, err)
		require.Equal(t, []int64{3, 1}, chunk.GetIds().GetIntId().GetData())
	})
	t.Run("proxy_duplicate_after_window", func(t *testing.T) {
		a := input([]int64{9, 2}, []int64{1, 2}, []int64{10, 20})
		b := input([]int64{7, 9}, []int64{3, 4}, []int64{15, 30})
		s := NewOrderedReduceStream(req, schema, []ReduceStream{splitTestQuery(t, a), splitTestQuery(t, b)}, false, 1)
		defer s.Close()
		for _, id := range []int64{9, 7, 2} {
			chunk, err := s.Recv()
			require.NoError(t, err)
			require.Equal(t, []int64{id}, chunk.GetIds().GetIntId().GetData())
		}
		_, err := s.Recv()
		require.ErrorIs(t, err, merr.ErrDataIntegrity)
		_, batchErr := NewConcatAndCheckPKOperator(schema).Run(t.Context(), nil, []*internalpb.RetrieveResults{a, b})
		require.ErrorIs(t, batchErr, merr.ErrDataIntegrity)
	})

	t.Run("proxy_nulls_first_and_pk_tiebreak", func(t *testing.T) {
		order := proto.CloneOf(req)
		order.OrderByFields[0].Ascending = false
		order.OrderByFields[0].NullsFirst = true
		a := input([]int64{9, 6}, []int64{1, 2}, []int64{0, 30})
		b := input([]int64{8, 5}, []int64{3, 4}, []int64{0, 30})
		for _, r := range []*internalpb.RetrieveResults{a, b} {
			typeutil.SetFieldDataValidData(r.FieldsData[1], []bool{false, true})
		}
		s := NewOrderedReduceStream(order, schema, []ReduceStream{splitTestQuery(t, a), splitTestQuery(t, b)}, false, 1)
		defer s.Close()
		for _, id := range []int64{8, 9, 5, 6} {
			chunk, err := s.Recv()
			require.NoError(t, err)
			require.Equal(t, []int64{id}, chunk.GetIds().GetIntId().GetData())
		}
	})
	t.Run("late_worker_error_precedes_first_output", func(t *testing.T) {
		a := splitTestQuery(t, input([]int64{1}, []int64{10}, []int64{10}))
		a.chunks = append(a.chunks, &internalpb.RetrieveResults{Status: merr.Status(merr.ErrServiceUnavailable)})
		s := NewOrderedReduceStream(req, schema, []ReduceStream{a}, true, 1)
		defer s.Close()
		chunk, err := s.Recv()
		require.ErrorIs(t, err, merr.ErrServiceUnavailable)
		require.Nil(t, chunk)
	})
}

type unaryGroupWorker struct {
	querypb.QueryNodeClient
	request *querypb.QueryRequest
}

func (w *unaryGroupWorker) QueryReduceStream(context.Context, ...grpc.CallOption) (querypb.QueryNode_QueryReduceStreamClient, error) {
	return nil, status.Error(codes.Unimplemented, "unary peer")
}

func (w *unaryGroupWorker) QuerySegments(_ context.Context, req *querypb.QueryRequest, _ ...grpc.CallOption) (*internalpb.RetrieveResults, error) {
	w.request = req
	return &internalpb.RetrieveResults{Status: merr.Success()}, nil
}

func TestUnaryGroupWorkerGetsKeyOrderBeforeLimit(t *testing.T) {
	req := &querypb.QueryRequest{Req: &internalpb.RetrieveRequest{GroupByFieldIds: []int64{201, 200}, Limit: 1}}
	w := &unaryGroupWorker{}
	_, stream, err := OpenQueryReduceStream(t.Context(), w, req, 32, true)
	require.NoError(t, err)
	require.NoError(t, stream.Close())
	require.Empty(t, req.Req.OrderByFields)
	require.Len(t, w.request.Req.OrderByFields, 2)
	for i, field := range w.request.Req.OrderByFields {
		require.Equal(t, req.Req.GroupByFieldIds[i], field.FieldId)
		require.True(t, field.Ascending)
		require.False(t, field.NullsFirst)
	}
}

func TestOrderedGroupsCompositeNullableKeys(t *testing.T) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{FieldID: 100, Name: "a", DataType: schemapb.DataType_Int64, Nullable: true},
		{FieldID: 101, Name: "b", DataType: schemapb.DataType_Int64},
	}}
	req := &internalpb.RetrieveRequest{GroupByFieldIds: []int64{100, 101}, OrderByFields: []*planpb.OrderByField{{FieldId: 101, Ascending: false}}, Limit: -1}
	child := func(keys, second []int64, valid []bool) ReduceStream {
		a := makeInt64Field(100, "a", keys)
		typeutil.SetFieldDataValidData(a, valid)
		return splitTestQuery(t, &internalpb.RetrieveResults{FieldsData: []*schemapb.FieldData{a, makeInt64Field(101, "b", second)}})
	}
	stream := NewOrderedReduceStream(req, schema, []ReduceStream{
		child([]int64{2, 0, 1}, []int64{7, 7, 5}, []bool{true, false, true}),
		child([]int64{1, 0, 2}, []int64{7, 7, 5}, []bool{true, false, true}),
	}, false, 1)
	defer stream.Close()
	// Sort b DESC, break ties by a ASC NULLS LAST. Merge the two (NULL,7) heads.
	for _, key := range []struct {
		a, b  int64
		valid bool
	}{{1, 7, true}, {2, 7, true}, {0, 7, false}, {1, 5, true}, {2, 5, true}} {
		got, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, []int64{key.a}, got.FieldsData[0].GetScalars().GetLongData().Data)
		require.Equal(t, []bool{key.valid}, typeutil.GetFieldDataValidData(got.FieldsData[0]))
		require.Equal(t, []int64{key.b}, got.FieldsData[1].GetScalars().GetLongData().Data)
	}
	_, err := stream.Recv()
	require.ErrorIs(t, err, io.EOF)
}
