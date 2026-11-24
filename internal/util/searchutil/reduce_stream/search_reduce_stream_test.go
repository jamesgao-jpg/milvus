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
	"io"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type testSearchStream struct {
	chunks []*internalpb.SearchResults
	err    error
	closed bool
}

type gatedHybridStream struct {
	testSearchStream
	release <-chan struct{}
	last    *internalpb.SearchResults
	ctx     context.Context
	cancel  context.CancelFunc
}

func (s *gatedHybridStream) Recv() (*internalpb.SearchResults, error) {
	if len(s.chunks) == 0 {
		select {
		case <-s.release:
			if s.last == nil {
				return nil, io.EOF
			}
			last := s.last
			s.last = nil
			return last, nil
		case <-s.ctx.Done():
			return nil, s.ctx.Err()
		}
	}
	return s.testSearchStream.Recv()
}

func (s *gatedHybridStream) Close() error {
	s.cancel()
	return s.testSearchStream.Close()
}

func TestUngroupedHybridSearchStream(t *testing.T) {
	req := &internalpb.SearchRequest{
		IsAdvanced: true, Nq: 1,
		SubReqs: []*internalpb.SubSearchRequest{{Nq: 1, Topk: 3}, {Nq: 1, Topk: 3}},
	}
	frame := func(index int64, end bool, ids ...int64) *internalpb.SearchResults {
		sub := &internalpb.SubSearchResults{ReqIndex: index, MetricType: "IP", NumQueries: 1, TopK: 3, StreamEnd: proto.Bool(end)}
		if !end {
			data := newSearchData(1, 3)
			data.Topks[0] = int64(len(ids))
			for _, id := range ids {
				typeutil.AppendPKs(data.Ids, id)
				data.Scores = append(data.Scores, -float32(id))
			}
			sub.ResultData = data
		}
		return &internalpb.SearchResults{IsAdvanced: true, SubResults: []*internalpb.SubSearchResults{sub}}
	}
	consume := func(i int, inputs []ReduceStream) (*internalpb.SubSearchResults, error) {
		r := NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 3}, inputs, schemapb.DataType_Int64, 1024, 12)
		defer r.Close()
		out := &internalpb.SubSearchResults{ReqIndex: int64(i)}
		for {
			c, err := r.Recv()
			if err == io.EOF {
				return out, nil
			}
			if err != nil {
				return nil, err
			}
			out.ResultData = AppendSearchData(out.ResultData, c.ResultData)
			if len(out.ResultData.GetScores()) == 3 {
				return out, nil
			}
		}
	}
	for _, cancelRequest := range []bool{false, true} {
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()
		release := make(chan struct{})
		left := &testSearchStream{chunks: []*internalpb.SearchResults{frame(0, false, 1), frame(0, false, 2), frame(0, false, 3), frame(1, false, 9), frame(1, true), frame(0, true)}}
		rightCtx, closeRight := context.WithCancel(ctx)
		// B is empty on the right. Its EOF must be sufficient to finish B.
		right := &gatedHybridStream{testSearchStream: testSearchStream{chunks: []*internalpb.SearchResults{frame(1, true)}}, release: release, ctx: rightCtx, cancel: closeRight}
		bDone := make(chan *internalpb.SubSearchResults, 1)
		right.last = frame(0, true)
		done := make(chan error, 1)
		go func() {
			_, err := ReduceHybridSearchStreams(ctx, req, []ReduceStream{left, right}, func(i int, streams []ReduceStream) (*internalpb.SubSearchResults, error) {
				out, err := consume(i, streams)
				if i == 1 && err == nil {
					bDone <- out
				}
				return out, err
			})
			done <- err
		}()
		select {
		case b := <-bDone:
			require.Equal(t, []int64{9}, b.ResultData.Ids.GetIntId().Data)
		case err := <-done:
			t.Fatalf("request ended before B: %v", err)
		case <-ctx.Done():
			t.Fatal("B waited for A")
		}
		select {
		case <-done:
			t.Fatal("fusion could start before A finished")
		default:
		}
		if cancelRequest {
			cancel()
			require.ErrorIs(t, <-done, context.Canceled)
		} else {
			close(release)
			require.NoError(t, <-done)
		}
		require.True(t, left.closed && right.closed)
	}
	t.Run("old leader unary blob and old streaming frame", func(t *testing.T) {
		a, b := frame(0, false, 7), frame(1, false, 9)
		a.SubResults[0].StreamEnd, b.SubResults[0].StreamEnd = nil, nil
		var err error
		b.SubResults[0].SlicedBlob, err = proto.Marshal(b.SubResults[0].ResultData)
		require.NoError(t, err)
		b.SubResults[0].ResultData = nil
		a.SubResults = append(a.SubResults, b.SubResults...)
		a.ScannedTotalBytes = 5
		out, err := ReduceHybridSearchStreams(t.Context(), req, []ReduceStream{&testSearchStream{chunks: []*internalpb.SearchResults{a}}}, consume)
		require.NoError(t, err)
		require.Equal(t, []int64{7}, out.SubResults[0].ResultData.Ids.GetIntId().Data)
		require.Equal(t, []int64{9}, out.SubResults[1].ResultData.Ids.GetIntId().Data)
		require.EqualValues(t, 5, out.ScannedTotalBytes)
	})
	for _, tc := range []struct {
		name   string
		frames []*internalpb.SearchResults
		want   string
	}{
		{"out of range", []*internalpb.SearchResults{frame(2, true)}, "invalid Hybrid Search ReqIndex"},
		{"duplicate end", []*internalpb.SearchResults{frame(1, true), frame(1, true)}, "after ReqIndex"},
		{"missing end", []*internalpb.SearchResults{frame(1, true)}, "missing Hybrid Search end"},
		{"too many hits", []*internalpb.SearchResults{frame(0, false, 1, 2), frame(0, false, 3, 4)}, "exceeds topK"},
		{"trailing error", []*internalpb.SearchResults{frame(0, true), frame(1, true), {Status: merr.Status(merr.ErrCollectionNotLoaded)}}, "not loaded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := ReduceHybridSearchStreams(t.Context(), req, []ReduceStream{&testSearchStream{chunks: tc.frames}}, consume)
			require.Nil(t, out)
			require.ErrorContains(t, err, tc.want)
		})
	}
}

func (s *testSearchStream) Recv() (*internalpb.SearchResults, error) {
	if len(s.chunks) == 0 {
		if s.err != nil {
			return nil, s.err
		}
		return nil, io.EOF
	}
	chunk := s.chunks[0]
	s.chunks = s.chunks[1:]
	return chunk, nil
}

func (s *testSearchStream) Close() error {
	s.closed = true
	return nil
}

func (s *testSearchStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}

func TestSearchReduceStream(t *testing.T) {
	req := &internalpb.SearchRequest{Nq: 2, Topk: 3}
	build := func(channel string, ids []int64, scores []float32, valid []bool, vectors []float32, topks []int64) *testSearchStream {
		data := newSearchData(req.Nq, req.Topk)
		data.Topks, data.Scores, data.AllSearchCount = topks, scores, 12
		for _, id := range ids {
			typeutil.AppendPKs(data.Ids, id)
		}
		data.FieldsData = []*schemapb.FieldData{{
			Type: schemapb.DataType_FloatVector, FieldId: 42, ValidData: valid,
			Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{
				Dim: 1, Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: vectors}},
			}},
		}}
		blob, err := proto.Marshal(data)
		require.NoError(t, err)
		result := &internalpb.SearchResults{
			Status: merr.Success(), MetricType: "L2", SlicedBlob: blob,
			ChannelsMvcc: map[string]uint64{channel: 80}, NumQueries: req.Nq, TopK: req.Topk,
			ScannedTotalBytes: 500, CostAggregation: &internalpb.CostAggregation{TotalRelatedDataSize: 42},
		}
		stream := &testSearchStream{}
		err = SplitSearchResult(result, req.Nq, req.Topk, 1, func(chunk *internalpb.SearchResults) error {
			stream.chunks = append(stream.chunks, chunk)
			return nil
		})
		require.NoError(t, err)
		require.Len(t, stream.chunks, len(ids))
		require.NotNil(t, stream.chunks[0].GetCostAggregation())
		require.Nil(t, stream.chunks[1].GetCostAggregation())
		return stream
	}
	a := build("a", []int64{1, 6, 3, 4, 9, 10}, []float32{-1, -2, -3, -1, -2, -3},
		[]bool{true, false, true, false, true, true}, []float32{10, 30, 90, 100}, []int64{3, 3})
	b := build("b", []int64{1, 2, 5, 7, 8}, []float32{-1, -2, -4, -1, -2},
		[]bool{false, true, false, true, true}, []float32{20, 70, 80}, []int64{3, 2})
	empty := &testSearchStream{chunks: []*internalpb.SearchResults{{ChannelsMvcc: map[string]uint64{"empty": 80}}}}
	reducer := NewReduceStream(req, []ReduceStream{a, b, empty}, schemapb.DataType_Int64, 1024, 1)
	defer reducer.Close()
	result, err := reducer.Recv()
	require.NoError(t, err)
	require.Len(t, result.GetResultData().GetScores(), 1)
	for range 5 {
		chunk, err := reducer.Recv()
		require.NoError(t, err)
		require.Len(t, chunk.GetResultData().GetScores(), 1)
		require.Nil(t, chunk.GetCostAggregation())
		result.ResultData = AppendSearchData(result.GetResultData(), chunk.GetResultData())
	}
	// Batch retains the duplicate ID and breaks score ties on the smaller PK.
	require.Equal(t, []int64{1, 1, 2, 4, 7, 8}, result.GetResultData().GetIds().GetIntId().GetData())
	require.Equal(t, []int64{3, 3}, result.GetResultData().GetTopks())
	require.Equal(t, []float32{-1, -1, -2, -1, -1, -2}, result.GetResultData().GetScores())
	require.Equal(t, []bool{true, false, true, false, true, true}, result.GetResultData().GetFieldsData()[0].GetValidData())
	require.Equal(t, []float32{10, 20, 70, 80}, result.GetResultData().GetFieldsData()[0].GetVectors().GetFloatVector().GetData())
	require.EqualValues(t, 24, result.GetResultData().GetAllSearchCount())
	require.EqualValues(t, 84, result.GetCostAggregation().GetTotalRelatedDataSize())
	require.EqualValues(t, 1000, result.GetScannedTotalBytes())
	require.Equal(t, map[string]uint64{"a": 80, "b": 80, "empty": 80}, result.GetChannelsMvcc())
	require.Len(t, a.chunks, 1, "stop before reading another Chunk once the final vector's window is full")
	// A caller that explicitly keeps pulling can still read surplus final-vector hits.
	more, err := reducer.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{9}, more.GetResultData().GetIds().GetIntId().GetData())
	require.NoError(t, reducer.Close())
	require.True(t, a.closed && b.closed && empty.closed)
}

func TestSearchReduceStreamRecvErrorDiscardsPartialOutput(t *testing.T) {
	req := &internalpb.SearchRequest{Nq: 1, Topk: 2}
	data := newSearchData(1, 2)
	data.Topks, data.Scores = []int64{1}, []float32{1}
	typeutil.AppendPKs(data.Ids, int64(1))
	stream := &testSearchStream{
		chunks: []*internalpb.SearchResults{{ResultData: data}},
		err:    merr.WrapErrServiceUnavailable("leader disconnected"),
	}
	reducer := NewReduceStream(req, []ReduceStream{stream}, schemapb.DataType_Int64, 1024, 1024)
	result, err := reducer.Recv()
	require.Nil(t, result)
	require.ErrorIs(t, err, merr.ErrServiceUnavailable)
	require.Equal(t, merr.Code(stream.err), merr.Code(err))
	require.NoError(t, reducer.Close())
}

func TestSplitSearchResultStopsOnSendError(t *testing.T) {
	data := newSearchData(1, 2)
	data.Topks, data.Scores = []int64{2}, []float32{1, 0}
	typeutil.AppendPKs(data.Ids, int64(1))
	typeutil.AppendPKs(data.Ids, int64(2))
	calls := 0
	err := SplitSearchResult(&internalpb.SearchResults{ResultData: data}, 1, 2, 1, func(chunk *internalpb.SearchResults) error {
		calls++
		return context.Canceled
	})
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, 1, calls)
}

func TestOrderedSearchOversizedHitIsEmittedAlone(t *testing.T) {
	req := &internalpb.SearchRequest{Nq: 1, Topk: 2}
	ids := []string{"a", strings.Repeat("z", 256)}
	data := newSearchData(1, 2)
	data.Topks, data.Scores = []int64{2}, []float32{2, 1}
	for _, id := range ids {
		typeutil.AppendPKs(data.Ids, id)
	}
	children := []ReduceStream{
		&testSearchStream{chunks: []*internalpb.SearchResults{{ResultData: data}}},
		&testSearchStream{chunks: []*internalpb.SearchResults{{ResultData: data}}},
	}
	stream := NewShardReduceStream(req, children, schemapb.DataType_VarChar, 4096, 96)
	defer stream.Close()
	for _, id := range ids {
		chunk, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, []string{id}, chunk.GetResultData().GetIds().GetStrId().GetData())
	}
	_, err := stream.Recv()
	require.ErrorIs(t, err, io.EOF)
}

func TestSearchChunkPayloadBoundaries(t *testing.T) {
	column := func(id int64, dt schemapb.DataType, fillZero bool, values ...any) *schemapb.FieldData {
		builder, err := typeutil.NewFieldDataBuilder(dt, fillZero, len(values))
		require.NoError(t, err)
		for _, v := range values {
			builder.Add(v)
		}
		fd := builder.Build()
		fd.FieldId, fd.FieldName = id, "descriptor is not part of the hit"
		return fd
	}
	cases := []struct {
		name   string
		bytes  int
		groups []*schemapb.FieldData
		fields []*schemapb.FieldData
		want   []int // Expected hits per Chunk, computed from the payload widths.
	}{
		// Int64 PK + float32 score = 12 bytes, regardless of PK magnitude.
		{"plain_exact", 24, nil, nil, []int{2, 2, 2}},
		{"plain_crossing", 25, nil, nil, []int{3, 3}},
		// String PK length + 4-byte score + 8-byte element index per hit.
		{"varchar_elements", 42, nil, nil, []int{4, 2}},
		// The Int32 key always adds 4 bytes. The compact string key adds
		// 1, 0, 8, 0, 0, 1 bytes; NULL has no string payload.
		{"nullable_composite", 33, []*schemapb.FieldData{
			column(101, schemapb.DataType_Int32, true, nil, int32(2), int32(3), nil, int32(5), int32(6)),
			column(102, schemapb.DataType_VarChar, false, "a", nil, "bbbbbbbb", "", nil, "z"),
		}, nil, []int{2, 2, 2}},
		{"oversized_middle", 36, nil, []*schemapb.FieldData{
			column(103, schemapb.DataType_VarChar, true, "x", "", strings.Repeat("z", 80), "", "x", ""),
		}, []int{2, 1, 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &internalpb.SearchRequest{Nq: 1, Topk: 6}
			data := newSearchData(1, 6)
			data.Topks, data.Scores = []int64{6}, []float32{6, 5, 4, 3, 2, 1}
			pkType := schemapb.DataType_Int64
			if tc.name == "varchar_elements" {
				pkType = schemapb.DataType_VarChar
				for _, pk := range []string{"a", "bbb", "x", "z", "yyyy", "kk"} {
					typeutil.AppendPKs(data.Ids, pk)
				}
				data.ElementIndices = &schemapb.LongArray{Data: []int64{5, 4, 3, 2, 1, 0}}
			} else {
				for i := range data.Scores {
					typeutil.AppendPKs(data.Ids, int64(-1-i))
				}
			}
			data.FieldsData, data.GroupByFieldValues = tc.fields, tc.groups
			for _, group := range tc.groups {
				req.GroupByFieldIds = append(req.GroupByFieldIds, group.FieldId)
			}
			for _, stage := range []string{"worker", "delegator", "proxy"} {
				t.Run(stage, func(t *testing.T) {
					input := &internalpb.SearchResults{ResultData: data, Status: merr.Success()}
					var got []int
					var pos int64
					collect := func(chunk *internalpb.SearchResults) error {
						ids := chunk.GetResultData().GetIds()
						n := typeutil.GetSizeOfIDs(ids)
						got = append(got, n)
						for i := range n {
							require.Equal(t, typeutil.GetPK(data.Ids, pos), typeutil.GetPK(ids, int64(i)))
							pos++
						}
						return nil
					}
					if stage == "worker" {
						require.NoError(t, SplitSearchResult(input, 1, 6, tc.bytes, collect))
					} else {
						newReducer := NewReduceStream
						if stage == "delegator" {
							newReducer = NewShardReduceStream
						}
						s := newReducer(req, []ReduceStream{&testSearchStream{chunks: []*internalpb.SearchResults{input}}}, pkType, 4096, tc.bytes)
						defer s.Close()
						for {
							chunk, err := s.Recv()
							if err == io.EOF {
								break
							}
							require.NoError(t, err)
							require.NoError(t, collect(chunk))
						}
					}
					require.Equal(t, tc.want, got)
				})
			}
		})
	}
}
