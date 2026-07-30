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

package searchutil

import (
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

type fakeStreamRecv struct {
	chunk *internalpb.SearchResults
	err   error
}

type fakeReduceStream struct {
	mu         sync.Mutex
	recv       []fakeStreamRecv
	closeErr   error
	recvCalls  int
	closeCalls int
}

func (s *fakeReduceStream) Recv() (*internalpb.SearchResults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.recvCalls++
	if len(s.recv) == 0 {
		return nil, io.EOF
	}
	next := s.recv[0]
	s.recv = s.recv[1:]
	return next.chunk, next.err
}

func (s *fakeReduceStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeCalls++
	return s.closeErr
}

func (s *fakeReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, errors.New("not implemented")
}

func (s *fakeReduceStream) calls() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recvCalls, s.closeCalls
}

type blockingReduceStream struct {
	mu         sync.Mutex
	started    chan struct{}
	release    <-chan struct{}
	chunk      *internalpb.SearchResults
	recvCalled bool
	closeCalls int
}

func (s *blockingReduceStream) Recv() (*internalpb.SearchResults, error) {
	s.mu.Lock()
	if s.recvCalled {
		s.mu.Unlock()
		return nil, io.EOF
	}
	s.recvCalled = true
	close(s.started)
	s.mu.Unlock()

	<-s.release
	return s.chunk, nil
}

func (s *blockingReduceStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeCalls++
	return nil
}

func (s *blockingReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, errors.New("not implemented")
}

type testHit struct {
	id    int64
	score float32
}

func TestOrderedReduceStreamMergesANNHits(t *testing.T) {
	left := &fakeReduceStream{recv: []fakeStreamRecv{
		{chunk: newSearchChunk(1, 4, []testHit{{id: 1, score: 0.95}, {id: 4, score: 0.70}})},
		{chunk: newSearchChunk(1, 4, []testHit{{id: 7, score: 0.60}})},
	}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{
		{chunk: newSearchChunk(1, 4, []testHit{{id: 2, score: 0.90}, {id: 3, score: 0.80}, {id: 8, score: 0.50}})},
	}}

	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 4, MetricType: "IP"},
		[]ReduceStream{left, right},
		2,
	)
	require.NoError(t, err)

	assertSearchChunk(t, recvChunk(t, stream), []int64{1, 2}, []float32{0.95, 0.90}, []int64{2})
	assertSearchChunk(t, recvChunk(t, stream), []int64{3, 4}, []float32{0.80, 0.70}, []int64{2})

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)

	leftRecv, leftClose := left.calls()
	rightRecv, rightClose := right.calls()
	require.Equal(t, 1, leftRecv, "final topK should cancel the unread child tail")
	require.Equal(t, 1, rightRecv)
	require.Equal(t, 1, leftClose)
	require.Equal(t, 1, rightClose)
}

func TestOrderedReduceStreamReturnsPartialFinalChunk(t *testing.T) {
	left := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 3,
		[]testHit{{id: 1, score: 0.9}, {id: 3, score: 0.7}})}}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 3,
		[]testHit{{id: 2, score: 0.8}})}}}

	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 3, MetricType: "IP"},
		[]ReduceStream{left, right},
		2,
	)
	require.NoError(t, err)

	assertSearchChunk(t, recvChunk(t, stream), []int64{1, 2}, []float32{0.9, 0.8}, []int64{2})
	assertSearchChunk(t, recvChunk(t, stream), []int64{3}, []float32{0.7}, []int64{1})
}

func TestOrderedReduceStreamMergesMultipleQueries(t *testing.T) {
	left := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(2, 2,
		[]testHit{{id: 1, score: 0.90}},
		[]testHit{{id: 10, score: 0.80}},
	)}}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{
		{chunk: newSearchChunk(2, 2,
			[]testHit{{id: 2, score: 0.85}, {id: 3, score: 0.70}},
			nil,
		)},
		{chunk: newSearchChunk(2, 2,
			nil,
			[]testHit{{id: 11, score: 0.95}, {id: 12, score: 0.60}},
		)},
	}}

	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 2, Topk: 2, MetricType: "IP"},
		[]ReduceStream{left, right},
		3,
	)
	require.NoError(t, err)

	assertSearchChunk(t, recvChunk(t, stream), []int64{1, 2, 11}, []float32{0.90, 0.85, 0.95}, []int64{2, 1})
	assertSearchChunk(t, recvChunk(t, stream), []int64{10}, []float32{0.80}, []int64{0, 1})
}

func TestOrderedReduceStreamBreaksScoreTiesByPK(t *testing.T) {
	left := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 2,
		[]testHit{{id: 5, score: 0.9}})}}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 2,
		[]testHit{{id: 2, score: 0.9}})}}}

	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 2, MetricType: "IP"},
		[]ReduceStream{left, right},
		2,
	)
	require.NoError(t, err)

	assertSearchChunk(t, recvChunk(t, stream), []int64{2, 5}, []float32{0.9, 0.9}, []int64{2})
}

func TestOrderedReduceStreamComposesReducedChildStreams(t *testing.T) {
	request := &internalpb.SearchRequest{Nq: 1, Topk: 3, MetricType: "IP"}
	left, err := NewReduceStream(request, []ReduceStream{
		&fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 3,
			[]testHit{{id: 1, score: 0.95}, {id: 5, score: 0.50}})}}},
		&fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 3,
			[]testHit{{id: 3, score: 0.75}})}}},
	}, 2)
	require.NoError(t, err)

	right, err := NewReduceStream(request, []ReduceStream{
		&fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 3,
			[]testHit{{id: 2, score: 0.90}})}}},
		&fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 3,
			[]testHit{{id: 4, score: 0.70}})}}},
	}, 2)
	require.NoError(t, err)

	stream, err := NewReduceStream(request, []ReduceStream{left, right}, 2)
	require.NoError(t, err)

	assertSearchChunk(t, recvChunk(t, stream), []int64{1, 2}, []float32{0.95, 0.90}, []int64{2})
	assertSearchChunk(t, recvChunk(t, stream), []int64{3}, []float32{0.75}, []int64{1})
}

func TestOrderedReduceStreamUsesInternalScoreOrderForL2(t *testing.T) {
	leftChunk := newSearchChunk(1, 2, []testHit{{id: 1, score: -0.10}})
	leftChunk.MetricType = "L2"
	rightChunk := newSearchChunk(1, 2, []testHit{{id: 2, score: -0.20}})
	rightChunk.MetricType = "L2"

	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 2, MetricType: "L2"},
		[]ReduceStream{
			&fakeReduceStream{recv: []fakeStreamRecv{{chunk: leftChunk}}},
			&fakeReduceStream{recv: []fakeStreamRecv{{chunk: rightChunk}}},
		},
		2,
	)
	require.NoError(t, err)

	assertSearchChunk(t, recvChunk(t, stream), []int64{1, 2}, []float32{-0.10, -0.20}, []int64{2})
}

func TestOrderedReduceStreamAcceptsEncodedChunk(t *testing.T) {
	chunk := newSearchChunk(1, 1, []testHit{{id: 1, score: 0.9}})
	blob, err := proto.Marshal(chunk.GetResultData())
	require.NoError(t, err)
	chunk.ResultData = nil
	chunk.SlicedBlob = blob

	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 1, MetricType: "IP"},
		[]ReduceStream{&fakeReduceStream{recv: []fakeStreamRecv{{chunk: chunk}}}},
		1,
	)
	require.NoError(t, err)
	assertSearchChunk(t, recvChunk(t, stream), []int64{1}, []float32{0.9}, []int64{1})
}

func TestOrderedReduceStreamStartsMissingChildReceivesConcurrently(t *testing.T) {
	release := make(chan struct{})
	left := &blockingReduceStream{
		started: make(chan struct{}),
		release: release,
		chunk:   newSearchChunk(1, 2, []testHit{{id: 1, score: 0.9}}),
	}
	right := &blockingReduceStream{
		started: make(chan struct{}),
		release: release,
		chunk:   newSearchChunk(1, 2, []testHit{{id: 2, score: 0.8}}),
	}
	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 2, MetricType: "IP"},
		[]ReduceStream{left, right},
		2,
	)
	require.NoError(t, err)

	type recvResult struct {
		chunk *internalpb.SearchResults
		err   error
	}
	received := make(chan recvResult, 1)
	go func() {
		chunk, err := stream.Recv()
		received <- recvResult{chunk: chunk, err: err}
	}()

	for _, started := range []chan struct{}{left.started, right.started} {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("child Recv did not start concurrently")
		}
	}
	close(release)

	result := <-received
	require.NoError(t, result.err)
	assertSearchChunk(t, result.chunk, []int64{1, 2}, []float32{0.9, 0.8}, []int64{2})
	require.NoError(t, stream.Close())
}

func TestOrderedReduceStreamClosesChildrenOnRecvError(t *testing.T) {
	recvErr := errors.New("recv failed")
	left := &fakeReduceStream{recv: []fakeStreamRecv{{err: recvErr}}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newSearchChunk(1, 2,
		[]testHit{{id: 2, score: 0.8}})}}}
	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 2, MetricType: "IP"},
		[]ReduceStream{left, right},
		2,
	)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, recvErr)
	_, leftClose := left.calls()
	_, rightClose := right.calls()
	require.Equal(t, 1, leftClose)
	require.Equal(t, 1, rightClose)
}

func TestOrderedReduceStreamCloseIsIdempotent(t *testing.T) {
	closeErr := errors.New("close failed")
	child := &fakeReduceStream{closeErr: closeErr}
	stream, err := NewReduceStream(
		&internalpb.SearchRequest{Nq: 1, Topk: 1},
		[]ReduceStream{child},
		1,
	)
	require.NoError(t, err)

	require.ErrorIs(t, stream.Close(), closeErr)
	require.ErrorIs(t, stream.Close(), closeErr)
	_, closeCalls := child.calls()
	require.Equal(t, 1, closeCalls)

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.ErrClosedPipe)
}

func TestNewReduceStreamValidatesPlainANNSearch(t *testing.T) {
	child := &fakeReduceStream{}

	_, err := NewReduceStream(nil, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "requires a Search request")

	_, err = NewReduceStream(&internalpb.SearchRequest{Topk: 1}, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "positive nq")

	_, err = NewReduceStream(&internalpb.SearchRequest{Nq: 1}, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "positive topK")

	_, err = NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 1, IsAdvanced: true}, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "Plain ANN Search only")

	_, err = NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 1, GroupByFieldIds: []int64{101}}, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "Plain ANN Search only")

	_, err = NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 1, IsIterator: true}, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "Plain ANN Search only")

	_, err = NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 1}, []ReduceStream{child}, 0)
	require.ErrorContains(t, err, "positive Chunk size")

	_, err = NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 1}, []ReduceStream{nil}, 1)
	require.ErrorContains(t, err, "child stream 0 is nil")
}

func TestOrderedReduceStreamInterruptIsUnimplemented(t *testing.T) {
	stream, err := NewReduceStream(&internalpb.SearchRequest{Nq: 1, Topk: 1}, nil, 1)
	require.NoError(t, err)

	metadata, err := stream.Interrupt()
	require.Nil(t, metadata)
	require.ErrorContains(t, err, "Interrupt is not implemented")
}

func recvChunk(t *testing.T, stream ReduceStream) *internalpb.SearchResults {
	t.Helper()
	chunk, err := stream.Recv()
	require.NoError(t, err)
	return chunk
}

func assertSearchChunk(t *testing.T, chunk *internalpb.SearchResults, ids []int64, scores []float32, topks []int64) {
	t.Helper()
	require.True(t, merr.Ok(chunk.GetStatus()))
	require.Equal(t, ids, chunk.GetResultData().GetIds().GetIntId().GetData())
	require.Equal(t, scores, chunk.GetResultData().GetScores())
	require.Equal(t, topks, chunk.GetResultData().GetTopks())
	values := make([]int64, len(ids))
	for i, id := range ids {
		values[i] = id * 10
	}
	require.Equal(t, values, chunk.GetResultData().GetFieldsData()[0].GetScalars().GetLongData().GetData())
}

func newSearchChunk(nq, topK int64, hitsByQuery ...[]testHit) *internalpb.SearchResults {
	topks := make([]int64, nq)
	ids := make([]int64, 0)
	scores := make([]float32, 0)
	values := make([]int64, 0)
	for queryIndex := int64(0); queryIndex < nq; queryIndex++ {
		var hits []testHit
		if int(queryIndex) < len(hitsByQuery) {
			hits = hitsByQuery[queryIndex]
		}
		topks[queryIndex] = int64(len(hits))
		for _, hit := range hits {
			ids = append(ids, hit.id)
			scores = append(scores, hit.score)
			values = append(values, hit.id*10)
		}
	}

	return &internalpb.SearchResults{
		Status:     merr.Success(),
		MetricType: "IP",
		NumQueries: nq,
		TopK:       topK,
		ResultData: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topK,
			Topks:      topks,
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{Data: ids},
				},
			},
			Scores: scores,
			FieldsData: []*schemapb.FieldData{
				{
					Type:      schemapb.DataType_Int64,
					FieldName: "value",
					FieldId:   101,
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_LongData{
								LongData: &schemapb.LongArray{Data: values},
							},
						},
					},
				},
			},
		},
	}
}
