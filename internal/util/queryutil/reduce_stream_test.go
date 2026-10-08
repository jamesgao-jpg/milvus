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
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestSplitRetrieveResult(t *testing.T) {
	result := newQueryTestChunk([]int64{1, 2, 3}, []int64{10, 20, 30})
	result.ReqID = 100
	result.AllRetrieveCount = 3

	chunks, err := SplitRetrieveResult(result, queryChunkBytes(t, result, 2))

	require.NoError(t, err)
	require.Len(t, chunks, 2)
	require.Equal(t, []int64{1, 2}, chunks[0].GetIds().GetIntId().GetData())
	require.Equal(t, []int64{10, 20}, chunks[0].GetFieldsData()[0].GetScalars().GetLongData().GetData())
	require.Equal(t, int64(100), chunks[0].GetReqID())
	require.Equal(t, int64(3), chunks[0].GetAllRetrieveCount())
	require.Equal(t, []int64{3}, chunks[1].GetIds().GetIntId().GetData())
	require.Equal(t, []int64{30}, chunks[1].GetFieldsData()[0].GetScalars().GetLongData().GetData())
	require.Zero(t, chunks[1].GetReqID())
	require.Zero(t, chunks[1].GetAllRetrieveCount())
}

func TestSplitRetrieveResultAcceptsUnitThatCrossesThreshold(t *testing.T) {
	result := newQueryTestChunk([]int64{1, 2, 3}, []int64{10, 20, 30})
	chunkBytes := queryChunkBytes(t, result, 2) - 1

	chunks, err := SplitRetrieveResult(result, chunkBytes)
	require.NoError(t, err)
	require.Len(t, chunks, 2)
	require.Equal(t, []int64{1, 2}, queryTestIDs(chunks[0]))
	require.Equal(t, []int64{3}, queryTestIDs(chunks[1]))
}

func TestSplitRetrieveResultEmitsOversizedUnitAlone(t *testing.T) {
	largeID := int64(1 << 56)
	result := newQueryTestChunk([]int64{1, largeID}, []int64{10, largeID})
	firstUnitBytes := queryChunkBytes(t, newQueryTestChunk([]int64{1}, []int64{10}), 1)
	largeUnitBytes := queryChunkBytes(t, newQueryTestChunk([]int64{largeID}, []int64{largeID}), 1)
	chunkBytes := firstUnitBytes + 1
	require.Greater(t, largeUnitBytes, chunkBytes)

	chunks, err := SplitRetrieveResult(result, chunkBytes)
	require.NoError(t, err)
	require.Len(t, chunks, 2)
	require.Equal(t, []int64{1}, queryTestIDs(chunks[0]))
	require.Equal(t, []int64{largeID}, queryTestIDs(chunks[1]))
}

func TestOrderedReduceStreamMergesChildChunks(t *testing.T) {
	childA := &queryTestStream{recv: []queryTestRecv{
		{chunk: newQueryTestChunk([]int64{1, 3}, []int64{10, 30})},
		{chunk: newQueryTestChunk([]int64{5}, []int64{50})},
	}}
	childB := &queryTestStream{recv: []queryTestRecv{
		{chunk: newQueryTestChunk([]int64{2, 4, 6}, []int64{20, 40, 60})},
	}}
	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{IsIterator: true, Limit: 5},
		[]ReduceStream{childA, childB},
		queryChunkBytes(t, newQueryTestChunk([]int64{1, 2}, []int64{10, 20}), 2),
	)
	require.NoError(t, err)

	first, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1, 2}, queryTestIDs(first))

	second, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{3, 4}, queryTestIDs(second))

	third, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{5}, queryTestIDs(third))

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)
	require.Equal(t, 1, childA.closeCount())
	require.Equal(t, 1, childB.closeCount())
}

func TestOrderedReduceStreamAcceptsBoundedOrdinaryQuery(t *testing.T) {
	child := &queryTestStream{recv: []queryTestRecv{
		{chunk: newQueryTestChunk([]int64{1, 2}, []int64{10, 20})},
	}}
	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{Limit: 2},
		[]ReduceStream{child},
		queryChunkBytes(t, child.recv[0].chunk, 2),
	)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1, 2}, queryTestIDs(chunk))
}

func TestOrderedReduceStreamDefersOversizedUnitToNextChunk(t *testing.T) {
	largeID := int64(1 << 56)
	result := newQueryTestChunk([]int64{1, largeID}, []int64{10, largeID})
	child := &queryTestStream{recv: []queryTestRecv{{chunk: result}}}
	chunkBytes := queryChunkBytes(t, newQueryTestChunk([]int64{1}, []int64{10}), 1) + 1
	require.Greater(t, queryChunkBytes(t, newQueryTestChunk([]int64{largeID}, []int64{largeID}), 1), chunkBytes)

	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{IsIterator: true, Limit: 2},
		[]ReduceStream{child},
		chunkBytes,
	)
	require.NoError(t, err)
	first, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1}, queryTestIDs(first))
	second, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{largeID}, queryTestIDs(second))
}

func TestOrderedReduceStreamRejectsDuplicatePK(t *testing.T) {
	childA := &queryTestStream{recv: []queryTestRecv{{chunk: newQueryTestChunk([]int64{1}, []int64{10})}}}
	childB := &queryTestStream{recv: []queryTestRecv{{chunk: newQueryTestChunk([]int64{1}, []int64{20})}}}
	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{IsIterator: true, Limit: 2},
		[]ReduceStream{childA, childB},
		queryChunkBytes(t, newQueryTestChunk([]int64{1, 1}, []int64{10, 20}), 2),
	)
	require.NoError(t, err)

	chunk, err := stream.Recv()

	require.Nil(t, chunk)
	require.Error(t, err)
	require.Equal(t, 1, childA.closeCount())
	require.Equal(t, 1, childB.closeCount())
}

func TestOrderedReduceStreamRetainedBufferFailureClearsState(t *testing.T) {
	recvErr := errors.New("sibling receive failed")
	retainedChild := &queryTestStream{}
	failingChild := &queryTestStream{recv: []queryTestRecv{{err: recvErr}}}
	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{IsIterator: true, Limit: 2},
		[]ReduceStream{retainedChild, failingChild},
		1,
	)
	require.NoError(t, err)
	reducer := stream.(*OrderedReduceStream)

	err = reducer.childBuffers[0].accept(newQueryTestChunk([]int64{1}, []int64{10}))
	require.NoError(t, err)
	require.True(t, reducer.childBuffers[0].hasUnit())

	chunk, err := reducer.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, recvErr)
	for i := range reducer.childBuffers {
		require.Nil(t, reducer.childBuffers[i].chunk)
		require.Zero(t, reducer.childBuffers[i].cursor)
	}
	require.Equal(t, 1, retainedChild.closeCount())
	require.Equal(t, 1, failingChild.closeCount())
}

func TestOrderedReduceStreamPendingRecvFailureUnblocksOnClose(t *testing.T) {
	blocked := &closeUnblockingQueryStream{
		started: make(chan struct{}),
		done:    make(chan struct{}),
		closed:  make(chan struct{}),
	}
	recvErr := errors.New("sibling receive failed")
	failing := &errorAfterStartQueryStream{started: blocked.started, err: recvErr}
	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{IsIterator: true, Limit: 1},
		[]ReduceStream{blocked, failing},
		1,
	)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, recvErr)
	select {
	case <-blocked.done:
	case <-time.After(time.Second):
		t.Fatal("pending Query child Recv did not exit after reducer Close")
	}
	require.Equal(t, 1, blocked.closeCount())
	require.Equal(t, 1, failing.closeCount())
}

func TestOrderedReduceStreamPrematureEOFWithSiblingReturnsUnprovenTopK(t *testing.T) {
	earlyChild := &queryTestStream{recv: []queryTestRecv{
		{chunk: newQueryTestChunk([]int64{1}, []int64{10})},
	}}
	sibling := &queryTestStream{recv: []queryTestRecv{
		{chunk: newQueryTestChunk([]int64{2, 4}, []int64{20, 40})},
	}}
	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{IsIterator: true, Limit: 3},
		[]ReduceStream{earlyChild, sibling},
		queryChunkBytes(t, newQueryTestChunk([]int64{1, 2, 4}, []int64{10, 20, 40}), 3),
	)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1, 2, 4}, queryTestIDs(chunk))
	chunk, err = stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)
}

func newQueryTestChunk(ids, values []int64) *internalpb.RetrieveResults {
	return &internalpb.RetrieveResults{
		Status: merr.Success(),
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}},
		},
		FieldsData: []*schemapb.FieldData{
			{
				Type:    schemapb.DataType_Int64,
				FieldId: 101,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: values}},
					},
				},
			},
		},
	}
}

func queryChunkBytes(t *testing.T, result *internalpb.RetrieveResults, unitCount int) int {
	t.Helper()
	bytes := 0
	for row := 0; row < unitCount; row++ {
		unitBytes, err := (retrieveUnit{result: result, rowIndex: int64(row)}).reducibleByteSize()
		require.NoError(t, err)
		bytes += unitBytes
	}
	return bytes
}

func queryTestIDs(result *internalpb.RetrieveResults) []int64 {
	return result.GetIds().GetIntId().GetData()
}

type queryTestRecv struct {
	chunk *internalpb.RetrieveResults
	err   error
}

type queryTestStream struct {
	mu         sync.Mutex
	recv       []queryTestRecv
	closeCalls int
}

func (s *queryTestStream) Recv() (*internalpb.RetrieveResults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.recv) == 0 {
		return nil, io.EOF
	}
	next := s.recv[0]
	s.recv = s.recv[1:]
	return next.chunk, next.err
}

func (s *queryTestStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeCalls++
	return nil
}

func (*queryTestStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("not implemented")
}

func (s *queryTestStream) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeCalls
}

type closeUnblockingQueryStream struct {
	started    chan struct{}
	done       chan struct{}
	closed     chan struct{}
	startOnce  sync.Once
	doneOnce   sync.Once
	closeOnce  sync.Once
	mu         sync.Mutex
	closeCalls int
}

func (s *closeUnblockingQueryStream) Recv() (*internalpb.RetrieveResults, error) {
	s.startOnce.Do(func() { close(s.started) })
	<-s.closed
	s.doneOnce.Do(func() { close(s.done) })
	return nil, io.ErrClosedPipe
}

func (s *closeUnblockingQueryStream) Close() error {
	s.mu.Lock()
	s.closeCalls++
	s.mu.Unlock()
	s.closeOnce.Do(func() { close(s.closed) })
	return nil
}

func (*closeUnblockingQueryStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("not implemented")
}

func (s *closeUnblockingQueryStream) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeCalls
}

type errorAfterStartQueryStream struct {
	started    <-chan struct{}
	err        error
	mu         sync.Mutex
	closeCalls int
}

func (s *errorAfterStartQueryStream) Recv() (*internalpb.RetrieveResults, error) {
	<-s.started
	return nil, s.err
}

func (s *errorAfterStartQueryStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closeCalls++
	return nil
}

func (*errorAfterStartQueryStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("not implemented")
}

func (s *errorAfterStartQueryStream) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeCalls
}
