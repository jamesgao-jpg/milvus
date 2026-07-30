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

type fakeStreamRecv struct {
	chunk *internalpb.RetrieveResults
	err   error
}

type fakeReduceStream struct {
	mu         sync.Mutex
	recv       []fakeStreamRecv
	closeErr   error
	closeCalls int
}

func (s *fakeReduceStream) Recv() (*internalpb.RetrieveResults, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

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

func (s *fakeReduceStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("not implemented")
}

func (s *fakeReduceStream) getCloseCalls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeCalls
}

type blockingReduceStream struct {
	mu         sync.Mutex
	started    chan struct{}
	release    <-chan struct{}
	chunk      *internalpb.RetrieveResults
	recvCalled bool
	closeCalls int
}

func (s *blockingReduceStream) Recv() (*internalpb.RetrieveResults, error) {
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

func (s *blockingReduceStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("not implemented")
}

func TestOrderedReduceStreamMergesChunksByPK(t *testing.T) {
	children := []*fakeReduceStream{
		{recv: []fakeStreamRecv{{chunk: newIntQueryChunk(1, 4)}, {chunk: newIntQueryChunk(7)}}},
		{recv: []fakeStreamRecv{{chunk: newIntQueryChunk(2, 5, 8)}}},
		{recv: []fakeStreamRecv{{chunk: newIntQueryChunk(3, 6)}, {chunk: newIntQueryChunk(9)}}},
	}

	stream, err := NewReduceStream(
		&internalpb.RetrieveRequest{},
		[]ReduceStream{children[0], children[1], children[2]},
		3,
	)
	require.NoError(t, err)

	for _, expected := range [][]int64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}} {
		chunk, err := stream.Recv()
		require.NoError(t, err)
		require.Equal(t, expected, chunk.GetIds().GetIntId().GetData())
		require.Equal(t, multiplyByTen(expected), chunk.GetFieldsData()[0].GetScalars().GetLongData().GetData())
		require.True(t, merr.Ok(chunk.GetStatus()))
	}

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.EOF)
	for _, child := range children {
		require.Equal(t, 1, child.getCloseCalls())
	}
}

func TestOrderedReduceStreamReturnsPartialFinalChunk(t *testing.T) {
	child := &fakeReduceStream{recv: []fakeStreamRecv{
		{chunk: newIntQueryChunk()},
		{chunk: newIntQueryChunk(1, 2)},
	}}
	stream, err := NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{child}, 4)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1, 2}, chunk.GetIds().GetIntId().GetData())

	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)
}

func TestOrderedReduceStreamStartsMissingChildReceivesConcurrently(t *testing.T) {
	release := make(chan struct{})
	left := &blockingReduceStream{
		started: make(chan struct{}),
		release: release,
		chunk:   newIntQueryChunk(1),
	}
	right := &blockingReduceStream{
		started: make(chan struct{}),
		release: release,
		chunk:   newIntQueryChunk(2),
	}
	stream, err := NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{left, right}, 2)
	require.NoError(t, err)

	type recvResult struct {
		chunk *internalpb.RetrieveResults
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
	require.Equal(t, []int64{1, 2}, result.chunk.GetIds().GetIntId().GetData())
	require.NoError(t, stream.Close())
}

func TestOrderedReduceStreamRejectsDuplicatePK(t *testing.T) {
	left := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newIntQueryChunk(1)}}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newIntQueryChunk(1)}}}
	stream, err := NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{left, right}, 1)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.NoError(t, err)
	require.Equal(t, []int64{1}, chunk.GetIds().GetIntId().GetData())

	chunk, err = stream.Recv()
	require.Nil(t, chunk)
	require.ErrorContains(t, err, "duplicate PK 1")
	require.Equal(t, 1, left.getCloseCalls())
	require.Equal(t, 1, right.getCloseCalls())
}

func TestOrderedReduceStreamClosesChildrenOnRecvError(t *testing.T) {
	recvErr := errors.New("recv failed")
	left := &fakeReduceStream{recv: []fakeStreamRecv{{err: recvErr}}}
	right := &fakeReduceStream{recv: []fakeStreamRecv{{chunk: newIntQueryChunk(2)}}}
	stream, err := NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{left, right}, 2)
	require.NoError(t, err)

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, recvErr)
	require.Equal(t, 1, left.getCloseCalls())
	require.Equal(t, 1, right.getCloseCalls())
}

func TestOrderedReduceStreamCloseIsIdempotent(t *testing.T) {
	closeErr := errors.New("close failed")
	child := &fakeReduceStream{closeErr: closeErr}
	stream, err := NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{child}, 2)
	require.NoError(t, err)

	require.ErrorIs(t, stream.Close(), closeErr)
	require.ErrorIs(t, stream.Close(), closeErr)
	require.Equal(t, 1, child.getCloseCalls())

	chunk, err := stream.Recv()
	require.Nil(t, chunk)
	require.ErrorIs(t, err, io.ErrClosedPipe)
}

func TestNewReduceStreamValidatesPlainQuery(t *testing.T) {
	child := &fakeReduceStream{}

	_, err := NewReduceStream(nil, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "requires a Query request")

	_, err = NewReduceStream(&internalpb.RetrieveRequest{IsCount: true}, []ReduceStream{child}, 1)
	require.ErrorContains(t, err, "supports Plain Query only")

	_, err = NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{child}, 0)
	require.ErrorContains(t, err, "positive Chunk size")

	_, err = NewReduceStream(&internalpb.RetrieveRequest{}, []ReduceStream{nil}, 1)
	require.ErrorContains(t, err, "child stream 0 is nil")
}

func TestOrderedReduceStreamInterruptIsUnimplemented(t *testing.T) {
	stream, err := NewReduceStream(&internalpb.RetrieveRequest{}, nil, 1)
	require.NoError(t, err)

	metadata, err := stream.Interrupt()
	require.Nil(t, metadata)
	require.ErrorContains(t, err, "Interrupt is not implemented")
}

func newIntQueryChunk(ids ...int64) *internalpb.RetrieveResults {
	values := multiplyByTen(ids)
	return &internalpb.RetrieveResults{
		Status: merr.Success(),
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: ids},
			},
		},
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
	}
}

func multiplyByTen(values []int64) []int64 {
	result := make([]int64, len(values))
	for i, value := range values {
		result[i] = value * 10
	}
	return result
}
