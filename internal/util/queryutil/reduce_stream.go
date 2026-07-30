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
	"fmt"
	"io"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// ReduceStream is the common parent-side contract for reduced streams and
// transport-backed child streams. Exactly one caller must drive Recv, Close,
// and Interrupt serially.
type ReduceStream interface {
	Recv() (*internalpb.RetrieveResults, error)
	Close() error
	Interrupt() (*internalpb.RetrieveResults, error)
}

type childRecvCompletion struct {
	childIndex int
	chunk      *internalpb.RetrieveResults
	err        error
}

type orderedUnit struct {
	chunk    *internalpb.RetrieveResults
	rowIndex int64
}

type orderedOutputBuffer struct {
	units []orderedUnit
}

func (b *orderedOutputBuffer) isEmpty() bool {
	return len(b.units) == 0
}

type orderedChildBuffer struct {
	chunk  *internalpb.RetrieveResults
	cursor int64
}

func (b *orderedChildBuffer) hasUnit() bool {
	return b.chunk != nil && b.cursor < int64(typeutil.GetSizeOfIDs(b.chunk.GetIds()))
}

// OrderedReduceStream merges Plain Query child Chunks by primary key.
type OrderedReduceStream struct {
	childStreams         []ReduceStream
	childBuffers         []orderedChildBuffer
	childRecvTasks       []bool
	childDrained         []bool
	childRecvCompletions chan childRecvCompletion
	chunkSize            int
	lastPK               any
	hasLastPK            bool

	closed         bool
	finished       bool
	childrenClosed bool
	closeErr       error
}

// NewReduceStream creates the Plain Query OrderedReduceStream implementation.
func NewReduceStream(request *internalpb.RetrieveRequest, childStreams []ReduceStream, chunkSize int) (ReduceStream, error) {
	if request == nil {
		return nil, errors.New("NewReduceStream requires a Query request")
	}
	if request.GetIsCount() || len(request.GetGroupByFieldIds()) > 0 || len(request.GetAggregates()) > 0 || len(request.GetOrderByFields()) > 0 {
		return nil, errors.New("NewReduceStream currently supports Plain Query only")
	}
	if chunkSize <= 0 {
		return nil, fmt.Errorf("NewReduceStream requires a positive Chunk size, got %d", chunkSize)
	}

	for i, childStream := range childStreams {
		if childStream == nil {
			return nil, fmt.Errorf("NewReduceStream child stream %d is nil", i)
		}
	}

	return &OrderedReduceStream{
		childStreams:         append([]ReduceStream(nil), childStreams...),
		childBuffers:         make([]orderedChildBuffer, len(childStreams)),
		childRecvTasks:       make([]bool, len(childStreams)),
		childDrained:         make([]bool, len(childStreams)),
		childRecvCompletions: make(chan childRecvCompletion, max(1, len(childStreams))),
		chunkSize:            chunkSize,
	}, nil
}

// Recv returns the next ordered Query Chunk.
func (s *OrderedReduceStream) Recv() (*internalpb.RetrieveResults, error) {
	if s.finished {
		return nil, io.EOF
	}
	if s.closed {
		return nil, io.ErrClosedPipe
	}

	outputBuffer := s.allocBuffer()

	for !s.isChunkReady(outputBuffer) {
		readyBuffers, err := s.getReadyBuffers()
		if err != nil {
			return nil, s.fail(err)
		}

		oneReduceResult, err := s.produceNextUnits(readyBuffers)
		if err != nil {
			return nil, s.fail(err)
		}

		s.merge(outputBuffer, oneReduceResult)
	}

	if outputBuffer.isEmpty() {
		s.finished = true
		if err := s.closeChildren(); err != nil {
			return nil, err
		}
		return nil, io.EOF
	}

	chunk, err := s.createOutputChunk(outputBuffer)
	if err != nil {
		return nil, s.fail(err)
	}
	return chunk, nil
}

func (s *OrderedReduceStream) allocBuffer() *orderedOutputBuffer {
	return &orderedOutputBuffer{
		units: make([]orderedUnit, 0, s.chunkSize),
	}
}

func (s *OrderedReduceStream) isChunkReady(outputBuffer *orderedOutputBuffer) bool {
	return len(outputBuffer.units) >= s.chunkSize || s.allChildrenDrained()
}

func (s *OrderedReduceStream) getReadyBuffers() ([]*orderedChildBuffer, error) {
	for {
		allReady := true
		for i, childStream := range s.childStreams {
			if s.childDrained[i] || s.childBuffers[i].hasUnit() {
				continue
			}

			allReady = false
			if s.childRecvTasks[i] {
				continue
			}

			s.childRecvTasks[i] = true
			completionChannel := s.childRecvCompletions
			go func(childIndex int) {
				chunk, err := childStream.Recv()
				completionChannel <- childRecvCompletion{
					childIndex: childIndex,
					chunk:      chunk,
					err:        err,
				}
			}(i)
		}

		if allReady {
			readyBuffers := make([]*orderedChildBuffer, 0, len(s.childBuffers))
			for i := range s.childBuffers {
				if s.childBuffers[i].hasUnit() {
					readyBuffers = append(readyBuffers, &s.childBuffers[i])
				}
			}
			return readyBuffers, nil
		}

		received := <-s.childRecvCompletions
		s.childRecvTasks[received.childIndex] = false

		if errors.Is(received.err, io.EOF) {
			s.childDrained[received.childIndex] = true
			continue
		}
		if received.err != nil {
			return nil, fmt.Errorf("child stream %d Recv failed: %w", received.childIndex, received.err)
		}
		if received.chunk == nil {
			return nil, fmt.Errorf("child stream %d returned a nil Chunk", received.childIndex)
		}
		if !merr.Ok(received.chunk.GetStatus()) {
			return nil, fmt.Errorf("child stream %d returned a failed Chunk: %w", received.childIndex, merr.Error(received.chunk.GetStatus()))
		}

		if typeutil.GetSizeOfIDs(received.chunk.GetIds()) == 0 || len(received.chunk.GetFieldsData()) == 0 {
			continue
		}

		s.childBuffers[received.childIndex].chunk = received.chunk
		s.childBuffers[received.childIndex].cursor = 0
	}
}

func (s *OrderedReduceStream) produceNextUnits(readyBuffers []*orderedChildBuffer) (*orderedUnit, error) {
	if len(readyBuffers) == 0 {
		return nil, nil
	}

	results := make([]*internalpb.RetrieveResults, len(readyBuffers))
	cursors := make([]int64, len(readyBuffers))
	for i, buffer := range readyBuffers {
		results[i] = buffer.chunk
		cursors[i] = buffer.cursor
	}

	winner, _ := typeutil.SelectMinPK(results, cursors)
	if winner < 0 {
		return nil, errors.New("OrderedReduceStream found no selectable child row")
	}

	buffer := readyBuffers[winner]
	pk := typeutil.GetPK(buffer.chunk.GetIds(), buffer.cursor)
	if s.hasLastPK && pk == s.lastPK {
		return nil, merr.WrapErrDataIntegrityMsg("duplicate PK %v found across child streams", pk)
	}
	s.lastPK = pk
	s.hasLastPK = true

	oneReduceResult := &orderedUnit{
		chunk:    buffer.chunk,
		rowIndex: buffer.cursor,
	}

	buffer.cursor++
	if !buffer.hasUnit() {
		buffer.chunk = nil
		buffer.cursor = 0
	}

	return oneReduceResult, nil
}

func (s *OrderedReduceStream) merge(outputBuffer *orderedOutputBuffer, oneReduceResult *orderedUnit) {
	if oneReduceResult != nil {
		outputBuffer.units = append(outputBuffer.units, *oneReduceResult)
	}
}

func (s *OrderedReduceStream) createOutputChunk(outputBuffer *orderedOutputBuffer) (*internalpb.RetrieveResults, error) {
	selectedResults := make([]*internalpb.RetrieveResults, len(outputBuffer.units))
	selectedRows := make([]rowRef, len(outputBuffer.units))
	for i, unit := range outputBuffer.units {
		selectedResults[i] = unit.chunk
		selectedRows[i] = rowRef{
			resultIdx: i,
			rowIdx:    unit.rowIndex,
		}
	}

	chunk, err := buildMergedRetrieveResults(selectedResults, selectedRows, nil)
	if err != nil {
		return nil, err
	}
	chunk.Base = selectedResults[0].GetBase()
	chunk.Status = merr.Success()
	chunk.ReqID = selectedResults[0].GetReqID()
	return chunk, nil
}

func (s *OrderedReduceStream) allChildrenDrained() bool {
	for i := range s.childDrained {
		if !s.childDrained[i] {
			return false
		}
	}
	return true
}

func (s *OrderedReduceStream) fail(err error) error {
	return errors.Join(err, s.Close())
}

// Close idempotently closes every child stream and releases retained Chunks.
func (s *OrderedReduceStream) Close() error {
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	return s.closeChildren()
}

func (s *OrderedReduceStream) closeChildren() error {
	if s.childrenClosed {
		return s.closeErr
	}
	s.childrenClosed = true

	closeErrors := make([]error, 0, len(s.childStreams))
	for i := range s.childStreams {
		if err := s.childStreams[i].Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("close child stream %d: %w", i, err))
		}
		s.childBuffers[i].chunk = nil
		s.childBuffers[i].cursor = 0
		s.childRecvTasks[i] = false
	}
	s.lastPK = nil
	s.hasLastPK = false
	s.closeErr = errors.Join(closeErrors...)
	return s.closeErr
}

// Interrupt is reserved for the bidirectional gRPC integration.
func (s *OrderedReduceStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("OrderedReduceStream Interrupt is not implemented")
}
