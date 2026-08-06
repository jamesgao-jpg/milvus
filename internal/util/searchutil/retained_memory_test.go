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
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func TestRetainedMemoryAccountingDeduplicatesChunkOwners(t *testing.T) {
	chunk := newSearchChunk(1, 2, []testHit{{id: 1, score: 0.9}, {id: 2, score: 0.8}})
	chunkBytes := int64(proto.Size(chunk))
	childOwner := new(int)
	outputOwner := new(int)

	accounting := NewRetainedMemoryAccounting(10, 1, 2)
	accounting.SetMode(RetainedMemoryModeStreaming)
	accounting.RegisterReduceStream(RetainedMemoryFinalReduceStreamRole, 2)
	accounting.Retain(childOwner, RetainedMemoryFinalChildBuffer, chunk)
	accounting.Retain(outputOwner, RetainedMemoryFinalOutputBuffer, chunk)
	accounting.Release(childOwner, chunk)
	accounting.Release(outputOwner, chunk)

	snapshot := accounting.Finish(123)
	require.Equal(t, RetainedMemoryModeStreaming, snapshot.Mode)
	require.Equal(t, chunkBytes, snapshot.PeakRetainedBytes)
	require.Equal(t, int64(1), snapshot.PeakRetainedChunks)
	require.Equal(t, int64(2), snapshot.PeakRetainedUnits)
	require.Zero(t, snapshot.CurrentRetainedBytes)
	require.Zero(t, snapshot.CurrentRetainedChunks)
	require.Zero(t, snapshot.CurrentRetainedUnits)
	require.Equal(t, chunkBytes, snapshot.AcceptedBytesTotal)
	require.Equal(t, chunkBytes, snapshot.ReleasedBytesTotal)
	require.Equal(t, int64(123), snapshot.FinalResponseBytes)
	require.Equal(t, chunkBytes, snapshot.Categories[string(RetainedMemoryFinalChildBuffer)].PeakBytes)
	require.Equal(t, chunkBytes, snapshot.Categories[string(RetainedMemoryFinalOutputBuffer)].PeakBytes)
	require.Equal(t, []RetainedMemoryReduceStreamSnapshot{{Role: "final", ChildCount: 2}}, snapshot.ReduceStreams)
}

func TestRetainedMemoryAccountingTracksConcurrentRequestPeak(t *testing.T) {
	firstChunk := newSearchChunk(1, 1, []testHit{{id: 1, score: 0.9}})
	secondChunk := newSearchChunk(1, 1, []testHit{{id: 2, score: 0.8}})
	firstOwner := new(int)
	secondOwner := new(int)

	first := NewRetainedMemoryAccounting(20, 1, 1)
	second := NewRetainedMemoryAccounting(21, 1, 1)
	first.Retain(firstOwner, RetainedMemoryBatchResults, firstChunk)
	second.Retain(secondOwner, RetainedMemoryBatchResults, secondChunk)
	expectedPeak := int64(proto.Size(firstChunk) + proto.Size(secondChunk))
	first.Release(firstOwner, firstChunk)
	second.Release(secondOwner, secondChunk)

	firstSnapshot := first.Finish(0)
	secondSnapshot := second.Finish(0)
	require.Equal(t, firstSnapshot.GroupGeneration, secondSnapshot.GroupGeneration)
	require.Equal(t, expectedPeak, firstSnapshot.ActiveRequestsPeakRetainedBytes)
	require.Equal(t, expectedPeak, secondSnapshot.ActiveRequestsPeakRetainedBytes)
}
