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
	"sort"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/fastpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type RetainedMemoryCategory string
type RetainedMemoryReduceStreamRole string

const (
	RetainedMemoryModeBatch     = "batch"
	RetainedMemoryModeStreaming = "streaming"
)

const (
	RetainedMemoryBatchResults            RetainedMemoryCategory = "batch_results"
	RetainedMemoryPerVChannelChildBuffer  RetainedMemoryCategory = "per_vchannel_child_buffer"
	RetainedMemoryPerVChannelOutputBuffer RetainedMemoryCategory = "per_vchannel_output_buffer"
	RetainedMemoryFinalChildBuffer        RetainedMemoryCategory = "final_child_buffer"
	RetainedMemoryFinalOutputBuffer       RetainedMemoryCategory = "final_output_buffer"
	RetainedMemoryFinalChunkHandoff       RetainedMemoryCategory = "final_chunk_handoff"
)

const (
	RetainedMemoryPerVChannelReduceStreamRole RetainedMemoryReduceStreamRole = "per_vchannel"
	RetainedMemoryFinalReduceStreamRole       RetainedMemoryReduceStreamRole = "final"
)

type RetainedMemoryCategorySnapshot struct {
	CurrentBytes  int64 `json:"current_bytes"`
	PeakBytes     int64 `json:"peak_bytes"`
	CurrentChunks int64 `json:"current_chunks"`
	PeakChunks    int64 `json:"peak_chunks"`
	CurrentUnits  int64 `json:"current_units"`
	PeakUnits     int64 `json:"peak_units"`
}

type RetainedMemoryReduceStreamSnapshot struct {
	Role       string `json:"role"`
	ChildCount int    `json:"child_count"`
}

type RetainedMemorySnapshot struct {
	RequestID                       int64                                     `json:"request_id"`
	Mode                            string                                    `json:"mode"`
	NQ                              int64                                     `json:"nq"`
	TopK                            int64                                     `json:"top_k"`
	CurrentRetainedBytes            int64                                     `json:"current_retained_bytes"`
	PeakRetainedBytes               int64                                     `json:"peak_retained_bytes"`
	CurrentRetainedChunks           int64                                     `json:"current_retained_chunks"`
	PeakRetainedChunks              int64                                     `json:"peak_retained_chunks"`
	CurrentRetainedUnits            int64                                     `json:"current_retained_units"`
	PeakRetainedUnits               int64                                     `json:"peak_retained_units"`
	AcceptedBytesTotal              int64                                     `json:"accepted_bytes_total"`
	ReleasedBytesTotal              int64                                     `json:"released_bytes_total"`
	ActiveRequestsPeakRetainedBytes int64                                     `json:"active_requests_peak_retained_bytes"`
	FinalResponseBytes              int64                                     `json:"final_response_bytes"`
	GroupGeneration                 int64                                     `json:"group_generation"`
	StartedAtUTC                    string                                    `json:"started_at_utc"`
	FinishedAtUTC                   string                                    `json:"finished_at_utc"`
	DurationMilliseconds            int64                                     `json:"duration_milliseconds"`
	Categories                      map[string]RetainedMemoryCategorySnapshot `json:"categories"`
	ReduceStreams                   []RetainedMemoryReduceStreamSnapshot      `json:"reduce_streams"`
}

type retainedMemoryChunk struct {
	bytes               int64
	units               int64
	owners              map[any]RetainedMemoryCategory
	categoryOwnerCounts map[RetainedMemoryCategory]int
}

type retainedMemoryCategoryState struct {
	currentBytes  int64
	peakBytes     int64
	currentChunks int64
	peakChunks    int64
	currentUnits  int64
	peakUnits     int64
}

type RetainedMemoryAccounting struct {
	mu sync.Mutex

	requestID int64
	nq        int64
	topK      int64
	mode      string
	startedAt time.Time

	chunks        map[*internalpb.SearchResults]*retainedMemoryChunk
	categories    map[RetainedMemoryCategory]*retainedMemoryCategoryState
	reduceStreams []RetainedMemoryReduceStreamSnapshot

	currentBytes  int64
	peakBytes     int64
	currentChunks int64
	peakChunks    int64
	currentUnits  int64
	peakUnits     int64
	acceptedBytes int64
	releasedBytes int64

	groupGeneration  int64
	finished         bool
	finishedSnapshot RetainedMemorySnapshot
}

var retainedMemoryActiveRequests struct {
	sync.Mutex
	generation int64
	active     int64
	current    int64
	peak       int64
}

func NewRetainedMemoryAccounting(requestID, nq, topK int64) *RetainedMemoryAccounting {
	retainedMemoryActiveRequests.Lock()
	if retainedMemoryActiveRequests.active == 0 {
		retainedMemoryActiveRequests.generation++
		retainedMemoryActiveRequests.current = 0
		retainedMemoryActiveRequests.peak = 0
	}
	retainedMemoryActiveRequests.active++
	generation := retainedMemoryActiveRequests.generation
	retainedMemoryActiveRequests.Unlock()

	return &RetainedMemoryAccounting{
		requestID:       requestID,
		nq:              nq,
		topK:            topK,
		startedAt:       time.Now().UTC(),
		chunks:          make(map[*internalpb.SearchResults]*retainedMemoryChunk),
		categories:      make(map[RetainedMemoryCategory]*retainedMemoryCategoryState),
		groupGeneration: generation,
	}
}

func (a *RetainedMemoryAccounting) SetMode(mode string) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.finished {
		a.mode = mode
	}
}

func (a *RetainedMemoryAccounting) RegisterReduceStream(role RetainedMemoryReduceStreamRole, childCount int) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finished {
		return
	}
	a.reduceStreams = append(a.reduceStreams, RetainedMemoryReduceStreamSnapshot{
		Role:       string(role),
		ChildCount: childCount,
	})
}

func (a *RetainedMemoryAccounting) Retain(owner any, category RetainedMemoryCategory, chunk *internalpb.SearchResults) {
	if a == nil || owner == nil || chunk == nil {
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finished {
		return
	}

	retained := a.chunks[chunk]
	if retained == nil {
		retained = &retainedMemoryChunk{
			bytes:               int64(proto.Size(chunk)),
			units:               searchResultUnitCount(chunk),
			owners:              make(map[any]RetainedMemoryCategory),
			categoryOwnerCounts: make(map[RetainedMemoryCategory]int),
		}
		a.chunks[chunk] = retained
		a.currentBytes += retained.bytes
		a.currentChunks++
		a.currentUnits += retained.units
		a.acceptedBytes += retained.bytes
		a.updatePeaks()
		updateActiveRetainedMemory(retained.bytes)
	}

	if _, exists := retained.owners[owner]; exists {
		return
	}
	retained.owners[owner] = category
	retained.categoryOwnerCounts[category]++
	if retained.categoryOwnerCounts[category] == 1 {
		state := a.category(category)
		state.currentBytes += retained.bytes
		state.currentChunks++
		state.currentUnits += retained.units
		state.peakBytes = max(state.peakBytes, state.currentBytes)
		state.peakChunks = max(state.peakChunks, state.currentChunks)
		state.peakUnits = max(state.peakUnits, state.currentUnits)
	}
}

func (a *RetainedMemoryAccounting) Release(owner any, chunk *internalpb.SearchResults) {
	if a == nil || owner == nil || chunk == nil {
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.finished {
		return
	}

	retained := a.chunks[chunk]
	if retained == nil {
		return
	}
	category, exists := retained.owners[owner]
	if !exists {
		return
	}
	delete(retained.owners, owner)
	retained.categoryOwnerCounts[category]--
	if retained.categoryOwnerCounts[category] == 0 {
		delete(retained.categoryOwnerCounts, category)
		state := a.category(category)
		state.currentBytes -= retained.bytes
		state.currentChunks--
		state.currentUnits -= retained.units
	}

	if len(retained.owners) == 0 {
		delete(a.chunks, chunk)
		a.currentBytes -= retained.bytes
		a.currentChunks--
		a.currentUnits -= retained.units
		a.releasedBytes += retained.bytes
		updateActiveRetainedMemory(-retained.bytes)
	}
}

func (a *RetainedMemoryAccounting) Finish(finalResponseBytes int64) RetainedMemorySnapshot {
	if a == nil {
		return RetainedMemorySnapshot{}
	}

	a.mu.Lock()
	if a.finished {
		snapshot := a.finishedSnapshot
		a.mu.Unlock()
		return snapshot
	}
	a.finished = true
	finishedAt := time.Now().UTC()

	categories := make(map[string]RetainedMemoryCategorySnapshot, len(a.categories))
	for category, state := range a.categories {
		categories[string(category)] = RetainedMemoryCategorySnapshot{
			CurrentBytes:  state.currentBytes,
			PeakBytes:     state.peakBytes,
			CurrentChunks: state.currentChunks,
			PeakChunks:    state.peakChunks,
			CurrentUnits:  state.currentUnits,
			PeakUnits:     state.peakUnits,
		}
	}
	reduceStreams := append([]RetainedMemoryReduceStreamSnapshot(nil), a.reduceStreams...)
	sort.Slice(reduceStreams, func(i, j int) bool {
		if reduceStreams[i].Role == reduceStreams[j].Role {
			return reduceStreams[i].ChildCount < reduceStreams[j].ChildCount
		}
		return reduceStreams[i].Role < reduceStreams[j].Role
	})

	retainedMemoryActiveRequests.Lock()
	activeRequestsPeak := retainedMemoryActiveRequests.peak
	retainedMemoryActiveRequests.active--
	retainedMemoryActiveRequests.Unlock()

	a.finishedSnapshot = RetainedMemorySnapshot{
		RequestID:                       a.requestID,
		Mode:                            a.mode,
		NQ:                              a.nq,
		TopK:                            a.topK,
		CurrentRetainedBytes:            a.currentBytes,
		PeakRetainedBytes:               a.peakBytes,
		CurrentRetainedChunks:           a.currentChunks,
		PeakRetainedChunks:              a.peakChunks,
		CurrentRetainedUnits:            a.currentUnits,
		PeakRetainedUnits:               a.peakUnits,
		AcceptedBytesTotal:              a.acceptedBytes,
		ReleasedBytesTotal:              a.releasedBytes,
		ActiveRequestsPeakRetainedBytes: activeRequestsPeak,
		FinalResponseBytes:              finalResponseBytes,
		GroupGeneration:                 a.groupGeneration,
		StartedAtUTC:                    a.startedAt.Format(time.RFC3339Nano),
		FinishedAtUTC:                   finishedAt.Format(time.RFC3339Nano),
		DurationMilliseconds:            finishedAt.Sub(a.startedAt).Milliseconds(),
		Categories:                      categories,
		ReduceStreams:                   reduceStreams,
	}
	snapshot := a.finishedSnapshot
	a.mu.Unlock()
	return snapshot
}

func (a *RetainedMemoryAccounting) category(category RetainedMemoryCategory) *retainedMemoryCategoryState {
	state := a.categories[category]
	if state == nil {
		state = &retainedMemoryCategoryState{}
		a.categories[category] = state
	}
	return state
}

func (a *RetainedMemoryAccounting) updatePeaks() {
	a.peakBytes = max(a.peakBytes, a.currentBytes)
	a.peakChunks = max(a.peakChunks, a.currentChunks)
	a.peakUnits = max(a.peakUnits, a.currentUnits)
}

func updateActiveRetainedMemory(delta int64) {
	retainedMemoryActiveRequests.Lock()
	defer retainedMemoryActiveRequests.Unlock()
	retainedMemoryActiveRequests.current += delta
	retainedMemoryActiveRequests.peak = max(
		retainedMemoryActiveRequests.peak,
		retainedMemoryActiveRequests.current,
	)
}

func searchResultUnitCount(chunk *internalpb.SearchResults) int64 {
	data := chunk.GetResultData()
	if data == nil && len(chunk.GetSlicedBlob()) > 0 {
		data = &schemapb.SearchResultData{}
		if err := fastpb.UnmarshalSearchResultData(chunk.GetSlicedBlob(), data); err != nil {
			return 0
		}
	}
	if data == nil {
		return 0
	}
	return int64(typeutil.GetSizeOfIDs(data.GetIds()))
}
