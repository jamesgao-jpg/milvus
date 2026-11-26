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
	"math"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// CanStreamHybrid requires a logical stream for each sub-search.
func CanStreamHybrid(req *internalpb.SearchRequest) bool {
	if !req.GetIsAdvanced() || len(req.GetSubReqs()) == 0 {
		return false
	}
	for _, sub := range req.GetSubReqs() {
		if sub == nil {
			return false
		}
	}
	return true
}

// One physical reader writes the Chunks for this (vchannel, sub-search).
// It must not wait for the reducer to take A before it can read B on the
// same RPC. Under skew, pending can hold the whole vchannel window: NQ *
// topK * hits per group. This is a hit limit, not a byte limit.
type hybridSubStream struct {
	ctx     context.Context             // Cancels all reads on request or transport failure.
	mu      sync.Mutex                  // Protects pending, ended and closed.
	pending []*internalpb.SearchResults // Released as Recv takes Chunks, or on Close.
	ready   chan struct{}               // Wakes the single consumer when data or EOF arrives.
	ended   bool                        // The leader sent EOF for this ReqIndex (physical EOF for old leaders).
	closed  bool                        // The final reducer no longer needs hits; other indexes stay open.

	// Only the physical reader updates these counters, including after Close.
	counts   []int64 // Cumulative hits per query vector from this vchannel.
	hitLimit int64   // topK for plain hits, topK * group size for grouped hits.
	sawEmpty bool    // Preserve empty-result metadata without queuing repeated empty Chunks.
}

func (s *hybridSubStream) notify() {
	select {
	case s.ready <- struct{}{}:
	default:
	}
}

func (s *hybridSubStream) accept(sub *internalpb.SubSearchResults, req *internalpb.SubSearchRequest) error {
	if s.ended {
		return merr.WrapErrServiceInternalMsg("Hybrid Search data or duplicate end after ReqIndex %d ended", sub.GetReqIndex())
	}
	if sub.GetStreamEnd() {
		if sub.GetResultData() != nil || len(sub.GetSlicedBlob()) != 0 {
			return merr.WrapErrServiceInternalMsg("Hybrid Search end marker contains data")
		}
		s.end()
		return nil
	}
	chunk := &internalpb.SearchResults{
		MetricType: sub.GetMetricType(), NumQueries: req.GetNq(), TopK: req.GetTopk(),
		ResultData: sub.GetResultData(), SlicedBlob: sub.GetSlicedBlob(),
	}
	data, err := decodeChunk(chunk, req.GetNq(), req.GetTopk())
	if err != nil {
		return err
	}
	chunk.ResultData, chunk.SlicedBlob = data, nil
	for q, n := range data.GetTopks() {
		if n > s.hitLimit-s.counts[q] {
			return merr.WrapErrServiceInternalMsg("Hybrid Search ReqIndex %d exceeds topK window for vector %d", sub.GetReqIndex(), q)
		}
		s.counts[q] += n
	}
	if len(data.GetScores()) == 0 {
		if s.sawEmpty {
			return nil
		}
		s.sawEmpty = true
	}
	s.mu.Lock()
	if !s.closed {
		s.pending = append(s.pending, chunk)
	}
	s.mu.Unlock()
	s.notify()
	return nil
}

func (s *hybridSubStream) end() {
	s.mu.Lock()
	s.ended = true
	s.mu.Unlock()
	s.notify()
}

func (s *hybridSubStream) Recv() (*internalpb.SearchResults, error) {
	for {
		s.mu.Lock()
		if len(s.pending) != 0 {
			chunk := s.pending[0]
			s.pending[0] = nil
			s.pending = s.pending[1:]
			if len(s.pending) == 0 {
				s.pending = nil
			}
			s.mu.Unlock()
			return chunk, nil
		}
		done := s.ended || s.closed
		s.mu.Unlock()
		if done {
			return nil, io.EOF
		}
		select {
		case <-s.ctx.Done():
			return nil, context.Cause(s.ctx)
		case <-s.ready:
		}
	}
}

// Close discards this sub-search's queue only. Closing the shared RPC here
// would cancel siblings that have not yet reached their own window.
func (s *hybridSubStream) Close() error {
	s.mu.Lock()
	s.closed, s.pending = true, nil
	s.mu.Unlock()
	s.notify()
	return nil
}

func (s *hybridSubStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}

// ReduceHybridSearchStreams reads one RPC per vchannel and reduces one
// candidate window per sub-search. Fusion can start only after all reducers
// finish and every physical EOF confirms there was no trailing error.
func ReduceHybridSearchStreams(ctx context.Context, req *internalpb.SearchRequest, streams []ReduceStream,
	consume func(int, []ReduceStream) (*internalpb.SubSearchResults, error),
) (*internalpb.SearchResults, error) {
	group, groupCtx := errgroup.WithContext(ctx)
	subs := req.GetSubReqs()
	inputs := make([][]ReduceStream, len(subs))
	for i := range inputs {
		inputs[i] = make([]ReduceStream, len(streams))
	}
	metadata := make([]internalpb.SearchResults, len(streams))
	for ch, stream := range streams {
		channel := make([]*hybridSubStream, len(subs))
		for i, sub := range subs {
			size := int64(1)
			if sub.GetGroupByFieldId() > 0 || len(req.GetGroupByFieldIds()) > 0 {
				size = max(1, sub.GetGroupSize())
			}
			limit := int64(math.MaxInt64)
			if sub.GetTopk() <= math.MaxInt64/size {
				limit = sub.GetTopk() * size
			}
			channel[i] = &hybridSubStream{ctx: groupCtx, ready: make(chan struct{}, 1), counts: make([]int64, sub.GetNq()), hitLimit: limit}
			inputs[i][ch] = channel[i]
		}
		group.Go(func() error {
			var modern, legacy bool
			for {
				frame, err := stream.Recv()
				if err == io.EOF {
					for index, s := range channel {
						if modern && !s.ended {
							return merr.WrapErrServiceInternalMsg("missing Hybrid Search end for ReqIndex %d", index)
						}
						// An old leader (or unary fallback) has no per-index EOF.
						s.end()
					}
					return nil
				}
				if err != nil {
					return err
				}
				if err := merr.Error(frame.GetStatus()); err != nil {
					return err
				}
				if !frame.GetIsAdvanced() {
					return merr.WrapErrServiceInternalMsg("expected Hybrid Search response")
				}
				mergeSearchMetadata(&metadata[ch], frame)
				for _, sub := range frame.GetSubResults() {
					index := sub.GetReqIndex()
					if sub == nil || index < 0 || index >= int64(len(subs)) {
						return merr.WrapErrServiceInternalMsg("invalid Hybrid Search ReqIndex %d", index)
					}
					modern = modern || sub.StreamEnd != nil
					legacy = legacy || sub.StreamEnd == nil
					if modern && legacy {
						return merr.WrapErrServiceInternalMsg("mixed Hybrid Search framing on one vchannel")
					}
					if err := channel[index].accept(sub, subs[index]); err != nil {
						return err
					}
				}
			}
		})
	}
	result := &internalpb.SearchResults{Status: merr.Success(), IsAdvanced: true, NumQueries: req.GetNq(), SubResults: make([]*internalpb.SubSearchResults, len(subs))}
	for i := range inputs {
		group.Go(func() error {
			sub, err := consume(i, inputs[i])
			result.SubResults[i] = sub
			return err
		})
	}
	// The RPCs were opened before the group existed. A router/reducer error
	// must interrupt those Recv calls too, before Wait can join their readers.
	closed := make(chan struct{})
	var closeErr error
	go func() {
		defer close(closed)
		<-groupCtx.Done()
		for _, stream := range streams {
			if err := stream.Close(); err != nil && closeErr == nil {
				closeErr = err
			}
		}
	}()
	err := group.Wait()
	<-closed
	if err != nil {
		return nil, err
	}
	if err := context.Cause(ctx); err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	for i := range metadata {
		mergeSearchMetadata(result, &metadata[i])
	}
	return result, nil
}
