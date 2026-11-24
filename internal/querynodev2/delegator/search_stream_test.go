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

package delegator

import (
	"context"
	"io"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/cluster"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/reduce"
	searchreducestream "github.com/milvus-io/milvus/internal/util/searchutil/reduce_stream"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type testSearchChunks struct {
	chunks []*internalpb.SearchResults
	closed bool
}

func (s *testSearchChunks) Recv() (*internalpb.SearchResults, error) {
	if len(s.chunks) == 0 {
		return nil, io.EOF
	}
	next := s.chunks[0]
	s.chunks = s.chunks[1:]
	return next, nil
}

func (s *testSearchChunks) Close() error {
	s.closed = true
	return nil
}

func (s *testSearchChunks) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}

func TestDelegatorSearchReduceStreamBatchParity(t *testing.T) {
	t.Run("plain", func(t *testing.T) { testDelegatorSearchReduceStreamBatchParity(t, false) })
	t.Run("grouped", func(t *testing.T) { testDelegatorSearchReduceStreamBatchParity(t, true) })
}

func testDelegatorSearchReduceStreamBatchParity(t *testing.T, grouped bool) {
	paramtable.Init()
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{
		FieldID: 100, DataType: schemapb.DataType_Int64, IsPrimaryKey: true,
	}}}
	schemaPatch := mockey.Mock((*segments.Collection).Schema).Return(schema).Build()
	defer schemaPatch.UnPatch()

	ctx := context.Background()
	req := &querypb.SearchRequest{
		Req:              &internalpb.SearchRequest{Nq: 1, Topk: 3, MetricType: "IP"},
		StreamChunkBytes: 1,
	}
	makeResult := func(ids []int64, scores []float32, responseTime int64) *internalpb.SearchResults {
		return &internalpb.SearchResults{
			Status: merr.Success(), MetricType: "IP", NumQueries: 1, TopK: 3,
			CostAggregation: &internalpb.CostAggregation{ResponseTime: responseTime, TotalRelatedDataSize: 40},
			ResultData: &schemapb.SearchResultData{
				NumQueries: 1, TopK: 3, Topks: []int64{int64(len(ids))}, Scores: scores, AllSearchCount: 10,
				Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}}},
			},
		}
	}
	batch := []*internalpb.SearchResults{
		makeResult([]int64{1, 3, 5}, []float32{1, 0.75, 0.5}, 10),
		makeResult([]int64{1, 2, 4}, []float32{0.99, 0.8, 0.7}, 20),
		{Status: merr.Success(), CostAggregation: &internalpb.CostAggregation{ResponseTime: 30, TotalRelatedDataSize: 90}},
	}
	info := reduce.NewReduceSearchResultInfo(1, 3).WithMetricType("IP")
	if grouped {
		req.Req.GroupByFieldId, req.Req.GroupSize = 101, 2
		info.WithGroupByFieldIds([]int64{101}).WithGroupSize(2)
		// Two workers repeat PK 1. Group 7 fills before PKs 3 and 4,
		// but the lower-scoring NULL group must still reach Proxy.
		for i, values := range [][]any{{int64(7), int64(7), nil}, {int64(8), int64(7), int64(7)}} {
			b, err := typeutil.NewFieldDataBuilder(schemapb.DataType_Int64, false, len(values))
			require.NoError(t, err)
			for _, v := range values {
				b.Add(v)
			}
			batch[i].ResultData.GroupByFieldValue = b.Build()
		}
	}
	tasks := make([]subTask[*querypb.SearchRequest], len(batch))
	open := make([]*testSearchChunks, len(batch))
	for i, r := range batch {
		open[i] = &testSearchChunks{}
		require.NoError(t, searchreducestream.SplitSearchResult(proto.CloneOf(r), 1, 3, 1, func(chunk *internalpb.SearchResults) error {
			open[i].chunks = append(open[i].chunks, chunk)
			return nil
		}))
		worker := cluster.NewMockWorker(t)
		worker.EXPECT().SearchSegmentsStream(mock.Anything, req, int32(1)).Return(open[i], nil).Once()
		tasks[i] = subTask[*querypb.SearchRequest]{worker: worker, targetID: int64(i), req: req}
	}
	var merged *internalpb.SearchResults
	_, err := (&shardDelegator{collection: &segments.Collection{}}).reduceSearchStreams(ctx, req, tasks, func(chunk *internalpb.SearchResults) error {
		require.Len(t, chunk.GetResultData().GetScores(), 1)
		if merged == nil {
			merged = chunk
		} else {
			merged.ResultData = searchreducestream.AppendSearchData(merged.ResultData, chunk.ResultData)
		}
		return nil
	})
	require.NoError(t, err)
	want, err := segments.ReduceSearchResults(ctx, batch, info)
	require.NoError(t, err)
	wantData, err := segments.DecodeSearchResults(ctx, []*internalpb.SearchResults{want})
	require.NoError(t, err)
	require.Len(t, wantData, 1)
	require.True(t, proto.Equal(wantData[0], merged.GetResultData()), "Batch %v; stream %v", wantData[0], merged.GetResultData())
	require.True(t, proto.Equal(want.GetCostAggregation(), merged.GetCostAggregation()))
	if grouped {
		require.Equal(t, []int64{1, 2, 5}, merged.GetResultData().GetIds().GetIntId().GetData())
	} else {
		require.Equal(t, []int64{1, 2, 3}, merged.GetResultData().GetIds().GetIntId().GetData())
	}
	for _, stream := range open {
		require.True(t, stream.closed)
	}
}

func TestDelegatorSearchReduceStreamCancelsBlockedWorker(t *testing.T) {
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{
		FieldID: 100, DataType: schemapb.DataType_Int64, IsPrimaryKey: true,
	}}}
	schemaPatch := mockey.Mock((*segments.Collection).Schema).Return(schema).Build()
	defer schemaPatch.UnPatch()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	req := &querypb.SearchRequest{Req: &internalpb.SearchRequest{}, StreamChunkBytes: 1}
	started := make(chan struct{})
	a, b := cluster.NewMockWorker(t), cluster.NewMockWorker(t)
	a.EXPECT().SearchSegmentsStream(mock.Anything, req, int32(1)).RunAndReturn(func(context.Context, *querypb.SearchRequest, int32) (searchreducestream.ReduceStream, error) {
		<-started
		return nil, merr.ErrCollectionNotLoaded
	}).Once()
	b.EXPECT().SearchSegmentsStream(mock.Anything, req, int32(1)).RunAndReturn(func(ctx context.Context, _ *querypb.SearchRequest, _ int32) (searchreducestream.ReduceStream, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}).Once()
	_, err := (&shardDelegator{collection: &segments.Collection{}}).reduceSearchStreams(ctx, req, []subTask[*querypb.SearchRequest]{
		{req: req, targetID: 1, worker: a}, {req: req, targetID: 2, worker: b},
	}, nil)
	require.ErrorIs(t, err, merr.ErrCollectionNotLoaded)
	require.NoError(t, ctx.Err())
}

type firstChunkWorker[T any] struct {
	first   T
	ready   bool
	ctx     context.Context
	blocked chan<- struct{}
	closed  *atomic.Int32
}

func (w *firstChunkWorker[T]) Recv() (T, error) {
	if !w.ready {
		w.ready = true
		return w.first, nil
	}
	select {
	case w.blocked <- struct{}{}:
	default:
	}
	<-w.ctx.Done()
	var zero T
	return zero, w.ctx.Err()
}
func (w *firstChunkWorker[T]) Close() error { w.closed.Add(1); return nil }
func (w *firstChunkWorker[T]) Interrupt() (T, error) {
	var zero T
	return zero, merr.ErrServiceUnimplemented
}

type pinnedStreamServer struct {
	querypb.UnimplementedQueryNodeServer
	delegator ShardDelegator
	done      chan struct{}
}

func (s *pinnedStreamServer) SearchStream(stream querypb.QueryNode_SearchStreamServer) error {
	defer close(s.done)
	open, err := stream.Recv()
	if err != nil {
		return err
	}
	req := open.GetRequest()
	req.StreamChunkBytes = open.GetChunkBytes()
	return s.delegator.SearchReduceStream(stream.Context(), req, stream.Send)
}

func (s *pinnedStreamServer) QueryReduceStream(stream querypb.QueryNode_QueryReduceStreamServer) error {
	defer close(s.done)
	open, err := stream.Recv()
	if err != nil {
		return err
	}
	req := open.GetRequest()
	req.StreamChunkBytes = open.GetChunkBytes()
	return s.delegator.QueryReduceStream(stream.Context(), req, stream.Send)
}

// The next worker read never returns data. A whole-result leader would time
// out before Proxy saw anything. Closing the RPC must also release the pin.
func (s *DelegatorSuite) TestOrderedReduceStreamFirstChunk() {
	s.delegator.Start()
	paramtable.SetNodeID(1)
	s.initSegments()
	sd := s.delegator.(*shardDelegator)
	for _, op := range []string{"query", "search", "grouped_search"} {
		s.Run(op, func() {
			t := s.T()
			s.workerManager.ExpectedCalls = nil
			snap := sd.distribution.current.Load()
			refs := snap.inUse.Load()
			var opened, closed atomic.Int32
			blocked := make(chan struct{}, 1)
			w := cluster.NewMockWorker(t)
			s.workerManager.EXPECT().GetWorker(mock.Anything, mock.Anything).Return(w, nil)
			newIDs := func() *schemapb.IDs {
				first := int64(opened.Add(1)) * 10
				return &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{first, first + 1}}}}
			}
			if op != "query" {
				w.EXPECT().SearchSegmentsStream(mock.Anything, mock.Anything, int32(1)).RunAndReturn(func(ctx context.Context, _ *querypb.SearchRequest, _ int32) (searchreducestream.ReduceStream, error) {
					data := &schemapb.SearchResultData{NumQueries: 1, TopK: 10, Topks: []int64{2}, Scores: []float32{1, 0}, Ids: newIDs()}
					if op == "grouped_search" {
						data.GroupByFieldValue = &schemapb.FieldData{Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{
							Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: data.Ids.GetIntId()}},
						}}
					}
					return &firstChunkWorker[*internalpb.SearchResults]{ctx: ctx, blocked: blocked, closed: &closed, first: &internalpb.SearchResults{Status: merr.Success(), MetricType: "IP", ResultData: data}}, nil
				})
			} else {
				w.EXPECT().QuerySegmentsReduceStream(mock.Anything, mock.Anything, int32(1)).RunAndReturn(func(ctx context.Context, _ *querypb.QueryRequest, _ int32) (queryutil.ReduceStream, error) {
					ids := newIDs()
					return &firstChunkWorker[*internalpb.RetrieveResults]{ctx: ctx, blocked: blocked, closed: &closed, first: &internalpb.RetrieveResults{Ids: ids, Status: merr.Success(), FieldsData: []*schemapb.FieldData{{FieldId: 100, Type: schemapb.DataType_Int64, Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{Data: &schemapb.ScalarField_LongData{LongData: ids.GetIntId()}}}}}}}, nil
				})
			}
			listener := bufconn.Listen(1 << 16)
			server := grpc.NewServer()
			svc := &pinnedStreamServer{delegator: sd, done: make(chan struct{})}
			querypb.RegisterQueryNodeServer(server, svc)
			go server.Serve(listener)
			defer server.Stop()
			conn, err := grpc.NewClient("passthrough:///leader", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return listener.DialContext(ctx) }))
			require.NoError(t, err)
			defer conn.Close()
			client := querypb.NewQueryNodeClient(conn)
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			channels := []string{s.vchannelName}
			if op != "query" {
				rpc, err := client.SearchStream(ctx)
				require.NoError(t, err)
				req := &internalpb.SearchRequest{Base: commonpbutil.NewMsgBase(), FieldId: 101, Nq: 1, Topk: 10}
				if op == "grouped_search" {
					req.GroupByFieldId, req.GroupSize = 100, 2
				}
				require.NoError(t, rpc.Send(&querypb.SearchStreamRequest{Request: &querypb.SearchRequest{Req: req, DmlChannels: channels}, ChunkBytes: 1}))
				require.NoError(t, rpc.CloseSend())
				first, err := rpc.Recv()
				require.NoError(t, err)
				require.Len(t, first.GetResultData().GetScores(), 1)
				if op == "grouped_search" {
					require.Len(t, first.GetResultData().GetGroupByFieldValues(), 1)
					require.Equal(t, first.GetResultData().GetIds().GetIntId().GetData(), first.GetResultData().GroupByFieldValues[0].GetScalars().GetLongData().GetData())
				}
			} else {
				first, stream, err := queryutil.OpenQueryReduceStream(ctx, client, &querypb.QueryRequest{Req: &internalpb.RetrieveRequest{Base: commonpbutil.NewMsgBase(), Limit: -1}, DmlChannels: channels}, 1, false)
				require.NoError(t, err)
				defer stream.Close()
				require.Len(t, first.GetIds().GetIntId().GetData(), 1)
			}
			select {
			case <-blocked:
			case <-ctx.Done():
				t.Fatal("delegator never reached the blocked worker read")
			}
			require.Greater(t, snap.inUse.Load(), refs)
			cancel()
			select {
			case <-svc.done:
			case <-time.After(5 * time.Second):
				t.Fatal("cancellation did not release delegator")
			}
			require.Greater(t, opened.Load(), int32(1))
			require.Equal(t, opened.Load(), closed.Load())
			require.Equal(t, refs, snap.inUse.Load())
		})
	}
}
