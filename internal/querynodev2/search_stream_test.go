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

package querynodev2

import (
	"context"
	"io"
	"net"
	"testing"
	"time"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	searchreducestream "github.com/milvus-io/milvus/internal/util/searchutil/reduce_stream"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestReduceStreamRPC(t *testing.T) {
	node := &QueryNode{}
	patch := mockey.Mock((*QueryNode).searchRequest).To(func(_ *QueryNode, _ context.Context, req *querypb.SearchRequest, send func(*internalpb.SearchResults) error) (*internalpb.SearchResults, error) {
		assert.Equal(t, []string{"a"}, req.GetDmlChannels())
		assert.EqualValues(t, 1, req.GetTotalChannelNum())
		if req.GetReq().GetIsAdvanced() {
			// This request came from an older Proxy. It cannot interpret an
			// indexed EOF or a metadata-only trailer.
			assert.Nil(t, send)
			return &internalpb.SearchResults{IsAdvanced: true, ScannedTotalBytes: 7, SubResults: []*internalpb.SubSearchResults{{NumQueries: 1, TopK: 2}}}, nil
		}
		if req.GetReq().GetCollectionID() == 42 {
			return &internalpb.SearchResults{Status: merr.Status(merr.WrapErrCollectionNotLoaded(42))}, nil
		}
		ids := &schemapb.IDs{}
		typeutil.AppendPKs(ids, int64(2))
		typeutil.AppendPKs(ids, int64(3))
		return &internalpb.SearchResults{
			Status: merr.Success(), MetricType: "IP", ChannelsMvcc: map[string]uint64{"a": 10},
			ResultData: &schemapb.SearchResultData{NumQueries: 1, TopK: 2, Topks: []int64{2}, Scores: []float32{1, 0.5}, Ids: ids},
		}, nil
	}).Build()
	defer patch.UnPatch()

	listener := bufconn.Listen(1 << 16)
	server := grpc.NewServer()
	querypb.RegisterQueryNodeServer(server, node)
	go server.Serve(listener)
	defer server.Stop()

	conn, err := grpc.NewClient("passthrough:///querynode",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return listener.DialContext(ctx)
		}))
	require.NoError(t, err)
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	client := querypb.NewQueryNodeClient(conn)
	open := func(collectionID int64) querypb.QueryNode_SearchStreamClient {
		stream, err := client.SearchStream(ctx)
		require.NoError(t, err)
		require.NoError(t, stream.Send(&querypb.SearchStreamRequest{
			ChunkBytes: 1,
			Request: &querypb.SearchRequest{
				Req:         &internalpb.SearchRequest{CollectionID: collectionID, Nq: 1, Topk: 2},
				DmlChannels: []string{"a"}, TotalChannelNum: 1,
			},
		}))
		require.NoError(t, stream.CloseSend())
		return stream
	}

	stream := open(1)
	for i, id := range []int64{2, 3} {
		chunk, err := stream.Recv()
		require.NoError(t, err)
		require.True(t, merr.Ok(chunk.GetStatus()))
		require.Equal(t, []int64{id}, chunk.GetResultData().GetIds().GetIntId().GetData())
		if i == 0 {
			require.EqualValues(t, 10, chunk.GetChannelsMvcc()["a"])
		} else {
			require.Empty(t, chunk.GetChannelsMvcc())
		}
	}
	_, err = stream.Recv()
	require.ErrorIs(t, err, io.EOF)

	legacy, err := client.SearchStream(ctx)
	require.NoError(t, err)
	require.NoError(t, legacy.Send(&querypb.SearchStreamRequest{ChunkBytes: 1, Request: &querypb.SearchRequest{
		DmlChannels: []string{"a"}, TotalChannelNum: 1, Req: &internalpb.SearchRequest{IsAdvanced: true, SubReqs: []*internalpb.SubSearchRequest{{Nq: 1, Topk: 2}}},
	}}))
	require.NoError(t, legacy.CloseSend())
	first, err := legacy.Recv()
	require.NoError(t, err)
	require.Len(t, first.GetSubResults(), 1)
	require.Nil(t, first.SubResults[0].StreamEnd)
	require.EqualValues(t, 7, first.GetScannedTotalBytes())
	_, err = legacy.Recv()
	require.ErrorIs(t, err, io.EOF)

	failed := open(42)
	chunk, err := failed.Recv()
	require.NoError(t, err)
	require.ErrorIs(t, merr.Error(chunk.GetStatus()), merr.ErrCollectionNotLoaded)
	_, err = failed.Recv()
	require.ErrorIs(t, err, io.EOF)

	workerPatch := mockey.Mock((*QueryNode).SearchSegments).To(func(_ *QueryNode, _ context.Context, req *querypb.SearchRequest) (*internalpb.SearchResults, error) {
		assert.Equal(t, querypb.DataScope_Historical, req.GetScope())
		assert.Equal(t, []int64{7, 8}, req.GetSegmentIDs())
		ids := &schemapb.IDs{}
		typeutil.AppendPKs(ids, int64(4))
		typeutil.AppendPKs(ids, int64(5))
		return &internalpb.SearchResults{
			Status: merr.Success(), NumQueries: 1, TopK: 2,
			ResultData: &schemapb.SearchResultData{NumQueries: 1, TopK: 2, Topks: []int64{2}, Scores: []float32{1, 0.5}, Ids: ids},
		}, nil
	}).Build()
	defer workerPatch.UnPatch()

	worker, err := searchreducestream.OpenWorkerSearchStream(ctx, client, &querypb.SearchRequest{
		Req:         &internalpb.SearchRequest{Nq: 1, Topk: 2},
		DmlChannels: []string{"a"}, Scope: querypb.DataScope_Historical, SegmentIDs: []int64{7, 8},
	}, 1)
	require.NoError(t, err)
	defer worker.Close()
	for _, id := range []int64{4, 5} {
		chunk, err := worker.Recv()
		require.NoError(t, err)
		require.Equal(t, []int64{id}, chunk.GetResultData().GetIds().GetIntId().GetData())
	}
	_, err = worker.Recv()
	require.ErrorIs(t, err, io.EOF)

	queryResult := &internalpb.RetrieveResults{Ids: &schemapb.IDs{}, MvccTimestamp: 23, AllRetrieveCount: 7}
	for _, id := range []int64{10, 20} {
		typeutil.AppendPKs(queryResult.Ids, id)
	}
	queryResult.FieldsData = []*schemapb.FieldData{{
		FieldId: 101, Type: schemapb.DataType_Int64,
		Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
			Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{10, 20}}},
		}},
	}}
	queryPatch := mockey.Mock((*QueryNode).queryRequest).To(func(_ *QueryNode, _ context.Context, req *querypb.QueryRequest, send func(*internalpb.RetrieveResults) error) (*internalpb.RetrieveResults, error) {
		assert.EqualValues(t, 1, req.GetStreamChunkBytes())
		return queryResult, nil
	}).Build()
	defer queryPatch.UnPatch()
	queryWorkerPatch := mockey.Mock((*QueryNode).QuerySegments).To(func(_ *QueryNode, _ context.Context, req *querypb.QueryRequest) (*internalpb.RetrieveResults, error) {
		assert.Equal(t, []int64{7, 8}, req.GetSegmentIDs())
		return queryResult, nil
	}).Build()
	defer queryWorkerPatch.UnPatch()

	for _, asWorker := range []bool{false, true} {
		first, stream, err := queryutil.OpenQueryReduceStream(ctx, client, &querypb.QueryRequest{
			Req: &internalpb.RetrieveRequest{}, DmlChannels: []string{"a"}, SegmentIDs: []int64{7, 8},
		}, 1, asWorker)
		require.NoError(t, err)
		require.EqualValues(t, 23, first.GetMvccTimestamp())
		for i, id := range []int64{10, 20} {
			chunk, err := stream.Recv()
			require.NoError(t, err)
			require.Equal(t, []int64{id}, chunk.GetIds().GetIntId().GetData())
			require.Equal(t, []int64{id}, chunk.GetFieldsData()[0].GetScalars().GetLongData().GetData())
			if i == 0 {
				require.EqualValues(t, 7, chunk.GetAllRetrieveCount())
			} else {
				require.Zero(t, chunk.GetAllRetrieveCount())
			}
		}
		_, err = stream.Recv()
		require.ErrorIs(t, err, io.EOF)
		require.NoError(t, stream.Close())
	}
}
