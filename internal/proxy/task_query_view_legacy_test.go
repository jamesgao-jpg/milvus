// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package proxy

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/views/queryclient"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type fakeLegacyViewQueryClient struct {
	legacy queryclient.LegacyClient
}

func (c *fakeLegacyViewQueryClient) Legacy() queryclient.LegacyClient {
	return c.legacy
}

type fakeLegacyQueryClient struct {
	searchCalled int
	queryCalled  int
	searchResult *queryclient.LegacySearchResult
	queryResult  *queryclient.LegacyQueryResult
	err          error
}

type fakeLegacySearchStream struct{}

func (*fakeLegacySearchStream) Recv() (*internalpb.SearchResults, error) {
	return nil, io.EOF
}

func (*fakeLegacySearchStream) Close() error {
	return nil
}

func (*fakeLegacySearchStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, nil
}

var _ searchutil.ReduceStream = (*fakeLegacySearchStream)(nil)

type proxySearchStreamRecv struct {
	chunk *internalpb.SearchResults
	err   error
}

type proxySearchStream struct {
	recv       []proxySearchStreamRecv
	closeCalls int
}

func (s *proxySearchStream) Recv() (*internalpb.SearchResults, error) {
	if len(s.recv) == 0 {
		return nil, io.EOF
	}
	next := s.recv[0]
	s.recv = s.recv[1:]
	return next.chunk, next.err
}

func (s *proxySearchStream) Close() error {
	s.closeCalls++
	return nil
}

func (*proxySearchStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, errors.New("not implemented")
}

type proxyQueryStreamRecv struct {
	chunk *internalpb.RetrieveResults
	err   error
}

type proxyQueryStream struct {
	recv       []proxyQueryStreamRecv
	closeCalls int
}

func (s *proxyQueryStream) Recv() (*internalpb.RetrieveResults, error) {
	if len(s.recv) == 0 {
		return nil, io.EOF
	}
	next := s.recv[0]
	s.recv = s.recv[1:]
	return next.chunk, next.err
}

func (s *proxyQueryStream) Close() error {
	s.closeCalls++
	return nil
}

func (*proxyQueryStream) Interrupt() (*internalpb.RetrieveResults, error) {
	return nil, errors.New("not implemented")
}

var _ queryutil.ReduceStream = (*proxyQueryStream)(nil)

func (c *fakeLegacyQueryClient) Search(ctx context.Context, req *queryclient.LegacySearchRequest) (*queryclient.LegacySearchResult, error) {
	c.searchCalled++
	return c.searchResult, c.err
}

func (c *fakeLegacyQueryClient) Query(ctx context.Context, req *queryclient.LegacyQueryRequest) (*queryclient.LegacyQueryResult, error) {
	c.queryCalled++
	return c.queryResult, c.err
}

func TestQueryTaskExecuteUsesQueryViewLegacyClient(t *testing.T) {
	legacy := &fakeLegacyQueryClient{
		queryResult: &queryclient.LegacyQueryResult{
			Results: []*internalpb.RetrieveResults{
				{
					Base:   commonpbutil.NewMsgBase(commonpbutil.WithSourceID(101)),
					Status: merr.Success(),
				},
			},
		},
	}
	task := &queryTask{
		RetrieveRequest: &internalpb.RetrieveRequest{
			Base:         commonpbutil.NewMsgBase(commonpbutil.WithMsgID(1)),
			CollectionID: 1,
		},
		request:         &milvuspb.QueryRequest{DbName: "default"},
		viewQueryClient: &fakeLegacyViewQueryClient{legacy: legacy},
	}

	require.NoError(t, task.Execute(context.Background()))
	require.Equal(t, 1, legacy.queryCalled)
	require.Equal(t, 0, legacy.searchCalled)

	var sourceIDs []int64
	task.resultBuf.Range(func(result *internalpb.RetrieveResults) bool {
		sourceIDs = append(sourceIDs, result.GetBase().GetSourceID())
		return true
	})
	require.ElementsMatch(t, []int64{101}, sourceIDs)
}

func TestSearchTaskExecuteUsesQueryViewLegacyClient(t *testing.T) {
	legacy := &fakeLegacyQueryClient{
		searchResult: &queryclient.LegacySearchResult{
			Results: []*internalpb.SearchResults{
				{
					Base:   commonpbutil.NewMsgBase(commonpbutil.WithSourceID(202)),
					Status: merr.Success(),
				},
			},
		},
	}
	task := &searchTask{
		SearchRequest: &internalpb.SearchRequest{
			Base:         commonpbutil.NewMsgBase(commonpbutil.WithMsgID(2)),
			CollectionID: 1,
			Nq:           1,
		},
		request:         &milvuspb.SearchRequest{DbName: "default"},
		resultBuf:       typeutil.NewConcurrentSet[*internalpb.SearchResults](),
		viewQueryClient: &fakeLegacyViewQueryClient{legacy: legacy},
	}

	require.NoError(t, task.Execute(context.Background()))
	require.Equal(t, 1, legacy.searchCalled)
	require.Equal(t, 0, legacy.queryCalled)

	var sourceIDs []int64
	task.resultBuf.Range(func(result *internalpb.SearchResults) bool {
		sourceIDs = append(sourceIDs, result.GetBase().GetSourceID())
		return true
	})
	require.ElementsMatch(t, []int64{202}, sourceIDs)
	require.Zero(t, task.queryChannelsNode.Len())
}

func TestSearchTaskExecuteKeepsReduceStream(t *testing.T) {
	stream := &fakeLegacySearchStream{}
	legacy := &fakeLegacyQueryClient{
		searchResult: &queryclient.LegacySearchResult{Stream: stream},
	}
	task := &searchTask{
		SearchRequest: &internalpb.SearchRequest{
			Base:         commonpbutil.NewMsgBase(commonpbutil.WithMsgID(2)),
			CollectionID: 1,
			Nq:           1,
			Topk:         1,
		},
		request:         &milvuspb.SearchRequest{DbName: "default"},
		resultBuf:       typeutil.NewConcurrentSet[*internalpb.SearchResults](),
		viewQueryClient: &fakeLegacyViewQueryClient{legacy: legacy},
	}

	require.NoError(t, task.Execute(context.Background()))
	require.Same(t, stream, task.resultStream)
	require.Empty(t, task.resultBuf.Collect())
}

func TestSearchTaskConsumeResultStreamFailureDiscardsPartialResult(t *testing.T) {
	recvErr := errors.New("injected failure after output")
	stream := &proxySearchStream{recv: []proxySearchStreamRecv{
		{chunk: proxySearchChunk()},
		{err: recvErr},
	}}
	task := &searchTask{SearchRequest: &internalpb.SearchRequest{
		Nq:         1,
		Topk:       2,
		MetricType: "IP",
	}}

	result, _, err := task.consumeResultStream(stream, "IP", func(*internalpb.SearchResults) {})
	require.Nil(t, result)
	require.ErrorIs(t, err, recvErr)
	require.Equal(t, 1, stream.closeCalls)
}

func TestSearchTaskConsumeResultStreamPrematureEOFReturnsPartialResult(t *testing.T) {
	stream := &proxySearchStream{recv: []proxySearchStreamRecv{{chunk: proxySearchChunk()}}}
	task := &searchTask{SearchRequest: &internalpb.SearchRequest{
		Nq:         1,
		Topk:       2,
		MetricType: "IP",
	}}

	result, _, err := task.consumeResultStream(stream, "IP", func(*internalpb.SearchResults) {})
	require.NoError(t, err)
	require.Equal(t, []int64{1}, result.GetResults().GetIds().GetIntId().GetData())
	require.Equal(t, []int64{1}, result.GetResults().GetTopks())
	require.Equal(t, 1, stream.closeCalls)
}

func TestQueryTaskConsumeResultStreamFailureDiscardsPartialResult(t *testing.T) {
	recvErr := errors.New("injected failure after output")
	stream := &proxyQueryStream{recv: []proxyQueryStreamRecv{
		{chunk: proxyQueryChunk()},
		{err: recvErr},
	}}
	task := &queryTask{RetrieveRequest: &internalpb.RetrieveRequest{Limit: 2}}

	result, err := task.consumeQueryResultStream(stream)
	require.Nil(t, result)
	require.ErrorIs(t, err, recvErr)
	require.Equal(t, 1, stream.closeCalls)
}

func TestQueryTaskConsumeResultStreamPrematureEOFReturnsPartialResult(t *testing.T) {
	stream := &proxyQueryStream{recv: []proxyQueryStreamRecv{{chunk: proxyQueryChunk()}}}
	task := &queryTask{RetrieveRequest: &internalpb.RetrieveRequest{Limit: 2}}

	result, err := task.consumeQueryResultStream(stream)
	require.NoError(t, err)
	require.Equal(t, []int64{1}, result.GetIds().GetIntId().GetData())
	require.Equal(t, 1, stream.closeCalls)
}

func proxySearchChunk() *internalpb.SearchResults {
	return &internalpb.SearchResults{
		Status:     merr.Success(),
		MetricType: "IP",
		NumQueries: 1,
		TopK:       2,
		ResultData: &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       2,
			Topks:      []int64{1},
			Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{Data: []int64{1}},
			}},
			Scores: []float32{0.9},
		},
	}
}

func proxyQueryChunk() *internalpb.RetrieveResults {
	return &internalpb.RetrieveResults{
		Status: merr.Success(),
		Ids: &schemapb.IDs{IdField: &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{Data: []int64{1}},
		}},
		FieldsData: []*schemapb.FieldData{
			{
				Type:    schemapb.DataType_Int64,
				FieldId: 101,
				Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{10}}},
				}},
			},
		},
	}
}
