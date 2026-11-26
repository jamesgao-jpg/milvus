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

package hellomilvus

import (
	"cmp"
	"context"
	"crypto/rand"
	"net"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel/trace"
	collector "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

func TestReduceStreamBothHops(t *testing.T) {
	t.Setenv("OTEL_BSP_SCHEDULE_DELAY", "200")
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	require.NoError(t, err)
	recorder := &reduceStreamTraces{spans: make(map[string][]*tracepb.Span)}
	server := grpc.NewServer()
	collector.RegisterTraceServiceServer(server, recorder)
	go server.Serve(listener)
	t.Cleanup(server.Stop)

	s := &reduceStreamSuite{traces: recorder}
	s.WithMilvusConfig("trace.exporter", "otlp")
	s.WithMilvusConfig("trace.otlp.endpoint", listener.Addr().String())
	s.WithMilvusConfig("trace.otlp.method", "grpc")
	s.WithMilvusConfig("trace.otlp.secure", "false")
	s.WithMilvusConfig("proxy.reduceStreamChunkBytes", "1")
	s.WithMilvusConfig("indexCoord.segment.minSegmentNumRowsToEnableIndex", "1")
	s.WithMilvusConfig("queryNode.partialResultRequiredDataRatio", "1")
	s.WithMilvusConfig("queryCoord.autoBalance", "false")
	s.WithMilvusConfig("queryCoord.autoBalanceChannel", "false")
	suite.Run(t, s)
}

type reduceStreamSuite struct {
	integration.MiniClusterSuite
	traces *reduceStreamTraces
}

func (s *reduceStreamSuite) TestQueryAndSearch() {
	const shards, rows, dim, limit = 2, 64, 16, 8
	const groupField = "group_value"
	const indexType = integration.IndexHNSW
	ctx := s.Cluster.GetContext()
	c := s.Cluster
	c.AddQueryNode()
	name := "reduce_stream_" + funcutil.GenRandomStr()
	client := c.MilvusClient
	require := s.Require()

	collection := integration.ConstructSchemaOfVecDataType(name, dim, false, schemapb.DataType_FloatVector)
	collection.Fields = append(collection.Fields, &schemapb.FieldSchema{FieldID: 102, Name: groupField, DataType: schemapb.DataType_Int64})
	schema, err := proto.Marshal(collection)
	require.NoError(err)
	resp, err := client.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{
		CollectionName: name, Schema: schema, ShardsNum: shards,
	})
	require.NoError(merr.CheckRPCCall(resp, err))

	pks, _ := integration.GenerateChannelBalancedPrimaryKeys(integration.Int64Field, schemapb.DataType_Int64, rows, shards, 1)
	vectors := integration.NewFloatVectorFieldData(integration.FloatVecField, rows, dim)
	values := vectors.GetVectors().GetFloatVector().GetData()
	clear(values)
	groups := integration.NewInt64FieldData(groupField, rows)
	for i := range rows {
		values[i*dim] = float32(i+1) / rows
		groups.GetScalars().GetLongData().Data[i] = int64(i % 8)
	}
	insert, err := client.Insert(ctx, &milvuspb.InsertRequest{
		CollectionName: name, FieldsData: []*schemapb.FieldData{pks, vectors, groups}, NumRows: rows,
	})
	require.NoError(merr.CheckRPCCall(insert, err))
	flush, err := client.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{name}})
	require.NoError(merr.CheckRPCCall(flush, err))
	segments := flush.GetCollSegIDs()[name].GetData()
	require.Len(segments, shards, "the fixture must seal one segment per vchannel")
	s.WaitForFlush(ctx, segments, flush.GetCollFlushTs()[name], "", name)

	index, err := client.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
		CollectionName: name, FieldName: vectors.GetFieldName(),
		ExtraParams: integration.ConstructIndexParam(dim, indexType, metric.L2),
	})
	require.NoError(merr.CheckRPCCall(index, err))
	s.WaitForIndexBuilt(ctx, name, vectors.GetFieldName())
	load, err := client.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: name})
	require.NoError(merr.CheckRPCCall(load, err))
	s.WaitForLoad(ctx, name)
	s.moveLocalSegments(ctx, name, shards)

	kv := func(key, value string) *commonpb.KeyValuePair { return &commonpb.KeyValuePair{Key: key, Value: value} }
	expr := integration.Int64Field + " > 0"
	compare := func(query bool, read func(context.Context) proto.Message) {
		restore := c.MustModifyMilvusConfig(map[string]string{"proxy.enableReduceStream": "false"})
		defer restore()
		batchCtx, batchID := s.tracedContext(ctx)
		batch := read(batchCtx)
		c.MustModifyMilvusConfig(map[string]string{"proxy.enableReduceStream": "true"})
		streamCtx, streamID := s.tracedContext(ctx)
		stream := read(streamCtx)
		if query {
			s.requireBothHops(batchID, querypb.QueryNode_Query_FullMethodName, querypb.QueryNode_QuerySegments_FullMethodName, shards)
			s.requireBothHops(streamID, querypb.QueryNode_QueryReduceStream_FullMethodName, querypb.QueryNode_QueryReduceStream_FullMethodName, shards)
		} else {
			s.requireBothHops(batchID, querypb.QueryNode_Search_FullMethodName, querypb.QueryNode_SearchSegments_FullMethodName, shards)
			s.requireBothHops(streamID, querypb.QueryNode_SearchStream_FullMethodName, querypb.QueryNode_SearchSegmentsStream_FullMethodName, shards)
		}
		if !proto.Equal(batch, stream) {
			s.Require().Equal(prototext.Format(batch), prototext.Format(stream), "Batch vs. ReduceStream")
		}
	}
	window := kv("limit", strconv.Itoa(limit))
	group := kv("group_by_fields", groupField)
	for _, tc := range []struct {
		name   string
		fields []string
		params []*commonpb.KeyValuePair
	}{
		{"limited_offset", []string{integration.Int64Field, groupField}, []*commonpb.KeyValuePair{window, kv("offset", "2")}},
		{"unlimited", []string{integration.Int64Field, groupField}, nil},
		{"iterator", []string{integration.Int64Field}, []*commonpb.KeyValuePair{window, kv("iterator", "true")}},
		{"order_by", []string{integration.Int64Field, groupField}, []*commonpb.KeyValuePair{window, kv("offset", "2"), kv("order_by_fields", groupField+":desc,"+integration.Int64Field+":asc")}},
		{"group_by_keys", []string{groupField}, []*commonpb.KeyValuePair{group, kv("offset", "2"), kv("limit", "3")}},
		{"group_by_count", []string{groupField, "count(*)", "avg(" + integration.Int64Field + ")"}, []*commonpb.KeyValuePair{group, kv("offset", "1"), kv("limit", "3")}},
		{"group_by_order_by", []string{groupField, "count(*)", "sum(" + integration.Int64Field + ")"}, []*commonpb.KeyValuePair{group, kv("order_by_fields", groupField+":desc"), kv("limit", "3")}},
		{"global_aggregates", []string{"count(*)", "sum(" + groupField + ")", "avg(" + groupField + ")", "min(" + groupField + ")", "max(" + groupField + ")"}, nil},
	} {
		s.Run("query_"+tc.name, func() {
			compare(true, func(ctx context.Context) proto.Message {
				result, err := client.Query(ctx, &milvuspb.QueryRequest{CollectionName: name, Expr: expr, OutputFields: tc.fields, QueryParams: tc.params})
				s.Require().NoError(merr.CheckRPCCall(result, err))
				s.Require().NotEmpty(result.GetFieldsData())
				// Proxy builds the projection from maps. Compare columns by name without reordering rows.
				slices.Sort(result.OutputFields)
				slices.SortFunc(result.FieldsData, func(a, b *schemapb.FieldData) int {
					return cmp.Compare(a.GetFieldName(), b.GetFieldName())
				})
				return &milvuspb.QueryResults{FieldsData: result.GetFieldsData(), OutputFields: result.GetOutputFields()}
			})
		})
	}
	baseSearch := integration.ConstructSearchRequest("", name, groupField+" >= 2", vectors.GetFieldName(),
		schemapb.DataType_FloatVector, []string{groupField}, metric.L2, map[string]any{"ef": 64}, 1, dim, limit, -1)
	placeholder := &commonpb.PlaceholderGroup{}
	require.NoError(proto.Unmarshal(baseSearch.GetPlaceholderGroup(), placeholder))
	clear(placeholder.GetPlaceholders()[0].GetValues()[0])
	input, err := proto.Marshal(placeholder)
	require.NoError(err)
	baseSearch.SearchInput = &milvuspb.SearchRequest_PlaceholderGroup{PlaceholderGroup: input}
	iterator := kv("iterator", "true")
	searchGroup := []*commonpb.KeyValuePair{kv("group_by_field", groupField), kv("group_size", "2")}
	stringParam := func(value string) *schemapb.FunctionParamValue {
		return &schemapb.FunctionParamValue{Value: &schemapb.FunctionParamValue_StringValue{StringValue: value}}
	}
	scoreChain := func(stage schemapb.FunctionChainStage) *schemapb.FunctionChain {
		return &schemapb.FunctionChain{Stage: stage, Ops: []*schemapb.FunctionChainOp{{
			Op: "map", Outputs: []string{"$score"},
			Expr: &schemapb.FunctionChainExpr{
				Name: "num_combine",
				Args: []*schemapb.FunctionChainExprArg{
					{Arg: &schemapb.FunctionChainExprArg_Column{Column: &schemapb.FunctionChainColumnArg{Name: "$score"}}},
					{Arg: &schemapb.FunctionChainExprArg_Column{Column: &schemapb.FunctionChainColumnArg{Name: groupField}}},
				},
				Params: map[string]*schemapb.FunctionParamValue{"mode": stringParam("sum")},
			},
		}}}
	}
	l0 := scoreChain(schemapb.FunctionChainStage_FunctionChainStageL0Rerank)
	l1 := scoreChain(schemapb.FunctionChainStage_FunctionChainStageL1Rerank)
	l2 := scoreChain(schemapb.FunctionChainStage_FunctionChainStageL2Rerank)
	l2.Ops = append(l2.Ops, &schemapb.FunctionChainOp{
		Op: "sort", Inputs: []string{"$score", "$id"},
		Params: map[string]*schemapb.FunctionParamValue{"desc": {Value: &schemapb.FunctionParamValue_BoolValue{BoolValue: true}}},
	})
	for _, tc := range []struct {
		name   string
		params []*commonpb.KeyValuePair
		score  *schemapb.FunctionScore
		chains []*schemapb.FunctionChain
		agg    *commonpb.SearchAggregationSpec
	}{
		{name: "group_by", params: searchGroup},
		{name: "group_offset", params: []*commonpb.KeyValuePair{kv("group_by_field", groupField), kv("group_size", "3"), kv("topk", "2"), kv("offset", "1")}},
		{name: "composite_groups", params: []*commonpb.KeyValuePair{kv("group_by_fields", groupField+","+integration.Int64Field), kv("group_size", "2")}},
		{name: "aggregation", agg: &commonpb.SearchAggregationSpec{Fields: []string{groupField}, Size: 3, TopHits: &commonpb.TopHitsSpec{Size: 2}}},
		{name: "range_filter", params: []*commonpb.KeyValuePair{kv("params", `{"ef":64,"radius":0.7,"range_filter":0.1}`)}},
		{name: "iterative_filter", params: []*commonpb.KeyValuePair{kv("hints", "iterative_filter")}},
		{name: "iterator", params: []*commonpb.KeyValuePair{iterator}},
		{name: "iterator_v2", params: []*commonpb.KeyValuePair{iterator, kv("search_iter_v2", "true"), kv("search_iter_batch_size", "8"), kv("search_iter_id", "0c983c5c-214a-4473-9bad-7d07e7d5e5ac")}},
		{name: "boost", score: &schemapb.FunctionScore{Functions: []*schemapb.FunctionSchema{{
			Type:   schemapb.FunctionType_Rerank,
			Params: []*commonpb.KeyValuePair{kv("reranker", "boost"), kv("filter", groupField+" == 2"), kv("weight", "1.5")},
		}}}},
		{name: "rerank", score: &schemapb.FunctionScore{Functions: []*schemapb.FunctionSchema{{
			Type: schemapb.FunctionType_Rerank, InputFieldNames: []string{groupField},
			Params: []*commonpb.KeyValuePair{kv("reranker", "decay"), kv("origin", "4"), kv("scale", "4"), kv("offset", "1"), kv("decay", "0.5"), kv("function", "gauss")},
		}}}},
		{name: "l0_l1_l2", chains: []*schemapb.FunctionChain{l0, l1, l2}},
	} {
		s.Run("search_"+tc.name, func() {
			compare(false, func(ctx context.Context) proto.Message {
				req := proto.CloneOf(baseSearch)
				for _, p := range tc.params {
					i := slices.IndexFunc(req.SearchParams, func(kv *commonpb.KeyValuePair) bool { return kv.GetKey() == p.GetKey() })
					if i < 0 {
						req.SearchParams = append(req.SearchParams, p)
					} else {
						req.SearchParams[i] = p
					}
				}
				req.FunctionScore, req.FunctionChains, req.SearchAggregation = tc.score, tc.chains, tc.agg
				result, err := client.Search(ctx, req)
				s.Require().NoError(merr.CheckRPCCall(result, err))
				if tc.agg == nil {
					s.Require().NotEmpty(result.GetResults().GetScores())
				} else {
					s.Require().Len(result.GetResults().GetAggBuckets(), int(tc.agg.GetSize()))
				}
				return result.GetResults()
			})
		})
	}
	// Grouped Hybrid preserves the vchannel arrival order before fusion.
	// Weighted fusion scores the candidates independently of that order.
	for _, variant := range []string{"rrf", "weighted_offset", "weighted_empty", "group_by_weighted", "group_by_weighted_l0_l1_l2"} {
		s.Run("hybrid_"+variant, func() {
			compare(false, func(ctx context.Context) proto.Message {
				first, second := proto.CloneOf(baseSearch), proto.CloneOf(baseSearch)
				second.Dsl = groupField + " <= 5"
				req := &milvuspb.HybridSearchRequest{
					CollectionName: name, Requests: []*milvuspb.SearchRequest{first, second},
					OutputFields: []string{groupField}, RankParams: []*commonpb.KeyValuePair{window},
				}
				if strings.HasPrefix(variant, "group_by") {
					req.RankParams = append(req.RankParams, searchGroup...)
				}
				switch variant {
				case "group_by_weighted_l0_l1_l2":
					first.FunctionChains, second.FunctionChains = []*schemapb.FunctionChain{l0}, []*schemapb.FunctionChain{l1}
					fuse := proto.CloneOf(l2)
					weights := &schemapb.FunctionParamValue_ArrayValue{ArrayValue: &schemapb.FunctionParamArray{Values: []*schemapb.FunctionParamValue{
						{Value: &schemapb.FunctionParamValue_DoubleValue{DoubleValue: 0.7}},
						{Value: &schemapb.FunctionParamValue_DoubleValue{DoubleValue: 0.3}},
					}}}
					fuse.Ops = append([]*schemapb.FunctionChainOp{{Op: "merge", Params: map[string]*schemapb.FunctionParamValue{
						"strategy": stringParam("weighted"), "weights": {Value: weights},
					}}}, fuse.Ops...)
					req.FunctionChains = []*schemapb.FunctionChain{fuse}
				case "weighted_offset", "weighted_empty", "group_by_weighted":
					req.RankParams = append(req.RankParams, kv("strategy", "weighted"), kv("params", `{"weights":[0.7,0.3]}`))
					if variant == "weighted_empty" {
						second.Dsl = groupField + " < 0"
					}
					if variant == "weighted_offset" {
						req.RankParams = append(req.RankParams, kv("offset", "1"))
						for _, p := range second.SearchParams {
							if p.Key == "topk" {
								p.Value = "3"
							}
						}
						two := proto.CloneOf(placeholder)
						two.Placeholders[0].Values = append(two.Placeholders[0].Values, two.Placeholders[0].Values[0])
						input, err := proto.Marshal(two)
						s.Require().NoError(err)
						for _, sub := range req.Requests {
							sub.Nq, sub.SearchInput = 2, &milvuspb.SearchRequest_PlaceholderGroup{PlaceholderGroup: input}
						}
					}
				default:
					req.RankParams = append(req.RankParams, kv("strategy", "rrf"), kv("params", `{"k":60}`))
				}
				result, err := client.HybridSearch(ctx, req)
				s.Require().NoError(merr.CheckRPCCall(result, err))
				s.Require().NotEmpty(result.GetResults().GetScores())
				return result.GetResults()
			})
		})
	}
}

// A worker on the leader uses an in-process stream, which is valid but
// cannot prove the second gRPC hop. Place sealed segments on remote nodes.
func (s *reduceStreamSuite) moveLocalSegments(ctx context.Context, name string, shards int) {
	c := s.Cluster
	require := s.Require()
	desc, err := c.MilvusClient.DescribeCollection(ctx, &milvuspb.DescribeCollectionRequest{CollectionName: name})
	require.NoError(merr.CheckRPCCall(desc, err))
	var leaders *querypb.GetShardLeadersResponse
	require.Eventually(func() bool {
		resp, err := c.MixCoordClient.GetShardLeaders(ctx, &querypb.GetShardLeadersRequest{CollectionID: desc.GetCollectionID()})
		leaders = resp
		return merr.CheckRPCCall(resp, err) == nil && len(resp.GetShards()) == shards
	}, 20*time.Second, 100*time.Millisecond, "both vchannels must be serviceable")
	nodes := c.GetAllQueryNodes()
	require.Len(nodes, 2)
	for _, shard := range leaders.GetShards() {
		require.Len(shard.GetNodeIds(), 1)
		leaderID := shard.GetNodeIds()[0]
		for i, node := range nodes {
			if node.GetNodeID() != leaderID {
				continue
			}
			dist, err := node.MustGetClient(ctx).GetDataDistribution(ctx, &querypb.GetDataDistributionRequest{})
			require.NoError(merr.CheckRPCCall(dist, err))
			var local []int64
			for _, seg := range dist.GetSegments() {
				if seg.GetCollection() == desc.GetCollectionID() && seg.GetChannel() == shard.GetChannelName() {
					local = append(local, seg.GetID())
				}
			}
			if len(local) != 0 {
				status, err := c.MilvusClient.LoadBalance(ctx, &milvuspb.LoadBalanceRequest{
					CollectionName: name, SrcNodeID: leaderID, SealedSegmentIDs: local,
					DstNodeIDs: []int64{nodes[1-i].GetNodeID()},
				})
				require.NoError(merr.CheckRPCCall(status, err))
			}
		}
	}
}

func (s *reduceStreamSuite) tracedContext(ctx context.Context) (context.Context, trace.TraceID) {
	var id trace.TraceID
	_, err := rand.Read(id[:])
	s.Require().NoError(err)
	return metadata.AppendToOutgoingContext(ctx, "traceparent", "00-"+id.String()+"-0000000000000001-01"), id
}

func (s *reduceStreamSuite) requireBothHops(id trace.TraceID, leaderRPC, workerRPC string, shards int) {
	s.Require().Eventually(func() bool {
		s.traces.mu.Lock()
		defer s.traces.mu.Unlock()
		spans := s.traces.spans[string(id[:])]
		byID := make(map[string]*tracepb.Span, len(spans))
		for _, span := range spans {
			byID[string(span.GetSpanId())] = span
		}
		leadersWithWorker := make(map[string]struct{}, shards)
		for _, span := range spans {
			if !isReadRPCSpan(span, workerRPC) {
				continue
			}
			for ancestor := byID[string(span.GetParentSpanId())]; ancestor != nil; ancestor = byID[string(ancestor.GetParentSpanId())] {
				if isReadRPCSpan(ancestor, leaderRPC) {
					leadersWithWorker[string(ancestor.GetSpanId())] = struct{}{}
					break
				}
			}
		}
		return len(leadersWithWorker) == shards
	}, 20*time.Second, 100*time.Millisecond, "expected %s -> %s for both vchannels (trace %s)", leaderRPC, workerRPC, id)
}

func isReadRPCSpan(span *tracepb.Span, method string) bool {
	if span.GetKind() != tracepb.Span_SPAN_KIND_SERVER || span.GetName() != strings.TrimPrefix(method, "/") {
		return false
	}
	for _, attribute := range span.GetAttributes() {
		if attribute.GetKey() == "rpc.grpc.status_code" {
			// A parent may close a successful read once its candidate window is full.
			code := attribute.GetValue().GetIntValue()
			return code == int64(codes.OK) || code == int64(codes.Canceled)
		}
	}
	return false
}

type reduceStreamTraces struct {
	collector.UnimplementedTraceServiceServer
	mu    sync.Mutex
	spans map[string][]*tracepb.Span
}

func (r *reduceStreamTraces) Export(_ context.Context, req *collector.ExportTraceServiceRequest) (*collector.ExportTraceServiceResponse, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, resource := range req.GetResourceSpans() {
		for _, scope := range resource.GetScopeSpans() {
			for _, span := range scope.GetSpans() {
				key := string(span.GetTraceId())
				r.spans[key] = append(r.spans[key], span)
			}
		}
	}
	return &collector.ExportTraceServiceResponse{}, nil
}
