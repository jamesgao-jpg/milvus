package searchutil

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/stats"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
)

func TestSearchBenchmarkMetricsNilRecorder(t *testing.T) {
	var metrics *SearchBenchmarkMetrics
	metrics.AddANNDuration(time.Second)
	metrics.AddSplitDuration(time.Second)
	metrics.AddPerVchannelReduceDuration(time.Second)
	metrics.AddFinalReduceDuration(time.Second)
	metrics.AddFinalResponseDuration(time.Second)
	metrics.RecordFirstFinalChunk()
	metrics.RecordApplicationReceive(nil, nil)
	metrics.Finish(context.Background(), nil)
}

func TestSearchBenchmarkMetricsChildAccounting(t *testing.T) {
	parent := &SearchBenchmarkMetrics{role: "proxy", requestID: 10, startedAt: time.Now()}
	child := parent.Child("streaming", "qn@11", "v1", 11)
	result := benchmarkSearchResult(7, 1, 2, 3)
	response := &viewpb.SearchOnViewStreamResponse{
		Payload: &viewpb.SearchOnViewStreamResponse_Chunk{Chunk: result},
	}

	child.RecordSendAttempt(response)
	child.RecordSendComplete(response)
	child.RecordApplicationReceive(response, result)
	child.RecordApplicationConsume(2)
	child.RecordGRPCIn(100, 105)
	child.RecordGRPCOut(20, 25)
	parent.RecordFinalResultData(result.GetResultData())

	snapshot := child.snapshot()
	require.Equal(t, int64(1), snapshot.applicationMessages)
	require.Equal(t, int64(3), snapshot.applicationReceivedUnits)
	require.Equal(t, int64(2), snapshot.applicationConsumedUnits)
	require.Equal(t, int64(1), snapshot.grpcInMessages)
	require.Equal(t, int64(100), snapshot.grpcInPayloadBytes)
	require.Equal(t, int64(105), snapshot.grpcInWireBytes)
	require.Equal(t, int64(1), snapshot.grpcOutMessages)
	require.NotZero(t, snapshot.firstApplicationResponseAfter)

	parentSnapshot := parent.snapshot()
	require.Equal(t, int64(3), parentSnapshot.finalResultCount)
	require.NotEmpty(t, parentSnapshot.finalResultHash)
}

func TestSearchBenchmarkGRPCStatsHandler(t *testing.T) {
	metrics := &SearchBenchmarkMetrics{role: "proxy", startedAt: time.Now()}
	handler := searchBenchmarkGRPCStatsHandler{enabled: true}
	ctx := WithSearchBenchmarkMetrics(context.Background(), metrics)
	ctx = handler.TagRPC(ctx, &stats.RPCTagInfo{FullMethodName: viewpb.ViewQueryService_SearchOnViewStream_FullMethodName})
	handler.HandleRPC(ctx, &stats.OutHeader{RemoteAddr: &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 19530}})
	request := &viewpb.SearchOnViewStreamRequest{
		Payload: &viewpb.SearchOnViewStreamRequest_Request{
			Request: &viewpb.SearchOnViewRequest{
				LegacyReq: &internalpb.SearchRequest{ReqID: 99},
			},
		},
	}
	handler.HandleRPC(ctx, &stats.OutPayload{Payload: request, Length: 50, WireLength: 55})
	handler.HandleRPC(ctx, &stats.InPayload{Payload: &viewpb.SearchOnViewStreamResponse{
		Payload: &viewpb.SearchOnViewStreamResponse_Chunk{Chunk: benchmarkSearchResult(99, 1)},
	}, Length: 80, WireLength: 85})
	handler.HandleRPC(ctx, &stats.End{})

	snapshot := metrics.snapshot()
	require.Equal(t, int64(99), snapshot.requestID)
	require.Equal(t, int64(1), snapshot.grpcInMessages)
	require.Equal(t, int64(80), snapshot.grpcInPayloadBytes)
	require.Equal(t, int64(1), snapshot.grpcOutMessages)
	require.Equal(t, int64(50), snapshot.grpcOutPayloadBytes)
	require.Equal(t, "127.0.0.1:19530", snapshot.transportPeer)
	require.NotZero(t, snapshot.rpcDuration)
}

func benchmarkSearchResult(requestID int64, ids ...int64) *internalpb.SearchResults {
	return &internalpb.SearchResults{
		ReqID: requestID,
		ResultData: &schemapb.SearchResultData{
			Topks: []int64{int64(len(ids))},
			Ids: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}},
			},
		},
	}
}
