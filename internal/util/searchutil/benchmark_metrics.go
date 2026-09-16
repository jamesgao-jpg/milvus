package searchutil

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"sync"
	"time"

	"google.golang.org/grpc/stats"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

const searchBenchmarkLogMessage = "query view search benchmark metrics"

type searchBenchmarkMetricsKey struct{}
type searchBenchmarkStatsMetricsKey struct{}

type SearchBenchmarkMetrics struct {
	mu sync.Mutex

	parent   *SearchBenchmarkMetrics
	children []*SearchBenchmarkMetrics

	role      string
	mode      string
	method    string
	workNode  string
	vchannel  string
	requestID int64
	nodeID    int64
	startedAt time.Time

	annDuration                   time.Duration
	splitDuration                 time.Duration
	perVchannelReduceDuration     time.Duration
	finalReduceDuration           time.Duration
	finalResponseDuration         time.Duration
	rpcDuration                   time.Duration
	firstApplicationResponseAfter time.Duration
	firstFinalChunkAfter          time.Duration

	generatedResultBytes     int64
	generatedUnits           int64
	sendAttemptedMessages    int64
	sendAttemptedBytes       int64
	sendCompletedMessages    int64
	sendCompletedBytes       int64
	grpcInMessages           int64
	grpcInPayloadBytes       int64
	grpcInWireBytes          int64
	grpcOutMessages          int64
	grpcOutPayloadBytes      int64
	grpcOutWireBytes         int64
	applicationMessages      int64
	applicationBytes         int64
	applicationReceivedUnits int64
	applicationConsumedUnits int64
	finalResultCount         int64
	finalResultHash          string
	transportPeer            string
	transportStart           searchBenchmarkTransportSnapshot
	transportDelta           searchBenchmarkTransportSnapshot
	transportStarted         bool
	errorText                string
	finished                 bool
}

func SearchBenchmarkMetricsEnabled() bool {
	return paramtable.Get().ProxyCfg.EnableSearchBenchmarkMetrics.GetAsBool()
}

func NewSearchBenchmarkMetrics(role, mode string, requestID int64) *SearchBenchmarkMetrics {
	if !SearchBenchmarkMetricsEnabled() {
		return nil
	}
	return &SearchBenchmarkMetrics{
		role:      role,
		mode:      mode,
		requestID: requestID,
		nodeID:    paramtable.GetNodeID(),
		startedAt: time.Now(),
	}
}

func WithSearchBenchmarkMetrics(ctx context.Context, metrics *SearchBenchmarkMetrics) context.Context {
	if metrics == nil {
		return ctx
	}
	return context.WithValue(ctx, searchBenchmarkMetricsKey{}, metrics)
}

func SearchBenchmarkMetricsFromContext(ctx context.Context) *SearchBenchmarkMetrics {
	metrics, _ := ctx.Value(searchBenchmarkMetricsKey{}).(*SearchBenchmarkMetrics)
	return metrics
}

func (m *SearchBenchmarkMetrics) Child(mode, workNode, vchannel string, nodeID int64) *SearchBenchmarkMetrics {
	if m == nil {
		return nil
	}
	child := &SearchBenchmarkMetrics{
		parent:    m,
		role:      m.role,
		mode:      mode,
		workNode:  workNode,
		vchannel:  vchannel,
		requestID: m.requestID,
		nodeID:    nodeID,
		startedAt: time.Now(),
	}
	m.mu.Lock()
	m.children = append(m.children, child)
	m.mu.Unlock()
	return child
}

func (m *SearchBenchmarkMetrics) SetMode(mode string) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.mode = mode
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) SetRequest(req *viewpb.SearchOnViewRequest) {
	if m == nil || req == nil {
		return
	}
	m.mu.Lock()
	if legacy := req.GetLegacyReq(); legacy != nil {
		m.requestID = legacy.GetReqID()
	}
	if shard := req.GetShardId(); shard != nil {
		m.vchannel = shard.GetVchannel()
	}
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) AddANNDuration(duration time.Duration) {
	if m == nil {
		return
	}
	m.addDuration(&m.annDuration, duration)
}

func (m *SearchBenchmarkMetrics) AddSplitDuration(duration time.Duration) {
	if m == nil {
		return
	}
	m.addDuration(&m.splitDuration, duration)
}

func (m *SearchBenchmarkMetrics) AddPerVchannelReduceDuration(duration time.Duration) {
	if m == nil {
		return
	}
	m.addDuration(&m.perVchannelReduceDuration, duration)
}

func (m *SearchBenchmarkMetrics) AddFinalReduceDuration(duration time.Duration) {
	if m == nil {
		return
	}
	m.addDuration(&m.finalReduceDuration, duration)
}

func (m *SearchBenchmarkMetrics) AddFinalResponseDuration(duration time.Duration) {
	if m == nil {
		return
	}
	m.addDuration(&m.finalResponseDuration, duration)
}

func (m *SearchBenchmarkMetrics) addDuration(target *time.Duration, duration time.Duration) {
	m.mu.Lock()
	*target += duration
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordGeneratedResult(result *internalpb.SearchResults) {
	if m == nil || result == nil {
		return
	}
	m.mu.Lock()
	m.generatedResultBytes = int64(proto.Size(result))
	m.generatedUnits = searchResultUnits(result)
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordSendAttempt(message proto.Message) {
	if m == nil || message == nil {
		return
	}
	m.mu.Lock()
	m.sendAttemptedMessages++
	m.sendAttemptedBytes += int64(proto.Size(message))
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordSendComplete(message proto.Message) {
	if m == nil || message == nil {
		return
	}
	m.mu.Lock()
	m.sendCompletedMessages++
	m.sendCompletedBytes += int64(proto.Size(message))
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordApplicationReceive(message proto.Message, result *internalpb.SearchResults) {
	if m == nil || message == nil {
		return
	}
	m.mu.Lock()
	m.applicationMessages++
	m.applicationBytes += int64(proto.Size(message))
	m.applicationReceivedUnits += searchResultUnits(result)
	first := m.firstApplicationResponseAfter == 0
	if first {
		m.firstApplicationResponseAfter = time.Since(m.startedAt)
	}
	m.mu.Unlock()
	if first && m.parent != nil {
		m.parent.recordFirstApplicationResponse()
	}
}

func (m *SearchBenchmarkMetrics) RecordApplicationConsume(units int64) {
	if m == nil || units <= 0 {
		return
	}
	m.mu.Lock()
	m.applicationConsumedUnits += units
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) recordFirstApplicationResponse() {
	if m == nil {
		return
	}
	m.mu.Lock()
	if m.firstApplicationResponseAfter == 0 {
		m.firstApplicationResponseAfter = time.Since(m.startedAt)
	}
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordFirstFinalChunk() {
	if m == nil {
		return
	}
	m.mu.Lock()
	if m.firstFinalChunkAfter == 0 {
		m.firstFinalChunkAfter = time.Since(m.startedAt)
	}
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordGRPCIn(length, wireLength int) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.grpcInMessages++
	m.grpcInPayloadBytes += int64(length)
	m.grpcInWireBytes += int64(wireLength)
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordGRPCOut(length, wireLength int) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.grpcOutMessages++
	m.grpcOutPayloadBytes += int64(length)
	m.grpcOutWireBytes += int64(wireLength)
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) startTransport(peer string, tracker *searchBenchmarkConnectionTracker) {
	if m == nil || peer == "" || tracker == nil {
		return
	}
	m.mu.Lock()
	if !m.transportStarted {
		m.transportPeer = peer
		m.transportStart = tracker.snapshot(peer)
		m.transportStarted = true
	}
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) finishTransport(tracker *searchBenchmarkConnectionTracker) {
	if m == nil || tracker == nil {
		return
	}
	m.mu.Lock()
	if m.transportStarted {
		m.transportDelta = tracker.snapshot(m.transportPeer).since(m.transportStart)
	}
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) RecordFinalResultData(data *schemapb.SearchResultData) {
	if m == nil || data == nil {
		return
	}
	m.mu.Lock()
	for _, count := range data.GetTopks() {
		m.finalResultCount += count
	}
	hashData := &schemapb.SearchResultData{
		Ids:    data.GetIds(),
		Scores: data.GetScores(),
		Topks:  data.GetTopks(),
	}
	encoded, err := proto.MarshalOptions{Deterministic: true}.Marshal(hashData)
	if err == nil {
		hash := sha256.Sum256(encoded)
		m.finalResultHash = hex.EncodeToString(hash[:])
	}
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) recordPayloadIdentity(payload any) {
	switch value := payload.(type) {
	case *viewpb.SearchOnViewRequest:
		m.SetRequest(value)
	case *viewpb.SearchOnViewStreamRequest:
		m.SetRequest(value.GetRequest())
	case *viewpb.SearchOnViewResponse:
		if result := value.GetLegacyResults(); result != nil {
			m.setRequestID(result.GetReqID())
		}
	case *viewpb.SearchOnViewStreamResponse:
		if result := value.GetChunk(); result != nil {
			m.setRequestID(result.GetReqID())
		}
	}
}

func (m *SearchBenchmarkMetrics) setRequestID(requestID int64) {
	if m == nil || requestID == 0 {
		return
	}
	m.mu.Lock()
	m.requestID = requestID
	m.mu.Unlock()
}

func (m *SearchBenchmarkMetrics) Finish(ctx context.Context, err error) {
	if m == nil || m.parent != nil {
		return
	}
	m.mu.Lock()
	if m.finished {
		m.mu.Unlock()
		return
	}
	m.finished = true
	if err != nil {
		m.errorText = err.Error()
	}
	children := append([]*SearchBenchmarkMetrics(nil), m.children...)
	m.mu.Unlock()
	for _, child := range children {
		child.finishTransport(searchBenchmarkClientConnections)
		child.mu.Lock()
		if child.rpcDuration == 0 {
			child.rpcDuration = time.Since(child.startedAt)
		}
		child.mu.Unlock()
	}

	snapshot := m.snapshot()
	sort.Slice(children, func(i, j int) bool {
		return children[i].snapshot().workNode < children[j].snapshot().workNode
	})
	childWorkNodes := make([]string, 0, len(children))
	childVChannels := make([]string, 0, len(children))
	childTransportPeers := make([]string, 0, len(children))
	childNodeIDs := make([]int64, 0, len(children))
	childReceivedMessages := make([]int64, 0, len(children))
	childReceivedBytes := make([]int64, 0, len(children))
	childReceivedUnits := make([]int64, 0, len(children))
	childConsumedUnits := make([]int64, 0, len(children))
	childGRPCInMessages := make([]int64, 0, len(children))
	childGRPCInBytes := make([]int64, 0, len(children))
	childGRPCInWireBytes := make([]int64, 0, len(children))
	childConnectionReadBytes := make([]int64, 0, len(children))
	childTCPBytesReceived := make([]int64, 0, len(children))
	childTCPInfoAvailable := make([]bool, 0, len(children))
	childRPCNanos := make([]int64, 0, len(children))
	childFirstResponseNanos := make([]int64, 0, len(children))
	for _, child := range children {
		value := child.snapshot()
		childWorkNodes = append(childWorkNodes, value.workNode)
		childVChannels = append(childVChannels, value.vchannel)
		childTransportPeers = append(childTransportPeers, value.transportPeer)
		childNodeIDs = append(childNodeIDs, value.nodeID)
		childReceivedMessages = append(childReceivedMessages, value.applicationMessages)
		childReceivedBytes = append(childReceivedBytes, value.applicationBytes)
		childReceivedUnits = append(childReceivedUnits, value.applicationReceivedUnits)
		childConsumedUnits = append(childConsumedUnits, value.applicationConsumedUnits)
		childGRPCInMessages = append(childGRPCInMessages, value.grpcInMessages)
		childGRPCInBytes = append(childGRPCInBytes, value.grpcInPayloadBytes)
		childGRPCInWireBytes = append(childGRPCInWireBytes, value.grpcInWireBytes)
		childConnectionReadBytes = append(childConnectionReadBytes, int64(value.transportDelta.connectionReadBytes))
		childTCPBytesReceived = append(childTCPBytesReceived, int64(value.transportDelta.tcpBytesReceived))
		childTCPInfoAvailable = append(childTCPInfoAvailable, value.transportDelta.tcpInfoAvailable)
		childRPCNanos = append(childRPCNanos, value.rpcDuration.Nanoseconds())
		childFirstResponseNanos = append(childFirstResponseNanos, value.firstApplicationResponseAfter.Nanoseconds())
	}

	mlog.Info(ctx, searchBenchmarkLogMessage,
		mlog.String("role", snapshot.role),
		mlog.String("mode", snapshot.mode),
		mlog.String("method", snapshot.method),
		mlog.String("workNode", snapshot.workNode),
		mlog.FieldNodeID(snapshot.nodeID),
		mlog.FieldVChannel(snapshot.vchannel),
		mlog.Int64("requestID", snapshot.requestID),
		mlog.Duration("requestDuration", time.Since(snapshot.startedAt)),
		mlog.Duration("annDuration", snapshot.annDuration),
		mlog.Duration("splitDuration", snapshot.splitDuration),
		mlog.Duration("perVchannelReduceDuration", snapshot.perVchannelReduceDuration),
		mlog.Duration("finalReduceDuration", snapshot.finalReduceDuration),
		mlog.Duration("finalResponseDuration", snapshot.finalResponseDuration),
		mlog.Duration("rpcDuration", snapshot.rpcDuration),
		mlog.Duration("firstApplicationResponseAfter", snapshot.firstApplicationResponseAfter),
		mlog.Duration("firstFinalChunkAfter", snapshot.firstFinalChunkAfter),
		mlog.Int64("generatedResultBytes", snapshot.generatedResultBytes),
		mlog.Int64("generatedUnits", snapshot.generatedUnits),
		mlog.Int64("sendAttemptedMessages", snapshot.sendAttemptedMessages),
		mlog.Int64("sendAttemptedBytes", snapshot.sendAttemptedBytes),
		mlog.Int64("sendCompletedMessages", snapshot.sendCompletedMessages),
		mlog.Int64("sendCompletedBytes", snapshot.sendCompletedBytes),
		mlog.Int64("grpcInMessages", snapshot.grpcInMessages),
		mlog.Int64("grpcInPayloadBytes", snapshot.grpcInPayloadBytes),
		mlog.Int64("grpcInWireBytes", snapshot.grpcInWireBytes),
		mlog.Int64("grpcOutMessages", snapshot.grpcOutMessages),
		mlog.Int64("grpcOutPayloadBytes", snapshot.grpcOutPayloadBytes),
		mlog.Int64("grpcOutWireBytes", snapshot.grpcOutWireBytes),
		mlog.Int64("applicationReceivedMessages", snapshot.applicationMessages),
		mlog.Int64("applicationReceivedBytes", snapshot.applicationBytes),
		mlog.Int64("applicationReceivedUnits", snapshot.applicationReceivedUnits),
		mlog.Int64("applicationConsumedUnits", snapshot.applicationConsumedUnits),
		mlog.Int64("finalResultCount", snapshot.finalResultCount),
		mlog.String("finalResultHash", snapshot.finalResultHash),
		mlog.String("transportPeer", snapshot.transportPeer),
		mlog.String("transportAttribution", "peer_connection_interval"),
		mlog.Uint64("connectionReadBytes", snapshot.transportDelta.connectionReadBytes),
		mlog.Uint64("connectionWriteBytes", snapshot.transportDelta.connectionWriteBytes),
		mlog.Uint64("tcpBytesReceived", snapshot.transportDelta.tcpBytesReceived),
		mlog.Uint64("tcpBytesSent", snapshot.transportDelta.tcpBytesSent),
		mlog.Uint64("tcpBytesAcked", snapshot.transportDelta.tcpBytesAcked),
		mlog.Uint64("tcpNotSentBytes", snapshot.transportDelta.tcpNotSentBytes),
		mlog.Bool("tcpInfoAvailable", snapshot.transportDelta.tcpInfoAvailable),
		mlog.String("error", snapshot.errorText),
		mlog.Strings("childWorkNodes", childWorkNodes),
		mlog.Strings("childVChannels", childVChannels),
		mlog.Strings("childTransportPeers", childTransportPeers),
		mlog.Int64s("childNodeIDs", childNodeIDs),
		mlog.Int64s("childApplicationReceivedMessages", childReceivedMessages),
		mlog.Int64s("childApplicationReceivedBytes", childReceivedBytes),
		mlog.Int64s("childApplicationReceivedUnits", childReceivedUnits),
		mlog.Int64s("childApplicationConsumedUnits", childConsumedUnits),
		mlog.Int64s("childGRPCInMessages", childGRPCInMessages),
		mlog.Int64s("childGRPCInPayloadBytes", childGRPCInBytes),
		mlog.Int64s("childGRPCInWireBytes", childGRPCInWireBytes),
		mlog.Int64s("childConnectionReadBytes", childConnectionReadBytes),
		mlog.Int64s("childTCPBytesReceived", childTCPBytesReceived),
		mlog.Bools("childTCPInfoAvailable", childTCPInfoAvailable),
		mlog.Int64s("childRPCNanos", childRPCNanos),
		mlog.Int64s("childFirstResponseNanos", childFirstResponseNanos),
	)
}

type searchBenchmarkSnapshot struct {
	role, mode, method, workNode, vchannel, errorText, finalResultHash string
	requestID, nodeID                                                  int64
	startedAt                                                          time.Time
	annDuration, splitDuration, perVchannelReduceDuration              time.Duration
	finalReduceDuration, finalResponseDuration, rpcDuration            time.Duration
	firstApplicationResponseAfter, firstFinalChunkAfter                time.Duration
	generatedResultBytes, generatedUnits                               int64
	sendAttemptedMessages, sendAttemptedBytes                          int64
	sendCompletedMessages, sendCompletedBytes                          int64
	grpcInMessages, grpcInPayloadBytes, grpcInWireBytes                int64
	grpcOutMessages, grpcOutPayloadBytes, grpcOutWireBytes             int64
	applicationMessages, applicationBytes, applicationReceivedUnits    int64
	applicationConsumedUnits                                           int64
	finalResultCount                                                   int64
	transportPeer                                                      string
	transportDelta                                                     searchBenchmarkTransportSnapshot
}

func (m *SearchBenchmarkMetrics) snapshot() searchBenchmarkSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	return searchBenchmarkSnapshot{
		role: m.role, mode: m.mode, method: m.method, workNode: m.workNode,
		vchannel: m.vchannel, requestID: m.requestID, nodeID: m.nodeID,
		startedAt: m.startedAt, annDuration: m.annDuration, splitDuration: m.splitDuration,
		perVchannelReduceDuration: m.perVchannelReduceDuration,
		finalReduceDuration:       m.finalReduceDuration, finalResponseDuration: m.finalResponseDuration,
		rpcDuration: m.rpcDuration, firstApplicationResponseAfter: m.firstApplicationResponseAfter,
		firstFinalChunkAfter: m.firstFinalChunkAfter, generatedResultBytes: m.generatedResultBytes,
		generatedUnits: m.generatedUnits, sendAttemptedMessages: m.sendAttemptedMessages,
		sendAttemptedBytes: m.sendAttemptedBytes, sendCompletedMessages: m.sendCompletedMessages,
		sendCompletedBytes: m.sendCompletedBytes, grpcInMessages: m.grpcInMessages,
		grpcInPayloadBytes: m.grpcInPayloadBytes, grpcInWireBytes: m.grpcInWireBytes,
		grpcOutMessages: m.grpcOutMessages, grpcOutPayloadBytes: m.grpcOutPayloadBytes,
		grpcOutWireBytes: m.grpcOutWireBytes, applicationMessages: m.applicationMessages,
		applicationBytes: m.applicationBytes, applicationReceivedUnits: m.applicationReceivedUnits,
		applicationConsumedUnits: m.applicationConsumedUnits,
		finalResultCount:         m.finalResultCount, finalResultHash: m.finalResultHash,
		transportPeer: m.transportPeer, transportDelta: m.transportDelta, errorText: m.errorText,
	}
}

func searchResultUnits(result *internalpb.SearchResults) int64 {
	if result == nil {
		return 0
	}
	data := result.GetResultData()
	if len(result.GetSlicedBlob()) > 0 {
		data, _ = decodeChunk(result, result.GetNumQueries(), result.GetTopK())
	}
	if data == nil {
		return result.GetSlicedNumCount()
	}
	var total int64
	for _, count := range data.GetTopks() {
		total += count
	}
	return total
}

type searchBenchmarkGRPCStatsHandler struct {
	server, enabled bool
}

func NewSearchBenchmarkGRPCStatsHandler(server bool) stats.Handler {
	return searchBenchmarkGRPCStatsHandler{server: server, enabled: SearchBenchmarkMetricsEnabled()}
}

func (h searchBenchmarkGRPCStatsHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	if !h.enabled || !isSearchBenchmarkMethod(info.FullMethodName) {
		return ctx
	}
	metrics := SearchBenchmarkMetricsFromContext(ctx)
	if metrics == nil && h.server {
		mode := "batch"
		if info.FullMethodName == viewpb.ViewQueryService_SearchOnViewStream_FullMethodName {
			mode = "streaming"
		}
		metrics = NewSearchBenchmarkMetrics(paramtable.GetRole(), mode, 0)
	}
	if metrics == nil {
		return ctx
	}
	metrics.mu.Lock()
	metrics.method = info.FullMethodName
	metrics.mu.Unlock()
	ctx = WithSearchBenchmarkMetrics(ctx, metrics)
	return context.WithValue(ctx, searchBenchmarkStatsMetricsKey{}, metrics)
}

func (h searchBenchmarkGRPCStatsHandler) HandleRPC(ctx context.Context, event stats.RPCStats) {
	metrics, _ := ctx.Value(searchBenchmarkStatsMetricsKey{}).(*SearchBenchmarkMetrics)
	if metrics == nil {
		return
	}
	switch value := event.(type) {
	case *stats.InPayload:
		metrics.recordPayloadIdentity(value.Payload)
		metrics.RecordGRPCIn(value.Length, value.WireLength)
	case *stats.InHeader:
		if value.RemoteAddr != nil {
			metrics.startTransport(value.RemoteAddr.String(), h.connectionTracker())
		}
	case *stats.OutHeader:
		if value.RemoteAddr != nil {
			metrics.startTransport(value.RemoteAddr.String(), h.connectionTracker())
		}
	case *stats.OutPayload:
		metrics.recordPayloadIdentity(value.Payload)
		metrics.RecordGRPCOut(value.Length, value.WireLength)
	case *stats.End:
		metrics.finishTransport(h.connectionTracker())
		metrics.mu.Lock()
		metrics.rpcDuration = time.Since(metrics.startedAt)
		metrics.mu.Unlock()
		if h.server {
			metrics.Finish(ctx, value.Error)
		}
	}
}

func (h searchBenchmarkGRPCStatsHandler) connectionTracker() *searchBenchmarkConnectionTracker {
	if h.server {
		return searchBenchmarkServerConnections
	}
	return searchBenchmarkClientConnections
}

func (searchBenchmarkGRPCStatsHandler) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return ctx
}

func (searchBenchmarkGRPCStatsHandler) HandleConn(context.Context, stats.ConnStats) {}

func isSearchBenchmarkMethod(method string) bool {
	return method == viewpb.ViewQueryService_SearchOnView_FullMethodName ||
		method == viewpb.ViewQueryService_SearchOnViewStream_FullMethodName
}
