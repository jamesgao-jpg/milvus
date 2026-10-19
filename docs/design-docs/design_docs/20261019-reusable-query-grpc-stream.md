# Reusable Query gRPC Stream Prototype

## Status and Scope

Design v0 approved by the user on 2026-10-19, including one reusable RPC per
Proxy client/QueryNode pair, exclusive use, and request-scoped fallback when
busy. The user approved full implementation and the line-budget exception.
The benchmark-only prototype is implemented and has targeted local/ARM test
coverage, but is not production-ready. End-to-end experiment execution remains
pending. Keep it disabled by default and confined to
`codex/qv-reducestream-e2e-benchmark-20260915`.

The [Streaming Reduce design](20270722-streaming_reduce_design.md) remains
authoritative for reduction. This experimental extension changes only the
transport lifetime for bounded Plain Query requests to sealed QueryNodes. It
does not authorize production non-iterator streaming or M1B iterator sessions.
Search, Group By, aggregation, custom ordering, StreamingNode transport,
multiplexing, and component-lifetime persistent sessions are out of scope.

**Reuse the gRPC RPC, not the execution state.** Each request obtains a fresh
QueryPlan, child adapters, ReduceStreams, Buffers, output, and metadata. No
cursor, result, segment task, or QueryView lease survives between requests.

## Evidence and Current Boundary

**VERIFIED source baseline:** benchmark branch commit
`3ae2819bdf`. Source links below are pinned to that baseline.

| Existing behavior | Evidence |
| --- | --- |
| The Query adapter opens an RPC using the request context; `Close()` closes its send side and cancels it. | [grpc_stream.go, lines 43-103](https://github.com/jamesgao-jpg/milvus/blob/3ae2819bdf/internal/util/queryutil/grpc_stream.go#L43-L103) |
| The QueryNode client obtains the existing service and pins calls to the selected node. | [query_view_client.go, lines 66-89](https://github.com/jamesgao-jpg/milvus/blob/3ae2819bdf/internal/querynodev2/client/handler/query_view_client.go#L66-L89) |
| A Query obtains segment tasks, materializes its result, releases the tasks, and sends split Chunks. | [server.go, lines 217-299](https://github.com/jamesgao-jpg/milvus/blob/3ae2819bdf/internal/views/viewquery/server.go#L217-L299) |
| Proxy consumes the final stream within Execute and restarts complete attempts on retryable ViewErrors. | [task_query.go, lines 977-1031](https://github.com/jamesgao-jpg/milvus/blob/3ae2819bdf/internal/proxy/task_query.go#L977-L1031) |

The historical Proxy profile reporting 0.2% under `NewStream` / `newClientStream`
used Git commit `9d3dd3019283df4cfe77a180e4433666f7faaa98`, N32, and an
8,192-row Query with vector output. Matching stacks include Unary GetQueryPlan
creation as well as Query stream creation. This is a narrow sampled-CPU
attribution, not an upper bound on complete RPC lifecycle savings. Its binary Build ID
`ab33ced09ece335eaf903ef7cf0c319016afd529` is not a Git commit. Evidence is in
`unary_streaming_grpc_benchmark/milvus_result/test_0917_01/runs/20260917T030022Z-25203/`.

The standalone `result/test_1019_02` experiment found a 1.672x median paired
persistent/Unary QPS ratio for one child and 4 KiB responses. It did not profile
CPU or implement independent logical-request completion/error handling. That
result motivates this prototype; it does not establish a Milvus benefit.

[gRPC's performance guide](https://grpc.io/docs/guides/performance/) identifies
repeated RPC initiation as work that streaming can amortize, but also cautions
about fixed stream routing and scalability. Benefit here remains unverified.

## Ownership

```text
Proxy request A                         Proxy request B
  fresh QueryPlan                         fresh QueryPlan
  fresh final ReduceStream                fresh final ReduceStream
    fresh per-vchannel ReduceStream         fresh per-vchannel ReduceStream
      fresh child adapter                   fresh child adapter
              | exclusive use, then cleanup |
              +--------------+--------------+
                             |
               reusable Query gRPC RPC
                             |
                         QueryNode
                execute A; release A resources
                execute B; release B resources
```

Only the RPC and its fixed transport context are retained. Existing channels,
connections, routing, TLS, compression, message limits, keepalive, and flow
control remain unchanged. Do not create a new channel for each cached RPC.

## Selection and Bounded Reuse

Proposed single new benchmark-only switch:
`proxy.queryView.enableReusableQueryStream=false`. The prototype implements
this key. The existing `enableReduceStream` gate takes precedence.

| Configuration and request | Selected path |
| --- | --- |
| `enableReduceStream=false` | Existing Batch path |
| ReduceStream enabled; reuse disabled | Existing request-scoped streaming |
| Both enabled; bounded Plain Query on QN | Reusable RPC if available |
| Unsupported request or StreamingNode child | Existing path; no reusable RPC |

Keep selection inside the existing QueryNode client transport implementation;
the Proxy/legacy caller still obtains a request-scoped `queryutil.ReduceStream`.
Do not add pooling operations to the caller-facing interface.

For the first prototype, retain at most one reusable RPC, idle or leased, per
QueryNode ID in the existing QueryNode client. A concurrent request finding it
leased uses the existing request-scoped RPC immediately: no waiting queue,
multiplexing, or unbounded pool growth. Record this fallback explicitly.

An idle RPC expires after a proposed fixed 60 seconds. Node removal, client
shutdown, or any detected transport failure closes and removes it. Client
shutdown must cancel leased and idle RPCs before closing the underlying service.
Cache entries must not retain unregistered node IDs indefinitely.

Scope reuse by node identity, method, and fixed routing/security metadata. Never
reuse across a different credential or routing scope. Unhandled per-request
metadata must fall back to request-scoped transport rather than be dropped.
StreamingNode's channel/term routing is deliberately excluded from this draft.

## Wire Contract

Add a separate internal bidi method, proposed name `QueryOnViewStreamReusable`.
Do not reinterpret or change the existing Unary or request-scoped RPC methods.
The following describes required fields; it is not a generated-protobuf patch.

| Direction | Required contents |
| --- | --- |
| Request | Monotonically increasing request ID within this RPC; complete existing `QueryOnViewRequest`; remaining timeout; per-request trace/log context |
| Response | Matching request ID; optional existing `RetrieveResults` Chunk; optional completion status |
| Completion | Explicit success or error using `google.rpc.Status`, including existing typed `ViewError` details |

The implemented `completion_status` field packs `google.rpc.Status` into
`google.protobuf.Any`. Failed legacy result statuses retain their existing
`commonpb.Status` as a status detail; the adapter reconstructs the original
Milvus error rather than discarding its code. No existing RPC wire contract
or error classification is changed.

One RPC carries at most one active logical request. The next request is sent
only after successful completion and cleanup of the previous one. Each request
uses a fresh ID, including retries; the existing Proxy request ID remains a
separate diagnostic identifier and is not sufficient to distinguish attempts.

Every logical request must end with exactly one completion status. Successful
completion accompanies the final Chunk when there is one; empty success can be
a completion-only response. Error completion must not contain result rows.
The server releases request execution resources before sending completion.
The transport-level RPC remains open after successful logical completion.

A native RPC may close while idle without that closure being observed until
the next Send/Recv. Do not add a background reader or treat native EOF as that
next request's successful completion. The affected exchange fails, the RPC is
evicted, and a later independent request may open another RPC. There is no
hidden replay. Idle expiry still bounds unobserved stale entries.

Result metadata remains request-scoped in the existing `RetrieveResults`
representation, with existing aggregation rules and no duplication. Normally it
travels in the existing Chunks. If empty output requires metadata, completion
may accompany a metadata-only `RetrieveResults`; the adapter delivers that
result once before logical EOF. Do not defer required metadata to RPC trailers,
which are unavailable at each logical request's boundary.

The existing reducible-payload byte threshold and complete-Unit behavior are
unchanged. Request IDs, completion status, and result metadata are not reducible
payload. Do not split Units or alter Chunk sizing to improve measured reuse.

Malformed IDs, missing status, invalid response combinations, or data after
completion invalidate the RPC. Unexpected gRPC EOF before completion is an
error, not successful empty or partial output. This requirement applies to the
new protocol only; correcting legacy EOF semantics is a separate change.

## Recv and Close

The new adapter implements the existing `Recv()`, `Close()`, and stub
`Interrupt()` contract. It does not retry internally.

| Event | Adapter result | RPC disposition |
| --- | --- | --- |
| Chunk without completion | Return Chunk | Remains exclusively leased |
| Final Chunk with OK completion | Record completion; return Chunk | Remains leased until Close |
| OK completion without Chunk | Return logical EOF | Remains leased until Close |
| Recv after successful completion | Return logical EOF without another gRPC read | Remains leased until Close |
| Close after verified success | Release request resources | Return healthy RPC to idle cache |
| Close before verified completion | Abort immediately; do not drain unused Chunks | Cancel and evict RPC |
| Logical error or transport error | Return original classified error | Cancel and evict RPC |
| Cancellation, deadline, protocol error | Return corresponding error | Cancel and evict RPC |

Close remains idempotent. Recording completion does not return the RPC to the
cache by itself. Before reuse, remove and join request cancellation callbacks,
finish any in-flight Send/Recv, and release the adapter's request state. A late
cancel or deferred Close from request A must never affect request B.

This preserves early termination: reaching the parent's limit must not cause
background reads that transfer the remaining child result just to recover the
RPC. Some workloads may consequently evict most RPCs. Report that outcome
rather than extending request latency or changing reduction semantics.

The server clears request context, materialized result, Chunks, metadata, and
task references before accepting another request. Only idle transport state is
retained; do not cache QueryPlans or segment resources across requests.

## Context and Errors

Native gRPC metadata, status, cancellation, and deadlines belong to the entire
RPC, not each message exchange; see [RPC lifecycle](https://grpc.io/docs/what-is-grpc/core-concepts/#rpc-life-cycle).
Create the persistent RPC from the QueryNode client's lifecycle context, pinned
to the selected node, not from the first Query's deadline or trace context.

The Proxy adapter enforces the original request cancellation/deadline by
canceling and evicting its leased RPC. The server derives a fresh request
context from the transport context plus the sent remaining timeout and trace
context, then cancels it at completion. Do not copy arbitrary context values or
change the internal authentication model. Audit required context/metadata
consumers before implementing this boundary; unknown dependencies block reuse.

Serialize the existing status and typed details, reconstruct the error on the
client, and use the existing
[ViewError classification](https://github.com/jamesgao-jpg/milvus/blob/3ae2819bdf/internal/views/viewerror/error.go#L55-L58).
Do not replace errors with strings or introduce new retry codes.

| Observed failure | QueryView application behavior |
| --- | --- |
| `VIEW_INVALIDATED`, `VIEW_NOT_FOUND`, `ON_SHUTDOWN`, `NOT_PRIMARY` | Evict; let existing Execute restart the entire attempt within its attempt/deadline budget, even after internal final Chunks |
| Raw `UNAVAILABLE` without typed ViewError | Evict; no new QueryView re-plan condition |
| Cancel, deadline, or other non-retryable error | Evict; fail without a new application retry |

All rows, metadata, plans, Buffers, and reduction state from a failed attempt
are discarded. Never resume or replay inside the adapter. Retry classification
uses the original error before cleanup errors are combined.

**Transport limitation:** gRPC automatic retry applies to the RPC, not to each
request carried inside an established RPC. After response headers commit that
RPC, it is not automatically replayed; see the [retry guide](https://grpc.io/docs/guides/retry/).
Do not claim native transport retry parity with independent Unary RPCs or add
hidden replay to compensate. A failed logical request follows the application
policy above; a later independent request can open a fresh RPC.

## Observability and Review Gates

Keep native RPC lifetime metrics separate from logical Query metrics. A
persistent RPC may be active while no Query is active. Attribute phase timing,
result bytes, metadata, and status to the current logical request; do not retain
the first request's benchmark object in the transport context.

Record RPC opens, reuse hits, busy fallbacks, eviction reasons, logical
completions, and live idle/leased counts. Request/RPC IDs belong in retained
diagnostic records, not high-cardinality Prometheus labels. Connection bytes
remain connection-scoped; do not present them as exact per-request bytes.

Validation is defined separately in
`unary_streaming_grpc_benchmark/milvus_result/test_1019_01/TEST_PLAN.md`.
Correctness, cleanup, error propagation, and unchanged baseline paths gate
performance testing. Compare Batch, request-scoped streaming, and reusable
streaming in one optimized binary; profile Proxy and QN separately from timed
runs and normalize CPU and allocations per completed Query.

Approved design decisions include the new wire method and terminal-status
contract, single-entry-per-QN/busy-fallback policy, fixed idle expiry, and
disabled-by-default selection. A larger pool, multiplexing, and SN reuse remain
outside this version. The context/metadata audit is an implementation gate.
Implementation verification is recorded in
`unary_streaming_grpc_benchmark/milvus_result/test_1019_01/implementation_verification.md`.
The experiment runner, deployed E2E qualification, and performance results remain
pending. The feature-only `qv` branch is unchanged.
