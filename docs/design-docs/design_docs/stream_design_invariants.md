# Streaming Reduce Invariants

- **Created:** 2026-07-22
- **Status:** Active
- **Applies to:** Streaming Reduce design and implementation

## Abstraction Boundary

- `ReduceStream` owns stream flow: child `Recv()` scheduling, retained child
  input, drained state, backpressure, lifecycle, and output `CHUNK` assembly.
- `StreamReduceOperator` owns reduction semantics over input supplied by
  `ReduceStream`, including selection, ordering, deduplication, and aggregation.
- The operator consumes behavior-ready input and emits zero or more complete
  output rows, hits, or aggregate results back to `ReduceStream`.
- `ReduceStream` groups emitted output into parent-facing `CHUNK`s. The operator
  does not construct transport messages or call child streams.
- Partial aggregation state belongs to the Global Merge operator and is not
  exposed to `ReduceStream`.
- `ReduceStream` does not inspect or modify operator-owned reduction state.

The stream and operator are request-scoped and intentionally work together, but
their only semantic boundary is behavior-ready input and complete output units.

## Stream Lifecycle

Context cancellation is the stream lifecycle mechanism across RPC boundaries.
Each layer creates every child RPC context from its current request context and
retains the corresponding cancel function:

At Proxy, `parentCtx` is the Query/Search handler's `ctx`. Inside a streaming
server method, `parentCtx` is `server.Context()`.

```text
childCtx, cancel = context.WithCancel(parentCtx)
childStream = client.SearchStream(childCtx, request)

childStreams.append(childStream)
childCancels.append(cancel)

reducedStream = NewReduceStream(
    request,
    childStreams,
    childCancels,
)
defer reducedStream.Close()
```

`ReduceStream` owns its child streams, their cancel functions, and retained
`ChunkBuffer`s:

```text
ReduceStream:
    childStreams[]
    childCancels[]
    childChunkBuffers[]
```

`Close()` is idempotent and actively terminates owned child RPCs without
cancelling the parent request context:

```text
Close():
    for cancel in childCancels:
        cancel()

    CloseResources()
```

The following lifecycle rules are invariant:

- Parent cancellation automatically propagates to every `childCtx` derived
  from `parentCtx`.
- Local early termination calls `ReduceStream.Close()` to cancel child RPCs
  while `parentCtx` remains active.
- Every server method calls `defer reducedStream.Close()` immediately after
  creating its local reduced stream so all exit paths release retained buffers.
- Context cancellation unblocks gRPC child `Recv()` and server `Send()` calls;
  it does not automatically call application-owned `Close()` methods.
- Cross-process cancellation initiates termination asynchronously. Lower-level
  computation stops promptly only when it observes its cancelled context.
- `CloseSend()` is not a stream cancellation mechanism and does not replace the
  retained cancel function.

## Merge Behaviors

Every request selects exactly one of the following merge behaviors before child
data is consumed. Implementation support or rollout phase does not define an
additional merge behavior.

### Ordered Merge

```text
ReduceStream:
  retain at most one input CHUNK per active child
  call child Recv() until every active child has data or is drained

StreamReduceOperator:
  compare the retained heads
  consume the selected input unit
  emit zero or one complete ordered output unit

ReduceStream:
  repeat until one output CHUNK is full or the stream is drained
```

### Global Merge

```text
ReduceStream:
  retain at most one input CHUNK per active child
  provide any available retained input to the operator

StreamReduceOperator:
  consume available input into operator-owned global state
  emit no final output until every child is drained
  emit complete final output units during finalization

ReduceStream:
  group finalized output units into one or more output CHUNKs
```

### IgnoreOrder Merge

```text
ReduceStream:
  retain at most one input CHUNK per active child
  provide any available retained input to the operator

StreamReduceOperator:
  consume input without cross-child ordering or global state
  immediately emit zero or more complete output units

ReduceStream:
  repeat until one output CHUNK is full or the stream is drained
```

## Scenario Classification

| Merge behavior | Scenarios |
| --- | --- |
| Ordered Merge | Plain Query, Plain Search, Query `ORDER BY`, Search `ORDER BY`, rerank, and Hybrid Search |
| Global Merge | Query `GROUP BY`, Query `GROUP BY + ORDER BY`, Search `GROUP BY`, Search `GROUP BY + ORDER BY`, and Search Aggregation |
| IgnoreOrder Merge | Future paths that require neither cross-child ordering nor global state; Plain Query may move here if PK ordering is removed |

## Shared CHUNK Limits

- Each child `Recv()` returns at most the configured input `CHUNK` size.
- Each reduced-stream `Recv()` returns at most the configured output `CHUNK`
  size.
- The final reduced-result size determines the output `CHUNK` count and final
  `CHUNK` size, not the configured maximum output `CHUNK` size.
