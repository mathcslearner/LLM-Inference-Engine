# Continuous batching scheduler & runtime

**Milestone:** M9 (design doc: M9-T01; implementation: M9-T02 … M9-T10)
**Governs:** the `src/runtime/` module (the asynchronous engine — request/
sequence lifecycle, submission queue, per-request output channels, the single
engine-loop thread), the `src/scheduler/` module (pure per-step decision logic —
admission, token budget, decode-priority, preemption victim selection), the
ragged/batched extension of the M5 `ForwardRequest`/`Model` contract and the
`src/engine/batch.h` batch-assembly pass, and the two batched attention-kernel
entries (`PrefillAttentionVarlenF32`, `PagedDecodeAttentionBatchedF32`) M9 adds
to `src/kernels/`.
**Cites:** ADR-002 (module boundaries — this doc adds no new edge; it exercises
`runtime → {scheduler, engine, model, kvcache, sampling, tokenizer}` and keeps
`scheduler` a `core`-only leaf, §2), ADR-003 (error handling — per-request
recoverable `Status`, `CHECK` only for engine-internal invariants),
`docs/design/model-execution.md` (the `Model::forward`/`ForwardRequest` contract
§5, the `KvCache` interface §6, the KV correctness invariant §6.2, and §5.4's
"what M9 adds" reservation this doc discharges), `docs/design/optimized-cpu-
execution.md` (the `OptimizedModel` graph and workspace §6, and §11's M9
batching note), `docs/design/paged-kv-cache.md` (the block pool, block table,
`PagedKvCache`, the `free_blocks()`/`blocks_needed()` admission accessors §6.2,
the batched block-table shape §9.4, the resumable-exhaustion seam §10.2, and the
preemption interaction §11), `docs/design/cpu-backend.md` (the engine-loop thread
is an orchestration thread, never a compute thread §3.4), and
`src/engine/generator.{h,cpp}` (the single-sequence `Generate` whose stopping,
sampling, and resumable-error posture the batched loop generalizes).

This is the working contract for the continuous-batching runtime — the
architectural heart of the engine, so this doc carries more weight than any
other. Implementation tickets M9-T02 … M9-T10 must conform to it; if
implementation reveals a design flaw, this doc is updated in the same change with
a note on what changed and why (`docs/design/README.md`). The serving layer gets
its own doc (`docs/design/server.md`, M10-T01) that consumes the `runtime` public
API reserved here (§5); prefix caching (M11) extends the admission step (§6.5);
chunked prefill and flash-decoding (M12) extend the scheduler and batch shapes on
seams flagged here (§11).

---

## 1. Scope & non-goals

M5–M8 built a **single-request** engine: `engine::Generate(model, cache,
prompt_ids, options, on_token)` runs one sequence to completion — prefill in one
`kLast` forward, then one token per decode forward — against one private
`PagedKvCache` over a `BlockPool` the driver owns. Correct, but it serves one
request at a time: a second request waits for the first to finish, and a pool
sized for many sequences sits mostly idle.

M9 turns the engine into a **multi-request, continuously-batched system**:

- clients `submit` requests from any thread into a queue and receive a handle;
- a **single engine thread** runs a `step()` loop: every step it asks the
  **scheduler** which waiting sequences to prefill and which running sequences to
  decode (under a token budget and the block pool's free-block count), assembles
  the batch, runs the forward(s), samples one token per sequence, appends, checks
  stop conditions, and streams results back to per-request channels;
- when the pool runs dry mid-decode, the scheduler **preempts** the latest-arrived
  running sequence — frees its blocks and requeues it — resuming it later by
  re-prefilling prompt+generated (the M8-T08 resumable-error seam, §10.2 of the
  paged doc, made a first-class scheduler action);
- **cancellation** frees a request's resources and closes its channel promptly;
- a **per-request failure** (a sampler edge case, a bad token id) fails only that
  request — never the engine loop.

The load-bearing correctness property, and M9-T08's acceptance criterion, is the
**continuous-batching invariant**: N concurrent greedy requests produce output
**identical** to running each sequentially. This doc's batch-composition and
batch-invariance decisions (§7, §8) exist to make that invariant hold bit-for-bit,
not merely to Class-T tolerance.

**In scope (this doc):**

- the request/sequence state machine and its invariants (§3),
- the per-request output channel (§4),
- the runtime's public async API and threading model (§5),
- scheduling policy v1 — FCFS admission, token budget, decode-priority,
  block-availability, latest-arrived preemption (§6),
- batch composition (separate prefill/decode passes vs mixed — chosen and
  justified, §7),
- the ragged `ForwardRequest` extension, batch assembly, batched kernels, and the
  per-sequence **batch-invariance** guarantee (§8),
- the step loop pseudocode and the full invariant list (§9),
- preemption & recomputation mechanics (§10),
- cancellation & per-request failure isolation (§11),
- how M10/M11/M12/M15/M16 attach to these seams (§12),
- the per-ticket testing strategy (§13).

**Non-goals (of this doc, not the project) — each named with its milestone:**

- **No HTTP, SSE, chat templates, or backpressure policy.** The runtime exposes a
  handle with streaming/blocking token consumption; the socket-to-channel wiring,
  disconnect→cancel, and the bounded-queue backpressure policy are M10
  (`docs/design/server.md`). The channel here is **unbounded** (§4); a bounded
  variant is M10's call.
- **No prefix reuse at admission.** The scheduler admits a waiting sequence by
  prefilling its whole prompt. Walking the prompt's hash chain to adopt cached
  blocks and prefill only the suffix is M11-T04; §6.5 reserves the hook.
- **No chunked prefill.** A prompt is prefilled in one pass; splitting a long
  prompt into fixed-size chunks scheduled alongside decodes (so one long prompt
  cannot stall inter-token latency) is M12-T06 "scheduler v2", when
  `--max-num-batched-tokens` becomes the binding per-step budget. §6.4 records the
  v1 posture (reject a prompt longer than the batch-token budget at `submit`)
  that M12-T06 relaxes.
- **No flash-decoding / cache-split.** The batched decode kernel threads across
  (sequence, kv head) exactly as M8-T05 threads across kv heads; splitting one
  long context across threads is M12-T03.
- **No variable-tokens-per-step.** Every sequence advances exactly one token per
  decode step. Speculative decoding's accept-K-tokens step is M15-T06; §12 notes
  the `SchedulerOutput`/step-loop generalization it needs.
- **No swap-to-host preemption.** Preemption is evict-and-recompute only; on a
  CPU engine host RAM *is* the block pool, so there is no faster tier to swap KV
  into (§10.2). Recorded, not built.
- **No metrics endpoint.** The scheduler/pool already expose the counters (queue
  depth via the runtime, `BlockPoolStats`, preemption count) M16's `/metrics`
  reads; wiring them to Prometheus is M16.
- **`n = 1` per request.** One sequence per request (no beam search, no
  `best_of`); M10-T03 fixes `n = 1` for the API too. `Sequence` is nonetheless
  the scheduler-visible unit so a future `n > 1` is additive.

---

## 2. Module layout & layering

New files, under the two modules M9 introduces (`runtime`, `scheduler`) plus
additive files in `engine`/`kernels`:

| File | Module | Ticket | Responsibility |
|---|---|---|---|
| `src/runtime/request.h` | runtime | M9-T02 | `Request` (id, prompt, `SamplingParams`, arrival), `Sequence` (state, ids, cache handle, per-seq sampler/stop/detok state), the per-request output channel |
| `src/runtime/channel.{h,cpp}` | runtime | M9-T02 | Thread-safe per-request `OutputChannel` (ordered delivery, blocking + polling, close-on-finish/cancel) |
| `src/runtime/engine.{h,cpp}` | runtime | M9-T03/T08 | Public async API (`submit`/`cancel`/handle), submission queue, `Engine::step()`, the loop thread |
| `src/scheduler/scheduler.{h,cpp}` | scheduler | M9-T04 | Pure `schedule(inputs) → SchedulerOutput` decision logic |
| `src/engine/batch.{h,cpp}` | engine | M9-T05 | Flatten a `SchedulerOutput` into staged batch inputs (cu_seqlens, positions, per-seq caches, sampling metadata) |
| `src/kernels/attention.{h,cpp}` (+ per-ISA TUs) | kernels | M9-T06 | `PrefillAttentionVarlenF32` — ragged prefill over cu_seqlens |
| `src/kernels/paged_attention.{h,cpp}` (+ per-ISA TUs) | kernels | M9-T07 | `PagedDecodeAttentionBatchedF32` — batched decode over a `[B, max_blocks]` block-table tensor |

**Layering (ADR-002).** No new edge; M9 exercises edges already on the diagram
and keeps the scheduler a leaf:

- **`runtime` (layer 4, orchestration)** links `scheduler`, `engine`, `model`,
  `kvcache`, `sampling`, `tokenizer`, `memory`, `core` — every one already below
  it on the diagram (`server → runtime → scheduler`; `runtime → engine …`). The
  runtime is where the model, the pool, every per-sequence cache, and the
  scheduler live; it is the *only* thread that mutates block state, so admission
  arithmetic is exact (§5.3).
- **`scheduler` (layer 3) stays pure decision logic and links only `core`**
  (ADR-002 rule 4). This is the single most important layout decision in M9. The
  scheduler must *not* see `Sequence` (which lives in `runtime`, above it) or
  `BlockPool`/`kvcache` types: it consumes **plain descriptor structs** the
  runtime fills each step — a `PoolSnapshot{free_blocks, block_size, total_blocks}`
  and per-sequence `{id, arrival_index, num_computed_tokens, num_prompt_tokens,
  blocks_held}` — and returns a plain `SchedulerOutput` of ids and lengths
  (§6.2). ADR-002 would *permit* `scheduler → kvcache` (a downward edge), but
  nothing in the pure block-arithmetic needs a `kvcache` type — `blocks_needed`
  is `⌈(cur+add)/bs⌉ − ⌈cur/bs⌉`, reproduced in the scheduler from `block_size`
  — so keeping it `core`-only makes the whole scheduler table-testable with no
  pool, no allocator, no model (ADR-002 "scheduler and sampling logic stay
  testable on CPU-only CI forever"). `engine` and `scheduler` never depend on each
  other; `runtime` mediates (ADR-002 layer-3 note).
- **`engine` (layer 3)** gains `batch.h`; it already links `model`, `kvcache`,
  `sampling`, `tokenizer` (the M7-T04 edge). Batch assembly touches tensors and
  the `ForwardRequest` struct, so it belongs in `engine`, beside `Generate`, not
  in the tensor-free `scheduler`.
- **`kernels` (layer 1)** gains the two batched entries as additive functions in
  the existing `attention.{h,cpp}`/`paged_attention.{h,cpp}`; the `kvcache →
  kernels` and `model → kernels` edges that reach them are unchanged.

**Namespace note.** The single-sequence driver lives in namespace
`engine::engine` (`src/engine/`); the runtime is `engine::runtime`. The public
class is `engine::runtime::Engine` — distinct from the `engine::engine` namespace,
no clash (they differ by namespace vs type). Batch assembly is
`engine::engine::BatchInputs`/`AssembleBatch` (it is an `engine`-module type).

---

## 3. Request & sequence lifecycle (M9-T02)

### 3.1 `Request` and `Sequence`

A **`Request`** is the immutable client-supplied description; a **`Sequence`** is
the engine's mutable per-request execution state. One `Sequence` per `Request`
(`n = 1`), but the two are separate types so the API surface (`Request`) stays
free of engine internals and a future `n > 1` adds sequences without changing the
request.

```cpp
namespace engine::runtime {

using RequestId = std::uint64_t;  // engine-assigned, monotonic

struct Request {
  RequestId id = 0;
  std::vector<std::int32_t> prompt_ids;     // non-empty (validated at submit)
  sampling::SamplingParams params;          // validated at submit
  std::vector<std::int32_t> eos_ids;        // model-derived stop ids
  const tokenizer::Tokenizer* tokenizer = nullptr;  // for text/stop-strings
  bool skip_special_tokens = true;
  std::uint64_t arrival_index = 0;          // FCFS order key (submit order)
};

enum class SeqState { kWaiting, kRunning, kPreempted, kFinished,
                      kCancelled, kFailed };

class Sequence {
 public:
  // ... state accessor, token accessors, block-table handle, progress ...
 private:
  SeqState state_ = SeqState::kWaiting;
  const Request* request_;                  // borrowed; owned by the engine map
  std::vector<std::int32_t> generated_ids_;
  std::unique_ptr<kvcache::PagedKvCache> cache_;  // this sequence's KV
  sampling::Sampler sampler_;               // built from params (resolved seed)
  StopChecker stop_;                        // engine/stop.h, per-request
  std::int64_t num_computed_tokens_ = 0;    // positions committed to the cache
  FinishReason finish_reason_ = FinishReason::kLength;
  StopTrigger stop_trigger_ = StopTrigger::kNone;
  core::Status error_;                      // set on kFailed
};

}  // namespace engine::runtime
```

Rationale for the per-sequence members:

- **`cache_` is a `unique_ptr<PagedKvCache>` the sequence owns**, borrowing the
  engine's single shared `BlockPool` (paged doc §8.1: one pool, one cache per
  sequence). Dropping the sequence drops the cache, which returns every block
  (RAII, paged §7.3) — the basis for "no block leaks" on every finish/cancel/fail
  path.
- **`sampler_`, `stop_`, and a `DetokenizerStream` (inside `stop_`) are
  per-sequence and stateful** — they are exactly the objects the single-sequence
  `Generate` builds once and carries across the decode loop, lifted onto the
  sequence so the step loop is a fan-out of that loop. Building the `Sampler`
  resolves the seed once (M7-T02), so a request with no seed samples a fixed
  stochastic trajectory across steps regardless of batching. The batched sampler
  (§8.4) borrows each sequence's `Sampler` via `BatchRow`.
- **`num_computed_tokens_`** is the position count already committed to the cache
  (`== cache_->length()`). Prefill sets it to `prompt_len`; each decode step
  increments it by 1; preemption resets it to 0 (the cache is dropped) and resume
  recomputes prompt+generated. The scheduler reads it (as a descriptor field) to
  compute the next step's block need.

### 3.2 State machine

Every legal transition (acceptance criterion — the diagram):

```
                    submit
                      │
                      ▼
                 ┌─────────┐   admit (prefill)   ┌─────────┐
        ┌───────▶│ WAITING │────────────────────▶│ RUNNING │──────┐
        │        └─────────┘                     └─────────┘      │
        │             │                            │  │  ▲        │ stop / eos /
        │             │ cancel                     │  │  │ decode │ max_tokens
  requeue│            ▼                     preempt │  │  │(1 tok) │
  (head) │        ┌───────────┐  (evict + free      │  │  └────────┘
        │        │ CANCELLED │◀──────── blocks) ────┘  │        │
        │        └───────────┘   cancel         cancel │        ▼
        │             ▲     ▲                          │   ┌──────────┐
        │  ┌──────────┘     └───────────┐              │   │ FINISHED │
        └──│ PREEMPTED │                │              │   └──────────┘
           └───────────┘                │              │
                 │ cancel               │ per-request  ▼
                 └──────────────────────┘   fault   ┌────────┐
                                                    │ FAILED │
                                                    └────────┘
```

Transitions, exhaustively:

| From | To | Trigger |
|---|---|---|
| (none) | WAITING | `submit` accepted |
| WAITING | RUNNING | scheduler admits → prefill this step |
| WAITING | CANCELLED | `cancel` before admission |
| RUNNING | RUNNING | scheduler schedules a decode step (self-loop) |
| RUNNING | FINISHED | stop condition (eos / stop id / stop string / max_tokens) |
| RUNNING | PREEMPTED | scheduler preempts (pool exhausted, this is the victim) |
| RUNNING | CANCELLED | `cancel` during prefill/decode |
| RUNNING | FAILED | per-request fault (sampler/forward error scoped to this seq) |
| PREEMPTED | RUNNING | scheduler re-admits → re-prefill prompt+generated |
| PREEMPTED | CANCELLED | `cancel` while waiting to resume |

`FINISHED`, `CANCELLED`, `FAILED` are **terminal** — no transition leaves them.
Any transition not in this table is a programmer error → **`CHECK`** (M9-T02
acceptance: "illegal transition = CHECK failure"). The transition function is a
single `Sequence::Transition(SeqState next)` that `CHECK`s the (from, to) pair
against this table, so the invariant is enforced in one place, not scattered
across the loop.

**Note — PREEMPTED requeues at the head, not the tail.** A preempted sequence has
already done work (its prompt was prefilled, some tokens generated); putting it
back at the *head* of the waiting queue (ahead of never-started requests) bounds
the re-prefill cost and guarantees liveness (§10.3). This is the one deviation
from strict FCFS, and it is deliberate.

### 3.3 Finish info

On reaching a terminal state, the sequence's channel is closed with a
`FinishInfo`:

```cpp
struct FinishInfo {
  SeqState terminal;                        // kFinished / kCancelled / kFailed
  FinishReason finish_reason;               // engine/stop.h (kStop / kLength)
  StopTrigger stop_trigger;                 // finer: eos id / stop id / string / len
  std::string matched_stop;                 // the stop string, if any
  core::Status error;                       // ok unless kFailed
};
```

`FinishReason`/`StopTrigger` are reused verbatim from `src/engine/stop.h` (M7-T04)
— the same finish taxonomy the single-sequence `Generate` returns, so M10's
`finish_reason` mapping is backend-agnostic. `kCancelled` and `kFailed` are the
two runtime-only additions the batched loop needs and `Generate` did not.

---

## 4. Output channel (M9-T02)

Each sequence owns one `OutputChannel` — a thread-safe queue the engine thread
writes and one client thread reads:

```cpp
struct OutputItem {
  std::int32_t token_id;
  std::string text_delta;                   // safe-to-emit bytes (may be empty)
  std::optional<sampling::StepLogprobs> logprobs;  // when params.logprobs > 0
};

class OutputChannel {
 public:
  // Producer (engine thread): push one produced token, then close once.
  void Push(OutputItem item);               // ordered, single producer
  void Close(FinishInfo info);              // idempotent-guarded; closes once

  // Consumer (client thread):
  std::optional<OutputItem> Next();         // blocks until an item or close
  std::optional<OutputItem> TryNext();      // non-blocking poll
  std::optional<FinishInfo> finish() const; // set once closed
};
```

Design points:

- **Single producer, single consumer, mutex + condvar.** Only the engine thread
  pushes; only the handle's owner consumes. `Push`/`Close`/`Next` share one mutex;
  `Next` waits on a condvar the producer notifies. This is deliberately not
  lock-free: the forward pass dominates a step by orders of magnitude, so the
  queue lock is never contended on the hot path (a lock-free MPSC is a measured
  follow-up if a profile ever shows it, §12). `TryNext` gives M10's SSE loop a
  non-blocking poll for disconnect interleaving.
- **`OutputItem` mirrors the single-sequence `TokenEvent`** (`generator.h`): id +
  text delta + optional logprobs. The concatenation of every item's `text_delta`
  equals the sequence's full detokenized text (the streaming invariant M7-T04
  established and M10 relies on). The delta is produced by the sequence's
  `DetokenizerStream`/`StopChecker`, so held-back bytes (incomplete UTF-8 or a
  pending stop-string prefix) are released exactly as in `Generate`.
- **Close-once, and close carries the reason.** `Close` is guarded so a finish
  followed by a late cancel does not double-close; `finish()` returns the
  `FinishInfo` so a polling consumer sees why the stream ended without a sentinel
  item. After close, `Next` drains any buffered items then returns `nullopt`.
- **Unbounded.** v1 never blocks the producer (the engine thread must never block
  on a slow client — that would stall every other sequence in the batch). A
  bounded channel with a drop/backpressure policy is M10's decision, made where
  the socket write rate is visible.
- **Ownership / close semantics on finish/cancel** (M9-T02 acceptance): the
  channel outlives the `Sequence` (the handle holds a `shared_ptr<OutputChannel>`;
  the engine retires the `Sequence` as soon as it closes the channel, but the
  client can still drain buffered tokens). Delivery is in production order across
  threads — tested with a producer thread pushing a known sequence and a consumer
  asserting order and the terminal `FinishInfo`.

> **As built (M9-T02).** `src/runtime/request.h` (`Request`/`Sequence`,
> `SeqState`, `IsLegalTransition`, `FinishInfo`) + `src/runtime/sequence.cpp`,
> and `src/runtime/channel.{h,cpp}` (`OutputItem`/`OutputChannel`). Deviations
> from the sketch above, all deliberate:
> - **`Sequence::Create(const Request&, BlockPool*) → StatusOr<Sequence>`** is a
>   factory, not a public constructor: the `Sampler` and `StopChecker` come from
>   `StatusOr` factories, so construction is fallible and front-loads their
>   validation (bad params, or `stop_strings` without a tokenizer → the
>   `InvalidArgument` surfaces at `Create`, cache untouched). `Sequence` is
>   move-only with an out-of-line destructor/move-ctor (`OutputChannel` is
>   forward-declared in the header; its complete type lives in `sequence.cpp`).
> - **`IsLegalTransition(from, to)` is a public `constexpr` predicate** so the
>   test can enumerate the whole 6×6 matrix; `Sequence::Transition` is the one
>   caller that `CHECK`s it. The cache is created eagerly at `Create` (an empty
>   `PagedKvCache` holds zero blocks, so a `WAITING` sequence still owns nothing);
>   `ReleaseCache()`/`EnsureCache(pool)` make the preemption drop/rebuild explicit
>   for M9-T09.
> - **`OutputChannel::Close` returns `bool`** (the first close wins, later closes
>   are no-ops returning `false`), so the loop can tell whether it won the close
>   under a finish + late-cancel race. `Push` after close is a `CHECK` (the single
>   producer never pushes on a closed channel — the loop checks `closed()` first).
> - **Namespace footgun recorded:** `FinishReason`/`StopTrigger`/`StopChecker`
>   live in the sibling `engine::engine` namespace; from `engine::runtime` a
>   leading-`::` using-declaration (`using ::engine::engine::FinishReason;`) is
>   required — a bare `engine::engine::X` misresolves (§2's note). Tests deref
>   optionals via an explicit-guard `Require` helper, not `ASSERT_TRUE` + `->`,
>   to satisfy `bugprone-unchecked-optional-access`.

---

## 5. Public API & threading model (M9-T03)

### 5.1 The API

```cpp
namespace engine::runtime {

class RequestHandle {
 public:
  RequestId id() const;
  std::optional<OutputChannel::Item> next_token();   // blocking; nullopt at end
  FinishInfo await_completion();                      // blocks until terminal
  // (non-blocking poll via the underlying channel's TryNext)
 private:
  RequestId id_;
  std::shared_ptr<OutputChannel> channel_;
};

class Engine {
 public:
  static core::StatusOr<Engine> Create(model::Model& model, BlockPool& pool,
                                       EngineConfig config);

  // Client-thread API (any thread):
  core::StatusOr<RequestHandle> Submit(Request request);  // validates, enqueues
  void Cancel(RequestId id);                               // idempotent

  // Loop control:
  void Start();   // spawns the engine thread, which drives Step() to quiescence
  void Stop();    // signals shutdown, joins the thread, closes open channels
  bool Step();    // one scheduling+execution step; returns false when idle-empty
                  //   (exposed for deterministic single-thread tests)
};

}  // namespace engine::runtime
```

- **`Submit` front-loads validation** exactly as `Generate` does, returning the
  same status codes so M10 maps them to HTTP once: `ValidateSamplingParams`
  (M7-T01) → `InvalidArgument`; empty prompt → `InvalidArgument`; `stop_strings`
  without a tokenizer → `InvalidArgument`; `prompt_len > max_num_batched_tokens`
  → `InvalidArgument` (the v1 no-chunked-prefill limit, §6.4); `prompt_len +
  max_tokens > max_model_len` or `> pool token capacity` → `ResourceExhausted`.
  A rejected submit allocates no sequence, no cache, no blocks, and returns before
  the request enters the queue. On success it assigns the `RequestId` and
  `arrival_index`, constructs the `Sequence` (WAITING; the cache is created lazily
  at admission so a long queue holds no blocks), enqueues, notifies the loop, and
  returns a handle.
- **`Step()` is exposed** so tests drive the loop deterministically one step at a
  time (M9-T03/T08 mock-model tests) without racing the loop thread; `Start()`
  wraps it in `while (running) Step()` with a condvar idle-wait.

### 5.2 Threading model

```
   client threads                 engine thread (single)          client threads
  ┌───────────┐  Submit   ┌──────────────────────────────┐  Next  ┌───────────┐
  │ submitter │──────────▶│  submission queue (mtx+cv)    │        │ consumer  │
  │ submitter │──────────▶│         │                     │        │ consumer  │
  └───────────┘  Cancel   │         ▼                     │        └───────────┘
                          │   Engine::step():             │            ▲
                          │     drain submits/cancels     │            │
                          │     scheduler.schedule()      │  Push /    │
                          │     assemble → forward → sample──── Close ──┘
                          │     stop-check → deliver      │  (per-request
                          │     retire finished           │   OutputChannel)
                          └──────────────────────────────┘
```

- **Client threads → submission queue → single engine thread → per-request
  channels.** Submissions and cancels land in one mutex+condvar queue (a
  `std::deque<Command>` where a command is submit-a-request or cancel-an-id). The
  engine thread drains the queue at the top of each step, so admission and
  cancellation take effect on step boundaries — deterministic, and the ≤ one-step
  cancel latency M10-T04 needs. "lock-free-ish" (the roadmap's phrase): the queue
  is a plain locked deque because the lock is held only for the O(commands) drain,
  never across a forward; a genuinely lock-free MPSC is deferred as a measured
  optimization (there is no evidence the drain lock costs anything against a
  multi-millisecond forward).
- **The engine thread owns everything mutable**: the `Model&`, the single
  `BlockPool&`, every `Sequence` and its `PagedKvCache`, the `Scheduler`, the
  reused `BatchInputs` staging buffers, and one `BatchedSampler`. It is the sole
  mutator of block state, sequence state, and caches — so the scheduler's
  free-block arithmetic is exact (no other thread can allocate a block between the
  admission decision and the append), and no sequence-state mutation needs a lock.
- **The engine thread is an orchestration thread, never a compute thread**
  (cpu-backend.md §3.4). It *calls* the model's forward, which enters
  `parallel_for` regions on the `parallel` thread pool; the engine thread itself
  does no SIMD. So the pool's worker count is unaffected by the engine thread, and
  there is no nested-parallelism risk (the engine thread is the caller, not a
  worker).
- **Shutdown.** `Stop()` sets a flag, wakes the loop, and joins. Sequences still
  in flight have their channels closed with `kCancelled` (a drain-to-completion
  "graceful" shutdown is M10's `serve` concern; the runtime's `Stop` is abort-
  and-close so tests and the CLI tear down cleanly). No blocks leak: every live
  `Sequence` is destroyed on `Stop`, returning its blocks.

### 5.3 Concurrency invariants

- Exactly one thread (the engine thread) mutates block state, cache contents,
  and sequence state. Client threads touch only the submission queue (under its
  lock) and their own channel (under the channel lock).
- No client operation blocks the engine thread: `Submit`/`Cancel` only enqueue;
  the channel is unbounded so `Push` never waits.
- The engine thread never blocks on a client: it pushes to unbounded channels and
  never reads them.
- A stress test (M9-T03 acceptance): many submitter threads, random cancels,
  concurrent consumers — no deadlock, no lost/duplicated tokens, every channel
  eventually closes.

> **As built (M9-T03).** `src/runtime/engine.{h,cpp}` (`EngineConfig`,
> `RequestHandle`, `Engine`). Deviations from the §5 sketch, all deliberate:
> - **`Engine::Create` returns `StatusOr<std::unique_ptr<Engine>>`**, not
>   `StatusOr<Engine>`: the engine owns a `std::thread`, a mutex/condvar, and a
>   map of heap-pinned `Entry`s whose `Sequence::request_` borrows into the map,
>   so the object must be address-stable — it is non-movable and non-copyable.
>   `Create` also rejects a pool whose geometry does not match the model's
>   `cache_geometry()` (every forward would otherwise fail).
> - **`Submit` builds the `Sequence` on the client thread** (before assigning the
>   id): `Sequence::Create` is where the sampler/stop validation lives and the
>   client must receive that status synchronously. Constructing the empty
>   `PagedKvCache` touches no pool block state, so the single-mutator invariant
>   (§5.3) holds. The id and `arrival_index` are assigned under the queue lock at
>   enqueue time, so arrival order matches queue order. Peak-length rejection uses
>   `prompt_len + max_tokens - 1` (matching `Generate`) against both
>   `max_model_len` and the pool's token capacity.
> - **`Step()` ships the full §9.1 loop shape with two labelled placeholders** the
>   later tickets substitute *in place*: (a) admission is a plain FCFS walk under
>   `max_num_seqs` / the token budget / `free_blocks` (M9-T04 swaps in
>   `scheduler::Scheduler`; no preemption yet), and (b) execution runs one
>   `model::forward` **per sequence** — the body of `engine::Generate`'s decode
>   loop, lifted onto the `Sequence` — so a request's output is bit-identical to a
>   standalone `Generate` (a T03 test asserts exactly that). The batched passes
>   (M9-T05…T08) replace the per-sequence execution, preserving that output. The
>   decode set is snapshotted **before** admission, so a freshly admitted sequence
>   produces exactly one token this step (its prefill) and decodes from the next.
> - **Cancellation is handled entirely up front** (`RetireCancelled`, step 2
>   extended to RUNNING as well as WAITING/PREEMPTED): commands drain only at a
>   step boundary, so there is no in-flight batch when a cancel is applied, and a
>   cancel of a sequence in any live state takes effect immediately. This holds
>   for the batched loop too — M9-T08 **kept** RUNNING-cancel at the top of the
>   step (not moved to the end as first sketched here): a cancel that lands
>   *during* a batched forward simply waits in the queue and is applied at the
>   next boundary (≤ one step, §11.1; §9.4 as-built).
> - **`Stop` is abort-and-close and idempotent**; `~Engine` calls it. It joins the
>   thread, then closes every still-open channel `kCancelled` (including undrained
>   submissions) and drops every `Sequence` (RAII frees blocks). `Submit` after
>   `Stop` → `FailedPrecondition`. A per-request `forward`/sampler fault fails only
>   that sequence (`kFailed` + close + free), never the loop (ADR-003, §11.2) —
>   the isolation the T03 mock-fault test covers; graceful preemption of a
>   *decode*-time pool exhaustion is deferred to M9-T09 (in T03 it surfaces as a
>   per-request failure).
> - **`OutputChannel` gained `AwaitFinish()`** (blocks until closed, returns the
>   `FinishInfo` without consuming items) — what `RequestHandle::await_completion`
>   needs. `Entry::seq` is a `std::unique_ptr<Sequence>` (two-phase init: the
>   `Request` lands first so the `Sequence` can borrow it, then the sequence is
>   built) rather than an `optional`, to keep the loop clear of
>   `bugprone-unchecked-optional-access`.

---

## 6. Scheduling policy v1 (M9-T04)

### 6.1 Inputs and purity

`Scheduler::schedule` is a pure function of plain descriptors — no pool, no
model, no tensors (§2). The runtime fills the inputs each step from its own state
and one `pool.stats()`/`pool.free_blocks()` read:

```cpp
namespace engine::scheduler {

struct PoolSnapshot {
  std::int64_t free_blocks = 0;
  std::int64_t total_blocks = 0;
  int block_size = 0;
};

struct SeqDesc {                    // one per RUNNING or WAITING sequence
  RequestId id = 0;
  std::uint64_t arrival_index = 0;  // FCFS key
  std::int64_t num_computed_tokens = 0;   // committed to cache (== length)
  std::int64_t num_prompt_tokens = 0;     // for a WAITING seq: prefill length
  std::int64_t blocks_held = 0;           // physical blocks currently owned
};

struct ScheduleInputs {
  std::span<const SeqDesc> waiting;   // FCFS order (preempted at the head)
  std::span<const SeqDesc> running;
  PoolSnapshot pool;
  SchedulerConfig config;            // max_num_seqs, max_num_batched_tokens
};

struct SchedulerOutput {
  struct Prefill { RequestId id; std::int64_t num_tokens; };
  std::vector<Prefill> prefill;      // waiting seqs to admit (whole prompt in v1)
  std::vector<RequestId> decode;     // running seqs to step one token
  std::vector<RequestId> preempt;    // running seqs to evict this step
};

}  // namespace engine::scheduler
```

Pure and deterministic → table-driven unit tests (M9-T04 acceptance) with no
engine, model, or pool.

### 6.2 The policy

Applied in this order each step:

1. **Decode-first (no starvation).** Every RUNNING sequence is scheduled to decode
   one token. This is unconditional up to memory: a running sequence is never
   dropped in favor of admitting a waiting one, so decode starvation is
   impossible (M9-T04 acceptance). Compute the block demand of the decode set:
   `Σ blocks_needed(s.num_computed_tokens, 1)` over running `s`, where
   `blocks_needed(cur, add) = ⌈(cur+add)/bs⌉ − ⌈cur/bs⌉` (the same arithmetic as
   `BlockPool::blocks_needed`, recomputed from `block_size` so the scheduler needs
   no pool type).
2. **Preempt to fit the decode set.** While the decode set's block demand exceeds
   `pool.free_blocks`, **preempt the latest-arrived RUNNING sequence** (highest
   `arrival_index`), move it to `preempt`, remove it from the decode set, and add
   its `blocks_held` back to the notional free count. Repeat until the remaining
   decode set fits. Latest-arrived is the documented victim (M9-T04 acceptance):
   it has the least sunk cost to recompute and preserves progress of older
   sequences (closest to finishing, closest to freeing their blocks). The oldest
   running sequence is never preempted while it is alone (§10.3 liveness).
3. **Admit waiting sequences FCFS.** Walk `waiting` in arrival order (preempted
   sequences first, since they are at the head). Admit a sequence to `prefill`
   while **all** hold:
   - `num_scheduled_seqs < config.max_num_seqs` (batch-width cap), and
   - `prompt_tokens_so_far + s.num_prompt_tokens ≤ config.max_num_batched_tokens`
     (the per-step prefill-token budget), and
   - `blocks_needed(0, s.num_prompt_tokens) ≤ remaining_free_blocks` (block
     availability — decremented as each admission is granted).
   Stop at the first waiting sequence that fails the block or token check; do
   **not** skip it to admit a smaller one behind it (FCFS is order-preserving —
   head-of-line blocking is accepted in v1, the price of FCFS fairness; a
   priority policy is future work). A prompt longer than `max_num_batched_tokens`
   can never be admitted — it is rejected at `submit` (§6.4), so it never sits in
   the queue forever.

### 6.3 Admission reserves prompt blocks only (the trade-off)

Admission checks blocks for the **prompt** (`blocks_needed(0, prompt_len)`), not
for the prompt plus a decode horizon. Reserving a growth margin per sequence would
waste blocks (most sequences stop well short of `max_tokens`) and complicate the
arithmetic; instead, growth is handled by the **per-step decode check** (step 1–2):
if admitting sequences leaves too little headroom, the *next* step's decode demand
trips preemption of the latest-arrived (which will often be the just-admitted
sequence). This makes admission optimistic and preemption the safety valve —
simple, and it maximizes occupancy. The cost is that a sequence can be admitted
and then preempted a few steps later under memory pressure; the resumable-error
seam (§10) makes that correct and cheap. Stated so M11/M12 can revisit (M11's
shared-prefix admission changes the block count; M12-T06 chunked prefill changes
the token budget's meaning).

### 6.4 v1 prompt-length ceiling

Because prefill is one pass (no chunking until M12-T06), a prompt longer than
`max_num_batched_tokens` cannot be scheduled — it would exceed the per-step token
budget forever. v1 rejects such a prompt at `submit` with `InvalidArgument`
naming the limit, rather than silently accepting an un-schedulable request. This
is the posture M12-T06 relaxes: chunked prefill splits the prompt across steps, so
`max_num_batched_tokens` bounds per-step work without bounding prompt length.
Recorded here so the M12 change is a planned relaxation, not a surprise.

**As built (M9-T09): the ceiling is on the *peak* length, not the prompt.** A
preempted sequence resumes by re-prefilling `prompt ++ generated` (§10.2), itself
a single prefill pass bounded by the same budget, and the longest such re-prefill
a sequence can reach is its peak `prompt_len + max_tokens - 1`. So `submit`
rejects `peak > max_num_batched_tokens` (not merely `prompt_len > …`) — otherwise
a request could be admitted, run, be preempted after generating enough tokens, and
then never fit a resume, stalling forever at the head of `waiting_`. Since
`peak ≥ prompt_len`, the peak check subsumes the prompt-only ceiling. This
realizes §10.2's "config validation flags this" as an *exact per-request* check
rather than a blanket `max_num_batched_tokens ≥ max_model_len` config inequality
(which would reject a large-context model whose individual requests all fit the
budget). M12-T06 chunked prefill relaxes both the prompt and the resume case.

### 6.5 Seams reserved for later scheduler versions

- **M11 (prefix reuse):** admission (step 3) gains a hook, before counting prompt
  blocks, to walk the prompt's hash chain, adopt cached blocks (`Share`), and
  reduce the sequence's prefill length to the uncached suffix. `SchedulerOutput::
  Prefill` already carries an explicit `num_tokens` (not "the whole prompt"), so
  a suffix-only prefill needs no shape change.
- **M12-T06 (chunked prefill):** `Prefill::num_tokens` becomes a per-step chunk
  (< prompt_len), scheduled *alongside* decodes under one `max_num_batched_tokens`
  budget; the state machine gains no state (a chunk-in-progress sequence is
  RUNNING with `num_computed_tokens < num_prompt_tokens`).
- **M15-T06 (speculative):** `SchedulerOutput::decode` entries gain a
  tokens-this-step count (the accepted-window size), and the step loop appends a
  variable number of tokens per sequence.

### 6.6 As built (M9-T04)

`src/scheduler/scheduler.{h,cpp}` implements exactly this section. Notes:

- **`class Scheduler` with a `const` `Schedule` method**, stateless in v1 (a
  free function would do, but the class is the M11 hook's home per §6.5 — so
  call sites never churn). The block arithmetic is a public `constexpr
  BlocksNeeded(cur, add, bs)`, cross-checked bit-for-bit against
  `BlockPool::blocks_needed` in a test (the only test that touches a real pool);
  everything else is pure struct-in / struct-out, so the suite is `scheduler`-
  labelled with **no** model/pool/allocator and no `SCALAR_PASS`.
- **`preempt` is emitted latest-arrived-first** (the eviction order). The engine
  requeues each at the head of `waiting_` via `push_front`, which lands them in
  arrival order among themselves (§3.2). Ties on `arrival_index` break to the
  last such element in the `running` span (deterministic).
- **The scheduler does not special-case the oldest-alone sequence.** If a lone
  running sequence's decode does not fit, it is still preempted (empty decode
  set that step). Liveness (§10.3) is a *config-sizing* guarantee — the pool
  holds at least one `max_model_len` sequence — enforced where the pool is
  sized, not inside the pure policy.
- **`RequestId` is redeclared in `scheduler.h`** (the scheduler links only
  `core`, never `runtime/request.h`, which sits above it). `engine.cpp`
  `static_assert`s `runtime::RequestId` and `scheduler::RequestId` are the same
  type.
- **Runtime wiring (`Engine::ScheduleStep` + `Step`).** The M9-T03 placeholder
  `AdmitWaiting` is replaced: `ScheduleStep()` fills the descriptors from the
  engine-thread state — running `num_computed_tokens`/`blocks_held` read
  straight from each `Sequence`'s `PagedKvCache` (`length()` /
  `block_table().num_blocks()`), waiting `num_prompt_tokens` = prompt length +
  `num_generated()` (0 for a fresh sequence; the resume length for a preempted
  one, §10.2) — and `Step()` applies preempt (§9.1 step 4: `ReleaseCache` →
  PREEMPTED → head of `waiting_`) then prefill then decode. **Preemption
  mechanics land here** rather than waiting for M9-T09, because the scheduler
  can now legitimately emit a `preempt` (admission reserves prompt blocks only,
  §6.3) and the engine must act on it; the prefill path was generalized to
  re-prefill `prompt ++ generated` so a resumed sequence's next token is
  bit-identical to an uninterrupted run. M9-T09 keeps its own scope: validating
  preemption under the *batched* loop, the tiny-pool forced-preemption / no-leak
  acceptance, and the pool-sizing config validation.
- **`runtime → scheduler` links PUBLIC** (`engine.h` names `SchedulerOutput` in
  `ScheduleStep`'s private declaration). No new ADR edge — the edge is already
  on the diagram (`server → runtime → scheduler`).

---

## 7. Batch composition: two passes, not mixed

**Decision: each step runs at most two forwards — one ragged *prefill* forward
over the admitted sequences, then one batched *decode* forward over the running
sequences — never a single mixed prefill+decode forward.** Justified:

- **The two phases use different attention kernels.** Prefill runs the blocked,
  causal, online-softmax `PrefillAttentionF32` over a gathered `[Hkv, L, d]` K/V
  slab; decode runs `PagedDecodeAttentionF32` reading K/V *through the block table*
  (one query per head over the whole cache). A mixed batch would need a unified
  kernel handling both ragged multi-token causal attention and single-token paged
  attention in one pass — complexity with no upside on CPU.
- **Mixing buys nothing on CPU.** The motivation for a fused prefill+decode step
  (as in some GPU engines) is amortizing kernel-launch overhead; there is no launch
  overhead here — a forward is a function call into `parallel_for`. Two sequential
  forwards cost the same arithmetic as one fused forward and are far simpler.
- **Decode-priority is a *scheduling* property, not an execution-order one.** The
  scheduler guarantees every running sequence decodes every step (§6.2 step 1);
  running the prefill forward first or second within the step does not change which
  sequences advance. We run **prefill first** so a newly admitted sequence samples
  its first token (from the prefill's `kLast` per-sequence logits) *this* step and
  joins the decode batch *next* step — it does not idle a step. Its first decode
  therefore happens one step after admission, uniformly.
- **The continuous-batching invariant is preserved** (§8.5): each sequence's math
  is independent of its batch-mates in both passes, so batching changes only *when*
  a token is computed, never *what* it is.

Consequence for the step: prefill batch (0..N_admit sequences) → forward →
sample first tokens; decode batch (0..N_run sequences) → forward → sample next
tokens. Either batch may be empty (a step with only decodes, or only a prefill).

---

## 8. Ragged/batched forward & batch assembly (M9-T05…T07)

### 8.1 The additive `ForwardRequest` fields

model-execution.md §5.4 reserved this: `ForwardRequest` grows into a batch bundle
by **adding fields**, not changing the signature. The single-sequence path is the
`B = 1` special case (empty `cu_seqlens` ⇒ the existing scalar flow).

```cpp
struct ForwardRequest {
  std::span<const std::int32_t> token_ids;   // [Σ T_b], flattened, batch-major
  std::span<const std::int32_t> positions;   // [Σ T_b], per-seq absolute pos
  kvcache::KvCache* cache = nullptr;         // B==1 path (unchanged)
  LogitsMode logits_mode = LogitsMode::kLast;
  ActivationHook* hook = nullptr;
  // --- M9 additions (empty ⇒ single-sequence path) ---
  std::span<const std::int32_t> cu_seqlens;  // [B+1] prefix sums of T_b
  std::span<kvcache::KvCache* const> caches; // [B] one cache per sequence
};
```

- **`cu_seqlens`** ([B+1] cumulative sequence lengths, the standard varlen
  convention) delimits sequence `b`'s tokens as `[cu_seqlens[b], cu_seqlens[b+1])`
  in the flattened `token_ids`/`positions`. Prefill sequences have `T_b =
  prompt_len_b`; decode sequences have `T_b = 1`.
- **`caches`** is per-sequence (each sequence has its own `PagedKvCache` over the
  shared pool). K/V for sequence `b` is appended to `caches[b]` — see §8.3.
- **Output**: with `LogitsMode::kLast` and a batch, the model returns `[B, V]`
  (the last token's logits per sequence) — row-contiguous per sequence, exactly
  the `[num_seqs, V]` block `BatchedSampler` consumes (model-execution.md §5.2
  fixed per-sequence contiguity for this). `kAll` returns `[Σ T_b, V]`.

### 8.2 Batch assembly (`engine/batch.h`, M9-T05)

`AssembleBatch(SchedulerOutput, sequences) → BatchInputs&` flattens the scheduled
work into preallocated staging buffers in **one pass, no per-step allocation**
(the buffers grow-on-demand to a high-water mark, like the `Workspace`, §6.3 of
the optimized doc — and this closes that doc's deferred "pre-size from
`--max-num-batched-tokens`" note: the staging buffers *are* sized by the batch-
token budget):

```cpp
struct BatchInputs {                     // identical for prefill and decode
  std::vector<std::int32_t> token_ids;   // flattened
  std::vector<std::int32_t> positions;   // per-seq absolute positions
  std::vector<std::int32_t> cu_seqlens;  // [B+1]
  std::vector<kvcache::KvCache*> caches; // [B]
  std::vector<BatchRow> sample_rows;     // [B] sampler + context per seq
  // (M9-T05 also staged a [B, max_blocks] block_table + seq_lens for decode;
  //  M9-T07 removed them — the batched decode kernel self-sources per-sequence
  //  tables post-append, §8.4 as-built.)
};
```

Assembled fields, hand-verifiable (M9-T05 acceptance — exact tensor contents for
"2 prefills of different lengths", "3 decodes", "mixed"):

- **`token_ids`/`positions`**: prefill sequence `b` contributes its prompt ids at
  positions `[0, prompt_len_b)`; a decode sequence contributes its one new token
  at position `num_computed_tokens`.
- **`cu_seqlens`**: prefix sums of the per-sequence `T_b`.
- **`sample_rows`**: `BatchRow{&seq.sampler, seq.context()}` per sequence — the
  batched-sampler input (§8.4), letting one batch freely mix greedy/stochastic,
  temperatures, penalties, and logprob requests (M7-T06).
- ~~**`block_table`** (decode only): `[B, max_blocks]` int32~~ — **retired in
  M9-T07.** A pre-forward block-table snapshot is stale for any sequence whose
  decode token crosses a block boundary (the new block is allocated *inside* the
  forward, §8.3/§8.4 as-built), so the batched decode kernel self-sources
  per-sequence tables post-append instead. The assembly stages no block-table
  tensor or `seq_lens`; `BatchInputs` is the flattened
  token/position/cu_seqlens/caches/sample_rows bundle for both passes.

**As built (M9-T05, block-table field retired M9-T07):**
`src/engine/batch.{h,cpp}` — `BatchAssembler` +
`BatchInputs` (`engine::engine`). Realized choices, matching the acceptance
tests:

- **Input is a `BatchSeqInput` descriptor, not `SchedulerOutput` + `Sequence`.**
  ADR-002 keeps `engine` free of `runtime`/`scheduler` types (the scheduler is a
  `core`-only leaf; `Sequence` lives *above* `engine`). So the runtime fills a
  plain `BatchSeqInput{token_ids, cache, sampler, context}` per scheduled
  sequence — the §8.2 shorthand `AssembleBatch(SchedulerOutput, sequences)` is
  this descriptor input. No new ADR edge; `engine`'s `tensor` link moved
  PRIVATE→PUBLIC and `memory` was added (both already-below-it layer-1 modules)
  for `batch.h`'s public surface.
- **Two entry points** — `AssemblePrefill` / `AssembleDecode` — mirroring the §7
  two passes. A shared `Flatten` fills the common
  `token_ids`/`positions`/`cu_seqlens`/`caches`/`sample_rows`; decode adds
  `seq_lens` + `block_table`. **Positions are `cache->length() + t`** uniformly
  (a fresh prefill cache ⇒ `[0, T)`; a decode cache of length `L` ⇒ `L`), so the
  runtime never passes a start position.
- **No slot-mapping field.** §8.3 keeps K/V append per sequence through
  `PagedKvCache::append` (which computes its own slots and owns the exhaustion
  seam), so the batch carries no batch-level slot mapping — the roadmap's "slot
  mappings" line is subsumed by per-sequence append, recorded here.
- ~~**`block_table` via the abstract `paged_view(0)`**~~ — **removed in M9-T07**
  (the T05 decode assembly built a `[B, max_blocks]` block-table tensor + a
  `seq_lens` snapshot; the batched decode kernel now self-sources per-sequence
  tables post-append, so both are gone, see §8.4 as-built). **Allocation-free
  after warm-up:** the staging vectors keep capacity across steps
  (`staging_bytes()` is the stability metric); no tensor allocation remains, so
  the assembler dropped its `memory::Allocator` and `engine`'s public surface no
  longer names a tensor type (the M9-T05 `engine::tensor` PUBLIC→PRIVATE,
  `engine::memory` dropped).
- **Additive `ForwardRequest` fields landed with T05** (§8.1): `cu_seqlens` +
  `caches` (empty ⇒ the single-sequence path). Both backends
  (`ReferenceModel`/`OptimizedModel`) implement the batched forward as of M9-T07
  (M9-T05/T06 rejected a non-empty batch with `Unimplemented`).
  `BatchInputs::MakeForwardRequest` builds the batched request over the staging.

### 8.3 K/V append stays per sequence

**Decision: even in a batched forward, each sequence's K/V is appended through its
own `PagedKvCache::append`, sliced from the flattened rows by `cu_seqlens`.** The
model's layer loop, for each layer, slices the layer's K/V projections
`[Σ T_b, Hkv·d]` per sequence and calls `caches[b]->append(layer, k_b, v_b)`.
Rationale:

- It preserves the M8 per-forward layer protocol (`append(layer)` once per layer
  in order, the `pending_slots_` grow-then-scatter, the front-loaded validation,
  the exhaustion posture — paged §8.2) unchanged: batching is a loop *around* the
  existing append, not a new append path.
- It keeps the **exhaustion seam** (§10 / paged §10.2) exact: sequence `b`'s
  layer-0 append is where its block growth happens, so a pool-dry condition
  surfaces as that sequence's `ResourceExhausted` — which the loop turns into a
  preemption or a per-request failure (§10, §11), not a whole-batch abort.
- Attention then reads per sequence: prefill via `caches[b]->view(layer)` (the
  gather, paged §9.3) fed to `PrefillAttentionVarlenF32` (§8.4); decode via the
  batched block-table tensor + `PagedDecodeAttentionBatchedF32`.

The alternative — one giant append across the batch — would entangle the per-
sequence exhaustion boundary and gain nothing (the scatter is per-token regardless).

### 8.4 Batched kernels

- **`PrefillAttentionVarlenF32`** (M9-T06): the ragged prefill kernel. Loops over
  sequences (`cu_seqlens`), running the **unchanged** per-sequence
  `PrefillAttentionF32` recurrence over each sequence's `[Hkv, L_b, d]` slab —
  "shared kernels loop over sequences; still naive-but-correct" (roadmap). Threaded
  over (sequence, query-block, head) units. Because each sequence's attention is
  computed exactly as a standalone prefill, the batch's per-sequence outputs are
  **identical** to running each sequence alone (M9-T06 acceptance: batch of
  lengths {5,64,129} matches per-sequence runs exactly).

  **As built (M9-T06):** `src/kernels/attention.{h,cpp}` — `PrefillAttentionVarlenF32`.
  Realized choices:
  - **No new per-ISA code, no new arithmetic.** The "unchanged per-sequence
    recurrence" is realized *literally*: the varlen entry reuses the existing
    dispatched `PrefillUnits` variant (`scalar`/`neon`/`avx2`) verbatim — it adds
    only sequence-major unit bookkeeping in `attention.cpp` (the M8-T05
    argument taken one step further). So there is **no `{isa}/*` varlen TU** and
    **no new `internal::*Impl` template**; the blind AVX2 path is covered by
    construction and SCALAR_PASS exercises the shipped scalar bytes through the
    same variant. This differs from the ticket's "(+ per-ISA TUs)" shorthand,
    recorded here.
  - **Per-sequence K/V as pointer arrays, not one concatenated slab.** §8.3 reads
    each sequence's K/V through its own `caches[b]->view(layer)` — B separate
    gathered `[Hkv, L_b, d]` tensors — so the signature takes `const float* const*
    k_seqs/v_seqs` + a per-sequence `l_dims[]`, exactly the per-layer `vector<KvView>`
    the T07 model loop will hold. `cu_seqlens` is `int32` to match
    `ForwardRequest::cu_seqlens`/`BatchInputs`.
  - **Sequence-major unit space.** Sequence `b` owns the contiguous global units
    `[S_b, S_b + H·⌈T_b/kAttnQb⌉)`; the public entry parallelizes over the total
    `U = Σ_b U_b` with grain 1 (the single-sequence work item). The chunk splitter
    `detail::PrefillVarlenUnits` (the test seam) walks sequences (O(B), allocation-
    free) and, for each one overlapping the requested `[begin,end)`, synthesizes
    the *same* `PrefillArgs` a standalone call would build and hands the variant
    the local sub-range — so the fp32 arithmetic order per output row is unchanged
    from a standalone run. This gives per-sequence bit-identity **and** thread/chunk
    invariance by construction (both asserted with 0-tolerance allclose).
  - **`ForwardRequest`/model wiring is M9-T07.** The batched `forward` (per-sequence
    K/V append sliced by `cu_seqlens`, prefill branch → this kernel, decode branch →
    `PagedDecodeAttentionBatchedF32`) lands with T07, where it can be exercised
    end-to-end; both backends keep rejecting a non-empty batch with `Unimplemented`
    until then. No `src/` consumer change and no BASELINES entry in T06 (no perf
    claim — the per-sequence recurrence is unchanged; the whole-step number is the
    T08 throughput-sanity criterion). +8 gtest cases (×2 with SCALAR_PASS,
    `varlen_attention_kernel_test`) → 1311 ctest green.
- **`PagedDecodeAttentionBatchedF32`** (M9-T07): the batched decode kernel. Loops
  over sequences, running the **unchanged** per-sequence `PagedDecodeAttentionF32`
  recurrence for each, reading that sequence's block table + length. Threaded over
  (sequence, kv head). Per fixed (sequence, head) the arithmetic order is
  identical to the single-sequence kernel, so it is **bit-identical** to a
  sequential single-sequence decode of each member — the batched-decode
  correctness criterion (M9-T07 acceptance).

  **As built (M9-T07):** `src/kernels/paged_attention.{h,cpp}` —
  `PagedDecodeAttentionBatchedF32`. Realized choices:
  - **Per-sequence pointer arrays, not the `[B, max_blocks]` tensor.** The
    original §8.2 sketch had a decode assembly stage a `−1`-padded block-table
    tensor + `seq_lens` for this kernel. But **block growth happens inside the
    forward** — layer 0's per-sequence `append` (§8.3) allocates a new block when
    a decode token crosses a block boundary (`L % bs == 0`) — so a tensor
    snapshotted *before* the forward is one block short (and `seq_lens` one token
    short) exactly for those sequences. The model therefore self-sources each
    sequence's `paged_view(layer)` **after** the layer's appends and passes the
    kernel `const int32_t* const* block_tables` + `const int64_t* lengths`
    (post-append), plus the **shared** `k_slab`/`v_slab`/`block_stride`/
    `block_size` (all sequences share one `BlockPool`, so `paged_view` returns
    identical slab bases; the model validates this). This is the M9-T06
    varlen-prefill precedent (per-sequence K/V pointer arrays) applied to decode.
    Consequently the batch assembly (§8.2) stages **no** block-table tensor or
    lengths, and the reserved `[B, max_blocks]` shape (paged-kv-cache.md §9.4) is
    retired as-built.
  - **No new per-ISA code, no new arithmetic.** Unit space is batch-major
    `B·Hkv` (one (sequence, kv head) pair per unit); `detail::PagedDecodeBatched-
    Units` synthesizes the *same* `PagedDecodeArgs` a standalone call builds for
    each sequence and invokes the existing dispatched `PagedDecodeUnits` variant
    on that sequence's single kv head — so there is no `{isa}/*` batched TU and
    SCALAR_PASS exercises the shipped bytes. Decode parallel width grows from
    `Hkv` (single-sequence) to `B·Hkv`, partially relieving the M6-T05 idle-cores
    note for batched decode.
  - **Model wiring** (`OptimizedModel::BatchedAttention`): per-sequence K/V append
    sliced by `cu_seqlens`, then — every T_b == 1 over paged caches — this kernel
    on the post-append `paged_view`s; else (any T_b > 1, or non-paged caches) the
    varlen prefill kernel via `view()`. The reference backend realizes its batched
    forward as per-member single-sequence forwards concatenated (the oracle).
- **Batched sampling** (M7-T06, already built): the `[B, V]` logits block feeds
  `BatchedSampler::Sample(logits, V, sample_rows, out)`, which picks the same
  token+logprobs the single-sequence `Sampler` would, per row, by construction
  (it calls the same `detail::SampleRow`).

### 8.5 The batch-invariance invariant (why the CB invariant holds bit-for-bit)

Every operation in the forward is either **row-local** (embedding lookup, RMSNorm,
SiLU, residual add, RoPE — each token/row computed independently of other rows) or
**sequence-local** (attention — each sequence attends only its own K/V; the causal
mask and per-sequence cache guarantee no cross-sequence leakage). The one operation
that could differ is the GEMM: a batched GEMM (`PackedGemm` with `M = Σ T_b`)
computes row `m` by the *same* K-accumulation order as the single-row GEMV
(`PackedGemv`) — verified bit-for-bit today by `packed_gemm_test.GemvMatchesGemmRow`
(`EXPECT_EQ` across every row). Therefore a sequence's per-row outputs are
**bit-identical** whether it runs alone or batched with others. The batched sampler
is bit-identical by construction (§8.4). Chained across all layers and the sampler,
this gives the **continuous-batching invariant**: N concurrent greedy requests
produce byte-identical tokens to N sequential runs (M9-T08 acceptance), and the
stochastic case is identical too (each sequence's Philox counter is keyed by its
own step index, not its batch position, M7-T02).

This also answers M17-T04's open question ("batching-dependent numerics — decide
and document whether batch-invariance is guaranteed"): **on a fixed ISA it is
guaranteed, bit-for-bit**, because batching never reorders any reduction. (Across
ISAs it remains Class T, as everywhere — the exp polynomial and FMA order differ by
≤ a few ulp; that is orthogonal to batching.) M17-T04 records the final contract;
M9 supplies the guarantee.

---

## 9. The step loop & invariants (M9-T08)

### 9.1 Pseudocode

```
Engine::step():
  # 1. Absorb client commands (top of step → deterministic boundaries)
  for cmd in drain(submission_queue):
     if cmd is Submit:  make Sequence(WAITING); add to waiting; register channel
     if cmd is Cancel:  mark id cancelled  (handled in step 7 by state)

  # 2. Retire freshly cancelled WAITING/PREEMPTED sequences (free nothing / free blocks)
  for s in waiting + preempted where cancelled(s):
     drop s.cache (RAII frees blocks); close channel(kCancelled); erase s

  if waiting.empty() and running.empty():  return false   # idle

  # 3. Schedule
  out = scheduler.schedule({waiting-descs, running-descs, pool.snapshot(), config})

  # 4. Apply preemptions (before any forward, to free their blocks for this step)
  for id in out.preempt:
     s = seq(id); s.drop_cache()            # FreeAll → blocks back to pool
     s.state = PREEMPTED; move s to head of waiting; close nothing

  # 5. Prefill pass (admitted sequences), if any
  if out.prefill not empty:
     for p in out.prefill:                  # create/reset per-seq cache
        seq(p.id).ensure_cache(pool); seq(p.id).state = RUNNING
     batch = AssembleBatchPrefill(out.prefill, sequences)
     logits = model.forward(prefill ForwardRequest)      # [N_admit, V], kLast
     handle_forward_result(...)             # per-seq: propagate/fail (§11)
     sampler.Sample(logits, V, batch.sample_rows, results)
     for each admitted seq: append_and_deliver(seq, results[b])   # §9.2

  # 6. Decode pass (running sequences), if any
  if out.decode not empty:
     batch = AssembleBatchDecode(out.decode, sequences)
     logits = model.forward(decode ForwardRequest)       # [N_run, V], kLast
     handle_forward_result(...)
     sampler.Sample(logits, V, batch.sample_rows, results)
     for each running seq: append_and_deliver(seq, results[b])

  # 7. Retire finished / failed / cancelled-mid-step
  for s in running where terminal(s):
     close channel(FinishInfo); drop s.cache; erase s
  return true
```

`append_and_deliver(seq, result)`: append the sampled id to `seq.generated_ids`;
run `seq.stop_.Observe(id, ...)` → text delta + finish decision; `Push` the
`OutputItem` (id, delta, logprobs) to the channel; if the stop check fires or a
cancel is pending, set the terminal state (handled in step 7). This *is* the body
of the single-sequence `Generate` decode loop, run per sequence — which is why the
outputs match sequential runs.

### 9.2 Invariants (acceptance criterion — the explicit list)

1. **Block sufficiency:** a RUNNING sequence always holds every block it needs for
   the token it is about to produce this step (the scheduler admitted/kept it only
   after `blocks_needed` fit `free_blocks`, and the engine thread is the sole
   allocator, so no other actor drained the pool in between). This is the roadmap's
   named invariant.
2. **Single mutator:** only the engine thread mutates block state, cache contents,
   and sequence state. (Concurrency correctness, §5.3.)
3. **Batch invariance:** a sequence's produced tokens are bit-identical whether it
   runs alone or batched (§8.5). ⇒ the continuous-batching invariant.
4. **Streaming fidelity:** the concatenation of a channel's `text_delta`s equals
   the sequence's full detokenized text; each `OutputItem.token_id` is delivered
   exactly once, in order, before the channel closes.
5. **No leaks:** when no sequence is live, `pool.stats().used == 0`. Every terminal
   transition drops the sequence's cache (RAII), and preemption frees blocks
   immediately.
6. **Resume equivalence:** a preempted-then-resumed sequence produces output
   identical to an uninterrupted run (§10, the KV invariant as resumability).
7. **Cancel latency:** a cancel takes effect within one step (it is applied at the
   next step boundary; the sequence's channel closes `kCancelled` that step).
8. **FCFS among equals:** waiting sequences are admitted in arrival order (preempted
   sequences ahead of never-started ones, §3.2).

### 9.3 Throughput sanity (M9-T08 acceptance)

8 concurrent requests complete in well under 8× a single request's wall-clock —
because the decode steps batch (8 sequences' decode is one forward, not 8), the
per-token cost grows sublinearly with batch size until memory-bandwidth-bound. The
integration test records the ratio; it is a sanity check (the engine batches at
all), not a tuned number (M12 owns throughput tuning). The number is recorded, not
asserted against a threshold.

### 9.4 As built (M9-T08)

`src/runtime/engine.{h,cpp}` — `Step()`'s M9-T03/T04 per-sequence placeholder
execution is replaced in place by the two batched passes; steps 1–4 (drain,
retire-cancelled, schedule, apply-preempt) are unchanged. Realized choices, all
deliberate:

- **The batched passes reuse the pre-built pieces verbatim.** Step 5 collects
  the admitted sequences into `pass_entries_`, step 6 the running set; each calls
  one `RunBatchPass`, which fills `BatchSeqInput`s → `BatchAssembler::Assemble{
  Prefill,Decode}` (M9-T05) → `MakeForwardRequest(kLast)` → `model.forward`
  (M9-T07 batched) → one `BatchedSampler::Sample` over the `[B, V]` block (M7-T06)
  → per-row `DeliverSampled`. The engine thread owns one reused `assembler_`, one
  `batched_sampler_`, and the per-pass scratch (`pass_entries_`/`pass_seqs_`/
  `pass_results_`/`prefill_tokens_`), grown on demand and kept across steps, so a
  steady-state decode allocates nothing (§5.2).
- **B == 1 goes through the batched path too** — no single-sequence fast path in
  the normal loop. Bit-identity to the standalone `Generate` is the M9-T07
  guarantee (§8.5), so every existing T03/T04 mock test now exercises the batched
  loop unchanged, and the new fixture suite proves the real-model CB invariant on
  both backends. `ExecuteAndDeliver` (the single-sequence forward) survives only
  as the fault-recovery fallback below.
- **Two-tier per-request fault recovery** (the T08-scope realization of §11.2,
  which fully lands in M9-T10). A batched forward appends K/V per sequence and
  samples per row, so a fault must fail only the offending request:
  - *Forward fault* (`RecoverForwardFailure`): a mid-batch forward error can
    leave one cache mid-append. The loop snapshots each member's pre-forward
    `cache->length()`, and on failure `truncate`s every member back to it
    (clearing the in-progress-forward state) then re-runs each member through the
    single-sequence `ExecuteAndDeliver`. Healthy members deliver bit-identically;
    only the genuine faulter fails. A decode-time pool exhaustion surfaces here as
    a per-request failure — routing it to preemption is M9-T09.
  - *Sampler fault* (`RecoverSampleFailure`): the forward committed all K/V, so
    the loop must **not** re-run it. `BatchedSampler::Sample` reports the first bad
    row and leaves `out` unspecified (the per-row-status refinement is M9-T10, §14),
    so the loop re-samples each row over the **same** committed `[B, V]` logits via
    the single-sequence `SampleWithLogprobs` — bit-identical (same
    `detail::SampleRow`) — delivering healthy rows and failing only the bad ones.
    This is the path the `PerRequestFaultIsolated` test (a NaN-logits request
    batched with a healthy one) now exercises.
  Both recovery paths run only on the error path; the steady state is one batched
  forward + one batched sample per pass.
- **RUNNING-cancel stays at the top of the step** (`RetireCancelled`, step 2),
  not moved to the step end as §5.2's M9-T03 as-built note anticipated for T08.
  Commands are drained only at step boundaries (§5.2), so there is never an
  in-flight batch when a cancel is applied; handling it at the top is simpler and
  equivalent, and a cancel that arrives *during* a forward still lands in the
  queue and takes effect at the next boundary (≤ one step, §11.1). The in-flight
  token is produced but never delivered (the channel closes `kCancelled` first).
- **`runtime_batching_test.cpp`** is the milestone's headline suite (§13,
  `runtime` label, **SCALAR_PASS** — the first runtime suite in the forced-scalar
  pass): 8 concurrent greedy requests through the `Engine` == 8 sequential
  `Generate` runs, token-for-token, on tiny-llama (untied) and tiny-qwen2
  (tied+biases); staggered mid-flight arrivals matching their standalone runs
  (the mixed prefill+decode step); and a recorded throughput ratio (§9.3 — 8
  concurrent completed in ≈0.34× the 8-sequential wall-clock on the dev machine,
  well under 8×). `runtime_engine_test.cpp` gained the mock-driven CB-invariant
  and staggered-arrival cases and now drives its whole 28-case suite through the
  batched loop. No new ADR edge (`runtime → engine`/`sampling` already linked);
  no `src/` change beyond `engine.{h,cpp}`.

---

## 10. Preemption & recomputation (M9-T09)

### 10.1 Mechanics

When the scheduler emits a `preempt` id (pool exhausted, §6.2 step 2), the engine:

1. **Frees the victim's blocks:** drops (or `truncate(0)`s then keeps) its
   `PagedKvCache` → `BlockTable::FreeAll` returns every block to the pool
   (paged §7.3). The blocks are available *this* step for the decode/prefill that
   needed them.
2. **Sets state PREEMPTED** and requeues at the **head** of `waiting` (§3.2), ahead
   of never-started requests.
3. **Keeps the sequence's progress:** `generated_ids`, the `Sampler` (with its
   resolved seed), and the `StopChecker`/`DetokenizerStream` state are retained on
   the `Sequence`. Only the KV cache (recomputable) is discarded.

### 10.2 Resume = re-prefill prompt+generated

When the scheduler later re-admits a PREEMPTED sequence, the engine schedules a
**prefill of `prompt_ids ++ generated_ids`** (length = its full progress so far)
into a fresh cache. This rebuilds exactly the KV the sequence had, because the KV
for a token depends only on the tokens up to it (the KV invariant, model-execution
§6.2): re-prefilling the same token prefix reproduces the same K/V bit-for-bit, so
the sequence's *next* sampled token is identical to what it would have produced had
it never been preempted. **This is the M8-T08 resumable-error seam promoted to a
scheduler action** — M8-T08 proved `delivered ++ resumed == uninterrupted` for the
single-sequence exhaustion case; M9-T09 uses the same equivalence, driven by the
scheduler instead of a caller retry. Evict-and-recompute (recompute the KV) rather
than swap-out (copy KV to host and back): on a CPU engine host RAM already *is* the
pool, so there is no faster tier to swap to — recompute is the only sensible policy
(recorded, §1 non-goal).

Note the re-prefill counts against the prefill token budget (§6.2 step 3) like any
prefill, and can itself be too big to admit in one step under a tight
`max_num_batched_tokens` — which is exactly the case M12-T06 chunked prefill fixes
(re-prefill in chunks). In v1, `max_num_batched_tokens` must be ≥ the longest
`prompt+generated` a sequence can reach (bounded by `max_model_len`), or a very
long preempted sequence cannot resume; the config validation flags this.

### 10.3 Liveness

Preemption can never deadlock: the scheduler preempts the **latest-arrived** first
(§6.2 step 2), so the single oldest running sequence is preempted only if it is the
only running sequence *and* its own next token does not fit — but a single sequence's
full length is bounded by `max_model_len`, and the pool is sized (config validation)
to hold at least one `max_model_len` sequence, so the oldest-alone sequence always
fits and is never preempted. Therefore at least one sequence always makes forward
progress each step, and every sequence eventually becomes the oldest and completes.
M9-T09 acceptance (all requests complete despite forced preemptions on a tiny pool)
exercises exactly this.

### 10.4 As built (M9-T09)

The preemption *mechanics* (evict-and-recompute, requeue-at-head, resume by
re-prefill) landed in M9-T04 (§6.6) because the scheduler could already emit a
`preempt` and the engine had to act on it. M9-T09 completes the section:

- **Config-sizing liveness enforced (§10.2, §10.3).** Two checks make the
  liveness argument hold rather than assume it:
  - `Submit` rejects `peak > max_num_batched_tokens` (§6.4 as-built) so a
    preempted sequence's resume re-prefill always fits the budget — the exact
    per-request form of §10.2's "config validation flags this".
  - `Create` rejects an **explicitly pinned** `max_model_len` that exceeds the
    pool's token capacity (`num_blocks · block_size`): the pool must hold one
    full-length sequence so the oldest-alone sequence never needs preemption to
    fit its own next token. An *auto-resolved* `max_model_len` (from the model's
    `max_position_embeddings`) is **not** checked at `Create` — a large-context
    model over a small pool is legitimate, and `Submit`'s per-request
    peak-vs-pool-capacity check enforces the same guarantee for every admitted
    request.
- **A preemption counter** (`Engine::num_preemptions()`, bumped in the step-4
  apply loop) makes the forced-preemption acceptance test non-vacuous and is the
  counter M16 exports (§12).
- **Acceptance validated under the batched loop** on both real fixtures
  (`runtime_batching_test`, SCALAR_PASS) and the mock (`runtime_engine_test`): a
  tiny pool that cannot hold all sequences at once forces repeated preemptions,
  every request completes with output **identical** to its standalone `Generate`
  (the resume-equivalence invariant, bit-exact via the KV invariant on the real
  backend), `num_preemptions() > 0`, and `pool.stats().used == 0` at the end (no
  leaks).
- **Reactive decode-exhaustion routing (§11.2) is *not* part of M9-T09.** For
  the engine's own pool the scheduler's block accounting is exact — decode demand
  is charged against `free_blocks` before admission (§6.2) — so a decode append
  never exhausts and no reactive path fires. A decode-time `ResourceExhausted` is
  only reachable when the pool is shared with an external allocator that drains it
  between the scheduler snapshot and the append; routing that to graceful
  preemption instead of per-request failure belongs to M9-T10, where §11.2 places
  it alongside the per-row `BatchedSampler` status change and the isolation tests
  (and where it is deterministically testable). The M9-T08 code comments that
  forward-referenced this to "M9-T09" were corrected to M9-T10.

---

## 11. Cancellation & per-request failure isolation (M9-T10)

### 11.1 Cancellation

`Cancel(id)` enqueues a cancel command; it takes effect at the next step boundary
(§5.2), so cancel latency is ≤ one step (M10-T04's within-one-step requirement).
Effect by state:

- **WAITING / PREEMPTED:** the sequence is removed before scheduling (step 2); its
  cache (if any — a PREEMPTED sequence has none, a WAITING sequence created none)
  is dropped, freeing blocks; the channel closes `kCancelled`.
- **RUNNING:** the sequence is marked cancelled; at step 7 (or before the next
  step's scheduling) its cache is dropped (blocks freed) and the channel closes
  `kCancelled`. If a cancel arrives mid-step (after scheduling, during the
  forward), it is honored at the *next* boundary — the in-flight forward completes
  (it is already batched and running), the produced token is simply not delivered
  after the channel closes.

All three reclaim resources; M9-T10 acceptance verifies `pool.stats()` is restored
(no leaked blocks) for cancel-during-WAITING, -prefill, and -decode.

### 11.2 Per-request failure isolation

A fault scoped to one request must fail **only** that request, never the loop
(ADR-003; the engine loop treats request-derived data as recoverable `Status`,
never `CHECK`). Two sources:

- **Sampler edge case** (a non-finite logits row, a bad history id). `BatchedSampler`
  returns per-row results; the failing row's sequence transitions to FAILED with
  the row's `Status`, its channel closes `kFailed`, its blocks free; every other
  row in the batch is delivered normally. *(Required additive change flagged: the
  current `BatchedSampler::Sample` returns only the lowest-index row's error and
  leaves `out` unspecified on any error (batched_sampler.h). M9-T10 changes it to
  a per-row status — write each row's `Status` into `out[b]` and always fill every
  row — so one bad row does not discard the batch. This is an M9-T10 code change,
  recorded here so it is a planned interface refinement, not a silent divergence
  from M7-T06.)*
- **Per-sequence `forward` error** (a bad token id, a position overflow, or a
  `ResourceExhausted` from that sequence's own append). Because appends are per
  sequence (§8.3), the model can attribute the error to a sequence index; the loop
  fails that sequence (FAILED + close + free) and continues the others. A
  `ResourceExhausted` on a *decode* append is instead routed to preemption (§10),
  not failure, when the scheduler can free blocks by preempting a younger sequence;
  it becomes a failure only if the sequence is the sole occupant and genuinely
  cannot fit (the front-loaded-vs-mid-generation distinction of paged §10.2).
  *(This decode-exhaustion routing lands in M9-T10, not M9-T09: for the engine's
  own pool the scheduler is exact so it never fires — it is only reachable under a
  shared/externally-drained pool — and it is deterministically testable only
  alongside the per-row `BatchedSampler` status and the isolation harness here.
  §10.4 records the reassignment from the M9-T08 forward-reference.)*

A genuinely engine-internal invariant violation (a wrong-rank tensor the engine
built itself, an illegal state transition) remains a `CHECK` — those are bugs, not
request data. M9-T10 acceptance: an injected per-request fault leaves every other
concurrent request's output unchanged.

---

## 12. Interactions with later milestones

- **M10 (serving).** `docs/design/server.md` wires sockets to the runtime:
  `Submit` per HTTP request, stream `OutputChannel` items as SSE chunks, map
  `FinishInfo` → OpenAI `finish_reason`/`usage`, and disconnect → `Cancel`. The
  runtime's unbounded channel gets M10's bounded/backpressure policy. No runtime
  change beyond what §5 exposes.
- **M11 (prefix caching).** The admission hook (§6.5) adopts cached blocks and
  shortens prefill to the suffix; the scheduler-stats surface (prefill token count)
  is what M11-T04's "second identical prompt skips prefill" test asserts against.
- **M12 (perf).** Chunked prefill (T06) extends `Prefill::num_tokens` to a chunk
  and the budget's binding meaning (§6.4); flash-decoding (T03) changes only the
  batched decode kernel's threading (§1 non-goal); staging-buffer reuse (T01) is
  already the §8.2 posture.
- **M15 (speculative).** `SchedulerOutput::decode` gains a per-sequence
  tokens-this-step count; the step loop appends a variable number of tokens per
  sequence and the sampler verifies a window; cache rollback on rejection is
  `truncate` (paged §7.3). §6.5 flags the `SchedulerOutput` extension.
- **M16 (observability).** The scheduler and pool already hold the counters M16
  exports (queue depth, running-batch size histogram, preemption count,
  `BlockPoolStats`); M16 reads them from the engine thread and exposes `/metrics`.

---

## 13. Testing strategy (per ticket)

The correctness spine is the **continuous-batching invariant** (batched output ==
sequential output), reducible at every layer to the KV invariant and the batch-
invariance of the kernels (§8.5). Per ticket:

- **M9-T02 (`request.h`, `channel`).** State-machine unit tests: every legal
  transition succeeds, every illegal one `CHECK`-fails (death tests). Channel:
  ordered delivery across a producer thread and a consumer thread; blocking `Next`
  wakes on push and on close; `TryNext` polls; close-once under a finish+late-cancel
  race; buffered items drain after close.
- **M9-T03 (`engine` API + queue).** Mock `Model` (returns canned logits). Submit /
  consume / cancel from multiple client threads; `Step()`-driven determinism; a
  stress test (many submitters, random cancels, concurrent consumers) with no
  deadlock and every channel closed. No real forward yet.
- **M9-T04 (`scheduler`).** Table-driven, pure: admission blocked when blocks
  insufficient; token budget respected across mixed prefill sizes; decode always
  scheduled (starvation impossible); preemption picks the latest-arrived; FCFS
  order preserved; `max_num_seqs` capped. No engine/pool/model.
- **M9-T05 (`batch.h`).** Hand-verified assembled metadata for {2 prefills of
  different lengths}, {3 decodes}, {mixed}: `token_ids`/`positions`/`cu_seqlens`/
  `block_table` contents exact (`EXPECT_EQ` on every element); no per-step
  allocation after warm-up (staging high-water stable).
- **M9-T06 (`PrefillAttentionVarlenF32`).** Batch {lengths 5/64/129} outputs match
  three standalone `PrefillAttentionF32` runs **exactly** (bit-identical per
  sequence); thread-count invariance; `SCALAR_PASS`.
- **M9-T07 (batched decode).** Batched decode logits **bit-identical** to sequential
  single-sequence decode per member; heterogeneous cache lengths covered; the
  `[B, max_blocks]` block-table padding (`−1`) handled; `SCALAR_PASS`.
- **M9-T08 (loop integration).** The milestone's headline test: 8 concurrent greedy
  requests produce output **identical** to 8 sequential `Generate` runs (both
  fixtures). Mid-flight arrivals (staggered submission) join without perturbing
  running sequences (their outputs unchanged vs a run without the newcomer).
  Throughput ratio (< 8×) recorded.
- **M9-T09 (preemption).** Artificially tiny pool forces preemptions; all requests
  complete correctly; preempted-request output identical to an unpreempted run
  (greedy); `pool.stats().used == 0` at end (no leaks).
- **M9-T10 (cancel & isolation).** Cancel during WAITING / prefill / decode each
  reclaims blocks (stats-verified); an injected per-request sampler fault fails only
  that request, others' outputs unchanged.

Suites touching a kernel or a full forward register `SCALAR_PASS` so the forced-
scalar pass (`ENGINE_FORCE_ISA=scalar`) covers the shipped bytes on both CI
(x86-64) and the arm64 dev machine (CLAUDE.md).

---

## 14. Deferred / open items (recorded)

- **Lock-free MPSC submission queue and output channel** — a measured-perf
  follow-up only if a profile shows the locks matter against the forward (§4, §5.2).
- **Bounded output channel + backpressure** — M10 (`server.md`), where the socket
  write rate is visible.
- **Chunked prefill / scheduler v2 / `--max-num-batched-tokens` as the binding
  budget** — M12-T06; v1 rejects over-budget prompts at submit (§6.4).
- **Prefix-reuse admission** — M11-T04 (§6.5 hook).
- **Flash-decoding batched kernel** — M12-T03 (threading-only change to §8.4).
- **Variable-tokens-per-step (speculative)** — M15-T06 (§6.5, §12).
- **Priority / fairness beyond FCFS, head-of-line-blocking mitigation** — future;
  v1 is strict FCFS with preempted-at-head (§6.2).
- **`BatchedSampler` per-row status return** — an M9-T10 additive change to the
  M7-T06 interface (§11.2), flagged so it is a planned refinement.
- **Graceful drain-on-shutdown** — M10's `serve`; the runtime `Stop` is abort-and-
  close (§5.2).
