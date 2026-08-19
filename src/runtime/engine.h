#pragma once

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "model/model.h"
#include "runtime/channel.h"
#include "runtime/request.h"
#include "scheduler/scheduler.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

// Engine API & request queue (M9-T03; design: docs/design/scheduler-runtime.md
// §5). The runtime's public async surface: clients `Submit` requests from any
// thread and receive a `RequestHandle` streaming tokens back through the
// per-request `OutputChannel`; a single engine thread drives `Step()`, which
// drains submissions/cancellations at each boundary, admits waiting sequences,
// runs the forward(s), samples one token per sequence, and retires finished /
// cancelled / failed sequences.
//
// What lands in T03: the API, the mutex+condvar submission queue, the loop
// thread scaffold (`Start`/`Stop`), and a `Step()` whose *shape* is the full
// §9.1 loop with two labelled placeholders the later tickets substitute in
// place, not restructure:
//
//   * admission is now the `scheduler::Scheduler` decision component (M9-T04):
//     `ScheduleStep()` builds the plain descriptor inputs (§6.1) from the
//     engine-thread state, and `Step()` applies the resulting prefill / decode
//     / preempt decisions (§6.2). Preemption is evict-and-recompute
//     (`ReleaseCache` → re-prefill `prompt ++ generated` on re-admission, the
//     §10.2 resumable seam); the batched loop that validates it under memory
//     pressure is M9-T09.
//   * execution runs one `model::forward` **per sequence** over the existing
//     single-sequence `ForwardRequest` — the body of `engine::Generate`'s
//     decode loop, lifted onto the `Sequence` — so a request's output equals a
//     standalone `Generate` run today. The batched prefill/decode passes
//     (M9-T05…T08) replace the per-sequence execution, preserving that output.
//
// Layering (ADR-002): `runtime` is the layer-4 orchestration module; this
// header names `model::Model` (the forward contract) and
// `kvcache::BlockPool` (the shared block store), both already below `runtime`
// on the diagram — no new ADR edge.

namespace engine::runtime {

// Runtime configuration (design §5, §6). All three are validated at `Create`.
struct EngineConfig {
  // Maximum number of RUNNING sequences batched at once (§6.2 batch-width cap).
  int max_num_seqs = 256;
  // Per-step prefill-token budget (§6.2). A prompt longer than this can never
  // be admitted (no chunked prefill until M12-T06), so `Submit` rejects it
  // up front (§6.4).
  std::int64_t max_num_batched_tokens = 2048;
  // Absolute cap on a sequence's total length (prompt + generated). `0` →
  // `model.config().max_position_embeddings`. `Submit` also bounds a request
  // by the pool's token capacity (`num_blocks · block_size`).
  std::int64_t max_model_len = 0;
};

// A client's view of a submitted request (design §5.1). Copyable — it holds a
// `shared_ptr` to the sequence's `OutputChannel`, which outlives the engine's
// `Sequence` (the engine retires the sequence as soon as it closes the channel,
// but the client can still drain buffered tokens). Consumption is single-reader
// per handle.
class RequestHandle {
 public:
  RequestHandle() = default;  // an empty, invalid handle

  [[nodiscard]] RequestId id() const { return id_; }
  [[nodiscard]] bool valid() const { return channel_ != nullptr; }

  // Block until the next token is available or the stream ends; `nullopt` once
  // the channel is closed and drained (`await_completion` for the reason).
  [[nodiscard]] std::optional<OutputItem> next_token();
  // Non-blocking poll (the M10 SSE disconnect-interleave seam).
  [[nodiscard]] std::optional<OutputItem> try_next_token();
  // Block until the request reaches a terminal state; returns its `FinishInfo`.
  // Does not consume tokens — `next_token` still drains any buffered items.
  [[nodiscard]] FinishInfo await_completion();

 private:
  friend class Engine;
  RequestHandle(RequestId id, std::shared_ptr<OutputChannel> channel)
      : id_(id), channel_(std::move(channel)) {}

  RequestId id_ = 0;
  std::shared_ptr<OutputChannel> channel_;
};

// The multi-request, continuously-batched runtime (design §5). Non-movable and
// non-copyable: it owns a `std::thread`, a mutex/condvar, and a map of
// heap-pinned sequences whose `Sequence::request_` borrows into that map, so
// its address must stay stable. Construct via `Create` (returns a
// `unique_ptr`).
class Engine {
 public:
  // Build an engine over `model` and the shared block `pool`. `model` and
  // `pool` are borrowed and must outlive the engine. Fails `InvalidArgument`
  // if the config is non-positive or the pool's geometry does not match the
  // model's `cache_geometry()`.
  [[nodiscard]] static core::StatusOr<std::unique_ptr<Engine>> Create(
      model::Model& model, kvcache::BlockPool& pool, EngineConfig config);

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;
  ~Engine();

  // --- client-thread API (any thread) -------------------------------------

  // Validate and enqueue `request`, returning a handle to stream its output.
  // Front-loaded rejections (nothing enqueued, no sequence/cache/blocks
  // allocated), same status codes M10 maps to HTTP once (§5.1):
  //   * empty `prompt_ids` → InvalidArgument.
  //   * an invalid / not-yet-supported `params`, or `stop_strings` without a
  //     tokenizer → InvalidArgument (from `Sequence::Create`).
  //   * `prompt_len > max_num_batched_tokens` → InvalidArgument (the v1
  //     no-chunked-prefill ceiling, §6.4).
  //   * `prompt_len + max_tokens - 1` above `max_model_len` or the pool's token
  //     capacity → ResourceExhausted.
  //   * after `Stop` → FailedPrecondition.
  // On success `request.id` and `arrival_index` are engine-assigned (monotonic,
  // in submission order).
  [[nodiscard]] core::StatusOr<RequestHandle> Submit(Request request);

  // Request cancellation of `id`. Idempotent and non-blocking: it enqueues a
  // command applied at the next step boundary (≤ one step latency, §11.1); an
  // unknown or already-terminal id is a no-op.
  void Cancel(RequestId id);

  // --- loop control -------------------------------------------------------

  // Spawn the engine thread, which drives `Step()` to quiescence and then
  // idle-waits on the submission queue. CHECK-fails if already started.
  void Start();
  // Signal shutdown, join the engine thread, and close every still-open
  // channel with `kCancelled` (abort-and-close; a graceful drain is M10's
  // `serve`, §5.2). Idempotent. No blocks leak — every live sequence is
  // dropped, returning its blocks.
  void Stop();
  // Run one scheduling+execution step: drain commands, retire cancellations,
  // admit, execute, deliver, retire terminals. Returns `false` when there is
  // no work left (queue empty, no waiting/running sequences). Exposed so tests
  // drive the loop deterministically without the thread (§5.1).
  [[nodiscard]] bool Step();

  // --- diagnostics (engine-thread state; call only when the loop thread is not
  //     running — e.g. between manual `Step()` calls in single-threaded tests)
  [[nodiscard]] std::size_t num_waiting() const { return waiting_.size(); }
  [[nodiscard]] std::size_t num_running() const { return running_.size(); }
  [[nodiscard]] std::size_t num_live() const { return requests_.size(); }

 private:
  Engine(model::Model& model, kvcache::BlockPool& pool, EngineConfig config,
         std::int64_t max_model_len);

  // The engine's ownership record for one request: the immutable `Request`
  // (borrowed by the `Sequence`) and the mutable `Sequence`, heap-pinned behind
  // a `unique_ptr` so the `Sequence`'s back-pointer into `request` stays valid.
  struct Entry {
    explicit Entry(Request req) : request(std::move(req)) {}
    Request request;
    std::unique_ptr<Sequence> seq;  // set once `Sequence::Create` succeeds
    bool cancel_requested = false;
  };

  // A queued client command, applied at the next step boundary.
  struct SubmitCmd {
    std::unique_ptr<Entry> entry;
  };
  struct CancelCmd {
    RequestId id = 0;
  };

  // --- engine-thread internals (design §9.1) ------------------------------
  void Run();              // the loop-thread body
  void DrainCommands();    // step 1
  void RetireCancelled();  // step 2 (all states)
  [[nodiscard]] scheduler::SchedulerOutput ScheduleStep();  // step 3
  void ExecuteAndDeliver(Entry& entry, bool prefill);       // steps 5/6 per seq
  void RetireTerminal();                                    // step 7
  static void CloseCancelled(Entry& entry);  // transition + close a live seq
  void Shutdown();                           // post-join teardown

  model::Model& model_;
  kvcache::BlockPool& pool_;
  EngineConfig config_;
  std::int64_t max_model_len_ = 0;  // resolved at Create

  // Submission queue (client threads ↔ engine thread).
  mutable std::mutex cmd_mu_;
  std::condition_variable cmd_cv_;
  std::vector<std::variant<SubmitCmd, CancelCmd>> commands_;  // guarded
  RequestId next_id_ = 1;                                     // guarded
  std::uint64_t next_arrival_ = 0;                            // guarded
  bool stop_requested_ = false;                               // guarded

  // Engine-thread-owned state (no lock — single mutator, §5.3).
  std::unordered_map<RequestId, std::unique_ptr<Entry>> requests_;
  std::deque<RequestId> waiting_;   // FCFS order (preempted at head, §3.2)
  std::vector<RequestId> running_;  // sequences being decoded

  std::thread thread_;
};

}  // namespace engine::runtime
