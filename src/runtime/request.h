#pragma once

#include "core/status.h"
#include "engine/stop.h"
#include "kvcache/block_pool.h"
#include "kvcache/paged_cache.h"
#include "sampling/params.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Request & sequence abstractions (M9-T02; design:
// docs/design/scheduler-runtime.md §3). The runtime turns the M5-M8
// single-request `engine::Generate` into a multi-request, continuously-batched
// system; this header holds the two per-request objects the step loop
// (M9-T08) fans out over:
//
//   * `Request` — the immutable, client-supplied description (id, prompt ids,
//     `SamplingParams`, model EOS set, tokenizer, arrival order). It stays free
//     of engine internals so the M9-T03 public API can expose it directly.
//   * `Sequence` — the engine's mutable per-request execution state: the state
//     machine (§3.2), the generated ids, this sequence's own paged KV cache,
//     and the per-sequence `Sampler`/`StopChecker` the single-sequence
//     `Generate` built once and carried across the decode loop (lifted onto the
//     sequence so the batched loop is a fan-out of that loop).
//
// One `Sequence` per `Request` (`n == 1`); the two are separate types so a
// future `n > 1` adds sequences without changing the request surface.
//
// The per-request output channel (`OutputChannel`) lives in `channel.h`; a
// `Sequence` owns a `shared_ptr` to it (forward-declared here, so the channel's
// complete type is needed only in `sequence.cpp`).
//
// Layering (ADR-002): `runtime` is a layer-4 orchestration module and links
// every module named here (`engine`, `sampling`, `kvcache`, `tokenizer`,
// `core`) — all already below it on the diagram. No new ADR edge.

namespace engine::runtime {

// The single-sequence finish taxonomy from the M7-T04 stop machinery, reused
// verbatim (design §3.3). Leading `::` is required: from inside
// `engine::runtime` an unqualified `engine` resolves to the sibling
// `engine::engine` namespace, so `engine::engine::X` would misresolve (the
// namespace footgun the design's §2 note records).
using ::engine::engine::FinishReason;
using ::engine::engine::StopChecker;
using ::engine::engine::StopTrigger;

// Engine-assigned, monotonic request identifier.
using RequestId = std::uint64_t;

// A client's immutable request description (design §3.1). Prompt and params are
// validated at submit (M9-T03); `Sequence::Create` re-validates via
// `Sampler::Create`/`StopChecker::Create`, so a malformed request fails at
// sequence construction rather than mid-forward.
struct Request {
  RequestId id = 0;
  std::vector<std::int32_t> prompt_ids;  // non-empty (validated at submit)
  sampling::SamplingParams params;       // validated at submit
  std::vector<std::int32_t> eos_ids;     // model-derived stop ids
  const tokenizer::Tokenizer* tokenizer = nullptr;  // for text / stop-strings
  bool skip_special_tokens = true;
  std::uint64_t arrival_index = 0;  // FCFS order key (submit order)
};

// The lifecycle state of a `Sequence` (design §3.2). `kFinished`, `kCancelled`,
// and `kFailed` are terminal.
enum class SeqState : std::uint8_t {
  kWaiting,    // admitted to the engine, not yet prefilled
  kRunning,    // prefilled; being decoded
  kPreempted,  // evicted (cache freed) to relieve pool pressure; awaiting
               // resume
  kFinished,   // a stop condition fired (terminal)
  kCancelled,  // client cancelled (terminal)
  kFailed,     // a per-request fault (terminal)
};

// Human-readable state name (logs / tests / M10 diagnostics).
[[nodiscard]] std::string_view SeqStateName(SeqState state);

// The transition table of §3.2, as a pure predicate — `true` iff `from -> to`
// is a legal edge. Exposed so tests can enumerate the full state matrix and so
// `Sequence::Transition` enforces the table in one place. The initial state is
// `kWaiting` (the "(none) -> WAITING" row is construction, not a transition).
[[nodiscard]] constexpr bool IsLegalTransition(SeqState from, SeqState to) {
  switch (from) {
    case SeqState::kWaiting:
      // admit (prefill), or cancel before admission.
      return to == SeqState::kRunning || to == SeqState::kCancelled;
    case SeqState::kRunning:
      // decode self-loop, finish, preempt, cancel, or per-request fault.
      return to == SeqState::kRunning || to == SeqState::kFinished ||
             to == SeqState::kPreempted || to == SeqState::kCancelled ||
             to == SeqState::kFailed;
    case SeqState::kPreempted:
      // re-admit (re-prefill), or cancel while waiting to resume.
      return to == SeqState::kRunning || to == SeqState::kCancelled;
    case SeqState::kFinished:
    case SeqState::kCancelled:
    case SeqState::kFailed:
      return false;  // terminal — no outgoing edges.
  }
  return false;
}

// The finish descriptor a `Sequence`'s channel is closed with (design §3.3).
// `FinishReason`/`StopTrigger` are reused verbatim from `engine/stop.h` (the
// M7-T04 single-sequence finish taxonomy); `kCancelled`/`kFailed` are the two
// runtime-only terminal states `Generate` did not have.
struct FinishInfo {
  SeqState terminal = SeqState::kFinished;  // kFinished / kCancelled / kFailed
  FinishReason finish_reason = FinishReason::kLength;
  StopTrigger stop_trigger = StopTrigger::kNone;
  std::string matched_stop;  // the matched stop string, if any
  core::Status error;        // ok unless `terminal == kFailed`
};

class OutputChannel;  // channel.h

// A request's mutable execution state (design §3.1). Move-only; built by
// `Create`, which resolves the sampler seed once and front-loads sampler/stop
// validation. The step loop (M9-T08) drives it via `Transition` (the one
// CHECK-enforced state gate), `AppendGenerated`, and the progress mutators.
class Sequence {
 public:
  // Build the sequence for `request`, borrowing the engine's shared `pool` for
  // this sequence's KV cache. Returns `InvalidArgument` if the request's params
  // fail validation, or if `stop_strings` are set without a tokenizer (the
  // `Sampler`/`StopChecker` contracts). `pool` is non-owning and must outlive
  // the sequence. The channel is created open; the sequence starts `kWaiting`
  // with an empty (zero-block) cache.
  [[nodiscard]] static core::StatusOr<Sequence> Create(
      const Request& request, kvcache::BlockPool* pool);

  Sequence(Sequence&&) noexcept;
  Sequence& operator=(Sequence&&) = delete;
  Sequence(const Sequence&) = delete;
  Sequence& operator=(const Sequence&) = delete;
  ~Sequence();

  // --- state machine -------------------------------------------------------

  [[nodiscard]] SeqState state() const { return state_; }

  // Move to `next`, CHECK-failing on any edge not in the §3.2 table (M9-T02
  // acceptance: "illegal transition = CHECK failure"). The single place the
  // invariant is enforced.
  void Transition(SeqState next);

  [[nodiscard]] bool is_terminal() const {
    return state_ == SeqState::kFinished || state_ == SeqState::kCancelled ||
           state_ == SeqState::kFailed;
  }

  // --- request & progress accessors ---------------------------------------

  [[nodiscard]] const Request& request() const { return *request_; }
  [[nodiscard]] RequestId id() const { return request_->id; }

  [[nodiscard]] const std::vector<std::int32_t>& generated_ids() const {
    return generated_ids_;
  }
  [[nodiscard]] std::int64_t num_generated() const {
    return static_cast<std::int64_t>(generated_ids_.size());
  }
  // Positions committed to the cache (`== cache().length()` after a forward).
  [[nodiscard]] std::int64_t num_computed_tokens() const {
    return num_computed_tokens_;
  }

  [[nodiscard]] sampling::Sampler& sampler() { return sampler_; }
  [[nodiscard]] StopChecker& stop() { return stop_; }
  // The sequence's KV cache (null only between `ReleaseCache` and the next
  // `EnsureCache`, i.e. while preempted).
  [[nodiscard]] kvcache::PagedKvCache* cache() { return cache_.get(); }
  [[nodiscard]] const std::shared_ptr<OutputChannel>& channel() const {
    return channel_;
  }

  // --- mutators (driven by the step loop) ---------------------------------

  // Record a sampled token and advance the committed-position count by one.
  void AppendGenerated(std::int32_t id) {
    generated_ids_.push_back(id);
    ++num_computed_tokens_;
  }
  // Set the committed-position count directly (prefill sets it to
  // `prompt_len`).
  void set_num_computed_tokens(std::int64_t n) { num_computed_tokens_ = n; }

  // Preemption (M9-T09): drop the cache — returning every block to the pool
  // (RAII) — and reset the committed-position count. `EnsureCache` rebuilds an
  // empty cache before re-prefill.
  void ReleaseCache() {
    cache_.reset();
    num_computed_tokens_ = 0;
  }
  void EnsureCache(kvcache::BlockPool* pool) {
    if (cache_ == nullptr) {
      cache_ = std::make_unique<kvcache::PagedKvCache>(pool);
    }
  }

  // Record why the sequence finished (set before the terminal `Transition`).
  void set_finish(FinishReason reason, StopTrigger trigger,
                  std::string matched_stop) {
    finish_reason_ = reason;
    stop_trigger_ = trigger;
    matched_stop_ = std::move(matched_stop);
  }
  void set_error(core::Status error) { error_ = std::move(error); }

  // Assemble the channel-close descriptor from the current terminal state.
  // CHECK-fails if the sequence is not in a terminal state.
  [[nodiscard]] FinishInfo MakeFinishInfo() const;

 private:
  Sequence(const Request& request, std::unique_ptr<kvcache::PagedKvCache> cache,
           sampling::Sampler sampler, StopChecker stop,
           std::shared_ptr<OutputChannel> channel);

  SeqState state_ = SeqState::kWaiting;
  const Request* request_;  // borrowed; owned by the engine's request map
  std::vector<std::int32_t> generated_ids_;
  std::unique_ptr<kvcache::PagedKvCache> cache_;  // this sequence's KV
  sampling::Sampler sampler_;                     // built from params
  StopChecker stop_;                        // per-request stop bookkeeping
  std::shared_ptr<OutputChannel> channel_;  // outlives the sequence
  std::int64_t num_computed_tokens_ = 0;
  FinishReason finish_reason_ = FinishReason::kLength;
  StopTrigger stop_trigger_ = StopTrigger::kNone;
  std::string matched_stop_;
  core::Status error_;  // set on kFailed
};

}  // namespace engine::runtime
