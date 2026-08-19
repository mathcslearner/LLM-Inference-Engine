#pragma once

#include <cstdint>
#include <span>
#include <vector>

// Scheduler v1 (M9-T04; design: docs/design/scheduler-runtime.md §6). The pure
// decision component of the continuous-batching runtime: given the waiting
// queue, the running set, a snapshot of the block pool, and the token budget,
// it emits a `SchedulerOutput` naming which sequences to prefill (with their
// lengths), which to decode one token, and which to preempt this step.
//
// Purity is the point (§2, §6.1). The scheduler is a `core`-adjacent leaf that
// links **only** `core` (in fact only the standard library — no `core` symbol
// is needed) and depends on no engine, model, pool, allocator, or tensor type.
// It consumes plain descriptor structs the runtime fills each step and returns
// plain ids and lengths, so the whole policy is table-testable with no
// fixtures (ADR-002 rule 4: scheduler/sampling logic stay CPU-CI-testable
// forever). The block arithmetic is reproduced from `block_size` alone
// (`BlocksNeeded`), matching `kvcache::BlockPool::blocks_needed` bit-for-bit so
// the runtime and the scheduler agree on demand.
//
// This header intentionally redeclares `RequestId` (rather than including
// `runtime/request.h`, which lives *above* the scheduler): the runtime
// `static_assert`s the two aliases agree.

namespace engine::scheduler {

// Engine-assigned, monotonic request identifier. Kept in sync with
// `engine::runtime::RequestId` (asserted where the runtime bridges the two).
using RequestId = std::uint64_t;

// Physical-block budget the policy schedules against (design §6.1). A snapshot
// the runtime reads once per step from `BlockPool::free_blocks()` /
// `num_blocks()` / `block_size()`; the scheduler is the sole allocator's
// planner, so this snapshot is exact for the step it plans.
struct PoolSnapshot {
  std::int64_t free_blocks = 0;   // blocks available to allocate this step
  std::int64_t total_blocks = 0;  // pool capacity (advisory; not used by v1)
  int block_size = 0;             // tokens per block (`bs`); must be > 0
};

// One descriptor per RUNNING or WAITING sequence (design §6.1). The runtime
// fills these from its own per-sequence state and the sequence's cache each
// step; the scheduler never sees a `Sequence`, `PagedKvCache`, or `BlockPool`.
struct SeqDesc {
  RequestId id = 0;
  std::uint64_t arrival_index = 0;  // FCFS key (submission order)
  std::int64_t num_computed_tokens =
      0;                               // tokens committed to cache (== length)
  std::int64_t num_prompt_tokens = 0;  // WAITING: this step's prefill length
                                       // (prompt, or prompt+generated on
                                       // resume of a preempted sequence)
  std::int64_t blocks_held = 0;        // physical blocks currently owned
};

// Batch-width and per-step token budget (design §6.1). The two `EngineConfig`
// knobs the policy reads; `max_model_len` is a submit-time bound, not a
// scheduling one, so it is not here.
struct SchedulerConfig {
  int max_num_seqs = 256;                      // RUNNING batch-width cap (> 0)
  std::int64_t max_num_batched_tokens = 2048;  // per-step prefill budget (> 0)
};

// The scheduler's per-step inputs (design §6.1). `waiting` is in FCFS order
// (preempted sequences already moved to the head by the runtime); `running` is
// the current decode candidate set (any order — the policy sorts by
// `arrival_index` when choosing preemption victims).
struct ScheduleInputs {
  std::span<const SeqDesc> waiting;
  std::span<const SeqDesc> running;
  PoolSnapshot pool;
  SchedulerConfig config;
};

// The scheduler's decision (design §6.1). `prefill` carries an explicit
// per-sequence `num_tokens` (not "the whole prompt") so the M11 prefix-reuse
// and M12-T06 chunked-prefill seams need no shape change. `preempt` is emitted
// in eviction order (latest-arrived first); the runtime requeues each at the
// head of `waiting`, which lands them in arrival order among themselves.
struct SchedulerOutput {
  struct Prefill {
    RequestId id = 0;
    std::int64_t num_tokens = 0;
  };
  std::vector<Prefill> prefill;    // waiting seqs admitted this step
  std::vector<RequestId> decode;   // running seqs to advance one token
  std::vector<RequestId> preempt;  // running seqs to evict this step
};

// Blocks a sequence at `cur_tokens` must additionally allocate to hold
// `add_tokens` more, given block size `bs`. Reproduces
// `kvcache::BlockPool::blocks_needed` from `bs` alone (design §6.2). Public so
// the runtime and tests can cross-check it against the pool's own arithmetic.
[[nodiscard]] constexpr std::int64_t BlocksNeeded(std::int64_t cur_tokens,
                                                  std::int64_t add_tokens,
                                                  int bs) {
  const std::int64_t b = bs;
  const std::int64_t before = (cur_tokens + b - 1) / b;
  const std::int64_t after = (cur_tokens + add_tokens + b - 1) / b;
  return after - before;
}

// The scheduling policy (design §6.2). Stateless in v1 — a class so the M11
// prefix-adopt hook (§6.5) and later versions have a home without changing the
// call sites. Deterministic and pure: identical inputs → identical output.
class Scheduler {
 public:
  Scheduler() = default;

  // Plan one step. `CHECK`-fails on programmer-error inputs (non-positive
  // `block_size`/`max_num_seqs`/`max_num_batched_tokens`, negative counts).
  [[nodiscard]] SchedulerOutput Schedule(const ScheduleInputs& inputs) const;
};

}  // namespace engine::scheduler
