#include "scheduler/scheduler.h"

#include "core/check.h"

#include <algorithm>
#include <cstdint>

// Scheduler v1 policy implementation (design §6.2). Applied in three ordered
// passes each step: decode-first (no starvation), preempt-to-fit (latest-
// arrived victim), then FCFS admission (order-preserving, head-of-line
// blocking accepted). See scheduler.h for the input/output contract.

namespace engine::scheduler {

// `Schedule` reads no member state in v1 (the policy is a pure function of its
// inputs), but it is intentionally an instance method: the M11 prefix-reuse
// hook (design §6.5) adds per-scheduler state, and callers already hold a
// `Scheduler` instance. Kept non-static so that evolution needs no call-site
// churn.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
SchedulerOutput Scheduler::Schedule(const ScheduleInputs& inputs) const {
  const int bs = inputs.pool.block_size;
  CHECK(bs > 0, "Scheduler: block_size must be > 0 (got {})", bs);
  CHECK(inputs.config.max_num_seqs > 0,
        "Scheduler: max_num_seqs must be > 0 (got {})",
        inputs.config.max_num_seqs);
  CHECK(inputs.config.max_num_batched_tokens > 0,
        "Scheduler: max_num_batched_tokens must be > 0 (got {})",
        inputs.config.max_num_batched_tokens);

  SchedulerOutput out;

  // --- Pass 1: decode-first (design §6.2 step 1) --------------------------
  // Every RUNNING sequence is scheduled to decode one token; this is
  // unconditional up to memory, so decode starvation is impossible. Track the
  // decode set as a mutable working list (preemption removes from it) and the
  // notional free-block count as demand is charged/refunded.
  //
  // `demand` is the total blocks the decode set needs *beyond* what its members
  // already hold — a decode boundary crossing costs one block. A sequence is
  // kept in the decode set as long as it fits; the ones that don't are peeled
  // off latest-arrived-first below.
  std::vector<SeqDesc> decoding(inputs.running.begin(), inputs.running.end());
  std::int64_t free_blocks = inputs.pool.free_blocks;
  CHECK(free_blocks >= 0, "Scheduler: free_blocks must be >= 0 (got {})",
        free_blocks);

  auto decode_need = [bs](const SeqDesc& s) -> std::int64_t {
    CHECK(s.num_computed_tokens >= 0,
          "Scheduler: num_computed_tokens must be >= 0 (got {})",
          s.num_computed_tokens);
    CHECK(s.blocks_held >= 0, "Scheduler: blocks_held must be >= 0 (got {})",
          s.blocks_held);
    return BlocksNeeded(s.num_computed_tokens, 1, bs);
  };

  std::int64_t demand = 0;
  for (const SeqDesc& s : decoding) {
    demand += decode_need(s);
  }

  // --- Pass 2: preempt-to-fit (design §6.2 step 2) ------------------------
  // While the decode set's block demand exceeds the free pool, preempt the
  // latest-arrived RUNNING sequence (highest `arrival_index`; ties broken by
  // later position in `running`, i.e. the last such element). Move it to
  // `preempt`, drop it from the decode set, refund its held blocks, and reduce
  // demand by its decode need. Repeat until the remainder fits or the set is
  // empty (the oldest-alone liveness guarantee is a config-sizing property,
  // §10.3 — the scheduler still evicts a lone non-fitting sequence).
  while (demand > free_blocks && !decoding.empty()) {
    // Latest-arrived == max arrival_index; on ties pick the last one so the
    // choice is the later-positioned of equal arrivals (deterministic).
    std::size_t victim = 0;
    for (std::size_t i = 1; i < decoding.size(); ++i) {
      if (decoding[i].arrival_index >= decoding[victim].arrival_index) {
        victim = i;
      }
    }
    const SeqDesc& v = decoding[victim];
    out.preempt.push_back(v.id);
    demand -= decode_need(v);
    free_blocks += v.blocks_held;  // freed this step, available to admit below
    decoding.erase(decoding.begin() +
                   static_cast<std::vector<SeqDesc>::difference_type>(victim));
  }

  // Charge the surviving decode set's demand against the free pool; the
  // remaining headroom is what admission may consume.
  free_blocks -= demand;

  out.decode.reserve(decoding.size());
  for (const SeqDesc& s : decoding) {
    out.decode.push_back(s.id);
  }

  // --- Pass 3: FCFS admission (design §6.2 step 3) ------------------------
  // Walk `waiting` in arrival order (preempted sequences already at the head).
  // Admit while the batch-width, per-step token budget, and block-availability
  // checks all hold; stop at the first sequence that fails any check — do NOT
  // skip it to admit a smaller one behind it (FCFS is order-preserving;
  // head-of-line blocking is the accepted v1 cost).
  auto num_scheduled = static_cast<std::int64_t>(decoding.size());
  std::int64_t tokens_so_far = 0;
  for (const SeqDesc& s : inputs.waiting) {
    CHECK(s.num_prompt_tokens >= 0,
          "Scheduler: num_prompt_tokens must be >= 0 (got {})",
          s.num_prompt_tokens);
    if (num_scheduled >= inputs.config.max_num_seqs) {
      break;  // batch-width cap reached
    }
    if (tokens_so_far + s.num_prompt_tokens >
        inputs.config.max_num_batched_tokens) {
      break;  // per-step token budget exhausted
    }
    const std::int64_t need = BlocksNeeded(0, s.num_prompt_tokens, bs);
    if (need > free_blocks) {
      break;  // insufficient blocks for this prompt
    }
    out.prefill.push_back({.id = s.id, .num_tokens = s.num_prompt_tokens});
    tokens_so_far += s.num_prompt_tokens;
    free_blocks -= need;
    ++num_scheduled;
  }

  return out;
}

}  // namespace engine::scheduler
