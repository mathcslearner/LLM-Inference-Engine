#include "scheduler/scheduler.h"

#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "tensor/dtype.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

// Scheduler v1 tests (M9-T04; design: docs/design/scheduler-runtime.md §6). The
// pure decision component: table-driven over plain descriptor structs, no
// engine / model / pool / allocator on the hot path (one test cross-checks
// `BlocksNeeded` against a real `BlockPool::blocks_needed`, the only fixture).
// Verifies the five §13 acceptance properties — admission blocked when blocks
// insufficient, token budget respected across mixed prefill sizes, decode
// starvation impossible, latest-arrived preemption, FCFS order preserved — plus
// the `max_num_seqs` cap, determinism, and the output invariants.

namespace {

using engine::scheduler::BlocksNeeded;
using engine::scheduler::RequestId;
using engine::scheduler::ScheduleInputs;
using engine::scheduler::Scheduler;
using engine::scheduler::SchedulerOutput;
using engine::scheduler::SeqDesc;

// --- descriptor builders ----------------------------------------------------

// A WAITING sequence: `prefill` tokens to admit at arrival order `arrival`.
[[nodiscard]] SeqDesc Waiting(RequestId id, std::uint64_t arrival,
                              std::int64_t prefill) {
  return {.id = id,
          .arrival_index = arrival,
          .num_computed_tokens = 0,
          .num_prompt_tokens = prefill,
          .blocks_held = 0};
}

// A RUNNING sequence: `computed` tokens committed, holding `held` blocks.
[[nodiscard]] SeqDesc Running(RequestId id, std::uint64_t arrival,
                              std::int64_t computed, std::int64_t held) {
  return {.id = id,
          .arrival_index = arrival,
          .num_computed_tokens = computed,
          .num_prompt_tokens = 0,
          .blocks_held = held};
}

[[nodiscard]] ScheduleInputs Inputs(const std::vector<SeqDesc>& waiting,
                                    const std::vector<SeqDesc>& running,
                                    std::int64_t free_blocks, int block_size,
                                    int max_num_seqs,
                                    std::int64_t max_num_batched_tokens) {
  return ScheduleInputs{
      .waiting = waiting,
      .running = running,
      .pool = {.free_blocks = free_blocks,
               .total_blocks = free_blocks,
               .block_size = block_size},
      .config = {.max_num_seqs = max_num_seqs,
                 .max_num_batched_tokens = max_num_batched_tokens}};
}

[[nodiscard]] std::vector<RequestId> PrefillIds(const SchedulerOutput& out) {
  std::vector<RequestId> ids;
  ids.reserve(out.prefill.size());
  for (const auto& p : out.prefill) {
    ids.push_back(p.id);
  }
  return ids;
}

// =============================== BlocksNeeded ===============================

TEST(BlocksNeededTest, HandTableBs8) {
  EXPECT_EQ(BlocksNeeded(0, 6, 8), 1);   // prefill 6 from empty
  EXPECT_EQ(BlocksNeeded(6, 1, 8), 0);   // decode within block 0
  EXPECT_EQ(BlocksNeeded(7, 1, 8), 0);   // fills block 0's last slot (pos 7)
  EXPECT_EQ(BlocksNeeded(8, 1, 8), 1);   // pos 8 crosses into block 1
  EXPECT_EQ(BlocksNeeded(0, 16, 8), 2);  // exactly two full blocks
  EXPECT_EQ(BlocksNeeded(0, 17, 8), 3);  // one token into a third
  EXPECT_EQ(BlocksNeeded(16, 0, 8), 0);  // no new tokens
  EXPECT_EQ(BlocksNeeded(0, 0, 8), 0);
}

// The scheduler reproduces `BlockPool::blocks_needed` from `block_size` alone;
// this cross-checks the two are bit-for-bit identical over a sweep (the only
// test that touches a real pool).
TEST(BlocksNeededTest, MatchesBlockPool) {
  using engine::kvcache::BlockPool;
  using engine::kvcache::CacheGeometry;
  for (const int bs : {8, 16, 32, 64}) {
    const CacheGeometry geom{.num_layers = 2,
                             .num_kv_heads = 2,
                             .head_dim = 16,
                             .dtype = engine::tensor::DataType::kFloat32};
    auto pool = BlockPool::Create(geom, bs, /*num_blocks=*/64, nullptr);
    ASSERT_TRUE(pool.ok());
    for (std::int64_t cur = 0; cur <= 200; ++cur) {
      for (std::int64_t add = 0; add <= 130; ++add) {
        EXPECT_EQ(BlocksNeeded(cur, add, bs), pool->blocks_needed(cur, add))
            << "bs=" << bs << " cur=" << cur << " add=" << add;
      }
    }
  }
}

// ================================ empty/basic ==============================

TEST(SchedulerTest, EmptyInputsEmptyOutput) {
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs({}, {}, 100, 16, 8, 2048));
  EXPECT_TRUE(out.prefill.empty());
  EXPECT_TRUE(out.decode.empty());
  EXPECT_TRUE(out.preempt.empty());
}

TEST(SchedulerTest, DecodeOnlyAllScheduledNoPreempt) {
  // Three running sequences, ample blocks, no waiting → all decode.
  const std::vector<SeqDesc> running = {
      Running(1, 0, /*computed=*/10, /*held=*/1), Running(2, 1, 20, 2),
      Running(3, 2, 30, 2)};
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs({}, running, 100, 16, 8, 2048));
  EXPECT_EQ(out.decode, (std::vector<RequestId>{1, 2, 3}));
  EXPECT_TRUE(out.prefill.empty());
  EXPECT_TRUE(out.preempt.empty());
}

TEST(SchedulerTest, PrefillOnlyFromEmptyRunning) {
  const std::vector<SeqDesc> waiting = {Waiting(1, 0, 30), Waiting(2, 1, 10)};
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs(waiting, {}, 100, 16, 8, 2048));
  EXPECT_EQ(PrefillIds(out), (std::vector<RequestId>{1, 2}));
  EXPECT_EQ(out.prefill[0].num_tokens, 30);
  EXPECT_EQ(out.prefill[1].num_tokens, 10);
  EXPECT_TRUE(out.decode.empty());
  EXPECT_TRUE(out.preempt.empty());
}

// ================= decode starvation impossible (§6.2 step 1) ==============

TEST(SchedulerTest, DecodeScheduledEvenWhenNoBlocksToAdmit) {
  // No free blocks at all: running sequences that need no new block still
  // decode; no waiting sequence is admitted. Decode is never starved to admit.
  const std::vector<SeqDesc> running = {
      Running(1, 0, /*computed=*/4, /*held=*/1),  // pos 4 → no boundary cross
      Running(2, 1, 5, 1)};
  const std::vector<SeqDesc> waiting = {Waiting(3, 2, 8)};
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, running, /*free_blocks=*/0, 8, 8, 2048));
  EXPECT_EQ(out.decode, (std::vector<RequestId>{1, 2}));
  EXPECT_TRUE(out.prefill.empty());  // nothing to admit — no free blocks
  EXPECT_TRUE(out.preempt.empty());  // decoders need no new block, so no evict
}

// =============== admission blocked on block availability (§6.2 step 3) ======

TEST(SchedulerTest, AdmissionBoundaryOnFreeBlocks) {
  // bs=8, prompt=16 needs exactly 2 blocks. free=2 admits; free=1 rejects.
  const std::vector<SeqDesc> waiting = {Waiting(1, 0, 16)};
  EXPECT_EQ(
      Scheduler{}.Schedule(Inputs(waiting, {}, 2, 8, 8, 2048)).prefill.size(),
      1U);
  EXPECT_TRUE(
      Scheduler{}.Schedule(Inputs(waiting, {}, 1, 8, 8, 2048)).prefill.empty());
}

TEST(SchedulerTest, DecodeDemandConsumesBlocksBeforeAdmission) {
  // A running decode at a block boundary needs one new block; with free=1 that
  // block is spent on the decode, leaving none for the waiting prompt.
  const std::vector<SeqDesc> running = {
      Running(1, 0, /*computed=*/8, /*held=*/1)};  // pos 8 crosses to block 1
  const std::vector<SeqDesc> waiting = {Waiting(2, 1, 8)};  // needs 1 block
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, running, /*free_blocks=*/1, 8, 8, 2048));
  EXPECT_EQ(out.decode, (std::vector<RequestId>{1}));
  EXPECT_TRUE(out.prefill.empty());  // the one free block went to the decode
  EXPECT_TRUE(out.preempt.empty());  // exactly one block, fits the decode
}

// ============== token budget across mixed prefill sizes (§6.2 step 3) =======

TEST(SchedulerTest, TokenBudgetRespectedHeadOfLineBlocking) {
  // Budget 100, prompts [40, 50, 30, 10]. Admits 40 then 50 (=90). The next is
  // 30 → 90+30=120 > 100, so admission STOPS — it does not skip 30 to admit the
  // 10 behind it (FCFS is order-preserving, head-of-line blocking accepted).
  const std::vector<SeqDesc> waiting = {Waiting(1, 0, 40), Waiting(2, 1, 50),
                                        Waiting(3, 2, 30), Waiting(4, 3, 10)};
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, {}, 1000, 8, 8, /*max_num_batched_tokens=*/100));
  EXPECT_EQ(PrefillIds(out), (std::vector<RequestId>{1, 2}));
}

TEST(SchedulerTest, TokenBudgetExactFit) {
  const std::vector<SeqDesc> waiting = {Waiting(1, 0, 60), Waiting(2, 1, 40)};
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, {}, 1000, 8, 8, /*max_num_batched_tokens=*/100));
  EXPECT_EQ(PrefillIds(out), (std::vector<RequestId>{1, 2}));  // 60+40 == 100
}

// ===================== max_num_seqs batch-width cap =========================

TEST(SchedulerTest, MaxNumSeqsCapsAdmissionCountingRunning) {
  // 2 running + cap 3 → at most 1 admission even though 3 wait and
  // blocks/tokens are ample.
  const std::vector<SeqDesc> running = {Running(1, 0, 4, 1),
                                        Running(2, 1, 4, 1)};
  const std::vector<SeqDesc> waiting = {Waiting(3, 2, 8), Waiting(4, 3, 8),
                                        Waiting(5, 4, 8)};
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, running, 1000, 8, /*max_num_seqs=*/3, 2048));
  EXPECT_EQ(out.decode.size(), 2U);
  EXPECT_EQ(PrefillIds(out), (std::vector<RequestId>{3}));
}

TEST(SchedulerTest, MaxNumSeqsAlreadyMetAdmitsNothing) {
  const std::vector<SeqDesc> running = {Running(1, 0, 4, 1),
                                        Running(2, 1, 4, 1)};
  const std::vector<SeqDesc> waiting = {Waiting(3, 2, 8)};
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, running, 1000, 8, /*max_num_seqs=*/2, 2048));
  EXPECT_TRUE(out.prefill.empty());
}

// ===================== latest-arrived preemption (§6.2 step 2) ==============

TEST(SchedulerTest, PreemptsLatestArrivedToFitDecode) {
  // Three running, each at a block boundary → each decode needs one new block.
  // free=2 fits only two decodes; the latest-arrived (id 3, arrival 2) is
  // preempted. Its held block is refunded but not enough to admit anything.
  const std::vector<SeqDesc> running = {
      Running(1, 0, 8, 1), Running(2, 1, 8, 1), Running(3, 2, 8, 1)};
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs({}, running, /*free_blocks=*/2, 8, 8, 2048));
  EXPECT_EQ(out.preempt, (std::vector<RequestId>{3}));
  EXPECT_EQ(out.decode, (std::vector<RequestId>{1, 2}));
}

TEST(SchedulerTest, PreemptionVictimIsMaxArrivalNotPosition) {
  // running order is not arrival order: the victim is the max arrival_index
  // (id 2, arrival 9), regardless of its position in the running span.
  const std::vector<SeqDesc> running = {
      Running(1, 0, 8, 1), Running(2, 9, 8, 1), Running(3, 3, 8, 1)};
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs({}, running, /*free_blocks=*/2, 8, 8, 2048));
  EXPECT_EQ(out.preempt, (std::vector<RequestId>{2}));
  EXPECT_EQ(out.decode, (std::vector<RequestId>{1, 3}));
}

TEST(SchedulerTest, CascadingPreemptionUntilDecodeFits) {
  // free=0, three boundary-crossing decodes (each needs 1 block, holds 1).
  // Preempting one refunds 1 block and drops 1 demand (gap +2), so two
  // preemptions (the two latest, 3 then 2) are needed before the lone oldest
  // (1) fits.
  const std::vector<SeqDesc> running = {
      Running(1, 0, 8, 1), Running(2, 1, 8, 1), Running(3, 2, 8, 1)};
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs({}, running, /*free_blocks=*/0, 8, 8, 2048));
  EXPECT_EQ(out.preempt, (std::vector<RequestId>{3, 2}));  // latest-first order
  EXPECT_EQ(out.decode, (std::vector<RequestId>{1}));
}

TEST(SchedulerTest, PreemptionRefundEnablesAdmission) {
  // free=0, one boundary-crossing decode (id 1, holds 2 blocks) and a waiting
  // prompt needing 1 block. Preempting id 1 refunds 2 blocks; the decode set is
  // then empty (0 demand), so 2 blocks are free — the waiting prompt is
  // admitted this step. (Documents the refund-then-admit interaction.)
  const std::vector<SeqDesc> running = {
      Running(1, 0, /*computed=*/8, /*held=*/2)};
  const std::vector<SeqDesc> waiting = {Waiting(2, 1, 8)};  // needs 1 block
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, running, /*free_blocks=*/0, 8, 8, 2048));
  EXPECT_EQ(out.preempt, (std::vector<RequestId>{1}));
  EXPECT_TRUE(out.decode.empty());
  EXPECT_EQ(PrefillIds(out), (std::vector<RequestId>{2}));
}

TEST(SchedulerTest, LoneRunningNonFittingIsStillPreempted) {
  // The scheduler does not special-case the oldest-alone sequence (liveness is
  // a config-sizing guarantee, §10.3): a single running seq whose decode does
  // not fit is preempted, leaving an empty decode set.
  const std::vector<SeqDesc> running = {
      Running(1, 0, /*computed=*/8, /*held=*/1)};
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs({}, running, /*free_blocks=*/0, 8, 8, 2048));
  EXPECT_EQ(out.preempt, (std::vector<RequestId>{1}));
  EXPECT_TRUE(out.decode.empty());
}

// ===================== FCFS order preservation (§6.2 step 3) ================

TEST(SchedulerTest, AdmissionIsAPrefixOfWaitingInOrder) {
  // Admit as many as blocks allow, always a leading prefix of `waiting`.
  const std::vector<SeqDesc> waiting = {Waiting(10, 0, 8), Waiting(11, 1, 8),
                                        Waiting(12, 2, 8), Waiting(13, 3, 8)};
  // bs=8, each needs 1 block; free=2 admits exactly the first two.
  const SchedulerOutput out =
      Scheduler{}.Schedule(Inputs(waiting, {}, /*free_blocks=*/2, 8, 8, 2048));
  EXPECT_EQ(PrefillIds(out), (std::vector<RequestId>{10, 11}));
}

TEST(SchedulerTest, ResumeUsesPromptPlusGeneratedLength) {
  // A preempted sequence at the head of `waiting` carries a longer prefill
  // length (prompt ++ generated); the scheduler admits it by that length and
  // emits it in `num_tokens`.
  const std::vector<SeqDesc> waiting = {Waiting(1, 0, /*prefill=*/24),
                                        Waiting(2, 1, 8)};
  const SchedulerOutput out = Scheduler{}.Schedule(
      Inputs(waiting, {}, /*free_blocks=*/1000, 8, 8, 2048));
  ASSERT_EQ(out.prefill.size(), 2U);
  EXPECT_EQ(out.prefill[0].id, 1U);
  EXPECT_EQ(out.prefill[0].num_tokens, 24);
}

// ============================== determinism =================================

TEST(SchedulerTest, DeterministicAcrossCalls) {
  const std::vector<SeqDesc> running = {Running(1, 0, 8, 1),
                                        Running(2, 3, 8, 1)};
  const std::vector<SeqDesc> waiting = {Waiting(3, 5, 16), Waiting(4, 6, 8)};
  const auto a = Scheduler{}.Schedule(Inputs(waiting, running, 3, 8, 8, 100));
  const auto b = Scheduler{}.Schedule(Inputs(waiting, running, 3, 8, 8, 100));
  EXPECT_EQ(PrefillIds(a), PrefillIds(b));
  EXPECT_EQ(a.decode, b.decode);
  EXPECT_EQ(a.preempt, b.preempt);
}

// ==================== randomized invariant fuzz ============================

// One random case: the descriptor vectors that back the schedule inputs, plus
// the config knobs to schedule under.
struct FuzzCase {
  std::vector<SeqDesc> waiting;
  std::vector<SeqDesc> running;
  std::int64_t free_blocks = 0;
  int max_seqs = 1;
  std::int64_t budget = 1;
};

constexpr int kFuzzBs = 8;

[[nodiscard]] FuzzCase MakeFuzzCase(std::mt19937_64& rng) {
  FuzzCase c;
  const int n_run = static_cast<int>(rng() % 6);
  const int n_wait = static_cast<int>(rng() % 6);
  c.running.reserve(static_cast<std::size_t>(n_run));
  c.waiting.reserve(static_cast<std::size_t>(n_wait));
  std::uint64_t arrival = 0;
  RequestId id = 1;
  for (int i = 0; i < n_run; ++i) {
    const auto computed = static_cast<std::int64_t>(rng() % 40);
    const std::int64_t held =
        BlocksNeeded(0, std::max<std::int64_t>(computed, 1), kFuzzBs);
    c.running.push_back(Running(id++, arrival++, computed, held));
  }
  for (int i = 0; i < n_wait; ++i) {
    const auto prefill = static_cast<std::int64_t>(1 + (rng() % 40));
    c.waiting.push_back(Waiting(id++, arrival++, prefill));
  }
  c.free_blocks = static_cast<std::int64_t>(rng() % 20);
  c.max_seqs = static_cast<int>(1 + (rng() % 6));
  c.budget = static_cast<std::int64_t>(1 + (rng() % 80));
  return c;
}

[[nodiscard]] std::int64_t HeldById(const std::vector<SeqDesc>& running,
                                    RequestId id) {
  for (const auto& s : running) {
    if (s.id == id) {
      return s.blocks_held;
    }
  }
  return 0;
}

[[nodiscard]] std::int64_t DecodeNeedById(const std::vector<SeqDesc>& running,
                                          RequestId id) {
  for (const auto& s : running) {
    if (s.id == id) {
      return BlocksNeeded(s.num_computed_tokens, 1, kFuzzBs);
    }
  }
  return 0;
}

void CheckFuzzInvariants(const FuzzCase& c, const SchedulerOutput& out) {
  // decode ⊎ preempt == running (each running id appears exactly once).
  std::vector<RequestId> partition = out.decode;
  partition.insert(partition.end(), out.preempt.begin(), out.preempt.end());
  std::ranges::sort(partition);
  std::vector<RequestId> run_ids;
  run_ids.reserve(c.running.size());
  for (const auto& s : c.running) {
    run_ids.push_back(s.id);
  }
  std::ranges::sort(run_ids);
  EXPECT_EQ(partition, run_ids);

  // Batch width cap: admission never pushes the scheduled count past
  // max_num_seqs. Decode-first is unconditional, so when the (arbitrary,
  // possibly over-cap) running set already exceeds the cap, no admission
  // happens; the honest invariant is on the admitted count.
  const auto width_headroom = std::max<std::int64_t>(
      0, c.max_seqs - static_cast<std::int64_t>(out.decode.size()));
  EXPECT_LE(static_cast<std::int64_t>(out.prefill.size()), width_headroom);

  // Token budget over admitted prompts.
  std::int64_t admitted_tokens = 0;
  for (const auto& p : out.prefill) {
    admitted_tokens += p.num_tokens;
  }
  EXPECT_LE(admitted_tokens, c.budget);

  // Block accounting: surviving decode demand + admitted prompt blocks fit the
  // free pool plus the preempted sequences' refunded blocks.
  std::int64_t refunded = 0;
  for (const RequestId pid : out.preempt) {
    refunded += HeldById(c.running, pid);
  }
  std::int64_t demand = 0;
  for (const RequestId did : out.decode) {
    demand += DecodeNeedById(c.running, did);
  }
  for (const auto& p : out.prefill) {
    demand += BlocksNeeded(0, p.num_tokens, kFuzzBs);
  }
  EXPECT_LE(demand, c.free_blocks + refunded);

  // Prefill is a prefix of waiting (order-preserving FCFS).
  for (std::size_t i = 0; i < out.prefill.size(); ++i) {
    EXPECT_EQ(out.prefill[i].id, c.waiting[i].id);
  }
}

// Over many random inputs, the output must satisfy the structural invariants
// (§9.2): decode ⊎ preempt == running; the surviving decode set + admitted
// prompts fit the free pool (accounting for preemption refunds); the batch
// width and token budget hold; prefill is a prefix of waiting.
TEST(SchedulerTest, RandomizedInvariants) {
  std::mt19937_64 rng(0xC0FFEE);
  for (int iter = 0; iter < 2000; ++iter) {
    const FuzzCase c = MakeFuzzCase(rng);
    const SchedulerOutput out = Scheduler{}.Schedule(Inputs(
        c.waiting, c.running, c.free_blocks, kFuzzBs, c.max_seqs, c.budget));
    CheckFuzzInvariants(c, out);
  }
}

}  // namespace
