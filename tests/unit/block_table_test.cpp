#include "kvcache/block_table.h"

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "tensor/dtype.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

// BlockTable tests (M8-T03; design: docs/design/paged-kv-cache.md §7). The
// per-sequence logical→physical block map: append-with-boundary-crossing block
// allocation, hand-verified prefill (T tokens) and decode (1 token) slot
// mappings, all-or-nothing exhaustion, truncate (partial-tail-kept), and
// blocks-returned-to-pool on free. Pure bookkeeping over BlockPool — no model,
// no kernels, so no SCALAR_PASS.
//
// The design's §7.2 worked example uses bs = 4 for compactness, but
// BlockPool::Create only accepts bs ∈ {8,16,32,64} (the §4 divisibility
// constraint). These tests use bs = 8 — the smallest valid size — and shape the
// pool's LIFO free list to reproduce the same non-trivial physical-block
// ordering the example illustrates.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::StatusOr;
using engine::kvcache::BlockPool;
using engine::kvcache::BlockPoolStats;
using engine::kvcache::BlockTable;
using engine::kvcache::CacheGeometry;
using engine::tensor::DataType;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] CacheGeometry TinyGeom() {
  // tiny-llama: L=2, Hkv=2, d=16.
  return CacheGeometry{.num_layers = 2,
                       .num_kv_heads = 2,
                       .head_dim = 16,
                       .dtype = DataType::kFloat32};
}

constexpr int kBs = 8;  // smallest valid block size.

[[nodiscard]] BlockPool MakePool(std::int64_t num_blocks) {
  return Unwrap(BlockPool::Create(TinyGeom(), kBs, num_blocks, nullptr));
}

// Shape the pool's LIFO free list so the next two Allocate() calls return
// `first` then `second` (both < num_blocks). BlockPool hands out ascending ids
// from a fresh pool, so we allocate a prefix, then release the two targets in
// the order that leaves them on top of the stack.
void PrimeFreeList(BlockPool& pool, std::int32_t first, std::int32_t second) {
  const std::int32_t hi = std::max(first, second);
  std::vector<std::int32_t> taken;
  for (std::int32_t i = 0; i <= hi; ++i) {
    taken.push_back(Unwrap(pool.Allocate()));
  }
  // Release the non-target blocks first (they sink to the bottom of the stack),
  // then `second`, then `first` — so the LIFO pops `first` then `second` for
  // the next two Allocate() calls.
  for (const std::int32_t id : taken) {
    if (id != first && id != second) {
      pool.Release(id);
    }
  }
  pool.Release(second);
  pool.Release(first);
}

// ----------------------------- construction -----------------------------

TEST(BlockTableCreate, StartsEmpty) {
  BlockPool pool = MakePool(4);
  const BlockTable table(&pool);
  EXPECT_EQ(table.num_tokens(), 0);
  EXPECT_EQ(table.num_blocks(), 0);
  EXPECT_TRUE(table.blocks().empty());
  EXPECT_EQ(table.block_size(), kBs);
  EXPECT_EQ(table.pool(), &pool);
  EXPECT_EQ(pool.stats().used, 0);
}

// ----------------------------- prefill / decode slot mappings -------------

TEST(BlockTableAppend, PrefillFromEmptyHandVerified) {
  // §7.2 analogue at bs = 8: prefill T = 12 from empty needs ⌈12/8⌉ = 2 blocks.
  // Prime the free list so Allocate returns [5, 2] → blocks_ = [5, 2].
  BlockPool pool = MakePool(8);
  PrimeFreeList(pool, /*first=*/5, /*second=*/2);
  BlockTable table(&pool);

  const std::vector<std::int64_t> slots = Unwrap(table.AppendTokens(12));

  EXPECT_EQ(table.num_tokens(), 12);
  EXPECT_EQ(table.num_blocks(), 2);
  const std::vector<std::int32_t> want_blocks = {5, 2};
  EXPECT_EQ(
      std::vector<std::int32_t>(table.blocks().begin(), table.blocks().end()),
      want_blocks);
  // Positions 0..7 → block 5 (slots 40..47); 8..11 → block 2 (slots 16..19).
  // The straddle at pos 7→8 crosses from physical block 5 to block 2.
  const std::vector<std::int64_t> want = {40, 41, 42, 43, 44, 45,
                                          46, 47, 16, 17, 18, 19};
  EXPECT_EQ(slots, want);
  // slot(pos) agrees with the mapping for every committed position.
  for (std::int64_t pos = 0; pos < 12; ++pos) {
    EXPECT_EQ(table.slot(pos), want[static_cast<std::size_t>(pos)]);
  }
}

TEST(BlockTableAppend, DecodeNoAllocationHandVerified) {
  // After the prefill above, decode one token at num_tokens_ = 12 → 13. Still
  // ⌈13/8⌉ = 2 blocks, so no allocation; pos 12 → blocks_[1]·8 + 4 = 2·8+4=20.
  BlockPool pool = MakePool(8);
  PrimeFreeList(pool, 5, 2);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(12));
  const std::int64_t used_before = pool.stats().used;

  const std::vector<std::int64_t> slots = Unwrap(table.AppendTokens(1));

  EXPECT_EQ(slots, (std::vector<std::int64_t>{20}));
  EXPECT_EQ(table.num_tokens(), 13);
  EXPECT_EQ(table.num_blocks(), 2);
  EXPECT_EQ(pool.stats().used, used_before);  // no new block
}

TEST(BlockTableAppend, DecodeBoundaryCrossingAllocates) {
  // Grow to exactly 16 tokens (2 full blocks), then decode 16 → 17: ⌈17/8⌉ −
  // ⌈16/8⌉ = 3 − 2 = 1 new block. pos 16 → new_block·8 + 0.
  BlockPool pool = MakePool(8);
  PrimeFreeList(pool, 5, 2);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(16));
  EXPECT_EQ(table.num_blocks(), 2);
  const std::int64_t used_before = pool.stats().used;

  const std::vector<std::int64_t> slots = Unwrap(table.AppendTokens(1));

  EXPECT_EQ(table.num_tokens(), 17);
  EXPECT_EQ(table.num_blocks(), 3);
  EXPECT_EQ(pool.stats().used, used_before + 1);
  const std::int32_t new_block = table.blocks()[2];
  EXPECT_EQ(
      slots,
      (std::vector<std::int64_t>{static_cast<std::int64_t>(new_block) * kBs}));
}

// ----------------------------- growth invariants -----------------------------

TEST(BlockTableAppend, TokenByTokenGrowthMatchesBatch) {
  // Appending one token at a time must produce the same block set and per-token
  // slots as appending them all at once.
  BlockPool batch_pool = MakePool(16);
  BlockTable batch(&batch_pool);
  const std::vector<std::int64_t> batch_slots = Unwrap(batch.AppendTokens(20));

  BlockPool step_pool = MakePool(16);
  BlockTable step(&step_pool);
  std::vector<std::int64_t> step_slots;
  for (int i = 0; i < 20; ++i) {
    std::vector<std::int64_t> s = Unwrap(step.AppendTokens(1));
    ASSERT_EQ(s.size(), 1U);
    step_slots.push_back(s[0]);
  }
  EXPECT_EQ(batch_slots, step_slots);
  EXPECT_EQ(batch.num_blocks(), step.num_blocks());
  EXPECT_EQ(batch.num_blocks(), 3);  // ⌈20/8⌉
}

TEST(BlockTableAppend, NumBlocksTracksCeilAndSlotsUnique) {
  // Sweep a mix of prefill/decode appends; after each, num_blocks ==
  // ⌈num_tokens/bs⌉ and every committed slot across history is distinct.
  BlockPool pool = MakePool(32);
  BlockTable table(&pool);
  const std::vector<std::int64_t> steps = {1, 7, 1, 1, 8, 3, 16, 1};
  std::int64_t total = 0;
  for (const std::int64_t c : steps) {
    const std::vector<std::int64_t> slots = Unwrap(table.AppendTokens(c));
    total += c;
    EXPECT_EQ(table.num_tokens(), total);
    EXPECT_EQ(table.num_blocks(), (total + kBs - 1) / kBs);
    EXPECT_EQ(static_cast<std::int64_t>(slots.size()), c);
  }
  std::set<std::int64_t> seen;
  for (std::int64_t pos = 0; pos < total; ++pos) {
    EXPECT_TRUE(seen.insert(table.slot(pos)).second)
        << "duplicate slot at pos " << pos;
  }
}

TEST(BlockTableAppend, ExactBlockBoundary) {
  // Append exactly bs tokens → 1 block; one more → 2 blocks.
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(kBs));
  EXPECT_EQ(table.num_blocks(), 1);
  (void)Unwrap(table.AppendTokens(1));
  EXPECT_EQ(table.num_blocks(), 2);
}

// ----------------------------- validation -----------------------------

TEST(BlockTableAppend, RejectsNonPositiveCount) {
  BlockPool pool = MakePool(4);
  BlockTable table(&pool);
  EXPECT_TRUE(IsInvalidArgument(table.AppendTokens(0).status()));
  EXPECT_TRUE(IsInvalidArgument(table.AppendTokens(-3).status()));
  EXPECT_EQ(table.num_tokens(), 0);
  EXPECT_EQ(pool.stats().used, 0);
}

// ----------------------------- all-or-nothing exhaustion ------------------

TEST(BlockTableAppend, ExhaustionLeavesTableAndPoolUntouched) {
  // Pool of 3 blocks. One table already holds 1 block; a second requests 16
  // tokens (needs 2 blocks) with only 2 free — succeeds. Then a third request
  // needs more than remain and must roll back cleanly.
  BlockPool pool = MakePool(3);
  BlockTable other(&pool);
  (void)Unwrap(other.AppendTokens(1));  // consumes 1 block, 2 free

  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(9));  // needs ⌈9/8⌉ = 2 blocks → 0 free
  EXPECT_EQ(pool.stats().free, 0);

  // Snapshot, then attempt an append that needs a block the pool cannot give.
  const std::int64_t tokens_before = table.num_tokens();
  const std::int64_t blocks_before = table.num_blocks();
  const std::vector<std::int32_t> ids_before(table.blocks().begin(),
                                             table.blocks().end());
  const BlockPoolStats stats_before = pool.stats();

  const StatusOr<std::vector<std::int64_t>> grow =
      table.AppendTokens(8);  // +1 block
  EXPECT_TRUE(IsResourceExhausted(grow.status()));

  // Table and pool exactly as before the failed append.
  EXPECT_EQ(table.num_tokens(), tokens_before);
  EXPECT_EQ(table.num_blocks(), blocks_before);
  EXPECT_EQ(
      std::vector<std::int32_t>(table.blocks().begin(), table.blocks().end()),
      ids_before);
  EXPECT_EQ(pool.stats().used, stats_before.used);
  EXPECT_EQ(pool.stats().free, stats_before.free);
}

TEST(BlockTableAppend, RecoversAfterExhaustion) {
  // After a rolled-back multi-block append, a smaller one that fits succeeds —
  // proving the partially-taken blocks were returned.
  BlockPool pool = MakePool(2);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(8));  // 1 block, 1 free

  // Request 3 blocks' worth (24 tokens from 8 → needs 3 more): only 1 free.
  EXPECT_TRUE(IsResourceExhausted(table.AppendTokens(24).status()));
  EXPECT_EQ(pool.stats().free, 1);  // the one block was not consumed

  // A single-block append still works.
  EXPECT_TRUE(table.AppendTokens(8).ok());
  EXPECT_EQ(table.num_blocks(), 2);
}

// ----------------------------- truncate -----------------------------

TEST(BlockTableTruncate, KeepsPartialTailBlock) {
  // 20 tokens → 3 blocks. Truncate to 10 → ⌈10/8⌉ = 2 blocks; one released.
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(20));
  const std::int32_t released = table.blocks()[2];
  const std::int64_t used_before = pool.stats().used;

  ASSERT_TRUE(table.Truncate(10).ok());

  EXPECT_EQ(table.num_tokens(), 10);
  EXPECT_EQ(table.num_blocks(), 2);
  EXPECT_EQ(pool.stats().used, used_before - 1);
  EXPECT_EQ(pool.refcount(released), 0);  // returned to the pool
  // The surviving prefix keeps its exact slots.
  for (std::int64_t pos = 0; pos < 10; ++pos) {
    const std::int64_t block =
        table.blocks()[static_cast<std::size_t>(pos / kBs)];
    EXPECT_EQ(table.slot(pos), (block * kBs) + (pos % kBs));
  }
}

TEST(BlockTableTruncate, ReAppendReusesRetainedTail) {
  // Truncate mid-block, then append back into the same partial block: the
  // retained positions are unchanged and new positions land in the same
  // physical block (no allocation).
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(12));  // 2 blocks
  const std::int32_t tail = table.blocks()[1];
  ASSERT_TRUE(table.Truncate(10).ok());  // still 2 blocks (⌈10/8⌉=2)
  EXPECT_EQ(table.num_blocks(), 2);
  const std::int64_t used_before = pool.stats().used;

  const std::vector<std::int64_t> slots =
      Unwrap(table.AppendTokens(2));  // pos 10,11

  EXPECT_EQ(pool.stats().used, used_before);  // no new block
  const std::int64_t base = static_cast<std::int64_t>(tail) * kBs;
  EXPECT_EQ(slots, (std::vector<std::int64_t>{base + 2, base + 3}));
}

TEST(BlockTableTruncate, ExactBoundaryReleasesWholeBlocks) {
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(24));  // 3 blocks
  ASSERT_TRUE(table.Truncate(8).ok());   // exactly 1 block
  EXPECT_EQ(table.num_blocks(), 1);
  EXPECT_EQ(pool.stats().used, 1);
}

TEST(BlockTableTruncate, ToZeroReleasesAll) {
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(20));
  ASSERT_TRUE(table.Truncate(0).ok());
  EXPECT_EQ(table.num_tokens(), 0);
  EXPECT_EQ(table.num_blocks(), 0);
  EXPECT_EQ(pool.stats().used, 0);
}

TEST(BlockTableTruncate, NoOpAtCurrentLength) {
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(12));
  const BlockPoolStats before = pool.stats();
  ASSERT_TRUE(table.Truncate(12).ok());
  EXPECT_EQ(table.num_tokens(), 12);
  EXPECT_EQ(table.num_blocks(), 2);
  EXPECT_EQ(pool.stats().used, before.used);
}

TEST(BlockTableTruncate, RejectsOutOfRange) {
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(12));
  const BlockPoolStats before = pool.stats();
  EXPECT_TRUE(IsInvalidArgument(table.Truncate(13)));  // > num_tokens
  EXPECT_TRUE(IsInvalidArgument(table.Truncate(-1)));
  // State untouched by rejected truncates.
  EXPECT_EQ(table.num_tokens(), 12);
  EXPECT_EQ(pool.stats().used, before.used);
}

// ----------------------------- free-on-completion ------------------------

TEST(BlockTableFree, FreeAllReturnsBlocks) {
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(20));
  EXPECT_EQ(pool.stats().used, 3);
  table.FreeAll();
  EXPECT_EQ(pool.stats().used, 0);
  EXPECT_EQ(table.num_tokens(), 0);
  EXPECT_EQ(table.num_blocks(), 0);
}

TEST(BlockTableFree, DestructorReturnsBlocks) {
  BlockPool pool = MakePool(8);
  {
    BlockTable table(&pool);
    (void)Unwrap(table.AppendTokens(20));
    EXPECT_EQ(pool.stats().used, 3);
  }
  EXPECT_EQ(pool.stats().used, 0);  // RAII reclamation
}

TEST(BlockTableFree, InterleavedLifetimesReclaimIndependently) {
  BlockPool pool = MakePool(16);
  BlockTable a(&pool);
  (void)Unwrap(a.AppendTokens(8));  // 1 block
  {
    BlockTable b(&pool);
    (void)Unwrap(b.AppendTokens(16));  // 2 blocks
    EXPECT_EQ(pool.stats().used, 3);
  }
  EXPECT_EQ(pool.stats().used, 1);  // only b's blocks returned
  a.FreeAll();
  EXPECT_EQ(pool.stats().used, 0);
}

// ----------------------------- move -----------------------------

TEST(BlockTableMove, MovedToOwnsBlocksMovedFromEmpty) {
  BlockPool pool = MakePool(8);
  std::optional<BlockTable> src(std::in_place, &pool);
  const std::vector<std::int64_t> slots = Unwrap(src->AppendTokens(20));
  const std::vector<std::int32_t> ids(src->blocks().begin(),
                                      src->blocks().end());

  const BlockTable dst(std::move(*src));
  EXPECT_EQ(src->num_tokens(), 0);  // moved-from emptied
  EXPECT_TRUE(src->blocks().empty());
  EXPECT_EQ(dst.num_tokens(), 20);
  EXPECT_EQ(std::vector<std::int32_t>(dst.blocks().begin(), dst.blocks().end()),
            ids);

  // Destroying the moved-from source releases nothing.
  src.reset();
  EXPECT_EQ(pool.stats().used, 3);
  // dst still maps every position.
  for (std::int64_t pos = 0; pos < 20; ++pos) {
    EXPECT_EQ(dst.slot(pos), slots[static_cast<std::size_t>(pos)]);
  }
}

// ----------------------------- death tests -----------------------------

TEST(BlockTableDeath, NullPoolAborts) {
  EXPECT_DEATH({ const BlockTable table(nullptr); }, "must not be null");
}

TEST(BlockTableDeath, SlotOutOfRangeAborts) {
  BlockPool pool = MakePool(8);
  BlockTable table(&pool);
  (void)Unwrap(table.AppendTokens(4));
  EXPECT_DEATH((void)table.slot(4), "out of range");
  EXPECT_DEATH((void)table.slot(-1), "out of range");
}

TEST(BlockTableDeath, SlotOnEmptyAborts) {
  BlockPool pool = MakePool(8);
  const BlockTable table(&pool);
  EXPECT_DEATH((void)table.slot(0), "out of range");
}

}  // namespace
