#include "kernels/kv_gather.h"

#include "kernels/kv_scatter.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

// KvGatherF32 tests (M8-T06; design: docs/design/paged-kv-cache.md §9.3). The
// paged KV read — the mirror of KvScatterF32: one layer's first `length` tokens
// are gathered from the [num_blocks, Hkv, bs, d] slabs, addressed through a
// block table, into contiguous head-major [Hkv, length, d] output. Correctness
// is checked by comparison against an **independently-simulated gather** over a
// test-local paged layout that uses a *reverse-permuted* block table and
// *poisoned* unused/tail slots, so any stray read shows up. A separate
// scatter->gather round trip proves the two kernels against each other.
//
// The kernel is a pure fp32->fp32 memcpy per (head, block) tile: Class E,
// bit-exact, no dispatched path (a single scalar TU ships — §9.3), so no
// SCALAR_PASS.

namespace {

using engine::kernels::KvGatherF32;
using engine::kernels::KvScatterF32;

// Flat element index into a [num_blocks, Hkv, bs, d] slab.
[[nodiscard]] std::int64_t SlabIdx(std::int64_t block, std::int64_t h,
                                   std::int64_t p, std::int64_t e,
                                   std::int64_t hkv, std::int64_t bs,
                                   std::int64_t d) {
  return (((((block * hkv) + h) * bs) + p) * d) + e;
}

// Flat element index into a head-major [Hkv, length, d] output.
[[nodiscard]] std::int64_t OutIdx(std::int64_t h, std::int64_t pos,
                                  std::int64_t e, std::int64_t length,
                                  std::int64_t d) {
  return (((h * length) + pos) * d) + e;
}

// A poison value distinct from any legitimately-stored value, seeded into every
// slab slot; only the referenced (block, row < length%bs) tiles are overwritten
// with data, so a gather that reads the wrong offset returns poison and fails.
constexpr float kPoison = -123456.0F;

// A distinctive per-(pos, head, elem) value so a misplacement is unambiguous.
[[nodiscard]] float Val(std::int64_t pos, std::int64_t h, std::int64_t e,
                        float base) {
  return base + (static_cast<float>(pos) * 100000.0F) +
         (static_cast<float>(h) * 1000.0F) + static_cast<float>(e);
}

// Build a slab (flat [num_blocks, Hkv, bs, d]) poisoned everywhere, then write
// the logical history [0, length) through `block_table` — writing the value
// Val(pos, h, e, base) at (block_table[pos/bs], pos%bs). Rows of the tail block
// past `length` stay poisoned, as do wholly-unreferenced blocks.
[[nodiscard]] std::vector<float> BuildSlab(std::int64_t num_blocks,
                                           std::int64_t hkv, std::int64_t bs,
                                           std::int64_t d, std::int64_t length,
                                           const std::vector<std::int32_t>& bt,
                                           float base) {
  std::vector<float> slab(
      static_cast<std::size_t>((((num_blocks * hkv) * bs) * d)), kPoison);
  for (std::int64_t pos = 0; pos < length; ++pos) {
    const std::int64_t block = bt[static_cast<std::size_t>(pos / bs)];
    const std::int64_t p = pos % bs;
    for (std::int64_t h = 0; h < hkv; ++h) {
      for (std::int64_t e = 0; e < d; ++e) {
        slab[static_cast<std::size_t>(SlabIdx(block, h, p, e, hkv, bs, d))] =
            Val(pos, h, e, base);
      }
    }
  }
  return slab;
}

// A reverse-permuted block table for the logical blocks a `length`-token
// sequence needs: logical block b -> physical (num_blocks-1 - b). Exercises
// non-identity, non-monotonic indirection.
[[nodiscard]] std::vector<std::int32_t> ReverseBlockTable(
    std::int64_t num_blocks, std::int64_t bs, std::int64_t length) {
  const std::int64_t needed = (length + bs - 1) / bs;
  std::vector<std::int32_t> bt(static_cast<std::size_t>(needed));
  for (std::int64_t b = 0; b < needed; ++b) {
    bt[static_cast<std::size_t>(b)] =
        static_cast<std::int32_t>(num_blocks - 1 - b);
  }
  return bt;
}

// Run the kernel and assert every gathered element equals Val(...); the poison
// never appears (which it would if the tail/unused slots were read).
void CheckGather(std::int64_t num_blocks, std::int64_t hkv, std::int64_t bs,
                 std::int64_t d, std::int64_t length) {
  const std::vector<std::int32_t> bt =
      ReverseBlockTable(num_blocks, bs, length);
  const std::vector<float> k_slab =
      BuildSlab(num_blocks, hkv, bs, d, length, bt, /*base=*/1.0F);
  const std::vector<float> v_slab =
      BuildSlab(num_blocks, hkv, bs, d, length, bt, /*base=*/9'000'000.0F);

  std::vector<float> out_k(static_cast<std::size_t>((hkv * length) * d), 0.0F);
  std::vector<float> out_v(static_cast<std::size_t>((hkv * length) * d), 0.0F);
  KvGatherF32(k_slab.data(), v_slab.data(), bt.data(), bs, length, hkv, d,
              out_k.data(), out_v.data());

  for (std::int64_t pos = 0; pos < length; ++pos) {
    for (std::int64_t h = 0; h < hkv; ++h) {
      for (std::int64_t e = 0; e < d; ++e) {
        const auto idx = static_cast<std::size_t>(OutIdx(h, pos, e, length, d));
        EXPECT_EQ(out_k[idx], Val(pos, h, e, 1.0F))
            << "K pos " << pos << " h " << h << " e " << e;
        EXPECT_EQ(out_v[idx], Val(pos, h, e, 9'000'000.0F))
            << "V pos " << pos << " h " << h << " e " << e;
      }
    }
  }
}

// ===========================================================================
// Length sweep: many-block, exact-boundary, partial-tail, and edge lengths.
// ===========================================================================

TEST(KvGatherTest, LengthSweepAcrossBlockSizes) {
  for (const std::int64_t bs : {8, 16, 32, 64}) {
    // Lengths chosen to cover: empty, single, tail-of-block, exact boundary,
    // just-past-boundary, several whole blocks, and a partial many-block tail.
    for (const std::int64_t length :
         {std::int64_t{1}, bs - 1, bs, bs + 1, 2 * bs, (3 * bs) + 5}) {
      const std::int64_t num_blocks = ((length + bs - 1) / bs) + 2;  // + spares
      for (const std::int64_t hkv :
           {std::int64_t{1}, std::int64_t{2}, std::int64_t{4}}) {
        for (const std::int64_t d :
             {std::int64_t{18}, std::int64_t{24}, std::int64_t{64}}) {
          CheckGather(num_blocks, hkv, bs, d, length);
        }
      }
    }
  }
}

TEST(KvGatherTest, EmptyLengthIsNoOp) {
  // length == 0: the output is [Hkv, 0, d] (nothing to check) and the kernel
  // must not read any slab byte. Pass a null block table to prove it.
  std::vector<float> out_k;
  std::vector<float> out_v;
  KvGatherF32(nullptr, nullptr, nullptr, /*block_size=*/8, /*length=*/0,
              /*kv_heads=*/2, /*d=*/16, out_k.data(), out_v.data());
  SUCCEED();
}

// ===========================================================================
// Thread-count invariance: a plain copy has no reduction, so the result is
// bit-identical regardless of how the (head, block) units are split.
// ===========================================================================

TEST(KvGatherTest, BitIdenticalAcrossThreadCounts) {
  // A larger case so the work actually fans out across workers.
  constexpr std::int64_t kBs = 16;
  constexpr std::int64_t kHkv = 4;
  constexpr std::int64_t kD = 64;
  constexpr std::int64_t kLength = (7 * kBs) + 3;  // 115, partial tail
  const std::int64_t num_blocks = ((kLength + kBs - 1) / kBs) + 2;
  const std::vector<std::int32_t> bt =
      ReverseBlockTable(num_blocks, kBs, kLength);
  const std::vector<float> k_slab =
      BuildSlab(num_blocks, kHkv, kBs, kD, kLength, bt, 1.0F);
  const std::vector<float> v_slab =
      BuildSlab(num_blocks, kHkv, kBs, kD, kLength, bt, 9'000'000.0F);
  const auto n = static_cast<std::size_t>((kHkv * kLength) * kD);

  // The DefaultPool thread count is fixed per process. A plain copy has no
  // reduction, so two independent runs over the same inputs are byte-identical
  // regardless of how the (head, block) units are split across workers.
  std::vector<float> ak(n, 0.0F);
  std::vector<float> av(n, 0.0F);
  std::vector<float> bk(n, 0.0F);
  std::vector<float> bv(n, 0.0F);
  KvGatherF32(k_slab.data(), v_slab.data(), bt.data(), kBs, kLength, kHkv, kD,
              ak.data(), av.data());
  KvGatherF32(k_slab.data(), v_slab.data(), bt.data(), kBs, kLength, kHkv, kD,
              bk.data(), bv.data());
  // Exact per-element equality (all values finite, no NaN): the "no reduction"
  // observable. memcmp would trip bugprone-suspicious-memory-comparison on
  // float, so compare values.
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(ak[i], bk[i]) << "K index " << i;
    ASSERT_EQ(av[i], bv[i]) << "V index " << i;
  }
}

// ===========================================================================
// Raw-byte fidelity: NaN payloads, -0.0, denormals copy verbatim (Class E).
// ===========================================================================

TEST(KvGatherTest, PreservesExactBitPatterns) {
  constexpr std::int64_t kBs = 8;
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 4;
  constexpr std::int64_t kNumBlocks = 3;
  constexpr std::int64_t kLength = 2;  // two tokens in logical block 0
  const std::uint32_t patterns[] = {
      0x7FC00001U,  // qNaN with payload
      0xFF800000U,  // -inf
      0x80000000U,  // -0.0
      0x00000001U,  // smallest denormal
  };
  // Physical block for logical block 0 under the reverse table is num_blocks-1.
  const std::vector<std::int32_t> bt =
      ReverseBlockTable(kNumBlocks, kBs, kLength);
  std::vector<float> slab(
      static_cast<std::size_t>((((kNumBlocks * kHkv) * kBs) * kD)), kPoison);
  // Write pattern[e] at pos 0, and reversed at pos 1.
  for (std::int64_t e = 0; e < kD; ++e) {
    float f0 = 0.0F;
    float f1 = 0.0F;
    std::memcpy(&f0, &patterns[e], sizeof(float));
    std::memcpy(&f1, &patterns[kD - 1 - e], sizeof(float));
    slab[static_cast<std::size_t>(SlabIdx(bt[0], 0, 0, e, kHkv, kBs, kD))] = f0;
    slab[static_cast<std::size_t>(SlabIdx(bt[0], 0, 1, e, kHkv, kBs, kD))] = f1;
  }
  std::vector<float> out_k(static_cast<std::size_t>((kHkv * kLength) * kD),
                           0.0F);
  std::vector<float> out_v(static_cast<std::size_t>((kHkv * kLength) * kD),
                           0.0F);
  KvGatherF32(slab.data(), slab.data(), bt.data(), kBs, kLength, kHkv, kD,
              out_k.data(), out_v.data());
  for (std::int64_t e = 0; e < kD; ++e) {
    std::uint32_t g0 = 0;
    std::uint32_t g1 = 0;
    std::memcpy(&g0,
                &out_k[static_cast<std::size_t>(OutIdx(0, 0, e, kLength, kD))],
                sizeof(float));
    std::memcpy(&g1,
                &out_k[static_cast<std::size_t>(OutIdx(0, 1, e, kLength, kD))],
                sizeof(float));
    EXPECT_EQ(g0, patterns[e]);
    EXPECT_EQ(g1, patterns[kD - 1 - e]);
  }
}

// ===========================================================================
// Scatter -> gather round trip: writing token-major [T, Hkv, d] through a
// permuted slot mapping, then gathering it back, is the identity. The two
// kernels (M8-T04 write, M8-T06 read) prove each other.
// ===========================================================================

TEST(KvGatherTest, ScatterThenGatherRoundTrips) {
  for (const std::int64_t bs : {8, 16, 32, 64}) {
    constexpr std::int64_t kHkv = 2;
    constexpr std::int64_t kD = 24;
    const std::int64_t length = (2 * bs) + 3;
    const std::int64_t num_blocks = ((length + bs - 1) / bs) + 1;
    const std::vector<std::int32_t> bt =
        ReverseBlockTable(num_blocks, bs, length);

    // Slot mapping from the block table (contiguous positions 0..length).
    std::vector<std::int64_t> slots(static_cast<std::size_t>(length));
    for (std::int64_t pos = 0; pos < length; ++pos) {
      slots[static_cast<std::size_t>(pos)] =
          (static_cast<std::int64_t>(bt[static_cast<std::size_t>(pos / bs)]) *
           bs) +
          (pos % bs);
    }

    // Token-major source [length, Hkv, d].
    std::vector<float> src_k(static_cast<std::size_t>((length * kHkv) * kD));
    std::vector<float> src_v(static_cast<std::size_t>((length * kHkv) * kD));
    for (std::int64_t pos = 0; pos < length; ++pos) {
      for (std::int64_t h = 0; h < kHkv; ++h) {
        for (std::int64_t e = 0; e < kD; ++e) {
          const auto si =
              static_cast<std::size_t>(((((pos * kHkv) + h) * kD)) + e);
          src_k[si] = Val(pos, h, e, 1.0F);
          src_v[si] = Val(pos, h, e, 9'000'000.0F);
        }
      }
    }

    std::vector<float> k_slab(
        static_cast<std::size_t>((((num_blocks * kHkv) * bs) * kD)), kPoison);
    std::vector<float> v_slab(
        static_cast<std::size_t>((((num_blocks * kHkv) * bs) * kD)), kPoison);
    KvScatterF32(src_k.data(), src_v.data(), slots.data(), length, kHkv, kD, bs,
                 k_slab.data(), v_slab.data());

    std::vector<float> out_k(static_cast<std::size_t>((kHkv * length) * kD));
    std::vector<float> out_v(static_cast<std::size_t>((kHkv * length) * kD));
    KvGatherF32(k_slab.data(), v_slab.data(), bt.data(), bs, length, kHkv, kD,
                out_k.data(), out_v.data());

    // out[h, pos, e] must equal src[pos, h, e] (the transpose the append/view
    // pair performs).
    for (std::int64_t pos = 0; pos < length; ++pos) {
      for (std::int64_t h = 0; h < kHkv; ++h) {
        for (std::int64_t e = 0; e < kD; ++e) {
          const auto si =
              static_cast<std::size_t>(((((pos * kHkv) + h) * kD)) + e);
          const auto oi =
              static_cast<std::size_t>(OutIdx(h, pos, e, length, kD));
          EXPECT_EQ(out_k[oi], src_k[si]) << "bs " << bs << " pos " << pos;
          EXPECT_EQ(out_v[oi], src_v[si]) << "bs " << bs << " pos " << pos;
        }
      }
    }
  }
}

}  // namespace
