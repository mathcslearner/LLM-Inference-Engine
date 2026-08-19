#include "kernels/kv_scatter.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

// KvScatterF32 tests (M8-T04; design: docs/design/paged-kv-cache.md §9.1). The
// paged KV write: a batch of new K/V vectors is scattered into the per-layer
// slabs at a slot mapping. Correctness is checked by **readback against an
// independently-simulated paged layout** — a test-local plain-array model that
// writes each (token, head) row by explicit (block, in-block-row) decomposition
// — across boundary-straddling prefills and single-token decodes.
//
// The kernel is a pure fp32->fp32 memcpy per (token, head): Class E, bit-exact,
// no dispatched path (a single scalar TU ships — §9.1), so no SCALAR_PASS.

namespace {

using engine::kernels::KvScatterF32;

// Flat element index into a token-major [T, Hkv, d] source block.
[[nodiscard]] std::int64_t SrcIdx(std::int64_t t, std::int64_t h,
                                  std::int64_t e, std::int64_t hkv,
                                  std::int64_t d) {
  return ((((t * hkv) + h) * d) + e);
}

// Flat element index into a [num_blocks, Hkv, bs, d] slab.
[[nodiscard]] std::int64_t SlabIdx(std::int64_t block, std::int64_t h,
                                   std::int64_t p, std::int64_t e,
                                   std::int64_t hkv, std::int64_t bs,
                                   std::int64_t d) {
  return (((((block * hkv) + h) * bs) + p) * d) + e;
}

// A layer's K (or V) slab: [num_blocks, Hkv, bs, d] fp32, flat. Held as a plain
// vector so the test owns the exact bytes and can compare full contents.
struct Slab {
  std::int64_t num_blocks;
  std::int64_t hkv;
  std::int64_t bs;
  std::int64_t d;
  std::vector<float> data;

  Slab(std::int64_t nb, std::int64_t h, std::int64_t b, std::int64_t dd)
      : num_blocks(nb),
        hkv(h),
        bs(b),
        d(dd),
        data(static_cast<std::size_t>((((nb * h) * b) * dd))) {}

  [[nodiscard]] std::int64_t size() const {
    return (((num_blocks * hkv) * bs) * d);
  }
};

// Fill a slab with an index-derived sentinel so any stray write (or a write to
// the wrong offset) shows up as a mismatch against the reference, which starts
// from the identical sentinel.
void FillSentinel(Slab& s) {
  for (std::int64_t i = 0; i < s.size(); ++i) {
    s.data[static_cast<std::size_t>(i)] = -777.0F - static_cast<float>(i);
  }
}

// Independent reference: write token-major src [T, Hkv, d] into a slab at the
// slot mapping, decomposing each flat slot into (block, in-block row) plainly.
void SimScatter(const std::vector<float>& src,
                const std::vector<std::int64_t>& slot_mapping, std::int64_t hkv,
                std::int64_t d, Slab& slab) {
  const std::int64_t bs = slab.bs;
  for (std::int64_t t = 0; t < std::ssize(slot_mapping); ++t) {
    const std::int64_t slot = slot_mapping[static_cast<std::size_t>(t)];
    const std::int64_t block = slot / bs;
    const std::int64_t p = slot % bs;
    for (std::int64_t h = 0; h < hkv; ++h) {
      for (std::int64_t e = 0; e < d; ++e) {
        const float val =
            src[static_cast<std::size_t>(SrcIdx(t, h, e, hkv, d))];
        const std::int64_t idx = SlabIdx(block, h, p, e, hkv, bs, d);
        slab.data[static_cast<std::size_t>(idx)] = val;
      }
    }
  }
}

// A distinctive source block: value = base + t*100000 + h*1000 + e, so every
// (t, h, e) is unique and a misplacement is unambiguous. `base` separates K/V.
[[nodiscard]] std::vector<float> MakeSource(std::int64_t t, std::int64_t hkv,
                                            std::int64_t d, float base) {
  std::vector<float> src(static_cast<std::size_t>((t * hkv) * d));
  for (std::int64_t tt = 0; tt < t; ++tt) {
    for (std::int64_t h = 0; h < hkv; ++h) {
      for (std::int64_t e = 0; e < d; ++e) {
        src[static_cast<std::size_t>(SrcIdx(tt, h, e, hkv, d))] =
            base + (static_cast<float>(tt) * 100000.0F) +
            (static_cast<float>(h) * 1000.0F) + static_cast<float>(e);
      }
    }
  }
  return src;
}

[[nodiscard]] bool SlabsEqual(const Slab& a, const Slab& b) {
  return std::memcmp(a.data.data(), b.data.data(),
                     a.data.size() * sizeof(float)) == 0;
}

// Run the kernel and the reference on fresh sentinel-filled slabs; assert the
// full slab contents are bit-identical (placement correct AND nothing else
// touched, since both start from the same sentinel).
void CheckScatter(const std::vector<std::int64_t>& slot_mapping,
                  std::int64_t hkv, std::int64_t d, std::int64_t bs,
                  std::int64_t num_blocks) {
  const auto t = static_cast<std::int64_t>(slot_mapping.size());
  const std::vector<float> src_k = MakeSource(t, hkv, d, 1.0F);
  const std::vector<float> src_v = MakeSource(t, hkv, d, 9'000'000.0F);

  Slab k_kernel(num_blocks, hkv, bs, d);
  Slab v_kernel(num_blocks, hkv, bs, d);
  Slab k_sim(num_blocks, hkv, bs, d);
  Slab v_sim(num_blocks, hkv, bs, d);
  FillSentinel(k_kernel);
  FillSentinel(v_kernel);
  FillSentinel(k_sim);
  FillSentinel(v_sim);

  KvScatterF32(src_k.data(), src_v.data(), slot_mapping.data(), t, hkv, d, bs,
               k_kernel.data.data(), v_kernel.data.data());
  SimScatter(src_k, slot_mapping, hkv, d, k_sim);
  SimScatter(src_v, slot_mapping, hkv, d, v_sim);

  EXPECT_TRUE(SlabsEqual(k_kernel, k_sim));
  EXPECT_TRUE(SlabsEqual(v_kernel, v_sim));
}

// ===========================================================================
// Prefill: a batch of T tokens over a non-monotonic physical block order,
// straddling block boundaries.
// ===========================================================================

TEST(KvScatterTest, PrefillStraddlesBoundariesNonMonotonicBlocks) {
  // bs=8, 3 blocks [5, 2, 7] (non-monotonic, as the pool's LIFO free list can
  // hand out), 20 tokens (positions 0..19). slot(pos) = block[pos/8]*8 + pos%8.
  constexpr std::int64_t kBs = 8;
  const std::int32_t blocks[] = {5, 2, 7};
  std::vector<std::int64_t> slots;
  slots.reserve(20);
  for (std::int64_t pos = 0; pos < 20; ++pos) {
    slots.push_back((blocks[pos / kBs] * kBs) + (pos % kBs));
  }
  // Sanity on a couple of hand-worked slots: pos 7 -> block 5 slot 47; pos 8 ->
  // block 2 slot 16 (the straddle); pos 16 -> block 7 slot 56.
  EXPECT_EQ(slots[7], 47);
  EXPECT_EQ(slots[8], 16);
  EXPECT_EQ(slots[16], 56);
  CheckScatter(slots, /*hkv=*/2, /*d=*/16, kBs, /*num_blocks=*/8);
}

// ===========================================================================
// Decode: a single token, into a mid-block slot and a fresh block's slot 0.
// ===========================================================================

TEST(KvScatterTest, DecodeSingleTokenMidBlock) {
  // One token at position 12 in block 2 (bs=8): slot 2*8+4 = 20.
  CheckScatter({20}, /*hkv=*/2, /*d=*/16, /*bs=*/8, /*num_blocks=*/4);
}

TEST(KvScatterTest, DecodeSingleTokenFreshBlockRowZero) {
  // One token at the first row of a freshly allocated block b=3: slot 3*8 = 24.
  CheckScatter({24}, /*hkv=*/3, /*d=*/24, /*bs=*/8, /*num_blocks=*/4);
}

// ===========================================================================
// Sequential appends accumulate on one pair of slabs (decode after prefill).
// ===========================================================================

TEST(KvScatterTest, SequentialAppendsAccumulate) {
  constexpr std::int64_t kBs = 16;
  constexpr std::int64_t kHkv = 2;
  constexpr std::int64_t kD = 16;
  constexpr std::int64_t kNumBlocks = 4;
  Slab k(kNumBlocks, kHkv, kBs, kD);
  Slab v(kNumBlocks, kHkv, kBs, kD);
  Slab ks(kNumBlocks, kHkv, kBs, kD);
  Slab vs(kNumBlocks, kHkv, kBs, kD);
  FillSentinel(k);
  FillSentinel(v);
  FillSentinel(ks);
  FillSentinel(vs);

  // Prefill 10 tokens into block 1 (slots 16..25), then two decode steps into
  // block 1 (slot 26) and block 1 (slot 27).
  const std::vector<std::vector<std::int64_t>> batches = {
      {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}, {26}, {27}};
  float base_k = 1.0F;
  float base_v = 5'000'000.0F;
  for (const auto& slots : batches) {
    const auto t = static_cast<std::int64_t>(slots.size());
    const std::vector<float> src_k = MakeSource(t, kHkv, kD, base_k);
    const std::vector<float> src_v = MakeSource(t, kHkv, kD, base_v);
    KvScatterF32(src_k.data(), src_v.data(), slots.data(), t, kHkv, kD, kBs,
                 k.data.data(), v.data.data());
    SimScatter(src_k, slots, kHkv, kD, ks);
    SimScatter(src_v, slots, kHkv, kD, vs);
    base_k += 1'000'000.0F;
    base_v += 1'000'000.0F;
  }
  EXPECT_TRUE(SlabsEqual(k, ks));
  EXPECT_TRUE(SlabsEqual(v, vs));
}

// A later append overwriting the same slot lands the new bytes (immutability is
// a cache-level invariant, not the kernel's — the kernel copies what it is
// handed).
TEST(KvScatterTest, OverwriteSlotTakesNewBytes) {
  constexpr std::int64_t kBs = 8;
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 8;
  constexpr std::int64_t kNumBlocks = 2;
  Slab k(kNumBlocks, kHkv, kBs, kD);
  Slab v(kNumBlocks, kHkv, kBs, kD);
  const std::vector<float> first_k = MakeSource(1, kHkv, kD, 1.0F);
  const std::vector<float> second_k = MakeSource(1, kHkv, kD, 42.0F);
  const std::vector<float> zeros(static_cast<std::size_t>(kHkv * kD), 0.0F);
  const std::vector<std::int64_t> slot = {3};
  KvScatterF32(first_k.data(), zeros.data(), slot.data(), 1, kHkv, kD, kBs,
               k.data.data(), v.data.data());
  KvScatterF32(second_k.data(), zeros.data(), slot.data(), 1, kHkv, kD, kBs,
               k.data.data(), v.data.data());
  // Slot 3 -> block 0 row 3.
  for (std::int64_t e = 0; e < kD; ++e) {
    const std::int64_t idx = SlabIdx(0, 0, 3, e, kHkv, kBs, kD);
    EXPECT_EQ(k.data[static_cast<std::size_t>(idx)],
              second_k[static_cast<std::size_t>(e)]);
  }
}

// ===========================================================================
// Geometry sweep + a large permuted mapping.
// ===========================================================================

TEST(KvScatterTest, GeometrySweep) {
  for (const std::int64_t bs : {8, 16, 32, 64}) {
    for (const std::int64_t hkv : {1, 2, 8}) {
      for (const std::int64_t d : {16, 24, 64}) {
        // Fill two whole blocks (2*bs tokens) at blocks 1 and 3.
        std::vector<std::int64_t> slots;
        slots.reserve(static_cast<std::size_t>(2 * bs));
        for (std::int64_t p = 0; p < bs; ++p) {
          slots.push_back((1 * bs) + p);
        }
        for (std::int64_t p = 0; p < bs; ++p) {
          slots.push_back((3 * bs) + p);
        }
        CheckScatter(slots, hkv, d, bs, /*num_blocks=*/4);
      }
    }
  }
}

TEST(KvScatterTest, LargePermutedMapping) {
  constexpr std::int64_t kBs = 16;
  constexpr std::int64_t kHkv = 2;
  constexpr std::int64_t kD = 64;
  constexpr std::int64_t kNumBlocks = 100;
  // 1000 tokens scattered across a random permutation of distinct slots.
  std::vector<std::int64_t> all_slots;
  all_slots.reserve(static_cast<std::size_t>(kNumBlocks * kBs));
  for (std::int64_t s = 0; s < kNumBlocks * kBs; ++s) {
    all_slots.push_back(s);
  }
  std::mt19937_64 rng(0xC0FFEE);
  std::shuffle(all_slots.begin(), all_slots.end(), rng);
  all_slots.resize(1000);
  CheckScatter(all_slots, kHkv, kD, kBs, kNumBlocks);
}

// ===========================================================================
// Raw-byte fidelity: NaN payloads, -0.0, and denormals copy verbatim (memcpy,
// no arithmetic — the Class E guarantee).
// ===========================================================================

TEST(KvScatterTest, PreservesExactBitPatterns) {
  constexpr std::int64_t kBs = 8;
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 4;
  constexpr std::int64_t kNumBlocks = 2;
  const std::uint32_t patterns[] = {
      0x7FC00001U,  // qNaN with payload
      0xFF800000U,  // -inf
      0x80000000U,  // -0.0
      0x00000001U,  // smallest denormal
  };
  std::vector<float> src(static_cast<std::size_t>(kHkv * kD));
  for (std::int64_t e = 0; e < kD; ++e) {
    std::memcpy(&src[static_cast<std::size_t>(e)], &patterns[e], sizeof(float));
  }
  Slab k(kNumBlocks, kHkv, kBs, kD);
  Slab v(kNumBlocks, kHkv, kBs, kD);
  const std::vector<std::int64_t> slot = {5};
  KvScatterF32(src.data(), src.data(), slot.data(), 1, kHkv, kD, kBs,
               k.data.data(), v.data.data());
  // Slot 5 -> block 0 row 5: k bit-equals patterns[e].
  for (std::int64_t e = 0; e < kD; ++e) {
    const std::int64_t idx = SlabIdx(0, 0, 5, e, kHkv, kBs, kD);
    std::uint32_t got = 0;
    std::memcpy(&got, &k.data[static_cast<std::size_t>(idx)], sizeof(float));
    EXPECT_EQ(got, patterns[e]);
  }
}

}  // namespace
