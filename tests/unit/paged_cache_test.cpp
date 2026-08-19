#include "kvcache/paged_cache.h"

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "kvcache/block_table.h"
#include "kvcache/kv_cache.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// PagedKvCache tests (M8-T04; design: docs/design/paged-kv-cache.md §8). The
// paged implementation of the M5 KvCache interface: the layer-0-grows /
// layers-1..L−1-reuse append protocol (§8.2), the KvScatterF32 write, the
// zero-copy paged_view fast path (§8.3), front-loaded validation, truncate,
// advisory capacity, and RAII block reclamation. Reads are verified by
// reconstructing token values from paged_view + slab arithmetic and comparing
// against the known appended values (the independent paged-layout model, §9.1)
// — end-to-end append -> scatter -> read back. `unit` label; the scatter kernel
// is ISA-independent (Class E), so no SCALAR_PASS.

namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::StatusOr;
using engine::kvcache::BlockPool;
using engine::kvcache::CacheGeometry;
using engine::kvcache::KvCache;
using engine::kvcache::PagedKvCache;
using engine::kvcache::PagedKvView;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

constexpr int kBs = 8;  // smallest valid block size (§4).

[[nodiscard]] CacheGeometry TinyGeom() {
  return CacheGeometry{.num_layers = 2,
                       .num_kv_heads = 2,
                       .head_dim = 16,
                       .dtype = DataType::kFloat32};
}

[[nodiscard]] BlockPool MakePool(CacheGeometry geom, std::int64_t num_blocks) {
  return Unwrap(BlockPool::Create(geom, kBs, num_blocks, nullptr));
}

// Shape the pool's LIFO free list so the next two Allocate()s return `first`
// then `second` (mirrors block_table_test's helper).
void PrimeFreeList(BlockPool& pool, std::int32_t first, std::int32_t second) {
  const std::int32_t hi = std::max(first, second);
  std::vector<std::int32_t> taken;
  taken.reserve(static_cast<std::size_t>(hi) + 1);
  for (std::int32_t i = 0; i <= hi; ++i) {
    taken.push_back(Unwrap(pool.Allocate()));
  }
  for (const std::int32_t id : taken) {
    if (id != first && id != second) {
      pool.Release(id);
    }
  }
  pool.Release(second);
  pool.Release(first);
}

// A unique, exactly-representable value for (layer, pos, head, elem, kv). All
// terms are integers well under 2^24, so float holds them exactly and any
// misplacement is caught bit-for-bit. kv = 0 for K, 1 for V.
[[nodiscard]] float Expect(int layer, std::int64_t pos, std::int64_t h,
                           std::int64_t e, int kv) {
  return static_cast<float>(
      (((layer + 1) * 1'000'000) + (kv * 500'000) + (pos * 1'000) + (h * 100)) +
      e);
}

// A token-major [count, Hkv, d] block for `layer`, positions [start,
// start+count), K if kv==0 else V — filled with Expect(...).
[[nodiscard]] Tensor MakeKV(int layer, std::int64_t start, std::int64_t count,
                            const CacheGeometry& geom, int kv) {
  Tensor b = Unwrap(ops::zeros(Shape{count, geom.num_kv_heads, geom.head_dim},
                               DataType::kFloat32));
  auto* p = b.data_ptr<float>();
  const std::int64_t hkv = geom.num_kv_heads;
  const std::int64_t d = geom.head_dim;
  for (std::int64_t t = 0; t < count; ++t) {
    for (std::int64_t h = 0; h < hkv; ++h) {
      for (std::int64_t e = 0; e < d; ++e) {
        p[((((t * hkv) + h) * d) + e)] = Expect(layer, start + t, h, e, kv);
      }
    }
  }
  return b;
}

// Append the same token batch to every layer (layer 0 grows, 1..L−1 reuse).
void AppendAllLayers(PagedKvCache& cache, const CacheGeometry& geom,
                     std::int64_t start, std::int64_t count) {
  for (int layer = 0; layer < geom.num_layers; ++layer) {
    ASSERT_TRUE(cache
                    .append(layer, MakeKV(layer, start, count, geom, 0),
                            MakeKV(layer, start, count, geom, 1))
                    .ok());
  }
}

// Read every committed (pos, head, elem) of `layer` through paged_view and
// check it equals the appended value — for both K and V.
void VerifyLayer(const PagedKvCache& cache, const CacheGeometry& geom,
                 int layer, std::int64_t expected_len) {
  const PagedKvView pv = Unwrap(cache.paged_view(layer));
  ASSERT_EQ(pv.length, expected_len);
  ASSERT_EQ(pv.block_size, kBs);
  ASSERT_EQ(pv.num_blocks, (expected_len + kBs - 1) / kBs);
  const std::int64_t head_stride =
      static_cast<std::int64_t>(kBs) * geom.head_dim;  // bs·d
  for (std::int64_t pos = 0; pos < expected_len; ++pos) {
    const std::int32_t block = pv.block_table[pos / kBs];
    const std::int64_t p = pos % kBs;
    for (std::int64_t h = 0; h < geom.num_kv_heads; ++h) {
      const std::int64_t base =
          (block * pv.block_stride) + (h * head_stride) + (p * geom.head_dim);
      for (std::int64_t e = 0; e < geom.head_dim; ++e) {
        EXPECT_EQ(pv.k_slab[base + e], Expect(layer, pos, h, e, 0))
            << "K layer " << layer << " pos " << pos << " h " << h << " e "
            << e;
        EXPECT_EQ(pv.v_slab[base + e], Expect(layer, pos, h, e, 1))
            << "V layer " << layer << " pos " << pos << " h " << h << " e "
            << e;
      }
    }
  }
}

// Check `cache.view(layer)` (the contiguous gather, M8-T06) returns a
// head-major [Hkv, expected_len, d] snapshot whose every element equals the
// appended value — for both K and V.
void VerifyView(const PagedKvCache& cache, const CacheGeometry& geom, int layer,
                std::int64_t expected_len) {
  const engine::kvcache::KvView kv = Unwrap(cache.view(layer));
  ASSERT_EQ(kv.k.shape().rank(), 3);
  ASSERT_EQ(kv.k.shape().dim(0), geom.num_kv_heads);
  ASSERT_EQ(kv.k.shape().dim(1), expected_len);
  ASSERT_EQ(kv.k.shape().dim(2), geom.head_dim);
  ASSERT_TRUE(kv.k.is_contiguous());
  ASSERT_TRUE(kv.v.is_contiguous());
  const float* kp = kv.k.data_ptr<float>();
  const float* vp = kv.v.data_ptr<float>();
  const std::int64_t d = geom.head_dim;
  for (std::int64_t h = 0; h < geom.num_kv_heads; ++h) {
    for (std::int64_t pos = 0; pos < expected_len; ++pos) {
      for (std::int64_t e = 0; e < d; ++e) {
        const std::int64_t idx = (((h * expected_len) + pos) * d) + e;
        EXPECT_EQ(kp[idx], Expect(layer, pos, h, e, 0))
            << "K layer " << layer << " pos " << pos << " h " << h;
        EXPECT_EQ(vp[idx], Expect(layer, pos, h, e, 1))
            << "V layer " << layer << " pos " << pos << " h " << h;
      }
    }
  }
}

// ===========================================================================
// Construction, geometry, empty state.
// ===========================================================================

TEST(PagedKvCacheTest, ConstructsEmpty) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  const PagedKvCache cache(&pool);
  EXPECT_EQ(cache.length(), 0);
  EXPECT_EQ(cache.geometry().num_layers, geom.num_layers);
  EXPECT_EQ(cache.geometry().num_kv_heads, geom.num_kv_heads);
  EXPECT_EQ(cache.geometry().head_dim, geom.head_dim);
  // Advisory capacity: no owned blocks yet, 8 free blocks * bs.
  EXPECT_EQ(cache.capacity(), 8 * kBs);
}

// ===========================================================================
// The §7.2 walk: prefill grows on layer 0, decode reuses / allocates, values
// land at the hand-worked slots, and all layers stay in agreement.
// ===========================================================================

TEST(PagedKvCacheTest, PrefillThenDecodeWalk) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PrimeFreeList(pool, /*first=*/5, /*second=*/2);  // blocks_ becomes [5, 2]
  PagedKvCache cache(&pool);

  // Prefill 12 tokens (positions 0..11): layer 0 allocates 2 blocks.
  const std::int64_t free_before = pool.free_blocks();
  AppendAllLayers(cache, geom, /*start=*/0, /*count=*/12);
  EXPECT_EQ(cache.length(), 12);
  EXPECT_EQ(pool.free_blocks(), free_before - 2);  // only layer 0 grew

  // Block table is [5, 2]; hand-worked slots match paged_view.
  const PagedKvView pv0 = Unwrap(cache.paged_view(0));
  ASSERT_EQ(pv0.num_blocks, 2);
  EXPECT_EQ(pv0.block_table[0], 5);
  EXPECT_EQ(pv0.block_table[1], 2);
  VerifyLayer(cache, geom, 0, 12);
  VerifyLayer(cache, geom, 1, 12);

  // Decode one token at position 12 (block 2, no allocation).
  const std::int64_t free_mid = pool.free_blocks();
  AppendAllLayers(cache, geom, /*start=*/12, /*count=*/1);
  EXPECT_EQ(cache.length(), 13);
  EXPECT_EQ(pool.free_blocks(), free_mid);  // no new block
  VerifyLayer(cache, geom, 0, 13);
  VerifyLayer(cache, geom, 1, 13);

  // Decode tokens up to position 15 (fills block 2), then position 16 crosses
  // into a fresh third block.
  AppendAllLayers(cache, geom, /*start=*/13, /*count=*/3);  // -> len 16
  EXPECT_EQ(cache.length(), 16);
  EXPECT_EQ(cache.block_table().num_blocks(), 2);
  const std::int64_t free_pre_cross = pool.free_blocks();
  AppendAllLayers(cache, geom, /*start=*/16, /*count=*/1);  // -> len 17
  EXPECT_EQ(cache.length(), 17);
  EXPECT_EQ(cache.block_table().num_blocks(), 3);
  EXPECT_EQ(pool.free_blocks(), free_pre_cross - 1);  // one new block
  VerifyLayer(cache, geom, 0, 17);
  VerifyLayer(cache, geom, 1, 17);
}

// Token-by-token appends produce the same stored bytes as one batched prefill.
TEST(PagedKvCacheTest, TokenByTokenMatchesBatched) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool_a = MakePool(geom, 8);
  BlockPool pool_b = MakePool(geom, 8);
  PagedKvCache batched(&pool_a);
  PagedKvCache stepwise(&pool_b);

  AppendAllLayers(batched, geom, 0, 20);
  for (std::int64_t pos = 0; pos < 20; ++pos) {
    AppendAllLayers(stepwise, geom, pos, 1);
  }

  EXPECT_EQ(batched.length(), stepwise.length());
  for (int layer = 0; layer < geom.num_layers; ++layer) {
    VerifyLayer(batched, geom, layer, 20);
    VerifyLayer(stepwise, geom, layer, 20);
  }
}

// A larger geometry (more heads, wider d) round-trips too.
TEST(PagedKvCacheTest, WiderGeometryRoundTrips) {
  const CacheGeometry geom{.num_layers = 3,
                           .num_kv_heads = 4,
                           .head_dim = 24,
                           .dtype = DataType::kFloat32};
  BlockPool pool = MakePool(geom, 16);
  PagedKvCache cache(&pool);
  AppendAllLayers(cache, geom, 0, 30);  // spans multiple blocks at bs=8
  for (int layer = 0; layer < geom.num_layers; ++layer) {
    VerifyLayer(cache, geom, layer, 30);
  }
}

// ===========================================================================
// Append validation — each rejected append leaves the cache/pool untouched.
// ===========================================================================

TEST(PagedKvCacheTest, AppendValidationFrontLoaded) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  const std::int64_t free0 = pool.free_blocks();

  auto ok_block = [&](std::int64_t t) { return MakeKV(0, 0, t, geom, 0); };

  // Out-of-range layer (would be layer 0's slot but index is invalid).
  EXPECT_TRUE(IsInvalidArgument(cache.append(2, ok_block(2), ok_block(2))));
  // Wrong rank.
  const Tensor rank2 =
      Unwrap(ops::zeros(Shape{2, geom.head_dim}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, rank2, rank2)));
  // Wrong Hkv / d.
  const Tensor bad_hkv = Unwrap(ops::zeros(
      Shape{2, geom.num_kv_heads + 1, geom.head_dim}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, bad_hkv, bad_hkv)));
  const Tensor bad_d = Unwrap(ops::zeros(
      Shape{2, geom.num_kv_heads, geom.head_dim + 1}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, bad_d, bad_d)));
  // T == 0.
  const Tensor empty = Unwrap(ops::zeros(
      Shape{0, geom.num_kv_heads, geom.head_dim}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, empty, empty)));
  // Wrong dtype.
  const Tensor f16 = Unwrap(ops::zeros(
      Shape{2, geom.num_kv_heads, geom.head_dim}, DataType::kFloat16));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, f16, f16)));
  // Non-contiguous (inner slice of a [2, Hkv, 2d] block).
  const Tensor wide = Unwrap(ops::zeros(
      Shape{2, geom.num_kv_heads, 2 * static_cast<std::int64_t>(geom.head_dim)},
      DataType::kFloat32));
  const Tensor strided = Unwrap(wide.slice(2, 0, geom.head_dim));
  ASSERT_FALSE(strided.is_contiguous());
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, strided, strided)));
  // k and v disagree on T.
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, ok_block(2), ok_block(3))));

  // Nothing advanced.
  EXPECT_EQ(cache.length(), 0);
  EXPECT_EQ(pool.free_blocks(), free0);
}

// ===========================================================================
// The per-forward layer protocol (§8.2).
// ===========================================================================

TEST(PagedKvCacheTest, LayerProtocolEnforced) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);

  // Layer 1 before layer 0 -> InvalidArgument (expects layer 0 next).
  EXPECT_TRUE(IsInvalidArgument(
      cache.append(1, MakeKV(1, 0, 2, geom, 0), MakeKV(1, 0, 2, geom, 1))));
  EXPECT_EQ(cache.length(), 0);

  // Open a forward at layer 0 (2 tokens).
  ASSERT_TRUE(
      cache.append(0, MakeKV(0, 0, 2, geom, 0), MakeKV(0, 0, 2, geom, 1)).ok());
  // Repeating layer 0 mid-forward -> expects layer 1 next.
  EXPECT_TRUE(IsInvalidArgument(
      cache.append(0, MakeKV(0, 0, 2, geom, 0), MakeKV(0, 0, 2, geom, 1))));
  // Layer 1 with a mismatched token batch (3 != 2) -> InvalidArgument.
  EXPECT_TRUE(IsInvalidArgument(
      cache.append(1, MakeKV(1, 0, 3, geom, 0), MakeKV(1, 0, 3, geom, 1))));
  // Layer 1 with the right batch completes the forward.
  ASSERT_TRUE(
      cache.append(1, MakeKV(1, 0, 2, geom, 0), MakeKV(1, 0, 2, geom, 1)).ok());
  EXPECT_EQ(cache.length(), 2);

  // A new forward may open at layer 0 again.
  ASSERT_TRUE(
      cache.append(0, MakeKV(0, 2, 1, geom, 0), MakeKV(0, 2, 1, geom, 1)).ok());
  ASSERT_TRUE(
      cache.append(1, MakeKV(1, 2, 1, geom, 0), MakeKV(1, 2, 1, geom, 1)).ok());
  EXPECT_EQ(cache.length(), 3);
  VerifyLayer(cache, geom, 0, 3);
  VerifyLayer(cache, geom, 1, 3);
}

// Single-layer model: layer 0 is always the next expected layer, so repeated
// appends each open and close a forward.
TEST(PagedKvCacheTest, SingleLayerRepeatedAppends) {
  const CacheGeometry geom{.num_layers = 1,
                           .num_kv_heads = 2,
                           .head_dim = 16,
                           .dtype = DataType::kFloat32};
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  for (std::int64_t pos = 0; pos < 5; ++pos) {
    ASSERT_TRUE(
        cache.append(0, MakeKV(0, pos, 1, geom, 0), MakeKV(0, pos, 1, geom, 1))
            .ok());
  }
  EXPECT_EQ(cache.length(), 5);
  VerifyLayer(cache, geom, 0, 5);
}

// ===========================================================================
// Exhaustion at layer 0: nothing written, state unchanged, recovery works.
// ===========================================================================

TEST(PagedKvCacheTest, ExhaustionAtLayerZeroLeavesStateUnchanged) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 1);  // one block: 8 token slots
  PagedKvCache cache(&pool);
  // Fill the single block.
  AppendAllLayers(cache, geom, 0, kBs);
  EXPECT_EQ(cache.length(), kBs);
  EXPECT_EQ(pool.free_blocks(), 0);

  // A next append needs a second block; layer 0 -> ResourceExhausted.
  const std::int64_t len_before = cache.length();
  EXPECT_TRUE(IsResourceExhausted(
      cache.append(0, MakeKV(0, kBs, 1, geom, 0), MakeKV(0, kBs, 1, geom, 1))));
  EXPECT_EQ(cache.length(), len_before);
  EXPECT_EQ(pool.free_blocks(), 0);

  // Truncate frees the block; a smaller append then succeeds.
  ASSERT_TRUE(cache.truncate(0).ok());
  EXPECT_EQ(pool.free_blocks(), 1);
  AppendAllLayers(cache, geom, 0, 4);
  EXPECT_EQ(cache.length(), 4);
  VerifyLayer(cache, geom, 0, 4);
}

// ===========================================================================
// Truncate / reset.
// ===========================================================================

TEST(PagedKvCacheTest, TruncateReleasesBlocksAndKeepsTail) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  AppendAllLayers(cache, geom, 0, 20);  // 3 blocks (bs=8)
  EXPECT_EQ(cache.block_table().num_blocks(), 3);
  const std::int64_t used_before = pool.stats().used;

  // Truncate to 10 tokens -> ⌈10/8⌉ = 2 blocks kept, one released.
  ASSERT_TRUE(cache.truncate(10).ok());
  EXPECT_EQ(cache.length(), 10);
  EXPECT_EQ(cache.block_table().num_blocks(), 2);
  EXPECT_EQ(pool.stats().used, used_before - 1);
  VerifyLayer(cache, geom, 0, 10);  // surviving tokens intact

  // Re-append overwrites the partial tail slot correctly.
  AppendAllLayers(cache, geom, 10, 2);  // -> len 12, still 2 blocks
  EXPECT_EQ(cache.length(), 12);
  EXPECT_EQ(cache.block_table().num_blocks(), 2);
  VerifyLayer(cache, geom, 0, 12);

  // Out-of-range truncate rejected, state untouched.
  EXPECT_TRUE(IsInvalidArgument(cache.truncate(13)));
  EXPECT_TRUE(IsInvalidArgument(cache.truncate(-1)));
  EXPECT_EQ(cache.length(), 12);

  // reset() returns everything.
  cache.reset();
  EXPECT_EQ(cache.length(), 0);
  EXPECT_EQ(cache.block_table().num_blocks(), 0);
  EXPECT_EQ(pool.stats().used, 0);
}

// Truncate mid-forward resets the layer protocol.
TEST(PagedKvCacheTest, TruncateMidForwardResetsProtocol) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  ASSERT_TRUE(
      cache.append(0, MakeKV(0, 0, 2, geom, 0), MakeKV(0, 0, 2, geom, 1)).ok());
  // Truncate after a layer-0-only append; the protocol resets to expect layer
  // 0.
  ASSERT_TRUE(cache.truncate(0).ok());
  ASSERT_TRUE(
      cache.append(0, MakeKV(0, 0, 1, geom, 0), MakeKV(0, 0, 1, geom, 1)).ok());
  ASSERT_TRUE(
      cache.append(1, MakeKV(1, 0, 1, geom, 0), MakeKV(1, 0, 1, geom, 1)).ok());
  EXPECT_EQ(cache.length(), 1);
}

// ===========================================================================
// paged_view / view surfaces.
// ===========================================================================

// ===========================================================================
// view(layer): the contiguous gather (M8-T06, GatherLayerKV) — the prefill read
// path. Returns a fresh head-major [Hkv, len, d] snapshot of the committed
// history.
// ===========================================================================

TEST(PagedKvCacheTest, ViewGathersCommittedHistory) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  // Cross several block boundaries so the gather walks a multi-block table.
  AppendAllLayers(cache, geom, 0, 20);  // bs=8 -> 3 blocks
  VerifyView(cache, geom, 0, 20);
  VerifyView(cache, geom, 1, 20);
}

TEST(PagedKvCacheTest, ViewMatchesPagedViewElementwise) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  AppendAllLayers(cache, geom, 0, 11);  // partial tail block
  // Both VerifyView and VerifyLayer check the same Expect() values — view via
  // the gather, paged_view via the raw slabs — so agreement is transitive.
  VerifyView(cache, geom, 0, 11);
  VerifyLayer(cache, geom, 0, 11);
}

TEST(PagedKvCacheTest, ViewOnEmptyCacheIsZeroLength) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  const PagedKvCache cache(&pool);
  const engine::kvcache::KvView kv = Unwrap(cache.view(0));
  EXPECT_EQ(kv.k.shape().dim(0), geom.num_kv_heads);
  EXPECT_EQ(kv.k.shape().dim(1), 0);
  EXPECT_EQ(kv.k.shape().dim(2), geom.head_dim);
}

TEST(PagedKvCacheTest, ViewRejectsBadLayer) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  const PagedKvCache cache(&pool);
  EXPECT_TRUE(IsInvalidArgument(cache.view(-1).status()));
  EXPECT_TRUE(IsInvalidArgument(cache.view(geom.num_layers).status()));
}

// A view is a fresh snapshot: a later append does not mutate a view taken
// earlier (it owns its storage, not a slab alias).
TEST(PagedKvCacheTest, ViewIsAFreshSnapshot) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  AppendAllLayers(cache, geom, 0, 4);
  const engine::kvcache::KvView snap = Unwrap(cache.view(0));
  ASSERT_EQ(snap.k.shape().dim(1), 4);
  // Append more; the earlier snapshot's length and bytes are unchanged.
  AppendAllLayers(cache, geom, 4, 4);
  EXPECT_EQ(snap.k.shape().dim(1), 4);
  const float* kp = snap.k.data_ptr<float>();
  const std::int64_t d = geom.head_dim;
  for (std::int64_t h = 0; h < geom.num_kv_heads; ++h) {
    for (std::int64_t pos = 0; pos < 4; ++pos) {
      for (std::int64_t e = 0; e < d; ++e) {
        const std::int64_t idx = (((h * 4) + pos) * d) + e;
        EXPECT_EQ(kp[idx], Expect(0, pos, h, e, 0));
      }
    }
  }
}

// Mid-forward, a layer whose K/V has not been appended yet exposes only its
// previously committed length — matching SimpleKvCache's per-layer fill (so a
// view never returns unwritten slot bytes).
TEST(PagedKvCacheTest, ViewMidForwardUsesPerLayerVisibleLength) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  AppendAllLayers(cache, geom, 0, 5);  // both layers committed at 5
  // Open a new forward: append layer 0 only (positions 5..7).
  ASSERT_TRUE(
      cache.append(0, MakeKV(0, 5, 3, geom, 0), MakeKV(0, 5, 3, geom, 1)).ok());
  // Layer 0 sees the new tokens (8), layer 1 still sees the committed 5.
  EXPECT_EQ(Unwrap(cache.view(0)).k.shape().dim(1), 8);
  EXPECT_EQ(Unwrap(cache.view(1)).k.shape().dim(1), 5);
  VerifyView(cache, geom, 0, 8);
  VerifyView(cache, geom, 1, 5);
  // Complete the forward; both layers now see 8.
  ASSERT_TRUE(
      cache.append(1, MakeKV(1, 5, 3, geom, 0), MakeKV(1, 5, 3, geom, 1)).ok());
  VerifyView(cache, geom, 0, 8);
  VerifyView(cache, geom, 1, 8);
}

TEST(PagedKvCacheTest, ViewAfterTruncate) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  AppendAllLayers(cache, geom, 0, 20);
  ASSERT_TRUE(cache.truncate(6).ok());
  VerifyView(cache, geom, 0, 6);
  VerifyView(cache, geom, 1, 6);
}

TEST(PagedKvCacheTest, PagedViewRejectsBadLayer) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  const PagedKvCache cache(&pool);
  EXPECT_TRUE(IsInvalidArgument(cache.paged_view(-1).status()));
  EXPECT_TRUE(IsInvalidArgument(cache.paged_view(geom.num_layers).status()));
  // A valid layer on an empty cache is a well-formed zero-length view.
  const PagedKvView pv = Unwrap(cache.paged_view(0));
  EXPECT_EQ(pv.length, 0);
  EXPECT_EQ(pv.num_blocks, 0);
}

// ===========================================================================
// RAII, move, and pool sharing.
// ===========================================================================

TEST(PagedKvCacheTest, DestructionReturnsBlocks) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  {
    PagedKvCache cache(&pool);
    AppendAllLayers(cache, geom, 0, 20);
    EXPECT_GT(pool.stats().used, 0);
  }
  EXPECT_EQ(pool.stats().used, 0);  // no leaked blocks
}

TEST(PagedKvCacheTest, MoveLeavesSourceEmpty) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache src(&pool);
  AppendAllLayers(src, geom, 0, 10);
  const std::int64_t used = pool.stats().used;

  const PagedKvCache dst(std::move(src));
  EXPECT_EQ(dst.length(), 10);
  EXPECT_EQ(pool.stats().used, used);  // moved, not double-released
  VerifyLayer(dst, geom, 0, 10);
}

TEST(PagedKvCacheTest, TwoCachesShareOnePoolDisjointly) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 16);
  PagedKvCache a(&pool);
  PagedKvCache b(&pool);
  AppendAllLayers(a, geom, 0, 12);
  AppendAllLayers(b, geom, 0, 12);
  // Each reads back its own values (Expect is layer/pos-keyed, identical
  // inputs, but they must not clobber each other's blocks — distinct physical
  // blocks).
  VerifyLayer(a, geom, 0, 12);
  VerifyLayer(b, geom, 0, 12);
  const auto blocks_a = a.block_table().blocks();
  const auto blocks_b = b.block_table().blocks();
  for (const std::int32_t ba : blocks_a) {
    for (const std::int32_t bb : blocks_b) {
      EXPECT_NE(ba, bb);
    }
  }
}

// Usable through the base-class reference (the consumer's view).
TEST(PagedKvCacheTest, UsableThroughInterface) {
  const CacheGeometry geom = TinyGeom();
  BlockPool pool = MakePool(geom, 8);
  PagedKvCache cache(&pool);
  KvCache& iface = cache;
  ASSERT_TRUE(
      iface.append(0, MakeKV(0, 0, 3, geom, 0), MakeKV(0, 0, 3, geom, 1)).ok());
  ASSERT_TRUE(
      iface.append(1, MakeKV(1, 0, 3, geom, 0), MakeKV(1, 0, 3, geom, 1)).ok());
  EXPECT_EQ(iface.length(), 3);
  EXPECT_EQ(iface.geometry().num_layers, geom.num_layers);
  EXPECT_TRUE(iface.paged_view(0).ok());  // the paged override is reached
}

}  // namespace
