#include "kvcache/simple_cache.h"

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

// SimpleKvCache mechanics (M5-T06; design: docs/design/model-execution.md
// §6.2). Pure cache unit tests: construction validation, the append/view
// transpose, per-layer length semantics, truncate, over-capacity, append-only
// immutability, and per-sequence independence. The end-to-end KV-cache
// *invariant* (token-by-token decode == full recompute) lives in
// attention_test.cpp, where the real tiny-llama attention machinery already
// lives; M5-T07 elevates it to full-model logits.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::IsResourceExhausted;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::kvcache::CacheGeometry;
using engine::kvcache::KvView;
using engine::kvcache::SimpleKvCache;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] CacheGeometry Geom(int layers, int kv_heads, int head_dim) {
  return CacheGeometry{.num_layers = layers,
                       .num_kv_heads = kv_heads,
                       .head_dim = head_dim,
                       .dtype = DataType::kFloat32};
}

// A seeded token-major [T, Hkv, d] block.
[[nodiscard]] Tensor Block(std::int64_t t, std::int64_t hkv, std::int64_t d,
                           std::uint64_t seed) {
  Tensor b = Unwrap(ops::zeros(Shape{t, hkv, d}, DataType::kFloat32));
  EXPECT_TRUE(ops::fill_normal(b, 0.0, 1.0, seed).ok());
  return b;
}

// ===========================================================================
// Construction validation.
// ===========================================================================

TEST(SimpleKvCacheTest, CreateValidatesGeometryAndCapacity) {
  EXPECT_TRUE(SimpleKvCache::Create(Geom(2, 2, 4), 16).ok());

  EXPECT_TRUE(IsInvalidArgument(
      SimpleKvCache::Create(Geom(0, 2, 4), 16).status()));  // layers
  EXPECT_TRUE(IsInvalidArgument(
      SimpleKvCache::Create(Geom(2, 0, 4), 16).status()));  // kv_heads
  EXPECT_TRUE(IsInvalidArgument(
      SimpleKvCache::Create(Geom(2, 2, 0), 16).status()));  // head_dim
  EXPECT_TRUE(IsInvalidArgument(
      SimpleKvCache::Create(Geom(2, 2, 4), 0).status()));  // capacity

  CacheGeometry bad_dtype = Geom(2, 2, 4);
  bad_dtype.dtype = DataType::kBFloat16;
  EXPECT_TRUE(IsUnimplemented(SimpleKvCache::Create(bad_dtype, 16).status()));
}

TEST(SimpleKvCacheTest, FreshCacheReportsGeometryAndZeroLength) {
  const CacheGeometry geom = Geom(3, 2, 4);
  const SimpleKvCache cache = Unwrap(SimpleKvCache::Create(geom, 16));
  EXPECT_EQ(cache.geometry().num_layers, 3);
  EXPECT_EQ(cache.geometry().num_kv_heads, 2);
  EXPECT_EQ(cache.geometry().head_dim, 4);
  EXPECT_EQ(cache.length(), 0);
  EXPECT_EQ(cache.capacity(), 16);
  EXPECT_EQ(cache.layer_length(0), 0);
}

// ===========================================================================
// append -> view: head-major layout.
// ===========================================================================

TEST(SimpleKvCacheTest, AppendThenViewIsHeadMajor) {
  constexpr std::int64_t kHkv = 2;
  constexpr std::int64_t kD = 3;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(1, kHkv, kD), 8));

  const Tensor k1 = Block(2, kHkv, kD, 7);  // T=2
  const Tensor v1 = Block(2, kHkv, kD, 8);
  ASSERT_TRUE(cache.append(0, k1, v1).ok());
  const Tensor k2 = Block(1, kHkv, kD, 9);  // T=1
  const Tensor v2 = Block(1, kHkv, kD, 10);
  ASSERT_TRUE(cache.append(0, k2, v2).ok());

  const KvView kv = Unwrap(cache.view(0));
  ASSERT_EQ(kv.k.shape().dim(0), kHkv);
  ASSERT_EQ(kv.k.shape().dim(1), 3);  // len = 2 + 1
  ASSERT_EQ(kv.k.shape().dim(2), kD);
  EXPECT_TRUE(kv.k.is_contiguous());
  EXPECT_EQ(kv.k.dtype(), DataType::kFloat32);

  // Head-major view[h, s, e] == token-major source[s_within_block, h, e].
  for (std::int64_t h = 0; h < kHkv; ++h) {
    for (std::int64_t e = 0; e < kD; ++e) {
      EXPECT_FLOAT_EQ((kv.k.item<float>({h, 0, e})),
                      (k1.item<float>({0, h, e})));
      EXPECT_FLOAT_EQ((kv.k.item<float>({h, 1, e})),
                      (k1.item<float>({1, h, e})));
      EXPECT_FLOAT_EQ((kv.k.item<float>({h, 2, e})),
                      (k2.item<float>({0, h, e})));
      EXPECT_FLOAT_EQ((kv.v.item<float>({h, 2, e})),
                      (v2.item<float>({0, h, e})));
    }
  }
  EXPECT_EQ(cache.length(), 3);
  EXPECT_EQ(cache.layer_length(0), 3);
}

TEST(SimpleKvCacheTest, ViewOfEmptyLayerIsZeroLength) {
  const SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, 2, 4), 8));
  const KvView kv = Unwrap(cache.view(1));
  EXPECT_EQ(kv.k.shape().dim(0), 2);  // Hkv
  EXPECT_EQ(kv.k.shape().dim(1), 0);  // len
  EXPECT_EQ(kv.k.shape().dim(2), 4);  // d
  EXPECT_EQ(kv.k.numel(), 0);
}

// ===========================================================================
// Malformed appends / views.
// ===========================================================================

TEST(SimpleKvCacheTest, AppendRejectsMalformedBlocks) {
  constexpr std::int64_t kHkv = 2;
  constexpr std::int64_t kD = 4;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, kHkv, kD), 8));
  const Tensor ok = Block(2, kHkv, kD, 1);

  // layer out of range.
  EXPECT_TRUE(IsInvalidArgument(cache.append(-1, ok, ok)));
  EXPECT_TRUE(IsInvalidArgument(cache.append(2, ok, ok)));

  // wrong rank.
  const Tensor rank2 = Unwrap(ops::zeros(Shape{2, kD}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, rank2, rank2)));

  // wrong Hkv / d.
  const Tensor bad_hkv = Block(2, kHkv + 1, kD, 2);
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, bad_hkv, bad_hkv)));
  const Tensor bad_d = Block(2, kHkv, kD + 1, 3);
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, bad_d, bad_d)));

  // T == 0.
  const Tensor empty =
      Unwrap(ops::zeros(Shape{0, kHkv, kD}, DataType::kFloat32));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, empty, empty)));

  // wrong dtype.
  const Tensor f16 = Unwrap(ops::zeros(Shape{2, kHkv, kD}, DataType::kFloat16));
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, f16, f16)));

  // non-contiguous (inner slice of a [2, Hkv, 2d] block).
  const Tensor wide = Block(2, kHkv, 2 * kD, 4);
  const Tensor strided = Unwrap(wide.slice(2, 0, kD));
  ASSERT_FALSE(strided.is_contiguous());
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, strided, strided)));

  // k and v disagree on T.
  const Tensor k_t2 = Block(2, kHkv, kD, 5);
  const Tensor v_t3 = Block(3, kHkv, kD, 6);
  EXPECT_TRUE(IsInvalidArgument(cache.append(0, k_t2, v_t3)));

  // None of the rejected appends advanced the cache.
  EXPECT_EQ(cache.layer_length(0), 0);
  EXPECT_EQ(cache.length(), 0);
}

TEST(SimpleKvCacheTest, ViewRejectsBadLayer) {
  const SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, 2, 4), 8));
  EXPECT_TRUE(IsInvalidArgument(cache.view(-1).status()));
  EXPECT_TRUE(IsInvalidArgument(cache.view(2).status()));
}

// ===========================================================================
// Capacity.
// ===========================================================================

TEST(SimpleKvCacheTest, OverCapacityAppendIsRejectedAndLeavesStateUnchanged) {
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 2;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(1, kHkv, kD), 3));

  ASSERT_TRUE(
      cache.append(0, Block(2, kHkv, kD, 1), Block(2, kHkv, kD, 2)).ok());
  EXPECT_EQ(cache.layer_length(0), 2);
  const KvView before = Unwrap(cache.view(0));

  // 2 + 2 = 4 > capacity 3 -> ResourceExhausted, no write.
  const Tensor k = Block(2, kHkv, kD, 3);
  EXPECT_TRUE(IsResourceExhausted(cache.append(0, k, k)));
  EXPECT_EQ(cache.layer_length(0), 2);
  const KvView after = Unwrap(cache.view(0));
  for (std::int64_t i = 0; i < before.k.numel(); ++i) {
    EXPECT_FLOAT_EQ(before.k.data_ptr<float>()[i],
                    after.k.data_ptr<float>()[i]);
  }

  // Exact-fill append (2 + 1 == 3) succeeds.
  ASSERT_TRUE(
      cache.append(0, Block(1, kHkv, kD, 4), Block(1, kHkv, kD, 5)).ok());
  EXPECT_EQ(cache.layer_length(0), 3);
  EXPECT_EQ(cache.capacity(), 3);
}

// ===========================================================================
// length() is the minimum fill across layers.
// ===========================================================================

TEST(SimpleKvCacheTest, LengthIsMinAcrossLayers) {
  constexpr std::int64_t kHkv = 2;
  constexpr std::int64_t kD = 4;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, kHkv, kD), 8));

  ASSERT_TRUE(
      cache.append(0, Block(3, kHkv, kD, 1), Block(3, kHkv, kD, 2)).ok());
  // Layer 0 has 3, layer 1 still 0 -> committed length 0 (a mid-forward state).
  EXPECT_EQ(cache.layer_length(0), 3);
  EXPECT_EQ(cache.layer_length(1), 0);
  EXPECT_EQ(cache.length(), 0);

  ASSERT_TRUE(
      cache.append(1, Block(3, kHkv, kD, 3), Block(3, kHkv, kD, 4)).ok());
  EXPECT_EQ(cache.length(), 3);
}

// ===========================================================================
// truncate / reset.
// ===========================================================================

TEST(SimpleKvCacheTest, TruncateDropsTailAndResyncsLayers) {
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 2;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, kHkv, kD), 16));
  for (int layer = 0; layer < 2; ++layer) {
    const auto seed = static_cast<std::uint64_t>(layer);
    ASSERT_TRUE(cache
                    .append(layer, Block(5, kHkv, kD, 10 + seed),
                            Block(5, kHkv, kD, 20 + seed))
                    .ok());
  }
  ASSERT_EQ(cache.length(), 5);

  // Out-of-range truncations.
  EXPECT_TRUE(IsInvalidArgument(cache.truncate(-1)));
  EXPECT_TRUE(IsInvalidArgument(cache.truncate(6)));

  // Drop to 3, then continue appending: the layer resumes at 3.
  ASSERT_TRUE(cache.truncate(3).ok());
  EXPECT_EQ(cache.length(), 3);
  EXPECT_EQ(cache.view(0).value().k.shape().dim(1), 3);
  ASSERT_TRUE(
      cache.append(0, Block(2, kHkv, kD, 30), Block(2, kHkv, kD, 31)).ok());
  EXPECT_EQ(cache.layer_length(0), 5);

  // reset() == truncate(0), on both layers.
  cache.reset();
  EXPECT_EQ(cache.length(), 0);
  EXPECT_EQ(cache.layer_length(0), 0);
  EXPECT_EQ(cache.layer_length(1), 0);
}

// truncate re-synchronizes layers that disagree after a partial forward.
TEST(SimpleKvCacheTest, TruncateResyncsAfterPartialForward) {
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 2;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, kHkv, kD), 8));
  ASSERT_TRUE(
      cache.append(0, Block(4, kHkv, kD, 1), Block(4, kHkv, kD, 2)).ok());
  ASSERT_TRUE(
      cache.append(1, Block(2, kHkv, kD, 3), Block(2, kHkv, kD, 4)).ok());
  ASSERT_EQ(cache.length(), 2);  // min(4, 2)

  ASSERT_TRUE(cache.truncate(2).ok());
  EXPECT_EQ(cache.layer_length(0), 2);
  EXPECT_EQ(cache.layer_length(1), 2);
}

// ===========================================================================
// Append-only immutability of committed positions.
// ===========================================================================

TEST(SimpleKvCacheTest, CommittedPositionsAreImmutableAcrossAppends) {
  constexpr std::int64_t kHkv = 2;
  constexpr std::int64_t kD = 3;
  SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(1, kHkv, kD), 16));
  ASSERT_TRUE(
      cache.append(0, Block(4, kHkv, kD, 1), Block(4, kHkv, kD, 2)).ok());
  const KvView snap = Unwrap(cache.view(0));  // [Hkv, 4, d] snapshot

  ASSERT_TRUE(
      cache.append(0, Block(3, kHkv, kD, 3), Block(3, kHkv, kD, 4)).ok());
  const KvView grown = Unwrap(cache.view(0));  // [Hkv, 7, d]
  ASSERT_EQ(grown.k.shape().dim(1), 7);

  // The first 4 positions of every head are byte-for-byte unchanged.
  for (std::int64_t h = 0; h < kHkv; ++h) {
    for (std::int64_t s = 0; s < 4; ++s) {
      for (std::int64_t e = 0; e < kD; ++e) {
        EXPECT_FLOAT_EQ((snap.k.item<float>({h, s, e})),
                        (grown.k.item<float>({h, s, e})));
        EXPECT_FLOAT_EQ((snap.v.item<float>({h, s, e})),
                        (grown.v.item<float>({h, s, e})));
      }
    }
  }
}

// ===========================================================================
// Per-sequence independence.
// ===========================================================================

TEST(SimpleKvCacheTest, TwoCachesAreIndependent) {
  constexpr std::int64_t kHkv = 1;
  constexpr std::int64_t kD = 2;
  SimpleKvCache a = Unwrap(SimpleKvCache::Create(Geom(1, kHkv, kD), 8));
  const SimpleKvCache b = Unwrap(SimpleKvCache::Create(Geom(1, kHkv, kD), 8));
  ASSERT_TRUE(a.append(0, Block(3, kHkv, kD, 1), Block(3, kHkv, kD, 2)).ok());
  EXPECT_EQ(a.length(), 3);
  EXPECT_EQ(b.length(), 0);
}

// The contiguous cache does not support the paged decode fast path (§8.3): the
// interface's default `paged_view` is Unimplemented, so the consumer falls back
// to `view()` + the contiguous decode kernel.
TEST(SimpleKvCacheTest, PagedViewIsUnimplemented) {
  const SimpleKvCache cache = Unwrap(SimpleKvCache::Create(Geom(2, 2, 4), 8));
  EXPECT_TRUE(IsUnimplemented(cache.paged_view(0).status()));
}

}  // namespace
