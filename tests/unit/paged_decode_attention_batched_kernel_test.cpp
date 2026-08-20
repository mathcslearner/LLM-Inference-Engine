#include "common/paths.h"
#include "kernels/attention.h"
#include "kernels/dispatch.h"
#include "kernels/internal/attention_impl.h"
#include "kernels/internal/paged_attention_common.h"
#include "kernels/paged_attention.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_cache.h"
#include "memory/allocator.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Batched paged decode-attention kernel (M9-T07; design:
// docs/design/scheduler-runtime.md §8.4). Decodes B sequences (one query per
// head each) together, reading K/V through per-sequence block tables over the
// shared per-layer slabs. The acceptance criterion is **bitwise** equality with
// a sequential single-sequence PagedDecodeAttentionF32 of each member —
// heterogeneous cache lengths (including exact block-boundary lengths, the
// stale-snapshot case), GQA ratios, head dims, batch sizes, and thread counts.
// Reuses the same per-ISA PagedDecodeUnits as the single-sequence kernel, so
// the forced-scalar pass exercises the shipped bytes: registered SCALAR_PASS.
namespace engine::kernels {
namespace {

namespace ops = engine::tensor::ops;
using engine::core::StatusOr;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

// One member's decode problem: q [1, H, d], logical k/v [Hkv, L, d].
struct Problem {
  Tensor q;
  Tensor k;
  Tensor v;
  std::int64_t heads;
  std::int64_t kv_heads;
  std::int64_t d;
  std::int64_t l_dim;
  float scale;
};

[[nodiscard]] Problem MakeProblem(std::int64_t heads, std::int64_t kv_heads,
                                  std::int64_t d, std::int64_t l_dim,
                                  double std_dev, std::uint64_t seed) {
  Problem p{
      .q = Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32)),
      .k = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
      .v = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
      .heads = heads,
      .kv_heads = kv_heads,
      .d = d,
      .l_dim = l_dim,
      .scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)))};
  EXPECT_TRUE(ops::fill_normal(p.q, 0.0, std_dev, seed).ok());
  EXPECT_TRUE(ops::fill_normal(p.k, 0.0, std_dev, seed + 1).ok());
  EXPECT_TRUE(ops::fill_normal(p.v, 0.0, std_dev, seed + 2).ok());
  return p;
}

// The single-sequence contiguous decode kernel — the per-member bit-identity
// reference.
[[nodiscard]] Tensor ContiguousDecode(const Problem& p) {
  Tensor out = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  DecodeAttentionF32(p.q.data_ptr<float>(), p.k.data_ptr<float>(),
                     p.v.data_ptr<float>(), out.data_ptr<float>(), p.heads,
                     p.kv_heads, p.d, p.l_dim, p.scale);
  return out;
}

// A test-local paged materialization of B members' logical K/V into ONE shared
// pair of slabs (the batched contract: all sequences share one BlockPool). Each
// member gets its own block table; physical block ids are handed out in
// descending order across all members so logical order != physical order and
// members interleave. Unused physical blocks and every tail slot past a
// member's length are poisoned, so the kernel must honour the per-member tables
// and lengths, never a contiguous fallback.
struct BatchedLayout {
  std::vector<float> k_slab;
  std::vector<float> v_slab;
  std::vector<std::vector<std::int32_t>> tables;  // per member
  std::vector<const std::int32_t*> table_ptrs;
  std::vector<std::int64_t> lengths;
  std::int64_t num_blocks = 0;
  std::int64_t block_stride = 0;
};

[[nodiscard]] BatchedLayout MakeBatchedLayout(const std::vector<Problem>& probs,
                                              std::int64_t bs,
                                              std::int64_t extra_blocks) {
  const std::int64_t hkv = probs[0].kv_heads;
  const std::int64_t d = probs[0].d;
  const std::int64_t block_stride = hkv * bs * d;
  const auto num = static_cast<std::int64_t>(probs.size());

  std::vector<std::int64_t> needed(static_cast<std::size_t>(num));
  std::int64_t total_needed = 0;
  for (std::int64_t m = 0; m < num; ++m) {
    needed[static_cast<std::size_t>(m)] =
        (probs[static_cast<std::size_t>(m)].l_dim + bs - 1) / bs;
    total_needed += needed[static_cast<std::size_t>(m)];
  }
  const std::int64_t num_blocks = total_needed + extra_blocks;
  const float kPoison = 1e30F;

  BatchedLayout lay;
  lay.num_blocks = num_blocks;
  lay.block_stride = block_stride;
  lay.k_slab.assign(static_cast<std::size_t>(num_blocks * block_stride),
                    kPoison);
  lay.v_slab.assign(static_cast<std::size_t>(num_blocks * block_stride),
                    kPoison);
  lay.tables.resize(static_cast<std::size_t>(num));
  lay.lengths.resize(static_cast<std::size_t>(num));

  // Hand out physical ids descending; the first `extra_blocks` (0..extra-1)
  // stay unused (poisoned).
  std::int64_t next_phys = num_blocks - 1;
  for (std::int64_t m = 0; m < num; ++m) {
    const Problem& p = probs[static_cast<std::size_t>(m)];
    auto& table = lay.tables[static_cast<std::size_t>(m)];
    table.resize(static_cast<std::size_t>(needed[static_cast<std::size_t>(m)]));
    for (std::int64_t lb = 0; lb < needed[static_cast<std::size_t>(m)]; ++lb) {
      table[static_cast<std::size_t>(lb)] =
          static_cast<std::int32_t>(next_phys--);
    }
    // Scatter member m's logical key s → its physical block, slot s%bs.
    const float* k = p.k.data_ptr<float>();
    const float* v = p.v.data_ptr<float>();
    for (std::int64_t h = 0; h < hkv; ++h) {
      for (std::int64_t s = 0; s < p.l_dim; ++s) {
        const std::int64_t phys = table[static_cast<std::size_t>(s / bs)];
        const std::int64_t off =
            (phys * block_stride) + (h * bs * d) + ((s % bs) * d);
        const float* ksrc = k + (((h * p.l_dim) + s) * d);
        const float* vsrc = v + (((h * p.l_dim) + s) * d);
        std::copy(ksrc, ksrc + d,
                  lay.k_slab.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(vsrc, vsrc + d,
                  lay.v_slab.begin() + static_cast<std::ptrdiff_t>(off));
      }
    }
    lay.lengths[static_cast<std::size_t>(m)] = p.l_dim;
  }
  // Collect table pointers after all inner vectors are built (stable data()).
  lay.table_ptrs.reserve(static_cast<std::size_t>(num));
  for (std::int64_t m = 0; m < num; ++m) {
    lay.table_ptrs.push_back(lay.tables[static_cast<std::size_t>(m)].data());
  }
  return lay;
}

// Runs the batched kernel over a concatenated [B, H, d] q/out; returns out.
[[nodiscard]] Tensor BatchedDecode(const std::vector<Problem>& probs,
                                   const BatchedLayout& lay, std::int64_t bs) {
  const auto num = static_cast<std::int64_t>(probs.size());
  const std::int64_t heads = probs[0].heads;
  const std::int64_t kv_heads = probs[0].kv_heads;
  const std::int64_t d = probs[0].d;
  const Tensor q = Unwrap(ops::zeros(Shape{num, heads, d}, DataType::kFloat32));
  Tensor out = Unwrap(ops::zeros(Shape{num, heads, d}, DataType::kFloat32));
  for (std::int64_t m = 0; m < num; ++m) {
    const float* src = probs[static_cast<std::size_t>(m)].q.data_ptr<float>();
    std::copy(src, src + (heads * d), q.data_ptr<float>() + (m * heads * d));
  }
  PagedDecodeAttentionBatchedF32(
      q.data_ptr<float>(), lay.k_slab.data(), lay.v_slab.data(),
      lay.table_ptrs.data(), lay.lengths.data(), num, heads, kv_heads, d, bs,
      lay.block_stride, probs[0].scale, out.data_ptr<float>());
  return out;
}

void ExpectMembersBitExact(const std::vector<Problem>& probs,
                           const BatchedLayout& lay, std::int64_t bs,
                           const char* label) {
  const Tensor out = BatchedDecode(probs, lay, bs);
  const auto num = static_cast<std::int64_t>(probs.size());
  for (std::int64_t m = 0; m < num; ++m) {
    const Tensor single = ContiguousDecode(probs[static_cast<std::size_t>(m)]);
    const Tensor row = Unwrap(out.slice(0, m, m + 1));
    EXPECT_TRUE(Unwrap(ops::allclose(row, single, 0.0, 0.0)).allclose)
        << label << " member " << m << " not bit-identical to single-sequence";
  }
}

// (1) Bit-exact vs per-member single-sequence decode across block sizes and
// heterogeneous cache lengths, including exact block-boundary lengths and
// many-block spans — the M9-T07 acceptance. H=4, Hkv=2, d=64.
TEST(PagedDecodeBatchedKernelTest, BitExactVsPerMemberAcrossLengths) {
  const std::int64_t block_sizes[] = {8, 16, 32, 64};
  std::uint64_t seed = 1000;
  for (const std::int64_t bs : block_sizes) {
    // Heterogeneous lengths within one batch, including exact multiples of bs
    // (block-boundary) and many-block spans.
    std::vector<Problem> probs;
    const std::int64_t lens[] = {1, bs, (2 * bs) + 1, 63, 128, 300};
    for (const std::int64_t l : lens) {
      probs.push_back(MakeProblem(4, 2, 64, l, 1.0, seed += 10));
    }
    const BatchedLayout lay = MakeBatchedLayout(probs, bs, /*extra_blocks=*/3);
    const std::string label = "bs=" + std::to_string(bs);
    ExpectMembersBitExact(probs, lay, bs, label.c_str());
  }
}

// (2) GQA group sizes {1, 2, full, g=12>chunk} × head dims crossing the vector
// widths (18, 24 force NEON d%4 and AVX2 d%8 tails; 24 = Qwen head_dim), in one
// batch with heterogeneous lengths. bs=16.
TEST(PagedDecodeBatchedKernelTest, GqaConfigsAndHeadDims) {
  struct Cfg {
    std::int64_t heads;
    std::int64_t kv_heads;
  };
  const Cfg cfgs[] = {{.heads = 4, .kv_heads = 4},
                      {.heads = 4, .kv_heads = 2},
                      {.heads = 8, .kv_heads = 1},
                      {.heads = 24, .kv_heads = 2}};  // g = 12 > chunk
  const std::int64_t dims[] = {18, 24, 64, 128};
  std::uint64_t seed = 2000;
  for (const Cfg& c : cfgs) {
    for (const std::int64_t d : dims) {
      std::vector<Problem> probs;
      for (const std::int64_t l : {16, 65, 130}) {
        probs.push_back(
            MakeProblem(c.heads, c.kv_heads, d, l, 1.0, seed += 10));
      }
      const BatchedLayout lay =
          MakeBatchedLayout(probs, 16, /*extra_blocks=*/2);
      const std::string label = "H=" + std::to_string(c.heads) +
                                " Hkv=" + std::to_string(c.kv_heads) +
                                " d=" + std::to_string(d);
      ExpectMembersBitExact(probs, lay, 16, label.c_str());
    }
  }
}

// (3) Batch sizes {1, 2, 5} — the B==1 batch reduces to the single-sequence
// kernel, larger batches interleave block ids across members.
TEST(PagedDecodeBatchedKernelTest, BatchSizes) {
  std::uint64_t seed = 3000;
  for (const std::int64_t num : {1, 2, 5}) {
    std::vector<Problem> probs;
    probs.reserve(static_cast<std::size_t>(num));
    for (std::int64_t m = 0; m < num; ++m) {
      probs.push_back(MakeProblem(4, 2, 64, 40 + (m * 17), 1.5, seed += 10));
    }
    const BatchedLayout lay = MakeBatchedLayout(probs, 16, /*extra_blocks=*/2);
    const std::string label = "B=" + std::to_string(num);
    ExpectMembersBitExact(probs, lay, 16, label.c_str());
  }
}

// Build the PagedDecodeBatchedArgs the public entry builds, to call the variant
// walker directly over a contiguous unit range.
[[nodiscard]] internal::PagedDecodeBatchedArgs ArgsFor(
    const std::vector<Problem>& probs, const BatchedLayout& lay,
    const Tensor& q, std::int64_t bs, Tensor* out) {
  return internal::PagedDecodeBatchedArgs{
      .q = q.data_ptr<float>(),
      .k_slab = lay.k_slab.data(),
      .v_slab = lay.v_slab.data(),
      .block_tables = lay.table_ptrs.data(),
      .lengths = lay.lengths.data(),
      .out = out->data_ptr<float>(),
      .num_seqs = static_cast<std::int64_t>(probs.size()),
      .heads = probs[0].heads,
      .kv_heads = probs[0].kv_heads,
      .d = probs[0].d,
      .group = probs[0].heads / probs[0].kv_heads,
      .block_size = bs,
      .block_stride = lay.block_stride,
      .scale = probs[0].scale};
}

// (4) The threaded public entry is bit-identical to a single serial walk over
// all (sequence, kv-head) units and to an arbitrary manual chunking — thread /
// chunk invariant (each unit's recurrence wholly in one call).
TEST(PagedDecodeBatchedKernelTest, ThreadingIsBitExactVsSerialWalk) {
  std::vector<Problem> probs;
  probs.reserve(3);
  std::uint64_t seed = 5000;
  for (std::int64_t m = 0; m < 3; ++m) {
    probs.push_back(MakeProblem(8, 4, 64, 100 + (m * 33), 1.5, seed += 10));
  }
  const BatchedLayout lay = MakeBatchedLayout(probs, 16, /*extra_blocks=*/2);
  const std::int64_t heads = probs[0].heads;
  const std::int64_t d = probs[0].d;
  const auto num = static_cast<std::int64_t>(probs.size());
  const std::int64_t units = num * probs[0].kv_heads;
  const auto fn = detail::PagedDecodeAttentionVariant(SelectedIsa());

  const Tensor q = Unwrap(ops::zeros(Shape{num, heads, d}, DataType::kFloat32));
  for (std::int64_t m = 0; m < num; ++m) {
    const float* src = probs[static_cast<std::size_t>(m)].q.data_ptr<float>();
    std::copy(src, src + (heads * d), q.data_ptr<float>() + (m * heads * d));
  }

  const Tensor threaded = BatchedDecode(probs, lay, 16);

  Tensor serial = Unwrap(ops::zeros(Shape{num, heads, d}, DataType::kFloat32));
  detail::PagedDecodeBatchedUnits(ArgsFor(probs, lay, q, 16, &serial), fn, 0,
                                  units);

  Tensor chunked = Unwrap(ops::zeros(Shape{num, heads, d}, DataType::kFloat32));
  const internal::PagedDecodeBatchedArgs cargs =
      ArgsFor(probs, lay, q, 16, &chunked);
  const std::int64_t bounds[] = {0, 1, 3, 6, units};
  for (std::size_t i = 0; i + 1 < std::size(bounds); ++i) {
    detail::PagedDecodeBatchedUnits(cargs, fn, bounds[i], bounds[i + 1]);
  }

  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(chunked, serial, 0.0, 0.0)).allclose);
}

// (5) End-to-end through real PagedKvCaches sharing ONE BlockPool: append each
// member token-by-token, then feed each cache's paged_view(layer) into the
// batched kernel — bit-exact vs per-member single-sequence decode. Proves the
// shared-slab / per-member-table contract over real caches.
TEST(PagedDecodeBatchedKernelTest, ThroughSharedPagedKvCaches) {
  constexpr int kLayers = 2;
  constexpr int kHkv = 2;
  constexpr int kHeads = 4;
  constexpr int kD = 64;
  constexpr int kBs = 16;
  const std::vector<std::int64_t> lens = {kBs, 40, (3 * kBs) + 5};
  const auto num = static_cast<std::int64_t>(lens.size());

  std::vector<Problem> probs;
  probs.reserve(lens.size());
  std::uint64_t seed = 6000;
  for (const std::int64_t l : lens) {
    probs.push_back(MakeProblem(kHeads, kHkv, kD, l, 1.0, seed += 10));
  }

  const kvcache::CacheGeometry geom{.num_layers = kLayers,
                                    .num_kv_heads = kHkv,
                                    .head_dim = kD,
                                    .dtype = DataType::kFloat32};
  memory::CpuAllocator alloc;
  auto pool =
      Unwrap(kvcache::BlockPool::Create(geom, kBs, /*num_blocks=*/64, &alloc));

  std::vector<kvcache::PagedKvCache> caches;
  caches.reserve(static_cast<std::size_t>(num));
  for (std::int64_t m = 0; m < num; ++m) {
    caches.emplace_back(&pool);
  }
  // Append token by token, all layers, for every member (interleaves block ids
  // across members in the shared pool).
  for (std::int64_t m = 0; m < num; ++m) {
    const Problem& p = probs[static_cast<std::size_t>(m)];
    for (std::int64_t s = 0; s < p.l_dim; ++s) {
      const Tensor kt =
          Unwrap(ops::zeros(Shape{1, kHkv, kD}, DataType::kFloat32));
      const Tensor vt =
          Unwrap(ops::zeros(Shape{1, kHkv, kD}, DataType::kFloat32));
      for (std::int64_t h = 0; h < kHkv; ++h) {
        const float* ks = p.k.data_ptr<float>() + (((h * p.l_dim) + s) * kD);
        const float* vs = p.v.data_ptr<float>() + (((h * p.l_dim) + s) * kD);
        std::copy(ks, ks + kD, kt.data_ptr<float>() + (h * kD));
        std::copy(vs, vs + kD, vt.data_ptr<float>() + (h * kD));
      }
      for (int layer = 0; layer < kLayers; ++layer) {
        ASSERT_TRUE(
            caches[static_cast<std::size_t>(m)].append(layer, kt, vt).ok());
      }
    }
  }

  const int layer = 1;
  std::vector<const std::int32_t*> tables;
  std::vector<std::int64_t> lengths;
  tables.reserve(static_cast<std::size_t>(num));
  lengths.reserve(static_cast<std::size_t>(num));
  const float* k_slab = nullptr;
  const float* v_slab = nullptr;
  std::int64_t stride = 0;
  for (std::int64_t m = 0; m < num; ++m) {
    const kvcache::PagedKvView pv =
        Unwrap(caches[static_cast<std::size_t>(m)].paged_view(layer));
    if (m == 0) {
      k_slab = pv.k_slab;
      v_slab = pv.v_slab;
      stride = pv.block_stride;
    }
    tables.push_back(pv.block_table);
    lengths.push_back(pv.length);
  }

  const Tensor q =
      Unwrap(ops::zeros(Shape{num, kHeads, kD}, DataType::kFloat32));
  const Tensor out =
      Unwrap(ops::zeros(Shape{num, kHeads, kD}, DataType::kFloat32));
  const std::int64_t qrow = static_cast<std::int64_t>(kHeads) * kD;
  for (std::int64_t m = 0; m < num; ++m) {
    const float* src = probs[static_cast<std::size_t>(m)].q.data_ptr<float>();
    std::copy(src, src + qrow, q.data_ptr<float>() + (m * qrow));
  }
  PagedDecodeAttentionBatchedF32(
      q.data_ptr<float>(), k_slab, v_slab, tables.data(), lengths.data(), num,
      kHeads, kHkv, kD, kBs, stride, probs[0].scale, out.data_ptr<float>());

  for (std::int64_t m = 0; m < num; ++m) {
    const Tensor single = ContiguousDecode(probs[static_cast<std::size_t>(m)]);
    const Tensor row = Unwrap(out.slice(0, m, m + 1));
    EXPECT_TRUE(Unwrap(ops::allclose(row, single, 0.0, 0.0)).allclose)
        << "member " << m;
  }
}

// (6) B == 0 is a no-op (an empty decode pass is legal): the kernel must not
// touch `out`.
TEST(PagedDecodeBatchedKernelTest, EmptyBatchNoOp) {
  const Tensor out = Unwrap(ops::zeros(Shape{1, 4, 64}, DataType::kFloat32));
  EXPECT_TRUE(ops::fill_normal(out, 0.0, 1.0, 42).ok());
  const Tensor before = Unwrap(ops::zeros(Shape{1, 4, 64}, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(before, out).ok());
  const std::int64_t stride = static_cast<std::int64_t>(2) * 16 * 64;
  PagedDecodeAttentionBatchedF32(out.data_ptr<float>(), nullptr, nullptr,
                                 nullptr, nullptr, /*num_seqs=*/0, 4, 2, 64, 16,
                                 stride, 0.125F, out.data_ptr<float>());
  EXPECT_TRUE(Unwrap(ops::allclose(out, before, 0.0, 0.0)).allclose);
}

// (7) A block_size that does not divide kAttnKb violates the bit-identity
// invariant and is a programmer error (CHECK at the entry).
TEST(PagedDecodeBatchedKernelDeathTest, RejectsNonDivisorBlockSize) {
  std::vector<Problem> probs = {MakeProblem(4, 2, 64, 40, 1.0, 7000)};
  const std::int64_t bs = 24;  // does not divide kAttnKb = 64
  const BatchedLayout lay = MakeBatchedLayout(probs, bs, /*extra_blocks=*/0);
  const Tensor q = Unwrap(ops::zeros(Shape{1, 4, 64}, DataType::kFloat32));
  const Tensor out = Unwrap(ops::zeros(Shape{1, 4, 64}, DataType::kFloat32));
  const std::int64_t row = static_cast<std::int64_t>(4) * 64;
  std::copy(probs[0].q.data_ptr<float>(), probs[0].q.data_ptr<float>() + row,
            q.data_ptr<float>());
  EXPECT_DEATH(PagedDecodeAttentionBatchedF32(
                   q.data_ptr<float>(), lay.k_slab.data(), lay.v_slab.data(),
                   lay.table_ptrs.data(), lay.lengths.data(), 1, 4, 2, 64, bs,
                   lay.block_stride, probs[0].scale, out.data_ptr<float>()),
               "block_size");
}

}  // namespace
}  // namespace engine::kernels
