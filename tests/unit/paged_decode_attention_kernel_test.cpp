#include "common/paths.h"
#include "cpu/ops.h"
#include "kernels/attention.h"
#include "kernels/dispatch.h"
#include "kernels/internal/attention_impl.h"
#include "kernels/internal/paged_attention_common.h"
#include "kernels/paged_attention.h"
#include "kvcache/block_pool.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_cache.h"
#include "memory/allocator.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Paged decode-attention kernel (M8-T05; design: docs/design/paged-kv-cache.md
// §9.2). Decode attention (one query per head over the whole cache) reading K/V
// **through a block table** over the paged slabs. The acceptance criterion is
// **bitwise** equality with the M6-T05 contiguous DecodeAttentionF32 on the
// same logical K/V materialized both ways — for cache lengths crossing many
// blocks and length exactly at a block boundary — across GQA ratios and thread
// counts (design §12). Also replays the HF attention goldens (Class T vs the
// oracle) and drives the kernel end-to-end through a real PagedKvCache. The
// per-ISA PagedDecodeUnits reuses the same Ops as the contiguous kernel, so the
// forced-scalar pass exercises the shipped bytes: registered SCALAR_PASS.
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

constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-4;

// One decode problem: q [1, H, d] (a T = 1 query), logical k/v [Hkv, L, d],
// seeded. q is rank-3 [1, H, d] so the same handle feeds cpu::attention and
// DecodeAttentionF32 (which read it as [H, d]).
struct Problem {
  Tensor q;
  Tensor k;  // [Hkv, L, d] logical, head-major
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

// The M6-T05 contiguous decode kernel — the bit-identity reference.
[[nodiscard]] Tensor ContiguousDecode(const Problem& p) {
  Tensor out = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  DecodeAttentionF32(p.q.data_ptr<float>(), p.k.data_ptr<float>(),
                     p.v.data_ptr<float>(), out.data_ptr<float>(), p.heads,
                     p.kv_heads, p.d, p.l_dim, p.scale);
  return out;
}

// A test-local paged materialization of the logical K/V — independent of
// KvScatterF32 (the M8-T04 kernel). Allocates plain float slabs shaped
// [num_blocks, Hkv, bs, d], a NON-monotonic block table (physical ids permuted,
// some physical blocks left unused), poisons every slot the kernel must not
// read (unused physical blocks + the tail block's slots past `length`), then
// scatters logical key s → block_table[s/bs], slot s%bs. Verifies the kernel
// honours the block table (not a contiguous fallback) and never reads poison.
struct PagedLayout {
  std::vector<float> k_slab;
  std::vector<float> v_slab;
  std::vector<std::int32_t> block_table;
  std::int64_t num_blocks;
  std::int64_t block_stride;
};

[[nodiscard]] PagedLayout MakePagedLayout(const Problem& p, std::int64_t bs,
                                          std::int64_t extra_blocks) {
  const std::int64_t needed = (p.l_dim + bs - 1) / bs;
  const std::int64_t num_blocks = needed + extra_blocks;
  const std::int64_t block_stride = p.kv_heads * bs * p.d;
  const float kPoison = 1e30F;

  PagedLayout lay{
      .k_slab = std::vector<float>(
          static_cast<std::size_t>(num_blocks * block_stride), kPoison),
      .v_slab = std::vector<float>(
          static_cast<std::size_t>(num_blocks * block_stride), kPoison),
      .block_table =
          std::vector<std::int32_t>(static_cast<std::size_t>(needed)),
      .num_blocks = num_blocks,
      .block_stride = block_stride};

  // Reverse-permuted physical assignment so logical order != physical order:
  // logical block lb → physical (num_blocks - 1 - lb). The first `extra_blocks`
  // physical ids are then never referenced (poison stays).
  for (std::int64_t lb = 0; lb < needed; ++lb) {
    lay.block_table[static_cast<std::size_t>(lb)] =
        static_cast<std::int32_t>(num_blocks - 1 - lb);
  }

  const float* k = p.k.data_ptr<float>();
  const float* v = p.v.data_ptr<float>();
  for (std::int64_t h = 0; h < p.kv_heads; ++h) {
    for (std::int64_t s = 0; s < p.l_dim; ++s) {
      const std::int64_t phys =
          lay.block_table[static_cast<std::size_t>(s / bs)];
      const std::int64_t off =
          (phys * block_stride) + (h * bs * p.d) + ((s % bs) * p.d);
      const float* ksrc = k + ((h * p.l_dim + s) * p.d);
      const float* vsrc = v + ((h * p.l_dim + s) * p.d);
      std::copy(ksrc, ksrc + p.d,
                lay.k_slab.begin() + static_cast<std::ptrdiff_t>(off));
      std::copy(vsrc, vsrc + p.d,
                lay.v_slab.begin() + static_cast<std::ptrdiff_t>(off));
    }
  }
  return lay;
}

[[nodiscard]] Tensor PagedDecode(const Problem& p, const PagedLayout& lay,
                                 std::int64_t bs) {
  Tensor out = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  PagedDecodeAttentionF32(p.q.data_ptr<float>(), lay.k_slab.data(),
                          lay.v_slab.data(), lay.block_table.data(),
                          lay.num_blocks, p.l_dim, p.heads, p.kv_heads, p.d, bs,
                          lay.block_stride, p.scale, out.data_ptr<float>());
  return out;
}

void ExpectBitExactVsContiguous(const Problem& p, std::int64_t bs,
                                std::int64_t extra_blocks, const char* label) {
  const PagedLayout lay = MakePagedLayout(p, bs, extra_blocks);
  const Tensor paged = PagedDecode(p, lay, bs);
  const Tensor contiguous = ContiguousDecode(p);
  EXPECT_TRUE(Unwrap(ops::allclose(paged, contiguous, 0.0, 0.0)).allclose)
      << label << ": paged decode not bit-identical to contiguous decode";
}

// (1) Bit-exact vs the contiguous kernel across block sizes and cache lengths,
// including lengths exactly at a block boundary and spanning many blocks — the
// M8-T05 acceptance. H=4, Hkv=2, d=64.
TEST(PagedDecodeAttentionKernelTest, BitExactVsContiguousAcrossLengths) {
  const std::int64_t block_sizes[] = {8, 16, 32, 64};
  std::uint64_t seed = 1000;
  for (const std::int64_t bs : block_sizes) {
    // Lengths straddling block + kAttnKb-unit boundaries; several exact
    // multiples of bs (the "length exactly at a block boundary" case); large
    // many-block spans.
    const std::int64_t lengths[] = {
        1,   bs - 1, bs,  bs + 1, 2 * bs, 63,  64,   65,
        127, 128,    129, 5 * bs, 8 * bs, 511, 2048, 4096};
    for (const std::int64_t l_dim : lengths) {
      if (l_dim < 1) {
        continue;
      }
      const Problem p = MakeProblem(4, 2, 64, l_dim, 1.0, seed += 10);
      const std::string label =
          "bs=" + std::to_string(bs) + " L=" + std::to_string(l_dim);
      ExpectBitExactVsContiguous(p, bs, /*extra_blocks=*/2, label.c_str());
    }
  }
}

// (2) GQA group sizes {1, 2, full, g=12>chunk} × head dims crossing the vector
// widths (18, 24 force NEON d%4 and AVX2 d%8 tails; 24 = Qwen head_dim). Cache
// length crosses several blocks. bs=16.
TEST(PagedDecodeAttentionKernelTest, GqaConfigs) {
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
      const Problem p =
          MakeProblem(c.heads, c.kv_heads, d, 130, 1.0, seed += 10);
      const std::string label = "H=" + std::to_string(c.heads) +
                                " Hkv=" + std::to_string(c.kv_heads) +
                                " d=" + std::to_string(d);
      ExpectBitExactVsContiguous(p, 16, /*extra_blocks=*/3, label.c_str());
    }
  }
}

// (3) Large-magnitude scores across many blocks: the running-max rescale (alpha
// far from 1) must stay bit-exact vs the contiguous kernel and within tolerance
// vs the oracle. std 2 ⇒ logit std ≈4 (§10).
TEST(PagedDecodeAttentionKernelTest, LargeMagnitudeRescale) {
  std::uint64_t seed = 4000;
  for (const std::int64_t l_dim : {512, 2048}) {
    const Problem p = MakeProblem(4, 2, 64, l_dim, 2.0, seed += 10);
    const std::string label = "large L=" + std::to_string(l_dim);
    ExpectBitExactVsContiguous(p, 16, /*extra_blocks=*/2, label.c_str());

    // vs the oracle (cpu::attention, T=1) — ties the paged path to the HF
    // chain.
    const PagedLayout lay = MakePagedLayout(p, 16, 2);
    const Tensor paged = PagedDecode(p, lay, 16);
    Tensor ref = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
    EXPECT_TRUE(engine::cpu::attention(p.q, p.k, p.v, p.scale, ref).ok());
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(paged, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << label << " oracle: " << r.Summary();
    std::cerr << "[paged-decode] " << label
              << " oracle max_abs_diff=" << r.max_abs_diff << '\n';
  }
}

// Build the PagedDecodeArgs the public entry builds, to call a variant directly
// over a contiguous kv-head range.
[[nodiscard]] internal::PagedDecodeArgs ArgsFor(const Problem& p,
                                                const PagedLayout& lay,
                                                std::int64_t bs, float* out) {
  return internal::PagedDecodeArgs{.q = p.q.data_ptr<float>(),
                                   .k_slab = lay.k_slab.data(),
                                   .v_slab = lay.v_slab.data(),
                                   .block_table = lay.block_table.data(),
                                   .out = out,
                                   .d = p.d,
                                   .length = p.l_dim,
                                   .group = p.heads / p.kv_heads,
                                   .block_size = bs,
                                   .block_stride = lay.block_stride,
                                   .scale = p.scale};
}

// (4) The threaded public entry is bit-identical to a single serial variant
// call over all kv-head units — and to an arbitrary manual chunking — so the
// result is invariant to thread count / chunking (each kv head wholly in one
// call).
TEST(PagedDecodeAttentionKernelTest, ThreadingIsBitExactVsSerialVariant) {
  const Problem p = MakeProblem(8, 8, 64, 200, 1.5, 5000);
  const PagedLayout lay = MakePagedLayout(p, 16, 2);
  const std::int64_t units = p.kv_heads;
  const auto fn = detail::PagedDecodeAttentionVariant(SelectedIsa());

  const Tensor threaded = PagedDecode(p, lay, 16);

  const Tensor serial =
      Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  fn(ArgsFor(p, lay, 16, serial.data_ptr<float>()), 0, units);

  const Tensor chunked =
      Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  const internal::PagedDecodeArgs cargs =
      ArgsFor(p, lay, 16, chunked.data_ptr<float>());
  const std::int64_t bounds[] = {0, 1, 3, units};
  for (std::size_t b = 0; b + 1 < std::size(bounds); ++b) {
    fn(cargs, bounds[b], bounds[b + 1]);
  }

  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(chunked, serial, 0.0, 0.0)).allclose);
}

// (5) The vector variant slot is populated on a SIMD host (skipped under the
// forced-scalar pass, where the selected variant IS the scalar one).
TEST(PagedDecodeAttentionKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::PagedDecodeAttentionVariant(isa)),
            reinterpret_cast<void*>(&scalar::PagedDecodeUnits));
}

// (6) End-to-end through a real PagedKvCache: append the logical K/V token by
// token across many blocks via the real append protocol, then feed
// paged_view(layer) to the kernel — bit-exact vs the contiguous decode. Proves
// the PagedKvView field semantics (block_stride, block_size, length) are what
// the kernel consumes (the M8-T07 contract), and that KvScatterF32's writes are
// read back correctly by this kernel. The pool is primed (allocate/release) so
// physical block ids do not coincide with logical ones.
TEST(PagedDecodeAttentionKernelTest, ThroughPagedKvCache) {
  constexpr int kLayers = 2;
  constexpr int kHkv = 2;
  constexpr int kHeads = 4;
  constexpr int kD = 64;
  constexpr int kBs = 16;
  const std::int64_t l_dim =
      (5 * kBs) + 7;  // 87 tokens: several full blocks + tail

  const Problem p = MakeProblem(kHeads, kHkv, kD, l_dim, 1.0, 6000);

  const kvcache::CacheGeometry geom{.num_layers = kLayers,
                                    .num_kv_heads = kHkv,
                                    .head_dim = kD,
                                    .dtype = DataType::kFloat32};
  memory::CpuAllocator alloc;
  auto pool =
      Unwrap(kvcache::BlockPool::Create(geom, kBs, /*num_blocks=*/64, &alloc));

  // Prime the free list so physical ids are permuted vs logical order: take a
  // few blocks and release them, then build the cache (any valid block table
  // must give the identical result — the priming just makes the mapping
  // non-trivial rather than accidentally the identity).
  std::vector<std::int32_t> primed(7);
  for (std::int32_t& block : primed) {
    block = static_cast<std::int32_t>(Unwrap(pool.Allocate()));
  }
  for (const std::int32_t block : primed) {
    pool.Release(block);
  }

  kvcache::PagedKvCache cache(&pool);

  // Append token by token (T=1 per append), all layers per token, in order.
  // Build per-token [1, Hkv, d] slices from the logical [Hkv, L, d] tensors.
  const float* kbase = p.k.data_ptr<float>();
  const float* vbase = p.v.data_ptr<float>();
  for (std::int64_t s = 0; s < l_dim; ++s) {
    const Tensor kt =
        Unwrap(ops::zeros(Shape{1, kHkv, kD}, DataType::kFloat32));
    const Tensor vt =
        Unwrap(ops::zeros(Shape{1, kHkv, kD}, DataType::kFloat32));
    for (std::int64_t h = 0; h < kHkv; ++h) {
      const float* ks = kbase + ((h * l_dim + s) * kD);
      const float* vs = vbase + ((h * l_dim + s) * kD);
      std::copy(ks, ks + kD, kt.data_ptr<float>() + (h * kD));
      std::copy(vs, vs + kD, vt.data_ptr<float>() + (h * kD));
    }
    for (int layer = 0; layer < kLayers; ++layer) {
      ASSERT_TRUE(cache.append(layer, kt, vt).ok());
    }
  }
  ASSERT_EQ(cache.length(), l_dim);

  const int layer = 1;
  const kvcache::PagedKvView pv = Unwrap(cache.paged_view(layer));
  const Tensor paged =
      Unwrap(ops::zeros(Shape{1, kHeads, kD}, DataType::kFloat32));
  PagedDecodeAttentionF32(p.q.data_ptr<float>(), pv.k_slab, pv.v_slab,
                          pv.block_table, pv.num_blocks, pv.length, kHeads,
                          kHkv, kD, pv.block_size, pv.block_stride, p.scale,
                          paged.data_ptr<float>());

  const Tensor contiguous = ContiguousDecode(p);
  EXPECT_TRUE(Unwrap(ops::allclose(paged, contiguous, 0.0, 0.0)).allclose)
      << "paged_view-fed decode not bit-identical to contiguous decode";
}

// (7) The M5 attention op goldens (HF ctx) replayed through the paged decode
// kernel — the l0_decode case directly, and the LAST context row of each
// prefill golden (causality makes that row the decode of that final token).
// Ties the paged decode path to the HF chain (Class T vs the oracle).
TEST(PagedDecodeAttentionKernelTest, MatchesFixtureGoldens) {
  const auto file = engine::model::SafetensorsFile::Open(
      engine::testing::FixturesDir() /
      "models/tiny-llama/expected/attention.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  constexpr float kFixtureScale = 0.25F;  // tiny-llama head_dim=16 → 16**-0.5
  constexpr std::int64_t kBs = 8;         // 8 | 16 ≤ head_dim; divides kAttnKb
  const char* names[] = {"l0_prefill_empty", "l1_prefill_empty",
                         "l0_prefill_continue", "l1_prefill_continue",
                         "l0_decode"};
  for (const char* name : names) {
    const std::string base(name);
    const Tensor q = Unwrap(file->tensor(base + ".q_rot"));  // [T, H, d]
    const Tensor k = Unwrap(file->tensor(base + ".k_all"));  // [Hkv, L, d]
    const Tensor v = Unwrap(file->tensor(base + ".v_all"));  // [Hkv, L, d]
    const Tensor ctx = Unwrap(file->tensor(base + ".ctx"));  // [T, H, d]

    const std::int64_t t_dim = q.shape().dim(0);
    const std::int64_t heads = q.shape().dim(1);
    const std::int64_t d = q.shape().dim(2);
    const std::int64_t kv_heads = k.shape().dim(0);
    const std::int64_t l_dim = k.shape().dim(1);

    // Wrap the last query row (t_dim-1) + this golden's K/V into a Problem so
    // we reuse the paged materializer. Copy so the Problem owns contiguous
    // data.
    const Problem p{
        .q = Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32)),
        .k = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
        .v = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
        .heads = heads,
        .kv_heads = kv_heads,
        .d = d,
        .l_dim = l_dim,
        .scale = kFixtureScale};
    const float* q_last = q.data_ptr<float>() + ((t_dim - 1) * heads * d);
    std::copy(q_last, q_last + (heads * d), p.q.data_ptr<float>());
    std::copy(k.data_ptr<float>(), k.data_ptr<float>() + (kv_heads * l_dim * d),
              p.k.data_ptr<float>());
    std::copy(v.data_ptr<float>(), v.data_ptr<float>() + (kv_heads * l_dim * d),
              p.v.data_ptr<float>());

    const PagedLayout lay = MakePagedLayout(p, kBs, /*extra_blocks=*/1);
    const Tensor out = PagedDecode(p, lay, kBs);

    const float* ctx_last = ctx.data_ptr<float>() + ((t_dim - 1) * heads * d);
    const Tensor ctx_row =
        Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));
    std::copy(ctx_last, ctx_last + (heads * d), ctx_row.data_ptr<float>());

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, ctx_row, kRtol, 1e-4));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
    std::cerr << "[paged-decode] golden " << name
              << " max_abs_diff=" << r.max_abs_diff << '\n';
  }
}

// (8) A block_size that does not divide kAttnKb violates the bit-identity
// invariant and is a programmer error (CHECK at the entry).
TEST(PagedDecodeAttentionKernelDeathTest, RejectsNonDivisorBlockSize) {
  const Problem p = MakeProblem(4, 2, 64, 40, 1.0, 7000);
  // bs = 24 does not divide kAttnKb = 64. Build a minimal layout by hand.
  const std::int64_t bs = 24;
  const std::int64_t needed = (p.l_dim + bs - 1) / bs;
  const std::int64_t block_stride = p.kv_heads * bs * p.d;
  std::vector<float> slab(static_cast<std::size_t>(needed * block_stride),
                          0.0F);
  std::vector<std::int32_t> table(static_cast<std::size_t>(needed));
  for (std::int64_t i = 0; i < needed; ++i) {
    table[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
  }
  const Tensor out =
      Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  EXPECT_DEATH(PagedDecodeAttentionF32(
                   p.q.data_ptr<float>(), slab.data(), slab.data(),
                   table.data(), needed, p.l_dim, p.heads, p.kv_heads, p.d, bs,
                   block_stride, p.scale, out.data_ptr<float>()),
               "block_size");
}

}  // namespace
}  // namespace engine::kernels
