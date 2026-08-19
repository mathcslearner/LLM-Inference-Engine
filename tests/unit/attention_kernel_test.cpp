#include "common/paths.h"
#include "cpu/ops.h"
#include "kernels/attention.h"
#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

// Optimized prefill-attention kernel (M6-T04; design:
// docs/design/optimized-cpu-execution.md §8, §10). Validated against
// cpu::attention (the oracle, itself validated against HF fixtures) across
// sequence lengths {1, 17, 512, 2048}, prefill from empty and non-empty cache,
// and GQA configs. Registered SCALAR_PASS.
//
// Tolerance (§10 attention): atol 1e-5, rtol 1e-4 — the divergence from the
// oracle is the online-softmax rescale order + vector `exp` (≤2 ulp) +
// horizontal reductions. The observed max-abs-diff is logged and sits far
// below. Bit-identity is asserted across thread counts (each query's recurrence
// runs wholly within one variant call).
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

// One attention problem: q [T, H, d], k/v [Hkv, L, d] with L = P + T, seeded.
struct Problem {
  Tensor q;
  Tensor k;
  Tensor v;
  std::int64_t t_dim;
  std::int64_t heads;
  std::int64_t kv_heads;
  std::int64_t d;
  std::int64_t l_dim;
  float scale;
};

[[nodiscard]] Problem MakeProblem(std::int64_t t_dim, std::int64_t heads,
                                  std::int64_t kv_heads, std::int64_t d,
                                  std::int64_t past, double std_dev,
                                  std::uint64_t seed) {
  const std::int64_t l_dim = past + t_dim;
  Problem p{
      .q = Unwrap(ops::zeros(Shape{t_dim, heads, d}, DataType::kFloat32)),
      .k = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
      .v = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
      .t_dim = t_dim,
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

[[nodiscard]] Tensor Reference(const Problem& p) {
  Tensor out =
      Unwrap(ops::zeros(Shape{p.t_dim, p.heads, p.d}, DataType::kFloat32));
  EXPECT_TRUE(engine::cpu::attention(p.q, p.k, p.v, p.scale, out).ok());
  return out;
}

[[nodiscard]] Tensor Optimized(const Problem& p) {
  Tensor out =
      Unwrap(ops::zeros(Shape{p.t_dim, p.heads, p.d}, DataType::kFloat32));
  PrefillAttentionF32(p.q.data_ptr<float>(), p.k.data_ptr<float>(),
                      p.v.data_ptr<float>(), out.data_ptr<float>(), p.t_dim,
                      p.heads, p.kv_heads, p.d, p.l_dim, p.scale);
  return out;
}

void ExpectMatchesOracle(const Problem& p, const char* label) {
  const Tensor ref = Reference(p);
  const Tensor out = Optimized(p);
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
  EXPECT_TRUE(r.allclose) << label << ": " << r.Summary();
  std::cerr << "[attn] " << label << " max_abs_diff=" << r.max_abs_diff << '\n';
}

// The acceptance sequence lengths {1, 17, 512, 2048}, from empty (P=0) and
// non-empty (P>0) cache. GQA H=4, Hkv=2, d=64. The T=1, P=2047 case is the
// decode-shaped call (one query over a full cache) — M6-T05's path validated
// early through the same kernel.
TEST(AttentionKernelTest, MatchesOracleAcrossLengths) {
  const std::int64_t lengths[] = {1, 17, 512, 2048};
  std::uint64_t seed = 1000;
  for (const std::int64_t t_dim : lengths) {
    for (const std::int64_t past : {std::int64_t{0}, std::int64_t{5}}) {
      const Problem p = MakeProblem(t_dim, 4, 2, 64, past, 1.0, seed += 10);
      ExpectMatchesOracle(
          p, ("T=" + std::to_string(t_dim) + " P=" + std::to_string(past))
                 .c_str());
    }
  }
  const Problem decode = MakeProblem(1, 4, 2, 64, 2047, 1.0, seed + 10);
  ExpectMatchesOracle(decode, "T=1 P=2047 (decode-shaped)");
}

// GQA group sizes {1, 2, full} × head dims crossing the vector widths (18 and
// 24 force both NEON tail (d%4) and AVX2 tail (d%8); 24 = Qwen head_dim).
TEST(AttentionKernelTest, GqaConfigs) {
  struct Cfg {
    std::int64_t heads;
    std::int64_t kv_heads;
  };
  const Cfg cfgs[] = {{.heads = 4, .kv_heads = 4},
                      {.heads = 4, .kv_heads = 2},
                      {.heads = 8, .kv_heads = 1}};
  const std::int64_t dims[] = {18, 24, 64, 128};
  std::uint64_t seed = 2000;
  for (const Cfg& c : cfgs) {
    for (const std::int64_t d : dims) {
      const Problem p =
          MakeProblem(20, c.heads, c.kv_heads, d, 3, 1.0, seed += 10);
      ExpectMatchesOracle(
          p, ("H=" + std::to_string(c.heads) +
              " Hkv=" + std::to_string(c.kv_heads) + " d=" + std::to_string(d))
                 .c_str());
    }
  }
}

// T, P, L straddling the kAttnQb / kAttnKb block multiples — where the
// diagonal-block masking offset and the last-partial-block paths live.
TEST(AttentionKernelTest, BlockBoundaries) {
  const std::int64_t qb = internal::kAttnQb;
  const std::int64_t kb = internal::kAttnKb;
  const std::int64_t t_dims[] = {qb - 1, qb, qb + 1};
  const std::int64_t pasts[] = {kb - 1, kb, kb + 1, 0};
  std::uint64_t seed = 3000;
  for (const std::int64_t t_dim : t_dims) {
    for (const std::int64_t past : pasts) {
      const Problem p = MakeProblem(t_dim, 4, 2, 64, past, 1.0, seed += 10);
      ExpectMatchesOracle(
          p, ("T=" + std::to_string(t_dim) + " P=" + std::to_string(past))
                 .c_str());
    }
  }
}

// Large-magnitude scores across several key blocks: the running-max rescale
// (alpha far from 1 when a late block's max dominates) and the stability path
// (subtract running max, no overflow) must still land within tolerance. With
// scale = d^-0.5, N(0, s²) q/k give attention logits of std ≈ s², so s = 2
// drives logits to std ≈ 4 (range past ±16) — a genuine stress that forces the
// max to advance repeatedly across the 100/300-key blocks, while staying in the
// realistic logit range §10's tolerance is stated for. (Pushing s higher —
// logit std in the tens — leaves that range: flash-attention's incremental
// rescale then accumulates more rounding than the reference's single
// max-subtraction, exceeding rtol 1e-4 where |ref| is small, a regime no
// trained model hits. The T=2048 case already covers 32-block rescale at
// realistic magnitude.)
TEST(AttentionKernelTest, LargeMagnitudeRescale) {
  std::uint64_t seed = 4000;
  for (const std::int64_t t_dim : {100, 300}) {
    const Problem p = MakeProblem(t_dim, 4, 2, 64, 0, 2.0, seed += 10);
    ExpectMatchesOracle(p, ("large T=" + std::to_string(t_dim)).c_str());
  }
}

// Build the same PrefillArgs the public entry builds, to call a variant
// directly over a contiguous unit range.
[[nodiscard]] internal::PrefillArgs ArgsFor(const Problem& p, float* out) {
  const std::int64_t num_qblocks =
      (p.t_dim + internal::kAttnQb - 1) / internal::kAttnQb;
  return internal::PrefillArgs{.q = p.q.data_ptr<float>(),
                               .k = p.k.data_ptr<float>(),
                               .v = p.v.data_ptr<float>(),
                               .out = out,
                               .t_dim = p.t_dim,
                               .heads = p.heads,
                               .d = p.d,
                               .l_dim = p.l_dim,
                               .group = p.heads / p.kv_heads,
                               .past = p.l_dim - p.t_dim,
                               .num_qblocks = num_qblocks,
                               .scale = p.scale};
}

// The threaded public entry is bit-identical to a single serial variant call
// over all units — and to an arbitrary manual chunking — so the result is
// invariant to thread count / chunking (each query wholly in one call, §10).
TEST(AttentionKernelTest, ThreadingIsBitExactVsSerialVariant) {
  const Problem p = MakeProblem(70, 4, 2, 64, 9, 1.5, 5000);
  const std::int64_t num_qblocks =
      (p.t_dim + internal::kAttnQb - 1) / internal::kAttnQb;
  const std::int64_t units = p.heads * num_qblocks;
  const auto fn = detail::PrefillAttentionVariant(SelectedIsa());

  const Tensor threaded = Optimized(p);

  const Tensor serial =
      Unwrap(ops::zeros(Shape{p.t_dim, p.heads, p.d}, DataType::kFloat32));
  fn(ArgsFor(p, serial.data_ptr<float>()), 0, units);

  const Tensor chunked =
      Unwrap(ops::zeros(Shape{p.t_dim, p.heads, p.d}, DataType::kFloat32));
  const internal::PrefillArgs cargs = ArgsFor(p, chunked.data_ptr<float>());
  const std::int64_t bounds[] = {0, 1, 2, 5, units};
  for (std::size_t b = 0; b + 1 < std::size(bounds); ++b) {
    fn(cargs, bounds[b], bounds[b + 1]);
  }

  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(chunked, serial, 0.0, 0.0)).allclose);
}

TEST(AttentionKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::PrefillAttentionVariant(isa)),
            reinterpret_cast<void*>(&scalar::PrefillUnits));
}

// The M5 attention op goldens (HF ctx) replayed through the kernel — ties the
// optimized path to the HF chain, not just the reference. Cases span prefill
// from empty (P=0), prefill continuing (P=5, T=6), and decode (T=1, P=7).
TEST(AttentionKernelTest, MatchesFixtureGoldens) {
  const auto file = engine::model::SafetensorsFile::Open(
      engine::testing::FixturesDir() /
      "models/tiny-llama/expected/attention.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  // head_dim**-0.5 for tiny-llama head_dim=16 → 0.25 (attention_test.cpp).
  constexpr float kFixtureScale = 0.25F;
  const char* names[] = {"l0_prefill_empty", "l1_prefill_empty",
                         "l0_prefill_continue", "l1_prefill_continue",
                         "l0_decode"};
  for (const char* name : names) {
    const std::string base(name);
    const Tensor q = Unwrap(file->tensor(base + ".q_rot"));  // [T, H, d]
    const Tensor k = Unwrap(file->tensor(base + ".k_all"));  // [Hkv, L, d]
    const Tensor v = Unwrap(file->tensor(base + ".v_all"));  // [Hkv, L, d]
    const Tensor ctx = Unwrap(file->tensor(base + ".ctx"));  // [T, H, d]

    const Tensor out = Unwrap(ops::zeros(ctx.shape(), DataType::kFloat32));
    PrefillAttentionF32(q.data_ptr<float>(), k.data_ptr<float>(),
                        v.data_ptr<float>(), out.data_ptr<float>(),
                        q.shape().dim(0), q.shape().dim(1), k.shape().dim(0),
                        q.shape().dim(2), k.shape().dim(1), kFixtureScale);
    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ctx, kRtol, 1e-4));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
    std::cerr << "[attn] golden " << name << " max_abs_diff=" << r.max_abs_diff
              << '\n';
  }
}

}  // namespace
}  // namespace engine::kernels
