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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

// Optimized decode-attention kernel (M6-T05; design:
// docs/design/optimized-cpu-execution.md §8, §10). The single-token
// specialization of the prefill kernel: one query per head attends the whole
// cache, threaded across kv heads with the g = H/Hkv query heads of a group
// sharing the streamed K/V. Validated against cpu::attention (the oracle) with
// a T = 1 query across cache lengths {1, 63, 64, 65, 2048} and GQA configs, and
// asserted **bit-identical to the M6-T04 prefill path** for the same single
// token (the acceptance cross-check). Registered SCALAR_PASS.
//
// Tolerance vs the oracle (§10 attention): atol 1e-5, rtol 1e-4 — the online
// rescale order + vector `exp`. The decode-vs-prefill assertion is instead
// bitwise (identical arithmetic order per output head, by construction).
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

// One decode problem: q [1, H, d] (a T = 1 query), k/v [Hkv, L, d], seeded. The
// q tensor is rank-3 [1, H, d] so the same handle feeds cpu::attention (which
// takes [T, H, d]) and DecodeAttentionF32 (which reads it as [H, d]).
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

// The M5 reference over the same single-token query: cpu::attention with T = 1.
[[nodiscard]] Tensor Reference(const Problem& p) {
  Tensor out = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  EXPECT_TRUE(engine::cpu::attention(p.q, p.k, p.v, p.scale, out).ok());
  return out;
}

[[nodiscard]] Tensor Decode(const Problem& p) {
  Tensor out = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  DecodeAttentionF32(p.q.data_ptr<float>(), p.k.data_ptr<float>(),
                     p.v.data_ptr<float>(), out.data_ptr<float>(), p.heads,
                     p.kv_heads, p.d, p.l_dim, p.scale);
  return out;
}

// The M6-T04 prefill kernel over the same single-token query (T = 1). The
// decode kernel must match this bit-for-bit (§8, §10 M6-T05 acceptance).
[[nodiscard]] Tensor Prefill(const Problem& p) {
  Tensor out = Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  PrefillAttentionF32(p.q.data_ptr<float>(), p.k.data_ptr<float>(),
                      p.v.data_ptr<float>(), out.data_ptr<float>(),
                      /*t_dim=*/1, p.heads, p.kv_heads, p.d, p.l_dim, p.scale);
  return out;
}

void ExpectMatchesOracle(const Problem& p, const char* label) {
  const Tensor ref = Reference(p);
  const Tensor out = Decode(p);
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
  EXPECT_TRUE(r.allclose) << label << ": " << r.Summary();
  std::cerr << "[decode] " << label << " max_abs_diff=" << r.max_abs_diff
            << '\n';
}

// Decode == prefill(T=1), bit-for-bit. Interleaving the group's query heads at
// the block level never touches another head's accumulator, so the fp32
// arithmetic order per output head is identical to the prefill path (§8).
void ExpectMatchesPrefillBitExact(const Problem& p, const char* label) {
  const Tensor prefill = Prefill(p);
  const Tensor decode = Decode(p);
  EXPECT_TRUE(Unwrap(ops::allclose(decode, prefill, 0.0, 0.0)).allclose)
      << label << ": decode not bit-identical to prefill(T=1)";
}

// The acceptance cache lengths {1, 63, 64, 65, 2048} (63/64/65 straddle the
// kAttnKb block multiple; 127/128/129 straddle the second) — vs the oracle and
// bit-exact vs the prefill path. GQA H=4, Hkv=2, d=64.
TEST(DecodeAttentionKernelTest, MatchesOracleAcrossCacheLengths) {
  const std::int64_t lengths[] = {1, 63, 64, 65, 127, 128, 129, 2048};
  std::uint64_t seed = 1000;
  for (const std::int64_t l_dim : lengths) {
    const Problem p = MakeProblem(4, 2, 64, l_dim, 1.0, seed += 10);
    const std::string label = "L=" + std::to_string(l_dim);
    ExpectMatchesOracle(p, label.c_str());
    ExpectMatchesPrefillBitExact(p, label.c_str());
  }
}

// GQA group sizes {1, 2, full} × head dims crossing the vector widths (18 and
// 24 force both NEON tail (d%4) and AVX2 tail (d%8); 24 = Qwen head_dim). Plus
// a group larger than kAttnDecodeGroupChunk (H=24, Hkv=2 → g=12) to exercise
// the group-chunk slicing (K/V re-streamed per slice; still bit-exact vs
// prefill). Cache length crosses two key blocks.
TEST(DecodeAttentionKernelTest, GqaConfigs) {
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
      ExpectMatchesOracle(p, label.c_str());
      ExpectMatchesPrefillBitExact(p, label.c_str());
    }
  }
}

// Large-magnitude scores across many key blocks: the running-max rescale (alpha
// far from 1) must still land within tolerance. std 2 ⇒ logit std ≈4 (§10).
TEST(DecodeAttentionKernelTest, LargeMagnitudeRescale) {
  std::uint64_t seed = 4000;
  for (const std::int64_t l_dim : {512, 2048}) {
    const Problem p = MakeProblem(4, 2, 64, l_dim, 2.0, seed += 10);
    const std::string label = "large L=" + std::to_string(l_dim);
    ExpectMatchesOracle(p, label.c_str());
    ExpectMatchesPrefillBitExact(p, label.c_str());
  }
}

// Build the DecodeArgs the public entry builds, to call a variant directly over
// a contiguous kv-head range.
[[nodiscard]] internal::DecodeArgs ArgsFor(const Problem& p, float* out) {
  return internal::DecodeArgs{.q = p.q.data_ptr<float>(),
                              .k = p.k.data_ptr<float>(),
                              .v = p.v.data_ptr<float>(),
                              .out = out,
                              .d = p.d,
                              .l_dim = p.l_dim,
                              .group = p.heads / p.kv_heads,
                              .scale = p.scale};
}

// The threaded public entry is bit-identical to a single serial variant call
// over all kv-head units — and to an arbitrary manual chunking — so the result
// is invariant to thread count / chunking (each kv head wholly in one call).
TEST(DecodeAttentionKernelTest, ThreadingIsBitExactVsSerialVariant) {
  const Problem p = MakeProblem(8, 8, 64, 200, 1.5, 5000);
  const std::int64_t units = p.kv_heads;
  const auto fn = detail::DecodeAttentionVariant(SelectedIsa());

  const Tensor threaded = Decode(p);

  const Tensor serial =
      Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  fn(ArgsFor(p, serial.data_ptr<float>()), 0, units);

  const Tensor chunked =
      Unwrap(ops::zeros(Shape{1, p.heads, p.d}, DataType::kFloat32));
  const internal::DecodeArgs cargs = ArgsFor(p, chunked.data_ptr<float>());
  const std::int64_t bounds[] = {0, 1, 3, units};
  for (std::size_t b = 0; b + 1 < std::size(bounds); ++b) {
    fn(cargs, bounds[b], bounds[b + 1]);
  }

  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(chunked, serial, 0.0, 0.0)).allclose);
}

TEST(DecodeAttentionKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::DecodeAttentionVariant(isa)),
            reinterpret_cast<void*>(&scalar::DecodeUnits));
}

// The M5 attention op goldens (HF ctx) replayed through the decode kernel — the
// l0_decode case directly, and the LAST context row of each prefill golden
// (causality makes that row the decode of that final token, so cpu::attention's
// ctx[T-1] is the decode reference). Ties the decode path to the HF chain.
TEST(DecodeAttentionKernelTest, MatchesFixtureGoldens) {
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

    const std::int64_t t_dim = q.shape().dim(0);
    const std::int64_t heads = q.shape().dim(1);
    const std::int64_t d = q.shape().dim(2);
    const std::int64_t kv_heads = k.shape().dim(0);
    const std::int64_t l_dim = k.shape().dim(1);

    // Decode the last query (row t_dim-1): it sits at absolute position L-1 and
    // attends all L keys, exactly the decode contract. Its q row is q[t_dim-1].
    const float* q_last = q.data_ptr<float>() + ((t_dim - 1) * heads * d);
    const Tensor out =
        Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));
    DecodeAttentionF32(q_last, k.data_ptr<float>(), v.data_ptr<float>(),
                       out.data_ptr<float>(), heads, kv_heads, d, l_dim,
                       kFixtureScale);

    // Compare against ctx[t_dim-1] (the reference context for that token).
    const float* ctx_last = ctx.data_ptr<float>() + ((t_dim - 1) * heads * d);
    const Tensor ctx_row =
        Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));
    std::copy(ctx_last, ctx_last + (heads * d), ctx_row.data_ptr<float>());

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, ctx_row, kRtol, 1e-4));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
    std::cerr << "[decode] golden " << name
              << " max_abs_diff=" << r.max_abs_diff << '\n';
  }
}

}  // namespace
}  // namespace engine::kernels
