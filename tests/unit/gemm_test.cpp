#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string>

// cpu::gemm goldens & properties (M5-T02; design:
// docs/design/model-execution.md §3.3, §12, §13). Class T against HF fixtures
// (torch reduces in its own order); bit-exact internally (single fp32
// accumulator, ascending k), which the widening-equivalence and
// blocking-independence tests below assert directly.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] Tensor OpsFixtureTensor(const SafetensorsFile& file,
                                      const std::string& name) {
  return Unwrap(file.tensor(name));
}

// Naive serial reference: C[m,n] = sum_k A[m,k] * B[n,k], single fp32
// accumulator ascending k — the exact per-element reduction cpu::gemm performs,
// so a matching result proves gemm's tiling/threading changes traversal only.
[[nodiscard]] Tensor NaiveGemm(const Tensor& a, const Tensor& b) {
  const std::int64_t m = a.shape().dim(0);
  const std::int64_t k = a.shape().dim(1);
  const std::int64_t n = b.shape().dim(0);
  Tensor c = Unwrap(ops::zeros(Shape{m, n}, DataType::kFloat32));
  const auto* ad = a.data_ptr<float>();
  const auto* bd = b.data_ptr<float>();
  auto* cd = c.data_ptr<float>();
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float acc = 0.0F;
      for (std::int64_t p = 0; p < k; ++p) {
        acc += ad[(i * k) + p] * bd[(j * k) + p];
      }
      cd[(i * n) + j] = acc;
    }
  }
  return c;
}

// --- Fixture goldens across shapes & storage dtypes ---

struct GemmCase {
  const char* name;
  bool has_bias;
};

// The cases in tiny-llama/expected/ops.safetensors (ops_meta.json). Real bf16
// projection weights on the fixture's dims + synthetic shape/dtype stress.
constexpr GemmCase kCases[] = {
    {.name = "q_proj_prefill", .has_bias = false},
    {.name = "gate_proj_wide", .has_bias = false},
    {.name = "down_proj", .has_bias = false},
    {.name = "lm_head_wide", .has_bias = false},
    {.name = "decode_gemv", .has_bias = false},
    {.name = "skinny_tall", .has_bias = false},
    {.name = "k_one", .has_bias = false},
    {.name = "odd_tails", .has_bias = true},
    {.name = "fp16_weight", .has_bias = false},
    {.name = "bf16_bias", .has_bias = true},
};

TEST(CpuGemmTest, MatchesFixtureAcrossShapesAndDtypes) {
  const auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();

  // All cases are small K (<= 176) fp32-accumulated matmuls; the only
  // divergence from the torch golden is reduction order (Class T). Observed
  // max-abs-diff is printed per case and stays far below this threshold
  // (worst observed ~7.6e-6 on the bf16/f16 cases).
  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-4;

  for (const GemmCase& c : kCases) {
    const std::string base(c.name);
    const Tensor a = OpsFixtureTensor(*file, base + ".a");
    const Tensor b = OpsFixtureTensor(*file, base + ".b");
    const Tensor expected = OpsFixtureTensor(*file, base + ".c");
    Tensor bias;
    const Tensor* bias_ptr = nullptr;
    if (c.has_bias) {
      bias = OpsFixtureTensor(*file, base + ".bias");
      bias_ptr = &bias;
    }

    Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
    const Status s = engine::cpu::gemm(a, b, bias_ptr, out);
    ASSERT_TRUE(s.ok()) << c.name << ": " << s.ToString();

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << c.name << ": " << r.Summary();
    std::cerr << "[gemm] " << c.name << " max_abs_diff=" << r.max_abs_diff
              << "\n";
  }
}

// --- On-the-fly widening equals half.h widening then f32 GEMM (bit-exact) ---

void ExpectWideningBitExact(const SafetensorsFile& file,
                            const std::string& base) {
  const Tensor a = OpsFixtureTensor(file, base + ".a");
  const Tensor b_half = OpsFixtureTensor(file, base + ".b");  // f16 or bf16
  const Tensor b_f32 = Unwrap(ops::cast(b_half, DataType::kFloat32));

  const Shape out_shape{a.shape().dim(0), b_half.shape().dim(0)};
  Tensor from_half = Unwrap(ops::zeros(out_shape, DataType::kFloat32));
  Tensor from_f32 = Unwrap(ops::zeros(out_shape, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::gemm(a, b_half, nullptr, from_half).ok());
  ASSERT_TRUE(engine::cpu::gemm(a, b_f32, nullptr, from_f32).ok());

  // rtol=atol=0: bit-exact. Widening a stored half to f32 (half.h operator
  // float(), the same conversion ops::cast uses) then multiplying is exactly
  // what gemm does per element, so the two agree to the last bit.
  const ops::AllCloseResult r =
      Unwrap(ops::allclose(from_half, from_f32, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << base << ": " << r.Summary();
}

TEST(CpuGemmTest, OnTheFlyWideningIsBitExact) {
  const auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  ExpectWideningBitExact(*file, "fp16_weight");     // f16 weight
  ExpectWideningBitExact(*file, "q_proj_prefill");  // bf16 weight
}

// --- Tiling/threading independence: gemm == naive serial loop (bit-exact) ---

TEST(CpuGemmTest, TilingAndThreadingAreBitExactVsSerial) {
  // Shapes spanning multiple M/N tiles (kTileM=kTileN=64) and K past one K
  // block (kTileK=256): tiling and thread partitioning must not perturb the
  // per-element accumulation order.
  struct Shape2 {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
  };
  const Shape2 shapes[] = {
      {.m = 130, .k = 300, .n = 70},
      {.m = 65, .k = 129, .n = 33},
      {.m = 1, .k = 512, .n = 200},
  };
  std::uint64_t seed = 12345;
  for (const Shape2& sh : shapes) {
    const Tensor a = Unwrap(ops::zeros(Shape{sh.m, sh.k}, DataType::kFloat32));
    const Tensor b = Unwrap(ops::zeros(Shape{sh.n, sh.k}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(a, 0.0, 1.0, seed++).ok());
    ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, seed++).ok());

    Tensor out = Unwrap(ops::zeros(Shape{sh.m, sh.n}, DataType::kFloat32));
    ASSERT_TRUE(engine::cpu::gemm(a, b, nullptr, out).ok());
    const Tensor naive = NaiveGemm(a, b);

    const ops::AllCloseResult r = Unwrap(ops::allclose(out, naive, 0.0, 0.0));
    EXPECT_TRUE(r.allclose)
        << sh.m << "x" << sh.k << "x" << sh.n << ": " << r.Summary();
  }
}

// --- Bias is added once, after the K accumulation ---

TEST(CpuGemmTest, BiasEqualsUnbiasedPlusBias) {
  const Tensor a = Unwrap(ops::zeros(Shape{7, 40}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{9, 40}, DataType::kFloat32));
  const Tensor bias = Unwrap(ops::zeros(Shape{9}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(a, 0.0, 1.0, 1).ok());
  ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, 2).ok());
  ASSERT_TRUE(ops::fill_normal(bias, 0.0, 1.0, 3).ok());

  Tensor biased = Unwrap(ops::zeros(Shape{7, 9}, DataType::kFloat32));
  Tensor unbiased = Unwrap(ops::zeros(Shape{7, 9}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::gemm(a, b, &bias, biased).ok());
  ASSERT_TRUE(engine::cpu::gemm(a, b, nullptr, unbiased).ok());

  const auto* bd = bias.data_ptr<float>();
  const auto* ub = unbiased.data_ptr<float>();
  const auto* bi = biased.data_ptr<float>();
  for (std::int64_t i = 0; i < 7; ++i) {
    for (std::int64_t j = 0; j < 9; ++j) {
      EXPECT_FLOAT_EQ(bi[(i * 9) + j], ub[(i * 9) + j] + bd[j])
          << i << "," << j;
    }
  }
}

// --- Error posture: recoverable Status naming the offending input ---

TEST(CpuGemmTest, RejectsMalformedInputs) {
  const Tensor a = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{5, 8}, DataType::kFloat32));
  Tensor c = Unwrap(ops::zeros(Shape{4, 5}, DataType::kFloat32));

  // Rank.
  {
    const Tensor a3 = Unwrap(ops::zeros(Shape{2, 4, 8}, DataType::kFloat32));
    const Status s = engine::cpu::gemm(a3, b, nullptr, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("a must be rank-2"), std::string::npos)
        << s.message();
  }
  // K mismatch.
  {
    const Tensor b_bad = Unwrap(ops::zeros(Shape{5, 7}, DataType::kFloat32));
    const Status s = engine::cpu::gemm(a, b_bad, nullptr, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("inner dims disagree"), std::string::npos)
        << s.message();
  }
  // Wrong c shape.
  {
    Tensor c_bad = Unwrap(ops::zeros(Shape{4, 6}, DataType::kFloat32));
    const Status s = engine::cpu::gemm(a, b, nullptr, c_bad);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("c must be"), std::string::npos) << s.message();
  }
  // a not f32.
  {
    const Tensor a_bf = Unwrap(ops::zeros(Shape{4, 8}, DataType::kBFloat16));
    const Status s = engine::cpu::gemm(a_bf, b, nullptr, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("a must be f32"), std::string::npos)
        << s.message();
  }
  // b an unsupported (integer) dtype.
  {
    const Tensor b_i32 = Unwrap(ops::zeros(Shape{5, 8}, DataType::kInt32));
    const Status s = engine::cpu::gemm(a, b_i32, nullptr, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("weight) dtype"), std::string::npos)
        << s.message();
  }
  // Non-contiguous a (inner-dim slice).
  {
    const Tensor wide = Unwrap(ops::zeros(Shape{4, 16}, DataType::kFloat32));
    const Tensor a_view = Unwrap(wide.slice(1, 0, 8));
    const Status s = engine::cpu::gemm(a_view, b, nullptr, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("a must be contiguous"), std::string::npos)
        << s.message();
  }
  // Bias length mismatch.
  {
    const Tensor bias_bad = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
    const Status s = engine::cpu::gemm(a, b, &bias_bad, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("bias must have length N"), std::string::npos)
        << s.message();
  }
  // Bias unsupported dtype.
  {
    const Tensor bias_i32 = Unwrap(ops::zeros(Shape{5}, DataType::kInt32));
    const Status s = engine::cpu::gemm(a, b, &bias_i32, c);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("bias dtype"), std::string::npos) << s.message();
  }
}

// --- Sanity performance bound (not a perf test): 512^3 < 1 s ---

TEST(CpuGemmTest, LargeSquareGemmUnderOneSecond) {
  constexpr std::int64_t kN = 512;
  const Tensor a = Unwrap(ops::zeros(Shape{kN, kN}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{kN, kN}, DataType::kFloat32));
  Tensor c = Unwrap(ops::zeros(Shape{kN, kN}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(a, 0.0, 1.0, 7).ok());
  ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, 8).ok());

  const auto start = std::chrono::steady_clock::now();
  ASSERT_TRUE(engine::cpu::gemm(a, b, nullptr, c).ok());
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double seconds = std::chrono::duration<double>(elapsed).count();
  std::cerr << "[gemm] 512x512x512 took " << seconds << " s\n";
  EXPECT_LT(seconds, 1.0);

  // Spot-check a few outputs against the serial reference so a fast-but-wrong
  // implementation cannot pass on timing alone.
  const auto* ad = a.data_ptr<float>();
  const auto* bd = b.data_ptr<float>();
  const auto* cd = c.data_ptr<float>();
  for (const std::int64_t idx :
       {std::int64_t{0}, std::int64_t{137}, (kN * kN) - 1}) {
    const std::int64_t i = idx / kN;
    const std::int64_t j = idx % kN;
    float acc = 0.0F;
    for (std::int64_t p = 0; p < kN; ++p) {
      acc += ad[(i * kN) + p] * bd[(j * kN) + p];
    }
    EXPECT_FLOAT_EQ(cd[idx], acc) << idx;
  }
}

}  // namespace
