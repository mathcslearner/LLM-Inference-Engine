#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "model/loader.h"
#include "model/modules.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

// cpu::rmsnorm goldens & properties, and the RmsNorm module (M5-T03; design:
// docs/design/model-execution.md §4.2, §12, §13). Class T against HF fixtures
// (torch reduces in its own order & uses rsqrt); bit-exact internally (single
// fp32 accumulator, ascending e), which the widening-equivalence and
// threading-independence tests assert directly.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::load_model;
using engine::model::RmsNorm;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] SafetensorsFile OpsFile() {
  auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

// Naive serial reference: the exact per-element computation cpu::rmsnorm
// performs (sum of squares ascending into one fp32 accumulator, inv =
// 1/sqrt(mean + eps), then scale) — so a matching result proves the threaded
// op's row partitioning changes traversal only. f32 x and weight.
[[nodiscard]] Tensor NaiveRmsNorm(const Tensor& x, const Tensor& weight,
                                  float eps) {
  const std::int64_t rows = x.shape().dim(0);
  const std::int64_t e = x.shape().dim(1);
  Tensor y = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  const auto* xd = x.data_ptr<float>();
  const auto* wd = weight.data_ptr<float>();
  auto* yd = y.data_ptr<float>();
  const auto e_f = static_cast<float>(e);
  for (std::int64_t r = 0; r < rows; ++r) {
    float sum_sq = 0.0F;
    for (std::int64_t j = 0; j < e; ++j) {
      const float v = xd[(r * e) + j];
      sum_sq += v * v;
    }
    const float inv = 1.0F / std::sqrt((sum_sq / e_f) + eps);
    for (std::int64_t j = 0; j < e; ++j) {
      yd[(r * e) + j] = xd[(r * e) + j] * inv * wd[j];
    }
  }
  return y;
}

// --- Fixture goldens across input dtype & eps ---

struct RmsNormCase {
  const char* name;
  float eps;  // ops_meta.json rmsnorm_cases[*].eps
};

constexpr RmsNormCase kCases[] = {
    {.name = "rmsnorm_f32", .eps = 1e-5F},
    {.name = "rmsnorm_bf16", .eps = 1e-5F},  // bf16 input (§12 acceptance)
    {.name = "rmsnorm_eps", .eps = 0.5F},    // eps dominates
};

TEST(CpuRmsNormTest, MatchesFixtureAcrossDtypesAndEps) {
  const SafetensorsFile file = OpsFile();
  // fp32-accumulated normalization; only divergence from torch is reduction
  // order + rsqrt vs 1/sqrt (Class T). Observed max-abs-diff printed per case.
  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-4;

  for (const RmsNormCase& c : kCases) {
    const std::string base(c.name);
    const Tensor x = Unwrap(file.tensor(base + ".x"));  // f32 or bf16
    const Tensor weight = Unwrap(file.tensor(base + ".weight"));
    const Tensor expected = Unwrap(file.tensor(base + ".y"));  // f32

    Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
    const Status s = engine::cpu::rmsnorm(x, weight, c.eps, out);
    ASSERT_TRUE(s.ok()) << c.name << ": " << s.ToString();

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << c.name << ": " << r.Summary();
    std::cerr << "[rmsnorm] " << c.name << " max_abs_diff=" << r.max_abs_diff
              << "\n";
  }
}

// --- On-the-fly widening equals half.h widening then f32 rmsnorm (bit-exact)
// ---

TEST(CpuRmsNormTest, OnTheFlyWideningIsBitExact) {
  const SafetensorsFile file = OpsFile();
  const Tensor x_bf16 = Unwrap(file.tensor("rmsnorm_bf16.x"));
  const Tensor w_bf16 = Unwrap(file.tensor("rmsnorm_bf16.weight"));
  const Tensor x_f32 = Unwrap(ops::cast(x_bf16, DataType::kFloat32));
  const Tensor w_f32 = Unwrap(ops::cast(w_bf16, DataType::kFloat32));

  Tensor from_half = Unwrap(ops::zeros(x_f32.shape(), DataType::kFloat32));
  Tensor from_f32 = Unwrap(ops::zeros(x_f32.shape(), DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::rmsnorm(x_bf16, w_bf16, 1e-5F, from_half).ok());
  ASSERT_TRUE(engine::cpu::rmsnorm(x_f32, w_f32, 1e-5F, from_f32).ok());

  // rtol=atol=0: widening a stored half via half.h operator float() (the same
  // conversion ops::cast uses) then computing is exactly what rmsnorm does per
  // element, so the two agree to the last bit.
  const ops::AllCloseResult r =
      Unwrap(ops::allclose(from_half, from_f32, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Threading independence: rmsnorm == naive serial loop (bit-exact) ---

TEST(CpuRmsNormTest, ThreadingIsBitExactVsSerial) {
  // Many rows (spread across threads) and E past trivial: row partitioning
  // must not perturb the per-element accumulation order.
  const Tensor x = Unwrap(ops::zeros(Shape{130, 300}, DataType::kFloat32));
  const Tensor weight = Unwrap(ops::zeros(Shape{300}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 111).ok());
  ASSERT_TRUE(ops::fill_normal(weight, 0.0, 1.0, 222).ok());

  Tensor out = Unwrap(ops::zeros(Shape{130, 300}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::rmsnorm(x, weight, 1e-5F, out).ok());
  const Tensor naive = NaiveRmsNorm(x, weight, 1e-5F);

  const ops::AllCloseResult r = Unwrap(ops::allclose(out, naive, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- In-place (y aliases x) equals out-of-place, for f32 x ---

TEST(CpuRmsNormTest, InPlaceMatchesOutOfPlace) {
  const Tensor x = Unwrap(ops::zeros(Shape{9, 64}, DataType::kFloat32));
  const Tensor weight = Unwrap(ops::zeros(Shape{64}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 3).ok());
  ASSERT_TRUE(ops::fill_normal(weight, 0.0, 1.0, 4).ok());

  Tensor out = Unwrap(ops::zeros(Shape{9, 64}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::rmsnorm(x, weight, 1e-5F, out).ok());

  Tensor inplace = Unwrap(ops::zeros(x.shape(), DataType::kFloat32));
  ASSERT_TRUE(ops::copy(inplace, x).ok());  // becomes both input and output
  ASSERT_TRUE(engine::cpu::rmsnorm(inplace, weight, 1e-5F, inplace).ok());

  const ops::AllCloseResult r = Unwrap(ops::allclose(inplace, out, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- eps is applied; a zero row stays finite (eps prevents 0/0) ---

TEST(CpuRmsNormTest, EpsIsAppliedAndZeroRowIsFinite) {
  const Tensor x = Unwrap(ops::zeros(Shape{2, 4}, DataType::kFloat32));
  const Tensor weight = Unwrap(ops::ones(Shape{4}, DataType::kFloat32));
  // Row 0 nonzero, row 1 all zeros (data_ptr is const-qualified — a const
  // handle still yields a writable pointer into its buffer).
  auto* xd = x.data_ptr<float>();
  for (int j = 0; j < 4; ++j) {
    xd[j] = static_cast<float>(j + 1);
  }

  Tensor small_eps = Unwrap(ops::zeros(Shape{2, 4}, DataType::kFloat32));
  Tensor big_eps = Unwrap(ops::zeros(Shape{2, 4}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::rmsnorm(x, weight, 1e-5F, small_eps).ok());
  ASSERT_TRUE(engine::cpu::rmsnorm(x, weight, 10.0F, big_eps).ok());

  // Different eps → different normalization of the nonzero row.
  EXPECT_NE(small_eps.data_ptr<float>()[0], big_eps.data_ptr<float>()[0]);
  // The all-zero row is finite and exactly zero (0 * inv * w) for both eps.
  for (int j = 0; j < 4; ++j) {
    EXPECT_TRUE(std::isfinite(small_eps.data_ptr<float>()[4 + j]));
    EXPECT_EQ(small_eps.data_ptr<float>()[4 + j], 0.0F);
  }
}

// --- Error posture: recoverable Status naming the offending input ---

TEST(CpuRmsNormTest, RejectsMalformedInputs) {
  const Tensor x = Unwrap(ops::zeros(Shape{3, 8}, DataType::kFloat32));
  const Tensor weight = Unwrap(ops::zeros(Shape{8}, DataType::kFloat32));
  Tensor y = Unwrap(ops::zeros(Shape{3, 8}, DataType::kFloat32));

  // x rank.
  {
    const Tensor x3 = Unwrap(ops::zeros(Shape{2, 3, 8}, DataType::kFloat32));
    const Status s = engine::cpu::rmsnorm(x3, weight, 1e-5F, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be rank-2"), std::string::npos)
        << s.message();
  }
  // weight length mismatch.
  {
    const Tensor w_bad = Unwrap(ops::zeros(Shape{7}, DataType::kFloat32));
    const Status s = engine::cpu::rmsnorm(x, w_bad, 1e-5F, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("weight length"), std::string::npos)
        << s.message();
  }
  // y wrong shape.
  {
    Tensor y_bad = Unwrap(ops::zeros(Shape{3, 7}, DataType::kFloat32));
    const Status s = engine::cpu::rmsnorm(x, weight, 1e-5F, y_bad);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be"), std::string::npos) << s.message();
  }
  // y not f32.
  {
    Tensor y_bf = Unwrap(ops::zeros(Shape{3, 8}, DataType::kBFloat16));
    const Status s = engine::cpu::rmsnorm(x, weight, 1e-5F, y_bf);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be f32"), std::string::npos)
        << s.message();
  }
  // x an unsupported (integer) dtype.
  {
    const Tensor x_i32 = Unwrap(ops::zeros(Shape{3, 8}, DataType::kInt32));
    const Status s = engine::cpu::rmsnorm(x_i32, weight, 1e-5F, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x dtype must be"), std::string::npos)
        << s.message();
  }
  // Non-contiguous x.
  {
    const Tensor wide = Unwrap(ops::zeros(Shape{3, 16}, DataType::kFloat32));
    const Tensor x_view = Unwrap(wide.slice(1, 0, 8));
    const Status s = engine::cpu::rmsnorm(x_view, weight, 1e-5F, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be contiguous"), std::string::npos)
        << s.message();
  }
}

// --- RmsNorm module (§4.2) ---

TEST(RmsNormModuleTest, CreateReportsHiddenSizeAndEps) {
  const Tensor w = Unwrap(ops::zeros(Shape{64}, DataType::kFloat32));
  const RmsNorm norm = Unwrap(RmsNorm::Create(w, 1e-5F));
  EXPECT_EQ(norm.hidden_size(), 64);
  EXPECT_EQ(norm.eps(), 1e-5F);
}

TEST(RmsNormModuleTest, CreateRejectsMalformedWeight) {
  // Weight not rank-1.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{2, 64}, DataType::kFloat32));
    const auto r = RmsNorm::Create(w, 1e-5F);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("weight must be rank-1"),
              std::string::npos)
        << r.status().message();
  }
  // Weight an unsupported dtype.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{64}, DataType::kInt32));
    const auto r = RmsNorm::Create(w, 1e-5F);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("weight dtype"), std::string::npos)
        << r.status().message();
  }
}

TEST(RmsNormModuleTest, ForwardEqualsOp) {
  const Tensor x = Unwrap(ops::zeros(Shape{5, 64}, DataType::kFloat32));
  const Tensor w = Unwrap(ops::zeros(Shape{64}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 9).ok());
  ASSERT_TRUE(ops::fill_normal(w, 0.0, 1.0, 10).ok());
  const RmsNorm norm = Unwrap(RmsNorm::Create(w, 1e-5F));

  Tensor via_module = Unwrap(ops::zeros(Shape{5, 64}, DataType::kFloat32));
  Tensor via_op = Unwrap(ops::zeros(Shape{5, 64}, DataType::kFloat32));
  ASSERT_TRUE(norm.forward(x, via_module).ok());
  ASSERT_TRUE(engine::cpu::rmsnorm(x, w, 1e-5F, via_op).ok());

  const ops::AllCloseResult r =
      Unwrap(ops::allclose(via_module, via_op, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

TEST(RmsNormModuleTest, ForwardRejectsBadActivations) {
  const Tensor w = Unwrap(ops::zeros(Shape{12}, DataType::kFloat32));
  const RmsNorm norm = Unwrap(RmsNorm::Create(w, 1e-5F));
  // x hidden dim != hidden_size.
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 8}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, 12}, DataType::kFloat32));
    const Status s = norm.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be"), std::string::npos) << s.message();
  }
  // x not f32 (the module requires fp32 activations even though the op widens).
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 12}, DataType::kBFloat16));
    Tensor y = Unwrap(ops::zeros(Shape{2, 12}, DataType::kFloat32));
    const Status s = norm.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be f32"), std::string::npos)
        << s.message();
  }
  // y wrong shape.
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 12}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, 5}, DataType::kFloat32));
    const Status s = norm.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be caller-allocated"), std::string::npos)
        << s.message();
  }
}

// End-to-end: a real bf16 norm weight, loaded zero-copy, forwarded through the
// RmsNorm module, matches a hand-computed fp32 RMSNorm on genuine checkpoint
// bytes. tiny-llama's norm weights are HF-initialized to ones (widened bit-
// exactly), so this exercises the loader → module → cpu::rmsnorm path.
TEST(RmsNormModuleTest, RealCheckpointNormMatchesManualFp32) {
  const auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const auto it = loaded->weights.find("layers.0.attn_norm.weight");
  ASSERT_NE(it, loaded->weights.end());
  const Tensor w = it->second;  // [64], bf16, zero-copy
  ASSERT_EQ(w.dtype(), DataType::kBFloat16);
  ASSERT_EQ(w.shape().dim(0), 64);

  const SafetensorsFile file = OpsFile();
  const Tensor x = Unwrap(file.tensor("rmsnorm_f32.x"));  // [16,64] f32

  constexpr float kEps = 1e-5F;
  const RmsNorm norm = Unwrap(RmsNorm::Create(w, kEps));
  Tensor y = Unwrap(ops::zeros(x.shape(), DataType::kFloat32));
  ASSERT_TRUE(norm.forward(x, y).ok());

  const Tensor w_f32 = Unwrap(ops::cast(w, DataType::kFloat32));
  const Tensor manual = NaiveRmsNorm(x, w_f32, kEps);
  const ops::AllCloseResult r = Unwrap(ops::allclose(y, manual, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

}  // namespace
