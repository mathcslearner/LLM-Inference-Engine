#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "model/config.h"
#include "model/modules.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// cpu::rope_apply / the Rope module goldens & properties (M5-T04; design:
// docs/design/model-execution.md §7, §12, §13). The oracle is HF's
// LlamaRotaryEmbedding output (rope.safetensors). HF forces fp32 internally;
// the reference forms the angle in fp64 and stores fp32, so agreement is
// Class T within the tolerances stated per test (observed max-abs-diff
// printed). inv_freq (the scaling-formula target) agrees far more tightly.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::IsUnimplemented;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::Rope;
using engine::model::RopeScaling;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// tiny-llama RoPE constants (fixture invariants — tiny_llama.CONFIG_JSON).
constexpr int kTinyHeadDim = 16;
constexpr float kTinyTheta = 10000.0F;
constexpr std::int64_t kTinyMaxPos = 128;
// llama3 (committed Llama-3.1 config).
constexpr int kL3HeadDim = 128;
constexpr float kL3Theta = 500000.0F;
// linear (synthetic, tiny dims).
constexpr float kLinearFactor = 4.0F;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

[[nodiscard]] SafetensorsFile RopeFile() {
  auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/rope.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

[[nodiscard]] std::span<const std::int32_t> PosSpan(const Tensor& p) {
  return {p.data_ptr<std::int32_t>(), static_cast<std::size_t>(p.numel())};
}

// Fresh, writable f32 copy of a fixture tensor (rope_apply mutates in place;
// the mapped fixture buffer must not be touched).
[[nodiscard]] Tensor CopyOf(const Tensor& src) {
  Tensor dst = Unwrap(ops::zeros(src.shape(), DataType::kFloat32));
  EXPECT_TRUE(ops::copy(dst, src).ok());
  return dst;
}

// Gathers `table` rows at each position into a contiguous [P, cols] tensor, so
// a table covering many positions can be compared against a sparse golden.
[[nodiscard]] Tensor GatherRows(const Tensor& table, const Tensor& positions) {
  const std::int64_t p = positions.numel();
  const std::int64_t cols = table.shape().dim(1);
  Tensor got = Unwrap(ops::zeros(Shape{p, cols}, DataType::kFloat32));
  const auto* td = table.data_ptr<float>();
  const auto* pd = positions.data_ptr<std::int32_t>();
  auto* gd = got.data_ptr<float>();
  for (std::int64_t i = 0; i < p; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      gd[(i * cols) + j] = td[(pd[i] * cols) + j];
    }
  }
  return got;
}

// --- tiny-llama cos/sin tables vs HF ---

TEST(RopeTest, TinyTableMatchesFixture) {
  const SafetensorsFile file = RopeFile();
  const Tensor cos_exp = Unwrap(file.tensor("tiny_table.cos"));  // [128, 8]
  const Tensor sin_exp = Unwrap(file.tensor("tiny_table.sin"));

  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, std::nullopt, kTinyMaxPos));
  EXPECT_EQ(rope.head_dim(), kTinyHeadDim);
  EXPECT_EQ(rope.num_positions(), kTinyMaxPos);

  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-4;
  const ops::AllCloseResult rc =
      Unwrap(ops::allclose(rope.cos(), cos_exp, kRtol, kAtol));
  EXPECT_TRUE(rc.allclose) << "cos: " << rc.Summary();
  const ops::AllCloseResult rs =
      Unwrap(ops::allclose(rope.sin(), sin_exp, kRtol, kAtol));
  EXPECT_TRUE(rs.allclose) << "sin: " << rs.Summary();
  std::cerr << "[rope] tiny_table cos max_abs_diff=" << rc.max_abs_diff
            << " sin max_abs_diff=" << rs.max_abs_diff << "\n";
}

// --- tiny-llama apply (half-rotation) vs HF, sparse & contiguous positions ---

struct ApplyCase {
  const char* name;  // fixture prefix
};

class RopeApplyTest : public testing::TestWithParam<ApplyCase> {};

TEST_P(RopeApplyTest, MatchesFixture) {
  const SafetensorsFile file = RopeFile();
  const std::string base(GetParam().name);
  const Tensor positions = Unwrap(file.tensor(base + ".positions"));  // i32
  const Tensor q_exp = Unwrap(file.tensor(base + ".q_out"));
  const Tensor k_exp = Unwrap(file.tensor(base + ".k_out"));

  Tensor q = CopyOf(Unwrap(file.tensor(base + ".q")));  // [T, H, d]
  Tensor k = CopyOf(Unwrap(file.tensor(base + ".k")));  // [T, Hkv, d]

  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, std::nullopt, kTinyMaxPos));
  ASSERT_TRUE(rope.apply(q, k, PosSpan(positions)).ok());

  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-4;
  const ops::AllCloseResult rq = Unwrap(ops::allclose(q, q_exp, kRtol, kAtol));
  EXPECT_TRUE(rq.allclose) << base << " q: " << rq.Summary();
  const ops::AllCloseResult rk = Unwrap(ops::allclose(k, k_exp, kRtol, kAtol));
  EXPECT_TRUE(rk.allclose) << base << " k: " << rk.Summary();
  std::cerr << "[rope] " << base << " q max_abs_diff=" << rq.max_abs_diff
            << " k max_abs_diff=" << rk.max_abs_diff << "\n";
}

INSTANTIATE_TEST_SUITE_P(TinyLlama, RopeApplyTest,
                         testing::Values(ApplyCase{"tiny_sparse"},
                                         ApplyCase{"tiny_contig"}),
                         [](const testing::TestParamInfo<ApplyCase>& info) {
                           return std::string(info.param.name);
                         });

// --- Half-rotation, not interleaved: rotating [1,0,0,0] at position 1 puts the
// sine into element d/2, not element 1 (a silent-correctness guard) ---

TEST(RopeTest, HalfRotationLayoutNotInterleaved) {
  constexpr int kD = 4;
  const Rope rope = Unwrap(Rope::Create(kD, 100.0F, std::nullopt, 2));
  Tensor q = Unwrap(ops::zeros(Shape{1, 1, kD}, DataType::kFloat32));
  q.data_ptr<float>()[0] = 1.0F;  // q = [1, 0, 0, 0]
  const std::vector<std::int32_t> pos = {1};
  Tensor k = Unwrap(ops::zeros(Shape{1, 1, kD}, DataType::kFloat32));
  ASSERT_TRUE(rope.apply(q, k, std::span<const std::int32_t>(pos)).ok());

  // inv_freq[0] = 1 → angle 1. Half layout: out[0]=cos(1), out[2]=sin(1); the
  // interleaved layout would instead write sin into out[1].
  const auto* out = q.data_ptr<float>();
  EXPECT_NEAR(out[0], std::cos(1.0F), 1e-6F);
  EXPECT_NEAR(out[2], std::sin(1.0F), 1e-6F);
  EXPECT_EQ(out[1], 0.0F);
  EXPECT_EQ(out[3], 0.0F);
}

// --- Position 0 is the identity (cos 0 = 1, sin 0 = 0), bit-exact ---

TEST(RopeTest, PositionZeroIsIdentity) {
  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, std::nullopt, kTinyMaxPos));
  Tensor q = Unwrap(ops::zeros(Shape{1, 4, kTinyHeadDim}, DataType::kFloat32));
  Tensor k = Unwrap(ops::zeros(Shape{1, 2, kTinyHeadDim}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(q, 0.0, 1.0, 11).ok());
  ASSERT_TRUE(ops::fill_normal(k, 0.0, 1.0, 12).ok());
  const Tensor q0 = CopyOf(q);
  const Tensor k0 = CopyOf(k);

  const std::vector<std::int32_t> pos = {0};
  ASSERT_TRUE(rope.apply(q, k, std::span<const std::int32_t>(pos)).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(q, q0, 0.0, 0.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(k, k0, 0.0, 0.0)).allclose);
}

// --- Threading independence: op == serial loop over the SAME tables ---

TEST(RopeTest, ThreadingIsBitExactVsSerial) {
  const std::int64_t t = 130;
  const int heads = 4;
  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, std::nullopt, t));
  const Tensor x =
      Unwrap(ops::zeros(Shape{t, heads, kTinyHeadDim}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 99).ok());

  std::vector<std::int32_t> pos(static_cast<std::size_t>(t));
  for (std::int64_t i = 0; i < t; ++i) {
    pos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
  }

  Tensor got = CopyOf(x);
  ASSERT_TRUE(engine::cpu::rope_apply(got, std::span<const std::int32_t>(pos),
                                      rope.cos(), rope.sin())
                  .ok());

  // Serial reference reading the same cos/sin rows the op used.
  const Tensor naive = CopyOf(x);
  const std::int64_t half = kTinyHeadDim / 2;
  const auto* cos = rope.cos().data_ptr<float>();
  const auto* sin = rope.sin().data_ptr<float>();
  auto* nd = naive.data_ptr<float>();
  for (std::int64_t ti = 0; ti < t; ++ti) {
    const float* cr = cos + (pos[static_cast<std::size_t>(ti)] * half);
    const float* sr = sin + (pos[static_cast<std::size_t>(ti)] * half);
    for (int h = 0; h < heads; ++h) {
      float* v = nd + (((ti * heads) + h) * kTinyHeadDim);
      for (std::int64_t j = 0; j < half; ++j) {
        const float lo = v[j];
        const float hi = v[j + half];
        v[j] = (lo * cr[j]) - (hi * sr[j]);
        v[j + half] = (hi * cr[j]) + (lo * sr[j]);
      }
    }
  }
  const ops::AllCloseResult r = Unwrap(ops::allclose(got, naive, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- llama3 rope_scaling: inv_freq (tight) and cos/sin (documented tol) ---

TEST(RopeTest, Llama3ScaledFrequenciesMatchFixture) {
  const SafetensorsFile file = RopeFile();
  const Tensor inv_exp = Unwrap(file.tensor("llama3.inv_freq"));     // [64]
  const Tensor positions = Unwrap(file.tensor("llama3.positions"));  // i32
  const Tensor cos_exp = Unwrap(file.tensor("llama3.cos"));          // [P, 64]
  const Tensor sin_exp = Unwrap(file.tensor("llama3.sin"));

  RopeScaling scaling;
  scaling.rope_type = "llama3";
  scaling.factor = 8.0F;
  scaling.low_freq_factor = 1.0F;
  scaling.high_freq_factor = 4.0F;
  scaling.original_max_position_embeddings = 8192;

  // num_positions must cover the largest golden position.
  std::int64_t max_pos = 0;
  for (std::int64_t i = 0; i < positions.numel(); ++i) {
    max_pos =
        std::max<std::int64_t>(max_pos, positions.data_ptr<std::int32_t>()[i]);
  }
  const Rope rope =
      Unwrap(Rope::Create(kL3HeadDim, kL3Theta, scaling, max_pos + 1));

  // inv_freq is the scaling-formula target — expect very tight agreement.
  const Tensor inv_got =
      Unwrap(ops::zeros(inv_exp.shape(), DataType::kFloat32));
  const std::span<const float> inv = rope.inv_freq();
  ASSERT_EQ(static_cast<std::int64_t>(inv.size()), inv_exp.numel());
  std::ranges::copy(inv, inv_got.data_ptr<float>());
  const ops::AllCloseResult ri =
      Unwrap(ops::allclose(inv_got, inv_exp, 1e-5, 1e-6));
  EXPECT_TRUE(ri.allclose) << "inv_freq: " << ri.Summary();

  // cos/sin at the golden positions: larger positions diverge under fp32-cosf
  // vs fp64-cos range reduction; tolerance set above the observed max.
  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-3;
  const Tensor cos_got = GatherRows(rope.cos(), positions);
  const Tensor sin_got = GatherRows(rope.sin(), positions);
  const ops::AllCloseResult rc =
      Unwrap(ops::allclose(cos_got, cos_exp, kRtol, kAtol));
  EXPECT_TRUE(rc.allclose) << "cos: " << rc.Summary();
  const ops::AllCloseResult rs =
      Unwrap(ops::allclose(sin_got, sin_exp, kRtol, kAtol));
  EXPECT_TRUE(rs.allclose) << "sin: " << rs.Summary();
  std::cerr << "[rope] llama3 inv_freq max_abs_diff=" << ri.max_abs_diff
            << " cos max_abs_diff=" << rc.max_abs_diff
            << " sin max_abs_diff=" << rs.max_abs_diff << "\n";
}

// --- linear rope_scaling: inv_freq and cos/sin ---

TEST(RopeTest, LinearScaledFrequenciesMatchFixture) {
  const SafetensorsFile file = RopeFile();
  const Tensor inv_exp = Unwrap(file.tensor("linear.inv_freq"));
  const Tensor positions = Unwrap(file.tensor("linear.positions"));
  const Tensor cos_exp = Unwrap(file.tensor("linear.cos"));
  const Tensor sin_exp = Unwrap(file.tensor("linear.sin"));

  RopeScaling scaling;
  scaling.rope_type = "linear";
  scaling.factor = kLinearFactor;

  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, scaling, kTinyMaxPos));

  const Tensor inv_got =
      Unwrap(ops::zeros(inv_exp.shape(), DataType::kFloat32));
  const std::span<const float> inv = rope.inv_freq();
  std::ranges::copy(inv, inv_got.data_ptr<float>());
  EXPECT_TRUE(Unwrap(ops::allclose(inv_got, inv_exp, 1e-5, 1e-6)).allclose);

  const Tensor cos_got = GatherRows(rope.cos(), positions);
  const Tensor sin_got = GatherRows(rope.sin(), positions);
  EXPECT_TRUE(Unwrap(ops::allclose(cos_got, cos_exp, 1e-4, 1e-4)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(sin_got, sin_exp, 1e-4, 1e-4)).allclose);
}

// --- Error paths ---

TEST(RopeTest, CreateRejectsOddHeadDim) {
  EXPECT_TRUE(IsInvalidArgument(
      Rope::Create(15, kTinyTheta, std::nullopt, 8).status()));
}

TEST(RopeTest, CreateRejectsNonPositiveTheta) {
  EXPECT_TRUE(
      IsInvalidArgument(Rope::Create(16, 0.0F, std::nullopt, 8).status()));
}

TEST(RopeTest, CreateRejectsZeroPositions) {
  EXPECT_TRUE(IsInvalidArgument(
      Rope::Create(16, kTinyTheta, std::nullopt, 0).status()));
}

TEST(RopeTest, CreateRejectsUnknownRopeType) {
  RopeScaling scaling;
  scaling.rope_type = "yarn";
  const Status s = Rope::Create(16, kTinyTheta, scaling, 8).status();
  ASSERT_TRUE(IsUnimplemented(s)) << s.ToString();
  EXPECT_NE(s.message().find("yarn"), std::string_view::npos) << s.message();
  EXPECT_NE(s.message().find("llama3"), std::string_view::npos) << s.message();
}

TEST(RopeTest, CreateRejectsLlama3MissingOriginalMaxPos) {
  RopeScaling scaling;
  scaling.rope_type = "llama3";
  scaling.factor = 8.0F;
  scaling.low_freq_factor = 1.0F;
  scaling.high_freq_factor = 4.0F;
  scaling.original_max_position_embeddings = 0;  // missing
  EXPECT_TRUE(
      IsInvalidArgument(Rope::Create(128, kL3Theta, scaling, 8).status()));
}

TEST(RopeTest, ApplyRejectsWrongHeadDim) {
  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, std::nullopt, kTinyMaxPos));
  Tensor q = Unwrap(ops::zeros(Shape{1, 4, 8}, DataType::kFloat32));  // d != 16
  Tensor k = Unwrap(ops::zeros(Shape{1, 2, 8}, DataType::kFloat32));
  const std::vector<std::int32_t> pos = {0};
  const Status s = rope.apply(q, k, std::span<const std::int32_t>(pos));
  ASSERT_TRUE(IsInvalidArgument(s)) << s.ToString();
  EXPECT_NE(s.message().find('q'), std::string_view::npos) << s.message();
}

TEST(RopeTest, ApplyRejectsOutOfRangePosition) {
  const Rope rope =
      Unwrap(Rope::Create(kTinyHeadDim, kTinyTheta, std::nullopt, 4));
  Tensor q = Unwrap(ops::zeros(Shape{1, 4, kTinyHeadDim}, DataType::kFloat32));
  Tensor k = Unwrap(ops::zeros(Shape{1, 2, kTinyHeadDim}, DataType::kFloat32));
  const std::vector<std::int32_t> pos = {4};  // table has rows 0..3
  const Status s = rope.apply(q, k, std::span<const std::int32_t>(pos));
  ASSERT_TRUE(IsInvalidArgument(s)) << s.ToString();
  EXPECT_NE(s.message().find("positions[0]"), std::string_view::npos)
      << s.message();
}

}  // namespace
