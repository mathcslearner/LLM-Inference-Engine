#include "common/paths.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

// cpu::silu_mul / cpu::add / cpu::softmax goldens & properties (M5-T03; design:
// docs/design/model-execution.md §4.2, §12, §13). Elementwise ops are bit-exact
// vs torch (same f32 arithmetic); softmax is Class T (torch subtracts max in
// its own order). Each numerical test states its tolerance and prints the
// observed max-abs-diff.
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

[[nodiscard]] SafetensorsFile OpsFile() {
  auto file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  EXPECT_TRUE(file.ok()) << file.status().ToString();
  return *std::move(file);
}

[[nodiscard]] bool AllFinite(const Tensor& t) {
  const auto* d = t.data_ptr<float>();
  for (std::int64_t i = 0; i < t.shape().numel(); ++i) {
    if (!std::isfinite(d[i])) {
      return false;
    }
  }
  return true;
}

// ================================ silu_mul ================================

TEST(CpuSiluMulTest, MatchesFixture) {
  const SafetensorsFile file = OpsFile();
  // silu is smooth; the only divergence from torch F.silu(gate)*up is fp32
  // rounding of x/(1+exp(-x)) vs x*sigmoid(x). Observed diff printed per case.
  constexpr double kRtol = 1e-4;
  constexpr double kAtol = 1e-4;
  for (const char* name : {"silu_mul", "silu_mul_unit", "silu_mul_large"}) {
    const std::string base(name);
    const Tensor gate = Unwrap(file.tensor(base + ".gate"));
    const Tensor up = Unwrap(file.tensor(base + ".up"));
    const Tensor expected = Unwrap(file.tensor(base + ".y"));

    Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
    const Status s = engine::cpu::silu_mul(gate, up, out);
    ASSERT_TRUE(s.ok()) << name << ": " << s.ToString();

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
    EXPECT_TRUE(AllFinite(out)) << name << ": non-finite output";
    std::cerr << "[silu_mul] " << name << " max_abs_diff=" << r.max_abs_diff
              << "\n";
  }
}

TEST(CpuSiluMulTest, AliasingIsBitExact) {
  const Tensor gate = Unwrap(ops::zeros(Shape{6, 20}, DataType::kFloat32));
  const Tensor up = Unwrap(ops::zeros(Shape{6, 20}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(gate, 0.0, 1.0, 1).ok());
  ASSERT_TRUE(ops::fill_normal(up, 0.0, 1.0, 2).ok());

  Tensor out = Unwrap(ops::zeros(Shape{6, 20}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::silu_mul(gate, up, out).ok());

  // y aliases gate.
  Tensor into_gate = Unwrap(ops::zeros(Shape{6, 20}, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(into_gate, gate).ok());
  ASSERT_TRUE(engine::cpu::silu_mul(into_gate, up, into_gate).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(into_gate, out, 0.0, 0.0)).allclose);

  // y aliases up.
  Tensor into_up = Unwrap(ops::zeros(Shape{6, 20}, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(into_up, up).ok());
  ASSERT_TRUE(engine::cpu::silu_mul(gate, into_up, into_up).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(into_up, out, 0.0, 0.0)).allclose);
}

TEST(CpuSiluMulTest, RejectsMalformedInputs) {
  const Tensor gate = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
  const Tensor up = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
  Tensor y = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
  // gate not f32.
  {
    const Tensor g = Unwrap(ops::zeros(Shape{4, 8}, DataType::kBFloat16));
    const Status s = engine::cpu::silu_mul(g, up, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("gate must be f32"), std::string::npos)
        << s.message();
  }
  // up shape mismatch.
  {
    const Tensor u = Unwrap(ops::zeros(Shape{4, 7}, DataType::kFloat32));
    const Status s = engine::cpu::silu_mul(gate, u, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("up must be"), std::string::npos) << s.message();
  }
  // y shape mismatch.
  {
    Tensor y_bad = Unwrap(ops::zeros(Shape{4, 7}, DataType::kFloat32));
    const Status s = engine::cpu::silu_mul(gate, up, y_bad);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be"), std::string::npos) << s.message();
  }
}

// ================================== add ===================================

TEST(CpuAddTest, MatchesFixture) {
  const SafetensorsFile file = OpsFile();
  const Tensor a = Unwrap(file.tensor("add_basic.a"));
  const Tensor b = Unwrap(file.tensor("add_basic.b"));
  const Tensor expected = Unwrap(file.tensor("add_basic.y"));

  Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::add(a, b, out).ok());
  // f32 add matches torch's f32 add bit-for-bit.
  EXPECT_TRUE(Unwrap(ops::allclose(out, expected, 0.0, 0.0)).allclose);
}

TEST(CpuAddTest, AliasingIsBitExact) {
  const Tensor a = Unwrap(ops::zeros(Shape{5, 9}, DataType::kFloat32));
  const Tensor b = Unwrap(ops::zeros(Shape{5, 9}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(a, 0.0, 1.0, 5).ok());
  ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, 6).ok());

  Tensor out = Unwrap(ops::zeros(Shape{5, 9}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::add(a, b, out).ok());

  Tensor into_a = Unwrap(ops::zeros(Shape{5, 9}, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(into_a, a).ok());
  ASSERT_TRUE(engine::cpu::add(into_a, b, into_a).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(into_a, out, 0.0, 0.0)).allclose);
}

TEST(CpuAddTest, RejectsMalformedInputs) {
  const Tensor a = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
  Tensor y = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
  // b shape mismatch.
  {
    const Tensor b = Unwrap(ops::zeros(Shape{4, 7}, DataType::kFloat32));
    const Status s = engine::cpu::add(a, b, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("b must be"), std::string::npos) << s.message();
  }
  // a not f32.
  {
    const Tensor a_bf = Unwrap(ops::zeros(Shape{4, 8}, DataType::kBFloat16));
    const Tensor b = Unwrap(ops::zeros(Shape{4, 8}, DataType::kFloat32));
    const Status s = engine::cpu::add(a_bf, b, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("a must be f32"), std::string::npos)
        << s.message();
  }
}

// ================================ softmax =================================

TEST(CpuSoftmaxTest, MatchesFixture) {
  const SafetensorsFile file = OpsFile();
  // Stable softmax (max-subtracted) vs torch's; divergence is reduction order
  // only (Class T). Observed diff printed per case.
  constexpr double kRtol = 1e-5;
  constexpr double kAtol = 1e-6;
  for (const char* name :
       {"softmax_typical", "softmax_large", "softmax_causal"}) {
    const std::string base(name);
    const Tensor x = Unwrap(file.tensor(base + ".x"));
    const Tensor expected = Unwrap(file.tensor(base + ".y"));

    Tensor out = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
    const Status s = engine::cpu::softmax(x, out);
    ASSERT_TRUE(s.ok()) << name << ": " << s.ToString();
    EXPECT_TRUE(AllFinite(out)) << name << ": non-finite output";

    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
    std::cerr << "[softmax] " << name << " max_abs_diff=" << r.max_abs_diff
              << "\n";
  }
}

TEST(CpuSoftmaxTest, RowsSumToOne) {
  const SafetensorsFile file = OpsFile();
  for (const char* name :
       {"softmax_typical", "softmax_large", "softmax_causal"}) {
    const Tensor x = Unwrap(file.tensor(std::string(name) + ".x"));
    Tensor out = Unwrap(ops::zeros(x.shape(), DataType::kFloat32));
    ASSERT_TRUE(engine::cpu::softmax(x, out).ok());
    const std::int64_t rows = out.shape().dim(0);
    const std::int64_t n = out.shape().dim(1);
    const auto* d = out.data_ptr<float>();
    for (std::int64_t r = 0; r < rows; ++r) {
      double sum = 0.0;
      for (std::int64_t j = 0; j < n; ++j) {
        sum += d[(r * n) + j];
      }
      EXPECT_NEAR(sum, 1.0, 1e-6) << name << " row " << r;
    }
  }
}

// A strict-upper-triangle -inf mask (softmax_causal): masked entries are
// exactly 0, and the reference produces no NaN — the property M5-T05 relies on.
TEST(CpuSoftmaxTest, CausalMaskedEntriesAreExactlyZero) {
  const SafetensorsFile file = OpsFile();
  const Tensor x = Unwrap(file.tensor("softmax_causal.x"));  // [8,8]
  Tensor out = Unwrap(ops::zeros(x.shape(), DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::softmax(x, out).ok());
  ASSERT_TRUE(AllFinite(out));

  const std::int64_t n = out.shape().dim(1);
  const auto* d = out.data_ptr<float>();
  for (std::int64_t i = 0; i < out.shape().dim(0); ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      if (j > i) {
        EXPECT_EQ(d[(i * n) + j], 0.0F) << "masked (" << i << "," << j << ")";
      } else {
        EXPECT_GT(d[(i * n) + j], 0.0F) << "valid (" << i << "," << j << ")";
      }
    }
  }
}

TEST(CpuSoftmaxTest, InPlaceIsBitExact) {
  const Tensor x = Unwrap(ops::zeros(Shape{7, 33}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 3.0, 8).ok());

  Tensor out = Unwrap(ops::zeros(Shape{7, 33}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::softmax(x, out).ok());

  Tensor inplace = Unwrap(ops::zeros(Shape{7, 33}, DataType::kFloat32));
  ASSERT_TRUE(ops::copy(inplace, x).ok());
  ASSERT_TRUE(engine::cpu::softmax(inplace, inplace).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(inplace, out, 0.0, 0.0)).allclose);
}

TEST(CpuSoftmaxTest, SingleColumnIsOne) {
  const Tensor x = Unwrap(ops::full(Shape{3, 1}, DataType::kFloat32, -12345.0));
  Tensor out = Unwrap(ops::zeros(Shape{3, 1}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::softmax(x, out).ok());
  const auto* d = out.data_ptr<float>();
  for (int r = 0; r < 3; ++r) {
    EXPECT_EQ(d[r], 1.0F) << r;  // exp(x-x)/exp(x-x) == 1 regardless of value
  }
}

TEST(CpuSoftmaxTest, RejectsMalformedInputs) {
  const Tensor x = Unwrap(ops::zeros(Shape{3, 8}, DataType::kFloat32));
  Tensor y = Unwrap(ops::zeros(Shape{3, 8}, DataType::kFloat32));
  // x rank.
  {
    const Tensor x3 = Unwrap(ops::zeros(Shape{2, 3, 8}, DataType::kFloat32));
    const Status s = engine::cpu::softmax(x3, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be rank-2"), std::string::npos)
        << s.message();
  }
  // x not f32.
  {
    const Tensor x_bf = Unwrap(ops::zeros(Shape{3, 8}, DataType::kBFloat16));
    const Status s = engine::cpu::softmax(x_bf, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be f32"), std::string::npos)
        << s.message();
  }
  // y wrong shape.
  {
    Tensor y_bad = Unwrap(ops::zeros(Shape{3, 7}, DataType::kFloat32));
    const Status s = engine::cpu::softmax(x, y_bad);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be"), std::string::npos) << s.message();
  }
}

}  // namespace
