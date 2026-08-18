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

#include <string>
#include <utility>

// ReferenceLinear (M5-T02; design: docs/design/model-execution.md §4.1): the
// reference Linear over a zero-copy checkpoint weight. Its forward is exactly
// cpu::gemm, exercised here through the abstract `Linear&` interface and
// against a real bf16 checkpoint weight loaded end-to-end.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::Linear;
using engine::model::load_model;
using engine::model::ReferenceLinear;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

TEST(ReferenceLinearTest, CreateReportsFeaturesAndBias) {
  const Tensor w =
      Unwrap(ops::zeros(Shape{5, 3}, DataType::kFloat32));  // [out, in]
  const ReferenceLinear no_bias = Unwrap(ReferenceLinear::Create(w));
  EXPECT_EQ(no_bias.in_features(), 3);
  EXPECT_EQ(no_bias.out_features(), 5);
  EXPECT_FALSE(no_bias.has_bias());

  const Tensor bias = Unwrap(ops::zeros(Shape{5}, DataType::kFloat32));
  const ReferenceLinear with_bias = Unwrap(ReferenceLinear::Create(w, bias));
  EXPECT_TRUE(with_bias.has_bias());
  EXPECT_EQ(with_bias.out_features(), 5);
}

TEST(ReferenceLinearTest, CreateRejectsMalformedWeightsAndBias) {
  // Weight not rank-2.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{5}, DataType::kFloat32));
    const auto r = ReferenceLinear::Create(w);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("weight must be rank-2"),
              std::string::npos)
        << r.status().message();
  }
  // Weight an unsupported dtype.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{5, 3}, DataType::kInt32));
    const auto r = ReferenceLinear::Create(w);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("weight dtype"), std::string::npos)
        << r.status().message();
  }
  // Bias length disagrees with out_features.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{5, 3}, DataType::kFloat32));
    const Tensor bias = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
    const auto r = ReferenceLinear::Create(w, bias);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("bias length"), std::string::npos)
        << r.status().message();
  }
}

// forward through the abstract Linear& must equal a direct cpu::gemm call.
TEST(ReferenceLinearTest, ForwardEqualsDirectGemm) {
  const Tensor w = Unwrap(ops::zeros(Shape{9, 40}, DataType::kFloat32));
  Tensor bias = Unwrap(ops::zeros(Shape{9}, DataType::kFloat32));
  const Tensor x = Unwrap(ops::zeros(Shape{7, 40}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w, 0.0, 1.0, 1).ok());
  ASSERT_TRUE(ops::fill_normal(bias, 0.0, 1.0, 2).ok());
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 3).ok());

  const ReferenceLinear linear = Unwrap(ReferenceLinear::Create(w, bias));
  const Linear& iface = linear;

  Tensor via_module = Unwrap(ops::zeros(Shape{7, 9}, DataType::kFloat32));
  ASSERT_TRUE(iface.forward(x, via_module).ok());

  Tensor via_gemm = Unwrap(ops::zeros(Shape{7, 9}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::gemm(x, w, &bias, via_gemm).ok());

  const ops::AllCloseResult r =
      Unwrap(ops::allclose(via_module, via_gemm, 0.0, 0.0));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// T == 1 (decode / GEMV) goes through the same forward with no special-casing.
TEST(ReferenceLinearTest, DecodeGemvShape) {
  const Tensor w = Unwrap(ops::zeros(Shape{6, 12}, DataType::kFloat32));
  const Tensor x = Unwrap(ops::zeros(Shape{1, 12}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w, 0.0, 1.0, 4).ok());
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 5).ok());

  const ReferenceLinear linear = Unwrap(ReferenceLinear::Create(w));
  Tensor y = Unwrap(ops::zeros(Shape{1, 6}, DataType::kFloat32));
  ASSERT_TRUE(linear.forward(x, y).ok());
  Tensor ref = Unwrap(ops::zeros(Shape{1, 6}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::gemm(x, w, nullptr, ref).ok());
  EXPECT_TRUE(Unwrap(ops::allclose(y, ref, 0.0, 0.0)).allclose);
}

TEST(ReferenceLinearTest, ForwardRejectsBadActivationShapes) {
  const Tensor w = Unwrap(ops::zeros(Shape{6, 12}, DataType::kFloat32));
  const ReferenceLinear linear = Unwrap(ReferenceLinear::Create(w));

  // x inner dim != in_features.
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 8}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, 6}, DataType::kFloat32));
    const Status s = linear.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be"), std::string::npos) << s.message();
  }
  // y wrong shape (rows or cols).
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 12}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, 5}, DataType::kFloat32));
    const Status s = linear.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be caller-allocated"), std::string::npos)
        << s.message();
  }
}

// End-to-end: a real bf16 projection weight, loaded zero-copy, forwarded
// through ReferenceLinear, matches the standalone gemm golden for the same
// input. This ties the loader → module → gemm path together on genuine
// checkpoint bytes.
TEST(ReferenceLinearTest, RealCheckpointWeightMatchesGolden) {
  const auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const auto weight_it = loaded->weights.find("layers.0.attn.q_proj.weight");
  ASSERT_NE(weight_it, loaded->weights.end());
  const Tensor q_proj = weight_it->second;  // [64, 64], bf16, zero-copy
  ASSERT_EQ(q_proj.dtype(), DataType::kBFloat16);

  const auto ops_file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  ASSERT_TRUE(ops_file.ok()) << ops_file.status().ToString();
  const Tensor a = Unwrap(ops_file->tensor("q_proj_prefill.a"));  // [16,64]
  const Tensor expected =
      Unwrap(ops_file->tensor("q_proj_prefill.c"));  // [16,64]

  const ReferenceLinear linear = Unwrap(ReferenceLinear::Create(q_proj));
  Tensor y = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
  ASSERT_TRUE(linear.forward(a, y).ok());

  const ops::AllCloseResult r = Unwrap(ops::allclose(y, expected, 1e-4, 1e-4));
  EXPECT_TRUE(r.allclose) << r.Summary();
  std::cerr << "[linear] real q_proj max_abs_diff=" << r.max_abs_diff << "\n";
}

}  // namespace
