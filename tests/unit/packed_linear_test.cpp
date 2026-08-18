#include "model/packed_linear.h"

#include "common/paths.h"
#include "core/status.h"
#include "model/loader.h"
#include "model/modules.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <utility>

// PackedLinear (M6-T02; design: docs/design/optimized-cpu-execution.md §3, §4).
// The optimized `Linear` over the packed weight layout. Its forward is the
// dispatched PackedGemm, validated here against ReferenceLinear (cpu::gemm, the
// oracle) through the shared `Linear&` interface, on random and real
// checkpoint weights. Registered SCALAR_PASS (§9) — the first `model`-labelled
// suite that runs a dispatched kernel, so the forced-scalar pass covers this
// module too.
namespace {

namespace ops = engine::tensor::ops;
using engine::core::IsInvalidArgument;
using engine::core::Status;
using engine::core::StatusOr;
using engine::model::Linear;
using engine::model::load_model;
using engine::model::PackedLinear;
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

// PackedLinear.forward vs ReferenceLinear.forward for the same weight/bias/x,
// within the §10 GEMM tolerance (scalar is exact, NEON/AVX2 within band).
void ExpectMatchesReference(const Tensor& w, const std::optional<Tensor>& bias,
                            const Tensor& x) {
  const std::int64_t t = x.shape().dim(0);
  const std::int64_t out = w.shape().dim(0);

  const ReferenceLinear ref = Unwrap(ReferenceLinear::Create(w, bias));
  const PackedLinear packed = Unwrap(PackedLinear::Create(w, bias));

  const Linear& ref_iface = ref;
  const Linear& packed_iface = packed;
  ASSERT_EQ(packed.in_features(), ref.in_features());
  ASSERT_EQ(packed.out_features(), ref.out_features());
  ASSERT_EQ(packed.has_bias(), ref.has_bias());

  Tensor y_ref = Unwrap(ops::zeros(Shape{t, out}, DataType::kFloat32));
  Tensor y_packed = Unwrap(ops::zeros(Shape{t, out}, DataType::kFloat32));
  ASSERT_TRUE(ref_iface.forward(x, y_ref).ok());
  ASSERT_TRUE(packed_iface.forward(x, y_packed).ok());

  const ops::AllCloseResult r =
      Unwrap(ops::allclose(y_packed, y_ref, 1e-4, 1e-4));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

// --- Create: features, bias flag, packed shape ---

TEST(PackedLinearTest, CreateReportsFeaturesAndPackedShape) {
  const Tensor w = Unwrap(ops::zeros(Shape{17, 3}, DataType::kFloat32));  // pad
  const PackedLinear linear = Unwrap(PackedLinear::Create(w));
  EXPECT_EQ(linear.in_features(), 3);
  EXPECT_EQ(linear.out_features(), 17);
  EXPECT_FALSE(linear.has_bias());

  // Packed shape [P=ceil(17/16)=2, K=3, kNr=16], storage dtype preserved.
  const Tensor& wp = linear.packed_weight();
  EXPECT_EQ(wp.shape().rank(), 3);
  EXPECT_EQ(wp.shape().dim(0), 2);
  EXPECT_EQ(wp.shape().dim(1), 3);
  EXPECT_EQ(wp.shape().dim(2), 16);
  EXPECT_EQ(wp.dtype(), DataType::kFloat32);

  const Tensor bias = Unwrap(ops::zeros(Shape{17}, DataType::kFloat32));
  const PackedLinear with_bias = Unwrap(PackedLinear::Create(w, bias));
  EXPECT_TRUE(with_bias.has_bias());
}

TEST(PackedLinearTest, CreateRejectsMalformedWeightsAndBias) {
  // Weight not rank-2.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{5}, DataType::kFloat32));
    const auto r = PackedLinear::Create(w);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("weight must be rank-2"),
              std::string::npos)
        << r.status().message();
  }
  // Weight an unsupported dtype.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{5, 3}, DataType::kInt32));
    const auto r = PackedLinear::Create(w);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("weight dtype"), std::string::npos)
        << r.status().message();
  }
  // Bias length disagrees with out_features.
  {
    const Tensor w = Unwrap(ops::zeros(Shape{5, 3}, DataType::kFloat32));
    const Tensor bias = Unwrap(ops::zeros(Shape{4}, DataType::kFloat32));
    const auto r = PackedLinear::Create(w, bias);
    EXPECT_TRUE(IsInvalidArgument(r.status()));
    EXPECT_NE(r.status().message().find("bias length"), std::string::npos)
        << r.status().message();
  }
}

// --- forward vs the reference across dtypes, shapes, bias ---

TEST(PackedLinearTest, MatchesReferenceF32) {
  const Tensor w = Unwrap(ops::zeros(Shape{33, 40}, DataType::kFloat32));
  const Tensor x = Unwrap(ops::zeros(Shape{9, 40}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w, 0.0, 1.0, 1).ok());
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 2).ok());
  ExpectMatchesReference(w, std::nullopt, x);

  Tensor bias = Unwrap(ops::zeros(Shape{33}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(bias, 0.0, 1.0, 3).ok());
  ExpectMatchesReference(w, bias, x);
}

TEST(PackedLinearTest, MatchesReferenceBf16AndF16) {
  const Tensor w_f32 = Unwrap(ops::zeros(Shape{64, 96}, DataType::kFloat32));
  const Tensor x = Unwrap(ops::zeros(Shape{5, 96}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w_f32, 0.0, 1.0, 11).ok());
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 12).ok());

  const Tensor w_bf16 = Unwrap(ops::cast(w_f32, DataType::kBFloat16));
  const Tensor w_f16 = Unwrap(ops::cast(w_f32, DataType::kFloat16));
  ExpectMatchesReference(w_bf16, std::nullopt, x);
  ExpectMatchesReference(w_f16, std::nullopt, x);

  // bf16 weight + bf16 bias (the Qwen2 q/k/v shape).
  const Tensor bias_f32 = Unwrap(ops::zeros(Shape{64}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(bias_f32, 0.0, 1.0, 13).ok());
  const Tensor bias_bf16 = Unwrap(ops::cast(bias_f32, DataType::kBFloat16));
  ExpectMatchesReference(w_bf16, bias_bf16, x);
}

// T == 1 (decode / GEMV) goes through the same forward, no special-casing.
TEST(PackedLinearTest, DecodeGemvShapeMatchesReference) {
  const Tensor w_f32 = Unwrap(ops::zeros(Shape{200, 128}, DataType::kFloat32));
  const Tensor x = Unwrap(ops::zeros(Shape{1, 128}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w_f32, 0.0, 1.0, 21).ok());
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 22).ok());
  const Tensor w = Unwrap(ops::cast(w_f32, DataType::kBFloat16));
  ExpectMatchesReference(w, std::nullopt, x);
}

TEST(PackedLinearTest, ForwardRejectsBadActivationShapes) {
  const Tensor w = Unwrap(ops::zeros(Shape{6, 12}, DataType::kFloat32));
  const PackedLinear linear = Unwrap(PackedLinear::Create(w));

  // x inner dim != in_features.
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 8}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, 6}, DataType::kFloat32));
    const Status s = linear.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be"), std::string::npos) << s.message();
  }
  // y wrong shape.
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 12}, DataType::kFloat32));
    Tensor y = Unwrap(ops::zeros(Shape{2, 5}, DataType::kFloat32));
    const Status s = linear.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("y must be caller-allocated"), std::string::npos)
        << s.message();
  }
  // x not fp32.
  {
    const Tensor x = Unwrap(ops::zeros(Shape{2, 12}, DataType::kBFloat16));
    Tensor y = Unwrap(ops::zeros(Shape{2, 6}, DataType::kFloat32));
    const Status s = linear.forward(x, y);
    EXPECT_TRUE(IsInvalidArgument(s));
    EXPECT_NE(s.message().find("x must be fp32"), std::string::npos)
        << s.message();
  }
}

// End-to-end: a real bf16 projection weight, loaded zero-copy, packed, and
// forwarded — matches the ops.safetensors golden (loader → pack → kernel).
TEST(PackedLinearTest, RealCheckpointWeightMatchesGolden) {
  const auto loaded =
      load_model(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  const auto weight_it = loaded->weights.find("layers.0.attn.q_proj.weight");
  ASSERT_NE(weight_it, loaded->weights.end());
  const Tensor q_proj = weight_it->second;  // [64, 64], bf16
  ASSERT_EQ(q_proj.dtype(), DataType::kBFloat16);

  const auto ops_file =
      SafetensorsFile::Open(engine::testing::FixturesDir() /
                            "models/tiny-llama/expected/ops.safetensors");
  ASSERT_TRUE(ops_file.ok()) << ops_file.status().ToString();
  const Tensor a = Unwrap(ops_file->tensor("q_proj_prefill.a"));
  const Tensor expected = Unwrap(ops_file->tensor("q_proj_prefill.c"));

  const PackedLinear linear = Unwrap(PackedLinear::Create(q_proj));
  Tensor y = Unwrap(ops::zeros(expected.shape(), DataType::kFloat32));
  ASSERT_TRUE(linear.forward(a, y).ok());

  const ops::AllCloseResult r = Unwrap(ops::allclose(y, expected, 1e-4, 1e-4));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

}  // namespace
