#include "common/paths.h"
#include "cpu/ops.h"
#include "kernels/dispatch.h"
#include "kernels/internal/norm_impl.h"
#include "kernels/norm.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

// Optimized RMSNorm kernel (M6-T03; design:
// docs/design/optimized-cpu-execution.md §10). Validated against cpu::rmsnorm
// (the oracle). The optimized backend pre-converts norm scales to fp32, so the
// kernel takes an fp32 weight; the reference is called with the same fp32
// weight so the only divergence is accumulation order. Registered SCALAR_PASS.
//
// Tolerance (§10 RMSNorm): rtol 1e-5 (exact 1/sqrtf required — never rsqrte).
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

constexpr double kRtol = 1e-5;
constexpr double kAtol = 1e-6;

// Hidden sizes {odd, 1024, 4096} plus vector-width crossings and tails.
constexpr std::int64_t kEs[] = {1, 3, 7, 8, 16, 17, 1023, 1024, 4096};

[[nodiscard]] Tensor Reference(const Tensor& x, const Tensor& w, float eps) {
  Tensor y = Unwrap(ops::zeros(x.shape(), DataType::kFloat32));
  EXPECT_TRUE(engine::cpu::rmsnorm(x, w, eps, y).ok());
  return y;
}

TEST(NormKernelTest, MatchesOracleAcrossShapes) {
  std::uint64_t seed = 100;
  const float eps = 1e-5F;
  for (const std::int64_t e : kEs) {
    const std::int64_t rows = 4;
    const Tensor x = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, seed++).ok());
    const Tensor w = Unwrap(ops::zeros(Shape{e}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(w, 1.0, 0.2, seed++).ok());

    const Tensor ref = Reference(x, w, eps);
    const Tensor out = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
    RmsNormF32(x.data_ptr<float>(), w.data_ptr<float>(), eps, rows, e,
               out.data_ptr<float>());

    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << "e=" << e << ": " << r.Summary();
    std::cerr << "[rmsnorm] e=" << e << " max_abs_diff=" << r.max_abs_diff
              << '\n';
  }
}

// Magnitude extremes and eps that dominates — no overflow beyond the oracle.
TEST(NormKernelTest, MagnitudeExtremesAndEps) {
  struct Case {
    double stddev;
    float eps;
  };
  const Case cases[] = {{.stddev = 1e-20, .eps = 1e-6F},
                        {.stddev = 1e20, .eps = 1e-5F},
                        {.stddev = 1.0, .eps = 0.5F}};
  std::uint64_t seed = 200;
  for (const Case& c : cases) {
    const std::int64_t rows = 4;
    const std::int64_t e = 64;
    const Tensor x = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(x, 0.0, c.stddev, seed++).ok());
    const Tensor w = Unwrap(ops::zeros(Shape{e}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(w, 1.0, 0.1, seed++).ok());

    const Tensor ref = Reference(x, w, c.eps);
    const Tensor out = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
    RmsNormF32(x.data_ptr<float>(), w.data_ptr<float>(), c.eps, rows, e,
               out.data_ptr<float>());
    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << "stddev=" << c.stddev << ": " << r.Summary();
  }
}

TEST(NormKernelTest, ThreadingIsBitExactVsSerialVariant) {
  const std::int64_t rows = 64;
  const std::int64_t e = 300;
  const float eps = 1e-5F;
  const Tensor x = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 300).ok());
  const Tensor w = Unwrap(ops::zeros(Shape{e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w, 1.0, 0.2, 301).ok());

  const Tensor threaded =
      Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  RmsNormF32(x.data_ptr<float>(), w.data_ptr<float>(), eps, rows, e,
             threaded.data_ptr<float>());

  const Tensor serial = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  detail::RmsNormRowsVariant(SelectedIsa())(x.data_ptr<float>(),
                                            w.data_ptr<float>(), eps, rows, e,
                                            serial.data_ptr<float>());

  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
}

TEST(NormKernelTest, InPlaceMatchesOutOfPlace) {
  const std::int64_t rows = 5;
  const std::int64_t e = 65;
  const float eps = 1e-5F;
  const Tensor x = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 400).ok());
  const Tensor w = Unwrap(ops::zeros(Shape{e}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(w, 1.0, 0.2, 401).ok());
  const Tensor out = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
  RmsNormF32(x.data_ptr<float>(), w.data_ptr<float>(), eps, rows, e,
             out.data_ptr<float>());
  RmsNormF32(x.data_ptr<float>(), w.data_ptr<float>(), eps, rows, e,
             x.data_ptr<float>());  // in place
  EXPECT_TRUE(Unwrap(ops::allclose(x, out, 0.0, 0.0)).allclose);
}

TEST(NormKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::RmsNormRowsVariant(isa)),
            reinterpret_cast<void*>(&scalar::RmsNormRows));
}

// The M5 rmsnorm goldens (HF RMSNorm) replayed through the kernel. The golden
// weight is f32/bf16 per case; the kernel needs fp32, so a bf16 weight is cast
// once (what the load-time conversion does, §4).
TEST(NormKernelTest, MatchesFixtureGoldens) {
  const auto file =
      engine::model::SafetensorsFile::Open(engine::testing::FixturesDir() /
                                           "models/tiny-llama/expected/"
                                           "ops.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  struct Case {
    const char* name;
    float eps;
  };
  const Case cases[] = {{.name = "rmsnorm_f32", .eps = 1e-5F},
                        {.name = "rmsnorm_eps", .eps = 0.5F}};
  for (const Case& c : cases) {
    const std::string base(c.name);
    const Tensor x = Unwrap(file->tensor(base + ".x"));  // f32
    const Tensor w = Unwrap(file->tensor(base + ".weight"));
    const Tensor w32 = w.dtype() == DataType::kFloat32
                           ? w
                           : Unwrap(ops::cast(w, DataType::kFloat32));
    const Tensor expected = Unwrap(file->tensor(base + ".y"));
    const std::int64_t rows = x.shape().dim(0);
    const std::int64_t e = x.shape().dim(1);
    const Tensor out = Unwrap(ops::zeros(Shape{rows, e}, DataType::kFloat32));
    RmsNormF32(x.data_ptr<float>(), w32.data_ptr<float>(), c.eps, rows, e,
               out.data_ptr<float>());
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, 1e-4, 1e-5));
    EXPECT_TRUE(r.allclose) << c.name << ": " << r.Summary();
  }
}

}  // namespace
}  // namespace engine::kernels
