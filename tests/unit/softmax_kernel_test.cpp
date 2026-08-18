#include "common/paths.h"
#include "cpu/ops.h"
#include "kernels/dispatch.h"
#include "kernels/internal/softmax_impl.h"
#include "kernels/softmax.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

// Optimized softmax kernel (M6-T03; design:
// docs/design/optimized-cpu-execution.md §10). Validated against cpu::softmax
// (the oracle, itself validated against HF fixtures). Registered SCALAR_PASS.
//
// Tolerance (§10 softmax): atol 1e-6, rtol 1e-4 — outputs lie in [0, 1] and the
// only divergence from the oracle is the vector `exp` (≤2 ulp) plus horizontal
// reduction order. The observed max-abs-diff is logged and sits far below.
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

constexpr double kAtol = 1e-6;
constexpr double kRtol = 1e-4;

// Hidden sizes crossing the vector widths (NEON 4, AVX2 8), odd tails, and the
// acceptance sizes {odd, 1024, 4096}. Rows small (softmax is row-parallel).
constexpr std::int64_t kNs[] = {1, 3, 5, 7, 8, 16, 17, 33, 1023, 1024, 4096};

[[nodiscard]] Tensor Reference(const Tensor& x) {
  Tensor y = Unwrap(ops::zeros(x.shape(), DataType::kFloat32));
  EXPECT_TRUE(engine::cpu::softmax(x, y).ok());
  return y;
}

TEST(SoftmaxKernelTest, MatchesOracleAcrossShapes) {
  std::uint64_t seed = 100;
  for (const std::int64_t n : kNs) {
    const std::int64_t rows = 5;
    const Tensor x = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(x, 0.0, 3.0, seed++).ok());

    const Tensor ref = Reference(x);
    const Tensor out = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
    SoftmaxF32(x.data_ptr<float>(), out.data_ptr<float>(), rows, n);

    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << "n=" << n << ": " << r.Summary();
    std::cerr << "[softmax] n=" << n << " max_abs_diff=" << r.max_abs_diff
              << '\n';
  }
}

// Large-magnitude rows: the stability path (subtract row max) must not overflow
// to NaN/inf, and the result must still match the oracle.
TEST(SoftmaxKernelTest, LargeMagnitudeStable) {
  const double scales[] = {1e4, 1e30};
  std::uint64_t seed = 200;
  for (const double scale : scales) {
    const std::int64_t rows = 4;
    const std::int64_t n = 512;
    const Tensor x = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(x, 0.0, scale, seed++).ok());

    const Tensor ref = Reference(x);
    const Tensor out = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
    SoftmaxF32(x.data_ptr<float>(), out.data_ptr<float>(), rows, n);

    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << "scale=" << scale << ": " << r.Summary();
  }
}

// A `-inf` causal-mask entry must map to exactly 0 (the contract the exp flush
// delivers), and the remaining row must still sum to 1.
TEST(SoftmaxKernelTest, CausalMaskEntriesAreExactlyZero) {
  const std::int64_t rows = 3;
  const std::int64_t n = 40;
  const Tensor x = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 1.0, 300).ok());
  auto* xd = x.data_ptr<float>();
  const float ninf = -std::numeric_limits<float>::infinity();
  // Causal shape: row r masks columns > r (as attention would).
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t j = r + 1; j < n; ++j) {
      xd[(r * n) + j] = ninf;
    }
  }
  const Tensor out = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  SoftmaxF32(xd, out.data_ptr<float>(), rows, n);
  const auto* od = out.data_ptr<float>();
  for (std::int64_t r = 0; r < rows; ++r) {
    float sum = 0.0F;
    for (std::int64_t j = 0; j < n; ++j) {
      if (j > r) {
        EXPECT_EQ(od[(r * n) + j], 0.0F) << "r=" << r << " j=" << j;
      }
      sum += od[(r * n) + j];
    }
    EXPECT_NEAR(sum, 1.0F, 1e-5) << "r=" << r;
  }
}

// The threaded public entry is bit-identical to a single serial variant call
// over the whole problem — so the result is invariant to thread count /
// chunking (each row is wholly computed within one variant call, §10).
TEST(SoftmaxKernelTest, ThreadingIsBitExactVsSerialVariant) {
  const std::int64_t rows = 64;
  const std::int64_t n = 300;
  const Tensor x = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 2.0, 400).ok());

  const Tensor threaded =
      Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  SoftmaxF32(x.data_ptr<float>(), threaded.data_ptr<float>(), rows, n);

  const Tensor serial = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  detail::SoftmaxRowsVariant(SelectedIsa())(x.data_ptr<float>(),
                                            serial.data_ptr<float>(), rows, n);

  // Also an arbitrary manual chunking, proving invariance directly.
  const Tensor chunked = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  const auto fn = detail::SoftmaxRowsVariant(SelectedIsa());
  const std::int64_t bounds[] = {0, 1, 3, 30, 31, rows};
  for (std::size_t b = 0; b + 1 < std::size(bounds); ++b) {
    const std::int64_t r0 = bounds[b];
    const std::int64_t r1 = bounds[b + 1];
    fn(x.data_ptr<float>() + (r0 * n), chunked.data_ptr<float>() + (r0 * n),
       r1 - r0, n);
  }

  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(chunked, serial, 0.0, 0.0)).allclose);
}

// y may alias x.
TEST(SoftmaxKernelTest, InPlaceMatchesOutOfPlace) {
  const std::int64_t rows = 7;
  const std::int64_t n = 65;
  const Tensor x = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(x, 0.0, 2.0, 500).ok());
  const Tensor out = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
  SoftmaxF32(x.data_ptr<float>(), out.data_ptr<float>(), rows, n);
  SoftmaxF32(x.data_ptr<float>(), x.data_ptr<float>(), rows, n);  // in place
  EXPECT_TRUE(Unwrap(ops::allclose(x, out, 0.0, 0.0)).allclose);
}

TEST(SoftmaxKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::SoftmaxRowsVariant(isa)),
            reinterpret_cast<void*>(&scalar::SoftmaxRows));
}

// The M5 softmax goldens (HF torch.softmax) replayed through the kernel — ties
// the optimized path to the HF chain, not just the reference.
TEST(SoftmaxKernelTest, MatchesFixtureGoldens) {
  const auto file =
      engine::model::SafetensorsFile::Open(engine::testing::FixturesDir() /
                                           "models/tiny-llama/expected/"
                                           "ops.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  const char* names[] = {"softmax_typical", "softmax_large", "softmax_causal"};
  for (const char* name : names) {
    const std::string base(name);
    const Tensor x = Unwrap(file->tensor(base + ".x"));
    const Tensor expected = Unwrap(file->tensor(base + ".y"));
    const std::int64_t rows = x.shape().dim(0);
    const std::int64_t n = x.shape().dim(1);
    const Tensor out = Unwrap(ops::zeros(Shape{rows, n}, DataType::kFloat32));
    SoftmaxF32(x.data_ptr<float>(), out.data_ptr<float>(), rows, n);
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
  }
}

}  // namespace
}  // namespace engine::kernels
