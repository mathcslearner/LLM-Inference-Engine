#include "common/paths.h"
#include "cpu/ops.h"
#include "kernels/activation.h"
#include "kernels/dispatch.h"
#include "kernels/elementwise.h"
#include "kernels/internal/activation_impl.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

// Optimized SwiGLU-combine kernel (M6-T03; design:
// docs/design/optimized-cpu-execution.md §10). Validated against cpu::silu_mul
// (the oracle). This file also carries the **residual-add** acceptance: the
// M6 residual add is the existing Class-E kernels::AddF32 (§10 lists its oracle
// as cpu::add), so a bit-identity parity test against cpu::add closes the
// ticket's "residual add" box rather than adding a redundant kernel.
// Registered SCALAR_PASS.
//
// Tolerance (§10 SiLU): rtol 1e-5, atol 1e-6, derived from the ≤2-ulp exp.
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

// Flat lengths crossing vector widths, tails, and the threading grain (32768).
constexpr std::int64_t kNs[] = {1, 3, 7, 8, 17, 176, 4096, 32771};

TEST(ActivationKernelTest, MatchesOracleAcrossShapes) {
  std::uint64_t seed = 100;
  for (const std::int64_t n : kNs) {
    const Tensor gate = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
    const Tensor up = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(gate, 0.0, 2.0, seed++).ok());
    ASSERT_TRUE(ops::fill_normal(up, 0.0, 2.0, seed++).ok());

    Tensor ref = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
    ASSERT_TRUE(engine::cpu::silu_mul(gate, up, ref).ok());

    const Tensor out = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
    SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
               out.data_ptr<float>(), n);

    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << "n=" << n << ": " << r.Summary();
    std::cerr << "[silu_mul] n=" << n << " max_abs_diff=" << r.max_abs_diff
              << '\n';
  }
}

// Saturating extremes: very negative gate → silu → 0; very positive → silu → g.
// Must stay finite and match the oracle (also saturating).
TEST(ActivationKernelTest, SaturatingExtremesStayFinite) {
  const std::int64_t n = 8;
  const float gates[] = {-1e4F, -100.0F, -10.0F, 0.0F,
                         10.0F, 100.0F,  1e4F,   88.0F};
  const Tensor gate = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  const Tensor up = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  for (std::int64_t i = 0; i < n; ++i) {
    gate.data_ptr<float>()[i] = gates[i];
    up.data_ptr<float>()[i] = 1.0F;
  }
  Tensor ref = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  ASSERT_TRUE(engine::cpu::silu_mul(gate, up, ref).ok());
  const Tensor out = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
             out.data_ptr<float>(), n);
  const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
  EXPECT_TRUE(r.allclose) << r.Summary();
}

TEST(ActivationKernelTest, ThreadingIsBitExactVsSerialVariant) {
  const std::int64_t n = 100000;  // multi-chunk (grain 32768)
  const Tensor gate = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  const Tensor up = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(gate, 0.0, 2.0, 300).ok());
  ASSERT_TRUE(ops::fill_normal(up, 0.0, 2.0, 301).ok());

  const Tensor threaded = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
             threaded.data_ptr<float>(), n);
  const Tensor serial = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  detail::SiluMulVariant(SelectedIsa())(gate.data_ptr<float>(),
                                        up.data_ptr<float>(),
                                        serial.data_ptr<float>(), n);
  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
}

TEST(ActivationKernelTest, InPlaceMatchesOutOfPlace) {
  const std::int64_t n = 177;
  const Tensor gate = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  const Tensor up = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  ASSERT_TRUE(ops::fill_normal(gate, 0.0, 2.0, 400).ok());
  ASSERT_TRUE(ops::fill_normal(up, 0.0, 2.0, 401).ok());
  const Tensor out = Unwrap(ops::zeros(Shape{1, n}, DataType::kFloat32));
  SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
             out.data_ptr<float>(), n);
  // Alias y onto gate.
  SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
             gate.data_ptr<float>(), n);
  EXPECT_TRUE(Unwrap(ops::allclose(gate, out, 0.0, 0.0)).allclose);
}

TEST(ActivationKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::SiluMulVariant(isa)),
            reinterpret_cast<void*>(&scalar::SiluMul));
}

TEST(ActivationKernelTest, MatchesSiluMulGolden) {
  const auto file =
      engine::model::SafetensorsFile::Open(engine::testing::FixturesDir() /
                                           "models/tiny-llama/expected/"
                                           "ops.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  for (const char* name : {"silu_mul", "silu_mul_unit"}) {
    const std::string base(name);
    const Tensor gate = Unwrap(file->tensor(base + ".gate"));
    const Tensor up = Unwrap(file->tensor(base + ".up"));
    const Tensor expected = Unwrap(file->tensor(base + ".y"));
    const std::int64_t n = gate.shape().dim(0) * gate.shape().dim(1);
    const Tensor out = Unwrap(ops::zeros(gate.shape(), DataType::kFloat32));
    SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
               out.data_ptr<float>(), n);
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(out, expected, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << name << ": " << r.Summary();
  }
}

// --- Residual add (kernels::AddF32) acceptance: Class E, bit-identical to
// cpu::add over [T, E] activation shapes (§10). ---

TEST(ActivationKernelTest, ResidualAddMatchesOracleBitExact) {
  const std::int64_t shapes[][2] = {{1, 4096}, {7, 1024}, {17, 63}};
  std::uint64_t seed = 500;
  for (const auto& s : shapes) {
    const Tensor a = Unwrap(ops::zeros(Shape{s[0], s[1]}, DataType::kFloat32));
    const Tensor b = Unwrap(ops::zeros(Shape{s[0], s[1]}, DataType::kFloat32));
    ASSERT_TRUE(ops::fill_normal(a, 0.0, 1.0, seed++).ok());
    ASSERT_TRUE(ops::fill_normal(b, 0.0, 1.0, seed++).ok());

    Tensor ref = Unwrap(ops::zeros(Shape{s[0], s[1]}, DataType::kFloat32));
    ASSERT_TRUE(engine::cpu::add(a, b, ref).ok());

    const Tensor out =
        Unwrap(ops::zeros(Shape{s[0], s[1]}, DataType::kFloat32));
    AddF32(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
           s[0] * s[1]);
    // Class E: exact bit equality.
    EXPECT_TRUE(Unwrap(ops::allclose(out, ref, 0.0, 0.0)).allclose)
        << s[0] << "x" << s[1];
  }
}

}  // namespace
}  // namespace engine::kernels
