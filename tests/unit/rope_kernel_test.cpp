#include "common/paths.h"
#include "cpu/ops.h"
#include "kernels/dispatch.h"
#include "kernels/internal/rope_impl.h"
#include "kernels/rope.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>

// Optimized RoPE-apply kernel (M6-T03; design:
// docs/design/optimized-cpu-execution.md §10). Validated against
// cpu::rope_apply (the oracle). The op is table-agnostic, so the synthetic
// tests build their own cos/sin tables; the golden replay uses the HF-derived
// tiny-llama tables. Registered SCALAR_PASS.
//
// Tolerance (§10 RoPE): rtol 1e-5, atol 1e-6 (FMA in the rotate; tables shared
// fp32). Positions are arbitrary (unsorted/repeated) — the batched/paged-ready
// contract (§8).
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

// Builds a synthetic cos/sin table [p_count, half] and a random x [t, hx, d].
struct Fixture {
  Tensor x;
  Tensor cos;
  Tensor sin;
  std::vector<std::int32_t> positions;
};

[[nodiscard]] Fixture MakeFixture(std::int64_t t, std::int64_t hx,
                                  std::int64_t d, std::int64_t p_count,
                                  const std::vector<std::int32_t>& positions,
                                  std::uint64_t seed) {
  Fixture f;
  f.x = Unwrap(ops::zeros(Shape{t, hx, d}, DataType::kFloat32));
  EXPECT_TRUE(ops::fill_normal(f.x, 0.0, 1.0, seed).ok());
  f.cos = Unwrap(ops::zeros(Shape{p_count, d / 2}, DataType::kFloat32));
  f.sin = Unwrap(ops::zeros(Shape{p_count, d / 2}, DataType::kFloat32));
  EXPECT_TRUE(ops::fill_uniform(f.cos, -1.0, 1.0, seed + 1).ok());
  EXPECT_TRUE(ops::fill_uniform(f.sin, -1.0, 1.0, seed + 2).ok());
  f.positions = positions;
  return f;
}

// Runs the oracle on a copy of x and returns the rotated copy.
[[nodiscard]] Tensor Reference(const Fixture& f) {
  Tensor x = Unwrap(ops::zeros(f.x.shape(), DataType::kFloat32));
  EXPECT_TRUE(ops::copy(x, f.x).ok()) << "copy";
  const std::span<const std::int32_t> pos(f.positions);
  EXPECT_TRUE(engine::cpu::rope_apply(x, pos, f.cos, f.sin).ok());
  return x;
}

// Runs the kernel on a copy of x and returns the rotated copy.
[[nodiscard]] Tensor Kernel(const Fixture& f) {
  Tensor x = Unwrap(ops::zeros(f.x.shape(), DataType::kFloat32));
  EXPECT_TRUE(ops::copy(x, f.x).ok());
  const std::int64_t t = f.x.shape().dim(0);
  const std::int64_t hx = f.x.shape().dim(1);
  const std::int64_t d = f.x.shape().dim(2);
  RopeApplyF32(x.data_ptr<float>(), t, hx, d, f.positions.data(),
               f.cos.data_ptr<float>(), f.sin.data_ptr<float>());
  return x;
}

// Head-dims incl. Qwen's decoupled 24 (half=12, exercises the AVX2 8-lane tail)
// and 64/128; GQA-style head counts (Q vs KV shapes are just different hx).
TEST(RopeKernelTest, MatchesOracleAcrossShapesAndPositions) {
  struct Case {
    std::int64_t t, hx, d;
    const char* label;
  };
  const Case cases[] = {
      {.t = 4, .hx = 8, .d = 64, .label = "q_h8_d64"},
      {.t = 4, .hx = 2, .d = 64, .label = "kv_h2_d64"},
      {.t = 3, .hx = 14, .d = 24, .label = "q_h14_d24"},
      {.t = 3, .hx = 2, .d = 24, .label = "kv_h2_d24"},
      {.t = 5, .hx = 4, .d = 128, .label = "q_h4_d128"},
  };
  const std::int64_t p_count = 40000;  // to include a "large" position.
  std::uint64_t seed = 100;
  for (const Case& c : cases) {
    // Positions {0, 1, large, ...} truncated/cycled to length t.
    const std::int32_t bank[] = {0, 1, 39999, 12345, 2};
    std::vector<std::int32_t> pos(static_cast<std::size_t>(c.t));
    for (std::int64_t i = 0; i < c.t; ++i) {
      pos[static_cast<std::size_t>(i)] = bank[i % 5];
    }
    const Fixture f = MakeFixture(c.t, c.hx, c.d, p_count, pos, seed);
    seed += 10;
    const Tensor ref = Reference(f);
    const Tensor out = Kernel(f);
    const ops::AllCloseResult r = Unwrap(ops::allclose(out, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << c.label << ": " << r.Summary();
    std::cerr << "[rope] " << c.label << " max_abs_diff=" << r.max_abs_diff
              << '\n';
  }
}

// Unsorted and repeated positions (batched/paged-ready contract, §8).
TEST(RopeKernelTest, UnsortedAndRepeatedPositions) {
  const std::vector<std::int32_t> pos = {7, 0, 7, 3, 0, 100};
  const Fixture f = MakeFixture(/*t=*/6, /*hx=*/4, /*d=*/64,
                                /*p_count=*/200, pos, 700);
  const Tensor ref = Reference(f);
  const Tensor out = Kernel(f);
  EXPECT_TRUE(Unwrap(ops::allclose(out, ref, kRtol, kAtol)).allclose);
}

// Rotating token t in isolation equals its slice of the full batched call —
// tokens are independent (the property that makes token-parallelism valid).
TEST(RopeKernelTest, TokensAreIndependent) {
  const std::vector<std::int32_t> pos = {5, 11, 2, 88};
  const Fixture f = MakeFixture(/*t=*/4, /*hx=*/3, /*d=*/32,
                                /*p_count=*/128, pos, 800);
  const Tensor full = Kernel(f);
  const std::int64_t hx = 3;
  const std::int64_t d = 32;
  const std::int64_t stride = hx * d;
  for (std::int64_t t = 0; t < 4; ++t) {
    const Tensor one = Unwrap(ops::zeros(f.x.shape(), DataType::kFloat32));
    ASSERT_TRUE(ops::copy(one, f.x).ok());
    const std::int32_t p = pos[static_cast<std::size_t>(t)];
    RopeApplyF32(one.data_ptr<float>() + (t * stride), 1, hx, d, &p,
                 f.cos.data_ptr<float>(), f.sin.data_ptr<float>());
    // Row t of `one` must equal row t of `full`; other rows untouched (== x).
    const float* got = one.data_ptr<float>() + (t * stride);
    const float* want = full.data_ptr<float>() + (t * stride);
    for (std::int64_t i = 0; i < stride; ++i) {
      EXPECT_EQ(got[i], want[i]) << "t=" << t << " i=" << i;
    }
  }
}

TEST(RopeKernelTest, ThreadingIsBitExactVsSerialVariant) {
  std::vector<std::int32_t> pos(64);
  for (std::int64_t i = 0; i < 64; ++i) {
    pos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>((i * 7) % 500);
  }
  const Fixture f = MakeFixture(/*t=*/64, /*hx=*/8, /*d=*/64,
                                /*p_count=*/500, pos, 900);
  const Tensor threaded = Kernel(f);
  const Tensor serial = Unwrap(ops::zeros(f.x.shape(), DataType::kFloat32));
  ASSERT_TRUE(ops::copy(serial, f.x).ok());
  detail::RopeRowsVariant(SelectedIsa())(serial.data_ptr<float>(), 64, 8, 64,
                                         pos.data(), f.cos.data_ptr<float>(),
                                         f.sin.data_ptr<float>());
  EXPECT_TRUE(Unwrap(ops::allclose(threaded, serial, 0.0, 0.0)).allclose);
}

TEST(RopeKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::RopeRowsVariant(isa)),
            reinterpret_cast<void*>(&scalar::RopeRows));
}

// The HF-derived tiny-llama RoPE goldens replayed through the kernel (both Q
// and K, sparse and contiguous position sets).
TEST(RopeKernelTest, MatchesFixtureGoldens) {
  const auto file =
      engine::model::SafetensorsFile::Open(engine::testing::FixturesDir() /
                                           "models/tiny-llama/expected/"
                                           "rope.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  const Tensor cos = Unwrap(file->tensor("tiny_table.cos"));
  const Tensor sin = Unwrap(file->tensor("tiny_table.sin"));

  for (const char* group : {"tiny_sparse", "tiny_contig"}) {
    const std::string base(group);
    const Tensor positions = Unwrap(file->tensor(base + ".positions"));
    std::vector<std::int32_t> pos(
        positions.data_ptr<std::int32_t>(),
        positions.data_ptr<std::int32_t>() + positions.shape().dim(0));
    for (const char* which : {"q", "k"}) {
      const Tensor in = Unwrap(file->tensor(base + "." + which));
      const Tensor expected = Unwrap(file->tensor(base + "." + which + "_out"));
      const Tensor out = Unwrap(ops::zeros(in.shape(), DataType::kFloat32));
      ASSERT_TRUE(ops::copy(out, in).ok());
      RopeApplyF32(out.data_ptr<float>(), in.shape().dim(0), in.shape().dim(1),
                   in.shape().dim(2), pos.data(), cos.data_ptr<float>(),
                   sin.data_ptr<float>());
      const ops::AllCloseResult r =
          Unwrap(ops::allclose(out, expected, 1e-4, 1e-5));
      EXPECT_TRUE(r.allclose) << group << "." << which << ": " << r.Summary();
    }
  }
}

}  // namespace
}  // namespace engine::kernels
