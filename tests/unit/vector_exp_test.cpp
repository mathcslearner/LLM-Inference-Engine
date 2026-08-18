#include "kernels/dispatch.h"
#include "kernels/internal/exp_common.h"
#include "kernels/internal/exp_impl.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Dedicated sweep for the vector `expf` polynomial (M6-T03; design:
// docs/design/optimized-cpu-execution.md §10). This is the one new numerical
// algorithm M6 introduces, so it is bounded here **independently of the
// softmax/SiLU kernels that use it**, against a correctly-rounded reference
// (std::exp in double, narrowed to float — stricter than std::expf, whose own
// error is ~0.5–1 ulp).
//
// Contract asserted (exp_common.h):
//  - x ∈ [kExpLo, kExpHi]  → ≤ 2 ulp (the §10 target).
//  - x < kExpLo (incl −inf) → exactly +0.0 (softmax causal-mask contract).
//  - x > kExpHi            → finite, saturating (immaterial to callers).
//  - NaN → NaN.
//
// Registered SCALAR_PASS: the scalar variant runs the same polynomial, so the
// forced-scalar pass proves it on every host (its only coverage there, since
// softmax/SiLU-scalar embed it).
namespace engine::kernels {
namespace {

// ulp distance between two finite same-sign floats via their monotone bit
// patterns. Both operands here are non-negative (exp output), so a plain
// unsigned-bit subtraction is the ulp count.
[[nodiscard]] std::int64_t UlpDiff(float a, float b) {
  const auto ia = std::bit_cast<std::uint32_t>(a);
  const auto ib = std::bit_cast<std::uint32_t>(b);
  return std::abs(static_cast<std::int64_t>(ia) -
                  static_cast<std::int64_t>(ib));
}

// The array-form variant for `isa`, plus a name for diagnostics.
struct Variant {
  detail::ExpF32Fn fn;
  std::string name;
};

std::vector<Variant> VariantsUnderTest() {
  std::vector<Variant> v;
  v.push_back({&scalar::ExpF32, "scalar"});
  const Isa sel = SelectedIsa();
  if (sel != Isa::kScalar) {
    v.push_back({detail::ExpF32Variant(sel), std::string(IsaName(sel))});
  }
  return v;
}

TEST(VectorExpTest, VectorSlotPopulated) {
  const Isa sel = SelectedIsa();
  if (sel == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass: no vector variant expected";
  }
  // Otherwise the selected variant must differ from scalar, else the sweep
  // below would be scalar-vs-reference twice (vacuous for the vector path).
  EXPECT_NE(reinterpret_cast<void*>(detail::ExpF32Variant(sel)),
            reinterpret_cast<void*>(&scalar::ExpF32));
}

TEST(VectorExpTest, WithinTwoUlpOverGuaranteedDomain) {
  constexpr std::int64_t kMaxUlp = 2;  // the §10 target.
  // Dense sweep across [kExpLo, kExpHi]; the step is well below a float ulp of
  // the argument so every representable input class in the range is hit, plus
  // an explicit tail near 0 (softmax's stability regime, x ≤ 0).
  constexpr int kSteps = 2'000'000;
  const double lo = internal::kExpLo;
  const double hi = internal::kExpHi;
  const double step = (hi - lo) / kSteps;

  for (const Variant& var : VariantsUnderTest()) {
    std::int64_t max_ulp = 0;
    float worst_x = 0.0F;
    std::vector<float> in;
    std::vector<float> out;
    in.reserve(kSteps + 1);
    for (int i = 0; i <= kSteps; ++i) {
      in.push_back(static_cast<float>(lo + (step * i)));
    }
    out.resize(in.size());
    var.fn(in.data(), out.data(), static_cast<std::int64_t>(in.size()));
    for (std::size_t i = 0; i < in.size(); ++i) {
      const auto ref = static_cast<float>(std::exp(static_cast<double>(in[i])));
      const std::int64_t ulp = UlpDiff(out[i], ref);
      if (ulp > max_ulp) {
        max_ulp = ulp;
        worst_x = in[i];
      }
    }
    std::cerr << "[vector_exp] " << var.name << " max_ulp=" << max_ulp
              << " at x=" << worst_x << '\n';
    EXPECT_LE(max_ulp, kMaxUlp) << var.name << ": worst at x=" << worst_x;
  }
}

TEST(VectorExpTest, FlushesBelowLoToZero) {
  const float below[] = {internal::kExpLo - 0.01F, -90.0F, -1000.0F,
                         -std::numeric_limits<float>::infinity()};
  for (const Variant& var : VariantsUnderTest()) {
    std::vector<float> in(std::begin(below), std::end(below));
    std::vector<float> out(in.size());
    var.fn(in.data(), out.data(), static_cast<std::int64_t>(in.size()));
    for (std::size_t i = 0; i < in.size(); ++i) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(out[i]), 0U)
          << var.name << ": exp(" << in[i] << ") should flush to +0.0";
    }
  }
}

TEST(VectorExpTest, SaturatesFiniteAboveHi) {
  const float above[] = {internal::kExpHi + 0.01F, 100.0F, 1000.0F};
  for (const Variant& var : VariantsUnderTest()) {
    std::vector<float> in(std::begin(above), std::end(above));
    std::vector<float> out(in.size());
    var.fn(in.data(), out.data(), static_cast<std::int64_t>(in.size()));
    for (std::size_t i = 0; i < in.size(); ++i) {
      EXPECT_TRUE(std::isfinite(out[i]))
          << var.name << ": exp(" << in[i] << ") should saturate finite";
      EXPECT_GT(out[i], 0.0F) << var.name;
    }
  }
}

TEST(VectorExpTest, NanPropagates) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const Variant& var : VariantsUnderTest()) {
    std::vector<float> in{nan};
    std::vector<float> out(1);
    var.fn(in.data(), out.data(), 1);
    EXPECT_TRUE(std::isnan(out[0])) << var.name;
  }
}

// exp(0) == 1 exactly (a fixed point of the reduction: n = 0, r = 0).
TEST(VectorExpTest, ExpZeroIsOne) {
  for (const Variant& var : VariantsUnderTest()) {
    std::vector<float> in{0.0F};
    std::vector<float> out(1);
    var.fn(in.data(), out.data(), 1);
    EXPECT_EQ(out[0], 1.0F) << var.name;
  }
}

}  // namespace
}  // namespace engine::kernels
