#include "tensor/half.h"

#include "tensor/dtype.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <type_traits>
#include <vector>

namespace {

using engine::tensor::bfloat16;
using engine::tensor::DataType;
using engine::tensor::dtype_of;
using engine::tensor::float16;

// The storage contract Tensor relies on: 2-byte, trivially copyable,
// bit_cast-able value types.
static_assert(sizeof(float16) == 2 && alignof(float16) == 2);
static_assert(sizeof(bfloat16) == 2 && alignof(bfloat16) == 2);
static_assert(std::is_trivially_copyable_v<float16> &&
              std::is_standard_layout_v<float16>);
static_assert(std::is_trivially_copyable_v<bfloat16> &&
              std::is_standard_layout_v<bfloat16>);

// The DTypeTraits mapping added in M1-T07.
static_assert(dtype_of<float16> == DataType::kFloat16);
static_assert(dtype_of<bfloat16> == DataType::kBFloat16);

// Conversions are usable in constant expressions, end to end.
static_assert(float16(1.0F).to_bits() == 0x3C00);
static_assert(static_cast<float>(float16::from_bits(0x3C00)) == 1.0F);
static_assert(float16(0x1p-25F).to_bits() == 0x0000);  // rounds at compile time
static_assert(bfloat16(1.0F).to_bits() == 0x3F80);
static_assert(static_cast<float>(bfloat16::from_bits(0x3F80)) == 1.0F);

struct BitsValueGolden {
  std::uint16_t bits;
  float value;
};

// Exactly-representable pairs: conversion must be exact in both directions.
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr auto kFp16Exact = std::to_array<BitsValueGolden>({
    {.bits = 0x0000, .value = 0.0F},
    {.bits = 0x8000, .value = -0.0F},
    {.bits = 0x3C00, .value = 1.0F},
    {.bits = 0xBC00, .value = -1.0F},
    {.bits = 0x4000, .value = 2.0F},
    {.bits = 0xC000, .value = -2.0F},
    {.bits = 0x3800, .value = 0.5F},
    {.bits = 0x1400, .value = 0x1p-10F},         // epsilon
    {.bits = 0x3555, .value = 0x555p-12F},       // nearest fp16 to 1/3
    {.bits = 0x3C01, .value = 1.0F + 0x1p-10F},  // 1 + ulp
    {.bits = 0x7BFF, .value = 65504.0F},         // max finite
    {.bits = 0xFBFF, .value = -65504.0F},
    {.bits = 0x0400, .value = 0x1p-14F},    // min normal
    {.bits = 0x03FF, .value = 0x3FFp-24F},  // max subnormal
    {.bits = 0x0001, .value = 0x1p-24F},    // min subnormal
    {.bits = 0x8001, .value = -0x1p-24F},
    {.bits = 0x7C00, .value = kInf},
    {.bits = 0xFC00, .value = -kInf},
});

constexpr auto kBf16Exact = std::to_array<BitsValueGolden>({
    {.bits = 0x0000, .value = 0.0F},
    {.bits = 0x8000, .value = -0.0F},
    {.bits = 0x3F80, .value = 1.0F},
    {.bits = 0xBF80, .value = -1.0F},
    {.bits = 0xC000, .value = -2.0F},
    {.bits = 0x3F00, .value = 0.5F},
    {.bits = 0x3C00, .value = 0x1p-7F},  // epsilon
    // 3.140625: the bf16-*rounded* value of pi, deliberately not pi itself.
    {.bits = 0x4049, .value = 0x1.92p1F},  // NOLINT(modernize-use-std-numbers)
    {.bits = 0x7F7F, .value = 0x1.FEp127F},  // max finite
    {.bits = 0xFF7F, .value = -0x1.FEp127F},
    {.bits = 0x0080, .value = 0x1p-126F},   // min normal
    {.bits = 0x007F, .value = 0x7Fp-133F},  // max subnormal
    {.bits = 0x0001, .value = 0x1p-133F},   // min subnormal
    {.bits = 0x7F80, .value = kInf},
    {.bits = 0xFF80, .value = -kInf},
});

template <typename Half, std::size_t N>
void ExpectExactGoldens(const std::array<BitsValueGolden, N>& goldens) {
  for (const BitsValueGolden& g : goldens) {
    const float widened = Half::from_bits(g.bits);
    EXPECT_EQ(widened, g.value) << fmt::format("bits = {:#06x}", g.bits);
    EXPECT_EQ(std::signbit(widened), std::signbit(g.value))
        << fmt::format("bits = {:#06x}", g.bits);
    EXPECT_EQ(Half(g.value).to_bits(), g.bits)
        << fmt::format("value = {:a}", g.value);
  }
}

TEST(Float16Test, ExactGoldenPairsConvertBothWays) {
  ExpectExactGoldens<float16>(kFp16Exact);
}

TEST(BFloat16Test, ExactGoldenPairsConvertBothWays) {
  ExpectExactGoldens<bfloat16>(kBf16Exact);
}

// --- fp32 → half rounding goldens ---

TEST(Float16Test, RoundsTiesToEven) {
  // fp16 ulp at 1.0 is 2^-10; 1 + 2^-11 is the exact midpoint.
  EXPECT_EQ(float16(1.0F + 0x1p-11F).to_bits(), 0x3C00);  // tie → even, down
  EXPECT_EQ(float16(1.0F + 0x3p-11F).to_bits(), 0x3C02);  // tie → even, up
  EXPECT_EQ(float16(std::nextafter(1.0F + 0x1p-11F, 2.0F)).to_bits(), 0x3C01);
  EXPECT_EQ(float16(std::nextafter(1.0F + 0x1p-11F, 0.0F)).to_bits(), 0x3C00);
  // Mantissa rounding carries into the exponent: just below 2.0 → 2.0.
  EXPECT_EQ(float16(std::nextafter(2.0F, 0.0F)).to_bits(), 0x4000);
}

TEST(Float16Test, OverflowRoundsToInfinity) {
  // 65520 is the midpoint between 65504 (max finite) and 2^16; ties-to-even
  // sends it past the largest finite value, i.e. to infinity.
  EXPECT_EQ(float16(65520.0F).to_bits(), 0x7C00);
  EXPECT_EQ(float16(-65520.0F).to_bits(), 0xFC00);
  EXPECT_EQ(float16(std::nextafter(65520.0F, 0.0F)).to_bits(), 0x7BFF);
  EXPECT_EQ(float16(65536.0F).to_bits(), 0x7C00);
  EXPECT_EQ(float16(std::numeric_limits<float>::max()).to_bits(), 0x7C00);
  EXPECT_EQ(float16(kInf).to_bits(), 0x7C00);
  EXPECT_EQ(float16(-kInf).to_bits(), 0xFC00);
}

TEST(Float16Test, SubnormalRounding) {
  // fp16 subnormal unit is 2^-24; 2^-25 is the midpoint between 0 and it.
  EXPECT_EQ(float16(0x1p-25F).to_bits(), 0x0000);   // tie with zero → even
  EXPECT_EQ(float16(-0x1p-25F).to_bits(), 0x8000);  // sign survives underflow
  EXPECT_EQ(float16(std::nextafter(0x1p-25F, 1.0F)).to_bits(), 0x0001);
  EXPECT_EQ(float16(0x3p-26F).to_bits(), 0x0001);  // 0.75 units, rounds up
  EXPECT_EQ(float16(0x3p-25F).to_bits(), 0x0002);  // 1.5 units, tie → even
  EXPECT_EQ(float16(0x1p-26F).to_bits(), 0x0000);
  EXPECT_EQ(float16(1e-30F).to_bits(), 0x0000);  // fp32-tiny → signed zero
  EXPECT_EQ(float16(-1e-30F).to_bits(), 0x8000);
  // Rounding can carry the max subnormal up into the minimum normal.
  EXPECT_EQ(float16(std::nextafter(0x1p-14F, 0.0F)).to_bits(), 0x0400);
  // Midpoint between max subnormal and min normal: even → min normal.
  EXPECT_EQ(float16(0x7FFp-25F).to_bits(), 0x0400);
}

TEST(Float16Test, NanConversions) {
  EXPECT_EQ(float16(std::bit_cast<float>(0x7FC00000U)).to_bits(), 0x7E00);
  EXPECT_EQ(float16(std::bit_cast<float>(0xFFC00000U)).to_bits(), 0xFE00);
  // Payload entirely in the truncated low 13 bits: the quiet bit is forced
  // so the result never reads as infinity.
  EXPECT_EQ(float16(std::bit_cast<float>(0x7F800001U)).to_bits(), 0x7E00);
  // Payload bits that fit are preserved.
  EXPECT_EQ(float16(std::bit_cast<float>(0x7FAAA000U)).to_bits(), 0x7D55);
  // Widening shifts the payload up and keeps NaN-ness.
  const float widened = float16::from_bits(0x7E01);
  EXPECT_TRUE(std::isnan(widened));
  EXPECT_EQ(std::bit_cast<std::uint32_t>(widened), 0x7FC02000U);
}

TEST(BFloat16Test, RoundsTiesToEven) {
  // bf16 ulp at 1.0 is 2^-7; 1 + 2^-8 is the exact midpoint.
  EXPECT_EQ(bfloat16(1.0F + 0x1p-8F).to_bits(), 0x3F80);  // tie → even, down
  EXPECT_EQ(bfloat16(1.0F + 0x3p-8F).to_bits(), 0x3F82);  // tie → even, up
  EXPECT_EQ(bfloat16(std::nextafter(1.0F + 0x1p-8F, 2.0F)).to_bits(), 0x3F81);
  EXPECT_EQ(bfloat16(std::numbers::pi_v<float>).to_bits(), 0x4049);
  EXPECT_EQ(bfloat16(std::nextafter(2.0F, 0.0F)).to_bits(), 0x4000);
}

TEST(BFloat16Test, OverflowRoundsToInfinity) {
  // 0x1.FFp127 is the midpoint between the max finite bf16 and 2^128.
  EXPECT_EQ(bfloat16(0x1.FFp127F).to_bits(), 0x7F80);
  EXPECT_EQ(bfloat16(std::nextafter(0x1.FFp127F, 0.0F)).to_bits(), 0x7F7F);
  EXPECT_EQ(bfloat16(std::numeric_limits<float>::max()).to_bits(), 0x7F80);
  EXPECT_EQ(bfloat16(kInf).to_bits(), 0x7F80);
  EXPECT_EQ(bfloat16(-kInf).to_bits(), 0xFF80);
}

TEST(BFloat16Test, SubnormalRounding) {
  // bf16 subnormal unit is 2^-133 (well inside fp32's subnormal range).
  EXPECT_EQ(bfloat16(0x1p-133F).to_bits(), 0x0001);
  EXPECT_EQ(bfloat16(0x1p-134F).to_bits(), 0x0000);  // tie with zero → even
  EXPECT_EQ(bfloat16(0x3p-134F).to_bits(), 0x0002);  // 1.5 units, tie → even
  EXPECT_EQ(bfloat16(-0x1p-140F).to_bits(), 0x8000);
  // Rounding carries the max fp32 value below 2^-126 into bf16's min normal.
  EXPECT_EQ(bfloat16(std::nextafter(0x1p-126F, 0.0F)).to_bits(), 0x0080);
}

TEST(BFloat16Test, NanConversions) {
  EXPECT_EQ(bfloat16(std::bit_cast<float>(0x7FC00000U)).to_bits(), 0x7FC0);
  EXPECT_EQ(bfloat16(std::bit_cast<float>(0xFFC00000U)).to_bits(), 0xFFC0);
  // Payload entirely in the truncated low 16 bits: quiet bit forced.
  EXPECT_EQ(bfloat16(std::bit_cast<float>(0x7F800001U)).to_bits(), 0x7FC0);
  // Payload bits that fit are preserved.
  EXPECT_EQ(bfloat16(std::bit_cast<float>(0x7FAA0000U)).to_bits(), 0x7FAA);
  const float widened = bfloat16::from_bits(0x7F81);
  EXPECT_TRUE(std::isnan(widened));
  EXPECT_EQ(std::bit_cast<std::uint32_t>(widened), 0x7F810000U);
}

// --- round-trip: exhaustive over every 16-bit pattern ---

// half → fp32 → half is bit-identical for all 65536 patterns, NaNs (quiet
// and signaling, any payload) and ±0 included — the header's NaN policy is
// chosen to make this hold.
template <typename Half>
void ExpectAllBitPatternsRoundTrip() {
  for (std::uint32_t b = 0; b <= 0xFFFFU; ++b) {
    const auto bits = static_cast<std::uint16_t>(b);
    const Half h = Half::from_bits(bits);
    const Half back{static_cast<float>(h)};
    ASSERT_EQ(back.to_bits(), bits) << fmt::format("bits = {:#06x}", b);
  }
}

TEST(Float16Test, EveryBitPatternRoundTripsThroughFloat) {
  ExpectAllBitPatternsRoundTrip<float16>();
}

TEST(BFloat16Test, EveryBitPatternRoundTripsThroughFloat) {
  ExpectAllBitPatternsRoundTrip<bfloat16>();
}

// --- property: narrowing picks the nearest value, ties to even ---

// Every finite value of Half, as fp32, sorted — an independent reference to
// judge rounding against (the widening direction it relies on is pinned by
// the golden and exhaustive tests above).
template <typename Half>
std::vector<float> AllFiniteValues() {
  std::vector<float> values;
  values.reserve(1U << 16U);
  for (std::uint32_t b = 0; b <= 0xFFFFU; ++b) {
    const float f = Half::from_bits(static_cast<std::uint16_t>(b));
    if (std::isfinite(f)) {
      values.push_back(f);
    }
  }
  std::ranges::sort(values);
  return values;
}

template <typename Half>
void ExpectRoundsToNearestEven(const std::vector<float>& values, float f) {
  const Half h(f);
  const auto fd = static_cast<double>(f);
  const double err = std::abs(static_cast<double>(static_cast<float>(h)) - fd);
  // The distance to the nearest representable value (one of the two
  // bracketing neighbors), computed exactly in double.
  const auto it = std::ranges::lower_bound(values, f);
  double best = std::numeric_limits<double>::infinity();
  if (it != values.end()) {
    best = std::min(best, std::abs(static_cast<double>(*it) - fd));
  }
  if (it != values.begin()) {
    best = std::min(best, std::abs(fd - static_cast<double>(*std::prev(it))));
  }
  ASSERT_EQ(err, best) << fmt::format("f = {:a}", f);
  if (it != values.end() && it != values.begin()) {
    const auto lo = static_cast<double>(*std::prev(it));
    const auto hi = static_cast<double>(*it);
    if (lo != hi && fd - lo == hi - fd) {
      // Exact tie: the chosen pattern has an even significand lsb (which is
      // the bit-pattern lsb, including across exponent boundaries).
      ASSERT_EQ(h.to_bits() & 1U, 0U) << fmt::format("f = {:a}", f);
    }
  }
}

TEST(Float16Test, RandomFloatsRoundToNearestEven) {
  const std::vector<float> values = AllFiniteValues<float16>();
  std::mt19937 rng(20260803U);
  std::uniform_real_distribution<float> normal_range(-65504.0F, 65504.0F);
  std::uniform_real_distribution<float> subnormal_range(-0x1p-14F, 0x1p-14F);
  for (int i = 0; i < 50000; ++i) {
    ExpectRoundsToNearestEven<float16>(values, normal_range(rng));
    ExpectRoundsToNearestEven<float16>(values, subnormal_range(rng));
  }
  // Random bit patterns hit every exponent; uniform sampling does not.
  std::uniform_int_distribution<std::uint32_t> any_bits;
  int tested = 0;
  while (tested < 50000) {
    const auto f = std::bit_cast<float>(any_bits(rng));
    if (!std::isfinite(f) || std::abs(f) > 65504.0F) {
      continue;
    }
    ExpectRoundsToNearestEven<float16>(values, f);
    ++tested;
  }
}

TEST(BFloat16Test, RandomFloatsRoundToNearestEven) {
  const std::vector<float> values = AllFiniteValues<bfloat16>();
  std::mt19937 rng(20260803U);
  // ±1.5e38 keeps the distribution's range representable in fp32; random
  // bit patterns below cover the tail up to the bf16 max.
  std::uniform_real_distribution<float> normal_range(-1.5e38F, 1.5e38F);
  std::uniform_real_distribution<float> subnormal_range(-0x1p-126F, 0x1p-126F);
  for (int i = 0; i < 50000; ++i) {
    ExpectRoundsToNearestEven<bfloat16>(values, normal_range(rng));
    ExpectRoundsToNearestEven<bfloat16>(values, subnormal_range(rng));
  }
  std::uniform_int_distribution<std::uint32_t> any_bits;
  int tested = 0;
  while (tested < 50000) {
    const auto f = std::bit_cast<float>(any_bits(rng));
    if (!std::isfinite(f) || std::abs(f) > 0x1.FEp127F) {
      continue;
    }
    ExpectRoundsToNearestEven<bfloat16>(values, f);
    ++tested;
  }
}

// --- float interop & numeric_limits ---

TEST(HalfTest, ImplicitWideningComposesWithFloatExpressions) {
  const float16 h(1.5F);
  EXPECT_EQ(h * 2.0F, 3.0F);
  EXPECT_EQ(h + bfloat16(0.5F), 2.0F);
  EXPECT_TRUE(float16(1.0F) == 1.0F);
  EXPECT_TRUE(float16::from_bits(0x0000) == float16::from_bits(0x8000));
  // IEEE comparison semantics come with the conversion: NaN != NaN.
  const float16 nan = std::numeric_limits<float16>::quiet_NaN();
  EXPECT_FALSE(nan == nan);
  EXPECT_TRUE(nan != nan);
}

TEST(HalfTest, FormatsAsFloatValue) {
  EXPECT_EQ(fmt::format("{}", float16(1.5F)), "1.5");
  EXPECT_EQ(fmt::format("{:.2f}", float16(1.5F)), "1.50");
  EXPECT_EQ(fmt::format("{}", bfloat16(-2.0F)), "-2");
}

TEST(Float16Test, NumericLimits) {
  using Limits = std::numeric_limits<float16>;
  static_assert(Limits::is_specialized && Limits::is_iec559);
  static_assert(Limits::digits == 11 && Limits::max_exponent == 16);
  EXPECT_EQ(static_cast<float>(Limits::max()), 65504.0F);
  EXPECT_EQ(static_cast<float>(Limits::lowest()), -65504.0F);
  EXPECT_EQ(static_cast<float>(Limits::min()), 0x1p-14F);
  EXPECT_EQ(static_cast<float>(Limits::denorm_min()), 0x1p-24F);
  EXPECT_EQ(static_cast<float>(Limits::epsilon()), 0x1p-10F);
  EXPECT_EQ(static_cast<float>(Limits::round_error()), 0.5F);
  EXPECT_TRUE(std::isinf(static_cast<float>(Limits::infinity())));
  EXPECT_TRUE(std::isnan(static_cast<float>(Limits::quiet_NaN())));
  EXPECT_TRUE(std::isnan(static_cast<float>(Limits::signaling_NaN())));
}

TEST(BFloat16Test, NumericLimits) {
  using Limits = std::numeric_limits<bfloat16>;
  static_assert(Limits::is_specialized && !Limits::is_iec559);
  static_assert(Limits::digits == 8 && Limits::max_exponent == 128);
  EXPECT_EQ(static_cast<float>(Limits::max()), 0x1.FEp127F);
  EXPECT_EQ(static_cast<float>(Limits::lowest()), -0x1.FEp127F);
  EXPECT_EQ(static_cast<float>(Limits::min()), 0x1p-126F);
  EXPECT_EQ(static_cast<float>(Limits::denorm_min()), 0x1p-133F);
  EXPECT_EQ(static_cast<float>(Limits::epsilon()), 0x1p-7F);
  EXPECT_EQ(static_cast<float>(Limits::round_error()), 0.5F);
  EXPECT_TRUE(std::isinf(static_cast<float>(Limits::infinity())));
  EXPECT_TRUE(std::isnan(static_cast<float>(Limits::quiet_NaN())));
  EXPECT_TRUE(std::isnan(static_cast<float>(Limits::signaling_NaN())));
}

}  // namespace
