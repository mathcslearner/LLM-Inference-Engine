#include "kernels/convert.h"

#include "tensor/half.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

// Conversion kernel tests (M3-T06; design: docs/design/cpu-backend.md §5,
// §6.3 Class E, §9). Tolerance statement: none — every comparison is bit
// exact against tensor/half.h, the single definition of conversion
// semantics, whose own correctness is pinned by the M1-T07 golden tests
// (half_test.cpp). Coverage is exhaustive where the domain allows it: all
// 2^16 fp16/bf16 patterns are widened, and all of them round-trip through
// the narrowing kernels (half.h documents half → fp32 → half as the
// identity for every pattern, NaNs included — so the round trip exercises
// narrowing on every normal, subnormal, inf, and NaN payload class,
// including the signaling payloads where hardware conversion units disagree
// with half.h and the kernels must blend in integer-rebuilt lanes).
//
// The public entries dispatch the host's best ISA and thread via
// parallel_for (the 2^16-element arrays span two chunks); the forced-scalar
// registration (unit-scalar/) re-runs everything through the scalar
// variants.

namespace engine::kernels {
namespace {

using tensor::bfloat16;
using tensor::float16;

constexpr std::size_t kPatternCount = 1U << 16U;

std::uint32_t Bits(float value) { return std::bit_cast<std::uint32_t>(value); }

// fp32 values that hit every narrowing path: exact values, RNE ties in the
// normal and subnormal ranges, overflow boundaries, ±0, ±inf, and NaNs
// (quiet, signaling-with-surviving-payload, signaling-with-truncated-
// payload, negative). Expectations are computed from half.h per element, so
// the list only needs to be *interesting*, not hand-labeled.
const std::vector<float>& NarrowingProbeValues() {
  static const std::vector<float> kValues = [] {
    static constexpr std::uint32_t kPatterns[] = {
        0x00000000U,  // +0
        0x80000000U,  // -0
        0x3F800000U,  // 1.0
        0xBF800000U,  // -1.0
        0x3F801000U,  // 1 + 2^-11 — fp16 RNE tie (rounds to even, 1.0)
        0x3F803000U,  // 1 + 3·2^-12 — fp16 RNE tie (rounds up)
        0x3F808000U,  // 1 + 2^-8 — bf16 RNE tie (rounds to even)
        0x3F818000U,  // 1 + 3·2^-8 — bf16 RNE tie (rounds up)
        0x477FE000U,  // 65504 — largest finite fp16
        0x477FEFFFU,  // just below the fp16 overflow threshold
        0x477FF000U,  // 65520 — smallest fp32 rounding to fp16 inf
        0x47800000U,  // 65536
        0x7F7FFFFFU,  // FLT_MAX (bf16: rounds to inf)
        0x38800000U,  // 2^-14 — smallest normal fp16
        0x387FC000U,  // just below 2^-14 (subnormal result, near-tie)
        0x38000000U,  // 2^-15 — subnormal fp16
        0x33800000U,  // 2^-24 — smallest subnormal fp16
        0x33C00000U,  // 1.5 · 2^-24 — subnormal RNE tie
        0x33000000U,  // 2^-25 — rounds to zero (tie to even)
        0x33000001U,  // just above 2^-25 — rounds to denorm_min
        0x32FFFFFFU,  // just below 2^-25 — rounds to zero
        0x00000001U,  // fp32 denorm_min
        0x807FFFFFU,  // largest negative fp32 subnormal
        0x00800000U,  // smallest normal fp32 (bf16 exact, fp16 → 0)
        0x7F800000U,  // +inf
        0xFF800000U,  // -inf
        0x7FC00000U,  // canonical quiet NaN
        0xFFC00000U,  // negative quiet NaN
        0x7FC01234U,  // quiet NaN with payload
        0x7F800001U,  // SNaN whose payload truncates to zero in fp16 & bf16
        0x7F802000U,  // SNaN surviving fp16 truncation (bit 13)
        0x7F810000U,  // SNaN surviving bf16 truncation (bit 16)
        0x7FBFFFFFU,  // SNaN with full payload
        0xFF923456U,  // negative SNaN with payload
    };
    std::vector<float> values;
    for (const std::uint32_t pattern : kPatterns) {
      values.push_back(std::bit_cast<float>(pattern));
    }
    // Deterministic bulk values spanning well past both narrow ranges.
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> mantissa(1.0F, 2.0F);
    std::uniform_int_distribution<int> exponent(-30, 20);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < 5000; ++i) {
      const float magnitude = std::ldexp(mantissa(rng), exponent(rng));
      values.push_back(coin(rng) == 0 ? magnitude : -magnitude);
    }
    return values;
  }();
  return kValues;
}

TEST(ConvertTest, Fp16WideningIsExhaustivelyBitExact) {
  std::vector<float16> in;
  in.reserve(kPatternCount);
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    in.push_back(float16::from_bits(static_cast<std::uint16_t>(p)));
  }
  std::vector<float> out(kPatternCount);
  Fp16ToFp32(in.data(), out.data(), static_cast<std::int64_t>(kPatternCount));
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    const auto want = static_cast<float>(in[p]);
    ASSERT_EQ(Bits(out[p]), Bits(want)) << "fp16 pattern 0x" << std::hex << p;
  }
}

TEST(ConvertTest, Bf16WideningIsExhaustivelyBitExact) {
  std::vector<bfloat16> in;
  in.reserve(kPatternCount);
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    in.push_back(bfloat16::from_bits(static_cast<std::uint16_t>(p)));
  }
  std::vector<float> out(kPatternCount);
  Bf16ToFp32(in.data(), out.data(), static_cast<std::int64_t>(kPatternCount));
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    const auto want = static_cast<float>(in[p]);
    ASSERT_EQ(Bits(out[p]), Bits(want)) << "bf16 pattern 0x" << std::hex << p;
  }
}

TEST(ConvertTest, Fp16NarrowingRoundTripsEveryPattern) {
  // half → fp32 is exact and fp32 → half restores every 16-bit pattern
  // (half.h contract), so narrowing must reproduce the identity on all
  // 2^16 inputs — normals, subnormals, ±inf, and every NaN payload.
  std::vector<float> widened(kPatternCount);
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    widened[p] =
        static_cast<float>(float16::from_bits(static_cast<std::uint16_t>(p)));
  }
  std::vector<float16> out(kPatternCount);
  Fp32ToFp16(widened.data(), out.data(),
             static_cast<std::int64_t>(kPatternCount));
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    ASSERT_EQ(out[p].to_bits(), static_cast<std::uint16_t>(p))
        << "fp16 pattern 0x" << std::hex << p;
  }
}

TEST(ConvertTest, Bf16NarrowingRoundTripsEveryPattern) {
  std::vector<float> widened(kPatternCount);
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    widened[p] =
        static_cast<float>(bfloat16::from_bits(static_cast<std::uint16_t>(p)));
  }
  std::vector<bfloat16> out(kPatternCount);
  Fp32ToBf16(widened.data(), out.data(),
             static_cast<std::int64_t>(kPatternCount));
  for (std::size_t p = 0; p < kPatternCount; ++p) {
    ASSERT_EQ(out[p].to_bits(), static_cast<std::uint16_t>(p))
        << "bf16 pattern 0x" << std::hex << p;
  }
}

TEST(ConvertTest, Fp16NarrowingMatchesHalfOnProbeValues) {
  const std::vector<float>& in = NarrowingProbeValues();
  const auto n = static_cast<std::int64_t>(in.size());
  std::vector<float16> out(in.size());
  Fp32ToFp16(in.data(), out.data(), n);
  for (std::size_t i = 0; i < in.size(); ++i) {
    ASSERT_EQ(out[i].to_bits(), float16(in[i]).to_bits())
        << "input 0x" << std::hex << Bits(in[i]);
  }
}

TEST(ConvertTest, Bf16NarrowingMatchesHalfOnProbeValues) {
  const std::vector<float>& in = NarrowingProbeValues();
  const auto n = static_cast<std::int64_t>(in.size());
  std::vector<bfloat16> out(in.size());
  Fp32ToBf16(in.data(), out.data(), n);
  for (std::size_t i = 0; i < in.size(); ++i) {
    ASSERT_EQ(out[i].to_bits(), bfloat16(in[i]).to_bits())
        << "input 0x" << std::hex << Bits(in[i]);
  }
}

TEST(ConvertTest, UnalignedBasesAndTailSizesAreBitExact) {
  // Sizes crossing the 4/8-lane widths with bases offset off any natural
  // alignment (design §7, §9). Content coverage lives in the exhaustive
  // tests above; this pins the body/tail split and unaligned loads.
  constexpr std::int64_t kSizes[] = {1, 15, 16, 17, 4096};
  constexpr std::size_t kOffsets[] = {0, 1, 2, 3};
  constexpr std::size_t kMaxOffset = 3;
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::uint32_t> pattern16(0, 0xFFFFU);
  for (const std::int64_t n : kSizes) {
    const auto count = static_cast<std::size_t>(n);
    std::vector<float16> h16(count + kMaxOffset);
    std::vector<bfloat16> b16(count + kMaxOffset);
    std::vector<float> f32(count + kMaxOffset);
    for (std::size_t i = 0; i < count + kMaxOffset; ++i) {
      h16[i] = float16::from_bits(static_cast<std::uint16_t>(pattern16(rng)));
      b16[i] = bfloat16::from_bits(static_cast<std::uint16_t>(pattern16(rng)));
      f32[i] = static_cast<float>(h16[i]) * 1.5F;
    }
    for (const std::size_t offset : kOffsets) {
      std::vector<float> wide(count);
      Fp16ToFp32(h16.data() + offset, wide.data(), n);
      for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(Bits(wide[i]), Bits(static_cast<float>(h16[offset + i])))
            << "fp16→fp32 n=" << n << " offset=" << offset << " i=" << i;
      }
      Bf16ToFp32(b16.data() + offset, wide.data(), n);
      for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(Bits(wide[i]), Bits(static_cast<float>(b16[offset + i])))
            << "bf16→fp32 n=" << n << " offset=" << offset << " i=" << i;
      }
      std::vector<float16> narrow16(count);
      Fp32ToFp16(f32.data() + offset, narrow16.data(), n);
      for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(narrow16[i].to_bits(), float16(f32[offset + i]).to_bits())
            << "fp32→fp16 n=" << n << " offset=" << offset << " i=" << i;
      }
      std::vector<bfloat16> narrowb(count);
      Fp32ToBf16(f32.data() + offset, narrowb.data(), n);
      for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(narrowb[i].to_bits(), bfloat16(f32[offset + i]).to_bits())
            << "fp32→bf16 n=" << n << " offset=" << offset << " i=" << i;
      }
    }
  }
}

TEST(ConvertTest, ZeroLengthIsANoOp) {
  float out_f32 = 42.0F;
  float16 out_f16 = float16::from_bits(0x1234);
  bfloat16 out_bf16 = bfloat16::from_bits(0x5678);
  Fp16ToFp32(nullptr, &out_f32, 0);
  Bf16ToFp32(nullptr, &out_f32, 0);
  Fp32ToFp16(nullptr, &out_f16, 0);
  Fp32ToBf16(nullptr, &out_bf16, 0);
  EXPECT_EQ(out_f32, 42.0F);
  EXPECT_EQ(out_f16.to_bits(), 0x1234);
  EXPECT_EQ(out_bf16.to_bits(), 0x5678);
}

}  // namespace
}  // namespace engine::kernels
