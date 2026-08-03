#pragma once

#include "tensor/dtype.h"

#include <fmt/format.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

// Half-precision host value types (M1-T07; design: docs/design/tensor.md
// §3.1): `float16` (IEEE 754 binary16: 1 sign / 5 exponent, bias 15 / 10
// mantissa bits) and `bfloat16` (1 / 8, bias 127 / 7 — fp32 with the low 16
// mantissa bits dropped).
//
// This is a tensor_base header (ADR-002 Amendment 1): header-only, includes
// nothing from memory or the rest of tensor (dtype.h is a sibling leaf), may
// include core. No hardware float16 intrinsics or compiler extensions —
// every conversion is explicit integer bit manipulation over std::bit_cast,
// so results are bit-identical across compilers, architectures, and
// constant/runtime evaluation.
//
// Conversion policy (M1-T09's `cast` must match these functions exactly):
//  - Narrowing (fp32 → half) is an *explicit* constructor and rounds to
//    nearest, ties to even. Overflow → ±inf; underflow → ±0 (sign kept).
//  - Widening (half → fp32) is an *implicit* `operator float()` — it is
//    exact, so the types drop into float expressions the way the CPU
//    reference paths (M1-T08/09) use them.
//  - NaN: NaNs stay NaNs. The payload is truncated to the bits that fit;
//    if the truncated payload would be zero (which would read as inf), the
//    quiet bit is forced instead. Consequently half → fp32 → half is
//    bit-identical for every 16-bit pattern, NaNs (quiet or signaling)
//    included — tests rely on this.
//
// The types deliberately have no arithmetic or comparison operators of
// their own: operands implicitly widen to float, so `h * 2.0f`, `a < b`,
// and `h == 1.0f` all compute in fp32 with IEEE semantics (NaN != NaN,
// -0 == +0). Accumulate in float, narrow once at the end.

namespace engine::tensor {

namespace detail {

// fp32 → fp16 bit pattern, round-to-nearest-even.
[[nodiscard]] constexpr std::uint16_t Fp32ToFp16Bits(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
  const std::uint32_t abs = bits & 0x7FFFFFFFU;

  if (abs > 0x7F800000U) {  // NaN: truncate payload, never collapse to inf.
    auto payload = static_cast<std::uint16_t>((abs >> 13) & 0x03FFU);
    if (payload == 0U) {
      payload = 0x0200U;  // quiet bit
    }
    return static_cast<std::uint16_t>(sign | 0x7C00U | payload);
  }
  if (abs >= 0x477FF000U) {  // |value| >= 65520 rounds past 65504 → ±inf.
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (abs >= 0x38800000U) {  // Normal fp16 range: [2^-14, 65520).
    // Round the 13 dropped mantissa bits (add 0xFFF plus the kept lsb for
    // ties-to-even); the carry propagates into the exponent field, which is
    // then rebiased 127 → 15.
    const std::uint32_t rounded = abs + 0x00000FFFU + ((abs >> 13) & 1U);
    return static_cast<std::uint16_t>(sign | ((rounded >> 13) - (112U << 10)));
  }
  // Subnormal fp16 range (|value| < 2^-14), incl. fp32 subnormals and ±0.
  const std::uint32_t exponent = abs >> 23;
  if (exponent < 102U) {  // |value| < 2^-25: below half of 2^-24 → ±0.
    return sign;
  }
  // Result = round(mantissa.with_implicit_bit * 2^(exponent - 126)) in units
  // of 2^-24. A rounding carry out of the 10 mantissa bits lands in the
  // exponent field as the minimum normal — exactly right.
  const std::uint32_t mantissa = (abs & 0x007FFFFFU) | 0x00800000U;
  const std::uint32_t shift = 126U - exponent;  // 14..24
  const std::uint32_t rounded =
      mantissa + ((1U << (shift - 1U)) - 1U) + ((mantissa >> shift) & 1U);
  return static_cast<std::uint16_t>(sign | (rounded >> shift));
}

// fp16 bit pattern → fp32. Exact for every pattern (binary16 ⊂ binary32).
[[nodiscard]] constexpr float Fp16BitsToFp32(std::uint16_t half_bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half_bits & 0x8000U)
                             << 16;
  const std::uint32_t exponent = (half_bits >> 10) & 0x1FU;
  const std::uint32_t mantissa = half_bits & 0x03FFU;

  if (exponent == 0x1FU) {  // inf / NaN: payload shifts up, NaN-ness kept.
    return std::bit_cast<float>(sign | 0x7F800000U | (mantissa << 13));
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return std::bit_cast<float>(sign);  // ±0
    }
    // Subnormal: mantissa * 2^-24. Normalize: with the leading set bit at
    // position n, the value is 1.xxx * 2^(n - 24) → biased fp32 exponent
    // 127 + n - 24.
    const int n = 31 - std::countl_zero(mantissa);  // 0..9
    const std::uint32_t exp32 = 103U + static_cast<std::uint32_t>(n);
    const std::uint32_t man32 =
        (mantissa << (23U - static_cast<std::uint32_t>(n))) & 0x007FFFFFU;
    return std::bit_cast<float>(sign | (exp32 << 23) | man32);
  }
  // Normal: rebias 15 → 127.
  return std::bit_cast<float>(sign | ((exponent + 112U) << 23) |
                              (mantissa << 13));
}

// fp32 → bf16 bit pattern, round-to-nearest-even.
[[nodiscard]] constexpr std::uint16_t Fp32ToBf16Bits(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {  // NaN: truncate, keep NaN-ness.
    auto result = static_cast<std::uint16_t>(bits >> 16);
    if ((result & 0x007FU) == 0U) {
      result |= 0x0040U;  // quiet bit
    }
    return result;
  }
  // Round the 16 dropped bits; overflow to ±inf falls out of the carry.
  const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(rounded >> 16);
}

// bf16 bit pattern → fp32. Exact: reattach the 16 dropped mantissa bits.
[[nodiscard]] constexpr float Bf16BitsToFp32(std::uint16_t bf16_bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bf16_bits) << 16);
}

}  // namespace detail

// IEEE 754 binary16. Trivially copyable 2-byte value type, safe to place
// directly in Tensor storage and memcpy/bit_cast around. The default
// constructor leaves the value uninitialized, like a built-in float.
class float16 {
 public:
  float16() = default;

  // Narrowing, round-to-nearest-even (policy at the top of this header).
  explicit constexpr float16(float value)
      : bits_(detail::Fp32ToFp16Bits(value)) {}

  // Reinterpret a raw bit pattern (golden tests, fixtures, casts).
  [[nodiscard]] static constexpr float16 from_bits(std::uint16_t bits) {
    float16 result;
    result.bits_ = bits;
    return result;
  }
  [[nodiscard]] constexpr std::uint16_t to_bits() const { return bits_; }

  // Implicit: widening to fp32 is exact (see header comment).
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator float() const { return detail::Fp16BitsToFp32(bits_); }

 private:
  std::uint16_t bits_;
};

// bfloat16 (brain float): fp32's sign and exponent with the mantissa cut to
// 7 bits. Same value-type contract as float16.
class bfloat16 {
 public:
  bfloat16() = default;

  // Narrowing, round-to-nearest-even (policy at the top of this header).
  explicit constexpr bfloat16(float value)
      : bits_(detail::Fp32ToBf16Bits(value)) {}

  // Reinterpret a raw bit pattern (golden tests, fixtures, casts).
  [[nodiscard]] static constexpr bfloat16 from_bits(std::uint16_t bits) {
    bfloat16 result;
    result.bits_ = bits;
    return result;
  }
  [[nodiscard]] constexpr std::uint16_t to_bits() const { return bits_; }

  // Implicit: widening to fp32 is exact (see header comment).
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator float() const { return detail::Bf16BitsToFp32(bits_); }

 private:
  std::uint16_t bits_;
};

static_assert(sizeof(float16) == 2 && alignof(float16) == 2);
static_assert(sizeof(bfloat16) == 2 && alignof(bfloat16) == 2);
static_assert(std::is_trivially_copyable_v<float16> &&
              std::is_trivially_copyable_v<bfloat16>);

// dtype.h's compile-time mapping (specialized here because dtype.h is also a
// leaf header and cannot include this one).
template <>
struct DTypeTraits<float16> {
  static constexpr DataType value = DataType::kFloat16;
};
template <>
struct DTypeTraits<bfloat16> {
  static constexpr DataType value = DataType::kBFloat16;
};

}  // namespace engine::tensor

// numeric_limits, so generic code (M1-T08 fills/tolerances, property tests)
// can query representable ranges the standard way.
template <>
class std::numeric_limits<engine::tensor::float16> {
  using float16 = engine::tensor::float16;

 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr bool is_iec559 = true;  // binary16 is an IEEE 754 format.
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;
  static constexpr std::float_round_style round_style = std::round_to_nearest;

  static constexpr float16 min() { return float16::from_bits(0x0400); }
  static constexpr float16 lowest() { return float16::from_bits(0xFBFF); }
  static constexpr float16 max() { return float16::from_bits(0x7BFF); }
  static constexpr float16 epsilon() { return float16::from_bits(0x1400); }
  static constexpr float16 round_error() { return float16::from_bits(0x3800); }
  static constexpr float16 infinity() { return float16::from_bits(0x7C00); }
  static constexpr float16 quiet_NaN() { return float16::from_bits(0x7E00); }
  static constexpr float16 signaling_NaN() {
    return float16::from_bits(0x7D00);
  }
  static constexpr float16 denorm_min() { return float16::from_bits(0x0001); }
};

template <>
class std::numeric_limits<engine::tensor::bfloat16> {
  using bfloat16 = engine::tensor::bfloat16;

 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr bool is_iec559 = false;  // Not an IEEE interchange format.
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;
  static constexpr std::float_round_style round_style = std::round_to_nearest;

  static constexpr bfloat16 min() { return bfloat16::from_bits(0x0080); }
  static constexpr bfloat16 lowest() { return bfloat16::from_bits(0xFF7F); }
  static constexpr bfloat16 max() { return bfloat16::from_bits(0x7F7F); }
  static constexpr bfloat16 epsilon() { return bfloat16::from_bits(0x3C00); }
  static constexpr bfloat16 round_error() {
    return bfloat16::from_bits(0x3F00);
  }
  static constexpr bfloat16 infinity() { return bfloat16::from_bits(0x7F80); }
  static constexpr bfloat16 quiet_NaN() { return bfloat16::from_bits(0x7FC0); }
  static constexpr bfloat16 signaling_NaN() {
    return bfloat16::from_bits(0x7FA0);
  }
  static constexpr bfloat16 denorm_min() { return bfloat16::from_bits(0x0001); }
};

// Formats as the exact fp32 value, accepting float format specs:
// fmt::format("{:.3f}", float16(1.5f)) == "1.500".
template <>
struct fmt::formatter<engine::tensor::float16> : fmt::formatter<float> {
  template <typename FormatContext>
  auto format(engine::tensor::float16 value, FormatContext& ctx) const {
    return fmt::formatter<float>::format(static_cast<float>(value), ctx);
  }
};

template <>
struct fmt::formatter<engine::tensor::bfloat16> : fmt::formatter<float> {
  template <typename FormatContext>
  auto format(engine::tensor::bfloat16 value, FormatContext& ctx) const {
    return fmt::formatter<float>::format(static_cast<float>(value), ctx);
  }
};
