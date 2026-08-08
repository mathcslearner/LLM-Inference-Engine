#include "kernels/elementwise.h"

#include "kernels/internal/elementwise_impl.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

// Elementwise kernel tests (M3-T06; design: docs/design/cpu-backend.md §6.3
// Class E, §9). Tolerance statement: none — Class E results are compared as
// bit patterns. The public entry points dispatch the host's best ISA and
// thread via parallel_for; the reference is a direct single-threaded call to
// the scalar variant (the pre-M5 oracle, design §2.1), so one comparison
// simultaneously proves vector-vs-scalar bit-identity and that chunking
// cannot change results. Under the forced-scalar registration (unit-scalar/)
// the same comparisons re-run with the public entries pinned to scalar.
//
// The size sweep {0, 1, 15, 16, 17, 4096, 2^20 + 3} crosses every vector
// width (NEON 4, AVX2 8, the 16-element reduction step), the tail path, and
// the threading threshold (grain 32768: 4096 runs inline, 2^20 + 3 is
// multi-chunk); offsets {0, 1, 2, 3} deny the kernels any alignment
// assumption (design §7, §9).

namespace engine::kernels {
namespace {

constexpr std::int64_t kSizes[] = {0, 1, 15, 16, 17, 4096, (1 << 20) + 3};
constexpr std::size_t kOffsets[] = {0, 1, 2, 3};
constexpr std::size_t kMaxOffset = 3;

// Deterministic values spanning ~80 binades, with specials (±0, subnormals,
// ±inf, NaN) interleaved at a fixed cadence — elementwise ops must be
// bit-exact on every input class.
std::vector<float> TestValues(std::size_t n, std::uint32_t seed) {
  static constexpr float kSpecials[] = {
      0.0F,
      -0.0F,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(),
  };
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> mantissa(1.0F, 2.0F);
  std::uniform_int_distribution<int> exponent(-40, 40);
  std::uniform_int_distribution<int> coin(0, 1);
  std::vector<float> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (i % 17 == 16) {
      values[i] = kSpecials[(i / 17) % std::size(kSpecials)];
    } else {
      const float magnitude = std::ldexp(mantissa(rng), exponent(rng));
      values[i] = coin(rng) == 0 ? magnitude : -magnitude;
    }
  }
  return values;
}

void ExpectBitEqual(const float* actual, const float* expected,
                    std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    const auto a = std::bit_cast<std::uint32_t>(actual[i]);
    const auto e = std::bit_cast<std::uint32_t>(expected[i]);
    ASSERT_EQ(a, e) << "element " << i << ": got 0x" << std::hex << a
                    << ", want 0x" << e;
  }
}

TEST(ElementwiseTest, AddMatchesScalarBitExactAcrossSizesAndOffsets) {
  for (const std::int64_t n : kSizes) {
    const auto count = static_cast<std::size_t>(n);
    const std::vector<float> a = TestValues(count + kMaxOffset, 1);
    const std::vector<float> b = TestValues(count + kMaxOffset, 2);
    for (const std::size_t offset : kOffsets) {
      std::vector<float> out(count + kMaxOffset);
      std::vector<float> ref(count + kMaxOffset);
      AddF32(a.data() + offset, b.data() + offset, out.data() + offset, n);
      scalar::AddF32(a.data() + offset, b.data() + offset, ref.data() + offset,
                     n);
      ExpectBitEqual(out.data() + offset, ref.data() + offset, n);
    }
  }
}

TEST(ElementwiseTest, MulMatchesScalarBitExactAcrossSizesAndOffsets) {
  for (const std::int64_t n : kSizes) {
    const auto count = static_cast<std::size_t>(n);
    const std::vector<float> a = TestValues(count + kMaxOffset, 3);
    const std::vector<float> b = TestValues(count + kMaxOffset, 4);
    for (const std::size_t offset : kOffsets) {
      std::vector<float> out(count + kMaxOffset);
      std::vector<float> ref(count + kMaxOffset);
      MulF32(a.data() + offset, b.data() + offset, out.data() + offset, n);
      scalar::MulF32(a.data() + offset, b.data() + offset, ref.data() + offset,
                     n);
      ExpectBitEqual(out.data() + offset, ref.data() + offset, n);
    }
  }
}

TEST(ElementwiseTest, ScaleMatchesScalarBitExactAcrossSizesAndOffsets) {
  constexpr float kScales[] = {0.5F, -3.25F, 0.0F, -0.0F, 1.0F};
  for (const std::int64_t n : kSizes) {
    const auto count = static_cast<std::size_t>(n);
    const std::vector<float> x = TestValues(count + kMaxOffset, 5);
    for (const float s : kScales) {
      for (const std::size_t offset : kOffsets) {
        std::vector<float> out(count + kMaxOffset);
        std::vector<float> ref(count + kMaxOffset);
        ScaleF32(x.data() + offset, s, out.data() + offset, n);
        scalar::ScaleF32(x.data() + offset, s, ref.data() + offset, n);
        ExpectBitEqual(out.data() + offset, ref.data() + offset, n);
      }
    }
  }
}

TEST(ElementwiseTest, ExactAliasingIsSupported) {
  // out == a (and out == b for the symmetric ops) is documented as legal
  // (elementwise.h); a vector body that stored before loading would fail.
  constexpr std::int64_t kN = 4099;
  const std::vector<float> a = TestValues(kN, 6);
  const std::vector<float> b = TestValues(kN, 7);

  std::vector<float> expected(kN);
  scalar::AddF32(a.data(), b.data(), expected.data(), kN);
  std::vector<float> in_place = a;
  AddF32(in_place.data(), b.data(), in_place.data(), kN);
  ExpectBitEqual(in_place.data(), expected.data(), kN);

  scalar::MulF32(a.data(), b.data(), expected.data(), kN);
  in_place = b;
  MulF32(a.data(), in_place.data(), in_place.data(), kN);
  ExpectBitEqual(in_place.data(), expected.data(), kN);

  scalar::ScaleF32(a.data(), -1.5F, expected.data(), kN);
  in_place = a;
  ScaleF32(in_place.data(), -1.5F, in_place.data(), kN);
  ExpectBitEqual(in_place.data(), expected.data(), kN);
}

TEST(ElementwiseTest, ZeroLengthIsANoOp) {
  const float a = 1.0F;
  const float b = 2.0F;
  float out = 42.0F;
  AddF32(&a, &b, &out, 0);
  MulF32(&a, &b, &out, 0);
  ScaleF32(&a, 3.0F, &out, 0);
  EXPECT_EQ(out, 42.0F);
}

}  // namespace
}  // namespace engine::kernels
