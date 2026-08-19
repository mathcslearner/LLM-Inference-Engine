#include "sampling/philox.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>

// M7-T02 (design: docs/design/model-execution.md §15.3): the Philox4x32-10
// counter-based RNG the stochastic sampler draws from. Portable C++ — no ISA
// concern, no SCALAR_PASS. Covers: the standard Random123 known-answer vectors
// (also asserted at compile time), counter/key sensitivity, and the
// `[0, 1)`-double helper's range and reproducibility.

namespace {

using engine::sampling::Philox4x32_10;
using engine::sampling::Philox4x32Block;
using engine::sampling::PhiloxUniformDouble;

// The canonical Random123 kat_vectors entries for philox4x32 with 10 rounds:
// {key, counter} -> expected block. These pin the block function bit-for-bit.
constexpr std::array<std::uint32_t, 2> kZeroKey = {0, 0};
constexpr Philox4x32Block kZeroCtr = {0, 0, 0, 0};
constexpr Philox4x32Block kZeroExpected = {0x6627e8d5, 0xe169c58d, 0xbc57ac4c,
                                           0x9b00dbd8};

constexpr std::array<std::uint32_t, 2> kOnesKey = {0xffffffff, 0xffffffff};
constexpr Philox4x32Block kOnesCtr = {0xffffffff, 0xffffffff, 0xffffffff,
                                      0xffffffff};
constexpr Philox4x32Block kOnesExpected = {0x408f276d, 0x41c83b0e, 0xa20bc7c6,
                                           0x6d5451fd};

constexpr std::array<std::uint32_t, 2> kPiKey = {0xa4093822, 0x299f31d0};
constexpr Philox4x32Block kPiCtr = {0x243f6a88, 0x85a308d3, 0x13198a2e,
                                    0x03707344};
constexpr Philox4x32Block kPiExpected = {0xd16cfe09, 0x94fdcceb, 0x5001e420,
                                         0x24126ea1};

// The block function is `constexpr`, so the known-answer vectors are also a
// compile-time check.
static_assert(Philox4x32_10(kZeroKey, kZeroCtr) == kZeroExpected);
static_assert(Philox4x32_10(kOnesKey, kOnesCtr) == kOnesExpected);
static_assert(Philox4x32_10(kPiKey, kPiCtr) == kPiExpected);

TEST(PhiloxTest, KnownAnswerVectors) {
  EXPECT_EQ(Philox4x32_10(kZeroKey, kZeroCtr), kZeroExpected);
  EXPECT_EQ(Philox4x32_10(kOnesKey, kOnesCtr), kOnesExpected);
  EXPECT_EQ(Philox4x32_10(kPiKey, kPiCtr), kPiExpected);
}

TEST(PhiloxTest, IncrementingCounterChangesOutput) {
  const Philox4x32Block a = Philox4x32_10(kZeroKey, {0, 0, 0, 0});
  const Philox4x32Block b = Philox4x32_10(kZeroKey, {1, 0, 0, 0});
  EXPECT_NE(a, b);
}

TEST(PhiloxTest, DifferentKeysDiffer) {
  const Philox4x32Block a = Philox4x32_10({1, 0}, kZeroCtr);
  const Philox4x32Block b = Philox4x32_10({2, 0}, kZeroCtr);
  EXPECT_NE(a, b);
}

// ----------------------------------------------------- uniform double --

TEST(PhiloxTest, UniformIsInUnitInterval) {
  for (std::uint64_t step = 0; step < 4096; ++step) {
    const double u = PhiloxUniformDouble(/*seed=*/12345, step, /*draw=*/0);
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
}

TEST(PhiloxTest, UniformIsReproducible) {
  const double a = PhiloxUniformDouble(/*seed=*/777, /*step=*/42, /*draw=*/0);
  const double b = PhiloxUniformDouble(/*seed=*/777, /*step=*/42, /*draw=*/0);
  EXPECT_EQ(a, b);
}

TEST(PhiloxTest, UniformVariesWithSeedStepAndDraw) {
  const double base = PhiloxUniformDouble(/*seed=*/1, /*step=*/0, /*draw=*/0);
  EXPECT_NE(base, PhiloxUniformDouble(/*seed=*/2, /*step=*/0, /*draw=*/0));
  EXPECT_NE(base, PhiloxUniformDouble(/*seed=*/1, /*step=*/1, /*draw=*/0));
  EXPECT_NE(base, PhiloxUniformDouble(/*seed=*/1, /*step=*/0, /*draw=*/1));
}

// A coarse spread check: over many steps the draws should populate every decile
// of [0, 1). A 24-bit float generator could still do this, but it guards
// against a stuck or trivially-biased stream.
TEST(PhiloxTest, UniformCoversAllDeciles) {
  std::set<int> deciles;
  for (std::uint64_t step = 0; step < 10000; ++step) {
    const double u = PhiloxUniformDouble(/*seed=*/99, step, /*draw=*/0);
    deciles.insert(static_cast<int>(u * 10.0));
  }
  EXPECT_EQ(deciles.size(), 10U);
}

}  // namespace
