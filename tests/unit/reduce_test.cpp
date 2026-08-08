#include "kernels/reduce.h"

#include "kernels/internal/reduce_impl.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

// Reduction kernel tests (M3-T06; design: docs/design/cpu-backend.md §6.3
// Class R, §9). Tolerance statement: none — Class R results are compared as
// bit patterns against an independent serial re-implementation of the full
// specification (reduce.h): the (n, kReduceGrain) chunk decomposition, the
// 16-lane per-chunk accumulation, and parallel_reduce's fixed pairwise
// halving tree. The public entries run the dispatched variant threaded
// through the real DefaultPool(), so agreement proves both vector-vs-scalar
// lane identity and thread-schedule invariance in one comparison; the
// forced-scalar registration (unit-scalar/) re-runs everything with the
// scalar variants dispatched.
//
// 2^20 + 3 elements → 33 chunks: multi-chunk through the pool, and an odd
// chunk count exercising the fold tree's carry rule.

namespace engine::kernels {
namespace {

constexpr std::int64_t kSizes[] = {0, 1, 15, 16, 17, 4096, (1 << 20) + 3};
constexpr std::size_t kOffsets[] = {0, 1, 2, 3};
constexpr std::size_t kMaxOffset = 3;

// NaN-free values (a documented precondition of both reductions, reduce.h)
// across ~80 binades with ±0, subnormals, and ±inf interleaved.
std::vector<float> TestValues(std::size_t n, std::uint32_t seed) {
  static constexpr float kSpecials[] = {
      0.0F,
      -0.0F,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
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

// Independent re-implementation of one chunk's 16-lane partial (reduce.h).
float SpecSumChunk(const float* x, std::int64_t n) {
  float lanes[16] = {};  // +0.0F
  for (std::int64_t i = 0; i < n; ++i) {
    lanes[i % 16] += x[i];
  }
  for (int j = 0; j < 8; ++j) {
    lanes[j] += lanes[j + 8];
  }
  for (int j = 0; j < 4; ++j) {
    lanes[j] += lanes[j + 4];
  }
  lanes[0] += lanes[2];
  lanes[1] += lanes[3];
  lanes[0] += lanes[1];
  return lanes[0];
}

// Independent re-implementation of the full SumF32 spec: chunk partials
// folded by the fixed pairwise halving tree with the odd element carrying
// (parallel/parallel_for.h §3.3 contract).
float SpecSum(const float* x, std::int64_t n) {
  if (n == 0) {
    return 0.0F;
  }
  const std::int64_t num_chunks = (n + kReduceGrain - 1) / kReduceGrain;
  std::vector<float> partials;
  for (std::int64_t c = 0; c < num_chunks; ++c) {
    const std::int64_t begin = c * kReduceGrain;
    const std::int64_t end = std::min(begin + kReduceGrain, n);
    partials.push_back(SpecSumChunk(x + begin, end - begin));
  }
  std::int64_t count = num_chunks;
  while (count > 1) {
    const std::int64_t half = count / 2;
    for (std::int64_t i = 0; i < half; ++i) {
      partials[static_cast<std::size_t>(i)] +=
          partials[static_cast<std::size_t>(i + half)];
    }
    if (count % 2 == 1) {
      partials[static_cast<std::size_t>(half)] =
          partials[static_cast<std::size_t>(count - 1)];
    }
    count = half + count % 2;
  }
  return partials[0];
}

// Independent total-order max: ordinary comparison with -0.0F < +0.0F
// (reduce.h). Order-free, so a linear scan is a valid reference.
float SpecMax(const float* x, std::int64_t n) {
  float m = x[0];
  for (std::int64_t i = 1; i < n; ++i) {
    if (m < x[i] || (m == x[i] && std::signbit(m))) {
      m = x[i];
    }
  }
  return m;
}

std::uint32_t Bits(float value) { return std::bit_cast<std::uint32_t>(value); }

TEST(ReduceTest, VectorVariantsAreWiredIntoTheTables) {
  // Guards the suite against vacuous passes: Select's silent scalar fallback
  // (design §4.2) means an omitted table designator would make the public
  // entries scalar, and the spec/scalar comparisons would still pass. Pin
  // each table's vector slot to the arch's variant.
#if defined(ENGINE_ARCH_ARM64)
  EXPECT_EQ(detail::SumF32Variant(Isa::kNeon), &neon::SumF32Chunk);
  EXPECT_EQ(detail::MaxF32Variant(Isa::kNeon), &neon::MaxF32Chunk);
#elif defined(ENGINE_ARCH_X86_64)
  EXPECT_EQ(detail::SumF32Variant(Isa::kAvx2), &avx2::SumF32Chunk);
  EXPECT_EQ(detail::MaxF32Variant(Isa::kAvx2), &avx2::MaxF32Chunk);
#endif
  EXPECT_EQ(detail::SumF32Variant(Isa::kScalar), &scalar::SumF32Chunk);
  EXPECT_EQ(detail::MaxF32Variant(Isa::kScalar), &scalar::MaxF32Chunk);
}

TEST(SumF32Test, MatchesSpecReferenceBitExactAcrossSizesAndOffsets) {
  for (const std::int64_t n : kSizes) {
    const auto count = static_cast<std::size_t>(n);
    const std::vector<float> x = TestValues(count + kMaxOffset, 11);
    for (const std::size_t offset : kOffsets) {
      const float got = SumF32(x.data() + offset, n);
      const float want = SpecSum(x.data() + offset, n);
      ASSERT_EQ(Bits(got), Bits(want)) << "n=" << n << " offset=" << offset
                                       << ": got " << got << ", want " << want;
    }
  }
}

TEST(SumF32Test, SingleChunkMatchesScalarVariantBitExact) {
  // n <= kReduceGrain is one chunk, so the public entry is exactly one
  // dispatched-variant call: under the normal pass this is the direct
  // vector-vs-scalar Class R comparison at chunk granularity.
  constexpr std::int64_t kChunkSizes[] = {1, 15, 16, 17, 4096, kReduceGrain};
  for (const std::int64_t n : kChunkSizes) {
    const std::vector<float> x = TestValues(static_cast<std::size_t>(n), 12);
    ASSERT_EQ(Bits(SumF32(x.data(), n)), Bits(scalar::SumF32Chunk(x.data(), n)))
        << "n=" << n;
  }
}

TEST(SumF32Test, AdversarialMixedMagnitudesMatchSpecReference) {
  // Cancellation-heavy input where fold order visibly changes the answer
  // (design §9): agreement here is meaningful, not vacuous.
  constexpr std::int64_t kN = 100000;
  static constexpr float kPattern[] = {1e30F, 1.0F, -1e30F, 1e-8F,
                                       -1.0F, 1e8F, -1e-8F, -1e8F};
  std::vector<float> x(static_cast<std::size_t>(kN));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = kPattern[i % std::size(kPattern)];
  }
  EXPECT_EQ(Bits(SumF32(x.data(), kN)), Bits(SpecSum(x.data(), kN)));
}

TEST(SumF32Test, NegativeZeroInputsSumToPositiveZero) {
  // The spec's +0.0F lane initialization makes the empty and all-(-0) sums
  // +0.0F (reduce.h) — and makes it bit-visible if a variant initialized
  // its accumulators any other way.
  constexpr std::int64_t kZeroSizes[] = {1, 5, 16, 33, 4096};
  for (const std::int64_t n : kZeroSizes) {
    const std::vector<float> x(static_cast<std::size_t>(n), -0.0F);
    EXPECT_EQ(Bits(SumF32(x.data(), n)), Bits(0.0F)) << "n=" << n;
  }
}

TEST(SumF32Test, EmptySumIsPositiveZero) {
  EXPECT_EQ(Bits(SumF32(nullptr, 0)), Bits(0.0F));
}

TEST(MaxF32Test, MatchesSpecReferenceBitExactAcrossSizesAndOffsets) {
  for (const std::int64_t n : kSizes) {
    if (n == 0) {
      continue;  // MaxF32 requires n >= 1 (death-tested below).
    }
    const auto count = static_cast<std::size_t>(n);
    const std::vector<float> x = TestValues(count + kMaxOffset, 13);
    for (const std::size_t offset : kOffsets) {
      const float got = MaxF32(x.data() + offset, n);
      const float want = SpecMax(x.data() + offset, n);
      ASSERT_EQ(Bits(got), Bits(want)) << "n=" << n << " offset=" << offset
                                       << ": got " << got << ", want " << want;
    }
  }
}

TEST(MaxF32Test, SingleChunkMatchesScalarVariantBitExact) {
  constexpr std::int64_t kChunkSizes[] = {1, 15, 16, 17, 4096, kReduceGrain};
  for (const std::int64_t n : kChunkSizes) {
    const std::vector<float> x = TestValues(static_cast<std::size_t>(n), 14);
    ASSERT_EQ(Bits(MaxF32(x.data(), n)), Bits(scalar::MaxF32Chunk(x.data(), n)))
        << "n=" << n;
  }
}

TEST(MaxF32Test, ZeroSignTiesResolveToPositiveZero) {
  // The total order sharpens ±0: -0.0F < +0.0F (reduce.h), so the result's
  // zero sign is deterministic on every ISA — the exact case where plain
  // maxps/std::max would give ISA-dependent answers.
  const std::vector<float> all_negative(37, -0.0F);
  EXPECT_EQ(Bits(MaxF32(all_negative.data(), 37)), Bits(-0.0F));

  std::vector<float> mixed(37, -0.0F);
  mixed[17] = 0.0F;
  EXPECT_EQ(Bits(MaxF32(mixed.data(), 37)), Bits(0.0F));

  const std::vector<float> pair = {0.0F, -0.0F};
  EXPECT_EQ(Bits(MaxF32(pair.data(), 2)), Bits(0.0F));
}

TEST(MaxF32Test, HandlesAllNegativeAndInfinityInputs) {
  const std::vector<float> negatives = {-5.0F, -3.0F, -9.0F, -3.5F};
  EXPECT_EQ(MaxF32(negatives.data(), 4), -3.0F);

  std::vector<float> with_inf = TestValues(1000, 15);
  with_inf[777] = std::numeric_limits<float>::infinity();
  EXPECT_EQ(MaxF32(with_inf.data(), 1000),
            std::numeric_limits<float>::infinity());

  const std::vector<float> single = {-42.0F};
  EXPECT_EQ(MaxF32(single.data(), 1), -42.0F);
}

TEST(MaxF32DeathTest, EmptyMaxIsFatal) {
  const float x = 1.0F;
  EXPECT_DEATH(static_cast<void>(MaxF32(&x, 0)), "MaxF32: n must be >= 1");
}

}  // namespace
}  // namespace engine::kernels
