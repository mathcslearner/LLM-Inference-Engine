#pragma once

#include <array>
#include <cstdint>

// Philox4x32-10 counter-based RNG (Salmon, Moraes, Shaw & Dror, "Parallel
// Random Numbers: As Easy as 1, 2, 3", SC'11 — the generator behind Random123,
// JAX, cuRAND and PyTorch). Sampling (M7-T02; design:
// docs/design/model-execution.md §15.3) uses it as a *stateless* stream: the
// whole output is a pure function of the key and counter, so a draw is fully
// determined by its `(seed, step, draw)` coordinate with no mutable per-request
// RNG object to advance. That is exactly what makes a sampled token
// reproducible per `(seed, step)` and independent of batch composition (the T06
// batched sampler keys the same coordinate), and it lets `Sampler::Sample` stay
// `const`.
//
// The block function is `constexpr` so its known-answer vectors are checked at
// compile time (see philox_test.cpp).

namespace engine::sampling {

// One 128-bit Philox output block, four 32-bit words.
using Philox4x32Block = std::array<std::uint32_t, 4>;

namespace detail {

// 32x32 -> 64 multiply, split into the high and low 32-bit halves. `unsigned`
// 64-bit arithmetic is exact and constexpr-evaluable.
constexpr void MulHiLo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi,
                         std::uint32_t& lo) {
  const std::uint64_t product =
      static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
  hi = static_cast<std::uint32_t>(product >> 32);
  lo = static_cast<std::uint32_t>(product);
}

// The Philox4x32 single-round S-box (the multiply constants are from the paper
// / Random123).
constexpr Philox4x32Block Philox4x32Round(
    const Philox4x32Block& ctr, const std::array<std::uint32_t, 2>& key) {
  constexpr std::uint32_t kM0 = 0xD2511F53U;
  constexpr std::uint32_t kM1 = 0xCD9E8D57U;
  std::uint32_t hi0 = 0;
  std::uint32_t lo0 = 0;
  std::uint32_t hi1 = 0;
  std::uint32_t lo1 = 0;
  MulHiLo32(kM0, ctr[0], hi0, lo0);
  MulHiLo32(kM1, ctr[2], hi1, lo1);
  return {static_cast<std::uint32_t>(hi1 ^ ctr[1] ^ key[0]), lo1,
          static_cast<std::uint32_t>(hi0 ^ ctr[3] ^ key[1]), lo0};
}

// The per-round key bump (the golden-ratio / sqrt Weyl constants).
constexpr void BumpKey(std::array<std::uint32_t, 2>& key) {
  key[0] += 0x9E3779B9U;  // Weyl constant for the low key word
  key[1] += 0xBB67AE85U;  // Weyl constant for the high key word
}

}  // namespace detail

// Ten-round Philox4x32. The first round uses `key` unbumped; each subsequent
// round bumps first (9 bumps, 10 rounds) — the Random123 convention, so the
// standard known-answer vectors apply verbatim.
[[nodiscard]] constexpr Philox4x32Block Philox4x32_10(
    std::array<std::uint32_t, 2> key, Philox4x32Block ctr) {
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      detail::BumpKey(key);
    }
    ctr = detail::Philox4x32Round(ctr, key);
  }
  return ctr;
}

// A uniform double in [0, 1) with 53 bits of randomness, drawn from the
// `(seed, step, draw)` coordinate. `seed` splits into the two key words; `step`
// occupies the low two counter words and `draw` the third (so a step can draw
// several independent variates — T02 needs only `draw = 0`). 53 bits (float's
// 24 are too coarse for a ~150k-vocab tail) come from two output words.
[[nodiscard]] constexpr double PhiloxUniformDouble(std::uint64_t seed,
                                                   std::uint64_t step,
                                                   std::uint32_t draw) {
  const std::array<std::uint32_t, 2> key = {
      static_cast<std::uint32_t>(seed & 0xFFFFFFFFU),
      static_cast<std::uint32_t>(seed >> 32)};
  const Philox4x32Block ctr = {static_cast<std::uint32_t>(step & 0xFFFFFFFFU),
                               static_cast<std::uint32_t>(step >> 32), draw,
                               0U};
  const Philox4x32Block out = Philox4x32_10(key, ctr);
  const std::uint64_t bits =
      (static_cast<std::uint64_t>(out[1]) << 32) | out[0];
  // Top 53 bits scaled by 2^-53 -> a uniform in [0, 1).
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace engine::sampling
