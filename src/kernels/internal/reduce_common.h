#pragma once

#include <cmath>
#include <cstdint>

// Shared reduction helpers (M3-T06; design: cpu-backend.md §6.3). Plain,
// intrinsic-free C++, included by every per-ISA reduction TU and by
// reduce.cpp — the 16→1 lane fold and the ±0-aware max are literally the
// same code in every variant, which is how the Class-R bit-identity
// convention stays a single implementation rather than three that must
// agree.

namespace engine::kernels::internal {

// The fixed halving fold over the 16 accumulator lanes (reduce.h spec):
// lane[j] += lane[j + 8], then + 4, + 2, + 1. Mutates `lanes`.
[[nodiscard]] inline float FoldLanes16(float* lanes) {
  for (int half = 8; half >= 1; half /= 2) {
    for (int j = 0; j < half; ++j) {
      lanes[j] += lanes[j + half];
    }
  }
  return lanes[0];
}

// Total-order max on the NaN-free domain: ordinary fp comparison with ±0
// ties resolved as -0.0f < +0.0f (reduce.h). Equal non-zero values are
// bit-identical, so the tie branch only ever decides between zero signs.
[[nodiscard]] inline float TotalOrderMax(float a, float b) {
  if (a < b) {
    return b;
  }
  if (b < a) {
    return a;
  }
  return std::signbit(a) ? b : a;
}

}  // namespace engine::kernels::internal
