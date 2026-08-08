#include "kernels/internal/reduce_common.h"
#include "kernels/internal/reduce_impl.h"

// Scalar reduction variants (design §6.3 Class R). SumF32Chunk *is* the
// 16-lane specification, written directly: a 16-float accumulator array in
// place of NEON's 4×float32x4_t / AVX2's 2×__m256. The compiler may
// auto-vectorize the loop — any transform that preserves each lane's add
// sequence is exact, and without -ffast-math it may not reassociate.

namespace engine::kernels::scalar {

float SumF32Chunk(const float* x, std::int64_t n) {
  float lanes[16] = {};  // +0.0F init is part of the spec (reduce.h).
  for (std::int64_t i = 0; i < n; ++i) {
    lanes[i % 16] += x[i];
  }
  return internal::FoldLanes16(lanes);
}

float MaxF32Chunk(const float* x, std::int64_t n) {
  float m = x[0];
  for (std::int64_t i = 1; i < n; ++i) {
    m = internal::TotalOrderMax(m, x[i]);
  }
  return m;
}

}  // namespace engine::kernels::scalar
