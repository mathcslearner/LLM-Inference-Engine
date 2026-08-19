#pragma once

#include <cstdint>

// Public vector `expf` (M7-T06; polynomial from M6-T03, design:
// docs/design/optimized-cpu-execution.md §10). `out[i] = exp(in[i])` for
// `i in [0, n)`, using the shared degree-5 minimax polynomial (≤2 ulp of
// `std::expf`, bounded by `vector_exp_test`) dispatched to the host ISA
// (scalar / NEON / AVX2) exactly once per process. Any alignment; `out` may
// alias `in`.
//
// Numeric contract (inherited verbatim from internal/exp_common.h, the
// property the softmax mask relies on): `in[i] < kExpLo` (including −inf)
// maps to exactly `+0.0`, so a masked (`-inf`) logit produces a `0` weight
// with no rounding; `in[i] > kExpHi` saturates; NaN → NaN.
//
// **Unthreaded on purpose.** Unlike `SoftmaxF32`/`SumF32`, this entry never
// touches the thread pool — it is meant to be called *inside* a caller's own
// `parallel_for` body (the sampling module runs one exp per sequence row, one
// row per worker), where entering the pool would violate the no-nesting rule
// (parallel/parallel_for.h). Callers that want the row itself threaded wrap
// this in their own region.
//
// M6 kept the polynomial an internal detail of the softmax/SiLU kernels
// (`exp` shipped no public entry point); M7-T06 promotes it because the
// `sampling` module's reference softmax/log-softmax reuse it directly
// (docs/design/model-execution.md §15.2), which is a layer-2 → layer-1 edge
// ADR-002 already permits (as `model → kernels` is).

namespace engine::kernels {

// out[i] = exp(in[i]) for i in [0, n). n >= 0; out may alias in. Runs on the
// calling thread only (see the no-threading note above).
void ExpF32(const float* in, float* out, std::int64_t n);

}  // namespace engine::kernels
