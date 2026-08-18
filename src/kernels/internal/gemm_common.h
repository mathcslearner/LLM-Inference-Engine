#pragma once

#include "kernels/gemm.h"

#include <algorithm>
#include <cstdint>

// Shared, intrinsic-free helpers for the packed GEMM variants (M6-T02;
// design: optimized-cpu-execution.md §3.4). Included by every per-ISA GEMM TU.
// The register-tiled k-loop is ISA-specific (it lives in the per-ISA TU); the
// final store — masking the last panel's zero-pad columns and adding the
// once-converted fp32 bias (§3.5) — is identical across ISAs, so it lives here
// as one implementation each variant calls after spilling its accumulators.

namespace engine::kernels::internal {

// Store one `mr`-row × one-panel accumulator block `acc[mr][kNr]` (row-major,
// mr <= the ISA's MR) into `y[., n]` at rows [m_base, m_base+mr), panel p.
// Column col = p*kNr + r is written only when col < n (the pad is skipped —
// its accumulators are weight-zero regardless). Bias, when present, is fp32
// [n], added once here (after the K accumulation completed in the caller).
inline void StorePanelBlock(const float* acc, int mr, std::int64_t p,
                            std::int64_t n, const float* bias, float* y,
                            std::int64_t m_base) {
  const std::int64_t col0 = p * kNr;
  const std::int64_t ncols = std::min<std::int64_t>(kNr, n - col0);
  for (int mm = 0; mm < mr; ++mm) {
    float* y_row = y + ((m_base + mm) * n) + col0;
    const float* a_row = acc + (static_cast<std::int64_t>(mm) * kNr);
    if (bias != nullptr) {
      const float* bias_row = bias + col0;
      for (std::int64_t r = 0; r < ncols; ++r) {
        y_row[r] = a_row[r] + bias_row[r];
      }
    } else {
      for (std::int64_t r = 0; r < ncols; ++r) {
        y_row[r] = a_row[r];
      }
    }
  }
}

}  // namespace engine::kernels::internal
