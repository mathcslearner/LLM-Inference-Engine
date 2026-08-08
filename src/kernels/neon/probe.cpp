#include "kernels/internal/probe_impl.h"

#include <arm_neon.h>

#include <cstdint>

// NEON probe variant (arm64 builds only; design §4.4 — NEON is the AArch64
// baseline, so this TU carries no extra flags). The answer is computed with
// a real NEON intrinsic so the per-ISA TU machinery is genuinely exercised:
// a build that failed to include <arm_neon.h> support would not compile.

namespace engine::kernels::neon {

Isa Probe() {
  const int32x4_t lanes = vdupq_n_s32(static_cast<std::int32_t>(Isa::kNeon));
  return static_cast<Isa>(vgetq_lane_s32(lanes, 0));
}

}  // namespace engine::kernels::neon
