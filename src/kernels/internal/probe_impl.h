#pragma once

#include "kernels/dispatch.h"

// Per-ISA variants of the dispatch probe (M3-T05; design: cpu-backend.md
// §4.2). Internal: included only by probe.cpp and the per-ISA TUs that
// define these. Declarations are guarded by the same arch macros the
// build's source lists use (ENGINE_ARCH_ARM64 / ENGINE_ARCH_X86_64, defined
// by kernels' CMake from CMAKE_SYSTEM_PROCESSOR), so a KernelTable slot can
// be populated exactly when the variant's TU is present in the build — the
// declaration set and the source list cannot disagree without a compile
// error.

namespace engine::kernels {

namespace scalar {
// Compiled on every platform, no arch flags (design §4.4 rule 3).
Isa Probe();
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
// arm64 builds only; NEON is the AArch64 baseline, so no extra flags.
Isa Probe();
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// x86-64 builds only; compiled with per-source -mavx2 -mfma -mf16c. Must
// only be called when dispatch selected kAvx2 (the host executes AVX2).
Isa Probe();
}  // namespace avx2
#endif

}  // namespace engine::kernels
