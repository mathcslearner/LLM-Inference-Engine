#pragma once

#include <cstdint>

// Per-ISA variants of the elementwise kernels (M3-T06; design:
// cpu-backend.md §4.2). Internal: included only by elementwise.cpp, the
// per-ISA TUs that define these, and tests. Variants are single-threaded
// chunk bodies with the public signatures — threading lives in the public
// entry points (design §4.5 step 2). Declarations are guarded by the same
// arch macros the build's source lists use, so a KernelTable slot can be
// populated exactly when the variant's TU is present in the build.

namespace engine::kernels {

namespace scalar {
// Compiled on every platform, no arch flags (design §4.4 rule 3). The
// reference for the vector variants until the M5 cpu backend exists
// (design §2.1 bootstrap).
void AddF32(const float* a, const float* b, float* out, std::int64_t n);
void MulF32(const float* a, const float* b, float* out, std::int64_t n);
void ScaleF32(const float* x, float s, float* out, std::int64_t n);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void AddF32(const float* a, const float* b, float* out, std::int64_t n);
void MulF32(const float* a, const float* b, float* out, std::int64_t n);
void ScaleF32(const float* x, float s, float* out, std::int64_t n);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma -mf16c; must only run when dispatch
// selected kAvx2.
void AddF32(const float* a, const float* b, float* out, std::int64_t n);
void MulF32(const float* a, const float* b, float* out, std::int64_t n);
void ScaleF32(const float* x, float s, float* out, std::int64_t n);
}  // namespace avx2
#endif

}  // namespace engine::kernels
