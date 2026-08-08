#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Per-ISA variants of the conversion kernels (M3-T06; design:
// cpu-backend.md §4.2, §5). Internal: included only by convert.cpp, the
// per-ISA TUs, and tests. Variants traffic in raw 16-bit patterns
// (std::uint16_t) rather than the tensor half types so the vector TUs can
// load them directly; the public entry points (convert.h) own the one
// pointer reinterpretation. Semantics are bit-exact per element against
// tensor/half.h — see convert.h.

namespace engine::kernels {

namespace scalar {
void Fp16ToFp32(const std::uint16_t* in, float* out, std::int64_t n);
void Fp32ToFp16(const float* in, std::uint16_t* out, std::int64_t n);
void Bf16ToFp32(const std::uint16_t* in, float* out, std::int64_t n);
void Fp32ToBf16(const float* in, std::uint16_t* out, std::int64_t n);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void Fp16ToFp32(const std::uint16_t* in, float* out, std::int64_t n);
void Fp32ToFp16(const float* in, std::uint16_t* out, std::int64_t n);
void Bf16ToFp32(const std::uint16_t* in, float* out, std::int64_t n);
void Fp32ToBf16(const float* in, std::uint16_t* out, std::int64_t n);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma -mf16c (the fp16 paths use the F16C
// conversion unit); must only run when dispatch selected kAvx2.
void Fp16ToFp32(const std::uint16_t* in, float* out, std::int64_t n);
void Fp32ToFp16(const float* in, std::uint16_t* out, std::int64_t n);
void Bf16ToFp32(const std::uint16_t* in, float* out, std::int64_t n);
void Fp32ToBf16(const float* in, std::uint16_t* out, std::int64_t n);
}  // namespace avx2
#endif

namespace detail {

// Test seam (M3 audit): the table entry Select would return for `isa` —
// see internal/elementwise_impl.h for the rationale.
using ConvertWidenFn = void (*)(const std::uint16_t*, float*, std::int64_t);
using ConvertNarrowFn = void (*)(const float*, std::uint16_t*, std::int64_t);
[[nodiscard]] ConvertWidenFn Fp16ToFp32Variant(Isa isa);
[[nodiscard]] ConvertNarrowFn Fp32ToFp16Variant(Isa isa);
[[nodiscard]] ConvertWidenFn Bf16ToFp32Variant(Isa isa);
[[nodiscard]] ConvertNarrowFn Fp32ToBf16Variant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
