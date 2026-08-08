#pragma once

#include "tensor/half.h"

#include <cstdint>

// fp16/bf16 ↔ fp32 conversion kernels (M3-T06; design:
// docs/design/cpu-backend.md §5). Numerics class E (§6.3): every entry is
// bit-exact per element against the tensor/half.h definitions — the single
// source of truth for conversion semantics (round-to-nearest-even
// narrowing; exact widening; NaN payloads truncated, quiet bit forced only
// when the truncated payload would read as inf; subnormals handled exactly).
// The vector variants use hardware conversion units for the fast path and
// reconstruct NaN lanes with integer ops, because hardware quiets every
// SNaN while half.h preserves surviving payloads verbatim.
//
// Shape rules: n >= 0 (n == 0 is a no-op); any pointer alignment (design
// §7); `in` and `out` must not overlap (the element sizes differ).

namespace engine::kernels {

// out[i] = float(in[i]) — exact widening.
void Fp16ToFp32(const tensor::float16* in, float* out, std::int64_t n);

// out[i] = float16(in[i]) — RNE narrowing per half.h.
void Fp32ToFp16(const float* in, tensor::float16* out, std::int64_t n);

// out[i] = float(in[i]) — exact widening.
void Bf16ToFp32(const tensor::bfloat16* in, float* out, std::int64_t n);

// out[i] = bfloat16(in[i]) — RNE narrowing per half.h.
void Fp32ToBf16(const float* in, tensor::bfloat16* out, std::int64_t n);

}  // namespace engine::kernels
