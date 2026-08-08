#include "kernels/internal/convert_impl.h"
#include "tensor/half.h"

#include <arm_neon.h>

// NEON conversion variants (arm64 builds only). Contract: bit-exact per
// element against tensor/half.h (convert.h, design §5).
//
// fp16 paths use the hardware conversion unit (FCVTL/FCVTN implement RNE
// with exact subnormal handling; fp16 *conversion* is ARMv8.0 baseline) —
// except for NaN inputs, where hardware sets the quiet bit on every SNaN
// while half.h preserves surviving payloads verbatim (quieting only a
// payload that would truncate to zero). NaN lanes are therefore rebuilt
// with the same integer ops half.h uses and blended in; non-NaN lanes are
// untouched hardware results. bf16 paths are pure integer arithmetic
// (shift for widening, RNE rounding add for narrowing) — bit-exact by
// construction, no hardware unit involved.
//
// Environmental precondition: FPCR.FZ16 stays at its architectural default
// of 0 — nothing in the engine sets FPCR. A nonzero FZ16 would flush fp16
// subnormals in FCVTL/FCVTN and silently diverge from half.h. (The AVX2
// F16C path has no analogous hazard: MXCSR's DAZ/FTZ do not apply to
// VCVTPH2PS/VCVTPS2PH — see avx2/convert.cpp.)
//
// Scalar tails apply half.h per element, exactly like the scalar variant.

namespace engine::kernels::neon {

void Fp16ToFp32(const std::uint16_t* in, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const uint16x4_t h = vld1_u16(in + i);
    const float32x4_t hw = vcvt_f32_f16(vreinterpret_f16_u16(h));
    const uint32x4_t w = vmovl_u16(h);
    const uint32x4_t habs = vandq_u32(w, vdupq_n_u32(0x7FFFU));
    const uint32x4_t is_nan = vcgtq_u32(habs, vdupq_n_u32(0x7C00U));
    // half.h widening: sign << 16 | 0x7F800000 | payload << 13 — NaN-ness
    // kept, payload (and its signaling bit) exact.
    const uint32x4_t sign = vshlq_n_u32(vandq_u32(w, vdupq_n_u32(0x8000U)), 16);
    const uint32x4_t man = vshlq_n_u32(vandq_u32(w, vdupq_n_u32(0x03FFU)), 13);
    const uint32x4_t nan_bits =
        vorrq_u32(vorrq_u32(sign, vdupq_n_u32(0x7F800000U)), man);
    vst1q_f32(out + i, vbslq_f32(is_nan, vreinterpretq_f32_u32(nan_bits), hw));
  }
  for (; i < n; ++i) {
    out[i] = static_cast<float>(tensor::float16::from_bits(in[i]));
  }
}

void Fp32ToFp16(const float* in, std::uint16_t* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t f = vld1q_f32(in + i);
    const uint16x4_t hw = vreinterpret_u16_f16(vcvt_f16_f32(f));
    const uint32x4_t bits = vreinterpretq_u32_f32(f);
    const uint32x4_t abs = vandq_u32(bits, vdupq_n_u32(0x7FFFFFFFU));
    const uint32x4_t is_nan = vcgtq_u32(abs, vdupq_n_u32(0x7F800000U));
    // half.h narrowing NaN rule: truncate the payload; force the quiet bit
    // only if the truncated payload would read as inf.
    const uint32x4_t sign16 =
        vshrq_n_u32(vandq_u32(bits, vdupq_n_u32(0x80000000U)), 16);
    const uint32x4_t payload =
        vandq_u32(vshrq_n_u32(abs, 13), vdupq_n_u32(0x03FFU));
    const uint32x4_t payload_or_quiet = vbslq_u32(
        vceqq_u32(payload, vdupq_n_u32(0U)), vdupq_n_u32(0x0200U), payload);
    const uint32x4_t nan32 =
        vorrq_u32(vorrq_u32(sign16, vdupq_n_u32(0x7C00U)), payload_or_quiet);
    vst1_u16(out + i, vbsl_u16(vmovn_u32(is_nan), vmovn_u32(nan32), hw));
  }
  for (; i < n; ++i) {
    out[i] = tensor::float16(in[i]).to_bits();
  }
}

void Bf16ToFp32(const std::uint16_t* in, float* out, std::int64_t n) {
  // Exact: reattach the 16 dropped mantissa bits (half.h widening). 16
  // elements per iteration — the op is a pure shift, so the loop is memory
  // bound and needs the unroll to keep up with what the compiler already
  // does to the scalar variant.
  std::int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const uint16x8_t h0 = vld1q_u16(in + i);
    const uint16x8_t h1 = vld1q_u16(in + i + 8);
    vst1q_f32(out + i, vreinterpretq_f32_u32(
                           vshlq_n_u32(vmovl_u16(vget_low_u16(h0)), 16)));
    vst1q_f32(out + i + 4, vreinterpretq_f32_u32(
                               vshlq_n_u32(vmovl_u16(vget_high_u16(h0)), 16)));
    vst1q_f32(out + i + 8, vreinterpretq_f32_u32(
                               vshlq_n_u32(vmovl_u16(vget_low_u16(h1)), 16)));
    vst1q_f32(out + i + 12, vreinterpretq_f32_u32(
                                vshlq_n_u32(vmovl_u16(vget_high_u16(h1)), 16)));
  }
  for (; i < n; ++i) {
    out[i] = static_cast<float>(tensor::bfloat16::from_bits(in[i]));
  }
}

namespace {

// One 4-lane step of the bf16 narrowing, producing the narrowed 16-bit
// result: NaN lanes truncate (quiet bit forced if the 7 surviving payload
// bits are zero), everything else rounds RNE via bits + 0x7FFF +
// lsb-of-result — half.h exactly; overflow to ±inf falls out of the carry.
// vshrn does shift-by-16 + narrow in one instruction, and the NaN blend
// happens at 16-bit width, so the common path is short.
inline uint16x4_t NarrowBf16Lanes(uint32x4_t bits) {
  const uint16x4_t trunc = vshrn_n_u32(bits, 16);
  const uint32x4_t abs = vandq_u32(bits, vdupq_n_u32(0x7FFFFFFFU));
  const uint16x4_t is_nan = vmovn_u32(vcgtq_u32(abs, vdupq_n_u32(0x7F800000U)));
  const uint16x4_t payload_zero =
      vceq_u16(vand_u16(trunc, vdup_n_u16(0x007FU)), vdup_n_u16(0U));
  const uint16x4_t nan16 =
      vbsl_u16(payload_zero, vorr_u16(trunc, vdup_n_u16(0x0040U)), trunc);
  const uint32x4_t lsb = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1U));
  const uint16x4_t rounded =
      vshrn_n_u32(vaddq_u32(vaddq_u32(bits, vdupq_n_u32(0x7FFFU)), lsb), 16);
  return vbsl_u16(is_nan, nan16, rounded);
}

}  // namespace

void Fp32ToBf16(const float* in, std::uint16_t* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const uint16x4_t s0 =
        NarrowBf16Lanes(vreinterpretq_u32_f32(vld1q_f32(in + i)));
    const uint16x4_t s1 =
        NarrowBf16Lanes(vreinterpretq_u32_f32(vld1q_f32(in + i + 4)));
    const uint16x4_t s2 =
        NarrowBf16Lanes(vreinterpretq_u32_f32(vld1q_f32(in + i + 8)));
    const uint16x4_t s3 =
        NarrowBf16Lanes(vreinterpretq_u32_f32(vld1q_f32(in + i + 12)));
    vst1q_u16(out + i, vcombine_u16(s0, s1));
    vst1q_u16(out + i + 8, vcombine_u16(s2, s3));
  }
  for (; i + 4 <= n; i += 4) {
    vst1_u16(out + i,
             NarrowBf16Lanes(vreinterpretq_u32_f32(vld1q_f32(in + i))));
  }
  for (; i < n; ++i) {
    out[i] = tensor::bfloat16(in[i]).to_bits();
  }
}

}  // namespace engine::kernels::neon
