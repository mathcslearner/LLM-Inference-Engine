#include "kernels/internal/convert_impl.h"
#include "tensor/half.h"

#include <immintrin.h>

// AVX2 conversion variants (x86-64 builds only; the fp16 paths are why F16C
// is folded into the kAvx2 dispatch gate, design §4.1). Contract: bit-exact
// per element against tensor/half.h (convert.h, design §5).
//
// fp16 paths use the F16C conversion unit (VCVTPH2PS/VCVTPS2PH implement
// RNE with exact subnormal handling; MXCSR's DAZ/FTZ do not apply to these
// instructions) — except for NaN inputs, where hardware sets the quiet bit
// on every SNaN while half.h preserves surviving payloads verbatim
// (quieting only a payload that would truncate to zero). NaN lanes are
// therefore rebuilt with the same integer ops half.h uses and blended in.
// bf16 paths are pure integer arithmetic — bit-exact by construction.
//
// Signed 32-bit compares stand in for unsigned ones throughout: every
// compared value (habs ≤ 0xFFFF, abs ≤ 0x7FFFFFFF) is non-negative as
// int32. Scalar tails apply half.h per element, like the scalar variant.

namespace engine::kernels::avx2 {

namespace {

// Order-preserving narrow of eight 32-bit values (each ≤ 0xFFFF) to eight
// 16-bit values. packus saturates from *signed* 32-bit input, so values in
// [0, 0xFFFF] pass through unchanged; the 128-bit halves are packed
// explicitly to keep element order (the 256-bit pack interleaves them).
[[nodiscard]] inline __m128i NarrowU32ToU16(__m256i v) {
  return _mm_packus_epi32(_mm256_castsi256_si128(v),
                          _mm256_extracti128_si256(v, 1));
}

// Same narrow for all-ones/all-zeros lane masks (signed saturation keeps -1
// as -1, i.e. 0xFFFFFFFF → 0xFFFF).
[[nodiscard]] inline __m128i NarrowMask32To16(__m256i mask) {
  return _mm_packs_epi32(_mm256_castsi256_si128(mask),
                         _mm256_extracti128_si256(mask, 1));
}

}  // namespace

void Fp16ToFp32(const std::uint16_t* in, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
    const __m256 hw = _mm256_cvtph_ps(h);
    const __m256i w = _mm256_cvtepu16_epi32(h);
    const __m256i habs = _mm256_and_si256(w, _mm256_set1_epi32(0x7FFF));
    const __m256i is_nan = _mm256_cmpgt_epi32(habs, _mm256_set1_epi32(0x7C00));
    // half.h widening: sign << 16 | 0x7F800000 | payload << 13 — NaN-ness
    // kept, payload (and its signaling bit) exact.
    const __m256i sign =
        _mm256_slli_epi32(_mm256_and_si256(w, _mm256_set1_epi32(0x8000)), 16);
    const __m256i man =
        _mm256_slli_epi32(_mm256_and_si256(w, _mm256_set1_epi32(0x03FF)), 13);
    const __m256i nan_bits = _mm256_or_si256(
        _mm256_or_si256(sign, _mm256_set1_epi32(0x7F800000)), man);
    _mm256_storeu_ps(out + i,
                     _mm256_blendv_ps(hw, _mm256_castsi256_ps(nan_bits),
                                      _mm256_castsi256_ps(is_nan)));
  }
  for (; i < n; ++i) {
    out[i] = static_cast<float>(tensor::float16::from_bits(in[i]));
  }
}

void Fp32ToFp16(const float* in, std::uint16_t* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 f = _mm256_loadu_ps(in + i);
    const __m128i hw =
        _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256i bits = _mm256_castps_si256(f);
    const __m256i abs = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
    const __m256i is_nan =
        _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(0x7F800000));
    // half.h narrowing NaN rule: truncate the payload; force the quiet bit
    // only if the truncated payload would read as inf.
    const __m256i sign16 = _mm256_srli_epi32(
        _mm256_and_si256(
            bits, _mm256_set1_epi32(static_cast<std::int32_t>(0x80000000U))),
        16);
    const __m256i payload =
        _mm256_and_si256(_mm256_srli_epi32(abs, 13), _mm256_set1_epi32(0x03FF));
    const __m256i payload_or_quiet =
        _mm256_blendv_epi8(payload, _mm256_set1_epi32(0x0200),
                           _mm256_cmpeq_epi32(payload, _mm256_setzero_si256()));
    const __m256i nan32 = _mm256_or_si256(
        _mm256_or_si256(sign16, _mm256_set1_epi32(0x7C00)), payload_or_quiet);
    const __m128i result =
        _mm_blendv_epi8(hw, NarrowU32ToU16(nan32), NarrowMask32To16(is_nan));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), result);
  }
  for (; i < n; ++i) {
    out[i] = tensor::float16(in[i]).to_bits();
  }
}

void Bf16ToFp32(const std::uint16_t* in, float* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
    // Exact: reattach the 16 dropped mantissa bits (half.h widening).
    const __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(out + i, _mm256_castsi256_ps(w));
  }
  for (; i < n; ++i) {
    out[i] = static_cast<float>(tensor::bfloat16::from_bits(in[i]));
  }
}

void Fp32ToBf16(const float* in, std::uint16_t* out, std::int64_t n) {
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(in + i));
    const __m256i abs = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
    const __m256i is_nan =
        _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(0x7F800000));
    // NaN: truncate to the top 16 bits, forcing the quiet bit if the 7
    // surviving payload bits are zero (half.h).
    const __m256i trunc = _mm256_srli_epi32(bits, 16);
    const __m256i payload_zero =
        _mm256_cmpeq_epi32(_mm256_and_si256(trunc, _mm256_set1_epi32(0x007F)),
                           _mm256_setzero_si256());
    const __m256i nan16 = _mm256_blendv_epi8(
        trunc, _mm256_or_si256(trunc, _mm256_set1_epi32(0x0040)), payload_zero);
    // Non-NaN: RNE via bits + 0x7FFF + lsb-of-result (half.h; the epi32 add
    // wraps modulo 2^32 exactly like half.h's uint32 arithmetic). Overflow
    // to ±inf falls out of the carry.
    const __m256i lsb =
        _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_add_epi32(
        _mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF)), lsb);
    const __m256i sel =
        _mm256_blendv_epi8(_mm256_srli_epi32(rounded, 16), nan16, is_nan);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), NarrowU32ToU16(sel));
  }
  for (; i < n; ++i) {
    out[i] = tensor::bfloat16(in[i]).to_bits();
  }
}

}  // namespace engine::kernels::avx2
