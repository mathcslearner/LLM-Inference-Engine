#include "kernels/internal/gemm_common.h"
#include "kernels/internal/gemm_impl.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdint>

// AVX2 packed-GEMM variants (x86-64 builds only; compiled with per-source
// -mavx2 -mfma -mf16c — the f16 path uses the F16C conversion unit). Design:
// optimized-cpu-execution.md §3.4. Register-tiled MR=6 × NR=16 (6 rows ×
// 2×__m256 accumulators = 12 ymm, plus the broadcast and 2 widened weight
// vectors — within the 16 ymm register file, §3.4). Each accumulator is one
// lane's running fp32 sum over k via `_mm256_fmadd_ps`; FMA contraction makes
// this Class T vs scalar (§10) yet bit-identical across thread counts (a given
// output's register is never split across threads).

namespace engine::kernels::avx2 {

namespace {

constexpr int kMr = 6;
constexpr int kVecPerPanel = kNr / 8;  // 2 __m256 vectors span the 16 lanes

struct Bf16Widen {
  static void Load(const std::uint16_t* p, __m256 w[kVecPerPanel]) {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8));
    // bf16 → f32: zero-extend to 32 bits, shift the 16 bits into the fp32
    // mantissa (half.h). No rounding — exact.
    w[0] =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16));
    w[1] =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16));
  }
};

struct F16Widen {
  static void Load(const std::uint16_t* p, __m256 w[kVecPerPanel]) {
    w[0] =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    w[1] = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8)));
  }
};

struct F32Widen {
  static void Load(const float* p, __m256 w[kVecPerPanel]) {
    w[0] = _mm256_loadu_ps(p);
    w[1] = _mm256_loadu_ps(p + 8);
  }
};

// One panel p × mr rows (rows [mb, mb+mr)), all k, into y. Split out of the
// tile loop to keep each function's nesting readable (and under the tidy
// cognitive-complexity threshold) — mirrors neon/gemm.cpp.
template <typename StoredT, typename Widen>
void ComputePanel(const float* x, std::int64_t k, std::int64_t n,
                  const StoredT* wp, const float* bias, float* y,
                  std::int64_t mb, int mr, std::int64_t p) {
  const StoredT* panel = wp + (p * k * kNr);
  __m256 acc[kMr][kVecPerPanel];
  for (int i = 0; i < mr; ++i) {
    for (int c = 0; c < kVecPerPanel; ++c) {
      acc[i][c] = _mm256_setzero_ps();
    }
  }
  for (std::int64_t kk = 0; kk < k; ++kk) {
    __m256 w[kVecPerPanel];
    Widen::Load(panel + (kk * kNr), w);
    for (int i = 0; i < mr; ++i) {
      const __m256 a = _mm256_set1_ps(x[((mb + i) * k) + kk]);
      for (int c = 0; c < kVecPerPanel; ++c) {
        acc[i][c] = _mm256_fmadd_ps(w[c], a, acc[i][c]);
      }
    }
  }
  float out[kMr * kNr];
  for (int i = 0; i < mr; ++i) {
    for (int c = 0; c < kVecPerPanel; ++c) {
      _mm256_storeu_ps(out + (i * kNr) + (static_cast<std::int64_t>(c) * 8),
                       acc[i][c]);
    }
  }
  internal::StorePanelBlock(out, mr, p, n, bias, y, mb);
}

template <typename StoredT, typename Widen>
void GemmTile(const float* x, std::int64_t k, std::int64_t n, const StoredT* wp,
              const float* bias, float* y, std::int64_t m0, std::int64_t m1,
              std::int64_t p0, std::int64_t p1) {
  for (std::int64_t mb = m0; mb < m1; mb += kMr) {
    const int mr = static_cast<int>(std::min<std::int64_t>(kMr, m1 - mb));
    for (std::int64_t p = p0; p < p1; ++p) {
      ComputePanel<StoredT, Widen>(x, k, n, wp, bias, y, mb, mr, p);
    }
  }
}

}  // namespace

void GemmTileBf16(const float* x, std::int64_t k, std::int64_t n,
                  const std::uint16_t* wp, const float* bias, float* y,
                  std::int64_t m0, std::int64_t m1, std::int64_t p0,
                  std::int64_t p1) {
  GemmTile<std::uint16_t, Bf16Widen>(x, k, n, wp, bias, y, m0, m1, p0, p1);
}

void GemmTileF16(const float* x, std::int64_t k, std::int64_t n,
                 const std::uint16_t* wp, const float* bias, float* y,
                 std::int64_t m0, std::int64_t m1, std::int64_t p0,
                 std::int64_t p1) {
  GemmTile<std::uint16_t, F16Widen>(x, k, n, wp, bias, y, m0, m1, p0, p1);
}

void GemmTileF32(const float* x, std::int64_t k, std::int64_t n,
                 const float* wp, const float* bias, float* y, std::int64_t m0,
                 std::int64_t m1, std::int64_t p0, std::int64_t p1) {
  GemmTile<float, F32Widen>(x, k, n, wp, bias, y, m0, m1, p0, p1);
}

}  // namespace engine::kernels::avx2
