#include "kernels/internal/gemm_common.h"
#include "kernels/internal/gemm_impl.h"

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>

// NEON packed-GEMM variants (arm64 builds only; design:
// optimized-cpu-execution.md §3.4 — NEON is the AArch64 baseline, no extra
// flags). Register-tiled MR=4 × NR=16 (4 rows × 4×float32x4 accumulators, plus
// the broadcast and the 4 widened weight vectors — well within the 32×128-bit
// register file, §3.4). Each accumulator is one lane's running fp32 sum over k
// via fused multiply-add (`vfmaq_n_f32`); FMA contraction is what makes this
// Class T vs the scalar variant (§10), but the per-output register is never
// split across threads, so it stays bit-identical across thread counts.

namespace engine::kernels::neon {

namespace {

constexpr int kMr = 4;
constexpr int kVecPerPanel = kNr / 4;  // 4 float32x4 vectors span the 16 lanes

// Widen one panel column Wp[p, kk, :] (16 stored elements) to 4 float32x4.
struct Bf16Widen {
  static void Load(const std::uint16_t* p, float32x4_t w[kVecPerPanel]) {
    const uint16x8_t lo = vld1q_u16(p);
    const uint16x8_t hi = vld1q_u16(p + 8);
    // bf16 → f32 is a 16-bit left shift into the fp32 mantissa (half.h).
    w[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(lo), 16));
    w[1] = vreinterpretq_f32_u32(vshll_high_n_u16(lo, 16));
    w[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(hi), 16));
    w[3] = vreinterpretq_f32_u32(vshll_high_n_u16(hi, 16));
  }
};

struct F16Widen {
  static void Load(const std::uint16_t* p, float32x4_t w[kVecPerPanel]) {
    w[0] = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
    w[1] = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p + 4)));
    w[2] = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p + 8)));
    w[3] = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p + 12)));
  }
};

struct F32Widen {
  static void Load(const float* p, float32x4_t w[kVecPerPanel]) {
    w[0] = vld1q_f32(p);
    w[1] = vld1q_f32(p + 4);
    w[2] = vld1q_f32(p + 8);
    w[3] = vld1q_f32(p + 12);
  }
};

// One panel p × mr rows (rows [mb, mb+mr)), all k, into y. Split out of the
// tile loop to keep each function's nesting readable (and under the tidy
// cognitive-complexity threshold).
template <typename StoredT, typename Widen>
void ComputePanel(const float* x, std::int64_t k, std::int64_t n,
                  const StoredT* wp, const float* bias, float* y,
                  std::int64_t mb, int mr, std::int64_t p) {
  const StoredT* panel = wp + (p * k * kNr);
  float32x4_t acc[kMr][kVecPerPanel];
  for (int i = 0; i < mr; ++i) {
    for (int c = 0; c < kVecPerPanel; ++c) {
      acc[i][c] = vdupq_n_f32(0.0F);
    }
  }
  for (std::int64_t kk = 0; kk < k; ++kk) {
    float32x4_t w[kVecPerPanel];
    Widen::Load(panel + (kk * kNr), w);
    for (int i = 0; i < mr; ++i) {
      const float a = x[((mb + i) * k) + kk];
      for (int c = 0; c < kVecPerPanel; ++c) {
        acc[i][c] = vfmaq_n_f32(acc[i][c], w[c], a);
      }
    }
  }
  float out[kMr * kNr];
  for (int i = 0; i < mr; ++i) {
    for (int c = 0; c < kVecPerPanel; ++c) {
      vst1q_f32(out + (i * kNr) + (static_cast<std::int64_t>(c) * 4),
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

}  // namespace engine::kernels::neon
