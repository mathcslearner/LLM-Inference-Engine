#include "kernels/internal/gemm_common.h"
#include "kernels/internal/gemm_impl.h"
#include "tensor/half.h"

#include <cstdint>

// Scalar packed-GEMM variants (M6-T02; design: optimized-cpu-execution.md
// §3.4). The forced-scalar test target and the portability floor. Written
// straight from the §3.2 layout with no register blocking: for each output
// row and panel, a `kNr`-wide fp32 accumulator array summed ascending-k, the
// weight widened per element through tensor/half.h — which makes the scalar
// packed result *bit-identical* to `cpu::gemm` / a naive triple loop (same
// single fp32 accumulator, same ascending-k order, same widening). The vector
// variants reassociate (FMA, horizontal folds) and are Class T instead (§10).

namespace engine::kernels::scalar {

namespace {

// One output tile — rows [m0, m1) × panels [p0, p1), all k — with the stored
// weight widened per element by `widen`. `StoredT` is the packed dtype.
template <typename StoredT, typename WidenFn>
void GemmTile(const float* x, std::int64_t k, std::int64_t n, const StoredT* wp,
              const float* bias, float* y, std::int64_t m0, std::int64_t m1,
              std::int64_t p0, std::int64_t p1, WidenFn widen) {
  for (std::int64_t m = m0; m < m1; ++m) {
    const float* x_row = x + (m * k);
    for (std::int64_t p = p0; p < p1; ++p) {
      const StoredT* panel = wp + (p * k * kNr);
      float acc[kNr] = {};  // +0.0 init; single accumulator per lane.
      for (std::int64_t kk = 0; kk < k; ++kk) {
        const float a = x_row[kk];
        const StoredT* wcol = panel + (kk * kNr);
        for (std::int64_t r = 0; r < kNr; ++r) {
          acc[r] += a * widen(wcol[r]);
        }
      }
      internal::StorePanelBlock(acc, /*mr=*/1, p, n, bias, y, m);
    }
  }
}

}  // namespace

void GemmTileBf16(const float* x, std::int64_t k, std::int64_t n,
                  const std::uint16_t* wp, const float* bias, float* y,
                  std::int64_t m0, std::int64_t m1, std::int64_t p0,
                  std::int64_t p1) {
  GemmTile(x, k, n, wp, bias, y, m0, m1, p0, p1, [](std::uint16_t bits) {
    return static_cast<float>(tensor::bfloat16::from_bits(bits));
  });
}

void GemmTileF16(const float* x, std::int64_t k, std::int64_t n,
                 const std::uint16_t* wp, const float* bias, float* y,
                 std::int64_t m0, std::int64_t m1, std::int64_t p0,
                 std::int64_t p1) {
  GemmTile(x, k, n, wp, bias, y, m0, m1, p0, p1, [](std::uint16_t bits) {
    return static_cast<float>(tensor::float16::from_bits(bits));
  });
}

void GemmTileF32(const float* x, std::int64_t k, std::int64_t n,
                 const float* wp, const float* bias, float* y, std::int64_t m0,
                 std::int64_t m1, std::int64_t p0, std::int64_t p1) {
  GemmTile(x, k, n, wp, bias, y, m0, m1, p0, p1, [](float v) { return v; });
}

}  // namespace engine::kernels::scalar
