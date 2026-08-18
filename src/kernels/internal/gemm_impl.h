#pragma once

#include "kernels/dispatch.h"

#include <cstdint>

// Per-ISA variants of the packed GEMM micro-kernel (M6-T02; design:
// optimized-cpu-execution.md §3.4, §5). Internal: included only by gemm.cpp,
// the per-ISA TUs that define these, and tests.
//
// A variant computes ONE output tile — rows [m0, m1) × panels [p0, p1) — over
// all K, and is a single-threaded chunk body (threading and the tile grid live
// in gemm.cpp's public entries, §5). The tile boundary is the whole ISA seam:
// register-tiling (the per-ISA `MR` row block, §3.4) and weight widening live
// inside each variant; the tile-grid partition and dtype dispatch do not. Each
// output element accumulates its K products in one fp32 accumulator in
// ascending-k order (bit-identical across thread counts & tile shapes, §10).
//
// `wp` is the packed weight (§3.2): panel p at `wp + p*k*kNr`, element (kk, r)
// at `wp[(p*k + kk)*kNr + r]`. `x` is [., k] row-major fp32; `y` is [., n]
// row-major fp32. Columns col = p*kNr + r with col >= n are the last panel's
// zero pad — never stored (their accumulators are weight-zero anyway). `bias`
// is null or fp32 [n], added once after the K accumulation. The 16-bit
// variants take `wp` as std::uint16_t bit patterns (bf16/f16); the f32
// variants take float.

namespace engine::kernels {

namespace scalar {
void GemmTileBf16(const float* x, std::int64_t k, std::int64_t n,
                  const std::uint16_t* wp, const float* bias, float* y,
                  std::int64_t m0, std::int64_t m1, std::int64_t p0,
                  std::int64_t p1);
void GemmTileF16(const float* x, std::int64_t k, std::int64_t n,
                 const std::uint16_t* wp, const float* bias, float* y,
                 std::int64_t m0, std::int64_t m1, std::int64_t p0,
                 std::int64_t p1);
void GemmTileF32(const float* x, std::int64_t k, std::int64_t n,
                 const float* wp, const float* bias, float* y, std::int64_t m0,
                 std::int64_t m1, std::int64_t p0, std::int64_t p1);
}  // namespace scalar

#if defined(ENGINE_ARCH_ARM64)
namespace neon {
void GemmTileBf16(const float* x, std::int64_t k, std::int64_t n,
                  const std::uint16_t* wp, const float* bias, float* y,
                  std::int64_t m0, std::int64_t m1, std::int64_t p0,
                  std::int64_t p1);
void GemmTileF16(const float* x, std::int64_t k, std::int64_t n,
                 const std::uint16_t* wp, const float* bias, float* y,
                 std::int64_t m0, std::int64_t m1, std::int64_t p0,
                 std::int64_t p1);
void GemmTileF32(const float* x, std::int64_t k, std::int64_t n,
                 const float* wp, const float* bias, float* y, std::int64_t m0,
                 std::int64_t m1, std::int64_t p0, std::int64_t p1);
}  // namespace neon
#endif

#if defined(ENGINE_ARCH_X86_64)
namespace avx2 {
// Compiled with per-source -mavx2 -mfma -mf16c (the f16 path uses F16C); must
// only run when dispatch selected kAvx2.
void GemmTileBf16(const float* x, std::int64_t k, std::int64_t n,
                  const std::uint16_t* wp, const float* bias, float* y,
                  std::int64_t m0, std::int64_t m1, std::int64_t p0,
                  std::int64_t p1);
void GemmTileF16(const float* x, std::int64_t k, std::int64_t n,
                 const std::uint16_t* wp, const float* bias, float* y,
                 std::int64_t m0, std::int64_t m1, std::int64_t p0,
                 std::int64_t p1);
void GemmTileF32(const float* x, std::int64_t k, std::int64_t n,
                 const float* wp, const float* bias, float* y, std::int64_t m0,
                 std::int64_t m1, std::int64_t p0, std::int64_t p1);
}  // namespace avx2
#endif

namespace detail {

// Test seam (M3 audit convention, internal/elementwise_impl.h): the variant
// Select would return for `isa`, scalar fallback included — lets tests assert
// the build's vector slot is actually populated, so a bit-compare against
// scalar is not silently scalar-vs-scalar.
using GemmTile16Fn = void (*)(const float*, std::int64_t, std::int64_t,
                              const std::uint16_t*, const float*, float*,
                              std::int64_t, std::int64_t, std::int64_t,
                              std::int64_t);
using GemmTile32Fn = void (*)(const float*, std::int64_t, std::int64_t,
                              const float*, const float*, float*, std::int64_t,
                              std::int64_t, std::int64_t, std::int64_t);
[[nodiscard]] GemmTile16Fn Bf16TileVariant(Isa isa);
[[nodiscard]] GemmTile16Fn F16TileVariant(Isa isa);
[[nodiscard]] GemmTile32Fn F32TileVariant(Isa isa);

}  // namespace detail

}  // namespace engine::kernels
