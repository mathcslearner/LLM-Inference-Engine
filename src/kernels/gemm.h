#pragma once

#include "tensor/dtype.h"

#include <cstdint>

// Packed-weight GEMM & GEMV (M6-T02; design:
// docs/design/optimized-cpu-execution.md §3, §5, §10). fp32 activations ×
// packed fp16/bf16/f32 weights, fp32 accumulation. The optimized analog of
// `cpu::gemm` (the oracle): every result here is validated against
// `cpu::gemm` within the §10 GEMM tolerance.
//
// The packed layout (§3.2): a checkpoint-order weight `W[N, K]` (N =
// out_features, K = in_features, row-major — the source of truth, cpu-backend
// §7) is repacked once at load into K-major panels of `kNr` output rows:
//
//   Wp[p, k, r] = W[p*kNr + r, k]   when p*kNr + r < N, else 0,
//   shape [PackedPanels(N), K, kNr], element (p,k,r) at ((p*K)+k)*kNr + r.
//
// So Wp[p, k, :] is one aligned `kNr`-wide contiguous vector — the kNr output
// channels' weights for input channel k. `kNr` is fixed across ISAs so the
// forced-scalar pass validates the same packed bytes the vector passes do
// (§3.2); only the micro-kernel that reads the panel is ISA-specific.
//
// Numerics (§10): each output element `y[m,n]` accumulates its K products in
// one fp32 register in ascending-k order, never split across threads — so the
// result is **bit-identical across thread counts** (asserted in tests), though
// Class T across ISAs (FMA contraction differs) and Class T vs the oracle.
//
// These are raw-pointer entry points: preconditions (non-null, contiguous,
// shape/dtype agreement, finite weights) are the CALLER's to enforce —
// `model::PackedLinear` front-loads all recoverable validation, so nothing
// inside a parallel region does anything but arithmetic (ADR-003, §5).

namespace engine::kernels {

// Panel width: kNr output rows per packed panel, fixed across all ISAs
// (§3.2 — the 16-lane Class-R convention of cpu-backend §6.3).
inline constexpr std::int64_t kNr = 16;

// Panels a weight of `n` output rows packs into: ceil(n / kNr).
[[nodiscard]] constexpr std::int64_t PackedPanels(std::int64_t n) {
  return (n + kNr - 1) / kNr;
}

// Element count of the packed buffer for a weight [n, k]:
// PackedPanels(n) * k * kNr (the last panel is zero-padded to a full kNr).
[[nodiscard]] constexpr std::int64_t PackedWeightElements(std::int64_t n,
                                                          std::int64_t k) {
  return PackedPanels(n) * k * kNr;
}

// Repack a checkpoint-order weight `w[n, k]` (row-major) into the panel layout
// `wp[PackedPanels(n), k, kNr]` (§3.2). A pure gather + zero-pad — no numeric
// conversion, so `wp` holds the SAME storage dtype as `w`; the 16-bit overload
// covers bf16/f16 (bit patterns copied verbatim), the float overload f32. The
// zero pad uses the dtype's `+0.0` bit pattern (all-zero for f32/f16/bf16).
// Threaded over panels; a pure permute, so the result is independent of the
// thread count. `wp` must have room for PackedWeightElements(n, k) elements.
// Requires n >= 1, k >= 1, non-null non-overlapping w/wp (CHECK).
void PackWeightPanels(const std::uint16_t* w, std::int64_t n, std::int64_t k,
                      std::uint16_t* wp);
void PackWeightPanels(const float* w, std::int64_t n, std::int64_t k,
                      float* wp);

// y[m, n] = x[m, k] * W^T (+ bias[n]) over the packed weight `wp` (the layout
// above, storage dtype `weight_dtype` ∈ {kFloat32, kFloat16, kBFloat16}).
//   x    : [m, k] contiguous fp32 activations.
//   wp   : packed weight, PackedWeightElements(n, k) elements of weight_dtype.
//   bias : null, or [n] contiguous fp32 (converted once at pack time, §3.5) —
//          added once after the K accumulation.
//   y    : [m, n] contiguous fp32, fully overwritten.
// Routes M == 1 to the GEMV panel-parallel path; M > 1 tiles the
// (m-block, panel-block) grid across threads. Requires m,k,n >= 1 and a
// supported weight_dtype (CHECK).
void PackedGemm(const float* x, std::int64_t m, std::int64_t k, const void* wp,
                tensor::DataType weight_dtype, std::int64_t n,
                const float* bias, float* y);

// The decode-shaped path (M == 1): one activation row over the packed weight,
// threaded across output-panel chunks (§5 — each thread streams a disjoint run
// of panels once). Identical result to PackedGemm with m == 1; exposed
// separately for the decode call site and direct testing. Requires k,n >= 1
// and a supported weight_dtype (CHECK).
void PackedGemv(const float* x, std::int64_t k, const void* wp,
                tensor::DataType weight_dtype, std::int64_t n,
                const float* bias, float* y);

}  // namespace engine::kernels
