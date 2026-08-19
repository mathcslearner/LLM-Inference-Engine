#pragma once

#include "kernels/internal/exp_common.h"

#include <algorithm>
#include <cstdint>
#include <limits>

// Shared, intrinsic-free core of the prefill-attention variants (M6-T04;
// design: optimized-cpu-execution.md §8, §10). Mirrors the GEMM idiom
// (gemm_common.h): the online-softmax control flow — query/key blocking, the
// per-row causal `n_valid`, the running max/sum rescale, the first-block
// `alpha = 0` case — is written **once** here as a template over an ISA `Ops`
// policy, so the scalar/NEON/AVX2 TUs differ only in four arithmetic
// primitives (dot+score, exp+sum, scale, axpy). This keeps the bug-prone
// masking/rescale logic single-sourced and makes the blind-written AVX2 TU a
// pure ISA swap of the primitives.
//
// The unit of parallel work is one (head, query-block) pair (design §5). Each
// unit writes only `out[t, h, :]` for t in its query block — disjoint from
// every other unit — and each query's recurrence runs wholly within one call,
// so results are bit-identical across thread counts by construction. No
// per-worker scratch is needed: the running output accumulator IS `out`
// itself (rescaled in place, divided by the denominator at the end), and the
// score row is a fixed `kAttnKb` stack array.

namespace engine::kernels::internal {

// Block constants (design §8; named constants, an M12-T02 tuning seam). kAttnQb
// query rows share each streamed K/V block; kAttnKb key columns per block: for
// d = 128, one K + V block is 64 KB, L1/L2-resident.
inline constexpr std::int64_t kAttnQb = 32;
inline constexpr std::int64_t kAttnKb = 64;

// One prefill-attention call, flattened to raw pointers + dims (design §8).
struct PrefillArgs {
  const float* q;            // [T, H, d]
  const float* k;            // [Hkv, L, d]
  const float* v;            // [Hkv, L, d]
  float* out;                // [T, H, d]
  std::int64_t t_dim;        // T new queries
  std::int64_t heads;        // H query heads
  std::int64_t d;            // head_dim
  std::int64_t l_dim;        // L = P + T
  std::int64_t group;        // H / Hkv (GQA group size)
  std::int64_t past;         // P = L − T
  std::int64_t num_qblocks;  // ceil(T / kAttnQb)
  float scale;
};

// The blocked online-softmax recurrence for units [unit_begin, unit_end)
// (design §8). `Ops` supplies the ISA arithmetic:
//   float DotScoreRow(q, k_block, n, d, scale, scores)
//       scores[j] = scale · (q · k_block[j]) for j in [0, n); returns max_j.
//   float ExpRowSum(scores, n, m_new)
//       scores[j] = exp(scores[j] − m_new) in place; returns Σ_j scores[j].
//   void  ScaleRow(out, s, d)          out[e] *= s.
//   void  AxpyRow(out, s, x, d)        out[e] += s · x[e].
template <typename Ops>
void PrefillUnitsImpl(const PrefillArgs& a, std::int64_t unit_begin,
                      std::int64_t unit_end) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  const std::int64_t d = a.d;
  const std::int64_t l_dim = a.l_dim;
  float scores[kAttnKb];

  for (std::int64_t unit = unit_begin; unit < unit_end; ++unit) {
    const std::int64_t h = unit / a.num_qblocks;
    const std::int64_t qb = unit % a.num_qblocks;
    const std::int64_t hk = h / a.group;  // GQA: kv head for query head h
    const std::int64_t q0 = qb * kAttnQb;
    const std::int64_t q1 = std::min(q0 + kAttnQb, a.t_dim);
    const float* k_head = a.k + (hk * l_dim * d);
    const float* v_head = a.v + (hk * l_dim * d);

    for (std::int64_t t = q0; t < q1; ++t) {
      const float* q_vec = a.q + (((t * a.heads) + h) * d);
      float* out_vec = a.out + (((t * a.heads) + h) * d);
      for (std::int64_t e = 0; e < d; ++e) {
        out_vec[e] = 0.0F;  // running weighted-V accumulator
      }
      float run_max = kNegInf;                // running row max m
      float run_den = 0.0F;                   // running denominator l
      const std::int64_t limit = a.past + t;  // inclusive causal key bound

      // Stream key blocks left→right up to the query's causal boundary; blocks
      // wholly past `limit` are never visited (the causal skip).
      for (std::int64_t kb = 0; kb <= limit; kb += kAttnKb) {
        const std::int64_t kb_len = std::min<std::int64_t>(kAttnKb, l_dim - kb);
        // Valid keys in this block: [kb, min(limit, kb + kb_len − 1)]. Masked
        // keys (the diagonal block's upper part) are simply not iterated, which
        // reproduces the reference's `−inf → 0` softmax contribution exactly.
        const std::int64_t n_valid =
            std::min<std::int64_t>(limit - kb + 1, kb_len);
        const float* k_block = k_head + (kb * d);
        const float* v_block = v_head + (kb * d);

        const float block_max =
            Ops::DotScoreRow(q_vec, k_block, n_valid, d, a.scale, scores);
        const float m_new = block_max > run_max ? block_max : run_max;
        // alpha rescales the prior accumulator/denominator to the new max.
        // First block: run_max = −inf ⇒ alpha = exp(−inf) = 0 (the exp
        // polynomial flushes below kExpLo), so the zeroed accumulator is
        // untouched and run_den starts at this block's sum.
        const float alpha = ExpF32Scalar(run_max - m_new);
        const float row_sum = Ops::ExpRowSum(scores, n_valid, m_new);

        run_den = (run_den * alpha) + row_sum;
        Ops::ScaleRow(out_vec, alpha, d);
        for (std::int64_t j = 0; j < n_valid; ++j) {
          Ops::AxpyRow(out_vec, scores[j], v_block + (j * d), d);
        }
        run_max = m_new;
      }

      // Every query has at least key s = 0 valid (0 ≤ P + t), so run_den > 0.
      Ops::ScaleRow(out_vec, 1.0F / run_den, d);
    }
  }
}

}  // namespace engine::kernels::internal
