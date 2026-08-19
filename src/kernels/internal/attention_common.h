#pragma once

#include "kernels/internal/exp_common.h"

#include <algorithm>
#include <cstdint>
#include <limits>

// Shared, intrinsic-free core of the prefill (M6-T04) and decode (M6-T05)
// attention variants (design: optimized-cpu-execution.md §8, §10). Mirrors the
// GEMM idiom (gemm_common.h): the online-softmax control flow — query/key
// blocking, the per-row causal `n_valid`, the running max/sum rescale, the
// first-block `alpha = 0` case — is written **once** here as templates over an
// ISA `Ops` policy (`PrefillUnitsImpl` and `DecodeUnitsImpl`, both driving the
// **same** four primitives), so the scalar/NEON/AVX2 TUs differ only in four
// arithmetic primitives (dot+score, exp+sum, scale, axpy). This keeps the
// bug-prone masking/rescale logic single-sourced and makes the blind-written
// AVX2 TU a pure ISA swap of the primitives.
//
// The unit of parallel work is one (head, query-block) pair for prefill, one kv
// head for decode (design §5). Each unit writes disjoint output rows and each
// query's recurrence runs wholly within one call, so results are bit-identical
// across thread counts by construction. No per-worker scratch is needed: the
// running output accumulator IS `out` itself (rescaled in place, divided by the
// denominator at the end), and the score row is a fixed `kAttnKb` stack array
// (decode adds only fixed `kAttnDecodeGroupChunk` per-query state, also stack).

namespace engine::kernels::internal {

// Block constants (design §8; named constants, an M12-T02 tuning seam). kAttnQb
// query rows share each streamed K/V block; kAttnKb key columns per block: for
// d = 128, one K + V block is 64 KB, L1/L2-resident.
inline constexpr std::int64_t kAttnQb = 32;
inline constexpr std::int64_t kAttnKb = 64;

// Decode (M6-T05): the `g = H / Hkv` query heads of a kv group are processed
// together so that kv head's K/V stream from memory once (design §8). Their
// per-query online-softmax state (running max/denominator) lives in fixed stack
// arrays of this size; a group larger than the chunk is processed in ≤-chunk
// slices, re-streaming K/V once per slice (g ≤ 7 for Llama-3.2-1B / Qwen2.5, so
// real models never chunk). An M12-T02 tuning seam like the block constants.
inline constexpr std::int64_t kAttnDecodeGroupChunk = 8;

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

// One decode-attention call: a single query per query head over the full cache
// (design §8, M6-T05). q/out are [H, d] (the T == 1 slice of the prefill
// [T, H, d] layout), k/v are [Hkv, L, d] head-major. Every query attends keys
// [0, L − 1] (P = L − 1, t = 0). No num_qblocks / t_dim / past: the single
// query fixes them.
struct DecodeArgs {
  const float* q;      // [H, d]
  const float* k;      // [Hkv, L, d]
  const float* v;      // [Hkv, L, d]
  float* out;          // [H, d]
  std::int64_t d;      // head_dim
  std::int64_t l_dim;  // L = P + 1 cached+new keys
  std::int64_t group;  // H / Hkv (GQA group size)
  float scale;
};

// One decode group slice: the `gn` query heads [hq0, hq0 + gn) of a kv group,
// against that kv head's cache (`k_head`/`v_head` [L, d]), online-softmaxed.
// Split out of DecodeUnitsImpl to keep each function's nesting readable (and
// under the tidy cognitive-complexity threshold — the gemm.cpp idiom).
// `scores`/`run_max`/`run_den` are caller-owned scratch (a fixed kAttnKb score
// row; per-query state of length ≥ gn). Each query head's block loop is
// identical to PrefillUnitsImpl with T = 1 (the bit-identity invariant, §8).
template <typename Ops>
void DecodeGroupSlice(const DecodeArgs& a, const float* k_head,
                      const float* v_head, std::int64_t hq0, std::int64_t gn,
                      std::int64_t limit, float* scores, float* run_max,
                      float* run_den) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  const std::int64_t d = a.d;
  const std::int64_t l_dim = a.l_dim;

  for (std::int64_t gi = 0; gi < gn; ++gi) {
    float* out_vec = a.out + ((hq0 + gi) * d);
    for (std::int64_t e = 0; e < d; ++e) {
      out_vec[e] = 0.0F;  // running weighted-V accumulator
    }
    run_max[gi] = kNegInf;
    run_den[gi] = 0.0F;
  }

  // Stream key blocks left→right once; every query head of the slice consumes
  // each block while it is cache-resident.
  for (std::int64_t kb = 0; kb <= limit; kb += kAttnKb) {
    const std::int64_t kb_len = std::min<std::int64_t>(kAttnKb, l_dim - kb);
    const std::int64_t n_valid = std::min<std::int64_t>(limit - kb + 1, kb_len);
    const float* k_block = k_head + (kb * d);
    const float* v_block = v_head + (kb * d);

    for (std::int64_t gi = 0; gi < gn; ++gi) {
      const float* q_vec = a.q + ((hq0 + gi) * d);
      float* out_vec = a.out + ((hq0 + gi) * d);
      const float block_max =
          Ops::DotScoreRow(q_vec, k_block, n_valid, d, a.scale, scores);
      const float m_new = block_max > run_max[gi] ? block_max : run_max[gi];
      const float alpha = ExpF32Scalar(run_max[gi] - m_new);
      const float row_sum = Ops::ExpRowSum(scores, n_valid, m_new);
      run_den[gi] = (run_den[gi] * alpha) + row_sum;
      Ops::ScaleRow(out_vec, alpha, d);
      for (std::int64_t j = 0; j < n_valid; ++j) {
        Ops::AxpyRow(out_vec, scores[j], v_block + (j * d), d);
      }
      run_max[gi] = m_new;
    }
  }

  for (std::int64_t gi = 0; gi < gn; ++gi) {
    float* out_vec = a.out + ((hq0 + gi) * d);
    Ops::ScaleRow(out_vec, 1.0F / run_den[gi], d);  // run_den > 0 (key 0)
  }
}

// The decode online-softmax recurrence for kv heads [unit_begin, unit_end)
// (design §8, M6-T05). Parallelized over kv heads (the public entry), so a unit
// is one kv head's `group` query heads — processed **key-block-outer,
// query-inner** (in DecodeGroupSlice) so each streamed K/V block is reused
// across the whole group (bandwidth: K/V through DRAM once per kv head, not
// once per query head). The `Ops` policy is the SAME four primitives as prefill
// (attention_impl.h).
//
// Bit-identity with the prefill path (M6-T05 acceptance, asserted bitwise): for
// each fixed query head the block sequence, the `n_valid` per block, the
// first-block `alpha = exp(−inf) = 0`, and the four Ops calls are identical to
// PrefillUnitsImpl with T = 1 — interleaving the blocks across the other query
// heads of the group never touches this head's accumulator, so the fp32
// arithmetic order per output is unchanged. Bit-identical across thread counts
// too (each kv head's whole recurrence runs within one call).
template <typename Ops>
void DecodeUnitsImpl(const DecodeArgs& a, std::int64_t unit_begin,
                     std::int64_t unit_end) {
  const std::int64_t d = a.d;
  const std::int64_t l_dim = a.l_dim;
  const std::int64_t group = a.group;
  const std::int64_t limit = l_dim - 1;  // inclusive causal bound (t = 0)
  float scores[kAttnKb];
  float run_max[kAttnDecodeGroupChunk];
  float run_den[kAttnDecodeGroupChunk];

  for (std::int64_t hk = unit_begin; hk < unit_end; ++hk) {
    const float* k_head = a.k + (hk * l_dim * d);
    const float* v_head = a.v + (hk * l_dim * d);
    const std::int64_t hq_base = hk * group;  // query heads [hq_base, +group)

    // Slice the group so the per-query run state fits the fixed stack arrays;
    // K/V are re-streamed once per slice (never, for g ≤
    // kAttnDecodeGroupChunk).
    for (std::int64_t g0 = 0; g0 < group; g0 += kAttnDecodeGroupChunk) {
      const std::int64_t gn =
          std::min<std::int64_t>(kAttnDecodeGroupChunk, group - g0);
      DecodeGroupSlice<Ops>(a, k_head, v_head, hq_base + g0, gn, limit, scores,
                            run_max, run_den);
    }
  }
}

}  // namespace engine::kernels::internal
