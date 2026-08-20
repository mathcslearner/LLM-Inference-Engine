#pragma once

#include "kernels/internal/attention_common.h"
#include "kernels/internal/exp_common.h"

#include <algorithm>
#include <cstdint>
#include <limits>

// Shared, intrinsic-free core of the paged decode-attention variant (M8-T05;
// design: docs/design/paged-kv-cache.md §9.2). It is the M6-T05 decode
// recurrence (attention_common.h: DecodeUnitsImpl / DecodeGroupSlice) with one
// change — instead of walking a contiguous `k_head + s·d`, it reads K/V
// **through a block table** over the paged slabs (§3.1: two per-layer slabs
// `[num_blocks, Hkv, bs, d]`, one K one V). Everything else — the kv-head
// parallel unit, the key-block-outer/query-inner group loop, the fixed
// `kAttnKb`-wide `scores` row, the running max/denominator rescale, the
// `kAttnDecodeGroupChunk` group slicing, the four `Ops` primitives — is reused
// verbatim so the numeric code is single-sourced with prefill/decode.
//
// **Bit-identity with the contiguous decode kernel (the M8-T05 acceptance).**
// Because `bs | kAttnKb` (§4, enforced at pool construction), one 64-key
// online-softmax unit is exactly `kAttnKb / bs` whole physical blocks. Per
// fixed query head, the unit walk (`kb += kAttnKb`), the per-unit `n_valid`,
// the first-unit `alpha = exp(−inf) = 0`, and the four Ops calls are the
// identical arithmetic in the identical order as `DecodeUnitsImpl`:
//   - `DotScoreRow` scores each key independently (per-key dot·scale into
//     `scores[j]`, then a `>`-fold for the row max), so calling it once per
//     physical block into `scores + b·bs` produces byte-identical
//     `scores[0..n_valid)` and a `>`-folded unit max identical to the single
//     contiguous call.
//   - `ExpRowSum` runs over the SAME contiguous 64-wide `scores` row with the
//     same `n_valid`, so its lane grouping + scalar tail are unchanged.
//   - `ScaleRow` / `AxpyRow` run per key in ascending `j`; only the V-row
//     ADDRESS changes (block-table indirection), never the value or the order.
// So the fp32 accumulation order per output element is unchanged → the result
// is **bit-identical** to `DecodeAttentionF32` on the same logical K/V, and
// bit-identical across thread counts (each kv head's recurrence in one call).

namespace engine::kernels::internal {

// One paged decode-attention call, flattened to raw pointers + dims (design
// §9.2). The slabs are `[num_blocks, Hkv, bs, d]` fp32: block id `blk` begins
// at `blk · block_stride` floats, kv head `hk`'s tile at `+ hk · bs · d`, slot
// `p` within that tile at `+ p · d` (§3.1). `block_table[lb]` is the physical
// block id for this sequence's logical block `lb`.
struct PagedDecodeArgs {
  const float* q;       // [H, d]
  const float* k_slab;  // layer K slab base [num_blocks, Hkv, bs, d]
  const float* v_slab;  // layer V slab base
  const std::int32_t* block_table;  // logical→physical, this sequence
  float* out;                       // [H, d]
  std::int64_t d;                   // head_dim
  std::int64_t length;        // L cached keys (last block possibly partial)
  std::int64_t group;         // H / Hkv (GQA group size)
  std::int64_t block_size;    // bs (divides kAttnKb)
  std::int64_t block_stride;  // Hkv·bs·d — one block id's float span
  float scale;
};

// One paged decode group slice: the `gn` query heads [hq0, hq0 + gn) of the kv
// head whose per-block tiles begin at `k_slab/v_slab + block_table[lb]·stride +
// head_off`. The direct analogue of DecodeGroupSlice with block-table
// addressing; each query head's block loop is identical to that path with the
// contiguous `k_head + s·d` replaced by the paged tiles (the bit-identity
// invariant above). `scores`/`run_max`/`run_den` are caller-owned scratch (a
// fixed kAttnKb score row; per-query state of length ≥ gn).
template <typename Ops>
void PagedDecodeGroupSlice(const PagedDecodeArgs& a, std::int64_t head_off,
                           std::int64_t hq0, std::int64_t gn,
                           std::int64_t limit, float* scores, float* run_max,
                           float* run_den) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  const std::int64_t d = a.d;
  const std::int64_t bs = a.block_size;

  for (std::int64_t gi = 0; gi < gn; ++gi) {
    float* out_vec = a.out + ((hq0 + gi) * d);
    for (std::int64_t e = 0; e < d; ++e) {
      out_vec[e] = 0.0F;  // running weighted-V accumulator
    }
    run_max[gi] = kNegInf;
    run_den[gi] = 0.0F;
  }

  // Stream 64-key units left→right once; every query head of the slice consumes
  // each unit while its blocks are cache-resident. A unit spans kAttnKb/bs
  // whole physical blocks (bs | kAttnKb), gathered into the same 64-wide score
  // row.
  for (std::int64_t kb = 0; kb <= limit; kb += kAttnKb) {
    const std::int64_t kb_len = std::min<std::int64_t>(kAttnKb, a.length - kb);
    const std::int64_t n_valid = std::min<std::int64_t>(limit - kb + 1, kb_len);
    const std::int64_t lb0 = kb / bs;  // first logical block of this unit

    for (std::int64_t gi = 0; gi < gn; ++gi) {
      const float* q_vec = a.q + ((hq0 + gi) * d);
      float* out_vec = a.out + ((hq0 + gi) * d);

      // Score the unit block-by-block into the contiguous scores row (each
      // block's keys land at scores + b·bs); fold the per-block maxes with `>`.
      float block_max = kNegInf;
      for (std::int64_t b = 0; b * bs < n_valid; ++b) {
        const std::int64_t phys = a.block_table[lb0 + b];
        const float* k_tile = a.k_slab + (phys * a.block_stride) + head_off;
        const std::int64_t n_b = std::min<std::int64_t>(bs, n_valid - (b * bs));
        const float bmax =
            Ops::DotScoreRow(q_vec, k_tile, n_b, d, a.scale, scores + (b * bs));
        block_max = bmax > block_max ? bmax : block_max;
      }

      const float m_new = block_max > run_max[gi] ? block_max : run_max[gi];
      const float alpha = ExpF32Scalar(run_max[gi] - m_new);
      const float row_sum = Ops::ExpRowSum(scores, n_valid, m_new);
      run_den[gi] = (run_den[gi] * alpha) + row_sum;
      Ops::ScaleRow(out_vec, alpha, d);

      // Weighted-V accumulate in ascending key order (identical to the
      // contiguous path); the V row for key j is in block lb0 + j/bs at slot
      // j%bs — walked block-outer so the table is read once per block.
      for (std::int64_t b = 0; b * bs < n_valid; ++b) {
        const std::int64_t phys = a.block_table[lb0 + b];
        const float* v_tile = a.v_slab + (phys * a.block_stride) + head_off;
        const std::int64_t n_b = std::min<std::int64_t>(bs, n_valid - (b * bs));
        for (std::int64_t p = 0; p < n_b; ++p) {
          Ops::AxpyRow(out_vec, scores[(b * bs) + p], v_tile + (p * d), d);
        }
      }
      run_max[gi] = m_new;
    }
  }

  for (std::int64_t gi = 0; gi < gn; ++gi) {
    float* out_vec = a.out + ((hq0 + gi) * d);
    Ops::ScaleRow(out_vec, 1.0F / run_den[gi], d);  // run_den > 0 (key 0)
  }
}

// One batched paged decode-attention call (scheduler-runtime.md §8.4, M9-T07).
// B sequences, one query per query head each, decoded together in a single
// forward. All sequences' layer K/V live in the SAME per-layer slabs (they
// share one BlockPool, so `paged_view(layer)` returns identical
// `k_slab`/`v_slab`/`block_size`/`block_stride` for every sequence); only the
// block table and cached length differ per sequence — hence the shared slabs
// plus the per-sequence `block_tables`/`lengths` arrays. `q`/`out` are the
// flattened `[B, H, d]` batch-major blocks (sequence b at `+ b·H·d`). The
// per-sequence recurrence is the **unchanged** `PagedDecodeUnitsImpl` driven on
// a `PagedDecodeArgs` synthesized per sequence, so each sequence's output is
// **bit-identical** to a standalone `PagedDecodeAttentionF32` run on it (the
// M9-T07 acceptance) — the batch only reorders which (sequence, kv-head) unit a
// thread picks up; no cross-sequence value enters a sequence's recurrence.
struct PagedDecodeBatchedArgs {
  const float* q;       // [B, H, d]
  const float* k_slab;  // shared layer K slab base [num_blocks, Hkv, bs, d]
  const float* v_slab;  // shared layer V slab base
  const std::int32_t* const* block_tables;  // [B] logical→physical per sequence
  const std::int64_t* lengths;              // [B] cached keys L_b per sequence
  float* out;                               // [B, H, d]
  std::int64_t num_seqs;                    // B
  std::int64_t heads;                       // H query heads
  std::int64_t kv_heads;                    // Hkv
  std::int64_t d;                           // head_dim
  std::int64_t group;                       // H / Hkv (GQA group size)
  std::int64_t block_size;                  // bs (divides kAttnKb)
  std::int64_t block_stride;                // Hkv·bs·d — one block id's span
  float scale;
};

// The paged decode online-softmax recurrence for kv heads [unit_begin,
// unit_end) (design §9.2, M8-T05). Parallelized over kv heads (the public
// entry), exactly like DecodeUnitsImpl: a unit is one kv head's `group` query
// heads, processed key-block-outer/query-inner so each kv head's blocks stream
// once. The `Ops` policy is the SAME four primitives as prefill/decode
// (attention_impl.h). Bit-identical to DecodeUnitsImpl on the same logical K/V
// (the invariant above) and across thread counts.
template <typename Ops>
void PagedDecodeUnitsImpl(const PagedDecodeArgs& a, std::int64_t unit_begin,
                          std::int64_t unit_end) {
  const std::int64_t group = a.group;
  const std::int64_t limit = a.length - 1;  // inclusive causal bound (t = 0)
  const std::int64_t head_span =
      a.block_size * a.d;  // one kv head within a block
  float scores[kAttnKb];
  float run_max[kAttnDecodeGroupChunk];
  float run_den[kAttnDecodeGroupChunk];

  for (std::int64_t hk = unit_begin; hk < unit_end; ++hk) {
    const std::int64_t head_off = hk * head_span;
    const std::int64_t hq_base = hk * group;  // query heads [hq_base, +group)

    // Slice the group so the per-query run state fits the fixed stack arrays;
    // K/V are re-streamed once per slice (never, for g ≤
    // kAttnDecodeGroupChunk).
    for (std::int64_t g0 = 0; g0 < group; g0 += kAttnDecodeGroupChunk) {
      const std::int64_t gn =
          std::min<std::int64_t>(kAttnDecodeGroupChunk, group - g0);
      PagedDecodeGroupSlice<Ops>(a, head_off, hq_base + g0, gn, limit, scores,
                                 run_max, run_den);
    }
  }
}

}  // namespace engine::kernels::internal
