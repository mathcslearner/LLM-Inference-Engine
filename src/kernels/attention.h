#pragma once

#include <cstdint>

// Optimized prefill attention (M6-T04; design:
// docs/design/optimized-cpu-execution.md §8, §10). The blocked, flash-style
// analog of `cpu::attention` (the oracle): causal GQA self-attention with an
// online (running max/sum) softmax, so no `[H·T, L]` score matrix is
// materialized — only a `kAttnKb`-wide score row per query at a time.
//
// Contract (identical to `cpu::attention`, so tests are 1:1):
//   q    : [T, H, d]  contiguous fp32, token-major (row t, head h at
//          q + (t·H + h)·d).
//   k, v : [Hkv, L, d] contiguous fp32, head-major (L = P + T cached+new keys;
//          kv head hk at k + hk·L·d, key s at +s·d).
//   out  : [T, H, d]  contiguous fp32, caller-allocated, fully overwritten.
//   scale: multiplies the completed q·k dot (HF order: matmul then scale).
//
// GQA by KV-head indexing with **no materialized repeat**: with g = H / Hkv,
// query head h reads kv head h / g. Causal mask: new query t sits at absolute
// position P + t and attends key positions [0, P + t] inclusive (P = L − T).
// Supports prefill continuing from a non-empty cache (P > 0) — it is a (P, T)
// choice of the same kernel.
//
// Numerics (§10): fp32 throughout; each output row's online-softmax recurrence
// runs entirely within one thread, so the result is **bit-identical across
// thread counts**. The online rescale order and the vector `exp` polynomial
// make it **Class T** across ISAs and vs the oracle (tolerance
// rtol 1e-4, atol 1e-5, §10) — not bitwise.
//
// Raw-pointer entry: all preconditions are the CALLER's (non-null contiguous
// fp32 operands; T, H, Hkv, d ≥ 1; H a multiple of Hkv; L ≥ T). The
// reference-vs-optimized test and the OptimizedModel caller front-load
// validation, so nothing inside a parallel region does anything but arithmetic
// (design §5). `out` must not alias q/k/v.

namespace engine::kernels {

// out[T, H, d] = softmax(scale · Q·Kᵀ + causal_mask) · V, blocked/online.
void PrefillAttentionF32(const float* q, const float* k, const float* v,
                         float* out, std::int64_t t_dim, std::int64_t heads,
                         std::int64_t kv_heads, std::int64_t d,
                         std::int64_t l_dim, float scale);

// Optimized decode attention (M6-T05; design: optimized-cpu-execution.md §8,
// §10). The single-token specialization of `PrefillAttentionF32`: one query per
// head attends the whole cache. It is exactly `PrefillAttentionF32` with T = 1
// (a `q`/`out` shaped `[1, H, d]` is `[H, d]`), reorganized to stream each kv
// head's K/V from memory once for all g = H / Hkv query heads of the group.
//
// Contract (the T = 1 case of the prefill contract, so tests are 1:1):
//   q    : [H, d]     contiguous fp32 (query head h at q + h·d).
//   k, v : [Hkv, L, d] contiguous fp32, head-major (L cached+new keys;
//          kv head hk at k + hk·L·d, key s at +s·d).
//   out  : [H, d]     contiguous fp32, caller-allocated, fully overwritten.
//   scale: multiplies the completed q·k dot (HF order: matmul then scale).
//
// The single query sits at absolute position L − 1 and attends key positions
// [0, L − 1] inclusive (every key — no causal masking within the cache).
// GQA by KV-head indexing with **no materialized repeat**: query head h reads
// kv head h / g. Threaded across kv heads (design §5): each kv head's whole
// recurrence runs within one thread, so the result is **bit-identical across
// thread counts**, and — by construction — **bit-identical to
// `PrefillAttentionF32` called with T = 1** on the same q/k/v (M6-T05
// acceptance; the arithmetic order per output head is unchanged). Class T vs
// the oracle and across ISAs (online rescale + vector `exp`, §10).
//
// Raw-pointer entry: all preconditions are the CALLER's (non-null contiguous
// fp32 operands; H, Hkv, d, L ≥ 1; H a multiple of Hkv). `out` must not alias
// q/k/v.
void DecodeAttentionF32(const float* q, const float* k, const float* v,
                        float* out, std::int64_t heads, std::int64_t kv_heads,
                        std::int64_t d, std::int64_t l_dim, float scale);

// Varlen (ragged-batch) prefill attention (M9-T06; design:
// docs/design/scheduler-runtime.md §8.4). Runs `PrefillAttentionF32` for each
// sequence of a ragged batch delimited by `cu_seqlens`, sharing the
// **unchanged** per-sequence recurrence — "shared kernels loop over sequences;
// still naive-but-correct" (roadmap). Each sequence's K/V is its own gathered
// head-major slab (the M8-T06 `cache.view(layer)` read path, §8.3), so K/V come
// in as per-sequence pointer arrays rather than one concatenated slab.
//
// Contract:
//   q       : [ΣT, H, d]   contiguous fp32, token-major. Sequence b's queries
//             are rows [cu_seqlens[b], cu_seqlens[b + 1]).
//   cu_seqlens: [B + 1] int32 prefix sums of the per-sequence query counts T_b
//             (cu_seqlens[0] == 0; T_b = cu_seqlens[b + 1] − cu_seqlens[b] ≥
//             1).
//   k_seqs,
//   v_seqs  : [B] arrays of pointers; k_seqs[b]/v_seqs[b] is that sequence's
//             [Hkv, L_b, d] head-major slab (L_b cached+new keys).
//   l_dims  : [B] the L_b = P_b + T_b per sequence (P_b = L_b − T_b past keys).
//   out     : [ΣT, H, d]   contiguous fp32, caller-allocated, fully
//   overwritten. scale   : multiplies the completed q·k dot (HF order).
//
// GQA, causal masking, and continuation-from-a-non-empty-cache (P_b > 0) are
// all exactly the standalone `PrefillAttentionF32` behavior applied per
// sequence.
// **Each sequence's output is bit-identical to a standalone
// `PrefillAttentionF32` run on that sequence** (M9-T06 acceptance) — the batch
// only reorders which (head, query-block) unit a thread picks up; no
// cross-sequence value ever enters a sequence's recurrence. Bit-identical
// across thread counts; Class T vs the oracle and across ISAs (inherited from
// the per-sequence kernel, §10).
//
// Raw-pointer entry: all preconditions are the CALLER's (non-null contiguous
// fp32 operands; B, H, Hkv, d ≥ 1; H a multiple of Hkv; T_b ≥ 1; L_b ≥ T_b;
// cu_seqlens monotone from 0). B == 0 is a no-op. `out` must not alias q/k/v.
void PrefillAttentionVarlenF32(const float* q, const std::int32_t* cu_seqlens,
                               std::int64_t num_seqs,
                               const float* const* k_seqs,
                               const float* const* v_seqs,
                               const std::int64_t* l_dims, float* out,
                               std::int64_t heads, std::int64_t kv_heads,
                               std::int64_t d, float scale);

}  // namespace engine::kernels
