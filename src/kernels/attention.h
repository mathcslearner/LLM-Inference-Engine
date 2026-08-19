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

}  // namespace engine::kernels
