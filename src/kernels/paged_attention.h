#pragma once

#include <cstdint>

// Optimized paged decode attention (M8-T05; design:
// docs/design/paged-kv-cache.md §9.2). The paged analogue of
// `DecodeAttentionF32` (kernels/attention.h): one query per head attends the
// whole cache, but K/V are read **through a block table** over the paged slabs
// instead of a contiguous `[Hkv, L, d]` gather — the decode fast path
// `PagedKvCache::paged_view` feeds (kv_cache.h §8.3), so decode never pays the
// per-step full-history copy `view()` would.
//
// Contract:
//   q          : [H, d]  contiguous fp32 (query head h at q + h·d).
//   k_slab,    : layer K/V slab bases, each `[num_blocks, Hkv, bs, d]` fp32,
//   v_slab       head-major block tiles (§3.1): block id `blk` at
//                `blk·block_stride`, kv head `hk` at `+ hk·bs·d`, slot `p` at
//                `+ p·d`.
//   block_table: [≥ ceil(length/bs)] int32, logical→physical for this sequence.
//   num_blocks : valid entries in `block_table` (a bound; unused tail entries
//                and unused physical blocks are never read).
//   length     : L cached keys; the single query sits at position L−1 and
//                attends [0, L−1] inclusive (no causal mask within the cache).
//   out        : [H, d]  contiguous fp32, caller-allocated, fully overwritten.
//   block_size : bs — a power of two dividing kAttnKb = 64 (pool-enforced).
//   block_stride: Hkv·bs·d — one block id's float span.
//   scale      : multiplies the completed q·k dot (HF order: matmul then
//   scale).
//
// GQA by KV-head indexing with **no materialized repeat**: query head h reads
// kv head h / g (g = H / Hkv). Threaded across kv heads (design §5, like
// `DecodeAttentionF32`): each kv head's whole recurrence runs within one
// thread, so the result is **bit-identical across thread counts** — and, by
// construction (`bs | kAttnKb`, §4), **bit-identical to `DecodeAttentionF32`**
// on the same logical K/V materialized contiguously (the M8-T05 acceptance,
// asserted bitwise). Class T vs the `cpu::attention` oracle and across ISAs
// (online rescale + vector `exp`, §10 of optimized-cpu-execution.md).
//
// Raw-pointer entry: all preconditions are the CALLER's (`PagedKvCache` /
// `OptimizedModel` front-load validation, design §5) — non-null contiguous fp32
// operands; H, Hkv, d, L ≥ 1; H a multiple of Hkv (group = H / Hkv);
// `ceil(length/bs) ≤ num_blocks`. `out` must not alias q/k/v. `block_size`
// dividing kAttnKb is CHECKed at the entry (the bit-identity invariant is a
// programmer contract, not a per-request input).

namespace engine::kernels {

// out[H, d] = softmax(scale · q·Kᵀ) · V over the whole paged cache, blocked and
// online, reading K/V through `block_table`.
void PagedDecodeAttentionF32(const float* q, const float* k_slab,
                             const float* v_slab,
                             const std::int32_t* block_table,
                             std::int64_t num_blocks, std::int64_t length,
                             std::int64_t heads, std::int64_t kv_heads,
                             std::int64_t d, std::int64_t block_size,
                             std::int64_t block_stride, float scale,
                             float* out);

}  // namespace engine::kernels
