#pragma once

#include <cstdint>

// KV scatter: write a batch of new K/V vectors into the paged KV slabs at a
// slot mapping (M8-T04; design: docs/design/paged-kv-cache.md §9.1). The paged
// analog of `SimpleKvCache::append`'s transpose-copy — it replaces the
// contiguous append path (§1).
//
// The kernel is **layout-agnostic** (design §2): it never sees `BlockPool` or
// `BlockTable`, only raw fp32 slab base pointers, a `slot_mapping` array (one
// flat slot per new token, computed by `BlockTable::AppendTokens`, §7.1), and
// the geometry. `PagedKvCache` passes the slab pointers down; the physical
// layout stays `kvcache`'s private decision (§2).
//
// For token `t` (in `[0, T)`) and kv head `h`: the source row is
// `src_k + (t·Hkv + h)·d` (token-major `[T, Hkv, d]`, as the projections
// produce it), and the destination is derived from `slot = slot_mapping[t]`:
// physical block `slot / bs`, in-block row `p = slot % bs`, giving
// `k_slab + ((block·Hkv + h)·bs + p)·d` (§3.1). Each (token, head) row is a
// `memcpy` of `d` floats — a pure fp32→fp32 copy, no arithmetic.
//
// Numerics: **Class E (bit-exact)** — a plain copy has no reduction and no
// ISA-specific work, so the result is bit-identical across thread counts AND
// across ISAs. There is therefore **no per-ISA TU** and no dispatch table: a
// single scalar TU ships (like the M6-T06 embedding gather). Tokens are the
// unit of parallel work; distinct tokens (and distinct heads within a token)
// write disjoint destinations, so the threaded scatter is race-free.
//
// Raw-pointer entry: preconditions (non-null pointers; contiguous fp32 `src_k`,
// `src_v` `[T, Hkv, d]`; every `slot_mapping[t]` naming a distinct, in-range
// slot in the slabs) are the CALLER's to enforce — `PagedKvCache::append`
// front-loads all recoverable validation, so nothing here does anything but
// move bytes (ADR-003; cpu-backend.md §5). Slots must be distinct across the
// batch (they are, being consecutive sequence positions), so the disjoint-write
// guarantee holds.

namespace engine::kernels {

// Writes `T` new tokens' K and V into the paged slabs at `slot_mapping`.
//   src_k, src_v : [T, Hkv, d] token-major, contiguous, fp32.
//   slot_mapping : [T]         one flat slot per new token (§7.1).
//   k_slab, v_slab: the layer's K and V slab bases (`[num_blocks, Hkv, bs,
//   d]`).
// `bs` is the pool's block size. Pure copy; bit-exact; thread-safe (disjoint
// slots). `T == 0` is a no-op.
void KvScatterF32(const float* src_k, const float* src_v,
                  const std::int64_t* slot_mapping, std::int64_t t_dim,
                  std::int64_t kv_heads, std::int64_t d,
                  std::int64_t block_size, float* k_slab, float* v_slab);

}  // namespace engine::kernels
