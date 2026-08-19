#pragma once

#include <cstdint>

// KV gather: read one layer's paged K/V history back into a contiguous
// head-major `[Hkv, length, d]` slab (M8-T06; design:
// docs/design/paged-kv-cache.md §9.3). The paged read counterpart of
// `KvScatterF32` (kv_scatter.h) — it is the body of `PagedKvCache::view` and
// the prefill read path: prefill (T > 1, including prefill-continuation)
// gathers each layer's cached K/V once and feeds the **unchanged** M6 blocked
// prefill kernel (`PrefillAttentionF32`), which reads exactly this `[Hkv,
// length, d]` contiguous layout (kv_cache.h `KvView`). A block-walking prefill
// kernel that avoids the gather is deferred to M12.
//
// Like the scatter, the kernel is **layout-agnostic** (design §2): it never
// sees `BlockPool`/`BlockTable`, only raw fp32 slab base pointers, an
// `int32_t*` block-id array, and the geometry/strides. `paged_gather`
// (paged_gather.h) passes the slab pointers down; the physical layout stays
// `kvcache`'s private decision.
//
// Layout. The slabs are `[num_blocks, Hkv, bs, d]` (§3.1). For logical block
// `b` (in `[0, ceil(length/bs))`) the physical block is `block_table[b]`; its
// tile for kv head `h` starts at `slab + block_table[b]·(Hkv·bs·d) + h·(bs·d)`
// and holds `bs` rows of `d` floats. The gather copies the first `rows` rows of
// each block — `rows = min(bs, length − b·bs)`, so the last block is clipped to
// `length % bs` (0 → a full last block) — into the head-major output at
// `out + h·(length·d) + (b·bs)·d`. Blocks and rows past `length` are never
// touched (the caller may leave them uninitialized/poisoned).
//
// Numerics: **Class E (bit-exact)** — a plain fp32→fp32 copy, no reduction and
// no ISA-specific work, so the result is bit-identical across thread counts AND
// across ISAs. There is therefore **no per-ISA TU** and no dispatch table: a
// single scalar TU ships (like `KvScatterF32` and the M6-T06 embedding gather),
// and the forced-scalar pass needs no separate registration. The unit of
// parallel work is `(kv head, logical block)`; distinct units write disjoint
// destination ranges, so the threaded gather is race-free.
//
// Raw-pointer entry: preconditions (non-null pointers; a caller-allocated
// `[Hkv, length, d]` contiguous fp32 output; every referenced `block_table[b]`
// naming an in-range physical block) are the CALLER's to enforce —
// `GatherLayerKV` front-loads all recoverable validation, so nothing here does
// anything but move bytes (ADR-003; cpu-backend.md §5).

namespace engine::kernels {

// Gathers `length` tokens of one layer's paged K and V into contiguous
// head-major outputs.
//   k_slab, v_slab : the layer's K and V slab bases (`[num_blocks, Hkv, bs,
//                    d]`, fp32).
//   block_table    : logical→physical block ids; at least `ceil(length/bs)`
//                    valid entries.
//   out_k, out_v   : `[Hkv, length, d]` contiguous fp32 (caller-allocated).
// `bs` is the pool's block size. Pure copy; bit-exact; thread-safe (disjoint
// destinations). `length == 0` is a no-op.
void KvGatherF32(const float* k_slab, const float* v_slab,
                 const std::int32_t* block_table, std::int64_t block_size,
                 std::int64_t length, std::int64_t kv_heads, std::int64_t d,
                 float* out_k, float* out_v);

}  // namespace engine::kernels
