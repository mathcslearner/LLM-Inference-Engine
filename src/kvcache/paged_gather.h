#pragma once

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "kvcache/block_table.h"
#include "kvcache/kv_cache.h"

#include <cstdint>

// `paged_gather` (M8-T06; design: docs/design/paged-kv-cache.md §9.3) — the
// prefill read path for the paged cache. `GatherLayerKV` walks a sequence's
// block table and copies one layer's first `length` tokens of K/V into fresh
// contiguous head-major `[Hkv, length, d]` tensors — exactly the layout the
// **unchanged** M6 `PrefillAttentionF32` reads (kv_cache.h `KvView`). This *is*
// the body of `PagedKvCache::view` (paged_cache.cpp); prefill and
// prefill-continuation (`P > 0` cached tokens + new tokens) gather once per
// layer and call the M6 blocked prefill kernel. Simple and correct; a
// block-walking prefill kernel that skips the gather is deferred to M12.
//
// Layering (ADR-002): `paged_gather` lives in `kvcache` and calls the layer-1
// `KvGatherF32` kernel over the `kvcache → kernels` downward edge (added
// M8-T04, PRIVATE — this header stays kernels-free, the call lives in the
// .cpp), exactly as `PagedKvCache::append` calls `KvScatterF32`.

namespace engine::kvcache {

// Gathers `layer`'s first `length` tokens of paged K/V into fresh contiguous
// head-major `[Hkv, length, d]` tensors (the `KvView` the M6 prefill kernel
// reads). `pool` supplies the slab bases + block size; `table` supplies the
// logical→physical block ids.
//
// Validation is front-loaded (recoverable, ADR-003): out-of-range `layer` →
// `InvalidArgument`; `length` outside `[0, table.num_tokens()]` →
// `InvalidArgument`. `length == 0` yields empty `[Hkv, 0, d]` views (matching
// `SimpleKvCache::view` on an empty cache). Allocation failure propagates from
// `Tensor::empty`. On success every element of both outputs is written (no
// stale/poison bytes are exposed).
//
// `pool` is taken by non-const reference because the slab accessors
// (`k_slab`/`v_slab`) are non-const; the gather only reads through them.
[[nodiscard]] core::StatusOr<KvView> GatherLayerKV(BlockPool& pool,
                                                   const BlockTable& table,
                                                   int layer,
                                                   std::int64_t length);

}  // namespace engine::kvcache
