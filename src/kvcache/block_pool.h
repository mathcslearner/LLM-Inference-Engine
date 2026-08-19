#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "memory/allocator.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

// BlockPool — the paged KV cache's block allocator (M8-T02; design:
// docs/design/paged-kv-cache.md §6). Pure bookkeeping over pre-allocated
// per-layer K/V slabs — fully unit-testable without a model or any kernel.
//
// One pool is shared by all sequences (M9); in the M8 single-request path the
// engine owns exactly one. The pool holds `num_blocks` fixed-size physical
// blocks; each block stores `block_size` (`bs`) contiguous token slots. For
// **each layer** the pool keeps **two slabs** (one K, one V), each shaped
// `[num_blocks, Hkv, bs, d]` fp32 contiguous (§3.1), so the pool owns `2·L`
// tensors. A physical block id names the same slot region in every layer's
// slabs; a sequence's block table (M8-T03) is layer-independent.
//
// "Backed by the M2 caching allocator" (ROADMAP) means the pool's *slabs* come
// from the passed `Allocator` — one allocation per slab at construction. The
// per-block free list is the KV block allocator itself and never calls upstream
// on the token hot path (§6.1).
//
// Refcounts live in the pool from day one so M11 prefix sharing is an
// extension, not a rewrite: a block's lifecycle is FREE (rc 0, on the free
// list) → OWNED (rc 1, `Allocate`) → SHARED (rc ≥ 2, `Share`), and `Release`
// walks it back (§6.3). The immutability invariant M11 rests on — a block with
// rc ≥ 1 is never rewritten in place because sequences only append to their
// exclusive rc-1 tail — is stated in §6.4.
//
// Layering (ADR-002 Amendment 5): `kvcache` links `tensor`/`memory`/`core`, and
// — as of M8-T04 — the downward `kvcache → kernels` edge (the paged
// scatter/decode kernels). This header takes `memory::Allocator*`, so
// `engine::memory` is a PUBLIC link dependency of `engine_kvcache`; `BlockPool`
// itself links no kernel (it is pure bookkeeping).

namespace engine::kvcache {

// Pool occupancy snapshot (design §6.2). `used + free == total` always.
struct BlockPoolStats {
  std::int64_t total = 0;    // num_blocks
  std::int64_t used = 0;     // blocks with refcount >= 1
  std::int64_t free = 0;     // blocks on the free list
  double utilization = 0.0;  // used / total (0 when total == 0)
};

class BlockPool {
 public:
  // Every slab is allocated at this alignment (==
  // CachingAllocator::kMaxAlignment, the pool-wide alignment the M2 caching
  // allocator can satisfy — §6.1). Fixing it here keeps the pool independent of
  // any one allocator's default.
  static constexpr std::size_t kSlabAlignment = 256;

  // --- Capacity arithmetic (design §5.2; pure, static, no allocation) ---

  // Bytes one physical block occupies across all layers:
  //   2 · num_layers · Hkv · bs · head_dim · itemsize   (the `2` = K and V).
  // Validates the geometry (positive dims, fp32 until M13) and `block_size`
  // (§4); overflow → InvalidArgument.
  [[nodiscard]] static core::StatusOr<std::int64_t> BytesPerBlock(
      const CacheGeometry& geom, int block_size);

  // Blocks that fit in `kv_budget_bytes`: ⌊budget / BytesPerBlock⌋. A result
  // < 1 (the model does not fit the budget) → ResourceExhausted; a non-positive
  // budget → InvalidArgument.
  [[nodiscard]] static core::StatusOr<std::int64_t> NumBlocksForBudget(
      const CacheGeometry& geom, int block_size, std::int64_t kv_budget_bytes);

  // --- Construction ---

  // Allocates the `2·L` slabs for `geom`, `block_size`, `num_blocks` from
  // `allocator` (null → the process default CPU allocator; the engine passes
  // the M2 CachingAllocator). Slabs are 256-byte aligned and zero-filled once
  // (§6.1: a partial block's unused slots read 0.0, so masked keys contribute
  // exactly 0.0 and the KV invariant stays bit-exact; also makes RSS resident
  // up front). Geometry must be positive and fp32 (non-f32 → Unimplemented, the
  // M13 INT8 seam); `block_size ∈ {8,16,32,64}` (§4 divisibility constraint);
  // `num_blocks ≥ 1`. Allocation failure propagates (kOutOfMemory /
  // kResourceExhausted). Logs one line with the pool's byte footprint.
  [[nodiscard]] static core::StatusOr<BlockPool> Create(
      CacheGeometry geom, int block_size, std::int64_t num_blocks,
      memory::Allocator* allocator);

  // Move-only, and movable only before any block is handed out
  // (CHECK(used_ == 0)): a BlockTable (M8-T03) holds a raw BlockPool*, so a
  // move after handout would dangle. `Create` move-constructs out before
  // returning; thereafter the pool stays put.
  BlockPool(BlockPool&& other) noexcept;
  BlockPool& operator=(BlockPool&&) = delete;
  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;
  ~BlockPool() = default;

  // --- Block lifecycle (thread-safe; off the token hot path — §6.2) ---

  // Allocate one fresh block: pop the free list (LIFO — reuses recently-freed,
  // cache-warm blocks), set refcount 1, return its id. Empty free list →
  // ResourceExhausted (M8-T08 posture; M11 eviction retries before this fires).
  [[nodiscard]] core::StatusOr<std::int32_t> Allocate();

  // Bump refcount — M11 prefix sharing adopts an existing block. CHECK on a
  // free (refcount-0) or out-of-range block: sharing what nobody owns is a
  // programmer error.
  void Share(std::int32_t block);

  // Drop one reference. refcount 1→0 returns the block to the free list.
  // Releasing a refcount-0 (or out-of-range) block is the "double-free" the
  // acceptance names → CHECK (programmer error, not recoverable).
  void Release(std::int32_t block);

  // --- Stats / queries ---

  [[nodiscard]] std::int32_t refcount(std::int32_t block) const;
  [[nodiscard]] BlockPoolStats stats() const;
  [[nodiscard]] std::int64_t free_blocks() const;  // M9 admission

  // Blocks a sequence at `cur_tokens` needs to admit `add_tokens` more —
  // boundary-crossing count ⌈(cur+add)/bs⌉ − ⌈cur/bs⌉. Pure arithmetic (M9
  // scheduler admission, M8-T03 append). `cur_tokens`/`add_tokens` must be
  // ≥ 0 (CHECK).
  [[nodiscard]] std::int64_t blocks_needed(std::int64_t cur_tokens,
                                           std::int64_t add_tokens) const;

  // --- Kernel-facing slab pointers + strides (immutable post-construction; no
  // lock). Element (float) strides, matching §3.1. Used by PagedKvCache and the
  // gather (M8-T04/T06); never by a kernel directly — kvcache passes them down.
  [[nodiscard]] float* k_slab(int layer);
  [[nodiscard]] float* v_slab(int layer);
  // Hkv·bs·d — one block id advances this many floats.
  [[nodiscard]] std::int64_t block_stride() const { return block_stride_; }
  // bs·d — one kv head within a block.
  [[nodiscard]] std::int64_t head_stride() const { return head_stride_; }
  // d — one token slot within a (block, head) tile.
  [[nodiscard]] std::int64_t row_stride() const { return row_stride_; }

  // --- Geometry / sizing ---

  [[nodiscard]] const CacheGeometry& geometry() const { return geom_; }
  [[nodiscard]] int block_size() const { return block_size_; }
  [[nodiscard]] std::int64_t num_blocks() const { return num_blocks_; }
  // Bytes in one K (or V) slab for one layer.
  [[nodiscard]] std::int64_t slab_bytes() const { return slab_bytes_; }
  // Total pool footprint: 2·L slabs (for the M8-T07 stats log).
  [[nodiscard]] std::int64_t total_bytes() const {
    return slab_bytes_ * 2 * geom_.num_layers;
  }

 private:
  BlockPool(CacheGeometry geom, int block_size, std::int64_t num_blocks,
            std::vector<tensor::Tensor> k_slabs,
            std::vector<tensor::Tensor> v_slabs, std::int64_t slab_bytes);

  [[nodiscard]] bool block_in_range(std::int32_t block) const {
    return block >= 0 && static_cast<std::int64_t>(block) < num_blocks_;
  }

  CacheGeometry geom_;
  int block_size_ = 0;
  std::int64_t num_blocks_ = 0;
  std::int64_t slab_bytes_ = 0;
  std::int64_t block_stride_ = 0;  // Hkv·bs·d
  std::int64_t head_stride_ = 0;   // bs·d
  std::int64_t row_stride_ = 0;    // d

  // `[num_blocks, Hkv, bs, d]` fp32 contiguous, one per layer for K and V.
  std::vector<tensor::Tensor> k_slabs_;
  std::vector<tensor::Tensor> v_slabs_;

  // Bookkeeping, guarded by `mutex_` (unique_ptr so BlockPool stays movable —
  // std::mutex is not).
  std::unique_ptr<std::mutex> mutex_;
  std::vector<std::int32_t> refcount_;   // per-block refcount
  std::vector<std::int32_t> free_list_;  // stack of free block ids (LIFO)
  std::int64_t used_ = 0;                // blocks with refcount >= 1
};

}  // namespace engine::kvcache
