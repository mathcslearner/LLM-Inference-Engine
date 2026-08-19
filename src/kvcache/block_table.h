#pragma once

#include "core/status.h"
#include "kvcache/block_pool.h"

#include <cstdint>
#include <span>
#include <vector>

// BlockTable — a single sequence's logical→physical block map (M8-T03; design:
// docs/design/paged-kv-cache.md §7). One table drives all layers (a physical
// block id names the same slot region in every layer's slabs, §3.1), so a
// `PagedKvCache` (M8-T04) owns exactly one table for the sequence it holds.
//
// Logical block `i` covers token positions `[i·bs, (i+1)·bs)` (`bs` = the
// pool's block size). Token at logical position `pos` lives in logical block
// `pos / bs`, in-block offset `pos % bs`, physical block `blocks_[pos / bs]`.
// Its **flat slot** — the addressing unit the M8-T04 scatter kernel consumes —
// is `slot(pos) = blocks_[pos / bs]·bs + (pos % bs)` (§7.1), a single int64
// naming "which physical block, which row within it."
//
// The table owns its blocks: `AppendTokens` `Allocate()`s from the pool on
// block-boundary crossings, and every owned block is `Release()`d on
// `Truncate`/`FreeAll`/destruction (RAII — a dropped `PagedKvCache` returns its
// blocks; the basis for M8-T08 "no leaked blocks, stats zero at end"). In M8
// every owned block has refcount exactly 1: sequences only append to their
// exclusive tail (the §6.4 immutability invariant), so `Release` is the only
// pool verb the table uses; `Share` stays unused until M11 prefix caching.
//
// Not thread-safe: one table per sequence, driven by the single engine thread.
// The cross-sequence contention (many tables sharing one pool) is covered by
// `BlockPool`'s own mutex, not here.
//
// Layering: `block_table.h` lives in the layer-2 `kvcache` module and adds no
// new link edge — it calls only `BlockPool` (same module). The `kvcache →
// kernels` downward edge (the paged scatter/decode kernels) is added by
// `PagedKvCache` in M8-T04.

namespace engine::kvcache {

class BlockTable {
 public:
  // `pool` is non-owning and must outlive the table (§7.1, §10.1). CHECK on a
  // null pool — a table without storage is a programmer error.
  explicit BlockTable(BlockPool* pool);

  // RAII: releases every owned block back to the pool.
  ~BlockTable();

  // Move-only, mirroring BlockPool. The moved-from table is left empty (no
  // owned blocks, zero tokens) so its destruction releases nothing. Move-assign
  // is deleted to avoid the release-then-steal ordering subtlety — a table is
  // constructed in place, never reseated.
  BlockTable(BlockTable&& other) noexcept;
  BlockTable& operator=(BlockTable&&) = delete;
  BlockTable(const BlockTable&) = delete;
  BlockTable& operator=(const BlockTable&) = delete;

  // Grow the sequence by `count` tokens at positions `[num_tokens_,
  // num_tokens_ + count)`; returns the `slot_mapping[count]` array (one flat
  // slot per new token, §7.1) the M8-T04 scatter kernel consumes.
  //
  // All-or-nothing (§7.2): `Allocate()`s the `blocks_needed` new physical
  // blocks into a scratch vector; if any allocation returns ResourceExhausted,
  // releases the ones already taken and returns ResourceExhausted — table and
  // pool are left exactly as before. `count <= 0` → InvalidArgument (a forward
  // always appends at least one token).
  [[nodiscard]] core::StatusOr<std::vector<std::int64_t>> AppendTokens(
      std::int64_t count);

  // Drop tokens past `new_len` ∈ [0, num_tokens_]. Blocks that become wholly
  // empty are released (tail-first) and popped; the partial surviving tail
  // block is kept — its stale slots are overwritten on the next append and read
  // masked-out until then (§7.3). Out-of-range `new_len` → InvalidArgument,
  // state untouched. Under M8's exclusive-tail rule no released block is
  // shared, so no copy-on-write (deferred to M11/M15, §6.4).
  [[nodiscard]] core::Status Truncate(std::int64_t new_len);

  // Release every owned block and reset to empty. Never fails; the destructor
  // calls it.
  void FreeAll();

  // The flat slot for a committed position (§7.1). CHECK(0 <= pos <
  // num_tokens_): querying an uncommitted position is a programmer error.
  [[nodiscard]] std::int64_t slot(std::int64_t pos) const;

  [[nodiscard]] std::int64_t num_tokens() const { return num_tokens_; }

  // Physical blocks owned, in logical order == ⌈num_tokens_ / bs⌉ (invariant).
  [[nodiscard]] std::int64_t num_blocks() const {
    return static_cast<std::int64_t>(blocks_.size());
  }

  // The logical→physical block ids, contiguous, in logical order. This is the
  // `const int32_t* block_table` the M8-T05 PagedDecodeAttentionF32 reads
  // (§8.3, §9.2) — a zero-copy view, valid until the next append/truncate.
  [[nodiscard]] std::span<const std::int32_t> blocks() const { return blocks_; }

  [[nodiscard]] int block_size() const { return pool_->block_size(); }
  [[nodiscard]] BlockPool* pool() const { return pool_; }

 private:
  BlockPool* pool_;                   // non-owning; outlives the table
  std::vector<std::int32_t> blocks_;  // logical block i → physical block id
  std::int64_t num_tokens_ = 0;       // committed tokens in this sequence
};

}  // namespace engine::kvcache
