#pragma once

#include "core/status.h"
#include "kvcache/block_pool.h"
#include "kvcache/block_table.h"
#include "kvcache/kv_cache.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <vector>

// `PagedKvCache` — the paged KV cache (M8-T04/T07; design:
// docs/design/paged-kv-cache.md §8). A second implementation of the abstract
// `KvCache` interface (kv_cache.h, M5-T05), replacing `SimpleKvCache`'s
// per-sequence contiguous storage with a shared block pool + a per-sequence
// block table. `Attention` / `OptimizedModel` / `Generate` consume it exactly
// as they consume `SimpleKvCache` (the interface is unchanged but for the one
// additive `paged_view` fast path, §8.3).
//
// It holds a `BlockPool*` (shared across sequences, non-owning, outlives the
// cache — the allocator lifetime rule, §6.1/§10.1) and one `BlockTable` (the
// sequence's logical→physical map).
//
// `append(layer, k, v)` under paging (§8.2): the block table is
// layer-independent, but `append` is called once per layer per forward in layer
// order. **Layer 0 grows the table** (all-or-nothing `AppendTokens`, computing
// the batch's slot mapping); layers 1..L−1 **reuse** that slot mapping. So the
// table advances atomically per forward-token-batch, a layer-0
// `ResourceExhausted` aborts the forward before any K/V is written, and layers
// never disagree. The K/V bytes are written by the `KvScatterF32` kernel
// (kernels/kv_scatter.h) — the paged replacement for v0's transpose-copy.
//
// `view(layer)` (the contiguous gather that feeds M6 prefill) lands with the
// paged gather in **M8-T06**; until then it returns `Unimplemented`. The decode
// path does not use it — it reads through `paged_view` (§8.3) with zero copy.
//
// Layering (ADR-002): `kvcache` links `tensor`/`memory`/`core`, and — new in
// M8-T04 — the downward `kvcache → kernels` edge (the scatter kernel). This
// header stays kernels-free (the kernel call lives in the .cpp), so the link is
// PRIVATE, exactly like `model → kernels` via `PackedLinear`.

namespace engine::kvcache {

class PagedKvCache final : public KvCache {
 public:
  // `pool` is non-owning and must outlive the cache (§6.1, §10.1). CHECK on a
  // null pool — a cache without storage is a programmer error. The pool fixes
  // the geometry and block size; construction cannot fail (no allocation here —
  // the slabs were allocated when the pool was created).
  explicit PagedKvCache(BlockPool* pool);

  // Move-only, mirroring `BlockTable`: move-construct (the moved-from cache is
  // left empty, its block table releasing nothing), move-assign deleted (a
  // cache is built in place, never reseated).
  PagedKvCache(PagedKvCache&& other) noexcept = default;
  PagedKvCache& operator=(PagedKvCache&&) = delete;
  PagedKvCache(const PagedKvCache&) = delete;
  PagedKvCache& operator=(const PagedKvCache&) = delete;
  ~PagedKvCache() override = default;

  [[nodiscard]] CacheGeometry geometry() const override {
    return pool_->geometry();
  }

  // Committed tokens: the sequence length. One block table drives all layers,
  // so all layers agree by construction after a completed forward (§8.1).
  [[nodiscard]] std::int64_t length() const override {
    return table_.num_tokens();
  }

  // **Advisory** under paging (§8.1): what this sequence could grow into given
  // the pool's *current* free blocks — its already-owned slot capacity
  // (`num_blocks·bs`, which includes the partial tail block's free slots) plus
  // the free pool (`free_blocks·bs`). With a shared pool another sequence may
  // consume free blocks first, so `append`'s `ResourceExhausted` (M8-T08) is
  // the binding capacity check; `Generate`'s up-front check becomes
  // best-effort.
  [[nodiscard]] std::int64_t capacity() const override;

  // Appends `k`, `v` (`[T, Hkv, d]`, token-major, contiguous, f32, `T >= 1`) to
  // `layer` (§8.2). Front-loaded validation (layer range, rank/shape/dtype/
  // contiguity, k.T == v.T) precedes any mutation, so a rejected append leaves
  // the cache untouched. The caller must append in layer order, once per layer
  // per forward: layer 0 first (grows the table), then 1..L−1 (reuse the slot
  // mapping). An out-of-order/repeated/short append is `InvalidArgument`; a
  // layer-0 pool exhaustion is `ResourceExhausted` (nothing written).
  [[nodiscard]] core::Status append(int layer, const tensor::Tensor& k,
                                    const tensor::Tensor& v) override;

  // Contiguous gather of `layer`'s `[Hkv, len, d]` history — the M6 prefill
  // read path. Lands with the paged gather in M8-T06; returns `Unimplemented`
  // here.
  [[nodiscard]] core::StatusOr<KvView> view(int layer) const override;

  // Zero-copy paged view of `layer` (the decode fast path, §8.3). Out-of-range
  // `layer` → `InvalidArgument`.
  [[nodiscard]] core::StatusOr<PagedKvView> paged_view(
      int layer) const override;

  // Drops tokens past `new_length` (§7.3, via `BlockTable::Truncate`),
  // releasing wholly-emptied blocks and resetting any in-progress forward's
  // pending state. `truncate(0)` == `reset()`. Out-of-range `new_length` →
  // `InvalidArgument`.
  [[nodiscard]] core::Status truncate(std::int64_t new_length) override;

  // Test/inspection accessors.
  [[nodiscard]] const BlockTable& block_table() const { return table_; }
  [[nodiscard]] BlockPool* pool() const { return pool_; }

 private:
  [[nodiscard]] bool layer_in_range(int layer) const {
    return layer >= 0 && layer < pool_->geometry().num_layers;
  }

  BlockPool* pool_;   // non-owning; outlives the cache (§6.1)
  BlockTable table_;  // this sequence's logical→physical map

  // Per-forward pending state (§8.2): the slot mapping computed on the layer-0
  // append and reused for layers 1..L−1, and the next layer index the protocol
  // expects (0 when no forward is in progress). `truncate` resets both.
  std::vector<std::int64_t> pending_slots_;
  int next_layer_ = 0;
};

}  // namespace engine::kvcache
