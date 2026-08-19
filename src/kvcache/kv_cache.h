#pragma once

#include "core/status.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>

// The KV-cache interface — v0 (M5; design: docs/design/model-execution.md §6).
//
// One `KvCache` object holds the K/V for **one sequence across all layers**;
// the engine owns one per active sequence (M9-T02's `Sequence` holds a
// per-sequence cache handle, so the object is per-sequence from day one, not a
// global store). The `Attention` module (model-execution.md §4.2) appends this
// call's new K/V per layer and reads the accumulated K/V back through this
// interface; the generation loop consults only `length`/`capacity`.
//
// This header is the **seam M8 keeps**: `PagedKvCache` becomes a second
// implementation of this exact interface (a shared block pool + per-sequence
// block table replacing the contiguous storage), and the attention/generator
// code that consumes the interface does not change (§6.4). It is deliberately
// pure-interface + POD — no implementation lives here.
//
// Layering (ADR-002 Amendment 5): `kvcache` is a layer-2 domain module linking
// only `tensor`/`memory`/`core` — and, as of M8-T04, the downward
// `kvcache → kernels` edge (the paged scatter/decode kernels; ADR-002 permits a
// layer-2→1 edge without amendment, as `model → kernels`). It never links
// `model`. The `model → kvcache` edge (`Attention` holds a `KvCache&`) is
// one-directional.
//
// The interface header lands with M5-T05 (not T06) because the `Attention`
// module consumes it; `SimpleKvCache`, the v0 contiguous implementation, and
// the token-by-token KV invariant test land with M5-T06 (§6.2). M8-T04 adds
// `PagedKvCache` (a shared block pool + per-sequence block table) as a second
// implementation of this exact interface, plus one additive accessor —
// `paged_view` — the paged decode fast path needs (§8.3); every other consumer
// (`Attention`, the generation loop) is oblivious to which implementation it
// holds.

namespace engine::kvcache {

// The cache geometry a model requires (model-execution.md §6.1). The caller
// constructs a matching `KvCache`; `Model::cache_geometry()` (M5-T07) reports
// what the model expects, and `append`/`view` validate against it.
struct CacheGeometry {
  int num_layers = 0;
  int num_kv_heads = 0;  // Hkv
  int head_dim = 0;      // d
  // v0: f32; M13 adds int8 KV. The stored K/V dtype (not the activation dtype).
  tensor::DataType dtype = tensor::DataType::kFloat32;
};

// A read view of one layer's accumulated K/V (model-execution.md §6.1, §6.2).
// Head-major `[Hkv, len, d]` — the layout M6-T05's decode attention reads one
// kv head's full history contiguously, matching HF's cache after
// `transpose(1, 2)`. Valid until the next `append` to that layer.
struct KvView {
  tensor::Tensor k;  // [Hkv, len, d], contiguous, head-major
  tensor::Tensor v;  // [Hkv, len, d]
};

// A zero-copy read view of one layer's paged K/V, addressed **through** a block
// table (paged-kv-cache.md §8.3). A paged cache cannot hand back a single
// contiguous `[Hkv, len, d]` slice (the history is scattered across blocks), so
// the decode fast path (M8-T05 `PagedDecodeAttentionF32`) reads slabs + block
// table directly instead of paying a full-history gather every step. Valid
// until the next `append`/`truncate` to this sequence. `SimpleKvCache` does not
// support it (default `Unimplemented`); `PagedKvCache` overrides `paged_view`.
struct PagedKvView {
  const float* k_slab = nullptr;              // layer's K slab base
  const float* v_slab = nullptr;              // layer's V slab base
  const std::int32_t* block_table = nullptr;  // logical→physical, this sequence
  std::int64_t num_blocks = 0;                // valid entries in block_table
  int block_size = 0;                         // bs
  std::int64_t length = 0;        // committed tokens (last block partial)
  std::int64_t block_stride = 0;  // Hkv·bs·d — one block id's float span
  // Hkv, d come from geometry().
};

// Per-sequence, all-layers, device-agnostic KV cache (model-execution.md §6.1).
// The four verbs the roadmap names for v0 — append, view, current length, reset
// — map to `append`, `view`, `length`, and `truncate(0)`. `truncate` is the one
// addition (M15-T04 rollback needs it; v0 provides it trivially by dropping the
// tail), and it makes `reset` its `new_length == 0` case.
class KvCache {
 public:
  virtual ~KvCache() = default;

  [[nodiscard]] virtual CacheGeometry geometry() const = 0;
  // Committed tokens (all layers agree on the fill after a completed forward).
  [[nodiscard]] virtual std::int64_t length() const = 0;
  // Max tokens this cache can hold (v0: fixed at construction).
  [[nodiscard]] virtual std::int64_t capacity() const = 0;

  // Appends T new tokens' K/V for one layer. `k`, `v`: `[T, Hkv, d]`
  // (token-major, as the `Attention` module produces them). Must be called once
  // per layer per forward, in layer order; advances that layer's fill. An
  // over-capacity append is `ResourceExhausted`; a geometry/shape/dtype
  // mismatch is `InvalidArgument` (per-request-recoverable, ADR-003).
  [[nodiscard]] virtual core::Status append(int layer, const tensor::Tensor& k,
                                            const tensor::Tensor& v) = 0;

  // Read view of `layer`'s K/V over `[0, current fill)`, including tokens
  // appended this forward — head-major `[Hkv, fill, d]`. Valid until the next
  // append to this layer. An out-of-range `layer` is `InvalidArgument`.
  [[nodiscard]] virtual core::StatusOr<KvView> view(int layer) const = 0;

  // Zero-copy paged view of `layer` (the decode fast path — §8.3). The default
  // is `Unimplemented`: a contiguous cache (`SimpleKvCache`) cannot provide
  // one, and the consumer (`OptimizedModel::forward`) falls back to `view()` +
  // the contiguous decode kernel when it sees `Unimplemented`. `PagedKvCache`
  // overrides it. An out-of-range `layer` is `InvalidArgument`.
  [[nodiscard]] virtual core::StatusOr<PagedKvView> paged_view(
      int layer) const {
    (void)layer;
    return core::UnimplementedError("paged_view: cache is not paged");
  }

  // Drop everything after `new_length` tokens (per layer). `truncate(0)` ==
  // `reset`. A `new_length` above the current length, or negative, is
  // `InvalidArgument`. (M15 rollback; also the general current-length mutator.)
  [[nodiscard]] virtual core::Status truncate(std::int64_t new_length) = 0;

  void reset() { (void)truncate(0); }

 protected:
  // Interface type: protect (not delete) the special members so a `KvCache&`
  // cannot be sliced while concrete implementations stay movable.
  KvCache() = default;
  KvCache(const KvCache&) = default;
  KvCache& operator=(const KvCache&) = default;
  KvCache(KvCache&&) = default;
  KvCache& operator=(KvCache&&) = default;
};

}  // namespace engine::kvcache
