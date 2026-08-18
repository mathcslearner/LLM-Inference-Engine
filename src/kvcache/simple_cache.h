#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <vector>

// `SimpleKvCache` — the v0 contiguous KV cache (M5-T06; design:
// docs/design/model-execution.md §6.2). One object holds the K/V for **one
// sequence across all layers**; a concrete implementation of the `KvCache`
// interface (kv_cache.h, M5-T05) the `Attention` module and the generation loop
// consume.
//
// Storage is two contiguous fp32 tensors, `[num_layers, Hkv, capacity, d]` each
// for K and V, allocated once at construction from `capacity` (caller-chosen:
// prompt length + max new tokens for the single-request loop). Head-major
// (`[…, Hkv, len, d]`) because M6-T05's decode attention reads one kv head's
// full history contiguously, and it matches HF's cache after `transpose(1, 2)`.
//
// `append(layer, k, v)` transposes the token-major `[T, Hkv, d]` block the
// `Attention` module produces into `[Hkv, T, d]` at the layer's fill offset;
// `view(layer)` **gathers** the layer's `[Hkv, fill, d]` history into a fresh
// contiguous tensor. The gather (rather than the "zero-copy slice" an earlier
// draft of §6.2 imagined) is deliberate: a `[Hkv, fill, d]` window of the
// `[…, capacity, d]` store is inner-strided whenever `fill < capacity`, and
// `cpu::attention` requires contiguous K/V. Copying keeps the T05 op contract
// untouched and mirrors the seam M8 keeps — v0's `view()` is the reference's
// "gather cached K/V for a layer" helper; M8's is a block-table gather; the
// `Attention` code above them is unchanged (§6.4). The gather is
// O(Hkv·fill·d), strictly cheaper than the attention it feeds.
//
// **Append-only** (§6.4): a committed position's K/V is never rewritten — only
// appended past, or dropped by `truncate`. This is the immutability M11's
// prefix sharing later rests on.
//
// Layering (ADR-002 Amendment 5): `kvcache` links `tensor`/`memory`/`core`
// only; it never links `model` or `cpu`. All validation is local.

namespace engine::kvcache {

class SimpleKvCache final : public KvCache {
 public:
  // Allocates the K/V stores for `geom` and `capacity` tokens. `capacity >= 1`
  // and every geometry field `>= 1` are required (InvalidArgument otherwise);
  // v0 stores fp32, so a non-f32 `geom.dtype` is Unimplemented (the field
  // exists for M13's INT8 KV — §6.4). Allocation failure propagates
  // (kOutOfMemory).
  [[nodiscard]] static core::StatusOr<SimpleKvCache> Create(
      CacheGeometry geom, std::int64_t capacity);

  [[nodiscard]] CacheGeometry geometry() const override { return geom_; }
  // Committed tokens = the fill every layer agrees on (the minimum per-layer
  // fill). Mid-forward, or after a failed forward, layers disagree and this
  // reports the smallest — the count safe to treat as committed.
  [[nodiscard]] std::int64_t length() const override;
  [[nodiscard]] std::int64_t capacity() const override { return capacity_; }

  // Appends `k`, `v` (`[T, Hkv, d]`, token-major, contiguous, f32, `T >= 1`) to
  // `layer`, transposing to head-major storage. Out-of-range `layer`, a
  // rank/shape/dtype/contiguity mismatch → InvalidArgument; an append past
  // `capacity` → ResourceExhausted (checked before any write — a rejected
  // append leaves the layer untouched). Undefined `k`/`v` are a programmer
  // error (CHECK). The interface requires this be called once per layer per
  // forward in layer order; that ordering is a caller contract, not enforced.
  [[nodiscard]] core::Status append(int layer, const tensor::Tensor& k,
                                    const tensor::Tensor& v) override;

  // Gathers `layer`'s `[Hkv, fill, d]` head-major history into a fresh
  // contiguous f32 tensor (an owned snapshot — always valid, the interface's
  // "until next append" is trivially satisfied). `fill == 0` yields a valid
  // `[Hkv, 0, d]` tensor. Out-of-range `layer` → InvalidArgument.
  [[nodiscard]] core::StatusOr<KvView> view(int layer) const override;

  // Drops every layer's fill to `new_length` (also re-synchronizing layers
  // that disagree after a mid-forward failure). `truncate(0)` == `reset()`. A
  // `new_length` negative or above the current committed length →
  // InvalidArgument. No storage is zeroed: append overwrites, and immutability
  // covers committed positions only.
  [[nodiscard]] core::Status truncate(std::int64_t new_length) override;

  // Debug/test accessor: this layer's own fill (may exceed `length()`
  // mid-forward). Out-of-range `layer` returns 0.
  [[nodiscard]] std::int64_t layer_length(int layer) const;

 private:
  SimpleKvCache(CacheGeometry geom, std::int64_t capacity, tensor::Tensor k,
                tensor::Tensor v)
      : geom_(geom),
        capacity_(capacity),
        k_store_(std::move(k)),
        v_store_(std::move(v)),
        fill_(static_cast<std::size_t>(geom.num_layers), 0) {}

  [[nodiscard]] bool layer_in_range(int layer) const {
    return layer >= 0 && layer < geom_.num_layers;
  }

  CacheGeometry geom_;
  std::int64_t capacity_ = 0;
  // Both `[num_layers, Hkv, capacity, d]`, contiguous, fp32.
  tensor::Tensor k_store_;
  tensor::Tensor v_store_;
  std::vector<std::int64_t> fill_;  // per-layer committed length
};

}  // namespace engine::kvcache
