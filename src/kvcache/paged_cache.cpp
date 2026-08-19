#include "kvcache/paged_cache.h"

#include "core/check.h"
#include "core/status.h"
#include "kernels/kv_scatter.h"
#include "kvcache/kv_cache.h"
#include "kvcache/paged_gather.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace engine::kvcache {

namespace {

// Requires a token-major append block: rank-3, contiguous, f32, `[T, Hkv, d]`
// matching the cache geometry with `T >= 1` (mirrors SimpleKvCache's check so
// the two implementations reject the same malformed inputs identically).
[[nodiscard]] core::Status RequireAppendBlock(const tensor::Tensor& t,
                                              const CacheGeometry& geom,
                                              const char* role) {
  if (t.shape().rank() != 3) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: {} must be rank-3 [T, {}, {}], got rank-{}",
        role, geom.num_kv_heads, geom.head_dim, t.shape().rank());
  }
  if (!t.is_contiguous()) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: {} must be contiguous", role);
  }
  if (t.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: {} must be f32, got {}", role,
        tensor::to_string(t.dtype()));
  }
  if (t.shape().dim(0) < 1 || t.shape().dim(1) != geom.num_kv_heads ||
      t.shape().dim(2) != geom.head_dim) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: {} must be [T>=1, {}, {}], got {}", role,
        geom.num_kv_heads, geom.head_dim, t.shape());
  }
  return core::OkStatus();
}

}  // namespace

PagedKvCache::PagedKvCache(BlockPool* pool) : pool_(pool), table_(pool) {
  CHECK(pool_ != nullptr, "PagedKvCache: pool must not be null");
}

std::int64_t PagedKvCache::capacity() const {
  const std::int64_t bs = pool_->block_size();
  // Owned slots (num_blocks·bs — includes the partial tail block's free slots)
  // plus the free pool. Advisory: another sequence may claim the free blocks
  // first (§8.1).
  return (table_.num_blocks() * bs) + (pool_->free_blocks() * bs);
}

core::Status PagedKvCache::append(int layer, const tensor::Tensor& k,
                                  const tensor::Tensor& v) {
  CHECK(k.defined() && v.defined(),
        "PagedKvCache::append: k/v must be defined tensors");
  if (!layer_in_range(layer)) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: layer {} out of range [0, {})", layer,
        pool_->geometry().num_layers);
  }
  const CacheGeometry& geom = pool_->geometry();
  RETURN_IF_ERROR(RequireAppendBlock(k, geom, "k"));
  RETURN_IF_ERROR(RequireAppendBlock(v, geom, "v"));
  if (k.shape().dim(0) != v.shape().dim(0)) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: k T ({}) and v T ({}) must match",
        k.shape().dim(0), v.shape().dim(0));
  }

  const std::int64_t t_dim = k.shape().dim(0);

  // Enforce the per-forward layer protocol (§8.2). Layer 0 opens a forward and
  // grows the table; layers 1..L−1 reuse its slot mapping. `next_layer_` is 0
  // between forwards.
  if (layer != next_layer_) {
    return core::InvalidArgumentError(
        "PagedKvCache::append: expected layer {} next (append in layer order, "
        "once per layer per forward), got {}",
        next_layer_, layer);
  }

  if (layer == 0) {
    // Grow the table all-or-nothing; on exhaustion nothing is written and the
    // cache is left exactly as before (BlockTable's guarantee). `next_layer_`
    // is still 0, so no forward is in progress after a failed open — clear any
    // stale slot mapping from a prior forward so the post-failure state is
    // unambiguous (a decode step can be retried once blocks free up, §10.2).
    auto grown = table_.AppendTokens(t_dim);
    if (!grown.ok()) {
      pending_slots_.clear();
      return grown.status();
    }
    pending_slots_ = *std::move(grown);
  } else {
    const auto pending = static_cast<std::int64_t>(pending_slots_.size());
    if (t_dim != pending) {
      // Layers within a forward must carry the same token batch layer 0 sized.
      return core::InvalidArgumentError(
          "PagedKvCache::append: layer {} T ({}) must match this forward's "
          "token batch ({} from layer 0)",
          layer, t_dim, pending);
    }
  }

  kernels::KvScatterF32(k.data_ptr<float>(), v.data_ptr<float>(),
                        pending_slots_.data(), t_dim, geom.num_kv_heads,
                        geom.head_dim, pool_->block_size(),
                        pool_->k_slab(layer), pool_->v_slab(layer));

  // Advance the protocol; wrapping to 0 marks the forward complete.
  next_layer_ = (layer + 1) % geom.num_layers;
  return core::OkStatus();
}

core::StatusOr<KvView> PagedKvCache::view(int layer) const {
  if (!layer_in_range(layer)) {
    return core::InvalidArgumentError(
        "PagedKvCache::view: layer {} out of range [0, {})", layer,
        pool_->geometry().num_layers);
  }
  // A layer whose K/V for this forward has not been appended yet (its slot
  // mapping is allocated but unwritten) must expose only its *committed*
  // history — exactly what SimpleKvCache::view does with its per-layer fill.
  // Layer 0 grows the table (bumping num_tokens); mid-forward, layers
  // >= next_layer_ have not written the new tokens, so they see
  // num_tokens - pending. Layers < next_layer_ (and a completed forward,
  // next_layer_ == 0 with no pending) see the full length.
  const std::int64_t committed = table_.num_tokens();
  const auto pending = static_cast<std::int64_t>(pending_slots_.size());
  const bool layer_written = (next_layer_ == 0) || (layer < next_layer_);
  const std::int64_t visible = layer_written ? committed : committed - pending;
  return GatherLayerKV(*pool_, table_, layer, visible);
}

core::StatusOr<PagedKvView> PagedKvCache::paged_view(int layer) const {
  if (!layer_in_range(layer)) {
    return core::InvalidArgumentError(
        "PagedKvCache::paged_view: layer {} out of range [0, {})", layer,
        pool_->geometry().num_layers);
  }
  return PagedKvView{
      .k_slab = pool_->k_slab(layer),
      .v_slab = pool_->v_slab(layer),
      .block_table = table_.blocks().data(),
      .num_blocks = table_.num_blocks(),
      .block_size = pool_->block_size(),
      .length = table_.num_tokens(),
      .block_stride = pool_->block_stride(),
  };
}

core::Status PagedKvCache::truncate(std::int64_t new_length) {
  RETURN_IF_ERROR(table_.Truncate(new_length));
  // A truncate resets any in-progress forward's pending state.
  pending_slots_.clear();
  next_layer_ = 0;
  return core::OkStatus();
}

}  // namespace engine::kvcache
