#include "kvcache/simple_cache.h"

#include "core/check.h"
#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace engine::kvcache {

namespace {

// Requires a token-major append block: rank-3, contiguous, f32, `[T, Hkv, d]`
// matching the cache geometry with `T >= 1`. `role` names the operand.
[[nodiscard]] core::Status RequireAppendBlock(const tensor::Tensor& t,
                                              const CacheGeometry& geom,
                                              const char* role) {
  if (t.shape().rank() != 3) {
    return core::InvalidArgumentError(
        "SimpleKvCache::append: {} must be rank-3 [T, {}, {}], got rank-{}",
        role, geom.num_kv_heads, geom.head_dim, t.shape().rank());
  }
  if (!t.is_contiguous()) {
    return core::InvalidArgumentError(
        "SimpleKvCache::append: {} must be "
        "contiguous",
        role);
  }
  if (t.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "SimpleKvCache::append: {} must be f32, got {}", role,
        tensor::to_string(t.dtype()));
  }
  if (t.shape().dim(0) < 1 || t.shape().dim(1) != geom.num_kv_heads ||
      t.shape().dim(2) != geom.head_dim) {
    return core::InvalidArgumentError(
        "SimpleKvCache::append: {} must be [T>=1, {}, {}], got {}", role,
        geom.num_kv_heads, geom.head_dim, t.shape());
  }
  return core::OkStatus();
}

}  // namespace

core::StatusOr<SimpleKvCache> SimpleKvCache::Create(CacheGeometry geom,
                                                    std::int64_t capacity) {
  if (geom.num_layers < 1 || geom.num_kv_heads < 1 || geom.head_dim < 1) {
    return core::InvalidArgumentError(
        "SimpleKvCache::Create: geometry must be positive, got num_layers={}, "
        "num_kv_heads={}, head_dim={}",
        geom.num_layers, geom.num_kv_heads, geom.head_dim);
  }
  if (capacity < 1) {
    return core::InvalidArgumentError(
        "SimpleKvCache::Create: capacity must be >= 1, got {}", capacity);
  }
  if (geom.dtype != tensor::DataType::kFloat32) {
    return core::UnimplementedError(
        "SimpleKvCache::Create: v0 stores fp32 K/V; dtype {} arrives with "
        "M13's "
        "INT8 KV cache",
        tensor::to_string(geom.dtype));
  }

  const tensor::Shape store_shape{geom.num_layers, geom.num_kv_heads, capacity,
                                  geom.head_dim};
  ASSIGN_OR_RETURN(tensor::Tensor k,
                   tensor::ops::zeros(store_shape, tensor::DataType::kFloat32));
  ASSIGN_OR_RETURN(tensor::Tensor v,
                   tensor::ops::zeros(store_shape, tensor::DataType::kFloat32));
  return SimpleKvCache(geom, capacity, std::move(k), std::move(v));
}

std::int64_t SimpleKvCache::length() const {
  return *std::ranges::min_element(fill_);
}

std::int64_t SimpleKvCache::layer_length(int layer) const {
  if (!layer_in_range(layer)) {
    return 0;
  }
  return fill_[static_cast<std::size_t>(layer)];
}

core::Status SimpleKvCache::append(int layer, const tensor::Tensor& k,
                                   const tensor::Tensor& v) {
  CHECK(k.defined() && v.defined(),
        "SimpleKvCache::append: k/v must be defined tensors");
  if (!layer_in_range(layer)) {
    return core::InvalidArgumentError(
        "SimpleKvCache::append: layer {} out of range [0, {})", layer,
        geom_.num_layers);
  }
  RETURN_IF_ERROR(RequireAppendBlock(k, geom_, "k"));
  RETURN_IF_ERROR(RequireAppendBlock(v, geom_, "v"));
  if (k.shape().dim(0) != v.shape().dim(0)) {
    return core::InvalidArgumentError(
        "SimpleKvCache::append: k T ({}) and v T ({}) must match",
        k.shape().dim(0), v.shape().dim(0));
  }

  const std::int64_t t_dim = k.shape().dim(0);
  const std::int64_t hkv = geom_.num_kv_heads;
  const std::int64_t d = geom_.head_dim;
  const auto li = static_cast<std::size_t>(layer);
  const std::int64_t old = fill_[li];
  const std::int64_t next = old + t_dim;
  if (next > capacity_) {
    return core::ResourceExhaustedError(
        "SimpleKvCache::append: layer {} fill {} + {} exceeds capacity {}",
        layer, old, t_dim, capacity_);
  }

  // Transpose token-major src[t, h, e] -> head-major store[layer, h, old+t, e].
  const std::int64_t layer_base = static_cast<std::int64_t>(layer) * hkv;
  const auto scatter = [&](const tensor::Tensor& src, tensor::Tensor& store) {
    const float* sp = src.data_ptr<float>();
    auto* dp = store.data_ptr<float>();
    for (std::int64_t t = 0; t < t_dim; ++t) {
      for (std::int64_t h = 0; h < hkv; ++h) {
        const float* s_vec = sp + (((t * hkv) + h) * d);
        float* d_vec = dp + ((((layer_base + h) * capacity_) + (old + t)) * d);
        for (std::int64_t e = 0; e < d; ++e) {
          d_vec[e] = s_vec[e];
        }
      }
    }
  };
  scatter(k, k_store_);
  scatter(v, v_store_);
  fill_[li] = next;
  return core::OkStatus();
}

core::StatusOr<KvView> SimpleKvCache::view(int layer) const {
  if (!layer_in_range(layer)) {
    return core::InvalidArgumentError(
        "SimpleKvCache::view: layer {} out of range [0, {})", layer,
        geom_.num_layers);
  }
  const std::int64_t hkv = geom_.num_kv_heads;
  const std::int64_t d = geom_.head_dim;
  const std::int64_t fill = fill_[static_cast<std::size_t>(layer)];

  // Gather the [Hkv, fill, d] head-major window into fresh contiguous tensors.
  ASSIGN_OR_RETURN(tensor::Tensor k_view,
                   tensor::ops::zeros(tensor::Shape{hkv, fill, d},
                                      tensor::DataType::kFloat32));
  ASSIGN_OR_RETURN(tensor::Tensor v_view,
                   tensor::ops::zeros(tensor::Shape{hkv, fill, d},
                                      tensor::DataType::kFloat32));
  const std::int64_t layer_base = static_cast<std::int64_t>(layer) * hkv;
  const auto gather = [&](const tensor::Tensor& store, tensor::Tensor& out) {
    const float* sp = store.data_ptr<float>();
    auto* op = out.data_ptr<float>();
    for (std::int64_t h = 0; h < hkv; ++h) {
      const float* s_head = sp + (((layer_base + h) * capacity_) * d);
      float* o_head = op + ((h * fill) * d);
      for (std::int64_t s = 0; s < fill * d; ++s) {
        o_head[s] = s_head[s];
      }
    }
  };
  gather(k_store_, k_view);
  gather(v_store_, v_view);
  return KvView{.k = std::move(k_view), .v = std::move(v_view)};
}

core::Status SimpleKvCache::truncate(std::int64_t new_length) {
  if (new_length < 0 || new_length > length()) {
    return core::InvalidArgumentError(
        "SimpleKvCache::truncate: new_length {} out of range [0, {}]",
        new_length, length());
  }
  std::ranges::fill(fill_, new_length);
  return core::OkStatus();
}

}  // namespace engine::kvcache
