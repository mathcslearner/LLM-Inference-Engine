#include "kvcache/paged_gather.h"

#include "core/status.h"
#include "kernels/kv_gather.h"
#include "kvcache/kv_cache.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <utility>

namespace engine::kvcache {

core::StatusOr<KvView> GatherLayerKV(BlockPool& pool, const BlockTable& table,
                                     int layer, std::int64_t length) {
  const CacheGeometry& geom = pool.geometry();
  if (layer < 0 || layer >= geom.num_layers) {
    return core::InvalidArgumentError(
        "GatherLayerKV: layer {} out of range [0, {})", layer, geom.num_layers);
  }
  if (length < 0 || length > table.num_tokens()) {
    return core::InvalidArgumentError(
        "GatherLayerKV: length {} out of range [0, {}]", length,
        table.num_tokens());
  }

  const std::int64_t hkv = geom.num_kv_heads;
  const std::int64_t d = geom.head_dim;

  // Fresh contiguous head-major [Hkv, length, d] outputs — every element is
  // overwritten by the gather (no need to zero-fill). `length == 0` yields
  // empty views, matching SimpleKvCache::view on an empty cache.
  ASSIGN_OR_RETURN(
      tensor::Tensor out_k,
      tensor::Tensor::empty(tensor::Shape{hkv, length, d},
                            tensor::DataType::kFloat32, tensor::Device::Cpu()));
  ASSIGN_OR_RETURN(
      tensor::Tensor out_v,
      tensor::Tensor::empty(tensor::Shape{hkv, length, d},
                            tensor::DataType::kFloat32, tensor::Device::Cpu()));

  kernels::KvGatherF32(pool.k_slab(layer), pool.v_slab(layer),
                       table.blocks().data(), pool.block_size(), length, hkv, d,
                       out_k.data_ptr<float>(), out_v.data_ptr<float>());

  return KvView{.k = std::move(out_k), .v = std::move(out_v)};
}

}  // namespace engine::kvcache
