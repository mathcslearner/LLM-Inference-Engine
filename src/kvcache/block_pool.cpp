#include "kvcache/block_pool.h"

#include "core/check.h"
#include "core/logging.h"
#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "memory/allocator.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace engine::kvcache {

namespace {

// `block_size` must be a power of two dividing kAttnKb = 64 — {8, 16, 32, 64}
// (paged-kv-cache.md §4). Load-bearing for M8-T05: one 64-key online-softmax
// unit must span an integer number of whole physical blocks so the paged decode
// kernel reproduces the M6-T05 reduction order bit-for-bit.
[[nodiscard]] bool IsValidBlockSize(int block_size) {
  return block_size == 8 || block_size == 16 || block_size == 32 ||
         block_size == 64;
}

// Validates geometry positivity and the fp32-only policy (the M13 INT8 seam).
[[nodiscard]] core::Status ValidateGeometry(const CacheGeometry& geom) {
  if (geom.num_layers < 1 || geom.num_kv_heads < 1 || geom.head_dim < 1) {
    return core::InvalidArgumentError(
        "BlockPool: geometry must be positive, got num_layers={}, "
        "num_kv_heads={}, head_dim={}",
        geom.num_layers, geom.num_kv_heads, geom.head_dim);
  }
  if (geom.dtype != tensor::DataType::kFloat32) {
    return core::UnimplementedError(
        "BlockPool: v0 stores fp32 K/V; dtype {} arrives with M13's INT8 KV "
        "cache",
        tensor::to_string(geom.dtype));
  }
  return core::OkStatus();
}

}  // namespace

core::StatusOr<std::int64_t> BlockPool::BytesPerBlock(const CacheGeometry& geom,
                                                      int block_size) {
  RETURN_IF_ERROR(ValidateGeometry(geom));
  if (!IsValidBlockSize(block_size)) {
    return core::InvalidArgumentError(
        "BlockPool: block_size must be one of {{8, 16, 32, 64}} (power of two "
        "dividing kAttnKb=64, paged-kv-cache.md §4), got {}",
        block_size);
  }
  // 2 · L · Hkv · bs · d · itemsize, in int64 with overflow checks. The
  // sub-terms are all modest (dims from a config), but the product can be large
  // for big models, so guard each multiply.
  std::int64_t elems = 2;
  const std::int64_t factors[] = {
      static_cast<std::int64_t>(geom.num_layers),
      static_cast<std::int64_t>(geom.num_kv_heads),
      static_cast<std::int64_t>(block_size),
      static_cast<std::int64_t>(geom.head_dim),
      static_cast<std::int64_t>(tensor::itemsize(geom.dtype))};
  for (const std::int64_t f : factors) {
    if (__builtin_mul_overflow(elems, f, &elems)) {
      return core::InvalidArgumentError(
          "BlockPool: bytes-per-block overflows int64 (num_layers={}, "
          "num_kv_heads={}, block_size={}, head_dim={})",
          geom.num_layers, geom.num_kv_heads, block_size, geom.head_dim);
    }
  }
  return elems;
}

core::StatusOr<std::int64_t> BlockPool::NumBlocksForBudget(
    const CacheGeometry& geom, int block_size, std::int64_t kv_budget_bytes) {
  if (kv_budget_bytes <= 0) {
    return core::InvalidArgumentError(
        "BlockPool: kv_budget_bytes must be positive, got {}", kv_budget_bytes);
  }
  ASSIGN_OR_RETURN(const std::int64_t bytes_per_block,
                   BytesPerBlock(geom, block_size));
  const std::int64_t num_blocks = kv_budget_bytes / bytes_per_block;
  if (num_blocks < 1) {
    return core::ResourceExhaustedError(
        "BlockPool: KV budget {} bytes is below one block ({} bytes/block) — "
        "the model does not fit the budget",
        kv_budget_bytes, bytes_per_block);
  }
  return num_blocks;
}

BlockPool::BlockPool(CacheGeometry geom, int block_size,
                     std::int64_t num_blocks,
                     std::vector<tensor::Tensor> k_slabs,
                     std::vector<tensor::Tensor> v_slabs,
                     std::int64_t slab_bytes)
    : geom_(geom),
      block_size_(block_size),
      num_blocks_(num_blocks),
      slab_bytes_(slab_bytes),
      block_stride_(static_cast<std::int64_t>(geom.num_kv_heads) * block_size *
                    geom.head_dim),
      head_stride_(static_cast<std::int64_t>(block_size) * geom.head_dim),
      row_stride_(geom.head_dim),
      k_slabs_(std::move(k_slabs)),
      v_slabs_(std::move(v_slabs)),
      mutex_(std::make_unique<std::mutex>()),
      refcount_(static_cast<std::size_t>(num_blocks), 0) {
  // Free list holds every block id; push descending so the first Allocate pops
  // block 0 (LIFO stack). The exact order is an implementation detail (the
  // contract is only "free→alloc reuses a free block"), but pinned for tests.
  free_list_.reserve(static_cast<std::size_t>(num_blocks));
  for (std::int64_t b = num_blocks - 1; b >= 0; --b) {
    free_list_.push_back(static_cast<std::int32_t>(b));
  }
}

BlockPool::BlockPool(BlockPool&& other) noexcept
    : geom_(other.geom_),
      block_size_(other.block_size_),
      num_blocks_(other.num_blocks_),
      slab_bytes_(other.slab_bytes_),
      block_stride_(other.block_stride_),
      head_stride_(other.head_stride_),
      row_stride_(other.row_stride_),
      k_slabs_(std::move(other.k_slabs_)),
      v_slabs_(std::move(other.v_slabs_)),
      mutex_(std::move(other.mutex_)),
      refcount_(std::move(other.refcount_)),
      free_list_(std::move(other.free_list_)),
      used_(other.used_) {
  // A BlockTable holds a raw BlockPool*; moving after a block is handed out
  // would dangle it. Create move-constructs out before any Allocate.
  CHECK(other.used_ == 0,
        "BlockPool moved after {} block(s) handed out — a BlockTable would "
        "dangle; move only before allocation",
        other.used_);
}

core::StatusOr<BlockPool> BlockPool::Create(CacheGeometry geom, int block_size,
                                            std::int64_t num_blocks,
                                            memory::Allocator* allocator) {
  RETURN_IF_ERROR(ValidateGeometry(geom));
  if (!IsValidBlockSize(block_size)) {
    return core::InvalidArgumentError(
        "BlockPool::Create: block_size must be one of {{8, 16, 32, 64}} "
        "(paged-kv-cache.md §4), got {}",
        block_size);
  }
  if (num_blocks < 1) {
    return core::InvalidArgumentError(
        "BlockPool::Create: num_blocks must be >= 1, got {}", num_blocks);
  }
  if (allocator == nullptr) {
    allocator = memory::DefaultCpuAllocator();
  }

  // Per-layer slab shape `[num_blocks, Hkv, bs, d]`; slab_bytes with overflow
  // guard.
  std::int64_t slab_elems = num_blocks;
  const std::int64_t elem_factors[] = {
      static_cast<std::int64_t>(geom.num_kv_heads),
      static_cast<std::int64_t>(block_size),
      static_cast<std::int64_t>(geom.head_dim)};
  for (const std::int64_t f : elem_factors) {
    if (__builtin_mul_overflow(slab_elems, f, &slab_elems)) {
      return core::InvalidArgumentError(
          "BlockPool::Create: slab element count overflows int64 "
          "(num_blocks={}, num_kv_heads={}, block_size={}, head_dim={})",
          num_blocks, geom.num_kv_heads, block_size, geom.head_dim);
    }
  }
  std::int64_t slab_bytes = 0;
  if (__builtin_mul_overflow(
          slab_elems, static_cast<std::int64_t>(tensor::itemsize(geom.dtype)),
          &slab_bytes)) {
    return core::InvalidArgumentError(
        "BlockPool::Create: slab byte size overflows int64");
  }

  const tensor::Shape slab_shape{num_blocks, geom.num_kv_heads, block_size,
                                 geom.head_dim};
  const auto num_layers = static_cast<std::size_t>(geom.num_layers);
  std::vector<tensor::Tensor> k_slabs;
  std::vector<tensor::Tensor> v_slabs;
  k_slabs.reserve(num_layers);
  v_slabs.reserve(num_layers);

  // Allocate 2·L slabs directly from the allocator at kSlabAlignment (256),
  // then zero-fill. We bypass tensor::ops::zeros because Tensor::empty pins a
  // 64-byte alignment; the design mandates 256 (§6.1). RAII drops any
  // already-allocated slab if a later allocation fails, so nothing leaks.
  for (int slab = 0; slab < 2 * geom.num_layers; ++slab) {
    ASSIGN_OR_RETURN(memory::Buffer buffer,
                     allocator->Allocate(static_cast<std::size_t>(slab_bytes),
                                         kSlabAlignment));
    if (buffer.data() != nullptr) {
      std::memset(buffer.data(), 0, static_cast<std::size_t>(slab_bytes));
    }
    ASSIGN_OR_RETURN(tensor::Tensor tensor,
                     tensor::Tensor::from_buffer(
                         std::make_shared<memory::Buffer>(std::move(buffer)),
                         /*byte_offset=*/0, slab_shape, geom.dtype));
    if (slab % 2 == 0) {
      k_slabs.push_back(std::move(tensor));
    } else {
      v_slabs.push_back(std::move(tensor));
    }
  }

  BlockPool pool(geom, block_size, num_blocks, std::move(k_slabs),
                 std::move(v_slabs), slab_bytes);
  LOG_INFO("kvcache",
           "BlockPool: {} blocks × {} tokens ({} bytes/block, {} bytes total) "
           "for {} layers, Hkv={}, d={}",
           num_blocks, block_size, pool.total_bytes() / num_blocks,
           pool.total_bytes(), geom.num_layers, geom.num_kv_heads,
           geom.head_dim);
  return pool;
}

core::StatusOr<std::int32_t> BlockPool::Allocate() {
  const std::lock_guard<std::mutex> lock(*mutex_);
  if (free_list_.empty()) {
    return core::ResourceExhaustedError(
        "BlockPool::Allocate: no free blocks ({} of {} in use)", used_,
        num_blocks_);
  }
  const std::int32_t block = free_list_.back();
  free_list_.pop_back();
  refcount_[static_cast<std::size_t>(block)] = 1;
  ++used_;
  return block;
}

void BlockPool::Share(std::int32_t block) {
  const std::lock_guard<std::mutex> lock(*mutex_);
  CHECK(block_in_range(block),
        "BlockPool::Share: block {} out of range [0, {})", block, num_blocks_);
  std::int32_t& rc = refcount_[static_cast<std::size_t>(block)];
  CHECK(rc > 0,
        "BlockPool::Share: block {} is free (refcount 0) — sharing what "
        "nobody owns is a programmer error",
        block);
  ++rc;
}

void BlockPool::Release(std::int32_t block) {
  const std::lock_guard<std::mutex> lock(*mutex_);
  CHECK(block_in_range(block),
        "BlockPool::Release: block {} out of range [0, {})", block,
        num_blocks_);
  std::int32_t& rc = refcount_[static_cast<std::size_t>(block)];
  CHECK(rc > 0,
        "BlockPool::Release: block {} is already free (refcount 0) — double "
        "free is a programmer error",
        block);
  --rc;
  if (rc == 0) {
    free_list_.push_back(block);
    --used_;
  }
}

std::int32_t BlockPool::refcount(std::int32_t block) const {
  const std::lock_guard<std::mutex> lock(*mutex_);
  CHECK(block_in_range(block),
        "BlockPool::refcount: block {} out of range [0, {})", block,
        num_blocks_);
  return refcount_[static_cast<std::size_t>(block)];
}

BlockPoolStats BlockPool::stats() const {
  const std::lock_guard<std::mutex> lock(*mutex_);
  BlockPoolStats s;
  s.total = num_blocks_;
  s.used = used_;
  s.free = num_blocks_ - used_;
  s.utilization = num_blocks_ > 0 ? static_cast<double>(used_) /
                                        static_cast<double>(num_blocks_)
                                  : 0.0;
  return s;
}

std::int64_t BlockPool::free_blocks() const {
  const std::lock_guard<std::mutex> lock(*mutex_);
  return num_blocks_ - used_;
}

std::int64_t BlockPool::blocks_needed(std::int64_t cur_tokens,
                                      std::int64_t add_tokens) const {
  CHECK(cur_tokens >= 0 && add_tokens >= 0,
        "BlockPool::blocks_needed: cur_tokens ({}) and add_tokens ({}) must be "
        ">= 0",
        cur_tokens, add_tokens);
  const std::int64_t bs = block_size_;
  const std::int64_t before = (cur_tokens + bs - 1) / bs;
  const std::int64_t after = (cur_tokens + add_tokens + bs - 1) / bs;
  return after - before;
}

float* BlockPool::k_slab(int layer) {
  CHECK(layer >= 0 && layer < geom_.num_layers,
        "BlockPool::k_slab: layer {} out of range [0, {})", layer,
        geom_.num_layers);
  return k_slabs_[static_cast<std::size_t>(layer)].data_ptr<float>();
}

float* BlockPool::v_slab(int layer) {
  CHECK(layer >= 0 && layer < geom_.num_layers,
        "BlockPool::v_slab: layer {} out of range [0, {})", layer,
        geom_.num_layers);
  return v_slabs_[static_cast<std::size_t>(layer)].data_ptr<float>();
}

}  // namespace engine::kvcache
