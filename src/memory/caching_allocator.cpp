#include "memory/caching_allocator.h"

#include "core/check.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <fmt/format.h>

#include <bit>
#include <cstddef>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace engine::memory {

namespace {

[[nodiscard]] constexpr bool IsPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

constexpr std::size_t kMinClassBytes = 512;
constexpr std::size_t kLargeStepBytes = std::size_t{1} << 20U;  // 1 MiB

// The size-class table (design §7.3): 512B minimum, powers of two up to
// 1MiB, then 1MiB steps. Rounding waste is bounded at 2x for small blocks
// and ~1/1024 for big ones. Requires bytes > 0.
[[nodiscard]] constexpr std::size_t SizeClass(std::size_t bytes) {
  if (bytes <= kMinClassBytes) {
    return kMinClassBytes;
  }
  if (bytes <= kLargeStepBytes) {
    return std::bit_ceil(bytes);
  }
  if (bytes > std::numeric_limits<std::size_t>::max() - (kLargeStepBytes - 1)) {
    return bytes;  // rounding up would overflow; upstream OOM handles it
  }
  return (bytes + kLargeStepBytes - 1) / kLargeStepBytes * kLargeStepBytes;
}

static_assert(SizeClass(1) == 512);
static_assert(SizeClass(512) == 512);
static_assert(SizeClass(513) == 1024);
static_assert(SizeClass(kLargeStepBytes) == kLargeStepBytes);
static_assert(SizeClass(kLargeStepBytes + 1) == 2 * kLargeStepBytes);
static_assert(SizeClass((3 * kLargeStepBytes) - 5) == 3 * kLargeStepBytes);
static_assert(SizeClass(std::numeric_limits<std::size_t>::max()) ==
              std::numeric_limits<std::size_t>::max());

[[nodiscard]] bool IsExhausted(const core::Status& status) {
  // Device exhaustion is kResourceExhausted (design §4.2), host exhaustion
  // kOutOfMemory; the pool is device-agnostic, so both trigger the
  // release-and-retry courtesy.
  return core::IsResourceExhausted(status) || core::IsOutOfMemory(status);
}

}  // namespace

CachingAllocator::CachingAllocator(Allocator* upstream) : upstream_(upstream) {
  CHECK(upstream_ != nullptr, "CachingAllocator requires an upstream");
}

CachingAllocator::~CachingAllocator() {
  // No lock: if another thread still races the pool's destructor, no locking
  // discipline can save it.
  CHECK(live_.empty(),
        "CachingAllocator destroyed with {} live Buffer(s) holding {} bytes",
        live_.size(), stats_.bytes_allocated);
  // free_lists_ destruction returns every cached block via its upstream
  // deleter.
}

core::StatusOr<Buffer> CachingAllocator::Allocate(std::size_t bytes,
                                                  std::size_t alignment) {
  CHECK(IsPowerOfTwo(alignment), "alignment must be a power of two, got {}",
        alignment);
  CHECK(alignment <= kMaxAlignment,
        "alignment {} exceeds the pool's fixed {}-byte upstream alignment",
        alignment, kMaxAlignment);
  const tensor::Device device = upstream_->device();
  if (bytes == 0) {
    // Bypasses the cache and the stats: there is no block to pool, and the
    // deleter deliberately does not reference the pool.
    return Buffer(nullptr, 0, device, [](void*) {});
  }

  const std::size_t class_bytes = SizeClass(bytes);
  const std::scoped_lock lock(mutex_);

  Buffer storage;
  if (auto it = free_lists_.find(class_bytes);
      it != free_lists_.end() && !it->second.empty()) {
    storage = std::move(it->second.back());
    it->second.pop_back();
    ++stats_.hit_count;
  } else {
    ++stats_.miss_count;
    core::StatusOr<Buffer> allocated =
        upstream_->Allocate(class_bytes, kMaxAlignment);
    if (!allocated.ok() && IsExhausted(allocated.status())) {
      // The caching-allocator OOM courtesy: return every cached block to the
      // upstream and retry once. The victims must actually be freed before
      // the retry, so they are destroyed here, under the lock (upstream
      // deleters are self-contained and cannot re-enter the pool).
      std::vector<Buffer> victims;
      ReleaseCachedLocked(victims);
      victims.clear();
      allocated = upstream_->Allocate(class_bytes, kMaxAlignment);
    }
    if (!allocated.ok()) {
      return allocated.status();
    }
    storage = *std::move(allocated);
    stats_.bytes_reserved += class_bytes;
  }

  void* const data = storage.data();
  stats_.bytes_allocated += class_bytes;
  live_.emplace(data, LiveBlock{.storage = std::move(storage),
                                .class_bytes = class_bytes});
  // The wrapper reports the requested size (the CpuAllocator convention);
  // the full class_bytes remain writable.
  return Buffer(data, bytes, device, [this](void* ptr) { Return(ptr); });
}

tensor::Device CachingAllocator::device() const { return upstream_->device(); }

CachingAllocator::Stats CachingAllocator::stats() const {
  const std::scoped_lock lock(mutex_);
  return stats_;
}

std::size_t CachingAllocator::release_cached() {
  std::vector<Buffer> victims;
  std::size_t released = 0;
  {
    const std::scoped_lock lock(mutex_);
    released = ReleaseCachedLocked(victims);
  }
  // Destroy outside the lock: no reason to hold the pool mutex across
  // upstream frees.
  victims.clear();
  return released;
}

void CachingAllocator::Return(void* ptr) {
  const std::scoped_lock lock(mutex_);
  const auto it = live_.find(ptr);
  CHECK(it != live_.end(), "CachingAllocator: {} is not a live block",
        fmt::ptr(ptr));
  const std::size_t class_bytes = it->second.class_bytes;
  stats_.bytes_allocated -= class_bytes;
  free_lists_[class_bytes].push_back(std::move(it->second.storage));
  live_.erase(it);
}

std::size_t CachingAllocator::ReleaseCachedLocked(
    std::vector<Buffer>& victims) {
  std::size_t released = 0;
  for (auto& [class_bytes, blocks] : free_lists_) {
    released += class_bytes * blocks.size();
    for (Buffer& block : blocks) {
      victims.push_back(std::move(block));
    }
  }
  free_lists_.clear();
  stats_.bytes_reserved -= released;
  return released;
}

}  // namespace engine::memory
