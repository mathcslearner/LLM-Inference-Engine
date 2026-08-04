#pragma once

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

// CachingAllocator — size-class caching pool over any upstream Allocator
// (M2-T06; design: docs/design/cuda-backend.md §7.3–§7.4). The engine's
// memory-pooling backbone: in practice the upstream is a CudaAllocator, but
// the pool is device-agnostic and works over any Allocator, so it is
// toolkit-free and compiled on every configuration.
//
// This class is the exception the Buffer ownership model anticipated
// (allocator.h): its deleters capture `this` to return blocks to the free
// list instead of freeing upstream. LIFETIME REQUIREMENT: the
// CachingAllocator must outlive every Buffer it allocated. Destroying the
// pool while Buffers are live is programmer error (CHECK); for the same
// reason the class is neither copyable nor movable.
//
// Stream semantics (design §7.4): the pool is stream-agnostic. A block freed
// and immediately reused on any stream is safe only under the §6.4 rule that
// callers never destroy a Buffer while device work using it is in flight —
// the same contract raw cudaFree imposes, but a violation here corrupts the
// next tenant of the block instead of being driver-managed.

namespace engine::memory {

// Thread-safe (one internal mutex — allocation is never on the token hot
// path, tensor.md §6). Requests are rounded up to a size class (512B
// minimum, powers of two up to 1MiB, 1MiB steps above); a freed block
// re-enters its class's free list, so any request in the class reuses it
// exactly. Misses go upstream; upstream exhaustion (kResourceExhausted or
// kOutOfMemory — the pool is device-agnostic, so both count) triggers
// release_cached() plus one retry before the error propagates. The class
// table is an implementation constant: the contract is only that
// alloc→free→alloc of an equal size hits the cache.
class CachingAllocator final : public Allocator {
 public:
  // Every upstream allocation is requested at this fixed alignment, and
  // Allocate CHECK-rejects requests above it. Fixing one pool-wide alignment
  // (cudaMalloc's 256-byte guarantee, ≥ every alignment the engine uses)
  // means any cached block satisfies any admissible request, so free lists
  // never need alignment keys.
  static constexpr std::size_t kMaxAlignment = 256;

  // `upstream` is non-owning and must outlive this allocator.
  explicit CachingAllocator(Allocator* upstream);

  // CHECKs that no Buffer is live (bytes_allocated == 0), then returns all
  // cached blocks to the upstream.
  ~CachingAllocator() override;

  CachingAllocator(CachingAllocator&&) = delete;
  CachingAllocator& operator=(CachingAllocator&&) = delete;

  // Memory of at least `bytes` bytes from the cache or the upstream.
  // `alignment` must be a power of two no larger than kMaxAlignment (both
  // CHECKed — alignments are caller-authored constants). `bytes == 0`
  // returns an engaged Buffer with data() == nullptr, bypassing the cache
  // and the stats. The returned Buffer reports the *requested* size;
  // stats are denominated in class-rounded bytes.
  [[nodiscard]] core::StatusOr<Buffer> Allocate(std::size_t bytes,
                                                std::size_t alignment) override;

  // The upstream's device; every returned Buffer's device() matches.
  [[nodiscard]] tensor::Device device() const override;

  // All sizes in class-rounded bytes, so bytes_reserved is exactly the
  // pool's upstream footprint. Invariant: bytes_reserved == bytes_allocated
  // + the total size of cached free blocks.
  struct Stats {
    std::size_t bytes_allocated = 0;  // currently handed out to live Buffers
    std::size_t bytes_reserved = 0;   // allocated + cached free blocks
    std::uint64_t hit_count = 0;      // served from a free list
    std::uint64_t miss_count = 0;     // required an upstream Allocate (counted
                                      // once per request, even when the
                                      // OOM-retry path allocates twice)
  };
  [[nodiscard]] Stats stats() const;

  // Returns all cached (free) blocks to the upstream. Live Buffers are
  // unaffected. Returns the number of bytes released (class-rounded — the
  // exact drop in bytes_reserved). The upstream deleters run outside the
  // pool's mutex.
  std::size_t release_cached();

 private:
  // A block currently handed out: the upstream Buffer kept alive while the
  // caller holds the pool's wrapper, plus the size class it returns to.
  struct LiveBlock {
    Buffer storage;
    std::size_t class_bytes;
  };

  // Deleter target: moves `ptr`'s block from live_ back to its free list.
  void Return(void* ptr);

  // Moves every cached block into `victims` (destroyed by the caller, which
  // decides whether to hold the lock) and drops bytes_reserved accordingly.
  // Returns the class-rounded byte total. Requires mutex_ held.
  std::size_t ReleaseCachedLocked(std::vector<Buffer>& victims);

  Allocator* const upstream_;

  mutable std::mutex mutex_;
  std::unordered_map<void*, LiveBlock> live_;
  std::unordered_map<std::size_t, std::vector<Buffer>> free_lists_;
  Stats stats_;
};

}  // namespace engine::memory
