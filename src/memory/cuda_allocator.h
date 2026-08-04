#pragma once

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <cstddef>

// CudaAllocator — naive CUDA device-memory allocator (M2-T05; design:
// docs/design/cuda-backend.md §7.1).
//
// This header is toolkit-free (design §2.2): includable from any TU on any
// build. The implementation touches the CUDA toolkit and is compiled only on
// CUDA builds (cuda_allocator.cpp); CPU-only builds compile
// cuda_allocator_stub.cpp, whose factories return Unimplemented — the
// source-list seam. The implementation reuses the cuda module's device
// introspection and error mapping, the `memory → cuda` edge added by ADR-002
// Amendment 2.
//
// Deleters are self-contained — they capture only the device index, never the
// allocator — so a Buffer may outlive the CudaAllocator that produced it (the
// M1 Allocator contract, unchanged).

namespace engine::memory {

// One allocator per (used) device, over cudaMalloc/cudaFree. Thread-safe:
// Allocate mutates no allocator state and the toolkit calls are themselves
// thread-safe. Every returned Buffer is tagged with the construction-time
// Device, and its memory lives on that device regardless of the caller's
// current-device state.
class CudaAllocator final : public Allocator {
 public:
  // cudaMalloc's alignment guarantee. Allocate CHECK-rejects requests above
  // it — nothing in the engine wants more (revisit if something does).
  static constexpr std::size_t kGuaranteedAlignment = 256;

  // InvalidArgument (naming the index and the count) if `device_index` is
  // outside [0, cuda::device_count()) — which is every index on a CUDA build
  // with no visible device. On CPU-only builds: Unimplemented ("built
  // without CUDA", design §2.2 taxonomy).
  [[nodiscard]] static core::StatusOr<CudaAllocator> Create(int device_index);

  // Device memory of at least `bytes` bytes. `alignment` must be a power of
  // two no larger than kGuaranteedAlignment (both CHECKed — alignments are
  // caller-authored constants, not runtime input). `bytes == 0` returns an
  // engaged Buffer with data() == nullptr whose deleter still runs exactly
  // once. Device OOM → kResourceExhausted (design §4.2): an operating
  // condition the engine plans around, distinct from host kOutOfMemory.
  [[nodiscard]] core::StatusOr<Buffer> Allocate(std::size_t bytes,
                                                std::size_t alignment) override;

  [[nodiscard]] tensor::Device device() const override { return device_; }

 private:
  explicit CudaAllocator(int device_index)
      : device_(tensor::Device::Cuda(device_index)) {}

  tensor::Device device_;
};

// Process-wide CudaAllocator for `device_index`, lazily created on first use
// and intentionally leaked (the DefaultCpuAllocator pattern — the pointer
// stays valid during static destruction). What Tensor::empty allocates from
// for kCUDA devices when given no explicit allocator. Failure taxonomy is
// CudaAllocator::Create's.
[[nodiscard]] core::StatusOr<Allocator*> DefaultCudaAllocator(int device_index);

}  // namespace engine::memory
