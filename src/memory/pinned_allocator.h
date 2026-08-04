#pragma once

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <cstddef>

// PinnedCpuAllocator — page-locked host memory (M2-T07; design:
// docs/design/cuda-backend.md §7.2).
//
// This header is toolkit-free (design §2.2): includable from any TU on any
// build. The implementation touches the CUDA toolkit and is compiled only on
// CUDA builds (pinned_allocator.cpp); CPU-only builds compile
// pinned_allocator_stub.cpp, whose factory returns Unimplemented — the
// source-list seam. Like the other CUDA allocators it reuses the cuda
// module's introspection and error mapping (ADR-002 Amendment 2).
//
// Pinnedness is a property of the ALLOCATION, not a DeviceType: device() is
// Device::Cpu(), and tensors over pinned memory are ordinary CPU tensors —
// host-dereferenceable, printable, allclose-able. What pinning buys is
// true-async cudaMemcpyAsync (design §8.2): transfers from/to pageable host
// memory are only stream-ordered, transfers from/to pinned memory actually
// overlap. The transfer path does not need to know (the driver detects
// registered ranges); callers that care — the engine's I/O staging buffers,
// later — simply choose this allocator.

namespace engine::memory {

// Page-locked host memory over cudaHostAlloc (portable flag, so allocations
// are pinned for every CUDA context), freed with cudaFreeHost. Thread-safe:
// Allocate mutates no allocator state and the toolkit calls are themselves
// thread-safe. Deleters are fully self-contained (they capture nothing —
// cudaFreeHost needs no device context), so a Buffer may outlive the
// allocator that produced it (the M1 Allocator contract, unchanged).
class PinnedCpuAllocator final : public Allocator {
 public:
  // cudaHostAlloc returns page-aligned memory, comfortably above this;
  // mirroring CudaAllocator, Allocate CHECK-rejects requests above 256 so
  // the CachingAllocator's fixed upstream alignment works over either.
  static constexpr std::size_t kGuaranteedAlignment = 256;

  // Unavailable if no CUDA device is visible (pinning requires a CUDA
  // driver context; cudaHostAlloc would fail anyway — this just reports it
  // eagerly, the CudaAllocator::Create stance). On CPU-only builds:
  // Unimplemented ("built without CUDA", design §2.2 taxonomy).
  [[nodiscard]] static core::StatusOr<PinnedCpuAllocator> Create();

  // Page-locked host memory of at least `bytes` bytes. `alignment` must be
  // a power of two no larger than kGuaranteedAlignment (both CHECKed —
  // alignments are caller-authored constants, not runtime input).
  // `bytes == 0` returns an engaged Buffer with data() == nullptr whose
  // deleter still runs exactly once. Exhaustion (pinned memory is a scarcer
  // resource than pageable host memory — the OS must lock the pages) →
  // kResourceExhausted via the design §4.2 mapping.
  [[nodiscard]] core::StatusOr<Buffer> Allocate(std::size_t bytes,
                                                std::size_t alignment) override;

  // Pinned memory is host memory (file comment above).
  [[nodiscard]] tensor::Device device() const override {
    return tensor::Device::Cpu();
  }

 private:
  PinnedCpuAllocator() = default;
};

}  // namespace engine::memory
