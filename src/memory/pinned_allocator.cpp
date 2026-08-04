// CUDA-build implementation of memory/pinned_allocator.h (M2-T07; design:
// docs/design/cuda-backend.md §7.2). CPU-only builds compile
// pinned_allocator_stub.cpp instead (the source-list seam, design §2.2).

#include "memory/pinned_allocator.h"

#include "core/check.h"
#include "core/logging.h"
#include "core/status.h"
#include "cuda/cuda_check.h"
#include "cuda/cuda_utils.h"
#include "memory/allocator.h"

#include <fmt/format.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace engine::memory {

namespace {

[[nodiscard]] constexpr bool IsPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// The Buffer deleter. cudaFreeHost needs no device context, so unlike
// CudaAllocator's deleter there is no device to restore; failure — only
// plausible with a poisoned context (design §4.3), where the process is
// dying anyway — is logged and dropped (a deleter can neither return Status
// nor abort from a destructor, design §7.1).
void FreePinned(void* ptr) {
  if (ptr == nullptr) {
    return;  // zero-size Buffer: nothing was allocated
  }
  if (const cudaError_t err = cudaFreeHost(ptr); err != cudaSuccess) {
    LOG_ERROR("memory", "deleter: cudaFreeHost failed, leaking: {}: {}",
              cudaGetErrorName(err), cudaGetErrorString(err));
  }
  // Do not leave a pending error for an unrelated later launch-error check.
  // Deliberately unconditional; see the matching note in cuda_allocator.cpp.
  (void)cudaGetLastError();
}

}  // namespace

core::StatusOr<PinnedCpuAllocator> PinnedCpuAllocator::Create() {
  const int count = cuda::device_count();
  if (count == 0) {
    return core::UnavailableError(
        "PinnedCpuAllocator: no visible CUDA device — pinned host memory "
        "requires a CUDA driver context");
  }
  return PinnedCpuAllocator();
}

core::StatusOr<Buffer> PinnedCpuAllocator::Allocate(std::size_t bytes,
                                                    std::size_t alignment) {
  CHECK(IsPowerOfTwo(alignment), "alignment must be a power of two, got {}",
        alignment);
  CHECK(alignment <= kGuaranteedAlignment,
        "alignment {} exceeds cudaHostAlloc's {}-byte guarantee", alignment,
        kGuaranteedAlignment);
  if (bytes == 0) {
    return Buffer(nullptr, 0, device(), &FreePinned);
  }
  void* data = nullptr;
  if (const cudaError_t err =
          cudaHostAlloc(&data, bytes, cudaHostAllocPortable);
      err != cudaSuccess) {
    // A failed cudaHostAlloc also latches the last-error slot; clear it so
    // this (non-sticky) failure is not misread by a later launch-error check.
    (void)cudaGetLastError();
    return cuda::ToStatus(
        err, fmt::format("cudaHostAlloc of {} pinned bytes", bytes));
  }
  return Buffer(data, bytes, device(), &FreePinned);
}

}  // namespace engine::memory
