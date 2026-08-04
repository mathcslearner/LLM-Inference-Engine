// CUDA-build implementation of memory/cuda_allocator.h (M2-T05; design:
// docs/design/cuda-backend.md §7.1). CPU-only builds compile
// cuda_allocator_stub.cpp instead (the source-list seam, design §2.2).

#include "memory/cuda_allocator.h"

#include "core/check.h"
#include "core/logging.h"
#include "core/status.h"
#include "cuda/cuda_check.h"
#include "cuda/cuda_utils.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <fmt/format.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace engine::memory {

namespace {

[[nodiscard]] constexpr bool IsPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// The Buffer deleter. Deliberately raw cudaSetDevice/cudaFree rather than
// ScopedSetDevice + CUDA_CHECK: a deleter can neither return Status nor
// abort from a destructor, so every failure here — only plausible with a
// poisoned context (design §4.3), where the process is dying anyway — is
// logged and dropped (design §7.1).
void FreeOnDevice(int device_index, void* ptr) {
  if (ptr == nullptr) {
    return;  // zero-size Buffer: nothing was allocated
  }
  int previous = -1;
  const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
  if (const cudaError_t err = cudaSetDevice(device_index); err != cudaSuccess) {
    // Proceed anyway: cudaFree accepts pointers from any device; setting the
    // owning device first merely keeps the free on the allocation's context.
    LOG_WARN("memory", "deleter: cudaSetDevice({}) failed: {}: {}",
             device_index, cudaGetErrorName(err), cudaGetErrorString(err));
  }
  if (const cudaError_t err = cudaFree(ptr); err != cudaSuccess) {
    LOG_ERROR("memory", "deleter: cudaFree on cuda:{} failed, leaking: {}: {}",
              device_index, cudaGetErrorName(err), cudaGetErrorString(err));
  }
  // Do not leave a pending error for an unrelated later launch-error check.
  (void)cudaGetLastError();
  if (have_previous) {
    (void)cudaSetDevice(previous);
  }
}

}  // namespace

core::StatusOr<CudaAllocator> CudaAllocator::Create(int device_index) {
  const int count = cuda::device_count();
  if (device_index < 0 || device_index >= count) {
    return core::InvalidArgumentError(
        "CudaAllocator: device index {} out of range: {} visible CUDA "
        "device(s)",
        device_index, count);
  }
  return CudaAllocator(device_index);
}

core::StatusOr<Buffer> CudaAllocator::Allocate(std::size_t bytes,
                                               std::size_t alignment) {
  CHECK(IsPowerOfTwo(alignment), "alignment must be a power of two, got {}",
        alignment);
  CHECK(alignment <= kGuaranteedAlignment,
        "alignment {} exceeds cudaMalloc's {}-byte guarantee", alignment,
        kGuaranteedAlignment);
  const int device_index = device_.index();
  const Buffer::Deleter deleter = [device_index](void* ptr) {
    FreeOnDevice(device_index, ptr);
  };
  if (bytes == 0) {
    return Buffer(nullptr, 0, device_, deleter);
  }
  const cuda::ScopedSetDevice guard(device_index);
  void* data = nullptr;
  if (const cudaError_t err = cudaMalloc(&data, bytes); err != cudaSuccess) {
    // A failed cudaMalloc also latches the last-error slot; clear it so this
    // (non-sticky) failure is not misread by a later launch-error check.
    (void)cudaGetLastError();
    return cuda::ToStatus(err, fmt::format("cudaMalloc of {} bytes on {}",
                                           bytes, device_.ToString()));
  }
  return Buffer(data, bytes, device_, deleter);
}

core::StatusOr<Allocator*> DefaultCudaAllocator(int device_index) {
  // Guards lazy creation; created instances are immutable afterward.
  static std::mutex mutex;
  // Intentionally leaked, like the instances (DefaultCpuAllocator rationale).
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static std::vector<CudaAllocator*>* const instances =
      new std::vector<CudaAllocator*>();

  const int count = cuda::device_count();
  if (device_index < 0 || device_index >= count) {
    return core::InvalidArgumentError(
        "DefaultCudaAllocator: device index {} out of range: {} visible CUDA "
        "device(s)",
        device_index, count);
  }
  const std::scoped_lock lock(mutex);
  if (instances->empty()) {
    instances->resize(static_cast<std::size_t>(count), nullptr);
  }
  CudaAllocator*& slot = (*instances)[static_cast<std::size_t>(device_index)];
  if (slot == nullptr) {
    ASSIGN_OR_RETURN(CudaAllocator allocator, Create(device_index));
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    slot = new CudaAllocator(std::move(allocator));
  }
  return slot;
}

}  // namespace engine::memory
