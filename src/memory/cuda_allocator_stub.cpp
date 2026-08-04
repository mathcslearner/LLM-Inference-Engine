// CPU-only-build stub for memory/cuda_allocator.h (M2-T05; design:
// docs/design/cuda-backend.md §2.2): same symbols, no CUDA toolkit. CUDA
// builds compile cuda_allocator.cpp instead (the source-list seam).
//
// Create and DefaultCudaAllocator are the only producers of instances and
// both return Unimplemented here, so no CudaAllocator can ever exist on a
// CPU-only build; Allocate is a link-satisfier that cannot be reached.

#include "core/check.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "memory/cuda_allocator.h"

#include <cstddef>

namespace engine::memory {

core::StatusOr<CudaAllocator> CudaAllocator::Create(int device_index) {
  return core::UnimplementedError(
      "CudaAllocator::Create({}): engine built without CUDA support",
      device_index);
}

core::StatusOr<Buffer> CudaAllocator::Allocate(std::size_t bytes,
                                               std::size_t alignment) {
  CHECK(false,
        "CudaAllocator::Allocate({}, {}): engine built without CUDA support",
        bytes, alignment);
  return core::UnimplementedError("engine built without CUDA support");
}

core::StatusOr<Allocator*> DefaultCudaAllocator(int device_index) {
  return core::UnimplementedError(
      "DefaultCudaAllocator({}): engine built without CUDA support",
      device_index);
}

}  // namespace engine::memory
