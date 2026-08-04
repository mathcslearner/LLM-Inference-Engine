// CPU-only-build stub for memory/pinned_allocator.h (M2-T07; design:
// docs/design/cuda-backend.md §2.2): same symbols, no CUDA toolkit. CUDA
// builds compile pinned_allocator.cpp instead (the source-list seam).
//
// Create is the only producer of instances and returns Unimplemented here,
// so no PinnedCpuAllocator can ever exist on a CPU-only build; Allocate is
// a link-satisfier that cannot be reached.

#include "core/check.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "memory/pinned_allocator.h"

#include <cstddef>

namespace engine::memory {

core::StatusOr<PinnedCpuAllocator> PinnedCpuAllocator::Create() {
  return core::UnimplementedError(
      "PinnedCpuAllocator::Create(): engine built without CUDA support");
}

core::StatusOr<Buffer> PinnedCpuAllocator::Allocate(std::size_t bytes,
                                                    std::size_t alignment) {
  CHECK(false,
        "PinnedCpuAllocator::Allocate({}, {}): engine built without CUDA "
        "support",
        bytes, alignment);
  return core::UnimplementedError("engine built without CUDA support");
}

}  // namespace engine::memory
