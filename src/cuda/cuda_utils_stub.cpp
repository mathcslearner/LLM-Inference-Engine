// CPU-only-build stub for cuda/cuda_utils.h (M2-T03; design:
// docs/design/cuda-backend.md §2.2): same symbols, no CUDA toolkit. CUDA
// builds compile cuda_utils.cpp instead (the source-list seam).
//
// Behavior per the design's failure taxonomy: introspection returns benign
// values (device_count() == 0, so "is GPU work possible" needs no build-flag
// awareness); requesting a CUDA resource returns Unimplemented with a
// message distinguishing "built without CUDA" from "no such device".

#include "core/check.h"
#include "core/status.h"
#include "cuda/cuda_utils.h"

namespace engine::cuda {

int device_count() { return 0; }

core::StatusOr<DeviceProperties> GetDeviceProperties(int index) {
  return core::UnimplementedError(
      "GetDeviceProperties({}): engine built without CUDA support", index);
}

ScopedSetDevice::ScopedSetDevice(int index) {
  CHECK(false, "ScopedSetDevice({}): engine built without CUDA support", index);
}

// Unreachable (construction always CHECK-fails) but must exist for the
// linker. References previous_, which only the CUDA implementation reads,
// so -Wunused-private-field stays clean in this TU.
ScopedSetDevice::~ScopedSetDevice() { static_cast<void>(previous_); }

}  // namespace engine::cuda
