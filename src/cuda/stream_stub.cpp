// CPU-only-build stub for cuda/stream.h (M2-T04; design:
// docs/design/cuda-backend.md §2.2): same symbols, no CUDA toolkit. CUDA
// builds compile stream.cpp instead (the source-list seam).
//
// The factories are the only producers of engaged wrappers and they always
// return Unimplemented here, so no engaged CudaStream or CudaEvent can ever
// exist on a CPU-only build. The remaining members are link-satisfiers:
// destruction only ever sees disengaged wrappers (a no-op), and the
// operational members CHECK — reaching them requires an engaged wrapper,
// which is impossible.

#include "core/check.h"
#include "core/status.h"
#include "cuda/stream.h"

namespace engine::cuda {

// NOLINTBEGIN(readability-convert-member-functions-to-static): these define
// members declared in the shared toolkit-free header; the stub bodies never
// touch member state (they are unreachable), but the signatures are fixed.

core::StatusOr<CudaStream> CudaStream::Create() {
  return core::UnimplementedError(
      "CudaStream::Create(): engine built without CUDA support");
}

void CudaStream::Destroy() noexcept {
  CHECK(stream_ == nullptr, "engaged CudaStream on a CPU-only build");
}

core::Status CudaStream::Synchronize() {
  CHECK(false, "CudaStream::Synchronize(): engine built without CUDA support");
  return core::OkStatus();
}

core::Status CudaStream::WaitEvent(const CudaEvent& /*event*/) {
  CHECK(false, "CudaStream::WaitEvent(): engine built without CUDA support");
  return core::OkStatus();
}

core::StatusOr<CudaEvent> CudaEvent::Timing() {
  return core::UnimplementedError(
      "CudaEvent::Timing(): engine built without CUDA support");
}

core::StatusOr<CudaEvent> CudaEvent::Sync() {
  return core::UnimplementedError(
      "CudaEvent::Sync(): engine built without CUDA support");
}

void CudaEvent::Destroy() noexcept {
  CHECK(event_ == nullptr, "engaged CudaEvent on a CPU-only build");
}

core::Status CudaEvent::Record(const CudaStream& /*stream*/) {
  CHECK(false, "CudaEvent::Record(): engine built without CUDA support");
  return core::OkStatus();
}

core::Status CudaEvent::Synchronize() {
  CHECK(false, "CudaEvent::Synchronize(): engine built without CUDA support");
  return core::OkStatus();
}

core::StatusOr<bool> CudaEvent::Query() const {
  CHECK(false, "CudaEvent::Query(): engine built without CUDA support");
  return false;
}

core::StatusOr<float> CudaEvent::ElapsedMs(const CudaEvent& /*start*/,
                                           const CudaEvent& /*stop*/) {
  CHECK(false, "CudaEvent::ElapsedMs(): engine built without CUDA support");
  return 0.0F;
}

CudaStream& DefaultStream(int device_index) {
  CHECK(false, "DefaultStream({}): engine built without CUDA support",
        device_index);
  // Unreachable: CHECK(false) aborts. A return statement would need an
  // object that cannot exist on this build.
  __builtin_unreachable();
}

// NOLINTEND(readability-convert-member-functions-to-static)

}  // namespace engine::cuda
