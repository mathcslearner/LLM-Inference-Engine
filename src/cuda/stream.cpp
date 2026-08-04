// CUDA-build implementation of cuda/stream.h (M2-T04; design:
// docs/design/cuda-backend.md §6). CPU-only builds compile stream_stub.cpp
// instead (the source-list seam, design §2.2).

#include "cuda/stream.h"

#include "core/check.h"
#include "core/status.h"
#include "cuda/cuda_check.h"
#include "cuda/cuda_utils.h"

#include <cuda_runtime.h>

#include <mutex>
#include <utility>
#include <vector>

namespace engine::cuda {

core::StatusOr<CudaStream> CudaStream::Create() {
  int device = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&device));
  cudaStream_t stream = nullptr;
  CUDA_RETURN_IF_ERROR(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return CudaStream(stream, device);
}

void CudaStream::Destroy() noexcept {
  if (stream_ == nullptr) {
    return;
  }
  // Drain-on-destruction (design §6.2): a CudaStream never dies with work
  // in flight, which also discharges the buffer-lifetime rule for fixture
  // teardown (design §6.4 rule 3). Destruction cannot return Status; a
  // failure here is sticky/terminal by policy (design §4.3) → CHECK.
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamDestroy(stream_));
  stream_ = nullptr;
  device_index_ = -1;
}

core::Status CudaStream::Synchronize() {
  CHECK(stream_ != nullptr, "Synchronize() on a moved-from CudaStream");
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
  return core::OkStatus();
}

core::Status CudaStream::WaitEvent(const CudaEvent& event) {
  CHECK(stream_ != nullptr, "WaitEvent() on a moved-from CudaStream");
  CUDA_RETURN_IF_ERROR(
      cudaStreamWaitEvent(stream_, event.get(), cudaEventWaitDefault));
  return core::OkStatus();
}

core::StatusOr<CudaEvent> CudaEvent::Create(unsigned int flags, bool timing) {
  cudaEvent_t event = nullptr;
  CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&event, flags));
  return CudaEvent(event, timing);
}

core::StatusOr<CudaEvent> CudaEvent::Timing() {
  return Create(cudaEventDefault, /*timing=*/true);
}

core::StatusOr<CudaEvent> CudaEvent::Sync() {
  return Create(cudaEventDisableTiming, /*timing=*/false);
}

void CudaEvent::Destroy() noexcept {
  if (event_ == nullptr) {
    return;
  }
  // Drain-on-destruction, mirroring CudaStream (design §6.2). Synchronizing
  // a never-recorded event succeeds immediately.
  CUDA_CHECK(cudaEventSynchronize(event_));
  CUDA_CHECK(cudaEventDestroy(event_));
  event_ = nullptr;
}

core::Status CudaEvent::Record(const CudaStream& stream) {
  CHECK(event_ != nullptr, "Record() on a moved-from CudaEvent");
  CUDA_RETURN_IF_ERROR(cudaEventRecord(event_, stream.get()));
  recorded_ = true;
  return core::OkStatus();
}

core::Status CudaEvent::Synchronize() {
  CHECK(event_ != nullptr, "Synchronize() on a moved-from CudaEvent");
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(event_));
  return core::OkStatus();
}

core::StatusOr<bool> CudaEvent::Query() const {
  CHECK(event_ != nullptr, "Query() on a moved-from CudaEvent");
  const cudaError_t err = cudaEventQuery(event_);
  if (err == cudaSuccess) {
    return true;
  }
  if (err == cudaErrorNotReady) {
    return false;
  }
  return ToStatus(err, "cudaEventQuery");
}

core::StatusOr<float> CudaEvent::ElapsedMs(const CudaEvent& start,
                                           const CudaEvent& stop) {
  if (!start.is_timing() || !stop.is_timing()) {
    return core::InvalidArgumentError(
        "ElapsedMs requires Timing() events, got start: {}, stop: {}",
        start.is_timing() ? "Timing" : "Sync",
        stop.is_timing() ? "Timing" : "Sync");
  }
  // Pre-check that both events were ever recorded: a never-recorded event
  // counts as complete for Query()/WaitEvent() (CUDA semantics), so it
  // would pass the completion pre-check below and then make
  // cudaEventElapsedTime fail with cudaErrorInvalidResourceHandle → an
  // opaque kInternal, the exact escape this taxonomy exists to prevent.
  if (!start.recorded_ || !stop.recorded_) {
    return core::FailedPreconditionError(
        "ElapsedMs: event never recorded (start recorded: {}, stop "
        "recorded: {})",
        start.recorded_, stop.recorded_);
  }
  // Pre-check completion so the caller sees FailedPrecondition rather than
  // cudaErrorNotReady mapped to an opaque kInternal (design §4.2 table).
  ASSIGN_OR_RETURN(const bool start_done, start.Query());
  ASSIGN_OR_RETURN(const bool stop_done, stop.Query());
  if (!start_done || !stop_done) {
    return core::FailedPreconditionError(
        "ElapsedMs: event not yet completed (start done: {}, stop done: {})",
        start_done, stop_done);
  }
  float ms = 0.0F;
  CUDA_RETURN_IF_ERROR(cudaEventElapsedTime(&ms, start.get(), stop.get()));
  return ms;
}

CudaStream& DefaultStream(int device_index) {
  const int count = device_count();
  CHECK(device_index >= 0 && device_index < count,
        "DefaultStream({}): device index out of range [0, {})", device_index,
        count);
  // Lazily created and intentionally leaked (design §6.3): the vector and
  // the streams it points to live for the process, so DefaultStream
  // references never dangle and no stream is destroyed during static
  // teardown (when the CUDA runtime may already be gone).
  static std::mutex mutex;
  static std::vector<CudaStream*>& streams = *new std::vector<CudaStream*>();
  const std::lock_guard<std::mutex> lock(mutex);
  if (streams.size() < static_cast<std::size_t>(count)) {
    streams.resize(static_cast<std::size_t>(count), nullptr);
  }
  CudaStream*& slot = streams[static_cast<std::size_t>(device_index)];
  if (slot == nullptr) {
    const ScopedSetDevice guard(device_index);
    core::StatusOr<CudaStream> created = CudaStream::Create();
    // Failure to create a stream on a valid, visible device is not an
    // operating condition the engine plans around → fatal.
    CHECK_OK(created.status());
    slot = new CudaStream(*std::move(created));
  }
  return *slot;
}

}  // namespace engine::cuda
