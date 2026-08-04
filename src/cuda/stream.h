#pragma once

#include "core/check.h"
#include "core/status.h"
#include "cuda/stream_handle.h"

#include <utility>

// RAII stream & event wrappers and the per-device engine default stream
// (M2-T04; design: docs/design/cuda-backend.md §6).
//
// This header is toolkit-free (design §2.2): the CUDA handle types appear
// only as pointers to the forward-declared opaque structs CUstream_st /
// CUevent_st — inside CUDA-compiled TUs these are exactly cudaStream_t /
// cudaEvent_t — so any TU on any build can include it. On CPU-only builds
// the out-of-line members come from a stub (stream_stub.cpp): the factories
// return Unimplemented, DefaultStream CHECKs.
//
// Stream model (design §6.1): all engine streams are created
// cudaStreamNonBlocking — the legacy default stream (stream 0) is never
// used for work. Streams are cheap but not free (~µs to create): they are
// long-lived objects owned by long-lived components, never created
// per-operation.
//
// Lifetime rules (design §6.2):
//  - An event may outlive the stream it was recorded on: CUDA guarantees
//    event completion state survives stream destruction, and our stream
//    destructor synchronizes first anyway.
//  - Moved-from wrappers hold a null handle; every member CHECKs
//    engagement (mirroring Tensor's undefined-handle policy). Recording an
//    event on a destroyed stream, or get() on a moved-from wrapper, is
//    programmer error → CHECK.
//  - Thread safety (design §11): follows the Buffer model — move-only,
//    owned by one place; concurrent use of one object requires external
//    synchronization. Distinct streams/events are freely usable from
//    different threads.

struct CUevent_st;  // the pointee of cudaEvent_t

namespace engine::cuda {

class CudaEvent;

// Move-only RAII stream, created cudaStreamNonBlocking on the device that
// is current at construction. Destruction synchronizes the stream first
// (CHECK on failure — destruction can't return Status, and sync errors are
// sticky/terminal by policy, design §4.3), then destroys it: a CudaStream
// never dies with work in flight.
class CudaStream {
 public:
  [[nodiscard]] static core::StatusOr<CudaStream> Create();

  ~CudaStream() { Destroy(); }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept
      : stream_(std::exchange(other.stream_, nullptr)),
        device_index_(std::exchange(other.device_index_, -1)) {}

  // Drains and destroys the current stream (if engaged) first, then steals
  // `other`'s handle — the Buffer move model.
  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      Destroy();
      stream_ = std::exchange(other.stream_, nullptr);
      device_index_ = std::exchange(other.device_index_, -1);
    }
    return *this;
  }

  // Blocks the host until all work enqueued so far completes. The sync
  // point where deferred execution errors surface as Status (design §4.3).
  [[nodiscard]] core::Status Synchronize();

  // Makes *this* stream wait (device-side, non-blocking for the host) until
  // `event` — as last recorded — has completed. If `event` was never
  // recorded, this is a no-op success (CUDA semantics).
  [[nodiscard]] core::Status WaitEvent(const CudaEvent& event);

  // The raw cudaStream_t, for toolkit calls; never null.
  [[nodiscard]] CUstream_st* get() const {
    CHECK(stream_ != nullptr, "get() on a moved-from CudaStream");
    return stream_;
  }

  // Opaque handle for toolkit-free APIs (design §6.3). Non-owning: it must
  // not outlive this stream.
  [[nodiscard]] StreamHandle handle() const { return StreamHandle(get()); }

  // The device the stream was created on.
  [[nodiscard]] int device_index() const {
    CHECK(stream_ != nullptr, "device_index() on a moved-from CudaStream");
    return device_index_;
  }

 private:
  CudaStream(CUstream_st* stream, int device_index)
      : stream_(stream), device_index_(device_index) {}

  // Synchronizes and destroys the stream if engaged; disengages.
  void Destroy() noexcept;

  CUstream_st* stream_ = nullptr;
  int device_index_ = -1;
};

// Move-only RAII event. Two flavors: Timing() (default flags) and Sync()
// (cudaEventDisableTiming — cheaper, for ordering only; ElapsedMs rejects
// it). Destruction synchronizes then destroys, mirroring CudaStream.
class CudaEvent {
 public:
  [[nodiscard]] static core::StatusOr<CudaEvent> Timing();
  [[nodiscard]] static core::StatusOr<CudaEvent> Sync();

  ~CudaEvent() { Destroy(); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept
      : event_(std::exchange(other.event_, nullptr)), timing_(other.timing_) {}

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      Destroy();
      event_ = std::exchange(other.event_, nullptr);
      timing_ = other.timing_;
    }
    return *this;
  }

  // Captures the work enqueued on `stream` so far; overwrites any previous
  // record.
  [[nodiscard]] core::Status Record(const CudaStream& stream);

  // Blocks the host until the event — as last recorded — has completed. A
  // sync point where deferred execution errors surface (design §4.3).
  // Success immediately if the event was never recorded (CUDA semantics).
  [[nodiscard]] core::Status Synchronize();

  // True iff the event — as last recorded — has completed (true if never
  // recorded, CUDA semantics). Does not block.
  [[nodiscard]] core::StatusOr<bool> Query() const;

  // Milliseconds between the two events' completed records. Both events
  // must be Timing() flavor (else InvalidArgument) and completed (else
  // FailedPrecondition — pre-checked so cudaErrorNotReady never surfaces as
  // an opaque kInternal).
  [[nodiscard]] static core::StatusOr<float> ElapsedMs(const CudaEvent& start,
                                                       const CudaEvent& stop);

  // The raw cudaEvent_t, for toolkit calls; never null.
  [[nodiscard]] CUevent_st* get() const {
    CHECK(event_ != nullptr, "get() on a moved-from CudaEvent");
    return event_;
  }

  // True for Timing() events, false for Sync() events.
  [[nodiscard]] bool is_timing() const {
    CHECK(event_ != nullptr, "is_timing() on a moved-from CudaEvent");
    return timing_;
  }

 private:
  CudaEvent(CUevent_st* event, bool timing) : event_(event), timing_(timing) {}

  // Synchronizes and destroys the event if engaged; disengages.
  void Destroy() noexcept;

  CUevent_st* event_ = nullptr;
  bool timing_ = false;
};

// The engine default stream for `device_index`: engine-owned, lazily
// created, intentionally leaked (same pattern and rationale as
// DefaultCpuAllocator, tensor.md §6) — one non-blocking stream per device,
// so call-sites have a stream without plumbing one. Thread-safe (lazy init
// is lock-guarded, design §11). NOT the CUDA legacy default stream.
//
// `device_index` outside [0, device_count()) is programmer error → CHECK
// (validate data-driven indices against device_count() first, the
// ScopedSetDevice stance); stream creation failure on a valid device is
// also fatal. On CPU-only builds every call CHECKs.
[[nodiscard]] CudaStream& DefaultStream(int device_index);

}  // namespace engine::cuda
