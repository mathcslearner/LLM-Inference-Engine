#pragma once

#include "core/status.h"

#include <cstddef>
#include <string>

// Device introspection & device-scope RAII (M2-T03; design:
// docs/design/cuda-backend.md §5).
//
// This header is toolkit-free (design §2.2): it is includable from any TU on
// any build, CUDA or CPU-only. The CUDA error macros (`CUDA_CHECK`,
// `CUDA_RETURN_IF_ERROR`) and `ToStatus` live in cuda/cuda_check.h, which
// includes the CUDA toolkit and is for CUDA-compiled TUs only.
//
// On CPU-only builds (ENGINE_ENABLE_CUDA off) these symbols are provided by
// a stub (cuda_utils_stub.cpp): `device_count()` returns 0, so "is GPU work
// possible" is exactly `device_count() > 0` with no build-flag awareness.

namespace engine::cuda {

// Number of visible CUDA devices. 0 on machines without a GPU or driver AND
// on CPU-only builds. Never fails: enumeration errors (cudaErrorNoDevice
// etc.) are treated as "0 devices" and logged once at debug level. Memoized
// after the first enumeration (the device count does not change
// mid-process); the first successful enumeration logs one info line per
// device (name, compute capability, memory — design §4.4).
[[nodiscard]] int device_count();

struct DeviceProperties {
  std::string name;              // "NVIDIA A100-SXM4-80GB"
  int compute_capability_major;  // 8
  int compute_capability_minor;  // 0
  int multiprocessor_count;
  std::size_t total_memory_bytes;
};

// InvalidArgument if `index` is outside [0, device_count()). On CPU-only
// builds: Unimplemented (design §2.2 taxonomy — the message distinguishes
// "built without CUDA" from "no such device").
[[nodiscard]] core::StatusOr<DeviceProperties> GetDeviceProperties(int index);

// RAII: cudaSetDevice(index) on construction, restores the previously
// current device on destruction. Construction CHECKs — a bad index here is
// programmer error; validate data-driven indices against device_count()
// before entering a scope. On CPU-only builds construction always CHECKs.
// Not copyable/movable; scoped use only.
class ScopedSetDevice {
 public:
  explicit ScopedSetDevice(int index);
  ~ScopedSetDevice();

  ScopedSetDevice(const ScopedSetDevice&) = delete;
  ScopedSetDevice& operator=(const ScopedSetDevice&) = delete;
  ScopedSetDevice(ScopedSetDevice&&) = delete;
  ScopedSetDevice& operator=(ScopedSetDevice&&) = delete;

 private:
  int previous_ = 0;
};

}  // namespace engine::cuda
