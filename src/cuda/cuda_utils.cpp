// CUDA-build implementation of cuda/cuda_utils.h and cuda/cuda_check.h
// (M2-T03; design: docs/design/cuda-backend.md §4–§5). CPU-only builds
// compile cuda_utils_stub.cpp instead (the source-list seam, design §2.2).

#include "cuda/cuda_utils.h"

#include "core/check.h"
#include "core/logging.h"
#include "core/status.h"
#include "cuda/cuda_check.h"

#include <fmt/format.h>

#include <cuda_runtime.h>

#include <string>
#include <string_view>

namespace engine::cuda {
namespace {

// Design §4.2 code table. The switch is deliberately coarse; the error name
// embedded in the message preserves full fidelity.
[[nodiscard]] core::StatusCode CodeFor(cudaError_t err) {
  switch (err) {
    case cudaErrorMemoryAllocation:
      return core::StatusCode::kResourceExhausted;
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidValue:
      return core::StatusCode::kInvalidArgument;
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorNotSupported:
      return core::StatusCode::kUnavailable;
    default:
      return core::StatusCode::kInternal;
  }
}

// Errors that poison the CUDA context (design §4.3): every subsequent call
// returns the same sticky error and device memory is unreliable, so they are
// terminal for the process by policy. The well-known device-side set; not
// exhaustive — an unlisted sticky error is still handled correctly (it maps
// to a Status like any other), it just lacks the marker.
[[nodiscard]] bool IsStickyError(cudaError_t err) {
  switch (err) {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorLaunchTimeout:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorECCUncorrectable:
    case cudaErrorAssert:
      return true;
    default:
      return false;
  }
}

}  // namespace

namespace detail {

std::string DescribeCudaError(cudaError_t err) {
  return fmt::format("{}: {}{}", cudaGetErrorName(err), cudaGetErrorString(err),
                     IsStickyError(err)
                         ? " [sticky: CUDA context is corrupted, error is "
                           "terminal for the process]"
                         : "");
}

core::Status MakeCudaStatus(cudaError_t err, const char* expr, const char* file,
                            int line) {
  return ToStatus(err, fmt::format("{} at {}:{}", expr, file, line));
}

}  // namespace detail

core::Status ToStatus(cudaError_t err, std::string_view what) {
  DCHECK(err != cudaSuccess);
  return {CodeFor(err),
          fmt::format("{}: {}", what, detail::DescribeCudaError(err))};
}

int device_count() {
  // Memoized: the visible device count does not change mid-process. The
  // lambda runs exactly once (thread-safe static init), so the
  // enumeration-failure debug line and the per-device discovery info lines
  // (design §4.4) are logged exactly once.
  static const int count = [] {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
      LOG_DEBUG("cuda",
                "device enumeration failed ({}: {}); treating as 0 "
                "devices",
                cudaGetErrorName(err), cudaGetErrorString(err));
      return 0;
    }
    for (int i = 0; i < n; ++i) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
        continue;  // discovery logging is best-effort
      }
      LOG_INFO("cuda",
               "device {}: {} (compute capability {}.{}, {} SMs, "
               "{} MiB)",
               i, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
               prop.totalGlobalMem / (1024 * 1024));
    }
    return n;
  }();
  return count;
}

core::StatusOr<DeviceProperties> GetDeviceProperties(int index) {
  const int count = device_count();
  if (index < 0 || index >= count) {
    return core::InvalidArgumentError(
        "CUDA device index {} out of range [0, {})", index, count);
  }
  cudaDeviceProp prop{};
  CUDA_RETURN_IF_ERROR(cudaGetDeviceProperties(&prop, index));
  DeviceProperties properties;
  properties.name = prop.name;
  properties.compute_capability_major = prop.major;
  properties.compute_capability_minor = prop.minor;
  properties.multiprocessor_count = prop.multiProcessorCount;
  properties.total_memory_bytes = prop.totalGlobalMem;
  return properties;
}

ScopedSetDevice::ScopedSetDevice(int index) {
  CUDA_CHECK(cudaGetDevice(&previous_));
  CUDA_CHECK(cudaSetDevice(index));
}

ScopedSetDevice::~ScopedSetDevice() { CUDA_CHECK(cudaSetDevice(previous_)); }

}  // namespace engine::cuda
