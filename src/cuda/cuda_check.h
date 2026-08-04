#pragma once

#include "core/check.h"
#include "core/status.h"

#include <cuda_runtime.h>

#include <string>
#include <string_view>

// CUDA error macros & Status mapping (M2-T03; design:
// docs/design/cuda-backend.md §4).
//
// INTERNAL HEADER — includes the CUDA toolkit. It may be included only by
// CUDA-compiled TUs (the .cpp/.cu sources of the toolkit-touching modules
// and CUDA-only tests), never by a public header; on CPU-only builds it does
// not compile. Public, toolkit-free declarations live in cuda/cuda_utils.h
// (design §2.2).
//
// ADR-003's CHECK-vs-Status boundary, specialized for CUDA:
//
//   // Fatal. For programmer errors: misuse of our own wrappers, impossible
//   // arguments, corrupted state.
//   CUDA_CHECK(cudaSetDevice(idx));
//
//   // Recoverable. On failure returns an engine::core::Status with the
//   // failing expression, file:line, and the CUDA error name and
//   // description embedded. The enclosing function must return
//   // Status/StatusOr.
//   CUDA_RETURN_IF_ERROR(cudaMalloc(&p, bytes));
//
// Both evaluate their argument exactly once. Data-, environment-, or
// resource-driven failures (OOM, no device, invalid user-supplied device
// index) return Status; violations of our own invariants CHECK. When in
// doubt inside the backend, prefer CUDA_RETURN_IF_ERROR — a caller can
// always CHECK_OK a Status, but not the reverse.

namespace engine::cuda {

// Maps a CUDA error to a Status (design §4.2): cudaErrorMemoryAllocation →
// kResourceExhausted (device OOM is an operating condition the scheduler
// plans around — distinct from host kOutOfMemory); invalid device/value →
// kInvalidArgument; no device / insufficient driver / not supported →
// kUnavailable; everything else → kInternal. The message always embeds the
// CUDA error name and description (no fidelity lost to the coarse code
// mapping) plus `what`, and marks sticky errors — those that poison the
// CUDA context and are terminal by policy (design §4.3).
// Never returns OK for cudaSuccess — callers branch before calling.
[[nodiscard]] core::Status ToStatus(cudaError_t err, std::string_view what);

namespace detail {

// Builds CUDA_RETURN_IF_ERROR's Status: ToStatus with the failing
// expression and its source location as `what`.
[[nodiscard]] core::Status MakeCudaStatus(cudaError_t err, const char* expr,
                                          const char* file, int line);

// Formats "cudaErrorFoo: <description>[ (sticky ...)]" for CUDA_CHECK's
// failure message.
[[nodiscard]] std::string DescribeCudaError(cudaError_t err);

}  // namespace detail
}  // namespace engine::cuda

// Fatal unless `expr_` — a cudaError_t expression, evaluated exactly once —
// is cudaSuccess. Aborts via the CHECK machinery with the failing
// expression, source location, and CUDA error name/description.
#define CUDA_CHECK(expr_)                                                     \
  do {                                                                        \
    const cudaError_t engine_cuda_check_err_ = (expr_);                       \
    if (engine_cuda_check_err_ != cudaSuccess) [[unlikely]] {                 \
      ::engine::core::detail::CheckFailed(                                    \
          "CUDA_CHECK(" #expr_ ")", __FILE__, __LINE__, __func__,             \
          ::engine::cuda::detail::DescribeCudaError(engine_cuda_check_err_)); \
    }                                                                         \
  } while (0)

// Evaluates `expr_` (a cudaError_t expression) exactly once and returns a
// ToStatus-mapped Status from the enclosing function if it is not
// cudaSuccess. The enclosing function must return Status or StatusOr<T>.
#define CUDA_RETURN_IF_ERROR(expr_)                          \
  do {                                                       \
    const cudaError_t engine_cuda_rie_err_ = (expr_);        \
    if (engine_cuda_rie_err_ != cudaSuccess) [[unlikely]] {  \
      return ::engine::cuda::detail::MakeCudaStatus(         \
          engine_cuda_rie_err_, #expr_, __FILE__, __LINE__); \
    }                                                        \
  } while (0)
