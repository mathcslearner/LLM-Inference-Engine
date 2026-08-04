// CUDA-build implementation of tensor/transfer_detail.h (M2-T07; design:
// docs/design/cuda-backend.md §8). CPU-only builds compile transfer_stub.cpp
// instead (the source-list seam, design §2.2).
//
// This TU is why ADR-002 Amendment 3 adds the `tensor → cuda` edge: the
// null-handle contract ("null = the engine default stream", design §6.3)
// and the §4.2 error mapping are cuda-module state and code that must not
// be forked.

#include "core/status.h"
#include "cuda/cuda_check.h"
#include "cuda/stream.h"
#include "cuda/stream_handle.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"
#include "tensor/transfer_detail.h"

#include <fmt/format.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace engine::tensor::detail {

core::Status DeviceCopy(const Tensor& dst, const Tensor& src,
                        cuda::StreamHandle stream, bool synchronize) {
  const bool dst_cuda = dst.device().is_cuda();
  // Explicit kinds rather than cudaMemcpyDefault: the direction is already
  // classified, and the explicit kind keeps the driver's pointer validation.
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if (!dst_cuda) {
    kind = cudaMemcpyDeviceToHost;
  } else if (!src.device().is_cuda()) {
    kind = cudaMemcpyHostToDevice;
  }
  // The destination-relevant device (design §8.1). Its index was validated
  // when the CUDA tensor was allocated, so DefaultStream's CHECK cannot fire.
  const int device_index =
      dst_cuda ? dst.device().index() : src.device().index();
  cudaStream_t resolved = stream.is_default()
                              ? cuda::DefaultStream(device_index).get()
                              : stream.get();
  const std::size_t bytes = static_cast<std::size_t>(dst.numel()) *
                            static_cast<std::size_t>(itemsize(dst.dtype()));
  // On failure, clear the latched last-error slot before returning — the
  // allocator convention (cuda_allocator.cpp, M2-T05 refinement): a
  // non-sticky failure left in the slot would be misattributed by the next
  // kernel launcher's mandatory cudaGetLastError (design §9.2). Sticky
  // errors re-latch on every call, so clearing loses nothing for them.
  if (const cudaError_t err =
          cudaMemcpyAsync(dst.data(), src.data(), bytes, kind, resolved);
      err != cudaSuccess) {
    (void)cudaGetLastError();
    return cuda::ToStatus(
        err, fmt::format("cudaMemcpyAsync of {} bytes ({} -> {})", bytes,
                         src.device().ToString(), dst.device().ToString()));
  }
  if (synchronize) {
    if (const cudaError_t err = cudaStreamSynchronize(resolved);
        err != cudaSuccess) {
      (void)cudaGetLastError();
      return cuda::ToStatus(err, "cudaStreamSynchronize after transfer");
    }
  }
  return core::OkStatus();
}

}  // namespace engine::tensor::detail
