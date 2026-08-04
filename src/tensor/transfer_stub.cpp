// CPU-only-build stub for tensor/transfer_detail.h (M2-T07; design:
// docs/design/cuda-backend.md §2.2): same symbol, no CUDA toolkit. CUDA
// builds compile transfer.cpp instead (the source-list seam).
//
// Unreachable through supported factories — CUDA tensors cannot be created
// on a CPU-only build (Tensor::empty returns Unimplemented) — but a Status
// rather than a CHECK, per the design §2.2 taxonomy: a hand-built
// CUDA-tagged tensor asking for a transfer is requesting a CUDA resource.

#include "core/status.h"
#include "cuda/stream_handle.h"
#include "tensor/tensor.h"
#include "tensor/transfer_detail.h"

namespace engine::tensor::detail {

core::Status DeviceCopy(const Tensor& /*dst*/, const Tensor& /*src*/,
                        cuda::StreamHandle /*stream*/, bool /*synchronize*/) {
  return core::UnimplementedError(
      "device copy: engine built without CUDA support");
}

}  // namespace engine::tensor::detail
