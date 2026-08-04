#pragma once

#include "core/status.h"
#include "cuda/stream_handle.h"
#include "tensor/tensor.h"

// Internal seam for device-involving transfers (M2-T07; design:
// docs/design/cuda-backend.md §2.1, §8).
//
// INTERNAL HEADER — the toolkit-free boundary between the always-compiled
// public entry points (ops::copy's stream overload in ops.cpp, Tensor::to in
// tensor.cpp) and the toolkit-touching implementation: CUDA builds compile
// transfer.cpp (cudaMemcpyAsync on the resolved stream), CPU-only builds
// compile transfer_stub.cpp (Unimplemented) — the source-list seam (design
// §2.2). Not a public surface; include only from src/tensor/ TUs.

namespace engine::tensor::detail {

// Enqueues the H2D / D2H / same-device D2D copy of src's bytes into dst on
// `stream` (null = the engine default stream of the destination-relevant
// device: H2D dst's, D2H src's, D2D the common device — design §8.1) and,
// when `synchronize` is set, blocks until it completes (the sync point where
// deferred execution errors surface as Status, design §4.3).
//
// Preconditions, validated by the public entry points (not re-checked here):
// dst and src are defined and contiguous with identical shape and dtype,
// numel() > 0, at least one is on a CUDA device, and a CUDA pair shares one
// device index. H2H never reaches this seam.
[[nodiscard]] core::Status DeviceCopy(const Tensor& dst, const Tensor& src,
                                      cuda::StreamHandle stream,
                                      bool synchronize);

}  // namespace engine::tensor::detail
