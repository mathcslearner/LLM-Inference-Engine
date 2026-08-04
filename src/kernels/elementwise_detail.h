#pragma once

#include "core/status.h"
#include "cuda/stream_handle.h"
#include "tensor/tensor.h"

#include <cstdint>

// Internal seam between the always-compiled elementwise launchers and the
// toolkit-touching launches (M2-T08; design: docs/design/cuda-backend.md
// §2.2, §9 — the transfer_detail.h pattern from M2-T07).
//
// INTERNAL HEADER — the toolkit-free boundary: validation and the
// numel() == 0 guard live in elementwise.cpp on every configuration (so the
// "kernels validate inputs and return Status" acceptance criterion runs on
// CPU-only CI); CUDA builds compile elementwise.cu (dispatch + launch on
// the resolved stream), CPU-only builds compile elementwise_stub.cpp
// (Unimplemented) — the source-list seam (design §2.2). Not a public
// surface; include only from src/kernels/ TUs.
//
// Preconditions, validated by elementwise.cpp (not re-checked here): all
// tensors defined, contiguous, one shape, numel() > 0, on one common CUDA
// device; dtypes are floating (fp32/fp16/bf16) — equal across operands for
// Binary/Scale, any pair for Cast. Launches ENQUEUE on `stream` (null = the
// engine default stream of the tensors' device, design §6.3) and never
// synchronize (§6.4/§9.2).

namespace engine::kernels::detail {

enum class BinaryOp : std::uint8_t {
  kAdd,
  kMul,
};

// dst[i] = a[i] <op> b[i], computed in float per element (§9.3).
[[nodiscard]] core::Status LaunchBinary(BinaryOp op, const tensor::Tensor& dst,
                                        const tensor::Tensor& a,
                                        const tensor::Tensor& b,
                                        cuda::StreamHandle stream);

// dst[i] = src[i] * scalar; the double is narrowed to float once, host-side.
[[nodiscard]] core::Status LaunchScale(const tensor::Tensor& dst,
                                       const tensor::Tensor& src, double scalar,
                                       cuda::StreamHandle stream);

// dst[i] = convert(src[i]) between any two floating dtypes (widen to float,
// re-narrow round-to-nearest-even — the half.h policy, §9.3).
[[nodiscard]] core::Status LaunchCast(const tensor::Tensor& dst,
                                      const tensor::Tensor& src,
                                      cuda::StreamHandle stream);

}  // namespace engine::kernels::detail
