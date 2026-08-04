#pragma once

#include "core/status.h"
#include "cuda/stream_handle.h"
#include "tensor/tensor.h"

// Elementwise GPU kernels: add, mul, scale, cast (M2-T08; design:
// docs/design/cuda-backend.md §9). These four exist to exercise the kernel
// infrastructure — launch helpers, dtype dispatch, validation,
// Status-on-launch-error, the CPU-comparison test pattern — not to be fast
// (naive grid-stride loops; performance is an M2 non-goal, §1).
//
// This header is public and toolkit-free (design §2.2). The uniform
// launcher contract (§9.2):
//
//  - `dst` is pre-allocated by the caller — launchers never allocate
//    (factories do). All operands must be contiguous, share one shape, and
//    live on the same CUDA device → InvalidArgument otherwise (so CPU
//    tensors are InvalidArgument on every build; the "built without CUDA"
//    Unimplemented taxonomy is unreachable through supported factories).
//    Dispatch-unsupported dtypes → Unimplemented. Undefined handle → CHECK.
//  - Calls ENQUEUE on `stream` and return without synchronizing — never a
//    hidden sync (§6.4; completion must be observed via event or stream
//    synchronize before dst is consumed elsewhere). A null StreamHandle
//    means the engine default stream of the tensors' device (§6.3). Launch
//    errors surface immediately as Status (§4.3); asynchronous execution
//    errors surface at the caller's sync point.
//  - numel() == 0 is valid and returns OK without launching.
//  - Elementwise math widens each element to float and re-narrows to the
//    tensor dtype, round-to-nearest-even (§9.3).
//
// Aliasing: for the same-dtype kernels (add/mul/scale), a dst that is
// IDENTICAL to an input (same data pointer) is well-defined — element i
// depends only on the inputs' element i. Partially overlapping views, and
// any dst/src aliasing for cast (element widths differ across the pair),
// are undefined behavior — ops::copy's memcpy rule.

namespace engine::kernels {

// dst[i] = a[i] + b[i]. Same shape, same floating dtype (fp32/fp16/bf16),
// same CUDA device, all contiguous.
[[nodiscard]] core::Status add(const tensor::Tensor& dst,
                               const tensor::Tensor& a, const tensor::Tensor& b,
                               cuda::StreamHandle stream);

// dst[i] = a[i] * b[i]. Requirements as for add.
[[nodiscard]] core::Status mul(const tensor::Tensor& dst,
                               const tensor::Tensor& a, const tensor::Tensor& b,
                               cuda::StreamHandle stream);

// dst[i] = src[i] * scalar. dst/src as for add; `scalar` is narrowed to
// float once on the host (per-element math is float anyway, §9.3).
[[nodiscard]] core::Status scale(const tensor::Tensor& dst,
                                 const tensor::Tensor& src, double scalar,
                                 cuda::StreamHandle stream);

// dst[i] = convert(src[i]) for any pair of {fp32, fp16, bf16} dtypes —
// including equal dtypes (a device deep copy). Other dtypes on either side
// → Unimplemented (integer casts stay CPU-only until a consumer exists,
// §9.1). Same shape, same CUDA device, both contiguous. Unlike ops::cast,
// dst is pre-allocated (the §9.2 launchers-don't-allocate rule).
[[nodiscard]] core::Status cast(const tensor::Tensor& dst,
                                const tensor::Tensor& src,
                                cuda::StreamHandle stream);

}  // namespace engine::kernels
