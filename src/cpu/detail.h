#pragma once

#include "core/status.h"
#include "tensor/dtype.h"
#include "tensor/half.h"  // IWYU pragma: keep — Widen() widens float16/bfloat16.
#include "tensor/tensor.h"

// Shared validation & widening helpers for the reference `cpu::` ops
// (M5; design: docs/design/model-execution.md §3.3, §13). Private to the `cpu`
// module — only the op `.cpp` files include this; no public header exposes it.
// gemm.cpp predates this header and keeps its own equivalent local helpers
// (its messages are pinned by an 8-case error-path test); the norm/activation
// ops added in M5-T03 share these.
//
// Layering (ADR-002, model-execution.md §2.1): `cpu` links `tensor` only for
// widening — never `kernels`. `Widen` therefore goes through tensor/half.h's
// value-type `operator float()` (bit-exact, constexpr), not kernels/convert.

namespace engine::cpu::detail {

// Widen one stored element to fp32 through tensor/half.h's value-type
// conversions (operator float() — bit-exact, constexpr). Templated on the
// storage type so hot loops carry no per-element dtype branch.
template <typename StoredT>
[[nodiscard]] inline float Widen(StoredT v) {
  return static_cast<float>(v);
}

// Requires a contiguous, rank-`rank` tensor. `op`/`role` name it in error
// messages (the actionability rule, model-execution.md §13). Definedness is a
// programmer error checked at the call site (CHECK), not here.
[[nodiscard]] inline core::Status RequireContiguousRank(const tensor::Tensor& t,
                                                        int rank,
                                                        const char* op,
                                                        const char* role) {
  if (t.shape().rank() != rank) {
    return core::InvalidArgumentError("{}: {} must be rank-{}, got rank-{}", op,
                                      role, rank, t.shape().rank());
  }
  if (!t.is_contiguous()) {
    return core::InvalidArgumentError("{}: {} must be contiguous", op, role);
  }
  return core::OkStatus();
}

// Requires `t` to be exactly kFloat32 (activations / attention scores are fp32,
// model-execution.md §3.1, §3.3).
[[nodiscard]] inline core::Status RequireF32(const tensor::Tensor& t,
                                             const char* op, const char* role) {
  if (t.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError("{}: {} must be f32, got {}", op, role,
                                      tensor::to_string(t.dtype()));
  }
  return core::OkStatus();
}

}  // namespace engine::cpu::detail
