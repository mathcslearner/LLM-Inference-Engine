#pragma once

#include "core/status.h"
#include "tensor/dtype.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Floating-dtype dispatch for kernel launchers (M2-T08; design:
// docs/design/cuda-backend.md §9.1).
//
// INTERNAL HEADER — includes the CUDA toolkit (the cuda/cuda_check.h
// precedent, design §2.2): the whole point of the macro is to bind an alias
// to the DEVICE type for a dtype (§9.3), and __half/__nv_bfloat16 only
// exist in toolkit headers. It may be included only by CUDA-compiled
// .cu TUs, never by a public header; on CPU-only builds it does not
// compile.
//
// DISPATCH_FLOATING_TYPES(dtype, op_name, alias, body...) evaluates to a
// core::Status: it switches on `dtype`, binds `alias` (a caller-chosen type
// name — parameterized so cast's src×dst dispatch can nest without
// shadowing) to the device type for that dtype, and returns the result of
// invoking `body` — a lambda returning Status, so CUDA_RETURN_IF_ERROR
// works inside it. Non-floating and reserved dtypes return Unimplemented
// naming the op (§9.2; M2 kernels are floating-only — integer variants are
// added when a consumer exists). The trailing-variadic parameter keeps
// top-level commas in the lambda body out of the macro grammar.
//
// Usage, inside a function returning Status:
//
//   return DISPATCH_FLOATING_TYPES(dst.dtype(), "add", scalar_t, [&] {
//     KernelFor<scalar_t><<<config.grid, config.block, 0, stream>>>(...);
//     CUDA_RETURN_IF_ERROR(cudaGetLastError());
//     return ::engine::core::OkStatus();
//   });
//
// `dtype_` may be evaluated twice (once to switch, once in the failure
// message); pass an accessor or a local, not an expression with effects.
#define DISPATCH_FLOATING_TYPES(dtype_, op_name_, alias_, ...)   \
  [&]() -> ::engine::core::Status {                              \
    switch (dtype_) {                                            \
      case ::engine::tensor::DataType::kFloat32: {               \
        using alias_ = float;                                    \
        return (__VA_ARGS__)();                                  \
      }                                                          \
      case ::engine::tensor::DataType::kFloat16: {               \
        using alias_ = __half;                                   \
        return (__VA_ARGS__)();                                  \
      }                                                          \
      case ::engine::tensor::DataType::kBFloat16: {              \
        using alias_ = __nv_bfloat16;                            \
        return (__VA_ARGS__)();                                  \
      }                                                          \
      case ::engine::tensor::DataType::kInt8:                    \
      case ::engine::tensor::DataType::kUInt8:                   \
      case ::engine::tensor::DataType::kInt32:                   \
      case ::engine::tensor::DataType::kInt64:                   \
      case ::engine::tensor::DataType::kBool:                    \
      case ::engine::tensor::DataType::kFP8E4M3:                 \
      case ::engine::tensor::DataType::kInt4:                    \
        break;                                                   \
    }                                                            \
    return ::engine::core::UnimplementedError(                   \
        "{}: no GPU kernel for dtype {} (floating dtypes only: " \
        "fp32/fp16/bf16)",                                       \
        (op_name_), ::engine::tensor::to_string(dtype_));        \
  }()
