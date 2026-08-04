// CUDA-build implementation of kernels/elementwise_detail.h (M2-T08;
// design: docs/design/cuda-backend.md §9). CPU-only builds compile
// elementwise_stub.cpp instead (the source-list seam, design §2.2).
//
// Kernels are naive grid-stride loops (§9.4 — they exist to exercise the
// infrastructure, performance is an M2 non-goal) with internal linkage
// (§9.1). Elementwise math widens each element to float and re-narrows to
// the tensor dtype (§9.3); the device conversion intrinsics round to
// nearest-even, matching the half.h host policy.
//
// The mandatory post-launch cudaGetLastError (§9.2) reads *and clears* the
// thread's last-error slot, so an unharvested error latched by earlier
// unrelated work would be misattributed to this launch. Accepted: every
// engine call site harvests its errors immediately (the allocators and
// transfers even clear the slot on their own failures), and anything that
// slips through is sticky/terminal by policy anyway (§4.3).

#include "core/status.h"
#include "cuda/cuda_check.h"
#include "cuda/cuda_utils.h"
#include "cuda/stream.h"
#include "cuda/stream_handle.h"
#include "kernels/dispatch.h"
#include "kernels/elementwise_detail.h"
#include "kernels/launch.h"
#include "tensor/half.h"
#include "tensor/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace engine::kernels::detail {
namespace {

// The host half types (tensor/half.h) are layout-identical to the device
// types, so a Tensor's storage is reinterpreted at this boundary — and only
// here, inside the .cu (§9.3).
static_assert(sizeof(tensor::float16) == sizeof(__half));
static_assert(sizeof(tensor::bfloat16) == sizeof(__nv_bfloat16));
static_assert(std::is_trivially_copyable_v<tensor::float16>);
static_assert(std::is_trivially_copyable_v<tensor::bfloat16>);

template <typename T>
__device__ inline float WidenToFloat(T value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    static_assert(std::is_same_v<T, float>);
    return value;
  }
}

template <typename T>
__device__ inline T NarrowFromFloat(float value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    static_assert(std::is_same_v<T, float>);
    return value;
  }
}

struct AddOp {
  __device__ static float Apply(float a, float b) { return a + b; }
};

struct MulOp {
  __device__ static float Apply(float a, float b) { return a * b; }
};

template <typename Op, typename T>
__global__ void BinaryKernel(T* dst, const T* a, const T* b, std::int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[i] =
        NarrowFromFloat<T>(Op::Apply(WidenToFloat(a[i]), WidenToFloat(b[i])));
  }
}

template <typename T>
__global__ void ScaleKernel(T* dst, const T* src, float scalar,
                            std::int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[i] = NarrowFromFloat<T>(WidenToFloat(src[i]) * scalar);
  }
}

template <typename SrcT, typename DstT>
__global__ void CastKernel(DstT* dst, const SrcT* src, std::int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dst[i] = NarrowFromFloat<DstT>(WidenToFloat(src[i]));
  }
}

// Steps 2–3 of the §9.2 launcher contract, shared: set the device, resolve
// the stream (null = the engine default stream of the tensors' device,
// §6.3 — the index was validated when the CUDA tensor was allocated, so
// DefaultStream's CHECK cannot fire).
[[nodiscard]] cudaStream_t ResolveStream(const tensor::Tensor& dst,
                                         cuda::StreamHandle stream) {
  return stream.is_default() ? cuda::DefaultStream(dst.device().index()).get()
                             : stream.get();
}

}  // namespace

core::Status LaunchBinary(BinaryOp op, const tensor::Tensor& dst,
                          const tensor::Tensor& a, const tensor::Tensor& b,
                          cuda::StreamHandle stream) {
  cuda::ScopedSetDevice device_guard(dst.device().index());
  const cudaStream_t resolved = ResolveStream(dst, stream);
  const std::int64_t n = dst.numel();
  const LaunchConfig config = LaunchConfig1D(n);
  const char* name = op == BinaryOp::kAdd ? "add" : "mul";
  return DISPATCH_FLOATING_TYPES(dst.dtype(), name, scalar_t, [&] {
    auto* dst_data = static_cast<scalar_t*>(dst.data());
    const auto* a_data = static_cast<const scalar_t*>(a.data());
    const auto* b_data = static_cast<const scalar_t*>(b.data());
    if (op == BinaryOp::kAdd) {
      BinaryKernel<AddOp, scalar_t><<<config.grid, config.block, 0, resolved>>>(
          dst_data, a_data, b_data, n);
    } else {
      BinaryKernel<MulOp, scalar_t><<<config.grid, config.block, 0, resolved>>>(
          dst_data, a_data, b_data, n);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return core::OkStatus();
  });
}

core::Status LaunchScale(const tensor::Tensor& dst, const tensor::Tensor& src,
                         double scalar, cuda::StreamHandle stream) {
  cuda::ScopedSetDevice device_guard(dst.device().index());
  const cudaStream_t resolved = ResolveStream(dst, stream);
  const std::int64_t n = dst.numel();
  const LaunchConfig config = LaunchConfig1D(n);
  // Narrowed once, host-side: per-element math is float regardless of the
  // tensor dtype (§9.3), so double precision would be discarded anyway.
  const float scalar_f = static_cast<float>(scalar);
  return DISPATCH_FLOATING_TYPES(dst.dtype(), "scale", scalar_t, [&] {
    auto* dst_data = static_cast<scalar_t*>(dst.data());
    const auto* src_data = static_cast<const scalar_t*>(src.data());
    ScaleKernel<scalar_t><<<config.grid, config.block, 0, resolved>>>(
        dst_data, src_data, scalar_f, n);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return core::OkStatus();
  });
}

core::Status LaunchCast(const tensor::Tensor& dst, const tensor::Tensor& src,
                        cuda::StreamHandle stream) {
  cuda::ScopedSetDevice device_guard(dst.device().index());
  const cudaStream_t resolved = ResolveStream(dst, stream);
  const std::int64_t n = dst.numel();
  const LaunchConfig config = LaunchConfig1D(n);
  return DISPATCH_FLOATING_TYPES(src.dtype(), "cast", src_t, [&] {
    return DISPATCH_FLOATING_TYPES(dst.dtype(), "cast", dst_t, [&] {
      auto* dst_data = static_cast<dst_t*>(dst.data());
      const auto* src_data = static_cast<const src_t*>(src.data());
      CastKernel<src_t, dst_t>
          <<<config.grid, config.block, 0, resolved>>>(dst_data, src_data, n);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      return core::OkStatus();
    });
  });
}

}  // namespace engine::kernels::detail
