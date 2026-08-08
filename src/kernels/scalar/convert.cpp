#include "kernels/internal/convert_impl.h"
#include "tensor/half.h"

// Scalar conversion variants: per-element application of the tensor/half.h
// types — bit-exactness against half.h holds by construction, and the
// forced-scalar pass therefore re-validates the goldens through this path.
// The half.h conversions are constexpr integer bit manipulation, so this is
// also reasonably efficient as the production fallback.

namespace engine::kernels::scalar {

void Fp16ToFp32(const std::uint16_t* in, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = static_cast<float>(tensor::float16::from_bits(in[i]));
  }
}

void Fp32ToFp16(const float* in, std::uint16_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = tensor::float16(in[i]).to_bits();
  }
}

void Bf16ToFp32(const std::uint16_t* in, float* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = static_cast<float>(tensor::bfloat16::from_bits(in[i]));
  }
}

void Fp32ToBf16(const float* in, std::uint16_t* out, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    out[i] = tensor::bfloat16(in[i]).to_bits();
  }
}

}  // namespace engine::kernels::scalar
