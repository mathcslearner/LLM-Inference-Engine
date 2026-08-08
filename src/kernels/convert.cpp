#include "kernels/convert.h"

#include "kernels/dispatch.h"
#include "kernels/internal/convert_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

namespace engine::kernels {

namespace {

// Elements per chunk (design §3.2). Class E — chunking-invariant results —
// so purely a performance knob, matching kElementwiseGrain's reasoning.
constexpr std::int64_t kConvertGrain = 32768;

using NarrowFn = void (*)(const float*, std::uint16_t*, std::int64_t);
using WidenFn = void (*)(const std::uint16_t*, float*, std::int64_t);

constexpr KernelTable<WidenFn> kFp16ToFp32Table = {
    .scalar = &scalar::Fp16ToFp32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Fp16ToFp32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Fp16ToFp32,
#endif
};

constexpr KernelTable<NarrowFn> kFp32ToFp16Table = {
    .scalar = &scalar::Fp32ToFp16,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Fp32ToFp16,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Fp32ToFp16,
#endif
};

constexpr KernelTable<WidenFn> kBf16ToFp32Table = {
    .scalar = &scalar::Bf16ToFp32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Bf16ToFp32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Bf16ToFp32,
#endif
};

constexpr KernelTable<NarrowFn> kFp32ToBf16Table = {
    .scalar = &scalar::Fp32ToBf16,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Fp32ToBf16,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Fp32ToBf16,
#endif
};

// The one place the half types become raw bit patterns for the variants
// (internal/convert_impl.h). float16/bfloat16 are standard-layout wrappers
// around a single std::uint16_t (static_asserted in tensor/half.h), so the
// object representations coincide.
template <typename Half>
const std::uint16_t* AsBits(const Half* p) {
  return reinterpret_cast<const std::uint16_t*>(p);
}
template <typename Half>
std::uint16_t* AsBits(Half* p) {
  return reinterpret_cast<std::uint16_t*>(p);
}

}  // namespace

namespace detail {

ConvertWidenFn Fp16ToFp32Variant(Isa isa) {
  return Select(kFp16ToFp32Table, isa);
}
ConvertNarrowFn Fp32ToFp16Variant(Isa isa) {
  return Select(kFp32ToFp16Table, isa);
}
ConvertWidenFn Bf16ToFp32Variant(Isa isa) {
  return Select(kBf16ToFp32Table, isa);
}
ConvertNarrowFn Fp32ToBf16Variant(Isa isa) {
  return Select(kFp32ToBf16Table, isa);
}

}  // namespace detail

void Fp16ToFp32(const tensor::float16* in, float* out, std::int64_t n) {
  static const WidenFn fn = Select(kFp16ToFp32Table);
  const std::uint16_t* bits = AsBits(in);
  parallel::parallel_for(parallel::DefaultPool(), n, kConvertGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(bits + begin, out + begin, end - begin);
                         });
}

void Fp32ToFp16(const float* in, tensor::float16* out, std::int64_t n) {
  static const NarrowFn fn = Select(kFp32ToFp16Table);
  std::uint16_t* bits = AsBits(out);
  parallel::parallel_for(parallel::DefaultPool(), n, kConvertGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(in + begin, bits + begin, end - begin);
                         });
}

void Bf16ToFp32(const tensor::bfloat16* in, float* out, std::int64_t n) {
  static const WidenFn fn = Select(kBf16ToFp32Table);
  const std::uint16_t* bits = AsBits(in);
  parallel::parallel_for(parallel::DefaultPool(), n, kConvertGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(bits + begin, out + begin, end - begin);
                         });
}

void Fp32ToBf16(const float* in, tensor::bfloat16* out, std::int64_t n) {
  static const NarrowFn fn = Select(kFp32ToBf16Table);
  std::uint16_t* bits = AsBits(out);
  parallel::parallel_for(parallel::DefaultPool(), n, kConvertGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(in + begin, bits + begin, end - begin);
                         });
}

}  // namespace engine::kernels
