#include "kernels/elementwise.h"

#include "kernels/dispatch.h"
#include "kernels/internal/elementwise_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

namespace engine::kernels {

namespace {

// Elements per chunk (design §3.2: a named per-call-site constant, which
// doubles as the serial threshold — below one full chunk the body runs
// inline on the caller). Class E results are chunking-invariant, so this is
// purely a performance knob: ~128 KiB of fp32 per chunk keeps the per-chunk
// scheduling cost negligible against memory-bound bodies.
constexpr std::int64_t kElementwiseGrain = 32768;

using BinaryFn = void (*)(const float*, const float*, float*, std::int64_t);
using ScaleFn = void (*)(const float*, float, float*, std::int64_t);

// Vector slots are populated under the same arch guards as their
// declarations, so a slot is null exactly when the variant's TU is absent
// from the build (design §4.2).
constexpr KernelTable<BinaryFn> kAddF32Table = {
    .scalar = &scalar::AddF32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::AddF32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::AddF32,
#endif
};

constexpr KernelTable<BinaryFn> kMulF32Table = {
    .scalar = &scalar::MulF32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::MulF32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::MulF32,
#endif
};

constexpr KernelTable<ScaleFn> kScaleF32Table = {
    .scalar = &scalar::ScaleF32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::ScaleF32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::ScaleF32,
#endif
};

}  // namespace

namespace detail {

ElementwiseBinaryFn AddF32Variant(Isa isa) { return Select(kAddF32Table, isa); }
ElementwiseBinaryFn MulF32Variant(Isa isa) { return Select(kMulF32Table, isa); }
ElementwiseScaleFn ScaleF32Variant(Isa isa) {
  return Select(kScaleF32Table, isa);
}

}  // namespace detail

void AddF32(const float* a, const float* b, float* out, std::int64_t n) {
  static const BinaryFn fn = Select(kAddF32Table);
  parallel::parallel_for(parallel::DefaultPool(), n, kElementwiseGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(a + begin, b + begin, out + begin, end - begin);
                         });
}

void MulF32(const float* a, const float* b, float* out, std::int64_t n) {
  static const BinaryFn fn = Select(kMulF32Table);
  parallel::parallel_for(parallel::DefaultPool(), n, kElementwiseGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(a + begin, b + begin, out + begin, end - begin);
                         });
}

void ScaleF32(const float* x, float s, float* out, std::int64_t n) {
  static const ScaleFn fn = Select(kScaleF32Table);
  parallel::parallel_for(parallel::DefaultPool(), n, kElementwiseGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(x + begin, s, out + begin, end - begin);
                         });
}

}  // namespace engine::kernels
