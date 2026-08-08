#include "kernels/reduce.h"

#include "core/check.h"
#include "kernels/dispatch.h"
#include "kernels/internal/reduce_common.h"
#include "kernels/internal/reduce_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <limits>

namespace engine::kernels {

namespace {

using ReduceChunkFn = float (*)(const float*, std::int64_t);

constexpr KernelTable<ReduceChunkFn> kSumF32Table = {
    .scalar = &scalar::SumF32Chunk,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::SumF32Chunk,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::SumF32Chunk,
#endif
};

constexpr KernelTable<ReduceChunkFn> kMaxF32Table = {
    .scalar = &scalar::MaxF32Chunk,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::MaxF32Chunk,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::MaxF32Chunk,
#endif
};

}  // namespace

namespace detail {

ReduceChunkFn SumF32Variant(Isa isa) { return Select(kSumF32Table, isa); }
ReduceChunkFn MaxF32Variant(Isa isa) { return Select(kMaxF32Table, isa); }

}  // namespace detail

float SumF32(const float* x, std::int64_t n) {
  static const ReduceChunkFn fn = Select(kSumF32Table);
  // parallel_reduce owns the cross-chunk fixed pairwise tree; the identity
  // is returned only for n == 0 (reduce.h: empty sum is +0.0F) and is never
  // combined otherwise.
  return parallel::parallel_reduce<float>(
      parallel::DefaultPool(), n, kReduceGrain, 0.0F,
      [&](std::int64_t begin, std::int64_t end) {
        return fn(x + begin, end - begin);
      },
      [](float a, float b) { return a + b; });
}

float MaxF32(const float* x, std::int64_t n) {
  CHECK(n >= 1, "MaxF32: n must be >= 1 (got {})", n);
  static const ReduceChunkFn fn = Select(kMaxF32Table);
  // The identity is unreachable (n >= 1); -inf documents the intent anyway.
  return parallel::parallel_reduce<float>(
      parallel::DefaultPool(), n, kReduceGrain,
      -std::numeric_limits<float>::infinity(),
      [&](std::int64_t begin, std::int64_t end) {
        return fn(x + begin, end - begin);
      },
      [](float a, float b) { return internal::TotalOrderMax(a, b); });
}

}  // namespace engine::kernels
