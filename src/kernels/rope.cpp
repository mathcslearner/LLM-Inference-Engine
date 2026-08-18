#include "kernels/rope.h"

#include "kernels/dispatch.h"
#include "kernels/internal/rope_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>

namespace engine::kernels {

namespace {

// One token is one unit of parallel work (design §5): every (head, pair)
// rotation under a token reads only that token's slice of x and its own cos/sin
// row, so tokens are independent and any partition is equivalent —
// bit-identical across thread counts.
constexpr std::int64_t kTokenGrain = 1;

using RopeRowsFn = void (*)(float*, std::int64_t, std::int64_t, std::int64_t,
                            const std::int32_t*, const float*, const float*);

constexpr KernelTable<RopeRowsFn> kRopeTable = {
    .scalar = &scalar::RopeRows,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::RopeRows,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::RopeRows,
#endif
};

}  // namespace

namespace detail {

RopeRowsFn RopeRowsVariant(Isa isa) { return Select(kRopeTable, isa); }

}  // namespace detail

void RopeApplyF32(float* x, std::int64_t t, std::int64_t hx, std::int64_t d,
                  const std::int32_t* positions, const float* cos,
                  const float* sin) {
  static const RopeRowsFn fn = Select(kRopeTable);
  const std::int64_t stride = hx * d;
  parallel::parallel_for(parallel::DefaultPool(), t, kTokenGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(x + (begin * stride), end - begin, hx, d,
                              positions + begin, cos, sin);
                         });
}

}  // namespace engine::kernels
