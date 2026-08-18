#include "kernels/norm.h"

#include "kernels/dispatch.h"
#include "kernels/internal/norm_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>

namespace engine::kernels {

namespace {

// One row is one unit of parallel work (design §5): rows are independent and
// each reduces its e squares in a single fp32 accumulator, so the result is
// bit-identical regardless of how rows partition across threads. Grain 1 — a
// row is a substantial body and sub-row splitting would break the
// single-accumulator reduction, so it is never done.
constexpr std::int64_t kRowGrain = 1;

using RmsNormRowsFn = void (*)(const float*, const float*, float, std::int64_t,
                               std::int64_t, float*);

constexpr KernelTable<RmsNormRowsFn> kRmsNormTable = {
    .scalar = &scalar::RmsNormRows,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::RmsNormRows,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::RmsNormRows,
#endif
};

}  // namespace

namespace detail {

RmsNormRowsFn RmsNormRowsVariant(Isa isa) { return Select(kRmsNormTable, isa); }

}  // namespace detail

void RmsNormF32(const float* x, const float* weight, float eps,
                std::int64_t rows, std::int64_t e, float* y) {
  static const RmsNormRowsFn fn = Select(kRmsNormTable);
  parallel::parallel_for(parallel::DefaultPool(), rows, kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(x + (begin * e), weight, eps, end - begin, e,
                              y + (begin * e));
                         });
}

}  // namespace engine::kernels
