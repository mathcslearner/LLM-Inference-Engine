#include "kernels/softmax.h"

#include "kernels/dispatch.h"
#include "kernels/internal/softmax_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>

namespace engine::kernels {

namespace {

// One row is one unit of parallel work (design §5): rows are independent and
// each row's max/sum reductions run in one fp32 accumulator, so the result is
// bit-identical regardless of how rows partition across threads. Grain 1 — a
// row (n elements) is already a substantial body; sub-row splitting would break
// the single-accumulator reduction, so it is never done.
constexpr std::int64_t kRowGrain = 1;

using SoftmaxRowsFn = void (*)(const float*, float*, std::int64_t,
                               std::int64_t);

constexpr KernelTable<SoftmaxRowsFn> kSoftmaxTable = {
    .scalar = &scalar::SoftmaxRows,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::SoftmaxRows,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::SoftmaxRows,
#endif
};

}  // namespace

namespace detail {

SoftmaxRowsFn SoftmaxRowsVariant(Isa isa) { return Select(kSoftmaxTable, isa); }

}  // namespace detail

void SoftmaxF32(const float* x, float* y, std::int64_t rows, std::int64_t n) {
  static const SoftmaxRowsFn fn = Select(kSoftmaxTable);
  parallel::parallel_for(parallel::DefaultPool(), rows, kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(x + (begin * n), y + (begin * n), end - begin, n);
                         });
}

}  // namespace engine::kernels
