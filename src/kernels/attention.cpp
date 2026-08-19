#include "kernels/attention.h"

#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>

namespace engine::kernels {

namespace {

// One (head, query-block) pair is one unit of parallel work (design §5, §8):
// units write disjoint output rows and each query's online-softmax recurrence
// runs wholly within one variant call, so the result is bit-identical across
// thread counts. Grain 1 — a unit (up to kAttnQb queries over L keys) is
// already a substantial body.
constexpr std::int64_t kAttnQGrain = 1;

using PrefillUnitsFn = void (*)(const internal::PrefillArgs&, std::int64_t,
                                std::int64_t);

constexpr KernelTable<PrefillUnitsFn> kPrefillTable = {
    .scalar = &scalar::PrefillUnits,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::PrefillUnits,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::PrefillUnits,
#endif
};

}  // namespace

namespace detail {

PrefillUnitsFn PrefillAttentionVariant(Isa isa) {
  return Select(kPrefillTable, isa);
}

}  // namespace detail

void PrefillAttentionF32(const float* q, const float* k, const float* v,
                         float* out, std::int64_t t_dim, std::int64_t heads,
                         std::int64_t kv_heads, std::int64_t d,
                         std::int64_t l_dim, float scale) {
  static const PrefillUnitsFn fn = Select(kPrefillTable);
  const std::int64_t num_qblocks =
      (t_dim + internal::kAttnQb - 1) / internal::kAttnQb;
  const internal::PrefillArgs args{.q = q,
                                   .k = k,
                                   .v = v,
                                   .out = out,
                                   .t_dim = t_dim,
                                   .heads = heads,
                                   .d = d,
                                   .l_dim = l_dim,
                                   .group = heads / kv_heads,
                                   .past = l_dim - t_dim,
                                   .num_qblocks = num_qblocks,
                                   .scale = scale};
  parallel::parallel_for(
      parallel::DefaultPool(), heads * num_qblocks, kAttnQGrain,
      [&](std::int64_t begin, std::int64_t end) { fn(args, begin, end); });
}

}  // namespace engine::kernels
