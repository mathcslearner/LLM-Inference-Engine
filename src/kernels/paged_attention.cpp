#include "kernels/paged_attention.h"

#include "core/check.h"
#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "kernels/internal/paged_attention_common.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>

namespace engine::kernels {

namespace {

// One kv head is one unit of paged-decode parallel work (design §9.2), exactly
// like the contiguous decode kernel: units write the disjoint output rows of
// their group's query heads, and each kv head's online-softmax recurrence runs
// wholly within one variant call, so the result is bit-identical across thread
// counts. Grain 1 — a unit streams the whole cache for g query heads.
constexpr std::int64_t kAttnHeadGrain = 1;

using PagedDecodeUnitsFn = void (*)(const internal::PagedDecodeArgs&,
                                    std::int64_t, std::int64_t);

constexpr KernelTable<PagedDecodeUnitsFn> kPagedDecodeTable = {
    .scalar = &scalar::PagedDecodeUnits,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::PagedDecodeUnits,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::PagedDecodeUnits,
#endif
};

}  // namespace

namespace detail {

PagedDecodeUnitsFn PagedDecodeAttentionVariant(Isa isa) {
  return Select(kPagedDecodeTable, isa);
}

}  // namespace detail

void PagedDecodeAttentionF32(const float* q, const float* k_slab,
                             const float* v_slab,
                             const std::int32_t* block_table,
                             std::int64_t num_blocks, std::int64_t length,
                             std::int64_t heads, std::int64_t kv_heads,
                             std::int64_t d, std::int64_t block_size,
                             std::int64_t block_stride, float scale,
                             float* out) {
  // The bit-identity guarantee (§4) requires a 64-key online-softmax unit to be
  // a whole number of physical blocks — a programmer contract enforced at pool
  // construction, re-asserted here (a violating bs would silently break the
  // "matches the contiguous kernel exactly" promise, ADR-003 CHECK territory).
  CHECK(block_size > 0 && internal::kAttnKb % block_size == 0,
        "PagedDecodeAttentionF32: block_size {} must be a positive divisor of "
        "kAttnKb {}",
        block_size, internal::kAttnKb);
  (void)num_blocks;  // a caller-side bound (ceil(length/bs) <= num_blocks)
  static const PagedDecodeUnitsFn fn = Select(kPagedDecodeTable);
  const internal::PagedDecodeArgs args{.q = q,
                                       .k_slab = k_slab,
                                       .v_slab = v_slab,
                                       .block_table = block_table,
                                       .out = out,
                                       .d = d,
                                       .length = length,
                                       .group = heads / kv_heads,
                                       .block_size = block_size,
                                       .block_stride = block_stride,
                                       .scale = scale};
  parallel::parallel_for(
      parallel::DefaultPool(), kv_heads, kAttnHeadGrain,
      [&](std::int64_t begin, std::int64_t end) { fn(args, begin, end); });
}

}  // namespace engine::kernels
