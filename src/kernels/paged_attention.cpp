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

// The batched sequence walk (M9-T07). Batch-major unit space: one (sequence, kv
// head) pair per unit, so unit u decodes to sequence b = u / Hkv and kv head
// hk = u % Hkv. For each unit we synthesize the SAME PagedDecodeArgs the
// standalone entry would build for sequence b and hand `fn` that single kv head
// [hk, hk + 1) — so the arithmetic order per output row is unchanged from a
// standalone PagedDecodeAttentionF32(seq_b) run (the bit-identity guarantee).
void PagedDecodeBatchedUnits(const internal::PagedDecodeBatchedArgs& a,
                             PagedDecodeUnitsFn fn, std::int64_t unit_begin,
                             std::int64_t unit_end) {
  const std::int64_t kv_heads = a.kv_heads;
  const std::int64_t row = a.heads * a.d;  // one sequence's q/out float span
  for (std::int64_t u = unit_begin; u < unit_end; ++u) {
    const std::int64_t b = u / kv_heads;
    const std::int64_t hk = u % kv_heads;
    const internal::PagedDecodeArgs sa{.q = a.q + (b * row),
                                       .k_slab = a.k_slab,
                                       .v_slab = a.v_slab,
                                       .block_table = a.block_tables[b],
                                       .out = a.out + (b * row),
                                       .d = a.d,
                                       .length = a.lengths[b],
                                       .group = a.group,
                                       .block_size = a.block_size,
                                       .block_stride = a.block_stride,
                                       .scale = a.scale};
    fn(sa, hk, hk + 1);
  }
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

void PagedDecodeAttentionBatchedF32(
    const float* q, const float* k_slab, const float* v_slab,
    const std::int32_t* const* block_tables, const std::int64_t* lengths,
    std::int64_t num_seqs, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t d, std::int64_t block_size, std::int64_t block_stride,
    float scale, float* out) {
  if (num_seqs == 0) {
    return;  // B == 0: no-op (an empty decode pass is legal, §7).
  }
  // Same bit-identity contract as the single-sequence entry (§4): a 64-key unit
  // must be a whole number of blocks. CHECKed once (a fixed pool constant).
  CHECK(block_size > 0 && internal::kAttnKb % block_size == 0,
        "PagedDecodeAttentionBatchedF32: block_size {} must be a positive "
        "divisor of kAttnKb {}",
        block_size, internal::kAttnKb);
  static const PagedDecodeUnitsFn fn = Select(kPagedDecodeTable);
  const internal::PagedDecodeBatchedArgs args{.q = q,
                                              .k_slab = k_slab,
                                              .v_slab = v_slab,
                                              .block_tables = block_tables,
                                              .lengths = lengths,
                                              .out = out,
                                              .num_seqs = num_seqs,
                                              .heads = heads,
                                              .kv_heads = kv_heads,
                                              .d = d,
                                              .group = heads / kv_heads,
                                              .block_size = block_size,
                                              .block_stride = block_stride,
                                              .scale = scale};
  // One (sequence, kv head) pair per unit: B·Hkv units, grain 1 (a unit streams
  // one sequence's whole cache for its group's query heads).
  parallel::parallel_for(
      parallel::DefaultPool(), num_seqs * kv_heads, kAttnHeadGrain,
      [&](std::int64_t begin, std::int64_t end) {
        detail::PagedDecodeBatchedUnits(args, fn, begin, end);
      });
}

}  // namespace engine::kernels
