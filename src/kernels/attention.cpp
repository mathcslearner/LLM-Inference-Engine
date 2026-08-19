#include "kernels/attention.h"

#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <algorithm>
#include <cstdint>

namespace engine::kernels {

namespace {

// One (head, query-block) pair is one unit of parallel work (design §5, §8):
// units write disjoint output rows and each query's online-softmax recurrence
// runs wholly within one variant call, so the result is bit-identical across
// thread counts. Grain 1 — a unit (up to kAttnQb queries over L keys) is
// already a substantial body.
constexpr std::int64_t kAttnQGrain = 1;

// One kv head is one unit of decode parallel work (design §5, §8): units write
// the disjoint output rows of their group's query heads, and each kv head's
// online-softmax recurrence runs wholly within one variant call, so the result
// is bit-identical across thread counts. Grain 1 — a unit streams the whole
// cache for g query heads.
constexpr std::int64_t kAttnHeadGrain = 1;

using PrefillUnitsFn = void (*)(const internal::PrefillArgs&, std::int64_t,
                                std::int64_t);
using DecodeUnitsFn = void (*)(const internal::DecodeArgs&, std::int64_t,
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

constexpr KernelTable<DecodeUnitsFn> kDecodeTable = {
    .scalar = &scalar::DecodeUnits,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::DecodeUnits,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::DecodeUnits,
#endif
};

}  // namespace

namespace detail {

PrefillUnitsFn PrefillAttentionVariant(Isa isa) {
  return Select(kPrefillTable, isa);
}

DecodeUnitsFn DecodeAttentionVariant(Isa isa) {
  return Select(kDecodeTable, isa);
}

// The varlen kernel's sequence walk (M9-T06). Sequence-major unit space:
// sequence b owns the contiguous global units [S_b, S_b + H·ceil(T_b/kAttnQb)),
// which the per-sequence PrefillUnitsImpl decodes exactly as it does for a
// standalone call (h = u / nqb, qb = u % nqb). For each sequence overlapping
// the requested [unit_begin, unit_end) chunk we synthesize the SAME PrefillArgs
// the standalone entry would build and hand `fn` the local sub-range — so the
// arithmetic order per output row is unchanged from a standalone run. The O(B)
// scan per chunk is allocation-free and negligible against the attention work
// (B ≤ max_num_seqs).
void PrefillVarlenUnits(const internal::PrefillVarlenArgs& a, PrefillUnitsFn fn,
                        std::int64_t unit_begin, std::int64_t unit_end) {
  const std::int64_t heads = a.heads;
  const std::int64_t d = a.d;
  std::int64_t seq_unit_base = 0;  // S_b: first global unit of sequence b
  for (std::int64_t b = 0; b < a.num_seqs; ++b) {
    const std::int64_t t0 = a.cu_seqlens[b];
    const std::int64_t t_dim = a.cu_seqlens[b + 1] - t0;
    const std::int64_t nqb =
        (t_dim + internal::kAttnQb - 1) / internal::kAttnQb;
    const std::int64_t seq_unit_end = seq_unit_base + (heads * nqb);

    const std::int64_t lo = std::max(unit_begin, seq_unit_base);
    const std::int64_t hi = std::min(unit_end, seq_unit_end);
    if (lo < hi) {
      const std::int64_t l_dim = a.l_dims[b];
      const internal::PrefillArgs args{.q = a.q + (t0 * heads * d),
                                       .k = a.k_seqs[b],
                                       .v = a.v_seqs[b],
                                       .out = a.out + (t0 * heads * d),
                                       .t_dim = t_dim,
                                       .heads = heads,
                                       .d = d,
                                       .l_dim = l_dim,
                                       .group = a.group,
                                       .past = l_dim - t_dim,
                                       .num_qblocks = nqb,
                                       .scale = a.scale};
      fn(args, lo - seq_unit_base, hi - seq_unit_base);
    }
    seq_unit_base = seq_unit_end;
  }
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

void DecodeAttentionF32(const float* q, const float* k, const float* v,
                        float* out, std::int64_t heads, std::int64_t kv_heads,
                        std::int64_t d, std::int64_t l_dim, float scale) {
  static const DecodeUnitsFn fn = Select(kDecodeTable);
  const internal::DecodeArgs args{.q = q,
                                  .k = k,
                                  .v = v,
                                  .out = out,
                                  .d = d,
                                  .l_dim = l_dim,
                                  .group = heads / kv_heads,
                                  .scale = scale};
  parallel::parallel_for(
      parallel::DefaultPool(), kv_heads, kAttnHeadGrain,
      [&](std::int64_t begin, std::int64_t end) { fn(args, begin, end); });
}

void PrefillAttentionVarlenF32(const float* q, const std::int32_t* cu_seqlens,
                               std::int64_t num_seqs,
                               const float* const* k_seqs,
                               const float* const* v_seqs,
                               const std::int64_t* l_dims, float* out,
                               std::int64_t heads, std::int64_t kv_heads,
                               std::int64_t d, float scale) {
  static const PrefillUnitsFn fn = Select(kPrefillTable);
  // Total sequence-major units = Σ_b H·ceil(T_b/kAttnQb) (design §8.4). One
  // (sequence, head, query-block) triple is a unit — same grain-1 work item as
  // the single-sequence prefill entry.
  std::int64_t total_units = 0;
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    const std::int64_t t_dim = cu_seqlens[b + 1] - cu_seqlens[b];
    const std::int64_t nqb =
        (t_dim + internal::kAttnQb - 1) / internal::kAttnQb;
    total_units += heads * nqb;
  }
  const internal::PrefillVarlenArgs args{.q = q,
                                         .cu_seqlens = cu_seqlens,
                                         .k_seqs = k_seqs,
                                         .v_seqs = v_seqs,
                                         .l_dims = l_dims,
                                         .out = out,
                                         .num_seqs = num_seqs,
                                         .heads = heads,
                                         .d = d,
                                         .group = heads / kv_heads,
                                         .scale = scale};
  parallel::parallel_for(parallel::DefaultPool(), total_units, kAttnQGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           detail::PrefillVarlenUnits(args, fn, begin, end);
                         });
}

}  // namespace engine::kernels
