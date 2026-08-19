#include "kernels/kv_scatter.h"

#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>
#include <cstring>

// KV scatter (M8-T04; design: paged-kv-cache.md §9.1). Like the M6-T06
// embedding gather, the scatter has no ISA-specific arithmetic — every
// (token, head) row is a `memcpy` of `d` floats — so there is no per-ISA TU:
// this one scalar TU ships and is exercised on every backend (the result is
// Class E, bit-identical across ISAs and thread counts). The forced-scalar
// pass therefore needs no separate registration for this kernel.

namespace engine::kernels {

namespace {

// One token is one unit of parallel work: token `t` (and every head within it)
// writes only into the slabs at `slot_mapping[t]`, disjoint from every other
// token's slots — so the threaded scatter is race-free and the result is
// bit-identical across thread counts (a plain copy has no reduction). The grain
// keeps a whole cache block's worth of tokens on one worker (cache-warm slabs).
constexpr std::int64_t kTokenGrain = 16;

void ScatterRange(const float* src_k, const float* src_v,
                  const std::int64_t* slot_mapping, std::int64_t kv_heads,
                  std::int64_t d, std::int64_t block_size, float* k_slab,
                  float* v_slab, std::int64_t begin, std::int64_t end) {
  const auto bytes = static_cast<std::size_t>(d) * sizeof(float);
  const std::int64_t head_stride = block_size * d;  // bs·d
  for (std::int64_t t = begin; t < end; ++t) {
    const std::int64_t slot = slot_mapping[t];
    const std::int64_t block = slot / block_size;
    const std::int64_t p = slot % block_size;
    // Base of the (block, head 0) tile's row p; advancing one head adds bs·d.
    const std::int64_t dst_base = ((block * kv_heads) * head_stride) + (p * d);
    const std::int64_t src_base = (t * kv_heads) * d;
    for (std::int64_t h = 0; h < kv_heads; ++h) {
      const std::int64_t src_off = src_base + (h * d);
      const std::int64_t dst_off = dst_base + (h * head_stride);
      std::memcpy(k_slab + dst_off, src_k + src_off, bytes);
      std::memcpy(v_slab + dst_off, src_v + src_off, bytes);
    }
  }
}

}  // namespace

void KvScatterF32(const float* src_k, const float* src_v,
                  const std::int64_t* slot_mapping, std::int64_t t_dim,
                  std::int64_t kv_heads, std::int64_t d,
                  std::int64_t block_size, float* k_slab, float* v_slab) {
  parallel::parallel_for(parallel::DefaultPool(), t_dim, kTokenGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           ScatterRange(src_k, src_v, slot_mapping, kv_heads, d,
                                        block_size, k_slab, v_slab, begin, end);
                         });
}

}  // namespace engine::kernels
