#include "kernels/kv_gather.h"

#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

// KV gather (M8-T06; design: paged-kv-cache.md §9.3). Like the M8-T04 scatter
// and the M6-T06 embedding gather, there is no ISA-specific arithmetic — every
// (head, block) tile is a `memcpy` of `rows·d` floats — so there is no per-ISA
// TU: this one scalar TU ships and is exercised on every backend (the result is
// Class E, bit-identical across ISAs and thread counts). The forced-scalar pass
// therefore needs no separate registration for this kernel.

namespace engine::kernels {

namespace {

// One unit of parallel work is one (kv head, logical block) tile: it copies the
// block's `rows` valid rows from the paged slab into the head-major output.
// Each (head, block) writes a disjoint destination range, so the threaded
// gather is race-free and the result is bit-identical across thread counts (a
// plain copy has no reduction). Work is flattened over `head-major` unit index
// `u = h·num_blocks + b`; the grain keeps a whole head's blocks on one worker
// (cache-warm output rows).
struct GatherArgs {
  const float* k_slab;
  const float* v_slab;
  const std::int32_t* block_table;
  std::int64_t block_stride;  // Hkv·bs·d — one physical block's float span
  std::int64_t head_stride;   // bs·d     — one kv head within a block
  std::int64_t block_size;    // bs
  std::int64_t length;
  std::int64_t d;
  std::int64_t num_blocks;  // ceil(length / bs)
  float* out_k;
  float* out_v;
};

void GatherRange(const GatherArgs& a, std::int64_t begin, std::int64_t end) {
  for (std::int64_t u = begin; u < end; ++u) {
    const std::int64_t h = u / a.num_blocks;
    const std::int64_t b = u % a.num_blocks;
    const std::int64_t base = b * a.block_size;  // first token of this block
    const std::int64_t rows =
        std::min(a.block_size, a.length - base);  // clip the tail block
    const auto bytes = static_cast<std::size_t>(rows * a.d) * sizeof(float);
    const std::int64_t phys = a.block_table[b];
    const std::int64_t src_off =
        (phys * a.block_stride) + (h * a.head_stride);  // block tile, head h
    const std::int64_t dst_off = (h * a.length * a.d) + (base * a.d);
    std::memcpy(a.out_k + dst_off, a.k_slab + src_off, bytes);
    std::memcpy(a.out_v + dst_off, a.v_slab + src_off, bytes);
  }
}

constexpr std::int64_t kBlockGrain = 4;

}  // namespace

void KvGatherF32(const float* k_slab, const float* v_slab,
                 const std::int32_t* block_table, std::int64_t block_size,
                 std::int64_t length, std::int64_t kv_heads, std::int64_t d,
                 float* out_k, float* out_v) {
  if (length == 0) {
    return;
  }
  const std::int64_t num_blocks = (length + block_size - 1) / block_size;
  const GatherArgs args{
      .k_slab = k_slab,
      .v_slab = v_slab,
      .block_table = block_table,
      .block_stride = kv_heads * block_size * d,
      .head_stride = block_size * d,
      .block_size = block_size,
      .length = length,
      .d = d,
      .num_blocks = num_blocks,
      .out_k = out_k,
      .out_v = out_v,
  };
  parallel::parallel_for(parallel::DefaultPool(), kv_heads * num_blocks,
                         kBlockGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           GatherRange(args, begin, end);
                         });
}

}  // namespace engine::kernels
