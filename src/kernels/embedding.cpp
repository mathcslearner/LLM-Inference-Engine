#include "kernels/embedding.h"

#include "core/check.h"
#include "kernels/dispatch.h"
#include "kernels/gemm.h"
#include "kernels/internal/convert_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

// Embedding lookup variants (M6-T06; design: optimized-cpu-execution.md §7).
// Unlike the arithmetic kernels, the gather has no ISA-specific control flow —
// the only per-element work is the fp16/bf16->fp32 widen, which the M3-T06
// conversion variants already provide bit-exact per ISA. So there are no
// per-ISA embedding TUs: this one TU dispatches the widen through the existing
// `convert` KernelTables and does the (inherently scalar) strided/contiguous
// gather itself (design §2 file table amended accordingly). `ENGINE_FORCE_ISA`
// therefore still selects the widen path, keeping the forced-scalar pass
// honest.

namespace engine::kernels {

namespace {

// One token is one unit of parallel work (design §5, kRowGrain): each token's
// output row is an independent gather, written by exactly one worker — so the
// result is bit-identical across thread counts (no split reduction anywhere).
constexpr std::int64_t kRowGrain = 1;

// Chunk width for the strided (packed) gather: a fixed stack staging buffer of
// 16-bit lanes, widened a chunk at a time. The widen is elementwise (Class E),
// so chunking is result-invariant — bit-exact vs a whole-row widen.
constexpr std::int64_t kGatherChunk = 512;

using WidenFn = void (*)(const std::uint16_t*, float*, std::int64_t);

constexpr KernelTable<WidenFn> kBf16ToFp32Table = {
    .scalar = &scalar::Bf16ToFp32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Bf16ToFp32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Bf16ToFp32,
#endif
};

constexpr KernelTable<WidenFn> kFp16ToFp32Table = {
    .scalar = &scalar::Fp16ToFp32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Fp16ToFp32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Fp16ToFp32,
#endif
};

// The dispatched 16-bit widen for a half dtype. A non-half dtype is a
// programmer error (the public entries route f32 down the copy path and reject
// every other dtype upstream in the module).
[[nodiscard]] WidenFn HalfWiden(tensor::DataType dtype) {
  switch (dtype) {
    case tensor::DataType::kBFloat16:
      return Select(kBf16ToFp32Table);
    case tensor::DataType::kFloat16:
      return Select(kFp16ToFp32Table);
    default:
      CHECK(false, "kernels embedding HalfWiden: dtype {} is not a half type",
            static_cast<int>(dtype));
  }
}

// Rows [begin, end) of a row-major table `[V, E]`: a contiguous per-row source,
// so f32 is a straight memcpy and a half row widens in one call (no staging).
void GatherRowMajor(const void* table, tensor::DataType dtype, std::int64_t e,
                    const std::int32_t* ids, std::int64_t begin,
                    std::int64_t end, float* y) {
  if (dtype == tensor::DataType::kFloat32) {
    const auto* src = static_cast<const float*>(table);
    for (std::int64_t t = begin; t < end; ++t) {
      std::memcpy(y + (t * e), src + (static_cast<std::int64_t>(ids[t]) * e),
                  static_cast<std::size_t>(e) * sizeof(float));
    }
    return;
  }
  const WidenFn widen = HalfWiden(dtype);
  const auto* src = static_cast<const std::uint16_t*>(table);
  for (std::int64_t t = begin; t < end; ++t) {
    widen(src + (static_cast<std::int64_t>(ids[t]) * e), y + (t * e), e);
  }
}

// Rows [begin, end) of the packed weight `[PackedPanels(V), E, kNr]`: logical
// row v is strided by kNr (panel v/kNr, lane v%kNr). f32 copies element by
// element; a half row is gathered into a stack buffer a chunk at a time, then
// widened.
void GatherPacked(const void* wp, tensor::DataType dtype, std::int64_t e,
                  const std::int32_t* ids, std::int64_t begin, std::int64_t end,
                  float* y) {
  if (dtype == tensor::DataType::kFloat32) {
    const auto* base = static_cast<const float*>(wp);
    for (std::int64_t t = begin; t < end; ++t) {
      const auto v = static_cast<std::int64_t>(ids[t]);
      const float* col = base + ((v / kNr) * e * kNr) + (v % kNr);
      float* dst = y + (t * e);
      for (std::int64_t k = 0; k < e; ++k) {
        dst[k] = col[k * kNr];
      }
    }
    return;
  }
  const WidenFn widen = HalfWiden(dtype);
  const auto* base = static_cast<const std::uint16_t*>(wp);
  for (std::int64_t t = begin; t < end; ++t) {
    const auto v = static_cast<std::int64_t>(ids[t]);
    const std::uint16_t* col = base + ((v / kNr) * e * kNr) + (v % kNr);
    float* dst = y + (t * e);
    for (std::int64_t off = 0; off < e; off += kGatherChunk) {
      const std::int64_t len = std::min<std::int64_t>(kGatherChunk, e - off);
      std::uint16_t buf[kGatherChunk];
      for (std::int64_t j = 0; j < len; ++j) {
        buf[j] = col[(off + j) * kNr];
      }
      widen(buf, dst + off, len);
    }
  }
}

}  // namespace

void EmbeddingLookupF32(const void* table, tensor::DataType dtype,
                        std::int64_t e, const std::int32_t* ids, std::int64_t t,
                        float* y) {
  parallel::parallel_for(parallel::DefaultPool(), t, kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           GatherRowMajor(table, dtype, e, ids, begin, end, y);
                         });
}

void EmbeddingLookupPackedF32(const void* wp, tensor::DataType dtype,
                              std::int64_t e, const std::int32_t* ids,
                              std::int64_t t, float* y) {
  parallel::parallel_for(parallel::DefaultPool(), t, kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           GatherPacked(wp, dtype, e, ids, begin, end, y);
                         });
}

}  // namespace engine::kernels
