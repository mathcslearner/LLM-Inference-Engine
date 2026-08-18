#include "kernels/gemm.h"

#include "core/check.h"
#include "kernels/dispatch.h"
#include "kernels/internal/gemm_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"

#include <algorithm>
#include <cstdint>

namespace engine::kernels {

namespace {

// --- Blocking & grain constants (§3.4, §5) ---
//
// Starting points with a stated cache budget, NOT tuned numbers — M12-T02
// tunes them behind these same kernels with a sweep (§1 non-goals, §3.4).
// They change traversal order only: each y[m,n] still sums its K terms in one
// fp32 accumulator ascending-k inside the tile variant, so results are
// bit-identical regardless of the block sizes (§10), exactly as cpu::gemm's
// tiling is.
//
// One parallel unit is one (m-block, panel-block) output tile. `kMc` rows and
// `kNc` panels are sized so a panel-block's bf16 weights (kNc*K*kNr*2 bytes)
// stay hot in L2 across the m-blocks that reuse them, and each variant's
// register-tiled MR-row sub-block reuses its kMc-row A slab across the block's
// panels. Grain 1: each tile is already a substantial unit of work.
constexpr std::int64_t kMc = 64;  // rows per m-block
constexpr std::int64_t kNc = 8;   // panels per panel-block (kNc*kNr cols)
constexpr std::int64_t kGemmTileGrain = 1;

// GEMV (decode): panels per parallel chunk. Each thread streams a disjoint
// contiguous run of packed panels once (bandwidth-bound, §5).
constexpr std::int64_t kGemvPanelGrain = 8;

// --- Per-dtype variant tables ---
using Tile16Fn = void (*)(const float*, std::int64_t, std::int64_t,
                          const std::uint16_t*, const float*, float*,
                          std::int64_t, std::int64_t, std::int64_t,
                          std::int64_t);
using Tile32Fn = void (*)(const float*, std::int64_t, std::int64_t,
                          const float*, const float*, float*, std::int64_t,
                          std::int64_t, std::int64_t, std::int64_t);

constexpr KernelTable<Tile16Fn> kBf16Table = {
    .scalar = &scalar::GemmTileBf16,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::GemmTileBf16,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::GemmTileBf16,
#endif
};

constexpr KernelTable<Tile16Fn> kF16Table = {
    .scalar = &scalar::GemmTileF16,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::GemmTileF16,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::GemmTileF16,
#endif
};

constexpr KernelTable<Tile32Fn> kF32Table = {
    .scalar = &scalar::GemmTileF32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::GemmTileF32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::GemmTileF32,
#endif
};

// Widen-agnostic call over the selected variant, closing over `wp`'s dtype so
// the tile grid and GEMV loops below are written once. The 16-bit and f32
// packed pointers are threaded through as void* by the public entry.
struct TileRunner {
  Tile16Fn f16bits = nullptr;  // bf16 or f16 variant
  Tile32Fn f32 = nullptr;
  const std::uint16_t* wp16 = nullptr;
  const float* wp32 = nullptr;

  void Run(const float* x, std::int64_t k, std::int64_t n, const float* bias,
           float* y, std::int64_t m0, std::int64_t m1, std::int64_t p0,
           std::int64_t p1) const {
    if (f32 != nullptr) {
      f32(x, k, n, wp32, bias, y, m0, m1, p0, p1);
    } else {
      f16bits(x, k, n, wp16, bias, y, m0, m1, p0, p1);
    }
  }
};

[[nodiscard]] TileRunner MakeRunner(const void* wp,
                                    tensor::DataType weight_dtype) {
  TileRunner r;
  switch (weight_dtype) {
    case tensor::DataType::kBFloat16:
      r.f16bits = Select(kBf16Table);
      r.wp16 = static_cast<const std::uint16_t*>(wp);
      return r;
    case tensor::DataType::kFloat16:
      r.f16bits = Select(kF16Table);
      r.wp16 = static_cast<const std::uint16_t*>(wp);
      return r;
    case tensor::DataType::kFloat32:
      r.f32 = Select(kF32Table);
      r.wp32 = static_cast<const float*>(wp);
      return r;
    default:
      CHECK(false, "PackedGemm: weight_dtype must be f32/f16/bf16, got {}",
            tensor::to_string(weight_dtype));
  }
}

// --- Pack routine (§3.2): pure gather + zero-pad, dtype-agnostic ---
template <typename T>
void PackImpl(const T* w, std::int64_t n, std::int64_t k, T* wp) {
  CHECK(w != nullptr && wp != nullptr, "PackWeightPanels: null pointer");
  CHECK(n >= 1 && k >= 1, "PackWeightPanels: n,k must be >= 1 (got {}, {})", n,
        k);
  const std::int64_t panels = PackedPanels(n);
  parallel::parallel_for(parallel::DefaultPool(), panels, /*grain=*/1,
                         [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t p = begin; p < end; ++p) {
                             for (std::int64_t kk = 0; kk < k; ++kk) {
                               T* dst = wp + (((p * k) + kk) * kNr);
                               for (std::int64_t r = 0; r < kNr; ++r) {
                                 const std::int64_t row = (p * kNr) + r;
                                 dst[r] = row < n ? w[(row * k) + kk] : T{0};
                               }
                             }
                           }
                         });
}

}  // namespace

void PackWeightPanels(const std::uint16_t* w, std::int64_t n, std::int64_t k,
                      std::uint16_t* wp) {
  PackImpl(w, n, k, wp);
}

void PackWeightPanels(const float* w, std::int64_t n, std::int64_t k,
                      float* wp) {
  PackImpl(w, n, k, wp);
}

void PackedGemv(const float* x, std::int64_t k, const void* wp,
                tensor::DataType weight_dtype, std::int64_t n,
                const float* bias, float* y) {
  CHECK(x != nullptr && wp != nullptr && y != nullptr, "PackedGemv: null");
  CHECK(k >= 1 && n >= 1, "PackedGemv: k,n must be >= 1 (got {}, {})", k, n);
  const TileRunner runner = MakeRunner(wp, weight_dtype);
  const std::int64_t panels = PackedPanels(n);
  parallel::parallel_for(parallel::DefaultPool(), panels, kGemvPanelGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           runner.Run(x, k, n, bias, y, /*m0=*/0, /*m1=*/1,
                                      begin, end);
                         });
}

void PackedGemm(const float* x, std::int64_t m, std::int64_t k, const void* wp,
                tensor::DataType weight_dtype, std::int64_t n,
                const float* bias, float* y) {
  CHECK(x != nullptr && wp != nullptr && y != nullptr, "PackedGemm: null");
  CHECK(m >= 1 && k >= 1 && n >= 1,
        "PackedGemm: m,k,n must be >= 1 (got {}, {}, {})", m, k, n);
  if (m == 1) {
    PackedGemv(x, k, wp, weight_dtype, n, bias, y);
    return;
  }
  const TileRunner runner = MakeRunner(wp, weight_dtype);
  const std::int64_t panels = PackedPanels(n);
  const std::int64_t m_blocks = (m + kMc - 1) / kMc;
  const std::int64_t p_blocks = (panels + kNc - 1) / kNc;
  const std::int64_t num_tiles = m_blocks * p_blocks;
  parallel::parallel_for(parallel::DefaultPool(), num_tiles, kGemmTileGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t tile = begin; tile < end; ++tile) {
                             const std::int64_t mb = tile / p_blocks;
                             const std::int64_t pb = tile % p_blocks;
                             const std::int64_t m0 = mb * kMc;
                             const std::int64_t m1 = std::min(m0 + kMc, m);
                             const std::int64_t p0 = pb * kNc;
                             const std::int64_t p1 = std::min(p0 + kNc, panels);
                             runner.Run(x, k, n, bias, y, m0, m1, p0, p1);
                           }
                         });
}

namespace detail {

GemmTile16Fn Bf16TileVariant(Isa isa) { return Select(kBf16Table, isa); }
GemmTile16Fn F16TileVariant(Isa isa) { return Select(kF16Table, isa); }
GemmTile32Fn F32TileVariant(Isa isa) { return Select(kF32Table, isa); }

}  // namespace detail

}  // namespace engine::kernels
