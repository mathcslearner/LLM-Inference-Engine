#include "kernels/activation.h"

#include "kernels/dispatch.h"
#include "kernels/internal/activation_impl.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"

#include <cstdint>

namespace engine::kernels {

namespace {

// Elements per chunk (design §5). A pure elementwise map, so chunking never
// changes results — this is purely a performance knob; ~32K fp32 keeps the
// per-chunk scheduling cost negligible against the exp-bound body.
constexpr std::int64_t kSiluGrain = 32768;

using SiluMulFn = void (*)(const float*, const float*, float*, std::int64_t);

constexpr KernelTable<SiluMulFn> kSiluMulTable = {
    .scalar = &scalar::SiluMul,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::SiluMul,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::SiluMul,
#endif
};

}  // namespace

namespace detail {

SiluMulFn SiluMulVariant(Isa isa) { return Select(kSiluMulTable, isa); }

}  // namespace detail

void SiluMulF32(const float* gate, const float* up, float* y, std::int64_t n) {
  static const SiluMulFn fn = Select(kSiluMulTable);
  parallel::parallel_for(parallel::DefaultPool(), n, kSiluGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           fn(gate + begin, up + begin, y + begin, end - begin);
                         });
}

}  // namespace engine::kernels
