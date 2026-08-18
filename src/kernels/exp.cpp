#include "kernels/dispatch.h"
#include "kernels/internal/exp_impl.h"

// Dispatch glue for the array-form vector exp (M6-T03). `exp` ships no public
// entry point — softmax/SiLU embed the lane helpers directly — so this TU
// carries only the KernelTable and the `detail::ExpF32Variant` test seam the
// ulp sweep uses to reach each ISA's variant on one host.

namespace engine::kernels {

namespace {

using ExpFn = void (*)(const float*, float*, std::int64_t);

constexpr KernelTable<ExpFn> kExpF32Table = {
    .scalar = &scalar::ExpF32,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::ExpF32,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::ExpF32,
#endif
};

}  // namespace

namespace detail {

ExpF32Fn ExpF32Variant(Isa isa) { return Select(kExpF32Table, isa); }

}  // namespace detail

}  // namespace engine::kernels
