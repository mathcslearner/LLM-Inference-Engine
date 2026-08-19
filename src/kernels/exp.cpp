#include "kernels/exp.h"

#include "kernels/dispatch.h"
#include "kernels/internal/exp_impl.h"

#include <cstdint>

// Dispatch glue for the vector exp (M6-T03 polynomial). Through M6 `exp`
// shipped no public entry point — softmax/SiLU embedded the lane helpers
// directly — so this TU carried only the KernelTable and the
// `detail::ExpF32Variant` test seam the ulp sweep uses to reach each ISA's
// variant on one host. M7-T06 adds the public unthreaded `ExpF32` entry
// (kernels/exp.h): the `sampling` module's reference softmax/log-softmax reuse
// the polynomial directly, one exp per sequence row inside the caller's own
// parallel region (so this entry deliberately does no threading of its own).

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

void ExpF32(const float* in, float* out, std::int64_t n) {
  // One indirect call after the first dispatch (memoized), then the whole
  // array runs on the calling thread — no pool, so a caller may invoke this
  // from inside its own parallel_for body (kernels/exp.h).
  static const ExpFn fn = Select(kExpF32Table);
  fn(in, out, n);
}

namespace detail {

ExpF32Fn ExpF32Variant(Isa isa) { return Select(kExpF32Table, isa); }

}  // namespace detail

}  // namespace engine::kernels
