#include "kernels/probe.h"

#include "kernels/dispatch.h"
#include "kernels/internal/probe_impl.h"

namespace engine::kernels {

namespace {

using ProbeFn = Isa (*)();

// The registry idiom every kernel table from M3-T06 on copies: vector slots
// are populated under the same arch guards as their declarations, so a slot
// is null exactly when the variant's TU is absent from the build (design
// §4.2). Omitted designators default to nullptr (KernelTable).
constexpr KernelTable<ProbeFn> kProbeTable = {
    .scalar = &scalar::Probe,
#if defined(ENGINE_ARCH_ARM64)
    .neon = &neon::Probe,
#endif
#if defined(ENGINE_ARCH_X86_64)
    .avx2 = &avx2::Probe,
#endif
};

}  // namespace

Isa DispatchProbe() {
  // Resolved once; dispatch cost after the first call is one indirect call —
  // the pattern §4.2 prescribes for every public kernel entry point.
  static const ProbeFn fn = Select(kProbeTable);
  return fn();
}

}  // namespace engine::kernels
