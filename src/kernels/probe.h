#pragma once

#include "kernels/dispatch.h"

// Dispatch probe (M3-T05; design: docs/design/cpu-backend.md §4.2, §9).
//
// Test/diagnostic infrastructure, not a compute kernel: DispatchProbe()
// runs through the same KernelTable/Select machinery every real kernel uses
// and reports which ISA variant actually executed, making dispatch — and
// the ENGINE_FORCE_ISA override — observable from tests. Its files are also
// the first instantiation of the §4.4 per-ISA TU layout that every kernel
// ticket from M3-T06 onward copies.

namespace engine::kernels {

// The Isa of the variant that executed. Every variant this build compiled
// is populated in the probe's table, so the result equals SelectedIsa().
// The vector variants compute their answer with real intrinsics, so a
// per-TU flag misconfiguration fails to compile rather than passing
// silently.
[[nodiscard]] Isa DispatchProbe();

}  // namespace engine::kernels
