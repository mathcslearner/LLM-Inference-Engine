#include "kernels/internal/probe_impl.h"

// Scalar probe variant: compiled on every platform with no arch flags — the
// always-available fallback and the forced-scalar test target (design
// §4.4 rule 3).

namespace engine::kernels::scalar {

Isa Probe() { return Isa::kScalar; }

}  // namespace engine::kernels::scalar
