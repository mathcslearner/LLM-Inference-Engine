// Device-code anchor: proves .cu sources compile with the pinned
// architectures, device C++20, and device warning flags (M2-T02). M2-T08's
// real .cu sources landed in `kernels`, not here — this module's real
// sources are still host .cpp — so the anchor stays until *this module*
// gains a real .cu source (likely with the cuBLAS/graph work, see
// ROADMAP.md).
namespace engine::cuda {
namespace {

// Trivial device code so nvcc emits SASS for every pinned architecture.
__global__ void anchor_kernel() {}

}  // namespace

// Must be external to anchor the TU. References (never launches) the kernel
// so it is not flagged as unused under -Werror=all-warnings.
void module_anchor() noexcept { static_cast<void>(&anchor_kernel); }

}  // namespace engine::cuda
