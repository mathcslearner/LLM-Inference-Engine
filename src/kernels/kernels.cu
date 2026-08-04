// Placeholder CUDA translation unit: keeps the engine_kernels archive
// non-empty and proves .cu sources compile with the pinned architectures,
// device C++20, and device warning flags (M2-T02). Replaced by real sources
// from M2-T08 on (see ROADMAP.md).
namespace engine::kernels {
namespace {

// Trivial device code so nvcc emits SASS for every pinned architecture.
__global__ void anchor_kernel() {}

}  // namespace

// Must be external to anchor the TU. References (never launches) the kernel
// so it is not flagged as unused under -Werror=all-warnings.
void module_anchor() noexcept { static_cast<void>(&anchor_kernel); }

}  // namespace engine::kernels
