// Placeholder translation unit: keeps the engine_kernels archive non-empty
// until the module gains real sources — the SIMD dispatch infrastructure and
// first vectorized kernels land in M3-T05/T06 (see ROADMAP.md).
namespace engine::kernels {

// NOLINTNEXTLINE(misc-use-internal-linkage): must be external to anchor the TU
void module_anchor() noexcept {}

}  // namespace engine::kernels
