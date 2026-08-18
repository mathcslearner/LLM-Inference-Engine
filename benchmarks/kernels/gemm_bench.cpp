#include "core/status.h"
#include "cpu/ops.h"
#include "kernels/dispatch.h"
#include "kernels/gemm.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

#if defined(ENGINE_HAVE_BLAS)
#if defined(__APPLE__)
// Accelerate's SDK headers predate our warning set; they are third-party. The
// CMake target drops -Werror for this opt-in Apple benchmark build so they do
// not fail the build (the default, BLAS-off build keeps full strictness).
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

// Packed-GEMM benchmark (M6-T02; design: optimized-cpu-execution.md §10). The
// M6-T02 acceptance: packed GEMM >= 5x the M5 naive cpu::gemm at 4096x4096x4096
// on the dev machine (advisory), recorded in BASELINES.md, with an
// Accelerate/BLAS sanity number alongside for context (not a target; BLAS is
// benchmark-only and never linked into src/, ADR-002).
//
// Both paths run threaded through the same DefaultPool. Usage:
//   gemm_bench [M] [N] [K] [iterations]     (defaults 4096 4096 4096 3)

namespace {

using Clock = std::chrono::steady_clock;
namespace ops = engine::tensor::ops;
namespace k = engine::kernels;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(engine::core::StatusOr<T> value) {
  if (!value.ok()) {
    fmt::print(stderr, "fatal: {}\n", value.status().ToString());
    std::exit(1);
  }
  return *std::move(value);
}

template <typename Fn>
double BestSeconds(const Fn& fn, int iterations) {
  fn();  // warmup
  double best = 1e100;
  for (int i = 0; i < iterations; ++i) {
    const Clock::time_point start = Clock::now();
    fn();
    const std::chrono::duration<double> elapsed = Clock::now() - start;
    best = std::min(best, elapsed.count());
  }
  return best;
}

void Report(const std::string& name, double seconds, std::int64_t m,
            std::int64_t n, std::int64_t kk, double baseline_seconds) {
  const double gflop = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(kk) / 1e9;
  fmt::print("{:<26} {:>9.3f} s {:>9.2f} GFLOP/s {:>8.2f}x\n", name, seconds,
             gflop / seconds, baseline_seconds / seconds);
}

}  // namespace

int main(int argc, char** argv) {
  const std::int64_t m = argc > 1 ? std::atoll(argv[1]) : 4096;
  const std::int64_t n = argc > 2 ? std::atoll(argv[2]) : 4096;
  const std::int64_t kk = argc > 3 ? std::atoll(argv[3]) : 4096;
  const int iterations = argc > 4 ? std::atoi(argv[4]) : 3;
  fmt::print("GEMM {}x{}x{} (M x N x K), best of {}, ISA: {}\n\n", m, n, kk,
             iterations, k::IsaName(k::SelectedIsa()));

  Tensor x = Unwrap(ops::zeros(Shape{m, kk}, DataType::kFloat32));
  (void)ops::fill_normal(x, 0.0, 1.0, 1);
  Tensor w_f32 = Unwrap(ops::zeros(Shape{n, kk}, DataType::kFloat32));
  (void)ops::fill_normal(w_f32, 0.0, 1.0, 2);
  const Tensor w_bf16 = Unwrap(ops::cast(w_f32, DataType::kBFloat16));

  Tensor y = Unwrap(ops::zeros(Shape{m, n}, DataType::kFloat32));

  // --- Baseline: the M5 naive cpu::gemm (bf16 & f32 weights) ---
  const double naive_bf16 = BestSeconds(
      [&] { (void)engine::cpu::gemm(x, w_bf16, nullptr, y); }, iterations);
  Report("cpu::gemm naive bf16", naive_bf16, m, n, kk, naive_bf16);
  const double naive_f32 = BestSeconds(
      [&] { (void)engine::cpu::gemm(x, w_f32, nullptr, y); }, iterations);
  Report("cpu::gemm naive f32", naive_f32, m, n, kk, naive_bf16);

  // --- Packed GEMM (bf16 & f32 weights) ---
  Tensor wp_bf16 = Unwrap(Tensor::empty(Shape{k::PackedPanels(n), kk, k::kNr},
                                        DataType::kBFloat16, w_bf16.device()));
  k::PackWeightPanels(reinterpret_cast<const std::uint16_t*>(w_bf16.data()), n,
                      kk, reinterpret_cast<std::uint16_t*>(wp_bf16.data()));
  const double packed_bf16 = BestSeconds(
      [&] {
        k::PackedGemm(x.data_ptr<float>(), m, kk, wp_bf16.data(),
                      DataType::kBFloat16, n, nullptr, y.data_ptr<float>());
      },
      iterations);
  Report("PackedGemm bf16", packed_bf16, m, n, kk, naive_bf16);

  Tensor wp_f32 = Unwrap(Tensor::empty(Shape{k::PackedPanels(n), kk, k::kNr},
                                       DataType::kFloat32, w_f32.device()));
  k::PackWeightPanels(w_f32.data_ptr<float>(), n, kk, wp_f32.data_ptr<float>());
  const double packed_f32 = BestSeconds(
      [&] {
        k::PackedGemm(x.data_ptr<float>(), m, kk, wp_f32.data(),
                      DataType::kFloat32, n, nullptr, y.data_ptr<float>());
      },
      iterations);
  Report("PackedGemm f32", packed_f32, m, n, kk, naive_f32);

#if defined(ENGINE_HAVE_BLAS)
  // Sanity context only (§10): C = A * W^T, row-major, f32. Not a target.
  const double blas_f32 = BestSeconds(
      [&] {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(m), static_cast<int>(n),
                    static_cast<int>(kk), 1.0F, x.data_ptr<float>(),
                    static_cast<int>(kk), w_f32.data_ptr<float>(),
                    static_cast<int>(kk), 0.0F, y.data_ptr<float>(),
                    static_cast<int>(n));
      },
      iterations);
  Report("BLAS sgemm f32 (context)", blas_f32, m, n, kk, naive_f32);
#else
  fmt::print(
      "\n(BLAS comparison off; configure -DENGINE_BENCH_BLAS=ON for "
      "context)\n");
#endif

  return 0;
}
