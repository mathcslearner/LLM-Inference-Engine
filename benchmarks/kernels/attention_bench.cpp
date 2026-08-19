#include "core/status.h"
#include "cpu/ops.h"
#include "kernels/attention.h"
#include "kernels/dispatch.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

// Prefill-attention benchmark (M6-T04; design: optimized-cpu-execution.md §8,
// §10). The M6-T04 obligation: time the optimized prefill attention vs the M5
// reference (cpu::attention) at 2k context, recorded in BASELINES.md (no perf
// target — "time vs the reference"). Both paths run threaded through the same
// DefaultPool; set ENGINE_NUM_THREADS / ENGINE_FORCE_ISA to decompose the
// win. Usage:
//   attention_bench [T] [H] [Hkv] [d] [iterations]   (defaults 2048 32 8 64 5)
//
// Note: the reference materializes an [H·T, L] fp32 score buffer (at the
// defaults H·T·L·4 = 32·2048·2048·4 ≈ 537 MB); the optimized path holds only a
// kAttnKb score row per query. Fine on the 16 GB dev machine — but recorded so
// the number is read with the reference's very different memory profile in
// view (§8).

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

}  // namespace

int main(int argc, char** argv) {
  const std::int64_t t_dim = argc > 1 ? std::atoll(argv[1]) : 2048;
  const std::int64_t heads = argc > 2 ? std::atoll(argv[2]) : 32;
  const std::int64_t kv_heads = argc > 3 ? std::atoll(argv[3]) : 8;
  const std::int64_t d = argc > 4 ? std::atoll(argv[4]) : 64;
  const int iterations = argc > 5 ? std::atoi(argv[5]) : 5;
  const std::int64_t l_dim = t_dim;  // prefill from empty: P = 0, L = T.
  const auto scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));

  fmt::print(
      "Prefill attention T={} H={} Hkv={} d={} (P=0), best of {}, ISA: {}\n\n",
      t_dim, heads, kv_heads, d, iterations, k::IsaName(k::SelectedIsa()));

  Tensor q = Unwrap(ops::zeros(Shape{t_dim, heads, d}, DataType::kFloat32));
  (void)ops::fill_normal(q, 0.0, 1.0, 1);
  Tensor key =
      Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32));
  (void)ops::fill_normal(key, 0.0, 1.0, 2);
  Tensor value =
      Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32));
  (void)ops::fill_normal(value, 0.0, 1.0, 3);
  Tensor out = Unwrap(ops::zeros(Shape{t_dim, heads, d}, DataType::kFloat32));

  const double ref = BestSeconds(
      [&] { (void)engine::cpu::attention(q, key, value, scale, out); },
      iterations);
  const double opt = BestSeconds(
      [&] {
        k::PrefillAttentionF32(q.data_ptr<float>(), key.data_ptr<float>(),
                               value.data_ptr<float>(), out.data_ptr<float>(),
                               t_dim, heads, kv_heads, d, l_dim, scale);
      },
      iterations);

  fmt::print("{:<28} {:>9.4f} s {:>8.2f}x\n", "cpu::attention (reference)", ref,
             1.0);
  fmt::print("{:<28} {:>9.4f} s {:>8.2f}x\n", "PrefillAttentionF32", opt,
             ref / opt);
  return 0;
}
