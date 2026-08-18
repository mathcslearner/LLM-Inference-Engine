#include "kernels/convert.h"
#include "kernels/dispatch.h"
#include "kernels/elementwise.h"
#include "kernels/internal/activation_impl.h"
#include "kernels/internal/convert_impl.h"
#include "kernels/internal/elementwise_impl.h"
#include "kernels/internal/exp_impl.h"
#include "kernels/internal/norm_impl.h"
#include "kernels/internal/reduce_impl.h"
#include "kernels/internal/rope_impl.h"
#include "kernels/internal/softmax_impl.h"
#include "kernels/reduce.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Kernel microbenchmarks, first version (M3-T06; design: cpu-backend.md §9).
// Deliberately simple — monotonic clock, fixed warmup, best-of-N — enough to
// record the first BASELINES.md entry (vectorized conversion vs scalar at 1M
// elements). M12-T01 matures this harness (ISA/thread sweeps, CSV output,
// stability discipline).
//
// Two rows per kernel: the scalar variant and the SelectedIsa() vector
// variant, both called directly as single-threaded chunk bodies — an
// apples-to-apples ISA comparison with no thread-pool noise. Usage:
//   kernels_bench [num_elements] [iterations]

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kWarmupIterations = 3;

// Best-of-N wall time for one call of `fn`.
template <typename Fn>
double BestSeconds(const Fn& fn, int iterations) {
  for (int i = 0; i < kWarmupIterations; ++i) {
    fn();
  }
  double best = 1e100;
  for (int i = 0; i < iterations; ++i) {
    const Clock::time_point start = Clock::now();
    fn();
    const std::chrono::duration<double> elapsed = Clock::now() - start;
    best = std::min(best, elapsed.count());
  }
  return best;
}

// One report line: seconds → ns/element and effective GB/s (bytes = read +
// written per element), plus speedup relative to a scalar-baseline time.
void Report(const std::string& name, double seconds, std::int64_t n,
            std::int64_t bytes_per_elem, double scalar_seconds) {
  const double ns_per_elem = seconds * 1e9 / static_cast<double>(n);
  const double gb_per_s =
      static_cast<double>(n * bytes_per_elem) / seconds / 1e9;
  fmt::print("{:<28} {:>10.3f} ns/elem {:>10.2f} GB/s {:>8.2f}x\n", name,
             ns_per_elem, gb_per_s, scalar_seconds / seconds);
}

// The dispatched variant's display name ("neon", "avx2", or "scalar" when
// forced / on hosts without a vector ISA).
std::string VariantName() {
  return std::string(engine::kernels::IsaName(engine::kernels::SelectedIsa()));
}

}  // namespace

int main(int argc, char** argv) {
  namespace k = engine::kernels;

  const std::int64_t n =
      argc > 1 ? static_cast<std::int64_t>(std::atoll(argv[1])) : (1 << 20);
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  if (n < 1) {
    fmt::print(stderr, "usage: {} [num_elements >= 1] [iterations]\n", argv[0]);
    return 1;
  }
  const auto count = static_cast<std::size_t>(n);
  fmt::print("n = {} elements, best of {} iterations, vector ISA: {}\n\n", n,
             iterations, VariantName());

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
  std::vector<float> a(count);
  std::vector<float> b(count);
  for (std::size_t i = 0; i < count; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }
  std::vector<float> out_f32(count);
  std::vector<std::uint16_t> h16(count);
  std::vector<std::uint16_t> out_u16(count);
  k::scalar::Fp32ToFp16(a.data(), h16.data(), n);

  // Keep results observable so no call can be optimized away.
  volatile float sink = 0.0F;

  struct Row {
    std::string name;
    std::int64_t bytes_per_elem;
    std::function<void()> scalar_fn;
    std::function<void()> vector_fn;  // null when the host has no vector ISA
  };

  const bool have_vector = k::SelectedIsa() != k::Isa::kScalar;
  std::vector<Row> rows;

  const auto add_row = [&](std::string name, std::int64_t bytes,
                           std::function<void()> scalar_fn,
                           std::function<void()> vector_fn) {
    rows.push_back({std::move(name), bytes, std::move(scalar_fn),
                    have_vector ? std::move(vector_fn) : nullptr});
  };

#if defined(ENGINE_ARCH_ARM64)
  namespace vec = engine::kernels::neon;
#elif defined(ENGINE_ARCH_X86_64)
  namespace vec = engine::kernels::avx2;
#else
  namespace vec = engine::kernels::scalar;
#endif

  add_row(
      "AddF32", 12,
      [&] { k::scalar::AddF32(a.data(), b.data(), out_f32.data(), n); },
      [&] { vec::AddF32(a.data(), b.data(), out_f32.data(), n); });
  add_row(
      "MulF32", 12,
      [&] { k::scalar::MulF32(a.data(), b.data(), out_f32.data(), n); },
      [&] { vec::MulF32(a.data(), b.data(), out_f32.data(), n); });
  add_row(
      "ScaleF32", 8,
      [&] { k::scalar::ScaleF32(a.data(), 1.5F, out_f32.data(), n); },
      [&] { vec::ScaleF32(a.data(), 1.5F, out_f32.data(), n); });
  add_row(
      "SumF32 (chunk)", 4, [&] { sink = k::scalar::SumF32Chunk(a.data(), n); },
      [&] { sink = vec::SumF32Chunk(a.data(), n); });
  add_row(
      "MaxF32 (chunk)", 4, [&] { sink = k::scalar::MaxF32Chunk(a.data(), n); },
      [&] { sink = vec::MaxF32Chunk(a.data(), n); });
  add_row(
      "Fp16ToFp32", 6,
      [&] { k::scalar::Fp16ToFp32(h16.data(), out_f32.data(), n); },
      [&] { vec::Fp16ToFp32(h16.data(), out_f32.data(), n); });
  add_row(
      "Fp32ToFp16", 6,
      [&] { k::scalar::Fp32ToFp16(a.data(), out_u16.data(), n); },
      [&] { vec::Fp32ToFp16(a.data(), out_u16.data(), n); });
  add_row(
      "Bf16ToFp32", 6,
      [&] { k::scalar::Bf16ToFp32(h16.data(), out_f32.data(), n); },
      [&] { vec::Bf16ToFp32(h16.data(), out_f32.data(), n); });
  add_row(
      "Fp32ToBf16", 6,
      [&] { k::scalar::Fp32ToBf16(a.data(), out_u16.data(), n); },
      [&] { vec::Fp32ToBf16(a.data(), out_u16.data(), n); });

  // M6-T03 flat kernels (per-element bodies, benched single-threaded per ISA).
  // ExpF32 exercises the new polynomial; SiluMulF32 is exp-bound (it embeds
  // it).
  add_row(
      "ExpF32", 8, [&] { k::scalar::ExpF32(a.data(), out_f32.data(), n); },
      [&] { vec::ExpF32(a.data(), out_f32.data(), n); });
  add_row(
      "SiluMulF32", 12,
      [&] { k::scalar::SiluMul(a.data(), b.data(), out_f32.data(), n); },
      [&] { vec::SiluMul(a.data(), b.data(), out_f32.data(), n); });

  for (const Row& row : rows) {
    const double scalar_seconds = BestSeconds(row.scalar_fn, iterations);
    Report(row.name + " scalar", scalar_seconds, n, row.bytes_per_elem,
           scalar_seconds);
    if (row.vector_fn) {
      const double vector_seconds = BestSeconds(row.vector_fn, iterations);
      Report(row.name + " " + VariantName(), vector_seconds, n,
             row.bytes_per_elem, scalar_seconds);
    }
    sink = sink + out_f32[0] + static_cast<float>(out_u16[0]);
  }

  // M6-T03 shaped kernels (row/token-parallel bodies, benched single-threaded
  // per ISA over the flat buffer reshaped to a model-like hidden size). ns/elem
  // is over the total element count so the throughput is comparable across
  // rows.
  constexpr std::int64_t kE = 4096;  // a model-like hidden dim.
  if (n >= kE) {
    const std::int64_t rows_c = n / kE;
    const std::int64_t total = rows_c * kE;
    std::vector<float> w(static_cast<std::size_t>(kE), 1.0F);

    const auto shaped = [&](const std::string& name, std::int64_t bytes,
                            const std::function<void()>& scalar_fn,
                            const std::function<void()>& vector_fn) {
      const double ss = BestSeconds(scalar_fn, iterations);
      Report(name + " scalar", ss, total, bytes, ss);
      if (have_vector) {
        const double vs = BestSeconds(vector_fn, iterations);
        Report(name + " " + VariantName(), vs, total, bytes, ss);
      }
      sink = sink + out_f32[0];
    };

    shaped(
        "RmsNormF32", 8,
        [&] {
          k::scalar::RmsNormRows(a.data(), w.data(), 1e-5F, rows_c, kE,
                                 out_f32.data());
        },
        [&] {
          vec::RmsNormRows(a.data(), w.data(), 1e-5F, rows_c, kE,
                           out_f32.data());
        });
    shaped(
        "SoftmaxF32", 8,
        [&] { k::scalar::SoftmaxRows(a.data(), out_f32.data(), rows_c, kE); },
        [&] { vec::SoftmaxRows(a.data(), out_f32.data(), rows_c, kE); });

    // RoPE: reshape to [t, hx, d] with d = 128, hx = 8 (stride 1024).
    constexpr std::int64_t kD = 128;
    constexpr std::int64_t kHx = 8;
    constexpr std::int64_t kStride = kHx * kD;
    if (n >= kStride) {
      const std::int64_t t = n / kStride;
      std::vector<std::int32_t> pos(static_cast<std::size_t>(t));
      for (std::int64_t i = 0; i < t; ++i) {
        pos[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
      }
      // cos²+sin² = 1 keeps the in-place rotation norm-preserving, so repeated
      // rotations over the timing loop neither overflow nor underflow.
      const float kC = 0.6F;
      const float kS = 0.8F;
      std::vector<float> cos(static_cast<std::size_t>(t * (kD / 2)), kC);
      std::vector<float> sin(static_cast<std::size_t>(t * (kD / 2)), kS);
      std::vector<float> rx(a.begin(), a.begin() + (t * kStride));
      shaped(
          "RopeApplyF32", 4,
          [&] {
            k::scalar::RopeRows(rx.data(), t, kHx, kD, pos.data(), cos.data(),
                                sin.data());
          },
          [&] {
            vec::RopeRows(rx.data(), t, kHx, kD, pos.data(), cos.data(),
                          sin.data());
          });
      sink = sink + rx[0];
    }
  }

  return sink < 1e100 ? 0 : 1;  // Use sink so it cannot be dropped.
}
