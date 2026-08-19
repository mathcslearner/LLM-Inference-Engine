#include "core/status.h"
#include "cpu/ops.h"
#include "kernels/attention.h"
#include "kernels/dispatch.h"
#include "kernels/paged_attention.h"
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
#include <vector>

// Attention benchmark (M6-T04 prefill, M6-T05 decode; design:
// optimized-cpu-execution.md §8, §10). Times the optimized kernels vs the M5
// reference (cpu::attention), recorded in BASELINES.md (no perf target — "time
// vs the reference"). Both paths run threaded through the same DefaultPool; set
// ENGINE_NUM_THREADS / ENGINE_FORCE_ISA to decompose the win. Usage:
//   attention_bench [mode] [T] [H] [Hkv] [d] [iterations]
//     mode = prefill (default) or decode; prefill defaults 2048 32 8 64 5,
//     decode defaults L=2048 32 8 64 200 (T is forced to 1; the T slot is the
//     cache length L).
//
// prefill: the reference materializes an [H·T, L] fp32 score buffer (at the
// defaults H·T·L·4 = 32·2048·2048·4 ≈ 537 MB); the optimized path holds only a
// kAttnKb score row per query. decode: one query per head over the full cache,
// so both the reference ([H·1, L]) and the optimized paths are cheap — each
// timed sample runs a batch of calls so the per-call time is measurable, and
// DecodeAttentionF32 is compared against both cpu::attention(T=1) and
// PrefillAttentionF32(T=1). The decode kernel threads over the Hkv kv heads
// only (§8), so its parallel width is Hkv, not H·qblocks — read the thread
// scaling with that in view.

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

int RunPrefill(std::int64_t t_dim, std::int64_t heads, std::int64_t kv_heads,
               std::int64_t d, int iterations);
int RunDecode(std::int64_t l_dim, std::int64_t heads, std::int64_t kv_heads,
              std::int64_t d, int iterations);
int RunPaged(std::int64_t l_dim, std::int64_t heads, std::int64_t kv_heads,
             std::int64_t d, int iterations);

}  // namespace

int main(int argc, char** argv) {
  // Optional leading mode: "prefill" (default), "decode", or "paged" (M8-T05
  // paged decode vs contiguous decode). If present, the numeric args shift by
  // one.
  int arg = 1;
  bool decode = false;
  bool paged = false;
  if (argc > 1) {
    const std::string mode = argv[1];
    if (mode == "decode") {
      decode = true;
      arg = 2;
    } else if (mode == "paged") {
      paged = true;
      arg = 2;
    } else if (mode == "prefill") {
      arg = 2;
    }
  }
  const auto opt = [&](int i, std::int64_t dflt) -> std::int64_t {
    return argc > i ? std::atoll(argv[i]) : dflt;
  };
  const std::int64_t first = opt(arg, 2048);  // decode/paged: L; prefill: T
  const std::int64_t heads = opt(arg + 1, 32);
  const std::int64_t kv_heads = opt(arg + 2, 8);
  const std::int64_t d = opt(arg + 3, 64);
  const int default_iterations = decode || paged ? 200 : 5;
  const int iterations =
      argc > arg + 4 ? std::atoi(argv[arg + 4]) : default_iterations;
  if (decode) {
    return RunDecode(first, heads, kv_heads, d, iterations);
  }
  if (paged) {
    return RunPaged(first, heads, kv_heads, d, iterations);
  }
  return RunPrefill(first, heads, kv_heads, d, iterations);
}

namespace {

int RunPrefill(std::int64_t t_dim, std::int64_t heads, std::int64_t kv_heads,
               std::int64_t d, int iterations) {
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

int RunDecode(std::int64_t l_dim, std::int64_t heads, std::int64_t kv_heads,
              std::int64_t d, int iterations) {
  // Each timed sample runs a batch of decode calls (each is µs-scale), so the
  // reported per-call time is above timer noise.
  constexpr int kBatch = 100;
  const auto scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));

  fmt::print(
      "Decode attention L={} H={} Hkv={} d={} (T=1), best per-call of {}×{}, "
      "ISA: {}\n\n",
      l_dim, heads, kv_heads, d, iterations, kBatch,
      k::IsaName(k::SelectedIsa()));

  // q/out are the single-token [1, H, d] slice.
  Tensor q = Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));
  (void)ops::fill_normal(q, 0.0, 1.0, 1);
  Tensor key =
      Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32));
  (void)ops::fill_normal(key, 0.0, 1.0, 2);
  Tensor value =
      Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32));
  (void)ops::fill_normal(value, 0.0, 1.0, 3);
  Tensor out = Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));

  const double ref =
      BestSeconds(
          [&] {
            for (int b = 0; b < kBatch; ++b) {
              (void)engine::cpu::attention(q, key, value, scale, out);
            }
          },
          iterations) /
      kBatch;
  const double prefill =
      BestSeconds(
          [&] {
            for (int b = 0; b < kBatch; ++b) {
              k::PrefillAttentionF32(
                  q.data_ptr<float>(), key.data_ptr<float>(),
                  value.data_ptr<float>(), out.data_ptr<float>(),
                  /*t_dim=*/1, heads, kv_heads, d, l_dim, scale);
            }
          },
          iterations) /
      kBatch;
  const double dec = BestSeconds(
                         [&] {
                           for (int b = 0; b < kBatch; ++b) {
                             k::DecodeAttentionF32(
                                 q.data_ptr<float>(), key.data_ptr<float>(),
                                 value.data_ptr<float>(), out.data_ptr<float>(),
                                 heads, kv_heads, d, l_dim, scale);
                           }
                         },
                         iterations) /
                     kBatch;

  fmt::print("{:<28} {:>10.3f} us {:>8.2f}x\n", "cpu::attention (reference)",
             ref * 1e6, 1.0);
  fmt::print("{:<28} {:>10.3f} us {:>8.2f}x\n", "PrefillAttentionF32 (T=1)",
             prefill * 1e6, ref / prefill);
  fmt::print("{:<28} {:>10.3f} us {:>8.2f}x\n", "DecodeAttentionF32", dec * 1e6,
             ref / dec);
  return 0;
}

// Paged decode (M8-T05): PagedDecodeAttentionF32 reading K/V through a block
// table vs the contiguous DecodeAttentionF32 — the per-call cost of the block
// indirection. The logical K/V are materialized into paged slabs
// [num_blocks, Hkv, bs, d] with a sequential block table (blocks laid out in
// order — the common decode-time layout). Both are bit-identical; this is the
// overhead measurement the M8-T07 ≤10% decode-regression bound reads against.
int RunPaged(std::int64_t l_dim, std::int64_t heads, std::int64_t kv_heads,
             std::int64_t d, int iterations) {
  constexpr int kBatch = 100;
  constexpr std::int64_t kBs = 16;  // divides kAttnKb = 64
  const auto scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));

  fmt::print(
      "Paged decode L={} H={} Hkv={} d={} bs={} (T=1), best per-call of "
      "{}×{}, ISA: {}\n\n",
      l_dim, heads, kv_heads, d, kBs, iterations, kBatch,
      k::IsaName(k::SelectedIsa()));

  Tensor q = Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));
  (void)ops::fill_normal(q, 0.0, 1.0, 1);
  Tensor key =
      Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32));
  (void)ops::fill_normal(key, 0.0, 1.0, 2);
  Tensor value =
      Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32));
  (void)ops::fill_normal(value, 0.0, 1.0, 3);
  Tensor out = Unwrap(ops::zeros(Shape{1, heads, d}, DataType::kFloat32));

  // Materialize the logical K/V into paged slabs with a sequential block table.
  const std::int64_t num_blocks = (l_dim + kBs - 1) / kBs;
  const std::int64_t block_stride = kv_heads * kBs * d;
  std::vector<float> k_slab(
      static_cast<std::size_t>(num_blocks * block_stride));
  std::vector<float> v_slab(
      static_cast<std::size_t>(num_blocks * block_stride));
  std::vector<std::int32_t> table(static_cast<std::size_t>(num_blocks));
  for (std::int64_t b = 0; b < num_blocks; ++b) {
    table[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(b);
  }
  for (std::int64_t h = 0; h < kv_heads; ++h) {
    for (std::int64_t s = 0; s < l_dim; ++s) {
      const std::int64_t off =
          ((s / kBs) * block_stride) + (h * kBs * d) + ((s % kBs) * d);
      const float* ks = key.data_ptr<float>() + ((h * l_dim + s) * d);
      const float* vs = value.data_ptr<float>() + ((h * l_dim + s) * d);
      std::copy(ks, ks + d, k_slab.begin() + static_cast<std::ptrdiff_t>(off));
      std::copy(vs, vs + d, v_slab.begin() + static_cast<std::ptrdiff_t>(off));
    }
  }

  const double dec = BestSeconds(
                         [&] {
                           for (int b = 0; b < kBatch; ++b) {
                             k::DecodeAttentionF32(
                                 q.data_ptr<float>(), key.data_ptr<float>(),
                                 value.data_ptr<float>(), out.data_ptr<float>(),
                                 heads, kv_heads, d, l_dim, scale);
                           }
                         },
                         iterations) /
                     kBatch;
  const double pag =
      BestSeconds(
          [&] {
            for (int b = 0; b < kBatch; ++b) {
              k::PagedDecodeAttentionF32(
                  q.data_ptr<float>(), k_slab.data(), v_slab.data(),
                  table.data(), num_blocks, l_dim, heads, kv_heads, d, kBs,
                  block_stride, scale, out.data_ptr<float>());
            }
          },
          iterations) /
      kBatch;

  fmt::print("{:<28} {:>10.3f} us {:>8.2f}x\n", "DecodeAttentionF32 (contig)",
             dec * 1e6, 1.0);
  fmt::print("{:<28} {:>10.3f} us {:>8.2f}x\n", "PagedDecodeAttentionF32",
             pag * 1e6, dec / pag);
  return 0;
}

}  // namespace
