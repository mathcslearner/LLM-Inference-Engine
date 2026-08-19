#include "cpu/ops.h"
#include "kernels/attention.h"
#include "kernels/dispatch.h"
#include "kernels/internal/attention_common.h"
#include "kernels/internal/attention_impl.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <vector>

// Varlen (ragged-batch) prefill-attention kernel (M9-T06; design:
// docs/design/scheduler-runtime.md §8.4). The kernel loops the **unchanged**
// per-sequence PrefillAttentionF32 recurrence over the sequences of a ragged
// batch delimited by cu_seqlens. The correctness spine is that each sequence's
// output is **bit-identical** to a standalone PrefillAttentionF32 run on that
// sequence (M9-T06 acceptance) — asserted with an exact (0-tolerance) allclose
// per sequence. A secondary Class-T check vs cpu::attention (the oracle) ties
// the batch to the reference chain. Registered SCALAR_PASS so the forced-scalar
// pass exercises the shipped scalar bytes.
namespace engine::kernels {
namespace {

namespace ops = engine::tensor::ops;
using engine::core::StatusOr;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

template <typename T>
[[nodiscard]] T Unwrap(StatusOr<T> value) {
  EXPECT_TRUE(value.ok()) << value.status().ToString();
  return *std::move(value);
}

constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-4;

// One sequence of a batch: q [T, H, d], k/v [Hkv, L, d] with L = P + T, seeded.
// Mirrors attention_kernel_test's Problem so the standalone-vs-batch comparison
// is 1:1.
struct Seq {
  Tensor q;
  Tensor k;
  Tensor v;
  std::int64_t t_dim;
  std::int64_t l_dim;
};

[[nodiscard]] Seq MakeSeq(std::int64_t t_dim, std::int64_t kv_heads,
                          std::int64_t heads, std::int64_t d, std::int64_t past,
                          double std_dev, std::uint64_t seed) {
  const std::int64_t l_dim = past + t_dim;
  Seq s{.q = Unwrap(ops::zeros(Shape{t_dim, heads, d}, DataType::kFloat32)),
        .k = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
        .v = Unwrap(ops::zeros(Shape{kv_heads, l_dim, d}, DataType::kFloat32)),
        .t_dim = t_dim,
        .l_dim = l_dim};
  EXPECT_TRUE(ops::fill_normal(s.q, 0.0, std_dev, seed).ok());
  EXPECT_TRUE(ops::fill_normal(s.k, 0.0, std_dev, seed + 1).ok());
  EXPECT_TRUE(ops::fill_normal(s.v, 0.0, std_dev, seed + 2).ok());
  return s;
}

// The standalone (single-sequence) kernel output for one sequence — the oracle
// the batched output must match bit-for-bit per sequence.
[[nodiscard]] Tensor Standalone(const Seq& s, std::int64_t heads,
                                std::int64_t kv_heads, std::int64_t d,
                                float scale) {
  Tensor out = Unwrap(ops::zeros(Shape{s.t_dim, heads, d}, DataType::kFloat32));
  PrefillAttentionF32(s.q.data_ptr<float>(), s.k.data_ptr<float>(),
                      s.v.data_ptr<float>(), out.data_ptr<float>(), s.t_dim,
                      heads, kv_heads, d, s.l_dim, scale);
  return out;
}

// The cpu::attention reference (the M5 oracle) for one sequence.
[[nodiscard]] Tensor Reference(const Seq& s, std::int64_t heads, std::int64_t d,
                               float scale) {
  Tensor out = Unwrap(ops::zeros(Shape{s.t_dim, heads, d}, DataType::kFloat32));
  EXPECT_TRUE(engine::cpu::attention(s.q, s.k, s.v, scale, out).ok());
  return out;
}

// Runs the varlen kernel over a batch of sequences and returns the flattened
// [ΣT, H, d] output (NaN-poisoned first, so an un-written row is caught).
struct Batch {
  Tensor q_flat;                     // [ΣT, H, d]
  Tensor out_flat;                   // [ΣT, H, d]
  std::vector<std::int32_t> cu;      // [B + 1]
  std::vector<const float*> k_ptrs;  // [B]
  std::vector<const float*> v_ptrs;  // [B]
  std::vector<std::int64_t> l_dims;  // [B]
};

[[nodiscard]] Batch RunVarlen(const std::vector<Seq>& seqs, std::int64_t heads,
                              std::int64_t kv_heads, std::int64_t d,
                              float scale) {
  std::int64_t total_t = 0;
  for (const Seq& s : seqs) {
    total_t += s.t_dim;
  }
  Batch b{.q_flat =
              Unwrap(ops::zeros(Shape{total_t, heads, d}, DataType::kFloat32)),
          .out_flat =
              Unwrap(ops::zeros(Shape{total_t, heads, d}, DataType::kFloat32)),
          .cu = {},
          .k_ptrs = {},
          .v_ptrs = {},
          .l_dims = {}};
  // Poison the output so any row the kernel fails to overwrite is a NaN
  // mismatch.
  auto* out_data = b.out_flat.data_ptr<float>();
  for (std::int64_t i = 0; i < total_t * heads * d; ++i) {
    out_data[i] = std::numeric_limits<float>::quiet_NaN();
  }
  // Flatten q row-by-row into the contiguous [ΣT, H, d] buffer and collect the
  // per-sequence K/V pointers + cu_seqlens.
  auto* q_data = b.q_flat.data_ptr<float>();
  std::int32_t offset = 0;
  b.cu.reserve(seqs.size() + 1);
  b.k_ptrs.reserve(seqs.size());
  b.v_ptrs.reserve(seqs.size());
  b.l_dims.reserve(seqs.size());
  b.cu.push_back(0);
  for (const Seq& s : seqs) {
    const std::int64_t rows = s.t_dim * heads * d;
    const float* src = s.q.data_ptr<float>();
    std::copy(src, src + rows,
              q_data + (static_cast<std::int64_t>(offset) * heads * d));
    offset += static_cast<std::int32_t>(s.t_dim);
    b.cu.push_back(offset);
    b.k_ptrs.push_back(s.k.data_ptr<float>());
    b.v_ptrs.push_back(s.v.data_ptr<float>());
    b.l_dims.push_back(s.l_dim);
  }
  PrefillAttentionVarlenF32(
      b.q_flat.data_ptr<float>(), b.cu.data(),
      static_cast<std::int64_t>(seqs.size()), b.k_ptrs.data(), b.v_ptrs.data(),
      b.l_dims.data(), b.out_flat.data_ptr<float>(), heads, kv_heads, d, scale);
  return b;
}

// Extract sequence b's [T_b, H, d] rows from a flattened [ΣT, H, d] batch
// output.
[[nodiscard]] Tensor SliceSeq(const Batch& b, std::size_t idx,
                              std::int64_t heads, std::int64_t d) {
  const std::int64_t t0 = b.cu[idx];
  const std::int64_t t_dim = b.cu[idx + 1] - t0;
  Tensor out = Unwrap(ops::zeros(Shape{t_dim, heads, d}, DataType::kFloat32));
  const float* src = b.out_flat.data_ptr<float>() + (t0 * heads * d);
  std::copy(src, src + (t_dim * heads * d), out.data_ptr<float>());
  return out;
}

[[nodiscard]] float Scale(std::int64_t d) {
  return static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));
}

// The headline acceptance case: a batch of {5, 64, 129} matches three
// standalone PrefillAttentionF32 runs **exactly** (bit-identical per sequence).
TEST(VarlenAttentionKernelTest, BatchMatchesStandalonePerSequence) {
  constexpr std::int64_t kHeads = 4;
  constexpr std::int64_t kKvHeads = 2;
  constexpr std::int64_t kD = 64;
  const float scale = Scale(kD);
  std::vector<Seq> seqs;
  seqs.push_back(MakeSeq(5, kKvHeads, kHeads, kD, 0, 1.0, 100));
  seqs.push_back(MakeSeq(64, kKvHeads, kHeads, kD, 0, 1.0, 200));
  seqs.push_back(MakeSeq(129, kKvHeads, kHeads, kD, 0, 1.0, 300));

  const Batch b = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    const Tensor solo = Standalone(seqs[i], kHeads, kKvHeads, kD, scale);
    const Tensor batched = SliceSeq(b, i, kHeads, kD);
    EXPECT_TRUE(Unwrap(ops::allclose(batched, solo, 0.0, 0.0)).allclose)
        << "seq " << i << " (T=" << seqs[i].t_dim << ") not bit-identical";
  }
}

// Mixed past lengths and block boundaries in one batch, including a T=1 member
// (a decode-shaped sequence inside a prefill batch) and continuation P>0
// (re-prefill-after-preemption shape). Distinct seeds per member so any
// cross-sequence leakage would surface. Bit-identical per sequence.
TEST(VarlenAttentionKernelTest, MixedPastAndBlockBoundaries) {
  constexpr std::int64_t kHeads = 4;
  constexpr std::int64_t kKvHeads = 2;
  constexpr std::int64_t kD = 64;
  const float scale = Scale(kD);
  const std::int64_t t_dims[] = {1, 31, 32, 33, 64, 65, 129, 200};
  const std::int64_t pasts[] = {0, 3, 64, 100, 0, 7, 0, 5};
  std::vector<Seq> seqs;
  seqs.reserve(std::size(t_dims));
  std::uint64_t seed = 1000;
  for (std::size_t i = 0; i < std::size(t_dims); ++i) {
    seqs.push_back(
        MakeSeq(t_dims[i], kKvHeads, kHeads, kD, pasts[i], 1.0, seed += 17));
  }
  const Batch b = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    const Tensor solo = Standalone(seqs[i], kHeads, kKvHeads, kD, scale);
    const Tensor batched = SliceSeq(b, i, kHeads, kD);
    EXPECT_TRUE(Unwrap(ops::allclose(batched, solo, 0.0, 0.0)).allclose)
        << "seq " << i << " (T=" << seqs[i].t_dim << " P=" << pasts[i] << ")";
  }
}

// GQA group sizes {1, 2, full} × head dims crossing the vector widths (18 and
// 24 force NEON (d%4) and AVX2 (d%8) tails; 24 = Qwen head_dim).
TEST(VarlenAttentionKernelTest, GqaAndHeadDims) {
  struct Cfg {
    std::int64_t heads;
    std::int64_t kv_heads;
  };
  const Cfg cfgs[] = {{.heads = 4, .kv_heads = 4},
                      {.heads = 4, .kv_heads = 2},
                      {.heads = 8, .kv_heads = 1}};
  const std::int64_t dims[] = {18, 24, 64, 128};
  std::uint64_t seed = 2000;
  for (const Cfg& c : cfgs) {
    for (const std::int64_t d : dims) {
      const float scale = Scale(d);
      std::vector<Seq> seqs;
      seqs.push_back(MakeSeq(5, c.kv_heads, c.heads, d, 0, 1.0, seed += 11));
      seqs.push_back(MakeSeq(64, c.kv_heads, c.heads, d, 3, 1.0, seed += 11));
      seqs.push_back(MakeSeq(129, c.kv_heads, c.heads, d, 0, 1.0, seed += 11));
      const Batch b = RunVarlen(seqs, c.heads, c.kv_heads, d, scale);
      for (std::size_t i = 0; i < seqs.size(); ++i) {
        const Tensor solo = Standalone(seqs[i], c.heads, c.kv_heads, d, scale);
        const Tensor batched = SliceSeq(b, i, c.heads, d);
        EXPECT_TRUE(Unwrap(ops::allclose(batched, solo, 0.0, 0.0)).allclose)
            << "H=" << c.heads << " Hkv=" << c.kv_heads << " d=" << d << " seq "
            << i;
      }
    }
  }
}

// B == 1 is bit-identical to the non-varlen entry (the single-sequence path).
TEST(VarlenAttentionKernelTest, SingleSequenceIsPrefillAttention) {
  constexpr std::int64_t kHeads = 4;
  constexpr std::int64_t kKvHeads = 2;
  constexpr std::int64_t kD = 64;
  const float scale = Scale(kD);
  std::vector<Seq> seqs;
  seqs.push_back(MakeSeq(70, kKvHeads, kHeads, kD, 9, 1.5, 5000));
  const Batch b = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);
  const Tensor solo = Standalone(seqs[0], kHeads, kKvHeads, kD, scale);
  const Tensor batched = SliceSeq(b, 0, kHeads, kD);
  EXPECT_TRUE(Unwrap(ops::allclose(batched, solo, 0.0, 0.0)).allclose);
}

// Build the PrefillVarlenArgs the public entry builds, to drive the sequence
// walk directly over a contiguous unit range.
[[nodiscard]] internal::PrefillVarlenArgs ArgsFor(const Batch& b,
                                                  std::int64_t heads,
                                                  std::int64_t kv_heads,
                                                  std::int64_t d, float scale) {
  return internal::PrefillVarlenArgs{
      .q = b.q_flat.data_ptr<float>(),
      .cu_seqlens = b.cu.data(),
      .k_seqs = b.k_ptrs.data(),
      .v_seqs = b.v_ptrs.data(),
      .l_dims = b.l_dims.data(),
      .out = b.out_flat.data_ptr<float>(),
      .num_seqs = static_cast<std::int64_t>(b.k_ptrs.size()),
      .heads = heads,
      .d = d,
      .group = heads / kv_heads,
      .scale = scale};
}

// The threaded public entry is bit-identical to a single serial sequence-walk
// over all units — and to an arbitrary manual chunking straddling sequence
// boundaries — so the result is invariant to thread count / chunking.
TEST(VarlenAttentionKernelTest, ThreadingIsBitExactVsSerialVariant) {
  constexpr std::int64_t kHeads = 4;
  constexpr std::int64_t kKvHeads = 2;
  constexpr std::int64_t kD = 64;
  const float scale = Scale(kD);
  std::vector<Seq> seqs;
  seqs.push_back(MakeSeq(70, kKvHeads, kHeads, kD, 9, 1.5, 5000));
  seqs.push_back(MakeSeq(33, kKvHeads, kHeads, kD, 0, 1.5, 6000));
  seqs.push_back(MakeSeq(5, kKvHeads, kHeads, kD, 2, 1.5, 7000));

  const Batch threaded = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);

  // Total units = Σ H·ceil(T_b/kAttnQb).
  std::int64_t units = 0;
  for (const Seq& s : seqs) {
    units += kHeads * ((s.t_dim + internal::kAttnQb - 1) / internal::kAttnQb);
  }
  const auto fn = detail::PrefillAttentionVariant(SelectedIsa());

  // Serial: one call over all units. RunVarlen leaves valid output; the serial
  // walk rewrites every row (PrefillUnitsImpl zeroes each output row before
  // accumulating), so no explicit reset is needed.
  const Batch serial = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);
  detail::PrefillVarlenUnits(ArgsFor(serial, kHeads, kKvHeads, kD, scale), fn,
                             0, units);

  // Manual odd chunking with bounds straddling sequence boundaries.
  const Batch chunked = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);
  const internal::PrefillVarlenArgs cargs =
      ArgsFor(chunked, kHeads, kKvHeads, kD, scale);
  const std::int64_t bounds[] = {0, 1, 3, units / 2, units - 1, units};
  for (std::size_t i = 0; i + 1 < std::size(bounds); ++i) {
    detail::PrefillVarlenUnits(cargs, fn, bounds[i], bounds[i + 1]);
  }

  EXPECT_TRUE(
      Unwrap(ops::allclose(threaded.out_flat, serial.out_flat, 0.0, 0.0))
          .allclose);
  EXPECT_TRUE(Unwrap(ops::allclose(chunked.out_flat, serial.out_flat, 0.0, 0.0))
                  .allclose);
}

// The vector variant slot is populated on a vector ISA (mirrors the prefill
// kernel's guard; varlen reuses the same PrefillUnits variant).
TEST(VarlenAttentionKernelTest, VectorVariantSlotPopulated) {
  const Isa isa = SelectedIsa();
  if (isa == Isa::kScalar) {
    GTEST_SKIP() << "forced-scalar pass";
  }
  EXPECT_NE(reinterpret_cast<void*>(detail::PrefillAttentionVariant(isa)),
            reinterpret_cast<void*>(&scalar::PrefillUnits));
}

// Class-T tie to the M5 oracle (cpu::attention) per sequence: belt-and-braces
// beyond the per-sequence bit-identity, replaying a mixed batch through the
// reference chain within tolerance.
TEST(VarlenAttentionKernelTest, MatchesOracleClassT) {
  constexpr std::int64_t kHeads = 8;
  constexpr std::int64_t kKvHeads = 2;
  constexpr std::int64_t kD = 64;
  const float scale = Scale(kD);
  std::vector<Seq> seqs;
  seqs.push_back(MakeSeq(5, kKvHeads, kHeads, kD, 0, 1.0, 8000));
  seqs.push_back(MakeSeq(64, kKvHeads, kHeads, kD, 0, 1.0, 8100));
  seqs.push_back(MakeSeq(129, kKvHeads, kHeads, kD, 0, 1.0, 8200));
  const Batch b = RunVarlen(seqs, kHeads, kKvHeads, kD, scale);
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    const Tensor ref = Reference(seqs[i], kHeads, kD, scale);
    const Tensor batched = SliceSeq(b, i, kHeads, kD);
    const ops::AllCloseResult r =
        Unwrap(ops::allclose(batched, ref, kRtol, kAtol));
    EXPECT_TRUE(r.allclose) << "seq " << i << ": " << r.Summary();
    std::cerr << "[varlen] oracle seq " << i
              << " max_abs_diff=" << r.max_abs_diff << '\n';
  }
}

// B == 0 is a no-op: the poisoned (NaN) output is left untouched.
TEST(VarlenAttentionKernelTest, EmptyBatchIsNoOp) {
  constexpr std::int64_t kHeads = 4;
  constexpr std::int64_t kD = 64;
  const Tensor out =
      Unwrap(ops::zeros(Shape{1, kHeads, kD}, DataType::kFloat32));
  auto* out_data = out.data_ptr<float>();
  for (std::int64_t i = 0; i < kHeads * kD; ++i) {
    out_data[i] = std::numeric_limits<float>::quiet_NaN();
  }
  const std::int32_t cu[] = {0};
  PrefillAttentionVarlenF32(nullptr, cu, 0, nullptr, nullptr, nullptr, out_data,
                            kHeads, 2, kD, Scale(kD));
  for (std::int64_t i = 0; i < kHeads * kD; ++i) {
    EXPECT_TRUE(std::isnan(out_data[i]));
  }
}

}  // namespace
}  // namespace engine::kernels
