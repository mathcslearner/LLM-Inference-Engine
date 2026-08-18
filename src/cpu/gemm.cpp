#include "core/check.h"
#include "core/status.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cstdint>

namespace engine::cpu {

namespace {

// Cache-blocking tile sizes (model-execution.md §2.1: "cache-blocked but
// simple"). One (mb, nb) output tile of at most kTileM x kTileN is one unit of
// parallel work; the K dimension is walked in kTileK-sized blocks *inside* the
// tile so a B slab stays hot across the tile's rows. These change traversal
// order only, never the per-element reduction order (each C[m,n] still sums its
// K terms ascending into one fp32 accumulator), so results are bit-identical
// regardless of their values — they are not tuning knobs the way an optimized
// kernel's would be.
constexpr std::int64_t kTileM = 64;
constexpr std::int64_t kTileN = 64;
constexpr std::int64_t kTileK = 256;

// One fp32 output tile is one parallel chunk (grain 1 over the flattened tile
// grid): the reference favors obviousness over scheduling granularity.
constexpr std::int64_t kTileGrain = 1;

// Widen one stored weight/bias element to fp32 through tensor/half.h's
// value-type conversions (operator float() — bit-exact, constexpr). Templated
// on the storage type so the K loop carries no per-element dtype branch.
template <typename StoredT>
[[nodiscard]] inline float Widen(StoredT v) {
  return static_cast<float>(v);
}

// C[M,N] = A[M,K] * B[N,K]^T (+ bias[N]), with B stored as WeightT and bias as
// BiasT. `has_bias` gates the bias read at compile time (BiasT is float when
// absent). Called once per output tile.
template <typename WeightT, typename BiasT, bool kHasBias>
void GemmTile(const float* a, const WeightT* b, const BiasT* bias, float* c,
              std::int64_t n_dim, std::int64_t k_dim, std::int64_t m0,
              std::int64_t m1, std::int64_t n0, std::int64_t n1) {
  for (std::int64_t m = m0; m < m1; ++m) {
    const float* a_row = a + (m * k_dim);
    for (std::int64_t n = n0; n < n1; ++n) {
      const WeightT* b_row = b + (n * k_dim);
      // Single fp32 accumulator, ascending k, K blocked only for locality:
      // the running sum is one value walked in order, so blocking does not
      // change which additions happen in which order.
      float acc = 0.0F;
      for (std::int64_t k0 = 0; k0 < k_dim; k0 += kTileK) {
        const std::int64_t k1 = std::min(k0 + kTileK, k_dim);
        for (std::int64_t k = k0; k < k1; ++k) {
          acc += a_row[k] * Widen<WeightT>(b_row[k]);
        }
      }
      if constexpr (kHasBias) {
        acc += Widen<BiasT>(bias[n]);
      }
      c[(m * n_dim) + n] = acc;
    }
  }
}

// Runs the tiled GEMM for a fixed (WeightT, BiasT, kHasBias) instantiation.
template <typename WeightT, typename BiasT, bool kHasBias>
void GemmImpl(const float* a, const WeightT* b, const BiasT* bias, float* c,
              std::int64_t m_dim, std::int64_t n_dim, std::int64_t k_dim) {
  const std::int64_t m_tiles = (m_dim + kTileM - 1) / kTileM;
  const std::int64_t n_tiles = (n_dim + kTileN - 1) / kTileN;
  const std::int64_t num_tiles = m_tiles * n_tiles;
  parallel::parallel_for(
      parallel::DefaultPool(), num_tiles, kTileGrain,
      [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t tile = begin; tile < end; ++tile) {
          const std::int64_t mt = tile / n_tiles;
          const std::int64_t nt = tile % n_tiles;
          const std::int64_t m0 = mt * kTileM;
          const std::int64_t m1 = std::min(m0 + kTileM, m_dim);
          const std::int64_t n0 = nt * kTileN;
          const std::int64_t n1 = std::min(n0 + kTileN, n_dim);
          GemmTile<WeightT, BiasT, kHasBias>(a, b, bias, c, n_dim, k_dim, m0,
                                             m1, n0, n1);
        }
      });
}

// Selects the BiasT instantiation for a known WeightT, dispatching on the bias
// storage dtype (or the no-bias path). `bias_data` is null iff there is no
// bias.
template <typename WeightT>
[[nodiscard]] core::Status DispatchBias(const float* a, const WeightT* b,
                                        const tensor::Tensor* bias, float* c,
                                        std::int64_t m_dim, std::int64_t n_dim,
                                        std::int64_t k_dim) {
  if (bias == nullptr) {
    GemmImpl<WeightT, float, false>(a, b, nullptr, c, m_dim, n_dim, k_dim);
    return core::OkStatus();
  }
  switch (bias->dtype()) {
    case tensor::DataType::kFloat32:
      GemmImpl<WeightT, float, true>(a, b, bias->data_ptr<float>(), c, m_dim,
                                     n_dim, k_dim);
      return core::OkStatus();
    case tensor::DataType::kFloat16:
      GemmImpl<WeightT, tensor::float16, true>(
          a, b, bias->data_ptr<tensor::float16>(), c, m_dim, n_dim, k_dim);
      return core::OkStatus();
    case tensor::DataType::kBFloat16:
      GemmImpl<WeightT, tensor::bfloat16, true>(
          a, b, bias->data_ptr<tensor::bfloat16>(), c, m_dim, n_dim, k_dim);
      return core::OkStatus();
    default:
      return core::InvalidArgumentError(
          "cpu::gemm: bias dtype must be f32/f16/bf16, got {}",
          tensor::to_string(bias->dtype()));
  }
}

// Requires a defined, contiguous, rank-`rank` float tensor. `role` names it in
// error messages (the actionability rule, model-execution.md §13).
[[nodiscard]] core::Status RequireContiguous(const tensor::Tensor& t, int rank,
                                             const char* role) {
  if (t.shape().rank() != rank) {
    return core::InvalidArgumentError(
        "cpu::gemm: {} must be rank-{}, got rank-{}", role, rank,
        t.shape().rank());
  }
  if (!t.is_contiguous()) {
    return core::InvalidArgumentError("cpu::gemm: {} must be contiguous", role);
  }
  return core::OkStatus();
}

}  // namespace

core::Status gemm(const tensor::Tensor& a, const tensor::Tensor& b,
                  const tensor::Tensor* bias, tensor::Tensor& c) {
  // Undefined handles are a programmer error at the call site (CHECK);
  // malformed *shapes/dtypes* are recoverable (ADR-003).
  CHECK(a.defined() && b.defined() && c.defined(),
        "cpu::gemm: a/b/c must be defined tensors");

  RETURN_IF_ERROR(RequireContiguous(a, 2, "a"));
  RETURN_IF_ERROR(RequireContiguous(b, 2, "b"));
  RETURN_IF_ERROR(RequireContiguous(c, 2, "c"));

  if (a.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError(
        "cpu::gemm: a must be f32 (activations are fp32), got {}",
        tensor::to_string(a.dtype()));
  }
  if (c.dtype() != tensor::DataType::kFloat32) {
    return core::InvalidArgumentError("cpu::gemm: c must be f32, got {}",
                                      tensor::to_string(c.dtype()));
  }

  const std::int64_t m_dim = a.shape().dim(0);
  const std::int64_t k_dim = a.shape().dim(1);
  const std::int64_t n_dim = b.shape().dim(0);
  if (b.shape().dim(1) != k_dim) {
    return core::InvalidArgumentError(
        "cpu::gemm: inner dims disagree — a is [{}, {}], b is [{}, {}] (a.K "
        "must equal b.K)",
        m_dim, k_dim, n_dim, b.shape().dim(1));
  }
  if (m_dim < 1 || n_dim < 1 || k_dim < 1) {
    return core::InvalidArgumentError(
        "cpu::gemm: M, N, K must all be >= 1 — got M={}, N={}, K={}", m_dim,
        n_dim, k_dim);
  }
  if (c.shape().dim(0) != m_dim || c.shape().dim(1) != n_dim) {
    return core::InvalidArgumentError(
        "cpu::gemm: c must be [{}, {}] to hold A*B^T, got [{}, {}]", m_dim,
        n_dim, c.shape().dim(0), c.shape().dim(1));
  }
  if (bias != nullptr) {
    CHECK(bias->defined(), "cpu::gemm: bias handle is non-null but undefined");
    RETURN_IF_ERROR(RequireContiguous(*bias, 1, "bias"));
    if (bias->shape().dim(0) != n_dim) {
      return core::InvalidArgumentError(
          "cpu::gemm: bias must have length N={}, got {}", n_dim,
          bias->shape().dim(0));
    }
  }

  const auto* a_data = a.data_ptr<float>();
  auto* c_data = c.data_ptr<float>();
  switch (b.dtype()) {
    case tensor::DataType::kFloat32:
      return DispatchBias<float>(a_data, b.data_ptr<float>(), bias, c_data,
                                 m_dim, n_dim, k_dim);
    case tensor::DataType::kFloat16:
      return DispatchBias<tensor::float16>(a_data,
                                           b.data_ptr<tensor::float16>(), bias,
                                           c_data, m_dim, n_dim, k_dim);
    case tensor::DataType::kBFloat16:
      return DispatchBias<tensor::bfloat16>(a_data,
                                            b.data_ptr<tensor::bfloat16>(),
                                            bias, c_data, m_dim, n_dim, k_dim);
    default:
      return core::InvalidArgumentError(
          "cpu::gemm: b (weight) dtype must be f32/f16/bf16, got {}",
          tensor::to_string(b.dtype()));
  }
}

}  // namespace engine::cpu
