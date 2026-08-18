#include "core/check.h"
#include "core/status.h"
#include "cpu/detail.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/tensor.h"

#include <cmath>
#include <cstdint>

namespace engine::cpu {

namespace {

// One row is one unit of parallel work: rows are independent and each reduces
// its E squares in a single ascending fp32 accumulator, so the result is
// bit-identical regardless of how rows are partitioned across threads.
constexpr std::int64_t kRowGrain = 1;

// y[t, :] = x[t, :] * rsqrt(mean(x[t, :]²) + eps) * weight, for a known
// (input, weight) storage-dtype pair. The mean of squares and the scaling both
// run in fp32; x/weight are widened per element via detail::Widen.
template <typename XT, typename WT>
void RmsNormImpl(const XT* x, const WT* weight, float eps, float* y,
                 std::int64_t rows, std::int64_t e_dim) {
  const auto e_f = static_cast<float>(e_dim);
  parallel::parallel_for(
      parallel::DefaultPool(), rows, kRowGrain,
      [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t r = begin; r < end; ++r) {
          const XT* x_row = x + (r * e_dim);
          float* y_row = y + (r * e_dim);
          // Sum of squares, ascending e, single fp32 accumulator.
          float sum_sq = 0.0F;
          for (std::int64_t e = 0; e < e_dim; ++e) {
            const float v = detail::Widen<XT>(x_row[e]);
            sum_sq += v * v;
          }
          // rsqrt(mean + eps) — HF computes variance as the mean of squares
          // (mean over the hidden dim), then 1/sqrt(variance + eps).
          const float inv_rms = 1.0F / std::sqrt((sum_sq / e_f) + eps);
          for (std::int64_t e = 0; e < e_dim; ++e) {
            y_row[e] = detail::Widen<XT>(x_row[e]) * inv_rms *
                       detail::Widen<WT>(weight[e]);
          }
        }
      });
}

// Selects the WT instantiation for a known XT, dispatching on the weight
// storage dtype.
template <typename XT>
[[nodiscard]] core::Status DispatchWeight(const XT* x,
                                          const tensor::Tensor& weight,
                                          float eps, float* y,
                                          std::int64_t rows,
                                          std::int64_t e_dim) {
  switch (weight.dtype()) {
    case tensor::DataType::kFloat32:
      RmsNormImpl<XT, float>(x, weight.data_ptr<float>(), eps, y, rows, e_dim);
      return core::OkStatus();
    case tensor::DataType::kFloat16:
      RmsNormImpl<XT, tensor::float16>(x, weight.data_ptr<tensor::float16>(),
                                       eps, y, rows, e_dim);
      return core::OkStatus();
    case tensor::DataType::kBFloat16:
      RmsNormImpl<XT, tensor::bfloat16>(x, weight.data_ptr<tensor::bfloat16>(),
                                        eps, y, rows, e_dim);
      return core::OkStatus();
    default:
      return core::InvalidArgumentError(
          "cpu::rmsnorm: weight dtype must be f32/f16/bf16, got {}",
          tensor::to_string(weight.dtype()));
  }
}

}  // namespace

core::Status rmsnorm(const tensor::Tensor& x, const tensor::Tensor& weight,
                     float eps, tensor::Tensor& y) {
  CHECK(x.defined() && weight.defined() && y.defined(),
        "cpu::rmsnorm: x/weight/y must be defined tensors");

  // Rank/contiguity here; the storage-dtype gate is the dispatch switches
  // below (they name x/weight in the error), matching cpu::gemm's idiom.
  RETURN_IF_ERROR(detail::RequireContiguousRank(x, 2, "cpu::rmsnorm", "x"));
  RETURN_IF_ERROR(
      detail::RequireContiguousRank(weight, 1, "cpu::rmsnorm", "weight"));
  RETURN_IF_ERROR(detail::RequireContiguousRank(y, 2, "cpu::rmsnorm", "y"));
  RETURN_IF_ERROR(detail::RequireF32(y, "cpu::rmsnorm", "y"));

  const std::int64_t rows = x.shape().dim(0);
  const std::int64_t e_dim = x.shape().dim(1);
  if (weight.shape().dim(0) != e_dim) {
    return core::InvalidArgumentError(
        "cpu::rmsnorm: weight length {} must equal x's hidden dim {}",
        weight.shape().dim(0), e_dim);
  }
  if (rows < 1 || e_dim < 1) {
    return core::InvalidArgumentError(
        "cpu::rmsnorm: x must be [T>=1, E>=1], got [{}, {}]", rows, e_dim);
  }
  if (y.shape().dim(0) != rows || y.shape().dim(1) != e_dim) {
    return core::InvalidArgumentError(
        "cpu::rmsnorm: y must be [{}, {}] to match x, got [{}, {}]", rows,
        e_dim, y.shape().dim(0), y.shape().dim(1));
  }

  auto* y_data = y.data_ptr<float>();
  switch (x.dtype()) {
    case tensor::DataType::kFloat32:
      return DispatchWeight<float>(x.data_ptr<float>(), weight, eps, y_data,
                                   rows, e_dim);
    case tensor::DataType::kFloat16:
      return DispatchWeight<tensor::float16>(x.data_ptr<tensor::float16>(),
                                             weight, eps, y_data, rows, e_dim);
    case tensor::DataType::kBFloat16:
      return DispatchWeight<tensor::bfloat16>(x.data_ptr<tensor::bfloat16>(),
                                              weight, eps, y_data, rows, e_dim);
    default:
      return core::InvalidArgumentError(
          "cpu::rmsnorm: x dtype must be f32/f16/bf16, got {}",
          tensor::to_string(x.dtype()));
  }
}

}  // namespace engine::cpu
