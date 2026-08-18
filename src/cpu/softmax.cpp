#include "core/check.h"
#include "core/status.h"
#include "cpu/detail.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace engine::cpu {

namespace {

// One row is one unit of parallel work: rows are independent, and each row's
// max/sum reductions run in a fixed left-to-right order, so the result is
// bit-identical regardless of how rows are partitioned across threads.
constexpr std::int64_t kRowGrain = 1;

}  // namespace

core::Status softmax(const tensor::Tensor& x, tensor::Tensor& y) {
  CHECK(x.defined() && y.defined(),
        "cpu::softmax: x/y must be defined tensors");
  RETURN_IF_ERROR(detail::RequireContiguousRank(x, 2, "cpu::softmax", "x"));
  RETURN_IF_ERROR(detail::RequireF32(x, "cpu::softmax", "x"));

  const std::int64_t rows = x.shape().dim(0);
  const std::int64_t n_dim = x.shape().dim(1);
  if (rows < 1 || n_dim < 1) {
    return core::InvalidArgumentError(
        "cpu::softmax: x must be [R>=1, N>=1], got [{}, {}]", rows, n_dim);
  }
  RETURN_IF_ERROR(detail::RequireContiguousRank(y, 2, "cpu::softmax", "y"));
  RETURN_IF_ERROR(detail::RequireF32(y, "cpu::softmax", "y"));
  if (y.shape().dim(0) != rows || y.shape().dim(1) != n_dim) {
    return core::InvalidArgumentError(
        "cpu::softmax: y must be [{}, {}] to match x, got [{}, {}]", rows,
        n_dim, y.shape().dim(0), y.shape().dim(1));
  }

  const auto* x_data = x.data_ptr<float>();
  auto* y_data = y.data_ptr<float>();
  parallel::parallel_for(parallel::DefaultPool(), rows, kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t r = begin; r < end; ++r) {
                             const float* x_row = x_data + (r * n_dim);
                             float* y_row = y_data + (r * n_dim);
                             // Subtract the row max for stability: a `-inf`
                             // entry (causal mask) maps to exp(-inf) == 0; an
                             // all-`-inf` row yields NaN (caller error,
                             // documented). max/sum accumulate in a single fp32
                             // value.
                             float row_max = x_row[0];
                             for (std::int64_t j = 1; j < n_dim; ++j) {
                               row_max = std::max(row_max, x_row[j]);
                             }
                             float sum = 0.0F;
                             for (std::int64_t j = 0; j < n_dim; ++j) {
                               const float e = std::exp(x_row[j] - row_max);
                               y_row[j] = e;
                               sum += e;
                             }
                             const float inv_sum = 1.0F / sum;
                             for (std::int64_t j = 0; j < n_dim; ++j) {
                               y_row[j] *= inv_sum;
                             }
                           }
                         });
  return core::OkStatus();
}

}  // namespace engine::cpu
