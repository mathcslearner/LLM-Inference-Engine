#include "core/check.h"
#include "core/status.h"
#include "cpu/detail.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/tensor.h"

#include <cmath>
#include <cstdint>

namespace engine::cpu {

namespace {

// One row is one unit of parallel work (pure elementwise — any partition is
// equivalent; row grain keeps the traversal obvious and matches rmsnorm).
constexpr std::int64_t kRowGrain = 1;

// Validates that `t` is a contiguous f32 [rows, cols] tensor. `op`/`role` name
// it in errors.
[[nodiscard]] core::Status RequireF32Matrix(const tensor::Tensor& t,
                                            std::int64_t rows,
                                            std::int64_t cols, const char* op,
                                            const char* role) {
  RETURN_IF_ERROR(detail::RequireContiguousRank(t, 2, op, role));
  RETURN_IF_ERROR(detail::RequireF32(t, op, role));
  if (t.shape().dim(0) != rows || t.shape().dim(1) != cols) {
    return core::InvalidArgumentError("{}: {} must be [{}, {}], got [{}, {}]",
                                      op, role, rows, cols, t.shape().dim(0),
                                      t.shape().dim(1));
  }
  return core::OkStatus();
}

// silu(v) = v / (1 + exp(-v)) = v * sigmoid(v) (HF F.silu). exp(-v) saturates
// to +inf for very negative v, giving silu -> 0 with no NaN; for very positive
// v it underflows to 0, giving silu -> v. Computed in fp32.
[[nodiscard]] inline float Silu(float v) { return v / (1.0F + std::exp(-v)); }

}  // namespace

core::Status silu_mul(const tensor::Tensor& gate, const tensor::Tensor& up,
                      tensor::Tensor& y) {
  CHECK(gate.defined() && up.defined() && y.defined(),
        "cpu::silu_mul: gate/up/y must be defined tensors");
  RETURN_IF_ERROR(
      detail::RequireContiguousRank(gate, 2, "cpu::silu_mul", "gate"));
  RETURN_IF_ERROR(detail::RequireF32(gate, "cpu::silu_mul", "gate"));

  const std::int64_t rows = gate.shape().dim(0);
  const std::int64_t cols = gate.shape().dim(1);
  if (rows < 1 || cols < 1) {
    return core::InvalidArgumentError(
        "cpu::silu_mul: gate must be [T>=1, I>=1], got [{}, {}]", rows, cols);
  }
  RETURN_IF_ERROR(RequireF32Matrix(up, rows, cols, "cpu::silu_mul", "up"));
  RETURN_IF_ERROR(RequireF32Matrix(y, rows, cols, "cpu::silu_mul", "y"));

  const auto* gate_data = gate.data_ptr<float>();
  const auto* up_data = up.data_ptr<float>();
  auto* y_data = y.data_ptr<float>();
  const std::int64_t n = rows * cols;
  parallel::parallel_for(parallel::DefaultPool(), n, cols * kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t i = begin; i < end; ++i) {
                             y_data[i] = Silu(gate_data[i]) * up_data[i];
                           }
                         });
  return core::OkStatus();
}

core::Status add(const tensor::Tensor& a, const tensor::Tensor& b,
                 tensor::Tensor& y) {
  CHECK(a.defined() && b.defined() && y.defined(),
        "cpu::add: a/b/y must be defined tensors");
  RETURN_IF_ERROR(detail::RequireContiguousRank(a, 2, "cpu::add", "a"));
  RETURN_IF_ERROR(detail::RequireF32(a, "cpu::add", "a"));

  const std::int64_t rows = a.shape().dim(0);
  const std::int64_t cols = a.shape().dim(1);
  if (rows < 1 || cols < 1) {
    return core::InvalidArgumentError(
        "cpu::add: a must be [T>=1, E>=1], got [{}, {}]", rows, cols);
  }
  RETURN_IF_ERROR(RequireF32Matrix(b, rows, cols, "cpu::add", "b"));
  RETURN_IF_ERROR(RequireF32Matrix(y, rows, cols, "cpu::add", "y"));

  const auto* a_data = a.data_ptr<float>();
  const auto* b_data = b.data_ptr<float>();
  auto* y_data = y.data_ptr<float>();
  const std::int64_t n = rows * cols;
  parallel::parallel_for(parallel::DefaultPool(), n, cols * kRowGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t i = begin; i < end; ++i) {
                             y_data[i] = a_data[i] + b_data[i];
                           }
                         });
  return core::OkStatus();
}

}  // namespace engine::cpu
