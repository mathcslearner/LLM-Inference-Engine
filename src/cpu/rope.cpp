#include "core/check.h"
#include "core/status.h"
#include "cpu/detail.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <span>

namespace engine::cpu {

namespace {

// One token is one unit of parallel work: every (head, pair) rotation under a
// token reads only that token's slice of x and the cos/sin row for its
// position, so tokens are independent and any partition is equivalent —
// bit-identical across thread counts.
constexpr std::int64_t kTokenGrain = 1;

}  // namespace

core::Status rope_apply(tensor::Tensor& x,
                        std::span<const std::int32_t> positions,
                        const tensor::Tensor& cos, const tensor::Tensor& sin) {
  CHECK(x.defined() && cos.defined() && sin.defined(),
        "cpu::rope_apply: x/cos/sin must be defined tensors");

  RETURN_IF_ERROR(detail::RequireContiguousRank(x, 3, "cpu::rope_apply", "x"));
  RETURN_IF_ERROR(detail::RequireF32(x, "cpu::rope_apply", "x"));
  RETURN_IF_ERROR(
      detail::RequireContiguousRank(cos, 2, "cpu::rope_apply", "cos"));
  RETURN_IF_ERROR(detail::RequireF32(cos, "cpu::rope_apply", "cos"));
  RETURN_IF_ERROR(
      detail::RequireContiguousRank(sin, 2, "cpu::rope_apply", "sin"));
  RETURN_IF_ERROR(detail::RequireF32(sin, "cpu::rope_apply", "sin"));

  const std::int64_t t_dim = x.shape().dim(0);
  const std::int64_t heads = x.shape().dim(1);
  const std::int64_t d = x.shape().dim(2);
  const auto n_pos = static_cast<std::int64_t>(positions.size());
  if (t_dim < 1 || heads < 1 || d < 1) {
    return core::InvalidArgumentError(
        "cpu::rope_apply: x must be [T>=1, Hx>=1, d>=1], got [{}, {}, {}]",
        t_dim, heads, d);
  }
  if (d % 2 != 0) {
    return core::InvalidArgumentError(
        "cpu::rope_apply: head dim d must be even, got {}", d);
  }
  const std::int64_t half = d / 2;
  const std::int64_t table_rows = cos.shape().dim(0);
  if (cos.shape().dim(1) != half) {
    return core::InvalidArgumentError(
        "cpu::rope_apply: cos must be [P, {}] (d/2), got [{}, {}]", half,
        table_rows, cos.shape().dim(1));
  }
  if (sin.shape().dim(0) != table_rows || sin.shape().dim(1) != half) {
    return core::InvalidArgumentError(
        "cpu::rope_apply: sin must match cos shape [{}, {}], got [{}, {}]",
        table_rows, half, sin.shape().dim(0), sin.shape().dim(1));
  }
  if (n_pos != t_dim) {
    return core::InvalidArgumentError(
        "cpu::rope_apply: positions length {} must equal T {}", n_pos, t_dim);
  }
  // Position bounds precede the rotation: an out-of-range position would index
  // past the cos/sin tables (an out-of-bounds read). Serial and cheap (T).
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t p = positions[static_cast<std::size_t>(t)];
    if (p < 0 || p >= table_rows) {
      return core::InvalidArgumentError(
          "cpu::rope_apply: positions[{}] = {} is out of range for the "
          "cos/sin table of length {}",
          t, p, table_rows);
    }
  }

  auto* x_data = x.data_ptr<float>();
  const auto* cos_data = cos.data_ptr<float>();
  const auto* sin_data = sin.data_ptr<float>();

  parallel::parallel_for(parallel::DefaultPool(), t_dim, kTokenGrain,
                         [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t t = begin; t < end; ++t) {
                             const std::int64_t p =
                                 positions[static_cast<std::size_t>(t)];
                             const float* cos_row = cos_data + (p * half);
                             const float* sin_row = sin_data + (p * half);
                             float* x_tok = x_data + (t * heads * d);
                             for (std::int64_t h = 0; h < heads; ++h) {
                               float* v = x_tok + (h * d);
                               for (std::int64_t j = 0; j < half; ++j) {
                                 const float c = cos_row[j];
                                 const float s = sin_row[j];
                                 // Read both halves of the pair before writing
                                 // either — this is what makes the in-place
                                 // rotation correct.
                                 const float lo = v[j];
                                 const float hi = v[j + half];
                                 v[j] = (lo * c) - (hi * s);
                                 v[j + half] = (hi * c) + (lo * s);
                               }
                             }
                           }
                         });
  return core::OkStatus();
}

}  // namespace engine::cpu
