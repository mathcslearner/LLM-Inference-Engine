#include "core/check.h"
#include "core/status.h"
#include "cpu/detail.h"
#include "cpu/ops.h"
#include "parallel/parallel_for.h"
#include "parallel/thread_pool.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <limits>

namespace engine::cpu {

namespace {

// One (head, query-token) pair is one unit of parallel work: its scores row and
// its context vector read only that query's slice of q/out and the shared k/v,
// so the pairs are independent and any partition is equivalent — bit-identical
// across thread counts. Rows are laid out `row = h * T + t` (query-major within
// a head), matching the [H·T, L] scores buffer.
constexpr std::int64_t kRowGrain = 1;

// Dimensions of one attention call, shared by the two passes.
struct AttnDims {
  std::int64_t t_dim;  // T new queries
  std::int64_t heads;  // H query heads
  std::int64_t d;      // head_dim
  std::int64_t l_dim;  // L = P + T cached+new keys
  std::int64_t group;  // H / Hkv (GQA group size)
  std::int64_t past;   // P = L - T
  std::int64_t rows;   // H · T (units of parallel work)
};

// Pass 1: scores[row, s] = scale · (q_row · k[hk, s]) for s <= P + t, else
// -inf. The dot accumulates over d in a single ascending fp32 accumulator;
// `scale` multiplies the completed dot (HF order: matmul then scaling).
void ScorePass(const float* q_data, const float* k_data, float* score_data,
               const AttnDims& dims, float scale) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  parallel::parallel_for(
      parallel::DefaultPool(), dims.rows, kRowGrain,
      [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
          const std::int64_t h = row / dims.t_dim;
          const std::int64_t t = row % dims.t_dim;
          const std::int64_t hk = h / dims.group;    // GQA: kv head for query h
          const std::int64_t limit = dims.past + t;  // inclusive key boundary
          const float* q_vec = q_data + (((t * dims.heads) + h) * dims.d);
          float* score_row = score_data + (row * dims.l_dim);
          for (std::int64_t s = 0; s < dims.l_dim; ++s) {
            if (s > limit) {
              score_row[s] = kNegInf;
              continue;
            }
            const float* k_vec = k_data + (((hk * dims.l_dim) + s) * dims.d);
            float dot = 0.0F;
            for (std::int64_t e = 0; e < dims.d; ++e) {
              dot += q_vec[e] * k_vec[e];
            }
            score_row[s] = dot * scale;
          }
        }
      });
}

// Pass 2: out[t, h, :] = Σ_s prob[row, s] · v[hk, s, :]. Each output element
// sums over s in a single ascending fp32 accumulator; the masked (now 0)
// probabilities contribute nothing, so the sum can run over all L keys.
void ContextPass(const float* score_data, const float* v_data, float* out_data,
                 const AttnDims& dims) {
  parallel::parallel_for(
      parallel::DefaultPool(), dims.rows, kRowGrain,
      [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
          const std::int64_t h = row / dims.t_dim;
          const std::int64_t t = row % dims.t_dim;
          const std::int64_t hk = h / dims.group;
          const float* prob_row = score_data + (row * dims.l_dim);
          float* out_vec = out_data + (((t * dims.heads) + h) * dims.d);
          for (std::int64_t e = 0; e < dims.d; ++e) {
            float acc = 0.0F;
            for (std::int64_t s = 0; s < dims.l_dim; ++s) {
              acc +=
                  prob_row[s] * v_data[(((hk * dims.l_dim) + s) * dims.d) + e];
            }
            out_vec[e] = acc;
          }
        }
      });
}

}  // namespace

core::Status attention(const tensor::Tensor& q, const tensor::Tensor& k,
                       const tensor::Tensor& v, float scale,
                       tensor::Tensor& out) {
  CHECK(q.defined() && k.defined() && v.defined() && out.defined(),
        "cpu::attention: q/k/v/out must be defined tensors");

  RETURN_IF_ERROR(detail::RequireContiguousRank(q, 3, "cpu::attention", "q"));
  RETURN_IF_ERROR(detail::RequireF32(q, "cpu::attention", "q"));
  RETURN_IF_ERROR(detail::RequireContiguousRank(k, 3, "cpu::attention", "k"));
  RETURN_IF_ERROR(detail::RequireF32(k, "cpu::attention", "k"));
  RETURN_IF_ERROR(detail::RequireContiguousRank(v, 3, "cpu::attention", "v"));
  RETURN_IF_ERROR(detail::RequireF32(v, "cpu::attention", "v"));
  RETURN_IF_ERROR(
      detail::RequireContiguousRank(out, 3, "cpu::attention", "out"));
  RETURN_IF_ERROR(detail::RequireF32(out, "cpu::attention", "out"));

  const std::int64_t t_dim = q.shape().dim(0);
  const std::int64_t heads = q.shape().dim(1);
  const std::int64_t d = q.shape().dim(2);
  const std::int64_t kv_heads = k.shape().dim(0);
  const std::int64_t l_dim = k.shape().dim(1);
  if (t_dim < 1 || heads < 1 || d < 1) {
    return core::InvalidArgumentError(
        "cpu::attention: q must be [T>=1, H>=1, d>=1], got [{}, {}, {}]", t_dim,
        heads, d);
  }
  if (kv_heads < 1 || l_dim < 1 || k.shape().dim(2) != d) {
    return core::InvalidArgumentError(
        "cpu::attention: k must be [Hkv>=1, L>=1, {}] (d matching q), got {}",
        d, k.shape());
  }
  if (v.shape().dim(0) != kv_heads || v.shape().dim(1) != l_dim ||
      v.shape().dim(2) != d) {
    return core::InvalidArgumentError(
        "cpu::attention: v must match k shape [{}, {}, {}], got {}", kv_heads,
        l_dim, d, v.shape());
  }
  if (heads % kv_heads != 0) {
    return core::InvalidArgumentError(
        "cpu::attention: H ({}) must be a positive multiple of Hkv ({})", heads,
        kv_heads);
  }
  if (l_dim < t_dim) {
    return core::InvalidArgumentError(
        "cpu::attention: cache length L ({}) must be >= T ({}) — L = P + T "
        "with "
        "P >= 0 cached positions",
        l_dim, t_dim);
  }
  if (out.shape().dim(0) != t_dim || out.shape().dim(1) != heads ||
      out.shape().dim(2) != d) {
    return core::InvalidArgumentError(
        "cpu::attention: out must be [{}, {}, {}] to match q, got {}", t_dim,
        heads, d, out.shape());
  }

  // GQA group size and the causal offset: new query `t` sits at cache position
  // `P + t`, so it attends to key positions `[0, P + t]` inclusive (§6.3).
  const AttnDims dims{.t_dim = t_dim,
                      .heads = heads,
                      .d = d,
                      .l_dim = l_dim,
                      .group = heads / kv_heads,
                      .past = l_dim - t_dim,
                      .rows = heads * t_dim};

  // Scores `[H·T, L]`, causal-masked with -inf above each query's boundary,
  // then softmaxed in place — the -inf entries map to exactly 0 (cpu::softmax's
  // documented mask contract). Every row has at least the s=0 key valid (0 <=
  // P + t always), so no row is all -inf.
  core::StatusOr<tensor::Tensor> scores = tensor::ops::zeros(
      tensor::Shape{dims.rows, l_dim}, tensor::DataType::kFloat32);
  RETURN_IF_ERROR(scores.status());
  auto* score_data = scores->data_ptr<float>();

  ScorePass(q.data_ptr<float>(), k.data_ptr<float>(), score_data, dims, scale);
  RETURN_IF_ERROR(cpu::softmax(*scores, *scores));
  ContextPass(score_data, v.data_ptr<float>(), out.data_ptr<float>(), dims);

  return core::OkStatus();
}

}  // namespace engine::cpu
