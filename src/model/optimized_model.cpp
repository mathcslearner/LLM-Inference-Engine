#include "model/optimized_model.h"

#include "core/logging.h"
#include "core/status.h"
#include "kernels/activation.h"
#include "kernels/attention.h"
#include "kernels/elementwise.h"
#include "kernels/norm.h"
#include "kernels/paged_attention.h"
#include "kernels/rope.h"
#include "kvcache/kv_cache.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/modules.h"
#include "model/optimized_embedding.h"
#include "model/packed_linear.h"
#include "model/workspace.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::model {

namespace {

// Removes and returns the canonical weight `name` (recoverable miss, exactly
// like ReferenceModel::Create so the error-path messages match).
[[nodiscard]] core::StatusOr<tensor::Tensor> TakeWeight(
    std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name) {
  const auto it = weights.find(name);
  if (it == weights.end()) {
    return core::InvalidArgumentError(
        "OptimizedModel: missing required weight '{}'", name);
  }
  tensor::Tensor w = std::move(it->second);
  weights.erase(it);
  return w;
}

// The canonical weight `name` if present (a copy — zero-copy handle), else
// nullopt. Used for optional per-projection biases (Qwen2 q/k/v).
[[nodiscard]] std::optional<tensor::Tensor> OptionalWeight(
    const std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name) {
  const auto it = weights.find(name);
  if (it == weights.end()) {
    return std::nullopt;
  }
  return it->second;
}

// Builds a PackedLinear for canonical `name` (+ optional `name`.bias when
// `has_bias` and the entry is present), returned as an owning Linear handle so
// the layer struct holds it behind the interface. Repacks at construction (§3).
[[nodiscard]] core::StatusOr<std::unique_ptr<Linear>> BuildPacked(
    std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name, bool has_bias) {
  ASSIGN_OR_RETURN(const tensor::Tensor weight, TakeWeight(weights, name));
  std::optional<tensor::Tensor> bias;
  if (has_bias) {
    bias = OptionalWeight(weights, name.substr(0, name.size() - 7) + ".bias");
  }
  ASSIGN_OR_RETURN(PackedLinear linear, PackedLinear::Create(weight, bias));
  return std::unique_ptr<Linear>(
      std::make_unique<PackedLinear>(std::move(linear)));
}

// Converts a norm scale weight `[E]` to fp32 once at build (§4). Validates it
// like RmsNorm::Create — rank-1, contiguous, f32/f16/bf16 — so the message
// names the problem before the model is built.
[[nodiscard]] core::StatusOr<tensor::Tensor> BuildNormScale(
    std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name) {
  ASSIGN_OR_RETURN(const tensor::Tensor w, TakeWeight(weights, name));
  if (w.shape().rank() != 1) {
    return core::InvalidArgumentError(
        "OptimizedModel: norm weight '{}' must be rank-1 [E], got rank-{}",
        name, w.shape().rank());
  }
  if (!w.is_contiguous()) {
    return core::InvalidArgumentError(
        "OptimizedModel: norm weight '{}' must be contiguous", name);
  }
  return tensor::ops::cast(w, tensor::DataType::kFloat32);
}

void Emit(ActivationHook* hook, std::string_view name, int layer,
          const tensor::Tensor& tensor) {
  if (hook != nullptr) {
    const ActivationEvent event{.name = name, .layer = layer, .tensor = tensor};
    hook->on_activation(event);
  }
}

}  // namespace

core::StatusOr<std::unique_ptr<OptimizedModel>> OptimizedModel::Create(
    LoadedModel loaded) {
  const ModelConfig& config = loaded.config;
  std::unordered_map<std::string, tensor::Tensor>& weights = loaded.weights;

  const int head_dim = config.head_dim;
  const int num_heads = config.num_heads;
  const int num_kv_heads = config.num_kv_heads;

  LOG_INFO("model", "repacking weights for the optimized backend ({} layers)",
           config.num_layers);

  // lm_head first (§7): it is the authoritative packed copy for a tied model,
  // so the embedding shares its storage rather than duplicating [V, E].
  ASSIGN_OR_RETURN(tensor::Tensor final_norm_scale,
                   BuildNormScale(weights, "final_norm.weight"));
  const NormWeights final_norm{.scale = std::move(final_norm_scale),
                               .eps = config.rms_norm_eps};

  ASSIGN_OR_RETURN(const tensor::Tensor lm_head_weight,
                   TakeWeight(weights, "lm_head.weight"));
  ASSIGN_OR_RETURN(PackedLinear lm_head_packed,
                   PackedLinear::Create(lm_head_weight));
  // The [V, E] source handle is dropped now; for a tied model the loader
  // aliased embed_tokens.weight to the same handle, so it is already gone.

  // Embedding: tied → share the packed lm_head's storage (one physical copy,
  // §7); untied → the model's own [V, E] table.
  std::optional<OptimizedEmbedding> embed;
  if (config.tie_word_embeddings) {
    embed = OptimizedEmbedding::FromPackedLinear(lm_head_packed);
    // Drop the tied embed_tokens handle (the loader keeps a separate alias
    // entry when tied); ignore a miss — the shared packed copy is the source.
    weights.erase("embed_tokens.weight");
  } else {
    ASSIGN_OR_RETURN(tensor::Tensor embed_weight,
                     TakeWeight(weights, "embed_tokens.weight"));
    ASSIGN_OR_RETURN(OptimizedEmbedding untied,
                     OptimizedEmbedding::FromTable(std::move(embed_weight)));
    embed = std::move(untied);
  }
  auto lm_head = std::unique_ptr<Linear>(
      std::make_unique<PackedLinear>(std::move(lm_head_packed)));

  // One shared RoPE table for the whole model (§2.2): the cos/sin tables are
  // position-only, so a single copy serves every layer (the reference builds
  // one per layer; on a real model that would be hundreds of MB duplicated).
  ASSIGN_OR_RETURN(
      Rope rope, Rope::Create(head_dim, config.rope_theta, config.rope_scaling,
                              config.max_position_embeddings));

  std::vector<Layer> layers;
  layers.reserve(static_cast<std::size_t>(config.num_layers));
  for (int i = 0; i < config.num_layers; ++i) {
    const std::string pre = "layers." + std::to_string(i) + ".";
    Layer layer;

    ASSIGN_OR_RETURN(layer.attn_norm.scale,
                     BuildNormScale(weights, pre + "attn_norm.weight"));
    layer.attn_norm.eps = config.rms_norm_eps;

    ASSIGN_OR_RETURN(layer.q_proj,
                     BuildPacked(weights, pre + "attn.q_proj.weight",
                                 config.attention_bias));
    ASSIGN_OR_RETURN(layer.k_proj,
                     BuildPacked(weights, pre + "attn.k_proj.weight",
                                 config.attention_bias));
    ASSIGN_OR_RETURN(layer.v_proj,
                     BuildPacked(weights, pre + "attn.v_proj.weight",
                                 config.attention_bias));
    ASSIGN_OR_RETURN(layer.o_proj,
                     BuildPacked(weights, pre + "attn.o_proj.weight",
                                 config.attention_bias));

    ASSIGN_OR_RETURN(layer.mlp_norm.scale,
                     BuildNormScale(weights, pre + "mlp_norm.weight"));
    layer.mlp_norm.eps = config.rms_norm_eps;

    ASSIGN_OR_RETURN(layer.gate_proj,
                     BuildPacked(weights, pre + "mlp.gate_proj.weight",
                                 /*has_bias=*/false));
    ASSIGN_OR_RETURN(layer.up_proj,
                     BuildPacked(weights, pre + "mlp.up_proj.weight",
                                 /*has_bias=*/false));
    ASSIGN_OR_RETURN(layer.down_proj,
                     BuildPacked(weights, pre + "mlp.down_proj.weight",
                                 /*has_bias=*/false));

    // Validate the layer's projection shapes against (H, Hkv, d, E, I) — the
    // same agreement Attention::Create / Mlp::Create enforce, front-loaded so a
    // decoupled head_dim (E != H·d, Qwen) is honored, not assumed.
    RETURN_IF_ERROR([&]() -> core::Status {
      const std::int64_t e = layer.q_proj->in_features();
      const std::int64_t q_out =
          static_cast<std::int64_t>(num_heads) * head_dim;
      const std::int64_t kv_out =
          static_cast<std::int64_t>(num_kv_heads) * head_dim;
      if (layer.q_proj->out_features() != q_out) {
        return core::InvalidArgumentError(
            "OptimizedModel: layer {} q_proj out {} != H·d {}", i,
            layer.q_proj->out_features(), q_out);
      }
      if (layer.k_proj->out_features() != kv_out ||
          layer.v_proj->out_features() != kv_out) {
        return core::InvalidArgumentError(
            "OptimizedModel: layer {} k/v_proj out must be Hkv·d {}", i,
            kv_out);
      }
      if (layer.o_proj->in_features() != q_out ||
          layer.o_proj->out_features() != e) {
        return core::InvalidArgumentError(
            "OptimizedModel: layer {} o_proj must map H·d {} → E {}", i, q_out,
            e);
      }
      if (layer.gate_proj->in_features() != e ||
          layer.up_proj->in_features() != e) {
        return core::InvalidArgumentError(
            "OptimizedModel: layer {} gate/up_proj in must be E {}", i, e);
      }
      if (layer.gate_proj->out_features() != layer.up_proj->out_features() ||
          layer.down_proj->in_features() != layer.gate_proj->out_features() ||
          layer.down_proj->out_features() != e) {
        return core::InvalidArgumentError(
            "OptimizedModel: layer {} MLP intermediate/E shapes disagree", i);
      }
      return core::OkStatus();
    }());

    layers.push_back(std::move(layer));
    LOG_INFO("model", "  packed layer {}/{}", i + 1, config.num_layers);
  }

  const kvcache::CacheGeometry geometry{.num_layers = config.num_layers,
                                        .num_kv_heads = num_kv_heads,
                                        .head_dim = head_dim,
                                        .dtype = tensor::DataType::kFloat32};

  Workspace workspace =
      Workspace::Create(config.hidden_size, num_heads, num_kv_heads, head_dim,
                        config.intermediate_size);

  LOG_INFO("model", "repack complete: optimized backend ready");

  return std::make_unique<OptimizedModel>(OptimizedModel(
      config, geometry, *std::move(embed), std::move(rope), std::move(layers),
      final_norm, std::move(lm_head), std::move(workspace)));
}

core::Status OptimizedModel::ForwardLayer(const ForwardRequest& request,
                                          const Layer& layer, int layer_index,
                                          std::int64_t t_dim, float scale) {
  const std::int64_t e = config_.hidden_size;

  // Workspace prefix views (value types over shared storage — writing through
  // them writes the workspace, not the view metadata, so they are non-const).
  const tensor::Tensor x = workspace_.x(t_dim);
  const tensor::Tensor h = workspace_.h(t_dim);
  tensor::Tensor tmp = workspace_.tmp(t_dim);
  const tensor::Tensor r = workspace_.r(t_dim);
  tensor::Tensor q = workspace_.q(t_dim);
  tensor::Tensor k = workspace_.k(t_dim);
  tensor::Tensor v = workspace_.v(t_dim);
  const tensor::Tensor ctx = workspace_.ctx(t_dim);

  // --- Attention block: h = attn_norm(x); q/k/v = proj(h); RoPE; attn; o_proj.
  kernels::RmsNormF32(x.data_ptr<float>(),
                      layer.attn_norm.scale.data_ptr<float>(),
                      layer.attn_norm.eps, t_dim, e, h.data_ptr<float>());
  RETURN_IF_ERROR(layer.q_proj->forward(h, q));
  RETURN_IF_ERROR(layer.k_proj->forward(h, k));
  RETURN_IF_ERROR(layer.v_proj->forward(h, v));

  kernels::RopeApplyF32(q.data_ptr<float>(), t_dim, config_.num_heads,
                        config_.head_dim, request.positions.data(),
                        rope_.cos().data_ptr<float>(),
                        rope_.sin().data_ptr<float>());
  kernels::RopeApplyF32(k.data_ptr<float>(), t_dim, config_.num_kv_heads,
                        config_.head_dim, request.positions.data(),
                        rope_.cos().data_ptr<float>(),
                        rope_.sin().data_ptr<float>());

  // Attention: append this layer's K/V and read the accumulated cache back.
  if (!request.caches.empty()) {
    // Batched/ragged path (M9-T07): per-sequence append + batched attention.
    RETURN_IF_ERROR(
        BatchedAttention(request, layer_index, k, v, q, ctx, scale, t_dim));
  } else {
    // Single-sequence path. Append this call's K/V (token-major [T, Hkv, d]).
    ASSIGN_OR_RETURN(const tensor::Tensor k3,
                     k.reshape(tensor::Shape{t_dim, config_.num_kv_heads,
                                             config_.head_dim}));
    ASSIGN_OR_RETURN(const tensor::Tensor v3,
                     v.reshape(tensor::Shape{t_dim, config_.num_kv_heads,
                                             config_.head_dim}));
    RETURN_IF_ERROR(request.cache->append(layer_index, k3, v3));

    // ctx [T, H, d] (contiguous with the [T, H·d] ctx slot). Decode (T == 1)
    // prefers the zero-copy paged fast path (paged-kv-cache.md §8.3): reading
    // K/V through the block table avoids the per-step full-history gather
    // `view()` performs, which is what keeps M8-T07 under the ≤10% decode
    // regression bound. On a non-paged cache (`SimpleKvCache` →
    // `Unimplemented`) it falls back to `view()` + the contiguous decode
    // kernel, so nothing changes for that path. Prefill always gathers `[Hkv,
    // P+T, d]` head-major via `view()` (the M8-T06 read path; a block-walking
    // paged prefill kernel is M12).
    if (t_dim == 1) {
      core::StatusOr<kvcache::PagedKvView> paged =
          request.cache->paged_view(layer_index);
      if (paged.ok()) {
        const kvcache::PagedKvView& pv = *paged;
        kernels::PagedDecodeAttentionF32(
            q.data_ptr<float>(), pv.k_slab, pv.v_slab, pv.block_table,
            pv.num_blocks, pv.length, config_.num_heads, config_.num_kv_heads,
            config_.head_dim, pv.block_size, pv.block_stride, scale,
            ctx.data_ptr<float>());
      } else if (core::IsUnimplemented(paged.status())) {
        ASSIGN_OR_RETURN(const kvcache::KvView kv,
                         request.cache->view(layer_index));
        kernels::DecodeAttentionF32(
            q.data_ptr<float>(), kv.k.data_ptr<float>(), kv.v.data_ptr<float>(),
            ctx.data_ptr<float>(), config_.num_heads, config_.num_kv_heads,
            config_.head_dim, kv.k.shape().dim(1), scale);
      } else {
        return paged.status();
      }
    } else {
      ASSIGN_OR_RETURN(const kvcache::KvView kv,
                       request.cache->view(layer_index));
      kernels::PrefillAttentionF32(
          q.data_ptr<float>(), kv.k.data_ptr<float>(), kv.v.data_ptr<float>(),
          ctx.data_ptr<float>(), t_dim, config_.num_heads, config_.num_kv_heads,
          config_.head_dim, kv.k.shape().dim(1), scale);
    }
  }
  RETURN_IF_ERROR(layer.o_proj->forward(ctx, tmp));

  // r = x + attn_out.
  kernels::AddF32(x.data_ptr<float>(), tmp.data_ptr<float>(),
                  r.data_ptr<float>(), t_dim * e);

  // --- MLP block: h = mlp_norm(r); gate/up = proj(h); silu-mul; down; x = r+m.
  kernels::RmsNormF32(r.data_ptr<float>(),
                      layer.mlp_norm.scale.data_ptr<float>(),
                      layer.mlp_norm.eps, t_dim, e, h.data_ptr<float>());
  tensor::Tensor gate = workspace_.gate(t_dim);
  tensor::Tensor up = workspace_.up(t_dim);
  RETURN_IF_ERROR(layer.gate_proj->forward(h, gate));
  RETURN_IF_ERROR(layer.up_proj->forward(h, up));
  kernels::SiluMulF32(gate.data_ptr<float>(), up.data_ptr<float>(),
                      gate.data_ptr<float>(),
                      t_dim * config_.intermediate_size);
  RETURN_IF_ERROR(layer.down_proj->forward(gate, tmp));
  kernels::AddF32(r.data_ptr<float>(), tmp.data_ptr<float>(),
                  x.data_ptr<float>(), t_dim * e);

  return core::OkStatus();
}

core::Status OptimizedModel::BatchedAttention(
    const ForwardRequest& request, int layer_index, const tensor::Tensor& k,
    const tensor::Tensor& v, const tensor::Tensor& q, const tensor::Tensor& ctx,
    float scale, std::int64_t t_dim) {
  const auto num_seqs = static_cast<std::int64_t>(request.caches.size());
  const std::int64_t hkv = config_.num_kv_heads;
  const std::int64_t d = config_.head_dim;
  const std::int64_t heads = config_.num_heads;
  const std::int32_t* cu = request.cu_seqlens.data();  // [B+1]

  // 1. Per-sequence K/V append, sliced from the flattened [ΣT, Hkv·d] rows by
  // cu_seqlens (§8.3). Each sequence appends through its own PagedKvCache, so
  // the M8 per-forward layer protocol, slot growth, and exhaustion seam are
  // unchanged — batching is a loop around the existing append.
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    const std::int64_t t0 = cu[b];
    const std::int64_t t1 = cu[b + 1];
    ASSIGN_OR_RETURN(const tensor::Tensor k_b, k.slice(0, t0, t1));
    ASSIGN_OR_RETURN(const tensor::Tensor v_b, v.slice(0, t0, t1));
    ASSIGN_OR_RETURN(const tensor::Tensor k3,
                     k_b.reshape(tensor::Shape{t1 - t0, hkv, d}));
    ASSIGN_OR_RETURN(const tensor::Tensor v3,
                     v_b.reshape(tensor::Shape{t1 - t0, hkv, d}));
    RETURN_IF_ERROR(request.caches[static_cast<std::size_t>(b)]->append(
        layer_index, k3, v3));
  }

  // 2. Decode fast path: every sequence contributes one token (ΣT == B) and the
  // caches are paged. Read each sequence's block table + length through the
  // zero-copy paged_view AFTER the append (so a boundary-crossing token's fresh
  // block is included — a pre-forward snapshot could not carry it), and run the
  // batched paged decode kernel over the shared slabs.
  if (t_dim == num_seqs) {
    core::StatusOr<kvcache::PagedKvView> pv0 =
        request.caches[0]->paged_view(layer_index);
    if (pv0.ok()) {
      batch_block_tables_.clear();
      batch_lengths_.clear();
      const float* k_slab = pv0->k_slab;
      const float* v_slab = pv0->v_slab;
      const std::int64_t bs = pv0->block_size;
      const std::int64_t stride = pv0->block_stride;
      for (std::int64_t b = 0; b < num_seqs; ++b) {
        ASSIGN_OR_RETURN(
            const kvcache::PagedKvView pv,
            request.caches[static_cast<std::size_t>(b)]->paged_view(
                layer_index));
        // All sequences share one BlockPool, so the layer slabs/strides must
        // match; the kernel takes one shared slab pair (§8.4).
        if (pv.k_slab != k_slab || pv.v_slab != v_slab || pv.block_size != bs ||
            pv.block_stride != stride) {
          return core::InvalidArgumentError(
              "OptimizedModel: batched decode requires all sequences to share "
              "one block pool (layer {}, sequence {})",
              layer_index, b);
        }
        batch_block_tables_.push_back(pv.block_table);
        batch_lengths_.push_back(pv.length);
      }
      kernels::PagedDecodeAttentionBatchedF32(
          q.data_ptr<float>(), k_slab, v_slab, batch_block_tables_.data(),
          batch_lengths_.data(), num_seqs, heads, hkv, d, bs, stride, scale,
          ctx.data_ptr<float>());
      return core::OkStatus();
    }
    if (!core::IsUnimplemented(pv0.status())) {
      return pv0.status();  // a real cache failure is not masked.
    }
    // Unimplemented → non-paged caches in a decode batch: fall through to the
    // varlen path (bit-identical: varlen prefill with T_b == 1 == decode).
  }

  // 3. Varlen path (any T_b > 1, or the non-paged decode fallback). Gather each
  // sequence's [Hkv, L_b, d] head-major history via view(layer) and run the
  // ragged prefill kernel over the per-sequence pointer arrays (§8.4).
  batch_views_.clear();
  batch_views_.reserve(static_cast<std::size_t>(num_seqs));
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    ASSIGN_OR_RETURN(
        kvcache::KvView kv,
        request.caches[static_cast<std::size_t>(b)]->view(layer_index));
    batch_views_.push_back(std::move(kv));
  }
  batch_k_ptrs_.clear();
  batch_v_ptrs_.clear();
  batch_l_dims_.clear();
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    const kvcache::KvView& kv = batch_views_[static_cast<std::size_t>(b)];
    batch_k_ptrs_.push_back(kv.k.data_ptr<float>());
    batch_v_ptrs_.push_back(kv.v.data_ptr<float>());
    batch_l_dims_.push_back(kv.k.shape().dim(1));
  }
  kernels::PrefillAttentionVarlenF32(
      q.data_ptr<float>(), cu, num_seqs, batch_k_ptrs_.data(),
      batch_v_ptrs_.data(), batch_l_dims_.data(), ctx.data_ptr<float>(), heads,
      hkv, d, scale);
  return core::OkStatus();
}

core::StatusOr<tensor::Tensor> OptimizedModel::RunStack(
    const ForwardRequest& request, std::int64_t t_dim) {
  ActivationHook* hook = request.hook;
  const std::int64_t hidden = config_.hidden_size;
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(config_.head_dim)));

  // Embedding → residual stream x [ΣT, E].
  tensor::Tensor x = workspace_.x(t_dim);
  RETURN_IF_ERROR(embed_.forward(request.token_ids, x));
  Emit(hook, "embeddings", -1, x);

  for (std::size_t i = 0; i < layers_.size(); ++i) {
    RETURN_IF_ERROR(
        ForwardLayer(request, layers_[i], static_cast<int>(i), t_dim, scale));
    if (hook != nullptr) {
      const std::string name = "layers." + std::to_string(i);
      const tensor::Tensor xv = workspace_.x(t_dim);
      Emit(hook, name, static_cast<int>(i), xv);
    }
  }

  // Final norm over all ΣT positions (matches the golden / reference; §5.2).
  tensor::Tensor normed = workspace_.h(t_dim);
  kernels::RmsNormF32(x.data_ptr<float>(), final_norm_.scale.data_ptr<float>(),
                      final_norm_.eps, t_dim, hidden, normed.data_ptr<float>());
  Emit(hook, "final_norm", -1, normed);
  return normed;
}

core::StatusOr<tensor::Tensor> OptimizedModel::forward(
    const ForwardRequest& request) {
  // The ragged/batched forward (cu_seqlens + per-sequence caches, §8) runs its
  // own validation + logits gather.
  if (!request.cu_seqlens.empty() || !request.caches.empty()) {
    return ForwardBatched(request);
  }
  // Front-load every input check (§5.3) so a failure never touches the cache —
  // identical to ReferenceModel::forward (the error-path tests match 1:1).
  const auto t_dim = static_cast<std::int64_t>(request.token_ids.size());
  if (t_dim < 1) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: token_ids must be non-empty (T >= 1)");
  }
  const auto num_positions =
      static_cast<std::int64_t>(request.positions.size());
  if (num_positions != t_dim) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: positions length {} must equal token_ids "
        "length {}",
        num_positions, t_dim);
  }
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t id = request.token_ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= config_.vocab_size) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: token_ids[{}] = {} out of range [0, {})", t,
          id, config_.vocab_size);
    }
  }
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t pos = request.positions[static_cast<std::size_t>(t)];
    if (pos < 0 || pos >= config_.max_position_embeddings) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: positions[{}] = {} out of range [0, {})", t,
          pos, config_.max_position_embeddings);
    }
  }
  if (request.cache == nullptr) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: cache must be non-null");
  }
  const kvcache::CacheGeometry cache_geom = request.cache->geometry();
  if (cache_geom.num_layers != geometry_.num_layers ||
      cache_geom.num_kv_heads != geometry_.num_kv_heads ||
      cache_geom.head_dim != geometry_.head_dim ||
      cache_geom.dtype != geometry_.dtype) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: cache geometry (layers={}, kv_heads={}, "
        "head_dim={}) does not match the model's (layers={}, kv_heads={}, "
        "head_dim={})",
        cache_geom.num_layers, cache_geom.num_kv_heads, cache_geom.head_dim,
        geometry_.num_layers, geometry_.num_kv_heads, geometry_.head_dim);
  }
  const std::int64_t p_len = request.cache->length();
  if (p_len + t_dim > request.cache->capacity()) {
    return core::ResourceExhaustedError(
        "OptimizedModel::forward: cache length {} + {} new tokens exceeds "
        "capacity {}",
        p_len, t_dim, request.cache->capacity());
  }

  // Size the workspace for this call (grow-on-demand, §6.3). This is the only
  // allocation that can fail; it happens before any kernel runs or the cache is
  // touched, so a failure leaves the cache unchanged.
  RETURN_IF_ERROR(workspace_.EnsureCapacity(t_dim));

  ActivationHook* hook = request.hook;
  const std::int64_t vocab = config_.vocab_size;

  ASSIGN_OR_RETURN(const tensor::Tensor normed, RunStack(request, t_dim));

  // lm_head projection. kLast projects only the last position (GEMV); kAll
  // projects every position (GEMM). Logits are freshly allocated, caller-owned
  // (§6.3) — never a workspace view.
  if (request.logits_mode == LogitsMode::kLast) {
    ASSIGN_OR_RETURN(const tensor::Tensor last,
                     normed.slice(0, t_dim - 1, t_dim));  // [1, E], contiguous
    ASSIGN_OR_RETURN(tensor::Tensor logits,
                     tensor::ops::zeros(tensor::Shape{1, vocab},
                                        tensor::DataType::kFloat32));
    RETURN_IF_ERROR(lm_head_->forward(last, logits));
    Emit(hook, "logits", -1, logits);
    return logits;
  }
  ASSIGN_OR_RETURN(tensor::Tensor logits,
                   tensor::ops::zeros(tensor::Shape{t_dim, vocab},
                                      tensor::DataType::kFloat32));
  RETURN_IF_ERROR(lm_head_->forward(normed, logits));
  Emit(hook, "logits", -1, logits);
  return logits;
}

core::StatusOr<std::int64_t> OptimizedModel::ValidateBatched(
    const ForwardRequest& request) const {
  // Front-load every batch check (§5.3, §8) so a failure never touches a cache.
  const auto num_seqs = static_cast<std::int64_t>(request.caches.size());
  if (num_seqs == 0 || request.cu_seqlens.empty()) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: a batched forward needs non-empty "
        "caches and cu_seqlens");
  }
  const auto n_cu = static_cast<std::int64_t>(request.cu_seqlens.size());
  if (n_cu != num_seqs + 1) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: cu_seqlens length {} must be caches "
        "length + 1 = {}",
        n_cu, num_seqs + 1);
  }
  if (request.cu_seqlens[0] != 0) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: cu_seqlens[0] must be 0, got {}",
        request.cu_seqlens[0]);
  }
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    if (request.cu_seqlens[static_cast<std::size_t>(b + 1)] <=
        request.cu_seqlens[static_cast<std::size_t>(b)]) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: cu_seqlens must be strictly increasing "
          "(each sequence has >= 1 token); violated at {}",
          b);
    }
  }
  const std::int64_t t_dim =
      request.cu_seqlens[static_cast<std::size_t>(num_seqs)];  // ΣT
  const auto n_tok = static_cast<std::int64_t>(request.token_ids.size());
  const auto n_pos = static_cast<std::int64_t>(request.positions.size());
  if (n_tok != t_dim || n_pos != t_dim) {
    return core::InvalidArgumentError(
        "OptimizedModel::forward: token_ids/positions length must equal "
        "cu_seqlens.back() = {}",
        t_dim);
  }
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t id = request.token_ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= config_.vocab_size) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: token_ids[{}] = {} out of range [0, {})", t,
          id, config_.vocab_size);
    }
    const std::int32_t pos = request.positions[static_cast<std::size_t>(t)];
    if (pos < 0 || pos >= config_.max_position_embeddings) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: positions[{}] = {} out of range [0, {})", t,
          pos, config_.max_position_embeddings);
    }
  }
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    kvcache::KvCache* cache = request.caches[static_cast<std::size_t>(b)];
    if (cache == nullptr) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: caches[{}] is null", b);
    }
    const kvcache::CacheGeometry cg = cache->geometry();
    if (cg.num_layers != geometry_.num_layers ||
        cg.num_kv_heads != geometry_.num_kv_heads ||
        cg.head_dim != geometry_.head_dim || cg.dtype != geometry_.dtype) {
      return core::InvalidArgumentError(
          "OptimizedModel::forward: caches[{}] geometry does not match the "
          "model's",
          b);
    }
    const std::int64_t t_b =
        request.cu_seqlens[static_cast<std::size_t>(b + 1)] -
        request.cu_seqlens[static_cast<std::size_t>(b)];
    if (cache->length() + t_b > cache->capacity()) {
      return core::ResourceExhaustedError(
          "OptimizedModel::forward: caches[{}] length {} + {} new tokens "
          "exceeds capacity {}",
          b, cache->length(), t_b, cache->capacity());
    }
  }
  return t_dim;
}

core::StatusOr<tensor::Tensor> OptimizedModel::ForwardBatched(
    const ForwardRequest& request) {
  ASSIGN_OR_RETURN(const std::int64_t t_dim, ValidateBatched(request));
  const auto num_seqs = static_cast<std::int64_t>(request.caches.size());

  RETURN_IF_ERROR(workspace_.EnsureCapacity(t_dim));

  ActivationHook* hook = request.hook;
  const std::int64_t hidden = config_.hidden_size;
  const std::int64_t vocab = config_.vocab_size;

  ASSIGN_OR_RETURN(const tensor::Tensor normed, RunStack(request, t_dim));

  // lm_head. kLast → [B, V]: gather each sequence's last hidden row
  // (cu_seqlens[b+1]−1) into a [B, E] buffer (the free `tmp` slot), one GEMM.
  // kAll → [ΣT, V]. Row m of a batched GEMM is bit-identical to the
  // single-row GEMV of that row (packed_gemm_test.GemvMatchesGemmRow), so a
  // sequence's logits match a standalone run (§8.5). Logits are caller-owned.
  if (request.logits_mode == LogitsMode::kLast) {
    const tensor::Tensor gathered = workspace_.tmp(num_seqs);  // [B, E]
    auto* dst = gathered.data_ptr<float>();
    const auto* src = normed.data_ptr<float>();
    for (std::int64_t b = 0; b < num_seqs; ++b) {
      const std::int64_t last =
          request.cu_seqlens[static_cast<std::size_t>(b + 1)] - 1;
      const float* row = src + (last * hidden);
      std::copy(row, row + hidden, dst + (b * hidden));
    }
    ASSIGN_OR_RETURN(tensor::Tensor logits,
                     tensor::ops::zeros(tensor::Shape{num_seqs, vocab},
                                        tensor::DataType::kFloat32));
    RETURN_IF_ERROR(lm_head_->forward(gathered, logits));
    Emit(hook, "logits", -1, logits);
    return logits;
  }
  ASSIGN_OR_RETURN(tensor::Tensor logits,
                   tensor::ops::zeros(tensor::Shape{t_dim, vocab},
                                      tensor::DataType::kFloat32));
  RETURN_IF_ERROR(lm_head_->forward(normed, logits));
  Emit(hook, "logits", -1, logits);
  return logits;
}

}  // namespace engine::model
