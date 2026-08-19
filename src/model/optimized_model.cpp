#include "model/optimized_model.h"

#include "core/logging.h"
#include "core/status.h"
#include "kernels/activation.h"
#include "kernels/attention.h"
#include "kernels/elementwise.h"
#include "kernels/norm.h"
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

#include <cmath>
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
                                          std::int64_t t_dim,
                                          std::int64_t /*p_len*/, float scale) {
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

  // Append this call's K/V (token-major [T, Hkv, d]) and read the accumulated
  // [Hkv, P+T, d] head-major view back (§8, the SimpleKvCache gather seam).
  ASSIGN_OR_RETURN(
      const tensor::Tensor k3,
      k.reshape(tensor::Shape{t_dim, config_.num_kv_heads, config_.head_dim}));
  ASSIGN_OR_RETURN(
      const tensor::Tensor v3,
      v.reshape(tensor::Shape{t_dim, config_.num_kv_heads, config_.head_dim}));
  RETURN_IF_ERROR(request.cache->append(layer_index, k3, v3));
  ASSIGN_OR_RETURN(const kvcache::KvView kv, request.cache->view(layer_index));
  const std::int64_t l_dim = kv.k.shape().dim(1);

  // ctx [T, H, d] (contiguous with the [T, H·d] ctx slot). Decode (T == 1) uses
  // the single-token specialization; prefill uses the blocked kernel.
  if (t_dim == 1) {
    kernels::DecodeAttentionF32(q.data_ptr<float>(), kv.k.data_ptr<float>(),
                                kv.v.data_ptr<float>(), ctx.data_ptr<float>(),
                                config_.num_heads, config_.num_kv_heads,
                                config_.head_dim, l_dim, scale);
  } else {
    kernels::PrefillAttentionF32(q.data_ptr<float>(), kv.k.data_ptr<float>(),
                                 kv.v.data_ptr<float>(), ctx.data_ptr<float>(),
                                 t_dim, config_.num_heads, config_.num_kv_heads,
                                 config_.head_dim, l_dim, scale);
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

core::StatusOr<tensor::Tensor> OptimizedModel::forward(
    const ForwardRequest& request) {
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
  const std::int64_t hidden = config_.hidden_size;
  const std::int64_t vocab = config_.vocab_size;
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(config_.head_dim)));

  // Embedding → residual stream x [T, E].
  tensor::Tensor x = workspace_.x(t_dim);
  RETURN_IF_ERROR(embed_.forward(request.token_ids, x));
  Emit(hook, "embeddings", -1, x);

  for (std::size_t i = 0; i < layers_.size(); ++i) {
    RETURN_IF_ERROR(ForwardLayer(request, layers_[i], static_cast<int>(i),
                                 t_dim, p_len, scale));
    if (hook != nullptr) {
      const std::string name = "layers." + std::to_string(i);
      const tensor::Tensor xv = workspace_.x(t_dim);
      Emit(hook, name, static_cast<int>(i), xv);
    }
  }

  // Final norm over all T positions (matches the golden / reference; §5.2).
  const tensor::Tensor normed = workspace_.h(t_dim);
  kernels::RmsNormF32(x.data_ptr<float>(), final_norm_.scale.data_ptr<float>(),
                      final_norm_.eps, t_dim, hidden, normed.data_ptr<float>());
  Emit(hook, "final_norm", -1, normed);

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

}  // namespace engine::model
