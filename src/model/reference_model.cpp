#include "model/reference_model.h"

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/modules.h"
#include "tensor/dtype.h"
#include "tensor/ops.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::model {

namespace {

// Removes and returns the canonical weight `name` from `weights`. A required
// weight the loader guarantees present, but the model must not CHECK on
// caller-constructed input (the registry test builds a LoadedModel directly),
// so a miss is a recoverable InvalidArgument naming it.
[[nodiscard]] core::StatusOr<tensor::Tensor> TakeWeight(
    std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name) {
  const auto it = weights.find(name);
  if (it == weights.end()) {
    return core::InvalidArgumentError(
        "ReferenceModel: missing required weight '{}'", name);
  }
  tensor::Tensor w = std::move(it->second);
  weights.erase(it);
  return w;
}

// Returns the canonical weight `name` if present (a copy — bias handles are
// shared with the map's storage, zero-copy), else nullopt. Used for optional
// per-projection biases: the weight map only carries `.bias` entries for the
// architectures/projections that have them (Qwen2 biases q/k/v, not o), so the
// presence of the entry is the authority — no arch logic is hardcoded here.
[[nodiscard]] std::optional<tensor::Tensor> OptionalWeight(
    const std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name) {
  const auto it = weights.find(name);
  if (it == weights.end()) {
    return std::nullopt;
  }
  return it->second;
}

// Builds a `ReferenceLinear` for canonical `name` (+ optional `name`.bias when
// `has_bias` and the bias entry is present), returned as an owning `Linear`
// handle so `Attention`/`Mlp` can hold it behind the interface.
[[nodiscard]] core::StatusOr<std::unique_ptr<Linear>> BuildLinear(
    std::unordered_map<std::string, tensor::Tensor>& weights,
    const std::string& name, bool has_bias) {
  ASSIGN_OR_RETURN(tensor::Tensor weight, TakeWeight(weights, name));
  std::optional<tensor::Tensor> bias;
  if (has_bias) {
    bias = OptionalWeight(weights, name.substr(0, name.size() - 7) + ".bias");
  }
  ASSIGN_OR_RETURN(ReferenceLinear linear,
                   ReferenceLinear::Create(std::move(weight), std::move(bias)));
  return std::unique_ptr<Linear>(
      std::make_unique<ReferenceLinear>(std::move(linear)));
}

// Delivers one activation event to `hook` iff a hook is attached. The event
// (and its name/tensor view) is built only inside the guard, so a null hook is
// zero cost — no ActivationEvent, no string (§11).
void Emit(ActivationHook* hook, std::string_view name, int layer,
          const tensor::Tensor& tensor) {
  if (hook != nullptr) {
    const ActivationEvent event{.name = name, .layer = layer, .tensor = tensor};
    hook->on_activation(event);
  }
}

}  // namespace

core::StatusOr<std::unique_ptr<ReferenceModel>> ReferenceModel::Create(
    LoadedModel loaded) {
  const ModelConfig& config = loaded.config;
  std::unordered_map<std::string, tensor::Tensor>& weights = loaded.weights;

  const int head_dim = config.head_dim;
  const int num_heads = config.num_heads;
  const int num_kv_heads = config.num_kv_heads;

  // Embedding + final norm + lm_head. lm_head binds to whatever handle the map
  // holds under `lm_head.weight` — M4 aliases it to `embed_tokens.weight` when
  // tied, so tying is a binding decision, not a pointer-equality assumption.
  ASSIGN_OR_RETURN(tensor::Tensor embed_weight,
                   TakeWeight(weights, "embed_tokens.weight"));
  ASSIGN_OR_RETURN(Embedding embed, Embedding::Create(std::move(embed_weight)));

  ASSIGN_OR_RETURN(tensor::Tensor final_norm_weight,
                   TakeWeight(weights, "final_norm.weight"));
  ASSIGN_OR_RETURN(
      RmsNorm final_norm,
      RmsNorm::Create(std::move(final_norm_weight), config.rms_norm_eps));

  ASSIGN_OR_RETURN(std::unique_ptr<Linear> lm_head,
                   BuildLinear(weights, "lm_head.weight", /*has_bias=*/false));

  std::vector<DecoderLayer> layers;
  layers.reserve(static_cast<std::size_t>(config.num_layers));
  for (int i = 0; i < config.num_layers; ++i) {
    const std::string pre = "layers." + std::to_string(i) + ".";

    ASSIGN_OR_RETURN(tensor::Tensor attn_norm_w,
                     TakeWeight(weights, pre + "attn_norm.weight"));
    ASSIGN_OR_RETURN(RmsNorm attn_norm, RmsNorm::Create(std::move(attn_norm_w),
                                                        config.rms_norm_eps));

    ASSIGN_OR_RETURN(std::unique_ptr<Linear> q_proj,
                     BuildLinear(weights, pre + "attn.q_proj.weight",
                                 config.attention_bias));
    ASSIGN_OR_RETURN(std::unique_ptr<Linear> k_proj,
                     BuildLinear(weights, pre + "attn.k_proj.weight",
                                 config.attention_bias));
    ASSIGN_OR_RETURN(std::unique_ptr<Linear> v_proj,
                     BuildLinear(weights, pre + "attn.v_proj.weight",
                                 config.attention_bias));
    ASSIGN_OR_RETURN(std::unique_ptr<Linear> o_proj,
                     BuildLinear(weights, pre + "attn.o_proj.weight",
                                 config.attention_bias));

    ASSIGN_OR_RETURN(Rope rope, Rope::Create(head_dim, config.rope_theta,
                                             config.rope_scaling,
                                             config.max_position_embeddings));
    ASSIGN_OR_RETURN(
        Attention attn,
        Attention::Create(std::move(q_proj), std::move(k_proj),
                          std::move(v_proj), std::move(o_proj), std::move(rope),
                          num_heads, num_kv_heads, head_dim));

    ASSIGN_OR_RETURN(tensor::Tensor mlp_norm_w,
                     TakeWeight(weights, pre + "mlp_norm.weight"));
    ASSIGN_OR_RETURN(RmsNorm mlp_norm, RmsNorm::Create(std::move(mlp_norm_w),
                                                       config.rms_norm_eps));

    ASSIGN_OR_RETURN(
        std::unique_ptr<Linear> gate_proj,
        BuildLinear(weights, pre + "mlp.gate_proj.weight", /*has_bias=*/false));
    ASSIGN_OR_RETURN(
        std::unique_ptr<Linear> up_proj,
        BuildLinear(weights, pre + "mlp.up_proj.weight", /*has_bias=*/false));
    ASSIGN_OR_RETURN(
        std::unique_ptr<Linear> down_proj,
        BuildLinear(weights, pre + "mlp.down_proj.weight", /*has_bias=*/false));
    ASSIGN_OR_RETURN(Mlp mlp,
                     Mlp::Create(std::move(gate_proj), std::move(up_proj),
                                 std::move(down_proj)));

    ASSIGN_OR_RETURN(DecoderLayer layer,
                     DecoderLayer::Create(std::move(attn_norm), std::move(attn),
                                          std::move(mlp_norm), std::move(mlp)));
    layers.push_back(std::move(layer));
  }

  const kvcache::CacheGeometry geometry{.num_layers = config.num_layers,
                                        .num_kv_heads = num_kv_heads,
                                        .head_dim = head_dim,
                                        .dtype = tensor::DataType::kFloat32};

  // The parameterized constructor is private (accessible here); make_unique
  // then moves the temporary via the public move constructor — no raw `new`.
  return std::make_unique<ReferenceModel>(
      ReferenceModel(config, geometry, std::move(embed), std::move(layers),
                     std::move(final_norm), std::move(lm_head)));
}

core::StatusOr<tensor::Tensor> ReferenceModel::forward(
    const ForwardRequest& request) {
  // The ragged/batched forward (cu_seqlens + per-sequence caches) lands in
  // M9-T07; until then a non-empty batch is rejected rather than silently
  // ignored (scheduler-runtime.md §8.1). The single-sequence path below is the
  // B==1 special case (both spans empty).
  if (!request.cu_seqlens.empty() || !request.caches.empty()) {
    return core::UnimplementedError(
        "ReferenceModel::forward: batched (cu_seqlens/caches) forward is not "
        "implemented until M9-T07");
  }
  // Front-load every input check (§5.3) so a failure never touches the cache:
  // the per-layer append inside attention only runs once all of these pass.
  const auto t_dim = static_cast<std::int64_t>(request.token_ids.size());
  if (t_dim < 1) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: token_ids must be non-empty (T >= 1)");
  }
  const auto num_positions =
      static_cast<std::int64_t>(request.positions.size());
  if (num_positions != t_dim) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: positions length {} must equal token_ids "
        "length {}",
        num_positions, t_dim);
  }
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t id = request.token_ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= config_.vocab_size) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: token_ids[{}] = {} out of range [0, {})", t,
          id, config_.vocab_size);
    }
  }
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t pos = request.positions[static_cast<std::size_t>(t)];
    if (pos < 0 || pos >= config_.max_position_embeddings) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: positions[{}] = {} out of range [0, {})", t,
          pos, config_.max_position_embeddings);
    }
  }
  if (request.cache == nullptr) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: cache must be non-null");
  }
  const kvcache::CacheGeometry cache_geom = request.cache->geometry();
  if (cache_geom.num_layers != geometry_.num_layers ||
      cache_geom.num_kv_heads != geometry_.num_kv_heads ||
      cache_geom.head_dim != geometry_.head_dim ||
      cache_geom.dtype != geometry_.dtype) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: cache geometry (layers={}, kv_heads={}, "
        "head_dim={}) does not match the model's (layers={}, kv_heads={}, "
        "head_dim={})",
        cache_geom.num_layers, cache_geom.num_kv_heads, cache_geom.head_dim,
        geometry_.num_layers, geometry_.num_kv_heads, geometry_.head_dim);
  }
  if (request.cache->length() + t_dim > request.cache->capacity()) {
    return core::ResourceExhaustedError(
        "ReferenceModel::forward: cache length {} + {} new tokens exceeds "
        "capacity {}",
        request.cache->length(), t_dim, request.cache->capacity());
  }

  const std::int64_t hidden = config_.hidden_size;
  ActivationHook* hook = request.hook;

  // Embedding → x [T, E].
  ASSIGN_OR_RETURN(tensor::Tensor x,
                   tensor::ops::zeros(tensor::Shape{t_dim, hidden},
                                      tensor::DataType::kFloat32));
  RETURN_IF_ERROR(embed_.forward(request.token_ids, x));
  Emit(hook, "embeddings", -1, x);

  // N decoder layers; x is ping-ponged into a fresh buffer each layer.
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    ASSIGN_OR_RETURN(tensor::Tensor next,
                     tensor::ops::zeros(tensor::Shape{t_dim, hidden},
                                        tensor::DataType::kFloat32));
    RETURN_IF_ERROR(layers_[i].forward(
        x, request.positions, static_cast<int>(i), *request.cache, next));
    x = std::move(next);
    if (hook != nullptr) {
      const std::string name = "layers." + std::to_string(i);
      Emit(hook, name, static_cast<int>(i), x);
    }
  }

  // Final norm over all T positions (matches the golden; cheap in the
  // reference).
  ASSIGN_OR_RETURN(tensor::Tensor normed,
                   tensor::ops::zeros(tensor::Shape{t_dim, hidden},
                                      tensor::DataType::kFloat32));
  RETURN_IF_ERROR(final_norm_.forward(x, normed));
  Emit(hook, "final_norm", -1, normed);

  // lm_head projection. kLast restricts it to the final position's hidden state
  // (a real compute saving that is also correct, §5.2); kAll projects every
  // position.
  const std::int64_t vocab = config_.vocab_size;
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
