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

#include <algorithm>
#include <cstddef>
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
  // The ragged/batched forward (cu_seqlens + per-sequence caches, §8) is run
  // per member and concatenated. The single-sequence path below is the B==1
  // special case (both spans empty).
  if (!request.cu_seqlens.empty() || !request.caches.empty()) {
    return ForwardBatched(request);
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

core::StatusOr<std::int64_t> ReferenceModel::ValidateBatched(
    const ForwardRequest& request) const {
  // Front-load the batch checks (§5.3, §8) so a failure never touches any
  // cache.
  const auto num_seqs = static_cast<std::int64_t>(request.caches.size());
  if (num_seqs == 0 || request.cu_seqlens.empty()) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: a batched forward needs non-empty caches and "
        "cu_seqlens");
  }
  const auto n_cu = static_cast<std::int64_t>(request.cu_seqlens.size());
  if (n_cu != num_seqs + 1) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: cu_seqlens length {} must be caches length + "
        "1 = {}",
        n_cu, num_seqs + 1);
  }
  if (request.cu_seqlens[0] != 0) {
    return core::InvalidArgumentError(
        "ReferenceModel::forward: cu_seqlens[0] must be 0, got {}",
        request.cu_seqlens[0]);
  }
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    if (request.cu_seqlens[static_cast<std::size_t>(b + 1)] <=
        request.cu_seqlens[static_cast<std::size_t>(b)]) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: cu_seqlens must be strictly increasing "
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
        "ReferenceModel::forward: token_ids/positions length must equal "
        "cu_seqlens.back() = {}",
        t_dim);
  }
  for (std::int64_t t = 0; t < t_dim; ++t) {
    const std::int32_t id = request.token_ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= config_.vocab_size) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: token_ids[{}] = {} out of range [0, {})", t,
          id, config_.vocab_size);
    }
    const std::int32_t pos = request.positions[static_cast<std::size_t>(t)];
    if (pos < 0 || pos >= config_.max_position_embeddings) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: positions[{}] = {} out of range [0, {})", t,
          pos, config_.max_position_embeddings);
    }
  }
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    kvcache::KvCache* cache = request.caches[static_cast<std::size_t>(b)];
    if (cache == nullptr) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: caches[{}] is null", b);
    }
    const kvcache::CacheGeometry cg = cache->geometry();
    if (cg.num_layers != geometry_.num_layers ||
        cg.num_kv_heads != geometry_.num_kv_heads ||
        cg.head_dim != geometry_.head_dim || cg.dtype != geometry_.dtype) {
      return core::InvalidArgumentError(
          "ReferenceModel::forward: caches[{}] geometry does not match the "
          "model's",
          b);
    }
    const std::int64_t t_b =
        request.cu_seqlens[static_cast<std::size_t>(b + 1)] -
        request.cu_seqlens[static_cast<std::size_t>(b)];
    if (cache->length() + t_b > cache->capacity()) {
      return core::ResourceExhaustedError(
          "ReferenceModel::forward: caches[{}] length {} + {} new tokens "
          "exceeds capacity {}",
          b, cache->length(), t_b, cache->capacity());
    }
  }
  return t_dim;
}

core::StatusOr<tensor::Tensor> ReferenceModel::ForwardBatched(
    const ForwardRequest& request) {
  // Validated up front (nothing mutated on failure); per-member field checks
  // are re-run inside each sub-forward (which pass), so only an execution error
  // (e.g. shared-pool exhaustion) can surface mid-batch.
  ASSIGN_OR_RETURN(const std::int64_t t_dim, ValidateBatched(request));
  const auto num_seqs = static_cast<std::int64_t>(request.caches.size());
  const std::int64_t vocab = config_.vocab_size;
  const bool last_only = request.logits_mode == LogitsMode::kLast;
  const std::int64_t out_rows = last_only ? num_seqs : t_dim;
  ASSIGN_OR_RETURN(tensor::Tensor logits,
                   tensor::ops::zeros(tensor::Shape{out_rows, vocab},
                                      tensor::DataType::kFloat32));
  auto* out = logits.data_ptr<float>();

  // Run each member as a standalone single-sequence forward (recursion into the
  // B==1 path) over its token/position subspan and its own cache; concatenate
  // the per-member logits. Bit-identical to N standalone runs by construction
  // (§8.5). Hooks are suppressed per member (a batched hook stream is a debug
  // concern deferred; the oracle only needs correct logits).
  for (std::int64_t b = 0; b < num_seqs; ++b) {
    const std::int64_t t0 = request.cu_seqlens[static_cast<std::size_t>(b)];
    const std::int64_t t1 = request.cu_seqlens[static_cast<std::size_t>(b + 1)];
    const auto off = static_cast<std::size_t>(t0);
    const auto len = static_cast<std::size_t>(t1 - t0);
    const ForwardRequest sub{
        .token_ids = request.token_ids.subspan(off, len),
        .positions = request.positions.subspan(off, len),
        .cache = request.caches[static_cast<std::size_t>(b)],
        .logits_mode = request.logits_mode,
        .hook = nullptr};
    core::StatusOr<tensor::Tensor> member = forward(sub);
    if (!member.ok()) {
      return core::Status(member.status().code(),
                          "ReferenceModel::forward: sequence " +
                              std::to_string(b) + ": " +
                              std::string(member.status().message()));
    }
    const auto* src = member->data_ptr<float>();
    const std::int64_t rows = last_only ? 1 : (t1 - t0);
    const std::int64_t base = last_only ? b : t0;
    std::copy(src, src + (rows * vocab), out + (base * vocab));
  }
  return logits;
}

}  // namespace engine::model
