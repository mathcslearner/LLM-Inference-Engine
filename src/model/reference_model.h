#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/modules.h"
#include "tensor/tensor.h"

#include <memory>
#include <vector>

// The reference (oracle) `Model` implementation (M5-T07; design:
// docs/design/model-execution.md §4.3, §5, §8). Assembles the M5 modules —
// `Embedding` → N `DecoderLayer`s → `RmsNorm` (final) → lm_head `Linear` — from
// a `LoadedModel`, honoring tied embeddings, and runs the fp32 forward pass via
// the `cpu::` ops. Deliberately never optimized (cpu-backend.md §2.1): it is
// the correctness oracle every M6 kernel validates against.
//
// The architecture registry (`registry.h`, M5-T08) wraps this with `BuildModel`
// and the HF-arch-string dispatch: `ReferenceModel::Create` is the reference
// family builder registered for both `LlamaForCausalLM` and `Qwen2ForCausalLM`.

namespace engine::model {

class ReferenceModel final : public Model {
 public:
  // Binds every module's weights from `loaded.weights` by canonical name
  // (weight_map.h): `embed_tokens.weight`, `final_norm.weight`,
  // `lm_head.weight`, and per layer `attn_norm`/`attn.{q,k,v,o}_proj`(+`.bias`
  // when the map carries it)/`mlp_norm`/`mlp.{gate,up,down}_proj`. Tied
  // embeddings are honored as a binding decision (§4.3): lm_head binds to
  // whatever handle the map holds under `lm_head.weight`, which M4 aliases to
  // `embed_tokens.weight` when tied — no pointer-equality assumption. A missing
  // required weight, or a module-shape disagreement, → InvalidArgument naming
  // it; an unsupported `rope_scaling` → Unimplemented (from `Rope::Create`).
  [[nodiscard]] static core::StatusOr<std::unique_ptr<ReferenceModel>> Create(
      LoadedModel loaded);

  [[nodiscard]] core::StatusOr<tensor::Tensor> forward(
      const ForwardRequest& request) override;

  [[nodiscard]] const ModelConfig& config() const override { return config_; }
  [[nodiscard]] kvcache::CacheGeometry cache_geometry() const override {
    return geometry_;
  }

 private:
  // The ragged/batched forward (M9-T07; scheduler-runtime.md §8): the oracle
  // realizes it as "run the single-sequence forward per member and concatenate
  // logits" — bit-identical to standalone runs by construction, and the
  // cheapest reference for the optimized batched path. Validates the whole
  // batch before touching any cache.
  [[nodiscard]] core::StatusOr<tensor::Tensor> ForwardBatched(
      const ForwardRequest& request);

  // Front-loads every batch check (§5.3, §8) — cu_seqlens well-formedness,
  // token/position lengths + ranges, and per-member cache null/geometry/
  // capacity — returning ΣT on success. On error nothing is mutated.
  [[nodiscard]] core::StatusOr<std::int64_t> ValidateBatched(
      const ForwardRequest& request) const;

  ReferenceModel(ModelConfig config, kvcache::CacheGeometry geometry,
                 Embedding embed, std::vector<DecoderLayer> layers,
                 RmsNorm final_norm, std::unique_ptr<Linear> lm_head)
      : config_(std::move(config)),
        geometry_(geometry),
        embed_(std::move(embed)),
        layers_(std::move(layers)),
        final_norm_(std::move(final_norm)),
        lm_head_(std::move(lm_head)) {}

  ModelConfig config_;
  kvcache::CacheGeometry geometry_;
  Embedding embed_;
  std::vector<DecoderLayer> layers_;
  RmsNorm final_norm_;
  std::unique_ptr<Linear> lm_head_;
};

}  // namespace engine::model
