#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/modules.h"
#include "model/optimized_embedding.h"
#include "model/packed_linear.h"
#include "model/workspace.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <memory>
#include <vector>

// The optimized `Model` implementation (M6-T07; design:
// docs/design/optimized-cpu-execution.md §2.2, §5, §6, §9). The `kOptimized`
// backend behind the same `Model` interface as `ReferenceModel`: at build it
// repacks every projection weight into the K-major panel layout (§3, via
// `PackedLinear`) and honors tied embeddings by sharing the packed lm_head's
// one physical copy (§7); at `forward` it runs a parallel-implementation graph
// (NOT the reference `DecoderLayer` classes — §2.2) out of one reused
// `Workspace`, calling the dispatched `kernels::` GEMM/norm/activation/RoPE/
// attention kernels. Every kernel is validated against the `cpu::` oracle
// (§10); the whole model is validated against `ReferenceModel` token-for-token
// (M6-T07 acceptance).
//
// This header names no `kernels` type (the packed bytes are plain
// `tensor::Tensor`s inside `PackedLinear`/`OptimizedEmbedding`), so the
// `model → kernels` edge stays a PRIVATE link (§2.1), exactly as PackedLinear.
//
// The architecture registry (`registry.h`) dispatches to
// `OptimizedModel::Create` for `Backend::kOptimized`; the same Llama/Qwen2
// family builder serves both backends (§9).

namespace engine::model {

class OptimizedModel final : public Model {
 public:
  // Binds every module's weights from `loaded.weights` by canonical name
  // (weight_map.h), identically to `ReferenceModel::Create` but building
  // optimized modules: `PackedLinear` for every projection (q/k/v/o, gate/up/
  // down, lm_head), the norm scales converted to fp32 once (§4), one shared
  // `Rope` table (a deliberate divergence from the reference's per-layer copies
  // — the tables are position-only, so one suffices; §2.2), and an
  // `OptimizedEmbedding` that shares the packed lm_head storage when tied
  // (`config.tie_word_embeddings`, §7) or holds the model's own `[V, E]` table
  // otherwise. Repacking is progress-logged per layer. A missing required
  // weight, or a module-shape disagreement, → InvalidArgument naming it; an
  // unsupported `rope_scaling` → Unimplemented (from `Rope::Create`).
  [[nodiscard]] static core::StatusOr<std::unique_ptr<OptimizedModel>> Create(
      LoadedModel loaded);

  [[nodiscard]] core::StatusOr<tensor::Tensor> forward(
      const ForwardRequest& request) override;

  [[nodiscard]] const ModelConfig& config() const override { return config_; }
  [[nodiscard]] kvcache::CacheGeometry cache_geometry() const override {
    return geometry_;
  }

 private:
  // fp32 norm scale + eps, converted once at build (§4): the kernel entry
  // `kernels::RmsNormF32` takes a `const float*`, unlike the reference
  // `RmsNorm` which keeps a checkpoint handle and widens per call.
  struct NormWeights {
    tensor::Tensor scale;  // [E], fp32
    float eps = 0.0F;
  };

  // One decoder layer's optimized modules. Projections are held behind the
  // `Linear` interface (owning `PackedLinear`s) so the shapes were validated at
  // build; the forward reads them as `Linear&`.
  struct Layer {
    NormWeights attn_norm;
    std::unique_ptr<Linear> q_proj;
    std::unique_ptr<Linear> k_proj;
    std::unique_ptr<Linear> v_proj;
    std::unique_ptr<Linear> o_proj;
    NormWeights mlp_norm;
    std::unique_ptr<Linear> gate_proj;
    std::unique_ptr<Linear> up_proj;
    std::unique_ptr<Linear> down_proj;
  };

  OptimizedModel(ModelConfig config, kvcache::CacheGeometry geometry,
                 OptimizedEmbedding embed, Rope rope, std::vector<Layer> layers,
                 NormWeights final_norm, std::unique_ptr<Linear> lm_head,
                 Workspace workspace)
      : config_(std::move(config)),
        geometry_(geometry),
        embed_(std::move(embed)),
        rope_(std::move(rope)),
        layers_(std::move(layers)),
        final_norm_(std::move(final_norm)),
        lm_head_(std::move(lm_head)),
        workspace_(std::move(workspace)) {}

  // One decoder layer's forward, in place on the workspace residual stream.
  // Handles both the single-sequence path (`request.caches` empty) and the
  // ragged/batched path (M9-T07) — the non-attention ops run identically over
  // the flattened `[ΣT, E]` workspace; only K/V append + the attention kernel
  // branch (`BatchedAttention`).
  [[nodiscard]] core::Status ForwardLayer(const ForwardRequest& request,
                                          const Layer& layer, int layer_index,
                                          std::int64_t t_dim, float scale);

  // The batched attention section of a layer (M9-T07): per-sequence K/V append
  // sliced by `cu_seqlens`, then the batched paged-decode kernel (every T_b ==
  // 1 over paged caches) or the varlen prefill kernel (any T_b > 1, or
  // non-paged caches). `q`/`k`/`v`/`ctx` are the layer's workspace projections.
  [[nodiscard]] core::Status BatchedAttention(
      const ForwardRequest& request, int layer_index, const tensor::Tensor& k,
      const tensor::Tensor& v, const tensor::Tensor& q,
      const tensor::Tensor& ctx, float scale, std::int64_t t_dim);

  // Embedding → N layers → final norm over `t_dim` flattened tokens; returns
  // the normed `[t_dim, E]` workspace view (the `h` slot). Shared by the
  // single-sequence and batched forward paths. Assumes validation + workspace
  // sizing done by the caller.
  [[nodiscard]] core::StatusOr<tensor::Tensor> RunStack(
      const ForwardRequest& request, std::int64_t t_dim);

  // The ragged/batched forward (M9-T07; scheduler-runtime.md §8): validates the
  // batch, sizes the workspace to ΣT, runs the stack, and projects logits —
  // `[B, V]` for kLast (last row per sequence), `[ΣT, V]` for kAll.
  [[nodiscard]] core::StatusOr<tensor::Tensor> ForwardBatched(
      const ForwardRequest& request);

  // Front-loads every batch check (§5.3, §8) — cu_seqlens well-formedness,
  // token/position lengths + ranges, and per-member cache null/geometry/
  // capacity — returning ΣT on success. On error nothing is mutated.
  [[nodiscard]] core::StatusOr<std::int64_t> ValidateBatched(
      const ForwardRequest& request) const;

  ModelConfig config_;
  kvcache::CacheGeometry geometry_;
  OptimizedEmbedding embed_;
  Rope rope_;  // one shared table, applied to Q and K every layer (§2.2)
  std::vector<Layer> layers_;
  NormWeights final_norm_;
  std::unique_ptr<Linear> lm_head_;
  Workspace workspace_;

  // Reusable per-layer scratch for the batched attention paths (M9-T07),
  // cleared and refilled each layer so a steady-state batched decode allocates
  // nothing (the Workspace/BatchedSampler discipline). The varlen path holds
  // the B gathered K/V views alive while their data pointers feed the kernel.
  std::vector<const std::int32_t*> batch_block_tables_;
  std::vector<std::int64_t> batch_lengths_;
  std::vector<kvcache::KvView> batch_views_;
  std::vector<const float*> batch_k_ptrs_;
  std::vector<const float*> batch_v_ptrs_;
  std::vector<std::int64_t> batch_l_dims_;
};

}  // namespace engine::model
