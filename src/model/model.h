#pragma once

#include "core/status.h"
#include "kvcache/kv_cache.h"
#include "model/config.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

// The `Model::forward` contract and the activation-hook seam (M5-T07; design:
// docs/design/model-execution.md §5, §11). `Model` and `kvcache::KvCache` are
// the polymorphic contracts both backends share: which implementation runs is a
// construction-time choice (§8), invisible to the generation loop and the tests
// above the interface. M5 ships the reference (`reference_model.h`); M6 adds
// the optimized implementation behind the same interface.
//
// Layering (ADR-002 Amendment 5, model-execution.md §2.1): this header names
// `kvcache::CacheGeometry` (returned by value) and `kvcache::KvCache` (the
// forward parameter), so `model → kvcache` is a PUBLIC link and this include is
// part of the model's public surface — the one place the ADR's "no public model
// header re-exports kvcache types" note is relaxed, because the `Model`
// contract itself is stated in terms of the cache.

namespace engine::model {

// Which positions the forward pass returns logits for (§5.2).
enum class LogitsMode : std::uint8_t {
  kLast,  // logits for the final position only → [1, V] (steady-state decode)
  kAll,   // logits for every position → [T, V] (perplexity, spec-verify)
};

// A borrowed view of an intermediate activation, delivered to an
// `ActivationHook` during `forward` (§11). The `name` matches the golden
// fixture keys — "embeddings", "layers.{i}", "final_norm", "logits" — so a
// debug hook can dump exactly the tensors the goldens hold and localize a
// per-layer regression. (M14-T02 adds "linear_input:<canonical weight name>"
// events for calibration; those land with that milestone, not M5-T07.) The
// `tensor` is borrowed and valid only during the `on_activation` call.
struct ActivationEvent {
  std::string_view name;
  int layer = -1;  // -1 for non-layer stages
  const tensor::Tensor& tensor;
};

// The single debug/observability seam (§11): M5's per-layer dumping, M14's
// calibration capture, and M16's per-layer spans all attach here. A hook is
// invoked with a *view*, not a materialized copy, so memory-bounded
// accumulation is possible. When no hook is attached (`ForwardRequest.hook ==
// nullptr`, the default), the forward pass makes no hook calls at all and
// builds no `ActivationEvent` — the nullptr check is the compile-time-friendly
// zero- cost gate M16-T03 needs.
class ActivationHook {
 public:
  virtual ~ActivationHook() = default;
  virtual void on_activation(const ActivationEvent& event) = 0;

 protected:
  ActivationHook() = default;
  ActivationHook(const ActivationHook&) = default;
  ActivationHook& operator=(const ActivationHook&) = default;
  ActivationHook(ActivationHook&&) = default;
  ActivationHook& operator=(ActivationHook&&) = default;
};

// One sequence's forward-pass inputs (§5.1). A struct, not loose parameters,
// because M9-T05 grows it into the batch-assembly bundle (cu_seqlens,
// per-sequence caches, per-request sampling metadata) — those are added as new
// fields, not a new signature. The single-sequence path is the `B == 1` special
// case: `cu_seqlens`/`caches` empty ⇒ the existing scalar flow over `cache`
// (scheduler-runtime.md §8.1).
struct ForwardRequest {
  // [Σ T_b], flattened batch-major. For B==1 this is the sequence's tokens.
  std::span<const std::int32_t> token_ids;  // each in [0, V)
  // [Σ T_b], per-token absolute positions; size == token_ids.size().
  std::span<const std::int32_t> positions;
  kvcache::KvCache* cache = nullptr;  // B==1 path: non-null, length P → P+T
  LogitsMode logits_mode = LogitsMode::kLast;
  ActivationHook* hook = nullptr;  // nullptr = no capture, zero cost

  // --- M9 batch additions (empty ⇒ the single-sequence path above) ---------
  // [B+1] prefix sums of the per-sequence token counts T_b: sequence b owns the
  // flattened range [cu_seqlens[b], cu_seqlens[b+1]) of token_ids/positions.
  // Prefill sequences have T_b = prompt_len_b; decode sequences have T_b = 1.
  // The `{}` default (empty) keeps existing single-sequence designated
  // initializers warning-clean (they omit these trailing fields under
  // -Wmissing-designated-field-initializers). It reads as redundant to
  // clang-tidy (a span default-constructs empty) — a two-tool conflict the
  // compiler warning wins, so the tidy check is suppressed here.
  std::span<const std::int32_t>
      cu_seqlens{};  // NOLINT(readability-redundant-member-init)
  // [B] one KV cache per sequence (each its own PagedKvCache over the shared
  // pool). K/V for sequence b is appended to caches[b] (§8.3). When non-empty,
  // `cache` is ignored and the forward runs the ragged batch. The batched
  // forward lands in M9-T07; until then both backends reject a non-empty
  // batch with `Unimplemented` (never silently ignore it). `{}` as above.
  std::span<kvcache::KvCache* const>
      caches{};  // NOLINT(readability-redundant-member-init)
};

// The executable model: embedding → N decoder layers → final norm → lm_head
// (§5). One `forward` per sequence, appending this call's K/V to the cache.
// Backend-agnostic — the generation loop (§10) and the goldens above it touch
// only this interface and `KvCache`.
class Model {
 public:
  virtual ~Model() = default;

  // Runs the forward pass for one sequence, appending this call's K/V to
  // `request.cache`. Returns freshly allocated fp32 logits: [1, V] for kLast,
  // [T, V] for kAll (full vocab, contiguous per position). Malformed inputs are
  // recoverable Status (§5.3): a bad T/positions/id/position, a null or
  // geometry-mismatched cache → InvalidArgument; an over-capacity cache →
  // ResourceExhausted. On error the cache is left unchanged (validation is
  // front-loaded, ADR-003).
  [[nodiscard]] virtual core::StatusOr<tensor::Tensor> forward(
      const ForwardRequest& request) = 0;

  [[nodiscard]] virtual const ModelConfig& config() const = 0;

  // The cache geometry this model requires (layers, Hkv, d, dtype). The caller
  // constructs a matching `KvCache` (§6.1); `forward` rejects a cache whose
  // geometry differs.
  [[nodiscard]] virtual kvcache::CacheGeometry cache_geometry() const = 0;

 protected:
  Model() = default;
  Model(const Model&) = default;
  Model& operator=(const Model&) = default;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
};

}  // namespace engine::model
