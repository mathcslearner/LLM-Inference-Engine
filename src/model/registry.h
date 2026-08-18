#pragma once

#include "core/status.h"
#include "model/loader.h"
#include "model/model.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// The architecture registry & builder (M5-T08; design:
// docs/design/model-execution.md §9). Maps the raw HF architecture string
// (`ModelConfig::architecture_name`, e.g. "LlamaForCausalLM",
// "Qwen2ForCausalLM") to a builder that turns a loaded checkpoint into an
// executable `Model`. `BuildModel` is the single entry point the generation
// loop (M5-T09) and the server (M17) construct models through; unknown
// architectures yield a clean `Unimplemented` listing the supported ones.
//
// Adding an architecture is *one registration call* — there is no switch to
// edit (the M5-T08 acceptance criterion, verified by a test-local dummy arch).
// The built-in Llama/Qwen2 families are registered lazily on first use of the
// registry (see registry.cpp): `engine_model` is a static library, so a TU
// whose only content is a global self-registering object would be dropped by
// the linker — the built-ins therefore populate inside the registry accessor,
// not via static initializers.

namespace engine::model {

// Which `Model` implementation the builder constructs (design §8). M5 ships
// `kReference` (the `cpu::`-op oracle); M6 adds `kOptimized` behind the same
// `Model` interface, selected by the same enum, so a single test can build both
// and assert token-for-token equality (M6-T07).
//
// This enum lives in `model` (not `engine`) because `BuildOptions` — a
// parameter of `BuildModel`, which is part of the model graph the roadmap fixes
// in `src/model/registry.h` — names it, and `model` cannot depend on `engine`
// (ADR-002: engine → model). M5-T09's `engine/backend.h` re-exports it for the
// generation-loop call sites; the definition is here. (Design §2/§8 originally
// sketched it in `engine/backend.h`; this ticket relocates it and updates the
// doc.)
enum class Backend : std::uint8_t {
  kReference,  // M5: ReferenceModel + ReferenceLinear + cpu:: ops (the oracle)
  kOptimized,  // M6: repacked weights, dispatched kernels (Unimplemented in M5)
};

// Options threaded from `BuildModel` into the resolved builder.
struct BuildOptions {
  Backend backend = Backend::kReference;
};

// Turns a loaded checkpoint into an executable `Model`. Consumes `model` (its
// weights are moved into the modules). Signature shared by every registered
// architecture; the family builder for Llama/Qwen2 forwards to
// `ReferenceModel::Create` (or, in M6, the optimized implementation, per
// `options.backend`).
using ModelBuilder = std::function<core::StatusOr<std::unique_ptr<Model>>(
    LoadedModel model, const BuildOptions& options)>;

// Registers `builder` under the HF architecture string `hf_arch_name`. Returns
// `InvalidArgument` for an empty name and `AlreadyExists` if the name is
// already registered (duplicate registration is a programming bug worth
// surfacing, not a silent overwrite). Thread-safe.
core::Status RegisterArchitecture(std::string_view hf_arch_name,
                                  ModelBuilder builder);

// The registered architecture strings, sorted. Used to compose the
// unknown-architecture error message and by tests asserting family coverage.
[[nodiscard]] std::vector<std::string> SupportedArchitectures();

// Builds an executable `Model` from `model`, dispatching on
// `model.config.architecture_name`. Consumes `model`. An unregistered
// architecture → `Unimplemented` listing the supported strings; a
// `kOptimized` backend in M5 → `Unimplemented` (M6 fills it in). Any error the
// resolved builder returns (missing weight, shape/rope mismatch) propagates
// unchanged.
[[nodiscard]] core::StatusOr<std::unique_ptr<Model>> BuildModel(
    LoadedModel model, const BuildOptions& options = {});

}  // namespace engine::model
