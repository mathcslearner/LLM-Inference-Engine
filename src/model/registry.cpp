#include "model/registry.h"

#include "core/status.h"
#include "model/loader.h"
#include "model/model.h"
#include "model/optimized_model.h"
#include "model/reference_model.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::model {

namespace {

// The family builder shared by Llama and Qwen2 (design §9): the two families
// differ only in `attention_bias` (Qwen2 biases q/k/v) and config field
// *values*, both of which flow through the same `Create` — the diff is
// config/wiring, not layer code, so one builder serves both, and it serves both
// backends. `kReference` builds the `cpu::`-op oracle (M5); `kOptimized` builds
// the repacked-weight, dispatched-kernel `OptimizedModel` (M6-T07). Both
// implement the same `Model` interface, so a single test builds both and
// asserts token-for-token equality.
core::StatusOr<std::unique_ptr<Model>> BuildFamily(
    LoadedModel loaded, const BuildOptions& options) {
  if (options.backend == Backend::kOptimized) {
    ASSIGN_OR_RETURN(std::unique_ptr<OptimizedModel> model,
                     OptimizedModel::Create(std::move(loaded)));
    return std::unique_ptr<Model>(std::move(model));
  }
  ASSIGN_OR_RETURN(std::unique_ptr<ReferenceModel> model,
                   ReferenceModel::Create(std::move(loaded)));
  return std::unique_ptr<Model>(std::move(model));
}

// The process-wide registry. A `std::map` (sorted) so `SupportedArchitectures`
// and the error message list names deterministically. Guarded by `mutex`; the
// function-local statics are initialized on first use (thread-safe under C++11
// magic statics) and pre-populated with the built-in families — this is why
// the built-ins survive static-library linking (registry.h's note): they are
// added here, not by a global self-registering object a static lib would drop.
struct Registry {
  std::mutex mutex;
  std::map<std::string, ModelBuilder, std::less<>> builders;
};

Registry& GetRegistry() {
  static Registry* const registry = [] {
    auto* r = new Registry();
    r->builders.emplace("LlamaForCausalLM", BuildFamily);
    r->builders.emplace("Qwen2ForCausalLM", BuildFamily);
    return r;
  }();
  return *registry;
}

}  // namespace

core::Status RegisterArchitecture(std::string_view hf_arch_name,
                                  ModelBuilder builder) {
  if (hf_arch_name.empty()) {
    return core::InvalidArgumentError(
        "RegisterArchitecture: architecture name must be non-empty");
  }
  Registry& registry = GetRegistry();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  const auto [it, inserted] =
      registry.builders.try_emplace(std::string(hf_arch_name));
  if (!inserted) {
    return core::AlreadyExistsError(
        "RegisterArchitecture: architecture '{}' is already registered",
        hf_arch_name);
  }
  it->second = std::move(builder);  // move-assign only after the insert wins
  return core::OkStatus();
}

std::vector<std::string> SupportedArchitectures() {
  Registry& registry = GetRegistry();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  std::vector<std::string> names;
  names.reserve(registry.builders.size());
  for (const auto& [name, builder] : registry.builders) {
    names.push_back(name);
  }
  return names;  // already sorted (std::map key order)
}

core::StatusOr<std::unique_ptr<Model>> BuildModel(LoadedModel model,
                                                  const BuildOptions& options) {
  const std::string arch = model.config.architecture_name;

  ModelBuilder builder;
  {
    Registry& registry = GetRegistry();
    const std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.builders.find(arch);
    if (it != registry.builders.end()) {
      builder = it->second;  // copy out, then release the lock before building
    }
  }

  if (!builder) {
    std::string supported;
    for (const std::string& name : SupportedArchitectures()) {
      if (!supported.empty()) {
        supported += ", ";
      }
      supported += name;
    }
    return core::UnimplementedError(
        "BuildModel: unsupported architecture '{}'; supported: {}", arch,
        supported);
  }

  return builder(std::move(model), options);
}

}  // namespace engine::model
