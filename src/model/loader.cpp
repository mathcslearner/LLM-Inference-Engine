#include "model/loader.h"

#include "core/logging.h"
#include "core/status.h"
#include "model/checkpoint.h"
#include "model/config.h"
#include "model/weight_map.h"
#include "tensor/dtype.h"
#include "tensor/tensor.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::model {
namespace {

// The dtype-acceptance policy for *weights* (design §5): the checkpoint
// dtype is preserved, and only float dtypes are model weights until the
// quantization milestones. SafetensorsFile stays format-complete — integer
// activation/token-id fixtures still parse; the line is drawn here.
[[nodiscard]] bool IsSupportedWeightDtype(tensor::DataType dtype) {
  switch (dtype) {
    case tensor::DataType::kFloat32:
    case tensor::DataType::kFloat16:
    case tensor::DataType::kBFloat16:
      return true;
    default:
      return false;
  }
}

// PyTorch pickle checkpoint files, for the §3.1 actionable error: a
// directory with `.bin`/`.pt` weights and no safetensors should say
// "convert", not just "not found".
[[nodiscard]] std::vector<std::string> PickleWeightFiles(
    const std::filesystem::path& dir) {
  std::vector<std::string> found;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    const std::filesystem::path& path = entry.path();
    const std::string extension = path.extension().string();
    if (extension == ".bin" || extension == ".pt" || extension == ".pth") {
      found.push_back(path.filename().string());
    }
  }
  std::ranges::sort(found);
  return found;
}

}  // namespace

core::StatusOr<LoadedModel> load_model(const std::filesystem::path& dir) {
  LOG_INFO("model", "loading model from {}", dir.string());

  // Stage 1: config (§3.2). First deliberately — a bad path or unsupported
  // architecture fails before any weight I/O.
  ModelConfig config;
  ASSIGN_OR_RETURN(config, LoadModelConfig(dir / "config.json"));
  LOG_INFO("model", "config: {} — {} layer(s), hidden {}, vocab {}",
           config.architecture_name, config.num_layers, config.hidden_size,
           config.vocab_size);

  // Stage 2: checkpoint discovery (§3.1/§3.6). On the "neither spelling"
  // NotFound, check for PyTorch pickle weights so the error says how to
  // proceed; pickle checkpoints are permanently out of scope (unsafe
  // format — safetensors exists precisely to replace it).
  auto checkpoint = Checkpoint::Open(dir);
  if (!checkpoint.ok()) {
    if (checkpoint.status().code() == core::StatusCode::kNotFound) {
      const std::vector<std::string> pickles = PickleWeightFiles(dir);
      if (!pickles.empty()) {
        return core::NotFoundError(
            "{}: no safetensors weights, but found PyTorch pickle "
            "checkpoint file(s): {} — the pickle format is unsupported "
            "(unsafe by design); convert the model to safetensors (e.g. "
            "with the `safetensors` Python package or a safetensors export "
            "of the model) and retry",
            dir.string(), fmt::join(pickles, ", "));
      }
    }
    return checkpoint.status();
  }
  LOG_INFO("model", "checkpoint: {} tensor(s)", checkpoint->names().size());

  // Stage 3: name mapping + shape validation (§4); every missing required
  // weight is listed in one error.
  WeightMap map;
  ASSIGN_OR_RETURN(map, BuildWeightMap(config, *checkpoint));
  RETURN_IF_ERROR(CheckNoMissingWeights(map.report));

  // Stage 4: materialize. Each unique checkpoint tensor is resolved once and
  // the handle reused, so a tied lm_head/embed_tokens pair shares one Tensor
  // (design §4's alias rule), and the dtype policy names the canonical
  // weight — more actionable than the raw checkpoint spelling.
  LoadedModel model;
  std::map<std::string, tensor::Tensor, std::less<>> by_source;
  for (const auto& [canonical, source] : map.canonical_to_source) {
    auto it = by_source.find(source);
    if (it == by_source.end()) {
      tensor::Tensor resolved;
      ASSIGN_OR_RETURN(resolved, checkpoint->tensor(source));
      it = by_source.emplace(source, std::move(resolved)).first;
    }
    const tensor::Tensor& weight = it->second;
    if (!IsSupportedWeightDtype(weight.dtype())) {
      return core::UnimplementedError(
          "weight \"{}\" (checkpoint tensor \"{}\") has dtype {}; only "
          "F32/F16/BF16 weights are supported — quantized/integer weights "
          "arrive in M12–M13",
          canonical, source, tensor::to_string(weight.dtype()));
    }
    model.weights.emplace(canonical, weight);
  }
  LOG_INFO("model", "load complete: {} weight(s)", model.weights.size());

  model.config = std::move(config);
  model.report = std::move(map.report);
  return model;
}

std::int64_t weight_resident_bytes(const LoadedModel& loaded) {
  std::int64_t total = 0;
  // Dedup by storage pointer: tied embeddings hand back the same Tensor for
  // "lm_head.weight" and "embed_tokens.weight", so the shared table is counted
  // once. Distinct weights have distinct data pointers even when several are
  // views into one mapped shard (each names a disjoint window), so summing
  // per-tensor `numel × itemsize` yields the total weight-data footprint.
  std::vector<const void*> seen;
  seen.reserve(loaded.weights.size());
  for (const auto& [name, weight] : loaded.weights) {
    const void* ptr = weight.defined() ? weight.data() : nullptr;
    if (ptr != nullptr && std::ranges::find(seen, ptr) != seen.end()) {
      continue;
    }
    if (ptr != nullptr) {
      seen.push_back(ptr);
    }
    total += weight.numel() * tensor::itemsize(weight.dtype());
  }
  return total;
}

}  // namespace engine::model
