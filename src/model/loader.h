#pragma once

#include "core/status.h"
#include "model/config.h"
#include "model/weight_map.h"
#include "tensor/tensor.h"

#include <filesystem>
#include <string>
#include <unordered_map>

// Model loader (M4-T07; design: docs/design/model-loading.md §3.7).
//
// `load_model` is the composition point of the M4 weight path: config parse
// (§3.2) → checkpoint open (§3.6) → weight-name mapping and shape validation
// (§4) → materializing every canonical weight as a zero-copy CPU tensor.
// Stages run in that order, so a bad directory or unsupported architecture
// fails before any weight I/O, and each stage's error names the offending
// file/field/tensor.
//
// Dtype policy (design §5): checkpoint dtypes are preserved — a bf16
// checkpoint loads as kBFloat16 views of the mapped file, never up-converted.
// Only F32/F16/BF16 weights are accepted; integer/bool/F8 tensors parse fine
// (SafetensorsFile is format-complete) but are rejected here as *weights*
// with `Unimplemented` until the quantization milestones (M12–M13).

namespace engine::model {

struct LoadedModel {
  ModelConfig config;
  // Canonical names (weight_map.h) → zero-copy CPU tensors backed by the
  // mapped checkpoint files, dtypes preserved. With tied embeddings,
  // "lm_head.weight" and "embed_tokens.weight" are the same Tensor handle
  // (shared storage).
  std::unordered_map<std::string, tensor::Tensor> weights;
  WeightMapReport report;  // missing is always empty here (a load error)
};

// Loads the model in directory `dir` (HF snapshot layout, design §3.1).
// Errors: `NotFound` for a missing directory/config.json or absent weights
// (with an actionable conversion hint when only PyTorch `.bin` pickles are
// present); `Unimplemented` for unsupported architectures and non-float
// weight dtypes; `InvalidArgument` for malformed config/checkpoint contents,
// shape mismatches, and missing required weights (all names listed).
[[nodiscard]] core::StatusOr<LoadedModel> load_model(
    const std::filesystem::path& dir);

}  // namespace engine::model
