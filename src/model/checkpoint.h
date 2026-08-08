#pragma once

#include "core/status.h"
#include "model/safetensors.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Unified checkpoint interface over single-file and sharded safetensors
// (M4-T05; design: docs/design/model-loading.md §3.1, §3.6).
//
// `Open` takes a model *directory* in the HF snapshot layout and resolves
// the weights per §3.1: `model.safetensors.index.json` (sharded) wins when
// both spellings are present, matching HF loading order; neither present is
// `NotFound` naming both. Behind the interface the two layouts are
// indistinguishable — the rest of the loader (weight mapping M4-T06,
// `load_model` M4-T07) consumes only `Checkpoint`.
//
// The index (`{"metadata": {...}, "weight_map": {name: shard}}`) is parsed
// eagerly and its shard files are checked to *exist* eagerly, so a missing
// shard surfaces at `Open`, not mid-load. Shards are mapped lazily on the
// first `tensor()` touching them (hence `tensor()` is non-const), then
// cached. Shard filenames must be bare filenames — a hostile index must not
// walk out of the model directory. At first map, each shard is checked for
// consistency with the index: the shard's tensor set must equal the set the
// index maps to it (which also rejects a name duplicated across shards).
// All input is untrusted: every defect is a recoverable error naming the
// offending file (and tensors), never a CHECK, never an exception (ADR-003).

namespace engine::model {

class Checkpoint {
 public:
  // Resolves and validates the weights in model directory `dir`. Errors:
  // `NotFound` for a missing directory, missing weights (both spellings
  // listed), or a shard file the index names but the directory lacks;
  // `InvalidArgument` for a malformed index or (single-file case) any
  // `SafetensorsFile::Open` defect.
  [[nodiscard]] static core::StatusOr<Checkpoint> Open(
      const std::filesystem::path& dir);

  // Tensor names, sorted lexicographically. Views into this object — they
  // are invalidated when it is destroyed or moved.
  [[nodiscard]] std::vector<std::string_view> names() const;
  [[nodiscard]] bool contains(std::string_view name) const;

  // Zero-copy view, sharing the shard's mapping (the shard stays mapped
  // while any returned Tensor is alive, independent of this object).
  // Non-const: the first call touching a shard maps and validates it, and a
  // defective shard (unparseable, or inconsistent with the index) surfaces
  // here rather than at `Open`. `NotFound` for unknown names.
  [[nodiscard]] core::StatusOr<tensor::Tensor> tensor(std::string_view name);

 private:
  Checkpoint() = default;

  // The shard, mapping + consistency-checking it on first touch.
  [[nodiscard]] core::StatusOr<const SafetensorsFile*> shard(
      std::size_t ordinal);

  std::filesystem::path dir_;
  std::string weights_path_;  // resolved index or single file; error context
  std::vector<std::string> shard_names_;  // unique shard filenames
  std::vector<std::optional<SafetensorsFile>> shards_;  // lazy, by ordinal
  // name → ordinal into shard_names_/shards_. std::less<> enables
  // string_view lookup; map order gives sorted names().
  std::map<std::string, std::size_t, std::less<>> index_;
};

}  // namespace engine::model
