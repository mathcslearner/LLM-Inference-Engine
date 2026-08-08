#include "model/checkpoint.h"

#include "core/logging.h"
#include "core/status.h"
#include "memory/allocator.h"
#include "model/mapped_file.h"
#include "model/safetensors.h"
#include "tensor/tensor.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace engine::model {
namespace {

using nlohmann::json;

// The two weight spellings of the HF snapshot layout (design §3.1).
constexpr std::string_view kSingleFile = "model.safetensors";
constexpr std::string_view kIndexFile = "model.safetensors.index.json";

// `"a", "b", "c"` — for listing every offender in one message (one
// round-trip fixes a broken checkpoint, §3.7).
[[nodiscard]] std::string JoinQuoted(const std::vector<std::string>& items) {
  std::string out;
  for (const std::string& item : items) {
    if (!out.empty()) {
      out += ", ";
    }
    out += fmt::format(R"("{}")", item);
  }
  return out;
}

// `4.9 GiB`-style size for the per-file map lines (design §3.7). Weight
// files are the one place human-scale sizes matter enough to format.
[[nodiscard]] std::string HumanSize(std::uintmax_t bytes) {
  constexpr std::string_view kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  auto value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(kUnits)) {
    value /= 1024.0;
    ++unit;
  }
  return unit == 0 ? fmt::format("{} B", bytes)
                   : fmt::format("{:.1f} {}", value, kUnits[unit]);
}

// The progress line for a freshly mapped weight file (§3.7): real models
// are tens of GB across many shards, and these lines are what makes a long
// load observable.
void LogMapped(const std::filesystem::path& path) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    LOG_INFO("model", "mapped {}", path.filename().string());
  } else {
    LOG_INFO("model", "mapped {} ({})", path.filename().string(),
             HumanSize(size));
  }
}

// A bare filename: resolving `dir / name` must stay inside `dir`, so a
// hostile index cannot address files outside the model directory.
[[nodiscard]] bool IsBareFilename(std::string_view name) {
  return !name.empty() && name != "." && name != ".." &&
         name.find('/') == std::string_view::npos &&
         name.find('\\') == std::string_view::npos;
}

struct ParsedIndex {
  std::map<std::string, std::size_t, std::less<>> index;
  std::vector<std::string> shard_names;
};

// Parses the index JSON. Only "weight_map" is interpreted; "metadata"
// (e.g. total_size) is informational, and unknown top-level keys are
// tolerated — the index is HF-tooling output, not a framing format, so
// additive fields must not break loading (contrast the strict per-entry
// validation in safetensors.cpp, where every byte is load-bearing).
core::StatusOr<ParsedIndex> ParseIndex(const memory::Buffer& buffer) {
  const char* begin = static_cast<const char*>(buffer.data());
  const json root =
      buffer.size_bytes() == 0
          ? json(json::value_t::discarded)
          : json::parse(begin, begin + buffer.size_bytes(),
                        /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded()) {
    return core::InvalidArgumentError("index is not valid JSON");
  }
  if (!root.is_object()) {
    return core::InvalidArgumentError("index must be a JSON object, got {}",
                                      root.type_name());
  }
  if (!root.contains("weight_map")) {
    return core::InvalidArgumentError(R"(index has no "weight_map" key)");
  }
  const json& weight_map = root["weight_map"];
  if (!weight_map.is_object()) {
    return core::InvalidArgumentError(
        R"("weight_map" must be a JSON object, got {})",
        weight_map.type_name());
  }
  ParsedIndex parsed;
  std::map<std::string, std::size_t, std::less<>> ordinals;
  for (const auto& [name, value] : weight_map.items()) {
    if (!value.is_string()) {
      return core::InvalidArgumentError(
          R"(weight_map entry "{}" must name a shard file (string), got {})",
          name, value.type_name());
    }
    const std::string shard = value.get<std::string>();
    if (!IsBareFilename(shard)) {
      return core::InvalidArgumentError(
          R"(weight_map entry "{}" names shard "{}", which is not a bare )"
          "filename",
          name, shard);
    }
    const auto [it, inserted] =
        ordinals.try_emplace(shard, parsed.shard_names.size());
    if (inserted) {
      parsed.shard_names.push_back(shard);
    }
    parsed.index.emplace(name, it->second);
  }
  return parsed;
}

// First-map consistency check (design §3.6): the shard's tensor set must
// equal the set the index maps to it. Dtype/shape live only in the shard
// header — the index has no copy to cross-check. A shard tensor the index
// maps to a *different* shard is the cross-shard duplicate case.
core::Status CheckShardConsistency(
    const std::map<std::string, std::size_t, std::less<>>& index,
    std::size_t ordinal, const SafetensorsFile& shard,
    const std::vector<std::string>& shard_names, std::string_view index_path) {
  std::vector<std::string> missing;
  for (const auto& [name, mapped_ordinal] : index) {
    if (mapped_ordinal == ordinal && !shard.contains(name)) {
      missing.push_back(name);
    }
  }
  std::vector<std::string> unexpected;
  for (const std::string_view name : shard.names()) {
    const auto it = index.find(name);
    if (it == index.end()) {
      unexpected.push_back(fmt::format("{} (not in the index)", name));
    } else if (it->second != ordinal) {
      unexpected.push_back(fmt::format("{} (the index maps it to {})", name,
                                       shard_names[it->second]));
    }
  }
  if (missing.empty() && unexpected.empty()) {
    return core::OkStatus();
  }
  std::string detail;
  if (!missing.empty()) {
    detail += fmt::format("tensors the index maps here are absent: {}",
                          JoinQuoted(missing));
  }
  if (!unexpected.empty()) {
    if (!detail.empty()) {
      detail += "; ";
    }
    detail += fmt::format("tensors the index does not map here are present: {}",
                          JoinQuoted(unexpected));
  }
  return core::InvalidArgumentError(R"(shard "{}" is inconsistent with {}: {})",
                                    shard_names[ordinal], index_path, detail);
}

}  // namespace

core::StatusOr<Checkpoint> Checkpoint::Open(const std::filesystem::path& dir) {
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) {
    return core::NotFoundError(
        "model directory {} does not exist (or is not "
        "a directory)",
        dir.string());
  }
  Checkpoint checkpoint;
  checkpoint.dir_ = dir;

  const std::filesystem::path index_path = dir / kIndexFile;
  if (std::filesystem::exists(index_path, ec)) {
    // Sharded. The index wins even when model.safetensors is also present
    // (HF loading order, design §3.1).
    std::shared_ptr<memory::Buffer> buffer;
    ASSIGN_OR_RETURN(buffer, MapFileReadOnly(index_path));
    auto parsed = ParseIndex(*buffer);
    if (!parsed.ok()) {
      return core::Status(parsed.status().code(),
                          fmt::format("{}: {}", index_path.string(),
                                      parsed.status().message()));
    }
    checkpoint.weights_path_ = index_path.string();
    checkpoint.shard_names_ = std::move(parsed->shard_names);
    checkpoint.index_ = std::move(parsed->index);
    checkpoint.shards_.resize(checkpoint.shard_names_.size());
    // Eager existence check so "missing shard" surfaces here, not mid-load
    // (§3.6). Mapping and header validation stay lazy.
    std::vector<std::string> absent;
    for (const std::string& shard_name : checkpoint.shard_names_) {
      if (!std::filesystem::exists(dir / shard_name, ec)) {
        absent.push_back(shard_name);
      }
    }
    if (!absent.empty()) {
      return core::NotFoundError(
          "{}: shard file(s) named by the index do "
          "not exist: {}",
          index_path.string(), JoinQuoted(absent));
    }
    return checkpoint;
  }

  const std::filesystem::path single_path = dir / kSingleFile;
  if (std::filesystem::exists(single_path, ec)) {
    ASSIGN_OR_RETURN(SafetensorsFile file, SafetensorsFile::Open(single_path));
    LogMapped(single_path);
    checkpoint.weights_path_ = single_path.string();
    checkpoint.shard_names_.emplace_back(kSingleFile);
    for (const std::string_view name : file.names()) {
      checkpoint.index_.emplace(std::string(name), 0);
    }
    // After the names are copied: names() views die with `file`.
    checkpoint.shards_.emplace_back(std::move(file));
    return checkpoint;
  }

  return core::NotFoundError("{}: found neither {} nor {}", dir.string(),
                             kSingleFile, kIndexFile);
}

std::vector<std::string_view> Checkpoint::names() const {
  std::vector<std::string_view> result;
  result.reserve(index_.size());
  for (const auto& [name, ordinal] : index_) {
    result.emplace_back(name);
  }
  return result;
}

bool Checkpoint::contains(std::string_view name) const {
  return index_.find(name) != index_.end();
}

core::StatusOr<const SafetensorsFile*> Checkpoint::shard(std::size_t ordinal) {
  std::optional<SafetensorsFile>& slot = shards_[ordinal];
  if (!slot.has_value()) {
    // SafetensorsFile::Open prefixes the shard path on its own errors.
    ASSIGN_OR_RETURN(SafetensorsFile file,
                     SafetensorsFile::Open(dir_ / shard_names_[ordinal]));
    RETURN_IF_ERROR(CheckShardConsistency(index_, ordinal, file, shard_names_,
                                          weights_path_));
    LogMapped(dir_ / shard_names_[ordinal]);
    slot.emplace(std::move(file));
  }
  return &*slot;
}

core::StatusOr<tensor::Tensor> Checkpoint::tensor(std::string_view name) {
  const auto it = index_.find(name);
  if (it == index_.end()) {
    return core::NotFoundError(R"({}: no tensor named "{}")", weights_path_,
                               name);
  }
  const SafetensorsFile* file = nullptr;
  ASSIGN_OR_RETURN(file, shard(it->second));
  return file->tensor(name);
}

}  // namespace engine::model
