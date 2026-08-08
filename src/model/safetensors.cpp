#include "model/safetensors.h"

#include "core/status.h"
#include "memory/allocator.h"
#include "model/mapped_file.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace engine::model {
namespace {

using nlohmann::json;

// Safetensors data is little-endian; both supported platforms are too, so
// the header length and tensor bytes are read without swapping. A
// hypothetical big-endian port must fail to compile, not corrupt silently.
static_assert(std::endian::native == std::endian::little,
              "safetensors parsing assumes a little-endian host");

// Sanity cap on the header length (design §3.4 check 1): rejects hostile or
// garbage values before JSON parsing is attempted. The reference
// implementation uses 100 MB.
constexpr std::uint64_t kMaxHeaderLen = 256ULL * 1024 * 1024;

// Safetensors container dtype strings → engine DataType. This table is
// distinct from dtype.h's from_string, which maps HF config.json spellings
// ("bfloat16"); each lives next to its format (design §5).
core::StatusOr<tensor::DataType> ParseSafetensorsDtype(std::string_view name) {
  struct Mapping {
    std::string_view name;
    tensor::DataType dtype;
  };
  constexpr Mapping kSupported[] = {
      {.name = "F32", .dtype = tensor::DataType::kFloat32},
      {.name = "F16", .dtype = tensor::DataType::kFloat16},
      {.name = "BF16", .dtype = tensor::DataType::kBFloat16},
      {.name = "I8", .dtype = tensor::DataType::kInt8},
      {.name = "U8", .dtype = tensor::DataType::kUInt8},
      {.name = "I32", .dtype = tensor::DataType::kInt32},
      {.name = "I64", .dtype = tensor::DataType::kInt64},
      {.name = "BOOL", .dtype = tensor::DataType::kBool},
      {.name = "F8_E4M3", .dtype = tensor::DataType::kFP8E4M3},
  };
  for (const Mapping& mapping : kSupported) {
    if (name == mapping.name) {
      return mapping.dtype;
    }
  }
  // Real safetensors dtypes with no engine DataType; none appear in target
  // checkpoints (design §5).
  constexpr std::string_view kKnownUnsupported[] = {"F64", "I16", "U16",
                                                    "U32", "U64", "F8_E5M2"};
  if (std::ranges::find(kKnownUnsupported, name) !=
      std::end(kKnownUnsupported)) {
    return core::UnimplementedError(
        R"(safetensors dtype "{}" is not supported by the engine)", name);
  }
  return core::InvalidArgumentError(R"(unrecognized safetensors dtype "{}")",
                                    name);
}

core::StatusOr<tensor::Shape> ParseShapeField(const json& value,
                                              std::string_view name) {
  if (!value.is_array()) {
    return core::InvalidArgumentError(
        R"(tensor "{}": "shape" must be an array, got {})", name,
        value.type_name());
  }
  std::vector<std::int64_t> dims;
  dims.reserve(value.size());
  for (const json& dim : value) {
    if (!dim.is_number_integer()) {
      return core::InvalidArgumentError(
          R"(tensor "{}": "shape" dims must be integers, got {})", name,
          dim.type_name());
    }
    if (dim.is_number_unsigned() &&
        dim.get<std::uint64_t>() >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) {
      return core::InvalidArgumentError(
          R"(tensor "{}": "shape" dim {} is out of range)", name,
          dim.get<std::uint64_t>());
    }
    dims.push_back(dim.get<std::int64_t>());
  }
  // FromDims rejects negative dims, rank > kMaxRank, and numel overflow.
  auto shape = tensor::Shape::FromDims(dims);
  if (!shape.ok()) {
    return core::Status(shape.status().code(),
                        fmt::format(R"(tensor "{}": invalid shape: {})", name,
                                    shape.status().message()));
  }
  return shape;
}

core::StatusOr<std::pair<std::uint64_t, std::uint64_t>> ParseDataOffsets(
    const json& value, std::string_view name, std::uint64_t data_size) {
  if (!value.is_array() || value.size() != 2) {
    return core::InvalidArgumentError(
        R"(tensor "{}": "data_offsets" must be a 2-element array)", name);
  }
  std::uint64_t offsets[2] = {0, 0};
  for (std::size_t i = 0; i < 2; ++i) {
    const json& element = value[i];
    if (!element.is_number_integer()) {
      return core::InvalidArgumentError(
          R"(tensor "{}": "data_offsets" entries must be integers, got {})",
          name, element.type_name());
    }
    // nlohmann stores non-negative integers as unsigned; a signed value
    // here is therefore negative.
    if (!element.is_number_unsigned()) {
      return core::InvalidArgumentError(
          R"(tensor "{}": "data_offsets" entry {} is negative)", name,
          element.get<std::int64_t>());
    }
    offsets[i] = element.get<std::uint64_t>();
  }
  if (offsets[0] > offsets[1]) {
    return core::InvalidArgumentError(
        R"(tensor "{}": data_offsets begin {} exceeds end {})", name,
        offsets[0], offsets[1]);
  }
  if (offsets[1] > data_size) {
    return core::InvalidArgumentError(
        R"(tensor "{}": data_offsets end {} exceeds the data section size {})",
        name, offsets[1], data_size);
  }
  return std::make_pair(offsets[0], offsets[1]);
}

core::StatusOr<std::map<std::string, std::string>> ParseMetadataBlock(
    const json& value) {
  if (!value.is_object()) {
    return core::InvalidArgumentError(
        R"("__metadata__" must be a JSON object, got {})", value.type_name());
  }
  std::map<std::string, std::string> metadata;
  for (const auto& [key, entry] : value.items()) {
    if (!entry.is_string()) {
      return core::InvalidArgumentError(
          R"("__metadata__" value for key "{}" must be a string, got {})", key,
          entry.type_name());
    }
    metadata.emplace(key, entry.get<std::string>());
  }
  return metadata;
}

struct RawEntry {
  tensor::DataType dtype = tensor::DataType::kFloat32;
  tensor::Shape shape;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

core::StatusOr<RawEntry> ParseEntry(std::string_view name, const json& value,
                                    std::uint64_t data_size,
                                    std::uint64_t data_start) {
  if (!value.is_object()) {
    return core::InvalidArgumentError(
        R"(tensor "{}": entry must be a JSON object, got {})", name,
        value.type_name());
  }
  for (const auto& element : value.items()) {
    const std::string& key = element.key();
    if (key != "dtype" && key != "shape" && key != "data_offsets") {
      return core::InvalidArgumentError(
          R"(tensor "{}": unexpected key "{}" in entry)", name, key);
    }
  }
  for (const std::string_view required : {"dtype", "shape", "data_offsets"}) {
    if (!value.contains(required)) {
      return core::InvalidArgumentError(
          R"(tensor "{}": entry is missing required key "{}")", name, required);
    }
  }
  const json& dtype_value = value["dtype"];
  if (!dtype_value.is_string()) {
    return core::InvalidArgumentError(
        R"(tensor "{}": "dtype" must be a string, got {})", name,
        dtype_value.type_name());
  }
  RawEntry entry;
  {
    auto dtype = ParseSafetensorsDtype(dtype_value.get<std::string>());
    if (!dtype.ok()) {
      return core::Status(
          dtype.status().code(),
          fmt::format(R"(tensor "{}": {})", name, dtype.status().message()));
    }
    entry.dtype = *dtype;
  }
  ASSIGN_OR_RETURN(entry.shape, ParseShapeField(value["shape"], name));
  std::pair<std::uint64_t, std::uint64_t> offsets;
  ASSIGN_OR_RETURN(offsets,
                   ParseDataOffsets(value["data_offsets"], name, data_size));
  entry.begin = offsets.first;
  entry.end = offsets.second;

  // Size identity (design §3.4 check 4): the byte range must equal
  // numel × itemsize exactly. kInt4 never occurs in plain safetensors, so
  // itemsize() is total here.
  const auto item_bytes =
      static_cast<std::uint64_t>(tensor::itemsize(entry.dtype));
  std::uint64_t expected_bytes = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(entry.shape.numel()),
                             item_bytes, &expected_bytes)) {
    return core::InvalidArgumentError(
        R"(tensor "{}": byte size overflows uint64_t ({} elements of {}))",
        name, entry.shape.numel(), to_string(entry.dtype));
  }
  if (entry.end - entry.begin != expected_bytes) {
    return core::InvalidArgumentError(
        R"(tensor "{}": data_offsets span {} bytes but {} elements of {} )"
        "need {}",
        name, entry.end - entry.begin, entry.shape.numel(),
        to_string(entry.dtype), expected_bytes);
  }
  // Stricter than the upstream spec (design §3.4): the tensor's absolute
  // file offset must be itemsize-aligned, so typed access through views is
  // never misaligned. Checking `begin` alone is not enough — an unpadded
  // header shifts the whole data section, and `begin % itemsize == 0` says
  // nothing about `(8 + header_len + begin) % itemsize`. Real HF
  // serializers pad the header so the data section starts 8-aligned.
  if ((data_start + entry.begin) % item_bytes != 0) {
    return core::InvalidArgumentError(
        R"(tensor "{}": absolute offset {} (data section at {} + begin {}) )"
        "is not aligned to the {}-byte itemsize of {}",
        name, data_start + entry.begin, data_start, entry.begin, item_bytes,
        to_string(entry.dtype));
  }
  return entry;
}

struct ParsedFile {
  std::size_t data_start = 0;
  std::map<std::string, RawEntry, std::less<>> entries;
  std::map<std::string, std::string> metadata;
};

// Cross-tensor layout (design §3.4 check 5): sorted by begin, ranges must
// be non-overlapping and gap-free, covering the data section exactly.
// Overlap is data corruption; a gap is bytes no tensor accounts for. The
// cursor walk handles zero-length ranges naturally: they must sit exactly
// on the current boundary, so one strictly inside another tensor's range
// is rejected as overlap.
core::Status ValidateDataSectionLayout(
    const std::map<std::string, RawEntry, std::less<>>& entries,
    std::uint64_t data_size) {
  struct Range {
    std::uint64_t begin;
    std::uint64_t end;
    const std::string* name;
  };
  std::vector<Range> ranges;
  ranges.reserve(entries.size());
  for (const auto& [name, entry] : entries) {
    ranges.push_back({.begin = entry.begin, .end = entry.end, .name = &name});
  }
  std::ranges::sort(ranges, [](const Range& a, const Range& b) {
    return std::tie(a.begin, a.end) < std::tie(b.begin, b.end);
  });
  std::uint64_t cursor = 0;
  const std::string* previous = nullptr;
  for (const Range& range : ranges) {
    if (range.begin < cursor) {
      return core::InvalidArgumentError(
          R"(tensors "{}" and "{}" have overlapping data ranges)", *previous,
          *range.name);
    }
    if (range.begin > cursor) {
      return core::InvalidArgumentError(
          R"(gap of {} bytes in the data section before tensor "{}")",
          range.begin - cursor, *range.name);
    }
    cursor = range.end;
    previous = range.name;
  }
  if (cursor != data_size) {
    return core::InvalidArgumentError(
        "data section is {} bytes but tensors account for only {}", data_size,
        cursor);
  }
  return core::OkStatus();
}

core::StatusOr<ParsedFile> ParseSafetensors(const memory::Buffer& buffer) {
  // Framing (design §3.4 check 1). All arithmetic in uint64: header_len is
  // attacker-controlled and must never be trusted into a size_t cast first.
  const std::uint64_t file_size = buffer.size_bytes();
  if (file_size < 8) {
    return core::InvalidArgumentError(
        "file is {} bytes; a safetensors file needs at least an 8-byte "
        "header length",
        file_size);
  }
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, buffer.data(), sizeof(header_len));
  if (header_len > kMaxHeaderLen) {
    return core::InvalidArgumentError(
        "header length {} exceeds the {} MiB sanity cap", header_len,
        kMaxHeaderLen / (1024ULL * 1024ULL));
  }
  if (header_len > file_size - 8) {
    return core::InvalidArgumentError(
        "header length {} exceeds the {} bytes after the length field",
        header_len, file_size - 8);
  }
  const std::uint64_t data_size = file_size - 8 - header_len;

  const char* header_begin = static_cast<const char*>(buffer.data()) + 8;
  const json root = json::parse(header_begin, header_begin + header_len,
                                /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded()) {
    return core::InvalidArgumentError("header is not valid JSON");
  }
  if (!root.is_object()) {
    return core::InvalidArgumentError("header must be a JSON object, got {}",
                                      root.type_name());
  }

  ParsedFile parsed;
  parsed.data_start = static_cast<std::size_t>(8 + header_len);
  for (const auto& [name, value] : root.items()) {
    if (name == "__metadata__") {
      ASSIGN_OR_RETURN(parsed.metadata, ParseMetadataBlock(value));
      continue;
    }
    RawEntry entry;
    ASSIGN_OR_RETURN(entry,
                     ParseEntry(name, value, data_size,
                                static_cast<std::uint64_t>(parsed.data_start)));
    parsed.entries.emplace(name, entry);
  }
  RETURN_IF_ERROR(ValidateDataSectionLayout(parsed.entries, data_size));
  return parsed;
}

}  // namespace

core::StatusOr<SafetensorsFile> SafetensorsFile::Open(
    const std::filesystem::path& path) {
  std::shared_ptr<memory::Buffer> buffer;
  ASSIGN_OR_RETURN(buffer, MapFileReadOnly(path));
  auto parsed = ParseSafetensors(*buffer);
  if (!parsed.ok()) {
    return core::Status(
        parsed.status().code(),
        fmt::format("{}: {}", path.string(), parsed.status().message()));
  }
  SafetensorsFile file;
  file.buffer_ = std::move(buffer);
  file.data_start_ = parsed->data_start;
  file.metadata_ = std::move(parsed->metadata);
  for (auto& [name, raw] : parsed->entries) {
    file.entries_.emplace(
        name,
        Entry{.dtype = raw.dtype, .shape = raw.shape, .begin = raw.begin});
  }
  file.path_ = path.string();
  return file;
}

std::vector<std::string_view> SafetensorsFile::names() const {
  std::vector<std::string_view> result;
  result.reserve(entries_.size());
  for (const auto& [name, entry] : entries_) {
    result.emplace_back(name);
  }
  return result;
}

bool SafetensorsFile::contains(std::string_view name) const {
  return entries_.find(name) != entries_.end();
}

core::StatusOr<tensor::Tensor> SafetensorsFile::tensor(
    std::string_view name) const {
  const auto it = entries_.find(name);
  if (it == entries_.end()) {
    return core::NotFoundError("{}: no tensor named \"{}\"", path_, name);
  }
  const Entry& entry = it->second;
  auto view = tensor::Tensor::from_buffer(
      buffer_, data_start_ + static_cast<std::size_t>(entry.begin), entry.shape,
      entry.dtype);
  if (!view.ok()) {
    // Reachable only for reserved dtypes (F8_E4M3): bounds were validated
    // at Open. Prefix file/tensor context either way.
    return core::Status(view.status().code(),
                        fmt::format("{}: tensor \"{}\": {}", path_, name,
                                    view.status().message()));
  }
  return view;
}

}  // namespace engine::model
