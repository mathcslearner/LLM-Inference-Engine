#pragma once

#include "core/check.h"
#include "core/status.h"

#include <array>
#include <cstdint>
#include <string_view>

// Element data types (M1-T02; design: docs/design/tensor.md §3).
//
// This is a tensor_base header (ADR-002 Amendment 1): header-only, includes
// nothing from memory or the rest of tensor, may include core.
//
// Sub-byte rule (kInt4): two elements per byte, the element with the lower
// index in the low nibble (little-nibble order, matching the AWQ/GPTQ
// repacking conventions targeted in M12). An int4 tensor's innermost
// dimension must be contiguous and even, and only the innermost dimension
// may be sub-byte-packed; strides stay element-denominated (§4). Until M12
// the type is name/size-mapped only — `Tensor::empty` rejects it with
// Unimplemented.

namespace engine::tensor {

// Values are stable; append, never renumber (they will reach fixtures and
// serialized test data).
enum class DataType : std::uint8_t {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kInt8 = 3,
  kUInt8 = 4,
  kInt32 = 5,
  kInt64 = 6,
  kBool = 7,
  // Reserved for later milestones; representable now (name/size-mapped), not
  // allocatable until their milestone lands (factories return Unimplemented).
  kFP8E4M3 = 8,  // FP8 KV cache (M13)
  kInt4 = 9,     // AWQ/GPTQ weight quantization (M12)
};

// Every DataType, for exhaustive iteration (dispatch tables, tests).
inline constexpr std::array<DataType, 10> kAllDataTypes = {
    DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16,
    DataType::kInt8,    DataType::kUInt8,   DataType::kInt32,
    DataType::kInt64,   DataType::kBool,    DataType::kFP8E4M3,
    DataType::kInt4};

// Classification. Every dtype is exactly one of: floating-point, integral,
// or kBool — kBool is neither floating nor integral (the NumPy/PyTorch kind
// model; it is a distinct 1-byte kind, not a 1-bit integer).
[[nodiscard]] constexpr bool is_floating_point(DataType dtype) {
  return dtype == DataType::kFloat32 || dtype == DataType::kFloat16 ||
         dtype == DataType::kBFloat16 || dtype == DataType::kFP8E4M3;
}

[[nodiscard]] constexpr bool is_integral(DataType dtype) {
  return dtype == DataType::kInt8 || dtype == DataType::kUInt8 ||
         dtype == DataType::kInt32 || dtype == DataType::kInt64 ||
         dtype == DataType::kInt4;
}

[[nodiscard]] constexpr bool is_sub_byte(DataType dtype) {
  return dtype == DataType::kInt4;
}

// Element width in bits. Valid for every DataType, including sub-byte.
[[nodiscard]] constexpr int itemsize_bits(DataType dtype) {
  switch (dtype) {
    case DataType::kInt4:
      return 4;
    case DataType::kInt8:
    case DataType::kUInt8:
    case DataType::kBool:
    case DataType::kFP8E4M3:
      return 8;
    case DataType::kFloat16:
    case DataType::kBFloat16:
      return 16;
    case DataType::kFloat32:
    case DataType::kInt32:
      return 32;
    case DataType::kInt64:
      return 64;
  }
  CHECK(false, "invalid DataType value {}", static_cast<int>(dtype));
}

// Element width in bytes. CHECK-fails for sub-byte dtypes (kInt4) — call
// sites that can meet kInt4 must reason in bits or rows (see the packing
// rule above), not elements.
[[nodiscard]] constexpr int itemsize(DataType dtype) {
  CHECK(!is_sub_byte(dtype),
        "itemsize() is undefined for sub-byte dtypes; use itemsize_bits()");
  return itemsize_bits(dtype) / 8;
}

// Canonical lowercase name, matching HuggingFace config.json / safetensors
// dtype strings so `model` can compare against them directly, e.g. "float16".
[[nodiscard]] inline std::string_view to_string(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat16:
      return "float16";
    case DataType::kBFloat16:
      return "bfloat16";
    case DataType::kInt8:
      return "int8";
    case DataType::kUInt8:
      return "uint8";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
    case DataType::kBool:
      return "bool";
    case DataType::kFP8E4M3:
      return "fp8_e4m3";
    case DataType::kInt4:
      return "int4";
  }
  CHECK(false, "invalid DataType value {}", static_cast<int>(dtype));
}

// Parses a canonical name (exact, case-sensitive). Names are data-driven
// input (config.json, fixtures), so unknown ones are InvalidArgument, never
// a CHECK.
[[nodiscard]] inline core::StatusOr<DataType> from_string(
    std::string_view name) {
  for (const DataType dtype : kAllDataTypes) {
    if (name == to_string(dtype)) {
      return dtype;
    }
  }
  return core::InvalidArgumentError("unknown dtype name: \"{}\"", name);
}

// Compile-time C++ type ↔ enum mapping. The primary template is
// intentionally undefined: using an unmapped type is a compile error.
// float16/bfloat16 are specialized in half.h (M1-T07); kFP8E4M3 and kInt4
// deliberately have no mapped C++ type until their milestones need one.
template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<float> {
  static constexpr DataType value = DataType::kFloat32;
};
template <>
struct DTypeTraits<std::int8_t> {
  static constexpr DataType value = DataType::kInt8;
};
template <>
struct DTypeTraits<std::uint8_t> {
  static constexpr DataType value = DataType::kUInt8;
};
template <>
struct DTypeTraits<std::int32_t> {
  static constexpr DataType value = DataType::kInt32;
};
template <>
struct DTypeTraits<std::int64_t> {
  static constexpr DataType value = DataType::kInt64;
};
template <>
struct DTypeTraits<bool> {
  static constexpr DataType value = DataType::kBool;
};

template <typename T>
inline constexpr DataType dtype_of = DTypeTraits<T>::value;

}  // namespace engine::tensor
