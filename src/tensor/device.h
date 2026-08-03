#pragma once

#include "core/check.h"
#include "core/status.h"

#include <fmt/format.h>

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

// Device abstraction (M1-T04; design: docs/design/tensor.md §5).
//
// This is a tensor_base header (ADR-002 Amendment 1): header-only, includes
// nothing from memory or the rest of tensor, may include core. It is also
// backend-agnostic: no CUDA headers, ever — a CUDA Device is *representable*
// on any build, but allocating on one returns Unimplemented until M2 (and
// Unavailable-style failures after, on non-CUDA builds).
//
// `Device` is just an address. The device *registry* (how many CUDA devices
// exist, their properties) is M2's problem (`src/cuda/`); nothing here
// validates `index` against hardware — validation happens where a device is
// *used* (allocation, transfer), where a `Status` can carry real context.

namespace engine::tensor {

// Values are stable; append, never renumber (they will reach fixtures and
// serialized test data).
enum class DeviceType : std::uint8_t {
  kCPU = 0,
  kCUDA = 1,
};

// Canonical lowercase name, e.g. "cpu".
[[nodiscard]] constexpr std::string_view to_string(DeviceType type) {
  switch (type) {
    case DeviceType::kCPU:
      return "cpu";
    case DeviceType::kCUDA:
      return "cuda";
  }
  CHECK(false, "invalid DeviceType value {}", static_cast<int>(type));
}

// Plain value type naming a memory/execution space. Cheap to copy,
// equality-comparable, fmt-formattable as "cpu" / "cuda:0".
//
// Invariants, enforced at construction (CHECK — Device values are built by
// our own code via the factories or Parse, never directly from external
// input): index >= 0 always, and index == 0 for kCPU. Enforcing them here
// is what keeps operator== and ToString/Parse round-trips consistent:
// without it a Device(kCPU, 5) would format as "cpu" yet compare unequal
// to Cpu().
class Device {
 public:
  constexpr Device() = default;  // cpu

  // CHECKs the invariants above. Data-driven specs go through Parse, which
  // returns InvalidArgument instead.
  constexpr Device(DeviceType type, int index) : type_(type), index_(index) {
    CHECK(index_ >= 0, "negative device index {}", index_);
    CHECK(type_ != DeviceType::kCPU || index_ == 0,
          "cpu device index must be 0, got {}", index_);
  }

  [[nodiscard]] static constexpr Device Cpu() { return {}; }
  [[nodiscard]] static constexpr Device Cuda(int index) {
    return {DeviceType::kCUDA, index};
  }

  [[nodiscard]] constexpr DeviceType type() const { return type_; }
  // Always 0 for kCPU; ordinal for kCUDA.
  [[nodiscard]] constexpr int index() const { return index_; }

  // Parses a device spec. Accepts exactly "cpu", "cuda" (== "cuda:0"), and
  // "cuda:N" with N a base-10 int; anything else — including "cpu:0", other
  // casings, whitespace, signs, or an out-of-int-range N — is
  // InvalidArgument. Specs are data-driven input (config, requests), so bad
  // ones are a Status, never a CHECK. Leading zeros parse ("cuda:07" == 7);
  // canonical output comes from ToString.
  [[nodiscard]] static core::StatusOr<Device> Parse(std::string_view spec) {
    if (spec == to_string(DeviceType::kCPU)) {
      return Cpu();
    }
    constexpr std::string_view kCuda = to_string(DeviceType::kCUDA);
    if (spec == kCuda) {
      return Cuda(0);
    }
    if (spec.size() > kCuda.size() + 1 && spec.starts_with(kCuda) &&
        spec[kCuda.size()] == ':') {
      const std::string_view digits = spec.substr(kCuda.size() + 1);
      int index = 0;
      const auto [end, ec] =
          std::from_chars(digits.data(), digits.data() + digits.size(), index);
      if (ec == std::errc{} && end == digits.data() + digits.size() &&
          index >= 0) {
        return Cuda(index);
      }
    }
    return core::InvalidArgumentError(
        "invalid device spec: \"{}\" (expected \"cpu\", \"cuda\", or "
        "\"cuda:N\")",
        spec);
  }

  // Canonical spec: "cpu" for CPU, "cuda:N" for CUDA (index always printed).
  [[nodiscard]] std::string ToString() const {
    if (type_ == DeviceType::kCPU) {
      return std::string(to_string(type_));
    }
    return fmt::format("{}:{}", to_string(type_), index_);
  }

  [[nodiscard]] constexpr bool is_cpu() const {
    return type_ == DeviceType::kCPU;
  }
  [[nodiscard]] constexpr bool is_cuda() const {
    return type_ == DeviceType::kCUDA;
  }

  friend constexpr bool operator==(const Device&, const Device&) = default;

 private:
  DeviceType type_ = DeviceType::kCPU;
  int index_ = 0;
};

}  // namespace engine::tensor

// "cpu", "cuda:1" — same text as Device::ToString.
template <>
struct fmt::formatter<engine::tensor::DeviceType> {
  static constexpr fmt::format_parse_context::iterator parse(
      fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  static fmt::format_context::iterator format(engine::tensor::DeviceType type,
                                              fmt::format_context& ctx) {
    return fmt::format_to(ctx.out(), "{}", engine::tensor::to_string(type));
  }
};

template <>
struct fmt::formatter<engine::tensor::Device> {
  static constexpr fmt::format_parse_context::iterator parse(
      fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  static fmt::format_context::iterator format(
      const engine::tensor::Device& device, fmt::format_context& ctx) {
    return fmt::format_to(ctx.out(), "{}", device.ToString());
  }
};
