#pragma once

#include "core/check.h"
#include "core/status.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>

// Shape & strides (M1-T03; design: docs/design/tensor.md §4).
//
// This is a tensor_base header (ADR-002 Amendment 1): header-only, includes
// nothing from memory or the rest of tensor, may include core.
//
// Strides are element-denominated, not byte-denominated (the PyTorch
// convention): stride math stays independent of dtype, and
// view_as_dtype/cast never rescale strides. Byte offsets are computed at
// access time as element_offset * itemsize. Negative strides are not
// supported anywhere in the engine (no flip views).

namespace engine::tensor {

// Fixed capacity: transformer inference needs ≤ 5 dims in practice
// (e.g. [batch, heads, seq, seq] attention weights); 8 leaves headroom
// without heap allocation.
inline constexpr int kMaxRank = 8;

// Fixed-capacity inline vector of int64_t — the shared representation of a
// Shape's dims and of Strides. Copyable, equality-comparable,
// fmt-formattable as "[2, 3, 4]". Append-only growth: unused trailing slots
// stay zero-initialized, which is what makes the defaulted operator==
// correct.
class DimVector {
 public:
  constexpr DimVector() = default;

  // Values are not validated here (Shape imposes its own dim constraints).
  constexpr DimVector(std::initializer_list<std::int64_t> values) {
    CHECK(values.size() <= static_cast<std::size_t>(kMaxRank),
          "{} values exceed kMaxRank ({})", values.size(), kMaxRank);
    for (const std::int64_t value : values) {
      push_back(value);
    }
  }

  [[nodiscard]] constexpr int size() const { return size_; }

  [[nodiscard]] constexpr std::int64_t operator[](int i) const {
    CHECK(i >= 0 && i < size_, "index {} out of range for size {}", i, size_);
    return values_[static_cast<std::size_t>(i)];
  }

  [[nodiscard]] constexpr std::int64_t& operator[](int i) {
    CHECK(i >= 0 && i < size_, "index {} out of range for size {}", i, size_);
    return values_[static_cast<std::size_t>(i)];
  }

  constexpr void push_back(std::int64_t value) {
    CHECK(size_ < kMaxRank, "DimVector is full: kMaxRank is {}", kMaxRank);
    values_[static_cast<std::size_t>(size_++)] = value;
  }

  [[nodiscard]] constexpr std::span<const std::int64_t> values() const {
    return {values_.data(), static_cast<std::size_t>(size_)};
  }

  friend bool operator==(const DimVector&, const DimVector&) = default;

 private:
  std::array<std::int64_t, kMaxRank> values_{};
  int size_ = 0;
};

// Strides deliberately reuse the same inline container; they are measured in
// ELEMENTS (see the header comment).
using Strides = DimVector;

namespace detail {

// Why a candidate dim list cannot form a Shape (kNone: it can).
enum class ShapeDefect : std::uint8_t {
  kNone,
  kRankTooLarge,
  kNegativeDim,
  kNumelOverflow,
};

struct DimsValidation {
  ShapeDefect defect = ShapeDefect::kNone;
  int index = 0;           // Offending dim, for kNegativeDim.
  std::int64_t numel = 1;  // Meaningful only when defect == kNone.
};

// The single validation path shared by Shape's CHECKing constructor and
// Shape::FromDims, so the two entry points cannot drift.
//
// Overflow is judged on the product of the NON-ZERO dims: it is the true
// numel when no dim is 0 (a 0 dim makes numel 0, which cannot overflow), and
// it bounds every row-major stride — RowMajorStrides treats 0-sized dims as
// size 1, so each stride is a product of a subset of the non-zero dims.
constexpr DimsValidation ValidateDims(std::span<const std::int64_t> dims) {
  DimsValidation result;
  if (dims.size() > static_cast<std::size_t>(kMaxRank)) {
    result.defect = ShapeDefect::kRankTooLarge;
    return result;
  }
  bool has_zero = false;
  std::int64_t product = 1;
  for (std::size_t i = 0; i < dims.size(); ++i) {
    const std::int64_t dim = dims[i];
    if (dim < 0) {
      result.defect = ShapeDefect::kNegativeDim;
      result.index = static_cast<int>(i);
      return result;
    }
    if (dim == 0) {
      has_zero = true;
    } else if (__builtin_mul_overflow(product, dim, &product)) {
      result.defect = ShapeDefect::kNumelOverflow;
      return result;
    }
  }
  result.numel = has_zero ? 0 : product;
  return result;
}

}  // namespace detail

// Dim list of a tensor. Value type: copyable, equality-comparable,
// fmt-formattable as "[2, 3, 4]". Rank 0 (a scalar) is valid and has
// numel() == 1; dims of 0 are valid and give numel() == 0.
//
// Error split (design §4): the initializer_list constructor CHECKs — inline
// dim literals are programmer-authored. Shapes derived from data (files,
// requests) must come through FromDims, which returns InvalidArgument.
class Shape {
 public:
  constexpr Shape() = default;  // Rank 0; numel() == 1.

  constexpr Shape(std::initializer_list<std::int64_t> dims) {
    const detail::DimsValidation validation =
        detail::ValidateDims({dims.begin(), dims.size()});
    CHECK(validation.defect != detail::ShapeDefect::kRankTooLarge,
          "rank {} exceeds kMaxRank ({})", dims.size(), kMaxRank);
    CHECK(validation.defect != detail::ShapeDefect::kNegativeDim,
          "negative dim {} at index {}", dims.begin()[validation.index],
          validation.index);
    CHECK(validation.defect != detail::ShapeDefect::kNumelOverflow,
          "dim product overflows int64_t");
    for (const std::int64_t dim : dims) {
      dims_.push_back(dim);
    }
    numel_ = validation.numel;
  }

  // Validating factory for shapes derived from data rather than written as
  // literals: negative dims, rank > kMaxRank, and numel overflow are
  // InvalidArgument, never a CHECK.
  [[nodiscard]] static core::StatusOr<Shape> FromDims(
      std::span<const std::int64_t> dims) {
    const detail::DimsValidation validation = detail::ValidateDims(dims);
    switch (validation.defect) {
      case detail::ShapeDefect::kNone:
        break;
      case detail::ShapeDefect::kRankTooLarge:
        return core::InvalidArgumentError("rank {} exceeds kMaxRank ({})",
                                          dims.size(), kMaxRank);
      case detail::ShapeDefect::kNegativeDim:
        return core::InvalidArgumentError(
            "negative dim {} at index {}",
            dims[static_cast<std::size_t>(validation.index)], validation.index);
      case detail::ShapeDefect::kNumelOverflow:
        return core::InvalidArgumentError(
            "dim product overflows int64_t for dims [{}]",
            fmt::join(dims, ", "));
    }
    Shape shape;
    for (const std::int64_t dim : dims) {
      shape.dims_.push_back(dim);
    }
    shape.numel_ = validation.numel;
    return shape;
  }

  [[nodiscard]] constexpr int rank() const { return dims_.size(); }

  // CHECKs 0 ≤ i < rank().
  [[nodiscard]] constexpr std::int64_t dim(int i) const { return dims_[i]; }

  // Cached at construction (under the overflow check); a cheap read.
  [[nodiscard]] constexpr std::int64_t numel() const { return numel_; }

  [[nodiscard]] constexpr std::span<const std::int64_t> dims() const {
    return dims_.values();
  }

  friend bool operator==(const Shape&, const Shape&) = default;

 private:
  DimVector dims_;
  std::int64_t numel_ = 1;
};

// Contiguous row-major (C-order) strides: the innermost stride is 1 and each
// outer stride is the product of the inner dims. Dims of size 0 contribute a
// factor of 1 (the PyTorch convention), keeping every stride positive; Shape
// validation guarantees these products cannot overflow.
[[nodiscard]] constexpr Strides RowMajorStrides(const Shape& shape) {
  Strides strides;
  for (int i = 0; i < shape.rank(); ++i) {
    strides.push_back(1);
  }
  std::int64_t running = 1;
  for (int i = shape.rank() - 1; i >= 0; --i) {
    strides[i] = running;
    running *= std::max<std::int64_t>(shape.dim(i), 1);
  }
  return strides;
}

// True iff `strides` lays `shape` out densely in row-major order. Dims of
// size 1 impose no constraint on their stride, and a shape with numel() == 0
// is contiguous under any strides (design §4). A rank mismatch is a
// programmer error.
[[nodiscard]] constexpr bool IsContiguous(const Shape& shape,
                                          const Strides& strides) {
  CHECK(strides.size() == shape.rank(), "strides rank {} != shape rank {}",
        strides.size(), shape.rank());
  if (shape.numel() == 0) {
    return true;
  }
  std::int64_t expected = 1;
  for (int i = shape.rank() - 1; i >= 0; --i) {
    const std::int64_t dim = shape.dim(i);
    if (dim == 1) {
      continue;  // A size-1 dim is never stepped over; any stride works.
    }
    if (strides[i] != expected) {
      return false;
    }
    expected *= dim;
  }
  return true;
}

}  // namespace engine::tensor

// "[2, 3, 4]"; "[]" for rank 0. One formatter covers DimVector and its
// Strides alias.
template <>
struct fmt::formatter<engine::tensor::DimVector> {
  static constexpr fmt::format_parse_context::iterator parse(
      fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  static fmt::format_context::iterator format(
      const engine::tensor::DimVector& dims, fmt::format_context& ctx) {
    return fmt::format_to(ctx.out(), "[{}]", fmt::join(dims.values(), ", "));
  }
};

template <>
struct fmt::formatter<engine::tensor::Shape> {
  static constexpr fmt::format_parse_context::iterator parse(
      fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  static fmt::format_context::iterator format(
      const engine::tensor::Shape& shape, fmt::format_context& ctx) {
    return fmt::format_to(ctx.out(), "[{}]", fmt::join(shape.dims(), ", "));
  }
};
