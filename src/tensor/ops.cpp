#include "tensor/ops.h"

#include "core/check.h"
#include "core/status.h"
#include "cuda/stream_handle.h"
#include "memory/allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"
#include "tensor/transfer_detail.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

namespace engine::tensor::ops {
namespace {

using core::InvalidArgumentError;
using core::OkStatus;
using core::Status;
using core::StatusOr;

template <typename T>
struct TypeTag {
  using type = T;
};

// Calls fn(TypeTag<T>{}) for the host C++ type mapped to `dtype`. Reserved
// dtypes (kInt4, kFP8E4M3) have no host type; no Tensor can carry them until
// their milestone (empty/view_as_dtype reject them), so reaching this switch
// with one is a programmer error.
template <typename Fn>
decltype(auto) DispatchDtype(DataType dtype, Fn&& fn) {
  switch (dtype) {
    case DataType::kFloat32:
      return fn(TypeTag<float>{});
    case DataType::kFloat16:
      return fn(TypeTag<float16>{});
    case DataType::kBFloat16:
      return fn(TypeTag<bfloat16>{});
    case DataType::kInt8:
      return fn(TypeTag<std::int8_t>{});
    case DataType::kUInt8:
      return fn(TypeTag<std::uint8_t>{});
    case DataType::kInt32:
      return fn(TypeTag<std::int32_t>{});
    case DataType::kInt64:
      return fn(TypeTag<std::int64_t>{});
    case DataType::kBool:
      return fn(TypeTag<bool>{});
    case DataType::kFP8E4M3:
    case DataType::kInt4:
      break;
  }
  CHECK(false, "no host type for dtype {}", to_string(dtype));
}

// Visits every logical index of `shape` in row-major order (the iteration
// order the fill determinism contract promises). Rank 0 visits once with the
// empty index; numel 0 never visits.
template <typename Visit>
void ForEachIndex(const Shape& shape, Visit&& visit) {
  const std::int64_t total = shape.numel();
  DimVector index;
  for (int i = 0; i < shape.rank(); ++i) {
    index.push_back(0);
  }
  for (std::int64_t element = 0; element < total; ++element) {
    visit(std::as_const(index));
    for (int i = shape.rank() - 1; i >= 0; --i) {
      if (++index[i] < shape.dim(i)) {
        break;
      }
      index[i] = 0;
    }
  }
}

// Element offset (in elements, per §4 strides) of a logical index.
[[nodiscard]] std::int64_t ElementOffset(const DimVector& index,
                                         const Strides& strides) {
  std::int64_t offset = 0;
  for (int i = 0; i < index.size(); ++i) {
    offset += index[i] * strides[i];
  }
  return offset;
}

// double → element type, per the M1-T07 narrowing policy for halves. Integer
// callers have already validated the range.
template <typename T>
[[nodiscard]] T NarrowDouble(double value) {
  if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
    return T(static_cast<float>(value));
  } else if constexpr (std::is_same_v<T, bool>) {
    return value != 0.0;
  } else {
    return static_cast<T>(value);
  }
}

// Widen an element to double for comparison/printing. Halves widen exactly
// (via their operator float); every integer dtype's values fit a double
// exactly except int64 beyond 2^53, which only loses reporting precision
// (comparison of integers is exact, done in T).
template <typename T>
[[nodiscard]] double WidenToDouble(T value) {
  return static_cast<double>(value);
}

// |v| as uint64, via two's-complement negation — well-defined for every
// value including INT64_MIN (whose magnitude, 2^63, does not fit int64).
[[nodiscard]] constexpr std::uint64_t UnsignedMagnitude(std::int64_t v) {
  const auto u = static_cast<std::uint64_t>(v);
  return v < 0 ? ~u + 1 : u;
}

// Whether integer value `v` is representable in integer dtype `dtype`.
[[nodiscard]] bool FitsIntegerDtype(DataType dtype, std::int64_t v) {
  switch (dtype) {
    case DataType::kInt8:
      return std::in_range<std::int8_t>(v);
    case DataType::kUInt8:
      return std::in_range<std::uint8_t>(v);
    case DataType::kInt32:
      return std::in_range<std::int32_t>(v);
    case DataType::kInt64:
      return true;
    case DataType::kBool:
      return v == 0 || v == 1;
    default:
      break;
  }
  CHECK(false, "FitsIntegerDtype: {} is not an integer dtype",
        to_string(dtype));
}

// Whether double `value` is integral and representable in integer/bool dtype
// `dtype`. The int64 upper bound is exclusive: 2^63 is exactly representable
// as a double but is one past kInt64's max.
[[nodiscard]] bool FitsIntegerDtype(DataType dtype, double value) {
  if (std::trunc(value) != value) {  // Also rejects NaN.
    return false;
  }
  if (dtype == DataType::kInt64) {
    return value >= -0x1p63 && value < 0x1p63;
  }
  if (value < -0x1p63 || value >= 0x1p63) {
    return false;  // ±inf or beyond int64; no smaller dtype can hold it.
  }
  return FitsIntegerDtype(dtype, static_cast<std::int64_t>(value));
}

// Whether `dtype` participates in cast: the floating and integer kinds,
// excluding kBool (neither kind, and float→bool semantics are ambiguous).
// Reserved dtypes slip through the predicate (kFP8E4M3 is floating, kInt4
// integral) but can never reach a conversion loop: no Tensor can carry them
// as a source, and as a target Tensor::empty returns Unimplemented first.
[[nodiscard]] constexpr bool IsCastableDtype(DataType dtype) {
  return is_floating_point(dtype) || is_integral(dtype);
}

// Converts one element Src → Dst per the cast policy in ops.h. Returns
// false when the value is not representable in Dst (integer targets only;
// floating targets absorb everything via ±inf/NaN).
template <typename Dst, typename Src>
[[nodiscard]] bool CastElement(Src value, Dst& out) {
  if constexpr (is_floating_point(dtype_of<Dst>)) {
    // Exact widen (int64 > 2^53 rounds to nearest), M1-T07 narrow.
    out = NarrowDouble<Dst>(WidenToDouble(value));
    return true;
  } else if constexpr (is_floating_point(dtype_of<Src>)) {
    // trunc keeps NaN and ±inf, which FitsIntegerDtype then rejects.
    const double truncated = std::trunc(WidenToDouble(value));
    if (!FitsIntegerDtype(dtype_of<Dst>, truncated)) {
      return false;
    }
    out = static_cast<Dst>(truncated);
    return true;
  } else {
    // int8 tensor elements are numbers, not text; this value-preserving
    // widening is exactly the conversion the check exists to question.
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    const auto widened = static_cast<std::int64_t>(value);
    if (!FitsIntegerDtype(dtype_of<Dst>, widened)) {
      return false;
    }
    out = static_cast<Dst>(widened);
    return true;
  }
}

// The seeded generator behind both fills. std::mt19937_64's output sequence
// is fully specified by the standard; the [0,1) / (0,1] transforms are pure
// arithmetic (see ops.h).
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed) {}

  // Uniform on [0, 1): 53 random bits scaled by 2^-53.
  [[nodiscard]] double NextUnit() {
    return static_cast<double>(engine_() >> 11) * 0x1.0p-53;
  }

  // Uniform on (0, 1] — safe as a log() argument.
  [[nodiscard]] double NextUnitNonzero() {
    return static_cast<double>((engine_() >> 11) + 1) * 0x1.0p-53;
  }

 private:
  std::mt19937_64 engine_;
};

// One element pair judged by the allclose criterion. `diff` is |a - b| in
// double, with the special cases: NaN when either value is NaN (never
// close), 0 for same-signed infinities (close), +inf for opposite ones.
struct ElementComparison {
  bool close;
  double diff;
  double a;  // widened operands, for the report
  double b;
};

template <typename T>
[[nodiscard]] ElementComparison CompareElement(T va, T vb, double rtol,
                                               double atol) {
  const double da = WidenToDouble(va);
  const double db = WidenToDouble(vb);
  if constexpr (!is_floating_point(dtype_of<T>)) {
    // Integer dtypes and kBool compare exactly, in T (int64 values beyond
    // 2^53 would lose precision in double).
    return {.close = va == vb, .diff = std::abs(da - db), .a = da, .b = db};
  } else {
    if (std::isnan(da) || std::isnan(db)) {
      return {.close = false,
              .diff = std::numeric_limits<double>::quiet_NaN(),
              .a = da,
              .b = db};
    }
    if (std::isinf(da) || std::isinf(db)) {
      const bool close = da == db;
      return {.close = close,
              .diff = close ? 0.0 : std::numeric_limits<double>::infinity(),
              .a = da,
              .b = db};
    }
    const double diff = std::abs(da - db);
    return {.close = diff <= atol + (rtol * std::abs(db)),
            .diff = diff,
            .a = da,
            .b = db};
  }
}

// Shared validation for the fills. Undefined/non-CPU are programmer errors;
// a non-floating dtype is data-dependent (tensors come from loaders/tests).
[[nodiscard]] Status ValidateFillTarget(const Tensor& tensor,
                                        const char* op_name) {
  CHECK(tensor.defined(), "{}: undefined Tensor", op_name);
  CHECK(tensor.device().is_cpu(), "{} requires a CPU tensor; device is {}",
        op_name, tensor.device().ToString());
  if (!is_floating_point(tensor.dtype())) {
    return InvalidArgumentError("{}: dtype {} is not floating-point", op_name,
                                to_string(tensor.dtype()));
  }
  return OkStatus();
}

// Fills `tensor` (validated, floating dtype) with generate() outputs in
// logical row-major order.
template <typename Generate>
void FillWith(const Tensor& tensor, Generate&& generate) {
  DispatchDtype(tensor.dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* data = tensor.data_ptr<T>();
    const Strides& strides = tensor.strides();
    ForEachIndex(tensor.shape(), [&](const DimVector& index) {
      data[ElementOffset(index, strides)] = NarrowDouble<T>(generate());
    });
  });
}

// Recursive worker for to_string: appends the sub-tensor at `dim` whose
// first element sits at `offset`, indented so nested rows align under the
// opening bracket.
template <typename T>
void AppendSlice(const Tensor& tensor, int dim, std::int64_t offset,
                 std::int64_t edge_items, std::string& out) {
  const int rank = tensor.shape().rank();
  if (dim == rank) {
    fmt::format_to(std::back_inserter(out), "{}", tensor.data_ptr<T>()[offset]);
    return;
  }
  const std::int64_t n = tensor.shape().dim(dim);
  const std::int64_t stride = tensor.strides()[dim];
  const bool innermost = dim + 1 == rank;
  const bool truncate = n > 2 * edge_items;

  out += '[';
  bool first = true;
  auto append_separator = [&] {
    if (!first) {
      if (innermost) {
        out += ", ";
      } else {
        out += ",\n";
        out.append(static_cast<std::size_t>(dim) + 1, ' ');
      }
    }
    first = false;
  };
  auto append_item = [&](std::int64_t i) {
    append_separator();
    AppendSlice<T>(tensor, dim + 1, offset + (i * stride), edge_items, out);
  };
  if (truncate) {
    for (std::int64_t i = 0; i < edge_items; ++i) {
      append_item(i);
    }
    append_separator();
    out += "...";
    for (std::int64_t i = n - edge_items; i < n; ++i) {
      append_item(i);
    }
  } else {
    for (std::int64_t i = 0; i < n; ++i) {
      append_item(i);
    }
  }
  out += ']';
}

}  // namespace

StatusOr<Tensor> zeros(const Shape& shape, DataType dtype, Device device,
                       memory::Allocator* allocator) {
  return full(shape, dtype, 0.0, device, allocator);
}

StatusOr<Tensor> ones(const Shape& shape, DataType dtype, Device device,
                      memory::Allocator* allocator) {
  return full(shape, dtype, 1.0, device, allocator);
}

StatusOr<Tensor> full(const Shape& shape, DataType dtype, double value,
                      Device device, memory::Allocator* allocator) {
  // Allocate first so a reserved dtype reports Unimplemented (its real
  // blocker), not a complaint about the fill value.
  ASSIGN_OR_RETURN(Tensor tensor,
                   Tensor::empty(shape, dtype, device, allocator));
  if (!is_floating_point(dtype) && !FitsIntegerDtype(dtype, value)) {
    return InvalidArgumentError("full: value {} is not representable in {}",
                                value, to_string(dtype));
  }
  if (tensor.numel() == 0) {
    return tensor;
  }
  DispatchDtype(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    std::fill_n(tensor.data_ptr<T>(), tensor.numel(), NarrowDouble<T>(value));
  });
  return tensor;
}

StatusOr<Tensor> arange(std::int64_t start, std::int64_t end, std::int64_t step,
                        DataType dtype, Device device,
                        memory::Allocator* allocator) {
  if (step == 0) {
    return InvalidArgumentError("arange: step must be non-zero");
  }
  std::int64_t span = 0;
  if (__builtin_sub_overflow(end, start, &span)) {
    return InvalidArgumentError(
        "arange: end - start overflows int64 for "
        "start={}, end={}",
        start, end);
  }
  std::int64_t count = 0;
  if ((step > 0) == (span > 0) && span != 0) {
    // ceil(|span| / |step|) in uint64: span == INT64_MIN with step == -1
    // would make the signed division itself UB, and its count (2^63) does
    // not fit int64 anyway.
    const std::uint64_t uspan = UnsignedMagnitude(span);
    const std::uint64_t ustep = UnsignedMagnitude(step);
    const std::uint64_t ucount = (uspan / ustep) + (uspan % ustep != 0 ? 1 : 0);
    if (ucount >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return InvalidArgumentError(
          "arange: element count {} overflows int64 for start={}, end={}, "
          "step={}",
          ucount, start, end, step);
    }
    count = static_cast<std::int64_t>(ucount);
  }
  ASSIGN_OR_RETURN(Tensor tensor,
                   Tensor::empty(Shape{count}, dtype, device, allocator));
  if (count == 0) {
    return tensor;
  }
  // Values are monotonic from start to last, so the extremes prove the whole
  // range representable. (count - 1) * step is within span, so no overflow.
  const std::int64_t last = start + ((count - 1) * step);
  if (!is_floating_point(dtype) &&
      (!FitsIntegerDtype(dtype, start) || !FitsIntegerDtype(dtype, last))) {
    return InvalidArgumentError(
        "arange: values [{}, {}] are not representable in {}", start, last,
        to_string(dtype));
  }
  DispatchDtype(dtype, [&](auto tag) {
    using T = typename decltype(tag)::type;
    T* data = tensor.data_ptr<T>();
    for (std::int64_t i = 0; i < count; ++i) {
      if constexpr (is_floating_point(dtype_of<T>)) {
        data[i] = NarrowDouble<T>(
            static_cast<double>(start) +
            (static_cast<double>(i) * static_cast<double>(step)));
      } else {
        data[i] = static_cast<T>(start + (i * step));
      }
    }
  });
  return tensor;
}

Status fill_uniform(const Tensor& tensor, double low, double high,
                    std::uint64_t seed) {
  RETURN_IF_ERROR(ValidateFillTarget(tensor, "fill_uniform"));
  if (!std::isfinite(low) || !std::isfinite(high) || low > high ||
      !std::isfinite(high - low)) {
    return InvalidArgumentError("fill_uniform: invalid range [{}, {})", low,
                                high);
  }
  Rng rng(seed);
  const double extent = high - low;
  FillWith(tensor, [&] { return low + (rng.NextUnit() * extent); });
  return OkStatus();
}

Status fill_normal(const Tensor& tensor, double mean, double stddev,
                   std::uint64_t seed) {
  RETURN_IF_ERROR(ValidateFillTarget(tensor, "fill_normal"));
  if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev < 0) {
    return InvalidArgumentError("fill_normal: invalid mean {} / stddev {}",
                                mean, stddev);
  }
  Rng rng(seed);
  bool has_pending = false;
  double pending = 0.0;
  FillWith(tensor, [&] {
    double z = 0.0;
    if (has_pending) {
      has_pending = false;
      z = pending;
    } else {
      const double u1 = rng.NextUnitNonzero();
      const double u2 = rng.NextUnit();
      const double radius = std::sqrt(-2.0 * std::log(u1));
      const double angle = 2.0 * std::numbers::pi * u2;
      pending = radius * std::sin(angle);
      has_pending = true;
      z = radius * std::cos(angle);
    }
    return mean + (stddev * z);
  });
  return OkStatus();
}

Status copy(const Tensor& dst, const Tensor& src) {
  CHECK(dst.defined() && src.defined(), "copy: undefined Tensor");
  CHECK(dst.device().is_cpu() && src.device().is_cpu(),
        "copy requires CPU tensors; devices are {} and {}",
        dst.device().ToString(), src.device().ToString());
  if (dst.dtype() != src.dtype()) {
    return InvalidArgumentError(
        "copy: dtype mismatch: {} vs {} (dtype conversion is cast)",
        to_string(dst.dtype()), to_string(src.dtype()));
  }
  if (dst.shape() != src.shape()) {
    return InvalidArgumentError("copy: shape mismatch: {} vs {}", dst.shape(),
                                src.shape());
  }
  if (dst.numel() == 0) {
    return OkStatus();
  }
  if (dst.is_contiguous() && src.is_contiguous()) {
    // Equal pointers with matching shape/dtype means identical contiguous
    // views: a no-op (and memcpy onto itself would be UB).
    if (dst.data() != src.data()) {
      std::memcpy(dst.data(), src.data(),
                  static_cast<std::size_t>(dst.numel()) *
                      static_cast<std::size_t>(itemsize(dst.dtype())));
    }
    return OkStatus();
  }
  DispatchDtype(dst.dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    const T* src_data = src.data_ptr<T>();
    T* dst_data = dst.data_ptr<T>();
    ForEachIndex(dst.shape(), [&](const DimVector& index) {
      dst_data[ElementOffset(index, dst.strides())] =
          src_data[ElementOffset(index, src.strides())];
    });
  });
  return OkStatus();
}

Status copy(const Tensor& dst, const Tensor& src, cuda::StreamHandle stream) {
  CHECK(dst.defined() && src.defined(), "copy: undefined Tensor");
  if (dst.dtype() != src.dtype()) {
    return InvalidArgumentError(
        "copy: dtype mismatch: {} vs {} (dtype conversion is cast)",
        to_string(dst.dtype()), to_string(src.dtype()));
  }
  if (dst.shape() != src.shape()) {
    return InvalidArgumentError("copy: shape mismatch: {} vs {}", dst.shape(),
                                src.shape());
  }
  if (!dst.is_contiguous() || !src.is_contiguous()) {
    return InvalidArgumentError(
        "copy: the stream overload requires contiguous tensors (dst strides "
        "{}, src strides {}); strided device copy is not implemented",
        dst.strides(), src.strides());
  }
  const bool dst_cuda = dst.device().is_cuda();
  const bool src_cuda = src.device().is_cuda();
  if (!dst_cuda && !src_cuda) {
    return copy(dst, src);  // H2H: the host memcpy path, stream ignored
  }
  if (dst_cuda && src_cuda && dst.device() != src.device()) {
    return core::UnimplementedError(
        "copy: cross-device copy ({} -> {}) is not implemented until the "
        "distributed milestone",
        src.device().ToString(), dst.device().ToString());
  }
  if (dst.numel() == 0) {
    return OkStatus();
  }
  // Equal pointers with matching shape/dtype and contiguity means identical
  // views: a no-op (and a self-memcpy on the device would be UB). Comparing
  // a host pointer with a device pointer here is sound because CUDA's
  // unified virtual addressing (all 64-bit platforms we target) guarantees
  // host and device allocations occupy disjoint address ranges.
  if (dst.data() == src.data()) {
    return OkStatus();
  }
  return detail::DeviceCopy(dst, src, stream, /*synchronize=*/false);
}

StatusOr<Tensor> cast(const Tensor& src, DataType dtype,
                      memory::Allocator* allocator) {
  CHECK(src.defined(), "cast: undefined Tensor");
  CHECK(src.device().is_cpu(), "cast requires a CPU tensor; device is {}",
        src.device().ToString());
  // Allocate first so a reserved target dtype reports Unimplemented (its
  // real blocker), mirroring full's ordering.
  ASSIGN_OR_RETURN(Tensor out,
                   Tensor::empty(src.shape(), dtype, src.device(), allocator));
  if (!IsCastableDtype(src.dtype()) || !IsCastableDtype(dtype)) {
    return InvalidArgumentError("cast: unsupported dtype pair {} -> {}",
                                to_string(src.dtype()), to_string(dtype));
  }
  if (src.numel() == 0) {
    return out;
  }
  Status status = OkStatus();
  DispatchDtype(src.dtype(), [&](auto src_tag) {
    using Src = typename decltype(src_tag)::type;
    DispatchDtype(dtype, [&](auto dst_tag) {
      using Dst = typename decltype(dst_tag)::type;
      const Src* src_data = src.data_ptr<Src>();
      Dst* dst_data = out.data_ptr<Dst>();
      const Strides& src_strides = src.strides();
      // The output is freshly allocated contiguous row-major, so the
      // logical row-major walk writes it linearly.
      std::int64_t linear = 0;
      ForEachIndex(src.shape(), [&](const DimVector& index) {
        if (!status.ok()) {
          return;
        }
        const Src value = src_data[ElementOffset(index, src_strides)];
        Dst converted{};
        if (!CastElement(value, converted)) {
          status = InvalidArgumentError(
              "cast: value {} at index {} is not representable in {}",
              WidenToDouble(value), index, to_string(dtype));
          return;
        }
        dst_data[linear++] = converted;
      });
    });
  });
  RETURN_IF_ERROR(status);
  return out;
}

std::string AllCloseResult::Summary() const {
  if (allclose) {
    return fmt::format(
        "allclose: OK — {} elements, max abs diff {:.6g} (rtol={:.6g}, "
        "atol={:.6g})",
        numel, max_abs_diff, rtol, atol);
  }
  return fmt::format(
      "allclose: FAILED — {} of {} elements mismatch; worst at {}: a={:.9g}, "
      "b={:.9g} (abs diff {:.6g}); max abs diff {:.6g} (rtol={:.6g}, "
      "atol={:.6g})",
      mismatch_count, numel, worst_index, worst_a, worst_b,
      std::abs(worst_a - worst_b), max_abs_diff, rtol, atol);
}

StatusOr<AllCloseResult> allclose(const Tensor& a, const Tensor& b,
                                  std::optional<double> rtol,
                                  std::optional<double> atol) {
  CHECK(a.defined() && b.defined(), "allclose: undefined Tensor");
  CHECK(a.device().is_cpu() && b.device().is_cpu(),
        "allclose requires CPU tensors; devices are {} and {}",
        a.device().ToString(), b.device().ToString());
  if (a.dtype() != b.dtype()) {
    return InvalidArgumentError("allclose: dtype mismatch: {} vs {}",
                                to_string(a.dtype()), to_string(b.dtype()));
  }
  if (a.shape() != b.shape()) {
    return InvalidArgumentError("allclose: shape mismatch: {} vs {}", a.shape(),
                                b.shape());
  }
  const AllCloseTolerance defaults = default_allclose_tolerance(a.dtype());
  AllCloseResult result;
  result.rtol = rtol.value_or(defaults.rtol);
  result.atol = atol.value_or(defaults.atol);
  result.numel = a.numel();
  // !(x >= 0) also rejects NaN.
  if (!(result.rtol >= 0.0) || !(result.atol >= 0.0)) {
    return InvalidArgumentError(
        "allclose: invalid tolerances rtol={}, "
        "atol={}",
        result.rtol, result.atol);
  }
  if (result.numel == 0) {
    return result;
  }
  double worst_badness = -1.0;
  DispatchDtype(a.dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    const T* data_a = a.data_ptr<T>();
    const T* data_b = b.data_ptr<T>();
    ForEachIndex(a.shape(), [&](const DimVector& index) {
      const T va = data_a[ElementOffset(index, a.strides())];
      const T vb = data_b[ElementOffset(index, b.strides())];
      const ElementComparison cmp =
          CompareElement(va, vb, result.rtol, result.atol);
      if (!std::isnan(cmp.diff)) {
        result.max_abs_diff = std::max(result.max_abs_diff, cmp.diff);
      }
      if (cmp.close) {
        return;
      }
      ++result.mismatch_count;
      // A NaN-involved mismatch is maximally bad and wins the worst slot.
      const double badness = std::isnan(cmp.diff)
                                 ? std::numeric_limits<double>::infinity()
                                 : cmp.diff;
      if (badness > worst_badness) {
        worst_badness = badness;
        result.worst_index = index;
        result.worst_a = cmp.a;
        result.worst_b = cmp.b;
      }
    });
  });
  result.allclose = result.mismatch_count == 0;
  return result;
}

std::string to_string(const Tensor& tensor, std::int64_t edge_items) {
  CHECK(edge_items > 0, "to_string: edge_items must be positive, got {}",
        edge_items);
  if (!tensor.defined()) {
    return "Tensor(undefined)";
  }
  CHECK(tensor.device().is_cpu(),
        "to_string requires a CPU tensor; device is {}",
        tensor.device().ToString());
  std::string out =
      fmt::format("Tensor(shape={}, dtype={}, device={})\n", tensor.shape(),
                  tensor::to_string(tensor.dtype()), tensor.device());
  DispatchDtype(tensor.dtype(), [&](auto tag) {
    using T = typename decltype(tag)::type;
    AppendSlice<T>(tensor, 0, 0, edge_items, out);
  });
  return out;
}

}  // namespace engine::tensor::ops
