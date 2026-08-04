#pragma once

#include "core/check.h"
#include "core/status.h"
#include "cuda/stream_handle.h"
#include "memory/allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <optional>
#include <string>

// CPU tensor ops (M1-T08/09; design: docs/design/tensor.md §2, §7.1, §10):
// factories, seeded random fills, copy/cast, element-wise comparison, and
// debug printing.
//
// Everything in this header runs on the host and dereferences tensor data, so
// per design §8 the value-touching entry points CHECK `is_cpu()`. (One
// exception: the M2-T07 stream overload of `copy` is device-aware and hands
// device placements to cudaMemcpyAsync — it never dereferences.) All ops are
// free functions: no hidden dispatch, no implicit copies, no broadcasting.
//
// Error split (per the design §7 table): dtype/shape/range problems in the
// arguments are data-dependent → InvalidArgument; operating on an undefined
// handle or a non-CPU tensor is a programmer error → CHECK; reserved dtypes
// surface as Unimplemented from Tensor::empty. Since M2-T05, Tensor::empty
// can allocate on CUDA devices — but these ops still CHECK is_cpu(), so a
// factory asked for a CUDA tensor allocates and then aborts in the fill (GPU
// fills arrive with the M2-T08 kernels; pass CUDA devices to Tensor::empty
// directly until then).
//
// Determinism contract for the seeded fills (load-bearing for golden tests):
// the engine is std::mt19937_64 (fully specified by the C++ standard), and
// the uniform/normal transforms are implemented here rather than with
// std::uniform_real_distribution / std::normal_distribution, whose output
// sequences are implementation-defined and differ between libstdc++ and
// libc++. Given the same seed, a fill produces the same values on every
// platform (the normal fill modulo sub-ulp libm variation; see fill_normal).

namespace engine::tensor::ops {

// --- Factories ---
//
// All factories allocate a fresh contiguous row-major tensor via
// Tensor::empty, so its policies apply unchanged: reserved dtypes (kInt4,
// kFP8E4M3) → Unimplemented, allocation failure → kOutOfMemory, null
// allocator → the process default for `device`. The subsequent fill CHECKs
// is_cpu() (file comment above), so only CPU devices are usable here.

// Tensor of `dtype` filled with zeros (false for kBool).
[[nodiscard]] core::StatusOr<Tensor> zeros(
    const Shape& shape, DataType dtype, Device device = Device::Cpu(),
    memory::Allocator* allocator = nullptr);

// Tensor of `dtype` filled with ones (true for kBool).
[[nodiscard]] core::StatusOr<Tensor> ones(
    const Shape& shape, DataType dtype, Device device = Device::Cpu(),
    memory::Allocator* allocator = nullptr);

// Tensor of `dtype` filled with `value`. Floating dtypes narrow `value`
// double → float → dtype under the M1-T07 rounding policy (±inf/NaN pass
// through). For integer dtypes `value` must be integral and in range, and
// for kBool exactly 0 or 1 — otherwise InvalidArgument (values often come
// from configs/tests, not literals).
[[nodiscard]] core::StatusOr<Tensor> full(
    const Shape& shape, DataType dtype, double value,
    Device device = Device::Cpu(), memory::Allocator* allocator = nullptr);

// 1-d tensor holding start, start + step, ... up to but excluding `end`
// (empty when the range is empty; `step` may be negative; step == 0, an
// end - start that overflows int64, or an element count that does →
// InvalidArgument). Values are computed in int64 arithmetic; for integer
// dtypes every value must be representable (checked via the extremes) →
// otherwise InvalidArgument; floating dtypes narrow per M1-T07 (a value
// beyond fp16 range becomes ±inf, matching NumPy).
[[nodiscard]] core::StatusOr<Tensor> arange(
    std::int64_t start, std::int64_t end, std::int64_t step = 1,
    DataType dtype = DataType::kInt64, Device device = Device::Cpu(),
    memory::Allocator* allocator = nullptr);

// --- Seeded random fills ---
//
// Fill an EXISTING tensor's elements in logical row-major order (the order
// is part of the determinism contract — it makes fills on strided views
// well-defined). The tensor is mutated through a `const Tensor&` because
// const-ness is shallow by design (§8): the handle is untouched, the bytes
// change. Floating dtypes only (InvalidArgument otherwise); non-contiguous
// views are supported and touch only the view's elements. Undefined handle
// or non-CPU tensor → CHECK.
//
// Values are generated in double and narrowed once per element (M1-T07
// policy). Both fills consume engine outputs in a fixed, documented order,
// so the same seed yields the same tensor contents every time.

// Uniform on [low, high): u = (mt19937_64() >> 11) * 2^-53 ∈ [0, 1), value =
// low + u * (high - low) — pure arithmetic, bit-identical on every platform.
// (Narrowing to the target dtype may round a near-high value up to `high`
// itself.) Requires finite low <= high with finite high - low.
[[nodiscard]] core::Status fill_uniform(const Tensor& tensor, double low,
                                        double high, std::uint64_t seed);

// Normal(mean, stddev) via Box–Muller on consecutive engine outputs: per
// pair of elements, u1 = ((mt19937_64() >> 11) + 1) * 2^-53 ∈ (0, 1], then
// u2 as in fill_uniform, r = sqrt(-2 ln u1), z0 = r cos(2π u2),
// z1 = r sin(2π u2); an odd element count discards the final z1. Requires
// finite mean and finite stddev >= 0. Caveat: log/cos/sin come from libm,
// which is not required to be correctly rounded — sequences are identical
// given a seed on a given platform, and cross-platform up to sub-ulp libm
// differences (absorbed in practice by narrowing to <= 32-bit dtypes; the
// golden tests are validated on both CI platforms and macOS).
[[nodiscard]] core::Status fill_normal(const Tensor& tensor, double mean,
                                       double stddev, std::uint64_t seed);

// --- Copy & cast ---

// Deep element-wise copy of src's values into dst, by logical index. Shape
// and dtype must match exactly — dtype conversion is only ever the named
// `cast` below, never implicit (design §1) — otherwise InvalidArgument.
// Either side may be a non-contiguous view; only dst's view elements are
// written. dst is mutated through a `const Tensor&` because const-ness is
// shallow (§8). Both-contiguous pairs take a memcpy fast path. dst and src
// aliasing overlapping bytes of one buffer is undefined behavior (the
// memcpy rule), except that a dst identical to src is a well-defined no-op.
// Undefined handle or non-CPU tensor → CHECK.
[[nodiscard]] core::Status copy(const Tensor& dst, const Tensor& src);

// Device-aware copy, async on `stream` where the memory allows it (M2-T07;
// design: docs/design/cuda-backend.md §8). Shape and dtype must match
// exactly, as above → InvalidArgument otherwise. Additionally:
//  - Both tensors must be contiguous → InvalidArgument otherwise (strided
//    device copy needs a kernel, deferred until a consumer exists; the
//    two-argument overload still handles strided CPU views).
//  - Supported placements: H2D, D2H, and same-device D2D. A CUDA pair on
//    different devices → Unimplemented (cross-device copy arrives with the
//    distributed milestone). CPU↔CPU delegates to the two-argument overload,
//    `stream` ignored — so this overload works on every build for host
//    pairs; on CPU-only builds a copy involving a CUDA-tagged tensor is
//    Unimplemented (unreachable through supported factories).
//  - A null StreamHandle means the engine default stream of the
//    destination-relevant device (H2D: dst's, D2H: src's, D2D: the common
//    device — design §6.3).
//  - The call ENQUEUES and returns: completion must be observed per design
//    §6.4 (event or stream synchronize) before dst is consumed outside
//    `stream`. Whether host-side overlap is real depends on pinnedness
//    (design §8.2, memory/pinned_allocator.h); the correctness discipline
//    is identical either way.
// A dst identical to src is a well-defined no-op; other overlapping aliases
// are undefined behavior (as above). Undefined handle → CHECK.
[[nodiscard]] core::Status copy(const Tensor& dst, const Tensor& src,
                                cuda::StreamHandle stream);

// Fresh contiguous row-major tensor of `dtype` on src's device holding
// src's values converted element-wise; src may be a strided view. Always
// allocates — a same-dtype cast is a deep copy, never the source handle —
// so ownership is predictable. Supported dtypes on both sides: the floating
// (kFloat32/kFloat16/kBFloat16) and integer (kInt8/kUInt8/kInt32/kInt64)
// kinds; kBool is unsupported in either direction → InvalidArgument. A
// reserved target dtype → Unimplemented (via Tensor::empty, which also
// supplies the OOM/overflow policies).
//
// Conversion policy (design §7.1, decided in M1-T09):
//  - Floating targets: the source widens to double (exact for every
//    supported source except int64 beyond 2^53, which rounds to nearest)
//    and narrows per M1-T07 (double → float → half), the same path the
//    factories and fills use — so fp32→fp16/bf16 matches the half.h
//    constructors bit-exactly, out-of-range values become ±inf, and NaN
//    passes through. Caveat: quiet NaNs only — the double/float hops are
//    hardware conversions that quiet a signaling NaN, unlike half.h's pure
//    bit conversions (irrelevant to inference; noted for exactness).
//  - Integer targets: floating sources truncate toward zero (C semantics);
//    a NaN, ±inf, or out-of-range result — including integer→integer
//    narrowing — is InvalidArgument naming the first offending value and
//    its logical index. Loud beats silent wrap in the correctness oracle.
// Undefined handle or non-CPU tensor → CHECK.
[[nodiscard]] core::StatusOr<Tensor> cast(
    const Tensor& src, DataType dtype, memory::Allocator* allocator = nullptr);

// --- allclose ---

struct AllCloseTolerance {
  double rtol;
  double atol;
};

// Default tolerances used when allclose's rtol/atol are not given, so a test
// accepting the default is still stating one (design §10). NumPy/PyTorch
// values: kFloat32 {1e-5, 1e-8}; kFloat16 {1e-3, 1e-5}; kBFloat16
// {1.6e-2, 1e-5}; integer dtypes and kBool compare exactly {0, 0}. Reserved
// dtypes (no comparable host values until their milestone) → CHECK.
[[nodiscard]] constexpr AllCloseTolerance default_allclose_tolerance(
    DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return {.rtol = 1e-5, .atol = 1e-8};
    case DataType::kFloat16:
      return {.rtol = 1e-3, .atol = 1e-5};
    case DataType::kBFloat16:
      return {.rtol = 1.6e-2, .atol = 1e-5};
    case DataType::kInt8:
    case DataType::kUInt8:
    case DataType::kInt32:
    case DataType::kInt64:
    case DataType::kBool:
      return {.rtol = 0.0, .atol = 0.0};
    case DataType::kFP8E4M3:
    case DataType::kInt4:
      break;
  }
  CHECK(false, "no default tolerance for reserved dtype {}",
        tensor::to_string(dtype));
}

// The verdict plus the max-abs-diff report the ticket asks for. Values are
// widened to double for reporting; the worst_* fields describe the mismatch
// with the largest absolute difference (a NaN-involved mismatch counts as
// maximally bad) and are meaningful only when allclose is false.
struct AllCloseResult {
  bool allclose = true;
  std::int64_t numel = 0;
  std::int64_t mismatch_count = 0;
  // Max |a - b| over all element pairs whose difference is not NaN (same-
  // signed infinities contribute 0; opposite-signed ones +inf).
  double max_abs_diff = 0.0;
  DimVector worst_index;  // multi-dimensional index of the worst mismatch
  double worst_a = 0.0;
  double worst_b = 0.0;
  double rtol = 0.0;  // tolerances actually applied
  double atol = 0.0;

  // One-line human-readable report; on failure it names the worst mismatch's
  // index and both values.
  [[nodiscard]] std::string Summary() const;
};

// Element-wise closeness: |a - b| <= atol + rtol * |b| (the NumPy criterion,
// evaluated in double). NaN is never close to anything; infinities are close
// only to the same-signed infinity; integer dtypes and kBool compare
// exactly. Empty tensors are close. rtol/atol default per
// default_allclose_tolerance(a.dtype()); negative or NaN tolerances, shape
// mismatch, or dtype mismatch → InvalidArgument. Strided views are compared
// element-wise by logical index. Undefined or non-CPU tensors → CHECK.
[[nodiscard]] core::StatusOr<AllCloseResult> allclose(
    const Tensor& a, const Tensor& b, std::optional<double> rtol = std::nullopt,
    std::optional<double> atol = std::nullopt);

// --- Debug printing ---

// Human-readable rendering for debugging and test failure messages: a
// one-line header followed by nested NumPy-style rows, e.g.
//
//   Tensor(shape=[2, 3], dtype=float32, device=cpu)
//   [[0, 1, 2],
//    [3, 4, 5]]
//
// Dims longer than 2 * edge_items summarize as "first edge_items, ...,
// last edge_items" (edge_items must be positive: CHECK). Strided views
// print their logical values. An undefined handle prints
// "Tensor(undefined)" — this is a debugging aid, not a place to abort.
// Non-CPU tensors → CHECK (§8). Never use on a hot path.
[[nodiscard]] std::string to_string(const Tensor& tensor,
                                    std::int64_t edge_items = 3);

}  // namespace engine::tensor::ops
