# Tensor library & device model

**Milestone:** M1 (design doc: M1-T01; implementation: M1-T02 … M1-T09)
**Governs:** `src/tensor/`, `src/memory/`
**Cites:** ADR-001 (language/toolchain), ADR-002 (module boundaries),
ADR-003 (error handling)

This is the working contract for the tensor subsystem. Implementation tickets
must conform to it; if implementation reveals a design flaw, this doc is
updated in the same change with a note on what changed and why
(`docs/design/README.md`).

---

## 1. Scope & non-goals

The tensor library is exactly the data-structure layer an inference engine
needs and nothing more: dense n-dimensional arrays with explicit memory
management, explicit device placement, and cheap aliasing views. It is the
vocabulary consumed by every module above layer 1 — model loading, KV cache,
kernels, sampling — so its ownership and view semantics must be unambiguous.

**Non-goals (permanent, by design):**

- **No autograd.** No gradient tracking, no tape, no `requires_grad`.
- **No lazy evaluation or graph tracing.** Every operation executes eagerly;
  scheduling and fusion happen in `engine`/`kernels`, not here.
- **No broadcasting.** The tensor type itself defines none. Individual ops
  (M1-T08+, kernels) may define their own limited shape rules (e.g. a
  bias-add accepting a 1-d vector), documented per-op; nothing NumPy-general.
- **No implicit anything.** No implicit device transfers, no implicit dtype
  promotion, no implicit copies to make a view contiguous. All movement and
  conversion is a named function call (`copy`, `cast`, M1-T09; host↔device
  transfers in M2).
- **No sparse, ragged, or nested tensors.** Dense strided layout only.
  (Paged KV storage is built *from* dense tensors in `kvcache`, not modeled
  here.)
- **No serialization.** Weight I/O belongs to `model` (safetensors, M4).
- **No operator overloading arithmetic.** `a + b` hides allocation and
  dispatch decisions an inference engine wants explicit; ops are free
  functions in `ops.h`.

---

## 2. Module layout & layering

Types and files (all paths per ROADMAP tickets):

| File | Module | Contents | Ticket |
|---|---|---|---|
| `src/tensor/dtype.h` | tensor | `DataType`, `DTypeTraits` | M1-T02 |
| `src/tensor/shape.h` | tensor | `Shape`, strides helpers | M1-T03 |
| `src/tensor/device.h` | tensor | `DeviceType`, `Device` | M1-T04 |
| `src/memory/allocator.h` | memory | `Allocator`, `Buffer`, `CpuAllocator` | M1-T05 |
| `src/tensor/half.h` | tensor | `float16`, `bfloat16` host types | M1-T07 |
| `src/tensor/tensor.h` | tensor | `Tensor` | M1-T06 |
| `src/tensor/ops.h` | tensor | factories, `allclose`, `copy`, `cast` | M1-T08/09 |

### 2.1 The `tensor_base` split

There is a dependency knot: `Tensor` holds a `Buffer`, so **tensor links
memory** (the layer-1 edge ADR-002 lists). But `Buffer` records the `Device`
it lives on, and `Device` is defined in `src/tensor/device.h` — so memory
needs three *header-only value types* (`dtype.h`, `shape.h`, `device.h`) that
live in the tensor directory.

Resolution: the tensor module builds **two targets**:

- `engine::tensor_base` — INTERFACE (header-only) target exporting exactly
  `dtype.h`, `shape.h`, `device.h`, `half.h`. These headers include nothing
  from `memory` or from the rest of `tensor`; they may include `core`.
- `engine::tensor` — the compiled library (`tensor.h`, `ops.h`, their
  `.cpp`). Links `engine::tensor_base`, `engine::memory`, `engine::core`.

`engine::memory` links `engine::tensor_base` and `engine::core`. There is no
link cycle (`tensor_base` is INTERFACE and depends only on core) and no
include cycle (the base headers are leaves). ADR-002 has been amended to
record this edge; the amendment lives in the ADR, this doc just restates it.

File locations do not change: the base headers stay in `src/tensor/` and are
included as `#include "tensor/device.h"` per ADR-002's include convention.

---

## 3. Data types (`dtype.h`, M1-T02)

```cpp
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
  // Reserved for later milestones; representable now, not allocatable until
  // their milestone lands (factories return Unimplemented).
  kFP8E4M3 = 8,  // FP8 KV cache (M13)
  kInt4 = 9,     // AWQ/GPTQ weight quantization (M12)
};

// Element width in bits. Valid for every DataType, including sub-byte.
[[nodiscard]] constexpr int itemsize_bits(DataType dtype);

// Element width in bytes. CHECK-fails for sub-byte dtypes (kInt4) — call
// sites that can meet kInt4 must reason in bits or rows, not elements.
[[nodiscard]] constexpr int itemsize(DataType dtype);

[[nodiscard]] std::string_view to_string(DataType dtype);          // "float16"
[[nodiscard]] core::StatusOr<DataType> from_string(std::string_view name);

[[nodiscard]] constexpr bool is_floating_point(DataType dtype);
[[nodiscard]] constexpr bool is_integral(DataType dtype);
[[nodiscard]] constexpr bool is_sub_byte(DataType dtype);          // kInt4

// Compile-time C++ type ↔ enum mapping. Specialized for float, int8_t, …,
// and (from M1-T07) float16/bfloat16. DTypeTraits<T>::value is the enum;
// dtype_of<T> is shorthand. Primary template is undefined: using an
// unmapped type is a compile error.
template <typename T> struct DTypeTraits;
template <typename T> inline constexpr DataType dtype_of = DTypeTraits<T>::value;

}  // namespace engine::tensor
```

**Decisions.**

- **Names** are lowercase HuggingFace-style strings (`"float32"`,
  `"bfloat16"`, `"int4"`, `"fp8_e4m3"`) so `model` can compare against
  `config.json` / safetensors dtype strings directly. `from_string` returns
  `InvalidArgument` on unknown names (data-driven input, never CHECK).
- **`kBool`** is 1 byte (matches NumPy/PyTorch and fixture data).
- **Sub-byte rule (`kInt4`):** two elements per byte, the element with the
  **lower index in the low nibble** (little-nibble order, matching AWQ/GPTQ
  repacking conventions we will target in M12). An int4 tensor's innermost
  dimension must be contiguous and even; strides remain element-denominated
  (§4) and only the innermost dim may be sub-byte-packed. Until M12 the type
  is name/size-mapped only: `Tensor::empty` rejects it with `Unimplemented`.
- **`kFP8E4M3`** is 1 byte; no host arithmetic type is provided until M13
  needs one.

---

## 4. Shape & strides (`shape.h`, M1-T03)

```cpp
namespace engine::tensor {

// Fixed capacity: transformer inference needs ≤ 5 dims in practice
// (e.g. [batch, heads, seq, seq] attention weights); 8 leaves headroom
// without heap allocation.
inline constexpr int kMaxRank = 8;

// Small inline vector of non-negative int64_t dims. Value type: copyable,
// equality-comparable, fmt-formattable as "[2, 3, 4]". Rank 0 (scalar) is
// valid and has numel() == 1. Dims of 0 are valid and give numel() == 0.
class Shape {
 public:
  Shape() = default;                          // rank 0
  Shape(std::initializer_list<int64_t> dims); // CHECKs rank ≤ kMaxRank, dims ≥ 0
  static core::StatusOr<Shape> FromDims(std::span<const int64_t> dims);

  [[nodiscard]] int rank() const;
  [[nodiscard]] int64_t dim(int i) const;     // CHECKs 0 ≤ i < rank()
  [[nodiscard]] int64_t numel() const;        // cached at construction
  [[nodiscard]] std::span<const int64_t> dims() const;
  friend bool operator==(const Shape&, const Shape&) = default;
};

// The shared fixed-capacity inline container (int64_t, capacity kMaxRank,
// append-only growth): Shape's dims and Strides both use it. Strides are
// measured in ELEMENTS.
class DimVector { /* size(), operator[], push_back(), ==, fmt-formattable */ };
using Strides = DimVector;

[[nodiscard]] Strides RowMajorStrides(const Shape& shape);
[[nodiscard]] bool IsContiguous(const Shape& shape, const Strides& strides);

}  // namespace engine::tensor
```

**Decisions.**

- **Strides are element-denominated**, not byte-denominated (PyTorch
  convention). Rationale: stride math stays independent of dtype, and
  `view_as_dtype` / `cast` don't rescale strides. Byte offsets are computed
  at access time as `element_offset * itemsize`. For sub-byte dtypes this is
  exact because only the innermost, contiguous dim may be packed (§3).
- **Overflow policy:** the constructor from a literal list CHECKs (inline
  literals are programmer-authored); `FromDims` returns `InvalidArgument`
  for negative dims, rank > `kMaxRank`, or a dim product that overflows
  `int64_t` — this is the entry point for shapes derived from files or
  requests. `numel()` is computed once, at construction, under the same
  overflow check, so it is a cheap cached read afterwards. Refined in
  M1-T03: overflow is judged on the product of the *non-zero* dims — that
  is the true numel when no dim is 0 (a 0 dim makes numel 0, which cannot
  overflow), and it bounds every row-major stride, so `RowMajorStrides`
  cannot overflow on a valid shape.
- **Contiguity** means: strides equal `RowMajorStrides(shape)` with the
  convention that dims of size 1 (and any dim, when `numel() == 0`) impose
  no constraint on their stride. Golden-case tests in M1-T03 pin this down.
- **`RowMajorStrides` treats size-0 dims as size 1** (PyTorch's
  convention, decided in M1-T03): strides stay positive, and the result is
  contiguous under the definition above.
- `Strides` deliberately reuses the same inline container; no negative
  strides are supported anywhere in the engine (no flip views — a non-goal
  that keeps every kernel's index math simple).

---

## 5. Device (`device.h`, M1-T04)

```cpp
namespace engine::tensor {

enum class DeviceType : std::uint8_t { kCPU = 0, kCUDA = 1 };

// Plain value type. Backend-agnostic: this header never includes CUDA
// headers; a CUDA Device is *representable* on any build, but allocating on
// one returns Unimplemented until M2 (and Unavailable-style failures after,
// on non-CUDA builds).
struct Device {
  DeviceType type = DeviceType::kCPU;
  int index = 0;  // Always 0 for kCPU; ordinal for kCUDA.

  [[nodiscard]] static constexpr Device Cpu() { return {DeviceType::kCPU, 0}; }
  [[nodiscard]] static constexpr Device Cuda(int index) {
    return {DeviceType::kCUDA, index};
  }
  // "cpu", "cuda:0". Parse accepts exactly these forms (plus bare "cuda" ==
  // "cuda:0"); anything else, including "cpu:1", is InvalidArgument.
  [[nodiscard]] static core::StatusOr<Device> Parse(std::string_view spec);
  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] bool is_cpu() const;
  [[nodiscard]] bool is_cuda() const;
  friend bool operator==(const Device&, const Device&) = default;
};

}  // namespace engine::tensor
```

The device *registry* (how many CUDA devices exist, their properties) is
M2's problem (`src/cuda/`); `Device` itself is just an address. Nothing in
layer 1 validates `index` against hardware at construction time — validation
happens where a device is *used* (allocation, M2 transfer), where a `Status`
can be returned with real context.

---

## 6. Buffer & Allocator (`allocator.h`, M1-T05)

```cpp
namespace engine::memory {

// Move-only owning handle to one contiguous allocation. Destroying an
// engaged Buffer invokes its deleter exactly once — including for zero-size
// buffers. A moved-from Buffer is disengaged: destruction is a no-op.
class Buffer {
 public:
  using Deleter = std::function<void(void*)>;

  Buffer() = default;  // disengaged
  Buffer(void* data, std::size_t size_bytes, tensor::Device device,
         Deleter deleter);
  Buffer(Buffer&&) noexcept;             // transfers ownership
  Buffer& operator=(Buffer&&) noexcept;  // destroys current, then transfers
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer();                             // runs deleter if engaged

  [[nodiscard]] void* data() const;      // nullptr iff size_bytes() == 0
  [[nodiscard]] std::size_t size_bytes() const;
  [[nodiscard]] tensor::Device device() const;
};

class Allocator {
 public:
  virtual ~Allocator() = default;

  // `alignment` must be a power of two (CHECK — a caller-authored constant).
  // bytes == 0 is valid and returns an engaged Buffer with data() == nullptr
  // whose deleter still runs exactly once. Failure (e.g. OOM) is
  // kOutOfMemory — allocation never throws (ADR-003).
  [[nodiscard]] virtual core::StatusOr<Buffer> Allocate(
      std::size_t bytes, std::size_t alignment) = 0;

  [[nodiscard]] virtual tensor::Device device() const = 0;
};

// Aligned host allocator; default alignment 64 bytes (cache line; also
// satisfies every SIMD width we care about on x86-64 hosts).
class CpuAllocator final : public Allocator {
 public:
  explicit CpuAllocator(std::size_t default_alignment = 64);
  // ...
};

// Process-wide CpuAllocator instance used when Tensor factories are not
// given an explicit allocator. Never destroyed (function-local static).
[[nodiscard]] Allocator* DefaultCpuAllocator();

}  // namespace engine::memory
```

**Ownership decisions.**

- **Deleters are self-contained.** A `Buffer` never holds a pointer back to
  its `Allocator`; the deleter closure captures whatever it needs to free
  the memory. Consequence: a `Buffer` may safely outlive the allocator that
  made it (unless the deleter itself dereferences the allocator — which the
  M2 caching pool will do, and will document as a lifetime requirement
  *there*; the plain `CpuAllocator`'s deleter is just `std::free`-shaped).
- **`std::function` as the deleter type.** It costs one indirect call at
  destruction — irrelevant, since buffers in this engine are long-lived
  (weights, KV pool slabs) and never created/destroyed on the token hot
  path. Flexibility (capturing pool state in M2) wins over a raw
  function-pointer-plus-context pair.
- `Buffer` does not expose a `release()`; nothing in the roadmap needs to
  disassemble ownership, and not having it keeps "deleter runs exactly once"
  trivially true.

**Allocator contract.**

- Implementations must be **thread-safe**: `Allocate` may be called
  concurrently from any thread. (`CpuAllocator` is trivially so;
  the M2 caching pool must lock.)
- Returned memory is uninitialized. There is no `Deallocate` on the
  interface — deallocation is the Buffer deleter's job, which is what makes
  RAII airtight.

---

## 7. Tensor (`tensor.h`, M1-T06)

```cpp
namespace engine::tensor {

// A view onto shared storage:
//   shared_ptr<memory::Buffer> + byte offset + Shape + Strides + DataType
//   (+ Device, denormalized from the buffer for cheap access).
//
// Copying a Tensor is CHEAP and SHALLOW: it copies the handle, not the
// data. Copies and views alias the same storage — a write through one is
// visible through all. Deep copies are explicit: ops::copy (M1-T09).
class Tensor {
 public:
  Tensor() = default;  // empty handle: no storage, rank-0, kFloat32, cpu

  // Allocates uninitialized, contiguous, row-major storage. allocator may
  // be null → the process default for `device` (DefaultCpuAllocator for
  // cpu; CUDA devices return Unimplemented until M2). Reserved dtypes
  // (kInt4, kFP8E4M3) return Unimplemented until their milestone.
  [[nodiscard]] static core::StatusOr<Tensor> empty(
      Shape shape, DataType dtype, Device device,
      memory::Allocator* allocator = nullptr);

  // Metadata.
  [[nodiscard]] const Shape& shape() const;
  [[nodiscard]] const Strides& strides() const;
  [[nodiscard]] DataType dtype() const;
  [[nodiscard]] Device device() const;
  [[nodiscard]] int64_t numel() const;
  [[nodiscard]] bool is_contiguous() const;
  [[nodiscard]] bool defined() const;   // false for the default-constructed handle
  [[nodiscard]] std::size_t byte_offset() const;

  // Views — new Tensors sharing this buffer. Data-dependent misuse (bad
  // ranges, non-contiguous reshape) is recoverable: InvalidArgument.
  [[nodiscard]] core::StatusOr<Tensor> slice(int dim, int64_t start,
                                             int64_t end) const;
  [[nodiscard]] core::StatusOr<Tensor> reshape(Shape new_shape) const;
  [[nodiscard]] core::StatusOr<Tensor> view_as_dtype(DataType new_dtype) const;

  // Raw access. data() is the buffer base plus byte_offset.
  [[nodiscard]] void* data() const;
  // Typed access: CHECKs dtype_of<T> == dtype(). Programmer error by
  // definition — the call site names T statically.
  template <typename T> [[nodiscard]] T* data_ptr() const;

  // CPU-only element accessor for tests and debugging. CHECKs: cpu device,
  // dtype match, rank match, indices in range. Applies strides — works on
  // non-contiguous views. Never use on a hot path.
  template <typename T> [[nodiscard]] T item(std::span<const int64_t> indices) const;
};

}  // namespace engine::tensor
```

**View semantics (the load-bearing part).**

- `slice(dim, start, end)`: half-open `[start, end)` along `dim`; result
  shares storage with `byte_offset += start * strides[dim] * itemsize`,
  same strides, `shape[dim] = end - start`. Slicing an outer dim keeps
  contiguity; slicing an inner dim produces a legitimate non-contiguous
  view. No negative indices, no step — nothing in the engine needs them.
- `reshape(new_shape)`: contiguous tensors only, `numel` must match;
  returns a view (never copies — an implicit copy here is exactly the kind
  of hidden cost §1 bans). Non-contiguous input → `InvalidArgument`
  telling the caller to `copy` first.
- `view_as_dtype(new_dtype)`: same `itemsize` only (e.g. `kInt8` ↔
  `kUInt8`, `kFloat16` ↔ `kBFloat16` bit reinterpretation for fixtures).
  Same shape/strides. Size-changing reinterpretation is out of scope until
  a ticket needs it — at which point this doc gets amended, not silently
  extended.
- Views of views compose through `byte_offset`; every view holds the
  `shared_ptr<Buffer>`, so the base allocation lives until the last view
  dies. There is no distinction between "owning" and "view" tensors — every
  Tensor is a shared handle, uniformly.

**Error-handling boundary** (per ADR-003; this table is the contract for
M1-T02 … M1-T09, so individual tickets don't decide ad hoc):

| Failure | Response | Why |
|---|---|---|
| Allocation failure, unsupported device/dtype in `empty` | `Status` (`kOutOfMemory` / `kUnimplemented`) | environment- and data-dependent |
| Bad range/shape in `slice`/`reshape`/`view_as_dtype` | `InvalidArgument` | shapes flow from configs/requests |
| Shape overflow via `Shape::FromDims` | `InvalidArgument` | file/request input |
| `from_string`/`Parse` of unknown dtype/device | `InvalidArgument` | file/request input |
| `data_ptr<T>` dtype mismatch | `CHECK` | T is written in source; wrong T is a bug |
| Host element access (`item`) on non-CPU tensor | `CHECK` | callers must know placement; §8 |
| Out-of-range `dim(i)`, invalid `alignment`, `item` misuse | `CHECK` | call-site-authored constants |

**Copy/move semantics (acceptance criterion, stated precisely).**

- `Tensor` is copyable and movable; both are `noexcept` except copy
  (shared_ptr copy — still cheap, one atomic increment). Copy is shallow:
  the copy aliases the same `Buffer`, same offset/shape/strides/dtype.
- Moved-from `Tensor` is the empty handle (`defined() == false`); the only
  valid operations on it are destruction, assignment, and `defined()`.
- Equality is **not** defined on `Tensor` (would it mean same handle or
  same values? — ambiguous, so neither; value comparison is
  `ops::allclose`, M1-T08).

---

## 8. Host/device data-access rules

1. **Placement is explicit and static.** A Tensor's `device()` is fixed at
   creation. Nothing migrates data implicitly.
2. **Host code may dereference only CPU tensors.** `data()` /
   `data_ptr<T>()` on a CUDA tensor return the *device* pointer — legal to
   hold and pass to kernels/cuBLAS (M2+), illegal to dereference on the
   host. The typed conveniences that imply host dereference (`item`,
   printing, `allclose`) CHECK `is_cpu()`.
3. **All transfers are named calls** on the M2 copy API, which will also
   define the synchronization story (streams, events). Until M2, every
   CUDA-device operation returns `Unimplemented` — the types are ready, the
   backend is not.
4. **Const-ness is shallow.** `const Tensor&` means the *handle* (metadata)
   is immutable; the pointed-to data is not (`data()` is const and returns
   a mutable pointer, like `std::span`). Deep immutability is not modeled —
   weights are "immutable by convention after load", enforced by module
   discipline, not the type system. (Modeling it would bifurcate every API
   into const/mutable flavors for no inference-engine benefit.)

---

## 9. Thread-safety guarantees

Stated once here; every class's header comment restates its own line.

- **`Tensor` follows the `shared_ptr` model.** Distinct `Tensor` handles —
  including copies and views of the same storage — may be used concurrently
  from different threads without synchronization (the control block is
  atomic). Concurrent access to a *single* `Tensor` object is safe only if
  all access is const.
- **Data races on the underlying bytes are the caller's problem.** The
  tensor layer never locks data. Concurrent reads are fine; any writer
  concurrent with other access requires external synchronization (in
  practice: the engine's phase structure — load, then read-only — provides
  it).
- **`Allocator` implementations are thread-safe** (§6). `Buffer` is not a
  concurrent type (it is move-only and owned by one place at a time; the
  `shared_ptr` wrapper is what gets shared).
- Nothing in layer 1 spawns threads.

---

## 10. Testing strategy

Per the project's correctness methodology, everything in M1 is CPU-only and
must pass on CPU-only CI. Mapping of guarantees → tests:

| Area | Tests (ticket) |
|---|---|
| dtype sizes, name round-trips, traits, sub-byte reporting | every enum value exhaustively (M1-T02) |
| shape/stride goldens, 0-d…5-d, overflow, contiguity of sliced strides | (M1-T03) |
| device parse/format/equality, invalid specs | (M1-T04) |
| alignment honored, zero-size behavior, move semantics, deleter-exactly-once via counting test allocator | (M1-T05) |
| views alias (write-through visible), slice shape/stride/offset goldens, reshape rejects non-contiguous, `data_ptr` CHECK on mismatch (death test), cheap-copy semantics | (M1-T06) |
| fp16/bf16 golden bit patterns (rounding boundaries, ±inf, NaN, subnormals), round-trip property test | (M1-T07) |
| factories per dtype, seeded-fill determinism (exact values), `allclose` worst-mismatch report | (M1-T08) |
| strided copies, all cast pairs, fp32→fp16 rounding matches M1-T07 exactly | (M1-T09) |

Numerical tests state tolerances explicitly; `allclose` carries per-dtype
defaults (defined in M1-T08) precisely so tests that accept the default are
still stating one. CHECK-death tests use gtest's `EXPECT_DEATH` and follow
the harness conventions in `tests/README.md`.

---

## 11. Deferred (known, intentionally not designed here)

- **CUDA allocators, caching pool, streams** — `docs/design/cuda-backend.md`
  (M2-T01). This doc constrains it only via the `Allocator`/`Buffer`
  interface and the stated thread-safety contract.
- **Pinned host memory** — an `Allocator` implementation in M2; no interface
  change anticipated.
- **kInt4 packed tensors** — representable dtype now; container semantics
  (packing, quant scales/zeros live *beside* the tensor in `quant`, not
  inside it) designed in M12's doc.
- **fp8** — M13.
