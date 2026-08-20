# CUDA backend (RETIRED)

> **Retired 2026-08-07 (M3-T01/T02, ADR-004).** The engine pivoted to
> CPU-first; the code this document governed was removed in M3-T02 (it
> survives in git history through the `post-M2 hardening` commits). Kept as
> the starting point for any future GPU backend. Nothing below is updated
> for the current tree — the device-agnostic caching pool allocator (§7.3)
> is the one piece that lives on, in `src/memory/`.

**Milestone:** M2 (design doc: M2-T01; implementation: M2-T02 … M2-T09)
**Governs:** `src/cuda/`, `src/kernels/`, and the CUDA-facing parts of
`src/memory/` (device/pinned allocators, caching pool)
**Cites:** ADR-001 (language/toolchain), ADR-002 (module boundaries),
ADR-003 (error handling), `docs/design/tensor.md` (Buffer/Allocator/Tensor
contracts this backend plugs into)

This is the working contract for the CUDA backend. Implementation tickets
must conform to it; if implementation reveals a design flaw, this doc is
updated in the same change with a note on what changed and why
(`docs/design/README.md`).

The three questions M2-T01 must answer, and where they are answered:

1. *How does a CUDA error inside a kernel surface to the caller?* — §4.3.
2. *Who synchronizes and when?* — §6.4.
3. *How do tests assert kernel correctness?* — §10.2.

---

## 1. Scope & non-goals

M2 brings up the paved road every later GPU ticket drives on: build-system
integration, error handling, device introspection, streams and events,
device-memory allocation (naive, then the caching pool that becomes the
engine's memory-pooling backbone), pinned host memory, host↔device
transfers, and the kernel-launch infrastructure with first trivial
elementwise kernels. After M2, `Tensor` works on GPU and a kernel ticket
consists of: write kernel, write launcher, test against the CPU reference
via the shared fixture.

**Non-goals (deferred, with owners):**

- **cuBLAS.** Handle wrappers and GEMM land with the GPU forward pass
  (M5-T02, `src/cuda/cublas.h`). Nothing in M2 creates a cuBLAS handle.
- **CUDA graphs.** M11-T05. M2's obligation is only to not preclude capture:
  no legacy-default-stream use, no hidden synchronization on hot paths
  (§6.1, §7.4).
- **Multi-GPU / NCCL.** The distributed milestone. M2 APIs take a device
  index where relevant but nothing coordinates across devices.
- **Real kernels.** Attention, norms, sampling, quant kernels arrive in
  their own milestones. M2 ships only `add`, `mul`, `scale`, `cast` —
  chosen to exercise the infrastructure, not to be fast.
- **Kernel performance.** M2 kernels are naive grid-stride loops. No
  vectorized loads, no occupancy tuning, no benchmarks. The correctness
  ladder requires naive-but-correct first; optimized variants
  come with the milestones that need them and must beat a measured baseline.
- **Stream-ordered allocation (`cudaMallocAsync`).** Considered and
  deferred; rationale in §7.5.
- **Unified/managed memory.** Never planned; explicit placement is a tensor
  design invariant (tensor.md §8).

---

## 2. Module layout & layering

Components and files:

| File | Module | Contents | Ticket |
|---|---|---|---|
| `src/cuda/cuda_utils.h/.cpp` | cuda | `device_count()`, `DeviceProperties`, `ScopedSetDevice` (toolkit-free) | M2-T03 |
| `src/cuda/cuda_check.h` | cuda | `CUDA_CHECK`, `CUDA_RETURN_IF_ERROR`, `ToStatus` (internal, includes the toolkit — split from cuda_utils.h in M2-T03, see §2.2) | M2-T03 |
| `src/cuda/stream.h/.cpp` | cuda | `CudaStream`, `CudaEvent`, per-device default stream | M2-T04 |
| `src/memory/cuda_allocator.h/.cpp` | memory | `CudaAllocator` (naive `cudaMalloc`) | M2-T05 |
| `src/memory/caching_allocator.h/.cpp` | memory | `CachingAllocator` + stats | M2-T06 |
| `src/memory/pinned_allocator.h/.cpp` | memory | `PinnedCpuAllocator` (`cudaHostAlloc`) | M2-T07 |
| `src/tensor/ops.h` (+`transfer_detail.h`, `transfer.cpp`/`transfer_stub.cpp`) | tensor | device-aware `copy(dst, src, stream)` overload, `Tensor::to` | M2-T07 |
| `src/kernels/launch.h` | kernels | grid/block helpers, `CUDA_1D_KERNEL_LOOP` | M2-T08 |
| `src/kernels/dispatch.h` | kernels | `DISPATCH_FLOATING_TYPES` | M2-T08 |
| `src/kernels/elementwise.h/.cu` | kernels | `add`, `mul`, `scale`, `cast` launchers + kernels | M2-T08 |
| `tests/common/cuda.h/.cpp` | tests | skip predicate (landed M2-T03 — its acceptance criterion needs it), `CudaTestFixture`, `expect_tensors_close` | M2-T09 |

### 2.1 ADR-002 conformance

Reviewed against the layer-1 edge list. Originally concluded "no amendment
needed"; **M2-T05 implementation proved that wrong** — see the second
bullet. ADR-002 Amendment 2 records the fix.

- `cuda → core` (Status, logging): `cuda` links `engine::core` directly.
  (*Corrected post-M2:* this section — and ADR-002 Amendment 2 — originally
  also claimed a direct `engine::tensor_base` link, but nothing in
  `src/cuda/` uses tensor types and the link never existed in CMake. It
  remains *allowed* per Amendment 1 should `cuda` ever need `Device`.) The
  originally listed `cuda → memory` edge was never used and is removed by
  Amendment 2.
- `memory`'s CUDA allocators: ADR-002 rule 3 already names `memory` as one
  of the four modules that may touch the CUDA toolkit behind
  `ENGINE_ENABLE_CUDA`. *Refined in M2-T05:* toolkit access is not enough —
  `CudaAllocator` also links `cuda` module symbols (`device_count()` per
  §5.2, `ScopedSetDevice`, and `ToStatus`/`CUDA_RETURN_IF_ERROR`, whose
  definitions live in `engine_cuda`), which is a real module dependency
  under ADR-002 rule 2, and both directions at once would be a cycle.
  Resolution: **ADR-002 Amendment 2 flips the edge to `memory → cuda`**
  (PRIVATE in CMake; `memory`'s public headers stay toolkit-free per §2.2).
  The alternative — raw toolkit calls plus a local copy of the §4.2 error
  mapping, the `tensor` seam pattern below — was rejected: it forks the one
  code table this design centralizes, and `memory` lacks `tensor`'s reason
  to avoid the edge (nothing above `memory` gains a `cuda` dependency it
  wouldn't already have). `memory` still must not include
  `tensor.h`/`ops.h`.
- `kernels → cuda, tensor` (listed): launchers consume `Tensor` handles and
  streams. `kernels` never includes anything from layer 2+ (model,
  scheduler, runtime) — the acceptance-criteria review point.
- `tensor`'s M2-T07 transfer code calls into `cuda` (memcpy on a stream).
  This design originally planned toolkit access without a module edge —
  the seam calling only `cudaMemcpyAsync`, never `engine::cuda` symbols.
  *Refined in M2-T07 (implementation): that was wrong, for the same reason
  the M2-T05 bullet above was* — the transfer contract needs cuda-module
  state and code, not just the toolkit: a null `StreamHandle` resolves to
  the process-wide `DefaultStream` (§6.3, §8.1), which cannot be forked
  without breaking FIFO ordering against every other user of the engine
  default stream, and memcpy/sync failures must map through the one §4.2
  code table. **ADR-002 Amendment 3 adds `tensor → cuda`** — PRIVATE, so
  `tensor`'s public headers stay toolkit-free: `ops.h`/`tensor.h` include
  only the toolkit-free `cuda/stream_handle.h`, and the toolkit-touching
  code sits behind an internal seam (`transfer_detail.h`, §8.1) whose CUDA
  implementation (`transfer.cpp`) is compiled only when
  `ENGINE_ENABLE_CUDA=ON` (CPU-only builds compile a stub, §2.2). Rule 3's
  toolkit list gains `tensor` (its transfer TU), recorded in the same
  amendment. The original concern — dragging the CUDA toolkit under every
  `tensor` consumer — does not materialize: a PRIVATE edge propagates no
  toolkit headers, and the link-closure cost was already paid by
  `memory → cuda`.

### 2.2 CPU-only builds (`ENGINE_ENABLE_CUDA=OFF`)

Per ADR-002 rule 3, every module builds and passes its CPU tests with CUDA
off — that is what CI runs. The pattern, uniform across the four
CUDA-touching modules:

- **Headers stay includable everywhere.** No public header of `cuda`,
  `kernels`, `memory`, or `tensor` includes a CUDA toolkit header. Toolkit
  includes live in `.cpp`/`.cu` files only. Public headers use opaque
  handles (§6.3) and plain types, so a CPU-only TU can include
  `cuda/stream.h` and compile. *Refined in M2-T03:* declarations that
  inherently name toolkit types (`ToStatus(cudaError_t, …)` and the error
  macros, which expand `cudaSuccess`/`cudaGetErrorName`) cannot live in a
  toolkit-free header, so they moved to an **internal** header
  (`cuda/cuda_check.h`) that includes `<cuda_runtime.h>` and may be
  included only by CUDA-compiled `.cpp`/`.cu` TUs — the sharing mechanism
  for toolkit-touching code, not a public surface. The rule above governs
  *public* headers; it is unchanged.
- **Sources split, not `#ifdef`-riddled.** Each CUDA-touching target lists
  its `.cu`/CUDA-dependent `.cpp` sources only when `ENGINE_ENABLE_CUDA` is
  ON. With CUDA off, a small `*_stub.cpp` provides the same symbols
  returning `Unimplemented("engine built without CUDA support")` (or, for
  introspection, benign values: `device_count()` returns 0). Behavioral
  `#ifdef`s inside function bodies are avoided; the seam is the source
  list.
- **The compile definition.** Targets that need it get
  `ENGINE_ENABLE_CUDA` as a PUBLIC compile definition from one place
  (`cmake/`), so stray `#ifdef`s (permitted in tests and the rare header
  that genuinely needs one) agree on the spelling.
- **Failure taxonomy on CPU-only builds:** requesting a CUDA resource →
  `Unimplemented` (consistent with M1's placeholder behavior; the message
  distinguishes "built without CUDA" from M1's "not yet implemented").
  On CUDA-enabled builds with no visible device, allocation/use on
  `cuda:N` with `N >= device_count()` → `InvalidArgument` (§5.2).

---

## 3. Build integration (M2-T02)

- **`ENGINE_ENABLE_CUDA`** (already declared, default ON): when ON,
  `enable_language(CUDA)` is attempted via `check_language`; if no toolkit
  is found, configuration **downgrades to OFF with a status message** rather
  than failing — the documented macOS/CI configure command keeps working
  without the explicit `-DENGINE_ENABLE_CUDA=OFF`, and an explicit ON with
  no toolkit still works for laptops. (CI passes OFF explicitly anyway;
  auto-detect is a convenience, not a load-bearing path.)
- **Toolkit floor: CUDA 12.x** (ADR-001). Enforced at configure time with a
  clear error naming the found version.
- **`CMAKE_CUDA_ARCHITECTURES = 80;86;89;90`** — real binaries for each
  (no PTX-only entries): A100 (sm_80), consumer Ampere (sm_86), Ada/L4/L40
  (sm_89), Hopper H100 (sm_90). Newer arches (Blackwell) run sm_90 SASS-
  incompatible → revisit when hardware matters; adding an arch is a
  one-line change and a rebuild. *Refined in M2-T02:* spelled
  `80-real;86-real;89-real;90-real` in CMake — a plain `N` entry embeds
  forward-compat PTX alongside the SASS, which would contradict the
  no-silent-JIT intent above; `-real` is SASS only.
- **Device code is C++20** (`CMAKE_CUDA_STANDARD 20`), matching host code;
  nvcc 12.x supports it.
- **Warnings-as-errors extends to device code.** The `engine_warnings`
  interface target grows CUDA-language generator expressions:
  `-Werror all-warnings` plus `--expt-relaxed-constexpr` if and when
  needed (each expt flag gets a comment justifying it). Host-side flags
  pass through via `-Xcompiler` only where nvcc doesn't accept them
  natively. Third-party headers stay warning-exempt as on the host side.
  *Refined in M2-T02:* the host set forwarded through `-Xcompiler`
  excludes `-Wpedantic` (fires on the `#line`-directive style of
  nvcc-generated host code) and `-Wold-style-cast` (CUDA headers/macros
  expand to C-style casts); host-only TUs keep the full set via a
  `$<COMPILE_LANGUAGE:CXX>` fence.
- **`.cu` files may exist only in** `src/cuda/`, `src/kernels/`,
  `src/memory/` (and `tests/`); `distributed` joins the list in its
  milestone. Greppable, reviewed, matches ADR-002 rule 3.
- **Separable compilation is OFF** until something needs device linking
  (no cross-TU device calls planned; keeps link simple and fast).
- *Refined in M2-T03:* CUDA builds `find_package(CUDAToolkit)` and
  `engine_cuda` links `CUDA::cudart_static` PUBLIC — host-compiled TUs that
  include `<cuda_runtime.h>` (allocators, transfer code, CUDA-only tests)
  need the runtime's include dirs and library, which only nvcc-compiled TUs
  get implicitly. Static runtime: no `libcudart.so` deploy dependency;
  shared is a one-line change if tooling ever needs it.
- **CPU-only exclusion:** with CUDA off, no target lists a `.cu` source and
  the stub sources take their place (§2.2). All existing M0/M1 targets are
  untouched either way.

---

## 4. Error handling (M2-T03)

### 4.1 The two macros

ADR-003's CHECK-vs-Status boundary, specialized for CUDA:

```cpp
// Fatal. For programmer errors: misuse of our own wrappers, impossible
// arguments, corrupted state. Logs file:line, the failing expression, the
// error name and cudaGetErrorString, then aborts (via CHECK machinery).
CUDA_CHECK(cudaSetDevice(idx));

// Recoverable. Evaluates a cudaError_t expression; on failure returns an
// engine::core::Status built by ToStatus (below), with file:line, the
// expression text, the CUDA error name and description embedded in the
// message. Usable in any function returning Status/StatusOr.
CUDA_RETURN_IF_ERROR(cudaMalloc(&p, bytes));
```

Both evaluate their argument exactly once. The choice between them follows
the same rule as everywhere else (ADR-003): data-driven, environment-driven,
or resource-driven failures (OOM, no device, invalid user-supplied device
index) return `Status`; violations of our own invariants CHECK. When in
doubt inside the backend, prefer `CUDA_RETURN_IF_ERROR` — a caller can
always CHECK_OK a Status, but not the reverse.

### 4.2 `cudaError_t → Status` mapping

One function, used by both macros and available directly:

```cpp
namespace engine::cuda {
// Never returns OK for cudaSuccess — callers branch before calling.
[[nodiscard]] core::Status ToStatus(cudaError_t err, std::string_view what);
}
```

| `cudaError_t` | `StatusCode` |
|---|---|
| `cudaErrorMemoryAllocation` | `kResourceExhausted` (device OOM is an operating condition, not a bug — the scheduler will *plan* around it; distinct from host `kOutOfMemory` which M1 uses for `malloc` failure) |
| `cudaErrorInvalidDevice`, `cudaErrorInvalidValue` | `kInvalidArgument` |
| `cudaErrorNoDevice`, `cudaErrorInsufficientDriver`, `cudaErrorNotSupported` | `kUnavailable` (machine/environment problem — matches tensor.md §5's "Unavailable-style failures" forecast) |
| everything else | `kInternal` |

The original CUDA error name (`cudaGetErrorName`) always appears in the
message, so no fidelity is lost by the coarse code mapping.

### 4.3 How a kernel error surfaces to the caller

Kernel launches are asynchronous and return no value; errors appear in two
distinct ways, and the backend handles each explicitly:

1. **Launch-time (synchronous) errors** — bad configuration, no kernel
   image for this arch, too much shared memory. Caught by
   `cudaGetLastError()` **immediately after every launch**. Every host-side
   launcher (§9.2) ends with `CUDA_RETURN_IF_ERROR(cudaGetLastError())`, so
   a bad launch surfaces to the caller as a `Status` from the launcher
   itself, synchronously, with the kernel's name in the message. No launch
   is ever fire-and-forget.
2. **Execution-time (asynchronous) errors** — illegal address, assert
   fired, launch timeout. These are reported by the *next* CUDA API call
   after the failure is detected, typically a synchronization point. All
   our sync entry points — `CudaStream::Synchronize()`,
   `CudaEvent::Synchronize()`, `cudaDeviceSynchronize` in tests, and the
   implicit sync inside blocking transfers — return `Status`, so the error
   surfaces there. **Policy: such errors are terminal for the process.**
   Almost all of them (notably `cudaErrorIllegalAddress`) leave the CUDA
   context poisoned — every subsequent call returns the same sticky error
   and device memory contents are unreliable. The runtime layer (M8+) will
   translate "engine hit a sticky CUDA error" into failing all in-flight
   requests and exiting; M2's obligation is only that the `Status` reaches
   the caller and that the mapping marks stickiness in the message. We do
   not attempt context recovery (`cudaDeviceReset` mid-flight is not a
   correctness tool).

Consequence for callers: code that enqueues work and never synchronizes
cannot observe execution errors — which is why the ownership rule in §6.4
requires every chain of async work to end at an observed sync point (in M2:
tests and transfers; later: the engine step loop).

### 4.4 Logging

The backend uses the M0-T06 logging subsystem with module logger `"cuda"`.
Errors returned as `Status` are *not* also logged at error level by the
backend (the caller decides severity — ADR-003's "no double reporting");
`CUDA_CHECK` failures log fatally by construction. Device discovery at
startup logs one info line per device (name, CC, memory).

---

## 5. Device utilities (M2-T03)

### 5.1 Introspection

```cpp
namespace engine::cuda {

// Number of visible CUDA devices. 0 on machines without a GPU or driver
// AND on CPU-only builds (the stub returns 0) — callers need no build-flag
// awareness; "is GPU work possible" is exactly `device_count() > 0`.
// Never fails: enumeration errors (cudaErrorNoDevice etc.) are treated as
// "0 devices" and logged once at debug level.
[[nodiscard]] int device_count();

struct DeviceProperties {
  std::string name;               // "NVIDIA A100-SXM4-80GB"
  int compute_capability_major;   // 8
  int compute_capability_minor;   // 0
  int multiprocessor_count;
  std::size_t total_memory_bytes;
};

// InvalidArgument if index is out of [0, device_count()). On CPU-only
// builds: Unimplemented per the §2.2 taxonomy (refined in M2-T03 — the
// code distinguishes "built without CUDA" from "no such device").
[[nodiscard]] core::StatusOr<DeviceProperties> GetDeviceProperties(int index);

// RAII: cudaSetDevice(index) on construction, restores the previous
// current device on destruction. Construction CHECKs (a bad index here is
// programmer error — validate data-driven indices before entering a scope).
// Not copyable/movable; scoped use only.
class ScopedSetDevice { … };

}  // namespace engine::cuda
```

`device_count()` memoizes its first enumeration — success *or* failure (a
failed probe memoizes as zero devices; the count does not change
mid-process, and a process whose first enumeration failed stays GPU-less).
It is the foundation of the test skip predicate (§10.1).

### 5.2 Where `Device{kCUDA, i}` is finally validated

Per tensor.md §5, layer 1 constructs CUDA `Device` values without hardware
validation. M2 is where use-sites validate: `CudaAllocator` construction
and every transfer entry point check `index < device_count()` and return
`InvalidArgument` (naming the index and the count) otherwise. No global
registry object — `device_count()` + per-use validation is sufficient
until tensor parallelism needs topology (distributed milestone).

---

## 6. Streams & events (M2-T04)

### 6.1 Stream model

- **All engine streams are explicitly created** with
  `cudaStreamNonBlocking`. The legacy default stream (stream 0) is never
  used for work: it synchronizes with everything, which both hides bugs
  and blocks CUDA-graph capture later (M11). Nothing in the engine relies
  on legacy-stream implicit sync.
- **In M2 there is one *engine default stream* per device** (§6.3) plus
  whatever streams tests create. Multi-stream orchestration (transfer
  overlap, per-request streams) is an engine-milestone concern; the
  wrappers support it, M2 does not exercise it beyond tests.
- **Streams are cheap but not free** (~µs to create): long-lived objects
  owned by long-lived components, never created per-operation.

### 6.2 Wrappers

```cpp
namespace engine::cuda {

// Move-only RAII stream (created cudaStreamNonBlocking on the device that
// is current at construction). Destruction synchronizes the stream first
// (CHECK on sticky errors — destruction can't return Status), then
// destroys it: a CudaStream never dies with work in flight.
class CudaStream {
 public:
  [[nodiscard]] static core::StatusOr<CudaStream> Create();
  // Blocks the host until all work enqueued so far completes. Surfaces
  // deferred execution errors (§4.3) as Status.
  [[nodiscard]] core::Status Synchronize();
  // Makes *this* stream wait (device-side, non-blocking for the host)
  // until `event` — as last recorded — has completed.
  [[nodiscard]] core::Status WaitEvent(const CudaEvent& event);
  [[nodiscard]] cudaStream_t get() const;   // for toolkit calls; never null
  …
};

// Move-only RAII event. Two flavors: Timing() (default flags) and
// Sync() (cudaEventDisableTiming — cheaper, for ordering only).
class CudaEvent {
 public:
  [[nodiscard]] static core::StatusOr<CudaEvent> Timing();
  [[nodiscard]] static core::StatusOr<CudaEvent> Sync();
  [[nodiscard]] core::Status Record(const CudaStream& stream);
  [[nodiscard]] core::Status Synchronize();          // host-blocking
  [[nodiscard]] core::StatusOr<bool> Query() const;  // completed?
  // Both events must be Timing() and completed; ms between their records.
  [[nodiscard]] static core::StatusOr<float> ElapsedMs(const CudaEvent& start,
                                                       const CudaEvent& stop);
  …
};

}  // namespace engine::cuda
```

Lifetime rules (documented on the classes, tested in M2-T04):

- An **event may outlive the stream** it was recorded on (CUDA guarantees
  event completion state survives stream destruction; our stream destructor
  synchronizes first anyway).
- A **stream must not outlive its device context** — moot in practice
  (contexts live for the process).
- Recording an event on a destroyed stream, or `get()` on a moved-from
  wrapper, is programmer error → CHECK (moved-from wrappers hold a null
  handle and every member CHECKs engagement, mirroring `Tensor`'s
  undefined-handle policy).

*Refined in M2-T04 (implementation):*

- **Toolkit-free spelling.** `stream.h` must be includable on every build
  (§2.2), so the sketch's `cudaStream_t`/`cudaEvent_t` appear as pointers
  to the forward-declared opaque structs `CUstream_st`/`CUevent_st` — the
  exact types the toolkit typedefs name, so CUDA-compiled TUs use them
  interchangeably with no cast.
- **Events drain on destruction too**, mirroring the stream destructor
  (synchronize, then destroy): destruction cannot return Status, and no
  wrapper dies with its work unobserved.
- **`ElapsedMs` misuse taxonomy:** a `Sync()`-flavor argument →
  `InvalidArgument`; a not-yet-completed record → `FailedPrecondition`
  (pre-checked via `Query`, so `cudaErrorNotReady` never surfaces as the
  §4.2 table's opaque `kInternal`).
- **Never-recorded events count as complete** (CUDA semantics, documented
  on the members): `Query()` → true, `Synchronize()` → immediate success,
  and `WaitEvent` on one is a no-op.
- *Refined post-M2 (hardening audit):* the two rules above interact badly
  in `ElapsedMs` — a never-recorded event passes the completion pre-check
  (it "counts as complete") and then `cudaEventElapsedTime` fails with
  `cudaErrorInvalidResourceHandle` → the opaque `kInternal` the taxonomy
  exists to prevent. `CudaEvent` therefore tracks whether `Record()` ever
  succeeded, and `ElapsedMs` pre-checks it: a never-recorded argument →
  `FailedPrecondition`, before the completion check.

### 6.3 The per-device default stream, and the opaque handle

```cpp
// Engine-owned, lazily created, intentionally leaked (same pattern and
// rationale as DefaultCpuAllocator, tensor.md §6): one non-blocking stream
// per device, so M2/M3 call-sites have a stream without plumbing one.
// Thread-safe. NOT the CUDA legacy default stream.
[[nodiscard]] CudaStream& DefaultStream(int device_index);
```

*Refined in M2-T04:* `DefaultStream` returns a reference and cannot return
Status, so both failure modes are fatal: an out-of-range device index is
programmer error → CHECK (validate data-driven indices against
`device_count()` first — the `ScopedSetDevice` stance, §5.2), and stream
creation failure on a valid, visible device is `CHECK_OK`ed (not an
operating condition the engine plans around).

APIs outside the `cuda` module that accept a stream (`tensor`'s transfer
overload §8, `kernels` launchers §9) take it as an **opaque handle** so
their public headers stay toolkit-free (§2.2):

```cpp
namespace engine::cuda {
// Trivially-copyable opaque wrapper around cudaStream_t (a pointer under
// the hood; declared in a toolkit-free header, cuda/stream_handle.h).
// StreamHandle{} (null) means "the engine default stream for the relevant
// device" at every API that accepts one.
class StreamHandle { … };
}
```

`CudaStream::handle()` yields one; inside CUDA-compiled TUs it converts to
`cudaStream_t`. This is the one deliberately thin spot in the type system:
a `StreamHandle` does not own or validate. Passing a handle to a destroyed
stream is UB, same as any dangling pointer; ownership discipline (streams
are long-lived, §6.1) is the mitigation.

### 6.4 Who synchronizes, and when

The ownership rule, stated once and cited by later designs:

1. **Whoever enqueues async work is responsible for ensuring its
   completion is observed before its results or resources are consumed
   outside that stream.** Same-stream consumption needs nothing (streams
   are FIFO). Cross-stream consumption uses `CudaEvent::Record` +
   `CudaStream::WaitEvent` (device-side, preferred). Host consumption uses
   `Synchronize()` (host-blocking, the only place deferred errors surface —
   §4.3).
2. **M2's blocking sync points are exactly:** D2H transfers into pageable
   memory (implicitly synchronous, §8.2), explicit `Tensor::to` without a
   caller-provided stream (defined to be synchronous, §8.3), test
   assertions (`expect_tensors_close` syncs before comparing, §10.2), and
   stream/event destructor drains. Allocators never synchronize (§7.4);
   kernel launchers never synchronize.
3. **Buffer lifetime vs. async work:** freeing device memory while
   enqueued work may still read/write it is UB. In M2 this contract is
   discharged trivially — tests synchronize before teardown, and
   `CudaStream`'s draining destructor covers fixture teardown ordering.
   The caching allocator's reuse story depends on this same contract
   (§7.4). The engine milestones inherit rule 3 as: a `Tensor`'s buffer
   must be kept alive until an observed sync point after its last enqueued
   use — which the step-loop structure provides naturally.

---

## 7. Device & pinned allocators (M2-T05/T06/T07)

All three implement the M1 `Allocator` interface (tensor.md §6) unchanged:
thread-safe `Allocate(bytes, alignment) → StatusOr<Buffer>`, self-contained
deleters, `bytes == 0` → engaged null buffer, no `Deallocate` on the
interface.

### 7.1 `CudaAllocator` (naive, M2-T05)

```cpp
namespace engine::memory {

// One per (used) device, over cudaMalloc/cudaFree. Thread-safe (the
// toolkit calls are). Alignment: cudaMalloc guarantees ≥256B; requests
// above 256 are CHECK-rejected (nothing in the engine wants more; revisit
// if something does). device() is the construction-time Device.
class CudaAllocator final : public Allocator {
 public:
  // InvalidArgument if device_index ≥ device_count(); Unimplemented on
  // CPU-only builds.
  [[nodiscard]] static core::StatusOr<CudaAllocator> Create(int device_index);
  …
};

// Process-wide instance per device, lazily created, intentionally leaked
// (DefaultCpuAllocator pattern). What Tensor::empty uses for kCUDA.
[[nodiscard]] core::StatusOr<Allocator*> DefaultCudaAllocator(int device_index);

}  // namespace engine::memory
```

- `cudaErrorMemoryAllocation` → `kResourceExhausted` (§4.2) — the M2-T05
  acceptance criterion ("huge allocation returns ResourceExhausted, not
  crash").
- The deleter captures the device index and calls
  `ScopedSetDevice` + `cudaFree`; a failing `cudaFree` in a deleter (only
  plausible with a poisoned context, §4.3) logs and drops the error —
  deleters can't return Status, and the process is dying anyway.
- `Tensor::empty(shape, dtype, Device::Cuda(i))` routes to
  `DefaultCudaAllocator(i)`; M1's `Unimplemented` for kCUDA becomes real
  allocation (CUDA builds) or stays `Unimplemented` (CPU-only builds).

*Refined in M2-T05 (implementation):*

- **The `Allocator` base gained protected defaulted move operations** so
  `Create` can return `StatusOr<CudaAllocator>` by value — the base's
  deleted copy constructor had suppressed derived moves. Protected, so a
  derived allocator is never moved through a base reference (slicing).
- **The deleter uses raw `cudaSetDevice`/`cudaFree`, not `ScopedSetDevice`**:
  `ScopedSetDevice` CHECKs on failure, but a deleter must neither return
  Status nor abort from a destructor, so every deleter-path failure is
  logged and dropped as this section already required. A failed
  `cudaSetDevice` still attempts the free (`cudaFree` accepts pointers from
  any device context). `Allocate` itself does use `ScopedSetDevice`.
- **A failed `cudaMalloc` clears the last-error slot** (`cudaGetLastError()`)
  so the non-sticky OOM is not misread by a later §4.3 launch-error check.
- The alignment ceiling constant is `CudaAllocator::kGuaranteedAlignment`
  (256); `Allocate` CHECKs both power-of-two and the ceiling.
- The `memory → cuda` module edge this class needs is ADR-002 Amendment 2
  (see §2.1).

### 7.2 `PinnedCpuAllocator` (M2-T07)

Page-locked host memory via `cudaHostAlloc` (portable flag), freed with
`cudaFreeHost`. **`device()` is `Device::Cpu()`** — pinnedness is a
property of the allocation, not a new `DeviceType`: pinned tensors are
ordinary CPU tensors (dereferenceable on host, printable, `allclose`-able);
what pinning buys is true-async `cudaMemcpyAsync` (§8.2). Nothing in the
type system tracks pinnedness; the transfer path doesn't need to know (the
driver detects registered ranges), and callers that care (the engine's I/O
staging buffers, later) simply choose this allocator. On CPU-only builds:
`Unimplemented`.

*Refined in M2-T07 (implementation):*

- **`Create()` factory, mirroring `CudaAllocator`**: the stub's
  `Unimplemented` lives there (only producer of instances), and on CUDA
  builds `Create` eagerly reports **`kUnavailable` when no CUDA device is
  visible** — pinning requires a driver context, and the eager check beats
  letting `cudaHostAlloc` fail later (`cudaErrorNoDevice` maps to the same
  code via §4.2).
- **Alignment ceiling `kGuaranteedAlignment` (256)**, the same constant and
  CHECK as `CudaAllocator`: `cudaHostAlloc` returns page-aligned memory,
  and the shared ceiling keeps `CachingAllocator`'s fixed upstream
  alignment (§7.3) valid over either upstream. Exhaustion →
  `kResourceExhausted` (the §4.2 mapping of `cudaErrorMemoryAllocation`) —
  which the pool's OOM-retry already catches.
- **Deleters capture nothing**: `cudaFreeHost` needs no device context, so
  unlike `CudaAllocator`'s deleter there is no device index to carry;
  failures log-and-drop as usual.
- **No default-instance accessor** (`DefaultPinnedAllocator`): no consumer
  exists yet; the engine's staging paths (M4+) will own their allocator
  explicitly. Add one when a caller actually wants ambient pinned memory.

### 7.3 `CachingAllocator` (M2-T06)

The memory-pooling backbone. A caching layer over any upstream `Allocator`
(in practice `CudaAllocator`; the design is device-agnostic and tests can
run it over `CpuAllocator` on CPU-only CI):

```cpp
namespace engine::memory {

class CachingAllocator final : public Allocator {
 public:
  // Non-owning upstream; must outlive this allocator.
  explicit CachingAllocator(Allocator* upstream);

  [[nodiscard]] core::StatusOr<Buffer> Allocate(std::size_t bytes,
                                                std::size_t alignment) override;
  [[nodiscard]] tensor::Device device() const override;  // upstream's

  struct Stats {
    std::size_t bytes_allocated;   // currently handed out to live Buffers
    std::size_t bytes_reserved;    // allocated + cached free blocks
    std::uint64_t hit_count;       // served from free list
    std::uint64_t miss_count;      // required an upstream Allocate
  };
  [[nodiscard]] Stats stats() const;

  // Returns all cached (free) blocks to the upstream. Live Buffers are
  // unaffected. Returns the number of bytes released.
  std::size_t release_cached();
  …
};

}  // namespace engine::memory
```

Design decisions:

- **Size classes.** Requests are rounded up to a size class; a freed block
  re-enters the free list of its class, so any request in the class reuses
  it exactly. Classes: 512B minimum, powers of two up to 1MiB, then 1MiB
  steps (large blocks are weights/KV slabs where fragmentation, not
  rounding waste, is the enemy — the same shape as PyTorch's split, minus
  block-splitting, which we defer until fragmentation data demands it).
  Rounding waste is bounded at 2× for small blocks and ~1/1024 for big
  ones. The class table is an implementation constant; the *contract* is
  only "alloc→free→alloc of an equal size hits the cache" (the M2-T06
  acceptance test).
- **Free lists per class, exact-class reuse only.** No best-fit search
  across classes, no splitting/coalescing in M2. Misses go upstream.
  Upstream `kResourceExhausted` triggers **`release_cached()` + one
  retry** before propagating — the standard caching-allocator OOM
  courtesy.
- **Thread safety: one mutex.** Allocation is never on the token hot path
  (tensor.md §6 rationale); uncontended mutex cost is irrelevant. No
  lock-free cleverness.
- **Deleters point back to the pool** — the exception tensor.md §6
  anticipated: the deleter returns the block to the free list rather than
  freeing upstream, so it captures `this`. **Lifetime requirement,
  documented on the class: the `CachingAllocator` must outlive every
  `Buffer` it allocated.** (The engine's usage — process-lifetime pools —
  satisfies this trivially; tests are the only place teardown order needs
  care.) Destruction CHECKs that `bytes_allocated == 0` (destroying a pool
  with live buffers is programmer error), then releases cached blocks.
- **Stats are exact and lock-protected**, updated on every transition; the
  M2-T06 acceptance test scripts a sequence and asserts exact counter
  values, and proves reuse by asserting `miss_count` doesn't grow.

*Refined in M2-T06 (implementation):*

- **One fixed pool-wide upstream alignment** closes the gap this section
  left open (a cached block's alignment vs. a later request's): every
  upstream allocation is requested at
  `CachingAllocator::kMaxAlignment` (256 — cudaMalloc's guarantee, ≥ every
  alignment the engine uses) and `Allocate` CHECK-rejects requests above
  it. Any cached block therefore satisfies any admissible request, so free
  lists are keyed by size class alone. The alternative — keying by
  `(class, alignment)` — would fragment the cache for no current caller.
- **Stats are denominated in class-rounded bytes**, so `bytes_reserved` is
  exactly the pool's upstream footprint and the invariant
  `bytes_reserved == bytes_allocated + cached bytes` holds; the Buffer
  handed to the caller still reports the *requested* size (the
  CpuAllocator convention). `miss_count` counts requests that went
  upstream, incremented once per request even when the OOM-retry path
  calls the upstream twice.
- **Zero-byte requests bypass the cache and the stats** (there is no block
  to pool); their deleter deliberately does not reference the pool.
- **The OOM courtesy triggers on `kResourceExhausted` or `kOutOfMemory`**:
  the pool is device-agnostic and a host upstream reports exhaustion as
  the latter. Other error codes propagate immediately, cache intact.
- **The class is non-movable** (deleters capture `this`), and the
  destructor CHECK ("no live Buffers") is the lifetime rule above made
  fatal.
- **Lock discipline**: the deleter path locks the pool mutex, so cached
  upstream Buffers are destroyed outside the lock in `release_cached()`
  (never holding the pool mutex across `cudaFree`); the OOM-retry path is
  the exception and frees under the lock, since the memory must actually
  be back with the upstream before the retry (safe — upstream deleters
  are self-contained and cannot re-enter the pool).

### 7.4 Stream semantics of the pool (decided: stream-agnostic)

The pool does **not** track streams. A block freed (Buffer destroyed) and
immediately reused on any stream is safe **only** under §6.4 rule 3 — the
caller must not destroy a Buffer while device work using it is still in
flight. This is the same contract raw `cudaFree` imposes (free is
stream-ordered with respect to nothing the caller didn't order), so caching
adds no new hazard *category*; it does widen the blast radius of a
violation from "driver-managed" to "silent data corruption via reuse",
which is why the contract is stated in both places. Per-stream free lists
and event-guarded reuse (the PyTorch design) are deliberate non-goals until
the engine actually overlaps streams (revisit at the multi-stream engine
milestone, with this section updated).

### 7.5 Alternative considered: `cudaMallocAsync` / stream-ordered pools

CUDA's built-in pool allocator would give us driver-managed caching and
stream-ordered semantics for free. Rejected for M2: (a) the engine's
long-term memory story (paged KV cache, M7+) needs *our* pool's stats,
eviction hooks, and deterministic reuse, which the opaque driver pool
doesn't expose; (b) `Allocator`/`Buffer` are synchronous-by-design
(`Allocate` returns memory usable now, deleters take no stream) and
warping the M1 interface around stream-ordered semantics buys nothing at
M2's single-stream stage; (c) observability — `bytes_reserved` and
hit/miss stats are acceptance criteria and later Prometheus metrics.
Revisit only if fragmentation data in the KV-cache milestones argues for
it.

---

## 8. Host↔device transfers (M2-T07)

### 8.1 API

The M1-T09 `copy` contract extends rather than changes — identical shape
and dtype required, `InvalidArgument` otherwise (conversion stays `cast`'s
job, CPU-only for now):

```cpp
namespace engine::tensor::ops {

// Existing (M1-T09) — now defined for same-device CPU↔CPU only.
[[nodiscard]] core::Status copy(const Tensor& dst, const Tensor& src);

// M2-T07: device-aware, async on `stream` where the memory allows it.
// Supports H2D, D2H, D2D (same device; cross-device is Unimplemented
// until the distributed milestone), and H2H (delegates to the CPU path,
// stream ignored). Both tensors must be CONTIGUOUS (InvalidArgument
// otherwise) — strided device copy needs a kernel; deferred until a
// consumer exists. Null StreamHandle = the destination-relevant device's
// engine default stream.
[[nodiscard]] core::Status copy(const Tensor& dst, const Tensor& src,
                                cuda::StreamHandle stream);

}  // namespace engine::tensor::ops
```

```cpp
// M2-T07, on Tensor: allocate a contiguous tensor on `device` (default
// allocator for that device) and copy *this into it. A no-op returning
// *this (same handle, cheap) when already on `device`. Without a stream
// argument the call is SYNCHRONOUS — on return the result is safe to use
// from the host or any stream. The stream overload enqueues on `stream`
// and returns immediately; the result must not be consumed before a sync
// point per §6.4 (the M2 acceptance tests demonstrate the event pattern).
// Source must be contiguous (InvalidArgument otherwise) until strided
// device copy exists.
[[nodiscard]] core::StatusOr<Tensor> Tensor::to(Device device) const;
[[nodiscard]] core::StatusOr<Tensor> Tensor::to(Device device,
                                                cuda::StreamHandle stream) const;
```

Layering per §2.1: declarations stay in `tensor`'s toolkit-free headers
(`StreamHandle` comes from the toolkit-free `cuda/stream_handle.h`);
definitions live in a `transfer.cpp` compiled into `engine_tensor` only on
CUDA builds (stub returns `Unimplemented` otherwise), calling
`cudaMemcpyAsync` directly.

*Refined in M2-T07 (implementation):*

- **The seam moved below the public functions.** Validation and H2H
  delegation are build-invariant, so `copy(dst, src, stream)` lives in
  `ops.cpp` and `Tensor::to` in `tensor.cpp` (always compiled); they
  dispatch device-involving copies to one internal function —
  `detail::DeviceCopy(dst, src, stream, synchronize)` in
  `transfer_detail.h` — implemented by `transfer.cpp` (CUDA builds) or
  `transfer_stub.cpp` (`Unimplemented`, CPU-only). Consequences: the
  "mismatched shapes/dtypes → InvalidArgument" acceptance criterion is
  testable on CPU-only CI, and **H2H pairs work on every build** (the stub
  taxonomy applies only to copies that actually involve a CUDA tensor —
  unreachable through supported factories on CPU-only builds).
- The public layer also owns the small-print: `numel() == 0` succeeds
  without touching the toolkit (zero-size tensors may hold null data
  pointers), an identical dst/src view is a well-defined no-op (mirroring
  the two-argument `copy`), and the cross-device `Unimplemented` fires
  before the seam, so `DeviceCopy`'s CUDA pair always shares one device.
- `transfer.cpp` passes **explicit `cudaMemcpyKind`s** rather than
  `cudaMemcpyDefault`: the direction is already classified, and the
  explicit kind keeps the driver's pointer validation.
- The module edge this TU needs is ADR-002 Amendment 3 (§2.1).

### 8.2 Synchronization semantics (the honest table)

`cudaMemcpyAsync` on a stream is only as async as the host memory allows:

| Direction | Host memory | Actual behavior |
|---|---|---|
| H2D | pinned | truly async on the stream |
| H2D | pageable | the *driver* may stage/synchronize; treated as "async-unsafe": safe usage is identical to pinned (§6.4 discipline) but no overlap is promised |
| D2H | pinned | truly async on the stream |
| D2H | pageable | effectively synchronous with respect to the host thread — but **still ordered on the stream**, so correctness discipline is unchanged |
| D2D | — | truly async |

The design does not fork the API on pinnedness (the driver detects it;
§7.2). What it promises is *correctness* under §6.4 for every row;
*overlap* is promised only for pinned rows — the distinction that matters
starting with chunked prefill and weight streaming, which is exactly why
`PinnedCpuAllocator` ships in the same ticket.

### 8.3 Defaults chosen for safety

The stream-less `Tensor::to(device)` synchronizes the transfer before
returning (internally: enqueue on the device's default stream, then
`Synchronize`). Rationale: the stream-less spelling will be used by model
loading (M4/M5) and tests, where surprise-async is a footgun and the sync
cost is irrelevant; hot paths that care will pass a stream explicitly.
Same-device `to` returning the same handle (not a copy) mirrors
`reshape`'s "never copies" philosophy; a deep copy is `ops::copy` onto a
fresh tensor, as always.

*Refined in M2-T07 (implementation):* the source-contiguity requirement is
checked **before** the same-device fast path, so a non-contiguous `to` is
`InvalidArgument` even when no copy would have happened — the contract does
not depend on the destination, and relaxing later (when strided device copy
exists) cannot break anyone.

---

## 9. Kernel infrastructure (M2-T08)

### 9.1 Source organization

The pattern every kernel milestone follows:

- **`src/kernels/<area>.h`** — public, toolkit-free: `Status`-returning
  host launcher declarations taking `Tensor`s and a `StreamHandle`.
  Consumers (`engine`, tests) include only these.
- **`src/kernels/<area>.cu`** — `__global__` kernels + launcher
  definitions. Kernels get internal linkage via an anonymous namespace
  (not `static` — nvcc rejects `static` on `__global__` function
  templates) unless shared;
  cross-`.cu` sharing goes through an internal `detail/` header, not
  extern device symbols (separable compilation stays off, §3).
- **`src/kernels/launch.h`** (internal, safe for host TUs) — launch-config
  helpers: `LaunchConfig1D(n)` returning grid/block (block 256, grid
  capped at the grid-stride-loop-friendly limit) and the loop macro:

```cpp
// Grid-stride loop: correct for any n ≥ 0 with any launch config;
// performance-tuning the config never changes semantics.
#define CUDA_1D_KERNEL_LOOP(i, n)                                      \
  for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += static_cast<std::int64_t>(blockDim.x) * gridDim.x)
```

- **`src/kernels/dispatch.h`** — dtype dispatch:

```cpp
// Expands `body` once per floating dtype with `scalar_t` bound to the
// DEVICE type for that dtype; returns Unimplemented/InvalidArgument
// Status for non-floating dtypes (per call-site policy).
DISPATCH_FLOATING_TYPES(tensor.dtype(), "add", [&] { … launch<scalar_t>(…); });
```

  Integer-dtype dispatch variants are added when a consumer exists (M2
  kernels are floating-only; `cast` on GPU covers fp32↔fp16↔bf16 per the
  ticket, integer casts stay CPU-only until needed).

*Refined in M2-T08 (implementation):*

- **`dispatch.h` is an internal, toolkit-including header** (the
  `cuda_check.h` precedent, §2.2) — binding `scalar_t` to the device type
  inherently names `__half`/`__nv_bfloat16`, so it is includable only from
  CUDA-compiled TUs. The sketch above didn't classify it; the public
  surface remains `<area>.h` only.
- The macro signature gained two mechanical parameters:
  `DISPATCH_FLOATING_TYPES(dtype, op_name, alias, body...)`. The bound
  alias name is caller-chosen so `cast`'s src×dst dispatch can nest without
  shadowing, and the body is a trailing variadic so top-level commas in the
  lambda don't split macro arguments. Unsupported dtypes uniformly return
  `Unimplemented` naming the op (the §9.2 policy; no call site wants
  `InvalidArgument` yet).
- `launch.h` constants: block 256 (`kDefaultBlockSize`), grid cap 4096
  (`kMaxGridBlocks`). The cap is deliberately small so the grid-stride wrap
  path is testable at ~1M elements; any value is semantically equivalent.
  `LaunchConfig1D` CHECKs `n > 0` — the numel()==0 guard lives in the
  launcher (§9.2), so a zero grid can never be requested.

### 9.2 Launcher contract

Every host-side launcher, uniformly:

1. Validates: same shape, same dtype (where applicable), same device, all
   on kCUDA, contiguous (M2 kernels are contiguous-only; strided support
   is per-kernel, later, and always explicit) → `InvalidArgument`.
   Dispatch-unsupported dtype → `Unimplemented`.
2. `ScopedSetDevice` to the tensors' device.
3. Launches on the given stream (null handle → device default stream).
4. `CUDA_RETURN_IF_ERROR(cudaGetLastError())` — §4.3 rule: no
   fire-and-forget launches.
5. **Never synchronizes.** Enqueue-only; §6.4 puts sync on the caller.

`numel() == 0` is valid and returns OK without launching (grid of 0 is an
error in CUDA; the guard lives in the launcher, once).

*Refined in M2-T08 (implementation):* steps 1 and the numel guard live in
an **always-compiled** `elementwise.cpp`, with only the toolkit-touching
steps 2–5 behind an internal seam (`elementwise_detail.h`, implemented by
`elementwise.cu` on CUDA builds and `elementwise_stub.cpp` on CPU-only
builds) — the same split M2-T07 made for transfers, and for the same
reason: the "validate inputs and return Status" acceptance criterion runs
on CPU-only CI. Two consequences, both tested:

- Validation order is shape → dtype → contiguity → device, so every
  data-driven rejection is exercisable with CPU tensors on any build.
- Non-CUDA operands are `InvalidArgument` on **every** build (the device
  check precedes the seam); the stub's "built without CUDA" `Unimplemented`
  is unreachable through supported factories, exactly like the transfer
  stub (§8.1).

Aliasing, decided per-kernel here: for the same-dtype kernels
(add/mul/scale) a dst *identical* to an input is well-defined — element i
depends only on the inputs' element i. Partial overlap, and any dst/src
aliasing for `cast` (element widths differ), are undefined behavior
(ops::copy's memcpy rule). `cast` accepts the same-dtype diagonal (a device
deep copy); `scale`'s double scalar is narrowed to float once, host-side,
since per-element math is float anyway (§9.3).

### 9.3 Device half types

Kernels compute in `__half`/`__nv_bfloat16`; the host types
`tensor::float16`/`bfloat16` (half.h) are layout-identical (2-byte,
`uint16_t` payload) and get `reinterpret_cast` at the kernel boundary
inside `.cu` files only. A `static_assert` on size/triviality guards the
cast. half.h itself stays toolkit-free (it is a `tensor_base` header —
tensor.md §3.1); no `__CUDA_ARCH__` conditionals in it, ever. Elementwise
math in M2 kernels widens to float and re-narrows per element — matching
the CPU reference's fp32 accumulate behavior closely enough for the M2
tolerance table (§10.2); kernels that need exact-narrowing parity state it
per-kernel later.

### 9.4 M2's kernels

`add(dst, a, b, stream)`, `mul(dst, a, b, stream)` (elementwise),
`scale(dst, src, scalar, stream)` (scalar is host `double`, converted per
dtype), `cast(dst, src, stream)` for {fp32, fp16, bf16}² — out-of-place,
`dst` pre-allocated by the caller (launchers don't allocate; factories
do). These exist to exercise: launch helpers, dispatch, validation,
Status-on-launch-error, and the CPU-comparison test pattern — one kernel
per infrastructure feature would be over-engineering, four trivial kernels
cover it all.

---

## 10. GPU testing strategy (M2-T09)

### 10.1 Skip infrastructure

`tests/common/cuda.h` (linked into test targets via the existing
`tests/common` library):

```cpp
namespace engine::testing {
// True iff CUDA work is possible: built with CUDA and device_count() > 0.
[[nodiscard]] bool HasCudaDevice();
}

// In a fixture or test body:
#define ENGINE_SKIP_WITHOUT_CUDA()                                   \
  if (!::engine::testing::HasCudaDevice()) {                         \
    GTEST_SKIP() << "no CUDA device available";                      \
  }
```

Skips, never failures — the M2 acceptance criterion and the standing
CPU-CI rule (tests/README.md). GPU tests live in the same
`tests/unit/*_test.cpp` files and targets as everything else; there is no
separate GPU suite to forget to run. A `gpu` ctest label is added for
`ctest -L gpu` convenience on GPU machines.

### 10.2 The kernel-testing pattern (answers question 3)

Every GPU kernel test follows the correctness ladder: the CPU
reference is the oracle.

```
1. Build inputs on CPU with M1 seeded fills (deterministic).
2. Compute the expected result with the CPU implementation (cpu/ or
   tensor ops).
3. Tensor::to(cuda) the inputs; run the kernel on a fixture stream.
4. expect_tensors_close(gpu_out, cpu_expected, atol, rtol):
   synchronizes the stream (surfacing deferred errors, §4.3),
   copies D2H, and runs ops::allclose — failure messages carry
   allclose's worst-mismatch report.
5. Shapes exercised per kernel: 0, 1, block-multiple, non-multiple,
   > one grid's worth (grid-stride wrap) — the M2-T08 criterion.
```

Tolerances are stated explicitly per test; the M1-T08 per-dtype `allclose`
defaults are the starting point, and any kernel needing looser bounds
documents why in the test. `CudaTestFixture` (`tests/common/cuda.h`)
provides: the skip guard in `SetUp`, device 0 selection, a per-test
`CudaStream`, and a per-test `CachingAllocator` over the device allocator;
`TearDown` synchronizes the stream so §6.4/§7.4 teardown-order rules hold
mechanically. M2-T09 migrates the tests M2-T03…T08 wrote inline onto the
fixture and documents this pattern in `tests/README.md`.

> **Refined in M2-T09.** Three decisions the implementation pinned down:
>
> - **Helper signature.** The comparison helper is
>   `engine::testing::ExpectTensorsClose(actual, expected, CudaStream&,
>   optional<double> rtol, optional<double> atol)` — the project's
>   PascalCase, mirroring `allclose`'s optional tolerances rather than bare
>   `(atol, rtol)` scalars. It takes a `CudaStream&` (not a `StreamHandle`)
>   so `tests/common` stays toolkit-free: synchronizing a raw handle would
>   need `cudaStreamSynchronize`, while `CudaStream::Synchronize()` is on
>   the toolkit-free surface. Null-handle launch tests pass
>   `DefaultStream(0)`. It is a free function (usable from `.cu` TUs and
>   non-fixture code) rather than a fixture member; the fixture is the usual
>   source of the stream argument.
> - **The `gpu` label mechanism.** Suites needing a device follow a naming
>   contract — `*GpuTest` / `*GpuDeathTest`, which deriving from
>   `CudaTestFixture` produces naturally — and `engine_add_tests` runs
>   `gtest_discover_tests` twice per target with complementary
>   `TEST_FILTER`s. The GPU set gets the single label `<tree>-gpu` (not the
>   list `<tree>;gpu`: the module flattens list-valued PROPERTIES at the
>   CMake 3.26 floor); `ctest -L` matches by regex, so `-L gpu` and
>   `-L <tree>` both select correctly.
> - **Fixture members are `std::optional`s** (`ScopedSetDevice`, stream,
>   pool), engaged only after the skip guard passes — so the fixture
>   compiles and skips cleanly on CPU-only builds without touching a stub —
>   and torn down in reverse order after the stream sync.

### 10.3 What runs where

| Environment | What runs |
|---|---|
| CPU-only CI (GitHub Actions) | everything except GPU tests, which skip via the predicate; CUDA-touching modules build their stub paths |
| GPU dev machine | full suite; `ctest -L gpu` for the GPU subset |
| Optional GPU CI | M2-T09 adds a dormant workflow file for a self-hosted/manual runner; not wired to required checks |

Leak checking: allocator tests bracket alloc/free cycles with
`cudaMemGetInfo` deltas (tolerating driver noise by asserting return to
within a small epsilon of the starting free-bytes, and exact equality of
the pool's own `bytes_reserved` accounting).

---

## 11. Thread-safety guarantees

Restating the tensor.md §9 style for the new types:

- **`CudaStream` / `CudaEvent`** follow the `Buffer` model: move-only,
  owned by one place; concurrent use of one object requires external
  synchronization. Distinct streams/events are freely usable from
  different threads (CUDA's API is thread-safe).
- **`DefaultStream(i)` and the leaked default allocators** are safe to
  call from any thread (lazy init is lock-guarded).
- **All `Allocator` implementations remain thread-safe** per the M1
  contract: `CudaAllocator` trivially, `CachingAllocator` via its mutex
  (stress-tested per the M2-T06 criterion), `PinnedCpuAllocator`
  trivially.
- **CUDA's own thread model** (current-device is thread-local) is confined:
  engine code never assumes a current device — every entry point that
  needs one sets it via `ScopedSetDevice` (allocators, launchers,
  transfers). No engine API requires the caller to have set a device.
- Nothing in M2 spawns threads.

---

## 12. Deferred (known, intentionally not designed here)

- **cuBLAS handle/workspace management** — M5-T02, `src/cuda/cublas.h`.
- **Multi-stream engine orchestration & per-stream pool semantics** —
  engine milestones; §7.4 is revisited then.
- **CUDA graphs** — M11-T05; M2's stream rules (§6.1) keep capture viable.
- **Cross-device D2D & NCCL** — distributed milestone.
- **Strided device copies and kernels** — added per-consumer, per-kernel.
- **Kernel performance work** — per-milestone, always against a measured
  baseline (`benchmarks/BASELINES.md`).
- **FP8/INT4 device paths** — M12/M13 with their quant designs.
