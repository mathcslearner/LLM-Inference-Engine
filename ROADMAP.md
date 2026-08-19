# LLM Inference Engine — Development Roadmap (v2, CPU-first)

A CPU-first LLM inference engine for decoder-only transformer models (Llama, Qwen,
and similar families), written in C++20 with hand-vectorized SIMD kernels (NEON on
Apple Silicon arm64, AVX2 on Linux x86-64). Performance-competitive with existing
CPU engines (llama.cpp-class) while emphasizing clean architecture, maintainability,
and testability.

**This is the second-generation roadmap.** The project began as a multi-GPU CUDA
engine (v1 roadmap — archived at `docs/archive/ROADMAP-v1.md` by M3-T01). After
Milestone 2 it pivoted to CPU-first: the development environment has no NVIDIA
hardware, and the post-M2 audit demonstrated that CUDA written blind accumulates
unverifiable risk (a compile-breaking bug survived five "done" tickets because no
CUDA compiler ever saw the code). The decision and its alternatives are recorded in
ADR-004 (written in M3-T01). Milestones 0–2 below are v1 history, preserved verbatim
with their completion markers; M3 retires M2's CUDA-specific deliverables and stands
up the CPU compute substrate in their place. Everything from M3 onward is the
CPU-first plan. This file became `ROADMAP.md` with M3-T01 — the canonical name
always points at the current plan, and the GPU past lives in exactly two
clearly-labeled places: `docs/archive/` and the ADR record.

The end-state feature set is unchanged from v1 except where GPUs are inherent
(tensor parallelism and CUDA graphs are dropped; a GPU backend remains a future
direction): paged KV cache, continuous batching, prefix caching, custom SIMD
kernels, chunked prefill, weight-only quantization (INT8/INT4, AWQ/GPTQ formats,
INT8 KV cache), OpenAI-compatible HTTP serving with SSE streaming, speculative
decoding, and Prometheus metrics. Two tracks are deepened beyond v1 scope: a
**quantization algorithms & evaluation toolkit** (M14 — the engine produces and
evaluates its own quantized checkpoints, not just serves them) and a
**speculative-decoding measurement study** (M15-T07).

This roadmap is organized as **milestones** executed in **linear order**. Each milestone
contains **tickets** — independent, focused tasks sized for roughly one development
session (1–4 hours, typically a few hundred lines of code including tests). Tickets
within a milestone are also ordered linearly; dependencies are listed explicitly.

---

## How to work this roadmap

- **Execute tickets in order.** IDs are `M<milestone>-T<ticket>` (e.g. `M6-T03`).
  Dependencies always point backwards; if you work linearly you never need to skip.
- **Definition of done for every ticket:** code compiles with zero warnings under the
  project warning set, `clang-format` clean, tests written **in the same ticket** and
  passing via `ctest`, all listed acceptance criteria met, public APIs documented with
  doc comments.
- **Mark progress** by appending ` — ✅ DONE (YYYY-MM-DD)` to the ticket heading when
  complete, so the roadmap doubles as a status tracker.
- **Design docs come first.** Milestones that introduce a new subsystem begin with a
  design-doc ticket (`docs/design/*.md`) or an ADR (`docs/adr/*.md`). Implementation
  tickets must conform to the accepted design; if implementation reveals the design is
  wrong, update the doc in the same PR and note the change.
- **Testing policy:** unit tests for every module; golden-data tests (fixtures generated
  from HuggingFace via `tools/` Python scripts) for anything numerical. The correctness
  chain is: HF fixtures validate the scalar reference implementations; the scalar
  reference validates every vectorized (NEON/AVX2) and optimized kernel. The suite runs
  the best ISA the host offers **plus a forced-scalar pass** (dispatch override), so
  CI on x86-64 (AVX2 + scalar) and the arm64 dev machine (NEON + scalar) together
  cover every backend; no test is silently skipped for lack of hardware.
- **Performance claims are measured:** optimization tickets record before/after numbers
  in `benchmarks/BASELINES.md`. No perf change without a benchmark delta.
- **Never break abstractions:** the module boundaries listed in `CLAUDE.md` are
  load-bearing. A ticket that needs to reach across a boundary is a signal the design
  doc needs amending first.

### Module map (established across milestones)

| Module | Directory | Introduced |
|---|---|---|
| Core utilities (status, logging, config) | `src/core/` | M0 |
| Tensor library | `src/tensor/` | M1 |
| Memory management (allocators, pools) | `src/memory/` | M1–M2 |
| Parallel runtime (thread pool, parallel_for) | `src/parallel/` | M3 |
| SIMD kernels (scalar/NEON/AVX2) | `src/kernels/` | M3, M6 |
| CPU reference backend (correctness oracle) | `src/cpu/` | M5 |
| Model loader & configs | `src/model/` | M4 |
| Tokenizer | `src/tokenizer/` | M4 |
| KV cache | `src/kvcache/` | M5, M8 |
| Sampling | `src/sampling/` | M7 |
| Execution engine (model runner) | `src/engine/` | M5–M6 |
| Scheduler | `src/scheduler/` | M9 |
| Runtime (engine loop, requests) | `src/runtime/` | M9 |
| Server / API | `src/server/` | M10 |
| Quantization (containers, kernels, algorithms) | `src/quant/` | M13–M14 |
| Speculative decoding | `src/spec/` | M15 |
| Metrics & profiling | `src/metrics/` | M16 |
| Benchmarks | `benchmarks/` | M6, M12, M16 |
| Dev tooling (fixture generation, scripts) | `tools/` | M4 |

Retired at the pivot (M3-T02): `src/cuda/` (streams, events, device utils) and the
planned `src/distributed/` (NCCL tensor parallelism). The device-agnostic pieces of
M2 live on: the caching pool allocator becomes the KV block pool's backing store, and
the M1 `Device` abstraction keeps its reserved `kCUDA` value against a future GPU
backend.

---

## Milestone 0 — Project Foundation & Tooling

**Overview.** Establish the repository skeleton, build system, dependency management,
test harness, linting, CI, logging, and error-handling primitives. Nothing
model-related yet — this milestone exists so every later ticket lands in a project
with professional infrastructure: one command to build, one command to test, style
enforced mechanically, and decisions recorded as ADRs.

**Architecture documents:** ADR-001 (language & toolchain), ADR-002 (repository
layout & module boundaries), ADR-003 (error-handling policy) — written in M0-T08.

### M0-T01 · Repository skeleton & CMake build system — ✅ DONE (2026-08-03)
Create the top-level `CMakeLists.txt` (C++20, CMake ≥ 3.26), the directory layout from
the module map (empty module subdirectories with stub `CMakeLists.txt`), a project-wide
warning set (`-Wall -Wextra -Werror` baseline), and a placeholder `main` executable.
**Depends on:** none.
**Acceptance criteria:**
- `cmake -B build && cmake --build build` succeeds on Linux x86-64 with GCC 12+ and Clang 16+.
- Debug/Release configurations work; warnings-as-errors is on for project code only.
- Directory layout matches the module map; each module builds as its own static library target (empty for now).

### M0-T02 · Third-party dependency management — ✅ DONE (2026-08-03)
Wire up dependency fetching via CMake `FetchContent` (or CPM.cmake) with pinned
versions: `fmt`, `spdlog`, `nlohmann_json`, `GoogleTest`. Document the policy for
adding dependencies.
**Depends on:** M0-T01.
**Acceptance criteria:**
- All dependencies pinned to exact tags/commits; a clean clone builds with no system-installed libraries beyond the toolchain.
- `docs/dependencies.md` lists each dependency, version, license, and why it was chosen.

### M0-T03 · Test harness — ✅ DONE (2026-08-03)
Integrate GoogleTest with CTest. Create `tests/unit/` and `tests/integration/` trees, a
shared `tests/common/` helper library, and one example test per tree.
**Depends on:** M0-T02.
**Acceptance criteria:**
- `ctest --test-dir build` discovers and runs all tests; a deliberately failing test fails the run.
- Adding a new test file requires only adding it to one CMake list (documented in `tests/README.md`).

### M0-T04 · Formatting & static analysis — ✅ DONE (2026-08-03)
Add `.clang-format` (project style, documented rationale), `.clang-tidy` with a curated
check set, and a `scripts/check-format.sh` that verifies formatting without modifying files.
**Depends on:** M0-T01.
**Acceptance criteria:**
- `scripts/check-format.sh` exits non-zero on misformatted code; `scripts/format.sh` fixes it.
- `clang-tidy` runs clean on existing code; the enabled check list is committed and commented.

### M0-T05 · Continuous integration (CPU) — ✅ DONE (2026-08-03)
GitHub Actions workflow: build (GCC + Clang, Debug + Release) and run tests on every
push/PR, with ccache and dependency caching. GPU jobs are out of scope (added later as
manual/self-hosted).
**Depends on:** M0-T03, M0-T04.
**Acceptance criteria:**
- CI runs build + ctest + format check; a red build blocks merging.
- Cold-cache CI run < 15 min; warm-cache < 5 min.

### M0-T06 · Logging subsystem — ✅ DONE (2026-08-03)
Thin wrapper over spdlog in `src/core/logging.h`: leveled macros (`LOG_DEBUG` …
`LOG_ERROR`), per-module logger names, level configurable via environment variable and
programmatic API. No raw spdlog usage outside the wrapper.
**Depends on:** M0-T02.
**Acceptance criteria:**
- Log level switchable at runtime; messages include timestamp, level, module, source location.
- Unit tests verify level filtering and formatting; hot-path macro compiles to nothing above the configured level in Release.

### M0-T07 · Error-handling primitives — ✅ DONE (2026-08-03)
Implement `Status` and `StatusOr<T>` (or `Result<T, Error>`) in `src/core/status.h`
with an error-code taxonomy (`InvalidArgument`, `OutOfMemory`, `NotFound`,
`Unimplemented`, `Internal`, `ResourceExhausted`, …), plus `RETURN_IF_ERROR`,
`ASSIGN_OR_RETURN`, and fatal `CHECK`/`DCHECK` macros.
**Depends on:** M0-T06.
**Acceptance criteria:**
- Policy: recoverable errors return `Status`; programmer errors use `CHECK`; exceptions are not used across module boundaries. Documented in the header.
- Unit tests cover propagation macros, error message composition, and `StatusOr` move semantics.

### M0-T08 · Documentation scaffolding & founding ADRs — ✅ DONE (2026-08-03)
Create `docs/` structure: `docs/adr/` with a template, `docs/design/` for subsystem
designs. Write ADR-001 (C++20/CUDA choice, alternatives considered incl. Rust),
ADR-002 (repo layout & module dependency rules — which modules may depend on which),
ADR-003 (error-handling policy).
**Depends on:** M0-T07.
**Acceptance criteria:**
- ADR template includes Status/Context/Decision/Consequences sections.
- ADR-002 contains an explicit module dependency diagram (e.g. `server → runtime → scheduler/engine → tensor/cuda → core`); no cycles.

---

## Milestone 1 — Tensor Library & Core Abstractions (CPU)

**Overview.** Build the foundational data structures every other module consumes:
dtypes, shapes, devices, allocators, buffers, and the `Tensor` type. Kept deliberately
minimal — this is not a general ML framework; it is exactly the tensor abstraction an
inference engine needs (dense, mostly-contiguous, explicit memory management, no
autograd). Getting ownership and view semantics right here prevents whole classes of
bugs later.

**Architecture documents:** `docs/design/tensor.md` (M1-T01).

### M1-T01 · Design doc: tensor library & device model — ✅ DONE (2026-08-03)
Write `docs/design/tensor.md`: supported dtypes (now + planned), shape/stride
representation, ownership model (shared buffer + views), allocator interface, device
abstraction, host/device data-access rules, and explicit non-goals (autograd, lazy
eval, broadcasting beyond what inference needs).
**Depends on:** M0-T08.
**Acceptance criteria:**
- Doc covers all sections above with API sketches for `Tensor`, `Buffer`, `Allocator`, `Device`.
- Explicitly specifies thread-safety guarantees and copy/move semantics of `Tensor`.

### M1-T02 · DataType enum & traits — ✅ DONE (2026-08-03)
`src/tensor/dtype.h`: `DataType` enum (`kFloat32`, `kFloat16`, `kBFloat16`, `kInt8`,
`kUInt8`, `kInt32`, `kInt64`, `kBool`, plus reserved `kFP8E4M3`, `kInt4` for later),
`itemsize()`, `to_string()`/`from_string()`, and a compile-time `DTypeTraits<T>`
mapping C++ types ↔ enum values.
**Depends on:** M1-T01.
**Acceptance criteria:**
- Unit tests cover every enum value's size, name round-trip, and trait mapping.
- `kInt4` reports sub-byte handling explicitly (itemsize in bits API or documented packing rule).

### M1-T03 · Shape & strides — ✅ DONE (2026-08-03)
`src/tensor/shape.h`: `Shape` (small inline vector of dims), `numel()`, row-major
stride computation, `is_contiguous()` check, dim validation, equality, formatting.
**Depends on:** M1-T02.
**Acceptance criteria:**
- Unit tests: 0-d through 5-d shapes, numel overflow detection, contiguity for sliced strides, stride computation golden cases.

### M1-T04 · Device abstraction — ✅ DONE (2026-08-03)
`src/tensor/device.h`: `Device{DeviceType type; int index;}` with `DeviceType::kCPU`
and `DeviceType::kCUDA`, parsing (`"cuda:0"`), equality, formatting. CUDA devices are
representable now but any attempt to allocate on them returns `Unimplemented` until M2.
**Depends on:** M1-T02.
**Acceptance criteria:**
- Unit tests for parsing, formatting, equality, invalid inputs.
- No CUDA headers included — this header stays backend-agnostic.

### M1-T05 · Allocator interface & CPU allocator — ✅ DONE (2026-08-03)
`src/memory/allocator.h`: abstract `Allocator` (`allocate(bytes, alignment) →
StatusOr<Buffer>`), `Buffer` as a move-only owning handle (pointer, size, device,
deleter). Implement `CpuAllocator` with configurable alignment (default 64B).
**Depends on:** M1-T04, M0-T07.
**Acceptance criteria:**
- Unit tests: alignment honored, zero-size allocation defined behavior, Buffer move semantics, deleter invoked exactly once (tracked via test allocator).
- Allocation failures return `Status`, never throw.

### M1-T06 · Tensor core type — ✅ DONE (2026-08-03)
`src/tensor/tensor.h`: `Tensor` = shared `Buffer` + `Shape` + strides + `DataType` +
`Device` + byte offset. Factory `Tensor::empty(shape, dtype, device, allocator)`.
Views: `slice(dim, start, end)`, `reshape` (contiguous only), `view_as_dtype`
(same-size only). Typed `data_ptr<T>()` with dtype check; CPU-only element accessor
for tests.
**Depends on:** M1-T05.
**Acceptance criteria:**
- Unit tests: views share the buffer (write-through visible), slicing produces correct shapes/strides/offsets, reshape rejects non-contiguous, dtype-checked access fails loudly on mismatch.
- `Tensor` is cheap to copy (shared buffer semantics documented and tested).

### M1-T07 · Half-precision host support — ✅ DONE (2026-08-03)
`src/tensor/half.h`: `float16` and `bfloat16` value types for host code (bit-accurate
conversion to/from `float`, including rounding, inf/nan, subnormals for fp16), wired
into `DTypeTraits`.
**Depends on:** M1-T06.
**Acceptance criteria:**
- Golden bit-pattern tests: known fp32↔fp16 and fp32↔bf16 pairs including rounding boundaries, ±inf, NaN preservation.
- Conversion round-trip property test over random floats within representable range.

### M1-T08 · Tensor factories & comparison utilities — ✅ DONE (2026-08-03)
`src/tensor/ops.h` (CPU): `zeros/ones/full/arange`, seeded uniform/normal fill,
element-wise `allclose(a, b, rtol, atol)` with per-dtype default tolerances and a
max-abs-diff report, and human-readable tensor printing for debugging.
**Depends on:** M1-T07.
**Acceptance criteria:**
- Unit tests for each factory across dtypes; `allclose` failure message reports index and values of worst mismatch.
- Random fills are deterministic given a seed (test asserts exact values).

### M1-T09 · CPU copy & cast — ✅ DONE (2026-08-03)
`src/tensor/ops.h`: `copy(dst, src)` (same shape, handles non-contiguous views) and
`cast(src, dtype)` between all floating dtypes and int types on CPU.
**Depends on:** M1-T08.
**Acceptance criteria:**
- Unit tests: contiguous and strided copies, all supported cast pairs, precision-loss cases (fp32→fp16 rounding) match the M1-T07 conversion functions exactly.

### Post-M1 hardening — ✅ DONE (2026-08-03)
Audit-driven fixes before M2, in one change. Bugs: `arange` count math was UB
for span `INT64_MIN` / step `-1` (now computed in uint64; count overflow →
`InvalidArgument`); `Device` was an open aggregate whose documented
`index == 0`-for-CPU invariant was unenforced (now a class CHECKing its
invariants at construction); `Buffer` moves left the source's `device()`
unchanged (now reset, so moved-from == default-constructed). Contract
corrections (header comments promised more than the code guaranteed):
`Tensor::data()` nullness, moved-from `Tensor` metadata, `cast`'s
signaling-NaN quieting vs half.h, `DefaultCpuAllocator` lifetime wording;
design-doc API sketches synced (`kAllDataTypes`/`kNumDataTypes`, fmt
formatters, `CpuAllocator` convenience overload, `item(initializer_list)`).
Test hardening: `DeviceType` stable-value and `kAllDataTypes`
density/anchor static_asserts, death tests for previously uncovered CHECK
paths, boundary cases (rank-8 `FromDims`, numel at `INT64_MAX`,
`cuda:INT_MAX` parse), half→int and int64-double-rounding casts, rank-0
copy/cast, non-contiguous self-copy, multi-dim print truncation, half
printing, and normal-fill goldens made libm-tolerant (uniform stays
bit-exact).

---

## Milestone 2 — CUDA Backend Foundation

**Overview.** Bring up the CUDA layer: build-system integration, error handling,
streams/events, device memory allocation (including the caching pool allocator that
becomes the engine's memory-pooling backbone), host↔device transfers, and the kernel
infrastructure with first trivial kernels. After this milestone, `Tensor` works on GPU
and every later kernel ticket has a paved road: dispatch helpers, test fixtures, and a
correctness-vs-CPU testing pattern.

> **Status at the pivot (2026-08-07).** This milestone was completed and audited, but
> its CUDA-specific deliverables are **retired by M3-T02**: no CUDA toolkit ever
> compiled the device-side code, and the project pivoted to CPU-first (ADR-004,
> M3-T01). What lives on: the device-agnostic **caching pool allocator** (M2-T06 —
> becomes the KV block pool's backing store), the `check-tidy` compile-database
> discipline, the source-seam and oracle-testing patterns (reused for SIMD dispatch
> in M3), and the M1 `Device` abstraction. The tickets below are preserved verbatim
> as project history; their design doc moves to `docs/design/retired/cuda-backend.md`
> (M3-T01), kept as the starting point for any future GPU backend.

**Architecture documents:** `docs/design/cuda-backend.md` (M2-T01).

### M2-T01 · Design doc: CUDA backend — ✅ DONE (2026-08-04)
Write `docs/design/cuda-backend.md`: stream model (which streams exist, who owns
them), error-handling strategy (CUDA errors → `Status`), allocator strategy (naive vs
caching pool, stream-ordered semantics), kernel source organization (`src/kernels/`
layout, header/impl split, dispatch conventions), supported architectures
(sm_80/86/89/90), and GPU testing strategy.
**Depends on:** M1-T09.
**Acceptance criteria:**
- Doc answers: how does a CUDA error inside a kernel surface to the caller? Who synchronizes and when? How do tests assert kernel correctness?
- Reviewed against ADR-002 module rules (kernels may not depend on scheduler/runtime).

### M2-T02 · CMake CUDA integration — ✅ DONE (2026-08-04)
Enable CUDA as a first-class language behind an `ENGINE_ENABLE_CUDA` option (default
ON, auto-detect). Set `CMAKE_CUDA_ARCHITECTURES` (80;86;89;90), C++20 for device code,
and make the CPU-only build (CI) compile cleanly with all CUDA targets excluded.
**Depends on:** M2-T01.
**Acceptance criteria:**
- With CUDA toolkit present: `.cu` files compile into `src/cuda/` and `src/kernels/` targets.
- With `ENGINE_ENABLE_CUDA=OFF` (or no toolkit): full build + tests pass; `Device::kCUDA` operations return `Unimplemented`.

### M2-T03 · CUDA error handling & device utilities — ✅ DONE (2026-08-04)
`src/cuda/cuda_utils.h`: `CUDA_CHECK` (fatal) and `CUDA_RETURN_IF_ERROR` (→ `Status`)
macros capturing file/line and error string; device introspection (`device_count()`,
`DeviceProperties` with name, SM count, memory, compute capability); `ScopedSetDevice`
RAII.
**Depends on:** M2-T02.
**Acceptance criteria:**
- GPU test: querying properties of device 0 returns sane values; deliberately bad call surfaces a `Status` with the CUDA error string embedded.
- Tests are skipped (not failed) on machines without a GPU, via a shared test predicate.

### M2-T04 · Stream & event wrappers — ✅ DONE (2026-08-04)
`src/cuda/stream.h`: RAII `CudaStream` (non-blocking), `CudaEvent` (timing and
sync variants), `record/wait/synchronize/elapsed_ms`, and a per-device default stream
accessor.
**Depends on:** M2-T03.
**Acceptance criteria:**
- GPU tests: event ordering across two streams via `stream_wait_event`, elapsed time of a known-duration workload > 0, destruction order safety (event outliving stream misuse is documented).

### M2-T05 · CUDA device allocator (naive) — ✅ DONE (2026-08-04)
`CudaAllocator` implementing the M1-T05 `Allocator` interface over
`cudaMalloc`/`cudaFree`, device-tagged `Buffer`s, and `Tensor::empty` support for
`Device::kCUDA`.
**Depends on:** M2-T04, M1-T05.
**Acceptance criteria:**
- GPU tests: allocate/free cycles leak-free (`cudaMemGetInfo` delta check), device tensors report correct device, huge-allocation failure returns `ResourceExhausted` (not crash).

### M2-T06 · Caching pool allocator — ✅ DONE (2026-08-04)
`src/memory/caching_allocator.h`: a caching allocator over the naive one — size-class
binning, free-list reuse, stats (`bytes_allocated`, `bytes_reserved`, hit/miss
counts), `release_cached()`. This is the memory-pooling foundation for the whole
engine.
**Depends on:** M2-T05.
**Acceptance criteria:**
- GPU tests: alloc→free→alloc of same size reuses the block (no cudaMalloc call — verified via stats), stats accurate through a scripted sequence, `release_cached()` returns memory to the driver.
- Concurrent allocation from multiple threads is safe (stress test).

### M2-T07 · Pinned memory & host↔device transfer — ✅ DONE (2026-08-04)
`PinnedCpuAllocator` for page-locked host memory; `copy(dst, src, stream)` supporting
H2D/D2H/D2D async transfers; `Tensor::to(device, stream)` returning a new tensor.
**Depends on:** M2-T06.
**Acceptance criteria:**
- GPU tests: round-trip H2D→D2H preserves bytes for every dtype; async copy on a stream + event sync semantics verified; pinned round-trip works.
- Copy between mismatched shapes/dtypes returns `InvalidArgument`.

### M2-T08 · Kernel launch infrastructure & first elementwise kernels — ✅ DONE (2026-08-04)
`src/kernels/`: launch-config helpers (grid/block calculation, `CUDA_1D_KERNEL_LOOP`),
a dtype-dispatch macro (`DISPATCH_FLOATING_TYPES`), and elementwise kernels: `add`,
`mul`, `scale`, and `cast` (fp32↔fp16↔bf16) operating on contiguous tensors.
**Depends on:** M2-T07.
**Acceptance criteria:**
- GPU tests: each kernel matches the CPU implementation via `allclose` across shapes (including non-multiple-of-blockDim sizes) and dtypes.
- Kernels validate inputs (shape/dtype/device match) and return `Status` on violation.

### M2-T09 · GPU test infrastructure — ✅ DONE (2026-08-04)
Shared `CudaTestFixture` (skips without GPU, sets device, provides stream +
allocator), `expect_tensors_close(gpu_tensor, cpu_reference)` helper that handles the
D2H copy, and documentation of the kernel-testing pattern in `tests/README.md`. Add an
optional CI workflow file for a self-hosted/manual GPU runner (may stay dormant).
**Depends on:** M2-T08.
**Acceptance criteria:**
- Existing GPU tests are migrated to the fixture; running the suite on a no-GPU machine reports skips, not failures.
- `tests/README.md` documents how to write a kernel test (CPU-reference pattern).

### Post-M2 hardening — ✅ DONE (2026-08-04)
Audit-driven fixes before M3, in one change. Bugs: `stream.cpp`'s event
factories routed through an anonymous-namespace free function calling
`CudaEvent`'s private constructor — a guaranteed compile error on the first
CUDA-enabled build (now a private static `CudaEvent::Create`);
`CudaEvent::ElapsedMs` on a never-recorded `Timing()` event escaped the
misuse taxonomy as an opaque `kInternal` (never-recorded counts complete
for `Query()`, then `cudaEventElapsedTime` rejects the handle) — events now
track whether `Record()` ever succeeded and `ElapsedMs` pre-checks it →
`FailedPrecondition`, with a GPU test. Tooling: the no-arg
`scripts/check-tidy.sh` sweep analyzed CUDA-only `.cpp` TUs that have no
compile-database entry on a CPU-only build (clang-tidy guessed flags,
failed on `<cuda_runtime.h>`, and would have turned CI red); it now skips
TUs absent from the compile database (listing them), rejects explicit
arguments without an entry, and fails if the filter drops everything.
Consistency: `transfer.cpp` now clears the latched CUDA last-error slot on
failure like both allocators (design §9.2's post-launch check must not
inherit stale errors). Tests: the cast GPU test now runs the full size
sweep (1 through the >1M grid-stride wrap) instead of one odd size. Docs
synced to code: removed the phantom `cuda → tensor_base` link from design
§2.1 and (as a dated correction) ADR-002 Amendment 2/3, recorded the
`device_count()` memoize-on-failure and anonymous-namespace-`__global__`
(nvcc rejects `static` on kernel templates) refinements, re-scoped the
`cuda.cu` anchor's stated lifetime (M2-T08's kernels landed in `kernels`),
fixed `tests/README.md`'s sweep-shapes step (0 is its own test), and added
load-bearing comments (UVA assumption in `ops::copy`'s identical-view
check, deleters' unconditional error-slot clear, post-launch
`cudaGetLastError` misattribution stance, `ENGINE_SKIP_WITHOUT_CUDA`
dangling-else hazard). Removed stale untracked leftovers of deleted
placeholder anchors (`src/{core,cuda,kernels,memory}/<module>.cpp`,
`docs/{adr,design}/.gitkeep`). Standing caveat, unchanged by this pass:
every GPU-only acceptance criterion in M2 is still validated by inspection
only — no CUDA toolkit has ever compiled or run this code; the first
session on a GPU machine must run the full suite (`ctest` incl. `-L gpu`)
before M3 work builds on it.

---

## Milestone 3 — The Pivot: CPU Backend Foundation

**Overview.** Execute the CPU-first pivot cleanly: record the decision (ADR-004),
excise the CUDA-specific code while keeping everything device-agnostic, and stand up
the CPU compute substrate that every later milestone builds on — a thread pool with
deterministic parallel-for, runtime SIMD dispatch (scalar/NEON/AVX2), and first
vectorized kernels validated against scalar references. This milestone plays the
role M2 played for CUDA: after it, every kernel ticket has a paved road — dispatch
helpers, threading primitives, test fixtures, and a correctness-vs-reference testing
pattern that runs *everywhere*, including CI.

**Architecture documents:** ADR-004 (the pivot), `docs/design/cpu-backend.md`
(M3-T03); ADR-002 amendment for module retirement (M3-T02).

### M3-T01 · ADR-004: the CPU-first pivot & doc restructuring — ✅ DONE (2026-08-07)
Write ADR-004: context (no NVIDIA hardware in the dev environment; post-M2 audit
findings on never-compiled code), the decision (CPU-first engine, NEON + AVX2, CPU
reference as oracle, llama.cpp-class performance target), alternatives considered
(rented cloud GPUs per milestone; Metal/MLX backend; abandoning the project),
consequences (M2's CUDA deliverables retired but preserved in history; GPU backend
becomes a future direction), and the platform matrix (macOS arm64 primary dev,
Linux x86-64 CI). Then restructure the docs so every *living* surface is purely
CPU-first and the GPU past is archival: move the v1 roadmap to
`docs/archive/ROADMAP-v1.md` (with a one-line archived-banner at the top) and
rename this file to `ROADMAP.md`; move `docs/design/cuda-backend.md` to
`docs/design/retired/` (banner pointing at ADR-004); rewrite `CLAUDE.md` to the
CPU-first scope (project description, module table, build notes — and compress the
M0–M2 status log to a short history paragraph pointing at the archive); update
`README` likewise. ADRs 001–003 stay in place untouched (they still govern; ADRs
are immutable).
**Depends on:** M2-T09.
**Acceptance criteria:**
- ADR-004 follows the template with alternatives and consequences; the rent-a-GPU and Metal options are honestly weighed, not strawmanned.
- Exactly one roadmap exists at the repository root (`ROADMAP.md`, this plan); `docs/archive/` and `docs/design/retired/` hold the v1 roadmap and CUDA design doc with banners.
- Grep check recorded in the ticket: outside `docs/archive/`, `docs/design/retired/`, the ADRs, and this roadmap's M0–M3 history sections, no doc references CUDA/GPU as a current target; CLAUDE.md's history section is ≤ 5 lines and links to the archive.

### M3-T02 · CUDA excision — ✅ DONE (2026-08-07)
Remove the CUDA-specific surface while preserving the device-agnostic core: delete
`src/cuda/`, the CUDA sources/stubs behind the source-list seams
(`cuda_allocator`, `pinned_allocator`, `transfer`, `elementwise` CUDA paths),
`cmake/cuda.cmake` + the `engine::cuda_build` target and `ENGINE_ENABLE_CUDA`
option, `tests/common/cuda.*`, the CUDA test TUs, and the dormant
`gpu-ci.yml`. Keep: `CachingAllocator` (device-agnostic — future KV block pool
backing), `CpuAllocator`, the whole tensor library (`Device` reverts to M1
semantics: `kCUDA` reserved, allocation on it → `Unimplemented`; `ops::copy`
drops its stream overload; `Tensor::to` reduces to the same-device/CPU cases).
Amend ADR-002: `cuda` module removed from the layer diagram, Amendments 2/3
edges retired, rule-3 toolkit list emptied.
**Depends on:** M3-T01.
**Acceptance criteria:**
- Full build + ctest + format + tidy green on macOS arm64 and in CI; zero references to CUDA in the build system or `src/` (grep-verified); `scripts/check-tidy.sh` no-arg sweep analyzes every TU (its compile-database filter reports nothing skipped).
- Removed tests are enumerated in the commit message; surviving test count is stated and every surviving test passes unchanged.
- ADR-002 amendment records the retirement; the module dependency diagram matches the actual CMake link graph.

> **Audit note (2026-08-08):** the excision commit (`ff2b5db`) did not enumerate
> the removed tests in its message as the second criterion requires; recorded
> here instead, since `main` history is immutable. Removed test TUs (12):
> `caching_allocator_cuda_test.cpp`, `cuda_allocator_cuda_test.cpp`,
> `cuda_allocator_test.cpp`, `cuda_check_test.cpp`, `cuda_fixture_test.cpp`,
> `cuda_utils_test.cpp`, `dispatch_test.cu`, `elementwise_test.cpp` (the CUDA
> kernel suite — the name was later reused by M3-T06's CPU suite),
> `pinned_allocator_test.cpp`, `stream_cuda_test.cu`, `stream_test.cpp`,
> `transfer_test.cpp` — plus the `tests/common/cuda.*` fixture helpers.
> 285 tests survived, all passing unchanged.

### M3-T03 · Design doc: CPU backend — ✅ DONE (2026-08-07)
Write `docs/design/cpu-backend.md`: threading model (persistent pool sized to
physical cores, `parallel_for` with deterministic static partitioning, reduction
ordering rules so results are bitwise reproducible at any thread count; OpenMP
considered and rejected/accepted with rationale), SIMD strategy (runtime dispatch
over {scalar, NEON, AVX2}; per-ISA translation units with per-TU compile flags;
`ENGINE_FORCE_ISA` override for testing; scalar path always compiled everywhere),
dtype policy (fp32 accumulation everywhere; fp16/bf16 storage converted via the
M1-T07 bit-exact conversions), kernel validation methodology (scalar validates
vectorized, HF fixtures validate scalar — the v1 oracle chain, CPU-only now),
aligned-allocation and weight-layout conventions, and what runs where in CI
(x86-64: AVX2 + forced-scalar; arm64 dev machine: NEON + forced-scalar; a macOS
arm64 CI job is added if GitHub runner minutes allow — decided here).
**Depends on:** M3-T02.
**Acceptance criteria:**
- Doc answers: how is a kernel proven correct on an ISA the CI runner lacks? Who owns threads (no nested parallelism v1)? How is determinism guaranteed across thread counts?
- Reviewed against ADR-002 module rules; the `parallel` module's position in the layer diagram is recorded.

### M3-T04 · Thread pool & parallel_for — ✅ DONE (2026-08-07)
`src/parallel/`: fixed-size worker pool (default: physical core count, configurable),
`parallel_for(range, body)` with static chunking, and a deterministic
`parallel_reduce` (fixed tree order independent of thread count). No exceptions
across the boundary (Status/CHECK policy); no nested parallelism (CHECK-enforced).
**Depends on:** M3-T03.
**Acceptance criteria:**
- Unit tests: correctness across range/chunk edge cases (empty, one element, range < threads), pool restart/shutdown cleanliness, CHECK on nested use.
- Determinism test: `parallel_reduce` over adversarial fp32 inputs yields bit-identical results at thread counts {1, 2, 8}.
- TSAN job added to CI for `parallel` tests; stress test (many small parallel_fors from a loop) races clean.

### M3-T05 · SIMD dispatch infrastructure — ✅ DONE (2026-08-07)
`src/kernels/`: CPU feature detection (arm64: NEON baseline; x86-64: cpuid for
AVX2/FMA), a kernel registry binding per-ISA implementations behind one function
pointer per kernel, `ENGINE_FORCE_ISA={scalar,neon,avx2}` env override (unknown or
unavailable ISA → clear startup error), and the CMake pattern for per-ISA TUs
(`src/kernels/avx2/*.cpp` compiled with `-mavx2 -mfma`, plain TUs elsewhere — no
target-wide arch flags, so illegal-instruction bugs are structurally impossible).
**Depends on:** M3-T04.
**Acceptance criteria:**
- Unit tests: dispatch selects the expected ISA per platform; forced-scalar override verified; forcing an ISA the host lacks fails with an actionable error.
- The registry pattern is documented in the design doc with the recipe for adding a kernel (used by every kernel ticket after this).

### M3-T06 · First vectorized kernels — ✅ DONE (2026-08-08)
Elementwise `add`/`mul`/`scale`, reductions (`sum`, `max`), and fp16/bf16 ↔ fp32
conversion kernels in scalar + NEON + AVX2 variants, dispatched per M3-T05,
threaded per M3-T04 above a size threshold. Scalar variants double as the
reference; results must be bit-identical across ISAs for these ops (pure
elementwise/conversion work — documented; reductions bit-identical given the
deterministic tree).
**Depends on:** M3-T05.
**Acceptance criteria:**
- Tests: each vectorized variant matches scalar bit-exactly across sizes {1, 15, 16, 17, 4096, 1M+} including unaligned heads/tails; conversions match the M1-T07 half.h goldens; forced-scalar pass runs the same suite.
- Microbenchmark scaffold (`benchmarks/kernels/`, first version): vectorized conversion ≥ 2× scalar at 1M elements on the dev machine (advisory number, recorded in `benchmarks/BASELINES.md` — the file's first entry).

---

## Milestone 4 — Model Loading & Tokenization

**Overview.** Read real model artifacts from disk: HuggingFace `config.json`,
safetensors checkpoints (single and sharded), and `tokenizer.json` byte-level BPE
tokenizers (the format used by Llama 3 and Qwen 2+). Also establishes the
golden-fixture tooling — Python scripts that use HuggingFace libraries to produce
reference outputs, which every numerical test from here on relies on. After this
milestone the engine can load a real model's weights into tensors and tokenize text
identically to HuggingFace.

**Architecture documents:** `docs/design/model-loading.md` (M4-T01).

### M4-T01 · Design doc: model loading & tokenization — ✅ DONE (2026-08-08)
Write `docs/design/model-loading.md`: the load pipeline (config → weight discovery →
mmap → tensor registry), the internal weight-naming convention and per-architecture
mapping tables, dtype policy at load (preserve checkpoint dtype), tokenizer scope
(byte-level BPE from `tokenizer.json`; sentencepiece explicitly deferred), and the
golden-fixture strategy.
**Depends on:** M3-T06.
**Acceptance criteria:**
- Doc includes the load-pipeline diagram and the tokenizer scope with rationale.
- Fixture strategy specifies where fixtures live (`tests/fixtures/`), how they're generated, and their size budget.

### M4-T02 · Golden-fixture generation tooling — ✅ DONE (2026-08-08)
`tools/gen_fixtures/`: Python package (pinned `requirements.txt`: torch, transformers,
tokenizers, safetensors) with a CLI that generates: (a) tokenizer encode/decode test
vectors for a given HF model, (b) a tiny random-weight model checkpoint
(safetensors + config) with recorded layer-by-layer and end-to-end outputs for a
fixed input. Deterministic via fixed seeds.
**Depends on:** M4-T01.
**Acceptance criteria:**
- Running the CLI twice produces byte-identical fixtures; a `Makefile`/script target regenerates everything.
- A tiny-Llama-style fixture (2 layers, small dims) + tokenizer vectors for one Llama-family and one Qwen-family tokenizer are committed under `tests/fixtures/` (small enough for git).
- `tools/README.md` documents usage; CI does not require Python (fixtures are committed).

### M4-T03 · Model config parser — ✅ DONE (2026-08-08)
`src/model/config.h`: parse HF `config.json` into `ModelConfig` (architecture string,
hidden/intermediate size, layer/head/kv-head counts, head dim, vocab size, RoPE theta
& scaling, norm epsilon, max position, tie-word-embeddings, torch_dtype), with
validation and unknown-field tolerance.
**Depends on:** M4-T02.
**Acceptance criteria:**
- Unit tests parse committed real config.json files (Llama-3-style, Qwen2-style) and assert every field; malformed/missing-required-field cases return `InvalidArgument` with the field name.

### M4-T04 · Safetensors file parser — ✅ DONE (2026-08-08)
`src/model/safetensors.h`: mmap a `.safetensors` file, parse the JSON header, expose
`{name → (dtype, shape, data span)}` with zero-copy views into the mapping. Handle
alignment, bounds validation, and dtype string mapping (F32/F16/BF16/I8/…).
**Depends on:** M4-T03.
**Acceptance criteria:**
- Unit tests against a fixture file: metadata correct, tensor bytes match expected values, truncated/corrupt header cases return errors (fuzz-ish negative tests).
- File stays mapped while any tensor view is alive (lifetime tested).

### M4-T05 · Sharded checkpoint support — ✅ DONE (2026-08-08)
Support `model.safetensors.index.json`: resolve tensor→shard mapping, open shards
lazily, present a unified `{name → tensor view}` interface identical to the
single-file case.
**Depends on:** M4-T04.
**Acceptance criteria:**
- Unit tests with a 2-shard fixture: all tensors resolvable, missing-shard and inconsistent-index errors surfaced clearly.

### M4-T06 · Weight-name mapping — ✅ DONE (2026-08-08)
`src/model/weight_map.h`: per-architecture mapping from HF checkpoint names
(`model.layers.0.self_attn.q_proj.weight`) to internal canonical names, with a report
of missing and unexpected weights at load.
**Depends on:** M4-T05.
**Acceptance criteria:**
- Unit tests: full mapping for the tiny-Llama fixture resolves every weight; deliberately removing a weight produces a load error naming it; extra weights produce a warning list.

### M4-T07 · Model loader — ✅ DONE (2026-08-08)
`src/model/loader.h`: `load_model(path) → StatusOr<LoadedModel>` combining config
parse, shard resolution, name mapping, and materializing weights as CPU tensors
(dtype preserved from checkpoint). Progress logging for large models.
**Depends on:** M4-T06.
**Acceptance criteria:**
- Integration test loads the tiny fixture end-to-end; spot-check tensor values against fixture-recorded expectations.
- Load errors (bad path, unsupported architecture) produce actionable messages.

### M4-T08 · Tokenizer model parsing — ✅ DONE (2026-08-08)
`src/tokenizer/`: parse `tokenizer.json` — vocab, merges, added/special tokens
(with `special`, `lstrip`/`rstrip` flags), byte-level alphabet mapping. Build the
in-memory structures for encoding (merge ranks) and decoding (id → token bytes).
**Depends on:** M4-T02 (fixtures).
**Acceptance criteria:**
- Unit tests: vocab size, specific token↔id pairs, special-token metadata for both fixture tokenizers.
- Unsupported tokenizer types (sentencepiece/unigram) rejected with a clear error.

### M4-T09 · BPE encoding — ✅ DONE (2026-08-08)
Implement byte-level BPE encode: pre-tokenization regex split (the GPT-2/Llama-3
pattern), byte-to-unicode mapping, merge loop, special-token splitting
(added tokens are matched before BPE). API: `encode(text, add_special_tokens) →
vector<int32>`.
**Depends on:** M4-T08.
**Acceptance criteria:**
- Golden tests: encodings byte-identical to HF `tokenizers` for the committed test vectors (ASCII, Unicode incl. CJK + emoji, whitespace edge cases, special tokens embedded in text) for both tokenizer fixtures.

### M4-T10 · Decoding & incremental detokenization — ✅ DONE (2026-08-08)
Implement `decode(ids, skip_special_tokens)` and an incremental
`DetokenizerStream` that emits valid UTF-8 as tokens arrive (buffering incomplete
multi-byte sequences) — required later for streaming generation.
**Depends on:** M4-T09.
**Acceptance criteria:**
- Golden tests: decode(encode(x)) == x for all test vectors; streaming decode emits identical total text with no invalid UTF-8 at any intermediate step (tested token-by-token, including a multi-token emoji).

> **Audit note (2026-08-08):** the post-milestone audit found this ticket's two
> golden acceptance loops (round-trip, token-by-token streaming) vacuous — a
> C++20 dangling-temporary range-for iterated zero times, so those criteria
> were unverified (and qwen2 streaming unexercised) until the audit fix. Also
> fixed in the same change: the M4-T04 safetensors alignment check is now
> absolute (`8 + header_len + begin`, not `begin` alone — an unpadded header
> shifts the data section); plus new loader missing-weight, weight-map
> warning-capture, and byte-alphabet-bijection tests. Details:
> docs/PROGRESS.md "M4 audit fixes".

---

## Milestone 5 — CPU Reference Engine

**Overview.** Implement a complete, unoptimized scalar forward pass and greedy
generation loop for Llama/Qwen-family models. This is the **correctness oracle**:
every optimized/vectorized kernel from now on is validated against it (and it, in
turn, is validated against HuggingFace fixtures). Speed is a non-goal; clarity is the
goal. This milestone also fixes the model-execution interfaces (module structure,
KV-cache interface) that the optimized backend (M6) will implement.

**Architecture documents:** `docs/design/model-execution.md` (M5-T01).

### M5-T01 · Design doc: model execution — ✅ DONE (2026-08-17)
Write `docs/design/model-execution.md`: the layer/module structure (Attention, MLP,
DecoderLayer, Model), how weights bind to modules, the forward-pass signature (token
ids + positions + KV cache → logits), batch/sequence dimension conventions, the
KV-cache interface (v0: append-only per-sequence), and how the reference and
optimized backends share the same interfaces.
**Depends on:** M4-T10.
**Acceptance criteria:**
- Doc specifies the exact `Model::forward` contract used by both backends, and the GQA layout conventions (head counts, head_dim, kv repeat).
- KV-cache interface v0 is specified with an explicit note on what M8 (paged) will change.

### M5-T02 · CPU GEMM & Linear layer — ✅ DONE (2026-08-17)
`src/cpu/`: a correct (`parallel_for`-threaded, cache-blocked but simple) fp32 GEMM;
weights in fp16/bf16 are converted to fp32 on the fly. `Linear` module (weight,
optional bias).
**Depends on:** M5-T01.
**Acceptance criteria:**
- Unit tests vs fixture GEMM results across shapes (including k=1, skinny/wide) within fp32 tolerance.
- A 512×512×512 GEMM completes in < 1 s (sanity, not perf).

### M5-T03 · CPU normalization & activation ops — ✅ DONE (2026-08-17)
RMSNorm (with epsilon, weight), SiLU, elementwise multiply (for SwiGLU), residual
add, and numerically-stable softmax — all CPU, fp32 accumulation.
**Depends on:** M5-T02.
**Acceptance criteria:**
- Golden tests vs fixture outputs for each op (including RMSNorm on bf16 inputs, softmax with large-magnitude logits).

### M5-T04 · Embedding & RoPE (CPU) — ✅ DONE (2026-08-17)
Token-embedding lookup; rotary position embeddings matching HF Llama exactly
(half-rotation layout, configurable theta, optional scaling factors parsed in M4-T03).
Precompute cos/sin tables.
**Depends on:** M5-T03.
**Acceptance criteria:**
- Golden tests: RoPE output matches fixture for positions {0, 1, large}, head dims from config; embedding lookup matches fixture rows.

### M5-T05 · Causal attention (CPU) — ✅ DONE (2026-08-17)
Naive causal self-attention with GQA: project QKV, apply RoPE, scores = QKᵀ/√d with
causal mask, softmax, weighted sum, output projection. Supports prefill (T tokens)
against an existing cache of length P.
**Depends on:** M5-T04.
**Acceptance criteria:**
- Golden tests vs fixture attention-layer outputs (prefill from empty cache, and prefill continuing from a non-empty cache) with GQA (kv_heads < heads) covered.

### M5-T06 · KV cache v0 — ✅ DONE (2026-08-17)
`src/kvcache/simple_cache.h`: per-sequence, per-layer contiguous append-only K/V
storage implementing the interface from M5-T01 (append, view, current length, reset).
CPU tensors for now; device-agnostic API.
**Depends on:** M5-T05.
**Acceptance criteria:**
- Unit test: decoding token-by-token with the cache produces logits equal to full-prompt recompute at every step (the fundamental KV-cache invariant), within fp32 tolerance.

> **Note (2026-08-17):** `Model::forward` is M5-T07, so the invariant landed at
> the **attention-chain** level (norms/MLP/residuals don't touch the cache) —
> full-prefill vs token-by-token, chunked prefill, and truncate-then-redecode,
> all bit-exact (max_abs_diff = 0). M5-T07 elevates the same check to
> full-model logits. `view()` gathers a contiguous snapshot rather than
> returning a zero-copy slice (`cpu::attention` needs contiguous K/V; this is
> the M8 gather seam) — design §6.2 updated.

### M5-T07 · Transformer block & full model forward (CPU) — ✅ DONE (2026-08-17)
Assemble `DecoderLayer` (attention + MLP + norms + residuals) and `Model` (embedding →
N layers → final norm → lm_head, honoring tied embeddings). Prefill returns logits for
the last position (and optionally all positions for testing).
**Depends on:** M5-T06.
**Acceptance criteria:**
- Golden test: end-to-end logits for the tiny-Llama fixture match HF within tolerance (report max abs diff; threshold documented in the test).
- Per-layer debug hook allows dumping intermediate activations (used to localize future regressions).

> **Note (2026-08-17):** landed `Mlp`/`DecoderLayer` (`model/modules.{h,cpp}`),
> the `Model`/`ActivationHook`/`ForwardRequest` contract (`model/model.h`), and
> `ReferenceModel` (`model/reference_model.{h,cpp}`). End-to-end tiny-llama
> logits match `activations.safetensors` at max_abs_diff = 3.7e-6 (tol 2e-4);
> the per-layer hook emits the fixture-named stages (`embeddings`/`layers.{i}`/
> `final_norm`/`logits`) and the KV invariant now holds at full-model logits,
> bit-exact. **`model → kvcache` became a PUBLIC CMake edge** because `model.h`
> returns `kvcache::CacheGeometry` by value — ADR-002 Amendment 5 clarified.
> `linear_input:` hook events are deferred to M14-T02 (they need the hook
> threaded through `Linear`/`Attention`/`Mlp`); the four stage events land here.

### M5-T08 · Architecture registry — ✅ DONE (2026-08-17)
`src/model/registry.h`: map `architectures[0]` strings (`LlamaForCausalLM`,
`Qwen2ForCausalLM`, …) to model builders; unsupported architectures produce a clean
error listing supported ones. Builder wires `ModelConfig` + weights → `Model`.
**Depends on:** M5-T07.
**Acceptance criteria:**
- Unit tests: registry resolves both families, rejects unknown; adding an architecture requires only a registration call (verified by a test-local dummy arch).

> **Note (2026-08-17):** `src/model/registry.{h,cpp}` — `BuildModel(LoadedModel,
> BuildOptions)` dispatching on `config.architecture_name`, `RegisterArchitecture`
> (returns `Status`: `AlreadyExists`/`InvalidArgument`), `SupportedArchitectures()`.
> One `BuildReferenceFamily` builder serves both Llama and Qwen2 (the diff is
> `attention_bias` + config values, both already handled by
> `ReferenceModel::Create`) and **both are registered in T08** — M5-T10 adds only
> the Qwen fixture. **`enum class Backend` was relocated from the sketched
> `engine/backend.h` into `registry.h`** because `BuildOptions` names it and
> `model` cannot depend on `engine` (ADR-002); `kOptimized` → `Unimplemented`
> until M6. Built-ins register lazily inside a mutex-guarded `GetRegistry`
> function-local static (a static-lib TU of self-registering globals would be
> dropped by the linker). +7 tests (701 green); design §2/§8/§9/§13 updated.

### M5-T09 · Greedy generation loop — ✅ DONE (2026-08-18)
`src/engine/generator.h`: prefill + autoregressive decode loop with greedy argmax,
EOS-token and max-new-tokens stopping. Returns generated ids; hooks for per-token
callbacks (streaming later).
**Depends on:** M5-T08.
**Acceptance criteria:**
- Golden test: greedy continuation of fixture prompts matches HF `generate(do_sample=False)` token-for-token for ≥ 32 tokens on the tiny model.
- Determinism test: two runs produce identical outputs.

> **Note (2026-08-18):** landed `src/engine/generator.{h,cpp}` (`Generate` —
> prefill in one `kLast` forward, argmax with **lowest-index tie-break**, decode
> one token per forward at the running cache position, `on_token` callback firing
> once per id after its append and before the next forward; EOS ids included in
> the output; front-loaded validation — empty prompt / `max_new_tokens ≤ 0` →
> InvalidArgument, and an **up-front capacity check** → ResourceExhausted since a
> `StatusOr<vector>` cannot carry a partial result) and `src/engine/backend.h`
> (`Backend`/`BuildOptions` re-export + header-only `BackendName`/`ParseBackend`;
> the `engine.cpp` anchor removed). New `ModelConfig::eos_token_ids` (M4-config
> addition the loop needs: HF's int-or-list `eos_token_id`, validated in
> `[0, vocab_size)`). New `tiny-llama/expected/generate.json` (`tiny-llama-generate`
> subcommand): three **well-separated** prompts (min top-2 logit gap > 1e-2 —
> four orders above the ~4e-6 HF-vs-reference logit diff), EOS suppressed so each
> reaches 40 tokens, HF `generate` cross-checked against a manual KV loop. The
> committed activation prompt is *ill-conditioned* (sub-1e-3 gaps make the greedy
> trajectory untestable across numerically-close paths) and deliberately not
> reused. +20 tests (721 green): golden token-for-token match, determinism, EOS/
> max-new-tokens stopping, callback ordering, non-empty-cache continuation (KV
> invariant), error paths, backend helpers, and 7 `eos_token_id` config tests.
> design §10/§12 updated.

### M5-T10 · Qwen-family support (CPU) — ✅ DONE (2026-08-18)
Implement Qwen2/2.5 differences (QKV bias, its config fields) on the shared modules;
add a tiny Qwen-style fixture.
**Depends on:** M5-T09.
**Acceptance criteria:**
- Golden logits + greedy-generation tests pass for the Qwen fixture, reusing existing modules (diff should be config/wiring, not new layer code).

> **Note (2026-08-18):** the ticket was **fixtures + tests, zero `src/` change** —
> the M5 modules, config parser, weight map, `ReferenceModel::Create`, and registry
> already carried every Qwen difference (per-arch `attention_bias` default, q/k/v
> biases with a bias-free o_proj, a `head_dim` decoupled from `hidden_size/heads`,
> tied embeddings), landed across M5-T02…T08 as the design promised. New
> `tools/gen_fixtures/tiny_qwen2.py` (`tiny-qwen2`) emits a 2-layer
> `Qwen2ForCausalLM` mirror of tiny-llama chosen to be distinct on every
> Qwen-relevant axis: **`head_dim=24` ≠ 64/4=16** (exercises the decoupled path §3.1
> that tiny-llama can't), **`attention_bias` omitted from config.json** (so the
> parser's per-arch default is what wires the biases end-to-end), **tied
> embeddings** (`lm_head.weight` dropped from the checkpoint — safetensors refuses
> the shared storage — reconstituted via the loader alias), Qwen2 `rope_theta=1e6`/
> `rms_norm_eps=1e-6`, no BOS. The q/k/v biases are filled with fixed non-zero noise
> (HF `_init_weights` zeros them, which would make the bias path invisible in the
> golden). New `tiny_qwen2_generate.py` (`tiny-qwen2-generate`) writes greedy
> `generate.json` for three **well-separated** prompts (min top-2 logit gap now
> **asserted** > 1e-2, a tightening over T09's diagnostic-only check; observed
> ≥0.10), HF `generate` cross-checked vs a manual KV loop. New
> `tests/unit/qwen2_family_test.cpp` (+8, → 729 green): config/loader wiring
> assertions, registry routing + bit-identical-to-direct build, end-to-end logits
> vs golden (max_abs_diff 3.9e-6, tol 2e-4), **biases load-bearing** (dropping them
> moves logits by ~0.98), greedy match/determinism/KV-invariant. `regen_fixtures.sh
> --verify` byte-clean; format + scoped tidy clean.

---

## Milestone 6 — Optimized CPU Execution Engine

**Overview.** Make the forward pass fast: packed-weight multithreaded GEMM,
vectorized layer kernels, and blocked attention — every optimized kernel validated
against the M5 scalar reference (which M5 validated against HuggingFace). Ends with
end-to-end generation on a real model (~1B class) at usable speed and the first
tokens/second baseline. This is the analog of v1's single-GPU milestone: deliberately
straightforward optimized kernels behind stable interfaces — the aggressive tuning
pass comes in M12 behind the same interfaces.

**Architecture documents:** `docs/design/optimized-cpu-execution.md` (M6-T01).

### M6-T01 · Design doc: optimized CPU execution — ✅ DONE (2026-08-18)
Write `docs/design/optimized-cpu-execution.md`: weight layout policy (checkpoint
tensors repacked at load into cache-blocked tiles; layout documented per kernel),
dtype policy (weights held fp16/bf16-packed, fp32 accumulation everywhere; where
conversion happens), threading integration (which loops parallelize — rows, heads,
query blocks; one pool, no oversubscription), workspace strategy (per-thread scratch
sized once, reused across layers), backend selection (`reference` vs `optimized`
behind the M5 interfaces, selectable for tests), and the kernel-validation
methodology (every optimized kernel tested against the scalar reference; tolerance
stated per kernel since fp32-accumulation order differs).
**Depends on:** M5-T10.
**Acceptance criteria:**
- Doc specifies the packed tile layout with a worked example, the workspace sizing formula, and where bitwise-vs-tolerance equality is expected and why.

### M6-T02 · Packed-weight GEMM & GEMV — ✅ DONE (2026-08-18)
`src/kernels/`: cache-blocked, register-tiled, `parallel_for`-threaded GEMM
(activations fp32 × packed fp16/bf16 weights, fp32 accumulate) and a decode-shaped
GEMV path (batch of single rows). Load-time weight repacking into the documented
tile layout.
**Depends on:** M6-T01.
**Acceptance criteria:**
- Tests vs the M5 reference GEMM across model shapes (qkv/o/gate/up/down/lm_head, k=1, skinny/wide) within documented tolerance, all ISAs plus forced-scalar.
- Bench: ≥ 5× the M5 naive GEMM at 4096×4096×4096 on the dev machine (advisory), recorded in BASELINES.md; a sanity comparison against Accelerate/BLAS on the same shape is documented for context (not a target).

### M6-T03 · Vectorized norm, activation & RoPE kernels — ✅ DONE (2026-08-18)
NEON/AVX2 variants of RMSNorm (fp32 accumulation), SiLU-and-mul (SwiGLU combine),
residual add, numerically-stable softmax, and RoPE application (arbitrary per-token
positions, supporting later batched/paged use) — all behind the M3-T05 dispatch.
**Depends on:** M6-T02.
**Acceptance criteria:**
- Tests vs scalar reference per kernel: hidden sizes {odd, 1024, 4096}, large-magnitude softmax inputs, RoPE positions {0, 1, large} and GQA head counts; tolerances documented per kernel.

### M6-T04 · Optimized prefill attention — ✅ DONE (2026-08-18)
Blocked causal attention with online softmax (flash-style, CPU): tiled QKᵀ and AV
over key blocks with running max/sum renormalization, fp32 accumulation, GQA via KV
head indexing (no materialized repeat), threaded across (head, query-block) pairs.
Supports prefill continuing from a non-empty cache.
**Depends on:** M6-T03.
**Acceptance criteria:**
- Tests vs the M5 reference attention: prefill from empty and non-empty cache, GQA configs, sequence lengths {1, 17, 512, 2048}, tolerance documented.
- Bench: time vs the reference attention at 2k context recorded in BASELINES.md.

### M6-T05 · Optimized decode attention — ✅ DONE (2026-08-18)
Single-token decode path: one query attending over the cached K/V per sequence,
vectorized dot-products and weighted sums, threaded across heads (and across
sequences when batched later).
**Depends on:** M6-T04.
**Acceptance criteria:**
- Tests vs reference across cache lengths {1, 63, 64, 65, 2048} and GQA configs; matches the prefill path's result for the same token.

### M6-T06 · Embedding & logits path
Embedding-lookup (ids → packed rows, handling repacked layouts) and the lm_head
GEMM/GEMV producing fp32 logits; tied embeddings honored (lookup and projection
share storage across different layouts — resolved in the design doc).
**Depends on:** M6-T05.
**Acceptance criteria:**
- Tests: lookup matches reference for random id sets (incl. repeated ids); logits match the reference within tolerance on the fixture model.

### M6-T07 · Optimized model forward & generation
Wire the optimized kernels into the M5 `Model` interface as the `optimized` backend:
weight repacking at load (with progress logging), workspace allocation, full
prefill + decode forward, greedy loop reused from M5.
**Depends on:** M6-T06.
**Acceptance criteria:**
- Tiny-fixture greedy generation on the optimized backend matches the reference backend token-for-token; end-to-end logits within documented tolerance.
- A real ~1B model (e.g. Llama-3.2-1B or Qwen2.5-0.5B, bf16) loads and generates coherent text on the dev machine (sample outputs committed to the PR description).

### M6-T08 · Generation benchmark & first baseline
`benchmarks/bench_generate`: measures prefill tokens/sec and decode tokens/sec for a
given model, prompt length, and thread count; prints a small report.
**Depends on:** M6-T07.
**Acceptance criteria:**
- Benchmark runs on the 1B model and reports stable numbers (±5% across runs); baseline recorded in `benchmarks/BASELINES.md` with hardware and thread-count fingerprint.
- A same-model llama.cpp number on the same machine is recorded alongside for context (methodology documented; parity is a goal for M12, not here).

---

## Milestone 7 — Sampling & Generation Controls

**Overview.** Replace greedy-only generation with a full sampling pipeline:
temperature, top-k, top-p, penalties, seeded per-request RNG, stop conditions, and
logprobs. Implemented first as a clear single-sequence pipeline with correctness and
statistical tests; vectorized batched sampling closes the milestone. These are
exactly the knobs the API layer (M10) will expose.

**Architecture documents:** sampling pipeline section added to
`docs/design/model-execution.md` (M7-T01).

### M7-T01 · SamplingParams & pipeline skeleton
Define `SamplingParams` (temperature, top_k, top_p, repetition/presence/frequency
penalties, seed, max_tokens, stop tokens/strings, logprobs count) with validation;
document the pipeline stage order in the design doc; implement the greedy path through
the new pipeline structure.
**Depends on:** M6-T08.
**Acceptance criteria:**
- Validation unit tests (rejects temperature<0, top_p∉(0,1], etc.).
- Greedy generation output unchanged (regression test passes).

### M7-T02 · Temperature, top-k, top-p sampling
Implement the core sampler over the final-position logits: temperature scale →
top-k filter → top-p (nucleus) filter → categorical sample with a per-request
counter-based RNG (Philox) so results are reproducible per (seed, step) and
independent of batch composition.
**Depends on:** M7-T01.
**Acceptance criteria:**
- Statistical tests: empirical distribution over 10k draws matches expected within chi-square tolerance for known logits; top-k/top-p masks verified exactly.
- Same seed ⇒ identical sequence across runs; different requests with different seeds are independent.

### M7-T03 · Repetition, presence & frequency penalties
Apply penalties over the request's token history (prompt + generated, matching
OpenAI/vLLM semantics — documented choice) before temperature.
**Depends on:** M7-T02.
**Acceptance criteria:**
- Unit tests with hand-computed logit adjustments for each penalty type and combinations; no-op when at default values (exact logits equality).

### M7-T04 · Stop conditions & finish reasons
EOS handling (including multi-EOS token sets from config), stop-token ids, stop
strings (matched on the incrementally-detokenized stream, handling stop strings that
span token boundaries), max_tokens; produce `finish_reason` (stop/length).
**Depends on:** M7-T03, M4-T10.
**Acceptance criteria:**
- Unit tests: stop string split across two tokens is caught and trailing text is trimmed; max_tokens exact; finish_reason correct in each case.

### M7-T05 · Logprobs
Return chosen-token logprob and top-N logprobs per step (computed from the same
logits the sampler saw, post-penalties or raw — documented, matching OpenAI
semantics).
**Depends on:** M7-T04.
**Acceptance criteria:**
- Unit tests: logprobs sum ≈ 1 in prob space over full vocab on a small-vocab fixture; top-N ordering matches reference computation; greedy chosen-token logprob equals max.

### M7-T06 · Batched sampling optimization
Optimize the hot path for batch-of-sequences sampling: vectorized softmax/filtering
(reusing the M3/M6 kernel infrastructure), partial-sort top-k instead of full sort,
`parallel_for` across sequences, per-request params and Philox states; the simple
single-sequence path retained as reference.
**Depends on:** M7-T05.
**Acceptance criteria:**
- Optimized sampler picks identical tokens to the reference sampler given identical RNG counters (bit-exact filtered distributions, exact tie-breaks — documented).
- Microbench: batched sampling for 64 sequences × 128k vocab under 5 ms on the dev machine (number recorded, threshold advisory).

---

## Milestone 8 — Paged KV Cache & Block Manager

**Overview.** Replace the contiguous per-sequence cache with a paged (block-based) KV
cache — the memory architecture that makes continuous batching and prefix caching
possible. Fixed-size token blocks are allocated from a host-memory block pool (backed
by the M2 caching allocator); sequences hold block tables mapping logical positions to
physical blocks. Attention kernels read through the indirection. Designed from day one
with reference counting so M11 (prefix caching) is an extension, not a rewrite.

**Architecture documents:** `docs/design/paged-kv-cache.md` (M8-T01).

### M8-T01 · Design doc: paged KV cache
Write `docs/design/paged-kv-cache.md`: physical layout
(`[num_blocks, 2, layer… ] `— choose and justify K/V layout per block for kernel
access patterns), block size choice (16 tokens default, rationale), capacity
calculation (`--kv-cache-memory` budget — absolute bytes or a fraction of host RAM —
after weights + workspace), block tables, slot mapping, reference counting, and the
preemption + future prefix-caching interactions.
**Depends on:** M7-T06.
**Acceptance criteria:**
- Doc specifies exact memory layout with a worked example and the formula for blocks-per-pool; refcount lifecycle diagram included.

### M8-T02 · Block pool & allocator
`src/kvcache/block_pool.h`: pool sized from config/free memory, free-list
allocate/free of block ids, per-block refcounts, stats (used/free/total). Pure
bookkeeping, fully unit-testable; block storage drawn from the M2 caching allocator.
**Depends on:** M8-T01.
**Acceptance criteria:**
- Unit tests: exhaustion returns `ResourceExhausted` (no crash); refcount double-free detected; stats accurate through scripted alloc/free/share sequences.

### M8-T03 · Block table & sequence cache handle
`src/kvcache/block_table.h`: per-sequence logical→physical mapping, append-token
(allocating blocks on boundary crossings), slot-mapping computation for a batch of
token positions, free-on-completion.
**Depends on:** M8-T02.
**Acceptance criteria:**
- Unit tests: growth across block boundaries, slot mappings for prefill (T tokens) and decode (1 token) hand-verified, blocks returned to pool on free.

### M8-T04 · KV write (scatter) kernel
Kernel writing a batch of new K/V vectors into paged storage given a slot mapping
(one entry per token). Replaces the contiguous append path.
**Depends on:** M8-T03.
**Acceptance criteria:**
- Tests: scattered writes land in the exact expected block/offset (readback comparison vs an independently-simulated paged layout), across boundary-straddling prefills and single-token decodes.

### M8-T05 · Paged decode attention kernel
Decode attention reading K/V through the block table (block-table pointer array per
sequence), GQA, fp32 accumulation.
**Depends on:** M8-T04.
**Acceptance criteria:**
- Tests: matches the M6-T05 contiguous decode kernel results exactly (same inputs materialized both ways) for cache lengths crossing many blocks, including length exactly at a block boundary.

### M8-T06 · Paged prefill attention path
Prefill attention over paged cache: gather cached K/V to a contiguous workspace for
the M6 blocked prefill path (simple, correct; fully paged-aware prefill comes with
the M12 tuning pass). Handles prefill-continuation (cache hit + new tokens)
correctly.
**Depends on:** M8-T05.
**Acceptance criteria:**
- Tests vs the reference backend: prefill with existing paged cache content matches; the gather routine independently tested.

### M8-T07 · Engine integration
Swap the paged cache into the single-request generation path behind the M5 cache
interface; wire capacity config (`--kv-cache-memory`); expose cache stats.
**Depends on:** M8-T06.
**Acceptance criteria:**
- End-to-end regression: tiny-fixture greedy output identical to pre-paging engine; 1B-model generation works with memory stats logged.
- `bench_generate` shows no more than 10% decode-throughput regression vs M6 baseline (recorded).

### M8-T08 · Exhaustion behavior & metrics
Defined behavior when the pool runs dry mid-generation (error for now — preemption
arrives with the scheduler in M9); cache usage metrics API (blocks used/free,
utilization) consumed later by scheduler and metrics endpoint.
**Depends on:** M8-T07.
**Acceptance criteria:**
- Test: a generation that would exceed capacity fails gracefully with `ResourceExhausted`, pool state fully reclaimed afterwards (no leaked blocks — asserted via stats).

---

## Milestone 9 — Continuous Batching Scheduler & Runtime

**Overview.** The engine becomes a multi-request system: an asynchronous runtime
accepts requests into a queue; a scheduler decides, every step, which sequences to
prefill and which to decode under a token budget and KV-block availability; batched
kernels execute ragged batches; results stream back per-request. Includes preemption
(evict + recompute) when memory runs out, and cancellation. This is the architectural
heart of the engine — the design doc matters more here than anywhere else.

**Architecture documents:** `docs/design/scheduler-runtime.md` (M9-T01).

### M9-T01 · Design doc: request lifecycle, scheduler & runtime
Write `docs/design/scheduler-runtime.md`: request/sequence state machine
(WAITING→RUNNING→(PREEMPTED)→FINISHED), the step loop, scheduling policy v1 (FCFS,
token budget, block-availability admission, decode-priority), batch composition
(separate prefill/decode passes vs mixed — choose and justify), threading model
(client threads → lock-free-ish queue → single engine thread → per-request output
channels), preemption policy (evict-and-recompute), and cancellation semantics.
**Depends on:** M8-T08.
**Acceptance criteria:**
- State machine diagram with every legal transition; step-loop pseudocode; explicit invariants (e.g. "a RUNNING sequence always holds all blocks it needs for its next token").

### M9-T02 · Request & sequence abstractions
`src/runtime/request.h`: `Request` (id, prompt ids, `SamplingParams`, arrival time),
`Sequence` (state, token ids, block table handle, generation progress), and a
thread-safe per-request output channel (tokens + finish info) supporting
blocking and polling consumption.
**Depends on:** M9-T01.
**Acceptance criteria:**
- Unit tests: state transitions enforced (illegal transition = CHECK failure), output channel delivers in order across threads, channel close semantics on finish/cancel.

### M9-T03 · Engine API & request queue
`src/runtime/engine.h`: public async API — `submit(request) → RequestHandle`,
`handle.next_token()/await_completion()`, `cancel(id)`; internal waiting queue;
`Engine::step()` skeleton that the loop thread will drive.
**Depends on:** M9-T02.
**Acceptance criteria:**
- Unit tests with a mock model: submit/consume/cancel from multiple client threads; no deadlocks under a stress test (many submitters, random cancels).

### M9-T04 · Scheduler v1
`src/scheduler/scheduler.h`: pure decision component — given queue + running set +
block-pool stats + token budget, emit `SchedulerOutput`: sequences to prefill (with
lengths), sequences to decode, sequences to preempt. FCFS admission, decode-first
priority, block-availability check via M8-T02 stats. No engine/backend
dependencies — deterministic and unit-testable.
**Depends on:** M9-T03.
**Acceptance criteria:**
- Table-driven unit tests: admission blocked when blocks insufficient, token budget respected across mixed prefill sizes, decode starvation impossible (decodes always scheduled first), preemption chooses the documented victim (latest-arrived).

### M9-T05 · Batch assembly
`src/engine/batch.h`: build the flattened batch inputs for a `SchedulerOutput` —
concatenated token ids, positions, sequence start offsets (cu_seqlens), slot mappings,
block-table tensor, per-request sampling metadata — assembled in one pass into
preallocated staging buffers (no per-step allocation).
**Depends on:** M9-T04.
**Acceptance criteria:**
- Unit tests: assembled metadata hand-verified for scenarios (2 prefills of different lengths, 3 decodes, mixed), tensor contents exact.

### M9-T06 · Varlen batched prefill attention
Extend prefill attention to ragged batches using cu_seqlens (per-sequence lengths,
shared kernels loop over sequences; still naive-but-correct).
**Depends on:** M9-T05.
**Acceptance criteria:**
- Tests: batch of {3 sequences, lengths 5/64/129} matches per-sequence single runs exactly (same outputs sequence-by-sequence).

### M9-T07 · Batched decode execution
Batched decode step: N sequences × 1 token through the full model using the paged
decode kernel over a batched block-table tensor; batched sampling (M7-T06) consumes
the result.
**Depends on:** M9-T06.
**Acceptance criteria:**
- Tests: batched decode logits match sequential single-sequence decode for each member; batch with heterogeneous cache lengths covered.

### M9-T08 · Engine loop integration
The continuous-batching loop: engine thread runs `step()` — schedule → assemble →
forward (prefills then decodes, or combined per design doc) → sample → append tokens →
deliver to channels → retire finished sequences. Streaming tokens flow to handles as
they're produced.
**Depends on:** M9-T07.
**Acceptance criteria:**
- Integration test: 8 concurrent greedy requests produce outputs identical to running each sequentially (the continuous-batching correctness invariant).
- Requests arriving mid-flight join batching without disturbing running sequences (test with staggered submission).
- Throughput sanity: 8 concurrent requests complete in well under 8× single-request time (recorded).

### M9-T09 · Preemption & recomputation
On block exhaustion during decode: preempt victim sequences (free their blocks,
state→PREEMPTED, back to queue head), resume later by re-prefilling
prompt+generated-so-far. Wire scheduler preemption decisions to engine actions.
**Depends on:** M9-T08.
**Acceptance criteria:**
- Test with an artificially tiny block pool: requests all complete correctly despite forced preemptions; preempted-request output identical to unpreempted run (greedy); no block leaks (pool stats zero at end).

### M9-T10 · Cancellation & per-request failure isolation
Cancel promptly frees blocks and closes the channel with `cancelled`; a per-request
error (e.g. sampler edge case) fails that request only, never the engine loop.
**Depends on:** M9-T09.
**Acceptance criteria:**
- Tests: cancel during WAITING, during prefill, during decode — all reclaim resources (stats-verified); injected per-request fault leaves other concurrent requests' outputs unchanged.

---

## Milestone 10 — Serving Layer (HTTP API)

**Overview.** Expose the engine over HTTP with an OpenAI-compatible API:
`/v1/completions` and `/v1/chat/completions` with SSE streaming, chat templates,
structured errors, health endpoints, and a production-shaped CLI + config system. After
this milestone the project is a *server* you can point existing OpenAI SDK clients at.

**Architecture documents:** `docs/design/server.md` + ADR-004 HTTP library choice
(M10-T01).

### M10-T01 · Design doc & ADR: server architecture
Write `docs/design/server.md` and ADR-004: HTTP library selection (evaluate e.g.
standalone Asio + a minimal HTTP layer, `cpp-httplib`, Drogon — criteria: streaming
support, thread model fit, dependency weight), server threading model relative to the
engine thread, API schema definitions, streaming design, backpressure policy.
**Depends on:** M9-T10.
**Acceptance criteria:**
- ADR records the library decision with a comparison table; design doc includes request-flow diagram from socket to engine channel and back.

### M10-T02 · HTTP server skeleton
`src/server/`: server bootstrap with the chosen library — `/health` (liveness) and
`/v1/models` endpoints, graceful startup/shutdown (drain-free for now), port/host
configuration, request logging with request ids.
**Depends on:** M10-T01.
**Acceptance criteria:**
- Integration test (real HTTP client): endpoints respond correctly; SIGTERM shuts the server down cleanly (test sends signal, asserts exit).

### M10-T03 · /v1/completions (non-streaming)
Parse and validate the OpenAI completions request (prompt, max_tokens, temperature,
top_p, stop, seed, logprobs, n=1 for now), map to `SamplingParams`, run via engine
handle, return the OpenAI response shape with `usage` token counts.
**Depends on:** M10-T02.
**Acceptance criteria:**
- Integration tests: valid request returns correct schema (checked field-by-field); the official `openai` Python client (or a schema-validating test) can parse the response; invalid params → 400 with structured error body.

### M10-T04 · SSE streaming (completions)
`stream: true` support: SSE chunks matching OpenAI's chunk schema, terminal `[DONE]`,
client-disconnect detection → engine cancel.
**Depends on:** M10-T03.
**Acceptance criteria:**
- Integration tests: streamed concatenation equals non-streamed text for the same seed; disconnect mid-stream cancels the engine request (verified via engine stats) within one step.

### M10-T05 · Chat templates
`src/server/chat_template.h`: render chat messages to prompt strings via built-in
templates per model family (Llama-3 header format, ChatML for Qwen), selected from
model config/tokenizer metadata with a config override. (Full Jinja evaluation is
deliberately out of scope — documented.)
**Depends on:** M10-T04.
**Acceptance criteria:**
- Golden tests: rendered prompts byte-identical to HF `apply_chat_template` fixtures for both families across system/user/assistant/multi-turn cases, including generation prompt suffix.

### M10-T06 · /v1/chat/completions (non-streaming + streaming)
Chat endpoint: message validation, template render, generation, chat response schema;
streaming variant with role/delta chunks.
**Depends on:** M10-T05.
**Acceptance criteria:**
- Integration tests mirror M10-T03/T04 for the chat schema; finish_reason and usage present and correct.

### M10-T07 · Error taxonomy & API robustness
Uniform error responses (OpenAI error JSON shape): 400 validation, 404 model, 429
queue-full, 500 internal with request id; malformed-JSON handling; engine `Status` →
HTTP mapping table; panic-free guarantee on arbitrary bodies.
**Depends on:** M10-T06.
**Acceptance criteria:**
- Table-driven integration tests for each error class; a small adversarial-body corpus (huge strings, wrong types, deep nesting) returns 4xx without crashing.

### M10-T08 · CLI & configuration system
`engine serve` CLI: flags (`--model`, `--host/--port`, `--kv-cache-memory`, `--threads`,
`--max-num-seqs`, `--max-model-len`, `--dtype`, …) + optional YAML config file with
documented precedence (flags > file > defaults); `--help` generated from one
definition source; config validation with actionable errors.
**Depends on:** M10-T07.
**Acceptance criteria:**
- Unit tests: precedence, validation failures name the offending key; `engine serve --help` output committed as a golden file (kept current by test).

### M10-T09 · Admission control & backpressure
Queue-depth limit → 429 with `Retry-After`; per-request queue timeout; max concurrent
streams; counters for rejected/timed-out requests.
**Depends on:** M10-T08.
**Acceptance criteria:**
- Load test script (committed under `tools/loadtest/`, may be Python): flooding beyond capacity yields 429s while in-flight requests complete correctly; no memory growth after the flood (RSS checked).

---

## Milestone 11 — Prefix Caching

**Overview.** Reuse KV blocks across requests that share prompt prefixes (system
prompts, few-shot preambles, multi-turn chat). Content-addressed block hashing over
the paged cache from M8: full blocks are keyed by the hash chain of their token
history; admission matches the longest cached prefix and shares those blocks
(refcounted), prefilling only the suffix. Big TTFT wins on real workloads.

**Architecture documents:** `docs/design/prefix-caching.md` (M11-T01).

### M11-T01 · Design doc: prefix caching
Write `docs/design/prefix-caching.md`: chained hash definition (hash of block tokens +
parent hash; include salt for collision policy), full-block-only granularity, cache
index structure, refcount interaction with the block pool, LRU eviction of
refcount-zero blocks, correctness argument (why shared blocks are immutable), metrics.
**Depends on:** M10-T09.
**Acceptance criteria:**
- Doc includes the immutability invariant proof-sketch and the eviction policy; hash collision handling decided and justified.

### M11-T02 · Block content hashing
Compute chained hashes as blocks fill (in `BlockTable`/cache write path): only
complete blocks are hashable; hash stored with the block.
**Depends on:** M11-T01.
**Acceptance criteria:**
- Unit tests: identical token prefixes yield identical hash chains; single-token difference in any position diverges all subsequent hashes; partial blocks are never hashed.

### M11-T03 · Cached-block index
`src/kvcache/prefix_index.h`: hash → block-id map with insert-on-fill,
lookup, refcount pinning on match, and remove-on-eviction. Owns nothing; coordinates
with the block pool.
**Depends on:** M11-T02.
**Acceptance criteria:**
- Unit tests: lookup hit pins the block (refcount asserted); index never returns a block whose content was freed (lifecycle stress test with scripted alloc/free).

### M11-T04 · Admission-time prefix reuse
On scheduling a new sequence: walk its prompt's hash chain, adopt the longest cached
prefix into its block table (shared, refcounted), schedule prefill only for the
remaining suffix; handle the edge case of a full-prompt hit (still recompute last
token for logits).
**Depends on:** M11-T03.
**Acceptance criteria:**
- Integration tests: two identical prompts — second skips prefix prefill (scheduler stats assert fewer prefill tokens) and produces token-identical greedy output; partial-overlap prompts reuse exactly the shared full blocks.

### M11-T05 · Eviction
When the pool needs blocks: evict refcount-zero cached blocks in LRU order
(index removal + pool return); allocation path falls back through eviction before
reporting exhaustion.
**Depends on:** M11-T04.
**Acceptance criteria:**
- Tests with a tiny pool: workload alternating unique prompts completes (eviction keeps up); previously-cached-then-evicted prefix re-prefills correctly; no leaks (stats zero at drain).

### M11-T06 · Metrics, toggle & end-to-end validation
`--enable-prefix-caching` flag (default on), hit-rate metrics (tokens reused / prompt
tokens), and an end-to-end benchmark scenario demonstrating the win.
**Depends on:** M11-T05.
**Acceptance criteria:**
- Correctness A/B test: a randomized multi-request workload produces identical outputs with caching on vs off (greedy).
- Benchmark: shared-system-prompt workload shows ≥ 2× TTFT improvement for cache-hit requests (recorded in `benchmarks/BASELINES.md`).

---

## Milestone 12 — CPU Performance Engineering & Latency

**Overview.** Close the performance gap: tuned GEMM/GEMV, long-context attention
scaling, kernel fusions, allocation and memory-bandwidth hygiene, and chunked
prefill for tail-latency control. Every optimization lands behind an existing
interface with the previous implementation kept as the test reference, and every
ticket must show its benchmark delta. The milestone ends with a rigorous comparison
against llama.cpp on identical hardware and workload.

**Architecture documents:** `docs/profiling.md` methodology notes begin here
(M12-T01); design docs amended in place.

### M12-T01 · Kernel microbenchmark harness
Mature the `benchmarks/kernels/` harness begun in M3-T06: monotonic-clock timing
with warmup + iterations, thread-count and ISA sweep support, CSV/markdown output,
run-to-run stability discipline (pinned thread count, documented machine-quiescing
procedure). Document the profiling workflow (`Instruments` on macOS, `perf` on
Linux) in `docs/profiling.md`.
**Depends on:** M11-T06.
**Acceptance criteria:**
- Harness benches existing kernels (GEMM, RMSNorm, decode attention) with ±3% run-to-run stability on the quiesced dev machine; used by every subsequent ticket in this milestone.

### M12-T02 · GEMM/GEMV tuning
Tune the M6 GEMM/GEMV: tile-size selection per shape class (documented sweep, not
folklore), software prefetching where it measurably helps, native fp16/bf16
arithmetic where the ISA offers it (NEON FP16/BF16 extensions — feature-detected),
and thread-partitioning tuned for the skinny decode shapes.
**Depends on:** M12-T01.
**Acceptance criteria:**
- Correctness suite unchanged (tolerances may not loosen); microbench deltas per shape class recorded in BASELINES.md; decode GEMV improvement reflected in end-to-end decode tokens/sec (recorded).

### M12-T03 · Long-context attention scaling
Optimize decode attention for long contexts: split the cache across threads with a
deterministic reduction pass (flash-decoding analog), tuned block sizes for the
prefill attention, and threading that stays efficient when batch × heads exceeds
core count.
**Depends on:** M12-T02.
**Acceptance criteria:**
- Correctness vs the M6 kernels across cache lengths {16, 512, 4k, 16k}; determinism preserved (bit-identical across thread counts).
- Bench: decode latency at 16k context improves ≥ 2× vs M6 kernel (recorded); scaling curve (latency vs context length) plotted in the PR.

### M12-T04 · Kernel fusions
Fuse: (a) residual-add + RMSNorm, (b) RoPE + KV-cache write, (c) SwiGLU combine
into the down-projection input where layout permits. Wire into the forward pass;
unfused paths remain for reference testing.
**Depends on:** M12-T03.
**Acceptance criteria:**
- Correctness: fused == unfused within tolerance on the fixture model (end-to-end logits equality test).
- Bench: per-layer improvement and end-to-end decode tokens/sec gain recorded.

### M12-T05 · Allocation & memory-bandwidth hygiene
Audit the step loop for per-step allocations and needless copies: workspace and
staging reuse everywhere, batch metadata assembled into preallocated buffers,
false-sharing check on hot shared state, huge-page/THP stance documented per
platform, thread-affinity policy decided and documented (and applied where it
measurably helps).
**Depends on:** M12-T04.
**Acceptance criteria:**
- A profile before/after is summarized in `docs/perf-notes.md`; steady-state decode performs zero heap allocations per step (asserted with a counting-allocator test hook); end-to-end change recorded (must be ≥ 0).

### M12-T06 · Chunked prefill
Scheduler v2 feature: split long prompts into fixed-size chunks scheduled alongside
decodes (bounding per-step latency); config knob `--max-num-batched-tokens` becomes
the binding budget; the prefill-continuation path (M8-T06) already supports it.
**Depends on:** M12-T05.
**Acceptance criteria:**
- Correctness: chunked prefill output identical to unchunked (greedy, multiple chunk sizes incl. chunk boundary == block boundary).
- Bench: P99 inter-token latency of concurrent decodes improves ≥ 3× when a long prompt arrives mid-stream (scenario test, numbers recorded).

### M12-T07 · Performance baseline vs llama.cpp
Document a rigorous comparison on fixed hardware: this engine vs llama.cpp (same
model, same quantization/dtype, same thread count, same prompts) for single-stream
latency and batched throughput; identify the top-3 remaining gaps with profiles.
**Depends on:** M12-T06.
**Acceptance criteria:**
- `docs/perf-comparison.md` with methodology (versions, hardware, commands), results table, and gap analysis; results also appended to BASELINES.md.

---

## Milestone 13 — Quantized Inference (Weight-Only)

**Overview.** Serve quantized models: weight-only INT8 (round-to-nearest at load) and
INT4 group-quantized checkpoints (AWQ/GPTQ formats) with custom SIMD dequant kernels,
plus INT8 KV-cache storage. Halves-to-quarters memory footprint — the difference
between a 1B and an 8B model fitting comfortably in host RAM — and, on CPU, a direct
decode-throughput win (decode is memory-bandwidth-bound, so smaller weights stream
faster). Quantized layers slot in behind the `Linear` interface; accuracy is
validated statistically against the unquantized baseline, not vibes.

**Architecture documents:** `docs/design/quantization.md` (M13-T01).

### M13-T01 · Design doc: quantization
Write `docs/design/quantization.md`: scope (weight-only; activations stay bf16/fp16),
supported formats (RTN INT8 per-channel; AWQ & GPTQ INT4 group-wise — layouts
documented bit-exactly), `QuantizedLinear` abstraction, kernel plan (fused SIMD
dequant-GEMV for decode, dequant-to-tiles feeding the M6 GEMM for prefill),
accuracy-validation methodology (top-1 agreement rate + logit MSE on a prompt corpus
vs the bf16/fp32 baseline).
**Depends on:** M12-T07.
**Acceptance criteria:**
- Layout diagrams for AWQ and GPTQ packed formats (bit positions, group scales/zeros); accuracy methodology with thresholds defined.

### M13-T02 · Quantized weight containers & config detection
Activate `kInt4`/packed handling in the tensor layer as opaque byte tensors +
metadata; parse HF `quantization_config` (method, bits, group size, sym/asym);
`QuantizedLinear` module skeleton holding qweight/scales/zeros.
**Depends on:** M13-T01.
**Acceptance criteria:**
- Unit tests: config detection for AWQ and GPTQ checkpoint configs; container shape/metadata validation errors are actionable.

### M13-T03 · INT8 weight-only path
Load-time RTN INT8 per-channel quantization of a bf16 checkpoint (`--dtype int8`);
dequant-to-tiles + M6 GEMM path for prefill; fused SIMD dequant-GEMV kernel for
decode (batch-1-ish shapes).
**Depends on:** M13-T02.
**Acceptance criteria:**
- Kernel tests vs Python-fixture dequant references (exact integer math).
- E2E: 1B model at INT8 — memory reduction ≈ 2× on weights (measured), top-1 token agreement vs bf16 ≥ 98% on the test corpus, greedy fixture outputs recorded; decode tokens/sec vs bf16 recorded (expected ≥ 1×).

### M13-T04 · AWQ/GPTQ checkpoint loading
Load pre-quantized safetensors: qweight/qzeros/scales (+ g_idx for GPTQ) unpacking
into `QuantizedLinear`, weight-map extensions, validation of group-size consistency
with config.
**Depends on:** M13-T03.
**Acceptance criteria:**
- Unit tests against a small real AWQ and GPTQ checkpoint fixture (generated via AutoAWQ/GPTQ in `tools/`, committed small): unpacked values match Python reference dequant exactly.

### M13-T05 · INT4 dequant kernels
Group-wise INT4 dequant-GEMV kernel for decode (fused SIMD dequant + dot-product,
both AWQ and GPTQ layouts) and a dequant-to-tiles path feeding the M6 GEMM for
prefill.
**Depends on:** M13-T04.
**Acceptance criteria:**
- Kernel tests: bit-exact dequant vs Python reference across group sizes {32, 64, 128}, then GEMV matches fp32 reference within tolerance.
- Microbench: INT4 GEMV ≥ 1.5× faster than bf16 GEMV at 4k×4k (memory-bound win, recorded).

### M13-T06 · End-to-end quantized serving
Run a real INT4 model (e.g. a 7B-class AWQ checkpoint) end-to-end through the server;
accuracy validation per M13-T01 methodology; benchmarks.
**Depends on:** M13-T05.
**Acceptance criteria:**
- Serving integration test passes with the quantized model; accuracy report committed (top-1 agreement, logit MSE); memory + throughput vs bf16 recorded in BASELINES.md.

### M13-T07 · INT8 KV cache
Optional `--kv-cache-dtype int8`: scaled INT8 KV storage (per-head or per-block
scales — documented choice), dequant inside the attention kernels, scale calibration
strategy (dynamic per-block absmax).
**Depends on:** M13-T06.
**Acceptance criteria:**
- Attention kernels with the INT8 cache match full-precision-cache results within documented tolerance; long-generation quality spot-check; KV memory halved (measured); throughput delta recorded.

---

## Milestone 14 — Quantization Algorithms & Evaluation Toolkit

**Overview.** M13 *serves* quantized checkpoints; this milestone *produces and
evaluates* them. Implement the quantizers themselves — RTN, AWQ (activation-aware
scaling), and GPTQ (Hessian-based error compensation) — in the engine, using the
engine's own forward pass for calibration, plus a perplexity evaluation harness and
a reproducible quality-vs-bits report. Everything cross-validated against the
reference Python implementations via committed fixtures. This is the milestone that
turns "loads AWQ checkpoints" into "understands quantization end to end."

**Architecture documents:** `docs/design/quantization-algorithms.md` (M14-T01).

### M14-T01 · Design doc: quantization algorithms & evaluation
Write `docs/design/quantization-algorithms.md`: the algorithm math written out (RTN
rounding & clipping; AWQ per-channel scale search minimizing layer output error over
calibration activations; GPTQ column-wise quantization with Hessian-driven error
compensation, Cholesky formulation, act-order option), the calibration data flow
(activation-capture hooks on the M5/M6 forward pass, storage budget, sample counts),
the evaluation methodology (perplexity definition, sliding-window evaluation,
corpus choice and committed subset within a size budget; top-1 agreement as the
secondary metric), and the C++/Python boundary (the engine implements the
algorithms; `tools/` produces reference outputs from AutoAWQ/GPTQ for
cross-validation fixtures only).
**Depends on:** M13-T07.
**Acceptance criteria:**
- The GPTQ update equations and the AWQ search objective are written out precisely enough to implement from the doc alone; evaluation methodology fixes corpus, window, and stride.

### M14-T02 · Activation capture & calibration pipeline
Forward-pass hooks capturing per-layer input activations (or sufficient statistics —
per the design doc) over a calibration set fed through the engine; deterministic
sample selection; memory-bounded accumulation (running Hessian / absmax statistics,
never materializing all activations).
**Depends on:** M14-T01.
**Acceptance criteria:**
- Unit tests: captured statistics for the tiny fixture match a `tools/` Python reference computation exactly (fp64 accumulation tolerance documented); memory ceiling asserted for a large calibration run.

### M14-T03 · Perplexity evaluation harness
`engine eval-ppl`: sliding-window perplexity over a token file using the engine's
forward pass (all-position logits path), with documented window/stride; committed
small evaluation corpus per the design doc.
**Depends on:** M14-T02.
**Acceptance criteria:**
- Validation: engine PPL on the tiny fixture matches a HuggingFace-computed fixture value within documented tolerance; run-to-run deterministic.
- CLI documented; runtime on the 1B model at the committed corpus size is recorded (feasibility check).

### M14-T04 · RTN quantizer & baseline report
`engine quantize --method rtn`: load a bf16/fp16 checkpoint, quantize weight-only
(INT8 per-channel; INT4 group-wise at {32, 64, 128}), and write an
engine-loadable checkpoint in the M13 container formats.
**Depends on:** M14-T03.
**Acceptance criteria:**
- Round-trip test: quantize → load → serve passes the M13 accuracy gates; quantized values match a Python RTN reference bit-exactly.
- First report table committed: PPL and top-1 agreement for fp baseline vs INT8 vs INT4-RTN across group sizes on the 1B model.

### M14-T05 · AWQ quantizer
Activation-aware scaling: per-channel scale search (grid over the mixing exponent,
minimizing layer output error on calibration activations per the design doc),
scale folding into adjacent ops, INT4 group-wise export in the AWQ layout M13
already loads.
**Depends on:** M14-T04.
**Acceptance criteria:**
- Cross-validation: on the tiny fixture with identical calibration data, chosen scales and quantized weights match AutoAWQ within documented tolerance (fixture from `tools/`).
- Report: AWQ INT4 beats RTN INT4 on PPL for the 1B model (the expected result — if not, the investigation is documented before the ticket closes).

### M14-T06 · GPTQ quantizer
Layer-wise GPTQ: Hessian accumulation from calibration activations (M14-T02),
column-wise quantization with error compensation via the Cholesky formulation,
optional act-order (`g_idx`), INT4 group-wise export in the GPTQ layout M13 loads.
**Depends on:** M14-T05.
**Acceptance criteria:**
- Cross-validation: tiny-fixture quantized weights match AutoGPTQ within documented tolerance given identical calibration data; numerical-stability fallback (Hessian damping) tested on an ill-conditioned layer.
- Report: GPTQ INT4 results added; runtime of quantizing the 1B model recorded.

### M14-T07 · Quality-vs-bits report & method comparison
The deliverable: `docs/quantization-report.md` — full matrix (fp baseline, INT8-RTN,
INT4 × {RTN, AWQ, GPTQ} × group sizes) of perplexity, top-1 agreement, memory
footprint, and decode tokens/sec, on the committed corpus and fixed hardware, with
a scripted regeneration path (`engine quantize` + `engine eval-ppl` invocations in
a committed script).
**Depends on:** M14-T06.
**Acceptance criteria:**
- Report committed with methodology and machine fingerprint; every number regenerable by the script; conclusions section compares methods honestly (including where they tie).

---

## Milestone 15 — Speculative Decoding

**Overview.** Accelerate decode with speculative execution: a cheap proposer drafts k
tokens, the target model verifies them in one batched forward pass, and a
rejection-sampling acceptance rule preserves the target distribution exactly. Two
proposers: n-gram/prompt-lookup (free, surprisingly effective on repetitive text) and
a small draft model. The correctness bar is unusual — greedy output must be
*identical* with speculation on or off, and sampled output must be *distributionally*
identical. Speculation is particularly attractive on CPU: decode is
memory-bandwidth-bound, so verifying k tokens in one pass amortizes the weight
streaming that dominates decode cost. The milestone closes with a measurement study
(M15-T07) treating acceptance rates and speedups as a research question, not a
checkbox.

**Architecture documents:** `docs/design/speculative-decoding.md` (M15-T01).

### M15-T01 · Design doc: speculative decoding
Write `docs/design/speculative-decoding.md`: `Proposer` interface, verification
forward (target scores k draft tokens + 1 in a single step — reuses varlen prefill
machinery), acceptance rule (greedy exact-match; stochastic rejection sampling with
the residual-distribution resample), KV-cache handling for speculated-then-rejected
tokens (cache rollback via block-table truncation), scheduler integration, and when
speculation is expected to help vs hurt (the CPU memory-bandwidth argument made
quantitative).
**Depends on:** M14-T07.
**Acceptance criteria:**
- Acceptance-rule math written out (with the residual distribution formula); KV rollback design covers block-boundary cases.

### M15-T02 · Proposer interface & n-gram proposer
`src/spec/proposer.h` + prompt-lookup proposer: propose k tokens by matching the
longest recent n-gram of the sequence against its own history (prompt + generated);
configurable n range and k.
**Depends on:** M15-T01.
**Acceptance criteria:**
- Unit tests: known-repetition sequences produce expected proposals; no-match produces empty proposal (engine falls back to normal decode); proposal never exceeds max_len budget.

### M15-T03 · Draft-model proposer
Load a second, smaller model (same tokenizer — validated) with its own KV cache
(paged, small pool); autoregressively draft k tokens per sequence per step; draft
cache maintained in sync with accepted tokens (including rollback).
**Depends on:** M15-T02.
**Acceptance criteria:**
- Unit tests with fixture models: drafts match running the small model standalone; draft KV state after mixed accept/reject steps equals recomputed-from-scratch state.

### M15-T04 · Verification forward & KV rollback
Target-model verification: score the k proposed tokens + 1 bonus position in one
forward (per sequence, batched across sequences); on rejection at position j, truncate
sequence and roll back paged KV (block-table truncation + slot invalidation) for both
target and draft caches.
**Depends on:** M15-T03.
**Acceptance criteria:**
- Tests: verification logits at each position match sequential decode logits exactly (same cache state); rollback then re-decode produces identical results to never having speculated (the cache-integrity invariant), including rollbacks across block boundaries.

### M15-T05 · Acceptance rule
Greedy mode: accept the longest prefix of drafts matching target argmax, then take
target's token at first mismatch. Sampling mode: per-position rejection sampling
(accept with prob min(1, p_target/p_draft), resample from normalized residual on
reject).
**Depends on:** M15-T04.
**Acceptance criteria:**
- Greedy: speculative output token-identical to non-speculative for fixture + real model over long generations (the golden invariant).
- Sampling: statistical test — empirical next-token distribution under speculation matches target-only distribution (chi-square over a small-vocab fixture, 100k trials).

### M15-T06 · Engine integration & metrics
Scheduler/engine support for speculative steps (variable tokens-per-step per
sequence), `--speculative-model` / `--num-speculative-tokens` / `--spec-method=ngram|draft`
config, metrics (proposals, acceptance rate, effective tokens/step), interaction with
stop conditions (stop mid-accepted-window handled).
**Depends on:** M15-T05.
**Acceptance criteria:**
- Integration: API test suite passes with speculation enabled; greedy A/B identical outputs; acceptance-rate metric exposed.
- Bench: ≥ 1.5× decode tokens/sec on a favorable pair (e.g. 7B-INT4 target + 0.5B draft, or n-gram on a repetitive workload) recorded in BASELINES.md; unfavorable-case regression documented.

### M15-T07 · Acceptance-rate & speedup study
The measurement deliverable: a systematic study across proposers (n-gram, draft
model), draft lengths k ∈ {1..8}, target/draft pairs, and workload types (repetitive
extraction, chat, code, open-ended prose), measuring acceptance rate, effective
tokens/step, and end-to-end speedup — with the memory-bandwidth model from M15-T01
compared against observed results.
**Depends on:** M15-T06.
**Acceptance criteria:**
- `docs/speculation-study.md` committed: methodology (models, corpora, hardware fingerprint), full results matrix, the predicted-vs-observed speedup comparison, and an honest recommendations section (default k, when to enable which proposer); every number regenerable by a committed script.

---

## Milestone 16 — Observability, Benchmarking & Profiling

**Overview.** Make the engine measurable in production and in development: a metrics
subsystem with Prometheus exposition, request-level latency instrumentation (TTFT,
inter-token latency), trace-based profiling integration, an offline throughput
benchmark, a realistic serving load generator with workload replay, and documented
performance methodology. Several earlier milestones added ad-hoc counters; this
milestone unifies them.

**Architecture documents:** `docs/design/observability.md` (part of M16-T01).

### M16-T01 · Metrics registry & Prometheus endpoint
`src/metrics/`: lightweight registry (counters, gauges, histograms with fixed
buckets; lock-cheap hot path), Prometheus text-format exposition at `/metrics`,
design notes in `docs/design/observability.md` (naming conventions, label discipline,
cardinality rules).
**Depends on:** M15-T07.
**Acceptance criteria:**
- Unit tests: metric registration/update/render (format validated against Prometheus exposition spec); concurrent-update stress test; hot-path counter increment benchmarked (< 50 ns).

### M16-T02 · Engine & request instrumentation
Instrument with the registry: TTFT, inter-token latency, end-to-end latency, queue
wait, tokens in/out, running/waiting sequence gauges, KV utilization, prefix-cache hit
rate, preemptions, batch-size histogram, spec-decode acceptance rate — replacing all
ad-hoc counters from earlier milestones.
**Depends on:** M16-T01.
**Acceptance criteria:**
- Integration test: a scripted workload produces expected metric values (e.g. request count, token counts exact; latency histograms populated); `/metrics` output reviewed against the naming doc.

### M16-T03 · Trace spans & profiling documentation
Chrome-trace (Perfetto-compatible) span emission around step phases (schedule,
assemble, prefill, decode, sample, output) and per-layer (compile-time-gated to zero
cost when off), written to a file on demand; extend `docs/profiling.md` (begun in
M12-T01): how to capture and read engine traces plus `perf`/Instruments profiles,
with annotated example screenshots.
**Depends on:** M16-T02.
**Acceptance criteria:**
- A captured trace opens in Perfetto/`chrome://tracing` with named phase and layer spans (manual verification, doc includes the capture command); disabled build has zero overhead (bench diff within noise).

### M16-T04 · Offline throughput benchmark
`benchmarks/bench_throughput`: run N prompts (synthetic or dataset file) at max
batching, report prompt & generation tokens/sec, per-phase breakdown, JSON + human
output; replaces/absorbs `bench_generate`.
**Depends on:** M16-T03.
**Acceptance criteria:**
- Deterministic workload mode for regression comparisons; results include config fingerprint (model, dtype, flags, commit); BASELINES.md updated via a documented process.

### M16-T05 · Serving load generator
`tools/loadtest/` (Python acceptable): open-loop Poisson arrivals at target RPS,
ShareGPT-format conversation replay, measures TTFT/ITL/E2E percentiles (P50/P90/P99),
goodput at SLO, outputs a report; drives the real HTTP API.
**Depends on:** M16-T04.
**Acceptance criteria:**
- Against a live server: report generated with sane percentiles; saturation behavior visible (latency knee as RPS increases across a sweep); README documents usage.

### M16-T06 · Structured request logging & slow-request tracing
Per-request structured log line (JSON option) with ids, token counts, timing
breakdown, finish reason; debug mode capturing per-step timelines for sampled slow
requests.
**Depends on:** M16-T05.
**Acceptance criteria:**
- Log schema documented and tested (parse the JSON in the test); slow-request trace triggers at a configurable threshold and contains per-phase timings.

### M16-T07 · Performance regression guardrail
A scripted perf-regression check (manual, or nightly on a designated quiesced
machine): runs the deterministic throughput benchmark and key kernel
microbenchmarks, compares against BASELINES.md with tolerance bands, fails on
regression; documented release-gate process.
**Depends on:** M16-T06.
**Acceptance criteria:**
- Script exits non-zero on a synthetic injected regression (test by perturbing a baseline); process documented in `docs/perf-process.md`.

---

## Milestone 17 — Hardening, Documentation & v0.1 Release

**Overview.** Production-grade polish: leak/stress/fuzz testing, determinism
guarantees, packaging (Docker), a real documentation site, and a tagged v0.1.0
release. This milestone turns a working engine into a credible open-source project —
the difference between "code that runs" and "software people can adopt."

**Architecture documents:** none new — this milestone finishes and publishes the
existing ones.

### M17-T01 · Sanitizer & leak hygiene
CI jobs: ASAN+UBSAN build running the full test suite; TSAN on
parallel/runtime/scheduler tests; fix everything found. (A CPU-only engine means the
*entire* surface is sanitizable in CI — no unsanitized device code.)
**Depends on:** M16-T07.
**Acceptance criteria:**
- ASAN/UBSAN/TSAN CI jobs green across the whole suite; suppression list (if any) empty or individually justified in-file.

### M17-T02 · Stress & soak testing
`tools/stress/`: hours-long soak driver (random request sizes, params, cancels,
client disconnects, occasional pathological inputs) against the server; memory-growth
and fd-leak assertions; documented soak procedure.
**Depends on:** M17-T01.
**Acceptance criteria:**
- 2-hour soak on a real model: zero crashes, RSS plateaus (< 1%/hour growth), block-pool stats return to zero at drain.

### M17-T03 · Fuzzing
libFuzzer targets: tokenizer encode/decode, safetensors header parser, API JSON
request parser; seed corpora committed; short fuzz runs wired into CI, long runs
documented.
**Depends on:** M17-T02.
**Acceptance criteria:**
- 10-minute CI fuzz of each target finds no crashes; any pre-existing findings fixed with regression tests added.

### M17-T04 · Determinism & compatibility guarantees
Document and test the determinism contract (same model+seed+params+engine version ⇒
identical output; caveats: batching-dependent numerics — decide and document whether
batch-invariance is guaranteed or not); model-support matrix validated at startup
(clear error for unsupported arch/dtype/head-dim combos).
**Depends on:** M17-T03.
**Acceptance criteria:**
- Determinism test suite (restart-to-restart, and documented batching caveats verified as described); startup validation errors tested for the top unsupported cases.

### M17-T05 · Packaging & release builds
Multi-stage `Dockerfile` (slim Debian/distroless base, pinned toolchain, multi-arch
amd64 + arm64), `docker run … engine serve --model …` works out of the box; release
CMake preset (LTO, per-ISA kernel TUs verified in Release); documented from-source
install for macOS (the primary dev platform); CI job building the image on tags.
**Depends on:** M17-T04.
**Acceptance criteria:**
- Image builds in CI for both architectures, serves a model on a plain host (documented smoke procedure), size and layer hygiene reviewed (no build tools in final stage); macOS from-source instructions verified from a clean clone.

### M17-T06 · Documentation site
`docs/` publishable (mkdocs-material or similar): getting started (install → serve →
query in 5 minutes), architecture overview with diagrams (from the design docs),
configuration reference (generated from the M10-T08 definition source), API reference,
model support matrix, performance guide, contribution guide.
**Depends on:** M17-T05.
**Acceptance criteria:**
- Site builds in CI; a newcomer path (getting-started) tested end-to-end verbatim; every design doc linked and current (stale sections updated).

### M17-T07 · Examples & client compatibility
`examples/`: curl scripts, Python with the official `openai` SDK, streaming client,
chat client; compatibility notes for common tools (documented tested versions).
**Depends on:** M17-T06.
**Acceptance criteria:**
- Every example runs against a live server in an integration test (or a documented manual matrix); `openai` SDK streaming + non-streaming verified.

### M17-T08 · v0.1.0 release
Semantic versioning policy, `CHANGELOG.md` (Keep-a-Changelog format, back-filled by
milestone), README polish (badges, feature list, benchmark table, architecture
diagram), license + NOTICE review for vendored code, tag + GitHub release with notes
and Docker image reference.
**Depends on:** M17-T07.
**Acceptance criteria:**
- `v0.1.0` tag published with release notes; README quick-start verified from a clean machine; version surfaced in `engine --version` and `/v1/models`.

---

## Future directions (post-v0.1, unordered)

Deliberately out of scope for the linear roadmap; each would begin with its own design
doc and milestone breakdown when scheduled:

- **Metal/MLX backend** — the dev machine's own GPU; the backend seams (reference vs
  optimized, ADR-002 layering) are designed to admit a third implementation.
- **CUDA backend revival** — the retired M2 foundation plus v1's M5/M11/M13 plans
  remain in history (`docs/archive/ROADMAP-v1.md`,
  `docs/design/retired/cuda-backend.md`) as the starting point if NVIDIA hardware
  enters the picture.
- **AVX-512 / ARM SVE kernel variants** — the per-ISA TU pattern (M3-T05) makes new
  ISAs additive.
- **Structured/guided decoding** — JSON-schema/grammar-constrained generation (logit
  masking via FSM).
- **LoRA serving** — multi-adapter batching, adapter hot-swap.
- **KV-cache offload & tiering** — NVMe/disk tiering for long-context and
  many-session workloads (especially natural on CPU where the cache already lives in
  host RAM).
- **Sliding-window & hybrid attention** — Mistral-style windowed attention, attention
  sinks.
- **Mixture-of-Experts (MoE)** — expert routing on CPU (Mixtral/Qwen-MoE families);
  memory-bound expert paging is an interesting CPU-specific angle.
- **Encoder-only & embedding serving** — BERT-family embedding endpoints.
- **State-space models & hybrids** — Mamba-style layers (Jamba-class).
- **Sub-4-bit & modern quantization** — INT2/ternary, importance-aware mixed
  precision; extends the M14 toolkit.
- **Batch-invariant determinism** — CPU fp32 accumulation with deterministic
  partitioning makes "same output regardless of batching" plausibly guaranteeable;
  investigate promoting it from caveat (M17-T04) to contract.
- **Horizontal serving** — multi-instance routing and KV-aware load balancing in
  front of N engine processes.
- **Vision-language models** — image encoder + projector, multimodal prompt
  plumbing.
