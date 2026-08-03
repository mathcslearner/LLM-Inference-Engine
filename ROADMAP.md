# LLM Inference Engine — Development Roadmap

A multi-GPU LLM inference engine for decoder-only transformer models (Llama, Qwen, and
similar families), written in C++20/CUDA, targeting NVIDIA GPUs on Linux x86-64.
Performance-competitive with existing engines while emphasizing clean architecture,
maintainability, and testability.

This roadmap is organized as **milestones** executed in **linear order**. Each milestone
contains **tickets** — independent, focused tasks sized for roughly one development
session (1–4 hours, typically a few hundred lines of code including tests). Tickets
within a milestone are also ordered linearly; dependencies are listed explicitly.

---

## How to work this roadmap

- **Execute tickets in order.** IDs are `M<milestone>-T<ticket>` (e.g. `M5-T03`).
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
  from HuggingFace via `tools/` Python scripts) for anything numerical; integration
  tests for cross-module behavior; GPU tests auto-skip when no CUDA device is present so
  the CPU-only CI stays green.
- **Never break abstractions:** the module boundaries listed in `CLAUDE.md` are
  load-bearing. A ticket that needs to reach across a boundary is a signal the design
  doc needs amending first.

### Module map (established across milestones)

| Module | Directory | Introduced |
|---|---|---|
| Core utilities (status, logging, config) | `src/core/` | M0 |
| Tensor library | `src/tensor/` | M1 |
| Memory management (allocators, pools) | `src/memory/` | M1–M2 |
| CUDA backend (streams, device utils) | `src/cuda/` | M2 |
| CUDA kernels | `src/kernels/` | M2 |
| CPU reference backend | `src/cpu/` | M4 |
| Model loader & configs | `src/model/` | M3 |
| Tokenizer | `src/tokenizer/` | M3 |
| KV cache | `src/kvcache/` | M4, M7 |
| Sampling | `src/sampling/` | M5–M6 |
| Execution engine (model runner) | `src/engine/` | M4–M5 |
| Scheduler | `src/scheduler/` | M8 |
| Runtime (engine loop, requests) | `src/runtime/` | M8 |
| Server / API | `src/server/` | M9 |
| Distributed (NCCL, tensor parallel) | `src/distributed/` | M13 |
| Quantization | `src/quant/` | M12 |
| Speculative decoding | `src/spec/` | M14 |
| Metrics & profiling | `src/metrics/` | M15 |
| Benchmarks | `benchmarks/` | M5, M11, M15 |
| Dev tooling (fixture generation, scripts) | `tools/` | M3 |

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

### M0-T06 · Logging subsystem
Thin wrapper over spdlog in `src/core/logging.h`: leveled macros (`LOG_DEBUG` …
`LOG_ERROR`), per-module logger names, level configurable via environment variable and
programmatic API. No raw spdlog usage outside the wrapper.
**Depends on:** M0-T02.
**Acceptance criteria:**
- Log level switchable at runtime; messages include timestamp, level, module, source location.
- Unit tests verify level filtering and formatting; hot-path macro compiles to nothing above the configured level in Release.

### M0-T07 · Error-handling primitives
Implement `Status` and `StatusOr<T>` (or `Result<T, Error>`) in `src/core/status.h`
with an error-code taxonomy (`InvalidArgument`, `OutOfMemory`, `NotFound`,
`Unimplemented`, `Internal`, `ResourceExhausted`, …), plus `RETURN_IF_ERROR`,
`ASSIGN_OR_RETURN`, and fatal `CHECK`/`DCHECK` macros.
**Depends on:** M0-T06.
**Acceptance criteria:**
- Policy: recoverable errors return `Status`; programmer errors use `CHECK`; exceptions are not used across module boundaries. Documented in the header.
- Unit tests cover propagation macros, error message composition, and `StatusOr` move semantics.

### M0-T08 · Documentation scaffolding & founding ADRs
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

### M1-T01 · Design doc: tensor library & device model
Write `docs/design/tensor.md`: supported dtypes (now + planned), shape/stride
representation, ownership model (shared buffer + views), allocator interface, device
abstraction, host/device data-access rules, and explicit non-goals (autograd, lazy
eval, broadcasting beyond what inference needs).
**Depends on:** M0-T08.
**Acceptance criteria:**
- Doc covers all sections above with API sketches for `Tensor`, `Buffer`, `Allocator`, `Device`.
- Explicitly specifies thread-safety guarantees and copy/move semantics of `Tensor`.

### M1-T02 · DataType enum & traits
`src/tensor/dtype.h`: `DataType` enum (`kFloat32`, `kFloat16`, `kBFloat16`, `kInt8`,
`kUInt8`, `kInt32`, `kInt64`, `kBool`, plus reserved `kFP8E4M3`, `kInt4` for later),
`itemsize()`, `to_string()`/`from_string()`, and a compile-time `DTypeTraits<T>`
mapping C++ types ↔ enum values.
**Depends on:** M1-T01.
**Acceptance criteria:**
- Unit tests cover every enum value's size, name round-trip, and trait mapping.
- `kInt4` reports sub-byte handling explicitly (itemsize in bits API or documented packing rule).

### M1-T03 · Shape & strides
`src/tensor/shape.h`: `Shape` (small inline vector of dims), `numel()`, row-major
stride computation, `is_contiguous()` check, dim validation, equality, formatting.
**Depends on:** M1-T02.
**Acceptance criteria:**
- Unit tests: 0-d through 5-d shapes, numel overflow detection, contiguity for sliced strides, stride computation golden cases.

### M1-T04 · Device abstraction
`src/tensor/device.h`: `Device{DeviceType type; int index;}` with `DeviceType::kCPU`
and `DeviceType::kCUDA`, parsing (`"cuda:0"`), equality, formatting. CUDA devices are
representable now but any attempt to allocate on them returns `Unimplemented` until M2.
**Depends on:** M1-T02.
**Acceptance criteria:**
- Unit tests for parsing, formatting, equality, invalid inputs.
- No CUDA headers included — this header stays backend-agnostic.

### M1-T05 · Allocator interface & CPU allocator
`src/memory/allocator.h`: abstract `Allocator` (`allocate(bytes, alignment) →
StatusOr<Buffer>`), `Buffer` as a move-only owning handle (pointer, size, device,
deleter). Implement `CpuAllocator` with configurable alignment (default 64B).
**Depends on:** M1-T04, M0-T07.
**Acceptance criteria:**
- Unit tests: alignment honored, zero-size allocation defined behavior, Buffer move semantics, deleter invoked exactly once (tracked via test allocator).
- Allocation failures return `Status`, never throw.

### M1-T06 · Tensor core type
`src/tensor/tensor.h`: `Tensor` = shared `Buffer` + `Shape` + strides + `DataType` +
`Device` + byte offset. Factory `Tensor::empty(shape, dtype, device, allocator)`.
Views: `slice(dim, start, end)`, `reshape` (contiguous only), `view_as_dtype`
(same-size only). Typed `data_ptr<T>()` with dtype check; CPU-only element accessor
for tests.
**Depends on:** M1-T05.
**Acceptance criteria:**
- Unit tests: views share the buffer (write-through visible), slicing produces correct shapes/strides/offsets, reshape rejects non-contiguous, dtype-checked access fails loudly on mismatch.
- `Tensor` is cheap to copy (shared buffer semantics documented and tested).

### M1-T07 · Half-precision host support
`src/tensor/half.h`: `float16` and `bfloat16` value types for host code (bit-accurate
conversion to/from `float`, including rounding, inf/nan, subnormals for fp16), wired
into `DTypeTraits`.
**Depends on:** M1-T06.
**Acceptance criteria:**
- Golden bit-pattern tests: known fp32↔fp16 and fp32↔bf16 pairs including rounding boundaries, ±inf, NaN preservation.
- Conversion round-trip property test over random floats within representable range.

### M1-T08 · Tensor factories & comparison utilities
`src/tensor/ops.h` (CPU): `zeros/ones/full/arange`, seeded uniform/normal fill,
element-wise `allclose(a, b, rtol, atol)` with per-dtype default tolerances and a
max-abs-diff report, and human-readable tensor printing for debugging.
**Depends on:** M1-T07.
**Acceptance criteria:**
- Unit tests for each factory across dtypes; `allclose` failure message reports index and values of worst mismatch.
- Random fills are deterministic given a seed (test asserts exact values).

### M1-T09 · CPU copy & cast
`src/tensor/ops.h`: `copy(dst, src)` (same shape, handles non-contiguous views) and
`cast(src, dtype)` between all floating dtypes and int types on CPU.
**Depends on:** M1-T08.
**Acceptance criteria:**
- Unit tests: contiguous and strided copies, all supported cast pairs, precision-loss cases (fp32→fp16 rounding) match the M1-T07 conversion functions exactly.

---

## Milestone 2 — CUDA Backend Foundation

**Overview.** Bring up the CUDA layer: build-system integration, error handling,
streams/events, device memory allocation (including the caching pool allocator that
becomes the engine's memory-pooling backbone), host↔device transfers, and the kernel
infrastructure with first trivial kernels. After this milestone, `Tensor` works on GPU
and every later kernel ticket has a paved road: dispatch helpers, test fixtures, and a
correctness-vs-CPU testing pattern.

**Architecture documents:** `docs/design/cuda-backend.md` (M2-T01).

### M2-T01 · Design doc: CUDA backend
Write `docs/design/cuda-backend.md`: stream model (which streams exist, who owns
them), error-handling strategy (CUDA errors → `Status`), allocator strategy (naive vs
caching pool, stream-ordered semantics), kernel source organization (`src/kernels/`
layout, header/impl split, dispatch conventions), supported architectures
(sm_80/86/89/90), and GPU testing strategy.
**Depends on:** M1-T09.
**Acceptance criteria:**
- Doc answers: how does a CUDA error inside a kernel surface to the caller? Who synchronizes and when? How do tests assert kernel correctness?
- Reviewed against ADR-002 module rules (kernels may not depend on scheduler/runtime).

### M2-T02 · CMake CUDA integration
Enable CUDA as a first-class language behind an `ENGINE_ENABLE_CUDA` option (default
ON, auto-detect). Set `CMAKE_CUDA_ARCHITECTURES` (80;86;89;90), C++20 for device code,
and make the CPU-only build (CI) compile cleanly with all CUDA targets excluded.
**Depends on:** M2-T01.
**Acceptance criteria:**
- With CUDA toolkit present: `.cu` files compile into `src/cuda/` and `src/kernels/` targets.
- With `ENGINE_ENABLE_CUDA=OFF` (or no toolkit): full build + tests pass; `Device::kCUDA` operations return `Unimplemented`.

### M2-T03 · CUDA error handling & device utilities
`src/cuda/cuda_utils.h`: `CUDA_CHECK` (fatal) and `CUDA_RETURN_IF_ERROR` (→ `Status`)
macros capturing file/line and error string; device introspection (`device_count()`,
`DeviceProperties` with name, SM count, memory, compute capability); `ScopedSetDevice`
RAII.
**Depends on:** M2-T02.
**Acceptance criteria:**
- GPU test: querying properties of device 0 returns sane values; deliberately bad call surfaces a `Status` with the CUDA error string embedded.
- Tests are skipped (not failed) on machines without a GPU, via a shared test predicate.

### M2-T04 · Stream & event wrappers
`src/cuda/stream.h`: RAII `CudaStream` (non-blocking), `CudaEvent` (timing and
sync variants), `record/wait/synchronize/elapsed_ms`, and a per-device default stream
accessor.
**Depends on:** M2-T03.
**Acceptance criteria:**
- GPU tests: event ordering across two streams via `stream_wait_event`, elapsed time of a known-duration workload > 0, destruction order safety (event outliving stream misuse is documented).

### M2-T05 · CUDA device allocator (naive)
`CudaAllocator` implementing the M1-T05 `Allocator` interface over
`cudaMalloc`/`cudaFree`, device-tagged `Buffer`s, and `Tensor::empty` support for
`Device::kCUDA`.
**Depends on:** M2-T04, M1-T05.
**Acceptance criteria:**
- GPU tests: allocate/free cycles leak-free (`cudaMemGetInfo` delta check), device tensors report correct device, huge-allocation failure returns `ResourceExhausted` (not crash).

### M2-T06 · Caching pool allocator
`src/memory/caching_allocator.h`: a caching allocator over the naive one — size-class
binning, free-list reuse, stats (`bytes_allocated`, `bytes_reserved`, hit/miss
counts), `release_cached()`. This is the memory-pooling foundation for the whole
engine.
**Depends on:** M2-T05.
**Acceptance criteria:**
- GPU tests: alloc→free→alloc of same size reuses the block (no cudaMalloc call — verified via stats), stats accurate through a scripted sequence, `release_cached()` returns memory to the driver.
- Concurrent allocation from multiple threads is safe (stress test).

### M2-T07 · Pinned memory & host↔device transfer
`PinnedCpuAllocator` for page-locked host memory; `copy(dst, src, stream)` supporting
H2D/D2H/D2D async transfers; `Tensor::to(device, stream)` returning a new tensor.
**Depends on:** M2-T06.
**Acceptance criteria:**
- GPU tests: round-trip H2D→D2H preserves bytes for every dtype; async copy on a stream + event sync semantics verified; pinned round-trip works.
- Copy between mismatched shapes/dtypes returns `InvalidArgument`.

### M2-T08 · Kernel launch infrastructure & first elementwise kernels
`src/kernels/`: launch-config helpers (grid/block calculation, `CUDA_1D_KERNEL_LOOP`),
a dtype-dispatch macro (`DISPATCH_FLOATING_TYPES`), and elementwise kernels: `add`,
`mul`, `scale`, and `cast` (fp32↔fp16↔bf16) operating on contiguous tensors.
**Depends on:** M2-T07.
**Acceptance criteria:**
- GPU tests: each kernel matches the CPU implementation via `allclose` across shapes (including non-multiple-of-blockDim sizes) and dtypes.
- Kernels validate inputs (shape/dtype/device match) and return `Status` on violation.

### M2-T09 · GPU test infrastructure
Shared `CudaTestFixture` (skips without GPU, sets device, provides stream +
allocator), `expect_tensors_close(gpu_tensor, cpu_reference)` helper that handles the
D2H copy, and documentation of the kernel-testing pattern in `tests/README.md`. Add an
optional CI workflow file for a self-hosted/manual GPU runner (may stay dormant).
**Depends on:** M2-T08.
**Acceptance criteria:**
- Existing GPU tests are migrated to the fixture; running the suite on a no-GPU machine reports skips, not failures.
- `tests/README.md` documents how to write a kernel test (CPU-reference pattern).

---

## Milestone 3 — Model Loading & Tokenization

**Overview.** Read real model artifacts from disk: HuggingFace `config.json`,
safetensors checkpoints (single and sharded), and `tokenizer.json` byte-level BPE
tokenizers (the format used by Llama 3 and Qwen 2+). Also establishes the
golden-fixture tooling — Python scripts that use HuggingFace libraries to produce
reference outputs, which every numerical test from here on relies on. After this
milestone the engine can load a real model's weights into tensors and tokenize text
identically to HuggingFace.

**Architecture documents:** `docs/design/model-loading.md` (M3-T01).

### M3-T01 · Design doc: model loading & tokenization
Write `docs/design/model-loading.md`: the load pipeline (config → weight discovery →
mmap → tensor registry), the internal weight-naming convention and per-architecture
mapping tables, dtype policy at load (preserve checkpoint dtype), tokenizer scope
(byte-level BPE from `tokenizer.json`; sentencepiece explicitly deferred), and the
golden-fixture strategy.
**Depends on:** M2-T09.
**Acceptance criteria:**
- Doc includes the load-pipeline diagram and the tokenizer scope with rationale.
- Fixture strategy specifies where fixtures live (`tests/fixtures/`), how they're generated, and their size budget.

### M3-T02 · Golden-fixture generation tooling
`tools/gen_fixtures/`: Python package (pinned `requirements.txt`: torch, transformers,
tokenizers, safetensors) with a CLI that generates: (a) tokenizer encode/decode test
vectors for a given HF model, (b) a tiny random-weight model checkpoint
(safetensors + config) with recorded layer-by-layer and end-to-end outputs for a
fixed input. Deterministic via fixed seeds.
**Depends on:** M3-T01.
**Acceptance criteria:**
- Running the CLI twice produces byte-identical fixtures; a `Makefile`/script target regenerates everything.
- A tiny-Llama-style fixture (2 layers, small dims) + tokenizer vectors for one Llama-family and one Qwen-family tokenizer are committed under `tests/fixtures/` (small enough for git).
- `tools/README.md` documents usage; CI does not require Python (fixtures are committed).

### M3-T03 · Model config parser
`src/model/config.h`: parse HF `config.json` into `ModelConfig` (architecture string,
hidden/intermediate size, layer/head/kv-head counts, head dim, vocab size, RoPE theta
& scaling, norm epsilon, max position, tie-word-embeddings, torch_dtype), with
validation and unknown-field tolerance.
**Depends on:** M3-T02.
**Acceptance criteria:**
- Unit tests parse committed real config.json files (Llama-3-style, Qwen2-style) and assert every field; malformed/missing-required-field cases return `InvalidArgument` with the field name.

### M3-T04 · Safetensors file parser
`src/model/safetensors.h`: mmap a `.safetensors` file, parse the JSON header, expose
`{name → (dtype, shape, data span)}` with zero-copy views into the mapping. Handle
alignment, bounds validation, and dtype string mapping (F32/F16/BF16/I8/…).
**Depends on:** M3-T03.
**Acceptance criteria:**
- Unit tests against a fixture file: metadata correct, tensor bytes match expected values, truncated/corrupt header cases return errors (fuzz-ish negative tests).
- File stays mapped while any tensor view is alive (lifetime tested).

### M3-T05 · Sharded checkpoint support
Support `model.safetensors.index.json`: resolve tensor→shard mapping, open shards
lazily, present a unified `{name → tensor view}` interface identical to the
single-file case.
**Depends on:** M3-T04.
**Acceptance criteria:**
- Unit tests with a 2-shard fixture: all tensors resolvable, missing-shard and inconsistent-index errors surfaced clearly.

### M3-T06 · Weight-name mapping
`src/model/weight_map.h`: per-architecture mapping from HF checkpoint names
(`model.layers.0.self_attn.q_proj.weight`) to internal canonical names, with a report
of missing and unexpected weights at load.
**Depends on:** M3-T05.
**Acceptance criteria:**
- Unit tests: full mapping for the tiny-Llama fixture resolves every weight; deliberately removing a weight produces a load error naming it; extra weights produce a warning list.

### M3-T07 · Model loader
`src/model/loader.h`: `load_model(path) → StatusOr<LoadedModel>` combining config
parse, shard resolution, name mapping, and materializing weights as CPU tensors
(dtype preserved from checkpoint). Progress logging for large models.
**Depends on:** M3-T06.
**Acceptance criteria:**
- Integration test loads the tiny fixture end-to-end; spot-check tensor values against fixture-recorded expectations.
- Load errors (bad path, unsupported architecture) produce actionable messages.

### M3-T08 · Tokenizer model parsing
`src/tokenizer/`: parse `tokenizer.json` — vocab, merges, added/special tokens
(with `special`, `lstrip`/`rstrip` flags), byte-level alphabet mapping. Build the
in-memory structures for encoding (merge ranks) and decoding (id → token bytes).
**Depends on:** M3-T02 (fixtures).
**Acceptance criteria:**
- Unit tests: vocab size, specific token↔id pairs, special-token metadata for both fixture tokenizers.
- Unsupported tokenizer types (sentencepiece/unigram) rejected with a clear error.

### M3-T09 · BPE encoding
Implement byte-level BPE encode: pre-tokenization regex split (the GPT-2/Llama-3
pattern), byte-to-unicode mapping, merge loop, special-token splitting
(added tokens are matched before BPE). API: `encode(text, add_special_tokens) →
vector<int32>`.
**Depends on:** M3-T08.
**Acceptance criteria:**
- Golden tests: encodings byte-identical to HF `tokenizers` for the committed test vectors (ASCII, Unicode incl. CJK + emoji, whitespace edge cases, special tokens embedded in text) for both tokenizer fixtures.

### M3-T10 · Decoding & incremental detokenization
Implement `decode(ids, skip_special_tokens)` and an incremental
`DetokenizerStream` that emits valid UTF-8 as tokens arrive (buffering incomplete
multi-byte sequences) — required later for streaming generation.
**Depends on:** M3-T09.
**Acceptance criteria:**
- Golden tests: decode(encode(x)) == x for all test vectors; streaming decode emits identical total text with no invalid UTF-8 at any intermediate step (tested token-by-token, including a multi-token emoji).

---

## Milestone 4 — CPU Reference Engine

**Overview.** Implement a complete, unoptimized CPU forward pass and greedy generation
loop for Llama/Qwen-family models. This is the **correctness oracle**: every GPU
kernel and every optimization from now on is validated against it (and it, in turn, is
validated against HuggingFace fixtures). Speed is a non-goal; clarity is the goal.
This milestone also fixes the model-execution interfaces (module structure, KV-cache
interface) that the GPU engine will implement.

**Architecture documents:** `docs/design/model-execution.md` (M4-T01).

### M4-T01 · Design doc: model execution
Write `docs/design/model-execution.md`: the layer/module structure (Attention, MLP,
DecoderLayer, Model), how weights bind to modules, the forward-pass signature (token
ids + positions + KV cache → logits), batch/sequence dimension conventions, the
KV-cache interface (v0: append-only per-sequence), and how CPU and GPU
implementations share the same interfaces.
**Depends on:** M3-T10.
**Acceptance criteria:**
- Doc specifies the exact `Model::forward` contract used by both backends, and the GQA layout conventions (head counts, head_dim, kv repeat).
- KV-cache interface v0 is specified with an explicit note on what M7 (paged) will change.

### M4-T02 · CPU GEMM & Linear layer
`src/cpu/`: a correct (OpenMP-parallel, cache-blocked but simple) fp32 GEMM; weights
in fp16/bf16 are converted to fp32 on the fly. `Linear` module (weight, optional
bias).
**Depends on:** M4-T01.
**Acceptance criteria:**
- Unit tests vs fixture GEMM results across shapes (including k=1, skinny/wide) within fp32 tolerance.
- A 512×512×512 GEMM completes in < 1 s (sanity, not perf).

### M4-T03 · CPU normalization & activation ops
RMSNorm (with epsilon, weight), SiLU, elementwise multiply (for SwiGLU), residual
add, and numerically-stable softmax — all CPU, fp32 accumulation.
**Depends on:** M4-T02.
**Acceptance criteria:**
- Golden tests vs fixture outputs for each op (including RMSNorm on bf16 inputs, softmax with large-magnitude logits).

### M4-T04 · Embedding & RoPE (CPU)
Token-embedding lookup; rotary position embeddings matching HF Llama exactly
(half-rotation layout, configurable theta, optional scaling factors parsed in M3-T03).
Precompute cos/sin tables.
**Depends on:** M4-T03.
**Acceptance criteria:**
- Golden tests: RoPE output matches fixture for positions {0, 1, large}, head dims from config; embedding lookup matches fixture rows.

### M4-T05 · Causal attention (CPU)
Naive causal self-attention with GQA: project QKV, apply RoPE, scores = QKᵀ/√d with
causal mask, softmax, weighted sum, output projection. Supports prefill (T tokens)
against an existing cache of length P.
**Depends on:** M4-T04.
**Acceptance criteria:**
- Golden tests vs fixture attention-layer outputs (prefill from empty cache, and prefill continuing from a non-empty cache) with GQA (kv_heads < heads) covered.

### M4-T06 · KV cache v0
`src/kvcache/simple_cache.h`: per-sequence, per-layer contiguous append-only K/V
storage implementing the interface from M4-T01 (append, view, current length, reset).
CPU tensors for now; device-agnostic API.
**Depends on:** M4-T05.
**Acceptance criteria:**
- Unit test: decoding token-by-token with the cache produces logits equal to full-prompt recompute at every step (the fundamental KV-cache invariant), within fp32 tolerance.

### M4-T07 · Transformer block & full model forward (CPU)
Assemble `DecoderLayer` (attention + MLP + norms + residuals) and `Model` (embedding →
N layers → final norm → lm_head, honoring tied embeddings). Prefill returns logits for
the last position (and optionally all positions for testing).
**Depends on:** M4-T06.
**Acceptance criteria:**
- Golden test: end-to-end logits for the tiny-Llama fixture match HF within tolerance (report max abs diff; threshold documented in the test).
- Per-layer debug hook allows dumping intermediate activations (used to localize future regressions).

### M4-T08 · Architecture registry
`src/model/registry.h`: map `architectures[0]` strings (`LlamaForCausalLM`,
`Qwen2ForCausalLM`, …) to model builders; unsupported architectures produce a clean
error listing supported ones. Builder wires `ModelConfig` + weights → `Model`.
**Depends on:** M4-T07.
**Acceptance criteria:**
- Unit tests: registry resolves both families, rejects unknown; adding an architecture requires only a registration call (verified by a test-local dummy arch).

### M4-T09 · Greedy generation loop
`src/engine/generator.h`: prefill + autoregressive decode loop with greedy argmax,
EOS-token and max-new-tokens stopping. Returns generated ids; hooks for per-token
callbacks (streaming later).
**Depends on:** M4-T08.
**Acceptance criteria:**
- Golden test: greedy continuation of fixture prompts matches HF `generate(do_sample=False)` token-for-token for ≥ 32 tokens on the tiny model.
- Determinism test: two runs produce identical outputs.

### M4-T10 · Qwen-family support (CPU)
Implement Qwen2/2.5 differences (QKV bias, its config fields) on the shared modules;
add a tiny Qwen-style fixture.
**Depends on:** M4-T09.
**Acceptance criteria:**
- Golden logits + greedy-generation tests pass for the Qwen fixture, reusing existing modules (diff should be config/wiring, not new layer code).

---

## Milestone 5 — Single-GPU Execution Engine

**Overview.** Port the forward pass to GPU: cuBLAS GEMMs plus custom CUDA kernels for
everything else, validated kernel-by-kernel against the CPU reference. Ends with
end-to-end generation on a real model (~1B class) on one GPU and a first
tokens/second baseline. The kernels here are deliberately straightforward — optimized
variants come in M11 behind the same interfaces.

**Architecture documents:** `docs/design/gpu-execution.md` (M5-T01).

### M5-T01 · Design doc: GPU execution
Write `docs/design/gpu-execution.md`: compute dtype policy (bf16 weights/activations,
fp32 accumulation in norms/softmax/logits), workspace-buffer strategy (pre-allocated,
reused across layers), stream discipline for the forward pass, kernel interface
conventions, and the kernel-validation methodology (every kernel tested vs CPU
reference).
**Depends on:** M4-T10.
**Acceptance criteria:**
- Doc specifies where fp32 accumulation is mandatory and why; workspace sizing formula documented.

### M5-T02 · cuBLAS integration & GPU Linear
`src/cuda/cublas.h`: handle wrapper (per-device, stream-bound), GEMM for
fp16/bf16 with fp32 accumulate, correct row-major↔column-major mapping. GPU `Linear`
module using it.
**Depends on:** M5-T01.
**Acceptance criteria:**
- GPU tests: GEMM matches CPU reference across transpose combinations and shapes used by the model (qkv, o, gate/up/down, lm_head); bias addition covered.

### M5-T03 · RMSNorm kernel
Row-parallel RMSNorm kernel (fp32 accumulation, vectorized loads where alignment
allows) for fp16/bf16.
**Depends on:** M5-T02.
**Acceptance criteria:**
- GPU tests vs CPU reference: hidden sizes {odd, 1024, 4096, 8192}, batch of rows, both dtypes, tolerance documented.

### M5-T04 · RoPE kernel
Kernel applying rotary embeddings to Q and K in-place given positions (arbitrary
per-token positions, supporting later batched/paged use).
**Depends on:** M5-T03.
**Acceptance criteria:**
- GPU tests vs CPU reference for contiguous and scattered position arrays, GQA head counts, both dtypes.

### M5-T05 · Activation & fusion-ready elementwise kernels
Kernels: SiLU-and-mul (SwiGLU combine, fused), residual add. Interfaces shaped so M11
fusions can swap in.
**Depends on:** M5-T04.
**Acceptance criteria:**
- GPU tests vs CPU reference across shapes/dtypes; in-place residual add verified.

### M5-T06 · Embedding & logits path
Embedding-lookup kernel (ids → rows); lm_head via cuBLAS with fp32 logits output;
handles tied embeddings.
**Depends on:** M5-T05.
**Acceptance criteria:**
- GPU tests: lookup matches CPU for random id sets (incl. repeated ids); logits match CPU reference within tolerance on fixture.

### M5-T07 · Prefill attention (GPU, naive)
Single-sequence prefill attention: cuBLAS batched GEMM for QKᵀ and AV, custom
masked-softmax kernel (fp32, causal mask, supports cache offset), GQA via KV head
indexing (no materialized repeat).
**Depends on:** M5-T06.
**Acceptance criteria:**
- GPU tests vs CPU reference: prefill from empty and non-empty cache, GQA, sequence lengths {1, 17, 512}, tolerance documented.

### M5-T08 · GPU KV cache v0 & append kernel
Contiguous per-sequence GPU K/V cache implementing the M4 interface; kernel appending
new K/V (post-RoPE) at the current length.
**Depends on:** M5-T07.
**Acceptance criteria:**
- GPU tests: append + view round-trip exact; incremental-vs-recompute invariant (M4-T06 test pattern) holds on GPU.

### M5-T09 · Decode attention kernel v0
Kernel for single-token decode: one query per sequence attending over the cached K/V
(one block per head, shared-memory reduction, fp32 accumulation, GQA).
**Depends on:** M5-T08.
**Acceptance criteria:**
- GPU tests vs CPU reference across cache lengths {1, 63, 64, 65, 2048} and GQA configs; matches prefill-path result for the same token.

### M5-T10 · GPU model forward & generation
Wire GPU modules into the M4 `Model` interface: weight upload at load (with progress
logging), workspace allocation, full prefill+decode forward, greedy loop reusing
M4-T09 (argmax on CPU from copied-back logits for now).
**Depends on:** M5-T09.
**Acceptance criteria:**
- Tiny-fixture greedy generation on GPU matches the CPU reference token-for-token.
- A real ~1B model (e.g. Llama-3.2-1B or Qwen2.5-0.5B) loads and generates coherent text (manual artifact: sample outputs committed to the PR description).

### M5-T11 · GPU argmax & throughput smoke benchmark
Batched argmax kernel (row-wise over logits) removing the D2H logits copy for greedy;
`benchmarks/bench_generate`: measures prefill and decode tokens/sec for a given model
and prompt lengths, prints a small report.
**Depends on:** M5-T10.
**Acceptance criteria:**
- Argmax kernel tested vs CPU (ties resolved to lowest index, documented).
- Benchmark runs on the 1B model and reports stable numbers (±5% across runs); baseline recorded in `benchmarks/BASELINES.md`.

---

## Milestone 6 — Sampling & Generation Controls

**Overview.** Replace greedy-only generation with a full sampling pipeline:
temperature, top-k, top-p, penalties, seeded per-request RNG, stop conditions, and
logprobs. Implemented first as a clear CPU/GPU-hybrid pipeline with correctness and
statistical tests; batched GPU sampling closes the milestone. These are exactly the
knobs the API layer (M9) will expose.

**Architecture documents:** sampling pipeline section added to
`docs/design/model-execution.md` (M6-T01).

### M6-T01 · SamplingParams & pipeline skeleton
Define `SamplingParams` (temperature, top_k, top_p, repetition/presence/frequency
penalties, seed, max_tokens, stop tokens/strings, logprobs count) with validation;
document the pipeline stage order in the design doc; implement the greedy path through
the new pipeline structure.
**Depends on:** M5-T11.
**Acceptance criteria:**
- Validation unit tests (rejects temperature<0, top_p∉(0,1], etc.).
- Greedy generation output unchanged (regression test passes).

### M6-T02 · Temperature, top-k, top-p sampling
Implement the core sampler on CPU over GPU-computed logits (D2H of final-position
logits): temperature scale → top-k filter → top-p (nucleus) filter → categorical
sample with a per-request counter-based RNG (Philox) so results are reproducible per
(seed, step).
**Depends on:** M6-T01.
**Acceptance criteria:**
- Statistical tests: empirical distribution over 10k draws matches expected within chi-square tolerance for known logits; top-k/top-p masks verified exactly.
- Same seed ⇒ identical sequence across runs; different requests with different seeds are independent.

### M6-T03 · Repetition, presence & frequency penalties
Apply penalties over the request's token history (prompt + generated, matching
OpenAI/vLLM semantics — documented choice) before temperature.
**Depends on:** M6-T02.
**Acceptance criteria:**
- Unit tests with hand-computed logit adjustments for each penalty type and combinations; no-op when at default values (exact logits equality).

### M6-T04 · Stop conditions & finish reasons
EOS handling (including multi-EOS token sets from config), stop-token ids, stop
strings (matched on the incrementally-detokenized stream, handling stop strings that
span token boundaries), max_tokens; produce `finish_reason` (stop/length).
**Depends on:** M6-T03, M3-T10.
**Acceptance criteria:**
- Unit tests: stop string split across two tokens is caught and trailing text is trimmed; max_tokens exact; finish_reason correct in each case.

### M6-T05 · Logprobs
Return chosen-token logprob and top-N logprobs per step (computed from the same
logits the sampler saw, post-penalties or raw — documented, matching OpenAI
semantics).
**Depends on:** M6-T04.
**Acceptance criteria:**
- Unit tests: logprobs sum ≈ 1 in prob space over full vocab on a small-vocab fixture; top-N ordering matches reference computation; greedy chosen-token logprob equals max.

### M6-T06 · Batched GPU sampling kernels
Move the hot path to GPU for batch-of-sequences sampling: fused
temperature+top-k+top-p sampling kernel(s) over batched logits with per-request
params and Philox states; CPU path retained as reference.
**Depends on:** M6-T05.
**Acceptance criteria:**
- GPU sampler output distribution matches CPU sampler statistically; given identical RNG counters, GPU and CPU pick identical tokens on identical filtered distributions.
- Microbench: batched GPU sampling for 64 sequences × 128k vocab under 1 ms on target hardware (number recorded, threshold advisory).

---

## Milestone 7 — Paged KV Cache & Block Manager

**Overview.** Replace the contiguous per-sequence cache with a paged (block-based) KV
cache — the memory architecture that makes continuous batching and prefix caching
possible. Fixed-size token blocks are allocated from a GPU block pool; sequences hold
block tables mapping logical positions to physical blocks. Attention kernels read
through the indirection. Designed from day one with reference counting so M10 (prefix
caching) is an extension, not a rewrite.

**Architecture documents:** `docs/design/paged-kv-cache.md` (M7-T01).

### M7-T01 · Design doc: paged KV cache
Write `docs/design/paged-kv-cache.md`: physical layout
(`[num_blocks, 2, layer… ] `— choose and justify K/V layout per block for kernel
access patterns), block size choice (16 tokens default, rationale), capacity
calculation (`gpu_memory_utilization` fraction after weights + workspace), block
tables, slot mapping, reference counting, and the preemption + future prefix-caching
interactions.
**Depends on:** M6-T06.
**Acceptance criteria:**
- Doc specifies exact memory layout with a worked example and the formula for blocks-per-GPU; refcount lifecycle diagram included.

### M7-T02 · Block pool & allocator
`src/kvcache/block_pool.h`: pool sized from config/free memory, free-list
allocate/free of block ids, per-block refcounts, stats (used/free/total). Pure
bookkeeping — no kernels — so it's fully unit-testable on CPU.
**Depends on:** M7-T01.
**Acceptance criteria:**
- Unit tests: exhaustion returns `ResourceExhausted` (no crash); refcount double-free detected; stats accurate through scripted alloc/free/share sequences.

### M7-T03 · Block table & sequence cache handle
`src/kvcache/block_table.h`: per-sequence logical→physical mapping, append-token
(allocating blocks on boundary crossings), slot-mapping computation for a batch of
token positions, free-on-completion.
**Depends on:** M7-T02.
**Acceptance criteria:**
- Unit tests: growth across block boundaries, slot mappings for prefill (T tokens) and decode (1 token) hand-verified, blocks returned to pool on free.

### M7-T04 · KV write (scatter) kernel
Kernel writing a batch of new K/V vectors into paged storage given a slot mapping
(one entry per token). Replaces the contiguous append kernel.
**Depends on:** M7-T03.
**Acceptance criteria:**
- GPU tests: scattered writes land in the exact expected block/offset (readback comparison vs a CPU-simulated paged layout), across boundary-straddling prefills and single-token decodes.

### M7-T05 · Paged decode attention kernel
Decode attention reading K/V through the block table (block-table pointer array per
sequence), GQA, fp32 accumulation.
**Depends on:** M7-T04.
**Acceptance criteria:**
- GPU tests: matches the M5-T09 contiguous decode kernel results exactly (same inputs materialized both ways) for cache lengths crossing many blocks, including length exactly at a block boundary.

### M7-T06 · Paged prefill attention path
Prefill attention over paged cache: gather cached K/V to a contiguous workspace for
the cuBLAS-based prefill path (simple, correct; flash-style paged prefill comes in
M11). Handles prefill-continuation (cache hit + new tokens) correctly.
**Depends on:** M7-T05.
**Acceptance criteria:**
- GPU tests vs CPU reference: prefill with existing paged cache content matches; gather kernel independently tested.

### M7-T07 · Engine integration
Swap the paged cache into the single-request GPU generation path behind the M4 cache
interface; wire capacity config (`gpu_memory_utilization`); expose cache stats.
**Depends on:** M7-T06.
**Acceptance criteria:**
- End-to-end regression: tiny-fixture greedy output identical to pre-paging engine; 1B-model generation works with memory stats logged.
- `bench_generate` shows no more than 10% decode-throughput regression vs M5 baseline (recorded).

### M7-T08 · Exhaustion behavior & metrics
Defined behavior when the pool runs dry mid-generation (error for now — preemption
arrives with the scheduler in M8); cache usage metrics API (blocks used/free,
utilization) consumed later by scheduler and metrics endpoint.
**Depends on:** M7-T07.
**Acceptance criteria:**
- Test: a generation that would exceed capacity fails gracefully with `ResourceExhausted`, pool state fully reclaimed afterwards (no leaked blocks — asserted via stats).

---

## Milestone 8 — Continuous Batching Scheduler & Runtime

**Overview.** The engine becomes a multi-request system: an asynchronous runtime
accepts requests into a queue; a scheduler decides, every step, which sequences to
prefill and which to decode under a token budget and KV-block availability; batched
kernels execute ragged batches; results stream back per-request. Includes preemption
(evict + recompute) when memory runs out, and cancellation. This is the architectural
heart of the engine — the design doc matters more here than anywhere else.

**Architecture documents:** `docs/design/scheduler-runtime.md` (M8-T01).

### M8-T01 · Design doc: request lifecycle, scheduler & runtime
Write `docs/design/scheduler-runtime.md`: request/sequence state machine
(WAITING→RUNNING→(PREEMPTED)→FINISHED), the step loop, scheduling policy v1 (FCFS,
token budget, block-availability admission, decode-priority), batch composition
(separate prefill/decode passes vs mixed — choose and justify), threading model
(client threads → lock-free-ish queue → single engine thread → per-request output
channels), preemption policy (evict-and-recompute), and cancellation semantics.
**Depends on:** M7-T08.
**Acceptance criteria:**
- State machine diagram with every legal transition; step-loop pseudocode; explicit invariants (e.g. "a RUNNING sequence always holds all blocks it needs for its next token").

### M8-T02 · Request & sequence abstractions
`src/runtime/request.h`: `Request` (id, prompt ids, `SamplingParams`, arrival time),
`Sequence` (state, token ids, block table handle, generation progress), and a
thread-safe per-request output channel (tokens + finish info) supporting
blocking and polling consumption.
**Depends on:** M8-T01.
**Acceptance criteria:**
- Unit tests: state transitions enforced (illegal transition = CHECK failure), output channel delivers in order across threads, channel close semantics on finish/cancel.

### M8-T03 · Engine API & request queue
`src/runtime/engine.h`: public async API — `submit(request) → RequestHandle`,
`handle.next_token()/await_completion()`, `cancel(id)`; internal waiting queue;
`Engine::step()` skeleton that the loop thread will drive.
**Depends on:** M8-T02.
**Acceptance criteria:**
- Unit tests with a mock model: submit/consume/cancel from multiple client threads; no deadlocks under a stress test (many submitters, random cancels).

### M8-T04 · Scheduler v1
`src/scheduler/scheduler.h`: pure decision component — given queue + running set +
block-pool stats + token budget, emit `SchedulerOutput`: sequences to prefill (with
lengths), sequences to decode, sequences to preempt. FCFS admission, decode-first
priority, block-availability check via M7-T02 stats. No CUDA dependencies —
deterministic and unit-testable.
**Depends on:** M8-T03.
**Acceptance criteria:**
- Table-driven unit tests: admission blocked when blocks insufficient, token budget respected across mixed prefill sizes, decode starvation impossible (decodes always scheduled first), preemption chooses the documented victim (latest-arrived).

### M8-T05 · Batch assembly
`src/engine/batch.h`: build the flattened device inputs for a `SchedulerOutput` —
concatenated token ids, positions, sequence start offsets (cu_seqlens), slot mappings,
block-table tensor, per-request sampling metadata — in pinned staging buffers with one
H2D copy.
**Depends on:** M8-T04.
**Acceptance criteria:**
- Unit tests: assembled metadata hand-verified for scenarios (2 prefills of different lengths, 3 decodes, mixed), tensor contents exact.

### M8-T06 · Varlen batched prefill attention
Extend prefill attention to ragged batches using cu_seqlens (per-sequence lengths,
shared kernels loop over sequences; still naive-but-correct).
**Depends on:** M8-T05.
**Acceptance criteria:**
- GPU tests: batch of {3 sequences, lengths 5/64/129} matches per-sequence single runs exactly (same outputs sequence-by-sequence).

### M8-T07 · Batched decode execution
Batched decode step: N sequences × 1 token through the full model using the paged
decode kernel over a batched block-table tensor; batched sampling (M6-T06) consumes
the result.
**Depends on:** M8-T06.
**Acceptance criteria:**
- GPU tests: batched decode logits match sequential single-sequence decode for each member; batch with heterogeneous cache lengths covered.

### M8-T08 · Engine loop integration
The continuous-batching loop: engine thread runs `step()` — schedule → assemble →
forward (prefills then decodes, or combined per design doc) → sample → append tokens →
deliver to channels → retire finished sequences. Streaming tokens flow to handles as
they're produced.
**Depends on:** M8-T07.
**Acceptance criteria:**
- Integration test: 8 concurrent greedy requests produce outputs identical to running each sequentially (the continuous-batching correctness invariant).
- Requests arriving mid-flight join batching without disturbing running sequences (test with staggered submission).
- Throughput sanity: 8 concurrent requests complete in well under 8× single-request time (recorded).

### M8-T09 · Preemption & recomputation
On block exhaustion during decode: preempt victim sequences (free their blocks,
state→PREEMPTED, back to queue head), resume later by re-prefilling
prompt+generated-so-far. Wire scheduler preemption decisions to engine actions.
**Depends on:** M8-T08.
**Acceptance criteria:**
- Test with an artificially tiny block pool: requests all complete correctly despite forced preemptions; preempted-request output identical to unpreempted run (greedy); no block leaks (pool stats zero at end).

### M8-T10 · Cancellation & per-request failure isolation
Cancel promptly frees blocks and closes the channel with `cancelled`; a per-request
error (e.g. sampler edge case) fails that request only, never the engine loop.
**Depends on:** M8-T09.
**Acceptance criteria:**
- Tests: cancel during WAITING, during prefill, during decode — all reclaim resources (stats-verified); injected per-request fault leaves other concurrent requests' outputs unchanged.

---

## Milestone 9 — Serving Layer (HTTP API)

**Overview.** Expose the engine over HTTP with an OpenAI-compatible API:
`/v1/completions` and `/v1/chat/completions` with SSE streaming, chat templates,
structured errors, health endpoints, and a production-shaped CLI + config system. After
this milestone the project is a *server* you can point existing OpenAI SDK clients at.

**Architecture documents:** `docs/design/server.md` + ADR-004 HTTP library choice
(M9-T01).

### M9-T01 · Design doc & ADR: server architecture
Write `docs/design/server.md` and ADR-004: HTTP library selection (evaluate e.g.
standalone Asio + a minimal HTTP layer, `cpp-httplib`, Drogon — criteria: streaming
support, thread model fit, dependency weight), server threading model relative to the
engine thread, API schema definitions, streaming design, backpressure policy.
**Depends on:** M8-T10.
**Acceptance criteria:**
- ADR records the library decision with a comparison table; design doc includes request-flow diagram from socket to engine channel and back.

### M9-T02 · HTTP server skeleton
`src/server/`: server bootstrap with the chosen library — `/health` (liveness) and
`/v1/models` endpoints, graceful startup/shutdown (drain-free for now), port/host
configuration, request logging with request ids.
**Depends on:** M9-T01.
**Acceptance criteria:**
- Integration test (real HTTP client): endpoints respond correctly; SIGTERM shuts the server down cleanly (test sends signal, asserts exit).

### M9-T03 · /v1/completions (non-streaming)
Parse and validate the OpenAI completions request (prompt, max_tokens, temperature,
top_p, stop, seed, logprobs, n=1 for now), map to `SamplingParams`, run via engine
handle, return the OpenAI response shape with `usage` token counts.
**Depends on:** M9-T02.
**Acceptance criteria:**
- Integration tests: valid request returns correct schema (checked field-by-field); the official `openai` Python client (or a schema-validating test) can parse the response; invalid params → 400 with structured error body.

### M9-T04 · SSE streaming (completions)
`stream: true` support: SSE chunks matching OpenAI's chunk schema, terminal `[DONE]`,
client-disconnect detection → engine cancel.
**Depends on:** M9-T03.
**Acceptance criteria:**
- Integration tests: streamed concatenation equals non-streamed text for the same seed; disconnect mid-stream cancels the engine request (verified via engine stats) within one step.

### M9-T05 · Chat templates
`src/server/chat_template.h`: render chat messages to prompt strings via built-in
templates per model family (Llama-3 header format, ChatML for Qwen), selected from
model config/tokenizer metadata with a config override. (Full Jinja evaluation is
deliberately out of scope — documented.)
**Depends on:** M9-T04.
**Acceptance criteria:**
- Golden tests: rendered prompts byte-identical to HF `apply_chat_template` fixtures for both families across system/user/assistant/multi-turn cases, including generation prompt suffix.

### M9-T06 · /v1/chat/completions (non-streaming + streaming)
Chat endpoint: message validation, template render, generation, chat response schema;
streaming variant with role/delta chunks.
**Depends on:** M9-T05.
**Acceptance criteria:**
- Integration tests mirror M9-T03/T04 for the chat schema; finish_reason and usage present and correct.

### M9-T07 · Error taxonomy & API robustness
Uniform error responses (OpenAI error JSON shape): 400 validation, 404 model, 429
queue-full, 500 internal with request id; malformed-JSON handling; engine `Status` →
HTTP mapping table; panic-free guarantee on arbitrary bodies.
**Depends on:** M9-T06.
**Acceptance criteria:**
- Table-driven integration tests for each error class; a small adversarial-body corpus (huge strings, wrong types, deep nesting) returns 4xx without crashing.

### M9-T08 · CLI & configuration system
`engine serve` CLI: flags (`--model`, `--host/--port`, `--gpu-memory-utilization`,
`--max-num-seqs`, `--max-model-len`, `--dtype`, …) + optional YAML config file with
documented precedence (flags > file > defaults); `--help` generated from one
definition source; config validation with actionable errors.
**Depends on:** M9-T07.
**Acceptance criteria:**
- Unit tests: precedence, validation failures name the offending key; `engine serve --help` output committed as a golden file (kept current by test).

### M9-T09 · Admission control & backpressure
Queue-depth limit → 429 with `Retry-After`; per-request queue timeout; max concurrent
streams; counters for rejected/timed-out requests.
**Depends on:** M9-T08.
**Acceptance criteria:**
- Load test script (committed under `tools/loadtest/`, may be Python): flooding beyond capacity yields 429s while in-flight requests complete correctly; no memory growth after the flood (RSS checked).

---

## Milestone 10 — Prefix Caching

**Overview.** Reuse KV blocks across requests that share prompt prefixes (system
prompts, few-shot preambles, multi-turn chat). Content-addressed block hashing over
the paged cache from M7: full blocks are keyed by the hash chain of their token
history; admission matches the longest cached prefix and shares those blocks
(refcounted), prefilling only the suffix. Big TTFT wins on real workloads.

**Architecture documents:** `docs/design/prefix-caching.md` (M10-T01).

### M10-T01 · Design doc: prefix caching
Write `docs/design/prefix-caching.md`: chained hash definition (hash of block tokens +
parent hash; include salt for collision policy), full-block-only granularity, cache
index structure, refcount interaction with the block pool, LRU eviction of
refcount-zero blocks, correctness argument (why shared blocks are immutable), metrics.
**Depends on:** M9-T09.
**Acceptance criteria:**
- Doc includes the immutability invariant proof-sketch and the eviction policy; hash collision handling decided and justified.

### M10-T02 · Block content hashing
Compute chained hashes as blocks fill (in `BlockTable`/cache write path): only
complete blocks are hashable; hash stored with the block.
**Depends on:** M10-T01.
**Acceptance criteria:**
- Unit tests: identical token prefixes yield identical hash chains; single-token difference in any position diverges all subsequent hashes; partial blocks are never hashed.

### M10-T03 · Cached-block index
`src/kvcache/prefix_index.h`: hash → block-id map with insert-on-fill,
lookup, refcount pinning on match, and remove-on-eviction. Owns nothing; coordinates
with the block pool.
**Depends on:** M10-T02.
**Acceptance criteria:**
- Unit tests: lookup hit pins the block (refcount asserted); index never returns a block whose content was freed (lifecycle stress test with scripted alloc/free).

### M10-T04 · Admission-time prefix reuse
On scheduling a new sequence: walk its prompt's hash chain, adopt the longest cached
prefix into its block table (shared, refcounted), schedule prefill only for the
remaining suffix; handle the edge case of a full-prompt hit (still recompute last
token for logits).
**Depends on:** M10-T03.
**Acceptance criteria:**
- Integration tests: two identical prompts — second skips prefix prefill (scheduler stats assert fewer prefill tokens) and produces token-identical greedy output; partial-overlap prompts reuse exactly the shared full blocks.

### M10-T05 · Eviction
When the pool needs blocks: evict refcount-zero cached blocks in LRU order
(index removal + pool return); allocation path falls back through eviction before
reporting exhaustion.
**Depends on:** M10-T04.
**Acceptance criteria:**
- Tests with a tiny pool: workload alternating unique prompts completes (eviction keeps up); previously-cached-then-evicted prefix re-prefills correctly; no leaks (stats zero at drain).

### M10-T06 · Metrics, toggle & end-to-end validation
`--enable-prefix-caching` flag (default on), hit-rate metrics (tokens reused / prompt
tokens), and an end-to-end benchmark scenario demonstrating the win.
**Depends on:** M10-T05.
**Acceptance criteria:**
- Correctness A/B test: a randomized multi-request workload produces identical outputs with caching on vs off (greedy).
- Benchmark: shared-system-prompt workload shows ≥ 2× TTFT improvement for cache-hit requests (recorded in `benchmarks/BASELINES.md`).

---

## Milestone 11 — Kernel Optimization & Latency Engineering

**Overview.** Close the performance gap: flash-style prefill attention, an optimized
paged decode kernel, key fusions, CUDA graphs for decode, and chunked prefill for
tail-latency control. Every optimization lands behind an existing interface with the
naive kernel kept as the test reference, and every ticket must show its benchmark
delta. A kernel microbenchmark harness comes first so claims are measured, not vibes.

**Architecture documents:** ADR-005 attention-kernel strategy (M11-T01).

### M11-T01 · ADR: attention strategy & microbenchmark harness
ADR-005: integrate FlashAttention-2 (vendored library) for prefill vs custom tiled
kernel — evaluate build cost, arch coverage, varlen+paged support; decide. Build the
kernel microbenchmark harness (`benchmarks/kernels/`): event-timed, warmup +
iterations, CSV/markdown output, used by all subsequent tickets.
**Depends on:** M10-T06.
**Acceptance criteria:**
- ADR records the decision with measurements/constraints; harness benches an existing kernel (RMSNorm) and produces stable numbers (±3% run-to-run).

### M11-T02 · Flash prefill attention
Land the chosen prefill attention (FA2 integration or custom tiled online-softmax
kernel) with varlen (cu_seqlens) support, behind the attention interface; naive path
retained under a flag as reference.
**Depends on:** M11-T01.
**Acceptance criteria:**
- Correctness: matches naive path within documented tolerance across GQA configs, lengths {1, 128, 2048, 8192}, batch raggedness.
- Bench: ≥ 3× naive prefill attention time at 2k context (recorded); end-to-end prefill throughput improvement recorded in BASELINES.md.

### M11-T03 · Optimized paged decode attention
Optimize the paged decode kernel: vectorized 16B loads, warp-level softmax reductions,
split-K across cache length with a reduction pass (flash-decoding style) for long
contexts, tuned launch configs per head-dim.
**Depends on:** M11-T02.
**Acceptance criteria:**
- Correctness vs naive paged kernel across cache lengths {16, 512, 4k, 16k}.
- Bench: ≥ 2× naive at 4k context; long-context (16k) decode latency sublinear vs naive (numbers recorded).

### M11-T04 · Kernel fusions
Fuse: (a) residual-add + RMSNorm, (b) RoPE + KV-cache write. Wire into the forward
pass; unfused paths remain for reference testing.
**Depends on:** M11-T03.
**Acceptance criteria:**
- Correctness: fused == unfused within tolerance on fixture model (end-to-end logits equality test).
- Bench: measurable per-layer improvement and end-to-end decode tokens/sec gain recorded.

### M11-T05 · CUDA graphs for decode
Capture the decode step as CUDA graphs for a bucketed set of batch sizes
(1,2,4,8,…,max); replay with updated inputs (static buffers/pointers designed in);
fallback to eager for uncaptured shapes; interaction with cuBLAS handled (workspace,
stream capture mode).
**Depends on:** M11-T04.
**Acceptance criteria:**
- Correctness: graph-replayed decode output identical to eager (token-for-token over a long generation).
- Bench: small-batch decode latency improvement ≥ 15% (recorded); graph capture time and memory overhead documented.

### M11-T06 · Chunked prefill
Scheduler v2 feature: split long prompts into fixed-size chunks scheduled alongside
decodes (bounding per-step latency); config knob `--max-num-batched-tokens` becomes
the binding budget; prefill-continuation path (M7-T06) already supports it.
**Depends on:** M11-T05.
**Acceptance criteria:**
- Correctness: chunked prefill output identical to unchunked (greedy, multiple chunk sizes incl. chunk boundary == block boundary).
- Bench: P99 inter-token latency of concurrent decodes improves ≥ 3× when a long prompt arrives mid-stream (scenario test, numbers recorded).

### M11-T07 · Sync & transfer hygiene
Eliminate unnecessary synchronization in the step loop: audit with nsys, move sampling
inputs/outputs to pinned staging with async copies, batch small H2D metadata copies
into one, remove hidden syncs (e.g. default-stream interactions, pageable copies).
**Depends on:** M11-T06.
**Acceptance criteria:**
- An nsys trace before/after is summarized in the PR (screenshots/notes in `docs/perf-notes.md`); step-loop CPU gap between kernels reduced; end-to-end decode tokens/sec change recorded (must be ≥ 0).

### M11-T08 · Performance baseline vs reference engine
Document a rigorous comparison on fixed hardware: this engine vs vLLM (same model,
same workload, same GPU) for offline throughput and serving latency; identify the
top-3 remaining gaps with profiles.
**Depends on:** M11-T07.
**Acceptance criteria:**
- `docs/perf-comparison.md` with methodology (versions, hardware, commands), results table, and gap analysis; results also appended to BASELINES.md.

---

## Milestone 12 — Quantization

**Overview.** Serve quantized models: weight-only INT8 (round-to-nearest at load) and
INT4 group-quantized checkpoints (AWQ/GPTQ formats) with custom dequant kernels, plus
FP8 KV-cache storage. Halves-to-quarters memory footprint, enabling larger models per
GPU. Quantized layers slot in behind the `Linear` interface; accuracy is validated
statistically against the fp16 baseline, not vibes.

**Architecture documents:** `docs/design/quantization.md` (M12-T01).

### M12-T01 · Design doc: quantization
Write `docs/design/quantization.md`: scope (weight-only; activations stay bf16/fp16),
supported formats (RTN INT8 per-channel; AWQ & GPTQ INT4 group-wise — layouts
documented bit-exactly), `QuantizedLinear` abstraction, kernel plan (dequant-GEMV for
decode, dequant-to-dense + cuBLAS for prefill), accuracy-validation methodology
(top-1 agreement rate + logit MSE on a prompt corpus vs fp16).
**Depends on:** M11-T08.
**Acceptance criteria:**
- Layout diagrams for AWQ and GPTQ packed formats (bit positions, group scales/zeros); accuracy methodology with thresholds defined.

### M12-T02 · Quantized weight containers & config detection
Activate `kInt4`/packed handling in the tensor layer as opaque byte tensors +
metadata; parse HF `quantization_config` (method, bits, group size, sym/asym);
`QuantizedLinear` module skeleton holding qweight/scales/zeros.
**Depends on:** M12-T01.
**Acceptance criteria:**
- Unit tests: config detection for AWQ and GPTQ checkpoint configs; container shape/metadata validation errors are actionable.

### M12-T03 · INT8 weight-only path
Load-time RTN INT8 per-channel quantization of a bf16 checkpoint (`--dtype int8`);
dequant kernel + cuBLAS GEMM path; dequant-GEMV fused kernel for decode (batch-1-ish
shapes).
**Depends on:** M12-T02.
**Acceptance criteria:**
- Kernel tests vs Python-fixture dequant references (exact integer math).
- E2E: 1B model at INT8 — memory reduction ≈ 2× on weights (measured), top-1 token agreement vs fp16 ≥ 98% on the test corpus, greedy fixture outputs recorded.

### M12-T04 · AWQ/GPTQ checkpoint loading
Load pre-quantized safetensors: qweight/qzeros/scales (+ g_idx for GPTQ) unpacking
into `QuantizedLinear`, weight-map extensions, validation of group-size consistency
with config.
**Depends on:** M12-T03.
**Acceptance criteria:**
- Unit tests against a small real AWQ and GPTQ checkpoint fixture (generated via AutoAWQ/GPTQ in `tools/`, committed small): unpacked values match Python reference dequant exactly.

### M12-T05 · INT4 dequant kernels
Group-wise INT4 dequant-GEMV kernel for decode (fused dequant + dot-product, both AWQ
and GPTQ layouts) and dequant-to-bf16 kernel feeding cuBLAS for prefill.
**Depends on:** M12-T04.
**Acceptance criteria:**
- Kernel tests: bit-exact dequant vs Python reference across group sizes {32, 64, 128}, then GEMV matches fp32 reference within tolerance.
- Microbench: INT4 GEMV ≥ 1.5× faster than bf16 GEMV at 4k×4k (memory-bound win, recorded).

### M12-T06 · End-to-end quantized serving
Run a real INT4 model (e.g. a 7B-class AWQ checkpoint) end-to-end through the server;
accuracy validation per M12-T01 methodology; benchmarks.
**Depends on:** M12-T05.
**Acceptance criteria:**
- Serving integration test passes with the quantized model; accuracy report committed (top-1 agreement, logit MSE); memory + throughput vs fp16 recorded in BASELINES.md.

### M12-T07 · FP8 KV cache
Optional `--kv-cache-dtype fp8`: scaled FP8-E4M3 KV storage (per-head or per-tensor
scales — documented choice), dequant inside attention kernels, scale calibration
strategy (dynamic per-block max).
**Depends on:** M12-T06.
**Acceptance criteria:**
- Attention kernels with FP8 cache match fp16-cache results within documented tolerance; long-generation quality spot-check; KV memory halved (measured); throughput delta recorded.

---

## Milestone 13 — Tensor Parallelism (Multi-GPU)

**Overview.** Scale to multiple GPUs with Megatron-style tensor parallelism:
column/row-sharded projections, head-sharded attention with all-reduce on output
projection, sharded MLP, vocab-sharded embedding/lm_head, and a per-rank KV cache.
Architecture: one process, one std::thread + CUDA device per rank, NCCL for
collectives — chosen for simplicity of a single scheduler (documented in the ADR with
the multi-process alternative). This is where the engine becomes genuinely multi-GPU.

**Architecture documents:** `docs/design/tensor-parallelism.md` + ADR-006 process
model (M13-T01).

### M13-T01 · Design doc & ADR: distributed architecture
Write `docs/design/tensor-parallelism.md` + ADR-006: process/thread model decision
(single-process multi-thread vs multi-process w/ shared-memory coordination — decide,
justify, note migration path for multi-node later), sharding map for every weight in
the Llama family (column: qkv/gate/up + vocab embedding; row: o_proj/down_proj +
lm_head input), collective points per layer (2 all-reduces), KV-cache sharding
(kv-heads split across ranks; replication fallback when kv_heads < tp_size), driver ↔
worker step protocol.
**Depends on:** M12-T07.
**Acceptance criteria:**
- Complete sharding table for all weights incl. edge cases (GQA kv-head divisibility rules → validation errors); step-protocol sequence diagram.

### M13-T02 · Communicator abstraction & NCCL wrapper
`src/distributed/`: `Communicator` interface (all_reduce, all_gather, broadcast,
barrier — stream-ordered) with a NCCL implementation (unique-id init across ranks) and
a single-rank no-op implementation (so TP=1 uses the same code path).
**Depends on:** M13-T01.
**Acceptance criteria:**
- Multi-GPU tests (auto-skip if < 2 GPUs): all_reduce/all_gather/broadcast correctness for bf16/fp32 across 2 ranks on real device buffers; no-op impl passes the same suite at world_size=1.

### M13-T03 · Worker runtime
Per-rank worker threads each owning a CUDA device, stream, allocator, and model
shard; driver broadcasts `SchedulerOutput`/batch metadata to workers; workers execute
the identical step in lockstep; rank-0 owns sampling results.
**Depends on:** M13-T02.
**Acceptance criteria:**
- Tests with a dummy model: N workers execute scripted steps in lockstep, metadata identical per rank (asserted), clean startup/shutdown with no thread leaks (run under TSAN in CI-CPU mode with a fake communicator).

### M13-T04 · Sharded weight loading
Rank-aware loader: each rank materializes only its shard (slicing safetensors views
per the M13-T01 sharding map — no full-weight materialization), including fused-qkv
splitting and quantized-weight sharding constraints (group-size divisibility).
**Depends on:** M13-T03.
**Acceptance criteria:**
- Unit tests: for the tiny fixture at TP=2, concatenating rank shards reconstructs the original weight exactly, for every layer type; per-rank memory ≈ 1/TP of weights (asserted within slack).

### M13-T05 · TP attention & MLP
Parallel decoder layer: column-parallel QKV (local heads only), local attention over
sharded KV cache, row-parallel o_proj + all-reduce; column-parallel gate/up,
row-parallel down + all-reduce; norms replicated.
**Depends on:** M13-T04.
**Acceptance criteria:**
- 2-GPU test: layer output on each rank (post all-reduce) matches the single-GPU layer bit-tolerance-wise on the fixture; GQA head-split correctness covered (kv_heads==tp and kv_heads>tp cases).

### M13-T06 · TP embedding, logits & sampling
Vocab-parallel embedding (partial lookup + all-reduce) and lm_head (local logits
shard + all-gather to rank 0); sampling runs on rank 0 over full logits; sampled
tokens broadcast to all ranks for the next step.
**Depends on:** M13-T05.
**Acceptance criteria:**
- 2-GPU test: full-model logits match single-GPU within tolerance; greedy generation token-identical to single-GPU on fixture.

### M13-T07 · TP end-to-end integration
Wire TP through engine/runtime/server: `--tensor-parallel-size` flag, per-rank paged
KV caches (capacity computed per rank), scheduler unchanged (device-count-aware stats
only), server unchanged.
**Depends on:** M13-T06.
**Acceptance criteria:**
- Integration: 2-GPU serving of a 7B-class model passes the API test suite; greedy outputs identical to TP=1 for the fixture model; continuous-batching invariant test (M8-T08) passes at TP=2.
- Bench: 7B model TP=2 vs TP=1 throughput and latency recorded in BASELINES.md.

### M13-T08 · Distributed failure handling
A CUDA/NCCL error on any rank propagates: all ranks abort the step, engine surfaces a
fatal error, process exits cleanly (no hang — NCCL abort + watchdog timeout);
documented operational behavior.
**Depends on:** M13-T07.
**Acceptance criteria:**
- Fault-injection test: injected worker failure terminates the engine within the watchdog window with a clear error, no deadlock (test has a timeout), resources released.

---

## Milestone 14 — Speculative Decoding

**Overview.** Accelerate decode with speculative execution: a cheap proposer drafts k
tokens, the target model verifies them in one batched forward pass, and a
rejection-sampling acceptance rule preserves the target distribution exactly. Two
proposers: n-gram/prompt-lookup (free, surprisingly effective on repetitive text) and
a small draft model. The correctness bar is unusual — greedy output must be
*identical* with speculation on or off, and sampled output must be *distributionally*
identical.

**Architecture documents:** `docs/design/speculative-decoding.md` (M14-T01).

### M14-T01 · Design doc: speculative decoding
Write `docs/design/speculative-decoding.md`: `Proposer` interface, verification
forward (target scores k draft tokens + 1 in a single step — reuses varlen prefill
machinery), acceptance rule (greedy exact-match; stochastic rejection sampling with
the residual-distribution resample), KV-cache handling for speculated-then-rejected
tokens (cache rollback via block-table truncation), scheduler integration, and when
speculation is expected to help vs hurt.
**Depends on:** M13-T08.
**Acceptance criteria:**
- Acceptance-rule math written out (with the residual distribution formula); KV rollback design covers block-boundary cases.

### M14-T02 · Proposer interface & n-gram proposer
`src/spec/proposer.h` + prompt-lookup proposer: propose k tokens by matching the
longest recent n-gram of the sequence against its own history (prompt + generated);
configurable n range and k.
**Depends on:** M14-T01.
**Acceptance criteria:**
- Unit tests: known-repetition sequences produce expected proposals; no-match produces empty proposal (engine falls back to normal decode); proposal never exceeds max_len budget.

### M14-T03 · Draft-model proposer
Load a second, smaller model (same tokenizer — validated) with its own KV cache
(paged, small pool); autoregressively draft k tokens per sequence per step; draft
cache maintained in sync with accepted tokens (including rollback).
**Depends on:** M14-T02.
**Acceptance criteria:**
- Unit tests with fixture models: drafts match running the small model standalone; draft KV state after mixed accept/reject steps equals recomputed-from-scratch state.

### M14-T04 · Verification forward & KV rollback
Target-model verification: score the k proposed tokens + 1 bonus position in one
forward (per sequence, batched across sequences); on rejection at position j, truncate
sequence and roll back paged KV (block-table truncation + slot invalidation) for both
target and draft caches.
**Depends on:** M14-T03.
**Acceptance criteria:**
- Tests: verification logits at each position match sequential decode logits exactly (same cache state); rollback then re-decode produces identical results to never having speculated (the cache-integrity invariant), including rollbacks across block boundaries.

### M14-T05 · Acceptance rule
Greedy mode: accept the longest prefix of drafts matching target argmax, then take
target's token at first mismatch. Sampling mode: per-position rejection sampling
(accept with prob min(1, p_target/p_draft), resample from normalized residual on
reject).
**Depends on:** M14-T04.
**Acceptance criteria:**
- Greedy: speculative output token-identical to non-speculative for fixture + real model over long generations (the golden invariant).
- Sampling: statistical test — empirical next-token distribution under speculation matches target-only distribution (chi-square over a small-vocab fixture, 100k trials).

### M14-T06 · Engine integration & metrics
Scheduler/engine support for speculative steps (variable tokens-per-step per
sequence), `--speculative-model` / `--num-speculative-tokens` / `--spec-method=ngram|draft`
config, metrics (proposals, acceptance rate, effective tokens/step), interaction with
stop conditions (stop mid-accepted-window handled).
**Depends on:** M14-T05.
**Acceptance criteria:**
- Integration: API test suite passes with speculation enabled; greedy A/B identical outputs; acceptance-rate metric exposed.
- Bench: ≥ 1.5× decode tokens/sec on a favorable pair (e.g. 7B target + 1B draft, or n-gram on repetitive workload) recorded in BASELINES.md; unfavorable-case regression documented.

---

## Milestone 15 — Observability, Benchmarking & Profiling

**Overview.** Make the engine measurable in production and in development: a metrics
subsystem with Prometheus exposition, request-level latency instrumentation (TTFT,
inter-token latency), NVTX profiling integration, an offline throughput benchmark, a
realistic serving load generator with workload replay, and documented performance
methodology. Several earlier milestones added ad-hoc counters; this milestone unifies
them.

**Architecture documents:** `docs/design/observability.md` (part of M15-T01).

### M15-T01 · Metrics registry & Prometheus endpoint
`src/metrics/`: lightweight registry (counters, gauges, histograms with fixed
buckets; lock-cheap hot path), Prometheus text-format exposition at `/metrics`,
design notes in `docs/design/observability.md` (naming conventions, label discipline,
cardinality rules).
**Depends on:** M14-T06.
**Acceptance criteria:**
- Unit tests: metric registration/update/render (format validated against Prometheus exposition spec); concurrent-update stress test; hot-path counter increment benchmarked (< 50 ns).

### M15-T02 · Engine & request instrumentation
Instrument with the registry: TTFT, inter-token latency, end-to-end latency, queue
wait, tokens in/out, running/waiting sequence gauges, KV utilization, prefix-cache hit
rate, preemptions, batch-size histogram, spec-decode acceptance rate — replacing all
ad-hoc counters from earlier milestones.
**Depends on:** M15-T01.
**Acceptance criteria:**
- Integration test: a scripted workload produces expected metric values (e.g. request count, token counts exact; latency histograms populated); `/metrics` output reviewed against the naming doc.

### M15-T03 · NVTX ranges & profiling documentation
NVTX ranges around step phases (schedule, assemble, prefill, decode, sample, output)
and per-layer (compile-time-gated to zero cost when off); `docs/profiling.md`: how to
capture and read nsys/ncu traces of the engine, with annotated example screenshots.
**Depends on:** M15-T02.
**Acceptance criteria:**
- nsys capture shows named ranges (manual verification, doc includes the capture command); disabled build has zero overhead (bench diff within noise).

### M15-T04 · Offline throughput benchmark
`benchmarks/bench_throughput`: run N prompts (synthetic or dataset file) at max
batching, report prompt & generation tokens/sec, per-phase breakdown, JSON + human
output; replaces/absorbs `bench_generate`.
**Depends on:** M15-T03.
**Acceptance criteria:**
- Deterministic workload mode for regression comparisons; results include config fingerprint (model, dtype, flags, commit); BASELINES.md updated via a documented process.

### M15-T05 · Serving load generator
`tools/loadtest/` (Python acceptable): open-loop Poisson arrivals at target RPS,
ShareGPT-format conversation replay, measures TTFT/ITL/E2E percentiles (P50/P90/P99),
goodput at SLO, outputs a report; drives the real HTTP API.
**Depends on:** M15-T04.
**Acceptance criteria:**
- Against a live server: report generated with sane percentiles; saturation behavior visible (latency knee as RPS increases across a sweep); README documents usage.

### M15-T06 · Structured request logging & slow-request tracing
Per-request structured log line (JSON option) with ids, token counts, timing
breakdown, finish reason; debug mode capturing per-step timelines for sampled slow
requests.
**Depends on:** M15-T05.
**Acceptance criteria:**
- Log schema documented and tested (parse the JSON in the test); slow-request trace triggers at a configurable threshold and contains per-phase timings.

### M15-T07 · Performance regression guardrail
A scripted perf-regression check (manual/self-hosted trigger): runs the deterministic
throughput benchmark and key kernel microbenchmarks, compares against BASELINES.md
with tolerance bands, fails on regression; documented release-gate process.
**Depends on:** M15-T06.
**Acceptance criteria:**
- Script exits non-zero on a synthetic injected regression (test by perturbing a baseline); process documented in `docs/perf-process.md`.

---

## Milestone 16 — Hardening, Documentation & v0.1 Release

**Overview.** Production-grade polish: leak/stress/fuzz testing, determinism
guarantees, packaging (Docker), a real documentation site, and a tagged v0.1.0
release. This milestone turns a working engine into a credible open-source project —
the difference between "code that runs" and "software people can adopt."

**Architecture documents:** none new — this milestone finishes and publishes the
existing ones.

### M16-T01 · Sanitizer & leak hygiene
CI jobs: ASAN+UBSAN build running the CPU test suite; TSAN on runtime/scheduler tests;
`compute-sanitizer` (memcheck) run documented and applied to the GPU kernel test suite
(manual/self-hosted); fix everything found.
**Depends on:** M15-T07.
**Acceptance criteria:**
- ASAN/UBSAN/TSAN CI jobs green; compute-sanitizer run on kernel tests reports zero errors (log committed to PR).

### M16-T02 · Stress & soak testing
`tools/stress/`: hours-long soak driver (random request sizes, params, cancels,
client disconnects, occasional pathological inputs) against the server; memory-growth
and fd-leak assertions; documented soak procedure.
**Depends on:** M16-T01.
**Acceptance criteria:**
- 2-hour soak on a real model: zero crashes, RSS and GPU memory plateau (< 1%/hour growth), block-pool stats return to zero at drain.

### M16-T03 · Fuzzing
libFuzzer targets: tokenizer encode/decode, safetensors header parser, API JSON
request parser; seed corpora committed; short fuzz runs wired into CI, long runs
documented.
**Depends on:** M16-T02.
**Acceptance criteria:**
- 10-minute CI fuzz of each target finds no crashes; any pre-existing findings fixed with regression tests added.

### M16-T04 · Determinism & compatibility guarantees
Document and test the determinism contract (same model+seed+params+engine version ⇒
identical output; caveats: batching-dependent numerics — decide and document whether
batch-invariance is guaranteed or not); model-support matrix validated at startup
(clear error for unsupported arch/dtype/head-dim combos).
**Depends on:** M16-T03.
**Acceptance criteria:**
- Determinism test suite (restart-to-restart, and documented batching caveats verified as described); startup validation errors tested for the top unsupported cases.

### M16-T05 · Docker packaging & release builds
Multi-stage `Dockerfile` (CUDA runtime base, pinned toolchain, slim final image),
`docker run … engine serve --model …` works out of the box; release CMake preset
(LTO, arch list); CI job building the image on tags.
**Depends on:** M16-T04.
**Acceptance criteria:**
- Image builds in CI, serves a model on a GPU host (documented smoke procedure), size and layer hygiene reviewed (no build tools in final stage).

### M16-T06 · Documentation site
`docs/` publishable (mkdocs-material or similar): getting started (install → serve →
query in 5 minutes), architecture overview with diagrams (from the design docs),
configuration reference (generated from the M9-T08 definition source), API reference,
model support matrix, performance guide, contribution guide.
**Depends on:** M16-T05.
**Acceptance criteria:**
- Site builds in CI; a newcomer path (getting-started) tested end-to-end verbatim; every design doc linked and current (stale sections updated).

### M16-T07 · Examples & client compatibility
`examples/`: curl scripts, Python with the official `openai` SDK, streaming client,
chat client; compatibility notes for common tools (documented tested versions).
**Depends on:** M16-T06.
**Acceptance criteria:**
- Every example runs against a live server in an integration test (or a documented manual matrix); `openai` SDK streaming + non-streaming verified.

### M16-T08 · v0.1.0 release
Semantic versioning policy, `CHANGELOG.md` (Keep-a-Changelog format, back-filled by
milestone), README polish (badges, feature list, benchmark table, architecture
diagram), license + NOTICE review for vendored code, tag + GitHub release with notes
and Docker image reference.
**Depends on:** M16-T07.
**Acceptance criteria:**
- `v0.1.0` tag published with release notes; README quick-start verified from a clean machine; version surfaced in `engine --version` and `/v1/models`.

---

## Future directions (post-v0.1, unordered)

Deliberately out of scope for the linear roadmap; each would begin with its own design
doc and milestone breakdown when scheduled:

- **Mixture-of-Experts (MoE)** — expert-parallel routing, fused MoE kernels (Mixtral/Qwen-MoE families).
- **Encoder-only & encoder-decoder models** — embedding serving (BERT-family), Whisper/T5-style cross-attention.
- **State-space models & hybrids** — Mamba-style layers, hybrid attention/SSM (Jamba-class).
- **Multi-node inference** — pipeline parallelism, multi-node NCCL bootstrap (builds on ADR-006's migration path).
- **Disaggregated prefill/decode** — separate prefill and decode fleets with KV transfer.
- **LoRA serving** — multi-adapter batching (S-LoRA-style), adapter hot-swap.
- **Structured/guided decoding** — JSON-schema/grammar-constrained generation (logit masking via FSM).
- **FP8 weights & compute** — FP8 GEMM on Hopper+ (cuBLASLt/CUTLASS), calibration.
- **Marlin/Machete-class quantized GEMM kernels** — faster INT4 at batch > 1.
- **Vision-language models** — image encoder + projector, multimodal prompt plumbing.
- **KV-cache offload** — CPU/NVMe tiering for long-context and many-session workloads.
- **Sliding-window & hybrid attention** — Mistral-style windowed attention, attention sinks.
- **Prefill/decode-aware autoscaling metrics** — SLO-driven admission policies.
- **Windows/other-accelerator portability** — only if/when a real need appears; the backend abstraction (ADR-002) keeps the door open.
