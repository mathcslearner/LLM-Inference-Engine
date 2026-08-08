# LLM Inference Engine

A production-grade, **CPU-first** LLM inference engine for decoder-only
transformer models (Llama, Qwen, and similar families). C++20 with
hand-vectorized SIMD kernels — NEON on Apple Silicon arm64 (the primary dev
platform), AVX2 on Linux x86-64 (CI and deployment) — behind runtime dispatch
over an always-present scalar reference. Performance-competitive with existing
CPU engines (llama.cpp-class) while prioritizing clean architecture,
maintainability, and thorough testing. The pivot from the original CUDA plan is
recorded in ADR-004.

**Feature scope (built incrementally):** paged KV cache, continuous batching,
prefix caching, custom SIMD kernels (GEMM, attention, norms, fusions), chunked
prefill, weight-only quantization (INT8/INT4, AWQ/GPTQ formats, INT8 KV cache)
*including our own quantizer implementations and evaluation harness*,
speculative decoding with a measurement study, OpenAI-compatible HTTP API with
SSE streaming, Prometheus metrics. **Inference only** — no training or
fine-tuning.

## The roadmap is the source of truth

All work follows **[ROADMAP.md](ROADMAP.md)** (the v2, CPU-first plan): linear
milestones (M0–M17) broken into tickets (`M<n>-T<m>`) sized for one focused
session (1–4 hours, a few hundred lines including tests). Work tickets
**strictly in order**. For each ticket:

1. Read the ticket, its dependencies, and the governing design doc in
   `docs/design/` (each milestone's first ticket usually writes one —
   implementation must conform).
2. Implement the feature **and its tests together** — a ticket without tests is
   not done.
3. Meet every acceptance criterion; build clean (warnings-as-errors), format
   clean.
4. Mark the ticket done in ROADMAP.md: append ` — ✅ DONE (YYYY-MM-DD)` to its
   heading.
5. If implementation reveals a design flaw, update the design doc in the same
   change and note what changed. Never silently diverge from a design doc.

## Architecture

Strict module boundaries (dependency rules recorded in ADR-002, current as of
Amendment 4; no cycles):

```
server → runtime → scheduler ─┐
                   engine ────┼→ model / tokenizer / kvcache / sampling / quant / spec
                              └→ tensor / memory / parallel / kernels / cpu → core
```

| Module | Path | Responsibility |
|---|---|---|
| core | `src/core/` | Status/StatusOr, logging, base utilities |
| tensor | `src/tensor/` | Dtypes, shapes, Tensor (dense, explicit memory, no autograd) |
| memory | `src/memory/` | Allocator interface, CPU allocator, caching pool |
| parallel | `src/parallel/` | Thread pool, deterministic parallel_for/reduce (M3-T04) |
| kernels | `src/kernels/` | SIMD kernels (scalar/NEON/AVX2 behind runtime dispatch) |
| cpu | `src/cpu/` | Unoptimized scalar reference implementations — the correctness oracle |
| model | `src/model/` | config.json, safetensors loading, weight mapping, architecture registry |
| tokenizer | `src/tokenizer/` | Byte-level BPE (tokenizer.json), incremental detokenization |
| kvcache | `src/kvcache/` | Paged block pool, block tables, prefix-cache index |
| sampling | `src/sampling/` | SamplingParams, penalties, top-k/p, stop conditions, logprobs |
| engine | `src/engine/` | Model runner: batch assembly, forward execution, backend selection |
| scheduler | `src/scheduler/` | Pure decision logic: admission, batching budget, preemption |
| runtime | `src/runtime/` | Engine loop thread, request lifecycle, output channels |
| server | `src/server/` | HTTP, OpenAI-compatible API, SSE streaming, chat templates |
| quant | `src/quant/` | Quantized weight containers, format loaders, quantizer algorithms |
| spec | `src/spec/` | Speculative decoding proposers and verification |
| metrics | `src/metrics/` | Metrics registry, Prometheus exposition, trace spans |
| tests | `tests/` | `unit/`, `integration/`, `fixtures/` (committed golden data) |
| tools | `tools/` | Python dev tooling: golden-fixture generation, load testing |
| benchmarks | `benchmarks/` | Throughput benchmarks, kernel microbenchmarks, BASELINES.md |
| docs | `docs/` | `adr/` (decisions), `design/` (subsystem designs), `archive/` + `design/retired/` (pre-pivot history) |

## Engineering conventions

- **Language:** C++20. Python only in `tools/` (fixture generation via
  HuggingFace, load testing, reference cross-checks) — never in the engine or
  its build.
- **Errors:** recoverable → `Status`/`StatusOr<T>` (`src/core/status.h`);
  programmer errors → `CHECK`; no exceptions across module boundaries
  (ADR-003).
- **Correctness methodology (load-bearing):** HuggingFace fixtures (generated
  by `tools/`, committed under `tests/fixtures/`) validate the scalar
  reference; the scalar reference validates every vectorized (NEON/AVX2) and
  optimized kernel. Never optimize without a reference path to test against.
- **Testing:** tests land in the same change as the feature. The suite runs
  the host's best ISA plus a forced-scalar pass (`ENGINE_FORCE_ISA=scalar`
  once M3-T05 lands), so CI (x86-64) and the dev machine (arm64) jointly
  cover every backend — no test is skipped for lack of hardware. Numerical
  tests state their tolerance explicitly.
- **Determinism:** `parallel_reduce` uses a fixed tree order — results are
  bit-identical across thread counts. Seeded fills are bit-exact
  cross-platform (see `src/tensor/ops.h`).
- **Performance claims are measured:** optimization tickets record
  before/after numbers in `benchmarks/BASELINES.md`. No perf change without a
  benchmark delta.
- **Style:** `.clang-format` enforced; warnings-as-errors; public APIs get doc
  comments. Match surrounding code's idiom.
- **Decisions:** anything architectural gets an ADR (`docs/adr/`); each
  subsystem has a design doc (`docs/design/`) written before its
  implementation milestone.

## Build & test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
scripts/check-format.sh
scripts/check-tidy.sh
```

**Per-ticket validation workflow:** on every iteration, run the incremental
build, the full ctest suite, and `scripts/check-format.sh` — all seconds.
The full-tree `scripts/check-tidy.sh` sweep takes minutes; while iterating,
scope it to the ticket's TUs instead (`scripts/check-tidy.sh
tests/unit/foo_test.cpp`). Headers have no compile-database entry, so pass a
TU that includes them — usually the ticket's test file — and
`HeaderFilterRegex` analyzes them through it. The sweep analyzes only TUs
present in the compile database (and says what it skipped); explicit
arguments without an entry are rejected.

Before handoff, tidy coverage follows from what changed (clang-tidy has no
cross-TU analysis, so untouched TUs with untouched headers cannot change
results — no blanket full sweep needed):
- Only new files or edited `.cpp` files → the scoped run on the ticket's
  TUs is provably sufficient.
- Edited an existing header → scope to every TU that includes it
  (`grep -rl 'tensor/tensor.h' src tests`), since a header change can
  perturb diagnostics at other call sites.
- Changed `.clang-tidy`, the toolchain pin, or a core header included
  everywhere (`core/status.h`, `core/check.h`) → run the full no-arg sweep;
  the includer set is the whole tree anyway.

CI runs the full sweep regardless (same pinned LLVM 20), so a missed edge
case surfaces as a red Actions run rather than slipping through.

**On the macOS dev machine, always build with Homebrew LLVM 20, not Apple
clang**:

```bash
CC="$(brew --prefix llvm@20)/bin/clang" CXX="$(brew --prefix llvm@20)/bin/clang++" \
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

Two reasons: it matches the project's pinned clang-format/clang-tidy toolchain
(LLVM 20, resolved by `scripts/clang-tools.sh`; Apple's clang uses different
version numbering), and Apple's Command Line Tools install on this machine is
broken — its libc++ headers are missing, so Apple clang cannot compile any
C++ until the CLT is reinstalled.

## Current status

**History (pre-pivot):** M0 (foundation & tooling), M1 (CPU tensor library),
and M2 (CUDA backend) were built and audited as a CUDA engine, 2026-08-03/04;
on 2026-08-07 the project pivoted to CPU-first (ADR-004) because no CUDA
hardware ever compiled or ran the device code. Details:
`docs/archive/ROADMAP-v1.md`, `docs/design/retired/cuda-backend.md`, git
history.

What survives from M0–M2 and remains load-bearing: the build/test/CI/tooling
discipline (M0), the complete tensor library — dtypes incl. reserved
kInt4/kFP8E4M3, Shape/Strides, `Device` (kCUDA reserved, never allocatable),
Buffer/Allocator, Tensor with views, fp16/bf16 host types with bit-exact
conversions, seeded fills/allclose/copy/cast (M1) — and the device-agnostic
`CachingAllocator` (M2-T06), which becomes the KV block pool's backing store.

M3-T01 done (2026-08-07: ADR-004; ADR-002 Amendment 4; v1 roadmap archived to
`docs/archive/ROADMAP-v1.md`; v2 roadmap promoted to `ROADMAP.md`;
cuda-backend design doc retired to `docs/design/retired/`; CLAUDE.md and
README rewritten CPU-first).
M3-T02 done (2026-08-07: CUDA excision — deleted `src/cuda/`,
`src/distributed/`, the CUDA/pinned allocators, the transfer path, the CUDA
kernel infrastructure, `cmake/cuda.cmake` + the `ENGINE_ENABLE_CUDA` option,
`tests/common/cuda.*`, 12 CUDA test TUs, and `gpu-ci.yml`; `Tensor::empty`/
`Tensor::to` return Unimplemented for the reserved kCUDA device;
`ops::copy` dropped its stream overload; kernels module reset to a
placeholder anchor; test harness gpu-label machinery removed —
285 tests, all green; `engine_project_files` now skips deleted-but-unstaged
ghosts).
M3-T03 done (2026-08-07: `docs/design/cpu-backend.md` — threading model
(persistent pool at physical cores, deterministic static partitioning,
fixed-tree `parallel_reduce`, no nested parallelism, OpenMP rejected with
rationale); SIMD dispatch (runtime `SelectedIsa()` over scalar/NEON/AVX2,
per-ISA TUs with per-TU flags, `ENGINE_FORCE_ISA` fatal on bad values,
kernel-table registry + add-a-kernel recipe); dtype policy (fp32
accumulation, half.h-exact conversions, F16C folded into the AVX2 gate);
validation methodology (oracle chain re-rooted on `src/cpu/`, platform
matrix answering how ISAs absent from CI are proven, numerics classes E/R/T
incl. the fixed 16-lane Class-R convention); alignment/weight-layout
conventions; CI additions (forced-scalar ctest pass, TSAN job for
`parallel`, macOS arm64 CI job decided *against* — private-repo 10× minute
multiplier — with revisit triggers). Flagged: `cpu → parallel` will need an
ADR-002 amendment at M5-T02).
M3-T04 done (2026-08-07: `src/parallel/` — `ThreadPool` (persistent
fixed-size pool, condvar-parked workers, static round-robin chunk
assignment, one region in flight, caller executes no chunks),
`DefaultPool()` sized by `ENGINE_NUM_THREADS` (non-numeric fatal, <1 clamps;
design doc §3.1 note) else `physical_core_count()` (sysctl/sysfs, SMT
collapsed); `parallel_for` with (n, grain)-only static partitioning and
inline path when num_chunks <= 1; `parallel_reduce` with chunk-id-indexed
cache-line-padded partials + fixed pairwise halving fold — bit-identical at
thread counts {1,2,8}, tested against an independent serial tree; nesting
CHECK via thread_local flag; `FunctionRef` spelled as const std::function&
for v1. Build/CI: `ENGINE_SANITIZE=thread` CMake option (global flags so
gtest is instrumented too), new `tsan` CI job running `ctest -L parallel`
(clang Debug), `engine_add_tests` gained LABELS. 309 tests green incl. 24
new `parallel`-labeled; TSAN-clean locally).
M3-T05 done (2026-08-07: `src/kernels/` SIMD dispatch — `Isa` enum +
`SelectedIsa()` (memoized once per process; arm64 → NEON unconditionally,
x86-64 → AVX2 iff cpuid AVX2+FMA+F16C+AVX+OSXSAVE and xgetbv confirms YMM
state, else scalar), `KernelTable<Fn>`/`Select` registry with silent
per-kernel scalar fallback for null slots (+ explicit-ISA overload for
tests), `ENGINE_FORCE_ISA={scalar,neon,avx2}` fatal on unknown/unavailable
values via testable `detail::ResolveIsa` seam; `DispatchProbe()` probe
kernel instantiating the per-ISA TU layout (`scalar|neon|avx2/probe.cpp`,
arch-guarded `internal/probe_impl.h`); CMake: PUBLIC `ENGINE_ARCH_ARM64`/
`ENGINE_ARCH_X86_64` from `CMAKE_SYSTEM_PROCESSOR`, per-ISA sources with
per-source `-mavx2;-mfma;-mf16c` (never target-wide), placeholder anchor
removed; `engine_add_tests` gained `SCALAR_PASS` (same binary re-registered
as `<tree>-scalar/` with `ENGINE_FORCE_ISA=scalar`, label `scalar`) —
design doc §4.2/§8.1 implementation notes added. 337 tests green
(dispatch suite runs twice: NEON + forced-scalar)).
M3-T06 done (2026-08-08: first vectorized kernels — `src/kernels/`
`elementwise.h` (AddF32/MulF32/ScaleF32), `reduce.h` (SumF32/MaxF32,
`kReduceGrain` exposed as part of the bit-exact spec), `convert.h`
(fp16/bf16 ↔ fp32 on `tensor::float16`/`bfloat16` pointers), each with
scalar+NEON+AVX2 variants in per-ISA TUs behind KernelTables, threaded via
`DefaultPool()` `parallel_for`/`parallel_reduce` (variants are
single-threaded chunk bodies). Numerics decisions recorded as design-doc §5
/ §6.3 implementation notes: vector fp16 conversions blend NaN lanes rebuilt
with half.h's integer ops (hardware FCVT/F16C quiets every SNaN — verified
on M2 — where half.h preserves surviving payloads); Class-R sum lanes
initialize to +0.0f (spec'd; all-(-0) sums are +0) and the grain is part of
the spec; MaxF32 ±0 ties sharpened to total-order -0 < +0 (NEON FMAX
native, AVX2 cmp-eq/AND blend, scalar signbit tie-break; NaN-free
precondition documented for both reductions). Tests: 23 new tests × both
ISA passes — size sweep {0,1,15,16,17,4096,2^20+3} × offsets {0..3}
bit-compared against the scalar variants, independent serial
re-implementation of the full sum spec (33-chunk odd-carry case),
exhaustive 2^16-pattern widening and narrowing round-trips, half.h probe
goldens incl. SNaN payload classes, aliasing/no-op/death cases — 383 tests
green. `benchmarks/` scaffold: `kernels_bench` (hand-rolled best-of-N,
`ENGINE_BUILD_BENCHMARKS` option) + first `BASELINES.md` entry — M2 NEON
vs scalar: fp16 conversions 4.06×/2.86× (≥2× advisory target met),
sum/max 6.7×/8.0×, memory-bound ops ~1×).
Next up: **M4-T01** (design doc: model loading & tokenization).
