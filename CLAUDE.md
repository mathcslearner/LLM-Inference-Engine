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
   heading. Append the detailed entry (what landed, non-obvious decisions,
   test count) to `docs/PROGRESS.md`, and refresh the compact status at the
   bottom of this file (milestone bullet + "Next up").
5. If implementation reveals a design flaw, update the design doc in the same
   change and note what changed. Never silently diverge from a design doc.

## Architecture

Strict module boundaries (dependency rules recorded in ADR-002, current as of
Amendment 5 — `model → kvcache`, added in M5-T01; no cycles):

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

**clang-tidy is local-only** (2026-08-08): CI's full-sweep tidy job outgrew
its 15-minute timeout and was removed — the scoped rules above are the
whole tidy gate now, so apply them diligently before handoff. Known
accepted gap: TUs absent from the arm64 dev machine's compile database
(the x86-64 cpuid path in `src/kernels/dispatch.cpp`, the `avx2/` per-ISA
TUs) are tidy-checked nowhere; CI's x86-64 build jobs still compile them
warnings-as-errors. When editing that code, review against `.clang-tidy`
by hand (e.g. the asm-output false-positive NOLINT in dispatch.cpp).

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

**History (pre-pivot):** M0 (foundation), M1 (CPU tensor library), and M2
(CUDA backend) were built as a CUDA engine (2026-08-03/04); on 2026-08-07
the project pivoted to CPU-first (ADR-004) — no CUDA hardware ever compiled
the device code. Details: `docs/archive/ROADMAP-v1.md`,
`docs/design/retired/cuda-backend.md`, git history.

What survives from M0–M2 and remains load-bearing: the build/test/CI/tooling
discipline (M0), the complete tensor library — dtypes incl. reserved
kInt4/kFP8E4M3, Shape/Strides, `Device` (kCUDA reserved, never allocatable),
Buffer/Allocator, Tensor with views, fp16/bf16 host types with bit-exact
conversions, seeded fills/allclose/copy/cast (M1) — and the device-agnostic
`CachingAllocator` (M2-T06), which becomes the KV block pool's backing store.

**Post-pivot progress** (compact; the detailed per-ticket log — what
landed, the non-obvious decisions, test counts — is
**[docs/PROGRESS.md](docs/PROGRESS.md)**; read it when you need the "why"
behind an existing subsystem):

- **M3 (CPU backend foundation) — complete** (2026-08-07/08): pivot
  transition (ADR-004, roadmap swap, CUDA excision);
  `docs/design/cpu-backend.md`; `src/parallel/` (deterministic ThreadPool,
  `parallel_for`/`parallel_reduce`, TSAN CI job); `src/kernels/` runtime
  ISA dispatch (scalar/NEON/AVX2 per-ISA TUs, `ENGINE_FORCE_ISA`,
  `SCALAR_PASS` test registration); first vectorized kernels
  (elementwise/reductions/fp16-bf16 conversions) + `benchmarks/` scaffold
  and first BASELINES.md entries; post-milestone audit fixes. CI's
  full-sweep tidy job was later retired (2026-08-08; see Build & test).
- **M4 (model loading & tokenization) — complete** (2026-08-08): T01
  `docs/design/model-loading.md`; T02 `tools/gen_fixtures/` + ~16 MB
  committed fixtures; T03 `src/model/config.h` (ModelConfig parser); T04
  safetensors parser (`Tensor::from_buffer`, `src/model/mapped_file`,
  `src/model/safetensors`); T05 sharded checkpoints
  (`src/model/checkpoint` — unified single-file/sharded interface, lazy
  shard mapping, index↔shard consistency checks); T06 weight-name mapping
  (`src/model/weight_map` — canonical namespace, per-arch HF tables,
  tied-embedding aliasing, shape validation, missing/unexpected/ignored
  report); T07 model loader (`src/model/loader` — `load_model(dir)`
  composing config → checkpoint → mapping → zero-copy weight registry;
  dtype policy F32/F16/BF16-only, pickle-checkpoint convert hint,
  per-stage/per-shard progress logging); T08 tokenizer model parsing
  (`src/tokenizer/tokenizer` + `bpe` — tokenizer.json → byte-level BPE
  structures: token↔id maps, merge ranks, id→raw-bytes, added-token
  registry, template BOS/EOS; §6.2 component whitelist enforced by name,
  amended from fixture reality: post_processor ByteLevel/Sequence
  accepted, `pair` template ignored); T09 BPE encoding (`encode()`
  byte-identical to HF for all committed vectors, both families:
  added-token longest-match split with lstrip/rstrip/normalized
  semantics → NFC (hand-written, tables generated from pinned Unicode
  16.0.0 by new `tools/gen_unicode/` → committed
  `src/tokenizer/unicode_data.inc`) → hand-written GPT-2/cl100k split
  matcher (`src/tokenizer/pretokenize` — the two fixture patterns only,
  selected at parse, rejected by name otherwise) → byte-level map →
  merge loop (`bpe_split`, leftmost-lowest-rank) → template inserts;
  invalid UTF-8 encodes as own pre-tokens per the encode_synthetic
  vectors); T10 decoding & incremental detokenization (`decode` — token
  bytes concatenated, specials dropped when skipping, then U+FFFD
  maximal-subpart lossy conversion matching HF's Rust
  `from_utf8_lossy`, pinned by the malformed-tail goldens; strict
  `classify_utf8_prefix`/`append_utf8_lossy` in unicode.h alongside the
  lenient encode-side codec; `src/tokenizer/detokenize` —
  `DetokenizerStream` push/finish over a ≤3-byte carry, streaming
  output bit-identical to batch decode, every emission valid UTF-8);
  post-milestone audit fixes (2026-08-08: two vacuous
  tokenizer_decode_test golden loops fixed — dangling-temporary
  range-for, C++20 — restoring the T10 round-trip/streaming acceptance
  coverage incl. qwen2 streaming; safetensors alignment check made
  absolute `(8 + header_len + begin) % itemsize`; loader missing-weight,
  weight-map LOG_WARN-capture, and 256-byte alphabet-bijection tests
  added; model-loading.md staleness synced). 597 tests green. M4 is
  pushed and CI-green (the first x86-64 CI build surfaced a GCC-only
  `-Wshadow` break in three tokenizer tests, fixed in `fix gcc errors`).
- **M5 (CPU reference engine) — in progress**: T01
  `docs/design/model-execution.md` — the model-execution contract
  M6/M8/M9/M11–M15 build on: module decomposition (`cpu` reference ops =
  the oracle, links `tensor`+`parallel`, never `kernels`; `model` graph
  with `Linear` as an *interface* so M6 repacking / M13 quant slot in;
  `kvcache` v0; `engine` greedy loop), the exact
  `Model::forward(ForwardRequest&)` contract (token-major `[T,E]`,
  per-token positions, `LogitsMode{kLast,kAll}`, recoverable-Status
  inputs), GQA layout (`h → h/(H/Hkv)`, no materialized repeat), KV cache
  v0 (`append`/`view`/`length`/`truncate`, head-major `[Hkv,len,d]`, the
  token-by-token==full-recompute invariant) with an explicit "what M8
  paging changes" account, HF half-rotation RoPE spec, the
  reference-vs-optimized backend seam, registry/builder, and activation
  hooks; **ADR-002 Amendment 5 (`model → kvcache`)** added,
  cpu-backend.md §2's provisional `cpu → parallel` flag retired
  (Amendment 4 already allowed it). Docs-only.

Next up: **M5-T02** (CPU GEMM & Linear layer).
