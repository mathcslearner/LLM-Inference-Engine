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
- **M5 (CPU reference engine) — complete** (2026-08-18): T01
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
  (Amendment 4 already allowed it). Docs-only. T02 CPU GEMM & `Linear`:
  `src/cpu/gemm.cpp` (`cpu::gemm`) — reference fp32 GEMM in NT form
  `C=A·Bᵀ(+bias)`, weight/bias f32/f16/bf16 widened per element via
  `half.h` (never links `kernels`), cache-blocked + `parallel_for` threaded
  but single-accumulator ascending-k so bit-identical across threads/tiles;
  `src/model/modules.{h,cpp}` — abstract `Linear` interface (§4.1) +
  `ReferenceLinear` (zero-copy checkpoint weight, `y` caller-allocated).
  New `tiny-llama/expected/ops.safetensors` (`tiny-llama-ops` subcommand,
  10 GEMM cases incl. real bf16 weights, k==1, skinny/wide, widening);
  +12 tests (609 green). T03 CPU norm/activation ops: `src/cpu/`
  `rmsnorm.cpp`/`activation.cpp` (silu_mul+add)/`softmax.cpp` + shared
  `detail.h` — `cpu::rmsnorm` (per-row `x*rsqrt(mean(x²)+eps)*weight`, HF
  order, f32/f16/bf16 `x`+weight widened, the pure fp32 forward with no HF
  `.to(input_dtype)` round-trip), `cpu::silu_mul` (SwiGLU, saturating
  silu), `cpu::add` (residual), `cpu::softmax` (max-subtracted, `-inf`→0
  for M5-T05's mask); all `parallel_for`-threaded with single-accumulator
  per-row reductions → bit-identical across threads, `y` caller-allocated,
  aliasing allowed. `src/model/modules.{h,cpp}` — `RmsNorm` module
  (zero-copy `[E]` weight + eps, f32-activation `forward`). 10 new
  `ops.safetensors` cases appended after the GEMM draws (GEMM bytes
  unchanged, `--verify` byte-clean); +23 tests (632 green). T04 Embedding &
  RoPE: `src/cpu/embedding.cpp` (`cpu::embedding_lookup` — gather +
  f32/f16/bf16→fp32 widen, `ids` as the `ForwardRequest.token_ids` span,
  out-of-range id pre-scanned → InvalidArgument naming index+value) and
  `src/cpu/rope.cpp` (`cpu::rope_apply` — in-place HF half-rotation on
  `[T,Hx,d]`, pairs `(j,j+d/2)` rotated by precomputed `cos`/`sin`, position
  bounds pre-scanned; both ops token-parallel and bit-identical across
  threads); `src/model/modules.{h,cpp}` — `Embedding` module (zero-copy
  `[V,E]` table; added as the M6-T06 tied-weight seam, design §4.2 updated)
  and `Rope` module (`Create` precomputes `cos`/`sin` `[num_pos, d/2]` +
  `inv_freq` from theta and `rope_scaling`: default/`linear`
  (`inv_freq/=factor`)/`llama3` (exact HF `_compute_llama3_parameters`),
  other → Unimplemented; `apply` rotates Q and K). inv_freq/angle formed in
  fp64, stored fp32 → Class T vs HF (range reduction diverges with position:
  negligible ≤127, ~5e-3 at 131071; inv_freq ~1 ulp). New
  `tools/gen_fixtures/tiny_llama_rope.py` (`tiny-llama-rope` →
  `rope.safetensors`: tiny table/apply, llama3 & linear scaled inv_freq +
  cos/sin) + one `embedding_edge` case appended to `ops.safetensors` (existing
  bytes byte-identical, `--verify` clean); +26 tests (658 green). T05 Causal
  attention: `src/cpu/attention.cpp` (`cpu::attention` — naive causal GQA:
  scores `[H·T,L]` materialized, `-inf` causal mask with new query `t` at cache
  position `P+t` attending keys `[0,P+t]`, softmax via `cpu::softmax` reusing
  its `-inf→0` contract, context contraction; GQA by KV-head indexing `h/g`, no
  materialized repeat; `scale` multiplies the completed dot per HF order; two
  `parallel_for` passes, single-accumulator → bit-identical across threads);
  `src/model/modules.{h,cpp}` — `Attention` module (owns q/k/v/o `Linear`s as
  `unique_ptr<Linear>` so M6/M13 slot in, move-only; `Create` validates shapes
  vs (H,Hkv,d) without assuming `E==H·d`; `forward` projects QKV → RoPE Q/K →
  `cache.append` → `cache.view` → `cpu::attention` → `o_proj`, validation
  front-loaded so no half-append); `src/kvcache/kv_cache.h` — abstract `KvCache`
  interface + `CacheGeometry`/`KvView` (§6.1), landed here (not T06) because
  `Attention` consumes it — `model → kvcache` edge (ADR-002 Amdt 5) wired
  PRIVATE, forward-declared in modules.h. `SimpleKvCache` + KV invariant stay
  T06; T05 used a test-local `FakeKvCache`. New
  `tools/gen_fixtures/tiny_llama_attention.py` (`tiny-llama-attention` →
  `attention.safetensors`: 5 cases (layer,P,T) — prefill-empty, prefill-continue
  P>0, decode T=1 — module I/O + op intermediates captured from HF eager
  attention; existing bytes untouched, `--verify` clean). +9 tests (667 green).
  T06 KV cache v0: `src/kvcache/simple_cache.{h,cpp}` (`SimpleKvCache` — the v0
  contiguous impl of the T05 `KvCache` interface; anchor `kvcache.cpp` removed).
  Two `[num_layers,Hkv,capacity,d]` fp32 stores allocated once at
  `Create(geom,capacity)`; `append` transposes token-major `[T,Hkv,d]` →
  head-major at the layer's fill (over-capacity→ResourceExhausted,
  rank/shape/dtype/contiguity/k-vs-v-T→InvalidArgument, all front-loaded so a
  rejected append leaves the layer untouched); `view` **gathers** a contiguous
  `[Hkv,fill,d]` snapshot (not a zero-copy slice — `cpu::attention` needs
  contiguous K/V, and this is the M8 gather seam; design §6.2 updated);
  `length()` = min per-layer fill; `truncate(n)` drops+resyncs all layers,
  `reset()`=`truncate(0)`. The KV invariant lands at attention-chain level
  (`Model::forward` is T07, which elevates it to logits): full-prefill vs
  token-by-token, chunked prefill, truncate-then-redecode — all **bit-exact**
  (max_abs_diff=0, since masked softmax-0 keys add exactly 0.0). `attention_test`
  swapped from the `FakeKvCache` double to the real `SimpleKvCache`
  (`PreloadHeadMajor` seeds P>0 through `append`), so the prefill-continuation
  goldens now cover the cache end-to-end. +13 tests, −1 Fake self-check
  (679 green). T07 Transformer block & full model forward: `src/model/`
  `modules.{h,cpp}` gain `Mlp` (SwiGLU `down(silu(gate)⊙up)`, move-only, owns
  gate/up/down `Linear`s) and `DecoderLayer` (pre-norm `h=attn_norm(x);
  r=x+attn(h); y=r+mlp(mlp_norm(r))`, move-only, owns 2×`RmsNorm`+`Attention`+
  `Mlp`); new `model/model.h` — the `Model::forward` contract
  (`LogitsMode{kLast,kAll}`, `ForwardRequest`, abstract `Model` with
  `forward`/`config`/`cache_geometry`) + the `ActivationHook`/`ActivationEvent`
  debug seam; new `model/reference_model.{h,cpp}` — `ReferenceModel::Create`
  binds modules from `LoadedModel.weights` by canonical name (honoring tied
  embeddings as a binding, not a pointer-equality assumption) and `forward` runs
  embedding → N layers → final norm → lm_head with front-loaded validation,
  `kLast` (project only the last row) vs `kAll`, and per-stage hook emission.
  End-to-end tiny-llama logits vs `activations.safetensors` max_abs_diff=3.7e-6
  (tol 2e-4); the per-layer hook emits `embeddings`/`layers.{i}`/`final_norm`/
  `logits`; the KV invariant now holds at full-model logits (bit-exact).
  **`model → kvcache` promoted PRIVATE→PUBLIC** in CMake (`model.h` returns
  `kvcache::CacheGeometry` by value — ADR-002 Amendment 5 clarified);
  `linear_input:` hook events deferred to M14-T02 (design §11 updated). +15
  tests (694 green). T08 Architecture registry: `src/model/registry.{h,cpp}` —
  `BuildModel(LoadedModel, BuildOptions)` dispatching on
  `config.architecture_name`; `RegisterArchitecture(string_view, ModelBuilder)
  → Status` (`AlreadyExists`/`InvalidArgument`); `SupportedArchitectures()`
  (sorted); unknown arch → `Unimplemented` listing supported. One
  `BuildReferenceFamily` builder serves both `LlamaForCausalLM` and
  `Qwen2ForCausalLM` (family diff = `attention_bias`+config values, already
  handled by `ReferenceModel::Create`) — **both registered in T08**, M5-T10
  adds only the Qwen fixture. **`enum class Backend` relocated from the sketched
  `engine/backend.h` into `registry.h`** (`BuildOptions` names it; `model` can't
  depend on `engine`, ADR-002); `kOptimized` → `Unimplemented` until M6.
  Built-ins register lazily inside a mutex-guarded `GetRegistry` function-local
  static (a static-lib TU of self-registering globals would be linker-dropped).
  `BuildModel` bit-identical to a direct `ReferenceModel::Create` (pass-through).
  design §2/§8/§9/§13 updated. +7 tests (701 green). T09 Greedy generation loop:
  `src/engine/generator.{h,cpp}` — `Generate(Model&, KvCache&, prompt_ids,
  GenerateOptions, TokenCallback)`: prefill in one `kLast` forward at positions
  `P0..P0+T-1` (`P0=cache.length()`, continues from a non-empty cache), strict-`>`
  argmax (lowest-index tie-break; NaN→`Internal`), decode one token/forward at the
  running length feeding the prior token back; EOS ids **included** in the output;
  `on_token` fires once per id, in order, after append/before next forward;
  front-loaded validation (empty prompt / `max_new_tokens≤0` → InvalidArgument;
  **up-front worst-case capacity check** → ResourceExhausted since `StatusOr<vec>`
  can't carry a partial result); `forward` errors propagate. `src/engine/backend.h`
  — `Backend`/`BuildOptions` re-export + header-only `BackendName`/`ParseBackend`;
  `engine.cpp` anchor removed. New `ModelConfig::eos_token_ids` (HF int-or-list
  `eos_token_id`, validated `[0,vocab)`; model-loading.md §3.2). New
  `tools/gen_fixtures/tiny_llama_generate.py` → `generate.json`: three prompts
  **selected for min top-2 logit gap > 1e-2** (the random model's sub-1e-3-gap
  regions make greedy trajectories ill-conditioned — HF generate vs a manual KV
  loop diverge there; well-separated prompts make token-for-token robust), EOS
  suppressed, HF `generate` cross-checked vs manual loop. design §10/§12 updated.
  +20 tests (721 green). T10 Qwen-family support: **zero `src/` change** — the
  config parser, weight map, `ReferenceModel::Create`, `Attention::Create`, and
  registry already carried every Qwen difference (per-arch `attention_bias`
  default, q/k/v biases with bias-free o_proj, `head_dim` decoupled from
  `hidden_size/heads`, tied embeddings), so T10 is a model fixture + goldens +
  tests. New `tools/gen_fixtures/tiny_qwen2.py` (`tiny-qwen2`): a 2-layer
  `Qwen2ForCausalLM` distinct from tiny-llama on every Qwen axis — **`head_dim=24`
  ≠ 64/4=16** (decoupled path §3.1), **`attention_bias` omitted from config.json**
  (parser default wires the biases), **tied embeddings** (`lm_head.weight` dropped
  — safetensors refuses shared storage — restored via loader alias), θ=1e6/eps=1e-6/
  no BOS; q/k/v biases filled with fixed non-zero noise (HF zero-inits them, which
  would make the bias path invisible in the golden). New `tiny_qwen2_generate.py`
  (`tiny-qwen2-generate`) → `generate.json`, three no-BOS prompts, min top-2 gap
  now **asserted** > 1e-2 (observed ≥0.10), HF `generate` vs manual KV loop
  cross-checked. New `tests/unit/qwen2_family_test.cpp`: config/loader wiring,
  registry routing + bit-identical build, end-to-end logits vs golden
  (max_abs_diff 3.9e-6, tol 2e-4), **biases load-bearing** (dropping them moves
  logits ~0.98), greedy match/determinism/KV-invariant. `--verify` byte-clean;
  format + scoped tidy clean. +8 tests (729 green).

- **M6 (optimized CPU execution engine) — complete** (2026-08-18): T01
  `docs/design/optimized-cpu-execution.md` — the contract M6-T02…T08 build on.
  **Docs-only.** Fixes: the **packed weight tile** (checkpoint `[N=out,K=in]`
  stays source of truth, derived at load into K-major panels of `NR=16` output
  rows `[ceil(N/NR),K,NR]`, checkpoint dtype widened in-register; `NR` fixed
  across ISAs so the forced-scalar pass validates the shipped bytes; worked
  example + `MR×NR` micro-kernel serving GEMM/skinny/GEMV from one layout);
  **dtype policy** (fp32 accumulation/activations everywhere, weights kept in
  checkpoint dtype, fp32 checkpoints not narrowed, norm/bias converted once at
  load); **workspace** (`OptimizedModel` owns one reused-across-layers
  `Workspace`, stated byte sizing formula, **grow-on-demand** → allocation-free
  decode; logits stay caller-owned for both backends); **parallel
  implementations** (not reuse of the M5 layer classes — reuse only the `Linear`
  interface via `PackedLinear` + `Rope` tables, so the oracle stays untouched and
  the graph is fusion-ready); **tied embeddings → one physical copy** (packed
  lm_head authoritative, lookup gathers row `v` from panel `v/NR` — no ~272 MB
  `[V,E]` duplicate on tied Qwen2.5-0.5B); **kernel-validation tolerance table**
  (bit-identical across thread counts always; bit-identical across ISAs only for
  the two pure-map ops; Class T across ISAs + vs oracle for everything with a
  multiply-accumulate — with stated tolerance recipes and a ≤2-ulp vector-`expf`
  sweep). `model → kernels` is a downward edge (no ADR amendment, PRIVATE link);
  a determinism-safe worker-index `parallel_for` overload lands with M6-T04;
  optimized suites register `SCALAR_PASS`. Cross-doc notes added to
  model-execution.md (§4.3/§5.2/§8/§14) and cpu-backend.md (§3.2/§6.3/§7). No
  `src/` change (729 green unchanged). T02 packed-weight GEMM & GEMV +
  `PackedLinear`: `src/kernels/gemm.{h,cpp}` — `kNr=16` (fixed across ISAs),
  `PackWeightPanels` (pure gather+zero-pad of `W[N,K]` → K-major panels
  `[ceil(N/16),K,16]`, dtype-preserving), and `PackedGemm`/`PackedGemv` behind a
  tile-shaped ISA seam (`{scalar,neon,avx2}/gemm.cpp`, `internal/gemm_impl.h` +
  `gemm_common.h`): register-tiled `MR×NR` (`MR=4` NEON, `6` AVX2, full-K in
  registers — design §3.4 amended), scalar bit-identical to a naive loop,
  NEON/AVX2 Class T; `PackedGemm` threads the `(kMc=64, kNc=8)` tile grid and
  routes `M==1` to panel-threaded `PackedGemv`. `src/model/packed_linear.{h,cpp}`
  — `PackedLinear` (third `Linear`): repacks at `Create`, bias→fp32 once, drops
  the checkpoint handle, `forward`=`PackedGemm`; header kernels-free so
  **`model → kernels` links PRIVATE** (downward edge, no ADR amendment). Tests
  (+30, → 759 green): `packed_gemm_test` (kernels, SCALAR_PASS) vs the
  `cpu::gemm` oracle across shapes/dtypes + layout/threading/GEMV-vs-GEMM
  invariants + `ops.safetensors` goldens; `packed_linear_test` (**model,
  SCALAR_PASS — first model suite in the forced-scalar pass**) vs
  `ReferenceLinear`. Bench: **8.76× bf16 / 8.48× f32** the naive `cpu::gemm` at
  4096³ (~112 GFLOP/s on the M2), ≥5× target met (BASELINES.md); the optional
  Accelerate context number couldn't build on the dev machine (Homebrew-clang vs
  CLT SDK, `-Welaborated-enum-base`) so it's deferred, wiring in place behind
  `-DENGINE_BENCH_BLAS`. Format + scoped tidy clean. T03 vectorized
  norm/activation/softmax/RoPE kernels: NEON/AVX2 (behind M3-T05 dispatch) +
  scalar for `RmsNormF32`/`SiluMulF32`/`SoftmaxF32`/`RopeApplyF32`
  (`src/kernels/{norm,activation,softmax,rope}.{h,cpp}` + per-ISA TUs), each
  validated against its `cpu::` oracle within a stated tolerance on the host ISA
  **and** the forced-scalar pass. The **vector `expf` polynomial** is the one new
  numerical algorithm (`internal/exp_common.h` scalar spec + `neon_exp.h`/
  `avx2_exp.h` lane helpers, Cephes/`avx_mathfun` lineage): shared by all three
  ISAs (scalar embeds it too, so forced-scalar exercises the shipped numeric
  code), with its **own** ≤2-ulp sweep (`vector_exp_test`, observed max **1
  ulp**) independent of the softmax/SiLU kernels that use it; `x < kExpLo` →
  exactly `+0.0` makes softmax's `-inf → 0` mask contract exact. RMSNorm uses an
  **exact** `1/sqrtf` (no raw `rsqrte`, §10); SiLU uses **exact** division; RoPE
  takes arbitrary per-token positions (batched/paged-ready). **Residual add reuses
  the existing Class-E `kernels::AddF32`** (no new kernel; parity-tested vs
  `cpu::add`). All kernels **bit-identical across thread counts** (no split
  reduction; tested threaded-vs-serial-variant). `exp` ships no public kernel —
  softmax/SiLU embed the lane helpers; `exp.cpp`+`internal/exp_impl.h` expose
  array-form variants only for the sweep. Observed vs-oracle (NEON): softmax
  6.0e-7, RMSNorm 2.4e-6, SiLU 9.5e-7, RoPE 2.4e-7 across sizes {odd,1024,4096},
  large-magnitude/causal softmax, positions {0,1,large}/unsorted and head-dims
  {24,64,128}; HF goldens replayed through each kernel. Bench (BASELINES.md
  M6-T03): NEON vs scalar — RMSNorm 2.10×, Softmax 3.08×; ExpF32/SiLU ~1.05×
  (streaming-bound / scalar auto-vectorizes); RoPE 0.70× recorded honestly (a
  M12 tuning item, correctness unaffected). AVX2 TUs written blind, hand-reviewed,
  proven by CI's x86-64 build. +64 test registrations → **823 green**. T04
  optimized prefill attention: `src/kernels/attention.{h,cpp}` +
  `internal/attention_common.h`/`attention_impl.h` + `{scalar,neon,avx2}/
  attention.cpp` — `PrefillAttentionF32`, blocked flash-style causal GQA
  attention with an online (running max/sum) softmax honoring the exact
  `cpu::attention` op contract; the recurrence written once as
  `PrefillUnitsImpl<Ops>` (GEMM-idiom split: shared control flow, ISA-only
  arithmetic — dot+score/exp+sum/scale/axpy), block constants `kAttnQb=32`/
  `kAttnKb=64`, threaded over `(head, query-block)` units. **No per-worker
  scratch and §6.4's worker-index `parallel_for` overload NOT added** — the
  online accumulator is `out` itself, so `src/parallel/` is untouched and the
  design (`optimized-cpu-execution.md` §6.1/§6.2/§6.4, `cpu-backend.md` §3.2)
  updated to record the deferral. Causal masking is a per-row `n_valid` (no
  written `−inf`); row-at-a-time score buffer. Class T vs the oracle, observed
  1.3e-6 across the acceptance sweep (T∈{1,17,512,2048}, P∈{0,5}, GQA
  {(4,4),(4,2),(8,1)}×d∈{18,24,64,128}, block-boundary straddles;
  `rtol 1e-4, atol 1e-5`); bit-identical across thread counts. AVX2 TU written
  blind, hand-reviewed, proven by CI. Bench (BASELINES.md M6-T04): **9.72×** the
  reference at 8-thread NEON on a 2k-context prefill (2.121 → 0.218 s), 12.89×
  single-thread. +14 test registrations → **837 green**. T05 optimized decode
  attention: the single-token specialization — one query per head attends the
  whole cache, threaded over `Hkv` kv heads (`kAttnHeadGrain=1`) with the
  `g=H/Hkv` query heads of a group processed **key-block-outer/query-inner** so
  each kv head's K/V streams from DRAM once. `src/kernels/attention.{h,cpp}` —
  `DecodeAttentionF32(q,k,v,out, H,Hkv,d,L, scale)`, `KernelTable` dispatch,
  `detail::DecodeAttentionVariant` seam; `internal/attention_common.h` —
  `DecodeUnitsImpl<Ops>` + `DecodeGroupSlice<Ops>` (per-slice body split out
  under the tidy cognitive-complexity threshold, gemm.cpp `ComputePanel` idiom),
  `DecodeArgs`, `kAttnDecodeGroupChunk=8`; scalar/neon/avx2 TUs each a one-line
  `DecodeUnitsImpl<…Ops>` instantiation reusing the **same four Ops primitives**
  as prefill (zero new intrinsics; AVX2 blind). **Bit-identical to
  `PrefillAttentionF32(T=1)` by construction, asserted bitwise** (the M8-T05
  paged-decode oracle) — per fixed query head the arithmetic order is identical;
  bit-identical across thread counts; Class T vs the oracle (observed 8.6e-6
  across cache lengths {1,63,64,65,127,128,129,2048}, GQA + a `g=12>chunk` slice
  case, d∈{18,24,64,128}, large-logit stress; `rtol 1e-4, atol 1e-5`). No
  per-worker scratch, allocation-free, `src/parallel/` untouched. Parallel width
  = `Hkv` only (decode leaves cores idle when `Hkv<cores` — the M12-T03
  flash-decoding motivator, recorded). Bench (BASELINES.md M6-T05): **5.25×** the
  reference at 8-thread NEON on a 2k-context decode step (1407 → 268 µs/call),
  7.15× single-thread NEON; decode ≈ prefill(T=1) (both hold one score row at
  T=1). +12 test registrations → **849 green**. T06 embedding & logits path:
  `src/kernels/embedding.{h,cpp}` — `EmbeddingLookupF32` (row-major `[V,E]`,
  untied) and `EmbeddingLookupPackedF32` (packed lm_head `[ceil(V/kNr),E,kNr]`,
  tied, gathering logical row `v` from panel `v/kNr` lane `v%kNr`). A pure
  gather + exact fp16/bf16→fp32 widen → **bit-identical** across thread counts,
  ISAs, and vs the `cpu::embedding_lookup` oracle. **No per-ISA embedding TU**
  (amends design §2): the only per-element work is the widen, dispatched through
  the M3-T06 `convert` variants via a `KernelTable<WidenFn>`, so `embedding.cpp`
  needs no NEON/AVX2 TU and the forced-scalar pass still exercises the scalar
  widen; the gather is inherently scalar (strided packed lanes chunked into a
  512-wide stack buffer, widened a chunk at a time — Class E, chunk-invariant).
  `src/model/optimized_embedding.{h,cpp}` — `OptimizedEmbedding` mirroring the M5
  `Embedding` surface: `FromTable` (untied zero-copy `[V,E]`) and
  `FromPackedLinear` (tied — holds the lm_head's `packed_weight()` by value; the
  refcounted `shared_ptr<Buffer>` keeps the **one physical copy** alive so the
  linear may then be moved behind a `unique_ptr<Linear>` without dangling —
  tested); `forward` front-loads the `y` checks + `[0,V)` id pre-scan (naming
  index+value like the oracle) so the raw gather never touches a bad row. The
  lm_head logits path is unchanged `PackedLinear::forward` (kAll GEMM / kLast
  GEMV) — no new logits kernel. `model → kernels` stays PRIVATE. +13 gtest cases
  (embedding_kernel_test/kernels + embedding_logits_test/model, both SCALAR_PASS)
  → **862 green** (875 ctest entries). No fixture change; `src/parallel/`
  untouched; format + scoped tidy clean. T07 optimized model forward &
  generation: `src/model/workspace.{h,cpp}` (`Workspace` — the
  reused-across-layers scratch, ten model-level fp32 slots `x/h/tmp/r` +
  `q/k/v/ctx` + `gate/up` exposed as `[T,width]` prefix views; monotone
  grow-on-demand with a high-water mark, all-or-nothing `EnsureCapacity` so a
  mid-way OOM leaves prior buffers intact and surfaces before any kernel/cache
  touch → allocation-free steady-state decode; `W_worker=0` as designed; §6.2
  `bytes()` asserted at tiny-llama's 25.6 KB) and
  `src/model/optimized_model.{h,cpp}` (`OptimizedModel` — the `kOptimized`
  backend, a **parallel-implementation graph** not the reference `DecoderLayer`
  classes so the oracle stays untouched: `Create` builds a `PackedLinear` per
  projection incl. lm_head, norm scales→fp32 once, **one shared `Rope`** for the
  whole model — cos/sin are position-only, avoiding ~200 MB of per-layer table
  duplication on a real model — and honors tied embeddings by sharing the packed
  lm_head storage before it's moved behind `unique_ptr<Linear>`; `forward` copies
  the reference's validation verbatim then runs embedding→N layers→final
  norm→lm_head out of the workspace via the dispatched
  RmsNorm/PackedGemm/RopeApply/Prefill|DecodeAttention/SiluMul/Add kernels,
  reading K/V through `cache.view`; logits caller-owned kLast-GEMV/kAll-GEMM; same
  hook events as the reference). Registry: `BuildReferenceFamily`→`BuildFamily`
  dispatches on `options.backend` (M5 `Unimplemented` guard gone). Driver:
  `src/main.cpp` grew an `engine generate` subcommand (load→tokenize→`Generate`→
  stream-detokenize, the §10 real-model harness until M9's server binary). +15
  gtest cases (`optimized_model_test`, model label, **SCALAR_PASS** — the first
  full-model forward under forced-scalar) → **877 green** (905 ctest entries):
  both backends built on tiny-llama (untied) + tiny-qwen2 (tied+biases,
  decoupled head_dim), logits vs reference (**2.4e-7** kAll / **1.8e-7** kLast)
  and vs the HF golden (**3.7e-6**/**3.9e-6**, matching the reference's own HF
  agreement), greedy token-for-token vs reference *and* `generate.json`, the KV
  invariant **bit-exact** (0), workspace sizing, and every front-loaded error
  path. Real ~1B model (Qwen2-0.5B-Instruct bf16) loads and generates coherent
  text on the dev machine (sample outputs in the PR / PROGRESS.md). No fixture
  change; `src/parallel/` untouched; format + scoped tidy clean. T08 generation
  benchmark & first baseline: new `benchmarks/bench_generate.cpp` (+ target and
  two smoke `add_test`s) drives the real greedy `Generate` loop and splits it by
  per-token callback timestamps — prefill tok/s = `prompt_len/`(start→first
  token), decode tok/s = `1000/`median-per-step (median = steady-state, absorbs
  isolated background spikes), synthetic random prompt ids (length-only,
  tokenizer-free, like `llama-bench -p N`), one warmup run, `--threads N` sets
  `ENGINE_NUM_THREADS` before `load_model` since `DefaultPool` sizes once at
  weight-packing time; report carries a host/thread/ISA/dtype fingerprint and a
  ±5% PASS/OVER verdict; backend-agnostic. Baseline (BASELINES.md M6-T08):
  Qwen2-0.5B bf16, 8-thread NEON optimized — **prefill ~133 / decode ~31 tok/s,
  decode run-to-run ±3.9% (PASS)**; 4t/2t advisory; full quiescing + sweep
  discipline deferred to M12-T01. Same-machine **llama.cpp** context number @
  `6d05498` (CPU-only NEON build, `-Wno-elaborated-enum-base` past the broken
  CLT SDK; bf16+f16 GGUF via `convert_hf_to_gguf.py`): f16 192/34 tok/s (prefill
  8t / decode 4t) → ballpark parity; llama's ARM CPU **bf16** GEMM is ~20×
  slower than its f16 (so our bf16-first design compares vs llama f16); the M2
  **efficiency cores throttle memory-bound decode in both engines** (llama
  decode faster at 4t than 8t) — an E-core-aware decode pool is an M12 lever,
  compounding the `Hkv`-only decode parallel width (§8, M12-T03 motivator). No
  `src/` change; `tools/.venv` restored to its pins after conversion; 907 ctest
  green; format + scoped tidy clean.

- **M7 (sampling & generation controls) — in progress**: T01 SamplingParams &
  pipeline skeleton (2026-08-18): new `sampling` module content (was
  anchor-only) — `src/sampling/params.{h,cpp}` (`SamplingParams`:
  temperature/top_k/top_p/repetition·presence·frequency penalties/seed/
  max_tokens/stop_token_ids/stop_strings/logprobs with OpenAI·vLLM defaults, a
  default value = identity "raw softmax" request; `ValidateSamplingParams` the
  single field-naming `InvalidArgument` validator shared with the M10 mapper;
  `SamplingParams::Greedy(n)` = temperature-0; `kMaxLogprobs=20`) and
  `src/sampling/sampler.{h,cpp}` (single-sequence reference `Sampler`:
  `Create` validates then **rejects any non-greedy-subset knob with
  `Unimplemented` naming the field** — no knob silently ignored, each later
  ticket drops its guard; `Sample(logits, SampleContext{prompt_ids,
  generated_ids})` dispatches on temperature, `==0` = strict-`>` lowest-index
  argmax ported verbatim from M5-T09's `ArgmaxLastRow` (NaN max→`Internal`,
  empty→`InvalidArgument`), `>0` = the T02 seam; history-stateless so
  `generated_ids.size()` is the step index the RNG/penalties will key on; `const`
  in T01 with a note that T02's RNG counter goes `mutable`). Engine routing:
  `GenerateOptions` swaps `max_new_tokens` for a `sampling::SamplingParams`
  field (`eos_ids` stays separate, model-derived), `Generate` builds one
  `Sampler` up front (front-loaded validation, cache untouched on error) and
  samples the last logits row — `engine → sampling` PUBLIC, the old tensor
  argmax dropped. **Breaking rename `max_new_tokens → sampling.max_tokens`**
  across 6 call sites (main.cpp, bench_generate, 3 test suites). design §15
  written (SamplingParams table, fixed stage order penalties→temp→top-k→top-p→
  select→logprobs, `Sampler` contract, routing, per-ticket forward map); §10
  re-annotated, §14 sampling bullet retired. +35 gtest cases (sampling_params
  18, sampler 16 — both `sampling` label, no SCALAR_PASS — + 1 generator
  rejection case) → 912 green (942 ctest); the greedy `generate.json` goldens on
  both backends pass unchanged (the regression guarantee). format + scoped tidy
  clean. T02 temperature/top-k/top-p sampling (2026-08-18): the
  `Sampler::Sample` `temperature > 0` branch — new `src/sampling/philox.h`
  (header-only **Philox4x32-10**, a `constexpr` pure block fn +
  `PhiloxUniformDouble(seed, step, draw)` → 53-bit `[0,1)`; counter derived from
  the step so there is **no mutable RNG state** and `Sample` stays `const`) and
  `src/sampling/stages.{h,cpp}` (`engine::sampling::detail`: `CheckFinite` /
  `ApplyTemperature` / `ApplyTopK` (torch.topk threshold semantics — boundary
  ties kept) / `Softmax` (double-accumulated, −inf→0 exactly) / `ApplyTopP`
  (ascending-index tie-break, crossing token inclusive, no-op at 1.0) /
  `SelectByCdf` (inverse-CDF ascending walk)). `Sampler::Create` drops the
  temperature/top-k/top-p guards (penalties/stop/logprobs still `Unimplemented`),
  resolves the seed (request seed else `random_device`+clock) and exposes
  `seed()`. `engine generate` CLI gains `--temperature/--top-k/--top-p/--seed`.
  Sampled sequences are same-machine reproducible only (`std::exp` ulp variance;
  `sampling` can't link `kernels`' exp — cross-platform contract is M17-T04's;
  recorded in stages.h/§15.2). design §15.2/§15.3/§15.5 updated. +41 gtest cases
  (philox 7 incl. Random123 KAT vectors `static_assert`-checked, stages 18 exact-
  mask, sampler +13 incl. chi-square over 20k draws, generator +3 stochastic) →
  953 green (983 ctest), all deterministic (fixed seeds); greedy goldens
  unchanged. format + scoped tidy clean.

Next up: **M7-T03** (repetition, presence & frequency penalties: apply penalties
over the request's token history (prompt + generated, OpenAI/vLLM semantics)
*before* temperature — the stage-1 seam in the pipeline; exact hand-computed
logit-adjustment tests per penalty type and combinations, no-op-at-default exact
equality; drops the three penalty `Unimplemented` guards in
`Sampler::Create`/`CheckImplementedSubset`).
