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

- **M7 (sampling & generation controls) — complete** (2026-08-18/19): T01 SamplingParams &
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
  unchanged. format + scoped tidy clean. T03 repetition/presence/frequency
  penalties (2026-08-19): new `detail::ApplyPenalties` (stages.{h,cpp}) — the
  pipeline's stage 1, applied to raw logits *before* temperature and ahead of the
  greedy argmax alike (a penalty can move the token in either mode). Documented
  HF/vLLM history split: `repetition_penalty` penalizes prompt ∪ generated once
  per distinct token (`x<0 ? x·r : x/r`, not compounded on recurrence);
  `frequency`/`presence` count generated tokens only (prompt-only token
  untouched), applied repetition→frequency→presence. History collected O(history)
  (unordered_map count + set seen), validated `[0,V)` before any mutation
  (`InvalidArgument`, logits untouched); exact no-op fast path at defaults
  (bitwise-identical to pre-T03), `-inf` stays `-inf`. `Sampler::Create` drops
  the three penalty `Unimplemented` guards; `Sample` penalizes in both branches
  (greedy keeps its allocation-free `ArgmaxRow` when no penalty active),
  `SampleStochastic` gained a `SampleContext` arg. CLI: `--repetition-penalty/
  --presence-penalty/--frequency-penalty`. design §15.2/§15.5 updated. +14 gtest
  (new `sampling_penalties_test` 12 exact dyadic-logit cases; sampler
  reject→accept/behavioural flips; generator +1 e2e penalty-trajectory, the
  Unimplemented-before-generating case moved onto `logprobs`) → 967 green (997
  ctest), deterministic; greedy `generate.json` goldens unchanged. format +
  scoped tidy clean. T04 stop conditions & finish reasons (2026-08-19): the
  generation loop's stopping generalized from EOS+max_tokens to the full set —
  new `src/engine/stop.{h,cpp}`: `StopStringMatcher` (pure byte-stream trim,
  holds back a pending stop-prefix so a stop string **split across token
  boundaries** is caught and trailing text trimmed; earliest match, list-order
  tie-break; no tokenizer/model dependency, reusable by M9) and `StopChecker`
  (per-request composite over a `DetokenizerStream`: `Observe` checks EOS set →
  request `stop_token_ids` → `stop_strings` → `max_tokens` **in that priority**,
  id-based stops kept un-trimmed and releasing the matcher's held tail, a
  stop-string match trimming its bytes + everything after; `stop_strings`
  without a tokenizer → front-loaded `InvalidArgument`; null tokenizer = the
  token-only path). `Generate` now returns **`GenerateResult`** (tokens +
  `finish_reason`/`stop_trigger`/`matched_stop`/detok `text`) and the callback
  takes a **`TokenEvent`** (id + safe-to-emit delta; concatenated deltas ==
  `text`). `GenerateOptions` gains an optional `tokenizer`; `engine_engine`
  links **`engine::tokenizer` PUBLIC** (ADR-002 diagram edge, first use, no
  amendment). Sampler `CheckImplementedSubset` drops the two stop guards (now
  accepts+ignores them; only logprobs guarded). Breaking API rippled through 6
  call sites (`.tokens`, `TokenEvent` callback); CLI gains `--stop`/
  `--stop-token-id` and prints `finish_reason`. design §10/§15.2/§15.3/§15.5
  updated. +25 gtest cases (new `stop_test` 21; `generator_test` +4 e2e
  finish-reason; `sampler_test` 2 reject→accept) → 992 green (1022 ctest),
  deterministic; greedy `generate.json` goldens unchanged. format + scoped tidy
  clean. T05 logprobs (2026-08-19): the final sampling stage — new
  `src/sampling/logprobs.h` (`TokenLogprob`/`StepLogprobs`), `detail::LogSoftmax`
  + `detail::ExtractLogprobs` (stage 6, stages.{h,cpp}), and a new
  `Sampler::SampleWithLogprobs` returning `SampleResult{token,
  optional<StepLogprobs>}` — the token-only `Sample` retained, both sharing one
  private impl. **Documented "which logits" choice** (§15.2 stage 6): natural-log
  softmax of the post-penalty/post-temperature logits *before* top-k/top-p
  masking, so logprobs describe the full vocabulary and sum to 1 (greedy uses the
  post-penalty row, no scaling) — OpenAI's convention, not the renormalized
  nucleus. The RNG draw is untouched, so `SampleWithLogprobs` picks the same
  token as `Sample`; greedy chosen logprob == max == top[0] (an acceptance
  criterion). **`Create` drops the last `Unimplemented` guard** (the whole
  `CheckImplementedSubset` gone) — every `SamplingParams` field is now
  implemented. `GenerateResult` gains a `logprobs` vector (index-aligned with
  `tokens`, empty unless requested); `TokenEvent` gains a `const StepLogprobs*`
  (the M10 SSE seam); the loop calls `SampleWithLogprobs`. CLI `engine generate
  --logprobs N`. design §10/§15.1/§15.2/§15.3/§15.5 updated. +~35 gtest (new
  `sampling_logprobs_test` 14; `sampler_test` +7 incl.
  `RejectsLogprobs`→`AcceptsLogprobs`; `generator_test` +4/−1 e2e) → 1042 ctest
  green; greedy `generate.json` goldens on both backends unchanged; format +
  scoped tidy clean. T06 batched sampling optimization (2026-08-19): new
  `src/sampling/batched_sampler.{h,cpp}` (`BatchedSampler` + `BatchRow`) samples
  one token per sequence from a `[batch, vocab]` block, `parallel_for` across
  sequences (grain 1 — row `b` only touches scratch slot `b`, so **`src/parallel/`
  untouched**, the §6.4 worker-index overload NOT needed; steady-state
  allocation-free via one owned `RowScratch` per slot grown on demand). **Picks
  bit-identical tokens+logprobs to the reference by construction**: the per-row
  pipeline is one shared `detail::SampleRow` (sampler.h) both call — not two
  paths that must match. Resolving the ticket's "vectorized softmax reusing M3/M6
  kernels" ∧ "bit-exact with reference" tension: promoted the M6-T03 polynomial
  to a **public unthreaded `kernels::ExpF32`** (`src/kernels/exp.{h,cpp}`; runs on
  the caller's thread so it's callable inside a `parallel_for` body) and switched
  the reference `detail::Softmax`/`LogSoftmax` to it → **`sampling → kernels`
  PRIVATE** (a layer-2→1 edge ADR-002 already permits, as `model → kernels`; the
  stale "sampling cannot link kernels" notes corrected). Sampled sequences now
  libm-independent per-ISA (toward M17-T04) but Class T across ISAs, so
  `batched_sampler_test` is **SCALAR_PASS**; 5 T05 logprob tolerances relaxed to
  fp32-exp class (~2.4e-7); greedy `generate.json` goldens argmax-only, unchanged.
  `ApplyTopP` now sorts **only positive-prob tokens** (post-top-k → sorts ~`k`
  survivors not the whole vocab; bit-identical to full-sort, never worse). Front-
  loaded shape validation; per-row errors captured + lowest-index returned after
  the region. Bench (BASELINES.md M7-T06; new `benchmarks/bench_sampling.cpp` + 2
  smoke tests): 64×128k 8t NEON — **greedy ~2.0 ms PASS** (advisory ≤5 ms), temp
  ~10 ms, top-k+top-p ~21 ms (was ~44 ms pre-`ApplyTopP` fix), logprobs ~22 ms;
  sampling configs DRAM-bandwidth-bound (fused-pass/E-core-aware pool = M12
  lever); threading ~3.5×@4t, ~4–5×@8t. +7 gtest (×2 SCALAR_PASS) +2 smoke →
  **1058 ctest green**; design §15.2/§15.5/§15.6 + optimized-cpu-execution.md §2
  updated; format + scoped tidy clean.

- **M8 (paged KV cache & block manager) — complete** (2026-08-19): T01
  `docs/design/paged-kv-cache.md` — the contract M8-T02…T08 (+ the
  M9/M11/M12/M13/M15 interactions) build on. **Docs-only.** Fixes: the
  **physical layout** (two per-layer K/V slabs `[num_blocks, Hkv, bs, d]` fp32,
  innermost `[bs,d]` tile contiguous = the head-major stream the M6 kernels read;
  resolves the roadmap's literal `[num_blocks,2,layer…]` — `2` = K/V axis as two
  slabs, per-layer for prefetch/TLB — with worked byte offsets for tiny-llama /
  Qwen2-0.5B); **block size 16** constrained to `bs ∈ {8,16,32,64}` by the
  **load-bearing `bs | kAttnKb=64` constraint** (a paged decode kernel bit-exact
  to M6-T05 needs a 64-key online-softmax unit to span whole blocks); the
  **capacity formula** (`num_blocks = ⌊kv_budget / (2·L·Hkv·bs·d·itemsize)⌋`,
  `--kv-cache-memory` absolute-or-fraction-of-host-RAM, `--kv-block-size`);
  **block-pool refcounting** + lifecycle diagram (FREE→OWNED→SHARED, double-free
  `CHECK`, dashed M11 `CACHED`) and the **immutability invariant** M11 rests on;
  **block table** slot mapping (`slot = blocks_[pos/bs]·bs + pos%bs`,
  all-or-nothing append, hand-worked mappings); **`PagedKvCache` keeps the M5
  `KvCache` interface** with one additive `paged_view(layer)` virtual
  (default-`Unimplemented`, the zero-copy decode fast path — a per-step full
  `view()` gather would blow M8-T07's ≤10% regression bound) and an **advisory
  `capacity()`**; **kernels** `KvScatterF32`/`PagedDecodeAttentionF32`/`paged_gather`
  as layout-agnostic raw-pointer entries over the new **`kvcache → kernels`**
  downward edge (ADR-002-permitted like `model → kernels`, no amendment).
  Cross-doc: model-execution.md §6.4 + optimized-cpu-execution.md §8 updated. No
  `src/` change (1058 tests unchanged). T02 block pool & allocator:
  `src/kvcache/block_pool.{h,cpp}` — `BlockPool`, pure bookkeeping over
  `2·num_layers` pre-allocated per-layer K/V slabs (`[num_blocks,Hkv,bs,d]` fp32,
  zero-filled once), fully unit-testable without a model or kernel. `Create` pulls
  the slabs straight from the passed `Allocator` (engine → M2 `CachingAllocator`;
  tests → default/fake) at **256-byte alignment** (`kSlabAlignment ==
  CachingAllocator::kMaxAlignment` — bypasses `ops::zeros`, which pins 64B) and
  the per-block free list is the KV block allocator itself (never upstream on the
  hot path); `Allocate` pops a LIFO stack + refcount 1, `Share`/`Release` are the
  M11-ready refcount verbs (FREE→OWNED→SHARED, §6.3), `refcount`/`stats`/
  `free_blocks`/`blocks_needed` round it out, one `std::mutex` off the token path.
  Static capacity helpers `BytesPerBlock`/`NumBlocksForBudget` (§5.2, overflow-
  guarded; host-RAM/fraction math deferred to T07). `block_size` validated to
  `{8,16,32,64}` (the §4 `bs | kAttnKb=64` constraint M8-T05 rests on). Decisions:
  value-returned pool with a `unique_ptr<std::mutex>` (movable) + a move-ctor
  `CHECK(used_==0)` so a post-handout move can't dangle T03's raw `BlockPool*`; no
  dtor CHECK (nothing dangles into a dead pool but that pointer, whose lifetime
  rule already covers it); extra `head/row_stride`+`slab/total_bytes` accessors
  the kernels/stats-log need. CMake: `engine_kvcache` gains `block_pool.cpp` +
  explicit PUBLIC `engine::memory` (header takes `memory::Allocator*`); no
  `kvcache → kernels` edge yet (M8-T04). +27 tests (new `block_pool_test`, label
  `unit`, no SCALAR_PASS — construction/validation, LIFO alloc/exhaustion/recovery,
  share refcounts, scripted stats invariant, hand-worked `blocks_needed`, §3.3
  capacity numbers, double-free/share-on-free/out-of-range/move-after-handout death
  tests, 8-thread stress) → **1085 green**. Format + scoped tidy clean. T03
  block table & sequence cache handle: `src/kvcache/block_table.{h,cpp}` —
  `BlockTable`, the per-sequence logical→physical block map over the M8-T02
  `BlockPool` (one table drives all layers, §3.1). `AppendTokens(count) →
  StatusOr<vector<int64_t>>` grows the sequence and returns the `slot(pos) =
  blocks_[pos/bs]·bs + pos%bs` mapping (the M8-T04 scatter kernel's addressing
  unit), allocating `blocks_needed` blocks on boundary crossings
  **all-or-nothing** (any `ResourceExhausted` releases the taken blocks and
  leaves table+pool byte-identical); `count ≤ 0` → InvalidArgument. `Truncate`
  releases wholly-empty blocks tail-first (lowest logical block back on top of
  the pool's LIFO free list) keeping the partial tail; `FreeAll`/destructor
  return every block (RAII — the M8-T08/M9 "no leaks" basis). Exposes
  `blocks()` as `std::span<const int32_t>` — the zero-copy `const int32_t*` the
  M8-T05 `PagedDecodeAttentionF32` reads (§8.3/§9.2). Move-only (moved-from
  emptied, move-assign deleted); not thread-safe by design (one table per
  sequence; the pool's mutex covers cross-sequence contention); every owned
  block is refcount-1 (the §6.4 exclusive-tail invariant), so only `Release` is
  used, `Share` waits for M11. **No new link edge** (`BlockTable` calls only
  `BlockPool`; `kvcache → kernels` still lands with M8-T04). +23 tests (new
  `block_table_test`, label `unit`, no SCALAR_PASS — hand-verified §7.2
  prefill/decode slot mappings at `bs = 8` with a primed free list, all-or-
  nothing exhaustion rollback, truncate/free/move, death tests) → **1108
  green**. design §7.2 updated to `bs = 8` + an "as built" notes block.
  Format + scoped tidy clean. T04 KV-write (scatter) kernel & `PagedKvCache`:
  new `src/kernels/kv_scatter.{h,cpp}` (`KvScatterF32(src_k, src_v,
  slot_mapping, T, Hkv, d, bs, k_slab, v_slab)` — writes token-major `[T,Hkv,d]`
  K/V into the paged slabs at the block table's slot mapping, a pure fp32→fp32
  `memcpy` of `d` floats per (token, head): **Class E** bit-exact, **single
  scalar TU** no per-ISA variant like the M6-T06 embedding gather,
  `parallel_for` over tokens with disjoint slots) and
  `src/kvcache/paged_cache.{h,cpp}` (`PagedKvCache : KvCache` — the paged impl
  of the M5 interface; composes `BlockPool*` + one `BlockTable`).
  `append(layer,k,v)` front-loads `SimpleKvCache`-identical validation, then
  **layer 0 grows the table** all-or-nothing (`AppendTokens` → `pending_slots_`,
  a `ResourceExhausted` aborting the forward with nothing written) and **layers
  1..L−1 reuse** that slot mapping, scattering each layer's slabs (§8.2); the
  **per-forward layer protocol is enforced** via `next_layer_`
  (out-of-order/repeated/wrong-`T` append → `InvalidArgument`, stricter than
  §8.2's documented contract). `length()` = committed tokens (one table, all
  layers agree); `capacity()` advisory `num_blocks·bs + free·bs` (corrects §8.1,
  which under-counts the partial-tail free slots); `truncate` delegates +
  resets the pending forward; `view` `Unimplemented` until the M8-T06 gather
  (decode uses `paged_view`). The interface gains **one additive virtual**,
  `paged_view(layer)` → `PagedKvView` (slab bases + block-table ptr + strides),
  default-`Unimplemented` on `KvCache`, overridden zero-copy by `PagedKvCache`
  (the decode fast path, §8.3) — the only consumer change M8-T07 needs.
  **`kvcache → kernels` PRIVATE downward edge added** (ADR-002-permitted, no
  amendment; `paged_cache.h` stays kernels-free); stale "arrives with M8-T04"
  comments in `block_pool.h`/`block_table.h` refreshed. +25 tests (new
  `kv_scatter_test` 8, `kernels` label, no SCALAR_PASS — readback vs an
  independently-simulated paged layout across a non-monotonic-block straddling
  prefill, mid-block/fresh-block decodes, geometry sweep, 1000-token permuted
  mapping, exact bit-pattern fidelity; new `paged_cache_test` 16, `unit` — the
  §7.2 walk verified through `paged_view`, token-by-token == batched, front-
  loaded validation, the layer protocol, layer-0 exhaustion + recovery,
  truncate/reset, RAII, pool sharing; +1 `simple_cache_test` default-`paged_view`)
  → **1133 green**. design §8/§9.1 gained "as built" notes; format + scoped tidy
  clean (kv_cache.h header edit → full includer set swept). T05 paged decode
  attention kernel: `src/kernels/paged_attention.{h,cpp}`
  (`PagedDecodeAttentionF32`) + `internal/paged_attention_common.h`
  (`PagedDecodeUnitsImpl`/`PagedDecodeGroupSlice`/`PagedDecodeArgs`) — decode
  attention (one query/head over the whole cache) reading K/V **through the
  block table** over the paged slabs, the zero-copy fast path
  `PagedKvCache::paged_view` (T04) feeds. Reuses M6-T05's `DecodeUnitsImpl`
  recurrence with the contiguous `k_head + s·d` walk replaced by a block-table
  walk: because `bs | kAttnKb=64` (§4, pool-enforced), a 64-key online-softmax
  unit is exactly `64/bs` whole physical blocks, so per unit it scores
  **block-by-block** (`DotScoreRow` per block into `scores + b·bs`, `>`-folded
  max), runs the identical `ExpRowSum`/`ScaleRow` over the same 64-wide `scores`
  row, and axpys V in ascending key order (only the row *address* changes) →
  **bit-identical to `DecodeAttentionF32`** on the same logical K/V (asserted
  bitwise), bit-identical across thread counts, threaded over `Hkv` kv heads.
  Per-ISA variants are **one-liners in the existing `{scalar,neon,avx2}/
  attention.cpp` TUs** reusing the *same* anonymous-namespace `Ops` (the literal
  reuse **is** the bit-identity guarantee — no separate `{isa}/
  paged_attention.cpp`); `paged_attention.cpp` holds only `KernelTable` dispatch
  + `parallel_for` + the `detail::PagedDecodeAttentionVariant` seam. `bs | kAttnKb`
  **CHECKed at the entry** (death-tested); `num_blocks` a caller-side bound only.
  No consumer wiring (the `OptimizedModel` swap is T07); `src/parallel/`
  untouched (accumulator is `out`); no new link edge. New
  `paged_decode_attention_kernel_test` (kernels, **SCALAR_PASS**, +engine::kvcache/
  memory): bitwise vs the contiguous kernel across `bs ∈ {8,16,32,64}` ×
  many-block/exact-boundary lengths, GQA (incl. `g=12>chunk`), d∈{18,24,64,128},
  thread-count invariance, HF goldens (Class T vs oracle), an **end-to-end pass
  through a real `PagedKvCache`/`BlockPool`** (append token-by-token → `paged_view`
  → kernel, bit-exact), and a `bs∤64` death test — the paged layout materialized
  test-locally (independent of `KvScatterF32`) with a **reverse-permuted** table +
  **poisoned** unused/tail slots. Bench (`attention_bench paged`; BASELINES.md
  M8-T05): isolated paged kernel **0.89×** the contiguous decode (274.7 vs
  245.8 µs, 8t NEON) — the block-indirection cost — but it **replaces the `view()`
  gather** the contiguous engine path needs (~12% decode traffic, §8), so the
  whole-step comparison is T07's job. **1133 → 1149 green** (+16). design
  paged-kv-cache.md §9.2 "as built" + optimized-cpu-execution.md §8 landed-note;
  format + scoped tidy clean (attention_impl.h header edit → attention TUs/tests/
  bench swept; avx2 one-liner hand-reviewed, CI-proven). T06 paged prefill
  attention path: `PagedKvCache::view(layer)` (Unimplemented since T04) now
  **gathers** the layer's committed `[Hkv, len, d]` history — the layout the
  **unchanged** M6 `PrefillAttentionF32` reads — so prefill/prefill-continuation
  through a `PagedKvCache` work on **both backends with zero consumer change**
  (`Attention::forward` and `OptimizedModel::forward` already do append→view→
  attention over a `KvCache&`). New `src/kernels/kv_gather.{h,cpp}`
  (`KvGatherF32`, the `KvScatterF32` mirror: one `memcpy` of `rows·d` floats per
  (head, block), `rows=min(bs, len−b·bs)` clipping the tail; **Class E** →
  single scalar TU, no per-ISA variant, no SCALAR_PASS) and
  `src/kvcache/paged_gather.{h,cpp}` (`GatherLayerKV` — validate, `Tensor::empty`
  outputs, call the kernel over the existing PRIVATE `kvcache → kernels` edge; no
  new link edge). `view` exposes a **per-layer visible length** (mid-forward, a
  not-yet-appended layer sees only its committed history — reproduces
  `SimpleKvCache::view`'s per-layer fill, one int compare). The single-layer
  `attention_test` golden stays on SimpleKvCache (its layer-1-in-isolation
  appends violate the paged layer-0-first protocol by design); reference-backend
  paged coverage lives in `model_test`. +19 tests (**1149 → 1168 green**):
  `kv_gather_test` (kernel vs a reverse-permuted/poisoned simulated layout +
  scatter→gather round trip), `paged_cache_test` (view gathers, per-layer
  mid-forward length, fresh-snapshot, truncate), `optimized_model_test`
  (**paged prefill-continuation bit-exact vs SimpleKvCache**, 1.8e-7 vs
  reference; paged KV invariant; paged greedy), `model_test` (reference-backend
  paged continuation + KV invariant). No BASELINES entry (no perf claim; decode
  still routes through `view()` on a paged cache until T07 swaps in `paged_view`).
  design §2 table + §9.3 "as built"; format + scoped tidy clean. T07 engine
  integration: the paged cache is now the `engine generate` default behind the
  unchanged M5 `KvCache` interface. **The one consumer change** —
  `OptimizedModel::ForwardLayer`'s decode branch (`T==1`) calls
  `cache.paged_view(layer)` first and runs `PagedDecodeAttentionF32` on the
  zero-copy slabs+block-table (§8.3), falling back to `view()`+the contiguous
  decode kernel on `Unimplemented` (so `SimpleKvCache` is untouched) and
  **propagating** any other status; `view()`'s full gather is now confined to
  prefill and the fallback, which is what keeps decode under the ≤10% bound.
  Capacity config landed as pure unit-tested helpers: `core::host_memory_bytes()`
  (§5.3 host-RAM helper, added to `engine_core`; removed from §13 deferred),
  `kvcache::kv_budget.{h,cpp}` (`ParseKvCacheMemory` absolute-`2GiB`/fraction-`0.6`,
  `ResolveKvBudgetBytes` = `f·host_ram−weights−workspace` with 3-term
  `ResourceExhausted`, `BlocksForTokens`), `model::weight_resident_bytes(loaded)`
  (tied-dedup checkpoint bytes, computed pre-`BuildModel`; chosen over a new
  `Model` virtual since the checkpoint bytes serve both backends), and
  `Workspace::BytesFor(config,T)`. `engine generate` gained
  `--kv-cache{paged,simple}`/`--kv-cache-memory SPEC`/`--kv-block-size` and logs
  `BlockPool::stats()`; `bench_generate` got the same, `DoRun` now takes
  `KvCache&` (the A/B). **CLI default divergence** (paged-kv-cache.md §5.1
  as-built): no-budget default = token-sized pool (not the M9 server's `0.9`
  fraction, which would zero-fill ~13 GB on the 16 GB dev box); pool slabs from
  the default CPU allocator (the M2 caching allocator wires in with M9's churning
  runtime). Acceptance met: tiny greedy identical on both fixtures (existing paged
  tests + unchanged `generate.json` goldens); Qwen2-0.5B byte-identical
  paged/simple output ("…Paris…") with stats logged; `bench_generate` decode
  **1.6–4.2%** below same-machine `--kv-cache simple` (≤10% bound, inside
  run-to-run noise; BASELINES.md M8-T07). +~30 tests (new `kv_budget_test` 20;
  `optimized_model_test` +4 ×SCALAR_PASS incl. decode-step bit-exactness +
  paged_view-error-propagation) → **1195 green**. design paged-kv-cache.md
  §5.1/§10.1/§13 + optimized-cpu-execution.md §8 "as built"; format + scoped tidy
  clean. T08 exhaustion behavior & metrics: pinned the pool-dry-mid-generation
  behavior down as a documented, **resumable** error and finalized the
  cache-usage metrics API. The error plumbing + RAII reclamation already existed
  (`BlockPool::Allocate` → `ResourceExhausted` → `BlockTable::AppendTokens` →
  `PagedKvCache::append` → `Model::forward` → `Generate`; a dropped cache returns
  every block), so T08 added the metrics, the graceful-CLI reporting, the
  contract docs, and the missing end-to-end tests. **Metrics**
  (`block_pool.{h,cpp}`): `BlockPoolStats` gained `peak_used` (high-water mark,
  bumped in `Allocate` on success), `exhaustions` (failed-`Allocate` count, one
  per drained append), and `block_size` — both counters monotone since `Create`,
  maintained under the existing mutex (two integer ops off any hot path), so they
  survive full reclamation and answer "did this pool ever run dry?" after the
  fact. **Contract** (`generator.h`): split into *front-loaded* (before any
  forward, cache untouched, nothing emitted — the up-front `capacity()` check is
  **exact** for a private paged cache, so an over-capacity run is rejected here)
  vs *mid-generation* (a shared paged pool drained by another sequence after the
  up-front check passes → the `forward` status propagates; the produced tokens
  are not in the returned `StatusOr` but were each delivered via `on_token`, the
  failing forward wrote nothing, and the cache holds a consistent
  `cache.length()` prefix → **resumable** from the last delivered token,
  reproducing the identical greedy trajectory — the seam M9 preemption/resume and
  M10 streaming build on). **Hardening** (`paged_cache.cpp`): a failed layer-0
  append clears `pending_slots_` so the post-failure state is unambiguous.
  **CLI** (`main.cpp`): `engine generate` reports pool stats (used/free/util,
  peak, exhaustions) on both success and failure, and on a mid-run failure prints
  the streamed tokens + `generation stopped: <status>` + pool state before
  propagating. +12 tests → **1207 green**: `block_pool_test` (+4, counters +
  reclamation-survival + concurrency-bounded), `paged_cache_test` (+2,
  mid-sequence boundary exhaustion with view/paged_view intact + recovery,
  exhaustion-then-drop reclaims all), `generator_test` (+2 reference backend,
  paged front-loaded rejection + the mid-generation-drain-then-resume acceptance),
  `optimized_model_test` (+2 ×SCALAR_PASS, the pool `Allocate → ResourceExhausted`
  chain from inside a `forward` propagating+recovering + paged greedy exhaustion
  reclaiming all blocks on the optimized backend). No BASELINES entry (no perf
  claim). design paged-kv-cache.md §6.2/§10.2/§12 + model-execution.md §10;
  format + scoped tidy clean.

- **M9 (continuous batching scheduler & runtime) — complete** (2026-08-19): T01
  `docs/design/scheduler-runtime.md` — the contract M9-T02…T10 (and the
  M10/M11/M12/M15/M16 interactions) build on: the runtime that turns the M5–M8
  single-request `engine::Generate` into a multi-request, continuously-batched
  system. **Docs-only.** Fixes: the **scheduler stays a `core`-only leaf**
  (ADR-002 rule 4 — it sits beside `engine`, `runtime` mediates; since
  `Request`/`Sequence` live in `runtime` *above* it, the scheduler consumes plain
  descriptor structs — `PoolSnapshot` + per-seq `{arrival, num_computed,
  num_prompt, blocks_held}` — and returns a plain `SchedulerOutput`, so the whole
  policy is table-testable with no pool/model/allocator; `blocks_needed` reproduced
  from `block_size`; **no new ADR edge in all of M9**); **two passes per step, not
  a mixed forward** (prefill-then-decode: different kernels, no CPU launch overhead
  to amortize, decode-priority is a scheduling not execution-order property;
  prefill first so a new seq samples its first token the same step); **batch
  invariance guaranteed bit-for-bit on a fixed ISA** — the CB correctness invariant
  (N concurrent == N sequential) reduces to every op being row-/sequence-local plus
  GEMM==GEMV-row (verified today by `packed_gemm_test.GemvMatchesGemmRow`), which
  pre-answers M17-T04's open batch-invariance question; **policy v1** (decode-first
  → no starvation, preempt latest-arrived, FCFS admission reserving prompt blocks
  only with per-step decode-check + preemption as the safety valve); **preemption =
  evict-and-recompute** resting on the M8-T08 resumable-error seam (free blocks,
  keep generated ids/sampler/stop state, requeue at head, resume by re-prefilling
  `prompt ++ generated` → bit-identical by the KV invariant; no swap-out on CPU;
  liveness argued via oldest-alone-always-fits); **per-request failure isolation**
  (a bad row/forward fails only that sequence, loop survives — flags a required
  additive `BatchedSampler` per-row-status change); full **state machine** +
  legal-transition table (`CHECK`-enforced) + **step-loop pseudocode** + the
  **explicit invariant list** (the three acceptance criteria). Also planned: the
  additive `ForwardRequest` fields (`cu_seqlens`, per-seq `caches`), per-sequence
  K/V append inside the batched forward, the two batched kernels
  (`PrefillAttentionVarlenF32`, `PagedDecodeAttentionBatchedF32`), staging-buffer
  reuse (discharges optimized-cpu-execution.md §6.3 workspace pre-sizing), config
  knobs, and per-ticket testing. Cross-doc pointers added in paged-kv-cache.md
  §9.4/§11, model-execution.md §5.4, optimized-cpu-execution.md §11. No `src/`
  change (1207 tests unchanged). T02 request & sequence abstractions
  (2026-08-19): the `runtime` module's first real content (was anchor-only) —
  `src/runtime/request.h` + `sequence.cpp` (`Request` = immutable client
  description; `Sequence` = mutable execution state: the §3.2 state machine,
  `generated_ids`, this sequence's own `PagedKvCache`, per-sequence
  `Sampler`/`StopChecker`; `SeqState` 6-state enum with 3 terminals;
  `FinishInfo` reusing the M7-T04 `FinishReason`/`StopTrigger` taxonomy +
  runtime-only `kCancelled`/`kFailed`) and `src/runtime/channel.{h,cpp}`
  (`OutputChannel` — SPSC mutex+condvar queue, unbounded, closed once with a
  `FinishInfo`; `Push`/`Close`(→bool, first-wins) producer, `Next`(blocking)/
  `TryNext`(polling)/`finish()` consumer; `OutputItem` mirrors `TokenEvent` by
  value). Decisions: **`Sequence::Create(req, pool) → StatusOr<Sequence>`** a
  fallible factory front-loading sampler/stop validation (move-only, out-of-line
  dtor over a forward-declared `OutputChannel`); **`IsLegalTransition` a public
  `constexpr` predicate** (the §3.2 table in one place, `Transition` `CHECK`s it
  — "illegal transition = CHECK failure"); cache created eagerly (empty = zero
  blocks) with `ReleaseCache`/`EnsureCache` for M9-T09; `Push`-after-close a
  `CHECK`; channel outlives the `Sequence` via `shared_ptr`. Namespace footgun
  recorded: the `engine::engine` finish taxonomy needs a leading-`::`
  using-decl from `engine::runtime`; tests deref optionals via an explicit-guard
  `Require` helper (`bugprone-unchecked-optional-access` doesn't model gtest
  `ASSERT_TRUE`). `engine_runtime` links `PUBLIC core/engine/sampling/kvcache/
  tokenizer` — no new ADR edge. design §3/§4 gained an "as built" note. +25
  tests (`channel_test` 12, `request_test` 13; `runtime` label, no SCALAR_PASS)
  → **1232 green**; format + scoped tidy clean. T03 engine API & request queue:
  `src/runtime/engine.{h,cpp}` — the runtime's public async surface + the
  single-engine-thread loop scaffold (§5, §9.1). `EngineConfig`
  (`max_num_seqs`/`max_num_batched_tokens`/`max_model_len`); `RequestHandle`
  (copyable `shared_ptr<OutputChannel>`: `next_token`/`try_next_token`/
  `await_completion`); `Engine` (`Create`/`Submit`/`Cancel`/`Start`/`Stop`/
  `Step`). Clients `Submit` from any thread into a mutex+condvar command queue
  (`variant<SubmitCmd, CancelCmd>`); the single engine thread drains at each step
  boundary, admits, executes, delivers to per-request channels, retires
  terminals. **`Create` returns `StatusOr<unique_ptr<Engine>>`, `Engine`
  non-movable/non-copyable** (owns a thread + mutex + a map of heap-pinned
  `Entry`s the `Sequence` borrows into; the private-ctor `new`-in-`unique_ptr`
  factory carries a documented `NewDeleteLeaks` NOLINT). **`Submit` does all
  fallible per-sequence work (`Sequence::Create` validation) on the client thread
  before assigning the id** under the queue lock (arrival order == queue order);
  rejections front-loaded (empty prompt / `prompt_len > max_num_batched_tokens`
  → InvalidArgument; peak `prompt_len + max_tokens - 1` over `max_model_len` or
  the pool token capacity → ResourceExhausted; geometry mismatch at `Create`;
  post-`Stop` → FailedPrecondition). **`Step()` ships the §9.1 loop shape with
  two placeholders the later tickets swap in place**: FCFS admission under
  `max_num_seqs`/token-budget/`free_blocks` (M9-T04 → `scheduler::Scheduler`; no
  preemption yet) and **per-sequence execution** (one `model::forward` over the
  single-sequence `ForwardRequest` = the `engine::Generate` decode-loop body
  lifted onto the `Sequence`), so a request's output is bit-identical to a
  standalone `Generate` (asserted). Decode set snapshotted before admission
  (admitted seqs produce one token via prefill, decode next step).
  **Cancellation handled up front in T03** (no in-flight batch at a boundary;
  batched loop moves RUNNING-cancel to step end in M9-T08). **`Stop`
  abort-and-close + idempotent** (`~Engine` calls it); per-request forward/sampler
  fault → that seq `kFailed` + close + free, loop survives (ADR-003; mock injects
  a NaN row). `OutputChannel` gained `AwaitFinish()`; `Entry::seq` is
  `unique_ptr<Sequence>` (two-phase init, avoids the optional-access lint).
  `engine_runtime` links `engine::model` PUBLIC (no new ADR edge). +25 tests
  (`runtime_engine_test`; `runtime` label, no SCALAR_PASS; stress = 6 submitters
  × 40 reqs, random cancels, clean ×20, no leaks) → **1257 green**; design §5
  "as built"; format + scoped tidy clean. T04 scheduler v1 (2026-08-19):
  `src/scheduler/scheduler.{h,cpp}` — the pure decision component (design §6),
  a `core`-only leaf (ADR-002 rule 4): `Scheduler::Schedule(ScheduleInputs) →
  SchedulerOutput` over plain descriptor structs (`waiting`/`running` `SeqDesc`
  + `PoolSnapshot` + `SchedulerConfig`), emitting `prefill` (admitted, explicit
  `num_tokens`), `decode` (one token/running seq), `preempt` (evicted).
  Three ordered passes: **decode-first** (unconditional up to memory →
  starvation impossible), **preempt-to-fit** (evict the latest-arrived — max
  `arrival_index` — refund its `blocks_held`, repeat; emitted latest-first; no
  oldest-alone special case — liveness is a config-sizing guarantee, §10.3),
  **FCFS admission** (walk `waiting` in order; admit while `max_num_seqs` /
  token budget / `BlocksNeeded(0,prompt) ≤ free` hold; **stop at first failure**,
  head-of-line blocking accepted). `class Scheduler` stateless in v1 (the M11
  hook's home, §6.5; documented static-NOLINT); block math a public `constexpr
  BlocksNeeded(cur,add,bs)` cross-checked bit-for-bit vs `BlockPool::
  blocks_needed`; `RequestId` redeclared here (`static_assert`ed == the runtime's
  in `engine.cpp`). Engine wiring: the M9-T03 placeholder `AdmitWaiting` →
  `Engine::ScheduleStep` (fills descriptors from each `Sequence`'s cache —
  `length()`/`block_table().num_blocks()`; waiting prefill = prompt +
  `num_generated()`) + a generalized `Step` that **applies preempt** (§9.1
  step 4: `ReleaseCache` → `kPreempted` → head of `waiting_`) before prefill →
  decode. **Preemption mechanics land in T04, not T09** (the scheduler can now
  emit `preempt`, so the engine must act): the prefill path re-prefills `prompt
  ++ generated` so a resumed sequence's next token is bit-identical to an
  uninterrupted run (§10.2, the M8-T08 resumable seam); M9-T09 keeps the batched-
  loop validation, tiny-pool forced-preemption/no-leak acceptance, and pool-
  sizing config check. `runtime → scheduler` links PUBLIC (`engine.h` names
  `SchedulerOutput`); no new ADR edge. +21 scheduler tests (`scheduler_test`,
  `scheduler` label, no SCALAR_PASS — decode-never-starved, admission block
  boundary, token-budget head-of-line, `max_num_seqs` cap, latest-arrived
  preemption incl. cascade/refund-then-admit, FCFS prefix, determinism, a
  2000-iter structural-invariant fuzz) + 1 engine test
  (`PreemptionResumesWithIdenticalOutput`: 2-block pool forces a real preemption,
  the resumed request's output == a standalone `Generate`, `used == 0` at end) →
  **1279 green**. design scheduler-runtime.md §6.6 "as built"; format + scoped
  tidy clean. T05 batch assembly (2026-08-19): `src/engine/batch.{h,cpp}` —
  `BatchAssembler` + `BatchInputs` (`engine::engine`) flatten the per-step
  scheduled work into staged inputs a batched forward (M9-T06/T07) consumes:
  concatenated `token_ids`, per-token absolute `positions`, `[B+1]` `cu_seqlens`
  prefix sums, per-sequence `caches`, per-request `sample_rows`
  (`sampling::BatchRow`), and — decode only — the `[B, max_blocks]` int32
  block-table tensor + `[B]` `seq_lens`. **Input is a plain `BatchSeqInput`
  descriptor, not `SchedulerOutput` + `Sequence`** — ADR-002 keeps `engine` free
  of `runtime`/`scheduler` types (scheduler is a `core`-only leaf; `Sequence`
  lives above `engine`), so the runtime fills `{token_ids, cache, sampler,
  context}` per scheduled seq and the assembler sits in `engine` beside
  `Generate` (the §8.2 `AssembleBatch(SchedulerOutput, sequences)` shorthand
  realized as this descriptor input). **Two entry points**
  (`AssemblePrefill`/`AssembleDecode`, the §7 two passes); a shared `Flatten`
  fills the common fields, decode adds seq_lens + block_table. **Positions are
  `cache->length() + t` uniformly** (fresh prefill ⇒ `[0,T)`; decode length `L` ⇒
  `L`). **No batch-level slot mapping** (§8.3 keeps K/V append per sequence
  through `PagedKvCache::append`, which owns its slots + exhaustion seam).
  **`block_table` via the abstract `paged_view(0)`** — row `b` = its block ids
  `−1`-padded to `max_blocks`, the tensor a zero-copy `slice`→`reshape` view over
  a 1-D int32 storage grown to a high-water mark; a `SimpleKvCache` propagates
  `paged_view`'s `Unimplemented`. **Allocation-free after warm-up**
  (`staging_bytes()` the stability metric; discharges optimized-cpu-execution.md
  §6.3's deferred "pre-size staging from `--max-num-batched-tokens`"). **Additive
  `ForwardRequest` fields landed here** (§8.1): `cu_seqlens` + `caches`, empty ⇒
  single-sequence path; the two new span members carry `{}` defaults so the ~24
  existing single-sequence designated-init sites stay
  `-Wmissing-designated-field-initializers`-clean (untouched), with an in-header
  `NOLINT(readability-redundant-member-init)` resolving the span-`{}` tidy
  conflict; both backends reject a non-empty batch with `Unimplemented` until
  M9-T07 (`BatchInputs::MakeForwardRequest` builds the batched request).
  CMake: `engine_engine` gains `batch.cpp`, `engine::tensor` PRIVATE→PUBLIC +
  `engine::memory` added (no ADR amendment). +16 tests (`batch_test` 13 `engine`;
  reference/optimized `ForwardRejectsBatchedRequest`, opt ×SCALAR_PASS) → **1295
  green**. design scheduler-runtime.md §8.2 + optimized-cpu-execution.md §6.3 "as
  built"/discharge; format + scoped tidy clean (model.h header edit → full
  includer set swept). T06 varlen batched prefill attention (2026-08-19):
  `kernels::PrefillAttentionVarlenF32` (`src/kernels/attention.{h,cpp}`) — the
  ragged-batch prefill kernel, running the **unchanged** per-sequence
  `PrefillAttentionF32` recurrence over each sequence of a batch delimited by
  `cu_seqlens`, so each sequence's output is **bit-identical** to a standalone
  run (the acceptance guarantee). Realized **without any new per-ISA code, new
  arithmetic, or new template**: the varlen entry reuses the already-dispatched
  `PrefillUnits` variant *verbatim* and adds only sequence-major unit bookkeeping
  in `attention.cpp` (M8-T05 reuse argument, one step further), so there is **no
  `{isa}/*` varlen TU** and SCALAR_PASS covers the shipped bytes — a deliberate
  deviation from the ticket's "(+ per-ISA TUs)" shorthand (as-built in §8.4). K/V
  come in as **per-sequence pointer arrays** (`const float* const* k_seqs/v_seqs`
  + `l_dims[]`), matching §8.3's per-sequence `caches[b]->view(layer)` read the
  T07 model loop will feed; `cu_seqlens` is `int32` (matches
  `ForwardRequest`/`BatchInputs`); continuation `P_b>0` is the standalone `(P,T)`
  case per sequence. Sequence-major unit space (sequence `b` owns
  `[S_b, S_b+H·⌈T_b/kAttnQb⌉)`), split by `detail::PrefillVarlenUnits` (test seam)
  which synthesizes the *same* `PrefillArgs` a standalone call builds → per-seq
  bit-identity + thread/chunk invariance by construction. **Model wiring deferred
  to M9-T07** (the batched `forward` — per-seq K/V append + prefill/decode kernel
  branches; both backends keep the `Unimplemented` batch guard until then). New
  `internal::PrefillVarlenArgs`; no `src/` consumer change, no link-edge change,
  no BASELINES entry (no perf claim; whole-step number is T08). +8 gtest
  (`varlen_attention_kernel_test` ×2 SCALAR_PASS: headline {5,64,129} case, mixed
  past/block-boundary, GQA×head-dims, B==1, thread/chunk invariance, oracle
  Class-T ~1e-6, B==0 no-op) → **1311 ctest green**. design scheduler-runtime.md
  §8.4 "as built" + optimized-cpu-execution.md §8 landed-note; format + scoped
  tidy clean (three attention headers edited → full includer set swept). T07
  batched decode execution (2026-08-19): the batched forward on **both** backends.
  New `kernels::PagedDecodeAttentionBatchedF32` (`src/kernels/paged_attention.{h,
  cpp}` + `internal/paged_attention_common.h`) — decodes B sequences over
  **per-sequence pointer arrays** (`block_tables[B]`/`lengths[B]`) + the shared
  `k_slab`/`v_slab`, reusing the M8-T05 `PagedDecodeUnits` variant verbatim
  (`detail::PagedDecodeBatchedUnits` synthesizes a per-sequence `PagedDecodeArgs`
  and calls the variant on one kv head), so each member is **bit-identical to a
  standalone `PagedDecodeAttentionF32`** by construction; no new per-ISA TU,
  SCALAR_PASS covers it; parallel width `Hkv→B·Hkv`. **Design correction:** the
  §8.2/§9.4-reserved `[B, max_blocks]` `−1`-padded block-table tensor is retired —
  block growth happens *inside* the forward (a boundary-crossing decode token
  allocates a new block during layer-0 append), so a pre-forward snapshot is
  stale; the model self-sources `paged_view(layer)` **after** the append instead.
  `BatchInputs::block_table`/`seq_lens`, `EnsureBlockTable`, and the assembler's
  `memory::Allocator` removed (dead once self-sourced); `engine`'s `tensor` link
  PUBLIC→PRIVATE, `engine::memory` dropped. `OptimizedModel::ForwardBatched`
  (front-loaded `ValidateBatched`) runs the flattened `[ΣT,E]` workspace through
  the shared `RunStack` and branches only `BatchedAttention` (per-seq append
  sliced by `cu_seqlens` → batched paged decode when every T_b==1 over paged
  caches, else varlen prefill via `view()`); `kLast` gathers last rows → `[B,V]`
  GEMM (row-bitwise = per-seq GEMV, §8.5). `ReferenceModel::ForwardBatched` = per-
  member single-seq forward concatenated (the oracle). The `Unimplemented` batch
  guards and the `ForwardLayer` unused `p_len` param are gone. +~30 tests (new
  `paged_decode_attention_batched_kernel_test` ×SCALAR_PASS; `optimized_model_test`
  headline batched-decode-vs-sequential bit-exactness incl. a boundary-crossing
  member + `[B,V]`→`BatchedSampler`; `model_test` reference concatenation +
  validation matrix; obsolete `ForwardRejectsBatchedRequest` removed) → **1335
  ctest green**. No BASELINES entry (whole-step throughput is M9-T08). design
  scheduler-runtime.md §8.2/§8.4 + paged-kv-cache.md §9.4 +
  optimized-cpu-execution.md §8 + model-execution.md §5.4 "as built"; format +
  scoped tidy clean (six headers edited → includer sets swept). T08 engine loop
  integration (2026-08-19): `Engine::Step()`'s M9-T03/T04 per-sequence
  placeholder execution (steps 5–6) replaced **in place** by the two batched
  passes of §9.1 — every request now flows through the batched forward + batched
  sampler; steps 1–4 (drain/retire-cancelled/schedule/preempt) unchanged. Pure
  composition of the M9-T05…T07 pieces in `src/runtime/engine.{h,cpp}` (no new
  `src/` file, no kernel, no ADR edge). `RunBatchPass` (per pass):
  `BuildPassInputs` → `BatchAssembler::Assemble{Prefill,Decode}` (M9-T05) →
  `MakeForwardRequest(kLast)` → `model.forward` (M9-T07 batched `[B,V]`) → one
  `BatchedSampler::Sample` (M7-T06) → per-row `DeliverSampled` (the single-seq
  `Generate` tail); the engine thread owns one reused `assembler_` +
  `batched_sampler_` + per-pass scratch (grown-on-demand, allocation-free
  steady-state decode, §5.2). **B==1 has no fast path** — a single-seq step runs
  the batched code, so the ~26 existing T03/T04 mock tests now drive the batched
  loop unchanged; `ExecuteAndDeliver` (single-seq forward) survives **only** as
  the fault fallback. **Two-tier fault recovery** (T08's §11.2 slice; the
  `BatchedSampler` per-row-status change is M9-T10): a *forward* fault
  (`RecoverForwardFailure`) truncates every member back to its snapshotted
  pre-forward length then re-runs each via `ExecuteAndDeliver` (healthy members
  bit-identical, only the faulter fails; a decode exhaustion is a per-request
  failure here, preemption is M9-T09); a *sampler* fault (`RecoverSampleFailure`)
  re-samples each row over the **same** committed `[B,V]` logits (bit-identical,
  same `detail::SampleRow`), failing only bad rows — the path
  `PerRequestFaultIsolated` (NaN-logits request batched with a healthy one)
  exercises. Deviation: RUNNING-cancel stays top-of-step (not moved to step end
  as the T03 note sketched) — commands drain only at boundaries so no in-flight
  batch exists, top-of-step is equivalent (§9.4 as-built). Tests: new
  `runtime_batching_test.cpp` (`runtime`, **SCALAR_PASS** — first runtime suite
  in the forced-scalar pass): 8 concurrent greedy == 8 sequential `Generate`
  token-for-token on tiny-llama + tiny-qwen2 optimized, staggered mid-flight
  arrivals, recorded throughput **≈0.34×** the 8-sequential wall-clock (< 8×,
  §9.3), pool `used==0`; `runtime_engine_test.cpp` +2 mock CB cases and a batched
  `CannedModel` path so its whole 28-case suite runs on the batched loop. **1335
  → 1345 ctest green.** design scheduler-runtime.md §9.4 "as built"; format +
  scoped tidy clean. T09 preemption & recomputation (2026-08-19): completed the
  preemption story — the *mechanics* (evict-and-recompute → resume by re-prefill
  `prompt ++ generated`) already landed in M9-T04, so T09 is the **config-sizing
  liveness** guarantees, a preemption **counter**, and **acceptance under the
  batched loop** (~40 lines in `src/runtime/`, no new file/kernel/ADR edge).
  `Engine::Submit` now rejects **peak** `prompt_len + max_tokens - 1 >
  max_num_batched_tokens` (not just `prompt_len`) so a preempted sequence's
  resume re-prefill always fits the budget (§10.2's "config validation flags
  this", enforced exactly per request rather than a blanket
  `max_num_batched_tokens ≥ max_model_len` inequality — which would reject a
  large-context model whose requests all fit; peak subsumes the prompt ceiling).
  `Engine::Create` now rejects an **explicitly pinned** `max_model_len` over the
  pool's token capacity (§10.3 — the pool must hold one full-length sequence so
  the oldest-alone never preempts); an auto-resolved `max_model_len` is *not*
  checked (a big-mpe model over a tiny pool is legit — `Submit`'s per-request
  peak-vs-capacity check covers it), so the existing `CannedModel` Create tests
  are untouched. New `Engine::num_preemptions()` counter (bumped in the §9.1
  step-4 apply loop; the M16 metric). **Scope correction:** the M9-T08 code
  comments forward-referencing the *reactive* decode-exhaustion → preemption
  routing to "M9-T09" are corrected to **M9-T10** — for the engine's own pool the
  scheduler is exact (decode demand charged before admission) so a decode append
  never exhausts; the reactive path is reachable only under an externally-drained
  shared pool and is testable only alongside M9-T10's per-row `BatchedSampler`
  status + isolation harness (§11.2 already places it there). +9 tests → **1354
  ctest green**: `runtime_batching_test` `RunPreemptionInvariant` (both fixtures,
  ×2 SCALAR_PASS — 4-block pool cycles 8 concurrent requests through preemption,
  each output **identical** to standalone `Generate`, `num_preemptions()>0`,
  `used==0`); `runtime_engine_test` repeated-preemption + the §10.2/§10.3 checks
  and boundaries; `PreemptionResumesWithIdenticalOutput` now asserts
  `num_preemptions()==1`. design scheduler-runtime.md §6.4 + §10.4 "as built" +
  §11.2 reassignment; format + scoped tidy clean. T10 cancellation &
  per-request failure isolation (2026-08-19): the final M9 ticket. Cancellation
  *mechanics* (boundary-drain → `RetireCancelled` → RAII block return) already
  landed in T03/T08, so T10 is three changes plus the missing acceptance tests.
  **`BatchedSampler` per-row status** (the §11.2/§14 planned refinement):
  `Sample` gained a caller-owned `std::span<core::Status> row_status` output and
  now **always fills every row** — Ok (with `out[b]` set) or that row's
  recoverable error (bad history id → InvalidArgument, non-finite row →
  Internal); the batch-level return covers only front-loaded shape/null
  validation of the engine-built inputs (`CHECK`ed by the engine), replacing
  M7-T06's lowest-index-error return so one bad row no longer discards the batch
  (additive signature; the reference `Sampler` untouched; all four call sites +
  model-execution.md §15.6 updated). Engine `RunBatchPass` reads `row_status`
  per row (healthy → deliver from the batched pass bit-for-bit, faulting → only
  that sequence FAILED); the T08 stopgap `RecoverSampleFailure` **deleted**.
  **Reactive decode-exhaustion routing** (§10.4 reassignment): a decode-time
  `ResourceExhausted` in `ExecuteAndDeliver` now returns false rather than
  failing; `RecoverForwardFailure` preempts the **latest-arrived other running
  sequence** (§6.2 policy, via a shared new `PreemptSequence` helper) and
  retries, failing only the **sole-occupant** case — reachable only under a
  shared/externally-drained pool (the engine's own scheduler is exact), tested
  via a mock that drains the pool from inside its decode forward. +9 tests
  (**1354 → 1363 ctest green**): `runtime_engine_test` cancel-after-prefill /
  cancel-while-preempted / decode-fault-with-healthy-neighbors /
  shared-pool-exhaustion-preempts-younger / sole-occupant-fails;
  `runtime_batching_test` Llama/Qwen mid-flight-cancel-isolation (real fixtures,
  ×2 SCALAR_PASS — survivors bit-identical to standalone `Generate`);
  `batched_sampler_test` `IsolatesPerRowErrors` (rewritten) + a `row_status`
  size-mismatch case. design scheduler-runtime.md §11.3 "as built" + §14 (item
  marked landed) + model-execution.md §15.6; format + scoped tidy clean.

Next up: **M10-T01** (design doc & ADR: server architecture) — the first ticket
of Milestone 10 (serving layer / OpenAI-compatible HTTP API). Write
`docs/design/server.md` and the HTTP-library-selection ADR (evaluate standalone
Asio + a minimal HTTP layer vs. `cpp-httplib` vs. Drogon — criteria: streaming
support, thread-model fit vs. the engine thread, dependency weight), the server
threading model relative to the runtime's engine thread, the OpenAI API schema
(`/v1/completions`, `/v1/chat/completions`), the SSE streaming design, and the
backpressure policy (the runtime's unbounded `OutputChannel` gains M10's bounded
policy). Acceptance (ROADMAP M10-T01): the ADR records the library decision with
a comparison table; the design doc includes a request-flow diagram from socket
to engine channel and back. Docs-only. Per ROADMAP M10-T01.
