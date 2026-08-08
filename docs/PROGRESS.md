# Progress log

The per-ticket work log for the CPU-first engine (post-pivot, ADR-004;
newest entries last). Each entry records what landed, the non-obvious
implementation decisions, and the test count at the time — the searchable
"why is it built this way" record, one step above git history. Division of
labor: ROADMAP.md carries the authoritative ✅ DONE markers, design docs
carry the binding decisions, CLAUDE.md keeps only a compact current
position pointing here. **When a ticket lands, append its detailed entry
here and refresh CLAUDE.md's short status in the same change.**

Pre-pivot history (M0–M2, the CUDA era): `docs/archive/ROADMAP-v1.md`,
`docs/design/retired/cuda-backend.md`, and git history.

## Milestone 3 — The Pivot: CPU Backend Foundation

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
M3 audit fixes done (2026-08-08: post-milestone audit of M3 against its
acceptance criteria, fixes in one change. Kernels: `detail::<Kernel>Variant
(Isa)` test seams + per-suite wiring asserts pin every vector table slot by
pointer identity, closing the silent-scalar-fallback vacuous-pass hole;
ENGINE_FORCE_ISA failure modes now exercised end-to-end via two `sh -c`
ctest registrations asserting the fatal message (CTest fails signal-deaths
even under PASS_REGULAR_EXPRESSION, hence the shell wrapper); convert sweep
gained the 2^20+3 size and output-side offsets (was input-only, capped at
4096). Parallel: `ThreadPool::Run` CHECKs the region flag (was a silent
self-deadlock; flag moved to `thread_pool.h` detail), body/combine
exceptions terminate via noexcept frames on the inline path too (was
propagate-vs-terminate depending on n vs grain), `parallel_reduce`'s slot
allocation converts bad_alloc to CHECK (ADR-003), `ReferenceTreeSum`
rewritten structurally independent (fresh-vector rounds vs in-place fold),
+2 death tests. Hygiene: `is_standard_layout` static_asserts in half.h
(convert.cpp's comment now true), FPCR.FZ16 precondition documented in
neon/convert.cpp, dispatch.h "fatal at startup" corrected to lazy
first-dispatch. Docs: CUDA-era leftovers purged from living surfaces —
dependencies.md toolkit/NCCL exemption retired, tensor.md "until M2"
phrasings + broken cuda-backend link fixed, ROADMAP future-directions paths
repointed to archive/retired, scripts' C++/CUDA wording and `cu cuh` globs
dropped, check-tidy's stale example replaced; M3-T02's removed-test
enumeration recorded as a ROADMAP audit note; CLAUDE.md history trimmed to
5 lines; design doc §3.4 + §4.2 audit implementation notes added. 393 tests
green. Known gap, deliberately not closed here: the milestone3 branch has
no CI run yet, so the AVX2 TUs have never been compiled and the TSAN job
has never executed — push before treating M3 as fully validated. [Gap
closed 2026-08-08: main's CI runs compile the AVX2 TUs green under
GCC+Clang warnings-as-errors and the TSAN job executes green.]).

## Milestone 4 — Model Loading & Tokenization

M4-T01 done (2026-08-08: `docs/design/model-loading.md` — load pipeline
(HF snapshot dir → config parse/arch registry {Llama, Qwen2} → weight
discovery → read-only mmap as shared `memory::Buffer` → safetensors header
parse with strict validation (bounds, non-overlap, gap-free, itemsize-aligned
offsets) → per-arch weight-name mapping → zero-copy `Tensor` registry →
`LoadedModel`); specifies a new public `Tensor::from_buffer` factory
(M4-T04, tensor.md note — the buffer-wrapping ctor is private); canonical
weight namespace (`layers.{i}.attn.*`, `attn_norm`/`mlp_norm`) + Llama/Qwen2
tables incl. Qwen2 QKV biases and tied-lm_head aliasing (same Tensor handle;
checkpoint copy ignored); dtype policy: preserve checkpoint dtype
({F32,F16,BF16} as weights, parser is format-complete, loader rejects the
rest — quant formats enter at that seam in M12–M13); tokenizer scope:
byte-level BPE from tokenizer.json only, sentencepiece/unigram explicitly
deferred (Llama-2-era only), component whitelist incl. NFC normalizer
(Qwen) + `ignore_merges` (Llama 3), pre-tokenization via hand-written
matcher over generated Unicode tables (no regex engine, no ICU;
`tools/gen_unicode/` codegen, pinned Unicode version), `DetokenizerStream`
with UTF-8 buffering + U+FFFD tail policy; fixture strategy: committed
under `tests/fixtures/` (tiny-llama bf16 + sharded copy + activation
goldens as safetensors, qwen2 name inventory, two real tokenizer.json +
vectors.json), generated by `tools/gen_fixtures/` (pinned reqs,
byte-identical regen, CI never runs Python), budget ≤5 MB/model fixture,
≤40 MB total, Llama license file committed alongside. No ADR amendment
needed — model/tokenizer use only listed layer-2→layer-1 edges. Docs-only
change).
M4-T02 done (2026-08-08: `tools/gen_fixtures/` — Python 3.11 package
(pinned requirements: torch 2.13.0, transformers 5.14.1, tokenizers 0.22.2,
safetensors 0.8.0, huggingface_hub 1.27.0; `tools/.python-version`), CLI
subcommands `tiny-llama` (random-weight 2-layer LlamaForCausalLM, hidden 64,
4 heads/2 KV heads, vocab 512, bf16; hand-authored config.json, single-file
+ deterministic 2-shard safetensors, fp32 hook-captured activation goldens
{embeddings, layers.{i}, final_norm, logits} + meta.json), `qwen2-names`
(header-only safetensors metadata fetch of Qwen/Qwen2-0.5B-Instruct — tied
lm_head + QKV biases — no weight download), `tokenizer-vectors` (byte-copied
unmodified tokenizer.json + LICENSE for llama3
(NousResearch/Meta-Llama-3-8B-Instruct mirror; meta-llama is gated — design
doc §7.3 note) and qwen2, 24 golden cases each incl. NFC composed/decomposed
discriminators, ZWJ emoji, contraction case pairs, digit-run grouping,
in-text specials); `tools/regen_fixtures.sh` with `--verify` (temp-dir regen
+ diff) — byte-identity verified. Determinism: fixed seed, 1-thread
deterministic torch, sorted-key JSON/safetensors, commit-hash-pinned
downloads. Design-doc §6.4 note: HF tokenizers rejects non-UTF-8 input, so
malformed-UTF-8 vectors are `encode_synthetic` pinning our byte-fallback
semantics (invalid runs standalone/string-final only, where implementations
provably agree); decode side is true HF golden. ~16 MB fixtures committed
under budget; `tools/README.md` + `tests/fixtures/README.md` (regen
discipline, provenance/licensing incl. committed Llama LICENSE); CI
untouched — no Python in the test path).
M4-T03 done (2026-08-08: `src/model/config.h/.cpp` — `ModelConfig`/
`RopeScaling`/`Architecture` + `ParseModelConfig`/`LoadModelConfig` per
design §3.2 (nlohmann PRIVATE, non-throwing parse, exception-free
accessors); registry {LlamaForCausalLM, Qwen2ForCausalLM} → else
Unimplemented listing supported; required {architectures, hidden_size,
intermediate_size, num_hidden_layers, num_attention_heads, vocab_size} +
max_position_embeddings (design-doc §3.2 note: no universal HF default,
always serialized); HF-semantics defaults incl. per-arch attention_bias
(Llama false / Qwen2 true, explicit wins), head_dim derived with
divisibility check when absent, rope_scaling null→nullopt with legacy
"type" key fallback (§3.2 note), torch_dtype via tensor::from_string;
validation: strict positivity, kv-heads divide heads, int narrowing
checks — all errors name the field (tested). Parse split into per-stage
helpers (clang-tidy cognitive-complexity 25). New fixtures
`tests/fixtures/models/configs/{llama3,qwen2}/config.json` — byte-copied
real configs via new `gen_fixtures model-configs` subcommand; Llama side is
the **3.1** mirror (NousResearch, pinned; upstream LICENSE committed) for
its rope_scaling block, Qwen2 exercises absence + omitted attention_bias;
design §7.1 layout + fixtures/tools READMEs updated. Tests: 30 in
`tests/unit/config_test.cpp` (label `model`) — every field asserted on
both real configs + tiny-llama, defaults/explicit-wins, missing-field
loop, wrong-type table, positivity table, GQA divisibility, overflow,
unknown arch/dtype, NotFound + path-prefixed parse errors. 423 tests
green; format + scoped tidy clean).
CI tidy retirement done (2026-08-08: the full-sweep tidy job exceeded its
15-minute timeout on every M4 push (runs cancelled, `ci` gate red despite
all other jobs green) — job removed from ci.yml (`needs` updated, header
comment records the rationale + accepted x86-64 coverage gap); tidy is
local-only now per the Build & test section above. The one finding CI
surfaced before timing out — misc-const-correctness on `Xgetbv0()`'s
eax/edx in src/kernels/dispatch.cpp, an asm-output false positive
invisible to arm64-local tidy (x86-64-only block) where const would not
even compile — suppressed with a reasoned NOLINTBEGIN/END. check-tidy.sh
header + cpu-backend.md §8 amendment + ci-signal-workflow memory updated;
423 tests green, format clean, scoped tidy on dispatch.cpp clean).
M4-T04 done (2026-08-08: safetensors file parser — `Tensor::from_buffer`
public factory added to `src/tensor/` (validated front door for wrapping
external storage as a contiguous view: overflow-checked window bounds →
InvalidArgument, reserved dtypes → Unimplemented, device from the buffer,
null buffer CHECK; tensor.md §7 "Extended in M4-T04" note); `src/model/
mapped_file.h/.cpp` `MapFileReadOnly` (PROT_READ/MAP_PRIVATE mmap wrapped
as a shared `memory::Buffer` whose deleter munmaps — the file stays mapped
while any tensor view is alive with no new lifetime machinery; empty file →
engaged zero-size Buffer; ENOENT → NotFound, non-regular file rejected);
`src/model/safetensors.h/.cpp` `SafetensorsFile`
(Open/names/contains/tensor/metadata; little-endian `static_assert`;
validation per design §3.4 checks 1–5: all framing arithmetic in uint64 +
256 MiB header cap, exception-free nlohmann parse, entries with exactly
{dtype, shape, data_offsets}, container dtype table
F32/F16/BF16/I8/U8/I32/I64/BOOL/F8_E4M3 loaded + F64/I16/U16/U32/U64/
F8_E5M2 → Unimplemented + unrecognized → InvalidArgument, `end − begin ==
numel × itemsize` overflow-checked, `begin % itemsize == 0`, sorted cursor
walk rejecting overlap/gap/uncovered tail — zero-length ranges must sit
exactly on a boundary; F8_E4M3 entries parse but `tensor()` returns
Unimplemented via from_buffer's reserved-dtype rule; every error carries
the file path and tensor name). `engine_model` gained the two TUs and now
links PUBLIC `engine::tensor` + `engine::memory`. Tests: 36 new —
from_buffer suite (incl. uint64-overflow windows, zero-size buffers,
null-buffer death), mapped_file suite (NUL-byte round-trip, directory
rejection, mapping outlives the original handle), safetensors suite
(synthetic round-trip, scalar + zero-numel, tensor-free files, tiny-llama
21-tensor name/shape/dtype spot checks, activations `input_ids` vs
meta.json ground truth, tensor bytes vs independent ifstream read, lifetime
test reading through a Tensor after the SafetensorsFile is destroyed, and
~25 fuzz-ish negatives asserting status code + message names the
file/tensor/field). 459 tests green; format + scoped tidy clean).
M4-T05 done (2026-08-08: sharded checkpoint support —
`src/model/checkpoint.h/.cpp` `Checkpoint` (Open/names/contains/tensor),
the unified weights interface the rest of the loader consumes; single-file
and sharded are indistinguishable behind it (design §3.1/§3.6). `Open`
takes the model *directory* and resolves discovery: index wins when both
spellings are present (HF loading order — tested with a deliberately
corrupt model.safetensors alongside a valid sharded set), neither →
NotFound naming both. Index parse is eager and exception-free; only
"weight_map" is interpreted — "metadata"/unknown top-level keys tolerated
(HF-tooling output, additive fields must not break loading — a reasoned
contrast to safetensors.cpp's strictness, noted in a comment). Shard
filenames must be bare (no separators, not ".."/empty — hostile index
must not escape the model dir) and are existence-checked eagerly so
"missing shard" fails at Open listing every absent shard; mapping +
header validation stay lazy per shard on first tensor() touch (tested:
a corrupt untouched shard doesn't block reading the healthy one), then
cached. First-map consistency check: shard tensor set must equal the set
the index maps to it — catches index-names-absent-tensor,
shard-tensor-not-in-index, and cross-shard duplicates (reported as "the
index maps it to <other shard>"), all offenders listed in one message.
Single-file case wraps SafetensorsFile as shard 0, names copied into the
unified index. Non-obvious: `tensor()` is non-const (lazy cache), and the
sketch's `StatusOr<const SafetensorsFile*>` internal accessor keeps
ASSIGN_OR_RETURN usable. Tests: 17 in `tests/unit/checkpoint_test.cpp`
(label `model`) — synthetic two-shard round-trip, tensor-outlives-
checkpoint, index-wins, lazy mapping, empty weight_map valid, metadata
tolerance, missing dir/neither-spelling/missing-shards NotFound cases,
malformed-index + non-bare-filename tables, all three inconsistency
shapes, unknown-name NotFound, and the acceptance criterion: the
committed 2-shard tiny-llama fixture resolves every tensor with metadata
and bytes memcmp-identical to the single-file fixture. 476 tests green;
format + scoped tidy clean).
M4-T06 done (2026-08-08: weight-name mapping — `src/model/weight_map.h/.cpp`:
the canonical weight namespace (design §4) and per-architecture HF→canonical
tables, expanded per layer from config with expected shapes substituted.
Two-layer API so the core logic is testable without tensor bytes:
`BuildWeightMap(config, WeightInventory)` over a name→Shape metadata view
(what the qwen2-weight-names.json fixture is), plus a
`BuildWeightMap(config, Checkpoint&)` convenience that extracts the
inventory from zero-copy views. Missing required weights are *recorded* in
`WeightMapReport.missing`, not an error at build — the separate policy step
`CheckNoMissingWeights` turns a non-empty list into one InvalidArgument
naming every missing weight (design §3.6's all-names-in-one-round-trip
rule; the split exists because the report struct carries `missing` and the
loader owns when to fail). Shape mismatch is an immediate InvalidArgument
naming weight, expected, and actual; expected dims live in plain
`vector<int64_t>`, not Shape, so hostile config dims can't trip Shape's
overflow CHECK. Tied embeddings map both `lm_head.weight` and
`embed_tokens.weight` to the same source name, so resolution shares one
Tensor handle structurally (shallow-copy semantics); a redundantly
serialized lm_head goes to `ignored` (alias wins, matching HF), as do
`rotary_emb.inv_freq` artifacts. Non-obvious decision: bias rows are gated
on `attention_bias` for *both* architectures — Llama gets q/k/v/o biases
(HF LlamaAttention biases o_proj too), Qwen2 q/k/v only (Qwen2Attention
leaves o_proj bias-free); design §4's Llama bullet was clarified in the
same change. `unexpected` is one aggregated WARN line, `ignored` DEBUG.
Tests: 12 in `tests/unit/weight_map_test.cpp` (label `model`; first test
target with a direct nlohmann_json dep, to read the fixture) — synthetic
complete-Llama mapping with report empty + spot-checked spellings,
missing-weight report + policy error (single and several, all listed),
stray tensor → unexpected, inv_freq → ignored, shape mismatch naming
weight/expected/actual, Llama bias gating both ways (absent-bias missing
list incl. o_proj; bias-with-flag-false → unexpected), tiny-llama fixture
resolving all 21 weights through Checkpoint, tied-config variant proving
same-data()-pointer aliasing + lm_head ignored, and the real Qwen2-0.5B
inventory (290 tensors, 24 layers): clean report, 291 canonicals, q/k/v
bias mapping, no o_proj bias demanded, lm_head aliased; removing a bias →
missing. 488 tests green; format + scoped tidy clean).
M4-T07 done (2026-08-08: model loader — `src/model/loader.h/.cpp`:
`load_model(dir) → StatusOr<LoadedModel>` composing the whole M4 weight
path: config parse (§3.2) → checkpoint discovery (§3.1/§3.6) → weight-name
mapping + shape validation + missing-weights policy (§4) → materializing
every canonical weight as a zero-copy CPU tensor, checkpoint dtypes
preserved. Config runs first deliberately, so a bad path or unsupported
architecture fails before any weight I/O. New logic beyond composition:
(1) the dtype-acceptance policy (design §5) — only F32/F16/BF16 pass as
*weights*; integer/bool tensors parse fine but the loader rejects them
with Unimplemented naming the canonical weight and pointing at M12–M13;
(2) the §3.1 pickle error — when discovery finds neither safetensors
spelling, the loader scans for `.bin`/`.pt`/`.pth` files and upgrades the
NotFound to an actionable convert-to-safetensors message listing them
(placed in loader.cpp, not Checkpoint::Open, to keep Checkpoint a pure
safetensors abstraction); (3) progress logging (§3.7) — one INFO line per
stage in loader.cpp, plus per-file `mapped model-…safetensors (4.9 GiB)`
lines added to checkpoint.cpp at the two places mapping actually happens
(the lazy shard() path and the eager single-file Open branch — the design
put these lines in §3.7 but lazy mapping means only Checkpoint can emit
them). Materialization resolves each unique checkpoint tensor once and
reuses the handle, so a tied lm_head/embed_tokens pair is literally the
same Tensor (data() pointer equality, not just equal bytes). Tests: 10 in
`tests/integration/loader_test.cpp` (label `model`; first real
integration-tree suite) — tiny-llama end-to-end (config fields, all 21
canonical names, empty report, bf16 preserved); value spot-checks against
fixture-recorded bytes (hardcoded bf16 bit patterns from the committed
model.safetensors plus byte-identity cross-reads vs direct HF-named
SafetensorsFile access); sharded layout staged with config into a scratch
dir, byte-identical to single-file, with the per-shard `mapped` log lines
asserted via SetLogStreamForTesting; tied-config variant (same-pointer
alias, lm_head → ignored); a synthetic complete 1-layer micro checkpoint
proving itself clean, then with one I32 weight → Unimplemented naming the
canonical weight and M12; and the four actionable-error cases (missing
dir, config-less dir, pickle-only dir naming the .bin and saying convert,
unknown architecture listing supported ones). 498 tests green; format +
scoped tidy clean).
M4-T08 done (2026-08-08: tokenizer model parsing — `src/tokenizer/` gains
its first real sources. `bpe.h/.cpp`: the GPT-2 byte-level alphabet as a
constexpr 256-entry bijection (printable-latin identity + 68 remapped to
U+0100+n) with both directions plus `map_bytes_to_alphabet` /
`unmap_alphabet_to_bytes` and a small self-contained UTF-8 codec (the full
Unicode tables stay in M4-T09 — the design's file table originally pinned
`tools/gen_unicode/` to T08, moved to T09 where its first consumer lives).
`tokenizer.h/.cpp`: `Tokenizer::from_file`/`from_json` (the
Parse/Load-style testable seam), enforcing the §6.2 whitelist —
`Unimplemented` naming the component for off-whitelist types/flags
(non-BPE models incl. unigram, NFKC, unknown pre/post-processors and
decoders, byte_fallback, unk_token, subword affixes, single_word,
Split.invert, non-null truncation/padding), `InvalidArgument` for
malformed input — and building the T09/T10 structures: transparent-hash
token→id map, dense id→token and id→raw-bytes vectors (base vocab
unmapped through the alphabet, added tokens verbatim), merge-rank map
keyed "left right", sorted added-token registry with id index, template
prefix/suffix ids, split pattern + NFC flag + ignore_merges. Vocab ids
are validated contiguous [0, n) (the dense tables' precondition); merges
are validated against the vocab including the merged product; added
tokens must extend the id space gaplessly or textually match the base
entry they shadow. encode/decode are declared but return Unimplemented
until T09/T10. Non-obvious: the committed fixtures falsified the §6.2
table's original post_processor row two ways (Llama-3 wraps
TemplateProcessing in a Sequence with an offsets-only ByteLevel, and its
template *carries* a `pair` key), so the whitelist gained
ByteLevel/Sequence and "pair rejected" became "pair ignored" — recorded
as an amendment block in the design doc, which also now pins both merge
serializations ("l r" strings and [l, r] arrays). bos_id/eos_id derive
solely from TemplateProcessing insertions: Llama-3 bos=128000, no eos;
Qwen2 neither. Tests: 46 in `tests/unit/tokenizer_test.cpp` (new
`tokenizer` ctest label) — golden parses of both real fixtures
(vocab_size 128256/151646, token↔id pairs, token_bytes "Ġworld"→" world",
special metadata incl. lstrip/rstrip, bos/eos), a config_test-style
JSON-assembly builder driving a 17-case parameterized whitelist suite
plus vocab/merge/added-token malformation cases, each asserting code and
message-names-the-offender; sentencepiece binary via from_json and
from_file. 544 tests green; format + scoped tidy clean).
