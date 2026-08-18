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
M4-T09 done (2026-08-08: BPE encoding — `Tokenizer::encode` runs the full
§6.4 pipeline and both fixture golden suites pass byte-identically (all
24 vectors × llama3/qwen2 × with/without specials, incl. the
encode_synthetic invalid-UTF-8 cases pinning our own semantics). New
pieces: `tools/gen_unicode/` (stdlib-only Python; downloads pinned
Unicode 16.0.0 UCD files — SHA-256-verified, cache gitignored — and
emits `src/tokenizer/unicode_data.inc`: \p{L}/\p{N}/White_Space ranges,
CCC, fully-expanded canonical decompositions, composition pairs minus
exclusions, NFC quick-check ranges, with a generation stamp);
`unicode.h/.cpp` (range/lookup tables + hand-written NFC —
decompose/reorder/compose with algorithmic Hangul, fast path when every
cp is QC=Yes with ccc 0 — plus the module's shared UTF-8 codec, moved
out of bpe.cpp); `pretokenize.h/.cpp` (SplitSpec selection: exactly the
two fixture pattern strings, differing only in digit-run bound 3 vs 1,
rejected by name otherwise — now enforced at *parse* time in from_json;
matcher = one helper per regex alternative in pattern order, incl. the
\s+(?!\S) backtracking rule and \s*[\r\n]+ last-CRLF rule); bpe.cpp
gains `bpe_split` (leftmost-lowest-rank rescan loop over contiguous
substring views — ranks are total so no other tie-break exists);
tokenizer.cpp gains the added-token longest-match scan (first-byte
buckets, length-desc; lstrip/rstrip consume adjacent whitespace,
normalized:false matches raw text before NFC, normalized:true in a
second pass after — embedded specials always match, add_special_tokens
only gates template inserts), the valid/invalid byte-run split (invalid
runs are their own pre-tokens), and ignore_merges short-circuit.
Non-obvious: HF's fullwidth-comma behavior (a single non-CRLF/L/N cp is
a valid alt-2 prefix, so "，世界" is ONE pre-token — an intuitive
punctuation-splits-here expectation is wrong, and the goldens caught
exactly that during development); contraction folding is
ASCII-only (documented gap: no golden exercises ſ→s); missing final
merge symbol → InvalidArgument naming it. Design amendments recorded
(§2 file table: generator emits the .inc, algorithms hand-written; §6.4
amendment block). Tests: +32 (unicode_test 11 — categories incl.
NBSP-is-\s and ZWSP-isn't, UTF-8 codec boundaries, NFC cases
cross-checked against Python unicodedata 16.0.0 incl. exclusions,
Hangul, reordering, blocking, idempotence, invalid-byte totality;
pretokenize_test 10 — selection + per-alternative semantics;
tokenizer_encode_test 10 — the two golden suites plus synthetic
micro-tokenizers for merge order, rank-beats-position, ignore_merges,
missing-symbol error, longest-match, lstrip/rstrip, normalized-flag
passes, template prefix/suffix/empty; tokenizer_test builder now uses
the real Qwen-2 pattern and asserts unknown-pattern rejection). 576
tests green; format + scoped tidy clean (tidy: generated .inc wrapped
in NOLINT for designated-initializers; MatchAt split into helpers to
pass cognitive-complexity).
M4-T10 done (2026-08-08: decoding & incremental detokenization — M4
complete. `Tokenizer::decode(ids, skip_special_tokens)` replaces the T09
stub: per-id range check (`InvalidArgument` naming id and position),
added tokens decode to raw content and are dropped only when skipping
*and* `special`, byte concatenation, then a lossy UTF-8 pass. The
non-obvious finding driving the design: the goldens show HF's decode is
NOT raw byte concatenation — `tokenizers` materializes the bytes via
Rust's `String::from_utf8_lossy`, i.e. strict UTF-8 (overlong/surrogate/
>U+10FFFF invalid) with the WHATWG maximal-subpart policy, one U+FFFD
per maximal subpart (`FF FE FD` → 3 replacements, truncated `E2 82` tail
→ 1). unicode.h gains that strict classifier (`classify_utf8_prefix`:
kComplete/kIncomplete/kInvalid + subpart length; `append_utf8_lossy`),
coexisting with the deliberately-lenient encode-side `decode_utf8`. New
`detokenize.h/.cpp`: `DetokenizerStream` (§6.5 API — push/finish over a
≤3-byte carried prefix) drains via the same classifier, so streaming
output is bit-identical to batch decode (the design's "up to the U+FFFD
policy" hedge proved unnecessary — tests assert exact equality);
finish() turns a truncated carry into one U+FFFD and resets (stream
reusable); a rejected push leaves state untouched. Golden pairing
pinned: `decoded` ↔ (ids, skip=false), `decoded_skip_special` ↔
(ids_with_special, skip=true) — recorded with the rest in a §6.5
amendment block. Tests: +16/−1 (tokenizer_decode_test 11 — both
families' 24-vector golden decode + encode↔decode round-trip +
token-by-token streaming with a test-local independent UTF-8 validator
asserting every emission valid and concat == batch; multi-token-emoji
buffering pinned on llama3's 3-token U+1F469; truncated-tail finish
policy vs batch agreement; out-of-range errors batch and stream incl.
state-preserved-after-error; mini-tokenizer non-special added token
survives skip_special_tokens [audit correction 2026-08-08: the file
defines 12 gtest cases, not 11, and the round-trip + token-by-token
streaming golden loops were vacuous until the M4 audit fix below];
unicode_test +4 — classifier
complete/incomplete/maximal-subpart shapes incl. overlong, surrogate,
broken-mid-sequence, and from_utf8_lossy-parity cases; the
DecodeIsUnimplementedUntilT10 stub test removed). 591 tests green;
format + scoped tidy clean.

M4 audit fixes done (2026-08-08: post-milestone audit of M4 against its
acceptance criteria, fixes in one change. Tokenizer: the two
tokenizer_decode_test golden loops (encode↔decode round-trip and
token-by-token streaming) ranged over `LoadVectors(...)["cases"]` — a
reference into a temporary destroyed before the first iteration under
C++20 — so both loops ran zero times: two T10 acceptance criteria were
unverified and qwen2 streaming was entirely unexercised. Fixed with
named locals plus the ≥-case-count guards the healthy golden loops
already had, and the round-trip assertion corrected to compare against
the golden `decoded` rather than the input text (a normalizing
tokenizer legitimately round-trips decomposed input to its NFC form —
qwen2's nfc_decomposed vector would fail the old assertion, which is
how the audit proved the loop never ran). Model: the safetensors
alignment check made absolute — `(8 + header_len + begin) % itemsize`,
not `begin % itemsize`, which an unpadded header defeats by shifting
the whole data section (design §3.4 audit note; synthetic test headers
now space-padded to an 8-aligned data section like real HF serializers;
new negative test pins the unpadded-header rejection). New coverage:
loader-level missing-weight failure naming the canonical weight
(micro-checkpoint gains an omit parameter); LOG_WARN capture in the
unexpected-tensor weight_map test (the "warning list" criterion is the
log line, not just the report vector); new bpe_test pins the 256-byte
alphabet bijection — total, injective, exact inverse, string
round-trip. Docs: model-loading.md §6.3 generator output corrected to
`unicode_data.inc`, §7.1 layout gains the committed tokenizer LICENSE
files, §7.2 gains huggingface_hub in the requirements list and precise
safetensors header-ordering wording, §8's T04 row matched to the tests
as written; T10 entry corrected (12 decode tests, not 11). 597 tests
green; format + scoped tidy clean. Known gap, deliberately left open
here (mirrors M3): every commit from T04 on is unpushed, so no CI run
has built M4 on x86-64 or compiled the AVX2 TUs against it — push and
confirm green before treating M4 as fully validated.)

M4 CI-gap fallout fixed (2026-08-08: the first CI run to build
T08–T10's code failed both gcc jobs — the three tokenizer-test
`INSTANTIATE_TEST_SUITE_P` name-generator lambdas named their parameter
`info`, which shadows the parameter of the generated gtest function the
macro expands them into; GCC's -Wshadow flags this, clang's does not,
and gcc had never compiled these TUs (clang-only dev machine, no CI
since T03). Renamed to `param_info` in tokenizer_test,
tokenizer_encode_test, tokenizer_decode_test. Verified by a full local
Homebrew GCC 15 warnings-as-errors build — clean, 597/597 in both the
gcc and clang trees; format + scoped tidy clean.)

## Milestone 5 — CPU Reference Engine

M5-T01 done (2026-08-17: `docs/design/model-execution.md` — the working
contract for model execution that M6 (optimized backend), M8 (paged
cache), M9 (batching), M11 (prefix caching), M12 (fusions), M13 (quant
layers), M14 (activation capture), and M15 (spec decoding) all build on.
Module decomposition: `cpu` = stateless clarity-first fp32 reference ops
(gemm/rmsnorm/silu_mul/add/softmax/embedding/rope/attention — the oracle,
links `tensor`+`parallel`, never `kernels`); `model` = the graph (`Linear`
*interface* + `ReferenceLinear`, `RmsNorm`/`Rope`/`Attention`/`Mlp`/
`DecoderLayer`, abstract `Model`, registry/builder); `kvcache` =
`KvCache` interface + `SimpleKvCache`; `engine` = greedy generator +
`Backend` seam. Fixes: (1) the exact `Model::forward(ForwardRequest&) →
StatusOr<Tensor>` contract — token-major `[T,E]` activations (no batch
dim; M9 flattens to `[Σ,E]`+cu_seqlens additively), per-token `positions`
vector, `KvCache*`, `LogitsMode{kLast→[1,V], kAll→[T,V]}` (kAll a real
mode for M14 ppl / M15 verify, not test-only), `ActivationHook*`; inputs
validated as recoverable `Status` per M9-T10, not CHECK. (2) GQA layout:
query head `h` reads kv head `h/(H/Hkv)` (repeat_interleave, HF
`repeat_kv`), no materialized repeat (M6-T04); `head_dim` may ≠
hidden/heads. (3) KV cache v0: per-sequence, all-layers, device-agnostic
`append`/`view`/`length`/`capacity`/`truncate` (the roadmap's
append/view/current-length/reset, + `truncate` for M15 rollback with
`reset`=`truncate(0)`); head-major `[Hkv,len,d]` storage; the KV
invariant (token-by-token == full recompute) as T06's acceptance; a full
§6.4 account of what M8 paging changes (storage→block pool+block tables
+refcounts, append→slot-mapping scatter, `view()` is the accessor that
does *not* survive paging → attention uses a gather helper,
`ResourceExhausted` on exhaustion, `dtype` field for M13 INT8 KV) vs what
stays (the abstract `KvCache` surface, per-sequence ownership,
append-only immutability for M11). (4) RoPE spec: HF half-rotation
(`rotate_half`, not interleaved), fp32 inv_freq tables, `rope_scaling`
none/linear/llama3 (else `Unimplemented`). (5) `Linear` is an interface
(opaque backend-owned weight) so M6 repacking / M13 `QuantizedLinear`
slot in with no layer-code change; tied embeddings = an overridable
binding, not a pointer-alias assumption (M6-T06). (6) backend seam:
`Model`/`KvCache` polymorphic, `Backend{kReference,kOptimized}`
construction-time choice, generator backend-agnostic (M6-T07 reuses it);
doc commits only to the `Model`/`Linear`-level seam, leaving M6-T01 free
on layer-class reuse. (7) registry keyed by HF `architecture_name`, one
family builder for Llama+Qwen2 (M5-T10 = wiring not new code). (8)
hooks: one `ActivationHook` seam (fixture-key names + `linear_input:*`
for M14-T02), nullptr = zero cost (M16-T03). (9) M5 fixture plan
(op/rope/attention goldens + generate.json + new tiny-qwen2, each landing
with its ticket, ≤5 MB/model). Per-ticket test table (T02–T10) with
Class-T tolerances, no SCALAR_PASS (calls `cpu::` directly). ADR
consequences: **ADR-002 Amendment 5 adds `model → kvcache`** (layer 2's
first intra-layer edge; alternatives — `Model` in `engine`, or raw K/V
views through forward — rejected with reasons); cpu-backend.md §2's
provisional `cpu → parallel` amendment flag retired (Amendment 4 already
allowed it — no amendment needed, M5-T02 links it directly). §15
placeholder left for M7-T01's sampling section. Docs-only change;
model-loading.md §7.1 + tests/fixtures/README.md point forward to the M5
fixture additions.)

M5-T02 done (2026-08-17: CPU GEMM + `Linear`). `src/cpu/gemm.cpp`
(`cpu::gemm`, replacing the placeholder `cpu.cpp`): the reference fp32
GEMM in the **NT form** `C[M,N] = A[M,K] · B[N,K]ᵀ (+ bias[N])` — the only
GEMM shape the reference provides, because every `Linear` is exactly it
(weight stored `[out,in]`, so the inner K loop is a contiguous dot over
both operands; attention's matmuls get their own loops in M5-T05). Weight
**and bias** may be f32/f16/bf16, widened per element through
`tensor/half.h`'s `operator float()` (`cpu` never links `kernels` —
model-execution.md §2.1); bias converted once and added after the K sum
(matches torch addmm rounding). Cache-blocked (kTileM=kTileN=64,
kTileK=256) and `parallel_for`-threaded over the flattened output-tile
grid, but each `C[m,n]` sums ascending-k into **one fp32 accumulator**, so
results are **bit-identical across thread counts and tile sizes** (Class T
only vs HF, which reduces in its own order). Recoverable `Status` for
malformed shapes/dtypes/contiguity/K-mismatch/bias-length, each message
naming the offending input (ADR-003); undefined handles are `CHECK`.
`src/model/modules.h`+`modules.cpp`: the abstract **`Linear` interface**
(model-execution.md §4.1 verbatim — special members protected+defaulted so
concrete impls stay movable for `StatusOr` return without slicing through a
`Linear&`) + `ReferenceLinear` (`Create` validates the `[out,in]` weight
and optional `[out]` bias; `forward` is `cpu::gemm` with `y`
caller-allocated — the M12 allocation-free-decode property — and its own
x/y-named errors). Non-obvious decisions: NT-only GEMM (no transpose flag,
YAGNI); bias kept in its storage dtype and widened on the fly rather than
copied to f32 at construction (preserves strict zero-copy — Qwen2 carries
bf16 biases); `y` is caller-allocated, InvalidArgument if wrong.
New fixture `tiny-llama/expected/ops.safetensors` (+`ops_meta.json`) via
`tools/gen_fixtures/tiny_llama_ops.py` (subcommand `tiny-llama-ops`,
byte-identical regen verified): 10 GEMM cases — real bf16 q/gate/down/
lm_head weights on the fixture dims, decode GEMV (M=1), skinny/wide, k==1,
odd tile-tails with f32 bias, and f16-weight / bf16-bias widening cases;
ground truth is fp32 `A @ B.float().T (+ bias.float())`. Tests (12): 6
`CpuGemmTest` — fixture goldens across shapes/dtypes (rtol/atol 1e-4;
worst observed max-abs-diff 7.6e-6), on-the-fly widening bit-exact vs
`ops::cast`-then-gemm, tiling/threading bit-exact vs a serial triple loop,
bias-added-at-end, 8 error paths each asserting code + message, and 512³ <
1 s (ran ~22 ms) with a serial spot-check; 6 `ReferenceLinearTest` —
Create accessors + malformed-weight/bias rejection, forward == direct gemm
through `Linear&`, T==1 GEMV, bad x/y shapes, and a real bf16 `q_proj`
loaded end-to-end via `load_model` matching the golden. Labels: gemm_test
`cpu`, linear_test `model`; ordinary portable tests (call `cpu::`
directly, no SCALAR_PASS). 609 tests green; format + scoped tidy clean.)
