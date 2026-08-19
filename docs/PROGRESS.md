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

M5-T03 done (2026-08-17: CPU normalization & activation ops). Four new
reference ops in `src/cpu/` (one `.cpp` each: `rmsnorm.cpp`,
`activation.cpp` = silu_mul + add, `softmax.cpp`) plus a private
`src/cpu/detail.h` sharing widening/validation helpers (`Widen`,
`RequireContiguousRank`, `RequireF32`) across the new ops; `gemm.cpp`
keeps its own battle-tested local helpers untouched. **`cpu::rmsnorm(x,
weight, eps, y)`** — per-row `y = x * rsqrt(mean(x²)+eps) * weight`, HF
LlamaRMSNorm order, mean-of-squares in a single ascending fp32
accumulator; `x` accepts f32/f16/bf16 (the "RMSNorm on bf16 input"
criterion) and `weight` f32/f16/bf16, each widened per element via
`half.h` (dispatched on the (x,weight) dtype pair, `cpu` never links
`kernels`); this is the **pure fp32 forward** — it deliberately omits HF's
intermediate `.to(input_dtype)` round-trip, matching the fp32-forward
goldens. **`cpu::silu_mul(gate, up, y)`** — SwiGLU `silu(gate) ⊙ up`,
`silu(v)=v/(1+exp(-v))` (saturating exp → no NaN on large-magnitude
gates), f32. **`cpu::add(a, b, y)`** — residual add, f32. **`cpu::softmax
(x, y)`** — row-wise last-dim, max-subtracted for stability; a `-inf`
entry maps to exactly 0 (the property M5-T05's causal mask needs), an
all-`-inf` row yields NaN (documented caller error). All are
`parallel_for`-threaded over rows with single-accumulator per-row
reductions, so **bit-identical across thread counts**; `y` is
caller-allocated and (silu_mul/add/softmax/rmsnorm-with-f32-x) may alias
an input. Recoverable `Status` naming the offending input for every
malformed shape/dtype/contiguity; undefined handles are `CHECK`.
`src/model/modules.{h,cpp}`: the **`RmsNorm` module** (concrete class, no
interface — unlike `Linear`) holding the zero-copy `[E]` weight + eps from
`config.rms_norm_eps`; `forward` requires fp32 activations (the M5 graph's
invariant) and calls `cpu::rmsnorm`. Non-obvious decisions: the op accepts
bf16 `x` at the boundary while the *module* enforces f32 (op-level
criterion vs graph invariant); softmax/add/silu are f32-only (attention
scores and activations are always f32 — §3.1/§3.3), only rmsnorm widens;
`Mlp` left to M5-T07 (this ticket is norms+activations, T07 composes the
MLP). Fixtures: extended `tools/gen_fixtures/tiny_llama_ops.py`
(`tiny-llama-ops`) with 10 new cases appended to `ops.safetensors`/
`ops_meta.json` **after** the GEMM draws so the committed GEMM tensors
stay byte-identical (verified: all 32 unchanged; `regen_fixtures.sh
--verify` byte-clean) — rmsnorm_f32/rmsnorm_bf16/rmsnorm_eps (eps 0.5),
softmax_typical/large(±1e4)/causal(-inf), silu_mul/silu_mul_unit(up=ones →
pure SiLU)/silu_mul_large(±100)/add_basic; ground truth all fp32
(`xf*rsqrt(mean(xf²)+eps)*w.float()`, `torch.softmax`, `F.silu(gate)*up`,
`a+b`). Tests (23): `rmsnorm_test.cpp` (11) — fixture goldens across
dtype/eps, on-the-fly-widening bit-exact vs `ops::cast`-then-rmsnorm,
threading bit-exact vs serial, in-place==out-of-place, eps-applied +
zero-row-finite, 6 error paths, RmsNorm module accessors/rejections/
forward==op/bad-activation/real-bf16-`attn_norm.weight`-loaded-via-
`load_model`; `activation_test.cpp` (12) — silu_mul/add/softmax fixture
goldens, aliasing bit-exact, softmax rows-sum-to-1 / causal-masked-exactly-
zero / in-place / N==1==1.0 / large-magnitude-finite, error paths. Observed
max-abs-diff: rmsnorm ≤9.5e-7 (tol 1e-4), silu_mul ≤2.4e-7 (tol 1e-4),
softmax ≤3.0e-8 (tol 1e-5). Labels: both `cpu`; ordinary portable tests
(call `cpu::` directly, no SCALAR_PASS). 632 tests green; format + scoped
tidy clean.

M5-T04 done (2026-08-17: Embedding & RoPE, CPU). Two new reference ops in
`src/cpu/` (`embedding.cpp`, `rope.cpp`), two new modules in
`src/model/modules.{h,cpp}`. **`cpu::embedding_lookup(table, ids, y)`** —
gathers `table[ids[t]]` widened f32/f16/bf16→fp32 per element (zero-copy
bf16 checkpoint table never up-converted at load); `ids` is a
`std::span<const std::int32_t>` (the `ForwardRequest.token_ids` span, passed
straight through — no i64 copy). Out-of-range id → `InvalidArgument` naming
index *and* value (pre-scanned before the threaded gather so no OOB read).
**`cpu::rope_apply(x, positions, cos, sin)`** — in-place HF half-rotation
RoPE on `x[T, Hx, d]` (Q with Hx=H, K with Hx=Hkv, called once per tensor):
per pair `(j, j+d/2)` rotated by `cos[p,j]`/`sin[p,j]`, both halves read
before either is written (in-place-safe). `positions` bounds pre-scanned →
`InvalidArgument` naming index/value (this is where §5's
`max(positions) < max_position_embeddings` materializes for the reference).
Both ops `parallel_for`-threaded over tokens with per-element maps →
bit-identical across thread counts. **`Embedding` module** (concrete, like
`RmsNorm`) — zero-copy `[V,E]` table, `forward(ids) → y` via the op;
introduced as a module (the design originally had the lookup as a bare op in
`ReferenceModel`) to be M6-T06's tied-weight sharing seam — design §4.2
updated to record the addition. **`Rope` module** — `Create(head_dim, theta,
rope_scaling, num_positions)` precomputes `cos`/`sin` `[num_positions, d/2]`
(only the first half stored; half-rotation reuses `cos[p,j]` for both
elements of a pair) and exposes `inv_freq`; `apply(q, k, positions)` calls
`cpu::rope_apply` twice. `rope_scaling`: absent/`default` → identity,
`linear` → `inv_freq /= factor` (HF `_compute_linear_scaling` form, not the
doc's literal `p/factor` — §7 clarified), `llama3` → exact HF
`_compute_llama3_parameters` piecewise wavelength scaling (validated
`factor>0`, `high≠low`, `orig_max_pos>0`); any other `rope_type` →
`Unimplemented` listing supported. **Non-obvious decisions:** `inv_freq` and
the angle `p·inv_freq[j]` formed in **fp64** (accuracy) and stored fp32,
while HF uses fp32 → agreement is Class T (fp32-`cosf` vs fp64-`cos` range
reduction diverges with position: negligible at the tiny fixture's max 127,
~5e-3 at 131071); `inv_freq` itself agrees to ~1 ulp so the scaling goldens
(which target `inv_freq`) stay tight; table construction is config
interpretation and lives in the `Rope` module, not in the stateless `cpu`
op. Fixtures: new `tools/gen_fixtures/tiny_llama_rope.py`
(`tiny-llama-rope` subcommand → `rope.safetensors`/`rope_meta.json`) — HF
`LlamaRotaryEmbedding`/`apply_rotary_pos_emb` goldens: `tiny_table` cos/sin
`[128,8]`, `tiny_sparse`/`tiny_contig` apply I/O in token-major `[T,H,d]`
(positions {0,1,127} with GQA H=4/Hkv=2, and {0..7}), `llama3` scaled
`inv_freq [64]` + cos/sin `[5,64]` from the committed Llama-3.1 config,
`linear` scaled `inv_freq [8]` + cos/sin (synthetic factor 4); plus one
`embedding_edge` case appended to `ops.safetensors` (real bf16 embed table,
ids {0, V−1, 42, 42, 255}) after the existing draws so the M5-T02/T03 GEMM/
norm bytes are untouched (verified byte-identical; `regen_fixtures.sh
--verify` clean). Observed max-abs-diff: embedding bit-exact (widening copy,
tol 0); rope tiny cos/sin ≤1.6e-6 (tol 1e-4), tiny apply ≤2.1e-6 (tol 1e-4),
llama3 inv_freq 6.0e-8 (tol 1e-6), llama3 cos/sin ≤4.7e-4 (tol 1e-3, the
range-reduction band). Tests (26): `embedding_test.cpp` (12) — ops +
activations goldens, widening equivalence, threading bit-exact, module
forward, 6 error paths; `rope_test.cpp` (14, incl. the 2 parameterized
apply cases) — tiny table/apply goldens, half-rotation-not-interleaved
guard, position-0 identity (bit-exact), threading bit-exact, llama3/linear
scaling goldens, 7 Create/apply error paths. Labels: both `cpu`; ordinary
portable tests. 658 tests green; format + scoped tidy clean.

M5-T05 done (2026-08-17: Causal attention, CPU). New reference op
`src/cpu/attention.cpp` (`cpu::attention`), new `Attention` module in
`src/model/modules.{h,cpp}`, and the KV-cache interface header
`src/kvcache/kv_cache.h`. **`cpu::attention(q, k, v, scale, out)`** — naive
causal GQA self-attention: q `[T,H,d]`, head-major cache k/v `[Hkv,L,d]`
(L=P+T), out `[T,H,d]`. Materializes scores `[H·T,L]`, causal-masks with
`-inf` (new query `t` at cache position `P+t` attends keys `[0,P+t]`),
softmaxes via `cpu::softmax` (reusing its documented `-inf → 0` mask
contract), then contracts against V. GQA by **KV-head indexing** — query head
`h` reads kv head `h/g` (`g=H/Hkv`, repeat_interleave), **no materialized
repeat**. `scale` (=1/sqrt(d)) multiplies the completed dot (HF order: matmul
then scaling). Two `parallel_for` passes over `(h,t)` rows, each output a
single ascending fp32 accumulator → bit-identical across thread counts. The
op's compute is split into `ScorePass`/`ContextPass` helpers (cognitive-
complexity budget). **`Attention` module** — owns q/k/v/o `Linear`s as owning
`std::unique_ptr<Linear>` (so M6 `PackedLinear`/M13 `QuantizedLinear` slot in;
the class is move-only) plus a `Rope`; `Create` validates projection shapes
against (H, Hkv, d) without assuming `E==H·d` (decoupled head_dim honored);
`forward(x, positions, layer, cache, y)` projects QKV → reshapes to
`[T,heads,d]` → RoPE on Q/K → `cache.append` (token-major `[T,Hkv,d]`) →
`cache.view` (head-major `[Hkv,P+T,d]`) → `cpu::attention` → reshape ctx →
`o_proj`. Validation front-loaded so a failure never half-appends; cache
errors propagate. **`src/kvcache/kv_cache.h`** — the abstract `KvCache`
interface + `CacheGeometry`/`KvView` (§6.1), landed here (not T06) because
`Attention` consumes it; the `model → kvcache` CMake edge (ADR-002 Amendment
5) wired PRIVATE, modules.h only forward-declares `KvCache`. `SimpleKvCache`
+ the token-by-token KV invariant remain M5-T06; T05 exercised the interface
through a test-local `FakeKvCache` double. **Non-obvious decisions:** the
score scale is applied *after* the dot (HF `matmul * scaling`, not seeded into
the accumulator); the cache-view golden comparison uses a looser 1e-3 band
(it's the raw projection+RoPE output, before o_proj/softmax average the GEMM
Class-T spread down) while the final `out` holds tight (2e-4, observed
≤2e-5); attention's masked scores use exact `-inf` (→ softmax 0), numerically
identical to HF's `finfo.min` additive mask. Fixtures: new
`tools/gen_fixtures/tiny_llama_attention.py` (`tiny-llama-attention` →
`attention.safetensors`/`attention_meta.json`) — drives each layer's
`self_attn` directly (fp32 tiny-llama, `_attn_implementation="eager"` via a
registered capture wrapper over `eager_attention_forward`, explicit additive
causal mask, `DynamicCache`-seeded past for P>0), storing per case (layer,P,T)
the module I/O and op intermediates. 5 cases: `l{0,1}_prefill_empty` (P=0,T=8),
`l{0,1}_prefill_continue` (P=5,T=6), `l0_decode` (P=7,T=1); GQA H=4/Hkv=2
inherent, both layers (distinct weights). 49 KB, `regen_fixtures.sh --verify`
byte-clean (existing ops/rope bytes untouched). Observed max-abs-diff: op
`ctx` ≤3.6e-7 (tol 1e-4); module `out` ≤1.8e-5 (tol 2e-4); cache view ≤5.8e-4
(tol 1e-3); bit-exact-vs-serial atol=rtol=0. Tests (9): `attention_test.cpp`
— op goldens (5 cases), bit-exact-vs-serial (threading), causal-mask-hides-
future, GQA-interleave-not-tiling, 7 op error paths; module goldens (5 cases,
incl. accumulated-cache-view check + length), `Create` shape validation, 4
`forward` error paths (incl. over-capacity → ResourceExhausted), `FakeKvCache`
self-check. Label `cpu`; ordinary portable tests. 667 tests green; format +
scoped tidy clean.

M5-T06 done (2026-08-17: KV cache v0). New `src/kvcache/simple_cache.{h,cpp}`
(`SimpleKvCache`, the v0 contiguous implementation of the M5-T05 `KvCache`
interface); the placeholder `kvcache.cpp` anchor TU removed. **Storage:** two
contiguous fp32 tensors `[num_layers, Hkv, capacity, d]` (K and V), allocated
once at `Create(geom, capacity)` from a caller-chosen capacity; head-major so
one kv head's history is contiguous (matches HF's cache after `transpose(1,2)`
and the layout M6-T05 decode attention reads). **`append(layer, k, v)`** takes
the token-major `[T,Hkv,d]` block the `Attention` module produces and
transposes it to `[Hkv,T,d]` at the layer's fill offset; validation
front-loaded (over-capacity → `ResourceExhausted`, rank/shape/dtype/contiguity
+ a k-vs-v T-agreement check → `InvalidArgument`) so a rejected append leaves
the layer untouched. **`view(layer)`** gathers the layer's `[Hkv,fill,d]`
history into a fresh contiguous tensor. **`length()`** = the minimum per-layer
fill (committed = what every layer agrees on; well-defined mid-forward and
after a failed forward). **`truncate(n)`** drops every layer's fill to `n`
(also re-synchronizing layers that disagree); `reset()` is `truncate(0)`.
**Non-obvious decisions:** (1) `view()` **copies** rather than returning a
zero-copy slice — a `[Hkv,fill,d]` window of the `[…,capacity,d]` store is
inner-strided whenever `fill < capacity`, and `cpu::attention` requires
contiguous K/V; copying keeps the T05 op contract untouched and mirrors the
seam M8 keeps (v0's `view()` is the reference's "gather cached K/V" helper;
M8's is a block-table gather; the `Attention` code above them is unchanged).
model-execution.md §6.2 updated to say gather-copy, not "zero-copy slice".
(2) the roadmap's acceptance invariant is stated at **logits** level, but
`Model::forward` is M5-T07 — so T06 lands the invariant at **attention-chain**
level (norms/MLP/residuals don't touch the cache), and T07 elevates the same
check to full-model logits. The invariant holds **bit-exactly** here
(max_abs_diff = 0): each reference op reduces a row in a single ascending fp32
accumulator, and a masked (softmax-0) key contributes `0.0 * v == 0.0` exactly,
so full-prefill and token-by-token schedules produce identical sums. Tests
(13): `simple_cache_test.cpp` (12) — Create validation (geometry/capacity/
non-f32→Unimplemented), append→view head-major transpose, empty-layer view,
malformed appends (layer OOR, rank/Hkv/d/dtype/contiguity/T=0/k-v-T-mismatch,
state-unchanged), view-bad-layer, over-capacity (ResourceExhausted + unchanged
+ exact-fill), length=min-across-layers, truncate (drop-tail/resync/reset,
range errors), append-only immutability, per-sequence independence; plus
`attention_test.cpp`'s new `KvCacheInvariantMatchesFullRecompute` — a two-layer
real-tiny-llama attention chain: full-prefill vs token-by-token, chunked
prefill `[0,5)+[5,T)`, and truncate-then-redecode (all diff 0, tol 1e-5). The
T05 `FakeKvCache` double was replaced by the real `SimpleKvCache` throughout
`attention_test.cpp` (a `PreloadHeadMajor` helper transposes the head-major
fixture past-blocks through the real `append`), so the prefill-continuation
goldens now exercise the cache end-to-end; the `FakeKvCache` self-check test
was dropped (superseded by `simple_cache_test.cpp`). 679 tests green; format +
scoped tidy clean.

M5-T07 done (2026-08-17: transformer block & full model forward). New
`src/model/modules.{h,cpp}` gain `Mlp` (SwiGLU — `down(silu(gate(x)) ⊙ up(x))`,
owns gate/up/down `Linear`s as `unique_ptr<Linear>`, move-only, `Create`
validates gate/up `[I,E]` + down `[E,I]`) and `DecoderLayer` (pre-norm:
`h=attn_norm(x); r=x+attn(h); y=r+mlp(mlp_norm(r))`, owns two `RmsNorm`s + one
`Attention` + one `Mlp`, move-only, `Create` validates all four agree on E,
threads `layer`/`cache` straight to `Attention`). New `src/model/model.h` — the
`Model::forward` contract (§5.1): `LogitsMode{kLast,kAll}`, `ForwardRequest`
(token_ids/positions/cache/logits_mode/hook), the abstract `Model`
(`forward`/`config`/`cache_geometry`), and the `ActivationHook`/`ActivationEvent`
seam (§11). New `src/model/reference_model.{h,cpp}` — `ReferenceModel::Create`
binds every module's weights from `LoadedModel.weights` by canonical name
(embedding, final norm, lm_head, and per-layer attn/mlp), consuming the map;
`forward` runs embedding → N `DecoderLayer`s → final norm → lm_head with
front-loaded validation, `kLast` (project only the last position, [1,V]) vs
`kAll` ([T,V]), and per-stage hook emission. **Non-obvious decisions:** (1)
**`model → kvcache` promoted from PRIVATE to PUBLIC** in CMake: `model.h`'s
`cache_geometry()` returns `kvcache::CacheGeometry` *by value* and
`ForwardRequest` holds a `kvcache::KvCache*`, so a consumer of model.h needs the
kvcache headers — modules.h alone (which only forward-declares `KvCache`) would
have kept it PRIVATE. ADR-002 Amendment 5's "no public model header re-exports
kvcache types" note is relaxed with a one-line clarification (the `Model`
contract is itself stated in terms of the cache). No cycle, no new edge — the
same `model → kvcache` edge, now public. (2) **`linear_input:<name>` hook events
deferred to M14-T02**: §11 lists them, but emitting them needs the hook threaded
through `Linear`/`Attention`/`Mlp` forwards (signature changes to landed
modules); T07's acceptance needs only the four fixture-named stage events
(`embeddings`/`layers.{i}`/`final_norm`/`logits`), which land here.
model-execution.md §11 updated to say so. (3) **`Rope` is copied per layer**
(each `Attention` owns a `Rope` by value) rather than shared — keeps
`Attention`'s signature untouched; noted as an M6 table-sharing item. (4)
bias binding reads the weight map, not arch flags: `BuildLinear` attaches
`.bias` only when `config.attention_bias` *and* the map carries the entry, so
Qwen2's q/k/v-but-not-o bias pattern (M5-T10) is wiring, not new code. (5)
the private ctor is wrapped via `make_unique(ReferenceModel(...))` (move), not
raw `new`, to satisfy the leak analyzer. **Observed max-abs-diff** (fixture =
HF's fp32 forward of the bf16 checkpoint, the same computation): end-to-end
logits 3.7e-6; per-stage `layers.0` 5.0e-7, `layers.1` 4.7e-7, `final_norm`
1.7e-5, `logits` 3.7e-6; `embeddings` bit-exact — all under the 2e-4 tolerance;
kLast-vs-kAll-last-row and the KV invariant are bit-exact (atol=rtol=0). Tests
(15): `model_test.cpp` — end-to-end logits (kAll), per-layer hook dump (event
order/names/layers + each stage vs golden), null-hook zero-cost, stage-isolated
DecoderLayer/final-norm/lm_head goldens, kLast==kAll-last-row, KV invariant at
logits (token-by-token, chunked prefill, truncate-then-redecode), `Mlp`
Create-validation + hand-checked SwiGLU numeric, `DecoderLayer`
Create-validation, `ReferenceModel::Create` missing-weight → InvalidArgument
(names it), geometry/config accessors, forward error paths (empty/size-mismatch/
id-OOR/pos-OOR/null-cache/geometry-mismatch/over-capacity, cache-unchanged).
Label `model`. `model → kvcache` CMake edge now PUBLIC. 694 tests green; format
+ scoped tidy clean (scoped over every `modules.h`/`model.h` includer).

M5-T08 done (2026-08-17: architecture registry). New
`src/model/registry.{h,cpp}` — `BuildModel(LoadedModel, BuildOptions) →
StatusOr<unique_ptr<Model>>` dispatching on `config.architecture_name` (the raw
HF string); `RegisterArchitecture(string_view, ModelBuilder) → Status`;
`SupportedArchitectures() → sorted vector<string>`. A single
`BuildReferenceFamily` builder serves both `LlamaForCausalLM` and
`Qwen2ForCausalLM` — the only family differences (`attention_bias`, config
values) already flow through `ReferenceModel::Create`, so **both strings are
registered in T08**; M5-T10 adds only the Qwen fixture, not a registration or
new layer code. Unknown arch → `Unimplemented` listing the supported names (same
wording shape as `ParseArchitecture`). **Non-obvious decisions:** (1) **`enum
class Backend { kReference, kOptimized }` relocated from the sketched
`engine/backend.h` into `registry.h`** — `BuildOptions` names it and `model`
cannot depend on `engine` (ADR-002: `engine → model`); M5-T09's
`engine/backend.h` will re-export it. `kOptimized` → `Unimplemented` until M6.
design §2/§8/§9 updated to match. (2) **Built-ins register lazily, not via
static initializers**: `engine_model` is a *static library*, so a TU whose only
content is a file-scope self-registering object gets dropped by the linker
(`--gc-sections`/archive-member selection). Instead a mutex-guarded
`GetRegistry()` function-local static pre-populates Llama/Qwen2 on first use;
`BuildModel` copies the resolved builder out under the lock and invokes it
unlocked (so a builder can't deadlock on re-entrant registration). (3)
`RegisterArchitecture` returns `Status` (not `void` as §9 sketched):
`AlreadyExists` on a duplicate name, `InvalidArgument` on empty — duplicate
registration is a real bug worth surfacing, and it's testable. Added
`SupportedArchitectures()` for the error string + tests. (4) In the `.cpp` the
builder is inserted via `try_emplace(key)` then `it->second = std::move(builder)`
rather than `try_emplace(key, std::move(builder))` — clang-tidy's
`performance-unnecessary-value-param` doesn't see the move through
`try_emplace`'s forwarding template, so a direct move-assign both clears the
false positive and reads clearer. Tests (7): `registry_test.cpp` — both families
in `SupportedArchitectures` (+ sorted); unknown → `Unimplemented` naming the
unknown *and* both supported; **`BuildModel` bit-identical to a direct
`ReferenceModel::Create`** (atol=rtol=0 — the builder is a pure pass-through, no
new numerics); the Qwen2 string routes to the same family builder (logits ==
Llama-labelled build of the same checkpoint); `kOptimized` → `Unimplemented`; a
test-local `DummyModel` + one `RegisterArchitecture` call is visible in
`SupportedArchitectures` and buildable via `BuildModel` (the extensibility
criterion); duplicate registration → `AlreadyExists`, empty name →
`InvalidArgument`. Label `model`, no `SCALAR_PASS` (pure portable C++). Also
refreshed the stale "M5-T08 adds…" comment in `reference_model.h` to present
tense. 701 tests green; format + scoped tidy clean (registry.cpp,
registry_test.cpp, and the `reference_model.h` includers).

M5-T09 done (2026-08-18: greedy generation loop). New
`src/engine/generator.{h,cpp}` — `Generate(Model&, KvCache&, prompt_ids,
GenerateOptions, TokenCallback) → StatusOr<vector<int32>>`: prefill the whole
prompt in one `kLast` forward at positions `P0..P0+T-1` (`P0 = cache.length()`,
so it continues from a non-empty cache), argmax the `[1,V]` logits, then decode
one token per forward at the running `cache.length()`, feeding the previous
token back. Returns the continuation (prompt excluded). `GenerateOptions{
max_new_tokens, eos_ids }`; `on_token` fires once per returned id, in order,
after that id's append and before the next forward (the streaming seam M10
uses). New `src/engine/backend.h` — `Backend`/`BuildOptions` re-export +
header-only `BackendName`/`ParseBackend`; the placeholder `engine.cpp` anchor
removed (generator.cpp now anchors `engine_engine`). **Non-obvious decisions:**
(1) **Greedy = strict-`>` argmax → lowest-index tie-break**, stated so the
determinism criterion doesn't rest on `std::max_element`'s incidental behavior;
a NaN max → `Internal` rather than a bogus token. (2) **EOS id is included in
the output** (HF-style: the matched id is the last element), and `on_token`
fires for it too. (3) **Capacity checked up front on the worst case**
(`P0 + T + (max_new_tokens − 1)`, ignoring early EOS) → `ResourceExhausted`
before anything is generated — a `StatusOr<vector>` can't carry a partial result
beside a Status, so failing mid-loop would silently drop the tokens already
produced. Documented as the one judgment call; M10 streaming can revisit. (4)
New **`ModelConfig::eos_token_ids`** (the M4-config addition the loop needs):
HF's `eos_token_id` serializes as an int *or* a list of ints (Llama-3 ships a
list), both parse into the set, validated in `[0, vocab_size)`; a wrong shape or
non-int element → `InvalidArgument` naming `eos_token_id`/`eos_token_id[i]`.
model-loading.md §3.2 updated. **The golden's conditioning was the real work.**
A random tiny model has low-entropy stretches with sub-1e-3 top-2 logit gaps;
there, HF `generate`, a manual KV loop, and our reference (mutually ~4e-6 apart
on logits) flip tokens independently — so matching "HF generate token-for-token"
is *ill-conditioned* on such prompts (the committed activation prompt is one:
HF's own generate and manual paths diverge at token 18 on it). Fix: a
tools-side search selects three prompts whose **minimum top-2 gap stays > 1e-2**
over all 40 steps (four orders above the reference error), so every
numerically-close path agrees; EOS is suppressed in the fixture so each reaches
full length regardless of a chance EOS, and the C++ golden runs with empty
`eos_ids`. New `tools/gen_fixtures/tiny_llama_generate.py`
(`tiny-llama-generate`) writes `tiny-llama/expected/generate.json` and
cross-checks HF `generate` == the manual KV loop before writing; registered in
`__main__.py` and `regen_fixtures.sh`; existing fixture bytes untouched. Tests
(+20 → 721 green): `generator_test.cpp` — token-for-token vs the golden (≥32
asserted, all 40 compared) built through the registry; two-runs-identical;
max-new-tokens stop (n=1,7 prefixes); EOS-on-first-token (returns exactly it) and
mid-sequence EOS (stops at first occurrence, inclusive); `on_token` fired once
per id in order with `cache.length() == prompt+i` at the i-th callback (proving
the append/forward ordering); continuation from a half-prefilled cache equals the
full-prompt golden (the KV invariant); error paths (empty prompt, `max_new=0`,
insufficient capacity → `ResourceExhausted` with nothing appended, out-of-range
prompt id propagated from `forward`); backend `Name`/`Parse` round-trip + unknown
→ `InvalidArgument`. Plus 7 `config_test.cpp` `eos_token_id` cases (absent/scalar/
list/null/wrong-type/list-element/out-of-range) and list-vs-scalar assertions on
the real Llama-3.1 and Qwen2 config goldens. Label `engine` (`ctest -L engine`);
no `SCALAR_PASS` (portable C++, calls the interfaces not `kernels::`). Format +
scoped tidy clean (generator.{cpp} + test, config.{cpp} + test, and the
`model/config.h` direct-includer TUs).

M5-T10 done (2026-08-18: Qwen-family support). **Zero `src/` change** — the
milestone's promise was that Qwen support would be config/wiring, and it held: the
config parser (`Qwen2ForCausalLM` → `kQwen2`, `attention_bias` per-arch default,
explicit `head_dim`), weight map (q/k/v `.bias` rows, o_proj bias-free, tied
`lm_head`→`embed_tokens` alias), `ReferenceModel::Create` (per-projection
`BuildLinear(has_bias=config.attention_bias)`), `Attention::Create` (validates
`H·d` without assuming `E==H·d`), and the registry (`Qwen2ForCausalLM` routed to
the shared `BuildReferenceFamily`) had all landed across M5-T02…T08. So T10 is
**a new model fixture, its greedy-generation golden, and one end-to-end test
suite**. New `tools/gen_fixtures/tiny_qwen2.py` (subcommand `tiny-qwen2`): a
random-weight 2-layer `Qwen2ForCausalLM` under the pinned toolchain
(transformers 5.14.1 / torch 2.13.0), deliberately distinct from tiny-llama on
every Qwen-relevant axis so the test can't pass vacuously — **`head_dim=24` ≠
`hidden_size/num_heads` (64/4=16)** so q_proj is `[96,64]`, kv `[48,64]`, o
`[64,96]` (the decoupled path design §3.1 flags but tiny-llama never covers);
**`attention_bias` omitted from config.json** so the parser's per-arch default is
what supplies the biases end-to-end; **tied embeddings** — `tie_word_embeddings:
true`, and because safetensors refuses to serialize the shared `lm_head`/`embed`
storage, the checkpoint **omits `lm_head.weight`** exactly like a real Qwen2-0.5B
export and the loader's tied-alias path reconstitutes it; Qwen2 defaults
`rope_theta=1e6`, `rms_norm_eps=1e-6`, no BOS, single-int `eos_token_id`. **The
non-obvious fixture decision:** HF `_init_weights` **zero-inits every Linear
bias**, so a freshly built Qwen2 has all-zero q/k/v biases — which would make the
bias path invisible (dropping the biases wouldn't move a logit, and the golden
wouldn't actually test bias addition). `_build_model` fills the q/k/v biases with
fixed small-magnitude noise (`randn * 0.5`, a dedicated seeded generator so it's
independent of the model-init RNG) so the golden genuinely exercises bias add and
the load-bearing test is meaningful; no sharded copy (an M4 concern tiny-llama
covers). New `tiny_qwen2_generate.py` (`tiny-qwen2-generate`) writes
`expected/generate.json`: three prompts (no BOS) 40-token greedy continuations,
HF `generate(do_sample=False)` cross-checked against a manual `DynamicCache`
prefill+decode loop (lowest-index argmax) before writing, EOS suppressed. Prompts
were chosen (offline search over random candidates) for a **well-separated**
trajectory — and unlike T09, the minimum top-2 logit gap (>1e-2, four orders above
the ~4e-6 reference error) is now **asserted** in the generator, not merely
printed; the committed prompts observe ≥0.10. Both subcommands registered in
`__main__.py` and `regen_fixtures.sh`; `--verify` confirms every committed fixture
(the new tiny-qwen2 tree included) regenerates byte-identically. New
`tests/unit/qwen2_family_test.cpp` (+8 tests → **729 green**, label `model`):
(1) config carries the Qwen2 defaults (arch `kQwen2`, `attention_bias` true by
default, `head_dim==24≠16`, tied, θ=1e6, eps=1e-6, `eos_token_ids=={3}`);
(2) loader binds q/k/v biases with a bias-free o_proj and aliases the tied lm_head
onto the embed handle (same `.data()`); (3) the registry builds `Qwen2ForCausalLM`
with `cache_geometry().head_dim==24` and is **bit-identical** to a direct
`ReferenceModel::Create`; (4) **end-to-end logits vs the golden** (max_abs_diff
**3.9e-6**, tol 2e-4); (5) **biases are load-bearing** — rebuilding with
`attention_bias=false` moves the logits ~0.98 out of tolerance, proving the bias
path is exercised and not silently skipped; (6) greedy continuation matches
`generate.json` token-for-token (40 ids, ≥32 asserted), (7) is deterministic
across two runs, and (8) the KV invariant holds on the Qwen path (a split
prefill/continue equals a single full run). Caught in review-of-own-work: the
golden-case loader first hit the **C++20 dangling-temporary range-for** bug
(iterating `GenerateGoldenRoot().at("cases")` over a destroyed temporary — the
same footgun CLAUDE.md records being fixed in the tokenizer tests) → fixed by
binding the root to a local. `registry_test.cpp`'s synthetic
`Qwen2ArchStringRoutesToFamilyBuilder` (tiny-llama relabelled, bias forced off)
kept as the isolated routing check, its stale "M5-T10 swaps in the real Qwen
fixture" comment pointed at the new suite. Format + scoped tidy clean
(`qwen2_family_test.cpp`, `registry_test.cpp`).

## Milestone 6 — Optimized CPU Execution Engine

M6-T01 done (2026-08-18: `docs/design/optimized-cpu-execution.md` — the working
contract M6-T02…T08 optimized kernels/backend build on, plus the layout/policy
seam M8/M9/M12/M13 inherit). **Docs-only.** The doc fixes six decisions the M5
docs deliberately deferred here:

1. **Packed weight tile (§3).** Checkpoint `W[N=out, K=in]` stays the source of
   truth (cpu-backend §7 constraint); the packed form is derived at load into
   **K-major panels of `NR = 16` output rows**, shape `[ceil(N/NR), K, NR]`, held
   in the checkpoint dtype (bf16/f16/f32, widened in-register in the
   micro-kernel). `NR` is **fixed across ISAs** (4×`float32x4`/2×`__m256`/scalar
   16-array — same rationale as cpu-backend §6.3's 16-lane Class-R convention) so
   the forced-scalar pass validates the *same packed bytes* that ship. Worked
   example (N=5,K=3,NR=4 illustrative + tiny-llama/qwen2/1B real shapes),
   micro-kernel (`MR×NR` fp32 register accumulators, `MR` per-ISA, one packed
   layout serves GEMM/skinny/GEMV), cache blocking that changes traversal order
   only (bit-identical). Acceptance bullet met.
2. **Workspace strategy (§6).** `OptimizedModel` owns one `Workspace` (model-level
   residual-stream + projection + MLP buffers, reused across layers; per-worker
   attention scratch). A stated **sizing formula** in bytes as a function of
   (T,E,H,Hkv,d,I,nthr), instantiated for tiny-llama (~26 KB) and Llama-3.2-1B
   (~61 MB prefill / ~120 KB decode). Acceptance bullet met. Policy:
   **grow-on-demand with a high-water mark, never shrink** → steady-state decode
   allocation-free (M12-T05's target, reached now without a config knob; the
   `max_forward_tokens` bound deferred to M9 when the scheduler supplies it).
3. **Logits lifetime pinned:** freshly-allocated **caller-owned** for *both*
   backends (retiring model-execution §5.2's "may return a workspace view"
   latitude) — one lifetime contract; the one per-step alloc M12-T05 may later
   remove via an optional `ForwardRequest` output buffer.
4. **Parallel implementations, not reuse of the M5 layer classes** (§2.2,
   resolving model-execution §8's open call): the optimized graph reuses only the
   `Linear` interface (`PackedLinear` slots into the existing
   `unique_ptr<Linear>`) and `Rope::Create`'s tables (config interpretation, not
   compute); everything else is new so it runs out of the `Workspace` and stays
   fusion-ready (M12-T04) without optimizing the oracle.
5. **Tied embeddings → one physical copy** (§7, resolving model-execution §4.3's
   "resolved in the design doc"): the packed lm_head is authoritative and the
   lookup gathers logical row `v` from panel `v/NR`, lane `v%NR` — no `[V,E]`
   duplicate (which would cost ~272 MB on a tied Qwen2.5-0.5B). One embedding
   kernel, two source layouts (packed-strided tied / contiguous untied).
6. **Kernel-validation tolerance table (§10, the acceptance-critical
   "bitwise-vs-tolerance and why"):** every M6 kernel **bit-identical across
   thread counts** (no reduction split across threads outside `parallel_reduce`);
   **bit-identical across ISAs only** for the two pure-map ops (residual add
   Class E; embedding gather-and-widen); **Class T (stated tolerance) across ISAs
   and vs the oracle** for everything with a multiply-accumulate (GEMM, norms,
   softmax, RoPE, attention) — FMA contraction, horizontal-reduction order, and
   vector transcendentals are the three rounding sources. Stated tolerance
   recipes (GEMM atol scaled by √K; RMSNorm forbids raw `rsqrte`; the vector
   `expf` polynomial — the one new numerical algorithm — gets its own ≤2-ulp
   sweep vs `std::expf`). Recorded+rejected alternative: cross-ISA bit-identity
   via FMA-everywhere (slow libm fallback on non-FMA x86; revisit as an M12
   tightening with numbers).

Supporting decisions in the doc: `model → kernels` is a layer-2→layer-1 **downward
edge, already allowed — no ADR amendment** (only intra-layer edges need one, like
Amendment 5's `model → kvcache`), CMake link PRIVATE; a determinism-safe
**worker-index `parallel_for` overload** (`void(int worker, begin, end)`) as the
per-thread-scratch mechanism (lands with M6-T04; the index selects scratch, never
chunk assignment); optimized-kernel test suites register with `SCALAR_PASS`
(first milestone where it covers a full model forward); the `cache.view` gather
cost quantified (~12% of decode memory traffic on 1B@4k) and marked as exactly
what M8-T05 removes. Cross-doc edits landed in the same change:
model-execution.md §4.3/§5.2/§8/§14 and cpu-backend.md §3.2/§6.3/§7 all point
here with dated notes (never a silent divergence). ROADMAP M6-T01 ticked; CLAUDE.md
status + "Next up: M6-T02" refreshed. No `src/` change, no test-count change
(729 green); no build/tidy impact (docs-only).

M6-T02 done (2026-08-18: packed-weight GEMM & GEMV + `PackedLinear`). The first
optimized compute kernels, validated against the `cpu::gemm` oracle. What landed:

- **Packed layout & pack routine** (`src/kernels/gemm.{h,cpp}`): `kNr = 16`
  (fixed across ISAs), `PackedPanels`/`PackedWeightElements`, and
  `PackWeightPanels` — a pure gather + `+0.0` zero-pad of the checkpoint weight
  `W[N,K]` into K-major panels `Wp[P=ceil(N/16), K, 16]`, dtype-preserving
  (bf16/f16 via a `std::uint16_t` overload, f32 via a `float` overload),
  threaded over panels (result thread-independent). Verified layout-exact
  (`Wp[p,k,r] == W[p·16+r,k]`, pad = 0) in tests.
- **Micro-kernels** (`{scalar,neon,avx2}/gemm.cpp` behind the M3-T05 dispatch,
  `internal/gemm_impl.h` decls + `detail::*TileVariant` test seams,
  `internal/gemm_common.h` shared `StorePanelBlock` for bias + pad-column
  masking): the ISA seam is a **tile** — rows `[m0,m1)` × panels `[p0,p1)`, all
  K, single-threaded chunk body. Register-tiled `MR×NR` fp32 accumulators
  (`MR = 4` NEON via `vfmaq_n_f32` + `vshll`/`vcvt_f32_f16` widen; `MR = 6`
  AVX2 via `_mm256_fmadd_ps` + F16C/`cvtepu16`+shift widen), full-K held in
  registers (no `y` reload — simpler than the doc's sketched `kKc` reload, same
  ascending-`k` single accumulator; **design §3.4 amended in place**). Scalar is
  bit-identical to a naive triple loop (same accumulator, order, `half.h`
  widen); NEON/AVX2 are Class T (FMA contraction).
- **Dispatch & threading** (`gemm.cpp`): three per-dtype `KernelTable`s, a
  `TileRunner` closing over `wp`'s dtype so the loops are written once.
  `PackedGemm` tiles the `(m-block=kMc=64, panel-block=kNc=8)` grid across
  threads (grain 1); routes `M == 1` to `PackedGemv`, which threads panel chunks
  (`kGemvPanelGrain = 8`). One layout serves prefill/skinny/decode. Raw-pointer
  entries CHECK preconditions; recoverable validation is `PackedLinear`'s job so
  the parallel region does only arithmetic.
- **`PackedLinear`** (`src/model/packed_linear.{h,cpp}`, a third `Linear`
  alongside `ReferenceLinear`/future `QuantizedLinear`): `Create` validates
  weight/bias exactly as `ReferenceLinear` (same messages), repacks the weight,
  converts bias → fp32 once (§3.5), and **drops the checkpoint handle** (packed
  copy authoritative). `forward` front-loads all shape/dtype/contiguity checks
  then calls `PackedGemm`. Header stays kernels-free (`packed_weight()` exposes
  only a `tensor::Tensor`), so the **`model → kernels` edge is PRIVATE** (a
  downward layer-2→layer-1 edge, no ADR amendment — design §2.1). CMake:
  `engine::kernels` PRIVATE on `engine_model`.
- **Tests** (`+30`, → 759 green). `packed_gemm_test` (kernels label, SCALAR_PASS):
  pack layout; PackedGemm/GEMV vs `cpu::gemm` across 12 model/1B shapes ×
  {f32,f16,bf16} × {bias,no-bias} within `atol = 1e-5·√K, rtol = 1e-4` (scalar
  exact); tiling/threading bit-identity vs the un-tiled variant (rtol=atol=0);
  GEMV row == GEMM row (decode==prefill); the vector-slot vacuity guard; the 10
  `ops.safetensors` goldens replayed. `packed_linear_test` (**model label,
  SCALAR_PASS — first model suite in the forced-scalar pass**): Create/features/
  packed shape, malformed-input rejection, forward vs `ReferenceLinear` across
  dtypes/shapes/bias + real tiny-llama bf16 checkpoint, decode shape, bad
  activation rejection.
- **Benchmark** (`benchmarks/kernels/gemm_bench.cpp`, `gemm_bench` target;
  optional `-DENGINE_BENCH_BLAS`, benchmark-only, never in `src/`): at 4096³ on
  the M2, **8.76× bf16 / 8.48× f32** the naive `cpu::gemm`, ~112 GFLOP/s —
  clears the ≥5× advisory (BASELINES.md). The Accelerate context number could
  not be built on the dev machine (Homebrew LLVM 20 vs the CommandLineTools SDK
  Accelerate headers → `-Welaborated-enum-base`, the broken-CLT situation); the
  BLAS wiring is in place for a capable host and the number is context-only, so
  it is deferred, not blocking (recorded in BASELINES.md + design §10).

Non-obvious decisions: (1) the variant boundary is a whole tile, not a leaf
micro-op — keeps the register-tiled k-loop and MR-blocking per-ISA while the
tile grid, dtype dispatch, and pad-masked store are written once. (2) Bit-identity
across thread counts is tested as `PackedGemm == un-tiled variant` (the single
process can't vary `DefaultPool` size), the same technique `cpu::gemm`'s
tiling-invariance test uses. (3) `PackedLinear::Create` takes the weight by
`const&` (it repacks, never retains — the honest signature for a non-retaining
builder, unlike `ReferenceLinear` which moves-to-store). `regen_fixtures.sh` not
run (no fixture change). Format clean; scoped tidy clean on the six new/edited
TUs (AVX2 reviewed by hand — not in the arm64 compile DB).

---

M6-T03 done (2026-08-18: vectorized norm, activation, softmax & RoPE kernels).
NEON/AVX2 (behind the M3-T05 dispatch) + always-present scalar for RMSNorm,
SiLU-and-mul, numerically-stable softmax, and RoPE-apply — each validated
against its `cpu::` oracle within a stated tolerance, on the host ISA and the
forced-scalar pass. What landed:

- **Vector `expf` polynomial — the one new numerical algorithm (design §10).**
  `src/kernels/internal/exp_common.h` is the shared scalar spec (Cephes/
  `avx_mathfun` lineage: two-part Cody-Waite `ln2` range reduction, degree-5
  minimax for `eʳ`, `2ⁿ` by exponent-field write); `neon_exp.h` / `avx2_exp.h`
  are the lane helpers mirroring it, included **only** from per-ISA TUs (the
  intrinsic-header rule). Domain contract: ≤2 ulp on `[-87.34, 88.38]`
  (**observed max 1 ulp**, scalar and NEON); `x < kExpLo` (incl. `-inf`) →
  exactly `+0.0` — `n` never drops below −126 so the pow2 trick can't break, and
  this makes softmax's `-inf → 0` mask contract exact; `x > kExpHi` → finite
  saturation (immaterial to callers). Its dedicated sweep (`vector_exp_test`,
  2M-point) is **independent of the kernels that use it** (design §10).
- **All three ISAs run one exp algorithm** (scalar embeds the polynomial too,
  not `std::exp`): the forced-scalar pass then actually exercises the shipped
  numerical code, and the only scalar-vs-vector difference is FMA/lane order.
- **RMSNorm** (`norm.{h,cpp}` + per-ISA `RmsNormRows`): fp32 weight (the
  optimized backend pre-converts norm scales, §4 — no per-element widen on the
  hot path), 16-lane sum-of-squares, an **exact** scalar `1/sqrtf` (design §10
  forbids a raw `rsqrte`). **Softmax** (`softmax.{h,cpp}`): row max + exp + sum
  + normalize, the exp via the polynomial. **SiLU-and-mul** (`activation.{h,cpp}`):
  `silu(g)·up` with **exact division** (never a reciprocal estimate, §10).
  **RoPE** (`rope.{h,cpp}`): in-place HF half-rotation, arbitrary per-token
  positions (unsorted/repeated → batched/paged-ready, §8).
- **Residual add is the pre-existing `kernels::AddF32`** (Class E, M3-T06) — design
  §10 names its oracle as `cpu::add`, so no new kernel; a bit-equality parity
  test in `activation_kernel_test` closes the ticket's residual-add box.
- **`exp` ships no public kernel** — softmax/SiLU embed the lane helpers. A small
  `exp.cpp` + `internal/exp_impl.h` expose array-form variants **only** for the
  ulp sweep's `detail::ExpF32Variant` seam.

Threading (design §5): row-parallel (norm/softmax), token-parallel (RoPE), flat
(SiLU), all grain-`1`/element-chunked, no reduction split across threads — so
every kernel is **bit-identical across thread counts**, tested by comparing the
threaded public entry against a single serial variant call over the whole
problem (the M6-T02 idiom), plus, for softmax, an explicit arbitrary
row-chunking. Observed vs-oracle max-abs-diff (NEON, thresholds set above):
softmax 6.0e-7 (`atol 1e-6, rtol 1e-4`), RMSNorm 2.4e-6 (`rtol 1e-5, atol 1e-6`),
SiLU 9.5e-7 (`rtol 1e-5, atol 1e-6`), RoPE 2.4e-7 (`rtol 1e-5, atol 1e-6`) —
across hidden sizes {odd, 1024, 4096}, large-magnitude/causal-masked softmax,
RoPE positions {0, 1, large}/unsorted/repeated and head-dims {24, 64, 128} with
GQA head counts. Each kernel also replays the M5 HF-derived goldens
(`ops.safetensors` rmsnorm/silu_mul/softmax, `rope.safetensors` apply) through
the optimized path, tying it to the HF chain, not just the reference.

Bench (`kernels_bench`, BASELINES.md M6-T03): NEON vs scalar single-threaded on
the M2 — RMSNorm **2.10×**, Softmax **3.08×** (the row-reduction wins);
ExpF32/SiLU ~1.05× (clang auto-vectorizes the branch-light scalar polynomial and
both are streaming-bound at 1M elements); RoPE **0.70×** — the vector body is
currently *slower* than the auto-vectorized scalar loop, recorded honestly
(correctness unaffected), a tuning item for M12-T02, not an M6 obligation. No
perf *target* for T03 (the design's only hard GEMM target is M6-T02's 5×); the
delta is recorded because the kernels are performance-motivated (CLAUDE.md).

Non-obvious decisions: (1) scalar variants use the polynomial exp, not
`std::exp`, so the forced-scalar pass covers the shipped numerical code and
cross-ISA divergence is FMA-only. (2) `-inf → 0` is delivered by the exp flush,
not a special case in softmax — one contract, one test surface. (3) RoPE takes a
token offset so the threaded chunks pass `positions + begin`, keeping positions
and `x` aligned per chunk. AVX2 TUs written blind (no x86-64 dev machine),
mirroring the NEON structure 1:1, reviewed by hand against `.clang-format`/
`.clang-tidy`; CI's x86-64 build (warnings-as-errors) + the exp sweep + all
suites are the AVX2 proof (the accepted arm64-CI gap, CLAUDE.md). `regen_fixtures.sh`
not run (no fixture change). +64 test registrations (32 new cases × SCALAR_PASS;
6 forced-scalar slot-guards `GTEST_SKIP`) → **823 green**. Format clean; scoped
tidy clean on the new/edited TUs.

M6-T04 done (2026-08-18: optimized prefill attention). Blocked, flash-style
causal GQA attention with an online (running max/sum) softmax, replacing the M5
reference's materialize-`[H·T, L]`-then-`cpu::softmax` math while honoring the
exact `cpu::attention` op contract (q `[T,H,d]`, k/v `[Hkv,L,d]` head-major, L =
P+T, scale on the completed dot, out `[T,H,d]`, GQA by `h/g` KV-head indexing).
`src/kernels/attention.{h,cpp}` — public `PrefillAttentionF32(q,k,v,out, T,H,Hkv,
d,L, scale)`, `KernelTable` dispatch (scalar/neon/avx2), `parallel_for` over
`H·ceil(T/kAttnQb)` `(head, query-block)` units at grain 1, `detail::
PrefillAttentionVariant` test seam. `src/kernels/internal/attention_common.h` —
the online-softmax recurrence written **once** as `PrefillUnitsImpl<Ops>` (the
GEMM-idiom split: shared control flow, ISA-only arithmetic), block constants
`kAttnQb=32`/`kAttnKb=64`, the `PrefillArgs` struct. `internal/attention_impl.h`
— per-ISA `PrefillUnits` decls + the variant seam. `scalar/neon/avx2/
attention.cpp` — each an `Ops` policy of four primitives (dot+score, exp+sum,
scale-row, axpy-row); NEON/AVX2 vectorize over `d` + the key row via the shared
`neon::Exp`/`avx2::Exp` polynomial, scalar embeds `ExpF32Scalar` so forced-scalar
runs the shipped numeric code.

Non-obvious decisions: (1) **No per-worker scratch, and §6.4's worker-index
`parallel_for` overload was NOT added** — the online accumulator IS `out`
itself (each unit's output rows are disjoint, rescaled in place, divided by the
denominator at the end), and the only working memory is a fixed `kAttnKb` stack
score row + two scalars per query. This diverges from the M6-T01 draft (which
planned per-worker attention scratch + the overload); rather than ship unused
plumbing, `src/parallel/` was left untouched and the design updated in the same
change — `optimized-cpu-execution.md` §6.1/§6.2/§6.4 (`W_worker = 0`) and
`cpu-backend.md` §3.2 now record the deferral, with the overload's ready design
retained for the first kernel that genuinely needs cross-`d` scratch (an M12
fusion / a non-`parallel_reduce` decode split). (2) **Causal masking is a per-row
`n_valid`, never a written `−inf`:** key blocks wholly past `limit = P+t` are
never visited (the causal skip), the diagonal block iterates only its valid keys,
and masked keys contribute exactly 0 — reproducing the reference's `−inf → 0`
softmax contract without a mask value. (3) **Row-at-a-time**, so `kAttnQb`
controls only K/V block reuse, not a `[kQb,kKb]` score tile (kept the score
buffer a `kAttnKb` stack array). (4) The five `attention.safetensors` HF goldens
are replayed through the kernel (ties the optimized path to the HF chain, not
just the reference).

Numerics (design §10): fp32 throughout; **bit-identical across thread counts**
(each query's recurrence wholly in one variant call — asserted vs a single serial
variant call and an arbitrary manual chunking, `EXPECT_EQ` bitwise); Class T
across ISAs and vs the oracle (online rescale order + vector exp). Observed max
vs `cpu::attention` (NEON): **1.3e-6** across the acceptance sweep
(T∈{1,17,512,2048}, P∈{0,5}, the decode-shaped T=1/P=2047, GQA
{(4,4),(4,2),(8,1)}×d∈{18,24,64,128}, and the `kAttnQb`/`kAttnKb` block-boundary
straddles), well inside the stated `rtol 1e-4, atol 1e-5`. The large-logit stress
uses q/k std 2 (logit std ≈4, observed ~1.4e-5, still in tolerance); pushing to
logit std in the tens — no trained model's regime — is where the online rescale's
incremental rounding exceeds the threshold where `|ref|` is small, recorded
honestly rather than papered over.

Bench (`attention_bench`, BASELINES.md M6-T04): 2k-context prefill
(T=L=2048, H=32, Hkv=8, d=64, Llama-3.2-1B-shaped) vs the reference on the M2 —
**9.72×** at 8-thread NEON (2.121 → 0.218 s), 12.89× single-thread NEON, 6.68×
single-thread scalar. The reference is memory/materialization-bound (its scalar
and NEON single-thread times are ~equal, dominated by the 537 MB score-matrix
traffic), so most of the win is the algorithm (no score-matrix write) plus the
`d`-vectorized dot/axpy. No perf *target* for T04 ("time vs the reference"); the
delta is recorded per CLAUDE.md.

AVX2 TU written blind (no x86-64 dev machine), mirroring the NEON structure 1:1,
reviewed by hand against `.clang-format`/`.clang-tidy` (CI's x86-64
warnings-as-errors build + all suites are its proof — the accepted arm64-CI gap).
`regen_fixtures.sh` not run (no fixture change). `src/parallel/` untouched.
+14 test registrations (7 new cases × SCALAR_PASS; 1 forced-scalar slot-guard
`GTEST_SKIP`) → **837 green**. Format clean; scoped tidy clean on the
new/edited TUs.

M6-T05 done (2026-08-18: optimized decode attention). The single-token
specialization of the M6-T04 prefill kernel: one query per head attends the
whole cache, threaded across kv heads with the `g = H/Hkv` query heads of a
group processed together so that kv head's K/V stream from memory once. Honors
the exact `cpu::attention` op contract at `T = 1` (q/out `[H, d]` = the T=1 slice
of `[T, H, d]`; k/v `[Hkv, L, d]` head-major; scale on the completed dot; GQA by
`h/g` KV-head indexing). `src/kernels/attention.{h,cpp}` — public
`DecodeAttentionF32(q,k,v,out, H,Hkv,d,L, scale)`, `KernelTable` dispatch
(scalar/neon/avx2), `parallel_for` over `Hkv` kv-head units at
`kAttnHeadGrain = 1`, `detail::DecodeAttentionVariant` test seam.
`internal/attention_common.h` — `DecodeUnitsImpl<Ops>` (a second control-flow
template alongside `PrefillUnitsImpl`, driving the **same four Ops primitives** —
zero new ISA arithmetic) + `DecodeGroupSlice<Ops>` (the per-slice body, split out
to stay under the tidy cognitive-complexity threshold — the gemm.cpp
`ComputePanel` idiom), `DecodeArgs`, the `kAttnDecodeGroupChunk = 8` constant.
`internal/attention_impl.h` — per-ISA `DecodeUnits` decls + the variant seam.
`scalar/neon/avx2/attention.cpp` — one `DecodeUnits` wrapper each (a one-line
`DecodeUnitsImpl<…Ops>` instantiation; the AVX2 one written blind).

Non-obvious decisions: (1) **Bit-identity with `PrefillAttentionF32(T=1)` by
construction, asserted bitwise** (not tolerance): the recurrence is
key-block-outer / query-inner, but for each *fixed* query head the block
sequence, the per-block `n_valid`, the first-block `alpha = exp(−inf) = 0`, and
the four Ops calls are the identical arithmetic in the identical order —
interleaving the blocks across the group's other query heads never touches this
head's accumulator (fp32 ops on disjoint `out` rows). This is stronger than the
ROADMAP's "matches the prefill path's result" (which only asked for tolerance)
and is exactly the oracle M8-T05's paged decode kernel ("matches the M6-T05
contiguous decode kernel results exactly") will be validated against. (2)
**Key-block-outer / query-inner** is what makes "stream K/V once per kv head"
real: each `kAttnKb` block is loaded once and reused across all `g` query heads
while cache-resident (vs prefill(T=1) re-reading K/V per query head). (3)
**No per-worker scratch, no allocation** (§6.1 carried over): `out` is the
accumulator; the only working memory is a `kAttnKb` stack score row + fixed
`kAttnDecodeGroupChunk` per-query `run_max`/`run_den` arrays. A group larger than
the chunk (no real model — g ≤ 7 for Llama-3.2-1B / Qwen2.5) is sliced,
re-streaming K/V once per slice; the `g = 12 > chunk` test exercises the slicing.
(4) **Parallel width is `Hkv`** (decode threads over kv heads only, design §8) —
a model with `Hkv < cores` (Qwen2.5-0.5B Hkv=2) leaves cores idle on decode, the
concrete M12-T03 flash-decoding motivator; recorded, not hidden.

Numerics (design §10): fp32 throughout; **bit-identical across thread counts**
(each kv head's whole recurrence within one variant call — asserted vs a single
serial call + arbitrary manual chunking, `EXPECT_EQ` bitwise); **bit-identical to
prefill(T=1)** (max-abs-diff 0); Class T across ISAs and vs the oracle. Observed
max vs `cpu::attention` (NEON): **8.6e-6** across cache lengths
{1, 63, 64, 65, 127, 128, 129, 2048}, GQA {(4,4),(4,2),(8,1)} + the `g=12` slice
case, d∈{18,24,64,128}, and the large-logit stress (q/k std 2, logit std ≈4) —
well inside `rtol 1e-4, atol 1e-5`. The five `attention.safetensors` HF goldens
are replayed through the decode kernel: `l0_decode` directly, and the **last ctx
row of each prefill golden** (causality makes that row the decode of that final
token) — tying the decode path to the HF chain, not just the reference
(max-abs-diff ≤1.2e-7).

Bench (`attention_bench decode`, BASELINES.md M6-T05): 2k-context decode step
(L=2048, H=32, Hkv=8, d=64, Llama-3.2-1B-shaped) on the M2 — **5.25×** the M5
reference at 8-thread NEON (1407 → 268 µs/call), 7.15× single-thread NEON, 3.72×
single-thread scalar. Decode is marginally faster than prefill(T=1) (268 vs
275 µs at 8-thread) — both hold one score row at T=1, so the only edge is decode
streaming K/V once per kv head vs prefill(T=1)'s `g×` re-reads; at 2k context
both are bandwidth-bound on the shared K/V, so the reuse win is ~2% and grows
with `g`. The decode kernel's value is the allocation-free single-token path
M6-T07 calls per generated token, not this margin. No perf *target* for T05
("time vs the reference"); the delta is recorded per CLAUDE.md.

AVX2 TU written blind (a one-line `DecodeUnitsImpl<Avx2Ops>` instantiation — no
new intrinsics, since the four Ops primitives are shared with prefill), proven by
CI's x86-64 warnings-as-errors build + all suites (the accepted arm64-CI gap).
`regen_fixtures.sh` not run (no fixture change). `src/parallel/` untouched.
+12 test registrations (6 new cases × SCALAR_PASS; 1 forced-scalar slot-guard
`GTEST_SKIP`) → **849 green**. Format clean; scoped tidy clean on the new/edited
TUs (the `model → kernels`/attention includers set).

### M6-T06 — embedding & logits path (2026-08-18)

The two ends of the optimized model: the embedding-lookup gather (ids → fp32
rows, both source layouts) and the lm_head logits projection (already
`PackedLinear::forward` from T02, so T06 adds no logits kernel — only its
fixture-model tests). Tied embeddings honored with **one physical copy**
(design §7): the packed lm_head is authoritative, and the lookup gathers logical
row `v` out of its `[ceil(V/kNr), E, kNr]` layout — no ~272 MB `[V, E]`
duplicate on a tied Qwen2.5-0.5B.

**Kernels (`src/kernels/embedding.{h,cpp}`).** `EmbeddingLookupF32` (row-major
`[V, E]` source, untied) and `EmbeddingLookupPackedF32` (packed lm_head source,
tied). A pure gather + exact fp16/bf16→fp32 widen — **bit-identical** across
thread counts, across ISAs, and vs the `cpu::embedding_lookup` oracle (the §10
"embedding lookup" row; one of only two pure-map kernels). Both thread over
tokens (`kRowGrain = 1`, each output row written by one worker); the row-major
path widens a whole row in one call, the packed path gathers each row's
`kNr`-strided 16-bit lanes into a fixed 512-wide stack buffer a chunk at a time
then widens the chunk (widen is Class E → chunk-invariant → bit-exact). f32
tables skip the widen (`memcpy`/strided copy).

**Non-obvious decision — no per-ISA embedding TU** (amends the design §2 file
table, updated in the same change). The gather has no ISA-specific control flow;
its only per-element arithmetic is the widen, which the M3-T06 `convert`
variants (`scalar/neon/avx2::{Bf16,Fp16}ToFp32`) already provide bit-exact per
`half.h` including NaN payloads. So `embedding.cpp` builds a
`KernelTable<WidenFn>` from those variants and `Select`s once — `ENGINE_FORCE_ISA`
still selects the widen path, the forced-scalar pass still exercises the scalar
widen, and a blind AVX2 embedding TU (a pure wrapper) is avoided. `model →
kernels` stays a PRIVATE link (the packed bytes are a plain `tensor::Tensor`).

**Module (`src/model/optimized_embedding.{h,cpp}`).** `OptimizedEmbedding`
mirrors the M5 `Embedding` surface (so M6-T07's `OptimizedModel` swaps it in
1:1): `FromTable` (untied — validation identical to `Embedding::Create`, table
retained zero-copy) and `FromPackedLinear` (tied — holds the lm_head's
`packed_weight()` **by value**; the refcounted `shared_ptr<Buffer>` keeps the
one physical copy alive independently of the `PackedLinear`, so the linear may
then be moved into a `unique_ptr<Linear>` without dangling — tested). `forward`
front-loads the `y` shape/dtype checks and the `[0, V)` id pre-scan (naming the
offending index+value exactly as the oracle) so a bad id never reaches the raw
gather, then dispatches the packed or row-major kernel. The lm_head logits path
is unchanged `PackedLinear::forward` — GEMM shape (kAll) or GEMV shape (kLast).

**Tests** (+13 gtest cases; ×2 for SCALAR_PASS on the ctest side → 875 ctest
entries, all green — was 849 gtest = 862 now). `embedding_kernel_test` (kernels
label, SCALAR_PASS): both layouts vs a single-threaded reference gather across
dtypes {f32,f16,bf16} × shapes (E > the 512 chunk and non-multiple, V with and
without a padded last panel) × id sets (random+repeats, edges {0,V−1}); packed
zero-pad-lane correctness at V−1; NaN/inf/subnormal bit patterns widen bit-exact
vs the oracle (raw-bit compare); row-major/packed vs `cpu::embedding_lookup` on
the real tiny-llama bf16 table; single-token (decode-shaped) path.
`embedding_logits_test` (model label, SCALAR_PASS): `OptimizedEmbedding`
create/forward validation mirrors the reference; untied (tiny-llama) vs the
reference `Embedding` bit-exact; tied (tiny-qwen2) shares the packed lm_head
storage (`source().data() == lm_head.packed_weight().data()`) and matches the
reference bit-exact, including after the linear is moved behind a
`unique_ptr<Linear>`; the design §7 shared-storage assertion (gathered row `v`
== checkpoint row `v` widened, sampled over the padded-panel edge); the logits
path (`PackedLinear` GEMM + last-row GEMV) vs `ReferenceLinear` (`rtol/atol
1e-4`, the T02 band) with GEMV == GEMM last row bit-identical, and vs the HF
`logits` golden (`rtol 1e-4, atol 2e-4`) on both fixtures.

Format clean; scoped tidy clean on the four new TUs (no existing header edited).
No fixture change (existing goldens sufficed). `src/parallel/` untouched.

### M6-T07 — optimized model forward & generation (2026-08-18)

**Ticket.** Wire the M6 kernels into the M5 `Model` interface as the
`kOptimized` backend: weight repacking at load (progress-logged), workspace
allocation, full prefill + decode forward, the M5 greedy loop reused verbatim.
Acceptance: tiny-fixture greedy generation token-for-token identical to the
reference backend + logits within tolerance; a real ~1B model loads and
generates coherent text on the dev machine.

**Workspace (`src/model/workspace.{h,cpp}`).** The reused-across-layers scratch
the optimized forward runs out of (design §6): ten model-level fp32 slots — the
four `c_stream` E-width buffers `x`/`h`/`tmp`/`r` (residual stream / norm output
/ projection output / post-attention residual), the projections
`q [T,H·d]`/`k`/`v [T,Hkv·d]`/`ctx [T,H·d]`, and the MLP `gate`/`up [T,I]` —
each a separate contiguous tensor exposed as a `[T,width]` prefix view.
`EnsureCapacity(T)` is monotone grow-on-demand with a high-water mark: it
allocates all ten (uninitialized `Tensor::empty` — every slot is written before
read) into locals and commits only when all succeed, so a mid-way OOM leaves the
prior buffers intact and surfaces `ResourceExhausted`/OOM **before any kernel
runs or the cache is touched** (front-loaded, ADR-003). Steady-state decode is
therefore allocation-free once the first `T=1` call has sized it. `bytes()`
computes the §6.2 formula (`4·[4·T·E + T·(H+2·Hkv)·d + T·H·d + 2·T·I]`) and is
asserted against the instantiated tiny-llama figure (25.6 KB at T=8) and its
monotone growth. `W_worker = 0` as designed (the attention accumulator is `out`,
§6.1). Per-slot tensors chosen over one arena + offsets — same guarantees,
simpler.

**OptimizedModel (`src/model/optimized_model.{h,cpp}`).** A second `Model`
behind the same interface as `ReferenceModel`, a **parallel-implementation
graph** (design §2.2 — not the reference `DecoderLayer` classes, so the oracle
stays untouched and the graph is fusion-ready). `Create` binds every module from
`LoadedModel.weights` by canonical name, identically to `ReferenceModel::Create`
but building optimized modules: a `PackedLinear` per projection (q/k/v/o,
gate/up/down, lm_head — repacked into the §3 K-major panels at construction,
bias→fp32 once, checkpoint handle dropped), norm scales converted to fp32 once at
build (§4 — the kernel takes a `const float*`, unlike the reference which widens
per call), and **one shared `Rope`** for the whole model (a deliberate divergence
from the reference's per-layer copies: the cos/sin tables are position-only, so
one suffices — on Qwen2.5-0.5B the per-layer tables would be ~200 MB duplicated).
Tied embeddings (§7) build the lm_head `PackedLinear` first, then
`OptimizedEmbedding::FromPackedLinear` shares its packed storage (one physical
copy, no `[V,E]` duplicate) before the linear is moved behind a
`unique_ptr<Linear>`; untied uses the model's own `[V,E]` table via `FromTable`.
Per-layer projection-shape validation is front-loaded (E not assumed `H·d`, so
Qwen's decoupled `head_dim` is honored), and repacking logs `packed layer i/N`.

`forward` copies `ReferenceModel::forward`'s validation block verbatim (same
order, same messages with the `OptimizedModel::` prefix — so the error-path tests
match 1:1), then `Workspace::EnsureCapacity(T)`, then embedding → N layers →
final norm → lm_head, all out of the workspace. Each layer (`ForwardLayer`) is
the pre-norm arrangement expressed with dispatched kernels:
`RmsNormF32`→packed q/k/v→`RopeApplyF32` (reading the shared table's `cos()`/
`sin()` directly, not via the reference's `Rope::apply`/`cpu::rope_apply`)→
`cache.append`→`cache.view`→`DecodeAttentionF32` (T==1) or `PrefillAttentionF32`
→packed o_proj→`AddF32` residual→`RmsNormF32`→packed gate/up→`SiluMulF32` (in
place)→packed down→`AddF32` back into the residual stream. The aliasing-friendly
kernels (RmsNorm/SiluMul/Add all permit `y` to alias an input) let the ten slots
be reused tightly. Logits are freshly allocated caller-owned (§6.3): `kLast`
projects only the last row (GEMV), `kAll` all rows (GEMM). The per-layer hook
emits the same `embeddings`/`layers.{i}`/`final_norm`/`logits` events as the
reference.

**Backend wiring (`src/model/registry.{cpp,h}`).** `BuildReferenceFamily` renamed
`BuildFamily`, dispatching on `options.backend` (`kOptimized` →
`OptimizedModel::Create`, else `ReferenceModel::Create`); the M5 `Unimplemented`
guard is gone. `registry.h` docs updated; the stale `registry_test`
"kOptimized is Unimplemented" case became "builds through the registry".

**Driver (`src/main.cpp`).** The placeholder `main` grew an `engine generate`
subcommand — the §10 real-model acceptance harness (M9 replaces it with the
server binary): `--model DIR --prompt STR [--backend reference|optimized]
[--max-new-tokens N] [--cache-capacity N] [--no-bos]`, loading through the
registry, tokenizing with `Tokenizer::from_file`, running `Generate` with a
`DetokenizerStream` streaming callback, and printing load/prefill/decode timing.
Links model/tokenizer/kvcache/engine (it sits at the server layer, above every
module).

**Validation.** `optimized_model_test` (+15 gtest cases, `model` label,
**SCALAR_PASS** — the first full-model forward under the forced-scalar pass; ×2
ctest entries → **905 ctest** total, was 875; **877 gtest**, was 862) builds
*both* backends on the same fixture (loading once per backend, since `BuildModel`
consumes the `LoadedModel`) on tiny-llama (untied, no bias) and tiny-qwen2 (tied
+ q/k/v biases, decoupled `head_dim`): registry build, tied/untied storage
sharing, missing-weight report, logits vs the reference (kLast/kAll), logits vs
the HF `activations.safetensors` golden, `kLast` == `kAll` last row, per-layer
hook parity, greedy token-for-token vs the reference *and* the `generate.json`
golden, determinism, the KV invariant (full vs token-by-token, chunked),
workspace reuse, and every front-loaded error path (malformed inputs,
over-capacity → cache unchanged). Plus `WorkspaceTest` for the sizing formula and
monotone growth.

Observed (NEON dev machine; band `rtol 2e-4, atol 2e-4`): optimized vs the
reference **2.4e-7** (kAll) / **1.8e-7** (kLast) on both fixtures; optimized vs
the HF golden **3.7e-6** (llama) / **3.9e-6** (qwen) — indistinguishable from the
reference's own HF agreement (3.7e-6 / 3.9e-6), so the optimized path is as
accurate against HF as the oracle. The **KV invariant is bit-exact**
(max-abs-diff 0), matching the reference — GEMV≡GEMM-row (T02) and
decode≡prefill(T=1) (T05) are bitwise, and each row's online-softmax recurrence
is T-independent within one thread. Full ctest green (host ISA + forced scalar);
format + scoped tidy clean (the new TUs + `registry.cpp`/`main.cpp` +
`registry_test.cpp`); `src/parallel/` untouched.

**Real ~1B model (acceptance b).** Qwen2-0.5B-Instruct (bf16, 24 layers,
`Qwen2ForCausalLM`, tied embeddings + q/k/v biases, `V=151936`, `E=896`,
`H=14`, `Hkv=2`, `d=64`, `I=4864`) downloaded to `~/models/` and driven with
`engine generate` on the M2 dev machine. It loads (~0.4–1.0 s warm/cold) and
generates coherent text; the **reference and optimized backends produce
byte-identical greedy output** — the token-for-token acceptance holds on a real
~1B-class model, not just the tiny fixtures. Samples (greedy, `--max-new-tokens`
as shown):

- Prompt `The capital of France is` →
  `_______.____。\nParis\nLondon\nGermany\nAustralia\n答案:\n\nA\n\n1000…`
  (base-completion rambling — no chat template, which is M10 — but coherent
  English and correctly surfaces "Paris"; identical on both backends).
- Chat-templated (`<|im_start|>…user\nName three primary colors.<|im_end|>
  <|im_start|>assistant\n`) → `Here are three examples of primary colors:\n- Red:
  typically associated with warmth, excitement, and excitement. It is usually a
  bright, vibrant, and brightly colored color. It is often used in…`

Informal driver timing (the formal ±5% baseline is M6-T08's `bench_generate`, not
this): optimized **decode 33.3 tok/s / prefill 47.4 tok/s** vs reference **10.4 /
12.2 tok/s** — ~3.2× decode, ~3.9× prefill on the same prompt (8-thread default),
the packed-GEMM + blocked-attention win end-to-end. The download stalled hf on
its final xet-verification step (byte-complete at 988,097,824 = the repo's
`content-length`); the completed file was placed and validated by loading (291
BF16 tensors, tied embeddings resolved by the loader alias), so no re-download was
needed.

### M6-T08 — generation benchmark & first baseline (2026-08-18)

New `benchmarks/bench_generate.cpp` (+ target and two smoke `add_test`s in
`benchmarks/CMakeLists.txt`): an end-to-end prefill/decode tokens-per-second
harness that drives the real greedy `Generate` loop and splits the run via its
per-token callback timestamps — **prefill tok/s** = `prompt_len /` (start → first
token), **decode tok/s** = `1000 /` the run's *median* per-step latency. Median
(not whole-window average) is the steady-state rate: on a non-quiesced machine a
single background hiccup inflates one step out of 128 (the `p90 ≪ max` gap)
without reflecting steady-state throughput, and it is what makes the per-run
number repeatable within the ±5% target. The prompt is synthetic random ids in
`[0, vocab)` (like `llama-bench -p N` — only length drives the compute, no
tokenizer dependency). One warmup run absorbs first-touch faults + workspace
grow-on-demand. Flags: `--model --backend --threads --prompt-len --new-tokens
--runs --seed --markdown`. `--threads N` sets `ENGINE_NUM_THREADS` **before**
`load_model` (weight packing already touches `DefaultPool`, which sizes once), so
a thread sweep is a shell loop. The report prints a hardware/thread/ISA/dtype
fingerprint (CPU brand via `sysctlbyname` on macOS), the per-run table, a summary
(best-of-N prefill, median decode, decode-step p50/p90/max), and a PASS/OVER
verdict against ±5%. Backend-agnostic: reference and optimized run the identical
path. Not registered with CTest for perf (run by hand, numbers in BASELINES.md),
but two tiny smoke `add_test`s on the committed tiny-llama fixture (both
backends, `--runs 2`) give CI runtime coverage of the harness — the binary also
builds warnings-as-errors in CI (benchmarks build by default).

**Baseline (BASELINES.md M6-T08).** Qwen2-0.5B-Instruct (bf16), M2, 8-thread NEON
optimized: **prefill ~133 tok/s (best), decode ~31 tok/s (median), decode
run-to-run ±3.9% — PASS** (step p50 32.1 ms / p90 37.4 ms). The 8-thread row is
the recorded baseline; 4t (~101/~26) and 2t (~57/~15) are advisory (light
background load), and the full quiescing + thread/ISA sweep discipline is
M12-T01. Honest stability caveat recorded: back-to-back invocations on this
laptop intermittently spike a decode step to 130–650 ms (OS/background, not the
engine) — run one config at a time on an idle machine for a clean ±5%.

**llama.cpp context number** (parity is M12, not here). Cloned + built
`ggml-org/llama.cpp` @ `6d05498`, CPU-only (`-DGGML_METAL=OFF -DGGML_ACCELERATE=OFF
-DGGML_BLAS=OFF -DLLAMA_CURL=OFF`), same Homebrew LLVM 20 toolchain (Apple clang
is broken here) with `-Wno-elaborated-enum-base` to clear the CLT-SDK
CoreFoundation headers (the same broken-CLT wall the M6-T02 Accelerate number
hit). Converted the same checkpoint to bf16 and f16 GGUF via
`convert_hf_to_gguf.py`. `llama-bench -p 128 -n 128 -r 5`: f16 = **192.5±8.3
pp128 / 18.2±8.4 tg128 (8t)**, **111.9±7.0 / 34.0±7.1 (4t)**; bf16 = 7.7±5.0 /
5.1±1.0 (8t). Findings: (1) llama.cpp's CPU **bf16** GEMM is unvectorized on ARM
(~20–25× slower than its f16), so its **f16** GGUF is the fair opponent for our
bf16-first model; (2) we sit at ballpark parity — within ~1.4× on prefill, ~par
on decode; (3) the M2 **efficiency cores throttle memory-bound decode in both
engines** (llama.cpp decode is *faster at 4 threads than 8*), motivating an
E-core-aware/P-core-only decode pool as an M12 lever, compounding our decode
kernel's `Hkv`-only parallel width (`Hkv=2` here; design §8, the M12-T03
motivator). Conversion pulled a few packages into `tools/.venv`; the pinned
fixture-generation environment was **restored** afterward (torch/transformers/
tokenizers/hf-hub/safetensors/numpy back at their pins, extras removed). The
llama.cpp build (`~/llama.cpp`) and the two ~1 GB GGUFs (`~/models/`) are kept
uncommitted for the M12 sweep and are safe to delete.

Design docs synced: `optimized-cpu-execution.md` §10 benchmark-obligations M6-T08
bullet marked landed with the numbers; §5 pool note updated (the consequential
finding is the heterogeneous cores, not the briefly-idle core). No `src/` change;
907 ctest entries green (905 prior + 2 smoke); format + scoped tidy clean.

## Milestone 7 — Sampling & Generation Controls

### M7-T01 — SamplingParams & pipeline skeleton (2026-08-18)

Replaced greedy-only generation's hard-coded argmax with a real (if
single-branch) sampling pipeline, without changing any output. New `sampling`
module content (the archive was an anchor-only placeholder until now):

- **`src/sampling/params.{h,cpp}`** — `SamplingParams` (temperature, top_k,
  top_p, repetition/presence/frequency penalties, optional seed, max_tokens,
  stop_token_ids, stop_strings, logprobs) with OpenAI/vLLM-convention defaults
  (a default-constructed value is the identity "sample from raw softmax"
  request) and `ValidateSamplingParams` — the single validation entry point
  (shared with the future M10 request mapper), returning `InvalidArgument`
  naming the offending field. `SamplingParams::Greedy(max_tokens)` builds the
  pre-M7 config (temperature 0). `kMaxLogprobs = 20` (the OpenAI cap).
- **`src/sampling/sampler.{h,cpp}`** — the single-sequence reference `Sampler`.
  `Create(params)` validates, then **rejects any parameter outside the greedy
  subset with `Unimplemented` naming the field** (the `Backend::kOptimized`-
  until-M6 posture — no requested knob silently ignored; each later ticket drops
  its guard). `Sample(logits, SampleContext{prompt_ids, generated_ids})`
  dispatches on `temperature`: the `== 0` branch is argmax with a strict-`>`
  lowest-index tie-break (NaN max → `Internal`, empty row → `InvalidArgument`) —
  the greedy logic moved verbatim from the M5-T09 loop's `ArgmaxLastRow`; the
  `> 0` branch is the T02 seam (currently `Unimplemented`). History-stateless
  (the engine owns the token vectors; `generated_ids.size()` is the step index
  the RNG/penalties will key on), so it is cheap per-request and ready for the
  batched path (T06) and M9's `Request`.

Engine routing (`src/engine/generator.{h,cpp}`): `GenerateOptions` swaps
`max_new_tokens` for a `sampling::SamplingParams sampling` field (`eos_ids` stays
a separate, model-derived field — config/tokenizer EOS, distinct from a
request's `stop_token_ids`). `Generate` builds one `Sampler` up front (its
validation/`Unimplemented` are front-loaded like the old `max_new_tokens <= 0`
check, `cache` untouched on error), reads the last logits row as a `[V]` span,
and replaces both argmax calls with `sampler.Sample(...)`. `engine → sampling`
linked PUBLIC (generator.h exposes `SamplingParams`); the old
`ArgmaxLastRow`/`tensor` argmax dropped (tensor stays a PRIVATE link for the row
read).

**Non-obvious decisions.** (1) `max_new_tokens → sampling.max_tokens` is a
deliberate breaking rename across the six call sites (main.cpp, bench_generate,
three test suites) — one source of truth, the OpenAI/roadmap name — over keeping
a duplicate field. (2) Not-yet-implemented knobs `Unimplemented` rather than
ignored, so a default `SamplingParams` (temperature 1.0) does **not** run in
T01; greedy call sites use `SamplingParams::Greedy(n)`. (3) `Sample` is `const`
in T01 (greedy is pure) with a header note that T02's RNG counter becomes
`mutable` (or the qualifier drops) — chosen over leaving it non-const-able and
carrying a NOLINT. (4) The NaN-max contract is exactly the ported M5 one: the
strict-`>` scan makes the max NaN only when the leading logit is NaN, not on any
NaN in the row — the sampler test asserts the real contract, not a stronger one.

Docs: model-execution.md §15 written (SamplingParams table, the fixed stage
order, the `Sampler` contract, engine routing, the per-ticket forward map); §10
snippet re-annotated with an M7-T01 update note; §14 sampling bullet retired to
"designed in §15".

Tests (+35 gtest cases → 912 green, 942 ctest entries): new
`sampling_params_test` (18 — defaults, `Greedy`, boundary acceptances, one
rejection per rule with the field named) and `sampler_test` (16 — greedy
argmax/tie-break/single-element/determinism, empty-row and NaN error posture,
the `Unimplemented` guard per not-yet-landed knob, the `InvalidArgument` vs
`Unimplemented` distinction), both portable C++ (`sampling` label, no
SCALAR_PASS). One `generator_test` case added (a stochastic request is rejected
`Unimplemented` before generating, cache untouched); the existing greedy goldens
across `generator_test`/`qwen2_family_test`/`optimized_model_test` (HF
`generate.json` token-for-token, determinism, KV invariant, EOS/max-cap,
callback, error paths) pass unchanged — the regression guarantee. Full ctest
green; format + scoped tidy clean.

### M7-T02 — Temperature, top-k, top-p sampling (2026-08-18)

Filled the `Sampler`'s `temperature > 0` branch with the stochastic pipeline —
temperature → top-k → top-p → a seeded categorical draw — so generation is no
longer greedy-only. Real ~0.5B-model output is now reproducible per seed
(`engine generate --temperature 0.8 --top-p 0.95 --seed 1` gives the same tokens
every run; a different seed diverges; `--temperature 0` stays bit-identical to
the old greedy path).

- **`src/sampling/philox.h`** (header-only) — Philox4x32-10 (Salmon et al.,
  SC'11; the Random123/JAX/PyTorch generator). A `constexpr` pure block function
  `Philox4x32_10(key, ctr)` plus `PhiloxUniformDouble(seed, step, draw)` →
  `[0, 1)` with 53 bits (two output words; float's 24 are too coarse for a
  ~150k-vocab tail). Keyed on the seed; the counter carries `(step_lo, step_hi,
  draw, 0)`. Because the whole output is a pure function of `(key, counter)`
  there is **no mutable RNG stream** — a draw is fully determined by its
  `(seed, step)` coordinate, which is what makes sampling reproducible per
  `(seed, step)`, batch-composition-independent, and lets `Sample` stay `const`.
- **`src/sampling/stages.{h,cpp}`** (`engine::sampling::detail`) — the stages
  factored out for exact-mask testing and as the T06 token-for-token reference:
  `CheckFinite` (NaN/+inf/all-−inf → `Internal`), `ApplyTemperature` (in-place
  divide), `ApplyTopK` (threshold semantics matching `torch.topk` — mask
  everything below the k-th largest, boundary ties kept, may keep > k; `nth_element`
  on a copy; no-op at `k == 0`/`k >= V`), `Softmax` (max-subtracted, `double`
  sum in ascending order, −inf → exactly 0), `ApplyTopP` (descending sort with
  ascending-index tie-break, keep the prefix reaching `top_p` inclusive of the
  crossing token, ≥1 kept, no-op at `top_p >= 1`), `SelectByCdf` (inverse-CDF
  walk in ascending index over `u * mass`, last-positive fallback).
- **`src/sampling/sampler.{h,cpp}`** — the `Create` guard (renamed
  `CheckGreedySubset` → `CheckImplementedSubset`) drops the temperature/top-k/
  top-p clauses; penalties/stop/logprobs still `Unimplemented`. `Create` resolves
  the seed (explicit request seed, else `random_device` mixed with a
  steady-clock tick) and stores it; new `seed()` accessor exposes it. `Sample`
  gains the stochastic branch (`SampleStochastic`: copy → temperature → top-k →
  softmax → top-p → Philox draw) and stays `const`.
- **Engine/CLI**: no `Generate` logic change (it already routed through the
  sampler in T01). `src/main.cpp` `engine generate` gains `--temperature`,
  `--top-k`, `--top-p`, `--seed` (greedy stays the default at temperature 0).

**Non-obvious decisions.** (1) `Sample` **stays `const`** — the T01 header note
predicted a `mutable` counter, but deriving the Philox counter from
`generated_ids.size()` means there is no stream state to advance, which is the
cleaner outcome and the one that guarantees batch-independence. (2) **Top-k uses
threshold (not exact-k) semantics** matching `torch.topk`/vLLM: boundary ties are
all kept, so a request can end up with > k candidates — deterministic and the
documented HF behaviour. (3) **`double` accumulation** in softmax and the CDF, but
sampled *token sequences* are only same-machine reproducible: `std::exp` differs
by ulps across libm's, and `sampling` cannot link `kernels`' shared exp
polynomial (ADR-002) — the cross-platform contract is M17-T04's. Recorded in
stages.h and §15.2. (4) Reference path allocates per-call scratch (a logits copy
and the prob buffer); allocation-freedom is explicitly T06's concern.

Docs: model-execution.md §15.2 (top-k/top-p semantics, inverse-CDF selection,
the Philox keying and portability caveat), §15.3 (the `const`/no-mutable-state
rationale, `seed()`, the implemented-subset guard), §15.5 (T02 marked done).

Tests (+41 gtest cases → 953 green, 983 ctest entries), all portable C++
(`sampling`/`engine` labels, no SCALAR_PASS) and **deterministic** (every
statistical test uses a fixed seed, so no flake): new `philox_test` (7 — the
three canonical Random123 known-answer vectors asserted at runtime *and* compile
time via `static_assert`, counter/key sensitivity, uniform range/reproducibility/
decile coverage), new `sampling_stages_test` (18 — exact behaviour of every
stage: finiteness posture, temperature exactness, top-k threshold-ties/no-op,
softmax normalization + −inf → 0, top-p crossing-token/tiny-p/no-op/tie-order,
CDF cumulative-walk/never-masked/clamp), `sampler_test` (+13, 3 reject→accept
flips: chi-square vs softmax at T=1 and T=0.5 over 20k draws within the dof-4
0.001 critical value, top-k/top-p support restriction, same-seed-identical,
different-seed-different, draw-depends-on-step-not-history-contents, `seed()`
echo + nullopt-distinct, stochastic empty/NaN/+inf error posture, never-samples
−inf), `generator_test` (+3, 1 rewrite: stochastic same-seed-identical /
different-seed-different / nullopt-seed-runs on tiny-llama; the
`Unimplemented`-before-generating case reworked to use a still-guarded
penalty knob). The greedy `generate.json` goldens across
`generator_test`/`qwen2_family_test`/`optimized_model_test` pass unchanged — the
regression guarantee. Full ctest green; format + scoped tidy clean.

### M7-T03 — Repetition, presence & frequency penalties (2026-08-19)

Landed the sampling pipeline's stage 1 — the three history-based penalties —
applied to the raw logits *before* temperature and ahead of the greedy argmax
alike, so a penalty can change the selected token in both the greedy and the
stochastic modes. `engine generate --repetition-penalty 1.3` (etc.) now steers a
real model away from immediate repetition.

- **`src/sampling/stages.{h,cpp}`** — new `detail::ApplyPenalties(logits,
  prompt_ids, generated_ids, repetition, presence, frequency)`. Documented
  history convention (matching HF + vLLM): `repetition_penalty` penalizes every
  token in the prompt **or** the generated output, once per distinct token
  (`x < 0 ? x·r : x/r`, not compounded on recurrence); `frequency`/`presence`
  count **generated** tokens only (a prompt-only token is untouched) — frequency
  subtracts `f · occurrences`, presence a flat `p` once. Applied in the order
  repetition → frequency → presence. History collected into an
  `unordered_map` count + `unordered_set` seen (O(history), not O(V)) and fully
  validated (`[0, V)`) **before** any logit is mutated, so a bad id leaves the
  logits untouched (`InvalidArgument` naming index + value). Exact no-op — a
  fast path that returns before scanning — when all three are at defaults, so the
  default pipeline stays bitwise identical to pre-T03. `-inf` stays `-inf`.
- **`src/sampling/sampler.{h,cpp}`** — `CheckImplementedSubset` drops the three
  penalty `Unimplemented` guards (stop-ids/stop-strings/logprobs still guarded).
  `Sample` runs penalties in both branches: the greedy branch keeps its
  allocation-free `ArgmaxRow` fast path when no penalty is active (the regression
  guarantee) and otherwise penalizes a logits copy before argmax; `SampleStochastic`
  gained a `SampleContext` parameter and applies penalties on the working copy
  after `CheckFinite`, before temperature. A `PenaltiesActive` helper gates the
  greedy fast path.
- **CLI**: `src/main.cpp` `engine generate` gains `--repetition-penalty`,
  `--presence-penalty`, `--frequency-penalty`.

**Non-obvious decisions.** (1) **History split is not uniform** — repetition
reads prompt ∪ generated, frequency/presence read generated only. This matches
HF (`RepetitionPenaltyLogitsProcessor` over `input_ids`) and OpenAI/vLLM
(frequency/presence over sampled tokens), and is tested explicitly with a
prompt-only token that repetition touches but the other two do not. (2) **Repetition
applies once per distinct token** — HF's gather/scatter is idempotent per token
(duplicate writes are identical); we dedup via the `seen` set so `x/r` is never
compounded, tested with a token appearing three times in the prompt. (3)
**Penalties apply to the greedy branch too** — the design places them at stage 1,
before the greedy/stochastic split, so `temperature 0` requests still honour
penalties (an argmax-flip test proves it). (4) The reference path allocates a
per-call logits copy in both branches when penalties are active; allocation-freedom
remains T06's concern.

Docs: model-execution.md §15.2 stage 1 (the history convention, order, greedy
applicability, no-op guarantee) and §15.5 (T03 marked done).

Tests (+14 gtest cases → 967 green, 997 ctest entries), portable C++
(`sampling`/`unit` labels, no SCALAR_PASS), deterministic: new
`sampling_penalties_test` (12 — every case dyadic so post-penalty logits are
asserted with **exact** equality: repetition/frequency/presence singly, r<1 and
negative-presence boosts, all-three-compose-in-order, defaults-exact-no-op,
masked −inf stays −inf, empty-history no-op, once-per-distinct-token, and the
out-of-range/negative id `InvalidArgument` with logits untouched);
`sampler_test` (3 reject→accept/behavioural flips: `AcceptsPenalties`, greedy
argmax shifts under a repetition penalty, empty-history greedy no-op,
out-of-range history through `Sample`); `generator_test` (+1, 1 rewrite: an
end-to-end penalty run that alters the greedy trajectory and reproduces exactly,
plus the `Unimplemented`-before-generating case moved onto `logprobs`). The
greedy `generate.json` goldens across `generator_test`/`qwen2_family_test`/
`optimized_model_test` pass unchanged. Full ctest green; format + scoped tidy
clean.

### M7-T04 — Stop conditions & finish reasons (2026-08-19)

Replaced the generation loop's EOS-and-max-tokens-only stopping with a full stop
pipeline: the model EOS set, the request's `stop_token_ids` and `stop_strings`
(matched on the incrementally detokenized stream, across token boundaries), and
`max_tokens`, producing a `finish_reason` (`stop`/`length`). The machinery lives
in the loop, not the sampler (design §15.2). `engine generate --stop "</s>"
--stop-token-id 42` now trims output at a stop string and reports why it stopped.

- **`src/engine/stop.{h,cpp}`** — new. `FinishReason{kStop,kLength}` +
  `FinishReasonName`, and the finer `StopTrigger`
  (`kNone/kEosId/kStopTokenId/kStopString/kMaxTokens`, for M10's `stop_reason`).
  **`StopStringMatcher`** — a pure byte-stream matcher (no tokenizer dependency,
  unit-testable in isolation, reusable by M9): `Feed(chunk)` appends and emits
  the safe prefix, **holding back** the longest suffix that could still start a
  stop string (so a stop split across `Feed`s is caught and the trailing text is
  trimmed); earliest-occurrence match, ties broken by stop-list order; naive
  O(held·Σ|stop|) scan; `Flush()` releases the held tail at stream end.
  **`StopChecker`** — the per-request composite over a `DetokenizerStream`
  (M4-T10): `Observe(id, count)` checks EOS → `stop_token_ids` → `stop_strings`
  → `max_tokens` in that priority; id-based stops keep the token's text un-trimmed
  and release the matcher's held tail, a stop-string match trims its bytes and
  everything after; `Finish()` releases held/residual text except after a
  stop-string match. `stop_strings` without a tokenizer → `Create` is
  `InvalidArgument` (front-loaded); a null tokenizer runs the token-only path
  (no text).
- **`src/engine/generator.{h,cpp}`** — `GenerateOptions` gains
  `const tokenizer::Tokenizer* tokenizer` (required iff `stop_strings` set) and
  `bool skip_special_tokens`. **`Generate` now returns `GenerateResult`** (tokens,
  `finish_reason`, `stop_trigger`, `matched_stop`, detokenized `text`) instead of
  a bare `vector`, and the callback takes a **`TokenEvent`** (id + safe-to-emit
  text delta; concatenated deltas equal `result.text`, the end-of-stream residue
  folded into the finishing token's delta). The loop builds a `StopChecker` up
  front (alongside the sampler — both validated before any forward), and a shared
  `consume` lambda samples → appends → `Observe` → folds `Finish()` on the last
  token → fires the callback. Capacity check, determinism, and greedy token
  trajectory are unchanged.
- **`src/sampling/sampler.{h,cpp}`** — `CheckImplementedSubset` drops the two
  stop guards; the sampler now **accepts and ignores** `stop_token_ids`/
  `stop_strings` (they are the loop's, not the sampler's). Only the logprobs
  (T05) guard remains.
- **CMake / edge**: `engine_engine` links **`engine::tokenizer` PUBLIC** (stop.h
  / generator.h name it) — `engine → tokenizer` is an already-sanctioned ADR-002
  edge, first used here; no ADR amendment. **Breaking API change** rippled through
  6 call sites (main.cpp, bench_generate, generator/qwen2_family/optimized_model
  tests): `Generate(...)` result is now `.tokens`, the callback signature is
  `const TokenEvent&`. CLI gains repeatable `--stop STR` / `--stop-token-id N`
  and prints `finish_reason`.

**Non-obvious decisions.** (1) **Priority is EOS → stop_token → stop_string →
max_tokens** (vLLM's), and it is *observable*: a token that is both EOS and
completes a stop string reports `kEosId` with its text kept (an id-based stop is
never trimmed), tested directly. (2) **The completing token is included in the
ids** but its stop-string bytes (and anything after) are trimmed from `text` —
matching vLLM's default `include_stop_str_in_output=false`. (3) **Streaming trim
correctness**: the matcher holds back any pending stop-prefix so a streaming
consumer never sees bytes that later turn out to be (part of) a stop string; on a
non-stop-string finish the held tail is real output, released by `Finish()` into
the last token's delta so concatenated deltas still equal `result.text`. (4)
**Stop strings match bytewise**, not UTF-8-validated (an ill-formed stop could
split a delta mid-codepoint) — left to the M10 request mapper, noted in stop.h.
(5) The `StopStringMatcher` deliberately has **no tokenizer/model dependency** so
it (and its cross-boundary tests) stand alone and M9's per-request loop can reuse
it.

Docs: model-execution.md §10 (new `GenerateResult`/`TokenEvent` snippet + M7-T04
update note, stopping bullet), §15.2 (the landed stop paragraph: matcher +
checker + priority), §15.3 (guard list), §15.5 (T04 done).

Tests (+25 gtest cases → 992 green, 1022 ctest entries), portable C++
(`engine`/`unit` labels, no SCALAR_PASS), deterministic: new **`stop_test`** (21
— matcher: single-chunk / two-chunk / three-chunk splits, hold-back released when
broken, overlap, earliest-occurrence + list-order tie-breaks, a UTF-8 stop split
across chunks, `Flush` idempotence, unbounded-hold guard; checker over a
synthetic byte-level BPE tokenizer: `max_tokens` exact + `length`, stop string
across tokens with trailing trim, stop within a single token, EOS/stop-token
`stop`, EOS-beats-stop-string priority, `max_tokens` releasing a held prefix,
stop_strings-without-tokenizer `InvalidArgument`, null-tokenizer no-text);
`generator_test` (+4 end-to-end: `max_tokens`→`kLength`, EOS→`kStop`/`kEosId`,
`stop_token_ids`→`kStop`/`kStopTokenId` inclusive, stop_strings-without-tokenizer
front-loaded `InvalidArgument`); `sampler_test` (2 reject→accept:
`AcceptsStopTokenIds`/`AcceptsStopStrings`). The greedy `generate.json` goldens
across `generator_test`/`qwen2_family_test`/`optimized_model_test` pass unchanged
(the regression guarantee). Full ctest green; format + scoped tidy clean.

### M7-T05 — Logprobs (2026-08-19)

Added the last sampling stage: per-step logprobs, the OpenAI-style chosen-token
logprob and top-N `(id, logprob)` alternatives, returned by a new
`Sampler::SampleWithLogprobs` and carried on `GenerateResult`/`TokenEvent`. This
drops the final `Unimplemented` guard — every `SamplingParams` field is now
implemented.

**Documented "which logits" choice (design §15.2 stage 6):** logprobs are the
natural-log softmax of the logits the sampler saw *after* penalties (stage 1)
and temperature (stage 2) but *before* the top-k/top-p truncation filters — the
model's full-vocabulary (temperature-scaled) belief, not the renormalized
truncated nucleus. So the reported distribution always covers the whole vocab
and sums to 1 in prob space (the acceptance criterion); greedy
(`temperature == 0`) uses the post-penalty logits with no scaling. Consequence:
with top-k/top-p active a reported token's logprob can be non-zero even though
the filter would never let it be sampled — matching OpenAI's "logprobs describe
the model, the filters describe the draw" convention, and vLLM's `raw`-style
(not `processed`) logprobs.

- **`src/sampling/logprobs.h`** — new, header-only. `TokenLogprob{id, logprob}`
  and `StepLogprobs{chosen_logprob, top}`. `top` is descending logprob, ascending-id
  tie-break, holding `min(params.logprobs, #finite)` entries (`-inf`-logit tokens
  excluded); `chosen_logprob` is always finite.
- **`src/sampling/stages.{h,cpp}`** — stage 6. `LogSoftmax` (log-sum-exp in
  `double`, ascending index order, `-inf` logit → `-inf` log-prob, `sum exp == 1`).
  `ExtractLogprobs(lp, chosen, top_n)` (`std::partial_sort` of the `min(top_n,
  #finite)` largest — `top_n ≤ kMaxLogprobs`; excludes masked entries).
- **`src/sampling/sampler.{h,cpp}`** — new `SampleResult{token,
  optional<StepLogprobs>}` and `SampleWithLogprobs`; `Sample` retained as the
  token-only path (used by the many call sites and the T06 batched reference),
  both routed through one private `SampleWithLogprobsImpl(…, want_logprobs)`.
  Greedy captures the log-softmax over the (penalized) row after argmax; the
  stochastic branch captures it over the post-penalty/post-temperature `work`
  buffer *before* `ApplyTopK` masks it. The RNG draw is untouched, so
  `SampleWithLogprobs` picks the same token as `Sample` (tested). **`Create` no
  longer calls `CheckImplementedSubset`** — the guard is gone. The default
  (`logprobs == 0`) greedy no-penalty path stays the allocation-free
  bitwise-unchanged `ArgmaxRow`.
- **`src/engine/generator.{h,cpp}`** — `GenerateResult` gains
  `std::vector<sampling::StepLogprobs> logprobs` (index-aligned with `tokens`,
  empty unless requested); `TokenEvent` gains
  `const sampling::StepLogprobs* logprobs` (the M10 SSE per-chunk seam; null
  unless requested). The loop calls `SampleWithLogprobs`, stores the entry, and
  points the event at it. Reserves the logprobs vector only when requested.
- **`src/main.cpp`** — `engine generate --logprobs N` prints, per step, the
  chosen token's logprob and the top-N alternatives (single-id decode for
  readability, best-effort).

Non-obvious decisions: (1) **Two public entry points**, not a changed `Sample`
return type — keeps the ~22 `Sample` call sites (and the doc's `Sample`
signature) intact while giving logprobs a clean home; both share one impl. (2)
**Full-vocab pre-truncation logprobs** — the alternative (renormalized nucleus)
trivializes "sum ≈ 1 over full vocab" and can't fill top-N when `top_k < N`. (3)
**Greedy chosen logprob == max** falls out for free: greedy token == argmax, so
`chosen_logprob == top[0].logprob == max(log-softmax)` — the third acceptance
criterion. (4) `sampling` still can't link the kernels exp polynomial (ADR-002),
so logprobs (like the draw) are same-machine reproducible only until M17-T04.

Tests (+~35 cases): new **`sampling_logprobs_test`** (14) — `LogSoftmax` exact
(uniform row, double reference, known ln-ratio probabilities, large-offset
stability, `-inf` handling, `sum exp == 1` over a 512-vocab row) and
`ExtractLogprobs` (chosen entry, top-N vs a full-sort reference for N∈{1,5,20},
ascending-id tie-break, masked-token exclusion, vocab clamp, N=0). **`sampler_test`**
(+~7) — `AcceptsLogprobs` (was `RejectsLogprobs`); logprobs-off → `nullopt`;
greedy chosen == max; greedy logprobs use the penalized row; stochastic tokens
unchanged vs the token-only path; stochastic full-vocab pre-truncation logprobs
sum to 1. **`generator_test`** (+4, −1) — dropped
`NotYetImplementedSamplingIsRejectedBeforeGenerating` (no unimplemented knob
remains); added greedy e2e (logprobs aligned with tokens, top-1 == chosen,
golden tokens unchanged), `TokenEvent.logprobs` matches the stored entry,
logprobs-off leaves the result empty and the event pointer null, stochastic
tokens unchanged with logprobs on. 1042 ctest green; greedy `generate.json`
goldens on both backends unchanged; format + scoped tidy clean.

### M7-T06 — Batched sampling optimization (2026-08-19)

Closed M7 with the batched sampler: `src/sampling/batched_sampler.{h,cpp}`
(`BatchedSampler` + `BatchRow`) samples one token per sequence from a
`[batch, vocab]` logits block, in parallel across sequences, for M9's
continuous-batching loop to consume. The load-bearing acceptance criterion —
**picks identical tokens (and logprobs) to the single-sequence reference sampler
given identical RNG counters, bit-for-bit** — is met **by construction**, not by
test: the per-row pipeline was factored into one shared `detail::SampleRow`
(sampler.h) that both the reference `Sampler` and the batched sampler call, so
there is a single arithmetic implementation.

**The design tension and how it was resolved.** The ticket asks for both
"vectorized softmax reusing the M3/M6 kernels" *and* bit-exact token identity
with the reference. Those force a shared exp — the reference computed softmax
with `std::exp(double)`, which no vector kernel reproduces bit-for-bit. Chosen
resolution (recorded in design §15.6): promote the M6-T03 polynomial to a public
**unthreaded** `kernels::ExpF32` (`src/kernels/exp.{h,cpp}`; runs on the calling
thread so it can be invoked inside a caller's own `parallel_for` body) and switch
the reference's `detail::Softmax`/`LogSoftmax` to it. Consequences: `sampling →
kernels` (PRIVATE) is a layer-2 → layer-1 edge ADR-002 already permits (as
`model → kernels` is — no amendment; the stale "sampling cannot link kernels"
notes were corrected across stages.h/logprobs.h and design §15.2); sampled
sequences become libm-independent for a fixed ISA (a step toward M17-T04) but
stay Class T across ISAs, so `batched_sampler_test` registers `SCALAR_PASS`; and
five T05 logprob assertions that pinned `std::exp(double)` values were
re-toleranced to fp32-exp class (~2.4e-7 rel.). Greedy `generate.json` goldens
are argmax-only and unchanged.

**The optimizations, all shared with the reference (so identity is free):**
(1) **vectorized softmax/log-softmax** via `kernels::ExpF32` (weights summed in
`double` ascending; `x < kExpLo → +0.0` keeps the `-inf → 0` mask exact);
(2) **partial-sort filtering** — `ApplyTopK` already used `nth_element`;
`ApplyTopP` now sorts **only the positive-probability tokens** (a zero can never
enter the nucleus), so a post-top-k row sorts its ~`k` survivors instead of the
whole vocabulary — bit-identical to a full sort + walk (the order is total, the
positive prefix unique), and never worse than the old full sort for an
all-positive row; (3) **`parallel_for` across the batch** at grain 1, where row
`b` only ever touches scratch slot `b`, so **`src/parallel/` is untouched** (the
deferred §6.4 worker-index overload was not needed) and a steady-state step is
**allocation-free** (`BatchedSampler` owns one `RowScratch` per slot, grown on
demand). Shape validation is front-loaded before the region; per-row recoverable
errors (bad history id → `InvalidArgument`, non-finite row → `Internal`) are
captured into a per-row status slot and the lowest-index one returned after the
region (matching a sequential loop). `rows.empty()` is a no-op success.

**Bench (BASELINES.md M7-T06):** new `benchmarks/bench_sampling.cpp` +
two smoke `add_test`s. 64 × 128k on the M2, 8-thread NEON — **greedy ~2.0 ms
(PASS the advisory ≤5 ms)**; temp ~10 ms, top-k 50 + top-p 0.9 ~21 ms (down from
~44 ms once `ApplyTopP` stopped full-sorting the 128k mostly-zero row), logprobs
~22 ms. The sampling configs are DRAM-bandwidth-bound on the several full-vocab
passes over 128k × 64 (and the E-cores throttle them, as in M6-T08 decode); a
fused-pass / E-core-aware pool is an M12 lever, recorded honestly. Threading ~3.5×
at 4t, ~4–5× at 8t; the 1-thread batched path equals the serial reference.

**Tests (+7 gtest cases, ×2 for SCALAR_PASS = +14 registrations, +2 bench smoke
→ 1058 ctest green):** new **`batched_sampler_test`** (SCALAR_PASS) —
bit-for-bit token + logprob identity vs the reference across a grid of batch
sizes {1,3,17,64} × vocab {1,7,50,1000,32000} × rotating per-row params (greedy /
temperatures / top-k / top-p / penalties / logprobs 0–20) with injected `-inf`
masks and ties; batch-composition independence (a row in a batch == sampled
alone); invariance across pools of 1/2/3/8 threads; error posture (shape
mismatches, null sampler, `B=0`, NaN row → Internal, bad history id →
InvalidArgument); allocation-free steady state (`scratch_bytes()` stable across
same-shape steps). Five `sampling_logprobs_test` tolerances relaxed to fp32-exp
class. Format + scoped tidy clean.

## Milestone 8 — Paged KV Cache & Block Manager

### M8-T01 — Design doc: paged KV cache (2026-08-19)

Docs-only. Wrote **`docs/design/paged-kv-cache.md`** — the working contract
M8-T02…T08 (and the M9/M11/M12/M13/M15 interactions) build on: the memory
architecture that replaces M5's per-sequence contiguous `SimpleKvCache` with a
shared pool of fixed-size token blocks, indexed per sequence by a block table,
with reference counting from day one so M11 prefix caching is an extension, not
a rewrite.

What the doc fixes (the non-obvious decisions):

- **Physical layout — two per-layer K/V slabs of head-major block tiles.** The
  pool is `2·num_layers` tensors, each `[num_blocks, Hkv, bs, d]` fp32; the
  innermost `[bs, d]` tile is contiguous, exactly the head-major stream the M6
  kernels read (`transpose(1,2)` of HF's cache). This resolves the roadmap's
  literal `[num_blocks, 2, layer…]` sketch: the `2` is the K/V axis realized as
  **two slabs** (K and V feed different kernel args, never addressed as a unit),
  and **per-layer** slabs (not one giant `[num_blocks,2,L,…]` tensor) keep one
  layer's working set in a dense address range for prefetch/TLB. Deviation from
  the literal sketch recorded per `docs/design/README.md`. Worked byte-offset
  examples for tiny-llama (d=16, 8 KiB/block) and Qwen2-0.5B (d=64, 384
  KiB/block → 2730 blocks / 43,680 tokens at a 1 GiB budget).
- **Block size 16, constrained to `bs ∈ {8,16,32,64}` — the load-bearing
  `bs | kAttnKb` constraint.** M6-T05's decode recurrence rescales the online
  softmax once per `kAttnKb = 64`-key unit (`DecodeGroupSlice`'s four `Ops`
  primitives). For the paged decode kernel (M8-T05) to match the contiguous M6
  kernel **bit-for-bit** (not just Class T), a 64-key unit must be an integer
  number of whole physical blocks, so the paged kernel gathers a unit's blocks
  into the same 64-wide score row and runs the identical max/expsum/scale/axpy.
  Enforced at pool construction; cross-noted in optimized-cpu-execution.md §8.
- **Capacity formula.** `bytes_per_block = 2·L·Hkv·bs·d·itemsize`; `num_blocks =
  ⌊kv_budget / bytes_per_block⌋`. `--kv-cache-memory` is absolute bytes or a
  fraction `f` of host RAM (`kv = f·host_ram − weights_resident − workspace`,
  with `sysctl hw.memsize` / `sysconf` detection); `--kv-block-size` selects
  `bs`. `num_blocks < 1` → `ResourceExhausted`.
- **Block pool refcounting + lifecycle diagram** (M8-T02): free-list
  allocate/free, `Share`/`Release`, FREE→OWNED→SHARED with the double-`Release`
  and `Share`-on-free `CHECK`s, and the dashed M11 `CACHED` state between "rc hit
  0" and "returned to free list." The **immutability invariant** (§6.4) — a
  refcount≥1 block is never rewritten because sequences only append to their
  exclusive refcount-1 tail, and M11 shares only full blocks — is stated now as
  the property M11's correctness proof rests on.
- **Block table + slot mapping** (M8-T03): `slot(pos) = blocks_[pos/bs]·bs +
  pos%bs`, all-or-nothing block allocation on append (a rejected append leaves
  table+pool untouched), hand-worked prefill/decode/boundary-straddle slot
  mappings, truncate/free returning blocks to the pool (RAII).
- **`PagedKvCache` keeps the M5 `KvCache` interface unchanged** — with **one
  additive virtual, `paged_view(layer)`**, default-`Unimplemented`
  (`SimpleKvCache` inherits it), overridden by the paged cache to hand back slab
  pointers + block table with zero copy for the decode hot path (a per-step
  full-history `view()` gather would blow M8-T07's ≤10% regression bound).
  `capacity()` documented as **advisory** under a shared pool (`append`'s
  `ResourceExhausted` is the binding check). This is the only change to code
  above the interface; `view()` keeps its "may copy" contract for the reference
  and prefill read paths.
- **Kernels** (layout-agnostic raw-pointer entries, the `kvcache → kernels`
  downward edge — permitted by ADR-002 like `model → kernels`, no amendment,
  `kv_cache.h`'s comment updated when T04 lands): `KvScatterF32` (M8-T04, a
  bit-exact single-TU copy replacing the contiguous append), `PagedDecodeAttentionF32`
  (M8-T05, the M6-T05 recurrence with block-table indirection, bit-identical by
  construction), and `paged_gather` → the unchanged M6 prefill kernel (M8-T06).
- **Engine integration & exhaustion** (M8-T07/T08): pool owned by the
  engine/runtime, outlives every cache; tiny-fixture greedy output identical to
  pre-paging (the KV invariant under a new backend); graceful `ResourceExhausted`
  on a dry pool with all blocks reclaimed (stats zero, no leaks).

Cross-doc edits in the same change: model-execution.md §6.4 (points to the new
doc; folds back the advisory-`capacity()` and `paged_view` clarifications),
optimized-cpu-execution.md §8 (the `bs | kAttnKb` constraint on the paged decode
kernel). No `src/` change; 1058 tests unchanged.

### M8-T02 — Block pool & allocator (2026-08-19)

New **`src/kvcache/block_pool.{h,cpp}`** — `BlockPool`, the paged KV cache's
block allocator (design: paged-kv-cache.md §6). Pure bookkeeping over
pre-allocated per-layer K/V slabs, fully unit-testable without a model or any
kernel. The first M8 implementation ticket; M8-T03's `BlockTable` and
M8-T04's `PagedKvCache`/scatter build directly on it.

What landed:

- **Storage (§3.1/§6.1).** `Create(geom, block_size, num_blocks, allocator)`
  allocates `2·num_layers` slabs (one K, one V per layer), each a
  `[num_blocks, Hkv, bs, d]` fp32 `tensor::Tensor`, and zero-fills them once.
  The slabs come straight from the passed `Allocator` (the engine's M2
  `CachingAllocator`; tests pass the default CPU allocator or a scriptable
  fake) — the per-block free list is the actual KV block allocator and never
  calls upstream on the hot path.
- **Free list + refcounts (§6.2/§6.3).** `Allocate` pops a LIFO free-list stack
  (reuses cache-warm blocks), sets refcount 1; `Share` bumps it (M11 prefix
  sharing); `Release` drops it, returning the block to the free list at 1→0.
  `refcount`/`stats`/`free_blocks`/`blocks_needed` round out the surface. One
  `std::mutex` guards the bookkeeping (allocation is off the token hot path).
- **Capacity arithmetic (§5.2).** Static pure helpers `BytesPerBlock` (=
  `2·L·Hkv·bs·d·itemsize`, overflow-guarded) and `NumBlocksForBudget` (=
  `⌊budget / bytes_per_block⌋`, `< 1` → `ResourceExhausted`). Host-RAM
  detection and the fraction-of-RAM subtraction (§5.3) stay with M8-T07, the
  first consumer of the fraction spelling (§13).

Non-obvious decisions (each an amendment to the design doc, recorded there):

- **Slabs via raw `Allocator::Allocate(…, 256)` + `Tensor::from_buffer` +
  `memset`, not `tensor::ops::zeros`.** `Tensor::empty`/`ops::zeros` pin a
  64-byte alignment; the design mandates 256 (`kSlabAlignment ==
  CachingAllocator::kMaxAlignment, §6.1`). Going through the allocator directly
  honors the doc for any allocator and lets the test assert `ptr % 256 == 0`.
- **Value-returned `BlockPool` with a `std::unique_ptr<std::mutex>`.** The
  design's `StatusOr<BlockPool>` signature means the pool is move-constructed
  out of `Create`; `std::mutex` is immovable, so it lives behind a `unique_ptr`.
  The move-ctor `CHECK(used_ == 0)` enforces §6.1's "moved out before any block
  is handed out" — a `BlockTable` (M8-T03) will hold a raw `BlockPool*`, so a
  post-handout move would dangle it. No dtor `CHECK`: unlike `CachingAllocator`,
  nothing dangles into a dead pool except that non-owning pointer, whose
  lifetime rule ("pool outlives every table", §10.1) already covers it.
- **`block_size` validated to `{8,16,32,64}`** at construction and in the
  capacity helpers — the §4 `bs | kAttnKb=64` divisibility constraint M8-T05
  rests on for its bit-exact match — with the rationale in the error message.
- **Extra strides/sizing accessors** beyond the design's `block_stride()`:
  `head_stride()`/`row_stride()` (§3.1 names all three; the scatter/decode
  kernels take all three), `slab_bytes()`/`total_bytes()` (the M8-T07 stats
  log), `block_size()`/`num_blocks()`/`geometry()`.

CMake: `engine_kvcache` gains `block_pool.cpp` and an explicit **PUBLIC
`engine::memory`** link (the header takes a `memory::Allocator*`; previously
`memory` was only transitive via `tensor`). No `kvcache → kernels` edge yet —
that lands with the scatter/decode kernels in M8-T04.

Tests: new **`tests/unit/block_pool_test.cpp`** (27 cases, label `unit`, no
`SCALAR_PASS` — pure bookkeeping): construction validation (geometry, dtype,
`block_size`, `num_blocks`; null-allocator default; slab alignment/distinctness/
zero-fill; `CachingAllocator`-backed round-trip freeing all slabs on drop; a
scriptable-upstream failure propagating with no leak); `Allocate`/`Release`
uniqueness, exhaustion → `ResourceExhausted` (stats unchanged, recovery after
release), LIFO reuse; `Share` refcount semantics; a scripted alloc/free/share
sequence asserting `used/free/utilization` and the `used+free==total` invariant
at every step; hand-worked `blocks_needed` boundary crossings; the §3.3
`BytesPerBlock`/`NumBlocksForBudget` worked numbers (tiny-llama 8 KiB,
Qwen2-0.5B 384 KiB → 2730 blocks, Llama-3-8B 4 MiB → 10240); death tests for
double-`Release`, `Share`-on-free, out-of-range ids/layers, and move-after-
handout; and an 8-thread concurrency stress leaving the invariant intact.
**1058 → 1085 ctest green** (+27). Format + scoped tidy clean.

### M8-T03 — Block table & sequence cache handle (2026-08-19)

New **`src/kvcache/block_table.{h,cpp}`** — `BlockTable`, the per-sequence
logical→physical block map over the M8-T02 `BlockPool` (design:
paged-kv-cache.md §7). One table drives all layers (a physical block id names
the same slot region in every layer's slabs, §3.1); a `PagedKvCache` (M8-T04)
owns exactly one. State is the §7.1 sketch verbatim — `BlockPool* pool_`
(non-owning), `std::vector<int32_t> blocks_` (logical block i → physical id),
`int64_t num_tokens_`.

What landed:

- **`AppendTokens(count) → StatusOr<vector<int64_t>>`** (§7.2). Grows the
  sequence by `count` tokens at positions `[num_tokens_, num_tokens_ + count)`
  and returns the `slot_mapping[count]` array (`slot(pos) = blocks_[pos/bs]·bs +
  pos%bs`) the M8-T04 scatter kernel will consume. `need =
  pool_->blocks_needed(num_tokens_, count)` new blocks are `Allocate`d into a
  scratch vector **all-or-nothing**: on any `ResourceExhausted` the taken blocks
  are `Release`d and the call returns `ResourceExhausted` with `blocks_`,
  `num_tokens_`, and the pool left byte-identical. `count <= 0` →
  `InvalidArgument` (a forward always appends ≥ 1 token — additive to the
  sketch, which named only the exhaustion path).
- **`Truncate(new_len)`** (§7.3): drops tokens past `new_len ∈ [0,
  num_tokens_]`, `Release`s the wholly-empty blocks **tail-first** (so the
  lowest logical block ends up on top of the pool's LIFO free list, reused
  first) and keeps the partial surviving tail; out-of-range → `InvalidArgument`,
  state untouched. **`FreeAll`** releases every block and resets; the destructor
  calls it (RAII — a dropped cache returns its blocks, the basis for M8-T08 "no
  leaks, stats zero at end").
- **Surface for the kernels/consumers**: `slot(pos)` (CHECK-guarded to committed
  positions), **`blocks()` returning `std::span<const int32_t>`** — the
  contiguous, logical-order `const int32_t*` the M8-T05 `PagedDecodeAttentionF32`
  reads (§8.3/§9.2), zero-copy, valid until the next append/truncate — plus
  `num_tokens()`/`num_blocks()`/`block_size()`/`pool()`.
- **Move-only, mirroring `BlockPool`**: move-construct leaves the source empty
  (its destructor then releases nothing); move-assign deleted (a table is built
  in place, never reseated). **Not thread-safe by design** — one table per
  sequence on the engine thread; the pool's mutex covers cross-sequence
  contention. In M8 every owned block has refcount exactly 1 (the §6.4
  exclusive-tail invariant), so `Release` is the only pool verb used; `Share`
  waits for M11.

CMake: `engine_kvcache` gains `block_table.cpp`; **no new link edge** — the
table calls only `BlockPool` (same module). The `kvcache → kernels` downward
edge still arrives with the scatter/decode kernels in M8-T04.

Tests: new **`tests/unit/block_table_test.cpp`** (23 cases, label `unit`, no
`SCALAR_PASS` — pure bookkeeping): the §7.2 hand-verified prefill (T=12 → blocks
`[5,2]`, slots `40..47,16..19`, straddle at pos 7→8), no-allocation decode (slot
20), and boundary-crossing-decode-allocates cases — reproduced at **`bs = 8`**
(the smallest `BlockPool`-valid size, since the doc's `bs = 4` is not
allocatable, §4) with a primed LIFO free list; token-by-token == batch growth;
`num_blocks == ⌈num_tokens/bs⌉` and all-slots-unique across a mixed append
sweep; exact-boundary growth; `count <= 0` rejection; all-or-nothing exhaustion
leaving table+pool byte-identical and recovering afterward; truncate
(partial-tail-kept, re-append-reuses-tail, exact-boundary, to-zero, no-op,
out-of-range); free-on-completion via `FreeAll`, destructor, and interleaved
lifetimes; move (moved-from emptied, source destruction releases nothing); and
death tests (null pool, `slot` out-of-range / on-empty). Design §7.2 updated to
`bs = 8` with an "as built" notes block. **1085 → 1108 ctest green** (+23).
Format + scoped tidy clean.

### M8-T04 — KV write (scatter) kernel & PagedKvCache (2026-08-19)

New **`src/kernels/kv_scatter.{h,cpp}`** — `KvScatterF32`, the paged KV-write
kernel (design: paged-kv-cache.md §9.1), and **`src/kvcache/paged_cache.{h,cpp}`**
— `PagedKvCache`, the paged implementation of the M5 `KvCache` interface (§8).
Together they replace the contiguous `SimpleKvCache::append` transpose-copy with
the paged append: grow a per-sequence block table, then scatter the new K/V into
the shared pool's slabs at the resulting slot mapping.

What landed:

- **`KvScatterF32(src_k, src_v, slot_mapping, T, Hkv, d, bs, k_slab, v_slab)`**
  (§9.1). For each (token, head): `memcpy` the `d`-float row from token-major
  `[T, Hkv, d]` source to the slab at `slot = slot_mapping[t]` (block `slot/bs`,
  in-block row `slot%bs`, §3.1). A pure fp32→fp32 copy — **Class E** (bit-exact
  across ISAs and thread counts), so a **single scalar TU** ships (no per-ISA
  variant, like the M6-T06 embedding gather), added to `engine_kernels`. Tokens
  are the `parallel_for` unit (grain 16); distinct tokens write disjoint slots,
  so the threaded scatter is race-free. Raw-pointer, layout-agnostic (§2): the
  kernel never sees `BlockPool`/`BlockTable`.
- **`PagedKvCache`** composes `BlockPool*` (shared, non-owning) + one
  `BlockTable`. `append(layer, k, v)` (§8.2): front-loaded validation (layer
  range, rank/contiguity/dtype/`[T≥1,Hkv,d]`, k.T==v.T — mirroring `SimpleKvCache`
  so both reject identically) precedes any mutation; **layer 0 grows the table**
  all-or-nothing (`AppendTokens` → `pending_slots_`, a `ResourceExhausted`
  aborting the forward with nothing written), **layers 1..L−1 reuse** that slot
  mapping, then `KvScatterF32` writes the layer's slabs. The **per-forward layer
  protocol is enforced** (`next_layer_`): an out-of-order, repeated, or
  wrong-token-count append is `InvalidArgument` (stricter than §8.2's documented
  contract — the check is one integer compare off the write path).
- **`length()`** = the sequence's committed tokens (all layers agree, one table).
  **`capacity()`** is advisory: `num_blocks·bs + free_blocks·bs` — a correction
  to the design's `num_tokens + free·bs`, which under-counts the partial tail
  block's free slots (recorded in §8's as-built notes). **`truncate`** delegates
  to `BlockTable::Truncate` and resets the pending forward state; **`view`** is
  `Unimplemented` until the M8-T06 paged gather (the decode path uses `paged_view`
  instead). Move-only, RAII (a dropped cache returns its blocks).
- **The one additive interface accessor**, `paged_view(layer)` (§8.3): a new
  `struct PagedKvView` (slab bases + block-table pointer + strides) and a
  default-`Unimplemented` virtual on `KvCache`. `PagedKvCache` overrides it
  zero-copy (the decode fast path that keeps M8-T07 under its ≤10% regression
  bound); `SimpleKvCache` inherits the default (its decode falls back to `view()`
  + the contiguous kernel). `kv_cache.h`'s header comment amended to document
  the accessor and the now-permitted `kvcache → kernels` edge.

CMake: `engine_kernels` gains `kv_scatter.cpp`; `engine_kvcache` gains
`paged_cache.cpp` and the **downward `kvcache → kernels` PRIVATE edge** (a
layer-2→1 edge ADR-002 permits without amendment, exactly as `model → kernels`;
`paged_cache.h` stays kernels-free, the `KvScatterF32` call in the .cpp). The
stale "arrives with M8-T04" remarks in `block_pool.h`/`block_table.h` were
refreshed.

Tests: new **`tests/unit/kv_scatter_test.cpp`** (8 cases, `kernels` label, no
`SCALAR_PASS` — no dispatched path) — readback vs an independently-simulated
paged layout (a plain-array model) across a non-monotonic-block boundary-
straddling prefill, single-token decodes (mid-block and fresh-block-row-0),
sequential accumulation, slot overwrite, a `{bs}×{Hkv}×{d}` geometry sweep, a
1000-token permuted mapping, and exact bit-pattern preservation (NaN payload,
−inf, −0.0, denormal). New **`tests/unit/paged_cache_test.cpp`** (16 cases,
`unit` label) — the §7.2 prefill/decode walk (values land at the hand-worked
slots `[5,2]`, verified through `paged_view`), token-by-token == batched, a
wider geometry, front-loaded validation (each leaving stats/length untouched),
the layer protocol (out-of-order/repeat/short/skip → `InvalidArgument`, single-
layer repeated appends), layer-0 exhaustion (nothing written, recovery works),
truncate/reset (blocks released, partial tail kept, re-append overwrites),
`view` Unimplemented, `paged_view` bad-layer, RAII reclamation, move, and two
caches sharing one pool disjointly. One case added to `simple_cache_test`
(`paged_view` default Unimplemented). **1108 → 1133 ctest green** (+25). Design
§8 gained an "as built" notes block, §9.1 an as-built note; format + scoped tidy
clean (kv_cache.h is a header edit, so its full includer set — model/engine/main
+ tests — was tidy-swept).

### M8-T05 — Paged decode attention kernel (2026-08-19)

**What landed.** `kernels::PagedDecodeAttentionF32` — decode attention (one
query per head over the whole cache) reading K/V **through a block table** over
the paged slabs, the zero-copy decode fast path that `PagedKvCache::paged_view`
(M8-T04) feeds. New files: `src/kernels/paged_attention.{h,cpp}` (public entry +
`KernelTable` dispatch + `parallel_for` over `kv_heads` + the
`detail::PagedDecodeAttentionVariant` test seam) and
`src/kernels/internal/paged_attention_common.h` (the `PagedDecodeUnitsImpl` /
`PagedDecodeGroupSlice` templates + `PagedDecodeArgs`). The per-ISA variants are
one-line instantiations added to the **existing** `{scalar,neon,avx2}/
attention.cpp` TUs. Design: paged-kv-cache.md §9.2 (+ an "as built" block).

**Bit-identity by construction (the acceptance).** Because `bs | kAttnKb = 64`
(§4, pool-enforced), one 64-key online-softmax unit is exactly `kAttnKb/bs`
whole physical blocks. The recurrence is M6-T05's `DecodeUnitsImpl` with the
contiguous `k_head + s·d` walk replaced by a block-table walk: per unit it
scores **block-by-block** (`DotScoreRow(q, k_tile, n_b, d, scale, scores +
b·bs)` per physical block, folding per-block maxes with `>`), runs the identical
`ExpRowSum`/`ScaleRow` over the same contiguous 64-wide `scores` row, and axpys
V in ascending key order (only the row *address* changes). So the fp32
accumulation order per output element is unchanged → **bitwise** equal to
`DecodeAttentionF32` on the same logical K/V, and bit-identical across thread
counts (each kv head's recurrence in one call).

**Non-obvious decisions.**
- *Per-ISA variants reuse the existing attention TUs, not new
  `{isa}/paged_attention.cpp` files.* The four `Ops` primitives are TU-local
  (anonymous namespace); reusing them **literally** is the bit-identity
  guarantee. A separate TU would force duplicating the Ops (a second thing to
  keep byte-identical) or hoisting them into per-ISA headers (churn, no benefit).
  So `paged_attention.cpp` carries only dispatch, and each attention TU gains a
  one-liner. Recorded in the kernels CMake comment + design §9.2.
- *`internal/paged_attention_common.h` is a new header, not additions to
  `attention_common.h`.* Keeps M6's header byte-untouched so its includer set
  isn't re-tidied on this ticket.
- *`bs | kAttnKb` is CHECKed at the public entry* (not merely a doc
  precondition): a violating block size silently breaks the "matches the
  contiguous kernel exactly" promise, so it is ADR-003 CHECK territory
  (death-tested). `num_blocks` stays a caller-side bound (`ceil(length/bs) ≤
  num_blocks`), never read past.
- *No consumer wiring here.* `OptimizedModel::forward` swapping to `paged_view`
  (with a `view()`+contiguous-kernel fallback for `SimpleKvCache`) is M8-T07 by
  the roadmap. `src/parallel/` untouched (no per-worker scratch — the online
  accumulator is `out` itself, as in M6). No new link edge (`kvcache → kernels`
  already exists from M8-T04; the kernel lib already links `parallel`).

**Tests** (`tests/unit/paged_decode_attention_kernel_test.cpp`, `kernels` label,
**SCALAR_PASS**; +engine::kvcache/engine::memory for the end-to-end case): 8
cases — **bitwise** (`allclose atol=rtol=0`) equality to `DecodeAttentionF32`
across `bs ∈ {8,16,32,64}` × many-block/exact-boundary lengths (`1..4096`, incl.
exact multiples of bs), GQA `{(4,4),(4,2),(8,1),(24,2: g=12>chunk)}` × d
{18,24,64,128}, large-magnitude rescale (+ Class T vs the `cpu::attention`
oracle, max ~8.6e-6), thread-count invariance (threaded == serial ==
manual-chunked), the HF attention goldens (max ≤1.2e-7), an **end-to-end pass
through a real `PagedKvCache`/`BlockPool`** (append token-by-token, feed
`paged_view` to the kernel — bit-exact vs contiguous, proving the `PagedKvView`
field semantics), and a death test for `bs ∤ 64`. The paged layout is
materialized test-locally (independent of `KvScatterF32`) with a
**reverse-permuted** block table and **poisoned** unused blocks + tail slots, so
a contiguous-fallback or an over-read would fail. **1133 → 1149 ctest green**
(+16 = 8 cases ×2 SCALAR_PASS).

**Bench** (`benchmarks/kernels/attention_bench.cpp` new `paged` mode;
BASELINES.md M8-T05): 2k-context decode step, 8-thread NEON — the isolated paged
kernel is **0.89×** the contiguous decode kernel (274.7 vs 245.8 µs), 0.87× at
1 thread — the block-indirection cost. But the paged kernel **replaces the
`view()` gather** the contiguous path needs in the engine (~12% decode
memory-traffic overhead, optimized-cpu-execution.md §8), so the whole-step
comparison is M8-T07's `bench_generate` regression check, not this
microbenchmark.

**Docs.** paged-kv-cache.md §9.2 "as built" block; optimized-cpu-execution.md §8
"M8-T05 landed" note (consumer swap deferred to M8-T07). Format + scoped tidy
clean (attention_impl.h is a header edit → its includer set — the attention TUs,
attention_kernel/decode_attention_kernel tests, the bench — was tidy-swept;
avx2/attention.cpp is not in the arm64 compile DB, its one-liner hand-reviewed).

### M8-T06 — Paged prefill attention path (2026-08-19)

**What landed.** The paged cache's read side — `PagedKvCache::view` (stubbed
`Unimplemented` since T04) now gathers the layer's committed history into a
contiguous head-major `[Hkv, len, d]` tensor, the layout the **unchanged** M6
`PrefillAttentionF32` reads. With that one method wired, prefill and
prefill-continuation through a `PagedKvCache` work on **both backends with zero
consumer change**: `Attention::forward` (reference) and `OptimizedModel::forward`
(optimized) already do `append(layer) → view(layer) → Prefill/DecodeAttention`
over a `KvCache&`. Two new files:

- `src/kernels/kv_gather.{h,cpp}` — `KvGatherF32(k_slab, v_slab, block_table, bs,
  length, Hkv, d, out_k, out_v)`, the exact mirror of `KvScatterF32`: one
  `memcpy` of `rows·d` floats per (head, logical block), `rows = min(bs, length −
  b·bs)` clipping the partial tail block, `parallel_for` over `Hkv·⌈length/bs⌉`
  disjoint tiles. **Class E** (pure fp32→fp32 copy) — bit-identical across ISAs
  and thread counts, so a **single scalar TU** ships (no per-ISA variant, no
  dispatch table, no `SCALAR_PASS`), exactly like `kv_scatter.cpp` and the M6-T06
  embedding gather.
- `src/kvcache/paged_gather.{h,cpp}` — `GatherLayerKV(pool, table, layer, length)
  → StatusOr<KvView>`, the thin `kvcache` wrapper: front-loaded validation
  (layer range, `length ∈ [0, num_tokens]`), allocates the two `[Hkv, length, d]`
  outputs with `Tensor::empty` (every element overwritten — no zero-fill), calls
  `KvGatherF32`. Header stays kernels-free → the second call over the existing
  PRIVATE `kvcache → kernels` edge (after `KvScatterF32`), so no ADR change and no
  new link edge.

**Non-obvious decisions.**

- **Kernel in `kernels`, not inline in `kvcache`.** The tile copy is a kernel so
  the threading (`parallel_for`) stays in the layer-1 module — `kvcache` never
  links `parallel` — and the physical layout stays `kvcache`-private behind a
  raw-pointer signature (M12/M13 can change it without touching the wrapper).
- **`view` exposes a per-layer visible length.** Layer 0 grows the table (bumping
  `length()`) while layers `≥ next_layer_` have their new slots allocated but
  unwritten mid-forward. So `view(layer)` gathers only the layer's *committed*
  length: `length()` when the forward is complete (`next_layer_ == 0`) or `layer <
  next_layer_`, else `length() − pending`. This reproduces `SimpleKvCache::view`'s
  per-layer-fill semantics bit-for-bit — a not-yet-appended layer sees the prior
  committed history, never stale slot bytes — for one integer compare. Real
  consumers always `view(layer)` right after `append(layer)`, so they always see
  the full length; the rule only matters for tests that view an un-appended layer.
- **Fresh tensor, not a workspace slot.** The `KvView` owns its storage, so the
  gather allocates per call; a workspace-resident prefill gather buffer is a
  T07/M12 option (decode never gathers — it reads `paged_view` zero-copy, §8.3).
- **The reference-backend single-layer attention golden test is not paged.** The
  `attention_test` `MatchesFixtureGolden` cases append to one layer in isolation
  (some at layer 1), which the paged layer-0-first protocol rejects by design.
  Reference-backend paged coverage lives in `model_test.cpp` instead (full model,
  layers appended in order — the protocol's natural shape).

**Tests (+19 → 1149 → 1168 ctest green).**

- `kv_gather_test` (kernels, no SCALAR_PASS): the kernel vs an independently
  simulated paged layout — a **reverse-permuted** block table with **poisoned**
  unused/tail slots (a stray read returns poison) — across `bs ∈ {8,16,32,64}` ×
  lengths {1, bs−1, bs, bs+1, 2bs, 3bs+5} × Hkv {1,2,4} × d {18,24,64}, empty
  length (null pointers, proven no-op), thread-count invariance, exact NaN/−0.0/
  denormal bit patterns, and a **scatter→gather round trip** (`KvScatterF32` with
  a permuted mapping then gather == identity) that proves the M8-T04 write and
  M8-T06 read kernels against each other.
- `paged_cache_test`: replaced the `ViewIsUnimplementedUntilT06` stub with view
  gathers matching `Expect()` (and agreeing with `paged_view`), empty-cache
  zero-length view, bad-layer rejection, a fresh-snapshot check (a later append
  does not mutate an earlier view), the **per-layer mid-forward visible length**,
  and view-after-truncate.
- `optimized_model_test` (model, SCALAR_PASS): **paged prefill-continuation**
  (chunk A then chunk B, `P > 0`) on the optimized backend is **bit-exact** to the
  same two-chunk run on `SimpleKvCache` (max_abs_diff 0 — identical K/V bytes into
  identical kernels) and within tolerance of a one-shot reference-backend full
  prefill (**1.8e-7**), both fixtures; the **KV invariant** (token-by-token ==
  full prefill) bit-exact through the paged cache; a paged greedy loop matching
  the simple-cache greedy (transitively `generate.json`).
- `model_test` (model): **reference-backend** paged prefill-continuation
  bit-exact vs simple + reproducing the full-prefill last-row logit, and the paged
  KV invariant.

**Docs.** paged-kv-cache.md §2 table (kv_gather + paged_gather rows) and §9.3
"as built" block. No BASELINES entry (no perf claim this ticket; decode still
routes through `view()` on a paged cache until T07 swaps in `paged_view`).
Format + scoped tidy clean (paged_cache.h header edit → its includer set —
paged_cache.cpp, paged_gather.cpp, paged_cache_test, paged_decode_attention_
kernel_test, optimized_model_test, model_test — was tidy-swept; kv_cache.h
untouched, so no full-tree sweep).

### M8-T07 — Engine integration (2026-08-19)

Swapped the paged cache into the single-request generation path behind the
unchanged M5 `KvCache` interface, wired the `--kv-cache-memory` capacity config,
and exposed cache stats. Per paged-kv-cache.md §5, §8.3, §10.

**The decode fast path (the only consumer-code change).**
`OptimizedModel::ForwardLayer`'s decode branch (`T == 1`) now calls
`request.cache->paged_view(layer_index)` first and runs
`kernels::PagedDecodeAttentionF32` on the returned slabs + block table (zero
copy, §8.3); on `Unimplemented` it falls back to `view()` + the contiguous
`DecodeAttentionF32`, so `SimpleKvCache` is untouched, and a non-`Unimplemented`
status **propagates** (a real cache failure is not masked into a fallback).
`view()` (a full-history gather) is now called only in the prefill branch
(`T > 1`) and the decode fallback — the paged decode step pays no per-step
gather, which is what keeps it under the ≤10% regression bound. Nothing else
above the `KvCache` interface changed. `ReferenceModel`/`Attention` untouched
(the oracle stays the oracle). `model → kernels` stays PRIVATE (added
`kernels/paged_attention.h` to `optimized_model.cpp` only).

**Capacity config (pure, unit-tested helpers).**
- `src/core/sysinfo.{h,cpp}` — `host_memory_bytes()` (§5.3 host-RAM helper:
  macOS `sysctl hw.memsize`, Linux `_SC_PHYS_PAGES · _SC_PAGE_SIZE`, else 0),
  the fraction spelling's first consumer (removed from paged-kv-cache.md §13
  deferred list). Added to `engine_core`.
- `src/kvcache/kv_budget.{h,cpp}` — `ParseKvCacheMemory` (absolute `2GiB`/
  `1500MiB`/`8000000000` with binary+decimal units, or a fractional `0.6`;
  disambiguation: unit suffix ⇒ absolute, bare value with `.` ⇒ fraction, bare
  integer ⇒ bytes; rejects out-of-range/garbage naming the input),
  `ResolveKvBudgetBytes` (absolute passes through; fraction computes
  `f·host_ram − weights − workspace`, host-RAM-unknown → `FailedPrecondition`,
  non-positive → `ResourceExhausted` naming all three terms — §5.3), and
  `BlocksForTokens` (`⌈tokens/bs⌉`, the `--cache-capacity` spelling). Added to
  `engine_kvcache`; depends only on `core` (no `kvcache → kernels` needed).
- `src/model/loader.{h,cpp}` — `weight_resident_bytes(loaded)` sums the distinct
  weight tensors' `numel × itemsize`, deduplicating the tied embedding by
  storage pointer (backend-independent, computed at the driver **before**
  `BuildModel` moves `loaded`). Chosen over a new `Model::memory_footprint`
  virtual: the checkpoint bytes are the authoritative weight footprint for both
  backends and are available pre-build, so no interface widening was needed.
- `src/model/workspace.{h,cpp}` — `Workspace::BytesFor(config dims, t)` static
  (the §6.2 formula for the `workspace_bytes` term without instantiating a
  workspace); `bytes()` refactored onto a shared file-local helper.

**Driver & bench.** `engine generate` (`src/main.cpp`) gained `--kv-cache
{paged,simple}` (default **paged**), `--kv-cache-memory SPEC`, and
`--kv-block-size N` (default 16). It owns `unique_ptr<BlockPool> pool` (declared
first) and `unique_ptr<KvCache> cache` (destroyed first, the §6.1 lifetime
rule), sizes the pool via the helpers above, and logs `BlockPool::stats()`
(blocks used/free/total, utilization) after generation — both `LOG_INFO` and a
printed summary line. Flag-parsing and cache-construction were split into
`ParseKvArg`/`ResolvePagedBlocks`/`BuildKvCache` helpers to keep each function
under the clang-tidy cognitive-complexity threshold. `bench_generate` got the
same `--kv-cache`/`--kv-block-size`; `DoRun` takes a `KvCache&`, so paged and
simple run the identical harness (the A/B). New `bench_generate_smoke_simple`
ctest keeps the contiguous path exercised.

**CLI default divergence (documented, paged-kv-cache.md §5.1 as-built).** With
neither `--kv-cache-memory` nor `--cache-capacity` given, the CLI sizes a
**token-sized pool** (`⌈(prompt + max_new)/bs⌉` blocks), not the design's `0.9`
fraction: a single-request generation needs only its own worst case, and a
`0.9` default would zero-fill ~13 GB on the 16 GB dev box (and in smoke runs).
`0.9` remains the M9 server's default. **Allocator:** the single-request pool
draws slabs from the default CPU allocator, not the M2 caching allocator (which
amortizes churn and buys nothing for one alloc/free) — the caching allocator
wires in with the M9 runtime.

**Acceptance.** (a) Tiny-fixture greedy **identical** to pre-paging on both
fixtures — the existing `PagedGreedyMatchesSimpleCache` / paged-KV-invariant
tests (now routing decode through `paged_view`) and the unchanged `generate.json`
goldens all pass. (b) Qwen2-0.5B-Instruct generates coherent text on both cache
kinds with **byte-identical** output (`The capital of France is … Paris …`) and
cache stats logged. (c) `bench_generate` decode **1.6–4.2%** below the
same-machine `--kv-cache simple` baseline (BASELINES.md M8-T07), inside the
≤10% bound and inside the machine's run-to-run noise — the `paged_view` fast
path offsets the ~11% isolated paged-kernel cost (M8-T05) by not gathering.

**Tests (+~30).** New `kv_budget_test` (20 cases: parse units/fractions/
rejections, resolution arithmetic incl. the three-term exhaustion message, the
host-RAM-unknown `FailedPrecondition`, `BlocksForTokens`, and
`core::host_memory_bytes() > 0`). New `optimized_model_test` cases (both
fixtures, SCALAR_PASS): `PagedDecodeStepMatchesSimpleDecodeStep` (single decode
step through `paged_view` bit-exact to the contiguous `view()` path),
`PagedViewErrorPropagatesNotFallback` (a decorator injecting an `Internal`
`paged_view` → forward propagates it), `WeightResidentBytesDeduplicatesTiedEmbedding`,
and `WorkspaceTest.BytesForMatchesInstantiated`. 1195 ctest green.

**Docs.** paged-kv-cache.md §5.1 (CLI default + allocator as-built), §10.1 (the
consumer swap, the helpers, the allocator choice), §13 (host-RAM helper landed);
optimized-cpu-execution.md §8 (the M8-T07 consumer-swap landed note);
BASELINES.md M8-T07 (the A/B). Format clean; scoped tidy clean on all changed
TUs (loader.h/workspace.h edits were additive declarations — no perturbation of
existing call sites; the TUs consuming the new symbols were tidy-swept).

### M8-T08 — Exhaustion behavior & metrics (2026-08-19)

The pool-dry-mid-generation behavior pinned down as a documented, resumable
error, plus the cache-usage metrics API finalized. The error plumbing and RAII
reclamation already existed (§10.2): `BlockPool::Allocate` → `ResourceExhausted`
→ `BlockTable::AppendTokens` (all-or-nothing) → `PagedKvCache::append` →
`Model::forward` → `Generate`, and a dropped `PagedKvCache`/`BlockTable` returns
every block. This ticket added the metrics, the graceful-CLI reporting, the
contract docs, and the end-to-end tests that were missing.

**Metrics (`src/kvcache/block_pool.{h,cpp}`).** `BlockPoolStats` gained three
fields — `peak_used` (high-water mark of `used`, bumped in `Allocate` on
success), `exhaustions` (count of `Allocate` calls that returned
`ResourceExhausted`, one per drained append attempt), and `block_size` (so a
consumer holding only the stats prints token capacity as `total · block_size`).
The two counters are monotone since `Create` and maintained under the mutex the
block lifecycle already holds — two integer ops, off nothing's hot path — so
they survive a sequence's blocks being reclaimed and answer "did this pool ever
run dry?" after the fact. The move-ctor carries them.

**Contract (`src/engine/generator.h`, docs).** Split the `Generate` error
contract into two classes: *front-loaded* (raised before any forward, cache
untouched, nothing emitted — including the up-front `capacity()` check, which is
**exact** for a private paged cache so a run that cannot fit is rejected here);
and *mid-generation* (any `forward` failure once decoding is under way, including
a `ResourceExhausted` from a **shared** paged pool drained by another sequence
after the up-front check passed). A mid-generation failure propagates the
`forward` status; the already-produced tokens are not in the returned `StatusOr`
(no partial result beside a Status) but were every one delivered through
`on_token` (the partial-result channel), the failing forward wrote nothing
(front-loaded inside `forward`), and the cache holds a consistent
`cache.length()`-token prefix — so generation is **resumable** from the last
delivered token, reproducing the identical greedy trajectory (the KV invariant
as a resumability guarantee). This is the seam M9 preemption/resume and M10
streaming-with-cancel build on.

**Hardening (`src/kvcache/paged_cache.cpp`).** A failed layer-0 `AppendTokens`
now explicitly clears `pending_slots_` (it previously kept the prior forward's
stale mapping; harmless since `next_layer_` stays 0, but the post-failure state
is now unambiguous — a decode step can be retried once blocks free up).

**CLI (`src/main.cpp`).** `engine generate` now reports pool stats — used/free/
util, peak, exhaustions — on **both** the success and the failure path (a
`print_pool_stats` lambda), and on a mid-run failure prints the streamed tokens
followed by `generation stopped: <status>` and the pool state before propagating
the error, rather than bailing with a bare error after the streamed text.

**Tests (+12 → 1207 green).** `block_pool_test` (+4): `block_size` exposed,
`peak_used` high-water/never-decreases, `exhaustions` counts failed allocations
only, both counters survive full reclamation; concurrency stress asserts the
counters stay bounded. `paged_cache_test` (+2): mid-sequence block-boundary
exhaustion (committed prefix + `view`/`paged_view` intact, `exhaustions`
recorded, competitor-release recovery), and exhaustion-then-drop reclaiming all
blocks (stats zero, peak/exhaustions retained). `generator_test` (+2, reference
backend): the paged front-loaded rejection (nothing allocated), and the
mid-generation acceptance — a hog in the token callback drains the shared pool,
the run fails gracefully with `ResourceExhausted`, the delivered prefix matches
the uninterrupted trajectory, `cache.length()` is consistent, resume from the
last delivered token reproduces the full trajectory token-for-token, and pool
stats return to zero after drop. `optimized_model_test` (+2, SCALAR_PASS): the
pool `Allocate → ResourceExhausted` chain from inside a `forward` (a hook drains
the pool between the capacity check and the layer-0 append) propagating +
recovering, and paged greedy exhaustion reclaiming all blocks on the optimized
backend — the `engine generate` default, covered under forced-scalar too.

No perf claim (no BASELINES entry — the counters are two integer ops under an
already-held lock). Docs: paged-kv-cache.md §6.2 (metrics fields + as-built),
§10.2 (the M8-T08 as-built: front-loaded vs mid-generation, resumability, CLI
reporting), §12 (test bullet); model-execution.md §10 (mid-generation
`ResourceExhausted` note). Format clean; scoped tidy clean on the edited TUs and
the `block_pool.h`/`generator.h` header-includer sets.

## Milestone 9 — Continuous Batching Scheduler & Runtime

### M9-T01 — Design doc: request lifecycle, scheduler & runtime (2026-08-19)

Docs-only. Wrote **`docs/design/scheduler-runtime.md`** — the working contract
M9-T02…T10 (and the M10/M11/M12/M15/M16 interactions) build on: the runtime that
turns the M5–M8 single-request `engine::Generate` into a multi-request,
continuously-batched system (submission queue → single engine thread → per-request
channels), the pure scheduler that decides each step's prefill/decode/preempt set,
the ragged extension of the M5 `ForwardRequest`/`Model` contract, and the two
batched attention-kernel entries. The architectural-heart milestone, so the doc
carries the state machine, step-loop pseudocode, and explicit invariants the ticket
demands.

What the doc fixes (the non-obvious decisions):

- **Scheduler stays a `core`-only leaf (ADR-002 rule 4).** The single most
  important layout decision: `scheduler` sits beside `engine`, `runtime` mediates,
  and — crucially — since `Request`/`Sequence` live in `runtime` (*above* the
  scheduler), the scheduler cannot see them. It consumes plain descriptor structs
  the runtime fills (`PoolSnapshot{free_blocks, block_size, total}` + per-seq
  `{id, arrival_index, num_computed_tokens, num_prompt_tokens, blocks_held}`) and
  returns a plain `SchedulerOutput` of ids+lengths. ADR-002 would *permit*
  `scheduler → kvcache`, but the block arithmetic (`blocks_needed = ⌈(cur+add)/bs⌉
  − ⌈cur/bs⌉`) is reproduced from `block_size` alone, so the whole scheduler is
  table-testable with no pool/model/allocator. No new ADR edge anywhere in M9 —
  only edges already on the diagram.
- **Two passes per step, not a mixed forward** (the roadmap's "choose and
  justify"): one ragged prefill forward, then one batched decode forward. The two
  phases use different attention kernels (blocked-causal-gather vs paged-block-
  table); mixing buys nothing on CPU (no kernel-launch overhead to amortize);
  decode-priority is a *scheduling* property (every running seq decodes every
  step), not an execution-order one. Prefill runs first so a newly admitted
  sequence samples its first token (from prefill's `kLast` logits) the same step
  and joins the decode batch next step.
- **Batch-invariance guaranteed bit-for-bit on a fixed ISA** — the continuous-
  batching correctness invariant (N concurrent == N sequential, M9-T08). Every
  forward op is row-local (embedding/RMSNorm/SiLU/add/RoPE) or sequence-local
  (attention: each seq attends only its own cache); the one cross-row op, GEMM, is
  bit-identical to the single-row GEMV per row — *verified today* by
  `packed_gemm_test.GemvMatchesGemmRow` (`EXPECT_EQ` every row). Batched sampling
  is bit-identical by construction (M7-T06's shared `detail::SampleRow`). This
  pre-answers **M17-T04**'s open "is batch-invariance guaranteed?" — yes on a fixed
  ISA, Class T across ISAs (orthogonal to batching).
- **Scheduling policy v1:** decode-first (unconditional up to memory → no
  starvation), preempt the **latest-arrived** running sequence to fit the decode
  set (documented victim; least sunk cost), then admit WAITING FCFS under
  `max_num_seqs` / `max_num_batched_tokens` / block availability. Admission
  reserves **prompt blocks only** — growth is handled by the per-step decode check
  + preemption (optimistic admission, preemption as the safety valve), the
  trade-off stated for M11/M12 to revisit.
- **Preemption = evict-and-recompute**, resting on the M8-T08 resumable-error seam
  promoted to a scheduler action: free the victim's blocks (`BlockTable::FreeAll`),
  keep `generated_ids`+sampler+stop state, requeue at the **head** of WAITING, and
  on resume re-prefill `prompt ++ generated` into a fresh cache — bit-identical
  next token by the KV invariant (M8-T08 proved `delivered ++ resumed ==
  uninterrupted`). No swap-out: on CPU host RAM *is* the pool. Liveness argued: the
  oldest-alone sequence always fits `max_model_len` (config-guaranteed) so it is
  never preempted → forward progress every step.
- **Per-request failure isolation** (M9-T10): a sampler edge case or a per-sequence
  `forward` error fails only that sequence (FAILED + close channel + free blocks);
  the loop survives (ADR-003 — request data is recoverable `Status`, never
  `CHECK`). Flagged a required additive change: `BatchedSampler::Sample` must return
  a **per-row** status (today it returns only the lowest-index error and leaves
  `out` unspecified) so one bad row does not discard the batch.
- **Full state machine** (WAITING→RUNNING→(PREEMPTED)→FINISHED, plus CANCELLED/
  FAILED) with an exhaustive legal-transition table enforced by one
  `Sequence::Transition` that `CHECK`s illegal (from,to) pairs (M9-T02's "illegal
  transition = CHECK"); **step-loop pseudocode**; per-request **unbounded** SPSC
  channel (mutex+condvar, backpressure deferred to M10); the **explicit invariant
  list** (block sufficiency, single mutator, batch invariance, streaming fidelity,
  no leaks, resume equivalence, ≤1-step cancel, FCFS-among-equals) — the three
  acceptance criteria.

Also planned in the doc: the additive `ForwardRequest` fields (`cu_seqlens`,
per-seq `caches`), per-sequence K/V append inside the batched forward (preserving
the M8 layer protocol + exhaustion seam), the two batched kernels
(`PrefillAttentionVarlenF32`, `PagedDecodeAttentionBatchedF32` — both looping the
unchanged per-sequence recurrence), staging-buffer reuse (discharging
optimized-cpu-execution.md §6.3's deferred workspace pre-sizing), config knobs
(`max_num_seqs`/`max_num_batched_tokens`/`max_model_len`), and the M10/M11/M12/M15/
M16 seams. Per-ticket testing strategy maps every M9-T02…T10 acceptance criterion
to a suite (mock-model stress for T03, tiny-pool forced preemption for T09, the
8-concurrent-vs-sequential identity test for T08).

Cross-doc edits in the same change: paged-kv-cache.md §9.4/§11 (pointers to the new
doc for batch assembly/admission/preemption); model-execution.md §5.4 (the batched
`ForwardRequest`/append/invariance pointer); optimized-cpu-execution.md §11 (the
scheduler/loop pointer + the §6.3 workspace-presizing discharge). No `src/` change;
1207 tests unchanged. Format clean (docs-only; no build/tidy delta).
