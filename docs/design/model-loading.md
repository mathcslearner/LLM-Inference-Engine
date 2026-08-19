# Model loading & tokenization

**Milestone:** M4 (design doc: M4-T01; implementation: M4-T02 … M4-T10)
**Governs:** `src/model/`, `src/tokenizer/`, `tools/gen_fixtures/`, and the
golden-fixture methodology (`tests/fixtures/`) every numerical test from M5
onward relies on.
**Cites:** ADR-002 (module boundaries, through Amendment 4), ADR-003 (error
handling), `docs/design/tensor.md` (Tensor/Buffer ownership, dtypes),
`docs/design/cpu-backend.md` §6 (the oracle chain this milestone's fixtures
sit at the root of), `docs/dependencies.md` (nlohmann_json).

This is the working contract for model loading and tokenization.
Implementation tickets must conform to it; if implementation reveals a design
flaw, this doc is updated in the same change with a note on what changed and
why (`docs/design/README.md`).

---

## 1. Scope & non-goals

After M4 the engine can take a local HuggingFace model directory — the
`config.json` + safetensors + `tokenizer.json` layout produced by
`huggingface-cli download` — and (a) materialize every weight as a CPU tensor
with checkpoint dtypes preserved, and (b) tokenize and detokenize text
byte-identically to HF `tokenizers`. This milestone also builds the
golden-fixture tooling: from here on, "correct" means "matches a committed
HuggingFace-generated fixture" (cpu-backend.md §6.1 — this milestone creates
the root of that oracle chain).

**In scope:** config parsing and the architecture registry; single-file and
sharded safetensors reading (mmap, zero-copy); the internal weight-naming
convention and per-architecture mapping tables; the load-time dtype policy;
byte-level BPE from `tokenizer.json` (encode, decode, incremental streaming
detokenization); the fixture-generation tooling in `tools/` and the fixture
layout, determinism, and size rules in `tests/fixtures/`.

**Non-goals (of this doc, not the project):**

- **No model execution.** Module structure, forward signatures, and KV-cache
  interfaces are M5-T01's doc. M4 ends at "weights are tensors in a registry."
- **No quantized checkpoint containers.** AWQ/GPTQ/INT8 formats are M12–M13;
  §5's dtype policy is written so they slot in without reshaping the loader.
- **No downloading.** Local paths only. Hub access, auth, and caching belong
  to user tooling (`huggingface-cli`), not the engine.
- **No sentencepiece / unigram tokenizers** — deferred with rationale in
  §6.1 (this is a scope decision the roadmap asks this doc to record).
- **No chat templates.** `tokenizer_config.json`'s Jinja template is consumed
  by `server` (M16). The tokenizer's job here is text ↔ ids.
- **No GGUF.** One checkpoint container (safetensors) until a concrete need
  appears (§9).

---

## 2. Module layout & layering

Two layer-2 modules land in M4. Per ADR-002 they may depend on layer 1
(`tensor`, `memory`, `parallel`, `kernels`, `cpu`) and `core`; they are
siblings and never include each other. No ADR amendment is needed — unlike
M3, every edge this milestone uses is already listed.

| Module | Links | Role |
|---|---|---|
| `model` (`src/model/`, ns `engine::model`) | `tensor`, `memory`, `core`; nlohmann_json (PRIVATE) | config.json → `ModelConfig`; safetensors → zero-copy `Tensor` views; weight-name mapping; `load_model` |
| `tokenizer` (`src/tokenizer/`, ns `engine::tokenizer`) | `core`; nlohmann_json (PRIVATE) | tokenizer.json → byte-level BPE encode/decode + streaming detokenizer |

JSON handling (both modules): nlohmann_json is an implementation detail —
**never** in a public header, linked PRIVATE. The build keeps exceptions
enabled, but per ADR-003 none may escape a module API: parsing uses the
non-throwing mode (`json::parse(first, last, /*cb=*/nullptr,
/*allow_exceptions=*/false)` → discarded value on syntax error) and the
exception-free accessor discipline (`contains`/`is_*` checks before reads,
no bare `operator[]`/`get<T>` on unchecked input). Malformed external input
is always a recoverable `InvalidArgument` naming the offending field/file —
never a CHECK and never a propagating exception (fuzz-ish tests in §8 hold
this line).

Files this milestone creates:

| File | Module | Contents | Ticket |
|---|---|---|---|
| `tools/gen_fixtures/` | tools | Python fixture generators + pinned requirements | M4-T02 |
| `src/model/config.h/.cpp` | model | `ModelConfig`, `Architecture`, parse/validate | M4-T03 |
| `src/model/mapped_file.h/.cpp` | model | read-only mmap wrapped as a shared `memory::Buffer` | M4-T04 |
| `src/model/safetensors.h/.cpp` | model | single-file parser, `{name → Tensor view}` | M4-T04 |
| `src/tensor/tensor.h` (addition) | tensor | `Tensor::from_buffer` factory (§3.5) | M4-T04 |
| `src/model/checkpoint.h/.cpp` | model | unified single-file / sharded interface | M4-T05 |
| `src/model/weight_map.h/.cpp` | model | canonical names, per-arch tables, load report | M4-T06 |
| `src/model/loader.h/.cpp` | model | `load_model(dir) → StatusOr<LoadedModel>` | M4-T07 |
| `src/tokenizer/tokenizer.h/.cpp` | tokenizer | public API, tokenizer.json parsing | M4-T08 |
| `src/tokenizer/unicode.h/.cpp` | tokenizer | categories, NFC, UTF-8 codec (§6.3); includes the generated `unicode_data.inc` | M4-T09 |
| `src/tokenizer/pretokenize.h/.cpp` | tokenizer | GPT-2/cl100k-family split matcher (§6.4) | M4-T09 |
| `src/tokenizer/bpe.h/.cpp` | tokenizer | byte-level alphabet (M4-T08), merge loop (M4-T09) | M4-T08/T09 |
| `src/tokenizer/detokenize.h/.cpp` | tokenizer | `DetokenizerStream` (§6.5) | M4-T10 |
| `tools/gen_unicode/` | tools | generator for `src/tokenizer/unicode_data.inc` (data tables only; the algorithms are hand-written in unicode.cpp) | M4-T09 |

`model` does not link `parallel`: v1 loading is single-threaded (mmap makes
materialization lazy — §3.4; parallel page-warming is deferred, §9).

Everything below assumes little-endian, as safetensors data is little-endian
and both supported platforms (arm64 macOS, x86-64 Linux) qualify; a
`static_assert(std::endian::native == std::endian::little)` in
`safetensors.cpp` turns a hypothetical big-endian port into a compile error
instead of silent garbage.

---

## 3. The load pipeline

```
model directory
   │
   ├─ config.json ──────────────► ModelConfig ──► architecture registry
   │                                   │              (kLlama, kQwen2)
   │                                   ▼
   ├─ weight discovery ─── single: model.safetensors
   │                       sharded: model.safetensors.index.json → shards
   │        │
   │        ▼
   │   mmap (read-only, shared memory::Buffer per file)
   │        │
   │        ▼
   │   safetensors header parse ──► {HF name → (dtype, shape, span)}
   │        │
   │        ▼
   │   weight-name mapping (per-arch table) ──► missing/unexpected report
   │        │
   │        ▼
   │   {canonical name → Tensor} registry  (zero-copy views, dtype preserved)
   │        │
   │        ▼
   └────► LoadedModel {config, weights, report}

tokenizer.json ──► Tokenizer (independent path, §6; engine composes in M5+)
```

### 3.1 Input contract

`load_model` takes a **directory** in the HF snapshot layout:

- `config.json` — required.
- Weights — exactly one of: `model.safetensors` (single file) or
  `model.safetensors.index.json` plus the shards it names. Both present →
  the index wins (matches HF loading order); neither → `NotFound` listing
  both spellings looked for.
- `tokenizer.json` — required for `Tokenizer::from_file`, but the *weight*
  loader neither reads nor requires it: `load_model` and the tokenizer are
  separate entry points composed by the engine (M5), so weight-only tests
  need no tokenizer fixture and vice versa.

Other snapshot files (`generation_config.json`, `tokenizer_config.json`,
`*.bin`) are ignored. PyTorch pickle checkpoints are out of scope
permanently (unsafe-by-design format; safetensors exists precisely to
replace it) — a directory with only `.bin` weights gets an actionable error
saying to convert.

### 3.2 Model config (M4-T03)

```cpp
// src/model/config.h
enum class Architecture { kLlama, kQwen2 };

struct RopeScaling {           // parsed here, interpreted in M5/M7
  std::string rope_type;       // "llama3", "linear", …
  float factor = 1.0f;
  float low_freq_factor = 1.0f;    // llama3-type only
  float high_freq_factor = 4.0f;   // llama3-type only
  std::int64_t original_max_position_embeddings = 0;
};

struct ModelConfig {
  Architecture architecture;
  std::string architecture_name;   // raw HF string, for messages
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  int num_layers = 0;              // HF: num_hidden_layers
  int num_heads = 0;               // HF: num_attention_heads
  int num_kv_heads = 0;            // HF: num_key_value_heads (GQA)
  int head_dim = 0;                // explicit or hidden_size / num_heads
  std::int64_t vocab_size = 0;
  float rope_theta = 10000.0f;
  std::optional<RopeScaling> rope_scaling;
  float rms_norm_eps = 1e-6f;
  std::int64_t max_position_embeddings = 0;
  bool tie_word_embeddings = false;
  bool attention_bias = false;     // per-arch default, see below
  tensor::DataType torch_dtype = tensor::DataType::kFloat32;
  std::vector<std::int32_t> eos_token_ids;  // int or list; empty when absent
};

[[nodiscard]] core::StatusOr<ModelConfig> ParseModelConfig(
    std::string_view json_text);                      // unit-testable seam
[[nodiscard]] core::StatusOr<ModelConfig> LoadModelConfig(
    const std::filesystem::path& path);               // reads + parses
```

Rules:

- **Architecture registry.** `architectures[0]` maps `"LlamaForCausalLM"` →
  `kLlama`, `"Qwen2ForCausalLM"` → `kQwen2`. Anything else →
  `Unimplemented` listing the supported strings. Adding an architecture =
  one registry entry + one §4 mapping table (+ fixtures); the roadmap's
  Llama-3/Qwen-2 focus is a floor, not a ceiling.
- **Unknown-field tolerance.** HF configs carry many keys we don't consume
  (`sliding_window`, `use_cache`, `bos_token_id`, …). Unknown keys are
  ignored silently — tolerance is what keeps minor upstream config churn
  from being a breaking event.
- **Required vs defaulted.** Required (error naming the field if absent or
  wrong JSON type): `architectures`, `hidden_size`, `intermediate_size`,
  `num_hidden_layers`, `num_attention_heads`, `vocab_size`. Defaulted when
  absent, matching HF `transformers` semantics: `num_key_value_heads` :=
  `num_attention_heads` (MHA), `head_dim` := `hidden_size / num_heads`,
  `rope_theta` := 10000, `rms_norm_eps` := 1e-6, `tie_word_embeddings` :=
  false, `torch_dtype` := float32. `attention_bias` defaults **per
  architecture** — false for Llama, true for Qwen2 (Qwen2's modeling code
  hardcodes QKV bias; its configs typically omit the field) — and an
  explicit JSON value wins over the default.
- **Validation** (all `InvalidArgument` naming field and value): every
  count/size strictly positive; `num_heads % num_kv_heads == 0`;
  `torch_dtype` must name a dtype we can represent —
  `tensor::from_string` already speaks HF spellings (`"bfloat16"`,
  dtype.h) by design. `hidden_size == num_heads * head_dim` is **not**
  enforced when `head_dim` is explicit (real checkpoints legitimately
  decouple them); it is the definition, not a constraint, when derived.
- **`rope_scaling`** is parsed into the struct above when present (null →
  `nullopt`). M4 stores it; interpretation (llama3 frequency scaling) is
  M5/M7's contract. Unknown `rope_type` values are *parsed*, not rejected —
  rejection would couple config parsing to execution capability; the
  executor rejects what it can't honor.
- **`eos_token_id`** (added M5-T09 for the generation loop) is optional and
  accepts HF's two serializations — a single int or a list of ints (Llama-3
  ships a list) — parsed into `eos_token_ids` (empty when absent or `null`).
  Each id is validated in `[0, vocab_size)`; a present field of any other
  shape, or a non-int element, is `InvalidArgument` naming `eos_token_id`
  (or `eos_token_id[i]`). The greedy loop stops on any of these ids; the
  caller may also pass its own stop set (e.g. the tokenizer's `eos_id()`).

*Implementation notes (M4-T03):*

- `max_position_embeddings` is **required** (it was in neither list above).
  It has no universal default — HF's class-level defaults differ per
  architecture (2048 for Llama, 32768 for Qwen2) — and both target families
  always serialize it, so a proper "missing field" error beats a
  per-arch default table or a confusing strictly-positive failure on 0.
- `rope_scaling`'s type key is read as `rope_type` falling back to the
  legacy spelling `type` (configs written before transformers 4.43, e.g.
  `{"type": "linear", ...}`) — transformers itself accepts both and
  normalizes to `rope_type`.
- When `head_dim` is *derived*, `hidden_size % num_heads != 0` is an error
  (truncating the definition would be silent nonsense); when explicit, no
  divisibility or product constraint applies, per the rule above.
- The committed config fixtures live under `tests/fixtures/models/configs/`
  (§7.1): the Llama side is deliberately **Llama-3.1** — its `rope_scaling`
  block exercises the presence path — and the Qwen2 side exercises absence
  plus the omitted-`attention_bias` default.

### 3.3 Weight discovery & mmap (M4-T04)

Files are mapped read-only and wrapped in the existing ownership vocabulary
rather than a new one:

```cpp
// src/model/mapped_file.h — maps `path` read-only (PROT_READ, MAP_PRIVATE);
// returns a Buffer whose deleter munmaps. Empty file → engaged zero-size
// Buffer (allocator.h convention). Errors: NotFound / by errno.
[[nodiscard]] core::StatusOr<std::shared_ptr<memory::Buffer>>
MapFileReadOnly(const std::filesystem::path& path);
```

This is exactly the `Buffer` contract from tensor.md §6: a self-contained
deleter, so the mapping lives until the last `shared_ptr` drops — and since
every `Tensor` holds a `shared_ptr<Buffer>` to its storage, **the file stays
mapped while any tensor view is alive** (M4-T04's lifetime acceptance
criterion) with no new lifetime machinery at all. The `Buffer` reports
`Device::Cpu()`.

The mapping is `PROT_READ`: weights are logically immutable, and a stray
write through `Tensor::data()` (which is mutable by design — tensor.md §8's
shallow const) faults loudly at the write site instead of silently
corrupting a checkpoint view. This is the documented behavior, not an
accident: model weights are the one tensor population that is never written.

v1 is POSIX `mmap` only — both supported platforms are POSIX. A Windows
port would add a `CreateFileMapping` branch behind the same function (§9).

### 3.4 Safetensors parsing (M4-T04)

Format recap (upstream spec, restated so the parser has a local contract):

```
[0..8)      u64 little-endian: header_len (JSON byte count)
[8..8+N)    UTF-8 JSON header, N = header_len
[8+N..EOF)  data section; per-tensor byte ranges are relative to 8+N
```

Header JSON: `{ "__metadata__"?: {str: str}, "<name>": { "dtype": "F16",
"shape": [d0, …], "data_offsets": [begin, end) }, … }`.

```cpp
// src/model/safetensors.h
class SafetensorsFile {
 public:
  [[nodiscard]] static core::StatusOr<SafetensorsFile> Open(
      const std::filesystem::path& path);

  [[nodiscard]] std::vector<std::string_view> names() const;  // sorted
  [[nodiscard]] bool contains(std::string_view name) const;
  // Zero-copy view into the mapping; shares the file's Buffer. NotFound
  // for unknown names.
  [[nodiscard]] core::StatusOr<tensor::Tensor> tensor(
      std::string_view name) const;
  // "__metadata__" if present (informational; never interpreted).
  [[nodiscard]] const std::map<std::string, std::string>& metadata() const;
};
```

**Validation posture.** External input, so every malformed shape is a
recoverable error with the tensor name and file path in the message
(`InvalidArgument` unless noted). Checks, in parse order:

1. file ≥ 8 bytes; `header_len` such that `8 + header_len ≤ file size`
   (guards both truncation and a hostile huge value — the arithmetic is done
   in `uint64` with explicit overflow checks, never trusted into a `size_t`
   cast first). A sanity cap of 256 MiB on `header_len` (reference
   implementation uses 100 MB) rejects garbage before JSON parsing is
   attempted.
2. header parses as a JSON object; every entry (except `__metadata__`) has
   exactly the keys `dtype` (string), `shape` (array of non-negative
   integers), `data_offsets` (2-array, `begin ≤ end`, both `≤` data-section
   size).
3. dtype string maps to a `tensor::DataType` (§5 table). Unknown-but-real
   safetensors dtypes (`F64`, `I16`, `U16`, `U32`, `U64`, `F8_E5M2`) →
   `Unimplemented`; unrecognized strings → `InvalidArgument`.
4. `end - begin == numel(shape) × itemsize(dtype)` exactly (kInt4 does not
   occur in plain safetensors; the sub-byte case is a quant-milestone
   concern).
5. Cross-tensor: ranges must be non-overlapping and, mirroring the
   reference implementation's strictness, gap-free — sorted by `begin`,
   consecutive ranges abut, and the last `end` equals the data-section
   size. Overlap is data corruption; a gap means bytes no tensor accounts
   for. Both reject the file.

Shape edge cases carried over from the tensor library: scalars
(`shape: []`) and zero-numel tensors are legal and map onto rank-0 /
zero-size `Tensor` views with the semantics tensor.md already defines.

**Alignment.** The mapping base is page-aligned, but a tensor's absolute
offset `8 + header_len + begin` has no alignment guarantee beyond what the
writer chose (the reference serializer pads the header so the *data section*
starts 8-aligned; individual tensors are only as aligned as their offsets
make them, and none of it is promised to readers). The parser therefore
assumes **nothing**: views may start at any byte. This is compatible with
cpu-backend.md §7 — kernels handle unaligned heads/tails; the 64-byte
*allocation* convention there applies to buffers we allocate, not to views
we borrow. If a future kernel measurably wants aligned weights, the remedy
is an explicit copy-to-aligned pass at load, listed in §9, not a parser
assumption. One real consequence today: an fp16/bf16 tensor whose offset is
odd would make `data_ptr<float16>()` UB-adjacent; in practice every dtype's
offsets in real checkpoints are itemsize-multiples, and the parser enforces
absolute alignment — `(8 + header_len + begin) % itemsize(dtype) == 0` (a
new check, stricter than the spec) — so the assumption is verified rather
than hoped. *(M4 audit, 2026-08-08: originally this checked only
`begin % itemsize`, which an unpadded header defeats — the whole data
section shifts, so a relatively-aligned tensor can still land on a
misaligned absolute address. The check now covers the absolute offset;
real HF serializers pad the header so the data section starts 8-aligned,
so no real checkpoint is affected.)*

### 3.5 Zero-copy tensor materialization: `Tensor::from_buffer` (M4-T04)

`Tensor`'s buffer-wrapping constructor is deliberately private (tensor.md
§7); `model` must not friend its way in. M4-T04 adds the missing public
factory to `src/tensor/tensor.h` — a general facility, not a safetensors
special case (tensor.md gets an implementation note in the same change):

```cpp
// Wraps existing storage as a contiguous row-major Tensor view. Validates
// byte_offset + numel×itemsize ≤ buffer size (InvalidArgument), rejects
// reserved dtypes (Unimplemented, consistent with empty()); null buffer is
// a programmer error (CHECK). Device comes from the buffer. The Tensor
// shares ownership: storage lives until the last handle dies.
[[nodiscard]] static core::StatusOr<Tensor> from_buffer(
    std::shared_ptr<memory::Buffer> buffer, std::size_t byte_offset,
    Shape shape, DataType dtype);
```

`SafetensorsFile::tensor()` is then: map dtype, build `Shape`, call
`from_buffer(file_buffer_, data_start + begin, shape, dtype)`. Materializing
a weight costs a header lookup and a handle copy — actual page-in happens
lazily on first touch, which is what makes load "instant" and §3.7's
progress logging honest about what it measures.

### 3.6 Sharded checkpoints (M4-T05)

`model.safetensors.index.json`: `{ "metadata": {"total_size": N},
"weight_map": { "<tensor name>": "<shard filename>" } }`.

```cpp
// src/model/checkpoint.h — the interface the rest of the loader consumes;
// single-file and sharded are indistinguishable behind it.
class Checkpoint {
 public:
  [[nodiscard]] static core::StatusOr<Checkpoint> Open(
      const std::filesystem::path& dir);   // resolves per §3.1 discovery
  [[nodiscard]] std::vector<std::string_view> names() const;
  [[nodiscard]] bool contains(std::string_view name) const;
  [[nodiscard]] core::StatusOr<tensor::Tensor> tensor(std::string_view name);
};
```

- The index is parsed eagerly (it is tiny); shard files are opened and
  mapped **lazily** on the first `tensor()` touching them, then cached in
  the `Checkpoint`. Shard filenames from the index are validated to be bare
  filenames (no path separators — a hostile index must not walk out of the
  model directory) and to exist at open time (existence is checked eagerly
  so "missing shard" surfaces at `Open`, not mid-load).
- Consistency checks at first map of each shard: every index entry pointing
  into that shard must exist in it, with the index and header agreeing the
  tensor exists (dtype/shape live only in the shard header; the index has
  no copy to cross-check). A name in a shard that the index doesn't mention
  is an inconsistency error, as is a `weight_map` name missing from its
  shard.
- Duplicate tensor names across shards: rejected at index parse (the JSON
  `weight_map` can't express it, but two shards *containing* the same name
  can — the per-shard check above catches it).

### 3.7 `load_model` (M4-T07)

```cpp
// src/model/loader.h
struct LoadedModel {
  ModelConfig config;
  // Canonical names (§4) → zero-copy CPU tensors, checkpoint dtypes.
  std::unordered_map<std::string, tensor::Tensor> weights;
  WeightMapReport report;   // §4: missing / unexpected / ignored
};

[[nodiscard]] core::StatusOr<LoadedModel> load_model(
    const std::filesystem::path& dir);
```

Stages, each with its failure mode: config (§3.2) → checkpoint open (§3.6)
→ weight map build (§4) → resolve every canonical name to a `Tensor`
(dtype policy §5 applied per weight) → report. Any *missing required
weight* fails the load (`InvalidArgument`, all missing names listed — not
just the first, so one round-trip fixes a broken checkpoint); unexpected
and ignored names are returned in the report and logged as warnings, never
errors.

Progress logging (real models are tens of GB even though page-in is lazy):
one info line per stage, plus per-shard map lines
(`mapped model-00003-of-00109.safetensors (4.9 GiB)`), via `core` logging.
No progress callbacks in v1 — the consumer is a CLI and logs suffice (§9).

---

## 4. Weight naming & per-architecture mapping (M4-T06)

**Canonical namespace.** Internal code (M5's modules, M6's optimized
backend) speaks these names exclusively; HF spellings appear nowhere outside
`weight_map.cpp`'s tables:

```
embed_tokens.weight
layers.{i}.attn_norm.weight
layers.{i}.attn.q_proj.weight      [+ .bias when attention_bias]
layers.{i}.attn.k_proj.weight      [+ .bias when attention_bias]
layers.{i}.attn.v_proj.weight      [+ .bias when attention_bias]
layers.{i}.attn.o_proj.weight
layers.{i}.mlp_norm.weight
layers.{i}.mlp.gate_proj.weight
layers.{i}.mlp.up_proj.weight
layers.{i}.mlp.down_proj.weight
final_norm.weight
lm_head.weight
```

Flat strings with the layer index inline; the registry is a plain map, not
a tree. `attn_norm`/`mlp_norm` name the *position* (pre-attention /
pre-MLP) rather than HF's `input_layernorm`/`post_attention_layernorm`,
which read as if the second norm followed attention's output — it precedes
the MLP.

**Per-architecture tables.** A table row is `{canonical pattern, HF
pattern, presence}` with `{i}` expanded from `config.num_layers`; presence
∈ {required, per-config (e.g. bias iff `attention_bias`), tied-alias}:

- **Llama** (`kLlama`): the exact list above with no biases in the default
  configuration (`attention_bias` defaults false for Llama). The bias rows
  are still per-config: a config that sets `attention_bias` adds biases on
  all four projections, o_proj included — HF's `LlamaAttention` biases
  o_proj, unlike Qwen2's. *(Clarified 2026-08-08, M4-T06: "no biases"
  described the default, not an architectural absence of the rows.)*
  HF prefix mapping `model.embed_tokens.weight → embed_tokens.weight`,
  `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight →
  layers.{i}.attn.*`, `model.layers.{i}.input_layernorm.weight →
  layers.{i}.attn_norm.weight`,
  `model.layers.{i}.post_attention_layernorm.weight →
  layers.{i}.mlp_norm.weight`, `model.norm.weight → final_norm.weight`,
  `lm_head.weight → lm_head.weight`.
- **Qwen2** (`kQwen2`): identical plus
  `model.layers.{i}.self_attn.{q,k,v}_proj.bias` (no o_proj bias),
  present because `attention_bias` defaults true (§3.2).

**Tied embeddings.** When `config.tie_word_embeddings`, canonical
`lm_head.weight` is a **tied-alias** of `embed_tokens.weight`: the registry
stores the *same* `Tensor` handle under both names (shared storage — free,
by tensor.md's shallow-copy semantics), the checkpoint is not required to
contain `lm_head.weight`, and if it does anyway the checkpoint copy is
*ignored* (alias wins, matching HF, which overwrites `lm_head` with the
embedding on load) and reported in `ignored`. Untied models require
`lm_head.weight` like any other weight.

**The report.**

```cpp
struct WeightMapReport {
  std::vector<std::string> missing;     // canonical names — load error
  std::vector<std::string> unexpected;  // HF names — warning
  std::vector<std::string> ignored;     // HF names matching the ignore list
};
```

- `missing`: required-or-per-config-present canonical names with no
  checkpoint tensor. Non-empty → `load_model` fails listing all of them.
- `ignored`: checkpoint names matching a small per-architecture ignore list
  of known non-weight artifacts (today: `model.layers.*.self_attn
  .rotary_emb.inv_freq`, emitted by older HF exports; plus the tied
  `lm_head.weight` case above). Logged at debug, not warning — they are
  expected.
- `unexpected`: everything else unclaimed. Warning list, never an error —
  a fine-tune with extra adapter tensors should load, loudly.

**Shape validation** happens here too (it needs config + mapping, and M5
should never see a malformed registry): every resolved weight's shape is
checked against its expectation from config — `embed_tokens
[vocab, hidden]`, `q_proj [heads×head_dim, hidden]`, `k/v_proj
[kv_heads×head_dim, hidden]`, `o_proj [hidden, heads×head_dim]`,
`gate/up_proj [intermediate, hidden]`, `down_proj [hidden, intermediate]`,
norms `[hidden]`, biases matching their projection's rows, `lm_head
[vocab, hidden]`. (HF linear weights are `[out_features, in_features]`;
that orientation is *recorded here* as the canonical storage convention M5
consumes.) Mismatch → `InvalidArgument` naming the weight, expected, and
actual.

---

## 5. Dtype policy at load

**Preserve the checkpoint dtype; convert at execution, never at load.**
A bf16 checkpoint loads as `kBFloat16` tensors bit-identical to the file
(they *are* the file — §3.4). M5's reference backend converts to fp32 on
the fly (cpu-backend.md §5's accumulation policy); M6+ kernels consume
fp16/bf16 directly. Load-time up-conversion would double memory for large
models and destroy the zero-copy property for zero benefit.

Safetensors dtype string → `tensor::DataType` (distinct from the
`torch_dtype` spellings `dtype.h` already maps — that table is HF config
names, this one is safetensors container names; both live next to their
format):

| safetensors | DataType | at M4, weights of this dtype… |
|---|---|---|
| `F32` | `kFloat32` | load |
| `F16` | `kFloat16` | load |
| `BF16` | `kBFloat16` | load |
| `I8` / `U8` / `I32` / `I64` / `BOOL` | `kInt8` / `kUInt8` / `kInt32` / `kInt64` / `kBool` | parse (the *parser* is format-complete) but the *loader* rejects them as weights: `Unimplemented`, "quantized/integer weights arrive in M12–M13" |
| `F8_E4M3` | `kFP8E4M3` | parse; loader rejects (reserved dtype until M13) |
| `F64`, `I16`, `U16`, `U32`, `U64`, `F8_E5M2` | — | `Unimplemented` at parse (no `DataType` exists; none appear in target checkpoints) |

The parser/loader split is deliberate: `SafetensorsFile` is a
general-purpose reader (fixtures use it for activation goldens, §7, where
`F32` activations and `I32`/`I64` token-id tensors are normal);
*weight-acceptance* policy belongs to `load_model`. `config.torch_dtype` is
advisory metadata — the per-tensor safetensors dtype is authoritative, and
no cross-check between them is performed (mixed-dtype checkpoints, e.g.
fp32 norms in an otherwise-bf16 model, are legal and real).

---

## 6. Tokenizer (M4-T08 … M4-T10)

### 6.1 Scope: byte-level BPE from `tokenizer.json` — and why

The engine's declared model scope is Llama-3-family and Qwen-2+-family
(CLAUDE.md). Both ship HF *fast tokenizer* `tokenizer.json` files whose
model is **byte-level BPE** (GPT-2 lineage: a 256-symbol byte alphabet, so
any UTF-8 input tokenizes with no unknown-token path). Supporting exactly
that single format covers the entire target scope with one implementation
and one golden methodology.

**Sentencepiece is explicitly deferred** (with unigram): it would serve
Llama-2-era models only, none of which are in scope; it is a second full
tokenizer model with its own normalization pipeline and its own golden
harness — the largest possible cost for zero in-scope coverage. A
`tokenizer.json` whose `model.type` is not `"BPE"` (or that is a
sentencepiece `.model` file, which isn't JSON at all) → `Unimplemented`
naming what was found and what is supported (M4-T08 acceptance).

### 6.2 `tokenizer.json` component whitelist

The HF `tokenizers` format is a pipeline description: `normalizer` →
`pre_tokenizer` → `model` → `post_processor`, plus `decoder` and
`added_tokens`. We implement the closure of what the two target families
use, and **reject anything else by name** (`Unimplemented`, naming the
component type) — silent partial support is how byte-identity dies:

| Stage | Supported | Notes |
|---|---|---|
| `normalizer` | absent/null; `NFC` | Llama 3: none. Qwen 2+: NFC (§6.3). Sequences of supported normalizers also accepted. |
| `pre_tokenizer` | `Split` (regex pattern, behavior `Isolated`, `invert: false`); `ByteLevel` (`add_prefix_space: false`, `use_regex: false`); `Sequence` of these | The regex is the GPT-2/cl100k family (§6.4). `ByteLevel` here does the byte→unicode mapping only. |
| `model` | `BPE`: `vocab`, `merges`, `ignore_merges`, `byte_fallback: false`, no `unk_token`, `continuing_subword_prefix`/`end_of_word_suffix` empty | `ignore_merges: true` (Llama 3): a pre-token already present in the vocab is emitted directly, skipping the merge loop. |
| `post_processor` | absent/null; `ByteLevel` (accepted and ignored); `TemplateProcessing` restricted to BOS/EOS insertion around the `single` sequence; `Sequence` of these | A `ByteLevel` post-processor only rewrites offsets, which we never produce — an encoding no-op. The `pair` template (`$B`) is *ignored*, not rejected — inference never encodes pairs (see the M4-T08 amendment below). |
| `decoder` | absent/null; `ByteLevel` (or a `Sequence` reducing to it) | Inverse byte↔unicode mapping — which is also what an absent decoder means for a byte-level model, so absent is accepted. |
| `added_tokens` | full support: id, content, `special`, `lstrip`/`rstrip`, `normalized`, `single_word` | The flags are parsed and *stored* in M4-T08; matching semantics used by encode are `special`, `lstrip`/`rstrip`, `normalized` (§6.4). `single_word` is stored but unimplemented — rejected if true, which neither target family sets. |

The exact component configuration of both target tokenizers is captured in
committed fixtures (§7), so "what the families actually use" is pinned by
data, not by this table's prose — if an upstream tokenizer revision adds a
component outside the whitelist, parsing fails loudly and the whitelist
grows deliberately.

> **Amendment (M4-T08, 2026-08-08).** The fixtures corrected this table's
> original prose in two places, exactly per the pinned-by-data rule above:
> the real Llama-3 `post_processor` is a `Sequence` of [`ByteLevel`,
> `TemplateProcessing`] and Qwen 2's is a bare `ByteLevel`, so `ByteLevel`
> (an encoding no-op) and `Sequence` joined the post_processor row; and
> Llama-3's `TemplateProcessing` *carries* a `pair` template alongside
> `single`, so "pair templates rejected" became "the `pair` key is ignored"
> — rejecting on its presence would have rejected Llama 3 itself. Two
> data-driven parsing notes from the same ticket: `merges` are accepted in
> both serializations (`"left right"` strings — unambiguous, the byte-level
> alphabet contains no space — and `["left", "right"]` arrays), and
> top-level `truncation`/`padding` must be null/absent (`Unimplemented`
> otherwise — either would change the id sequence).

### 6.3 Unicode machinery: generated tables, no ICU

Two pieces of real Unicode work hide in §6.2 and neither can be
approximated, because golden byte-identity is the acceptance criterion:

- **NFC normalization** (Qwen). Requires canonical decompositions,
  canonical combining classes, and the composition algorithm (with
  exclusions) — driven by Unicode data files.
- **Character categories** for the pre-tokenization pattern: `\p{L}`,
  `\p{N}`, and the pattern's whitespace semantics (§6.4).

Options considered: **ICU** (correct, but a huge dependency for two lookup
tables — violates the fewer-dependencies bar and its data versioning is
system-dependent, which breaks reproducibility); **hand-maintained
tables** (unmaintainable, unverifiable); **generated tables, committed as
source** — chosen. `tools/gen_unicode/` (Python, same pinned-requirements
regime as §7) downloads a *pinned Unicode version's* data files
(`UnicodeData.txt`, `DerivedNormalizationProps.txt`, …), and emits
`src/tokenizer/unicode_data.inc` (data only, included by the hand-written
`unicode.cpp` — see the §6.4 M4-T09 amendment): compact range tables + a
generation-stamp comment (Unicode version, generator hash). The tables are
committed like any other source; CI never runs the generator (fixture
rule, §7.2). Order
of magnitude: some tens of KB of tables — noise next to the fixtures.
The chosen Unicode version is pinned in the generator and stamped in the
output; upgrading it is a deliberate regenerate-and-diff change validated
against re-generated HF vectors.

Provided API (module-internal): `nfc_normalize(string_view) → string`
(with a fast is-ASCII/already-NFC skip path, since English text never
decomposes), `is_letter(char32_t)`, `is_number(char32_t)`,
`is_whitespace(char32_t)`.

### 6.4 Encoding (M4-T09)

`encode(text, add_special_tokens)`:

1. **Added-token split.** Scan for added-token *contents* (longest-match,
   e.g. via one pass of Aho–Corasick over the added vocabulary — a few
   hundred strings) — matched spans become their ids directly and are never
   BPE'd; `lstrip`/`rstrip` consume adjacent whitespace into the match per
   the stored flags; tokens with `normalized: false` (the typical special)
   match against the *raw* text, before normalization. This ordering —
   added tokens first — is the roadmap's "added tokens are matched before
   BPE" and matches HF.
2. **Normalize** each remaining text segment (§6.2's normalizer; identity
   for Llama 3, NFC for Qwen).
3. **Pre-tokenize:** split into pre-tokens with the pattern. The GPT-2 and
   cl100k/Llama-3 patterns are *fixed known regexes* over: case-marked
   contraction suffixes (`'s`, `'t`, `'re`, …; cl100k adds
   case-insensitivity), letter runs with optional leading non-letter byte,
   digit runs (GPT-2: unbounded; cl100k: 1–3 digits), punctuation runs with
   optional leading space, newline handling, and trailing-whitespace
   lookahead (`\s+(?!\S)`). **Decision: no regex engine.** `std::regex`
   cannot express `\p{L}`; RE2/PCRE2 is a heavyweight dependency for *two
   known patterns*; fancy-regex-compatible lookahead semantics would still
   need verifying against HF byte-for-byte. Instead `pretokenize.cpp`
   implements the pattern family as a hand-written matcher over decoded
   codepoints (categories from §6.3), parameterized by the small feature
   set above, selected by matching the `Split` pattern string from
   `tokenizer.json` against the known patterns — an unknown pattern string
   is `Unimplemented` (whitelist discipline), and the matcher's fidelity is
   pinned by the golden vectors, which exist precisely to catch
   lookahead/category edge cases.
4. **Byte-level map:** each pre-token's UTF-8 bytes map through the GPT-2
   byte→unicode bijection (the 256-entry `bytes_to_unicode` table —
   printable-latin identity, remapped controls) into a merge-alphabet
   string.
5. **BPE merge loop** per pre-token: if `ignore_merges` and the whole
   pre-token is in the vocab, emit its id; otherwise start from single
   symbols and repeatedly apply the lowest-rank adjacent merge until none
   applies (rank map from `merges`; a heap-driven implementation is fine
   but *any* tie-breaking beyond rank order must not exist — ranks are
   total). Result symbols → ids via `vocab`.
6. **Post-process:** when `add_special_tokens`, apply the
   `TemplateProcessing` BOS/EOS insertions (Llama 3 prepends
   `<|begin_of_text|>`; Qwen adds nothing).

API sketch:

```cpp
// src/tokenizer/tokenizer.h (as landed in M4-T08)
class Tokenizer {
 public:
  [[nodiscard]] static core::StatusOr<Tokenizer> from_file(
      const std::filesystem::path& tokenizer_json);
  // The testable seam, mirroring ParseModelConfig/LoadModelConfig.
  [[nodiscard]] static core::StatusOr<Tokenizer> from_json(
      std::string_view json_text);

  [[nodiscard]] core::StatusOr<std::vector<std::int32_t>> encode(
      std::string_view text, bool add_special_tokens) const;
  [[nodiscard]] core::StatusOr<std::string> decode(
      std::span<const std::int32_t> ids, bool skip_special_tokens) const;

  // Lookups over the parsed structures (M4-T08): base-vocab tokens in
  // alphabet space, added tokens by raw content; token_bytes is the raw
  // byte string an id decodes to (the §6.5 concatenation contract's unit).
  [[nodiscard]] std::int64_t vocab_size() const;
  [[nodiscard]] std::optional<std::int32_t> token_to_id(
      std::string_view token) const;
  [[nodiscard]] std::optional<std::string_view> id_to_token(
      std::int32_t id) const;
  [[nodiscard]] std::optional<std::string_view> token_bytes(
      std::int32_t id) const;
  [[nodiscard]] std::optional<std::int32_t> bos_id() const;
  [[nodiscard]] std::optional<std::int32_t> eos_id() const;
  [[nodiscard]] bool is_special(std::int32_t id) const;
  [[nodiscard]] const AddedToken* added_token(std::int32_t id) const;
};
```

Invalid UTF-8 *input* to `encode` is not an error: byte-level BPE is
total over bytes (step 4 consumes bytes, not codepoints; the
codepoint-level stages treat unpaired bytes per HF's behavior — pinned by
a malformed-input golden vector rather than prose). Out-of-range ids to
`decode` → `InvalidArgument`.

> **Implementation note (M4-T02).** "Per HF's behavior" turned out not to
> exist on the encode side: HF `tokenizers` rejects non-UTF-8 input at its
> API boundary (`TextInputSequence must be str` — verified against the
> pinned version), so no HF encode golden is possible. The malformed-input
> vectors therefore pin *this engine's* semantics, marked
> `"encode_synthetic": true` in `vectors.json`: each maximal invalid-byte
> run forms its own pre-token, is mapped through the byte→unicode table
> (step 4), and goes through the normal merge loop (step 5) with the real
> vocab/merges; valid segments around it encode normally. The committed
> cases place invalid runs only standalone or string-final — positions
> where segment-at-invalid-bytes and invalid-byte-as-uncategorized-
> codepoint pre-tokenizer implementations provably agree — so M4-T09 is
> free to implement either. The *decode* direction of these vectors is a
> true HF golden (decode of the resulting ids is well-defined in HF).

> **Amendment (M4-T09, 2026-08-08).** Decisions recorded as implemented:
>
> - **Matcher selection happens at parse time**, not first encode:
>   `from_json` matches the Split pattern string against exactly the two
>   fixture-pinned patterns (they differ only in the digit-run bound —
>   `\p{N}{1,3}` cl100k/Llama 3 vs `\p{N}` Qwen 2), so an off-whitelist
>   tokenizer fails at load. The matcher is parameterized by that single
>   knob (`SplitSpec.max_digit_run`).
> - **Invalid-byte strategy:** encode pre-splits each added-token-free
>   segment into maximal valid/invalid byte runs (the
>   segment-at-invalid-bytes option above); invalid runs bypass
>   normalize/pre-tokenize and go straight to steps 4–5. `nfc_normalize`
>   and `pretokenize` are nevertheless total over arbitrary bytes.
> - **Contraction case folding is ASCII-only.** Full Unicode simple
>   folding would additionally match exotica (ſ for s, K for k) inside
>   `(?i:...)`; no golden exercises that, and the goldens are the
>   contract. Documented in pretokenize.cpp.
> - **Unicode version pin: 16.0.0** (matches the tables in the pinned
>   `tokenizers` 0.22.2's Rust dependencies; the golden vectors and the
>   NFC unit cases — cross-checked against Python 3.13's unicodedata at
>   the same version — arbitrate). The generator emits
>   `unicode_data.inc` (data only) rather than the whole `unicode.cpp`
>   prose above suggested; the NFC/lookup algorithms are hand-written.
> - **BPE with an incomplete byte-level vocab** (a final merge symbol
>   missing — impossible for real byte-level files, which carry all 256
>   byte tokens) is `InvalidArgument` naming the symbol, not silent
>   omission.

### 6.5 Decoding & incremental detokenization (M4-T10)

Batch `decode`: ids → token strings (added tokens: raw content, dropped
when `skip_special_tokens` and `special`) → inverse byte-level map →
byte concatenation. The result of a lone token can be an *incomplete*
UTF-8 sequence — concatenation, not per-token validity, is the contract.

Streaming is where that bites, so `DetokenizerStream` owns the buffering:

```cpp
class DetokenizerStream {
 public:
  explicit DetokenizerStream(const Tokenizer& tokenizer,
                             bool skip_special_tokens);
  // Bytes decodable so far: everything buffered up to the last *complete*
  // UTF-8 sequence. Always valid UTF-8; possibly empty (mid-codepoint).
  [[nodiscard]] core::StatusOr<std::string> push(std::int32_t id);
  // End of stream: flush the residue. A trailing incomplete sequence
  // becomes U+FFFD (never raw malformed bytes).
  [[nodiscard]] std::string finish();
};
```

Invariant (M4-T10's acceptance criterion, restated as the contract): the
concatenation of all `push` outputs plus `finish` equals batch
`decode` of the same ids up to the U+FFFD policy on a malformed *tail*,
and every individual emission is valid UTF-8. The buffer holds only the
current incomplete sequence (≤ 3 bytes carried), so streaming adds O(1)
state per sequence — this type is what M9's streaming generation holds
per request.

> **Amendment (M4-T10, 2026-08-08).** Decisions recorded as implemented:
>
> - **Batch `decode` is not raw byte concatenation at its boundary.** HF
>   materializes the concatenated token bytes as a Rust `String` via
>   `from_utf8_lossy`: strict UTF-8 (overlong forms, surrogates, and
>   > U+10FFFF are invalid) with the WHATWG *maximal subpart* policy —
>   each longest prefix of a well-formed sequence that cannot be
>   completed becomes exactly one U+FFFD. The malformed-tail goldens pin
>   it (`FF FE FD` → three replacements; a truncated `E2 82` tail → one),
>   so `decode` concatenates and then applies that lossy conversion; its
>   output is always valid UTF-8. The classifier
>   (`classify_utf8_prefix`/`append_utf8_lossy`, unicode.h) is strict —
>   deliberately unlike the lenient `decode_utf8` the encode side uses —
>   and is shared with `DetokenizerStream`, which makes streaming output
>   *bit-identical* to batch decode: the "up to the U+FFFD policy" hedge
>   above turned out unnecessary, and the tests assert exact equality.
> - **Golden pairing:** `decoded` ↔ (`ids`, skip=false);
>   `decoded_skip_special` ↔ (`ids_with_special`, skip=true) — exactly
>   what the generator recorded.
> - **`finish` resets the carry**: a second `finish` returns `""` and the
>   stream is reusable for a fresh id sequence. A rejected (out-of-range)
>   `push` leaves the carried state untouched.
>
> **Consumer (M7-T04).** The generation loop's `StopChecker`
> (`src/engine/stop.{h,cpp}`, model-execution.md §15.2) owns one
> `DetokenizerStream` per request: it feeds each sampled id through `push`, then
> matches `stop_strings` on the emitted text (across token boundaries) and
> flushes the residue via `finish` at end of generation.

---

## 7. Golden fixtures

The load-bearing methodology (cpu-backend.md §6.1): HF fixtures validate
the scalar reference; the reference validates everything else. M4 creates
the root. Fixtures are **committed artifacts** — CI and every dev build
consume them from the repo and never run Python (M4-T02 acceptance).

### 7.1 Layout (`tests/fixtures/`)

```
tests/fixtures/
  README.md                       # regeneration commands, budget, licensing
  models/
    tiny-llama/
      config.json                 # LlamaForCausalLM, 2 layers, tiny dims
      model.safetensors           # single-file form
      sharded/                    # SAME tensors, 2-shard form (M4-T05)
        model.safetensors.index.json
        model-00001-of-00002.safetensors
        model-00002-of-00002.safetensors
      expected/
        activations.safetensors   # per-layer + end-to-end outputs (M5 consumes)
        meta.json                 # input ids, seed, generator versions
        ops.safetensors           # M5-T02: cpu::gemm goldens; M5-T03: cpu::
                                  #   rmsnorm/softmax/silu_mul/add; M5-T04:
                                  #   embedding_edge (+ ops_meta.json)
        rope.safetensors          # M5-T04: cpu::rope_apply / Rope goldens
                                  #   (+ rope_meta.json; model-execution.md §7)
        attention.safetensors     # M5-T05: cpu::attention / Attention goldens
                                  #   (+ attention_meta.json; model-execution.md
                                  #   §12)
        # M5 adds more op-level goldens alongside activations.safetensors —
        # generate.json — each landing with the ticket that consumes it
        # (model-execution.md §12).
    tiny-qwen2/                   # M5-T10: a tiny Qwen2ForCausalLM mirror of
      config.json                 #   tiny-llama — q/k/v biases (o_proj bias-free),
      model.safetensors           #   head_dim=24 (decoupled, != hidden/heads),
      expected/                   #   tied embeddings (lm_head dropped), theta 1e6.
        activations.safetensors   #   attention_bias omitted → parser default.
        meta.json                 #   No sharded copy (an M4 concern tiny-llama
        generate.json             #   already covers); greedy goldens like T09.
    qwen2-weight-names.json       # name inventory only — weight-map tests
    configs/                      # real config.json files — M4-T03 goldens
      llama3/  { config.json, LICENSE }   # Llama-3.1 (rope_scaling presence)
      qwen2/   { config.json }            # rope_scaling absence, bias default
  tokenizers/
    llama3/  { tokenizer.json, vectors.json, LICENSE }
    qwen2/   { tokenizer.json, vectors.json, LICENSE }
```

- **tiny-llama** (M4-T02): random-weight `LlamaForCausalLM`, ~2 layers,
  hidden ≈ 64, 4 heads / 2 KV heads, vocab ≈ 512, bf16 — well under 1 MB.
  Illustrative dims, pinned by the generator, not by this doc. The sharded
  copy duplicates it (~2×, still tiny) so single-file and sharded tests
  don't couple. `expected/` records layer-by-layer and end-to-end fp32
  outputs for a fixed input — stored as safetensors, which our own parser
  reads (§5's parser/loader split earning its keep), with `meta.json`
  carrying everything needed to regenerate byte-identically.
- **qwen2-weight-names.json**: the full HF weight-name inventory of a real
  Qwen2 checkpoint (names/shapes only, no bytes — a few KB) so M4-T06 tests
  Qwen2 mapping (biases, tied lm_head) without a second model fixture.
- **tokenizer fixtures**: the **real, unmodified** `tokenizer.json` of one
  Llama-3-family and one Qwen-2-family model, plus `vectors.json`:
  encode/decode goldens generated by HF `tokenizers` — ASCII; NFC-sensitive
  text (composed & decomposed forms of the same string); CJK; emoji
  including multi-codepoint ZWJ sequences (the streaming test's multi-token
  emoji); whitespace runs/tabs/newline clusters; contractions (upper and
  lower case); digit runs (where GPT-2 vs cl100k grouping differ); special
  tokens embedded mid-text; empty string; malformed UTF-8 (§6.4); each with
  ids for both `add_special_tokens` values and round-trip decoded text.

Trimming tokenizer vocabularies was considered and **rejected**: merges and
vocab are interdependent, so a trimmed file exercises a *different*
tokenizer, and byte-identity against the real artifact is the entire point.

### 7.2 Generation & determinism (M4-T02)

`tools/gen_fixtures/`: a Python package with a pinned `requirements.txt`
(exact `==` versions: torch, transformers, tokenizers, safetensors, and
huggingface_hub — the package that performs every pinned-revision
download) and a
CLI (`python -m gen_fixtures …`) with one subcommand per fixture family,
plus a `regen-all` target (Makefile or script) that rebuilds every
committed fixture. Rules:

- **Byte-identical regeneration** (M4-T02 acceptance): fixed seeds
  (`torch.manual_seed`), CPU-only generation, sorted/stable serialization
  (tensor names are passed to the safetensors serializer sorted; the
  library's header ordering — deterministic, dtype-major — is what lands
  on disk; JSON dumped with `sort_keys` and
  fixed separators), and no timestamps anywhere. Running the CLI twice
  diffs empty; the doc-of-record for "which versions produced this" is
  `meta.json` + `requirements.txt`, both committed.
- **CI never needs Python.** Fixtures are inputs to the C++ test suite like
  any other test data. Regeneration is a dev-machine act, reviewed like a
  code change (a fixture diff in review = a generator or version change
  that must be explained).
- `tools/README.md` documents setup (venv + pinned installs) and every
  subcommand (M4-T02 acceptance).

### 7.3 Size budget & licensing

Budget (enforced by review + a stated ceiling in `tests/fixtures/README.md`):

- per model fixture ≤ **5 MB**; per tokenizer file as-shipped (~7–9 MB for
  the target families); `tests/fixtures/` total ≤ **40 MB** working-tree.
  Expected M4 actual: ~20 MB working tree, and git's zlib packing takes
  JSON tokenizer files down ~3× (~8–10 MB repo growth) — acceptable for a
  clone-and-build repo; if a future milestone's fixtures threaten the
  ceiling, that milestone revisits storage (Git LFS was considered and
  rejected for v1: it breaks the "clean clone builds and tests with no
  extra tooling" property for ~megabytes of savings).
- Licensing: Qwen tokenizer files are Apache-2.0. Llama 3's tokenizer is
  distributed under the Llama Community License — redistribution is
  permitted with attribution/notice, so `tests/fixtures/tokenizers/llama3/`
  carries the upstream `LICENSE` file and `tests/fixtures/README.md` notes
  the provenance (exact upstream repo + revision). *Implementation note
  (M4-T02):* the `meta-llama` hub repo is gated (requires per-account
  license acceptance), so the committed files come from the ungated
  byte-exact mirror `NousResearch/Meta-Llama-3-8B-Instruct` at a pinned
  revision — which ships the same upstream `LICENSE`, committed alongside;
  the fixtures README records both repos. If license terms make
  committing any future artifact awkward, the fallback is a
  fetch-and-cache script — deliberately *not* chosen for M4 because it
  reintroduces network + Python into the test path this section exists to
  eliminate.

---

## 8. Testing strategy

Per cpu-backend.md §6.2's platform matrix: nothing in M4 is ISA-dependent —
no kernel work, no `SCALAR_PASS` registrations, no per-ISA concerns. These
are ordinary portable-C++ unit/integration tests, identical on arm64 and
x86-64. All external-input error paths assert `InvalidArgument`/
`Unimplemented`/`NotFound` *and* that the message names the offending
field/tensor/file (the actionability criterion, tested — not aspirational).

| Ticket | Tests |
|---|---|
| M4-T02 | Generator-side determinism is proven by the committed fixtures themselves (regen produces no diff — checked at generation time, documented in tools/README) |
| M4-T03 | Parse committed real Llama-3-style and Qwen2-style `config.json`; assert **every** `ModelConfig` field incl. per-arch `attention_bias` defaults, `head_dim` derivation, `rope_scaling` presence/absence; malformed cases: missing required field, wrong JSON type, negative dims, `num_heads % num_kv_heads != 0`, unknown architecture, non-JSON input |
| M4-T04 | Metadata (names/dtypes/shapes) spot-checked against the fixture; tensor bytes vs expected values; `from_buffer` bounds/reserved-dtype rejections; **lifetime**: drop the `SafetensorsFile`, read through a surviving `Tensor` (ASAN-clean = the shared-Buffer design working); fuzz-ish negatives: file < 8 bytes, header_len overflowing/exceeding file, header-cap breach, non-object JSON, missing keys, `begin > end`, offsets past EOF, numel×itemsize ≠ range, overlap, gap, unknown dtype string, unaligned `begin`, misaligned data-section start (unpadded header) |
| M4-T05 | 2-shard fixture: every tensor resolvable, values identical to the single-file fixture; missing shard file, index naming absent tensors, shard tensor absent from index, path-separator shard name, duplicate names across shards |
| M4-T06 | tiny-llama: every canonical name resolves, report empty; delete one weight → error naming it; add a stray → `unexpected` lists it; `inv_freq` → `ignored`; qwen2-weight-names.json: bias mapping + tied-`lm_head` aliasing (same `Tensor` handle under both names); shape-mismatch rejection |
| M4-T07 | Integration: `load_model(tiny-llama)` end-to-end, spot-check tensor values vs `expected/`; bad path, config-less dir, `.bin`-only dir, unsupported architecture — each message actionable |
| M4-T08 | Both fixture tokenizers: vocab size, specific token↔id pairs, special-token metadata (`special`, `lstrip`/`rstrip`); rejection: unigram-model json, sentencepiece binary, unknown pre_tokenizer/normalizer/post_processor types |
| M4-T09 | Golden encode equality vs `vectors.json` — every category in §7.1, both tokenizers, both `add_special_tokens` values (byte-identical id sequences) |
| M4-T10 | `decode(encode(x)) == x` for all vectors; `skip_special_tokens` both ways; streaming: token-by-token `push` over every vector — each emission valid UTF-8 (validated by a test-local checker), concatenation equals batch decode; the multi-token-emoji case; `finish` U+FFFD policy on a truncated id sequence |

Labels: `model` / `tokenizer` via `engine_add_tests(… LABELS)`, so
`ctest -L tokenizer` iterates the golden suite alone.

---

## 9. Deferred (known, intentionally not designed here)

- **Sentencepiece/unigram tokenizers** (§6.1) — first in line if a
  Llama-2-class model ever enters scope.
- **GGUF and quantized checkpoint containers** — M12–M13, with their own
  design doc; §5's parser/loader split is the seam they enter through.
- **Chat templates** (`tokenizer_config.json` Jinja) — `server`, M16.
- **Hub download / auth / cache** — permanently out; user tooling exists.
- **Copy-to-aligned weight materialization** — only if a kernel milestone
  measures a win (§3.4); would be an opt-in load mode, not a default.
- **Parallel load / page-cache warming** (`model → parallel` edge; readahead
  or touch-pass) — measure first; mmap laziness may be strictly better for
  time-to-first-token.
- **Windows file mapping** (§3.3) — behind `MapFileReadOnly` when a Windows
  port matters.
- **`single_word` added-token matching** (§6.2) — unused by target
  families; rejected loudly today.
- **Progress callbacks on `load_model`** (§3.7) — logs suffice until a UI
  consumer exists.
