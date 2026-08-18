# tools/ — Python dev tooling

Python lives only here — never in the engine or its build (CLAUDE.md). CI
never runs anything in this directory: the fixtures these tools generate are
committed under `tests/fixtures/` and consumed by the C++ test suite as
ordinary test data.

## gen_fixtures — golden-fixture generation (M4-T02)

Regenerates every committed fixture in `tests/fixtures/`. Deterministic by
construction: fixed seeds, single-threaded CPU math
(`torch.use_deterministic_algorithms`), sorted-key JSON/safetensors
serialization, hub downloads pinned to commit hashes, no timestamps.
Running it twice produces byte-identical output.

### Setup (once)

Requires Python 3.11 (pinned in `tools/.python-version`; with pyenv it is
picked up automatically) and network access to huggingface.co:

```bash
cd tools
python -m venv .venv
.venv/bin/pip install -r gen_fixtures/requirements.txt
```

### Regenerating

```bash
tools/regen_fixtures.sh            # regenerate tests/fixtures/ in place
tools/regen_fixtures.sh --verify   # regenerate into a temp dir, diff against
                                   # the committed tree (byte-identity check)
```

Or per fixture family (each accepts `--out DIR`, default `tests/fixtures/`):

```bash
cd tools
.venv/bin/python -m gen_fixtures tiny-llama         # tiny random-weight Llama
                                                    #   checkpoint (single-file +
                                                    #   2-shard) + activation goldens
.venv/bin/python -m gen_fixtures tiny-llama-ops     # op-level goldens for the CPU
                                                    #   reference (M5): GEMM cases
.venv/bin/python -m gen_fixtures qwen2-names        # Qwen2 weight-name/shape
                                                    #   inventory (header-only fetch,
                                                    #   no weight download)
.venv/bin/python -m gen_fixtures tokenizer-vectors  # real tokenizer.json files +
                                                    #   encode/decode golden vectors
.venv/bin/python -m gen_fixtures model-configs      # real config.json files for
                                                    #   the config-parser goldens
```

### Review discipline

A fixture diff in review means a generator or pinned-version change that must
be explained — fixtures never drift on their own. When regenerating:

1. Change the generator (or bump `gen_fixtures/requirements.txt` /
   a pinned hub revision in `gen_fixtures/common.py`) deliberately.
2. Run `tools/regen_fixtures.sh`, then `--verify` to confirm determinism.
3. Commit the generator change and the fixture diff together, with the reason
   in the commit message.

Hub sources are pinned by commit hash in `gen_fixtures/common.py`; provenance
and licensing are recorded in `tests/fixtures/README.md`. No HF account is
needed (the Llama 3 tokenizer comes from an ungated byte-exact mirror; see the
fixtures README), though unauthenticated downloads are rate-limited — set
`HF_TOKEN` if that bites.

## gen_unicode — tokenizer Unicode tables (M4-T09)

Regenerates `src/tokenizer/unicode_data.inc` — the category ranges (\p{L},
\p{N}, White_Space) and NFC data (combining classes, canonical
decompositions, composition pairs, quick-check ranges) the tokenizer's
pre-tokenizer and normalizer consume (design: model-loading.md §6.3).
Stdlib-only (no venv needed): it downloads a *pinned* Unicode version's UCD
files into `gen_unicode/cache/` (gitignored), verifies their SHA-256
against the pins in `__main__.py`, and emits the tables with a
generation-stamp header. Same review discipline as gen_fixtures: the
committed tables never drift on their own; upgrading the Unicode version is
a deliberate regenerate-and-diff change validated against re-generated HF
vectors.

```bash
cd tools
python3 -m gen_unicode      # rewrites src/tokenizer/unicode_data.inc
```
