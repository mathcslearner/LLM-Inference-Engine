"""Tokenizer fixtures: real tokenizer.json files + golden encode/decode vectors.

For each family (llama3, qwen2) this byte-copies the unmodified tokenizer.json
and upstream LICENSE at a pinned revision, then records vectors.json: for every
test case, ids with and without special tokens plus HF's decode output, all
produced by HF `tokenizers` under the pinned versions.

Malformed-UTF-8 cases are the one exception: HF's API rejects non-UTF-8 input
(`TextInputSequence must be str`), so no HF encode golden can exist. Those
cases carry "encode_synthetic": true and pin the engine's documented byte-
fallback semantics — each maximal invalid-byte run forms its own pre-token,
mapped through the GPT-2 byte alphabet and merged by the real BPE model
(tokenizer.model.tokenize, i.e. real vocab + merges). Inputs are shaped so
that invalid runs only ever appear standalone or string-final, where
segment-based and stream-based pre-tokenizer implementations provably agree.
The decode direction of these cases is a true HF golden.
"""

import base64
import unicodedata

from . import common

FAMILIES = {
    "llama3": (common.LLAMA3_REPO, common.LLAMA3_REVISION),
    "qwen2": (common.QWEN2_REPO, common.QWEN2_REVISION),
}

# (name, text) — shared valid-UTF-8 cases, per docs/design/model-loading.md §7.1.
COMMON_CASES = [
    ("empty", ""),
    ("ascii_simple", "Hello, world!"),
    ("ascii_sentence", "The quick brown fox jumps over the lazy dog."),
    ("nfc_composed", "café résumé naïve"),
    ("nfc_decomposed", unicodedata.normalize("NFD", "café résumé naïve")),
    ("cjk", "你好，世界！日本語のテキストです。한국어 조금."),
    ("emoji_simple", "I ❤️ tokenizers 🚀"),
    ("emoji_zwj", "👩‍👩‍👧‍👦 family, flag 🏳️‍🌈, wave 👋🏽"),
    ("whitespace_runs", "a  b   c\td\t\te   \t f"),
    ("leading_trailing_space", "   padded   "),
    ("newline_cluster", "line1\nline2\r\nline3\n\n\nline4\n"),
    ("tabs_only", "\t\t\t"),
    ("contractions_lower", "don't can't we're i'll it's you've"),
    ("contractions_upper", "DON'T CAN'T WE'RE I'LL IT'S YOU'VE"),
    ("contractions_mixed", "Don'T isn'T wE'Re"),
    ("digit_runs", "1 12 123 1234 12345 123456 1234567890"),
    ("digits_in_text", "pi is 3.14159, date 2026-08-08, v2.0.1"),
    ("punct_and_symbols", "foo_bar-baz.qux@example.com — «quotes» & #tags!"),
    ("mixed_scripts", "English中文mix日本語123abc"),
]

# Special tokens embedded mid-text differ per family.
FAMILY_CASES = {
    "llama3": [
        ("special_mid_text", "Hello <|end_of_text|> world"),
        ("special_chat_markers", "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"),
    ],
    "qwen2": [
        ("special_mid_text", "Hello <|endoftext|> world"),
        ("special_chat_markers", "<|im_start|>user\nhi<|im_end|>"),
    ],
}

# (name, raw bytes) — encode side is synthetic (see module docstring).
MALFORMED_CASES = [
    ("invalid_single_byte", b"\xff"),
    ("invalid_byte_run", b"\xff\xfe\xfd"),
    ("truncated_multibyte_tail", b"hi\xe2\x82"),
]


def _bytes_to_unicode() -> dict:
    """The GPT-2 byte→unicode bijection (mirrors the HF/OpenAI reference)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _special_frame(tokenizer) -> tuple[list, list]:
    """Prefix/suffix ids the post-processor adds around a single sequence."""
    with_special = tokenizer.encode("x", add_special_tokens=True).ids
    without = tokenizer.encode("x", add_special_tokens=False).ids
    assert len(without) == 1, "probe text 'x' must be a single token"
    idx = with_special.index(without[0])
    return with_special[:idx], with_special[idx + 1 :]


def _split_invalid_runs(raw: bytes) -> list:
    """Split into ('valid', str) and ('invalid', bytes) segments."""
    segments = []
    i = 0
    while i < len(raw):
        try:
            raw[i:].decode("utf-8")
            segments.append(("valid", raw[i:].decode("utf-8")))
            break
        except UnicodeDecodeError as err:
            if err.start > 0:
                segments.append(("valid", raw[i : i + err.start].decode("utf-8")))
            bad_start = i + err.start
            bad_end = bad_start + 1
            # Grow the invalid run over consecutive undecodable bytes.
            while bad_end < len(raw):
                probe = raw[bad_end:]
                try:
                    probe.decode("utf-8")
                    break
                except UnicodeDecodeError as probe_err:
                    if probe_err.start > 0:
                        break
                    bad_end += 1
            segments.append(("invalid", raw[bad_start:bad_end]))
            i = bad_end
    return segments


def _encode_malformed(tokenizer, byte_map: dict, raw: bytes) -> list:
    ids = []
    for kind, segment in _split_invalid_runs(raw):
        if kind == "valid":
            if segment:
                ids.extend(tokenizer.encode(segment, add_special_tokens=False).ids)
        else:
            mapped = "".join(byte_map[b] for b in segment)
            ids.extend(token.id for token in tokenizer.model.tokenize(mapped))
    return ids


def _build_vectors(tokenizer, family: str) -> list:
    byte_map = _bytes_to_unicode()
    prefix, suffix = _special_frame(tokenizer)
    cases = []
    for name, text in COMMON_CASES + FAMILY_CASES[family]:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        ids_with_special = tokenizer.encode(text, add_special_tokens=True).ids
        cases.append(
            {
                "name": name,
                "text": text,
                "ids": ids,
                "ids_with_special": ids_with_special,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
                "decoded_skip_special": tokenizer.decode(
                    ids_with_special, skip_special_tokens=True
                ),
            }
        )
    for name, raw in MALFORMED_CASES:
        ids = _encode_malformed(tokenizer, byte_map, raw)
        ids_with_special = prefix + ids + suffix
        cases.append(
            {
                "name": name,
                "text_b64": base64.b64encode(raw).decode("ascii"),
                "encode_synthetic": True,
                "ids": ids,
                "ids_with_special": ids_with_special,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
                "decoded_skip_special": tokenizer.decode(
                    ids_with_special, skip_special_tokens=True
                ),
            }
        )
    return cases


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tokenizer-vectors", help="tokenizer.json + golden encode/decode vectors"
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def run(args) -> None:
    import pathlib
    import shutil

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    for family, (repo, revision) in sorted(FAMILIES.items()):
        out = pathlib.Path(args.out) / "tokenizers" / family
        out.mkdir(parents=True, exist_ok=True)
        for filename in ("tokenizer.json", "LICENSE"):
            cached = hf_hub_download(repo, filename, revision=revision)
            shutil.copyfile(cached, out / filename)  # byte-copy, never re-serialized

        tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
        common.dump_json(
            {
                "repo": repo,
                "revision": revision,
                "generator_versions": common.generator_versions(),
                "cases": _build_vectors(tokenizer, family),
            },
            out / "vectors.json",
        )
        print(f"  wrote {out} ({repo}@{revision[:12]})")
