"""Generates src/tokenizer/unicode_data.inc from pinned Unicode data files.

Design: docs/design/model-loading.md §6.3. The tokenizer needs two pieces of
real Unicode machinery — NFC normalization (Qwen's normalizer) and character
categories for the pre-tokenization pattern (\\p{L}, \\p{N}, \\s) — and both
must be byte-identical to what HF `tokenizers` computes. Rather than depend
on ICU (huge, system-versioned data) or hand-maintain tables (unverifiable),
this script downloads a *pinned* Unicode version's data files, verifies
their SHA-256, and emits compact C++ range tables that are committed as
source. CI never runs this script (fixture rule, §7.2); regenerating is a
deliberate, reviewed change.

Usage (stdlib only, no requirements):

    python3 -m gen_unicode [--cache-dir DIR] [--out FILE]

The default output path assumes the script runs from `tools/`.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

UNICODE_VERSION = "16.0.0"
BASE_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd"

# SHA-256 of the pinned data files. A mismatch means unicode.org served
# different bytes than this script was written against — never overwrite the
# committed tables from unverified input.
PINNED_FILES = {
    "UnicodeData.txt":
        "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f",
    "PropList.txt":
        "53d614508e2a0b2305a8aa21cd60d993de9326cdf65993660dfcce4503548583",
    "DerivedNormalizationProps.txt":
        "4d4c03892dea9146d674b686e495df2d55a28d071ac474041d73518f887abddc",
}

LETTER_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo"}
NUMBER_CATEGORIES = {"Nd", "Nl", "No"}

# Hangul syllables decompose/compose algorithmically (Unicode §3.12); they
# must not appear in the emitted tables.
HANGUL_S_BASE = 0xAC00
HANGUL_S_COUNT = 11172


def fetch(cache_dir: Path, name: str) -> str:
    """Returns the file's text, downloading into cache_dir if needed."""
    path = cache_dir / name
    if not path.exists():
        url = f"{BASE_URL}/{name}"
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PINNED_FILES[name]:
        raise SystemExit(
            f"{path}: SHA-256 {digest} does not match the pin "
            f"{PINNED_FILES[name]} for Unicode {UNICODE_VERSION}")
    return path.read_text(encoding="utf-8")


def parse_unicode_data(text: str):
    """UnicodeData.txt → (categories, ccc, canonical decompositions).

    Range entries come as paired lines named "<..., First>" / "<..., Last>";
    the pair shares one category (and never carries a decomposition or a
    non-zero combining class).
    """
    categories = {}  # cp -> general category
    ccc = {}  # cp -> non-zero canonical combining class
    decomp = {}  # cp -> tuple of cps (canonical decomposition, unexpanded)
    range_first = None
    for line in text.splitlines():
        fields = line.split(";")
        cp = int(fields[0], 16)
        name, category = fields[1], fields[2]
        if name.endswith(", First>"):
            range_first = cp
            continue
        if name.endswith(", Last>"):
            assert range_first is not None
            for c in range(range_first, cp + 1):
                categories[c] = category
            range_first = None
            continue
        categories[cp] = category
        combining = int(fields[3])
        if combining != 0:
            ccc[cp] = combining
        decomposition = fields[5]
        if decomposition and not decomposition.startswith("<"):
            decomp[cp] = tuple(int(part, 16)
                               for part in decomposition.split())
    return categories, ccc, decomp


def parse_property_ranges(text: str, wanted: str):
    """PropList-style file → sorted list of (lo, hi) for property `wanted`."""
    ranges = []
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = [part.strip() for part in body.split(";")]
        if len(fields) < 2 or fields[1] != wanted:
            continue
        span = fields[0].split("..")
        lo = int(span[0], 16)
        hi = int(span[-1], 16)
        ranges.append((lo, hi))
    return merge_ranges(ranges)


def parse_nfc_quick_check(text: str):
    """DerivedNormalizationProps.txt → ranges with NFC_QC = No or Maybe."""
    ranges = []
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = [part.strip() for part in body.split(";")]
        if len(fields) < 3 or fields[1] != "NFC_QC":
            continue
        assert fields[2] in ("N", "M"), fields
        span = fields[0].split("..")
        ranges.append((int(span[0], 16), int(span[-1], 16)))
    return merge_ranges(ranges)


def merge_ranges(ranges):
    ranges = sorted(ranges)
    merged = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def category_ranges(categories, wanted):
    cps = sorted(cp for cp, cat in categories.items() if cat in wanted)
    ranges = []
    for cp in cps:
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], cp)
        else:
            ranges.append((cp, cp))
    return ranges


def full_decompositions(decomp):
    """Recursively expanded canonical decompositions (Hangul excluded)."""
    cache = {}

    def expand(cp):
        if cp in cache:
            return cache[cp]
        parts = decomp.get(cp)
        if parts is None:
            result = (cp,)
        else:
            result = tuple(c for part in parts for c in expand(part))
        cache[cp] = result
        return result

    full = {}
    for cp in decomp:
        if HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT:
            raise SystemExit(f"unexpected tabular Hangul decomposition "
                             f"for U+{cp:04X}")
        full[cp] = expand(cp)
    return full


def composition_pairs(decomp, exclusions):
    """Primary composites: two-element canonical decomps, not excluded.

    Full_Composition_Exclusion already covers singletons, non-starter
    decompositions, and the script-specific exclusion list, so the only
    filters needed here are arity and the exclusion set itself.
    """
    pairs = {}
    for cp, parts in decomp.items():
        if len(parts) != 2:
            continue
        if any(lo <= cp <= hi for lo, hi in exclusions):
            continue
        key = (parts[0], parts[1])
        assert key not in pairs, f"duplicate composition pair {key}"
        pairs[key] = cp
    return pairs


def emit_ranges(out, name, ranges):
    out.append(f"constexpr CodepointRange {name}[] = {{")
    line = "   "
    for lo, hi in ranges:
        entry = f" {{0x{lo:X}, 0x{hi:X}}},"
        if len(line) + len(entry) > 80:
            out.append(line)
            line = "   "
        line += entry
    out.append(line)
    out.append("};")
    out.append("")


def generate(cache_dir: Path) -> str:
    categories, ccc, decomp = parse_unicode_data(
        fetch(cache_dir, "UnicodeData.txt"))
    whitespace = parse_property_ranges(
        fetch(cache_dir, "PropList.txt"), "White_Space")
    norm_props = fetch(cache_dir, "DerivedNormalizationProps.txt")
    exclusions = parse_property_ranges(norm_props,
                                       "Full_Composition_Exclusion")
    nfc_no_or_maybe = parse_nfc_quick_check(norm_props)

    full_decomp = full_decompositions(decomp)
    pairs = composition_pairs(decomp, exclusions)

    generator_hash = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()[:16]

    out = []
    out.append("// Generated by tools/gen_unicode — do not edit.")
    out.append(f"// Unicode version: {UNICODE_VERSION}")
    out.append(f"// Generator: gen_unicode/__main__.py "
               f"(sha256 {generator_hash})")
    out.append("//")
    out.append("// Included by src/tokenizer/unicode.cpp inside an anonymous")
    out.append("// namespace that defines CodepointRange, CccRange,")
    out.append("// DecompEntry, and CompEntry. Design:")
    out.append("// docs/design/model-loading.md §6.3.")
    out.append("")
    out.append("// clang-format off")
    out.append("// NOLINTBEGIN(modernize-use-designated-initializers) —")
    out.append("// generated data rows; field names would triple the file.")
    out.append("")

    emit_ranges(out, "kLetterRanges",
                category_ranges(categories, LETTER_CATEGORIES))
    emit_ranges(out, "kNumberRanges",
                category_ranges(categories, NUMBER_CATEGORIES))
    emit_ranges(out, "kWhitespaceRanges", whitespace)
    emit_ranges(out, "kNfcNoOrMaybeRanges", nfc_no_or_maybe)

    ccc_ranges = []
    for cp in sorted(ccc):
        if (ccc_ranges and cp == ccc_ranges[-1][1] + 1
                and ccc[cp] == ccc_ranges[-1][2]):
            ccc_ranges[-1] = (ccc_ranges[-1][0], cp, ccc_ranges[-1][2])
        else:
            ccc_ranges.append((cp, cp, ccc[cp]))
    out.append("constexpr CccRange kCccRanges[] = {")
    line = "   "
    for lo, hi, value in ccc_ranges:
        entry = f" {{0x{lo:X}, 0x{hi:X}, {value}}},"
        if len(line) + len(entry) > 80:
            out.append(line)
            line = "   "
        line += entry
    out.append(line)
    out.append("};")
    out.append("")

    pool = []
    pool_index = {}
    entries = []
    for cp in sorted(full_decomp):
        seq = full_decomp[cp]
        if seq not in pool_index:
            pool_index[seq] = len(pool)
            pool.extend(seq)
        entries.append((cp, pool_index[seq], len(seq)))
    assert len(pool) < 0x10000, "DecompEntry.offset must fit in uint16"
    assert max(length for _, _, length in entries) <= 0xFF

    out.append("constexpr char32_t kDecompPool[] = {")
    line = "   "
    for cp in pool:
        entry = f" 0x{cp:X},"
        if len(line) + len(entry) > 80:
            out.append(line)
            line = "   "
        line += entry
    out.append(line)
    out.append("};")
    out.append("")

    out.append("constexpr DecompEntry kDecompEntries[] = {")
    line = "   "
    for cp, offset, length in entries:
        entry = f" {{0x{cp:X}, {offset}, {length}}},"
        if len(line) + len(entry) > 80:
            out.append(line)
            line = "   "
        line += entry
    out.append(line)
    out.append("};")
    out.append("")

    out.append("// key = (uint64_t(starter) << 32) | combining")
    out.append("constexpr CompEntry kCompEntries[] = {")
    line = "   "
    for (first, second), composed in sorted(pairs.items()):
        key = (first << 32) | second
        entry = f" {{0x{key:X}, 0x{composed:X}}},"
        if len(line) + len(entry) > 80:
            out.append(line)
            line = "   "
        line += entry
    out.append(line)
    out.append("};")
    out.append("")
    out.append("// NOLINTEND(modernize-use-designated-initializers)")
    out.append("// clang-format on")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="gen_unicode",
                                     description=__doc__.splitlines()[0])
    default_out = (Path(__file__).resolve().parent.parent.parent / "src" /
                   "tokenizer" / "unicode_data.inc")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parent / "cache",
                        help="where UCD downloads are cached")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="output .inc path")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate(args.cache_dir), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
