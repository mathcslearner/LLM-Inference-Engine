#!/usr/bin/env bash
# Regenerate every committed fixture in tests/fixtures/ (M4-T02).
#
#   tools/regen_fixtures.sh            regenerate in place
#   tools/regen_fixtures.sh --verify   regenerate into a temp dir and diff
#                                      against the committed tree (byte-
#                                      identity check; exits non-zero on drift)
#
# Requires the pinned venv: see tools/README.md.
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$TOOLS_DIR")"
PYTHON="${PYTHON:-$TOOLS_DIR/.venv/bin/python}"
FIXTURES_DIR="$REPO_ROOT/tests/fixtures"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: $PYTHON not found — create the venv per tools/README.md" >&2
  exit 1
fi

OUT_DIR="$FIXTURES_DIR"
VERIFY=0
if [[ "${1:-}" == "--verify" ]]; then
  VERIFY=1
  OUT_DIR="$(mktemp -d)"
  trap 'rm -rf "$OUT_DIR"' EXIT
fi

cd "$TOOLS_DIR"
echo "== tiny-llama =="
"$PYTHON" -m gen_fixtures tiny-llama --out "$OUT_DIR"
echo "== tiny-llama-ops =="
"$PYTHON" -m gen_fixtures tiny-llama-ops --out "$OUT_DIR"
echo "== tiny-llama-rope =="
"$PYTHON" -m gen_fixtures tiny-llama-rope --out "$OUT_DIR"
echo "== tiny-llama-attention =="
"$PYTHON" -m gen_fixtures tiny-llama-attention --out "$OUT_DIR"
echo "== tiny-llama-generate =="
"$PYTHON" -m gen_fixtures tiny-llama-generate --out "$OUT_DIR"
echo "== tiny-qwen2 =="
"$PYTHON" -m gen_fixtures tiny-qwen2 --out "$OUT_DIR"
echo "== tiny-qwen2-generate =="
"$PYTHON" -m gen_fixtures tiny-qwen2-generate --out "$OUT_DIR"
echo "== qwen2-names =="
"$PYTHON" -m gen_fixtures qwen2-names --out "$OUT_DIR"
echo "== tokenizer-vectors =="
"$PYTHON" -m gen_fixtures tokenizer-vectors --out "$OUT_DIR"
echo "== model-configs =="
"$PYTHON" -m gen_fixtures model-configs --out "$OUT_DIR"

if [[ "$VERIFY" == 1 ]]; then
  # README.md is hand-written, not generated.
  diff -r --exclude=README.md "$FIXTURES_DIR" "$OUT_DIR"
  echo "verify: regenerated fixtures are byte-identical to the committed tree"
fi
