#!/usr/bin/env bash
# Verifies that all C++/CUDA sources (tracked or not, minus .gitignore'd)
# match .clang-format, without
# modifying anything. Exits non-zero and names the offending files otherwise;
# run scripts/format.sh to fix. Used by CI (M0-T05).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/clang-tools.sh
source "${repo_root}/scripts/clang-tools.sh"

clang_format="$(resolve_clang_tool clang-format "${CLANG_FORMAT:-}")"
cd "$repo_root"

if ! engine_cxx_sources | xargs -0 "$clang_format" --dry-run --Werror; then
  echo "" >&2
  echo "Formatting check failed. Run scripts/format.sh to fix." >&2
  exit 1
fi
