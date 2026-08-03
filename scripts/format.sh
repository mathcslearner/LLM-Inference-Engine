#!/usr/bin/env bash
# Rewrites all tracked C++/CUDA sources in place to match .clang-format.
# Verify-only counterpart: scripts/check-format.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/clang-tools.sh
source "${repo_root}/scripts/clang-tools.sh"

clang_format="$(resolve_clang_tool clang-format "${CLANG_FORMAT:-}")"
cd "$repo_root"

engine_cxx_sources | xargs -0 "$clang_format" -i
