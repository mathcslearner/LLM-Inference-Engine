#!/usr/bin/env bash
# Runs clang-tidy (config: .clang-tidy) over C++ translation units. Any
# diagnostic fails the script (WarningsAsErrors: '*').
#
# Usage: scripts/check-tidy.sh [tu...]
#   Without arguments: all project TUs (tracked or not, minus .gitignore'd)
#   — the CI mode. With arguments: only the listed TUs (.cc/.cpp/.cxx), for
#   fast scoped runs while iterating on a ticket. Headers cannot be passed
#   directly — they have no compile-database entry; pass a TU that includes
#   them (usually the ticket's test file) and HeaderFilterRegex in
#   .clang-tidy analyzes them through it.
#
# Needs a compile database; configures ${ENGINE_BUILD_DIR:-build} if missing
# (CMAKE_EXPORT_COMPILE_COMMANDS is ON project-wide). CUDA TUs (*.cu) are
# excluded until the CUDA toolchain lands in M2 — clang-tidy needs the CUDA
# headers from the compile database to parse them.
set -euo pipefail

invocation_dir="$PWD"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/clang-tools.sh
source "${repo_root}/scripts/clang-tools.sh"

clang_tidy="$(resolve_clang_tool clang-tidy "${CLANG_TIDY:-}")"
cd "$repo_root"

# Validate any explicit TU arguments before starting analysis, resolving
# paths relative to the invocation directory (the script itself runs from
# the repository root).
targets=()
for arg in "$@"; do
  path="$arg"
  [[ "$path" == /* ]] || path="${invocation_dir}/${path}"
  case "$path" in
    *.cc | *.cpp | *.cxx) ;;
    *)
      echo "error: ${arg} is not a translation unit (.cc/.cpp/.cxx)." >&2
      echo "  Headers are analyzed through the TUs that include them" >&2
      echo "  (HeaderFilterRegex); pass such a TU — usually the ticket's" >&2
      echo "  test file — instead." >&2
      exit 1
      ;;
  esac
  if [[ ! -f "$path" ]]; then
    echo "error: no such file: ${arg}" >&2
    exit 1
  fi
  targets+=("$path")
done

build_dir="${ENGINE_BUILD_DIR:-${repo_root}/build}"
if [[ ! -f "${build_dir}/compile_commands.json" ]]; then
  echo "No compile database in ${build_dir}; configuring..." >&2
  cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
fi

extra_args=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Homebrew clang-tidy does not know the Xcode SDK location; without this it
  # fails to find the C++ standard library headers.
  extra_args+=(
    "--extra-arg-before=-isysroot"
    "--extra-arg-before=$(xcrun --show-sdk-path)"
  )
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  jobs="$(sysctl -n hw.ncpu)"
else
  jobs="$(nproc)"
fi

# NUL-delimited TU list for xargs -0: the explicit arguments if given,
# otherwise every project TU. engine_project_files scopes the full listing
# to project source dirs (tracked plus untracked, minus .gitignore'd) — new
# TUs are analyzed before they are staged, and non-project trees like the CI
# FetchContent dir never are.
list_targets() {
  if ((${#targets[@]} > 0)); then
    printf '%s\0' "${targets[@]}"
  else
    engine_project_files cc cpp cxx
  fi
}

# xargs exits non-zero if any clang-tidy invocation reports a diagnostic.
if ! list_targets |
  xargs -0 -P "$jobs" -n 1 \
    "$clang_tidy" -p "$build_dir" --quiet ${extra_args[@]+"${extra_args[@]}"}; then
  echo "" >&2
  echo "clang-tidy check failed (config: .clang-tidy)." >&2
  exit 1
fi
