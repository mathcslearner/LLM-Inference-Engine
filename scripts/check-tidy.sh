#!/usr/bin/env bash
# Runs clang-tidy (config: .clang-tidy) over all C++ translation units
# (tracked or not, minus .gitignore'd). Any diagnostic fails the script
# (WarningsAsErrors: '*').
#
# Needs a compile database; configures ${ENGINE_BUILD_DIR:-build} if missing
# (CMAKE_EXPORT_COMPILE_COMMANDS is ON project-wide). Headers are analyzed
# through the TUs that include them (HeaderFilterRegex in .clang-tidy).
# CUDA TUs (*.cu) are excluded until the CUDA toolchain lands in M2 —
# clang-tidy needs the CUDA headers from the compile database to parse them.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/clang-tools.sh
source "${repo_root}/scripts/clang-tools.sh"

clang_tidy="$(resolve_clang_tool clang-tidy "${CLANG_TIDY:-}")"
cd "$repo_root"

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

# xargs exits non-zero if any clang-tidy invocation reports a diagnostic.
# --cached --others --exclude-standard: tracked plus untracked files, minus
# anything .gitignore'd — new TUs are analyzed before they are staged.
if ! git ls-files -z --cached --others --exclude-standard -- '*.cc' '*.cpp' '*.cxx' |
  xargs -0 -P "$jobs" -n 1 \
    "$clang_tidy" -p "$build_dir" --quiet ${extra_args[@]+"${extra_args[@]}"}; then
  echo "" >&2
  echo "clang-tidy check failed (config: .clang-tidy)." >&2
  exit 1
fi
