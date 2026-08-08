# Shared helper sourced by check-format.sh / format.sh / check-tidy.sh.
# Not executable on its own.
#
# Resolves a pinned LLVM tool (clang-format, clang-tidy) and refuses to run a
# mismatched major version: formatting output and check sets differ across
# releases, so an unpinned tool would make local results disagree with CI.
# Note that Apple's Xcode tools use their own version numbers (e.g. "Apple
# clang-format version 17") that do not correspond to upstream LLVM releases;
# they are rejected by the same version check. Install the upstream toolchain
# instead (macOS: `brew install llvm@20`; Debian/Ubuntu: apt.llvm.org).

# The single project-wide LLVM pin. Bumping it is a deliberate change: re-run
# scripts/format.sh over the tree and update CI in the same commit.
ENGINE_LLVM_MAJOR="${ENGINE_LLVM_MAJOR:-20}"

# resolve_clang_tool <tool-name> <env-override>
# Echoes the path of a <tool-name> whose major version matches the pin.
# Search order: explicit override, versioned name on PATH, Homebrew kegs,
# bare name on PATH. Exits with an install hint if nothing suitable exists.
resolve_clang_tool() {
  local tool="$1" override="${2:-}"
  local -a candidates=()

  if [[ -n "$override" ]]; then
    candidates+=("$override")
  else
    candidates+=("${tool}-${ENGINE_LLVM_MAJOR}")
    if command -v brew >/dev/null 2>&1; then
      local keg
      for keg in "llvm@${ENGINE_LLVM_MAJOR}" llvm; do
        local prefix
        if prefix="$(brew --prefix "$keg" 2>/dev/null)"; then
          candidates+=("${prefix}/bin/${tool}")
        fi
      done
    fi
    candidates+=("$tool")
  fi

  local candidate version
  for candidate in "${candidates[@]}"; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    # e.g. "clang-format version 20.1.7", "Ubuntu clang-format version 20...",
    # "Apple clang-format version 17.0.0" (rejected: Apple numbering).
    version="$("$candidate" --version 2>/dev/null |
      sed -n 's/.*version \([0-9][0-9]*\)\..*/\1/p' | head -n1)"
    if [[ "$version" == "$ENGINE_LLVM_MAJOR" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  echo "error: no ${tool} with major version ${ENGINE_LLVM_MAJOR} found." >&2
  echo "  Install it (macOS: brew install llvm@${ENGINE_LLVM_MAJOR};" >&2
  echo "  Debian/Ubuntu: https://apt.llvm.org), or point the" >&2
  echo "  ${tool//-/_} env var (upper-case) at a matching binary." >&2
  exit 1
}

# The directories containing the project's own C++/CUDA code. File listings
# are scoped to these so untracked non-project trees (the CI FetchContent
# dir .deps/, IDE build dirs) can never leak into format/tidy, whether or
# not they are gitignore'd.
ENGINE_SOURCE_DIRS=(src tests benchmarks)

# engine_project_files <ext...>
# NUL-delimited list (for xargs -0) of files with the given extensions under
# ENGINE_SOURCE_DIRS: tracked files plus untracked ones (--others), so new
# files are checked before they are staged; --exclude-standard keeps
# .gitignore'd trees (build*/) out. Git pathspec globs match recursively.
# Runs from the repository root (the entry-point scripts cd there).
# Deleted-but-not-yet-staged files still appear in `git ls-files --cached`,
# so filter to files that exist on disk — otherwise every tool run between
# an `rm` and the `git add` fails on the ghosts.
engine_project_files() {
  local -a pathspecs=()
  local dir ext file
  for dir in "${ENGINE_SOURCE_DIRS[@]}"; do
    for ext in "$@"; do
      pathspecs+=("${dir}/*.${ext}")
    done
  done
  git ls-files -z --cached --others --exclude-standard -- "${pathspecs[@]}" |
    while IFS= read -r -d '' file; do
      [[ -e "$file" ]] && printf '%s\0' "$file"
    done
  return 0
}

# engine_cxx_sources
# All C++/CUDA sources, headers included — the formatting surface.
engine_cxx_sources() {
  engine_project_files h hpp cc cpp cxx cu cuh
}
