# Third-party dependencies, fetched at configure time via FetchContent.
#
# Policy (full rationale and per-dependency notes in docs/dependencies.md):
#   - Every dependency is pinned to an exact release archive with a SHA256
#     hash — stronger than a git tag, which can be moved after the fact.
#   - Archives are declared SYSTEM so third-party headers are treated as
#     system includes: the project warning set (-Werror) never applies to
#     code we don't own. Dependency targets must never link engine_warnings.
#     (FetchContent's EXCLUDE_FROM_ALL option would also be nice here, but it
#     requires CMake 3.28 and the project floor is 3.26.)
#   - No find_package fallbacks: a clean clone must build with nothing
#     installed beyond the compiler toolchain, and every build must use the
#     exact pinned versions.
#
# Upgrading a dependency means changing the URL and SHA256 here and updating
# docs/dependencies.md in the same commit.

include(FetchContent)

# ---------------------------------------------------------------------------
# fmt 11.2.0 (MIT) — string formatting; also backs spdlog (see below).
# ---------------------------------------------------------------------------
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/11.2.0.tar.gz
  URL_HASH SHA256=bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af
  SYSTEM)

# ---------------------------------------------------------------------------
# spdlog 1.15.3 (MIT) — logging backend for src/core/logging.h (M0-T06).
# SPDLOG_FMT_EXTERNAL makes spdlog use our fmt instead of its bundled copy,
# so exactly one fmt version exists in any binary (no ODR hazard).
# ---------------------------------------------------------------------------
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz
  URL_HASH SHA256=15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67
  SYSTEM)

# ---------------------------------------------------------------------------
# nlohmann_json 3.12.0 (MIT) — config.json / tokenizer.json parsing (M3).
# ---------------------------------------------------------------------------
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
  URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
  SYSTEM)

# ---------------------------------------------------------------------------
# GoogleTest 1.17.0 (BSD-3-Clause) — unit/integration test framework
# (harness wired up in M0-T03). Only fetched when tests are built.
# ---------------------------------------------------------------------------
if(ENGINE_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    SYSTEM
    EXCLUDE_FROM_ALL)
endif()

set(ENGINE_DEPENDENCIES fmt spdlog nlohmann_json)
if(ENGINE_BUILD_TESTS)
  list(APPEND ENGINE_DEPENDENCIES googletest)
endif()
FetchContent_MakeAvailable(${ENGINE_DEPENDENCIES})
