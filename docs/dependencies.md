# Third-party dependencies

All dependencies are fetched at configure time by CMake `FetchContent` from
[`cmake/dependencies.cmake`](../cmake/dependencies.cmake). A clean clone builds
with nothing installed beyond the compiler toolchain (and, from M2 on, the CUDA
toolkit). There are no `find_package` fallbacks: every build uses exactly the
pinned versions below.

## Current dependencies

| Dependency | Version | License | Role | Why this one |
|---|---|---|---|---|
| [fmt](https://github.com/fmtlib/fmt) | 11.2.0 | MIT | String formatting (`fmt::format`, `fmt::print`) | The de-facto standard C++ formatting library: type-safe, fast, and the API `std::format` was standardized from. Preferred over `std::format` because libstdc++/libc++ coverage is still uneven across our supported toolchains, and spdlog needs a fmt anyway. |
| [spdlog](https://github.com/gabime/spdlog) | 1.15.3 | MIT | Logging backend behind `src/core/logging.h` (M0-T06) | Mature, header-light, leveled/structured logging with compile-time level stripping — matches the M0-T06 requirement that hot-path log macros compile to nothing in Release. Built with `SPDLOG_FMT_EXTERNAL=ON` so it uses our pinned fmt rather than its bundled copy (one fmt per binary, no ODR risk). |
| [nlohmann_json](https://github.com/nlohmann/json) | 3.12.0 | MIT | Parsing `config.json`, `tokenizer.json`, safetensors headers (M3) | Best-in-class ergonomics for read-mostly JSON of small-to-moderate size, which is all the engine does with JSON (model/tokenizer metadata, API bodies). Parse throughput is not on any hot path, so ergonomics beat rapidjson-style speed. |
| [GoogleTest](https://github.com/google/googletest) | 1.17.0 | BSD-3-Clause | Unit/integration test framework + gmock (harness in M0-T03) | De-facto standard C++ test framework with first-class CTest discovery (`gtest_discover_tests`), rich assertions, and death tests (needed to test `CHECK` in M0-T07). Only fetched when `ENGINE_BUILD_TESTS=ON` (default `ON`). |

Exact pins live in `cmake/dependencies.cmake` as release-archive URLs with
SHA256 hashes. The versions above must stay in sync with that file.

## Policy for adding or upgrading a dependency

1. **Pin by content, not by name.** Fetch a release archive via `URL` with
   `URL_HASH SHA256=…`. Git tags are not acceptable pins on their own — they
   can be moved; a hash cannot.
2. **No system libraries.** Nothing may rely on a system-installed package
   beyond the compiler toolchain and the CUDA toolkit/NCCL (which are treated
   as toolchain components). No `find_package` fallback paths.
3. **Permissive licenses only** (MIT / BSD / Apache-2.0 class). Record the
   license in the table above. No copyleft in the engine or anything it links.
4. **Warnings stay ours.** Declare every dependency `SYSTEM` so the project
   `-Werror` warning set never applies to third-party code. Dependency
   targets must never link `engine_warnings`.
5. **Justify it here.** Every dependency gets a row in the table: version,
   license, role, and why it was chosen over alternatives. A dependency with
   architectural weight (e.g. an HTTP server library, an NCCL alternative)
   additionally needs an ADR in `docs/adr/`.
6. **Upgrades are atomic.** Changing a pin means updating the URL, the SHA256,
   and this document in the same commit, and building + running the full test
   suite before merging.
7. **Prefer fewer dependencies.** The bar for adding one is that writing and
   maintaining the code ourselves is clearly worse. Python dependencies for
   `tools/` are out of scope here — they never touch the engine build.
