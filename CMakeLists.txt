cmake_minimum_required(VERSION 3.26)

# De-duplicate link lines (CMake ≥ 3.29; guarded — 3.26 stays the floor).
# Once two dependency paths reach the same engine archive (e.g. a test
# linking engine::core directly and via engine::test_common), the OLD
# behavior repeats it on the link line and Apple's ld64 warns on every test
# link. NEW keeps static-link correctness and drops the repeats.
if(POLICY CMP0156)
  cmake_policy(SET CMP0156 NEW)
endif()

project(llm_inference_engine
  VERSION 0.1.0
  DESCRIPTION "CPU-first LLM inference engine for decoder-only transformers"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release for single-config generators when nothing was specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No CMAKE_BUILD_TYPE set; defaulting to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
    STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(ENGINE_BUILD_TESTS "Build unit and integration tests" ON)
option(ENGINE_BUILD_BENCHMARKS "Build microbenchmarks" ON)

# Sanitizer instrumentation (M3-T04; design: docs/design/cpu-backend.md §8.2).
# Applied globally — compile and link, third-party included — on purpose:
# under TSAN, mixing instrumented and uninstrumented code produces false
# positives, so GoogleTest must be instrumented too. That is why this is not
# on engine_warnings (which is scoped to project targets only). Must be set
# before dependencies are added so the FetchContent targets pick it up.
set(ENGINE_SANITIZE "" CACHE STRING "Sanitizer instrumentation (\"\" or \"thread\")")
if(ENGINE_SANITIZE STREQUAL "thread")
  add_compile_options(-fsanitize=thread -g)
  add_link_options(-fsanitize=thread)
elseif(NOT ENGINE_SANITIZE STREQUAL "")
  message(FATAL_ERROR
    "Unknown ENGINE_SANITIZE value \"${ENGINE_SANITIZE}\" (supported: \"\", \"thread\")")
endif()

# Third-party dependencies (pinned; policy in docs/dependencies.md).
include(cmake/dependencies.cmake)

# Project-wide warning set. Applied to project targets only: every engine
# target links `engine_warnings` PRIVATE, and third-party targets must never
# link it — that is how warnings-as-errors stays scoped to our own code.
add_library(engine_warnings INTERFACE)
set(ENGINE_WARNING_FLAGS
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wnon-virtual-dtor
  -Wold-style-cast
  -Werror)
target_compile_options(engine_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:${ENGINE_WARNING_FLAGS}>)

add_subdirectory(src)

# enable_testing() rather than include(CTest): we want test registration only,
# not the CDash dashboard targets (Experimental, Nightly, ...) CTest adds.
if(ENGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ENGINE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
