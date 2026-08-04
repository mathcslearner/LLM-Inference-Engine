cmake_minimum_required(VERSION 3.26)

# De-duplicate link lines (CMake ≥ 3.29; guarded — 3.26 stays the floor).
# Once two dependency paths reach the same engine archive (e.g. a test
# linking engine::core directly and via engine::test_common → engine::cuda,
# M2-T03), the OLD behavior repeats it on the link line and Apple's ld64
# warns on every test link. NEW keeps static-link correctness and drops the
# repeats.
if(POLICY CMP0156)
  cmake_policy(SET CMP0156 NEW)
endif()

project(llm_inference_engine
  VERSION 0.1.0
  DESCRIPTION "Multi-GPU LLM inference engine for decoder-only transformers"
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

# When ON, cmake/cuda.cmake (included below) attempts toolkit detection and
# downgrades to OFF with a status message if none is found — the documented
# `-DENGINE_ENABLE_CUDA=OFF` configure command is never required just to
# configure on a CUDA-less machine.
option(ENGINE_ENABLE_CUDA "Build the CUDA backend (device runtime and kernels)" ON)
option(ENGINE_BUILD_TESTS "Build unit and integration tests" ON)

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
# Fenced to CXX compiles: $<CXX_COMPILER_ID> is true for CUDA sources too
# (it names the compiler, not the source language), and nvcc rejects these
# flags. Device-code warnings are added in cmake/cuda.cmake.
target_compile_options(engine_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:${ENGINE_WARNING_FLAGS}>)

# CUDA toolkit detection, language enablement, and the engine::cuda_build
# definition target (M2-T02; docs/design/cuda-backend.md §3). May downgrade
# ENGINE_ENABLE_CUDA to OFF, so everything below sees the final value.
include(cmake/cuda.cmake)

add_subdirectory(src)

# enable_testing() rather than include(CTest): we want test registration only,
# not the CDash dashboard targets (Experimental, Nightly, ...) CTest adds.
if(ENGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
