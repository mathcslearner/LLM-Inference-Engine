cmake_minimum_required(VERSION 3.26)

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

# Declared now so the documented `-DENGINE_ENABLE_CUDA=OFF` configure command
# works from day one; the CUDA language and toolkit wiring land in M2.
option(ENGINE_ENABLE_CUDA "Build the CUDA backend (device runtime and kernels)" ON)

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
