# CUDA build integration (M2-T02; design: docs/design/cuda-backend.md §3).
#
# Included from the top-level CMakeLists.txt after `engine_warnings` is
# defined (this file appends device-code flags to it) and before
# add_subdirectory(src). Responsibilities:
#   - toolkit auto-detection, downgrading ENGINE_ENABLE_CUDA to OFF when no
#     toolkit is found (never a configure failure),
#   - CUDA language enablement with the pinned architecture list and C++20,
#   - the CUDA 12.x toolkit floor (ADR-001),
#   - warnings-as-errors for device code,
#   - the single definition point of the ENGINE_ENABLE_CUDA compile
#     definition (the `engine::cuda_build` INTERFACE target, linked PUBLIC
#     by the CUDA-touching modules: cuda, kernels, memory, tensor).

if(ENGINE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Pinned architectures (design §3): A100 (80), consumer Ampere (86),
    # Ada/L4/L40 (89), Hopper (90). `-real` = SASS only, deliberately no
    # embedded PTX: plain `90` would add forward-compat PTX for newer
    # arches (Blackwell), which the design excludes — newer hardware gets
    # an explicit new entry and a rebuild, not silent JIT. Must be set
    # before enable_language(CUDA). Honors an explicit user -D override.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real)
    endif()
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
      message(FATAL_ERROR
        "CUDA toolkit ${CMAKE_CUDA_COMPILER_VERSION} found, but the engine "
        "requires CUDA 12.x or newer (ADR-001). Upgrade the toolkit or "
        "configure with -DENGINE_ENABLE_CUDA=OFF.")
    endif()
    # Device code is C++20, matching host code (design §3).
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # Separable compilation stays at its OFF default until something needs
    # device linking (design §3): no cross-TU device calls are planned.
  else()
    # Downgrade, don't fail: the documented CPU-only configure keeps
    # working without an explicit -DENGINE_ENABLE_CUDA=OFF. Set as a
    # normal variable so the cache keeps the user's choice and a later
    # configure on a toolkit-equipped machine re-detects.
    message(STATUS "No CUDA toolkit found; building without the CUDA backend "
                   "(ENGINE_ENABLE_CUDA downgraded to OFF)")
    set(ENGINE_ENABLE_CUDA OFF)
  endif()
endif()

# engine::cuda_build — the one place the ENGINE_ENABLE_CUDA compile
# definition comes from (design §2.2), so every #ifdef in the tree agrees
# on the spelling. Linked PUBLIC by the CUDA-touching module targets;
# exists (empty) on CPU-only builds so link lines are unconditional.
add_library(engine_cuda_build INTERFACE)
add_library(engine::cuda_build ALIAS engine_cuda_build)

if(ENGINE_ENABLE_CUDA)
  target_compile_definitions(engine_cuda_build INTERFACE ENGINE_ENABLE_CUDA)

  # Warnings-as-errors extends to device code (design §3). nvcc natively:
  # every warning is an error. The host warning set is forwarded to nvcc's
  # host-compiler pass via -Xcompiler (one flag each; comma-joining is a
  # quoting hazard), except flags that misfire on nvcc-generated host code:
  #   -Wpedantic       — warns on the #line-directive style nvcc emits
  #   -Wold-style-cast — CUDA headers/macros expand to C-style casts
  set(ENGINE_CUDA_WARNING_FLAGS
    -Werror=all-warnings
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-Wshadow
    -Xcompiler=-Wconversion
    -Xcompiler=-Wsign-conversion
    -Xcompiler=-Wnon-virtual-dtor
    -Xcompiler=-Werror)
  target_compile_options(engine_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:${ENGINE_CUDA_WARNING_FLAGS}>)
endif()
