# SIMD kernels module: per-ISA variants behind runtime dispatch (M3-T05
# dispatch infrastructure; first vectorized kernels M3-T06; design:
# docs/design/cpu-backend.md §4). Rules, all load-bearing (§4.4):
#   - ENGINE_ARCH_ARM64 / ENGINE_ARCH_X86_64 are PUBLIC compile definitions
#     derived from CMAKE_SYSTEM_PROCESSOR, so the arch-guarded internal
#     headers see the same guards in every includer (library TUs and tests).
#   - Per-ISA TUs of ISAs the target cannot use are absent from the build,
#     not compiled-and-unused: neon/ only on arm64, avx2/ only on x86-64.
#   - Arch flags (-mavx2 -mfma -mf16c) appear only as per-source properties
#     on avx2/*.cpp — never target- or toolchain-wide — so the compiler
#     cannot emit AVX2 instructions outside those TUs and
#     illegal-instruction-on-older-hardware bugs are structurally
#     impossible. NEON TUs need no flags (baseline in AArch64).

add_library(engine_kernels STATIC
  dispatch.cpp
  probe.cpp
  elementwise.cpp
  reduce.cpp
  convert.cpp
  gemm.cpp
  exp.cpp
  softmax.cpp
  activation.cpp
  norm.cpp
  rope.cpp
  attention.cpp
  # embedding.cpp has no per-ISA TU (M6-T06): its only per-element work is the
  # fp16/bf16->fp32 widen, dispatched through the existing convert variants; the
  # gather itself is inherently scalar (optimized-cpu-execution.md §2, §7).
  embedding.cpp
  # kv_scatter.cpp has no per-ISA TU (M8-T04): the paged KV write is a pure
  # fp32->fp32 memcpy per (token, head) — Class E, bit-identical across ISAs and
  # thread counts, so a single scalar TU ships (paged-kv-cache.md §9.1), like the
  # embedding gather above.
  kv_scatter.cpp
  scalar/probe.cpp
  scalar/elementwise.cpp
  scalar/reduce.cpp
  scalar/convert.cpp
  scalar/gemm.cpp
  scalar/exp.cpp
  scalar/softmax.cpp
  scalar/activation.cpp
  scalar/norm.cpp
  scalar/rope.cpp
  scalar/attention.cpp)
add_library(engine::kernels ALIAS engine_kernels)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" engine_processor)
if(engine_processor MATCHES "^(arm64|aarch64)$")
  target_compile_definitions(engine_kernels PUBLIC ENGINE_ARCH_ARM64)
  target_sources(engine_kernels PRIVATE
    neon/probe.cpp
    neon/elementwise.cpp
    neon/reduce.cpp
    neon/convert.cpp
    neon/gemm.cpp
    neon/exp.cpp
    neon/softmax.cpp
    neon/activation.cpp
    neon/norm.cpp
    neon/rope.cpp
    neon/attention.cpp)
elseif(engine_processor MATCHES "^(x86_64|amd64)$")
  target_compile_definitions(engine_kernels PUBLIC ENGINE_ARCH_X86_64)
  set(engine_kernels_avx2_sources
    avx2/probe.cpp
    avx2/elementwise.cpp
    avx2/reduce.cpp
    avx2/convert.cpp
    avx2/gemm.cpp
    avx2/exp.cpp
    avx2/softmax.cpp
    avx2/activation.cpp
    avx2/norm.cpp
    avx2/rope.cpp
    avx2/attention.cpp)
  target_sources(engine_kernels PRIVATE ${engine_kernels_avx2_sources})
  set_source_files_properties(${engine_kernels_avx2_sources}
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

target_include_directories(engine_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_kernels
  PUBLIC engine::tensor
  PRIVATE engine::parallel engine_warnings)
