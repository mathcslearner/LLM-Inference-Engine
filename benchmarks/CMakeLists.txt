# Microbenchmarks (begun M3-T06; harness matured in M12-T01). Not registered
# with CTest — benchmarks are run by hand on a quiesced machine and their
# numbers recorded in benchmarks/BASELINES.md (CLAUDE.md: no perf claim
# without a benchmark delta).

add_executable(kernels_bench kernels/kernels_bench.cpp)
target_link_libraries(kernels_bench PRIVATE
  engine::kernels
  engine_warnings)

# gemm_bench (M6-T02): packed GEMM vs the M5 naive cpu::gemm at 4096^3, with an
# optional Accelerate/BLAS sanity number. BLAS is benchmark-only and never
# linked into src/ (ADR-002; the engine ships no BLAS dependency) — it enters
# behind an opt-in CMake flag here.
add_executable(gemm_bench kernels/gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE
  engine::kernels
  engine::cpu
  engine::tensor
  fmt::fmt)

option(ENGINE_BENCH_BLAS "Link a BLAS into gemm_bench for a context number" OFF)
if(ENGINE_BENCH_BLAS)
  # Third-party BLAS SDK headers (Accelerate) predate our -Werror warning set,
  # so this opt-in build does NOT link engine_warnings — the default (BLAS-off)
  # build below keeps full warnings-as-errors on our code. Apple ships
  # Accelerate; elsewhere any find_package(BLAS) target works. The engine proper
  # never sees this — gemm_bench only.
  if(APPLE)
    find_library(ENGINE_ACCELERATE Accelerate REQUIRED)
    target_link_libraries(gemm_bench PRIVATE ${ENGINE_ACCELERATE})
    target_compile_definitions(gemm_bench PRIVATE
      ENGINE_HAVE_BLAS ACCELERATE_NEW_LAPACK)
  else()
    find_package(BLAS REQUIRED)
    target_link_libraries(gemm_bench PRIVATE BLAS::BLAS)
    target_compile_definitions(gemm_bench PRIVATE ENGINE_HAVE_BLAS)
  endif()
else()
  target_link_libraries(gemm_bench PRIVATE engine_warnings)
endif()
