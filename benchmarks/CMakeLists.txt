# Microbenchmarks (begun M3-T06; harness matured in M12-T01). Not registered
# with CTest — benchmarks are run by hand on a quiesced machine and their
# numbers recorded in benchmarks/BASELINES.md (project rule: no perf claim
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

# attention_bench (M6-T04): optimized prefill attention vs the M5 reference
# (cpu::attention) at 2k context, in BASELINES.md (no perf target — time vs
# reference).
add_executable(attention_bench kernels/attention_bench.cpp)
target_link_libraries(attention_bench PRIVATE
  engine::kernels
  engine::cpu
  engine::tensor
  fmt::fmt
  engine_warnings)

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

# bench_generate (M6-T08): end-to-end prefill/decode tokens/sec through the real
# greedy Generate loop, with a run-to-run stability verdict (the ±5% acceptance
# criterion). Perf numbers are run by hand and recorded in BASELINES.md; the
# smoke test below is the only ctest-registered invocation.
add_executable(bench_generate bench_generate.cpp)
target_link_libraries(bench_generate PRIVATE
  engine::engine
  engine::kernels
  engine::parallel
  fmt::fmt
  engine_warnings)

# A fast smoke run on the committed tiny-llama fixture so CI exercises the
# harness end-to-end (arg parsing, both backends, the stats path) — kept tiny so
# it stays a correctness check, never a perf measurement. enable_testing() is
# called at the root only under ENGINE_BUILD_TESTS, so guard on it.
if(ENGINE_BUILD_TESTS)
  # Default (paged) cache, both backends.
  add_test(NAME bench_generate_smoke
    COMMAND bench_generate
      --model "${CMAKE_SOURCE_DIR}/tests/fixtures/models/tiny-llama"
      --prompt-len 8 --new-tokens 4 --runs 2 --backend optimized)
  add_test(NAME bench_generate_smoke_reference
    COMMAND bench_generate
      --model "${CMAKE_SOURCE_DIR}/tests/fixtures/models/tiny-llama"
      --prompt-len 8 --new-tokens 4 --runs 2 --backend reference)
  # The pre-paging contiguous cache (the A/B baseline path stays exercised).
  add_test(NAME bench_generate_smoke_simple
    COMMAND bench_generate
      --model "${CMAKE_SOURCE_DIR}/tests/fixtures/models/tiny-llama"
      --prompt-len 8 --new-tokens 4 --runs 2 --backend optimized
      --kv-cache simple)
endif()

# bench_sampling (M7-T06): the batched sampler's hot path over a synthetic
# [batch, vocab] logits block (no model load), median ms/step vs the serial
# reference loop. The 64×128k-under-5ms target is advisory; numbers recorded in
# BASELINES.md. The smoke run keeps it a tiny correctness check in CI.
add_executable(bench_sampling bench_sampling.cpp)
target_link_libraries(bench_sampling PRIVATE
  engine::sampling
  engine::parallel
  fmt::fmt
  engine_warnings)

if(ENGINE_BUILD_TESTS)
  add_test(NAME bench_sampling_smoke
    COMMAND bench_sampling --batch 4 --vocab 256 --runs 2 --config topk_topp)
  add_test(NAME bench_sampling_smoke_logprobs
    COMMAND bench_sampling --batch 4 --vocab 256 --runs 2 --config logprobs)
endif()
