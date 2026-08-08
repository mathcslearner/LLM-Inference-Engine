# Microbenchmarks (begun M3-T06; harness matured in M12-T01). Not registered
# with CTest — benchmarks are run by hand on a quiesced machine and their
# numbers recorded in benchmarks/BASELINES.md (CLAUDE.md: no perf claim
# without a benchmark delta).

add_executable(kernels_bench kernels/kernels_bench.cpp)
target_link_libraries(kernels_bench PRIVATE
  engine::kernels
  engine_warnings)
