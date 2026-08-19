# sampling: the per-request generation controls (M7). `SamplingParams` + its
# validation (params.{h,cpp}), the stochastic-sampling stages (stages.{h,cpp}:
# temperature/top-k/top-p/softmax/CDF draw) and the header-only Philox RNG
# (philox.h), the single-sequence reference `Sampler` (sampler.{h,cpp}) —
# penalties → temperature → top-k/p → selection → logprobs — and the batched
# optimized sampler (batched_sampler.{h,cpp}, M7-T06: parallel_for across
# sequences, sharing the reference's per-row core so it picks identical tokens)
# (design: docs/design/model-execution.md §15). Layer-2 domain module
# (ADR-002). `engine::core` is PUBLIC (Status/StatusOr in the public headers);
# `engine::parallel` is PUBLIC (batched_sampler.h exposes a `ThreadPool&`);
# `engine::kernels` is PRIVATE (the softmax/log-softmax stages reuse the shared
# vector `ExpF32` polynomial — a layer-2 → layer-1 edge ADR-002 already
# permits, as `model → kernels` is). The whole pipeline stays testable on
# CPU-only CI with no model or tensor dependency.
add_library(engine_sampling STATIC
  params.cpp
  sampler.cpp
  stages.cpp
  batched_sampler.cpp)
add_library(engine::sampling ALIAS engine_sampling)
target_include_directories(engine_sampling PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_sampling
  PUBLIC engine::core engine::parallel
  PRIVATE engine::kernels engine_warnings)
