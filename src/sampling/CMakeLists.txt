# sampling: the per-request generation controls (M7). `SamplingParams` + its
# validation (params.{h,cpp}) and the single-sequence reference `Sampler`
# (sampler.{h,cpp}) — penalties → temperature → top-k/p → selection → logprobs
# (design: docs/design/model-execution.md §15). Layer-2 domain module
# (ADR-002): it depends only on `core` (Status/StatusOr in the public headers),
# so the whole sampling pipeline stays testable on CPU-only CI with no model or
# tensor dependency. `engine::core` is PUBLIC — params.h/sampler.h expose it.
add_library(engine_sampling STATIC
  params.cpp
  sampler.cpp)
add_library(engine::sampling ALIAS engine_sampling)
target_include_directories(engine_sampling PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_sampling
  PUBLIC engine::core
  PRIVATE engine_warnings)
