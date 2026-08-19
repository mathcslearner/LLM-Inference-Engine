# engine: the generation loop (generator.{h,cpp}, M5-T09) and the backend
# selection seam (backend.h — a re-export of model::Backend plus the string↔enum
# helpers, header-only). Layer-3 module (ADR-002): engine → model, kvcache,
# tensor, core (all lower layers); it depends only on the abstract `Model` and
# `KvCache` interfaces, so it is backend-agnostic (design §2.1, §10).
#
# engine::core, engine::model, engine::kvcache, and engine::sampling are PUBLIC:
# the public headers expose Status/StatusOr (generator.h, backend.h),
# model::Model/Backend (generator.h, backend.h), kvcache::KvCache (generator.h),
# and sampling::SamplingParams (generator.h — GenerateOptions::sampling, M7-T01).
# engine::tensor is PRIVATE — only generator.cpp names tensor::Tensor (reading
# the logits row for the sampler); model.h re-exports tensor transitively, but
# the direct use is an implementation detail.
add_library(engine_engine STATIC generator.cpp)
add_library(engine::engine ALIAS engine_engine)
target_include_directories(engine_engine PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_engine
  PUBLIC engine::core engine::model engine::kvcache engine::sampling
  PRIVATE engine::tensor engine_warnings)
