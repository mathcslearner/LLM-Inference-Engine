# engine: the generation loop (generator.{h,cpp}, M5-T09), the stop-condition
# machinery (stop.{h,cpp}, M7-T04), and the backend selection seam (backend.h —
# a re-export of model::Backend plus the string↔enum helpers, header-only).
# Layer-3 module (ADR-002): engine → model, kvcache, sampling, tokenizer,
# tensor, core (all lower layers); it depends only on the abstract `Model` and
# `KvCache` interfaces, so it is backend-agnostic (design §2.1, §10).
#
# engine::core, engine::model, engine::kvcache, engine::sampling, and
# engine::tokenizer are PUBLIC: the public headers expose Status/StatusOr
# (generator.h, backend.h, stop.h), model::Model/Backend (generator.h,
# backend.h), kvcache::KvCache (generator.h), sampling::SamplingParams
# (generator.h/stop.h), and tokenizer::Tokenizer/DetokenizerStream (stop.h,
# generator.h — GenerateOptions::tokenizer, M7-T04). `engine → tokenizer` is an
# already-sanctioned edge (ADR-002 diagram), first used here. engine::tensor is
# PRIVATE — only generator.cpp names tensor::Tensor (reading the logits row for
# the sampler); model.h re-exports tensor transitively, but the direct use is an
# implementation detail.
add_library(engine_engine STATIC generator.cpp stop.cpp)
add_library(engine::engine ALIAS engine_engine)
target_include_directories(engine_engine PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_engine
  PUBLIC engine::core engine::model engine::kvcache engine::sampling
         engine::tokenizer
  PRIVATE engine::tensor engine_warnings)
