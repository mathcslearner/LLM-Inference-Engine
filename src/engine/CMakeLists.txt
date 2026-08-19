# engine: the generation loop (generator.{h,cpp}, M5-T09), the stop-condition
# machinery (stop.{h,cpp}, M7-T04), the batch assembler (batch.{h,cpp}, M9-T05),
# and the backend selection seam (backend.h — a re-export of model::Backend plus
# the string↔enum helpers, header-only).
# Layer-3 module (ADR-002): engine → model, kvcache, sampling, tokenizer,
# tensor, memory, core (all lower layers); it depends only on the abstract
# `Model` and `KvCache` interfaces, so it is backend-agnostic (design §2.1, §10).
#
# engine::core, engine::model, engine::kvcache, engine::sampling,
# engine::tokenizer, engine::tensor, and engine::memory are PUBLIC: the public
# headers expose Status/StatusOr (generator.h, backend.h, stop.h, batch.h),
# model::Model/Backend/ForwardRequest (generator.h, backend.h, batch.h),
# kvcache::KvCache (generator.h, batch.h), sampling::SamplingParams/BatchRow
# (generator.h/stop.h/batch.h), tokenizer::Tokenizer/DetokenizerStream (stop.h,
# generator.h), tensor::Tensor (batch.h — the block-table tensor), and
# memory::Allocator (batch.h — the staging allocator). `engine → tokenizer` is
# an already-sanctioned edge (ADR-002 diagram); engine::tensor moved PRIVATE →
# PUBLIC in M9-T05 (batch.h names tensor::Tensor in its public surface) and
# engine::memory is added here (both already-below-it layer-1 modules, no ADR
# amendment).
add_library(engine_engine STATIC generator.cpp stop.cpp batch.cpp)
add_library(engine::engine ALIAS engine_engine)
target_include_directories(engine_engine PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_engine
  PUBLIC engine::core engine::model engine::kvcache engine::sampling
         engine::tokenizer engine::tensor engine::memory
  PRIVATE engine_warnings)
