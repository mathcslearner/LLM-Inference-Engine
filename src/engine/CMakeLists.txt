# engine: the generation loop (generator.{h,cpp}, M5-T09), the stop-condition
# machinery (stop.{h,cpp}, M7-T04), the batch assembler (batch.{h,cpp}, M9-T05),
# and the backend selection seam (backend.h — a re-export of model::Backend plus
# the string↔enum helpers, header-only).
# Layer-3 module (ADR-002): engine → model, kvcache, sampling, tokenizer, core
# (all lower layers); it depends only on the abstract `Model` and `KvCache`
# interfaces, so it is backend-agnostic (design §2.1, §10).
#
# engine::core, engine::model, engine::kvcache, engine::sampling, and
# engine::tokenizer are PUBLIC: the public headers expose Status/StatusOr
# (generator.h, backend.h, stop.h, batch.h), model::Model/Backend/ForwardRequest
# (generator.h, backend.h, batch.h), kvcache::KvCache (generator.h, batch.h),
# sampling::SamplingParams/BatchRow (generator.h/stop.h/batch.h),
# tokenizer::Tokenizer/DetokenizerStream (stop.h, generator.h). `engine →
# tokenizer` is an already-sanctioned edge (ADR-002 diagram). engine::tensor is
# PRIVATE (generator.cpp reads a logits tensor internally; no public engine
# header names a tensor type since M9-T07 removed the batch block-table tensor —
# the batched decode kernel self-sources block tables post-append, §8.4).
add_library(engine_engine STATIC generator.cpp stop.cpp batch.cpp)
add_library(engine::engine ALIAS engine_engine)
target_include_directories(engine_engine PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_engine
  PUBLIC engine::core engine::model engine::kvcache engine::sampling
         engine::tokenizer
  PRIVATE engine::tensor engine_warnings)
