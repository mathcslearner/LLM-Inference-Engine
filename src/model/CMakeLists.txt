# model: config.json parsing, safetensors loading, weight mapping (M4;
# design: docs/design/model-loading.md). engine::core, engine::tensor, and
# engine::memory are PUBLIC because the headers expose Status/StatusOr,
# tensor::Tensor/DataType/Shape, and memory::Buffer; nlohmann_json is an
# implementation detail (never in a public header, design §2) and stays
# PRIVATE.
add_library(engine_model STATIC checkpoint.cpp config.cpp loader.cpp
                                mapped_file.cpp modules.cpp safetensors.cpp
                                weight_map.cpp)
add_library(engine::model ALIAS engine_model)
target_include_directories(engine_model PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cpu is PRIVATE: modules.cpp calls cpu::gemm, but no public model
# header exposes cpu types (the model graph's execution surface, M5-T02).
# engine::kvcache is PRIVATE too (ADR-002 Amendment 5, the `model → kvcache`
# edge): the Attention module's forward takes a `kvcache::KvCache&`, but
# modules.h only forward-declares it — no public model header re-exports
# kvcache types beyond the reference parameter.
target_link_libraries(engine_model
  PUBLIC engine::core engine::tensor engine::memory
  PRIVATE engine::cpu engine::kvcache nlohmann_json::nlohmann_json
          engine_warnings)
