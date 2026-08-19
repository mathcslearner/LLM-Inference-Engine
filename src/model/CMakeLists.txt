# model: config.json parsing, safetensors loading, weight mapping (M4;
# design: docs/design/model-loading.md). engine::core, engine::tensor, and
# engine::memory are PUBLIC because the headers expose Status/StatusOr,
# tensor::Tensor/DataType/Shape, and memory::Buffer; nlohmann_json is an
# implementation detail (never in a public header, design §2) and stays
# PRIVATE.
add_library(engine_model STATIC checkpoint.cpp config.cpp loader.cpp
                                mapped_file.cpp modules.cpp optimized_embedding.cpp
                                optimized_model.cpp packed_linear.cpp
                                reference_model.cpp registry.cpp safetensors.cpp
                                weight_map.cpp workspace.cpp)
add_library(engine::model ALIAS engine_model)
target_include_directories(engine_model PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cpu is PRIVATE: modules.cpp/reference_model.cpp call cpu:: ops, but no
# public model header exposes cpu types (the model graph's execution surface,
# M5-T02).
#
# engine::kernels is PRIVATE (M6-T02, the `model → kernels` downward edge —
# already allowed by ADR-002 §2.2, no amendment): packed_linear.cpp calls
# kernels::PackedGemm / PackWeightPanels and optimized_embedding.cpp (M6-T06)
# calls kernels::EmbeddingLookup*, but their headers expose only a
# tensor::Tensor for the packed bytes, never a kernels type
# (optimized-cpu-execution.md §2.1).
#
# engine::kvcache is PUBLIC (ADR-002 Amendment 5, the `model → kvcache` edge):
# the public `model.h` (M5-T07) states the `Model` contract in terms of the
# cache — `cache_geometry()` returns `kvcache::CacheGeometry` by value and
# `forward` takes a `ForwardRequest` holding a `kvcache::KvCache*` — so a
# consumer of model.h needs the kvcache headers. modules.h alone would keep it
# PRIVATE (it only forward-declares `KvCache`), but model.h's by-value return
# makes the edge part of the public surface.
target_link_libraries(engine_model
  PUBLIC engine::core engine::tensor engine::memory engine::kvcache
  PRIVATE engine::cpu engine::kernels nlohmann_json::nlohmann_json
          engine_warnings)
