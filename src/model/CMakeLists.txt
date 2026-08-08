# model: config.json parsing, safetensors loading, weight mapping (M4;
# design: docs/design/model-loading.md). engine::core, engine::tensor, and
# engine::memory are PUBLIC because the headers expose Status/StatusOr,
# tensor::Tensor/DataType/Shape, and memory::Buffer; nlohmann_json is an
# implementation detail (never in a public header, design §2) and stays
# PRIVATE.
add_library(engine_model STATIC checkpoint.cpp config.cpp mapped_file.cpp
                                safetensors.cpp weight_map.cpp)
add_library(engine::model ALIAS engine_model)
target_include_directories(engine_model PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_model
  PUBLIC engine::core engine::tensor engine::memory
  PRIVATE nlohmann_json::nlohmann_json engine_warnings)
