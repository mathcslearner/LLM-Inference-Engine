# model: config.json parsing, safetensors loading, weight mapping (M4;
# design: docs/design/model-loading.md). engine::core and engine::tensor_base
# are PUBLIC because config.h exposes Status/StatusOr and tensor::DataType;
# nlohmann_json is an implementation detail (never in a public header,
# design §2) and stays PRIVATE.
add_library(engine_model STATIC config.cpp)
add_library(engine::model ALIAS engine_model)
target_include_directories(engine_model PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_model
  PUBLIC engine::core engine::tensor_base
  PRIVATE nlohmann_json::nlohmann_json engine_warnings)
