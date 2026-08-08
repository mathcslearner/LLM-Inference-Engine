# tokenizer: tokenizer.json → byte-level BPE (M4-T08…T10; design:
# docs/design/model-loading.md §6). engine::core is PUBLIC because the
# headers expose Status/StatusOr; nlohmann_json is an implementation detail
# (never in a public header, design §2) and stays PRIVATE. unicode.cpp
# includes the generated unicode_data.inc tables (tools/gen_unicode, §6.3).
add_library(engine_tokenizer STATIC
  bpe.cpp
  pretokenize.cpp
  tokenizer.cpp
  unicode.cpp)
add_library(engine::tokenizer ALIAS engine_tokenizer)
target_include_directories(engine_tokenizer PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_tokenizer
  PUBLIC engine::core
  PRIVATE nlohmann_json::nlohmann_json engine_warnings)
