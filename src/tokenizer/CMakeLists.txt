add_library(engine_tokenizer STATIC tokenizer.cpp)
add_library(engine::tokenizer ALIAS engine_tokenizer)
target_include_directories(engine_tokenizer PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_tokenizer PRIVATE engine_warnings)
