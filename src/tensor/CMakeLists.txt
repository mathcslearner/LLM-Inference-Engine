add_library(engine_tensor STATIC tensor.cpp)
add_library(engine::tensor ALIAS engine_tensor)
target_include_directories(engine_tensor PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_tensor PRIVATE engine_warnings)
