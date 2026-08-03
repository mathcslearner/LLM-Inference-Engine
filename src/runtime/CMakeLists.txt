add_library(engine_runtime STATIC runtime.cpp)
add_library(engine::runtime ALIAS engine_runtime)
target_include_directories(engine_runtime PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_runtime PRIVATE engine_warnings)
