add_library(engine_distributed STATIC distributed.cpp)
add_library(engine::distributed ALIAS engine_distributed)
target_include_directories(engine_distributed PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_distributed PRIVATE engine_warnings)
