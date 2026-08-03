add_library(engine_memory STATIC memory.cpp)
add_library(engine::memory ALIAS engine_memory)
target_include_directories(engine_memory PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_memory PRIVATE engine_warnings)
