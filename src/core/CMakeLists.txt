add_library(engine_core STATIC core.cpp)
add_library(engine::core ALIAS engine_core)
target_include_directories(engine_core PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_core PRIVATE engine_warnings)
