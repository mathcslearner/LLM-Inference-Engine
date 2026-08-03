add_library(engine_cpu STATIC cpu.cpp)
add_library(engine::cpu ALIAS engine_cpu)
target_include_directories(engine_cpu PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_cpu PRIVATE engine_warnings)
