add_library(engine_sampling STATIC sampling.cpp)
add_library(engine::sampling ALIAS engine_sampling)
target_include_directories(engine_sampling PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_sampling PRIVATE engine_warnings)
