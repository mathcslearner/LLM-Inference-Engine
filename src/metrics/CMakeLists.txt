add_library(engine_metrics STATIC metrics.cpp)
add_library(engine::metrics ALIAS engine_metrics)
target_include_directories(engine_metrics PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_metrics PRIVATE engine_warnings)
