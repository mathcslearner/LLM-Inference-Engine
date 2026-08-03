add_library(engine_spec STATIC spec.cpp)
add_library(engine::spec ALIAS engine_spec)
target_include_directories(engine_spec PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_spec PRIVATE engine_warnings)
