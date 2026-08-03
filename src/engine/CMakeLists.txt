add_library(engine_engine STATIC engine.cpp)
add_library(engine::engine ALIAS engine_engine)
target_include_directories(engine_engine PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_engine PRIVATE engine_warnings)
