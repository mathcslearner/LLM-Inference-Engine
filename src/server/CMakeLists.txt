add_library(engine_server STATIC server.cpp)
add_library(engine::server ALIAS engine_server)
target_include_directories(engine_server PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_server PRIVATE engine_warnings)
