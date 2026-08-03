add_library(engine_scheduler STATIC scheduler.cpp)
add_library(engine::scheduler ALIAS engine_scheduler)
target_include_directories(engine_scheduler PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_scheduler PRIVATE engine_warnings)
