add_library(engine_kernels STATIC kernels.cpp)
add_library(engine::kernels ALIAS engine_kernels)
target_include_directories(engine_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_kernels PRIVATE engine_warnings)
