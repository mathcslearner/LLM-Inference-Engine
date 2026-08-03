add_library(engine_cuda STATIC cuda.cpp)
add_library(engine::cuda ALIAS engine_cuda)
target_include_directories(engine_cuda PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_cuda PRIVATE engine_warnings)
