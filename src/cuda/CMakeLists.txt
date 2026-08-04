# The CPU/CUDA split is the source list, not #ifdef-riddled bodies
# (docs/design/cuda-backend.md §2.2): CUDA builds compile the .cu sources,
# CPU-only builds compile stub .cpp sources with the same symbols.
if(ENGINE_ENABLE_CUDA)
  set(ENGINE_CUDA_SOURCES cuda.cu)
else()
  set(ENGINE_CUDA_SOURCES cuda.cpp)
endif()

add_library(engine_cuda STATIC ${ENGINE_CUDA_SOURCES})
add_library(engine::cuda ALIAS engine_cuda)
target_include_directories(engine_cuda PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cuda_build carries the ENGINE_ENABLE_CUDA compile definition
# (PUBLIC so dependents and test TUs agree on the spelling — design §2.2).
target_link_libraries(engine_cuda
  PUBLIC engine::cuda_build
  PRIVATE engine_warnings)
