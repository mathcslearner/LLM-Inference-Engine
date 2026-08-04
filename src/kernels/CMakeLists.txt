# The CPU/CUDA split is the source list, not #ifdef-riddled bodies
# (docs/design/cuda-backend.md §2.2): CUDA builds compile the .cu sources,
# CPU-only builds compile stub .cpp sources with the same symbols.
if(ENGINE_ENABLE_CUDA)
  set(ENGINE_KERNELS_SOURCES kernels.cu)
else()
  set(ENGINE_KERNELS_SOURCES kernels.cpp)
endif()

add_library(engine_kernels STATIC ${ENGINE_KERNELS_SOURCES})
add_library(engine::kernels ALIAS engine_kernels)
target_include_directories(engine_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cuda_build carries the ENGINE_ENABLE_CUDA compile definition
# (PUBLIC so dependents and test TUs agree on the spelling — design §2.2).
target_link_libraries(engine_kernels
  PUBLIC engine::cuda_build
  PRIVATE engine_warnings)
