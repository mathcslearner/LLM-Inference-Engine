# The CPU/CUDA split is the source list, not #ifdef-riddled bodies
# (docs/design/cuda-backend.md §2.2): CUDA builds compile the CUDA-touching
# sources, CPU-only builds compile stub .cpp sources with the same symbols.
# cuda.cu stays as a device-code anchor proving the pinned architectures
# compile (M2-T02) until this module gains a real .cu source (M2-T08's
# kernels landed in `kernels`, not here).
if(ENGINE_ENABLE_CUDA)
  set(ENGINE_CUDA_SOURCES cuda_utils.cpp stream.cpp cuda.cu)
else()
  set(ENGINE_CUDA_SOURCES cuda_utils_stub.cpp stream_stub.cpp)
endif()

add_library(engine_cuda STATIC ${ENGINE_CUDA_SOURCES})
add_library(engine::cuda ALIAS engine_cuda)
target_include_directories(engine_cuda PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cuda_build carries the ENGINE_ENABLE_CUDA compile definition
# (PUBLIC so dependents and test TUs agree on the spelling — design §2.2).
# engine::core: Status/StatusOr in the public cuda_utils.h API.
target_link_libraries(engine_cuda
  PUBLIC engine::cuda_build engine::core
  PRIVATE engine_warnings)
if(ENGINE_ENABLE_CUDA)
  # CUDA-compiled consumer TUs (other modules' .cpp/.cu, CUDA-only tests)
  # include cuda_runtime.h via cuda/cuda_check.h and link the CUDA runtime;
  # PUBLIC so they inherit the toolkit include dirs and library. Static
  # runtime: no libcudart.so dependency at deploy time (shared is a one-line
  # change if profiling tools ever need it).
  target_link_libraries(engine_cuda PUBLIC CUDA::cudart_static)
endif()
