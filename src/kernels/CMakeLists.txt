# The CPU/CUDA split is the source list, not #ifdef-riddled bodies
# (docs/design/cuda-backend.md §2.2): elementwise.cpp (validation — the
# always-compiled half of the §9.2 launcher contract) builds everywhere;
# CUDA builds add the .cu kernel/launch sources, CPU-only builds add stub
# .cpp sources with the same symbols.
set(ENGINE_KERNELS_SOURCES elementwise.cpp)
if(ENGINE_ENABLE_CUDA)
  list(APPEND ENGINE_KERNELS_SOURCES elementwise.cu)
else()
  list(APPEND ENGINE_KERNELS_SOURCES elementwise_stub.cpp)
endif()

add_library(engine_kernels STATIC ${ENGINE_KERNELS_SOURCES})
add_library(engine::kernels ALIAS engine_kernels)
target_include_directories(engine_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cuda_build carries the ENGINE_ENABLE_CUDA compile definition
# (PUBLIC so dependents and test TUs agree on the spelling — design §2.2).
# engine::tensor is PUBLIC: the launcher signatures in elementwise.h take
# Tensor and StreamHandle. engine::cuda is PRIVATE — the ADR-002
# `kernels → cuda` edge (ScopedSetDevice, DefaultStream, the §4.2 error
# mapping) is an implementation detail of the .cu sources; kernels' public
# headers stay toolkit-free.
target_link_libraries(engine_kernels
  PUBLIC engine::cuda_build engine::tensor
  PRIVATE engine::cuda engine_warnings)
