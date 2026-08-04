# The tensor module builds two targets (ADR-002 Amendment 1;
# docs/design/tensor.md §2.1):
#
# engine::tensor_base — INTERFACE (header-only): the plain value types
# dtype.h, shape.h, device.h, half.h. These headers include nothing from
# memory or the rest of tensor (only core), so engine::memory can link this
# target to hold a Device without creating a cycle. engine::core is INTERFACE
# because the headers use Status/StatusOr and fmt in consumer code.
add_library(engine_tensor_base INTERFACE)
add_library(engine::tensor_base ALIAS engine_tensor_base)
target_include_directories(engine_tensor_base INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_tensor_base INTERFACE engine::core)

# engine::tensor — the compiled library (tensor.h, ops.h and their .cpp).
# engine::memory is PUBLIC because tensor.h includes memory/allocator.h
# (Tensor holds a shared_ptr<Buffer>) — the layer-1 tensor → memory edge from
# ADR-002.
add_library(engine_tensor STATIC ops.cpp tensor.cpp)
add_library(engine::tensor ALIAS engine_tensor)
target_include_directories(engine_tensor PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::cuda_build carries the ENGINE_ENABLE_CUDA compile definition
# (tensor's device-aware transfer TU, M2-T07, is toolkit-touching per
# docs/design/cuda-backend.md §2.1).
target_link_libraries(engine_tensor
  PUBLIC engine::tensor_base engine::memory engine::cuda_build
  PRIVATE engine_warnings)
