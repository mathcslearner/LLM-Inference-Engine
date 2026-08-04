add_library(engine_memory STATIC allocator.cpp)
add_library(engine::memory ALIAS engine_memory)
target_include_directories(engine_memory PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::tensor_base is the memory → tensor_base edge (ADR-002 Amendment 1):
# Buffer/Allocator hold a tensor::Device; tensor_base carries engine::core
# (Status, CHECK) transitively. engine::cuda_build carries the
# ENGINE_ENABLE_CUDA compile definition (memory is a CUDA-touching module,
# ADR-002 rule 3; its device/pinned allocators land in M2-T05/T07).
target_link_libraries(engine_memory
  PUBLIC engine::tensor_base engine::cuda_build
  PRIVATE engine_warnings)
