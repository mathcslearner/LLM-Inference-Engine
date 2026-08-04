# The CPU/CUDA split is the source list, not #ifdef-riddled bodies
# (docs/design/cuda-backend.md §2.2): CUDA builds compile the
# toolkit-touching cuda_allocator.cpp, CPU-only builds a stub with the same
# symbols whose factories return Unimplemented.
if(ENGINE_ENABLE_CUDA)
  set(ENGINE_MEMORY_SOURCES allocator.cpp caching_allocator.cpp
                            cuda_allocator.cpp pinned_allocator.cpp)
else()
  set(ENGINE_MEMORY_SOURCES allocator.cpp caching_allocator.cpp
                            cuda_allocator_stub.cpp pinned_allocator_stub.cpp)
endif()

add_library(engine_memory STATIC ${ENGINE_MEMORY_SOURCES})
add_library(engine::memory ALIAS engine_memory)
target_include_directories(engine_memory PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::tensor_base is the memory → tensor_base edge (ADR-002 Amendment 1):
# Buffer/Allocator hold a tensor::Device; tensor_base carries engine::core
# (Status, CHECK) transitively. engine::cuda_build carries the
# ENGINE_ENABLE_CUDA compile definition (memory is a CUDA-touching module,
# ADR-002 rule 3). engine::cuda is the memory → cuda edge (ADR-002
# Amendment 2): the CUDA and pinned allocators reuse cuda's device
# introspection and error mapping. PRIVATE — only the .cpp files need it;
# memory's public headers stay toolkit-free.
target_link_libraries(engine_memory
  PUBLIC engine::tensor_base engine::cuda_build
  PRIVATE engine_warnings engine::cuda)
