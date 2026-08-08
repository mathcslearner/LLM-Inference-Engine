add_library(engine_memory STATIC allocator.cpp caching_allocator.cpp)
add_library(engine::memory ALIAS engine_memory)
target_include_directories(engine_memory PUBLIC ${PROJECT_SOURCE_DIR}/src)
# engine::tensor_base is the memory → tensor_base edge (ADR-002 Amendment 1):
# Buffer/Allocator hold a tensor::Device; tensor_base carries engine::core
# (Status, CHECK) transitively.
target_link_libraries(engine_memory
  PUBLIC engine::tensor_base
  PRIVATE engine_warnings)
