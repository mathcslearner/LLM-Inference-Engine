# kvcache: the KV-cache interface (kv_cache.h, M5-T05), its v0 contiguous
# implementation (SimpleKvCache, simple_cache.{h,cpp}, M5-T06), the paged
# block pool (BlockPool, block_pool.{h,cpp}, M8-T02), and the per-sequence
# block table (BlockTable, block_table.{h,cpp}, M8-T03). Layer-2 domain module
# (ADR-002 Amendment 5): `kvcache -> tensor, memory, core` only; it never links
# `model` or `cpu`. engine::core and engine::tensor are PUBLIC because the public
# headers expose Status/StatusOr, tensor::Tensor/DataType; engine::memory is
# PUBLIC because block_pool.h takes a memory::Allocator*. (The `kvcache ->
# kernels` downward edge arrives with the scatter/decode kernels in M8-T04.)
add_library(engine_kvcache STATIC simple_cache.cpp block_pool.cpp block_table.cpp)
add_library(engine::kvcache ALIAS engine_kvcache)
target_include_directories(engine_kvcache PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_kvcache
  PUBLIC engine::core engine::tensor engine::memory
  PRIVATE engine_warnings)
