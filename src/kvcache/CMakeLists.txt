# kvcache: the KV-cache interface (kv_cache.h, M5-T05), its v0 contiguous
# implementation (SimpleKvCache, simple_cache.{h,cpp}, M5-T06), the paged
# block pool (BlockPool, block_pool.{h,cpp}, M8-T02), the per-sequence block
# table (BlockTable, block_table.{h,cpp}, M8-T03), the paged cache
# (PagedKvCache, paged_cache.{h,cpp}, M8-T04), and the paged prefill read path
# (paged_gather.{h,cpp}, M8-T06). Layer-2 domain module (ADR-002
# Amendment 5): `kvcache -> tensor, memory, core` and — new in M8-T04 — the
# downward `kvcache -> kernels` edge (the paged KV-scatter kernel, called by
# PagedKvCache; a layer-2->1 edge ADR-002 permits without amendment, exactly as
# `model -> kernels`). It never links `model` or `cpu`. engine::core and
# engine::tensor are PUBLIC because the public headers expose Status/StatusOr,
# tensor::Tensor/DataType; engine::memory is PUBLIC because block_pool.h takes a
# memory::Allocator*. engine::kernels is PRIVATE: paged_cache.h stays
# kernels-free (the KvScatterF32 call lives in paged_cache.cpp), like
# `model -> kernels` via PackedLinear (M8-T06's KvGatherF32 call in
# paged_gather.cpp is the second such call).
add_library(engine_kvcache STATIC
  simple_cache.cpp
  block_pool.cpp
  block_table.cpp
  paged_cache.cpp
  paged_gather.cpp
  kv_budget.cpp)
add_library(engine::kvcache ALIAS engine_kvcache)
target_include_directories(engine_kvcache PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_kvcache
  PUBLIC engine::core engine::tensor engine::memory
  PRIVATE engine::kernels engine_warnings)
