# kvcache: the KV-cache interface (kv_cache.h, M5-T05) and its v0 contiguous
# implementation (SimpleKvCache, M5-T06). Layer-2 domain module (ADR-002
# Amendment 5): `kvcache -> tensor, memory, core` only; it never links `model`.
# engine::core and engine::tensor are PUBLIC because kv_cache.h exposes
# Status/StatusOr, tensor::Tensor/DataType. Still header-only until M5-T06 adds
# real sources, so the anchor TU keeps the archive non-empty.
add_library(engine_kvcache STATIC kvcache.cpp)
add_library(engine::kvcache ALIAS engine_kvcache)
target_include_directories(engine_kvcache PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_kvcache
  PUBLIC engine::core engine::tensor
  PRIVATE engine_warnings)
