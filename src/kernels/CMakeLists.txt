# SIMD kernels module (scalar/NEON/AVX2 — M3-T05/T06 onward). Currently a
# placeholder anchor; the per-ISA source-list pattern (per-TU arch flags, no
# target-wide -mavx2) will be specified in docs/design/cpu-backend.md.
add_library(engine_kernels STATIC kernels.cpp)
add_library(engine::kernels ALIAS engine_kernels)
target_include_directories(engine_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_kernels
  PUBLIC engine::tensor
  PRIVATE engine_warnings)
