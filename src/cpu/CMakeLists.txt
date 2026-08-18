# cpu: the reference backend — stateless, clarity-first fp32 ops that are the
# correctness oracle for every optimized kernel (M5; design:
# docs/design/cpu-backend.md §2.1, docs/design/model-execution.md §2). Layer 1:
# `cpu -> tensor, parallel` only — never `kernels` (model-execution.md §2.1):
# the reference must not depend on the dispatched kernels it exists to validate,
# so fp16/bf16 -> fp32 widening uses tensor/half.h, not kernels/convert.
# engine::tensor is PUBLIC (ops.h exposes tensor::Tensor); parallel is an
# implementation detail (threading is invisible in the op signatures) and stays
# PRIVATE.
add_library(engine_cpu STATIC gemm.cpp rmsnorm.cpp activation.cpp softmax.cpp
                              embedding.cpp rope.cpp attention.cpp)
add_library(engine::cpu ALIAS engine_cpu)
target_include_directories(engine_cpu PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_cpu
  PUBLIC engine::core engine::tensor
  PRIVATE engine::parallel engine_warnings)
