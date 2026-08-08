# Thread pool + deterministic parallel_for/parallel_reduce (M3-T04; design:
# docs/design/cpu-backend.md §3). Layer 1, beside memory: parallel → core
# only (ADR-002 Amendment 4) — it schedules index ranges, not tensors.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(engine_parallel STATIC thread_pool.cpp parallel_for.cpp)
add_library(engine::parallel ALIAS engine_parallel)
target_include_directories(engine_parallel PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_parallel
  PUBLIC engine::core Threads::Threads
  PRIVATE engine_warnings)
