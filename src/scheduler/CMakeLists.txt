# scheduler: the pure decision component of the continuous-batching runtime
# (M9-T04; design: docs/design/scheduler-runtime.md §6). `Scheduler::Schedule`
# maps plain descriptor structs (waiting/running SeqDescs + a PoolSnapshot +
# SchedulerConfig) to a SchedulerOutput of prefill/decode/preempt decisions —
# FCFS admission, decode-first priority, latest-arrived preemption — with no
# engine, model, pool, allocator, or tensor dependency (ADR-002 rule 4: a
# layer-3 leaf, table-testable on CPU-only CI forever). It links only
# `engine::core` for the CHECK macro (fmt); `runtime` mediates between it and
# `engine`.
add_library(engine_scheduler STATIC scheduler.cpp)
add_library(engine::scheduler ALIAS engine_scheduler)
target_include_directories(engine_scheduler PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_scheduler PUBLIC engine::core
                                       PRIVATE engine_warnings)
