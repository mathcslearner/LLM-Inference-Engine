# runtime: the orchestration layer that turns the single-request engine::Generate
# into a multi-request, continuously-batched system (M9; design:
# docs/design/scheduler-runtime.md). M9-T02 lands the two per-request objects the
# step loop fans out over — `Request`/`Sequence` (request.h + sequence.cpp: the
# §3.2 state machine, the per-sequence Sampler/StopChecker/PagedKvCache) — and
# the thread-safe per-request `OutputChannel` (channel.{h,cpp}, §4). M9-T03 adds
# the public async API and submission queue (engine.{h,cpp}, §5): `Engine`
# (`Submit`/`Cancel`/`Start`/`Stop`/`Step`), the mutex+condvar command queue,
# the single loop thread, and a `Step()` whose shape is the §9.1 loop with the
# FCFS-admission and per-sequence-execution placeholders the batched passes
# (M9-T05…T08) and the scheduler (M9-T04) substitute in place.
#
# Layer-4 orchestration module (ADR-002: server → runtime → scheduler; runtime →
# engine, model, kvcache, sampling, tokenizer, memory, core). No new ADR edge —
# every dependency is already below `runtime` on the diagram. The public headers
# expose Status/StatusOr, the `engine::engine` finish taxonomy (stop.h),
# sampling::Sampler/SamplingParams/StepLogprobs, kvcache::BlockPool/PagedKvCache,
# model::Model (the forward contract, engine.h), and tokenizer::Tokenizer, so
# those modules link PUBLIC.
add_library(engine_runtime STATIC channel.cpp engine.cpp sequence.cpp)
add_library(engine::runtime ALIAS engine_runtime)
target_include_directories(engine_runtime PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(engine_runtime
  PUBLIC engine::core engine::engine engine::model engine::sampling
         engine::kvcache engine::tokenizer
  PRIVATE engine_warnings)
