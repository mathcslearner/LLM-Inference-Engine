# One line per test file; engine_add_tests (tests/CMakeLists.txt) turns each
# into its own executable and registers it with CTest.
engine_add_tests(TREE unit SOURCES
  smoke_test.cpp)

engine_add_tests(TREE unit SOURCES
  check_test.cpp
  logging_test.cpp
  status_test.cpp
  DEPS engine::core)

engine_add_tests(TREE unit SOURCES
  device_test.cpp
  dtype_test.cpp
  half_test.cpp
  shape_test.cpp
  DEPS engine::tensor_base)

# caching_allocator_test.cpp exercises the device-agnostic pool (M2-T06 —
# kept through the CPU-first pivot, ADR-004) over scriptable fakes and the
# real CpuAllocator.
engine_add_tests(TREE unit SOURCES
  allocator_test.cpp
  caching_allocator_test.cpp
  DEPS engine::memory)

engine_add_tests(TREE unit SOURCES
  ops_test.cpp
  tensor_test.cpp
  DEPS engine::tensor)

# Ordinary portable-C++ tests — nothing ISA-dependent, no SCALAR_PASS
# (docs/design/model-loading.md §8). The `model` label mirrors `parallel`/
# `kernels`: `ctest -L model` selects the loading suites alone.
engine_add_tests(TREE unit SOURCES
  checkpoint_test.cpp
  config_test.cpp
  mapped_file_test.cpp
  safetensors_test.cpp
  DEPS engine::model
  LABELS model)

# weight_map_test reads the qwen2-weight-names.json inventory fixture
# directly, hence the test-only nlohmann_json dependency (the library's own
# JSON use stays PRIVATE — design §2).
engine_add_tests(TREE unit SOURCES
  weight_map_test.cpp
  DEPS engine::model nlohmann_json::nlohmann_json
  LABELS model)

# cpu reference ops (M5; design: docs/design/model-execution.md §13). Ordinary
# portable C++ — the suites call cpu:: directly (not kernels::), so no
# SCALAR_PASS and no per-ISA concern. gemm_test reads the ops.safetensors
# golden via engine::model's SafetensorsFile; linear_test additionally uses the
# loader. `cpu` label mirrors `model`/`kernels` for `ctest -L cpu`.
engine_add_tests(TREE unit SOURCES
  gemm_test.cpp
  DEPS engine::cpu engine::model
  LABELS cpu)

engine_add_tests(TREE unit SOURCES
  linear_test.cpp
  DEPS engine::model engine::cpu
  LABELS model)

# rmsnorm_test / activation_test (M5-T03): cpu::rmsnorm/softmax/silu_mul/add
# goldens + the RmsNorm module. Both read the ops.safetensors golden via
# engine::model's SafetensorsFile; rmsnorm_test additionally uses the loader for
# a real bf16 norm weight.
engine_add_tests(TREE unit SOURCES
  rmsnorm_test.cpp
  DEPS engine::cpu engine::model
  LABELS cpu)

engine_add_tests(TREE unit SOURCES
  activation_test.cpp
  DEPS engine::cpu engine::model
  LABELS cpu)

# embedding_test / rope_test (M5-T04): cpu::embedding_lookup / cpu::rope_apply
# goldens + the Embedding/Rope modules. Both read the ops/rope safetensors
# goldens via engine::model's SafetensorsFile; embedding_test additionally uses
# the loader for the real bf16 embed table and the activations golden.
engine_add_tests(TREE unit SOURCES
  embedding_test.cpp
  DEPS engine::cpu engine::model
  LABELS cpu)

engine_add_tests(TREE unit SOURCES
  rope_test.cpp
  DEPS engine::cpu engine::model
  LABELS cpu)

# attention_test (M5-T05): cpu::attention op goldens/properties + the Attention
# module over the loaded checkpoint, using the real SimpleKvCache (M5-T06) —
# so the prefill-continuation fixture cases exercise the cache end-to-end — plus
# the token-by-token KV invariant. Reads attention.safetensors via engine::model's
# SafetensorsFile and the loader; engine::kvcache supplies the cache.
engine_add_tests(TREE unit SOURCES
  attention_test.cpp
  DEPS engine::cpu engine::model engine::kvcache
  LABELS cpu)

# simple_cache_test (M5-T06): SimpleKvCache mechanics (append/view transpose,
# length, truncate, capacity, immutability) — pure cache, no model or kernels
# dependency, so no SCALAR_PASS.
engine_add_tests(TREE unit SOURCES
  simple_cache_test.cpp
  DEPS engine::kvcache engine::tensor)

# model_test (M5-T07): the DecoderLayer/Mlp modules, ReferenceModel end-to-end
# logits vs activations.safetensors, the per-layer activation hook, kLast/kAll
# consistency, and the KV-cache invariant elevated to full-model logits. Uses
# the loader + SafetensorsFile (engine::model), the cpu ops (via the modules),
# and the real SimpleKvCache (engine::kvcache). `model` label.
engine_add_tests(TREE unit SOURCES
  model_test.cpp
  DEPS engine::model engine::kvcache engine::cpu
  LABELS model)

# registry_test (M5-T08): the architecture registry & builder — both built-in
# families resolve, unknown → Unimplemented, BuildModel is bit-identical to a
# direct ReferenceModel::Create, and a test-local dummy arch is addable in one
# RegisterArchitecture call. Pure portable C++ — no SCALAR_PASS.
engine_add_tests(TREE unit SOURCES
  registry_test.cpp
  DEPS engine::model engine::kvcache engine::cpu
  LABELS model)

# tokenizer_test parses the committed real tokenizer.json fixtures (design
# §7.1, §8); the `tokenizer` label lets `ctest -L tokenizer` iterate the
# golden suite alone. Ordinary portable C++ — no SCALAR_PASS.
engine_add_tests(TREE unit SOURCES
  bpe_test.cpp
  pretokenize_test.cpp
  tokenizer_test.cpp
  unicode_test.cpp
  DEPS engine::tokenizer
  LABELS tokenizer)

# tokenizer_encode_test replays vectors.json (the committed HF goldens)
# through encode, hence the test-only nlohmann_json dependency — same
# pattern as weight_map_test. tokenizer_decode_test replays the same
# vectors through decode and the streaming detokenizer (M4-T10).
engine_add_tests(TREE unit SOURCES
  tokenizer_decode_test.cpp
  tokenizer_encode_test.cpp
  DEPS engine::tokenizer nlohmann_json::nlohmann_json
  LABELS tokenizer)

# The extra `parallel` label lets the tsan CI job (and local TSAN builds)
# select these suites with `ctest -L parallel` — TSAN's slowdown is spent
# where the threads are (design: docs/design/cpu-backend.md §8.2).
# parallel_stress_test.cpp is the many-small-regions TSAN workhorse.
engine_add_tests(TREE unit SOURCES
  parallel_test.cpp
  parallel_stress_test.cpp
  DEPS engine::parallel
  LABELS parallel)

# SCALAR_PASS registers kernel-dependent suites twice — the host's best ISA
# (unit/) and ENGINE_FORCE_ISA=scalar (unit-scalar/) — per design §8.1.
engine_add_tests(TREE unit SOURCES
  dispatch_test.cpp
  elementwise_test.cpp
  reduce_test.cpp
  convert_test.cpp
  DEPS engine::kernels
  LABELS kernels
  SCALAR_PASS)

# The ENGINE_FORCE_ISA failure modes, exercised through the real environment
# variable (M3 audit; design §4.3). The in-process death tests can only reach
# detail::ResolveIsa — SelectedIsa()'s memoization hides the env path — so
# these registrations prove a process with a bad value actually dies, with
# the actionable message, at its first dispatch. The binary runs under
# `sh -c "! …"`: CTest fails any test that terminates on a signal even when
# PASS_REGULAR_EXPRESSION matches, so the shell absorbs the expected abort
# (and the inversion fails the test if the binary unexpectedly survives);
# the message regex is the real assertion. The forced ISA for the
# unavailable case is whichever vector ISA this build does not carry
# (mirrors src/kernels/CMakeLists.txt's arch selection).
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" engine_test_processor)
if(engine_test_processor MATCHES "^(arm64|aarch64)$")
  set(engine_unavailable_isa avx2)
else()
  set(engine_unavailable_isa neon)
endif()
add_test(NAME unit/dispatch-env/unknown-force-isa-dies
  COMMAND sh -c
    "! '$<TARGET_FILE:unit_dispatch_test>' --gtest_filter=DispatchTest.SelectedIsaIsStable 2>&1")
set_tests_properties(unit/dispatch-env/unknown-force-isa-dies PROPERTIES
  ENVIRONMENT "ENGINE_FORCE_ISA=avx512"
  PASS_REGULAR_EXPRESSION
    "ENGINE_FORCE_ISA=\"avx512\" is not a supported value"
  LABELS "unit;kernels")
add_test(NAME unit/dispatch-env/unavailable-force-isa-dies
  COMMAND sh -c
    "! '$<TARGET_FILE:unit_dispatch_test>' --gtest_filter=DispatchTest.SelectedIsaIsStable 2>&1")
set_tests_properties(unit/dispatch-env/unavailable-force-isa-dies PROPERTIES
  ENVIRONMENT "ENGINE_FORCE_ISA=${engine_unavailable_isa}"
  PASS_REGULAR_EXPRESSION
    "ENGINE_FORCE_ISA=\"${engine_unavailable_isa}\" is not executable on this host"
  LABELS "unit;kernels")
