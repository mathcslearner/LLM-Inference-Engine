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
  DEPS engine::kernels
  LABELS kernels
  SCALAR_PASS)
