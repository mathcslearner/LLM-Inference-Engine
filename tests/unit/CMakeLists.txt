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
