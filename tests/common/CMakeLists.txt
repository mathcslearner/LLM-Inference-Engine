# Shared helpers available to every test via `#include "common/..."`.
# GTest::gtest is PUBLIC so helpers may use gtest types in their interfaces.
add_library(engine_test_common STATIC cuda.cpp paths.cpp)
add_library(engine::test_common ALIAS engine_test_common)
target_include_directories(engine_test_common PUBLIC ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(engine_test_common PRIVATE
  ENGINE_TEST_FIXTURES_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
# cuda/memory/tensor are PUBLIC since M2-T09: common/cuda.h's CudaTestFixture
# and ExpectTensorsClose expose CudaStream, CachingAllocator, and Tensor in
# their interfaces.
target_link_libraries(engine_test_common
  PUBLIC GTest::gtest engine::cuda engine::memory engine::tensor
  PRIVATE engine_warnings)
