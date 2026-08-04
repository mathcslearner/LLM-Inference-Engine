# Shared helpers available to every test via `#include "common/..."`.
# GTest::gtest is PUBLIC so helpers may use gtest types in their interfaces.
add_library(engine_test_common STATIC cuda.cpp paths.cpp)
add_library(engine::test_common ALIAS engine_test_common)
target_include_directories(engine_test_common PUBLIC ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(engine_test_common PRIVATE
  ENGINE_TEST_FIXTURES_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
target_link_libraries(engine_test_common
  PUBLIC GTest::gtest
  PRIVATE engine::cuda engine_warnings)
