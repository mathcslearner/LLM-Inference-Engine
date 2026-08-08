# One line per test file; engine_add_tests (tests/CMakeLists.txt) turns each
# into its own executable and registers it with CTest.
engine_add_tests(TREE integration SOURCES
  smoke_test.cpp)

# load_model end-to-end over the committed tiny-llama fixture (M4-T07).
engine_add_tests(TREE integration SOURCES
  loader_test.cpp
  DEPS engine::model
  LABELS model)
