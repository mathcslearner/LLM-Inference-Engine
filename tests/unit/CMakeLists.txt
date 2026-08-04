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

engine_add_tests(TREE unit SOURCES
  allocator_test.cpp
  DEPS engine::memory)

engine_add_tests(TREE unit SOURCES
  cuda_utils_test.cpp
  stream_test.cpp
  DEPS engine::cuda)

# Only buildable on CUDA builds: cuda_check_test.cpp includes
# cuda/cuda_check.h and therefore the CUDA toolkit (most of it runs without
# a GPU — ToStatus and the macros are pure host code); stream_cuda_test.cu
# launches its own delay kernel (M2-T08's kernels don't exist yet), so it is
# a .cu TU. GPU-dependent assertions use the shared skip predicate.
if(ENGINE_ENABLE_CUDA)
  engine_add_tests(TREE unit SOURCES
    cuda_check_test.cpp
    stream_cuda_test.cu
    DEPS engine::cuda)
endif()

engine_add_tests(TREE unit SOURCES
  ops_test.cpp
  tensor_test.cpp
  DEPS engine::tensor)
