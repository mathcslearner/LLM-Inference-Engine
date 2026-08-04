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

# caching_allocator_test.cpp compiles on every configuration: the pool is
# device-agnostic, so the size-class/stats/reuse/OOM-retry/concurrency
# contracts run over host upstreams everywhere, and the GPU acceptance tests
# skip without a device.
engine_add_tests(TREE unit SOURCES
  caching_allocator_test.cpp
  DEPS engine::memory)

# cuda_allocator_test.cpp compiles on every configuration (stub taxonomy on
# CPU-only builds, index validation + GPU-skipped allocation tests on CUDA
# builds); engine::cuda supplies device_count() for the index tests.
engine_add_tests(TREE unit SOURCES
  cuda_allocator_test.cpp
  DEPS engine::memory engine::cuda)

engine_add_tests(TREE unit SOURCES
  cuda_utils_test.cpp
  stream_test.cpp
  DEPS engine::cuda)

# Only buildable on CUDA builds: cuda_check_test.cpp includes
# cuda/cuda_check.h and therefore the CUDA toolkit (most of it runs without
# a GPU — ToStatus and the macros are pure host code); stream_cuda_test.cu
# launches its own delay kernel (M2-T08's kernels don't exist yet), so it is
# a .cu TU; cuda_allocator_cuda_test.cpp calls cudaMemGetInfo/cudaMemcpy for
# the M2-T05 leak and round-trip checks, and caching_allocator_cuda_test.cpp
# does the same for the M2-T06 released-to-driver and reuse checks.
# GPU-dependent assertions use the shared skip predicate.
if(ENGINE_ENABLE_CUDA)
  engine_add_tests(TREE unit SOURCES
    cuda_check_test.cpp
    stream_cuda_test.cu
    DEPS engine::cuda)
  engine_add_tests(TREE unit SOURCES
    cuda_allocator_cuda_test.cpp
    caching_allocator_cuda_test.cpp
    DEPS engine::memory engine::cuda)
endif()

# engine::cuda supplies device_count() for tensor_test's CUDA-device cases.
engine_add_tests(TREE unit SOURCES
  ops_test.cpp
  tensor_test.cpp
  DEPS engine::tensor engine::cuda)
