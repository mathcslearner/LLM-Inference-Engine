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

# M2-T09: self-tests for the shared GPU test infrastructure (CudaTestFixture
# + ExpectTensorsClose in tests/common/cuda.h). Compiles on every
# configuration; every test skips without a device.
engine_add_tests(TREE unit SOURCES
  cuda_fixture_test.cpp
  DEPS engine::tensor engine::cuda)

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

# M2-T07: both compile on every configuration. pinned_allocator_test links
# engine::tensor for the pinned-Tensor smoke test; transfer_test's
# validation/H2H cases run everywhere (design §8.1) and its GPU round-trip,
# stream/event, and pinned acceptance tests skip without a device.
engine_add_tests(TREE unit SOURCES
  pinned_allocator_test.cpp
  transfer_test.cpp
  DEPS engine::tensor engine::cuda)

# M2-T08: compiles on every configuration — the LaunchConfig1D goldens and
# the launcher validation contract (always-compiled elementwise.cpp) run
# everywhere; the kernel-vs-CPU-reference acceptance tests skip without a
# device. engine::cuda supplies CudaStream for the GPU tests.
engine_add_tests(TREE unit SOURCES
  elementwise_test.cpp
  DEPS engine::kernels engine::cuda)

# Only buildable on CUDA builds: dispatch_test.cu includes
# kernels/dispatch.h and therefore the CUDA toolkit (it runs without a GPU —
# the macro's switch and Unimplemented default are pure host code).
if(ENGINE_ENABLE_CUDA)
  engine_add_tests(TREE unit SOURCES
    dispatch_test.cu
    DEPS engine::kernels)
endif()
