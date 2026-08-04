# Test harness (M0-T03). Every test source file becomes its own executable so
# a crash in one file cannot take out unrelated tests, `ctest -j` parallelizes
# across files, and each binary stays small. Registration is one line in one
# list per tree — see tests/README.md.

include(GoogleTest)

# engine_add_tests(TREE <unit|integration> SOURCES <file...> [DEPS <target...>])
#
# For each source file `foo_test.cpp`, creates executable `<tree>_foo_test`
# linked against the shared test helpers and GoogleTest's default main, and
# registers its tests with CTest under the ctest-name prefix "<tree>/".
# DEPS lists the engine targets under test.
#
# Labels (M2-T09, design §10.1): tests in suites named `*GpuTest` /
# `*GpuDeathTest` — the CudaTestFixture naming contract (tests/common/cuda.h)
# — are labeled `<tree>-gpu`; everything else `<tree>`. The combined label is
# a single string because gtest_discover_tests(PROPERTIES) flattens
# list-valued properties at the CMake 3.26 floor; ctest -L matches labels by
# regex, so `-L gpu`, `-L unit`, and `-L integration` all select the right
# sets. Two discovery calls per target split the label sets (supported:
# per-target call counter since 3.10, TEST_FILTER since 3.22; a filter
# matching nothing registers nothing, without error).
function(engine_add_tests)
  cmake_parse_arguments(ARG "" "TREE" "SOURCES;DEPS" ${ARGN})
  foreach(source IN LISTS ARG_SOURCES)
    cmake_path(GET source STEM name)
    set(target ${ARG_TREE}_${name})
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
      engine_warnings
      engine::test_common
      GTest::gtest_main
      ${ARG_DEPS})
    # PRE_TEST defers test enumeration from build time to ctest time, which
    # stays correct for binaries that cannot run at build time (e.g. the
    # GPU-only tests arriving in M2).
    gtest_discover_tests(${target}
      TEST_PREFIX "${ARG_TREE}/"
      TEST_FILTER "*GpuTest.*:*GpuDeathTest.*"
      DISCOVERY_MODE PRE_TEST
      PROPERTIES LABELS "${ARG_TREE}-gpu")
    gtest_discover_tests(${target}
      TEST_PREFIX "${ARG_TREE}/"
      TEST_FILTER "-*GpuTest.*:*GpuDeathTest.*"
      DISCOVERY_MODE PRE_TEST
      PROPERTIES LABELS ${ARG_TREE})
  endforeach()
endfunction()

add_subdirectory(common)
add_subdirectory(unit)
add_subdirectory(integration)
