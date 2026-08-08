# Test harness (M0-T03). Every test source file becomes its own executable so
# a crash in one file cannot take out unrelated tests, `ctest -j` parallelizes
# across files, and each binary stays small. Registration is one line in one
# list per tree — see tests/README.md.

include(GoogleTest)

# engine_add_tests(TREE <unit|integration> SOURCES <file...> [DEPS <target...>]
#                  [LABELS <label...>] [SCALAR_PASS])
#
# For each source file `foo_test.cpp`, creates executable `<tree>_foo_test`
# linked against the shared test helpers and GoogleTest's default main, and
# registers its tests with CTest under the ctest-name prefix "<tree>/",
# labeled `<tree>` (ctest -L matches labels by regex, so `-L unit` and
# `-L integration` select the trees). DEPS lists the engine targets under
# test. LABELS appends extra ctest labels so a CI job can select a subset —
# e.g. the tsan job runs `ctest -L parallel` (M3-T04).
#
# SCALAR_PASS registers the same binary with ctest a second time, under the
# prefix "<tree>-scalar/" with ENGINE_FORCE_ISA=scalar in the environment
# and the extra label `scalar` — the forced-scalar pass (M3-T05; design:
# docs/design/cpu-backend.md §8.1). Same binary, same test sources, no
# #ifdef-forked expectations: a scalar-vs-vector behavioral difference fails
# exactly one of the two passes, which localizes it immediately. Only
# kernel-dependent suites opt in — dispatch resolution is per-process, so
# doubling suites that never dispatch buys nothing.
function(engine_add_tests)
  cmake_parse_arguments(ARG "SCALAR_PASS" "TREE" "SOURCES;DEPS;LABELS" ${ARGN})
  set(test_labels ${ARG_TREE} ${ARG_LABELS})
  foreach(source IN LISTS ARG_SOURCES)
    cmake_path(GET source STEM name)
    set(target ${ARG_TREE}_${name})
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
      engine_warnings
      engine::test_common
      GTest::gtest_main
      ${ARG_DEPS})
    # PRE_TEST defers test enumeration from build time to ctest time.
    gtest_discover_tests(${target}
      TEST_PREFIX "${ARG_TREE}/"
      DISCOVERY_MODE PRE_TEST
      PROPERTIES LABELS "${test_labels}")
    if(ARG_SCALAR_PASS)
      gtest_discover_tests(${target}
        TEST_PREFIX "${ARG_TREE}-scalar/"
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "${test_labels};scalar"
                   ENVIRONMENT "ENGINE_FORCE_ISA=scalar")
    endif()
  endforeach()
endfunction()

add_subdirectory(common)
add_subdirectory(unit)
add_subdirectory(integration)
