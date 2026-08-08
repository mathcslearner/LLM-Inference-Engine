# Test harness (M0-T03). Every test source file becomes its own executable so
# a crash in one file cannot take out unrelated tests, `ctest -j` parallelizes
# across files, and each binary stays small. Registration is one line in one
# list per tree — see tests/README.md.

include(GoogleTest)

# engine_add_tests(TREE <unit|integration> SOURCES <file...> [DEPS <target...>])
#
# For each source file `foo_test.cpp`, creates executable `<tree>_foo_test`
# linked against the shared test helpers and GoogleTest's default main, and
# registers its tests with CTest under the ctest-name prefix "<tree>/",
# labeled `<tree>` (ctest -L matches labels by regex, so `-L unit` and
# `-L integration` select the trees). DEPS lists the engine targets under
# test.
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
    # PRE_TEST defers test enumeration from build time to ctest time.
    gtest_discover_tests(${target}
      TEST_PREFIX "${ARG_TREE}/"
      DISCOVERY_MODE PRE_TEST
      PROPERTIES LABELS ${ARG_TREE})
  endforeach()
endfunction()

add_subdirectory(common)
add_subdirectory(unit)
add_subdirectory(integration)
