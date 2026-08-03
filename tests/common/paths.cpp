#include "common/paths.h"

namespace engine::testing {

std::filesystem::path FixturesDir() {
  // Injected by tests/common/CMakeLists.txt; points into the source tree so
  // tests are independent of the build directory location.
  return std::filesystem::path(ENGINE_TEST_FIXTURES_DIR);
}

}  // namespace engine::testing
