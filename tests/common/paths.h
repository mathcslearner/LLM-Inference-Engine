#pragma once

#include <filesystem>

namespace engine::testing {

// Absolute path to the committed golden-fixture tree (`tests/fixtures/` in
// the source checkout). Fixture-driven tests resolve their inputs against
// this rather than assuming a working directory.
std::filesystem::path FixturesDir();

}  // namespace engine::testing
