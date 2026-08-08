#include "model/mapped_file.h"

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

// MapFileReadOnly (M4-T04; design: docs/design/model-loading.md §3.3): a
// read-only mmap wrapped as a shared memory::Buffer whose deleter munmaps,
// so the mapping lives exactly as long as the last shared_ptr.

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::memory::Buffer;
using engine::model::MapFileReadOnly;
using engine::tensor::Device;

// Fresh per-test scratch directory under gtest's temp root.
[[nodiscard]] std::filesystem::path ScratchDir() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / "mapped_file_test" /
      info->name();
  std::filesystem::create_directories(dir);
  return dir;
}

[[nodiscard]] std::filesystem::path WriteFile(const std::filesystem::path& path,
                                              std::string_view bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  EXPECT_TRUE(out.good());
  return path;
}

TEST(MappedFileTest, MapsFileBytes) {
  std::string contents = "safetensors bytes ... end";
  contents[19] = '\0';  // an embedded NUL must survive the round trip
  const auto path = WriteFile(ScratchDir() / "data.bin", contents);
  auto buffer = MapFileReadOnly(path);
  ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
  ASSERT_NE(*buffer, nullptr);
  EXPECT_EQ((*buffer)->size_bytes(), contents.size());
  EXPECT_EQ((*buffer)->device(), Device::Cpu());
  ASSERT_NE((*buffer)->data(), nullptr);
  EXPECT_EQ(std::memcmp((*buffer)->data(), contents.data(), contents.size()),
            0);
}

TEST(MappedFileTest, EmptyFileIsEngagedZeroSizeBuffer) {
  const auto path = WriteFile(ScratchDir() / "empty.bin", "");
  auto buffer = MapFileReadOnly(path);
  ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
  ASSERT_NE(*buffer, nullptr);
  EXPECT_EQ((*buffer)->size_bytes(), 0U);
  EXPECT_EQ((*buffer)->data(), nullptr);
  EXPECT_EQ((*buffer)->device(), Device::Cpu());
}

TEST(MappedFileTest, MissingFileIsNotFound) {
  const auto path = ScratchDir() / "no-such-file.bin";
  auto buffer = MapFileReadOnly(path);
  ASSERT_FALSE(buffer.ok());
  EXPECT_TRUE(IsNotFound(buffer.status())) << buffer.status().ToString();
  EXPECT_NE(buffer.status().message().find(path.string()), std::string::npos)
      << buffer.status().ToString();
}

TEST(MappedFileTest, DirectoryIsInvalidArgument) {
  auto buffer = MapFileReadOnly(ScratchDir());
  ASSERT_FALSE(buffer.ok());
  EXPECT_TRUE(IsInvalidArgument(buffer.status())) << buffer.status().ToString();
  EXPECT_NE(buffer.status().message().find("not a regular file"),
            std::string::npos)
      << buffer.status().ToString();
}

TEST(MappedFileTest, MappingOutlivesOriginalHandle) {
  const std::string contents(4096, 'x');
  const auto path = WriteFile(ScratchDir() / "outlive.bin", contents);
  auto buffer = MapFileReadOnly(path);
  ASSERT_TRUE(buffer.ok()) << buffer.status().ToString();
  const std::shared_ptr<Buffer> copy = *buffer;
  buffer->reset();  // drop the original shared_ptr
  // The mapping must still be readable through the surviving copy
  // (sanitizers would flag a premature munmap).
  const auto* data = static_cast<const char*>(copy->data());
  for (std::size_t i = 0; i < copy->size_bytes(); i += 512) {
    EXPECT_EQ(data[i], 'x');
  }
}

}  // namespace
