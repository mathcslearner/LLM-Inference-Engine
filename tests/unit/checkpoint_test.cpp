#include "model/checkpoint.h"

#include "common/paths.h"
#include "core/status.h"
#include "model/safetensors.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

// Checkpoint (M4-T05; design: docs/design/model-loading.md §3.1, §3.6).
// Positive coverage runs the committed tiny-llama fixtures in both layouts
// (the sharded copy holds the same tensors as the single file, byte for
// byte); negative cases synthesize minimal checkpoint directories and break
// one thing per case, asserting the status code and that the message names
// the offending file/tensor (the actionability criterion).

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::core::StatusOr;
using engine::model::Checkpoint;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// --- helpers ---------------------------------------------------------------

[[nodiscard]] std::filesystem::path ScratchDir() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / "checkpoint_test" /
      info->name();
  std::filesystem::create_directories(dir);
  return dir;
}

void WriteFile(const std::filesystem::path& path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(out.good()) << path;
}

// Serializes a safetensors file: u64-LE header length + header + data.
[[nodiscard]] std::string BuildFileBytes(std::string_view header,
                                         std::string_view data) {
  std::string bytes(8, '\0');
  const std::uint64_t header_len = header.size();
  std::memcpy(bytes.data(), &header_len, sizeof(header_len));
  bytes.append(header);
  bytes.append(data);
  return bytes;
}

// A two-shard checkpoint: "a" F32 [2] = {1,2} in s1, "b" I8 [4] = {5,6,7,8}
// in s2. Individual tests overwrite pieces to break one invariant.
constexpr std::string_view kShard1Name = "s1.safetensors";
constexpr std::string_view kShard2Name = "s2.safetensors";
constexpr std::string_view kIndexJson =
    R"({"metadata":{"total_size":12},)"
    R"("weight_map":{"a":"s1.safetensors","b":"s2.safetensors"}})";

[[nodiscard]] std::string Shard1Bytes() {
  const float values[2] = {1.0F, 2.0F};
  std::string data(sizeof(values), '\0');
  std::memcpy(data.data(), &values, sizeof(values));
  return BuildFileBytes(
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", data);
}

[[nodiscard]] std::string Shard2Bytes() {
  const std::int8_t values[4] = {5, 6, 7, 8};
  std::string data(sizeof(values), '\0');
  std::memcpy(data.data(), &values, sizeof(values));
  return BuildFileBytes(
      R"({"b":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}})", data);
}

// Writes the intact two-shard checkpoint into a fresh scratch dir.
[[nodiscard]] std::filesystem::path WriteTwoShardCheckpoint() {
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json", kIndexJson);
  WriteFile(dir / kShard1Name, Shard1Bytes());
  WriteFile(dir / kShard2Name, Shard2Bytes());
  return dir;
}

// The negative-index workhorse: an index with `index_json` (shards intact)
// must fail Open with InvalidArgument, naming the index file and `needle`.
void ExpectIndexInvalid(std::string_view index_json, std::string_view needle) {
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json", index_json);
  WriteFile(dir / kShard1Name, Shard1Bytes());
  WriteFile(dir / kShard2Name, Shard2Bytes());
  const StatusOr<Checkpoint> checkpoint = Checkpoint::Open(dir);
  ASSERT_FALSE(checkpoint.ok()) << "index: " << index_json;
  EXPECT_TRUE(IsInvalidArgument(checkpoint.status()))
      << checkpoint.status().ToString();
  EXPECT_NE(checkpoint.status().message().find("model.safetensors.index.json"),
            std::string::npos)
      << checkpoint.status().ToString();
  EXPECT_NE(checkpoint.status().message().find(needle), std::string::npos)
      << checkpoint.status().ToString();
}

// --- synthetic positives ---------------------------------------------------

TEST(CheckpointTest, ShardedRoundTrip) {
  const std::filesystem::path dir = WriteTwoShardCheckpoint();
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  const std::vector<std::string_view> names = checkpoint->names();
  ASSERT_EQ(names.size(), 2U);
  EXPECT_EQ(names[0], "a");
  EXPECT_EQ(names[1], "b");
  EXPECT_TRUE(checkpoint->contains("a"));
  EXPECT_TRUE(checkpoint->contains("b"));
  EXPECT_FALSE(checkpoint->contains("c"));

  const StatusOr<Tensor> a = checkpoint->tensor("a");
  ASSERT_TRUE(a.ok()) << a.status().ToString();
  EXPECT_EQ(a->dtype(), DataType::kFloat32);
  EXPECT_EQ(a->shape(), (Shape{2}));
  EXPECT_EQ(a->item<float>({0}), 1.0F);
  EXPECT_EQ(a->item<float>({1}), 2.0F);

  const StatusOr<Tensor> b = checkpoint->tensor("b");
  ASSERT_TRUE(b.ok()) << b.status().ToString();
  EXPECT_EQ(b->dtype(), DataType::kInt8);
  EXPECT_EQ(b->item<std::int8_t>({0}), 5);
  EXPECT_EQ(b->item<std::int8_t>({3}), 8);
}

TEST(CheckpointTest, TensorOutlivesCheckpoint) {
  // Returned views share each shard's mapping, independent of the
  // Checkpoint's lifetime (same contract as SafetensorsFile).
  const std::filesystem::path dir = WriteTwoShardCheckpoint();
  StatusOr<Tensor> a = engine::core::NotFoundError("unset");
  {
    auto checkpoint = Checkpoint::Open(dir);
    ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();
    a = checkpoint->tensor("a");
  }
  ASSERT_TRUE(a.ok()) << a.status().ToString();
  EXPECT_EQ(a->item<float>({1}), 2.0F);
}

TEST(CheckpointTest, IndexWinsWhenBothSpellingsPresent) {
  // HF loading order (design §3.1): a deliberately corrupt
  // model.safetensors alongside a valid sharded set must be ignored.
  const std::filesystem::path dir = WriteTwoShardCheckpoint();
  WriteFile(dir / "model.safetensors", "garbage, not a safetensors file");
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();
  const StatusOr<Tensor> b = checkpoint->tensor("b");
  ASSERT_TRUE(b.ok()) << b.status().ToString();
  EXPECT_EQ(b->item<std::int8_t>({1}), 6);
}

TEST(CheckpointTest, ShardsAreMappedLazily) {
  // Open checks only shard existence; a corrupt shard surfaces on the first
  // tensor() touching it, and untouched shards are never parsed (§3.6).
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json", kIndexJson);
  WriteFile(dir / kShard1Name, Shard1Bytes());
  WriteFile(dir / kShard2Name, "garbage, not a safetensors file");
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  const StatusOr<Tensor> a = checkpoint->tensor("a");
  ASSERT_TRUE(a.ok()) << a.status().ToString();

  const StatusOr<Tensor> b = checkpoint->tensor("b");
  ASSERT_FALSE(b.ok());
  EXPECT_TRUE(IsInvalidArgument(b.status())) << b.status().ToString();
  EXPECT_NE(b.status().message().find(kShard2Name), std::string::npos)
      << b.status().ToString();
}

TEST(CheckpointTest, EmptyWeightMapIsAValidEmptyCheckpoint) {
  // Mirrors the tensor-free single file being valid (safetensors_test).
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json", R"({"weight_map":{}})");
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();
  EXPECT_TRUE(checkpoint->names().empty());
  EXPECT_FALSE(checkpoint->contains("a"));
}

TEST(CheckpointTest, MetadataAndUnknownTopLevelKeysAreTolerated) {
  // "metadata" is informational and never interpreted; additive HF-tooling
  // fields must not break loading.
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json",
            R"({"metadata":{"total_size":8},"future_field":[1,2],)"
            R"("weight_map":{"a":"s1.safetensors"}})");
  WriteFile(dir / kShard1Name, Shard1Bytes());
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();
  EXPECT_TRUE(checkpoint->tensor("a").ok());
}

// --- synthetic negatives ---------------------------------------------------

TEST(CheckpointTest, MissingDirectoryIsNotFound) {
  const StatusOr<Checkpoint> checkpoint =
      Checkpoint::Open(ScratchDir() / "no-such-subdir");
  ASSERT_FALSE(checkpoint.ok());
  EXPECT_TRUE(IsNotFound(checkpoint.status()))
      << checkpoint.status().ToString();
  EXPECT_NE(checkpoint.status().message().find("no-such-subdir"),
            std::string::npos)
      << checkpoint.status().ToString();
}

TEST(CheckpointTest, NeitherSpellingIsNotFoundListingBoth) {
  const std::filesystem::path dir = ScratchDir();  // exists, but empty
  const StatusOr<Checkpoint> checkpoint = Checkpoint::Open(dir);
  ASSERT_FALSE(checkpoint.ok());
  EXPECT_TRUE(IsNotFound(checkpoint.status()))
      << checkpoint.status().ToString();
  const engine::core::Status status = checkpoint.status();
  const std::string_view message = status.message();
  EXPECT_NE(message.find("model.safetensors"), std::string_view::npos)
      << checkpoint.status().ToString();
  EXPECT_NE(message.find("model.safetensors.index.json"),
            std::string_view::npos)
      << checkpoint.status().ToString();
}

TEST(CheckpointTest, MissingShardsFailAtOpenListingAll) {
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json", kIndexJson);
  // Neither shard written: both must be listed, so one round-trip fixes it.
  const StatusOr<Checkpoint> checkpoint = Checkpoint::Open(dir);
  ASSERT_FALSE(checkpoint.ok());
  EXPECT_TRUE(IsNotFound(checkpoint.status()))
      << checkpoint.status().ToString();
  const engine::core::Status status = checkpoint.status();
  const std::string_view message = status.message();
  EXPECT_NE(message.find(kShard1Name), std::string_view::npos)
      << checkpoint.status().ToString();
  EXPECT_NE(message.find(kShard2Name), std::string_view::npos)
      << checkpoint.status().ToString();
}

TEST(CheckpointTest, MalformedIndexIsInvalidNamingTheDefect) {
  ExpectIndexInvalid("{", "not valid JSON");
  ExpectIndexInvalid("", "not valid JSON");
  ExpectIndexInvalid("[]", "must be a JSON object");
  ExpectIndexInvalid(R"({"metadata":{"total_size":12}})", R"("weight_map")");
  ExpectIndexInvalid(R"({"weight_map":[]})", "must be a JSON object");
  ExpectIndexInvalid(R"({"weight_map":{"a":7}})", "must name a shard file");
}

TEST(CheckpointTest, NonBareShardFilenamesAreRejected) {
  // A hostile index must not address files outside the model directory.
  ExpectIndexInvalid(R"({"weight_map":{"a":"../evil.safetensors"}})",
                     "not a bare filename");
  ExpectIndexInvalid(R"({"weight_map":{"a":"sub/s1.safetensors"}})",
                     "not a bare filename");
  ExpectIndexInvalid(R"({"weight_map":{"a":"sub\\s1.safetensors"}})",
                     "not a bare filename");
  ExpectIndexInvalid(R"({"weight_map":{"a":".."}})", "not a bare filename");
  ExpectIndexInvalid(R"({"weight_map":{"a":""}})", "not a bare filename");
}

TEST(CheckpointTest, IndexNamingTensorAbsentFromShardIsInconsistent) {
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json",
            R"({"weight_map":{"a":"s1.safetensors",)"
            R"("phantom":"s1.safetensors"}})");
  WriteFile(dir / kShard1Name, Shard1Bytes());
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  // The first touch of s1 runs the consistency check, so even the tensor
  // that IS present fails until the checkpoint is repaired.
  const StatusOr<Tensor> a = checkpoint->tensor("a");
  ASSERT_FALSE(a.ok());
  EXPECT_TRUE(IsInvalidArgument(a.status())) << a.status().ToString();
  const engine::core::Status status = a.status();
  const std::string_view message = status.message();
  EXPECT_NE(message.find(kShard1Name), std::string_view::npos)
      << a.status().ToString();
  EXPECT_NE(message.find("phantom"), std::string_view::npos)
      << a.status().ToString();
  EXPECT_NE(message.find("absent"), std::string_view::npos)
      << a.status().ToString();
}

TEST(CheckpointTest, ShardTensorAbsentFromIndexIsInconsistent) {
  const std::filesystem::path dir = ScratchDir();
  // s1 holds "a", but the index only knows about a renamed "a2".
  WriteFile(dir / "model.safetensors.index.json",
            R"({"weight_map":{"a2":"s1.safetensors"}})");
  WriteFile(dir / kShard1Name, Shard1Bytes());
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  const StatusOr<Tensor> a2 = checkpoint->tensor("a2");
  ASSERT_FALSE(a2.ok());
  EXPECT_TRUE(IsInvalidArgument(a2.status())) << a2.status().ToString();
  const engine::core::Status status = a2.status();
  const std::string_view message = status.message();
  EXPECT_NE(message.find("not in the index"), std::string_view::npos)
      << a2.status().ToString();
}

TEST(CheckpointTest, DuplicateTensorAcrossShardsIsInconsistent) {
  // The JSON weight_map cannot express a duplicate, but two shards
  // *containing* the same name can (§3.6): the index maps "a" to s1, yet s2
  // also carries an "a". Mapping s2 must fail.
  const std::filesystem::path dir = ScratchDir();
  WriteFile(dir / "model.safetensors.index.json", kIndexJson);
  WriteFile(dir / kShard1Name, Shard1Bytes());
  const float values[2] = {9.0F, 9.0F};
  std::string duplicate_data(sizeof(values), '\0');
  std::memcpy(duplicate_data.data(), &values, sizeof(values));
  WriteFile(dir / kShard2Name,
            BuildFileBytes(
                R"({"b":{"dtype":"I8","shape":[4],"data_offsets":[0,4]},)"
                R"("a":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
                std::string(4, '\0') + duplicate_data));
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();

  const StatusOr<Tensor> b = checkpoint->tensor("b");
  ASSERT_FALSE(b.ok());
  EXPECT_TRUE(IsInvalidArgument(b.status())) << b.status().ToString();
  const engine::core::Status status = b.status();
  const std::string_view message = status.message();
  EXPECT_NE(message.find(kShard2Name), std::string_view::npos)
      << b.status().ToString();
  EXPECT_NE(message.find("maps it to"), std::string_view::npos)
      << b.status().ToString();
  EXPECT_NE(message.find(kShard1Name), std::string_view::npos)
      << b.status().ToString();
}

TEST(CheckpointTest, UnknownTensorNameIsNotFound) {
  const std::filesystem::path dir = WriteTwoShardCheckpoint();
  auto checkpoint = Checkpoint::Open(dir);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();
  const StatusOr<Tensor> missing = checkpoint->tensor("nope");
  ASSERT_FALSE(missing.ok());
  EXPECT_TRUE(IsNotFound(missing.status())) << missing.status().ToString();
  EXPECT_NE(missing.status().message().find("\"nope\""), std::string::npos)
      << missing.status().ToString();
}

// --- committed fixtures ----------------------------------------------------

TEST(CheckpointTest, SingleFileFixtureThroughCheckpoint) {
  // The single-file layout behind the same interface (tiny-llama's root has
  // model.safetensors and no index).
  auto checkpoint =
      Checkpoint::Open(engine::testing::FixturesDir() / "models/tiny-llama");
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().ToString();
  EXPECT_EQ(checkpoint->names().size(), 21U);
  ASSERT_TRUE(checkpoint->contains("model.norm.weight"));
  const StatusOr<Tensor> norm = checkpoint->tensor("model.norm.weight");
  ASSERT_TRUE(norm.ok()) << norm.status().ToString();
  EXPECT_EQ(norm->dtype(), DataType::kBFloat16);
  EXPECT_EQ(norm->shape(), (Shape{64}));
}

TEST(CheckpointTest, ShardedFixtureMatchesSingleFileBitForBit) {
  // The acceptance criterion: every tensor of the 2-shard fixture resolves,
  // and its metadata and bytes are identical to the single-file fixture's
  // (the generator writes the same tensors into both layouts).
  const std::filesystem::path root =
      engine::testing::FixturesDir() / "models/tiny-llama";
  auto sharded = Checkpoint::Open(root / "sharded");
  ASSERT_TRUE(sharded.ok()) << sharded.status().ToString();
  const auto single = SafetensorsFile::Open(root / "model.safetensors");
  ASSERT_TRUE(single.ok()) << single.status().ToString();

  const std::vector<std::string_view> sharded_names = sharded->names();
  const std::vector<std::string_view> single_names = single->names();
  ASSERT_EQ(sharded_names, single_names);

  for (const std::string_view name : single_names) {
    const StatusOr<Tensor> expected = single->tensor(name);
    ASSERT_TRUE(expected.ok()) << name << ": " << expected.status().ToString();
    const StatusOr<Tensor> actual = sharded->tensor(name);
    ASSERT_TRUE(actual.ok()) << name << ": " << actual.status().ToString();
    ASSERT_EQ(actual->dtype(), expected->dtype()) << name;
    ASSERT_EQ(actual->shape(), expected->shape()) << name;
    const std::size_t bytes =
        static_cast<std::size_t>(expected->numel()) *
        static_cast<std::size_t>(engine::tensor::itemsize(expected->dtype()));
    EXPECT_EQ(std::memcmp(actual->data(), expected->data(), bytes), 0)
        << name << ": sharded bytes differ from the single-file fixture";
  }
}

}  // namespace
