#include "model/safetensors.h"

#include "common/paths.h"
#include "core/status.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// SafetensorsFile (M4-T04; design: docs/design/model-loading.md §3.4).
// Positive coverage runs against the committed tiny-llama fixtures; the
// fuzz-ish negative table synthesizes minimal files and corrupts one thing
// per case, asserting both the status code and that the message names the
// offending file/tensor/field (the actionability criterion).

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::model::SafetensorsFile;
using engine::tensor::DataType;
using engine::tensor::Shape;
using engine::tensor::Tensor;

// --- helpers ---------------------------------------------------------------

[[nodiscard]] std::filesystem::path ScratchDir() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / "safetensors_test" /
      info->name();
  std::filesystem::create_directories(dir);
  return dir;
}

// Serializes a safetensors file exactly as given: u64-LE header length +
// header + data. Only the misaligned-data-section test wants this — every
// other case goes through the padding wrapper below.
[[nodiscard]] std::string BuildFileBytesUnpadded(std::string_view header,
                                                 std::string_view data) {
  std::string bytes(8, '\0');
  const std::uint64_t header_len = header.size();
  std::memcpy(bytes.data(), &header_len, sizeof(header_len));
  bytes.append(header);
  bytes.append(data);
  return bytes;
}

// Serializes a safetensors file, space-padding the header (legal trailing
// JSON whitespace) so the data section starts 8-aligned — what real HF
// serializers do, and what the parser's absolute-alignment check expects.
[[nodiscard]] std::string BuildFileBytes(std::string_view header,
                                         std::string_view data) {
  std::string padded(header);
  padded.append((8 - (8 + padded.size()) % 8) % 8, ' ');
  return BuildFileBytesUnpadded(padded, data);
}

// Writes `bytes` to a fresh file and opens it. Each call uses a distinct
// filename so one test can exercise several files.
[[nodiscard]] StatusOr<SafetensorsFile> OpenBytes(std::string_view bytes) {
  static int counter = 0;
  const std::filesystem::path path =
      ScratchDir() / ("case-" + std::to_string(counter++) + ".safetensors");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  EXPECT_TRUE(out.good());
  out.close();
  return SafetensorsFile::Open(path);
}

[[nodiscard]] StatusOr<SafetensorsFile> OpenSynthetic(std::string_view header,
                                                      std::string_view data) {
  return OpenBytes(BuildFileBytes(header, data));
}

// The negative-case workhorse: Open must fail with InvalidArgument and a
// message containing `needle` (and the file path, checked once in
// ErrorsNameTheFile rather than in every case).
void ExpectInvalid(std::string_view header, std::string_view data,
                   std::string_view needle) {
  const StatusOr<SafetensorsFile> file = OpenSynthetic(header, data);
  ASSERT_FALSE(file.ok()) << "header: " << header;
  EXPECT_TRUE(IsInvalidArgument(file.status())) << file.status().ToString();
  EXPECT_NE(file.status().message().find(needle), std::string::npos)
      << file.status().ToString();
}

// A valid two-tensor file: "a" F32 [2,2] = {1,2,3,4}, "b" I8 [4] = {5,6,7,8}.
constexpr std::string_view kTwoTensorHeader =
    R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
    R"("b":{"dtype":"I8","shape":[4],"data_offsets":[16,20]},)"
    R"("__metadata__":{"format":"pt"}})";

[[nodiscard]] std::string TwoTensorData() {
  std::string data(20, '\0');
  const float floats[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  std::memcpy(data.data(), &floats, sizeof(floats));
  const std::int8_t ints[4] = {5, 6, 7, 8};
  std::memcpy(data.data() + sizeof(floats), &ints, sizeof(ints));
  return data;
}

// --- synthetic positives -----------------------------------------------------

TEST(SafetensorsTest, SyntheticRoundTrip) {
  const auto file = OpenSynthetic(kTwoTensorHeader, TwoTensorData());
  ASSERT_TRUE(file.ok()) << file.status().ToString();

  const std::vector<std::string_view> names = file->names();
  ASSERT_EQ(names.size(), 2U);
  EXPECT_EQ(names[0], "a");
  EXPECT_EQ(names[1], "b");
  EXPECT_TRUE(file->contains("a"));
  EXPECT_TRUE(file->contains("b"));
  EXPECT_FALSE(file->contains("c"));
  ASSERT_EQ(file->metadata().size(), 1U);
  EXPECT_EQ(file->metadata().at("format"), "pt");

  const StatusOr<Tensor> a = file->tensor("a");
  ASSERT_TRUE(a.ok()) << a.status().ToString();
  EXPECT_EQ(a->dtype(), DataType::kFloat32);
  EXPECT_EQ(a->shape(), (Shape{2, 2}));
  EXPECT_TRUE(a->is_contiguous());
  // Data section starts at 8 + header_len, with the header space-padded to
  // the next 8-byte boundary; "a" begins at its offset 0.
  const std::size_t data_start = (8 + kTwoTensorHeader.size() + 7) / 8 * 8;
  EXPECT_EQ(a->byte_offset(), data_start);
  EXPECT_EQ(a->item<float>({0, 0}), 1.0F);
  EXPECT_EQ(a->item<float>({1, 1}), 4.0F);

  const StatusOr<Tensor> b = file->tensor("b");
  ASSERT_TRUE(b.ok()) << b.status().ToString();
  EXPECT_EQ(b->dtype(), DataType::kInt8);
  EXPECT_EQ(b->shape(), (Shape{4}));
  EXPECT_EQ(b->item<std::int8_t>({0}), 5);
  EXPECT_EQ(b->item<std::int8_t>({3}), 8);
}

TEST(SafetensorsTest, ScalarAndZeroNumelTensors) {
  // Scalar (shape []) and zero-numel tensors are legal (design §3.4); the
  // zero-length range abuts the scalar's end, keeping the layout gap-free.
  const float value = 7.0F;
  std::string data(sizeof(value), '\0');
  std::memcpy(data.data(), &value, sizeof(value));
  const auto file =
      OpenSynthetic(R"({"s":{"dtype":"F32","shape":[],"data_offsets":[0,4]},)"
                    R"("z":{"dtype":"F16","shape":[0],"data_offsets":[4,4]}})",
                    data);
  ASSERT_TRUE(file.ok()) << file.status().ToString();

  const StatusOr<Tensor> scalar = file->tensor("s");
  ASSERT_TRUE(scalar.ok()) << scalar.status().ToString();
  EXPECT_EQ(scalar->shape().rank(), 0);
  EXPECT_EQ(scalar->item<float>({}), 7.0F);

  const StatusOr<Tensor> zero = file->tensor("z");
  ASSERT_TRUE(zero.ok()) << zero.status().ToString();
  EXPECT_EQ(zero->dtype(), DataType::kFloat16);
  EXPECT_EQ(zero->numel(), 0);
}

TEST(SafetensorsTest, TensorFreeFilesAreValid) {
  const auto empty = OpenSynthetic("{}", "");
  ASSERT_TRUE(empty.ok()) << empty.status().ToString();
  EXPECT_TRUE(empty->names().empty());
  EXPECT_TRUE(empty->metadata().empty());

  const auto metadata_only = OpenSynthetic(R"({"__metadata__":{"k":"v"}})", "");
  ASSERT_TRUE(metadata_only.ok()) << metadata_only.status().ToString();
  EXPECT_TRUE(metadata_only->names().empty());
  EXPECT_EQ(metadata_only->metadata().at("k"), "v");
}

TEST(SafetensorsTest, F8E4M3ParsesButMaterializesUnimplemented) {
  // The parser is format-complete: an F8_E4M3 entry is valid metadata, but
  // the dtype is reserved until M13, so the view cannot be built (design §5).
  const auto file = OpenSynthetic(
      R"({"q":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}})",
      std::string(2, '\0'));
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  EXPECT_TRUE(file->contains("q"));
  const StatusOr<Tensor> q = file->tensor("q");
  ASSERT_FALSE(q.ok());
  EXPECT_TRUE(IsUnimplemented(q.status())) << q.status().ToString();
  EXPECT_NE(q.status().message().find("\"q\""), std::string::npos)
      << q.status().ToString();
}

TEST(SafetensorsTest, UnknownTensorNameIsNotFound) {
  const auto file = OpenSynthetic(kTwoTensorHeader, TwoTensorData());
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  const StatusOr<Tensor> missing = file->tensor("nope");
  ASSERT_FALSE(missing.ok());
  EXPECT_TRUE(IsNotFound(missing.status())) << missing.status().ToString();
  EXPECT_NE(missing.status().message().find("\"nope\""), std::string::npos)
      << missing.status().ToString();
}

// --- committed fixtures
// -------------------------------------------------------

TEST(SafetensorsTest, TinyLlamaFixtureMetadata) {
  const auto file = SafetensorsFile::Open(
      engine::testing::FixturesDir() / "models/tiny-llama/model.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();

  const std::vector<std::string_view> names = file->names();
  EXPECT_EQ(names.size(), 21U);
  EXPECT_TRUE(std::ranges::is_sorted(names));

  // Spot-check shapes against the tiny-llama geometry (hidden 64, 2 layers,
  // vocab 512, intermediate 176, 4 heads / 2 KV heads → head_dim 16).
  struct Expected {
    std::string_view name;
    Shape shape;
  };
  const Expected expected[] = {
      {.name = "model.embed_tokens.weight", .shape = Shape{512, 64}},
      {.name = "lm_head.weight", .shape = Shape{512, 64}},
      {.name = "model.layers.0.self_attn.q_proj.weight",
       .shape = Shape{64, 64}},
      {.name = "model.layers.1.self_attn.k_proj.weight",
       .shape = Shape{32, 64}},
      {.name = "model.layers.0.mlp.down_proj.weight", .shape = Shape{64, 176}},
      {.name = "model.layers.1.mlp.gate_proj.weight", .shape = Shape{176, 64}},
      {.name = "model.norm.weight", .shape = Shape{64}},
  };
  for (const Expected& e : expected) {
    ASSERT_TRUE(file->contains(e.name)) << e.name;
    const StatusOr<Tensor> t = file->tensor(e.name);
    ASSERT_TRUE(t.ok()) << t.status().ToString();
    EXPECT_EQ(t->dtype(), DataType::kBFloat16) << e.name;
    EXPECT_EQ(t->shape(), e.shape) << e.name;
    EXPECT_TRUE(t->is_contiguous()) << e.name;
  }
}

TEST(SafetensorsTest, ActivationsFixtureValuesMatchMetaJson) {
  // Cross-format ground truth: expected/meta.json records the exact
  // input_ids the generator fed the model; the I32 tensor in the goldens
  // file must decode to the same list.
  const auto file = SafetensorsFile::Open(
      engine::testing::FixturesDir() /
      "models/tiny-llama/expected/activations.safetensors");
  ASSERT_TRUE(file.ok()) << file.status().ToString();

  const StatusOr<Tensor> ids = file->tensor("input_ids");
  ASSERT_TRUE(ids.ok()) << ids.status().ToString();
  EXPECT_EQ(ids->dtype(), DataType::kInt32);
  ASSERT_EQ(ids->shape(), (Shape{1, 16}));
  const std::int32_t expected[16] = {1, 57,  300, 4,   128, 499, 42,  42,
                                     7, 260, 15,  511, 0,   33,  210, 5};
  for (std::int64_t i = 0; i < 16; ++i) {
    EXPECT_EQ((ids->item<std::int32_t>({0, i})), expected[i]) << i;
  }

  const StatusOr<Tensor> logits = file->tensor("logits");
  ASSERT_TRUE(logits.ok()) << logits.status().ToString();
  EXPECT_EQ(logits->dtype(), DataType::kFloat32);
  EXPECT_EQ(logits->shape(), (Shape{1, 16, 512}));
}

TEST(SafetensorsTest, TensorBytesMatchIndependentFileRead) {
  // The zero-copy view must expose exactly the file's bytes: compare a
  // tensor read through the parser against the same byte range read
  // independently with ifstream.
  const std::filesystem::path path =
      engine::testing::FixturesDir() / "models/tiny-llama/model.safetensors";
  const auto file = SafetensorsFile::Open(path);
  ASSERT_TRUE(file.ok()) << file.status().ToString();
  const StatusOr<Tensor> norm = file->tensor("model.norm.weight");
  ASSERT_TRUE(norm.ok()) << norm.status().ToString();
  const auto byte_count = static_cast<std::size_t>(norm->numel()) *
                          sizeof(engine::tensor::bfloat16);

  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in.good());
  in.seekg(static_cast<std::streamoff>(norm->byte_offset()));
  std::vector<char> disk_bytes(byte_count);
  in.read(disk_bytes.data(), static_cast<std::streamsize>(byte_count));
  ASSERT_TRUE(in.good());
  EXPECT_EQ(std::memcmp(norm->data(), disk_bytes.data(), byte_count), 0);
}

TEST(SafetensorsTest, FileStaysMappedWhileTensorAlive) {
  // Acceptance criterion: drop the SafetensorsFile, then read every byte
  // through a surviving Tensor. A premature munmap is a crash (and a
  // sanitizer report) here.
  Tensor survivor;
  {
    const auto file = SafetensorsFile::Open(
        engine::testing::FixturesDir() / "models/tiny-llama/model.safetensors");
    ASSERT_TRUE(file.ok()) << file.status().ToString();
    StatusOr<Tensor> embed = file->tensor("model.embed_tokens.weight");
    ASSERT_TRUE(embed.ok()) << embed.status().ToString();
    survivor = *std::move(embed);
  }
  const auto* bytes = static_cast<const unsigned char*>(survivor.data());
  const auto byte_count = static_cast<std::size_t>(survivor.numel()) *
                          sizeof(engine::tensor::bfloat16);
  unsigned int checksum = 0;
  for (std::size_t i = 0; i < byte_count; ++i) {
    checksum += bytes[i];
  }
  EXPECT_GT(checksum, 0U);  // random bf16 weights are not all-zero
}

// --- fuzz-ish negatives
// -------------------------------------------------------

TEST(SafetensorsTest, MissingFileIsNotFound) {
  const auto file = SafetensorsFile::Open(ScratchDir() / "absent.safetensors");
  ASSERT_FALSE(file.ok());
  EXPECT_TRUE(IsNotFound(file.status())) << file.status().ToString();
}

TEST(SafetensorsTest, ErrorsNameTheFile) {
  const StatusOr<SafetensorsFile> file = OpenBytes("1234");
  ASSERT_FALSE(file.ok());
  EXPECT_TRUE(IsInvalidArgument(file.status())) << file.status().ToString();
  EXPECT_NE(file.status().message().find(".safetensors"), std::string::npos)
      << file.status().ToString();
}

TEST(SafetensorsTest, RejectsTruncatedLengthField) {
  const StatusOr<SafetensorsFile> file = OpenBytes("1234");
  ASSERT_FALSE(file.ok());
  EXPECT_NE(file.status().message().find("at least an 8-byte"),
            std::string::npos)
      << file.status().ToString();
}

TEST(SafetensorsTest, RejectsHeaderLengthPastEof) {
  // header_len = 1000 but only 4 bytes follow the length field. Also the
  // hostile extreme: header_len = UINT64_MAX must not overflow the check.
  std::string bytes(12, '\0');
  std::uint64_t header_len = 1000;
  std::memcpy(bytes.data(), &header_len, sizeof(header_len));
  StatusOr<SafetensorsFile> file = OpenBytes(bytes);
  ASSERT_FALSE(file.ok());
  EXPECT_TRUE(IsInvalidArgument(file.status())) << file.status().ToString();
  EXPECT_NE(file.status().message().find("header length 1000"),
            std::string::npos)
      << file.status().ToString();

  header_len = std::numeric_limits<std::uint64_t>::max();
  std::memcpy(bytes.data(), &header_len, sizeof(header_len));
  file = OpenBytes(bytes);
  ASSERT_FALSE(file.ok());
  EXPECT_TRUE(IsInvalidArgument(file.status())) << file.status().ToString();
}

TEST(SafetensorsTest, RejectsHeaderLengthOverSanityCap) {
  std::string bytes(12, '\0');
  const std::uint64_t header_len = 257ULL * 1024 * 1024;  // cap is 256 MiB
  std::memcpy(bytes.data(), &header_len, sizeof(header_len));
  const StatusOr<SafetensorsFile> file = OpenBytes(bytes);
  ASSERT_FALSE(file.ok());
  EXPECT_TRUE(IsInvalidArgument(file.status())) << file.status().ToString();
  EXPECT_NE(file.status().message().find("sanity cap"), std::string::npos)
      << file.status().ToString();
}

TEST(SafetensorsTest, RejectsMalformedHeaderJson) {
  ExpectInvalid("this is not json", "", "not valid JSON");
  ExpectInvalid("[1,2,3]", "", "must be a JSON object");
}

TEST(SafetensorsTest, RejectsMalformedEntries) {
  ExpectInvalid(R"({"a":5})", "", "entry must be a JSON object");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1]}})", "",
                R"(missing required key "data_offsets")");
  ExpectInvalid(
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}})",
      std::string(4, '\0'), R"(unexpected key "extra")");
  ExpectInvalid(R"({"a":{"dtype":3,"shape":[1],"data_offsets":[0,4]}})",
                std::string(4, '\0'), R"("dtype" must be a string)");
}

TEST(SafetensorsTest, RejectsBadDtypes) {
  ExpectInvalid(R"({"a":{"dtype":"Q4_K","shape":[1],"data_offsets":[0,4]}})",
                std::string(4, '\0'), "unrecognized safetensors dtype");
  // Real safetensors dtypes with no engine DataType are Unimplemented, not
  // InvalidArgument (design §3.4 check 3).
  const auto f64 =
      OpenSynthetic(R"({"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}})",
                    std::string(8, '\0'));
  ASSERT_FALSE(f64.ok());
  EXPECT_TRUE(IsUnimplemented(f64.status())) << f64.status().ToString();
  EXPECT_NE(f64.status().message().find("F64"), std::string::npos)
      << f64.status().ToString();
}

TEST(SafetensorsTest, RejectsBadShapes) {
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":"x","data_offsets":[0,4]}})",
                std::string(4, '\0'), R"("shape" must be an array)");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}})",
                std::string(4, '\0'), "dims must be integers");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
                std::string(4, '\0'), "negative dim");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],)"
                R"("data_offsets":[0,4]}})",
                std::string(4, '\0'), "exceeds kMaxRank");
}

TEST(SafetensorsTest, RejectsBadDataOffsets) {
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",
                std::string(4, '\0'), "2-element array");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[-1,3]}})",
                std::string(4, '\0'), "is negative");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})",
                std::string(8, '\0'), "begin 8 exceeds end 4");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,100]}})",
                std::string(4, '\0'), "exceeds the data section size");
}

TEST(SafetensorsTest, RejectsSizeMismatchAndOverflow) {
  // 2 × F32 needs 8 bytes; the range provides 4.
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",
                std::string(4, '\0'), "need 8");
  // numel fits int64 but numel × itemsize overflows uint64.
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[9223372036854775807],)"
                R"("data_offsets":[0,0]}})",
                "", "overflows");
}

TEST(SafetensorsTest, RejectsUnalignedBegin) {
  // "b" starts at byte 2 — not a multiple of F32's 4-byte itemsize
  // (stricter than the upstream spec, design §3.4).
  ExpectInvalid(R"({"a":{"dtype":"I8","shape":[2],"data_offsets":[0,2]},)"
                R"("b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}})",
                std::string(6, '\0'), "not aligned");
}

TEST(SafetensorsTest, RejectsMisalignedDataSectionStart) {
  // begin 0 is relatively aligned, but an unpadded header puts the data
  // section itself at a non-multiple of F32's itemsize — only the
  // *absolute* offset check catches this (the M4 audit found the relative
  // check let a misaligned typed view through).
  constexpr std::string_view kHeader =
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  ASSERT_NE((8 + kHeader.size()) % 4, 0U) << "precondition: unpadded start "
                                             "must be misaligned for F32";
  const StatusOr<SafetensorsFile> file =
      OpenBytes(BuildFileBytesUnpadded(kHeader, std::string(4, '\0')));
  ASSERT_FALSE(file.ok());
  EXPECT_TRUE(IsInvalidArgument(file.status())) << file.status().ToString();
  EXPECT_NE(file.status().message().find("not aligned"), std::string::npos)
      << file.status().ToString();
  EXPECT_NE(file.status().message().find("absolute offset"), std::string::npos)
      << file.status().ToString();
}

TEST(SafetensorsTest, RejectsOverlapGapAndUncoveredTail) {
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
                R"("b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
                std::string(8, '\0'), "overlapping");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
                R"("b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})",
                std::string(12, '\0'), "gap of 4 bytes");
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                std::string(8, '\0'), "account for only 4");
  ExpectInvalid("{}", std::string(4, '\0'), "account for only 0");
}

TEST(SafetensorsTest, RejectsZeroLengthRangeInsideAnotherTensor) {
  // A zero-length range strictly inside another tensor's bytes is layout
  // corruption, not a harmless empty tensor — the cursor walk rejects it.
  ExpectInvalid(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
                R"("z":{"dtype":"F32","shape":[0],"data_offsets":[4,4]}})",
                std::string(8, '\0'), "overlapping");
}

TEST(SafetensorsTest, RejectsMalformedMetadata) {
  ExpectInvalid(R"({"__metadata__":5})", "", "must be a JSON object");
  ExpectInvalid(R"({"__metadata__":{"k":5}})", "", "must be a string");
}

}  // namespace
