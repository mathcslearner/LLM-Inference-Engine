#pragma once

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Safetensors single-file parser (M4-T04; design: docs/design/model-loading.md
// §3.4).
//
// Format: [0..8) u64-LE header length N; [8..8+N) UTF-8 JSON header
// `{ "__metadata__"?: {str: str}, "<name>": { "dtype", "shape",
// "data_offsets" }, … }`; [8+N..EOF) data section, per-tensor byte ranges
// relative to its start.
//
// This is a general-purpose reader, format-complete over the dtypes the
// engine represents (F32/F16/BF16/I8/U8/I32/I64/BOOL/F8_E4M3) — fixtures use
// it for fp32 activation goldens and integer token-id tensors, not just
// weights. Weight-acceptance policy (which dtypes may be *model weights*)
// belongs to `load_model` (M4-T07), not here. All input is untrusted:
// every malformed shape is a recoverable error naming the file (and tensor),
// never a CHECK and never an exception (ADR-003). Validation includes
// non-overlapping, gap-free coverage of the data section and
// itemsize-aligned *absolute* offsets — 8 + header_len + begin, since an
// unpadded header can shift relatively-aligned tensors off alignment
// (design §3.4 checks 1–5).

namespace engine::model {

class SafetensorsFile {
 public:
  // Maps `path` (read-only mmap; §3.3) and fully validates the header.
  // Errors: `NotFound` for a missing file; `InvalidArgument` for framing,
  // JSON, or layout defects; `Unimplemented` for real safetensors dtypes
  // the engine has no DataType for (F64, I16, U16, U32, U64, F8_E5M2) —
  // unrecognized dtype strings are `InvalidArgument`.
  [[nodiscard]] static core::StatusOr<SafetensorsFile> Open(
      const std::filesystem::path& path);

  // Tensor names, sorted lexicographically. Views into this object — they
  // are invalidated when it is destroyed or moved.
  [[nodiscard]] std::vector<std::string_view> names() const;
  [[nodiscard]] bool contains(std::string_view name) const;

  // Zero-copy view into the mapping, sharing the file's Buffer: the file
  // stays mapped while any returned Tensor is alive, independent of this
  // object's lifetime. `NotFound` for unknown names. An F8_E4M3 entry
  // parses but materializes as `Unimplemented` until its milestone
  // (Tensor::from_buffer's reserved-dtype rule; the loader rejects such
  // weights anyway — design §5).
  [[nodiscard]] core::StatusOr<tensor::Tensor> tensor(
      std::string_view name) const;

  // The "__metadata__" block if present (informational; never interpreted).
  [[nodiscard]] const std::map<std::string, std::string>& metadata() const {
    return metadata_;
  }

 private:
  struct Entry {
    tensor::DataType dtype = tensor::DataType::kFloat32;
    tensor::Shape shape;
    std::uint64_t begin = 0;  // relative to the data section
  };

  SafetensorsFile() = default;

  std::shared_ptr<memory::Buffer> buffer_;
  std::size_t data_start_ = 0;  // 8 + header_len
  // std::less<> enables string_view lookup; map order gives sorted names().
  std::map<std::string, Entry, std::less<>> entries_;
  std::map<std::string, std::string> metadata_;
  std::string path_;  // for error messages
};

}  // namespace engine::model
