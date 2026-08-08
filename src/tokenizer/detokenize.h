#pragma once

#include "core/status.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <string>

// Incremental detokenization (M4-T10; design: docs/design/model-loading.md
// §6.5): the per-request state streaming generation (M9) holds to turn ids
// into text as they are sampled.

namespace engine::tokenizer {

// Streams ids → UTF-8 text as tokens arrive. Token boundaries do not align
// with codepoint boundaries in byte-level BPE (a 4-byte emoji commonly
// spans three tokens), so emissions hold back the trailing incomplete
// UTF-8 sequence — at most 3 bytes of carried state — until later bytes
// complete it or prove it invalid. Every emission is valid UTF-8, and the
// concatenation of all push() outputs plus finish() is byte-identical to
// batch Tokenizer::decode of the same ids (both apply the same maximal-
// subpart U+FFFD policy, tokenizer.h).
//
// The tokenizer must outlive the stream (a reference is retained).
class DetokenizerStream {
 public:
  explicit DetokenizerStream(const Tokenizer& tokenizer,
                             bool skip_special_tokens);

  // The bytes decodable so far: everything carried plus this token's bytes,
  // up to the last complete UTF-8 sequence. Always valid UTF-8; possibly
  // empty (mid-codepoint, or a skipped special token). An out-of-range id
  // is `InvalidArgument` and leaves the stream state untouched.
  [[nodiscard]] core::StatusOr<std::string> push(std::int32_t id);

  // End of stream: flushes the residue. A trailing incomplete sequence
  // becomes U+FFFD (never raw malformed bytes) — exactly what batch decode
  // does with the same trailing bytes. Resets the carry: a second finish()
  // returns "" and the stream can be reused for a fresh id sequence.
  [[nodiscard]] std::string finish();

 private:
  const Tokenizer* tokenizer_;
  bool skip_special_tokens_;
  // Invariant between calls: a proper prefix of one well-formed UTF-8
  // sequence (≤ 3 bytes) — everything else has been emitted.
  std::string pending_;
};

}  // namespace engine::tokenizer
