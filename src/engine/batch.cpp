#include "engine/batch.h"

#include "core/status.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace engine::engine {

namespace {

// Bytes reserved by a vector (capacity, not size). For a pointer-element
// vector (e.g. the caches list) the intended element size is the pointer size
// itself, so the sizeof-pointer diagnostic is a false positive here.
template <typename T>
[[nodiscard]] std::size_t VectorBytes(const std::vector<T>& v) {
  return v.capacity() * sizeof(T);  // NOLINT(bugprone-sizeof-expression)
}

}  // namespace

model::ForwardRequest BatchInputs::MakeForwardRequest(
    model::LogitsMode mode, model::ActivationHook* hook) const {
  return model::ForwardRequest{
      .token_ids = token_ids,
      .positions = positions,
      .cache = nullptr,  // batched path: sequences carried by `caches`
      .logits_mode = mode,
      .hook = hook,
      .cu_seqlens = cu_seqlens,
      .caches = caches,
  };
}

core::Status BatchAssembler::Flatten(std::span<const BatchSeqInput> seqs,
                                     bool decode) {
  inputs_.token_ids.clear();
  inputs_.positions.clear();
  inputs_.cu_seqlens.clear();
  inputs_.caches.clear();
  inputs_.sample_rows.clear();

  inputs_.cu_seqlens.push_back(0);
  std::int32_t running = 0;
  for (std::size_t b = 0; b < seqs.size(); ++b) {
    const BatchSeqInput& s = seqs[b];
    if (s.cache == nullptr) {
      return core::InvalidArgumentError(
          "BatchAssembler: sequence {} has a null cache", b);
    }
    if (s.sampler == nullptr) {
      return core::InvalidArgumentError(
          "BatchAssembler: sequence {} has a null sampler", b);
    }
    const auto t_b = static_cast<std::int64_t>(s.token_ids.size());
    if (t_b < 1) {
      return core::InvalidArgumentError(
          "BatchAssembler: sequence {} has empty token_ids", b);
    }
    if (decode && t_b != 1) {
      return core::InvalidArgumentError(
          "BatchAssembler: decode sequence {} must contribute exactly one "
          "token (got {})",
          b, t_b);
    }

    // Positions are absolute: sequence b's tokens continue from its committed
    // cache length (a fresh prefill cache ⇒ [0, T); a decode cache of length L
    // ⇒ position L). §8.2.
    const std::int64_t base = s.cache->length();
    for (std::int64_t t = 0; t < t_b; ++t) {
      inputs_.token_ids.push_back(s.token_ids[static_cast<std::size_t>(t)]);
      inputs_.positions.push_back(static_cast<std::int32_t>(base + t));
    }
    running += static_cast<std::int32_t>(t_b);
    inputs_.cu_seqlens.push_back(running);
    inputs_.caches.push_back(s.cache);
    inputs_.sample_rows.push_back(
        sampling::BatchRow{.sampler = s.sampler, .context = s.context});
  }
  return core::OkStatus();
}

core::Status BatchAssembler::AssemblePrefill(
    std::span<const BatchSeqInput> seqs) {
  return Flatten(seqs, /*decode=*/false);
}

core::Status BatchAssembler::AssembleDecode(
    std::span<const BatchSeqInput> seqs) {
  // Decode differs only in the T_b == 1 constraint (Flatten enforces it). The
  // batched decode kernel self-sources block tables + lengths post-append
  // (§8.4 as-built), so no paged view is read at assembly time.
  return Flatten(seqs, /*decode=*/true);
}

std::size_t BatchAssembler::staging_bytes() const {
  return VectorBytes(inputs_.token_ids) + VectorBytes(inputs_.positions) +
         VectorBytes(inputs_.cu_seqlens) + VectorBytes(inputs_.caches) +
         VectorBytes(inputs_.sample_rows);
}

}  // namespace engine::engine
