#pragma once

#include "tensor/dtype.h"

#include <cstdint>

// Embedding lookup: ids -> fp32 rows, gathered from either a row-major
// checkpoint table or the packed lm_head layout (M6-T06; design:
// docs/design/optimized-cpu-execution.md §7, §10). The optimized analog of
// `cpu::embedding_lookup` (the oracle): a pure gather + exact fp16/bf16->fp32
// widen, with no multiply-accumulate — so every result here is
// **bit-identical** to the oracle, across thread counts AND across ISAs (the
// §10 table's "embedding lookup" row). The widen goes through the M3-T06
// conversion variants (kernels/convert), which are bit-exact per element vs
// tensor/half.h including NaN payloads.
//
// Two source layouts, one per tied/untied model (§7), sharing the widen path:
//
//   - EmbeddingLookupF32       : row-major table `[V, E]` (untied — the
//                                embedding table is its own zero-copy weight).
//   - EmbeddingLookupPackedF32 : the packed lm_head `[PackedPanels(V), E, kNr]`
//                                (tied — one physical copy, no `[V, E]`
//                                duplicate). Row `v` lives in panel `v/kNr` at
//                                lane `v%kNr`; its E elements are strided by
//                                `kNr` (the layout defined in kernels/gemm.h).
//
// Numerics: a per-token independent gather, so **bit-identical across thread
// counts** (each output row is written by exactly one worker); Class E widen,
// so **bit-identical across ISAs** too. No tolerance — this is one of the two
// pure-map kernels (§10).
//
// Raw-pointer entries: preconditions (non-null table/ids/y, contiguous fp32 `y`
// [t, e], every id in [0, V), a supported `dtype`) are the CALLER's to enforce
// — `model::OptimizedEmbedding` front-loads all recoverable validation,
// including the id-range pre-scan, so nothing inside a parallel region does
// anything but move+widen (ADR-003, §5).

namespace engine::kernels {

// y[i, :] = widen(table[ids[i], :]) for i in [0, t), row-major table `[V, E]`
// in storage `dtype` (kFloat32/kFloat16/kBFloat16). `y` is `[t, E]` fp32.
void EmbeddingLookupF32(const void* table, tensor::DataType dtype,
                        std::int64_t e, const std::int32_t* ids, std::int64_t t,
                        float* y);

// y[i, :] = widen(row ids[i] of the packed weight `wp`), where `wp` is the
// packed lm_head `[PackedPanels(V), E, kNr]` in storage `dtype`. Element k of
// logical row v sits at `wp[(v/kNr)*E*kNr + k*kNr + (v%kNr)]`. `y` is `[t, E]`
// fp32. Produces the identical result to EmbeddingLookupF32 over the same
// logical table (the "share storage across layouts" property, §7).
void EmbeddingLookupPackedF32(const void* wp, tensor::DataType dtype,
                              std::int64_t e, const std::int32_t* ids,
                              std::int64_t t, float* y);

}  // namespace engine::kernels
