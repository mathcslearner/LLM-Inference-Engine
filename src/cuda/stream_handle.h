#pragma once

// Opaque stream handle (M2-T04; design: docs/design/cuda-backend.md §6.3).
//
// APIs outside the cuda module that accept a stream (tensor's transfer
// overload, kernels launchers) take a StreamHandle so their public headers
// stay toolkit-free (design §2.2). cudaStream_t is a pointer to the opaque
// struct CUstream_st; forward-declaring that struct names the same type
// without including the toolkit, so inside CUDA-compiled TUs a handle
// converts to cudaStream_t with no cast.

struct CUstream_st;  // the pointee of cudaStream_t

namespace engine::cuda {

// Trivially-copyable, non-owning, non-validating wrapper around
// cudaStream_t. A default-constructed (null) handle means "the engine
// default stream for the relevant device" (DefaultStream, cuda/stream.h) at
// every API that accepts one.
//
// This is the one deliberately thin spot in the type system (design §6.3):
// passing a handle to a destroyed stream is UB, same as any dangling
// pointer. The mitigation is ownership discipline — streams are long-lived
// objects owned by long-lived components (design §6.1).
class StreamHandle {
 public:
  constexpr StreamHandle() = default;
  constexpr explicit StreamHandle(CUstream_st* stream) : stream_(stream) {}

  // The raw cudaStream_t. Null means "engine default stream": resolve via
  // DefaultStream before handing it to a toolkit call.
  [[nodiscard]] constexpr CUstream_st* get() const { return stream_; }
  [[nodiscard]] constexpr bool is_default() const { return stream_ == nullptr; }

 private:
  CUstream_st* stream_ = nullptr;
};

}  // namespace engine::cuda
