#include "tensor/tensor.h"

#include "core/status.h"
#include "memory/allocator.h"
#include "tensor/device.h"
#include "tensor/dtype.h"
#include "tensor/half.h"
#include "tensor/shape.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

namespace {

using engine::core::IsInvalidArgument;
using engine::core::IsOutOfMemory;
using engine::core::IsUnimplemented;
using engine::core::StatusOr;
using engine::memory::Allocator;
using engine::memory::Buffer;
using engine::tensor::DataType;
using engine::tensor::Device;
using engine::tensor::kAllDataTypes;
using engine::tensor::Shape;
using engine::tensor::Strides;
using engine::tensor::Tensor;

[[nodiscard]] bool IsAligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// Counts allocations and deleter invocations — the instrument for "storage
// lives until the last handle dies, then is freed exactly once".
class CountingAllocator final : public Allocator {
 public:
  [[nodiscard]] StatusOr<Buffer> Allocate(std::size_t bytes,
                                          std::size_t alignment) override {
    (void)alignment;
    ++alloc_count_;
    int* free_count = &free_count_;
    void* data = bytes == 0 ? nullptr : std::malloc(bytes);
    return Buffer(data, bytes, device(), [free_count](void* ptr) {
      ++*free_count;
      std::free(ptr);
    });
  }

  [[nodiscard]] Device device() const override { return Device::Cpu(); }

  [[nodiscard]] int alloc_count() const { return alloc_count_; }
  [[nodiscard]] int free_count() const { return free_count_; }

 private:
  int alloc_count_ = 0;
  int free_count_ = 0;
};

// Reports a CUDA device without being able to allocate — only to exercise
// the allocator/device mismatch CHECK in Tensor::empty.
class FakeCudaAllocator final : public Allocator {
 public:
  [[nodiscard]] StatusOr<Buffer> Allocate(std::size_t bytes,
                                          std::size_t alignment) override {
    (void)bytes;
    (void)alignment;
    return engine::core::InternalError("FakeCudaAllocator cannot allocate");
  }

  [[nodiscard]] Device device() const override { return Device::Cuda(0); }
};

// Unwraps a StatusOr<Tensor>, failing the test on error.
[[nodiscard]] Tensor Unwrap(StatusOr<Tensor> tensor) {
  EXPECT_TRUE(tensor.ok()) << tensor.status().ToString();
  return *std::move(tensor);
}

// Fills a float32 CPU tensor's storage with 0, 1, 2, ... in memory order.
void FillArange(const Tensor& tensor) {
  auto* data = tensor.data_ptr<float>();
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    data[i] = static_cast<float>(i);
  }
}

// --- empty: metadata goldens ---

TEST(TensorTest, EmptyMetadataGoldens) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2, 3, 4}, DataType::kFloat32, Device::Cpu()));
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.shape(), (Shape{2, 3, 4}));
  EXPECT_EQ(t.strides(), (Strides{12, 4, 1}));
  EXPECT_EQ(t.dtype(), DataType::kFloat32);
  EXPECT_EQ(t.device(), Device::Cpu());
  EXPECT_EQ(t.numel(), 24);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_EQ(t.byte_offset(), 0U);
  EXPECT_NE(t.data(), nullptr);
  EXPECT_TRUE(IsAligned(t.data(), 64));
}

TEST(TensorTest, EmptyScalar) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{}, DataType::kFloat32, Device::Cpu()));
  EXPECT_EQ(t.shape().rank(), 0);
  EXPECT_EQ(t.numel(), 1);
  EXPECT_TRUE(t.is_contiguous());
  ASSERT_NE(t.data(), nullptr);
  t.data_ptr<float>()[0] = 42.0F;
  EXPECT_EQ(t.item<float>({}), 42.0F);
}

TEST(TensorTest, EmptyZeroNumel) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2, 0, 3}, DataType::kFloat32, Device::Cpu()));
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data(), nullptr);
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, EmptyAllAllocatableDtypes) {
  for (const DataType dtype : kAllDataTypes) {
    StatusOr<Tensor> t = Tensor::empty(Shape{3}, dtype, Device::Cpu());
    if (dtype == DataType::kInt4 || dtype == DataType::kFP8E4M3) {
      ASSERT_FALSE(t.ok()) << to_string(dtype);
      EXPECT_TRUE(IsUnimplemented(t.status())) << t.status().ToString();
    } else {
      ASSERT_TRUE(t.ok()) << t.status().ToString();
      EXPECT_EQ(t->dtype(), dtype);
      EXPECT_NE(t->data(), nullptr);
    }
  }
}

TEST(TensorTest, TypedAccessWithHalfPrecisionDtypes) {
  // The half.h DTypeTraits mapping (M1-T07) makes the typed accessors work
  // for fp16/bf16 storage end to end.
  using engine::tensor::bfloat16;
  using engine::tensor::float16;
  const Tensor f16 =
      Unwrap(Tensor::empty(Shape{2, 2}, DataType::kFloat16, Device::Cpu()));
  f16.data_ptr<float16>()[3] = float16(1.5F);
  EXPECT_EQ(f16.item<float16>({1, 1}), 1.5F);
  const Tensor bf16 =
      Unwrap(Tensor::empty(Shape{2}, DataType::kBFloat16, Device::Cpu()));
  bf16.data_ptr<bfloat16>()[0] = bfloat16(-2.0F);
  EXPECT_EQ(bf16.item<bfloat16>({0}), -2.0F);
}

TEST(TensorTest, EmptyCudaIsUnimplemented) {
  const StatusOr<Tensor> t =
      Tensor::empty(Shape{2, 2}, DataType::kFloat32, Device::Cuda(0));
  ASSERT_FALSE(t.ok());
  EXPECT_TRUE(IsUnimplemented(t.status())) << t.status().ToString();
}

TEST(TensorTest, EmptyUsesExplicitAllocator) {
  CountingAllocator allocator;
  {
    const Tensor t = Unwrap(
        Tensor::empty(Shape{4}, DataType::kInt32, Device::Cpu(), &allocator));
    EXPECT_EQ(allocator.alloc_count(), 1);
    EXPECT_EQ(allocator.free_count(), 0);
  }
  EXPECT_EQ(allocator.free_count(), 1);
}

TEST(TensorTest, EmptyByteSizeOverflowIsInvalidArgument) {
  // numel fits int64_t but numel * itemsize(kInt64) does not.
  const StatusOr<Tensor> t =
      Tensor::empty(Shape{std::numeric_limits<std::int64_t>::max() / 2},
                    DataType::kInt64, Device::Cpu());
  ASSERT_FALSE(t.ok());
  EXPECT_TRUE(IsInvalidArgument(t.status())) << t.status().ToString();
}

TEST(TensorTest, EmptyAllocationFailurePropagatesOutOfMemory) {
  // Byte size fits int64_t; the allocator itself fails.
  const StatusOr<Tensor> t =
      Tensor::empty(Shape{std::numeric_limits<std::int64_t>::max() / 16},
                    DataType::kUInt8, Device::Cpu());
  ASSERT_FALSE(t.ok());
  EXPECT_TRUE(IsOutOfMemory(t.status())) << t.status().ToString();
}

// --- Shared-handle semantics (acceptance criterion: cheap shallow copy) ---

TEST(TensorTest, CopyIsShallowAndAliases) {
  CountingAllocator allocator;
  const Tensor t = Unwrap(
      Tensor::empty(Shape{6}, DataType::kFloat32, Device::Cpu(), &allocator));
  const Tensor copy = t;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(allocator.alloc_count(), 1);  // copying never allocates
  EXPECT_EQ(copy.data(), t.data());
  EXPECT_EQ(copy.shape(), t.shape());
  t.data_ptr<float>()[5] = 7.0F;           // write through one handle...
  EXPECT_EQ(copy.item<float>({5}), 7.0F);  // ...visible through the other
}

TEST(TensorTest, ViewKeepsStorageAliveAndDeleterRunsOnce) {
  CountingAllocator allocator;
  Tensor view;
  {
    const Tensor base = Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32,
                                             Device::Cpu(), &allocator));
    FillArange(base);
    view = Unwrap(base.slice(0, 1, 3));
  }  // base handle destroyed; the view still owns the storage
  EXPECT_EQ(allocator.free_count(), 0);
  EXPECT_EQ(view.item<float>({0, 0}), 3.0F);
  view = Tensor();  // last handle dies
  EXPECT_EQ(allocator.free_count(), 1);
}

TEST(TensorTest, MovedFromIsUndefined) {
  Tensor t = Unwrap(Tensor::empty(Shape{2}, DataType::kFloat32, Device::Cpu()));
  Tensor moved = std::move(t);
  EXPECT_TRUE(moved.defined());
  // Reading the moved-from Tensor is deliberate: it is specified to be the
  // empty handle.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(t.defined());
  Tensor assigned;
  assigned = std::move(moved);
  EXPECT_TRUE(assigned.defined());
}

TEST(TensorTest, DefaultHandleMetadata) {
  const Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_EQ(t.shape().rank(), 0);
  EXPECT_EQ(t.dtype(), DataType::kFloat32);
  EXPECT_EQ(t.device(), Device::Cpu());
  EXPECT_EQ(t.byte_offset(), 0U);
}

// --- slice ---

TEST(TensorTest, SliceOuterDimGoldens) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32, Device::Cpu()));
  FillArange(t);
  const Tensor s = Unwrap(t.slice(0, 1, 3));
  EXPECT_EQ(s.shape(), (Shape{2, 3}));
  EXPECT_EQ(s.strides(), (Strides{3, 1}));
  EXPECT_EQ(s.byte_offset(), 3U * sizeof(float));  // one row of 3 floats
  EXPECT_TRUE(s.is_contiguous());
  EXPECT_EQ(s.data(), t.data_ptr<float>() + 3);
  EXPECT_EQ(s.item<float>({0, 0}), 3.0F);    // t(1, 0)
  EXPECT_EQ(s.item<float>({1, 2}), 8.0F);    // t(2, 2)
  s.data_ptr<float>()[0] = 100.0F;           // write through the view...
  EXPECT_EQ(t.item<float>({1, 0}), 100.0F);  // ...visible in the base
}

TEST(TensorTest, SliceInnerDimIsNonContiguous) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32, Device::Cpu()));
  FillArange(t);
  const Tensor s = Unwrap(t.slice(1, 1, 3));
  EXPECT_EQ(s.shape(), (Shape{4, 2}));
  EXPECT_EQ(s.strides(), (Strides{3, 1}));  // unchanged
  EXPECT_EQ(s.byte_offset(), 1U * sizeof(float));
  EXPECT_FALSE(s.is_contiguous());
  EXPECT_EQ(s.item<float>({0, 0}), 1.0F);  // t(0, 1)
  EXPECT_EQ(s.item<float>({2, 1}), 8.0F);  // t(2, 2)
}

TEST(TensorTest, SliceOfSliceComposesOffsets) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 6}, DataType::kFloat32, Device::Cpu()));
  FillArange(t);
  const Tensor a = Unwrap(t.slice(0, 1, 4));
  const Tensor b = Unwrap(a.slice(1, 2, 5));
  EXPECT_EQ(b.shape(), (Shape{3, 3}));
  EXPECT_EQ(b.strides(), (Strides{6, 1}));
  EXPECT_EQ(b.byte_offset(), (1U * 6U + 2U) * sizeof(float));
  EXPECT_EQ(b.item<float>({0, 0}), 8.0F);   // t(1, 2)
  EXPECT_EQ(b.item<float>({2, 2}), 22.0F);  // t(3, 4)
}

TEST(TensorTest, SliceEmptyRange) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32, Device::Cpu()));
  const Tensor s = Unwrap(t.slice(0, 2, 2));
  EXPECT_TRUE(s.defined());
  EXPECT_EQ(s.shape(), (Shape{0, 3}));
  EXPECT_EQ(s.numel(), 0);
  // A zero-numel view of a non-empty tensor keeps a non-null pointer (base
  // plus offset); data() is null only for zero-sized buffers (tensor.h).
  EXPECT_EQ(s.byte_offset(), 2U * 3U * sizeof(float));
  EXPECT_EQ(s.data(), t.data_ptr<float>() + (2 * 3));
}

TEST(TensorTest, SliceFullRangeIsIdentityMetadata) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32, Device::Cpu()));
  const Tensor s = Unwrap(t.slice(0, 0, 4));
  EXPECT_EQ(s.shape(), t.shape());
  EXPECT_EQ(s.strides(), t.strides());
  EXPECT_EQ(s.byte_offset(), 0U);
  EXPECT_EQ(s.data(), t.data());
}

TEST(TensorTest, SliceInvalidArguments) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32, Device::Cpu()));
  for (const StatusOr<Tensor>& bad :
       {t.slice(-1, 0, 1), t.slice(2, 0, 1), t.slice(0, -1, 2),
        t.slice(0, 0, 5), t.slice(0, 3, 2), t.slice(1, 2, 4)}) {
    ASSERT_FALSE(bad.ok());
    EXPECT_TRUE(IsInvalidArgument(bad.status())) << bad.status().ToString();
  }
  const Tensor scalar =
      Unwrap(Tensor::empty(Shape{}, DataType::kFloat32, Device::Cpu()));
  EXPECT_TRUE(IsInvalidArgument(scalar.slice(0, 0, 0).status()));
}

// --- reshape ---

TEST(TensorTest, ReshapeContiguousSharesData) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2, 6}, DataType::kFloat32, Device::Cpu()));
  FillArange(t);
  const Tensor r = Unwrap(t.reshape(Shape{3, 4}));
  EXPECT_EQ(r.shape(), (Shape{3, 4}));
  EXPECT_EQ(r.strides(), (Strides{4, 1}));
  EXPECT_EQ(r.data(), t.data());
  EXPECT_EQ(r.item<float>({1, 1}), 5.0F);  // memory order preserved
  r.data_ptr<float>()[0] = 50.0F;
  EXPECT_EQ(t.item<float>({0, 0}), 50.0F);
}

TEST(TensorTest, ReshapeScalarAndBack) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{}, DataType::kFloat32, Device::Cpu()));
  const Tensor r = Unwrap(t.reshape(Shape{1, 1}));
  EXPECT_EQ(r.shape(), (Shape{1, 1}));
  const Tensor back = Unwrap(r.reshape(Shape{}));
  EXPECT_EQ(back.shape().rank(), 0);
}

TEST(TensorTest, ReshapeZeroNumel) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2, 0}, DataType::kFloat32, Device::Cpu()));
  const Tensor r = Unwrap(t.reshape(Shape{0, 4}));
  EXPECT_EQ(r.shape(), (Shape{0, 4}));
  EXPECT_EQ(r.numel(), 0);
}

TEST(TensorTest, ReshapeNumelMismatchIsInvalidArgument) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2, 6}, DataType::kFloat32, Device::Cpu()));
  const StatusOr<Tensor> r = t.reshape(Shape{5});
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(IsInvalidArgument(r.status())) << r.status().ToString();
}

TEST(TensorTest, ReshapeNonContiguousIsInvalidArgument) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4, 3}, DataType::kFloat32, Device::Cpu()));
  const Tensor s = Unwrap(t.slice(1, 0, 2));  // inner slice: non-contiguous
  ASSERT_FALSE(s.is_contiguous());
  const StatusOr<Tensor> r = s.reshape(Shape{8});
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(IsInvalidArgument(r.status())) << r.status().ToString();
}

// --- view_as_dtype ---

TEST(TensorTest, ViewAsDtypeBitReinterprets) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2}, DataType::kFloat32, Device::Cpu()));
  t.data_ptr<float>()[0] = 1.0F;
  const Tensor v = Unwrap(t.view_as_dtype(DataType::kInt32));
  EXPECT_EQ(v.dtype(), DataType::kInt32);
  EXPECT_EQ(v.shape(), t.shape());
  EXPECT_EQ(v.strides(), t.strides());
  EXPECT_EQ(v.data(), t.data());
  EXPECT_EQ(v.item<std::int32_t>({0}), 0x3F800000);  // bits of 1.0f
  v.data_ptr<std::int32_t>()[1] = 0x40000000;        // bits of 2.0f
  EXPECT_EQ(t.item<float>({1}), 2.0F);
}

TEST(TensorTest, ViewAsDtypeSameSizePairs) {
  const Tensor i8 =
      Unwrap(Tensor::empty(Shape{4}, DataType::kInt8, Device::Cpu()));
  EXPECT_EQ(Unwrap(i8.view_as_dtype(DataType::kUInt8)).dtype(),
            DataType::kUInt8);
  // fp16 ↔ bf16: same 16-bit width, different formats — the fixture-loading
  // reinterpretation case.
  const Tensor f16 =
      Unwrap(Tensor::empty(Shape{4}, DataType::kFloat16, Device::Cpu()));
  const Tensor bf16 = Unwrap(f16.view_as_dtype(DataType::kBFloat16));
  EXPECT_EQ(bf16.dtype(), DataType::kBFloat16);
  EXPECT_EQ(bf16.data(), f16.data());
}

TEST(TensorTest, ViewAsDtypePreservesOffsetOnViews) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4}, DataType::kFloat32, Device::Cpu()));
  const Tensor s = Unwrap(t.slice(0, 2, 4));
  const Tensor v = Unwrap(s.view_as_dtype(DataType::kInt32));
  EXPECT_EQ(v.byte_offset(), 2U * sizeof(float));
}

TEST(TensorTest, ViewAsDtypeSizeMismatchIsInvalidArgument) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{4}, DataType::kFloat32, Device::Cpu()));
  for (const DataType bad :
       {DataType::kFloat16, DataType::kInt8, DataType::kInt64}) {
    const StatusOr<Tensor> v = t.view_as_dtype(bad);
    ASSERT_FALSE(v.ok()) << to_string(bad);
    EXPECT_TRUE(IsInvalidArgument(v.status())) << v.status().ToString();
  }
}

TEST(TensorTest, ViewAsDtypeReservedIsUnimplemented) {
  const Tensor i8 =
      Unwrap(Tensor::empty(Shape{4}, DataType::kInt8, Device::Cpu()));
  EXPECT_TRUE(IsUnimplemented(i8.view_as_dtype(DataType::kFP8E4M3).status()));
  // Reserved wins over the width mismatch: kInt4 is Unimplemented, not
  // InvalidArgument.
  EXPECT_TRUE(IsUnimplemented(i8.view_as_dtype(DataType::kInt4).status()));
}

// --- Typed access: failure is loud (acceptance criterion) ---

TEST(TensorDeathTest, DataPtrDtypeMismatchAborts) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2}, DataType::kFloat32, Device::Cpu()));
  EXPECT_DEATH((void)t.data_ptr<std::int32_t>(), "data_ptr");
}

TEST(TensorDeathTest, ItemMisuseAborts) {
  const Tensor t =
      Unwrap(Tensor::empty(Shape{2, 3}, DataType::kFloat32, Device::Cpu()));
  EXPECT_DEATH((void)t.item<std::int32_t>({0, 0}), "data_ptr");  // wrong T
  EXPECT_DEATH((void)t.item<float>({0}), "rank");                // wrong rank
  EXPECT_DEATH((void)t.item<float>({0, 3}), "out of range");     // bad index
  EXPECT_DEATH((void)t.item<float>({-1, 0}), "out of range");
}

TEST(TensorDeathTest, UndefinedHandleAborts) {
  const Tensor t;
  EXPECT_DEATH((void)t.data(), "undefined Tensor");
  EXPECT_DEATH((void)t.slice(0, 0, 1), "undefined Tensor");
  EXPECT_DEATH((void)t.reshape(Shape{1}), "undefined Tensor");
  EXPECT_DEATH((void)t.view_as_dtype(DataType::kInt32), "undefined Tensor");
}

TEST(TensorDeathTest, AllocatorDeviceMismatchAborts) {
  FakeCudaAllocator allocator;
  EXPECT_DEATH((void)Tensor::empty(Shape{2}, DataType::kFloat32, Device::Cpu(),
                                   &allocator),
               "does not match");
}

}  // namespace
