#include "core/status.h"
#include "kernels/dispatch.h"
#include "tensor/dtype.h"

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <string_view>
#include <type_traits>

// DISPATCH_FLOATING_TYPES tests (M2-T08). A .cu TU on CUDA builds only —
// dispatch.h is an internal toolkit-including header (it binds device
// types) — but everything here is pure host code: the macro's switch and
// its Unimplemented default branch never launch anything, so these tests
// run on the GPU-less CUDA CI configuration too. The typed launch paths are
// exercised end-to-end by the GPU tests in elementwise_test.cpp.

namespace {

using engine::core::IsUnimplemented;
using engine::core::OkStatus;
using engine::core::Status;
using engine::tensor::DataType;

// The §9.3 device-type binding, per floating dtype.
template <typename Expected>
[[nodiscard]] Status ExpectBinds(DataType dtype) {
  return DISPATCH_FLOATING_TYPES(dtype, "test", scalar_t, [&] {
    static_assert(!std::is_void_v<scalar_t>);
    EXPECT_TRUE((std::is_same_v<scalar_t, Expected>));
    return OkStatus();
  });
}

TEST(DispatchFloatingTypesTest, BindsDeviceTypePerDtype) {
  EXPECT_TRUE(ExpectBinds<float>(DataType::kFloat32).ok());
  EXPECT_TRUE(ExpectBinds<__half>(DataType::kFloat16).ok());
  EXPECT_TRUE(ExpectBinds<__nv_bfloat16>(DataType::kBFloat16).ok());
}

TEST(DispatchFloatingTypesTest, NonFloatingDtypesAreUnimplemented) {
  for (DataType dtype :
       {DataType::kInt8, DataType::kUInt8, DataType::kInt32, DataType::kInt64,
        DataType::kBool, DataType::kFP8E4M3, DataType::kInt4}) {
    const Status status =
        DISPATCH_FLOATING_TYPES(dtype, "test-op", scalar_t, [&]() -> Status {
          ADD_FAILURE() << "body must not run for non-floating dtypes";
          return OkStatus();
        });
    EXPECT_TRUE(IsUnimplemented(status)) << status.ToString();
    // The message names the op (§9.2's per-call-site policy hook).
    EXPECT_NE(status.message().find("test-op"), std::string_view::npos);
  }
}

TEST(DispatchFloatingTypesTest, PropagatesBodyStatus) {
  const Status status = DISPATCH_FLOATING_TYPES(
      DataType::kFloat32, "test", scalar_t,
      [&] { return engine::core::InternalError("from the body"); });
  EXPECT_TRUE(engine::core::IsInternal(status)) << status.ToString();
}

}  // namespace
