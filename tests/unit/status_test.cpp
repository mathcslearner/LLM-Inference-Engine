#include "core/status.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

namespace {

using engine::core::AlreadyExistsError;
using engine::core::CancelledError;
using engine::core::FailedPreconditionError;
using engine::core::InternalError;
using engine::core::InvalidArgumentError;
using engine::core::IsAlreadyExists;
using engine::core::IsCancelled;
using engine::core::IsFailedPrecondition;
using engine::core::IsInternal;
using engine::core::IsInvalidArgument;
using engine::core::IsNotFound;
using engine::core::IsOutOfMemory;
using engine::core::IsOutOfRange;
using engine::core::IsResourceExhausted;
using engine::core::IsUnavailable;
using engine::core::IsUnimplemented;
using engine::core::NotFoundError;
using engine::core::OkStatus;
using engine::core::OutOfMemoryError;
using engine::core::OutOfRangeError;
using engine::core::ResourceExhaustedError;
using engine::core::Status;
using engine::core::StatusCode;
using engine::core::StatusCodeToString;
using engine::core::StatusOr;
using engine::core::UnavailableError;
using engine::core::UnimplementedError;

// ---------------------------------------------------------------- Status --

TEST(StatusTest, DefaultConstructedIsOk) {
  const Status status;
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kOk);
  EXPECT_EQ(status.message(), "");
  EXPECT_EQ(status.ToString(), "OK");
}

TEST(StatusTest, OkStatusEqualsDefault) { EXPECT_EQ(OkStatus(), Status()); }

TEST(StatusTest, ConstructorStoresCodeAndMessage) {
  const Status status(StatusCode::kNotFound, "no such block");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kNotFound);
  EXPECT_EQ(status.message(), "no such block");
}

TEST(StatusTest, OkCodeDiscardsMessage) {
  const Status status(StatusCode::kOk, "ignored");
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.message(), "");
  EXPECT_EQ(status, OkStatus());
}

TEST(StatusTest, ToStringComposesCodeAndMessage) {
  const Status status(StatusCode::kInvalidArgument, "bad size");
  EXPECT_EQ(status.ToString(), "INVALID_ARGUMENT: bad size");
}

TEST(StatusTest, StreamOperatorMatchesToString) {
  std::ostringstream os;
  os << NotFoundError("weight {}", "wq");
  EXPECT_EQ(os.str(), "NOT_FOUND: weight wq");
}

TEST(StatusTest, EqualityComparesCodeAndMessage) {
  EXPECT_EQ(Status(StatusCode::kInternal, "x"),
            Status(StatusCode::kInternal, "x"));
  EXPECT_NE(Status(StatusCode::kInternal, "x"),
            Status(StatusCode::kInternal, "y"));
  EXPECT_NE(Status(StatusCode::kInternal, "x"),
            Status(StatusCode::kUnavailable, "x"));
  EXPECT_NE(Status(StatusCode::kInternal, "x"), OkStatus());
}

TEST(StatusTest, CopyAndMovePreserveContents) {
  Status original(StatusCode::kResourceExhausted, "no free blocks");
  const Status copy =
      original;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(copy, original);
  const Status moved = std::move(original);
  EXPECT_EQ(moved, copy);
}

TEST(StatusTest, StatusCodeToStringCoversAllCodes) {
  EXPECT_EQ(StatusCodeToString(StatusCode::kOk), "OK");
  EXPECT_EQ(StatusCodeToString(StatusCode::kCancelled), "CANCELLED");
  EXPECT_EQ(StatusCodeToString(StatusCode::kInvalidArgument),
            "INVALID_ARGUMENT");
  EXPECT_EQ(StatusCodeToString(StatusCode::kNotFound), "NOT_FOUND");
  EXPECT_EQ(StatusCodeToString(StatusCode::kAlreadyExists), "ALREADY_EXISTS");
  EXPECT_EQ(StatusCodeToString(StatusCode::kFailedPrecondition),
            "FAILED_PRECONDITION");
  EXPECT_EQ(StatusCodeToString(StatusCode::kOutOfRange), "OUT_OF_RANGE");
  EXPECT_EQ(StatusCodeToString(StatusCode::kResourceExhausted),
            "RESOURCE_EXHAUSTED");
  EXPECT_EQ(StatusCodeToString(StatusCode::kOutOfMemory), "OUT_OF_MEMORY");
  EXPECT_EQ(StatusCodeToString(StatusCode::kUnimplemented), "UNIMPLEMENTED");
  EXPECT_EQ(StatusCodeToString(StatusCode::kUnavailable), "UNAVAILABLE");
  EXPECT_EQ(StatusCodeToString(StatusCode::kInternal), "INTERNAL");
}

TEST(StatusTest, FactoriesSetMatchingCodeAndPredicateHolds) {
  EXPECT_TRUE(IsCancelled(CancelledError("m")));
  EXPECT_TRUE(IsInvalidArgument(InvalidArgumentError("m")));
  EXPECT_TRUE(IsNotFound(NotFoundError("m")));
  EXPECT_TRUE(IsAlreadyExists(AlreadyExistsError("m")));
  EXPECT_TRUE(IsFailedPrecondition(FailedPreconditionError("m")));
  EXPECT_TRUE(IsOutOfRange(OutOfRangeError("m")));
  EXPECT_TRUE(IsResourceExhausted(ResourceExhaustedError("m")));
  EXPECT_TRUE(IsOutOfMemory(OutOfMemoryError("m")));
  EXPECT_TRUE(IsUnimplemented(UnimplementedError("m")));
  EXPECT_TRUE(IsUnavailable(UnavailableError("m")));
  EXPECT_TRUE(IsInternal(InternalError("m")));
}

TEST(StatusTest, PredicatesRejectOtherCodesAndOk) {
  EXPECT_FALSE(IsNotFound(InvalidArgumentError("m")));
  EXPECT_FALSE(IsNotFound(OkStatus()));
}

TEST(StatusTest, FactoriesComposeMessagesWithFmt) {
  const Status status =
      OutOfMemoryError("failed to allocate {} MiB on {}", 512, "cuda:0");
  EXPECT_EQ(status.message(), "failed to allocate 512 MiB on cuda:0");
  EXPECT_EQ(status.ToString(),
            "OUT_OF_MEMORY: failed to allocate 512 MiB on cuda:0");
}

// -------------------------------------------------------------- StatusOr --

TEST(StatusOrTest, HoldsValue) {
  const StatusOr<int> result = 42;
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.status(), OkStatus());
  EXPECT_EQ(*result, 42);
  EXPECT_EQ(result.value(), 42);
}

TEST(StatusOrTest, HoldsError) {
  const StatusOr<int> result = NotFoundError("no entry {}", 7);
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(IsNotFound(result.status()));
  EXPECT_EQ(result.status().message(), "no entry 7");
}

TEST(StatusOrTest, ImplicitConversionFromConvertibleValue) {
  // Exercises the converting constructor: const char* -> std::string.
  const StatusOr<std::string> result = "weights";
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, "weights");
}

TEST(StatusOrTest, ArrowOperatorReachesMembers) {
  StatusOr<std::string> result = std::string("abc");
  EXPECT_EQ(result->size(), 3U);
  result->push_back('d');
  EXPECT_EQ(*result, "abcd");
}

TEST(StatusOrTest, MutableAccessThroughDereference) {
  StatusOr<int> result = 1;
  *result = 5;
  EXPECT_EQ(result.value(), 5);
}

TEST(StatusOrTest, SupportsMoveOnlyTypes) {
  StatusOr<std::unique_ptr<int>> result = std::make_unique<int>(9);
  ASSERT_TRUE(result.ok());
  const std::unique_ptr<int> owned = std::move(result).value();
  ASSERT_NE(owned, nullptr);
  EXPECT_EQ(*owned, 9);
}

TEST(StatusOrTest, MoveConstructionTransfersValue) {
  StatusOr<std::string> source = std::string("payload");
  const StatusOr<std::string> target = std::move(source);
  ASSERT_TRUE(target.ok());
  EXPECT_EQ(*target, "payload");
}

TEST(StatusOrTest, MoveConstructionTransfersError) {
  StatusOr<int> source = InternalError("broken invariant");
  const StatusOr<int> target = std::move(source);
  ASSERT_FALSE(target.ok());
  EXPECT_EQ(target.status().message(), "broken invariant");
}

TEST(StatusOrTest, CopyConstructionSharesNothing) {
  StatusOr<std::string> source = std::string("original");
  StatusOr<std::string> copy = source;
  *copy += "-modified";
  EXPECT_EQ(*source, "original");
  EXPECT_EQ(*copy, "original-modified");
}

TEST(StatusOrTest, RvalueStatusMovesOut) {
  StatusOr<int> result = UnavailableError("retry later");
  const Status status = std::move(result).status();
  EXPECT_TRUE(IsUnavailable(status));
  EXPECT_EQ(status.message(), "retry later");
}

TEST(StatusOrDeathTest, ValueOnErrorCheckFails) {
  const StatusOr<int> result = InvalidArgumentError("bad");
  EXPECT_DEATH(static_cast<void>(result.value()),
               "StatusOr::value\\(\\) on error: INVALID_ARGUMENT: bad");
}

TEST(StatusOrDeathTest, ConstructionFromOkStatusCheckFails) {
  EXPECT_DEATH(static_cast<void>(StatusOr<int>(OkStatus())),
               "StatusOr must not be constructed from an OK status");
}

// ------------------------------------------------------ propagation macros --

Status PassThrough(const Status& status, int& evaluations) {
  RETURN_IF_ERROR((++evaluations, Status(status)));
  return OkStatus();
}

TEST(ReturnIfErrorTest, PropagatesError) {
  int evaluations = 0;
  const Status result = PassThrough(NotFoundError("missing"), evaluations);
  EXPECT_TRUE(IsNotFound(result));
  EXPECT_EQ(result.message(), "missing");
  EXPECT_EQ(evaluations, 1);
}

TEST(ReturnIfErrorTest, PassesThroughOkAndEvaluatesOnce) {
  int evaluations = 0;
  EXPECT_EQ(PassThrough(OkStatus(), evaluations), OkStatus());
  EXPECT_EQ(evaluations, 1);
}

Status ChainTwo(const Status& first, const Status& second) {
  RETURN_IF_ERROR(Status(first));
  RETURN_IF_ERROR(Status(second));
  return OkStatus();
}

TEST(ReturnIfErrorTest, StopsAtFirstError) {
  EXPECT_TRUE(IsCancelled(ChainTwo(CancelledError("1st"), OkStatus())));
  EXPECT_TRUE(IsInternal(ChainTwo(OkStatus(), InternalError("2nd"))));
  EXPECT_EQ(ChainTwo(OkStatus(), OkStatus()), OkStatus());
}

Status AssignFrom(StatusOr<int> input, int& out, int& evaluations) {
  ASSIGN_OR_RETURN(out, (++evaluations, std::move(input)));
  return OkStatus();
}

TEST(AssignOrReturnTest, AssignsValueOnOk) {
  int out = 0;
  int evaluations = 0;
  EXPECT_EQ(AssignFrom(41, out, evaluations), OkStatus());
  EXPECT_EQ(out, 41);
  EXPECT_EQ(evaluations, 1);
}

TEST(AssignOrReturnTest, ReturnsErrorWithoutAssigning) {
  int out = -1;
  int evaluations = 0;
  const Status result =
      AssignFrom(OutOfRangeError("index 9"), out, evaluations);
  EXPECT_TRUE(IsOutOfRange(result));
  EXPECT_EQ(result.message(), "index 9");
  EXPECT_EQ(out, -1);
  EXPECT_EQ(evaluations, 1);
}

Status DeclareAndUse(StatusOr<std::string> input, std::size_t& length) {
  ASSIGN_OR_RETURN(const std::string text, std::move(input));
  length = text.size();
  return OkStatus();
}

TEST(AssignOrReturnTest, LhsMayDeclareAVariable) {
  std::size_t length = 0;
  EXPECT_EQ(DeclareAndUse(std::string("four"), length), OkStatus());
  EXPECT_EQ(length, 4U);
}

Status MoveThrough(StatusOr<std::unique_ptr<int>> input,
                   std::unique_ptr<int>& out) {
  ASSIGN_OR_RETURN(out, std::move(input));
  return OkStatus();
}

TEST(AssignOrReturnTest, SupportsMoveOnlyTypes) {
  std::unique_ptr<int> out;
  EXPECT_EQ(MoveThrough(std::make_unique<int>(3), out), OkStatus());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(*out, 3);
}

Status TwoAssignsInOneScope(StatusOr<int> a, StatusOr<int> b, int& sum) {
  ASSIGN_OR_RETURN(const int first, std::move(a));
  ASSIGN_OR_RETURN(const int second, std::move(b));
  sum = first + second;
  return OkStatus();
}

TEST(AssignOrReturnTest, ComposesWithinOneScope) {
  int sum = 0;
  EXPECT_EQ(TwoAssignsInOneScope(20, 22, sum), OkStatus());
  EXPECT_EQ(sum, 42);
  EXPECT_TRUE(IsNotFound(TwoAssignsInOneScope(NotFoundError("a"), 1, sum)));
  EXPECT_TRUE(IsInternal(TwoAssignsInOneScope(1, InternalError("b"), sum)));
}

// ------------------------------------------------------ CHECK_OK / DCHECK_OK
// --

TEST(CheckOkTest, PassesOnOkStatusAndValue) {
  CHECK_OK(OkStatus());
  const StatusOr<int> result = 1;
  CHECK_OK(result);
  DCHECK_OK(OkStatus());
  DCHECK_OK(result);
}

TEST(CheckOkDeathTest, FailsOnErrorStatus) {
  EXPECT_DEATH(CHECK_OK(NotFoundError("gone")),
               "CHECK_OK\\(NotFoundError\\(\"gone\"\\)\\) failed at "
               ".*status_test\\.cpp:[0-9]+ .*: NOT_FOUND: gone");
}

TEST(CheckOkDeathTest, FailsOnErrorStatusOr) {
  const StatusOr<int> result = InternalError("bad state");
  EXPECT_DEATH(CHECK_OK(result), "INTERNAL: bad state");
}

#ifdef NDEBUG
TEST(CheckOkTest, DcheckOkCompilesToNothingInRelease) {
  int evaluations = 0;
  DCHECK_OK((++evaluations, InternalError("never evaluated")));
  EXPECT_EQ(evaluations, 0);
}
#else
TEST(CheckOkDeathTest, DcheckOkFailsOnErrorInDebug) {
  EXPECT_DEATH(DCHECK_OK(CancelledError("stop")), "CANCELLED: stop");
}
#endif

}  // namespace
