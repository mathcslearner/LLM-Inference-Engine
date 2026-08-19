#include "model/workspace.h"

#include "core/check.h"
#include "core/status.h"
#include "tensor/dtype.h"
#include "tensor/shape.h"
#include "tensor/tensor.h"

#include <cstdint>

namespace engine::model {

namespace {

// Allocates one [t, width] uninitialized fp32 slot. `empty` (not `zeros`) is
// safe because every slot is fully written by a kernel before it is read within
// a forward (§6.1): the residual stream is seeded by the embedding gather and
// each projection/norm writes its whole output before the next stage reads it.
[[nodiscard]] core::StatusOr<tensor::Tensor> AllocSlot(std::int64_t t,
                                                       std::int64_t width) {
  return tensor::Tensor::empty(tensor::Shape{t, width},
                               tensor::DataType::kFloat32,
                               tensor::Device::Cpu());
}

}  // namespace

Workspace Workspace::Create(std::int64_t e, int num_heads, int num_kv_heads,
                            int d, std::int64_t i) {
  CHECK(e > 0 && num_heads > 0 && num_kv_heads > 0 && d > 0 && i > 0,
        "Workspace::Create: dims must be positive (E={}, H={}, Hkv={}, d={}, "
        "I={})",
        e, num_heads, num_kv_heads, d, i);
  const std::int64_t q_rows = static_cast<std::int64_t>(num_heads) * d;
  const std::int64_t kv_rows = static_cast<std::int64_t>(num_kv_heads) * d;
  return {e, q_rows, kv_rows, i};
}

core::Status Workspace::EnsureCapacity(std::int64_t t) {
  CHECK(t >= 1, "Workspace::EnsureCapacity: t must be >= 1, got {}", t);
  if (t <= capacity_) {
    return core::OkStatus();
  }
  // Allocate every slot at the new capacity into locals first; only commit
  // once all succeed, so a mid-way OOM leaves the prior (smaller) buffers
  // intact and the workspace usable (front-loaded, ADR-003).
  ASSIGN_OR_RETURN(tensor::Tensor x, AllocSlot(t, e_));
  ASSIGN_OR_RETURN(tensor::Tensor h, AllocSlot(t, e_));
  ASSIGN_OR_RETURN(tensor::Tensor tmp, AllocSlot(t, e_));
  ASSIGN_OR_RETURN(tensor::Tensor r, AllocSlot(t, e_));
  ASSIGN_OR_RETURN(tensor::Tensor q, AllocSlot(t, q_rows_));
  ASSIGN_OR_RETURN(tensor::Tensor k, AllocSlot(t, kv_rows_));
  ASSIGN_OR_RETURN(tensor::Tensor v, AllocSlot(t, kv_rows_));
  ASSIGN_OR_RETURN(tensor::Tensor ctx, AllocSlot(t, q_rows_));
  ASSIGN_OR_RETURN(tensor::Tensor gate, AllocSlot(t, i_));
  ASSIGN_OR_RETURN(tensor::Tensor up, AllocSlot(t, i_));

  x_ = std::move(x);
  h_ = std::move(h);
  tmp_ = std::move(tmp);
  r_ = std::move(r);
  q_ = std::move(q);
  k_ = std::move(k);
  v_ = std::move(v);
  ctx_ = std::move(ctx);
  gate_ = std::move(gate);
  up_ = std::move(up);
  capacity_ = t;
  return core::OkStatus();
}

tensor::Tensor Workspace::Prefix(const tensor::Tensor& slot,
                                 std::int64_t t) const {
  CHECK(t >= 1 && t <= capacity_,
        "Workspace: prefix t={} out of range [1, {}] — EnsureCapacity first", t,
        capacity_);
  core::StatusOr<tensor::Tensor> view = slot.slice(0, 0, t);
  CHECK(view.ok(), "Workspace: slice failed: {}", view.status().ToString());
  return *std::move(view);
}

tensor::Tensor Workspace::x(std::int64_t t) const { return Prefix(x_, t); }
tensor::Tensor Workspace::h(std::int64_t t) const { return Prefix(h_, t); }
tensor::Tensor Workspace::tmp(std::int64_t t) const { return Prefix(tmp_, t); }
tensor::Tensor Workspace::r(std::int64_t t) const { return Prefix(r_, t); }
tensor::Tensor Workspace::q(std::int64_t t) const { return Prefix(q_, t); }
tensor::Tensor Workspace::k(std::int64_t t) const { return Prefix(k_, t); }
tensor::Tensor Workspace::v(std::int64_t t) const { return Prefix(v_, t); }
tensor::Tensor Workspace::ctx(std::int64_t t) const { return Prefix(ctx_, t); }
tensor::Tensor Workspace::gate(std::int64_t t) const {
  return Prefix(gate_, t);
}
tensor::Tensor Workspace::up(std::int64_t t) const { return Prefix(up_, t); }

std::int64_t Workspace::bytes() const {
  constexpr std::int64_t kF32 = 4;
  // c_stream = 4 E-width buffers (x/h/tmp/r) + q + 2·kv (k,v) + ctx + 2·I.
  const std::int64_t elems_per_row =
      (4 * e_) + q_rows_ + (2 * kv_rows_) + q_rows_ + (2 * i_);
  return kF32 * capacity_ * elems_per_row;
}

}  // namespace engine::model
