#include "kvcache/block_table.h"

#include "core/check.h"
#include "core/status.h"

#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace engine::kvcache {

BlockTable::BlockTable(BlockPool* pool) : pool_(pool) {
  CHECK(pool_ != nullptr, "BlockTable: pool must not be null");
}

BlockTable::~BlockTable() { FreeAll(); }

BlockTable::BlockTable(BlockTable&& other) noexcept
    : pool_(other.pool_),
      blocks_(std::move(other.blocks_)),
      num_tokens_(other.num_tokens_) {
  // Leave the moved-from table empty so its destructor releases nothing.
  other.blocks_.clear();
  other.num_tokens_ = 0;
}

core::StatusOr<std::vector<std::int64_t>> BlockTable::AppendTokens(
    std::int64_t count) {
  if (count <= 0) {
    return core::InvalidArgumentError(
        "BlockTable::AppendTokens: count must be positive, got {}", count);
  }

  const std::int64_t bs = pool_->block_size();
  const std::int64_t need = pool_->blocks_needed(num_tokens_, count);

  // All-or-nothing: gather every new block first; on exhaustion release the
  // ones already taken and leave the table + pool exactly as before (§7.2).
  std::vector<std::int32_t> fresh;
  fresh.reserve(static_cast<std::size_t>(need));
  for (std::int64_t i = 0; i < need; ++i) {
    core::StatusOr<std::int32_t> block = pool_->Allocate();
    if (!block.ok()) {
      for (const std::int32_t taken : fresh) {
        pool_->Release(taken);
      }
      return core::ResourceExhaustedError(
          "BlockTable::AppendTokens: pool exhausted growing {} tokens by {} "
          "(needed {} new blocks, {} free)",
          num_tokens_, count, need, pool_->free_blocks());
    }
    fresh.push_back(*block);
  }

  // Full success: commit the new blocks, then build the slot mapping for the
  // new positions [num_tokens_, num_tokens_ + count).
  blocks_.insert(blocks_.end(), fresh.begin(), fresh.end());
  std::vector<std::int64_t> slot_mapping;
  slot_mapping.reserve(static_cast<std::size_t>(count));
  const std::int64_t begin = num_tokens_;
  num_tokens_ += count;
  for (std::int64_t pos = begin; pos < num_tokens_; ++pos) {
    const std::int64_t block = blocks_[static_cast<std::size_t>(pos / bs)];
    slot_mapping.push_back((block * bs) + (pos % bs));
  }
  return slot_mapping;
}

core::Status BlockTable::Truncate(std::int64_t new_len) {
  if (new_len < 0 || new_len > num_tokens_) {
    return core::InvalidArgumentError(
        "BlockTable::Truncate: new_len {} out of range [0, {}]", new_len,
        num_tokens_);
  }
  const std::int64_t bs = pool_->block_size();
  // Blocks needed for new_len tokens: ⌈new_len / bs⌉. Release everything past
  // it (tail-first, so the lowest logical block ends up on top of the pool's
  // LIFO free list and is reused first on the next append).
  const std::int64_t keep = (new_len + bs - 1) / bs;
  while (std::ssize(blocks_) > keep) {
    pool_->Release(blocks_.back());
    blocks_.pop_back();
  }
  num_tokens_ = new_len;
  return core::OkStatus();
}

void BlockTable::FreeAll() {
  for (const std::int32_t block : blocks_) {
    pool_->Release(block);
  }
  blocks_.clear();
  num_tokens_ = 0;
}

std::int64_t BlockTable::slot(std::int64_t pos) const {
  CHECK(pos >= 0 && pos < num_tokens_,
        "BlockTable::slot: pos {} out of range [0, {})", pos, num_tokens_);
  const std::int64_t bs = pool_->block_size();
  const std::int64_t block = blocks_[static_cast<std::size_t>(pos / bs)];
  return (block * bs) + (pos % bs);
}

}  // namespace engine::kvcache
