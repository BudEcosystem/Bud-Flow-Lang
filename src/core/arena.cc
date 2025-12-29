// =============================================================================
// Bud Flow Lang - Arena Allocator Implementation
// =============================================================================

#include "bud_flow_lang/arena.h"

#include <hwy/aligned_allocator.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>

namespace bud {

// =============================================================================
// ArenaBlock Implementation
// =============================================================================

ArenaBlock::ArenaBlock(size_t size) {
    // Use Highway's aligned allocator for SIMD-friendly alignment
    // Highway handles SIMD alignment internally (HWY_ALIGNMENT)
    start_ = static_cast<char*>(hwy::AllocateAlignedBytes(size, nullptr, nullptr));

    if (start_) {
        current_ = start_;
        end_ = start_ + size;
    } else {
        spdlog::error("ArenaBlock: Failed to allocate {} bytes", size);
    }
}

ArenaBlock::~ArenaBlock() {
    if (start_) {
        hwy::FreeAlignedBytes(start_, nullptr, nullptr);
    }
}

ArenaBlock::ArenaBlock(ArenaBlock&& other) noexcept
    : start_(other.start_), current_(other.current_), end_(other.end_) {
    other.start_ = nullptr;
    other.current_ = nullptr;
    other.end_ = nullptr;
}

ArenaBlock& ArenaBlock::operator=(ArenaBlock&& other) noexcept {
    if (this != &other) {
        if (start_) {
            hwy::FreeAlignedBytes(start_, nullptr, nullptr);
        }
        start_ = other.start_;
        current_ = other.current_;
        end_ = other.end_;
        other.start_ = nullptr;
        other.current_ = nullptr;
        other.end_ = nullptr;
    }
    return *this;
}

void* ArenaBlock::allocate(size_t size, size_t alignment) {
    // Align current pointer
    uintptr_t current = reinterpret_cast<uintptr_t>(current_);
    uintptr_t aligned = alignUp(current, alignment);
    char* result = reinterpret_cast<char*>(aligned);

    // Check if we have enough space
    if (result + size > end_) {
        return nullptr;
    }

    current_ = result + size;
    return result;
}

// =============================================================================
// Arena Implementation
// =============================================================================

Arena::Arena(ArenaConfig config) : config_(std::move(config)) {
    // Pre-allocate first block
    addBlock(config_.initial_block_size);
}

Arena::Arena(Arena&& other) noexcept
    : config_(std::move(other.config_)),
      blocks_(std::move(other.blocks_)),
      current_block_(other.current_block_),
      total_allocated_(other.total_allocated_) {
    other.current_block_ = 0;
    other.total_allocated_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        blocks_ = std::move(other.blocks_);
        current_block_ = other.current_block_;
        total_allocated_ = other.total_allocated_;
        other.current_block_ = 0;
        other.total_allocated_ = 0;
    }
    return *this;
}

void* Arena::allocate(size_t size) {
    return allocate(size, config_.alignment);
}

void* Arena::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    // Ensure alignment is at least the configured minimum
    alignment = std::max(alignment, config_.alignment);

    // Try to allocate from current block
    if (current_block_ < blocks_.size()) {
        void* ptr = blocks_[current_block_].allocate(size, alignment);
        if (ptr) {
            total_allocated_ += size;
            return ptr;
        }
    }

    // Try subsequent blocks (in case of fragmentation)
    for (size_t i = current_block_ + 1; i < blocks_.size(); ++i) {
        void* ptr = blocks_[i].allocate(size, alignment);
        if (ptr) {
            current_block_ = i;
            total_allocated_ += size;
            return ptr;
        }
    }

    // Need to add a new block
    size_t block_size = config_.initial_block_size;
    if (config_.grow_exponentially && !blocks_.empty()) {
        block_size = std::min(blocks_.back().capacity() * 2, config_.max_block_size);
    }

    // Ensure block is large enough for this allocation
    block_size = std::max(block_size, size + alignment);
    addBlock(block_size);

    // Allocate from new block
    void* ptr = blocks_.back().allocate(size, alignment);
    if (ptr) {
        current_block_ = blocks_.size() - 1;
        total_allocated_ += size;
    }
    return ptr;
}

void Arena::reset() {
    for (auto& block : blocks_) {
        block.reset();
    }
    current_block_ = 0;
    total_allocated_ = 0;
}

size_t Arena::totalAllocated() const {
    return total_allocated_;
}

size_t Arena::totalCapacity() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.capacity();
    }
    return total;
}

void Arena::addBlock(size_t min_size) {
    blocks_.emplace_back(min_size);
}

// =============================================================================
// Thread-Local Arena
// =============================================================================

Arena& threadLocalArena() {
    thread_local Arena arena;
    return arena;
}

// =============================================================================
// ScopedArenaReset
// =============================================================================

ScopedArenaReset::ScopedArenaReset(Arena& arena)
    : arena_(arena), saved_allocated_(arena.totalAllocated()) {}

ScopedArenaReset::~ScopedArenaReset() {
    // Note: This is a simplified reset. A full implementation would
    // need to track the exact allocation point to reset to.
    // For now, we just log if allocations occurred.
    if (arena_.totalAllocated() != saved_allocated_) {
        spdlog::debug("ScopedArenaReset: Arena grew from {} to {} bytes", saved_allocated_,
                      arena_.totalAllocated());
    }
}

}  // namespace bud
