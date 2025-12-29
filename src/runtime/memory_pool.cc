// =============================================================================
// Bud Flow Lang - Memory Pool
// =============================================================================
//
// High-performance memory pool for SIMD-aligned allocations.
// Used for runtime data buffers.
//

#include "bud_flow_lang/common.h"

#include <hwy/aligned_allocator.h>
#include <spdlog/spdlog.h>

#include <mutex>
#include <vector>

namespace bud {

// =============================================================================
// Memory Pool
// =============================================================================

// Configuration struct (defined outside class for default arg visibility)
struct MemoryPoolConfig {
    size_t initial_size = 64 * 1024 * 1024;   // 64 MB
    size_t max_size = 1024 * 1024 * 1024;     // 1 GB
    size_t block_size = 4 * 1024 * 1024;      // 4 MB blocks
};

class MemoryPool {
public:
    using Config = MemoryPoolConfig;

    explicit MemoryPool(Config config = Config{})
        : config_(std::move(config)) {
        // Pre-allocate initial blocks
        size_t num_blocks = config_.initial_size / config_.block_size;
        for (size_t i = 0; i < num_blocks; ++i) {
            allocateBlock();
        }
        spdlog::debug("MemoryPool: Initialized with {} blocks ({} MB)",
                      blocks_.size(),
                      blocks_.size() * config_.block_size / (1024 * 1024));
    }

    ~MemoryPool() {
        for (auto& block : blocks_) {
            hwy::FreeAlignedBytes(block.ptr, nullptr, nullptr);
        }
    }

    // Allocate aligned memory
    void* allocate(size_t size) {
        std::lock_guard lock(mutex_);

        // Round up to alignment
        size = alignUp(size, kSimdAlignment);

        // Try to find space in existing blocks
        for (auto& block : blocks_) {
            if (block.used + size <= block.size) {
                void* ptr = static_cast<char*>(block.ptr) + block.used;
                block.used += size;
                total_allocated_ += size;
                return ptr;
            }
        }

        // Need new block
        size_t block_size = std::max(config_.block_size, size);
        if (totalCapacity() + block_size > config_.max_size) {
            spdlog::error("MemoryPool: Max size exceeded");
            return nullptr;
        }

        allocateBlock(block_size);
        auto& block = blocks_.back();
        void* ptr = block.ptr;
        block.used = size;
        total_allocated_ += size;
        return ptr;
    }

    // Reset pool (keep allocated blocks, just reset usage)
    void reset() {
        std::lock_guard lock(mutex_);
        for (auto& block : blocks_) {
            block.used = 0;
        }
        total_allocated_ = 0;
    }

    // Statistics
    size_t totalAllocated() const {
        std::lock_guard lock(mutex_);
        return total_allocated_;
    }

    size_t totalCapacity() const {
        std::lock_guard lock(mutex_);
        size_t total = 0;
        for (const auto& block : blocks_) {
            total += block.size;
        }
        return total;
    }

private:
    struct Block {
        void* ptr = nullptr;
        size_t size = 0;
        size_t used = 0;
    };

    void allocateBlock(size_t size = 0) {
        if (size == 0) {
            size = config_.block_size;
        }

        // Highway handles SIMD alignment internally (HWY_ALIGNMENT)
        void* ptr = hwy::AllocateAlignedBytes(size, nullptr, nullptr);
        if (ptr) {
            blocks_.push_back({ptr, size, 0});
        } else {
            spdlog::error("MemoryPool: Failed to allocate {} bytes", size);
        }
    }

    Config config_;
    std::vector<Block> blocks_;
    size_t total_allocated_ = 0;
    mutable std::mutex mutex_;
};

// =============================================================================
// Global Memory Pool
// =============================================================================

namespace {
MemoryPool* g_memory_pool = nullptr;
}

MemoryPool& getMemoryPool() {
    if (!g_memory_pool) {
        g_memory_pool = new MemoryPool();
    }
    return *g_memory_pool;
}

void* poolAllocate(size_t size) {
    return getMemoryPool().allocate(size);
}

void poolReset() {
    if (g_memory_pool) {
        g_memory_pool->reset();
    }
}

void shutdownMemoryPool() {
    delete g_memory_pool;
    g_memory_pool = nullptr;
}

}  // namespace bud
