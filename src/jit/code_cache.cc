// =============================================================================
// Bud Flow Lang - JIT Code Cache
// =============================================================================
//
// Caches compiled kernels to avoid recompilation.
// Uses hash of IR structure as cache key.
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/error.h"

#include <spdlog/spdlog.h>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace bud {
namespace jit {

// Forward declaration
class CompiledKernel;

// =============================================================================
// IR Hasher
// =============================================================================

class IRHasher {
public:
    using HashType = uint64_t;

    static HashType hash(const ir::IRBuilder& builder) {
        HashType h = 0;

        for (const auto* node : builder.nodes()) {
            // Combine op code
            h = combine(h, static_cast<uint64_t>(node->opCode()));

            // Combine type info
            h = combine(h, static_cast<uint64_t>(node->type().scalarType()));
            h = combine(h, node->type().elementCount());

            // Combine operand count
            h = combine(h, node->numOperands());
        }

        return h;
    }

private:
    static HashType combine(HashType h, uint64_t value) {
        // FNV-1a style combining
        h ^= value;
        h *= 0x100000001b3;
        return h;
    }
};

// =============================================================================
// Code Cache
// =============================================================================

class CodeCache {
public:
    struct CacheEntry {
        std::shared_ptr<CompiledKernel> kernel;
        size_t code_size;
        uint64_t compile_time_us;
        uint64_t hit_count;
    };

    struct Stats {
        size_t total_entries = 0;
        size_t total_hits = 0;
        size_t total_misses = 0;
        size_t total_bytes = 0;
        double hit_rate = 0.0;
    };

    CodeCache(size_t max_size_bytes = 64 * 1024 * 1024)
        : max_size_bytes_(max_size_bytes) {}

    // Look up a cached kernel
    std::shared_ptr<CompiledKernel> find(IRHasher::HashType key) {
        std::shared_lock lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            ++it->second.hit_count;
            ++stats_.total_hits;
            return it->second.kernel;
        }

        ++stats_.total_misses;
        return nullptr;
    }

    // Insert a compiled kernel
    void insert(IRHasher::HashType key,
                std::shared_ptr<CompiledKernel> kernel,
                size_t code_size,
                uint64_t compile_time_us) {
        std::unique_lock lock(mutex_);

        // Check if we need to evict
        while (current_size_bytes_ + code_size > max_size_bytes_ && !cache_.empty()) {
            evictLRU();
        }

        CacheEntry entry{
            std::move(kernel),
            code_size,
            compile_time_us,
            0
        };

        cache_[key] = std::move(entry);
        current_size_bytes_ += code_size;
        ++stats_.total_entries;
        stats_.total_bytes = current_size_bytes_;

        spdlog::debug("CodeCache: Inserted kernel (hash={:#x}, size={})",
                      key, code_size);
    }

    // Get cache statistics
    Stats stats() const {
        std::shared_lock lock(mutex_);
        Stats s = stats_;
        if (s.total_hits + s.total_misses > 0) {
            s.hit_rate = static_cast<double>(s.total_hits) /
                         (s.total_hits + s.total_misses);
        }
        return s;
    }

    // Clear the cache
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        current_size_bytes_ = 0;
        stats_ = {};
    }

    // Get current size
    size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

private:
    void evictLRU() {
        // Simple eviction: remove entry with lowest hit count
        if (cache_.empty()) return;

        auto min_it = cache_.begin();
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->second.hit_count < min_it->second.hit_count) {
                min_it = it;
            }
        }

        current_size_bytes_ -= min_it->second.code_size;
        spdlog::debug("CodeCache: Evicted kernel (hash={:#x})", min_it->first);
        cache_.erase(min_it);
    }

    mutable std::shared_mutex mutex_;
    std::unordered_map<IRHasher::HashType, CacheEntry> cache_;
    size_t max_size_bytes_;
    size_t current_size_bytes_ = 0;
    Stats stats_;
};

// =============================================================================
// Global Cache Instance
// =============================================================================

namespace {
CodeCache* g_code_cache = nullptr;
}

CodeCache& getCodeCache() {
    if (!g_code_cache) {
        g_code_cache = new CodeCache();
    }
    return *g_code_cache;
}

void resetCodeCache() {
    if (g_code_cache) {
        g_code_cache->clear();
    }
}

void shutdownCodeCache() {
    delete g_code_cache;
    g_code_cache = nullptr;
}

}  // namespace jit
}  // namespace bud
