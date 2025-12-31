#pragma once

// =============================================================================
// Bud Flow Lang - Persistent Kernel Cache
// =============================================================================
//
// Disk-based cache for compiled kernels to eliminate cold-start compilation.
// Uses hash-based keys for efficient lookup and LRU eviction for size limits.
//
// Design:
// - CacheKey: Unique identifier based on IR hash, dtype, size class, target features
// - PersistentKernelCache: Disk-backed storage with memory-mapped file index
// - Thread-safe for concurrent access
//

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/type_system.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace bud {

// Forward declarations
namespace ir {
class IRModule;
class IRBuilder;
}  // namespace ir

namespace jit {

// =============================================================================
// Size Class for Specialized Kernels
// =============================================================================

enum class SizeClass : uint8_t {
    kSmall = 0,   // < 1K elements
    kMedium = 1,  // 1K - 64K elements
    kLarge = 2,   // > 64K elements
    kUnknown = 255
};

// =============================================================================
// CacheKey - Unique identifier for cached kernels
// =============================================================================

class CacheKey {
  public:
    CacheKey();

    // Create key from IR module
    static CacheKey fromIR(const ir::IRModule& module);
    static CacheKey fromIR(const ir::IRBuilder& builder);

    // Accessors
    [[nodiscard]] uint64_t irHash() const { return ir_hash_; }
    [[nodiscard]] ScalarType dtype() const { return dtype_; }
    [[nodiscard]] SizeClass sizeClass() const { return size_class_; }
    [[nodiscard]] uint32_t targetFeatures() const { return target_features_; }
    [[nodiscard]] uint32_t compilerVersion() const { return compiler_version_; }

    // Setters (for testing)
    void setIrHash(uint64_t hash) { ir_hash_ = hash; }
    void setDtype(ScalarType dtype) { dtype_ = dtype; }
    void setSizeClass(SizeClass size_class) { size_class_ = size_class; }
    void setTargetFeatures(uint32_t features) { target_features_ = features; }
    void setCompilerVersion(uint32_t version) { compiler_version_ = version; }

    // String representation for filename/debugging
    [[nodiscard]] std::string toString() const;

    // Hash for unordered_map
    [[nodiscard]] size_t hash() const;

    // Equality comparison
    [[nodiscard]] bool operator==(const CacheKey& other) const;

  private:
    uint64_t ir_hash_ = 0;  // Hash of IR bytecode
    ScalarType dtype_ = ScalarType::kUnknown;
    SizeClass size_class_ = SizeClass::kSmall;
    uint32_t target_features_ = 0;   // CPU feature flags (AVX2, AVX-512, etc.)
    uint32_t compiler_version_ = 0;  // For cache invalidation on compiler updates

    // Detect current CPU features
    static uint32_t detectTargetFeatures();
};

}  // namespace jit
}  // namespace bud

// Hash function for std::unordered_map
template <>
struct std::hash<bud::jit::CacheKey> {
    size_t operator()(const bud::jit::CacheKey& key) const { return key.hash(); }
};

namespace bud {
namespace jit {

// =============================================================================
// Cache Statistics
// =============================================================================

struct CacheStats {
    size_t total_entries = 0;
    size_t total_bytes = 0;
    size_t hits = 0;
    size_t misses = 0;
    double hit_rate = 0.0;
};

// =============================================================================
// PersistentKernelCache - Disk-backed kernel cache
// =============================================================================

class PersistentKernelCache {
  public:
    // Constructor
    // cache_dir: Directory to store cached kernels
    // max_size_bytes: Maximum cache size (default 100MB)
    explicit PersistentKernelCache(const std::filesystem::path& cache_dir,
                                   size_t max_size_bytes = 100 * 1024 * 1024);

    ~PersistentKernelCache();

    // Disable copy/move
    PersistentKernelCache(const PersistentKernelCache&) = delete;
    PersistentKernelCache& operator=(const PersistentKernelCache&) = delete;

    // Load a cached kernel
    // Returns empty optional if not found or invalid
    [[nodiscard]] std::optional<std::vector<uint8_t>> load(const CacheKey& key);

    // Save a kernel to cache
    // Returns true on success
    bool save(const CacheKey& key, const std::vector<uint8_t>& kernel_data);

    // Clear the entire cache
    void clear();

    // Get cache statistics
    [[nodiscard]] CacheStats stats() const;

    // Check if cache contains a key
    [[nodiscard]] bool contains(const CacheKey& key) const;

    // Get cache directory
    [[nodiscard]] const std::filesystem::path& cacheDir() const { return cache_dir_; }

  private:
    // Get filename for a cache key
    [[nodiscard]] std::filesystem::path keyToPath(const CacheKey& key) const;

    // Evict entries to stay under size limit
    void evictIfNeeded(size_t new_entry_size);

    // Load index from disk
    void loadIndex();

    // Save index to disk
    void saveIndex();

    // Update LRU timestamp for an entry
    void touchEntry(const CacheKey& key);

    std::filesystem::path cache_dir_;
    size_t max_size_bytes_;

    // In-memory index for fast lookup
    struct IndexEntry {
        std::filesystem::path path;
        size_t size_bytes = 0;
        uint64_t last_access_time = 0;
    };

    mutable std::shared_mutex mutex_;
    std::unordered_map<CacheKey, IndexEntry> index_;
    size_t current_size_bytes_ = 0;

    // Stats tracking
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
    bool initialized_ = false;
};

// =============================================================================
// Global Cache Instance
// =============================================================================

// Get the global persistent cache instance
// Default location: ~/.cache/bud_flow_lang/ or platform equivalent
PersistentKernelCache& getPersistentCache();

// Initialize with custom cache directory
void initPersistentCache(const std::filesystem::path& cache_dir,
                         size_t max_size_bytes = 100 * 1024 * 1024);

}  // namespace jit
}  // namespace bud
