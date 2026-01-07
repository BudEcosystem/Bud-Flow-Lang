#pragma once

// =============================================================================
// Bud Flow Lang - Adaptive Kernel Cache
// =============================================================================
//
// Context-keyed storage for optimized kernel versions. This is the core
// component of the Adaptive JIT Compiler that:
//
// 1. Maps execution contexts to specialized kernel versions
// 2. Tracks Thompson Sampling state for each version
// 3. Manages tier promotion and version lifecycle
// 4. Supports profile persistence for warm start
//
// Based on:
// - JVM HotSpot inline cache architecture
// - V8 feedback vector design
// - YJIT Basic Block Versioning (ECOOP 2024)
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/continuous_tuner.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace bud {
namespace jit {

// =============================================================================
// Execution Tier (matches executor.cc but owned here for integration)
// =============================================================================

enum class ExecutionTier : uint8_t {
    kInterpreter = 0,  // Tier 0: Direct Highway dispatch
    kCopyPatch = 1,    // Tier 1: Copy-and-patch JIT
    kFusedKernel = 2,  // Tier 2: Fused Highway kernels
};

[[nodiscard]] const char* tierName(ExecutionTier tier);

// =============================================================================
// Context Key - Uniquely identifies an execution context
// =============================================================================

struct ContextKey {
    uint64_t ir_hash = 0;  // Hash of kernel IR
    ScalarType dtype = ScalarType::kFloat32;
    uint32_t size_bucket = 0;  // Size range bucket (0-4)
    uint32_t hardware_id = 0;  // Hardware fingerprint

    [[nodiscard]] bool operator==(const ContextKey& other) const {
        return ir_hash == other.ir_hash && dtype == other.dtype &&
               size_bucket == other.size_bucket && hardware_id == other.hardware_id;
    }

    [[nodiscard]] uint64_t hash() const {
        // FNV-1a hash combination
        uint64_t h = 0xcbf29ce484222325ULL;
        h ^= ir_hash;
        h *= 0x100000001b3ULL;
        h ^= static_cast<uint64_t>(dtype);
        h *= 0x100000001b3ULL;
        h ^= size_bucket;
        h *= 0x100000001b3ULL;
        h ^= hardware_id;
        h *= 0x100000001b3ULL;
        return h;
    }
};

struct ContextKeyHash {
    [[nodiscard]] size_t operator()(const ContextKey& key) const {
        return static_cast<size_t>(key.hash());
    }
};

// =============================================================================
// Kernel Version - A compiled kernel with Thompson state
// =============================================================================

struct KernelVersion {
    uint32_t id = 0;  // Version ID within context
    ExecutionTier tier = ExecutionTier::kInterpreter;
    void* code_ptr = nullptr;  // Compiled code pointer (JIT'd function)

    // Thompson Sampling state (for variant selection)
    scheduler::ThompsonState thompson_state;

    // Specialization guards
    size_t min_size = 0;               // Minimum valid input size
    size_t max_size = SIZE_MAX;        // Maximum valid input size
    bool is_size_specialized = false;  // Whether specialized for size range

    // Execution statistics
    uint64_t executions = 0;
    uint64_t total_time_ns = 0;
    float avg_time_ns = 0.0f;
    float min_time_ns = std::numeric_limits<float>::max();
    float max_time_ns = 0.0f;

    // Timestamps
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_executed_at;

    // Check if this version can handle the given size
    [[nodiscard]] bool canHandle(size_t input_size) const {
        return input_size >= min_size && input_size <= max_size;
    }
};

// =============================================================================
// Context Entry - All versions for a specific context
// =============================================================================

struct ContextEntry {
    ContextKey key;
    std::vector<KernelVersion> versions;
    size_t best_version = 0;  // Index of current best version
    uint64_t total_executions = 0;
    float cumulative_time_ns = 0.0f;

    // Size histogram for specialization detection
    static constexpr size_t kMaxSizeBuckets = 16;
    std::array<uint32_t, kMaxSizeBuckets> size_histogram{};
    std::array<size_t, kMaxSizeBuckets> bucket_sizes{};

    // Timestamps
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_accessed_at;

    // Record a size for specialization analysis
    void recordSize(size_t input_size);

    // Get the most common size (for specialization)
    [[nodiscard]] std::optional<size_t> dominantSize(float threshold = 0.5f) const;

    // Check if specialization would be beneficial
    [[nodiscard]] bool shouldSpecialize(size_t min_executions = 100) const;
};

// =============================================================================
// Tier Promotion Decision
// =============================================================================

struct TierDecision {
    ExecutionTier tier = ExecutionTier::kInterpreter;
    size_t version_idx = 0;
    bool should_promote = false;
    bool should_specialize = false;
    std::string reason;

    // For debugging/logging
    float thompson_variance = 0.0f;
    float predicted_speedup = 0.0f;
    uint64_t call_count = 0;
};

// =============================================================================
// Adaptive Kernel Cache Configuration
// =============================================================================

struct AdaptiveCacheConfig {
    // Version limits
    size_t max_versions_per_context = 8;
    size_t max_context_entries = 10000;

    // Thompson Sampling parameters
    float prior_alpha = 1.0f;
    float prior_beta = 1.0f;
    float exploration_bonus = 0.1f;
    size_t warmup_executions = 10;
    size_t min_samples_per_version = 5;

    // Tier promotion thresholds (Thompson-based)
    float variance_threshold = 0.1f;  // Max Thompson variance for confidence
    float min_speedup_ratio = 1.2f;   // Min predicted speedup to promote
    size_t min_calls_tier1 = 10;      // Min calls before Tier 0->1
    size_t min_calls_tier2 = 100;     // Min calls before Tier 1->2

    // Specialization thresholds
    float dominant_size_ratio = 0.8f;  // Size must be >80% of calls
    size_t min_executions_specialize = 100;

    // EMA decay for timing updates
    float time_decay = 0.99f;
};

// =============================================================================
// Adaptive Kernel Cache
// =============================================================================

class AdaptiveKernelCache {
  public:
    using VersionExecutor = std::function<float(const KernelVersion& version)>;

    explicit AdaptiveKernelCache(const AdaptiveCacheConfig& config = {});
    ~AdaptiveKernelCache();

    AdaptiveKernelCache(AdaptiveKernelCache&&) noexcept;
    AdaptiveKernelCache& operator=(AdaptiveKernelCache&&) noexcept;
    AdaptiveKernelCache(const AdaptiveKernelCache&) = delete;
    AdaptiveKernelCache& operator=(const AdaptiveKernelCache&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    [[nodiscard]] const AdaptiveCacheConfig& config() const { return config_; }
    void setConfig(const AdaptiveCacheConfig& config);

    // =========================================================================
    // Context Registration
    // =========================================================================

    /// Get or create a context entry for the given key
    ContextEntry& getOrCreateContext(const ContextKey& key);

    /// Check if context exists
    [[nodiscard]] bool hasContext(const ContextKey& key) const;

    /// Get context (returns nullptr if not found)
    [[nodiscard]] const ContextEntry* getContext(const ContextKey& key) const;

    // =========================================================================
    // Version Management
    // =========================================================================

    /// Add a new version to a context
    size_t addVersion(const ContextKey& key, KernelVersion version);

    /// Get the best version for execution (Thompson Sampling)
    [[nodiscard]] TierDecision selectVersion(const ContextKey& key, size_t input_size = 0);

    /// Get version by index
    [[nodiscard]] const KernelVersion* getVersion(const ContextKey& key, size_t version_idx) const;

    /// Record execution result and update Thompson state
    void recordExecution(const ContextKey& key, size_t version_idx, size_t input_size,
                         float time_ns);

    // =========================================================================
    // Tier Promotion
    // =========================================================================

    /// Check if context should be promoted to next tier
    [[nodiscard]] bool shouldPromote(const ContextKey& key) const;

    /// Get promotion decision with reasoning
    [[nodiscard]] TierDecision getPromotionDecision(const ContextKey& key) const;

    /// Manually promote a context to a higher tier
    bool promoteContext(const ContextKey& key, ExecutionTier new_tier, void* code_ptr = nullptr);

    // =========================================================================
    // Specialization
    // =========================================================================

    /// Check if context should be size-specialized
    [[nodiscard]] bool shouldSpecialize(const ContextKey& key) const;

    /// Get dominant size for specialization
    [[nodiscard]] std::optional<size_t> getDominantSize(const ContextKey& key) const;

    // =========================================================================
    // Statistics
    // =========================================================================

    struct CacheStats {
        size_t total_contexts = 0;
        size_t total_versions = 0;
        uint64_t total_selections = 0;
        uint64_t total_executions = 0;
        size_t tier0_contexts = 0;
        size_t tier1_contexts = 0;
        size_t tier2_contexts = 0;
        size_t specialized_versions = 0;
    };

    [[nodiscard]] CacheStats statistics() const;

    /// Reset all cache entries
    void clear();

    // =========================================================================
    // Persistence
    // =========================================================================

    /// Serialize cache state (Thompson states, not code)
    [[nodiscard]] std::string serialize() const;

    /// Deserialize cache state for warm start
    bool deserialize(const std::string& data);

    /// Save to file
    bool save(const std::string& path) const;

    /// Load from file
    bool load(const std::string& path);

    // =========================================================================
    // Hardware Detection
    // =========================================================================

    /// Get current hardware fingerprint
    [[nodiscard]] static uint32_t getHardwareId();

    /// Create context key from IR and execution parameters
    [[nodiscard]] static ContextKey makeKey(uint64_t ir_hash, ScalarType dtype, size_t input_size);

  private:
    AdaptiveCacheConfig config_;

    // Cache storage
    mutable std::shared_mutex mutex_;
    std::unordered_map<ContextKey, ContextEntry, ContextKeyHash> entries_;

    // Random number generator for Thompson Sampling
    mutable std::mt19937 rng_;

    // Statistics
    mutable std::atomic<uint64_t> total_selections_{0};
    mutable std::atomic<uint64_t> total_executions_{0};

    // Internal helpers
    void evictContextIfNeeded();
    void evictVersionIfNeeded(ContextEntry& entry);
    float normalizeReward(float time_ns, const ContextEntry& entry) const;
    size_t getSizeBucket(size_t input_size) const;
};

// =============================================================================
// Global Cache Instance
// =============================================================================

/// Get the global adaptive kernel cache
AdaptiveKernelCache& getGlobalCache();

/// Configure the global cache
void configureGlobalCache(const AdaptiveCacheConfig& config);

}  // namespace jit
}  // namespace bud
