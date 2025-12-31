#pragma once

// =============================================================================
// Bud Flow Lang - Profile-Guided Optimization Specializer
// =============================================================================
//
// Generates specialized kernels based on observed runtime patterns:
// - Common input sizes (generate size-specific kernels)
// - Hot paths (optimize frequently executed code paths)
// - Type patterns (specialize for common data types)
//
// Benefits:
// - Optimized code for most common cases
// - Reduced branch overhead
// - Better instruction cache utilization
//

#include "bud_flow_lang/ir.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bud {
namespace jit {

// =============================================================================
// Size Statistics
// =============================================================================

/// Statistics for a commonly observed size
struct SizeStats {
    size_t size = 0;             // The size value
    uint64_t count = 0;          // Number of times observed
    uint64_t total_time_ns = 0;  // Total execution time for this size
};

// =============================================================================
// ProfileData
// =============================================================================

/// Collected profile data for a kernel/operation
class ProfileData {
  public:
    ProfileData() = default;

    /// Record an execution with given size and time
    void recordExecution(size_t element_count, std::chrono::nanoseconds exec_time);

    /// Get total number of calls recorded
    [[nodiscard]] uint64_t totalCalls() const { return total_calls_; }

    /// Get total elements processed
    [[nodiscard]] uint64_t totalElements() const { return total_elements_; }

    /// Get total execution time
    [[nodiscard]] std::chrono::nanoseconds totalTime() const {
        return std::chrono::nanoseconds(total_time_ns_);
    }

    /// Get common sizes sorted by frequency (most common first)
    [[nodiscard]] std::vector<SizeStats> commonSizes() const;

    /// Check if there's enough data to recommend specialization
    [[nodiscard]] bool shouldSpecialize() const;

    /// Get dominant size if one exists (>80% of calls)
    [[nodiscard]] std::optional<size_t> dominantSize() const;

    /// Get average time per element
    [[nodiscard]] float averageTimePerElement() const;

    /// Serialize profile data to string
    [[nodiscard]] std::string serialize() const;

    /// Deserialize profile data from string
    bool deserialize(const std::string& data);

    /// Reset all profile data
    void reset();

  private:
    uint64_t total_calls_ = 0;
    uint64_t total_elements_ = 0;
    uint64_t total_time_ns_ = 0;

    // Size histogram (size -> count, time)
    static constexpr size_t kMaxTrackedSizes = 16;
    std::array<SizeStats, kMaxTrackedSizes> size_stats_{};
    size_t num_tracked_sizes_ = 0;

    // Minimum calls before recommending specialization
    static constexpr uint64_t kMinCallsForSpecialization = 50;

    void updateSizeStats(size_t size, uint64_t time_ns);
};

// =============================================================================
// SpecializedIR
// =============================================================================

/// Represents a size-specialized version of IR
class SpecializedIR {
  public:
    SpecializedIR() = default;
    explicit SpecializedIR(size_t size) : specialized_size_(size) {}

    /// Get the size this IR is specialized for
    [[nodiscard]] size_t specializedSize() const { return specialized_size_; }

    /// Check if this is a valid specialization
    [[nodiscard]] bool isValid() const { return specialized_size_ > 0; }

  private:
    size_t specialized_size_ = 0;
    // Additional specialized IR state would go here
};

// =============================================================================
// Hot Path
// =============================================================================

/// Represents a frequently executed code path
struct HotPath {
    std::vector<ir::ValueId> operations;
    uint64_t execution_count = 0;
    float avg_time_ns = 0.0f;
};

// =============================================================================
// PGO Result
// =============================================================================

/// Result of applying profile-guided optimization
struct PGOResult {
    std::vector<SpecializedIR> specializations;
    std::vector<HotPath> optimized_paths;
    bool success = false;
};

// =============================================================================
// Cache Stats
// =============================================================================

/// Statistics for the specialization cache
struct SpecializationCacheStats {
    size_t total_entries = 0;
    size_t hits = 0;
    size_t misses = 0;
    size_t evictions = 0;
};

// =============================================================================
// PGOSpecializer
// =============================================================================

/// Profile-guided optimization specializer
class PGOSpecializer {
  public:
    PGOSpecializer();

    // -------------------------------------------------------------------------
    // Enable/Disable
    // -------------------------------------------------------------------------

    /// Check if PGO is enabled
    [[nodiscard]] bool isEnabled() const { return enabled_; }

    /// Enable PGO
    void enable() { enabled_ = true; }

    /// Disable PGO
    void disable() { enabled_ = false; }

    // -------------------------------------------------------------------------
    // Size Specialization
    // -------------------------------------------------------------------------

    /// Create a size-specialized version of the IR
    [[nodiscard]] std::optional<SpecializedIR> specializeForSize(const ir::IRBuilder& builder,
                                                                 size_t size);

    /// Create specializations for multiple sizes
    [[nodiscard]] std::vector<SpecializedIR> specializeForSizes(const ir::IRBuilder& builder,
                                                                const std::vector<size_t>& sizes);

    /// Select best specialization for given size
    [[nodiscard]] const SpecializedIR*
    selectSpecialization(const std::vector<SpecializedIR>& specializations, size_t size) const;

    // -------------------------------------------------------------------------
    // Profile-Based Specialization
    // -------------------------------------------------------------------------

    /// Apply profile data to generate optimizations
    [[nodiscard]] std::optional<PGOResult> applyProfile(const ir::IRBuilder& builder,
                                                        const ProfileData& profile);

    /// Identify hot paths in the IR based on profile
    [[nodiscard]] std::vector<HotPath> identifyHotPaths(const ir::IRBuilder& builder,
                                                        const ProfileData& profile) const;

    /// Optimize hot paths based on profile data
    [[nodiscard]] std::optional<ir::IRBuilder> optimizeHotPaths(const ir::IRBuilder& builder,
                                                                const ProfileData& profile);

    // -------------------------------------------------------------------------
    // Caching
    // -------------------------------------------------------------------------

    /// Get or create a cached specialization
    [[nodiscard]] std::optional<SpecializedIR>
    getOrCreateSpecialization(const ir::IRBuilder& builder, size_t size);

    /// Get cache statistics
    [[nodiscard]] SpecializationCacheStats cacheStats() const;

    /// Set maximum cache size
    void setMaxCacheSize(size_t max_entries);

    /// Clear the specialization cache
    void clearCache();

    // -------------------------------------------------------------------------
    // Profile Management
    // -------------------------------------------------------------------------

    /// Load profile data from serialized string
    bool loadProfile(const std::string& data);

    /// Get currently loaded profile
    [[nodiscard]] std::optional<ProfileData> currentProfile() const;

    /// Set the current profile
    void setProfile(const ProfileData& profile);

  private:
    bool enabled_ = true;

    // Specialization cache
    mutable std::mutex cache_mutex_;
    std::unordered_map<size_t, SpecializedIR> cache_;
    size_t max_cache_size_ = 32;
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;
    size_t cache_evictions_ = 0;

    // Current profile
    std::optional<ProfileData> current_profile_;

    // Size thresholds for specialization
    static constexpr size_t kMaxSpecializationRatio = 4;  // Don't specialize if size differs by >4x

    void evictIfNeeded();
};

}  // namespace jit
}  // namespace bud
