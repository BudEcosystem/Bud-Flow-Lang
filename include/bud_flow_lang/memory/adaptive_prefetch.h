#pragma once

// =============================================================================
// Bud Flow Lang - Adaptive Prefetching
// =============================================================================
//
// Runtime-adaptive prefetch distance adjustment based on observed memory
// latency and cache miss rates.
//
// Features:
// - Automatically tunes prefetch distance for different workloads
// - Adapts to different memory subsystems (CPUs, cache sizes)
// - Cooperates with hardware prefetcher
//

#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/prefetch.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace bud {
namespace memory {

// =============================================================================
// Aggressiveness Levels
// =============================================================================

/// Controls how aggressively the prefetcher adapts
enum class Aggressiveness : uint8_t {
    kConservative = 0,  // Slow adaptation, minimize cache pollution
    kModerate = 1,      // Balanced adaptation
    kAggressive = 2,    // Fast adaptation, prioritize performance
};

// =============================================================================
// AdaptivePrefetcher
// =============================================================================

/// Prefetcher with runtime-adaptive distance adjustment
class AdaptivePrefetcher {
  public:
    /// Construct with auto-detected cache config
    AdaptivePrefetcher();

    /// Construct with explicit cache config
    explicit AdaptivePrefetcher(const CacheConfig& config);

    // -------------------------------------------------------------------------
    // Core Prefetch Operations
    // -------------------------------------------------------------------------

    /// Prefetch a single cache line
    void prefetch(const void* addr, PrefetchHint hint = PrefetchHint::kT0) const;

    /// Prefetch multiple cache lines covering a range
    template <typename T>
    void prefetchRange(const T* addr, size_t count, PrefetchHint hint = PrefetchHint::kT0) const {
        prefetchBytes(addr, count * sizeof(T), hint);
    }

    /// Prefetch for a tile of data
    template <typename T>
    void prefetchTile(const T* addr, size_t count, PrefetchHint hint = PrefetchHint::kT0) const {
        prefetchBytes(addr, count * sizeof(T), hint);
    }

    /// Prefetch range specified in bytes
    void prefetchBytes(const void* addr, size_t bytes, PrefetchHint hint = PrefetchHint::kT0) const;

    // -------------------------------------------------------------------------
    // Adaptive Distance Control
    // -------------------------------------------------------------------------

    /// Get current prefetch distance (in cache lines)
    [[nodiscard]] size_t distance() const { return distance_; }

    /// Get minimum allowed distance
    [[nodiscard]] size_t minDistance() const { return min_distance_; }

    /// Get maximum allowed distance
    [[nodiscard]] size_t maxDistance() const { return max_distance_; }

    /// Set minimum allowed distance
    void setMinDistance(size_t min_dist);

    /// Set maximum allowed distance
    void setMaxDistance(size_t max_dist);

    /// Record observed memory latency (nanoseconds)
    void recordLatency(uint64_t latency_ns);

    /// Get average observed latency
    [[nodiscard]] float averageLatency() const;

    /// Adjust distance based on recorded observations
    void adjustDistance();

    // -------------------------------------------------------------------------
    // Hardware Prefetcher Cooperation
    // -------------------------------------------------------------------------

    /// Prime the hardware prefetcher with a sequential access pattern
    void primeHardwarePrefetcher(const void* data, size_t stride, size_t count) const;

    /// Check if hardware prefetcher appears active (heuristic)
    [[nodiscard]] bool isHardwarePrefetcherActive() const;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Get current aggressiveness level
    [[nodiscard]] Aggressiveness aggressiveness() const { return aggressiveness_; }

    /// Set aggressiveness level
    void setAggressiveness(Aggressiveness level);

    /// Get adaptation rate (0.0 = no adaptation, 1.0 = instant adaptation)
    [[nodiscard]] float adaptationRate() const { return adaptation_rate_; }

    /// Set adaptation rate
    void setAdaptationRate(float rate);

    /// Enable or disable prefetching
    void setEnabled(bool enabled) { enabled_ = enabled; }

    /// Check if prefetching is enabled
    [[nodiscard]] bool isEnabled() const { return enabled_; }

  private:
    CacheConfig config_;
    Prefetcher base_prefetcher_;

    // Adaptive state
    size_t distance_ = 8;       // Current prefetch distance (cache lines)
    size_t min_distance_ = 2;   // Minimum distance
    size_t max_distance_ = 64;  // Maximum distance

    // Latency tracking (exponential moving average)
    float avg_latency_ = 0.0f;
    size_t latency_samples_ = 0;

    // Configuration
    Aggressiveness aggressiveness_ = Aggressiveness::kModerate;
    float adaptation_rate_ = 0.1f;  // How quickly to adapt
    bool enabled_ = true;

    // Target latency thresholds (nanoseconds)
    static constexpr float kLowLatencyThreshold = 50.0f;    // L1 hit
    static constexpr float kHighLatencyThreshold = 200.0f;  // L3/memory

    // Issue prefetch instruction
    static void issuePrefetch(const void* addr, PrefetchHint hint);
};

// =============================================================================
// AdaptivePrefetchStream
// =============================================================================

/// Stream prefetcher with adaptive distance
template <typename T>
class AdaptivePrefetchStream {
  public:
    /// Create adaptive prefetch stream
    AdaptivePrefetchStream(const T* data, size_t count, AdaptivePrefetcher& prefetcher)
        : data_(data), count_(count), prefetcher_(prefetcher) {}

    /// Prefetch for position i
    void prefetchFor(size_t i) const {
        if (!prefetcher_.isEnabled()) {
            return;
        }

        size_t prefetch_idx = i + prefetcher_.distance();
        if (prefetch_idx < count_) {
            prefetcher_.prefetch(&data_[prefetch_idx]);
        }
    }

    /// Get current prefetch distance
    [[nodiscard]] size_t distance() const { return prefetcher_.distance(); }

  private:
    const T* data_;
    size_t count_;
    AdaptivePrefetcher& prefetcher_;
};

}  // namespace memory
}  // namespace bud
