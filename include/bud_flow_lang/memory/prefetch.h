/**
 * @file prefetch.h
 * @brief Software prefetching utilities
 *
 * Provides portable software prefetch operations with configurable
 * locality hints for different cache levels.
 */

#pragma once

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/memory/cache_config.h"

#include <cstddef>
#include <cstdint>

namespace bud {
namespace memory {

/**
 * @brief Prefetch locality hint
 *
 * Controls which cache levels receive the prefetched data.
 */
enum class PrefetchHint : uint8_t {
    kT0 = 0,   // Prefetch to all cache levels (highest locality)
    kT1 = 1,   // Prefetch to L2 and above
    kT2 = 2,   // Prefetch to L3 and above
    kNTA = 3,  // Non-temporal prefetch (minimize cache pollution)
};

/**
 * @brief Software prefetcher with configurable distance and hints
 *
 * Provides methods for issuing software prefetch instructions with
 * automatic optimal distance calculation based on cache configuration.
 */
class Prefetcher {
  public:
    /**
     * @brief Construct prefetcher with auto-detected cache config
     */
    Prefetcher();

    /**
     * @brief Construct prefetcher with explicit cache config
     * @param config Cache configuration to use
     */
    explicit Prefetcher(const CacheConfig& config);

    /**
     * @brief Prefetch a single cache line
     *
     * @param addr Address to prefetch
     * @param hint Locality hint (default: T0 for highest locality)
     */
    void prefetch(const void* addr, PrefetchHint hint = PrefetchHint::kT0) const;

    /**
     * @brief Prefetch multiple cache lines covering a range
     *
     * @param addr Starting address
     * @param count Number of elements
     * @param hint Locality hint
     */
    template <typename T>
    void prefetchRange(const T* addr, size_t count, PrefetchHint hint = PrefetchHint::kT0) const {
        prefetchBytes(addr, count * sizeof(T), hint);
    }

    /**
     * @brief Prefetch range specified in bytes
     *
     * @param addr Starting address
     * @param bytes Number of bytes to prefetch
     * @param hint Locality hint
     */
    void prefetchBytes(const void* addr, size_t bytes, PrefetchHint hint = PrefetchHint::kT0) const;

    /**
     * @brief Get optimal prefetch distance for streaming access
     *
     * @param element_size Size of each element in bytes
     * @return Prefetch distance in number of elements
     */
    [[nodiscard]] size_t optimalDistance(size_t element_size) const;

    /**
     * @brief Get optimal prefetch distance in bytes
     * @return Prefetch distance in bytes
     */
    [[nodiscard]] size_t optimalDistanceBytes() const;

    /**
     * @brief Set custom prefetch distance
     * @param distance_bytes Distance in bytes
     */
    void setDistance(size_t distance_bytes);

    /**
     * @brief Enable or disable prefetching
     * @param enabled True to enable, false to disable (no-op prefetches)
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }

    /**
     * @brief Check if prefetching is enabled
     */
    [[nodiscard]] bool isEnabled() const { return enabled_; }

  private:
    CacheConfig config_;
    size_t distance_bytes_;  // Prefetch distance in bytes
    bool enabled_;

    // Issue single prefetch instruction
    static void issuePrefetch(const void* addr, PrefetchHint hint);
};

/**
 * @brief RAII prefetch stream for sequential access patterns
 *
 * Automatically prefetches ahead during sequential iteration.
 */
template <typename T>
class PrefetchStream {
  public:
    /**
     * @brief Create prefetch stream
     * @param data Pointer to start of data
     * @param count Total number of elements
     * @param prefetcher Prefetcher to use
     */
    PrefetchStream(const T* data, size_t count, const Prefetcher& prefetcher)
        : data_(data),
          count_(count),
          prefetcher_(prefetcher),
          distance_(prefetcher.optimalDistance(sizeof(T))) {}

    /**
     * @brief Prefetch for position i (call before accessing data[i])
     * @param i Current position
     */
    void prefetchFor(size_t i) const {
        size_t prefetch_idx = i + distance_;
        if (prefetch_idx < count_) {
            prefetcher_.prefetch(&data_[prefetch_idx]);
        }
    }

    /**
     * @brief Get prefetch distance in elements
     */
    [[nodiscard]] size_t distance() const { return distance_; }

  private:
    const T* data_;
    size_t count_;
    const Prefetcher& prefetcher_;
    size_t distance_;
};

}  // namespace memory
}  // namespace bud
