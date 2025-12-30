/**
 * @file cache_config.h
 * @brief Cache configuration detection and optimal tile size calculation
 *
 * Provides runtime detection of CPU cache sizes and calculation of
 * optimal tile sizes for cache-aware loop tiling.
 */

#pragma once

#include "bud_flow_lang/common.h"

#include <cstddef>
#include <cstdint>

namespace bud {
namespace memory {

/**
 * @brief Cache level enumeration
 */
enum class CacheLevel : uint8_t {
    kL1 = 1,
    kL2 = 2,
    kL3 = 3,
};

/**
 * @brief CPU cache configuration
 *
 * Holds detected cache sizes and provides methods for calculating
 * optimal tile sizes for cache-aware algorithms.
 */
class CacheConfig {
  public:
    /**
     * @brief Detect cache configuration from the current CPU
     * @return CacheConfig with detected values
     */
    [[nodiscard]] static CacheConfig detect();

    /**
     * @brief Create a manual cache configuration
     * @param l1_size L1 data cache size in bytes
     * @param l2_size L2 cache size in bytes
     * @param l3_size L3 cache size in bytes (0 if not present)
     * @param line_size Cache line size in bytes
     */
    explicit CacheConfig(size_t l1_size, size_t l2_size, size_t l3_size, size_t line_size);

    /**
     * @brief Default constructor with reasonable defaults
     */
    CacheConfig();

    // Accessors
    [[nodiscard]] size_t l1Size() const { return l1_size_; }
    [[nodiscard]] size_t l2Size() const { return l2_size_; }
    [[nodiscard]] size_t l3Size() const { return l3_size_; }
    [[nodiscard]] size_t lineSize() const { return line_size_; }

    /**
     * @brief Calculate optimal tile size for L1 cache
     *
     * Returns a tile size that fits working set in L1 cache with headroom
     * for other data. The returned size is aligned to SIMD width.
     *
     * @param element_size Size of each element in bytes
     * @param num_arrays Number of arrays accessed per iteration (default 3: a, b, out)
     * @param utilization_factor Fraction of cache to use (default 0.5 for safety)
     * @return Optimal tile size in number of elements
     */
    [[nodiscard]] size_t optimalTileSize(size_t element_size, size_t num_arrays = 3,
                                         float utilization_factor = 0.5f) const;

    /**
     * @brief Calculate optimal tile size for a specific cache level
     *
     * @param element_size Size of each element in bytes
     * @param level Target cache level
     * @param num_arrays Number of arrays accessed per iteration
     * @param utilization_factor Fraction of cache to use
     * @return Optimal tile size in number of elements
     */
    [[nodiscard]] size_t optimalTileSizeForLevel(size_t element_size, CacheLevel level,
                                                 size_t num_arrays = 3,
                                                 float utilization_factor = 0.5f) const;

    /**
     * @brief Get prefetch distance in elements
     *
     * @param element_size Size of each element in bytes
     * @return Prefetch distance in number of elements
     */
    [[nodiscard]] size_t prefetchDistance(size_t element_size) const;

  private:
    size_t l1_size_;    // L1 data cache size in bytes
    size_t l2_size_;    // L2 cache size in bytes
    size_t l3_size_;    // L3 cache size in bytes (0 if not present)
    size_t line_size_;  // Cache line size in bytes

    // Helper to align tile size to SIMD width
    [[nodiscard]] size_t alignToSimd(size_t elements, size_t element_size) const;

    // Get cache size for a specific level
    [[nodiscard]] size_t cacheSizeForLevel(CacheLevel level) const;
};

}  // namespace memory
}  // namespace bud
