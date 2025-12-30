/**
 * @file tiled_executor.h
 * @brief Cache-aware tiled execution for array operations
 *
 * Provides tiled (blocked) execution of array operations to maximize
 * cache utilization and memory bandwidth.
 */

#pragma once

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/prefetch.h"

#include <cstddef>
#include <functional>

namespace bud {
namespace memory {

/**
 * @brief Cache-aware tiled executor for array operations
 *
 * Executes array operations in cache-friendly tiles to maximize
 * data reuse and minimize cache misses.
 */
class TiledExecutor {
  public:
    /**
     * @brief Construct executor with auto-detected cache config
     */
    TiledExecutor();

    /**
     * @brief Construct executor with explicit cache config
     * @param config Cache configuration to use
     */
    explicit TiledExecutor(const CacheConfig& config);

    /**
     * @brief Execute binary operation with tiling
     *
     * @tparam Op Binary operation type: (T, T) -> T
     * @param out Output array
     * @param a First input array
     * @param b Second input array
     * @param count Number of elements
     * @param op Binary operation to apply
     */
    template <typename T, typename Op>
    void binaryOp(T* out, const T* a, const T* b, size_t count, Op op) const {
        const size_t tile_size = config_.optimalTileSize(sizeof(T), 3);
        binaryOpTiled(out, a, b, count, tile_size, op);
    }

    /**
     * @brief Execute binary operation with tiling and prefetching
     *
     * @tparam Op Binary operation type: (T, T) -> T
     * @param out Output array
     * @param a First input array
     * @param b Second input array
     * @param count Number of elements
     * @param op Binary operation to apply
     */
    template <typename T, typename Op>
    void binaryOpWithPrefetch(T* out, const T* a, const T* b, size_t count, Op op) const {
        const size_t tile_size = config_.optimalTileSize(sizeof(T), 3);
        binaryOpTiledWithPrefetch(out, a, b, count, tile_size, op);
    }

    /**
     * @brief Execute unary operation with tiling
     *
     * @tparam Op Unary operation type: (T) -> T
     * @param out Output array
     * @param a Input array
     * @param count Number of elements
     * @param op Unary operation to apply
     */
    template <typename T, typename Op>
    void unaryOp(T* out, const T* a, size_t count, Op op) const {
        const size_t tile_size = config_.optimalTileSize(sizeof(T), 2);
        unaryOpTiled(out, a, count, tile_size, op);
    }

    /**
     * @brief Execute ternary operation with tiling (e.g., FMA)
     *
     * @tparam Op Ternary operation type: (T, T, T) -> T
     * @param out Output array
     * @param a First input array
     * @param b Second input array
     * @param c Third input array
     * @param count Number of elements
     * @param op Ternary operation to apply
     */
    template <typename T, typename Op>
    void ternaryOp(T* out, const T* a, const T* b, const T* c, size_t count, Op op) const {
        const size_t tile_size = config_.optimalTileSize(sizeof(T), 4);
        ternaryOpTiled(out, a, b, c, count, tile_size, op);
    }

    /**
     * @brief Execute reduction with tiling
     *
     * @tparam T Element type
     * @tparam Op Reduction operation type: (T, T) -> T
     * @param a Input array
     * @param count Number of elements
     * @param init Initial value
     * @param op Reduction operation
     * @return Reduced value
     */
    template <typename T, typename Op>
    T reduce(const T* a, size_t count, T init, Op op) const {
        const size_t tile_size = config_.optimalTileSize(sizeof(T), 1);
        return reduceTiled(a, count, init, tile_size, op);
    }

    /**
     * @brief Execute dot product with tiling
     *
     * @param a First input array
     * @param b Second input array
     * @param count Number of elements
     * @return Dot product result
     */
    float dotProduct(const float* a, const float* b, size_t count) const;

    /**
     * @brief Execute dot product with tiling (double precision)
     *
     * @param a First input array
     * @param b Second input array
     * @param count Number of elements
     * @return Dot product result
     */
    double dotProduct(const double* a, const double* b, size_t count) const;

    /**
     * @brief Get current tile size
     */
    [[nodiscard]] size_t tileSize(size_t element_size, size_t num_arrays = 3) const {
        return config_.optimalTileSize(element_size, num_arrays);
    }

    /**
     * @brief Set minimum array size for tiling
     *
     * Arrays smaller than this will be processed without tiling.
     *
     * @param min_size Minimum size in elements
     */
    void setMinTilingSize(size_t min_size) { min_tiling_size_ = min_size; }

    /**
     * @brief Get minimum array size for tiling
     */
    [[nodiscard]] size_t minTilingSize() const { return min_tiling_size_; }

    /**
     * @brief Enable or disable prefetching
     */
    void setPrefetchEnabled(bool enabled) { prefetcher_.setEnabled(enabled); }

  private:
    CacheConfig config_;
    Prefetcher prefetcher_;
    size_t min_tiling_size_ = 256;  // Don't tile arrays smaller than this

    // Tiled implementations
    template <typename T, typename Op>
    void binaryOpTiled(T* out, const T* a, const T* b, size_t count, size_t tile_size, Op op) const;

    template <typename T, typename Op>
    void binaryOpTiledWithPrefetch(T* out, const T* a, const T* b, size_t count, size_t tile_size,
                                   Op op) const;

    template <typename T, typename Op>
    void unaryOpTiled(T* out, const T* a, size_t count, size_t tile_size, Op op) const;

    template <typename T, typename Op>
    void ternaryOpTiled(T* out, const T* a, const T* b, const T* c, size_t count, size_t tile_size,
                        Op op) const;

    template <typename T, typename Op>
    T reduceTiled(const T* a, size_t count, T init, size_t tile_size, Op op) const;
};

// =============================================================================
// Template Implementations
// =============================================================================

template <typename T, typename Op>
void TiledExecutor::binaryOpTiled(T* out, const T* a, const T* b, size_t count, size_t tile_size,
                                  Op op) const {
    // Skip tiling for small arrays
    if (count < min_tiling_size_) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = op(a[i], b[i]);
        }
        return;
    }

    // Process in tiles
    for (size_t tile_start = 0; tile_start < count; tile_start += tile_size) {
        const size_t tile_end = std::min(tile_start + tile_size, count);

        // Process current tile
        for (size_t i = tile_start; i < tile_end; ++i) {
            out[i] = op(a[i], b[i]);
        }
    }
}

template <typename T, typename Op>
void TiledExecutor::binaryOpTiledWithPrefetch(T* out, const T* a, const T* b, size_t count,
                                              size_t tile_size, Op op) const {
    if (count < min_tiling_size_ || !prefetcher_.isEnabled()) {
        binaryOpTiled(out, a, b, count, tile_size, op);
        return;
    }

    const size_t prefetch_dist = prefetcher_.optimalDistance(sizeof(T));

    for (size_t tile_start = 0; tile_start < count; tile_start += tile_size) {
        const size_t tile_end = std::min(tile_start + tile_size, count);

        // Prefetch next tile
        const size_t next_tile = tile_start + tile_size;
        if (next_tile < count) {
            const size_t prefetch_end = std::min(next_tile + tile_size, count);
            prefetcher_.prefetchRange(a + next_tile, prefetch_end - next_tile);
            prefetcher_.prefetchRange(b + next_tile, prefetch_end - next_tile);
        }

        // Process current tile with streaming prefetch
        for (size_t i = tile_start; i < tile_end; ++i) {
            if (i + prefetch_dist < count) {
                prefetcher_.prefetch(a + i + prefetch_dist);
                prefetcher_.prefetch(b + i + prefetch_dist);
            }
            out[i] = op(a[i], b[i]);
        }
    }
}

template <typename T, typename Op>
void TiledExecutor::unaryOpTiled(T* out, const T* a, size_t count, size_t tile_size, Op op) const {
    if (count < min_tiling_size_) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = op(a[i]);
        }
        return;
    }

    for (size_t tile_start = 0; tile_start < count; tile_start += tile_size) {
        const size_t tile_end = std::min(tile_start + tile_size, count);

        for (size_t i = tile_start; i < tile_end; ++i) {
            out[i] = op(a[i]);
        }
    }
}

template <typename T, typename Op>
void TiledExecutor::ternaryOpTiled(T* out, const T* a, const T* b, const T* c, size_t count,
                                   size_t tile_size, Op op) const {
    if (count < min_tiling_size_) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = op(a[i], b[i], c[i]);
        }
        return;
    }

    for (size_t tile_start = 0; tile_start < count; tile_start += tile_size) {
        const size_t tile_end = std::min(tile_start + tile_size, count);

        for (size_t i = tile_start; i < tile_end; ++i) {
            out[i] = op(a[i], b[i], c[i]);
        }
    }
}

template <typename T, typename Op>
T TiledExecutor::reduceTiled(const T* a, size_t count, T init, size_t tile_size, Op op) const {
    if (count < min_tiling_size_) {
        T result = init;
        for (size_t i = 0; i < count; ++i) {
            result = op(result, a[i]);
        }
        return result;
    }

    // Use multiple accumulators for better ILP
    constexpr size_t kNumAccumulators = 4;
    T accumulators[kNumAccumulators] = {init, init, init, init};

    for (size_t tile_start = 0; tile_start < count; tile_start += tile_size) {
        const size_t tile_end = std::min(tile_start + tile_size, count);
        const size_t tile_count = tile_end - tile_start;

        // Process tile with multiple accumulators
        size_t i = tile_start;
        for (; i + kNumAccumulators <= tile_end; i += kNumAccumulators) {
            accumulators[0] = op(accumulators[0], a[i]);
            accumulators[1] = op(accumulators[1], a[i + 1]);
            accumulators[2] = op(accumulators[2], a[i + 2]);
            accumulators[3] = op(accumulators[3], a[i + 3]);
        }

        // Handle remainder within tile
        for (; i < tile_end; ++i) {
            accumulators[0] = op(accumulators[0], a[i]);
        }
    }

    // Combine accumulators
    T result = op(accumulators[0], accumulators[1]);
    result = op(result, accumulators[2]);
    result = op(result, accumulators[3]);

    return result;
}

}  // namespace memory
}  // namespace bud
