/**
 * @file tiled_executor.cc
 * @brief Cache-aware tiled executor implementation
 */

#include "bud_flow_lang/memory/tiled_executor.h"

#include <algorithm>
#include <cmath>

// Include Highway for SIMD operations
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

namespace bud {
namespace memory {

TiledExecutor::TiledExecutor() : TiledExecutor(CacheConfig::detect()) {}

TiledExecutor::TiledExecutor(const CacheConfig& config) : config_(config), prefetcher_(config) {}

namespace {

// SIMD dot product implementation with tiling and multiple accumulators
template <typename T>
T dotProductSimd(const T* a, const T* b, size_t count, const CacheConfig& config,
                 const Prefetcher& prefetcher, size_t min_tiling_size) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);

    // Use 4 accumulators for instruction-level parallelism
    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    // Calculate tile size (targeting L1 cache)
    const size_t tile_size = config.optimalTileSize(sizeof(T), 2);
    const size_t prefetch_dist = prefetcher.optimalDistance(sizeof(T));

    // Process in tiles for cache efficiency
    for (size_t tile_start = 0; tile_start < count; tile_start += tile_size) {
        const size_t tile_end = std::min(tile_start + tile_size, count);

        // Prefetch next tile if prefetching is enabled
        if (prefetcher.isEnabled() && tile_start + tile_size < count) {
            const size_t next_tile = tile_start + tile_size;
            const size_t prefetch_end = std::min(next_tile + tile_size, count);
            prefetcher.prefetchRange(a + next_tile, prefetch_end - next_tile);
            prefetcher.prefetchRange(b + next_tile, prefetch_end - next_tile);
        }

        // Process tile with SIMD
        size_t i = tile_start;

        // Process 4 vectors at a time for better ILP
        for (; i + 4 * N <= tile_end; i += 4 * N) {
            // Streaming prefetch within tile
            if (prefetcher.isEnabled() && i + prefetch_dist < count) {
                prefetcher.prefetch(a + i + prefetch_dist, PrefetchHint::kT0);
                prefetcher.prefetch(b + i + prefetch_dist, PrefetchHint::kT0);
            }

            const auto a0 = hn::LoadU(d, a + i);
            const auto b0 = hn::LoadU(d, b + i);
            sum0 = hn::MulAdd(a0, b0, sum0);

            const auto a1 = hn::LoadU(d, a + i + N);
            const auto b1 = hn::LoadU(d, b + i + N);
            sum1 = hn::MulAdd(a1, b1, sum1);

            const auto a2 = hn::LoadU(d, a + i + 2 * N);
            const auto b2 = hn::LoadU(d, b + i + 2 * N);
            sum2 = hn::MulAdd(a2, b2, sum2);

            const auto a3 = hn::LoadU(d, a + i + 3 * N);
            const auto b3 = hn::LoadU(d, b + i + 3 * N);
            sum3 = hn::MulAdd(a3, b3, sum3);
        }

        // Process remaining full vectors in tile
        for (; i + N <= tile_end; i += N) {
            const auto va = hn::LoadU(d, a + i);
            const auto vb = hn::LoadU(d, b + i);
            sum0 = hn::MulAdd(va, vb, sum0);
        }

        // Handle partial vector at end of tile
        if (i < tile_end) {
            const size_t remaining = tile_end - i;
            const auto va = hn::LoadN(d, a + i, remaining);
            const auto vb = hn::LoadN(d, b + i, remaining);
            sum0 = hn::MulAdd(va, vb, sum0);
        }
    }

    // Combine accumulators: tree reduction for better accuracy
    sum0 = hn::Add(sum0, sum1);
    sum2 = hn::Add(sum2, sum3);
    sum0 = hn::Add(sum0, sum2);

    // Final horizontal reduction
    return hn::ReduceSum(d, sum0);
}

}  // namespace

float TiledExecutor::dotProduct(const float* a, const float* b, size_t count) const {
    if (count == 0)
        return 0.0f;

    // For small arrays, skip tiling overhead
    if (count < min_tiling_size_) {
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    return dotProductSimd(a, b, count, config_, prefetcher_, min_tiling_size_);
}

double TiledExecutor::dotProduct(const double* a, const double* b, size_t count) const {
    if (count == 0)
        return 0.0;

    // For small arrays, skip tiling overhead
    if (count < min_tiling_size_) {
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    return dotProductSimd(a, b, count, config_, prefetcher_, min_tiling_size_);
}

}  // namespace memory
}  // namespace bud
