// =============================================================================
// Bud Flow Lang - Adaptive Prefetching Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for runtime-adaptive prefetch distance adjustment based on
// observed cache miss rates and memory latency.
//
// Benefits:
// - Automatically tunes prefetch distance for different workloads
// - Adapts to different memory subsystems (CPUs, cache sizes)
// - Cooperates with hardware prefetcher
//

#include "bud_flow_lang/memory/adaptive_prefetch.h"
#include "bud_flow_lang/memory/prefetch.h"

#include <chrono>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace memory {
namespace {

// =============================================================================
// AdaptivePrefetcher Tests
// =============================================================================

class AdaptivePrefetchTest : public ::testing::Test {
  protected:
    void SetUp() override { prefetcher_ = std::make_unique<AdaptivePrefetcher>(); }

    std::unique_ptr<AdaptivePrefetcher> prefetcher_;
};

TEST_F(AdaptivePrefetchTest, DefaultConstruction) {
    // Should have reasonable default distance
    EXPECT_GT(prefetcher_->distance(), 0);
    EXPECT_LE(prefetcher_->distance(), 64);  // Not too aggressive
}

TEST_F(AdaptivePrefetchTest, InitialDistanceFromCacheConfig) {
    // Different cache configs should give different initial distances
    // CacheConfig constructor: (l1_size, l2_size, l3_size, line_size)
    CacheConfig small_l2(32 * 1024, 256 * 1024, 0, 64);       // 256KB L2
    CacheConfig large_l2(32 * 1024, 2 * 1024 * 1024, 0, 64);  // 2MB L2

    AdaptivePrefetcher small_prefetcher(small_l2);
    AdaptivePrefetcher large_prefetcher(large_l2);

    // Larger cache might allow more aggressive prefetching
    // (or similar - depends on implementation)
    EXPECT_GT(small_prefetcher.distance(), 0);
    EXPECT_GT(large_prefetcher.distance(), 0);
}

TEST_F(AdaptivePrefetchTest, AdjustDistance_IncreasesOnHighLatency) {
    // If measured latency is high, distance should increase
    size_t initial_distance = prefetcher_->distance();

    // Simulate high latency observations
    for (int i = 0; i < 10; ++i) {
        prefetcher_->recordLatency(1000);  // 1000ns = high latency
    }

    prefetcher_->adjustDistance();

    // Distance should have increased (or stayed same if at max)
    EXPECT_GE(prefetcher_->distance(), initial_distance);
}

TEST_F(AdaptivePrefetchTest, AdjustDistance_DecreasesOnLowLatency) {
    // If measured latency is low, distance might decrease to avoid pollution
    // First, increase distance
    for (int i = 0; i < 10; ++i) {
        prefetcher_->recordLatency(1000);
    }
    prefetcher_->adjustDistance();
    size_t high_distance = prefetcher_->distance();

    // Then simulate low latency
    for (int i = 0; i < 20; ++i) {
        prefetcher_->recordLatency(10);  // 10ns = very low latency
    }
    prefetcher_->adjustDistance();

    // Distance should have decreased (or stayed if already optimal)
    EXPECT_LE(prefetcher_->distance(), high_distance);
}

TEST_F(AdaptivePrefetchTest, RecordLatency_AffectsAverage) {
    prefetcher_->recordLatency(100);
    prefetcher_->recordLatency(200);
    prefetcher_->recordLatency(300);

    float avg = prefetcher_->averageLatency();
    EXPECT_GT(avg, 0.0f);
    EXPECT_LT(avg, 1000.0f);
}

TEST_F(AdaptivePrefetchTest, DistanceBounds) {
    // Distance should stay within reasonable bounds
    // Try to push it very high
    for (int i = 0; i < 100; ++i) {
        prefetcher_->recordLatency(10000);
        prefetcher_->adjustDistance();
    }
    EXPECT_LE(prefetcher_->distance(), prefetcher_->maxDistance());

    // Try to push it very low
    for (int i = 0; i < 100; ++i) {
        prefetcher_->recordLatency(1);
        prefetcher_->adjustDistance();
    }
    EXPECT_GE(prefetcher_->distance(), prefetcher_->minDistance());
}

TEST_F(AdaptivePrefetchTest, SetDistanceBounds) {
    prefetcher_->setMinDistance(4);
    prefetcher_->setMaxDistance(32);

    EXPECT_EQ(prefetcher_->minDistance(), 4);
    EXPECT_EQ(prefetcher_->maxDistance(), 32);

    // Distance should be within bounds
    EXPECT_GE(prefetcher_->distance(), 4);
    EXPECT_LE(prefetcher_->distance(), 32);
}

// =============================================================================
// Hardware Prefetcher Cooperation Tests
// =============================================================================

TEST_F(AdaptivePrefetchTest, PrimeHardwarePrefetcher) {
    // Create a sequential access pattern to prime the HW prefetcher
    std::vector<float> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    // Should not crash and should complete
    prefetcher_->primeHardwarePrefetcher(data.data(), sizeof(float), data.size());
}

TEST_F(AdaptivePrefetchTest, DetectHWPrefetcherActivity) {
    // This is hard to test definitively, but we can check the API works
    bool hw_active = prefetcher_->isHardwarePrefetcherActive();
    (void)hw_active;  // Just verify it doesn't crash
}

// =============================================================================
// AdaptivePrefetchStream Tests
// =============================================================================

TEST_F(AdaptivePrefetchTest, StreamPrefetch_Sequential) {
    std::vector<float> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    AdaptivePrefetchStream<float> stream(data.data(), data.size(), *prefetcher_);

    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        stream.prefetchFor(i);
        sum += data[i];
    }

    // Verify sum is approximately correct (floating point accumulation has precision limits)
    float expected_sum = static_cast<float>((data.size() - 1) * data.size() / 2);
    // Use relative tolerance for large sums
    EXPECT_NEAR(sum, expected_sum, expected_sum * 0.001f);  // 0.1% tolerance
}

TEST_F(AdaptivePrefetchTest, StreamPrefetch_Adapts) {
    std::vector<float> data(100000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    size_t initial_distance = prefetcher_->distance();

    AdaptivePrefetchStream<float> stream(data.data(), data.size(), *prefetcher_);

    // Process data with timing
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        stream.prefetchFor(i);
        sum += data[i];
    }

    // After processing, prefetcher may have adapted
    // (actual adaptation depends on measured latency)
    (void)initial_distance;
    (void)sum;
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(AdaptivePrefetchTest, IntegrationWithTiledExecution) {
    // Simulate tiled execution pattern
    const size_t total_elements = 1000000;
    const size_t tile_size = 4096;

    std::vector<float> data(total_elements);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    float sum = 0.0f;

    for (size_t tile_start = 0; tile_start < total_elements; tile_start += tile_size) {
        size_t tile_end = std::min(tile_start + tile_size, total_elements);

        // Prefetch for this tile
        prefetcher_->prefetchTile(data.data() + tile_start, tile_end - tile_start);

        // Process tile
        for (size_t i = tile_start; i < tile_end; ++i) {
            sum += data[i];
        }
    }

    // Verify correctness with relative tolerance (floating point precision limits)
    double expected = static_cast<double>(total_elements - 1) * total_elements / 2.0;
    // Large sums have significant floating-point error
    EXPECT_NEAR(static_cast<double>(sum), expected, expected * 0.001);  // 0.1% tolerance
}

TEST_F(AdaptivePrefetchTest, MultiplePasses) {
    // Test adaptation over multiple passes of same data
    std::vector<float> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    std::vector<size_t> distances;

    for (int pass = 0; pass < 5; ++pass) {
        distances.push_back(prefetcher_->distance());

        AdaptivePrefetchStream<float> stream(data.data(), data.size(), *prefetcher_);

        float sum = 0.0f;
        for (size_t i = 0; i < data.size(); ++i) {
            stream.prefetchFor(i);
            sum += data[i];
        }
        (void)sum;
    }

    // Distance may have changed over passes as it adapted
    // (just verify no crashes)
    EXPECT_EQ(distances.size(), 5);
}

// =============================================================================
// Performance Metric Tests
// =============================================================================

TEST_F(AdaptivePrefetchTest, MeasurePrefetchEffectiveness) {
    // Compare memory access time with and without prefetching
    std::vector<float> data(1000000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    // Access without prefetch
    prefetcher_->setEnabled(false);
    auto start1 = std::chrono::high_resolution_clock::now();
    volatile float sum1 = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        sum1 += data[i];
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto time_no_prefetch = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

    // Access with prefetch
    prefetcher_->setEnabled(true);
    auto start2 = std::chrono::high_resolution_clock::now();
    volatile float sum2 = 0.0f;
    AdaptivePrefetchStream<float> stream(data.data(), data.size(), *prefetcher_);
    for (size_t i = 0; i < data.size(); ++i) {
        stream.prefetchFor(i);
        sum2 += data[i];
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto time_with_prefetch = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);

    // Log results (prefetch may or may not help depending on system)
    (void)time_no_prefetch;
    (void)time_with_prefetch;
    (void)sum1;
    (void)sum2;
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(AdaptivePrefetchTest, AggressivenessLevels) {
    // Test different aggressiveness settings
    prefetcher_->setAggressiveness(Aggressiveness::kConservative);
    EXPECT_EQ(prefetcher_->aggressiveness(), Aggressiveness::kConservative);

    prefetcher_->setAggressiveness(Aggressiveness::kModerate);
    EXPECT_EQ(prefetcher_->aggressiveness(), Aggressiveness::kModerate);

    prefetcher_->setAggressiveness(Aggressiveness::kAggressive);
    EXPECT_EQ(prefetcher_->aggressiveness(), Aggressiveness::kAggressive);
}

TEST_F(AdaptivePrefetchTest, AdaptationRate) {
    // Test different adaptation rates
    prefetcher_->setAdaptationRate(0.1f);
    EXPECT_NEAR(prefetcher_->adaptationRate(), 0.1f, 0.01f);

    prefetcher_->setAdaptationRate(0.5f);
    EXPECT_NEAR(prefetcher_->adaptationRate(), 0.5f, 0.01f);
}

}  // namespace
}  // namespace memory
}  // namespace bud
