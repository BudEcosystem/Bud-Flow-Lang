// Dynamic Tier Threshold Tests
// Tests for runtime-configurable size tier thresholds

#include "bud_flow_lang/codegen/hwy_ops.h"

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace bud::simd {

class DynamicThresholdTest : public ::testing::Test {
  protected:
    void SetUp() override { gen_.seed(42); }

    std::vector<float> RandomFloats(size_t count) {
        std::vector<float> v(count);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (size_t i = 0; i < count; ++i) {
            v[i] = dist(gen_);
        }
        return v;
    }

    template <typename Func>
    double BenchmarkOp(Func&& op, int iterations = 100) {
        // Warmup
        for (int i = 0; i < 10; ++i) {
            op();
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            op();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / iterations;
    }

    std::mt19937 gen_;
};

// =============================================================================
// Threshold Configuration Tests
// =============================================================================

// Test that default thresholds are reasonable
TEST_F(DynamicThresholdTest, DefaultThresholds_AreReasonable) {
    TierThresholds thresholds = GetDefaultThresholds();

    // Small threshold should be small (< 256)
    EXPECT_GT(thresholds.small_to_medium, 0);
    EXPECT_LE(thresholds.small_to_medium, 256);

    // Medium threshold should be between small and 64KB
    EXPECT_GT(thresholds.medium_to_large, thresholds.small_to_medium);
    EXPECT_LE(thresholds.medium_to_large, 64 * 1024);
}

// Test that thresholds can be configured
TEST_F(DynamicThresholdTest, ThresholdConfiguration) {
    TierThresholds custom{128, 8192};

    SetGlobalThresholds(custom);
    TierThresholds retrieved = GetGlobalThresholds();

    EXPECT_EQ(retrieved.small_to_medium, 128);
    EXPECT_EQ(retrieved.medium_to_large, 8192);

    // Reset to defaults
    SetGlobalThresholds(GetDefaultThresholds());
}

// Test that tier selection uses thresholds correctly
TEST_F(DynamicThresholdTest, TierSelection_UsesThresholds) {
    TierThresholds thresholds{64, 4096};
    SetGlobalThresholds(thresholds);

    EXPECT_EQ(GetSizeTier(32), SizeTier::Small);
    EXPECT_EQ(GetSizeTier(64), SizeTier::Medium);  // At boundary
    EXPECT_EQ(GetSizeTier(100), SizeTier::Medium);
    EXPECT_EQ(GetSizeTier(4096), SizeTier::Large);  // At boundary
    EXPECT_EQ(GetSizeTier(10000), SizeTier::Large);

    SetGlobalThresholds(GetDefaultThresholds());
}

// =============================================================================
// Cache-Based Threshold Detection
// =============================================================================

// Test cache size detection
TEST_F(DynamicThresholdTest, CacheSizeDetection) {
    CacheInfo info = DetectCacheInfo();

    // L1 data cache should be between 16KB and 128KB typically
    if (info.l1_data_cache > 0) {
        EXPECT_GE(info.l1_data_cache, 16 * 1024);
        EXPECT_LE(info.l1_data_cache, 256 * 1024);
    }

    // L2 cache should be between 128KB and 4MB typically
    if (info.l2_cache > 0) {
        EXPECT_GE(info.l2_cache, 128 * 1024);
        EXPECT_LE(info.l2_cache, 8 * 1024 * 1024);
    }

    // Cache line should be 32-128 bytes
    if (info.cache_line_size > 0) {
        EXPECT_GE(info.cache_line_size, 32);
        EXPECT_LE(info.cache_line_size, 128);
    }
}

// Test threshold calculation from cache info
TEST_F(DynamicThresholdTest, ThresholdFromCacheInfo) {
    CacheInfo info{32 * 1024, 256 * 1024, 64};  // 32KB L1, 256KB L2

    TierThresholds thresholds = CalculateThresholdsFromCache(info);

    // Small threshold should be fraction of L1 (to fit working set)
    // For float, L1=32KB means ~8K floats, small should be <2K
    EXPECT_GT(thresholds.small_to_medium, 0);
    EXPECT_LT(thresholds.small_to_medium, info.l1_data_cache / sizeof(float) / 4);

    // Medium threshold should be based on L2
    EXPECT_GT(thresholds.medium_to_large, thresholds.small_to_medium);
}

// =============================================================================
// Auto-Tuning Tests
// =============================================================================

// Test auto-tuning finds reasonable thresholds
TEST_F(DynamicThresholdTest, AutoTune_FindsReasonableThresholds) {
    TierThresholds tuned = AutoTuneThresholds();

    // Thresholds should be positive and ordered
    EXPECT_GT(tuned.small_to_medium, 0);
    EXPECT_GT(tuned.medium_to_large, tuned.small_to_medium);

    // Should be within reasonable bounds
    EXPECT_LE(tuned.small_to_medium, 1024);
    EXPECT_LE(tuned.medium_to_large, 64 * 1024);
}

// Test that operations use correct tier based on thresholds
TEST_F(DynamicThresholdTest, Operations_RespectThresholds) {
    // Set specific thresholds
    TierThresholds thresholds{64, 4096};
    SetGlobalThresholds(thresholds);

    // Small array - should use small kernel
    {
        const size_t count = 32;
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        std::vector<float> result(count);

        AddSizedDynamic(a.data(), b.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(result[i], a[i] + b[i], 1e-5f);
        }
    }

    // Medium array
    {
        const size_t count = 1024;
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        std::vector<float> result(count);

        AddSizedDynamic(a.data(), b.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(result[i], a[i] + b[i], 1e-5f);
        }
    }

    // Large array
    {
        const size_t count = 16384;
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        std::vector<float> result(count);

        AddSizedDynamic(a.data(), b.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(result[i], a[i] + b[i], 1e-5f);
        }
    }

    SetGlobalThresholds(GetDefaultThresholds());
}

// =============================================================================
// Double Precision Tests
// =============================================================================

TEST_F(DynamicThresholdTest, DoubleOperations_RespectThresholds) {
    TierThresholds thresholds{64, 4096};
    SetGlobalThresholds(thresholds);

    const size_t count = 1024;
    std::vector<double> a(count), b(count), result(count);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (size_t i = 0; i < count; ++i) {
        a[i] = dist(gen_);
        b[i] = dist(gen_);
    }

    AddSizedDynamic(a.data(), b.data(), result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(result[i], a[i] + b[i], 1e-10);
    }

    SetGlobalThresholds(GetDefaultThresholds());
}

}  // namespace bud::simd
