/**
 * @file test_simple.cc
 * @brief Simple integration tests for memory optimization
 *
 * Tests basic operations with small to medium array sizes to verify
 * correctness of the memory optimization integration.
 */

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/numa_allocator.h"
#include "bud_flow_lang/memory/prefetch.h"
#include "bud_flow_lang/memory/tiled_executor.h"

#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace test {

// =============================================================================
// Simple Correctness Tests
// =============================================================================

class SimpleMemoryOptimizationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize runtime if not already done
        if (!isInitialized()) {
            RuntimeConfig config;
            config.enable_debug_output = false;
            auto result = initialize(config);
            ASSERT_TRUE(result.hasValue()) << "Failed to initialize runtime";
        }
    }

    // Helper to check approximate equality
    static bool approxEqual(float a, float b, float rel_tol = 1e-5f) {
        float diff = std::abs(a - b);
        float max_val = std::max(std::abs(a), std::abs(b));
        return diff <= rel_tol * max_val + 1e-8f;
    }
};

// Test 1: Small addition (10 elements)
TEST_F(SimpleMemoryOptimizationTest, SmallAddition_10) {
    auto a = Bunch::arange(10, 1.0f, 1.0f);
    auto b = Bunch::arange(10, 0.0f, 1.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    auto result = *a + *b;
    ASSERT_EQ(result.size(), 10);

    // Expected: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    const float* data = static_cast<const float*>(result.data());
    for (size_t i = 0; i < 10; ++i) {
        float expected = 2 * i + 1;
        EXPECT_TRUE(approxEqual(data[i], expected))
            << "Mismatch at index " << i << ": got " << data[i] << ", expected " << expected;
    }
}

// Test 2: Small multiplication (100 elements)
TEST_F(SimpleMemoryOptimizationTest, SmallMultiplication_100) {
    auto a = Bunch::arange(100, 1.0f, 0.5f);
    auto b = Bunch::fill(100, 2.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    auto result = *a * *b;
    ASSERT_EQ(result.size(), 100);

    const float* data = static_cast<const float*>(result.data());
    for (size_t i = 0; i < 100; ++i) {
        float expected = 2.0f * (1.0f + 0.5f * i);
        EXPECT_TRUE(approxEqual(data[i], expected)) << "Mismatch at index " << i;
    }
}

// Test 3: Medium reduction (1000 elements)
TEST_F(SimpleMemoryOptimizationTest, SmallReduction_1000) {
    auto a = Bunch::ones(1000);
    ASSERT_TRUE(a.hasValue());

    float sum = a->sum();
    EXPECT_TRUE(approxEqual(sum, 1000.0f)) << "Sum expected 1000, got " << sum;
}

// Test 4: Small dot product (1000 elements) - crosses tiling threshold
TEST_F(SimpleMemoryOptimizationTest, SmallDotProduct_1000) {
    auto a = Bunch::ones(1000);
    auto b = Bunch::fill(1000, 2.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 2000.0f)) << "Dot product expected 2000, got " << dot;
}

// Test 5: Small FMA (1000 elements)
TEST_F(SimpleMemoryOptimizationTest, SmallFMA_1000) {
    auto a = Bunch::fill(1000, 2.0f);
    auto b = Bunch::fill(1000, 3.0f);
    auto c = Bunch::fill(1000, 1.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());
    ASSERT_TRUE(c.hasValue());

    auto result = fma(*a, *b, *c);  // 2 * 3 + 1 = 7
    ASSERT_EQ(result.size(), 1000);

    const float* data = static_cast<const float*>(result.data());
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_TRUE(approxEqual(data[i], 7.0f)) << "FMA mismatch at index " << i;
    }
}

// Test 6: Unary operations chain (1000 elements)
TEST_F(SimpleMemoryOptimizationTest, UnaryChain_1000) {
    auto a = Bunch::fill(1000, 4.0f);
    ASSERT_TRUE(a.hasValue());

    auto result = a->sqrt().sqrt();  // sqrt(sqrt(4)) = sqrt(2) ~ 1.414
    ASSERT_EQ(result.size(), 1000);

    const float* data = static_cast<const float*>(result.data());
    float expected = std::sqrt(std::sqrt(4.0f));
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_TRUE(approxEqual(data[i], expected, 1e-4f)) << "Mismatch at index " << i;
    }
}

// Test 7: At tiling threshold (1024 elements)
TEST_F(SimpleMemoryOptimizationTest, AtTilingThreshold_1024) {
    auto a = Bunch::arange(1024, 0.0f, 1.0f);
    auto b = Bunch::ones(1024);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    // Test dot product at exactly the tiling threshold
    float dot = a->dot(*b);
    float expected = 1024.0f * 1023.0f / 2.0f;  // Sum of 0 to 1023
    EXPECT_TRUE(approxEqual(dot, expected, 1e-3f))
        << "Dot at threshold expected " << expected << ", got " << dot;
}

// Test 8: Just above tiling threshold (2048 elements)
TEST_F(SimpleMemoryOptimizationTest, AboveTilingThreshold_2048) {
    auto a = Bunch::ones(2048);
    auto b = Bunch::fill(2048, 0.5f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    auto result = *a + *b;
    float sum = result.sum();
    EXPECT_TRUE(approxEqual(sum, 2048.0f * 1.5f))
        << "Sum expected " << 2048.0f * 1.5f << ", got " << sum;
}

// =============================================================================
// Memory Optimization Control Tests
// =============================================================================

TEST_F(SimpleMemoryOptimizationTest, TilingCanBeDisabled) {
    // Disable tiling
    setTilingEnabled(false);
    EXPECT_FALSE(isTilingEnabled());

    // Operations should still work correctly
    auto a = Bunch::ones(2000);
    auto b = Bunch::ones(2000);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 2000.0f));

    // Re-enable
    setTilingEnabled(true);
    EXPECT_TRUE(isTilingEnabled());
}

TEST_F(SimpleMemoryOptimizationTest, PrefetchCanBeDisabled) {
    // Disable prefetching
    setPrefetchEnabled(false);
    EXPECT_FALSE(isPrefetchEnabled());

    // Operations should still work correctly
    auto a = Bunch::ones(2000);
    ASSERT_TRUE(a.hasValue());

    float sum = a->sum();
    EXPECT_TRUE(approxEqual(sum, 2000.0f));

    // Re-enable
    setPrefetchEnabled(true);
    EXPECT_TRUE(isPrefetchEnabled());
}

TEST_F(SimpleMemoryOptimizationTest, CacheConfigIsValid) {
    auto* config = getCacheConfig();
    ASSERT_NE(config, nullptr);

    // Cache sizes should be reasonable
    EXPECT_GT(config->l1Size(), 0);
    EXPECT_GT(config->l2Size(), config->l1Size());
    EXPECT_GE(config->l3Size(), config->l2Size());
    EXPECT_GT(config->lineSize(), 0);
    EXPECT_EQ(config->lineSize() % 8, 0);  // Should be aligned
}

// =============================================================================
// Edge Cases Near Thresholds
// =============================================================================

TEST_F(SimpleMemoryOptimizationTest, JustBelowThreshold_1023) {
    auto a = Bunch::ones(1023);
    auto b = Bunch::ones(1023);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 1023.0f));
}

TEST_F(SimpleMemoryOptimizationTest, JustAboveThreshold_1025) {
    auto a = Bunch::ones(1025);
    auto b = Bunch::ones(1025);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 1025.0f));
}

// =============================================================================
// Data Type Tests
// =============================================================================

TEST_F(SimpleMemoryOptimizationTest, Float32Operations) {
    auto a = Bunch::zeros(100, ScalarType::kFloat32);
    ASSERT_TRUE(a.hasValue());
    EXPECT_EQ(a->dtype(), ScalarType::kFloat32);
}

// =============================================================================
// Chained Operations
// =============================================================================

TEST_F(SimpleMemoryOptimizationTest, ChainedBinaryOps) {
    auto a = Bunch::ones(1000);
    auto b = Bunch::fill(1000, 2.0f);
    auto c = Bunch::fill(1000, 3.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());
    ASSERT_TRUE(c.hasValue());

    // (a + b) * c = (1 + 2) * 3 = 9
    auto result = (*a + *b) * *c;

    const float* data = static_cast<const float*>(result.data());
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_TRUE(approxEqual(data[i], 9.0f)) << "Chain mismatch at index " << i;
    }
}

TEST_F(SimpleMemoryOptimizationTest, ChainedUnaryOps) {
    auto a = Bunch::fill(1000, -4.0f);
    ASSERT_TRUE(a.hasValue());

    // abs(neg(-4)) = abs(4) = 4
    auto result = (-(*a)).abs();

    const float* data = static_cast<const float*>(result.data());
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_TRUE(approxEqual(data[i], 4.0f));
    }
}

}  // namespace test
}  // namespace bud
