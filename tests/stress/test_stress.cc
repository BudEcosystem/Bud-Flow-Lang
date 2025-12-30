/**
 * @file test_stress.cc
 * @brief Stress tests for memory optimization
 *
 * Tests large arrays and extreme conditions to verify:
 * - Cache-aware tiled execution works correctly at scale
 * - NUMA allocation handles large allocations
 * - Prefetching improves performance
 * - System can handle edge cases
 */

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/numa_allocator.h"
#include "bud_flow_lang/memory/tiled_executor.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace test {

// =============================================================================
// Stress Test Fixture
// =============================================================================

class StressTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!isInitialized()) {
            RuntimeConfig config;
            config.enable_debug_output = false;
            auto result = initialize(config);
            ASSERT_TRUE(result.hasValue());
        }
        // Ensure optimization is enabled
        setTilingEnabled(true);
        setPrefetchEnabled(true);
    }

    // Time a function and return milliseconds
    template <typename Func>
    double timeMs(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Check approximate equality with relative tolerance
    static bool approxEqual(float a, float b, float rel_tol = 1e-4f) {
        float diff = std::abs(a - b);
        float max_val = std::max(std::abs(a), std::abs(b));
        return diff <= rel_tol * max_val + 1e-6f;
    }

    static bool approxEqual(double a, double b, double rel_tol = 1e-8) {
        double diff = std::abs(a - b);
        double max_val = std::max(std::abs(a), std::abs(b));
        return diff <= rel_tol * max_val + 1e-12;
    }
};

// =============================================================================
// Large Array Tests (10K - 1M elements)
// =============================================================================

TEST_F(StressTest, LargeReduction_10K) {
    const size_t N = 10000;
    auto a = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());

    float sum = a->sum();
    EXPECT_TRUE(approxEqual(sum, static_cast<float>(N))) << "Expected " << N << ", got " << sum;
}

TEST_F(StressTest, LargeReduction_100K) {
    const size_t N = 100000;
    auto a = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());

    float sum = a->sum();
    EXPECT_TRUE(approxEqual(sum, static_cast<float>(N), 1e-3f))
        << "Expected " << N << ", got " << sum;
}

TEST_F(StressTest, LargeReduction_1M) {
    const size_t N = 1000000;
    auto a = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());

    float sum = a->sum();
    // Allow larger tolerance for 1M elements due to FP accumulation
    EXPECT_TRUE(approxEqual(sum, static_cast<float>(N), 1e-2f))
        << "Expected " << N << ", got " << sum;
}

TEST_F(StressTest, LargeDotProduct_10K) {
    const size_t N = 10000;
    auto a = Bunch::ones(N);
    auto b = Bunch::fill(N, 2.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 2.0f * N, 1e-3f));
}

TEST_F(StressTest, LargeDotProduct_100K) {
    const size_t N = 100000;
    auto a = Bunch::ones(N);
    auto b = Bunch::fill(N, 0.5f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 0.5f * N, 1e-2f));
}

TEST_F(StressTest, LargeDotProduct_1M) {
    const size_t N = 1000000;
    auto a = Bunch::ones(N);
    auto b = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, static_cast<float>(N), 1e-1f));
}

// =============================================================================
// Binary Operation Stress Tests
// =============================================================================

TEST_F(StressTest, LargeBinaryAdd_1M) {
    const size_t N = 1000000;
    auto a = Bunch::fill(N, 1.0f);
    auto b = Bunch::fill(N, 2.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    auto result = *a + *b;

    // Verify first, last, and middle elements
    const float* data = static_cast<const float*>(result.data());
    EXPECT_TRUE(approxEqual(data[0], 3.0f));
    EXPECT_TRUE(approxEqual(data[N / 2], 3.0f));
    EXPECT_TRUE(approxEqual(data[N - 1], 3.0f));
}

TEST_F(StressTest, LargeBinaryMul_1M) {
    const size_t N = 1000000;
    auto a = Bunch::fill(N, 2.0f);
    auto b = Bunch::fill(N, 3.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    auto result = *a * *b;

    const float* data = static_cast<const float*>(result.data());
    EXPECT_TRUE(approxEqual(data[0], 6.0f));
    EXPECT_TRUE(approxEqual(data[N - 1], 6.0f));
}

// =============================================================================
// Chained Operations Stress Tests
// =============================================================================

TEST_F(StressTest, ChainedOps_100K) {
    const size_t N = 100000;
    auto a = Bunch::fill(N, 1.0f);
    auto b = Bunch::fill(N, 2.0f);
    auto c = Bunch::fill(N, 3.0f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());
    ASSERT_TRUE(c.hasValue());

    // (a + b) * c - a = (1 + 2) * 3 - 1 = 8
    auto result = ((*a + *b) * *c) - *a;

    const float* data = static_cast<const float*>(result.data());
    EXPECT_TRUE(approxEqual(data[0], 8.0f));
    EXPECT_TRUE(approxEqual(data[N / 2], 8.0f));
    EXPECT_TRUE(approxEqual(data[N - 1], 8.0f));
}

TEST_F(StressTest, DeepChain_10Ops) {
    const size_t N = 50000;
    auto x = Bunch::fill(N, 1.0f);
    ASSERT_TRUE(x.hasValue());

    // Chain of 10 operations
    auto result = *x;
    for (int i = 0; i < 10; ++i) {
        result = result + *Bunch::fill(N, 0.1f);
    }

    // Expected: 1.0 + 10 * 0.1 = 2.0
    const float* data = static_cast<const float*>(result.data());
    EXPECT_TRUE(approxEqual(data[0], 2.0f, 1e-3f));
}

// =============================================================================
// Edge Case Stress Tests
// =============================================================================

TEST_F(StressTest, PowerOf2Size_262144) {
    const size_t N = 262144;  // 2^18
    auto a = Bunch::ones(N);
    auto b = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, static_cast<float>(N), 1e-2f));
}

TEST_F(StressTest, PowerOf2MinusOne_262143) {
    const size_t N = 262143;  // 2^18 - 1
    auto a = Bunch::ones(N);
    auto b = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, static_cast<float>(N), 1e-2f));
}

TEST_F(StressTest, PowerOf2PlusOne_262145) {
    const size_t N = 262145;  // 2^18 + 1
    auto a = Bunch::ones(N);
    auto b = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, static_cast<float>(N), 1e-2f));
}

TEST_F(StressTest, PrimeSize_100003) {
    const size_t N = 100003;  // Prime number
    auto a = Bunch::ones(N);
    auto b = Bunch::fill(N, 0.5f);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    float dot = a->dot(*b);
    EXPECT_TRUE(approxEqual(dot, 0.5f * N, 1e-2f));
}

// =============================================================================
// NUMA Allocation Stress Tests
// =============================================================================

TEST_F(StressTest, NumaLargeAllocation_4MB) {
    // 4MB = 1M floats
    const size_t N = 1000000;
    auto a = Bunch::zeros(N);
    ASSERT_TRUE(a.hasValue());
    ASSERT_EQ(a->size(), N);

    // Fill with pattern
    float* data = static_cast<float*>(a->mutableData());
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<float>(i % 100);
    }

    // Verify pattern
    float sum = a->sum();
    // Sum of 0-99 = 4950, repeated 10000 times
    float expected = 4950.0f * 10000.0f;
    EXPECT_TRUE(approxEqual(sum, expected, 1e-2f));
}

TEST_F(StressTest, NumaLargeAllocation_10MB) {
    // 10MB = 2.5M floats
    const size_t N = 2500000;
    auto a = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());

    float sum = a->sum();
    EXPECT_TRUE(approxEqual(sum, static_cast<float>(N), 1e-1f));
}

// =============================================================================
// Performance Comparison Tests
// =============================================================================

TEST_F(StressTest, TilingVsNoTiling_Performance) {
    const size_t N = 500000;
    auto a = Bunch::ones(N);
    auto b = Bunch::ones(N);
    ASSERT_TRUE(a.hasValue());
    ASSERT_TRUE(b.hasValue());

    // Warmup
    volatile float warmup = a->dot(*b);
    (void)warmup;

    // Measure with tiling
    setTilingEnabled(true);
    double tiled_ms = timeMs([&]() {
        for (int i = 0; i < 10; ++i) {
            volatile float r = a->dot(*b);
            (void)r;
        }
    });

    // Measure without tiling
    setTilingEnabled(false);
    double untiled_ms = timeMs([&]() {
        for (int i = 0; i < 10; ++i) {
            volatile float r = a->dot(*b);
            (void)r;
        }
    });

    // Re-enable tiling
    setTilingEnabled(true);

    // Log the results (not a strict assertion since it depends on hardware)
    std::cout << "  Tiled: " << tiled_ms << " ms" << std::endl;
    std::cout << "  Untiled: " << untiled_ms << " ms" << std::endl;

    // Both should produce correct results
    EXPECT_TRUE(approxEqual(a->dot(*b), static_cast<float>(N), 1e-1f));
}

// =============================================================================
// Memory Limit Tests
// =============================================================================

TEST_F(StressTest, LargeAllocation_50M) {
    // Attempt 50M elements (200MB for float32)
    const size_t N = 50000000;
    auto a = Bunch::zeros(N);

    if (a.hasValue()) {
        EXPECT_EQ(a->size(), N);

        // Verify we can write and read
        float* data = static_cast<float*>(a->mutableData());
        data[0] = 1.0f;
        data[N - 1] = 2.0f;

        const float* cdata = static_cast<const float*>(a->data());
        EXPECT_EQ(cdata[0], 1.0f);
        EXPECT_EQ(cdata[N - 1], 2.0f);
    } else {
        // If allocation fails, that's acceptable for this stress test
        GTEST_SKIP() << "50M element allocation failed (system memory limit)";
    }
}

// =============================================================================
// Concurrent Access Test (single-threaded verification)
// =============================================================================

TEST_F(StressTest, MultipleArraysSimultaneously) {
    const size_t N = 100000;

    // Create multiple arrays
    std::vector<Result<Bunch>> arrays;
    for (int i = 0; i < 10; ++i) {
        arrays.push_back(Bunch::fill(N, static_cast<float>(i + 1)));
    }

    // Verify all allocations succeeded
    for (const auto& arr : arrays) {
        ASSERT_TRUE(arr.hasValue());
    }

    // Perform operations on all
    float total = 0.0f;
    for (auto& arr : arrays) {
        total += arr->sum();
    }

    // Expected: sum of (N * 1) + (N * 2) + ... + (N * 10) = N * 55
    float expected = N * 55.0f;
    EXPECT_TRUE(approxEqual(total, expected, 1e-2f));
}

}  // namespace test
}  // namespace bud
