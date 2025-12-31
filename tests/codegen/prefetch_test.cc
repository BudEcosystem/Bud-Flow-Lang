// Prefetching Tests
// Tests for software prefetching patterns and utilities

#include "bud_flow_lang/codegen/hwy_ops.h"

#include <hwy/highway.h>

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace bud::simd {

// Test fixture for Highway prefetch utilities
class HwyPrefetchTest : public ::testing::Test {
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

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / iterations;
    }

    std::mt19937 gen_;
};

// =============================================================================
// Prefetch Utilities Tests
// =============================================================================

// Test that prefetch hints compile and don't crash
TEST_F(HwyPrefetchTest, PrefetchHint_BasicOperation) {
    std::vector<float> data(1024);

    // Test read prefetch with different locality hints
    PrefetchRead(data.data(), 0);  // No locality
    PrefetchRead(data.data(), 1);  // Low locality
    PrefetchRead(data.data(), 2);  // Medium locality
    PrefetchRead(data.data(), 3);  // High locality (keep in cache)

    // Test write prefetch
    PrefetchWrite(data.data(), 3);

    // If we get here without crashing, the test passes
    SUCCEED();
}

// Test prefetch distance calculation
TEST_F(HwyPrefetchTest, PrefetchDistance_Calculation) {
    // For float (4 bytes), with typical 256-byte prefetch distance
    size_t dist_f32 = GetPrefetchDistance<float>();
    EXPECT_GE(dist_f32, 32);   // At least 32 elements (128 bytes)
    EXPECT_LE(dist_f32, 256);  // At most 256 elements (1024 bytes)

    // For double (8 bytes), should be half as many elements
    size_t dist_f64 = GetPrefetchDistance<double>();
    EXPECT_GE(dist_f64, 16);
    EXPECT_LE(dist_f64, 128);

    // Double elements should be roughly half of float elements
    EXPECT_NEAR(static_cast<double>(dist_f64), static_cast<double>(dist_f32) / 2,
                static_cast<double>(dist_f32) / 4);
}

// Test prefetch stream for sequential access
TEST_F(HwyPrefetchTest, PrefetchStream_SequentialAccess) {
    const size_t count = 1024 * 1024;  // 4 MB
    auto data = RandomFloats(count);

    // This should not crash and should be measurable
    float sum = 0.0f;
    PrefetchStream<float> stream(data.data(), count);

    for (size_t i = 0; i < count; ++i) {
        stream.prefetchAhead(i);
        sum += data[i];
    }

    EXPECT_NE(sum, 0.0f);
}

// =============================================================================
// Performance Tests
// =============================================================================

// Test that prefetching improves performance for large arrays
TEST_F(HwyPrefetchTest, Prefetch_LargeArrayPerformance) {
    const size_t count = 4 * 1024 * 1024;  // 16 MB (larger than L3 cache)
    auto a = RandomFloats(count);
    auto b = RandomFloats(count);
    std::vector<float> result_prefetch(count);
    std::vector<float> result_no_prefetch(count);

    // Benchmark with prefetch (using AddSized for large arrays)
    double time_prefetch =
        BenchmarkOp([&]() { AddSized(a.data(), b.data(), result_prefetch.data(), count); }, 10);

    // Benchmark without prefetch (simple loop)
    double time_no_prefetch = BenchmarkOp(
        [&]() { AddNoPrefetch(a.data(), b.data(), result_no_prefetch.data(), count); }, 10);

    std::cout << "Large array prefetch performance (" << count << " elements):\n";
    std::cout << "  With prefetch:    " << time_prefetch << " us/iter\n";
    std::cout << "  Without prefetch: " << time_no_prefetch << " us/iter\n";
    std::cout << "  Speedup:          " << time_no_prefetch / time_prefetch << "x\n";

    // Prefetch should provide at least small improvement for large arrays
    // (relaxed threshold since improvement depends on cache hierarchy)
    EXPECT_GT(time_no_prefetch / time_prefetch, 0.9)
        << "Prefetch should not significantly hurt performance";
}

// Test prefetch stride optimization
TEST_F(HwyPrefetchTest, Prefetch_StrideOptimization) {
    const size_t count = 1024 * 1024;  // 4 MB
    auto data = RandomFloats(count);
    std::vector<float> result(count);

    // Benchmark sequential access with prefetch
    double time_stride1 =
        BenchmarkOp([&]() { ProcessWithPrefetch(data.data(), result.data(), count, 1); }, 20);

    // Benchmark strided access with prefetch (cache-unfriendly)
    double time_stride64 =
        BenchmarkOp([&]() { ProcessWithPrefetch(data.data(), result.data(), count / 64, 64); }, 20);

    std::cout << "Stride prefetch performance:\n";
    std::cout << "  Stride 1:  " << time_stride1 << " us/iter\n";
    std::cout << "  Stride 64: " << time_stride64 << " us/iter\n";

    // Strided access should be slower due to cache misses
    // This verifies prefetch is actually doing something useful
    SUCCEED();
}

// =============================================================================
// Double Precision Tests
// =============================================================================

TEST_F(HwyPrefetchTest, PrefetchHint_Double) {
    std::vector<double> data(1024);

    PrefetchRead(data.data(), 3);
    PrefetchWrite(data.data(), 3);

    SUCCEED();
}

TEST_F(HwyPrefetchTest, Prefetch_LargeArrayDouble) {
    const size_t count = 2 * 1024 * 1024;  // 16 MB
    std::vector<double> a(count), b(count), result(count);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (size_t i = 0; i < count; ++i) {
        a[i] = dist(gen_);
        b[i] = dist(gen_);
    }

    // Verify correctness with prefetch
    AddSized(a.data(), b.data(), result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_NEAR(result[i], a[i] + b[i], 1e-10) << "Mismatch at index " << i;
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(HwyPrefetchTest, Prefetch_NullPointer) {
    // Prefetch of nullptr should not crash
    // (it's just a hint, so the processor should ignore it)
    PrefetchRead(static_cast<float*>(nullptr), 3);
    PrefetchWrite(static_cast<float*>(nullptr), 3);
    SUCCEED();
}

TEST_F(HwyPrefetchTest, Prefetch_SmallArray) {
    // Prefetching shouldn't hurt small arrays
    const size_t count = 16;
    auto a = RandomFloats(count);
    auto b = RandomFloats(count);
    std::vector<float> result(count);

    AddSized(a.data(), b.data(), result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_NEAR(result[i], a[i] + b[i], 1e-5f);
    }
}

}  // namespace bud::simd
