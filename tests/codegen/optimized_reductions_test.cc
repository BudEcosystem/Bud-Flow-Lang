// =============================================================================
// Bud Flow Lang - Optimized Reductions Tests (TDD)
// =============================================================================
//
// Tests for high-performance reduction operations using multiple accumulators.
// Based on research showing 2-4x speedup from using 4-8 independent accumulators
// to hide instruction latency.
//
// Reference: https://en.algorithmica.org/hpc/simd/reduction/
//
// =============================================================================

#include "bud_flow_lang/codegen/hwy_ops.h"

#include <hwy/aligned_allocator.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace simd {
namespace {

// =============================================================================
// Test Fixtures and Utilities
// =============================================================================

class OptimizedReductionsTest : public ::testing::Test {
  protected:
    void SetUp() override { rng_.seed(42); }

    hwy::AlignedFreeUniquePtr<float[]> RandomFloats(size_t count, float min = -100.0f,
                                                    float max = 100.0f) {
        auto arr = hwy::AllocateAligned<float>(count);
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    hwy::AlignedFreeUniquePtr<double[]> RandomDoubles(size_t count, double min = -100.0,
                                                      double max = 100.0) {
        auto arr = hwy::AllocateAligned<double>(count);
        std::uniform_real_distribution<double> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    hwy::AlignedFreeUniquePtr<int32_t[]> RandomInts32(size_t count, int32_t min = -1000,
                                                      int32_t max = 1000) {
        auto arr = hwy::AllocateAligned<int32_t>(count);
        std::uniform_int_distribution<int32_t> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    hwy::AlignedFreeUniquePtr<int64_t[]> RandomInts64(size_t count, int64_t min = -1000,
                                                      int64_t max = 1000) {
        auto arr = hwy::AllocateAligned<int64_t>(count);
        std::uniform_int_distribution<int64_t> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    static bool ApproxEqual(float a, float b, float rel_tol = 1e-4f) {
        float diff = std::abs(a - b);
        float max_val = std::max(std::abs(a), std::abs(b));
        return diff <= rel_tol * max_val + 1e-6f;
    }

    static bool ApproxEqual(double a, double b, double rel_tol = 1e-10) {
        double diff = std::abs(a - b);
        double max_val = std::max(std::abs(a), std::abs(b));
        return diff <= rel_tol * max_val + 1e-12;
    }

    // Standard test sizes including larger sizes for performance testing
    static std::vector<size_t> TestSizes() {
        return {1,   7,   8,   15,  16,   31,   32,   63,   64,   127,
                128, 255, 256, 512, 1024, 2048, 4096, 8192, 16384};
    }

    // Large sizes for performance benchmarking
    static std::vector<size_t> LargeSizes() { return {1024, 4096, 16384, 65536, 262144, 1048576}; }

    std::mt19937 rng_;
};

// =============================================================================
// Multi-Accumulator ReduceSum Tests (Float32)
// =============================================================================

TEST_F(OptimizedReductionsTest, ReduceSumFast_Float32_Correctness) {
    // Test that ReduceSumFast produces same results as ReduceSum
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count, -10.0f, 10.0f);

        float result_fast = ReduceSumFast(a.get(), count);
        float result_orig = ReduceSum(a.get(), count);

        // Compute reference with double precision
        double expected = 0.0;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<double>(a[i]);
        }

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<float>(expected), 1e-3f))
            << "ReduceSumFast mismatch for count " << count << ": got " << result_fast
            << ", expected " << expected;

        // Also verify it's close to the original implementation
        // Note: Different accumulator order causes slight precision differences
        ASSERT_TRUE(ApproxEqual(result_fast, result_orig, 1e-3f))
            << "ReduceSumFast differs too much from ReduceSum for count " << count
            << " (fast=" << result_fast << ", orig=" << result_orig << ")";
    }
}

TEST_F(OptimizedReductionsTest, ReduceSumFast_Float32_EdgeCases) {
    // Single element
    {
        float single[] = {42.0f};
        EXPECT_FLOAT_EQ(ReduceSumFast(single, 1), 42.0f);
    }

    // Two elements
    {
        float two[] = {1.0f, 2.0f};
        EXPECT_FLOAT_EQ(ReduceSumFast(two, 2), 3.0f);
    }

    // All zeros
    {
        auto zeros = hwy::AllocateAligned<float>(1024);
        std::fill(zeros.get(), zeros.get() + 1024, 0.0f);
        EXPECT_FLOAT_EQ(ReduceSumFast(zeros.get(), 1024), 0.0f);
    }

    // All ones
    {
        auto ones = hwy::AllocateAligned<float>(1000);
        std::fill(ones.get(), ones.get() + 1000, 1.0f);
        EXPECT_FLOAT_EQ(ReduceSumFast(ones.get(), 1000), 1000.0f);
    }

    // Alternating positive and negative (should sum to 0)
    {
        auto alt = hwy::AllocateAligned<float>(1000);
        for (size_t i = 0; i < 1000; ++i) {
            alt[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }
        EXPECT_FLOAT_EQ(ReduceSumFast(alt.get(), 1000), 0.0f);
    }
}

TEST_F(OptimizedReductionsTest, ReduceSumFast_Float32_LargeArrays) {
    // Test with large arrays to verify performance path is correct
    for (size_t count : LargeSizes()) {
        auto a = RandomFloats(count, -1.0f, 1.0f);  // Small values to avoid precision loss

        float result_fast = ReduceSumFast(a.get(), count);

        // Compute reference with Kahan summation for accuracy
        double sum = 0.0, c = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double y = static_cast<double>(a[i]) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<float>(sum), 1e-3f))
            << "ReduceSumFast mismatch for large count " << count;
    }
}

// =============================================================================
// Multi-Accumulator ReduceSum Tests (Float64)
// =============================================================================

TEST_F(OptimizedReductionsTest, ReduceSumFast_Float64_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomDoubles(count, -10.0, 10.0);

        double result_fast = ReduceSumFast(a.get(), count);
        double result_orig = ReduceSum(a.get(), count);

        // Compute reference
        long double expected = 0.0L;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<long double>(a[i]);
        }

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<double>(expected), 1e-10))
            << "ReduceSumFast Float64 mismatch for count " << count;

        ASSERT_TRUE(ApproxEqual(result_fast, result_orig, 1e-12))
            << "ReduceSumFast Float64 differs from ReduceSum for count " << count;
    }
}

// =============================================================================
// Multi-Accumulator ReduceSum Tests (Int32)
// =============================================================================

TEST_F(OptimizedReductionsTest, ReduceSumFast_Int32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomInts32(count, -100, 100);  // Small values to avoid overflow

        int32_t result_fast = ReduceSumFast(a.get(), count);
        int32_t result_orig = ReduceSum(a.get(), count);

        // Compute reference
        int64_t expected = 0;
        for (size_t i = 0; i < count; ++i) {
            expected += a[i];
        }

        ASSERT_EQ(result_fast, static_cast<int32_t>(expected))
            << "ReduceSumFast Int32 mismatch for count " << count;

        ASSERT_EQ(result_fast, result_orig)
            << "ReduceSumFast Int32 differs from ReduceSum for count " << count;
    }
}

// =============================================================================
// Multi-Accumulator ReduceSum Tests (Int64)
// =============================================================================

TEST_F(OptimizedReductionsTest, ReduceSumFast_Int64_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomInts64(count, -100, 100);

        int64_t result_fast = ReduceSumFast(a.get(), count);
        int64_t result_orig = ReduceSum(a.get(), count);

        // Compute reference (use __int128 if available, otherwise accept small error)
        int64_t expected = 0;
        for (size_t i = 0; i < count; ++i) {
            expected += a[i];
        }

        ASSERT_EQ(result_fast, expected) << "ReduceSumFast Int64 mismatch for count " << count;

        ASSERT_EQ(result_fast, result_orig)
            << "ReduceSumFast Int64 differs from ReduceSum for count " << count;
    }
}

// =============================================================================
// Multi-Accumulator ReduceMin/Max Tests
// =============================================================================

TEST_F(OptimizedReductionsTest, ReduceMinFast_Float32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count);

        float result_fast = ReduceMinFast(a.get(), count);
        float result_orig = ReduceMin(a.get(), count);
        float expected = *std::min_element(a.get(), a.get() + count);

        EXPECT_FLOAT_EQ(result_fast, expected) << "ReduceMinFast mismatch for count " << count;
        EXPECT_FLOAT_EQ(result_fast, result_orig);
    }
}

TEST_F(OptimizedReductionsTest, ReduceMaxFast_Float32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count);

        float result_fast = ReduceMaxFast(a.get(), count);
        float result_orig = ReduceMax(a.get(), count);
        float expected = *std::max_element(a.get(), a.get() + count);

        EXPECT_FLOAT_EQ(result_fast, expected) << "ReduceMaxFast mismatch for count " << count;
        EXPECT_FLOAT_EQ(result_fast, result_orig);
    }
}

// =============================================================================
// Performance Comparison Tests (Benchmarks)
// =============================================================================

TEST_F(OptimizedReductionsTest, ReduceSumFast_Float32_Performance) {
    // This test verifies that ReduceSumFast is at least as fast as ReduceSum
    // In practice, it should be 2-4x faster for large arrays

    constexpr size_t kCount = 1048576;  // 1M elements
    constexpr int kIterations = 100;

    auto a = RandomFloats(kCount, -1.0f, 1.0f);

    // Warm-up
    volatile float sink = 0.0f;
    sink = ReduceSum(a.get(), kCount);
    sink = ReduceSumFast(a.get(), kCount);

    // Benchmark original
    auto start_orig = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        sink = ReduceSum(a.get(), kCount);
    }
    auto end_orig = std::chrono::high_resolution_clock::now();

    // Benchmark fast
    auto start_fast = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        sink = ReduceSumFast(a.get(), kCount);
    }
    auto end_fast = std::chrono::high_resolution_clock::now();

    auto time_orig =
        std::chrono::duration_cast<std::chrono::microseconds>(end_orig - start_orig).count();
    auto time_fast =
        std::chrono::duration_cast<std::chrono::microseconds>(end_fast - start_fast).count();

    double speedup = static_cast<double>(time_orig) / static_cast<double>(time_fast);

    std::cout << "ReduceSum performance (1M elements, " << kIterations << " iterations):\n"
              << "  Original: " << time_orig << " us\n"
              << "  Fast:     " << time_fast << " us\n"
              << "  Speedup:  " << speedup << "x\n";

    // Fast version should not be slower (allow 10% margin for noise)
    EXPECT_GE(speedup, 0.9) << "ReduceSumFast should not be significantly slower";

    // We expect at least 1.5x speedup with multi-accumulators
    // This is a soft expectation - don't fail the test but report
    if (speedup < 1.5) {
        std::cout << "  WARNING: Expected speedup >= 1.5x, got " << speedup << "x\n";
    }

    (void)sink;  // Prevent optimization
}

TEST_F(OptimizedReductionsTest, DotFast_Float32_Correctness) {
    // Dot product is essentially a fused multiply-reduce operation
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count, -10.0f, 10.0f);
        auto b = RandomFloats(count, -10.0f, 10.0f);

        float result_fast = DotFast(a.get(), b.get(), count);
        float result_orig = Dot(a.get(), b.get(), count);

        // Compute reference
        double expected = 0.0;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<float>(expected), 1e-3f))
            << "DotFast mismatch for count " << count;

        ASSERT_TRUE(ApproxEqual(result_fast, result_orig, 1e-5f))
            << "DotFast differs from Dot for count " << count;
    }
}

TEST_F(OptimizedReductionsTest, DotFast_Float64_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomDoubles(count, -10.0, 10.0);
        auto b = RandomDoubles(count, -10.0, 10.0);

        double result_fast = DotFast(a.get(), b.get(), count);
        double result_orig = Dot(a.get(), b.get(), count);

        // Compute reference
        long double expected = 0.0L;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
        }

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<double>(expected), 1e-10))
            << "DotFast Float64 mismatch for count " << count;
    }
}

// =============================================================================
// Mean with Multi-Accumulator Tests
// =============================================================================

TEST_F(OptimizedReductionsTest, MeanFast_Float32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count, -10.0f, 10.0f);

        float result_fast = MeanFast(a.get(), count);

        // Compute reference
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum += static_cast<double>(a[i]);
        }
        float expected = static_cast<float>(sum / count);

        ASSERT_TRUE(ApproxEqual(result_fast, expected, 1e-4f))
            << "MeanFast mismatch for count " << count << ": got " << result_fast << ", expected "
            << expected;
    }
}

// =============================================================================
// Variance with Multi-Accumulator Tests
// =============================================================================

TEST_F(OptimizedReductionsTest, VarianceFast_Float32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count < 2)
            continue;  // Need at least 2 elements for variance

        auto a = RandomFloats(count, -10.0f, 10.0f);

        float result_fast = VarianceFast(a.get(), count);

        // Compute reference (population variance)
        double mean = 0.0;
        for (size_t i = 0; i < count; ++i) {
            mean += a[i];
        }
        mean /= count;

        double var = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double diff = a[i] - mean;
            var += diff * diff;
        }
        var /= count;

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<float>(var), 1e-3f))
            << "VarianceFast mismatch for count " << count << ": got " << result_fast
            << ", expected " << var;
    }
}

// =============================================================================
// Sum of Squares with Multi-Accumulator Tests
// =============================================================================

TEST_F(OptimizedReductionsTest, SumOfSquaresFast_Float32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count, -10.0f, 10.0f);

        float result_fast = SumOfSquaresFast(a.get(), count);

        // Compute reference
        double expected = 0.0;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        }

        ASSERT_TRUE(ApproxEqual(result_fast, static_cast<float>(expected), 1e-3f))
            << "SumOfSquaresFast mismatch for count " << count;
    }
}

// =============================================================================
// L2 Norm (Euclidean Norm) Tests
// =============================================================================

TEST_F(OptimizedReductionsTest, NormL2Fast_Float32_Correctness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count, -10.0f, 10.0f);

        float result_fast = NormL2Fast(a.get(), count);

        // Compute reference
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; ++i) {
            sum_sq += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        }
        float expected = static_cast<float>(std::sqrt(sum_sq));

        ASSERT_TRUE(ApproxEqual(result_fast, expected, 1e-4f))
            << "NormL2Fast mismatch for count " << count;
    }
}

}  // namespace
}  // namespace simd
}  // namespace bud
