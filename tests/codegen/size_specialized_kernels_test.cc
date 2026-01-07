// Copyright 2024 The Bud Flow Lang Authors. All Rights Reserved.
//
// Tests for size-specialized kernels that optimize for different array sizes:
// - Small (<64): fully unrolled, minimal overhead
// - Medium (64-4096): 4x unrolled loop
// - Large (>4096): 8x unrolled + prefetching
//
// TDD: Write tests first, verify they fail, then implement.

#include "bud_flow_lang/codegen/hwy_ops.h"

#include "hwy/aligned_allocator.h"

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

// Size tier thresholds
constexpr size_t kSmallThreshold = 64;
constexpr size_t kMediumThreshold = 4096;

class SizeSpecializedKernelsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Seed for reproducibility
        rng_.seed(42);
    }

    std::vector<float> GenerateRandomFloats(size_t count) {
        std::vector<float> data(count);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(rng_);
        }
        return data;
    }

    std::vector<double> GenerateRandomDoubles(size_t count) {
        std::vector<double> data(count);
        std::uniform_real_distribution<double> dist(-100.0, 100.0);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(rng_);
        }
        return data;
    }

    std::vector<int32_t> GenerateRandomInt32s(size_t count) {
        std::vector<int32_t> data(count);
        std::uniform_int_distribution<int32_t> dist(-1000, 1000);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(rng_);
        }
        return data;
    }

    // Reference implementations for validation
    void AddReference(const float* a, const float* b, float* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    void MulReference(const float* a, const float* b, float* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    void FmaReference(const float* a, const float* b, const float* c, float* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    }

    float ReduceSumReference(const float* a, size_t count) {
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum += a[i];
        }
        return sum;
    }

    bool ApproxEqual(float a, float b, float rel_tol = 1e-3f) {
        if (std::isinf(a) || std::isinf(b))
            return a == b;
        if (std::isnan(a) || std::isnan(b))
            return false;
        float max_val = std::max(std::abs(a), std::abs(b));
        if (max_val < 1e-6f)
            return true;  // Both near zero
        return std::abs(a - b) / max_val < rel_tol;
    }

    bool ArraysApproxEqual(const float* a, const float* b, size_t count, float rel_tol = 1e-5f) {
        for (size_t i = 0; i < count; ++i) {
            if (!ApproxEqual(a[i], b[i], rel_tol)) {
                return false;
            }
        }
        return true;
    }

    std::mt19937 rng_;
};

// =============================================================================
// AddSized Tests - Size-specialized addition
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, AddSized_SmallArrays_Correctness) {
    // Test small arrays (< 64 elements) - should use fully unrolled version
    for (size_t count : {1, 2, 4, 8, 16, 32, 63}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        AddReference(a.data(), b.data(), expected.data(), count);
        AddSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "AddSized failed for small array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, AddSized_MediumArrays_Correctness) {
    // Test medium arrays (64-4096 elements) - should use 4x unrolled version
    for (size_t count : {64, 128, 256, 512, 1024, 2048, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        AddReference(a.data(), b.data(), expected.data(), count);
        AddSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "AddSized failed for medium array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, AddSized_LargeArrays_Correctness) {
    // Test large arrays (> 4096 elements) - should use 8x unrolled + prefetch
    for (size_t count : {4097, 8192, 16384, 65536, 262144}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        AddReference(a.data(), b.data(), expected.data(), count);
        AddSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "AddSized failed for large array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, AddSized_TierBoundaries) {
    // Test exact tier boundaries
    for (size_t count : {63, 64, 65, 4095, 4096, 4097}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        AddReference(a.data(), b.data(), expected.data(), count);
        AddSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "AddSized failed at tier boundary size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, AddSized_EdgeCases) {
    // Empty array
    std::vector<float> empty;
    AddSized(empty.data(), empty.data(), empty.data(), 0);  // Should not crash

    // Single element
    float a = 3.5f, b = 2.5f, result = 0.0f;
    AddSized(&a, &b, &result, 1);
    EXPECT_FLOAT_EQ(result, 6.0f);
}

// =============================================================================
// MulSized Tests - Size-specialized multiplication
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, MulSized_SmallArrays_Correctness) {
    for (size_t count : {1, 2, 4, 8, 16, 32, 63}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        MulReference(a.data(), b.data(), expected.data(), count);
        MulSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "MulSized failed for small array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, MulSized_MediumArrays_Correctness) {
    for (size_t count : {64, 128, 256, 512, 1024, 2048, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        MulReference(a.data(), b.data(), expected.data(), count);
        MulSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "MulSized failed for medium array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, MulSized_LargeArrays_Correctness) {
    for (size_t count : {4097, 8192, 16384, 65536}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        MulReference(a.data(), b.data(), expected.data(), count);
        MulSized(a.data(), b.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "MulSized failed for large array size " << count;
    }
}

// =============================================================================
// FmaSized Tests - Size-specialized fused multiply-add
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, FmaSized_SmallArrays_Correctness) {
    for (size_t count : {1, 2, 4, 8, 16, 32, 63}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        FmaReference(a.data(), b.data(), c.data(), expected.data(), count);
        FmaSized(a.data(), b.data(), c.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "FmaSized failed for small array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, FmaSized_MediumArrays_Correctness) {
    for (size_t count : {64, 128, 256, 512, 1024, 2048, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        FmaReference(a.data(), b.data(), c.data(), expected.data(), count);
        FmaSized(a.data(), b.data(), c.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "FmaSized failed for medium array size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, FmaSized_LargeArrays_Correctness) {
    for (size_t count : {4097, 8192, 16384, 65536}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        FmaReference(a.data(), b.data(), c.data(), expected.data(), count);
        FmaSized(a.data(), b.data(), c.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "FmaSized failed for large array size " << count;
    }
}

// =============================================================================
// ReduceSumSized Tests - Size-specialized reduction
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, ReduceSumSized_SmallArrays_Correctness) {
    for (size_t count : {1, 2, 4, 8, 16, 32, 63}) {
        auto a = GenerateRandomFloats(count);
        float expected = ReduceSumReference(a.data(), count);
        float result = ReduceSumSized(a.data(), count);

        ASSERT_TRUE(ApproxEqual(result, expected))
            << "ReduceSumSized failed for small array size " << count << " expected=" << expected
            << " got=" << result;
    }
}

TEST_F(SizeSpecializedKernelsTest, ReduceSumSized_MediumArrays_Correctness) {
    for (size_t count : {64, 128, 256, 512, 1024, 2048, 4096}) {
        auto a = GenerateRandomFloats(count);
        float expected = ReduceSumReference(a.data(), count);
        float result = ReduceSumSized(a.data(), count);

        ASSERT_TRUE(ApproxEqual(result, expected))
            << "ReduceSumSized failed for medium array size " << count << " expected=" << expected
            << " got=" << result;
    }
}

TEST_F(SizeSpecializedKernelsTest, ReduceSumSized_LargeArrays_Correctness) {
    for (size_t count : {4097, 8192, 16384, 65536, 262144}) {
        auto a = GenerateRandomFloats(count);
        float expected = ReduceSumReference(a.data(), count);
        float result = ReduceSumSized(a.data(), count);

        ASSERT_TRUE(ApproxEqual(result, expected))
            << "ReduceSumSized failed for large array size " << count << " expected=" << expected
            << " got=" << result;
    }
}

// =============================================================================
// Performance Tests - Verify size specialization provides speedup
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, AddSized_Performance_SmallArrays) {
    constexpr size_t kCount = 32;  // Small tier
    constexpr int kIterations = 100000;

    auto a = GenerateRandomFloats(kCount);
    auto b = GenerateRandomFloats(kCount);
    std::vector<float> result(kCount);

    // Warm up
    for (int i = 0; i < 1000; ++i) {
        AddSized(a.data(), b.data(), result.data(), kCount);
    }

    // Benchmark AddSized (size-specialized)
    auto start_sized = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        AddSized(a.data(), b.data(), result.data(), kCount);
    }
    auto end_sized = std::chrono::high_resolution_clock::now();

    // Benchmark standard Add
    auto start_std = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        Add(a.data(), b.data(), result.data(), kCount);
    }
    auto end_std = std::chrono::high_resolution_clock::now();

    double time_sized = std::chrono::duration<double, std::micro>(end_sized - start_sized).count();
    double time_std = std::chrono::duration<double, std::micro>(end_std - start_std).count();
    double speedup = time_std / time_sized;

    std::cout << "Small array (" << kCount << " elements) Add performance:\n"
              << "  Standard: " << time_std / kIterations << " us/iter\n"
              << "  Sized:    " << time_sized / kIterations << " us/iter\n"
              << "  Speedup:  " << speedup << "x\n";

    // Size-specialized should not be dramatically slower (allow for dispatch overhead)
    // Performance varies by platform, so we just verify no major regression
    EXPECT_GE(speedup, 0.5) << "AddSized should not be dramatically slower than Add";
}

TEST_F(SizeSpecializedKernelsTest, AddSized_Performance_LargeArrays) {
    constexpr size_t kCount = 262144;  // 256K elements - large tier
    constexpr int kIterations = 1000;

    auto a = GenerateRandomFloats(kCount);
    auto b = GenerateRandomFloats(kCount);
    std::vector<float> result(kCount);

    // Warm up
    for (int i = 0; i < 10; ++i) {
        AddSized(a.data(), b.data(), result.data(), kCount);
    }

    // Benchmark AddSized (size-specialized with prefetching)
    auto start_sized = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        AddSized(a.data(), b.data(), result.data(), kCount);
    }
    auto end_sized = std::chrono::high_resolution_clock::now();

    // Benchmark standard Add
    auto start_std = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        Add(a.data(), b.data(), result.data(), kCount);
    }
    auto end_std = std::chrono::high_resolution_clock::now();

    double time_sized = std::chrono::duration<double, std::micro>(end_sized - start_sized).count();
    double time_std = std::chrono::duration<double, std::micro>(end_std - start_std).count();
    double speedup = time_std / time_sized;

    std::cout << "Large array (" << kCount << " elements) Add performance:\n"
              << "  Standard: " << time_std / kIterations << " us/iter\n"
              << "  Sized:    " << time_sized / kIterations << " us/iter\n"
              << "  Speedup:  " << speedup << "x\n";

    // Performance varies by platform and cache behavior
    // We just ensure it's not dramatically slower (prefetching benefits vary)
    EXPECT_GE(speedup, 0.2)
        << "AddSized should not be dramatically slower than Add for large arrays";
}

TEST_F(SizeSpecializedKernelsTest, ReduceSumSized_Performance_LargeArrays) {
    constexpr size_t kCount = 1048576;  // 1M elements
    constexpr int kIterations = 100;

    auto a = GenerateRandomFloats(kCount);

    // Warm up
    volatile float sink = 0.0f;
    for (int i = 0; i < 10; ++i) {
        sink = ReduceSumSized(a.data(), kCount);
    }

    // Benchmark ReduceSumSized (size-specialized)
    auto start_sized = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        sink = ReduceSumSized(a.data(), kCount);
    }
    auto end_sized = std::chrono::high_resolution_clock::now();

    // Benchmark standard ReduceSum
    auto start_std = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        sink = ReduceSum(a.data(), kCount);
    }
    auto end_std = std::chrono::high_resolution_clock::now();

    // Benchmark ReduceSumFast (multi-accumulator)
    auto start_fast = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        sink = ReduceSumFast(a.data(), kCount);
    }
    auto end_fast = std::chrono::high_resolution_clock::now();

    double time_sized = std::chrono::duration<double, std::micro>(end_sized - start_sized).count();
    double time_std = std::chrono::duration<double, std::micro>(end_std - start_std).count();
    double time_fast = std::chrono::duration<double, std::micro>(end_fast - start_fast).count();

    std::cout << "Large array (" << kCount << " elements) ReduceSum performance:\n"
              << "  Standard:    " << time_std / kIterations << " us/iter\n"
              << "  Fast (4acc): " << time_fast / kIterations << " us/iter\n"
              << "  Sized:       " << time_sized / kIterations << " us/iter\n"
              << "  Speedup vs std:  " << time_std / time_sized << "x\n"
              << "  Speedup vs fast: " << time_fast / time_sized << "x\n";

    (void)sink;  // Prevent unused warning
}

// =============================================================================
// Double precision tests
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, AddSizedDouble_AllTiers) {
    for (size_t count : {16, 256, 8192}) {
        auto a = GenerateRandomDoubles(count);
        auto b = GenerateRandomDoubles(count);
        std::vector<double> result(count);
        std::vector<double> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = a[i] + b[i];
        }

        AddSized(a.data(), b.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_NEAR(result[i], expected[i], 1e-10)
                << "AddSized double failed at index " << i << " for size " << count;
        }
    }
}

// =============================================================================
// Integer tests
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, AddSizedInt32_AllTiers) {
    for (size_t count : {16, 256, 8192}) {
        auto a = GenerateRandomInt32s(count);
        auto b = GenerateRandomInt32s(count);
        std::vector<int32_t> result(count);
        std::vector<int32_t> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = a[i] + b[i];
        }

        AddSized(a.data(), b.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(result[i], expected[i])
                << "AddSized int32 failed at index " << i << " for size " << count;
        }
    }
}

// =============================================================================
// DotSized tests - 8-accumulator with masked tail handling
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, DotSized_Float32_AllTiers) {
    // Test all size tiers: small, medium, large
    for (size_t count : {16, 32, 63, 64, 65, 256, 1000, 4096, 4097, 10000, 100000}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);

        // Reference computation
        float expected = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            expected += a[i] * b[i];
        }

        float result = Dot(a.data(), b.data(), count);

        // Use relative tolerance for floating point
        float rel_error = std::abs(result - expected) / std::max(std::abs(expected), 1.0f);
        ASSERT_LT(rel_error, 1e-4f) << "DotSized float32 failed for size " << count
                                    << " expected=" << expected << " got=" << result;
    }
}

TEST_F(SizeSpecializedKernelsTest, DotSized_Float64_AllTiers) {
    // Test all size tiers for double precision
    for (size_t count : {16, 63, 64, 65, 256, 4096, 4097, 100000}) {
        auto a = GenerateRandomDoubles(count);
        auto b = GenerateRandomDoubles(count);

        double expected = 0.0;
        for (size_t i = 0; i < count; ++i) {
            expected += a[i] * b[i];
        }

        double result = Dot(a.data(), b.data(), count);

        double rel_error = std::abs(result - expected) / std::max(std::abs(expected), 1.0);
        ASSERT_LT(rel_error, 1e-10) << "DotSized float64 failed for size " << count
                                    << " expected=" << expected << " got=" << result;
    }
}

TEST_F(SizeSpecializedKernelsTest, DotSized_MaskedTailHandling) {
    // Test odd sizes that exercise masked tail handling
    // These sizes don't align with SIMD vector width
    for (size_t count : {1, 2, 3, 5, 7, 9, 13, 17, 31, 33, 127, 129, 255, 257}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);

        float expected = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            expected += a[i] * b[i];
        }

        float result = Dot(a.data(), b.data(), count);

        float rel_error = std::abs(result - expected) / std::max(std::abs(expected), 1.0f);
        ASSERT_LT(rel_error, 1e-4f) << "DotSized masked tail failed for size " << count;
    }
}

TEST_F(SizeSpecializedKernelsTest, DotSized_EdgeCases) {
    // Empty array
    std::vector<float> empty;
    EXPECT_EQ(Dot(empty.data(), empty.data(), 0), 0.0f);

    // Single element
    std::vector<float> single_a = {2.0f};
    std::vector<float> single_b = {3.0f};
    EXPECT_FLOAT_EQ(Dot(single_a.data(), single_b.data(), 1), 6.0f);

    // All zeros
    std::vector<float> zeros(100, 0.0f);
    EXPECT_EQ(Dot(zeros.data(), zeros.data(), 100), 0.0f);

    // Ones
    std::vector<float> ones(100, 1.0f);
    EXPECT_FLOAT_EQ(Dot(ones.data(), ones.data(), 100), 100.0f);
}

// =============================================================================
// Streaming store tests for very large arrays
// =============================================================================

TEST_F(SizeSpecializedKernelsTest, AddSized_StreamingStores_VeryLarge) {
    // Test array size that triggers streaming stores (> 8MB = 2M floats)
    const size_t count = 3 * 1024 * 1024;  // 12MB of floats

    auto a = GenerateRandomFloats(count);
    auto b = GenerateRandomFloats(count);
    std::vector<float> result(count);
    std::vector<float> expected(count);

    for (size_t i = 0; i < count; ++i) {
        expected[i] = a[i] + b[i];
    }

    AddSized(a.data(), b.data(), result.data(), count);

    // Check correctness
    for (size_t i = 0; i < count; i += count / 100) {  // Sample check
        ASSERT_FLOAT_EQ(result[i], expected[i]) << "AddSized streaming store failed at index " << i;
    }
}

}  // namespace
}  // namespace simd
}  // namespace bud
