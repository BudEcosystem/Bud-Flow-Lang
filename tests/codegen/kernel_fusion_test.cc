// Copyright 2024 The Bud Flow Lang Authors. All Rights Reserved.
//
// Tests for kernel fusion patterns that combine multiple operations into
// single optimized kernels to eliminate temporaries and improve cache efficiency.
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

class KernelFusionTest : public ::testing::Test {
  protected:
    void SetUp() override { rng_.seed(42); }

    std::vector<float> GenerateRandomFloats(size_t count, float min = -100.0f, float max = 100.0f) {
        std::vector<float> data(count);
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(rng_);
        }
        return data;
    }

    std::vector<double> GenerateRandomDoubles(size_t count, double min = -100.0,
                                              double max = 100.0) {
        std::vector<double> data(count);
        std::uniform_real_distribution<double> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(rng_);
        }
        return data;
    }

    bool ApproxEqual(float a, float b, float rel_tol = 1e-3f) {
        if (std::isinf(a) || std::isinf(b))
            return a == b;
        if (std::isnan(a) || std::isnan(b))
            return false;
        float max_val = std::max(std::abs(a), std::abs(b));
        if (max_val < 1e-6f)
            return true;
        return std::abs(a - b) / max_val < rel_tol;
    }

    bool ArraysApproxEqual(const float* a, const float* b, size_t count, float rel_tol = 1e-3f) {
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
// AddMul Fusion: out[i] = (a[i] + b[i]) * c[i]
// =============================================================================

TEST_F(KernelFusionTest, AddMul_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096, 16384}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        // Reference implementation
        for (size_t i = 0; i < count; ++i) {
            expected[i] = (a[i] + b[i]) * c[i];
        }

        // Fused kernel
        AddMul(a.data(), b.data(), c.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "AddMul failed for count " << count;
    }
}

TEST_F(KernelFusionTest, AddMul_EdgeCases) {
    // Empty array
    std::vector<float> empty;
    AddMul(empty.data(), empty.data(), empty.data(), empty.data(), 0);

    // Single element
    float a = 2.0f, b = 3.0f, c = 4.0f, result = 0.0f;
    AddMul(&a, &b, &c, &result, 1);
    EXPECT_FLOAT_EQ(result, (2.0f + 3.0f) * 4.0f);
}

// =============================================================================
// MulAdd Fusion: out[i] = a[i] * b[i] + c[i] (FMA - already exists but test integration)
// =============================================================================

TEST_F(KernelFusionTest, MulSub_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        // Reference: out = a * b - c
        for (size_t i = 0; i < count; ++i) {
            expected[i] = a[i] * b[i] - c[i];
        }

        // Existing MulSub has signature: (out, a, b, c, count)
        MulSub(result.data(), a.data(), b.data(), c.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "MulSub failed for count " << count;
    }
}

// =============================================================================
// NegMulAdd Fusion: out[i] = -a[i] * b[i] + c[i] (FNMA)
// =============================================================================

TEST_F(KernelFusionTest, NegMulAdd_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        // Reference: out = -a * b + c = c - a*b
        for (size_t i = 0; i < count; ++i) {
            expected[i] = -a[i] * b[i] + c[i];
        }

        // Existing NegMulAdd has signature: (out, a, b, c, count)
        NegMulAdd(result.data(), a.data(), b.data(), c.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "NegMulAdd failed for count " << count;
    }
}

// =============================================================================
// SubMul Fusion: out[i] = (a[i] - b[i]) * c[i]
// =============================================================================

TEST_F(KernelFusionTest, SubMul_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        auto c = GenerateRandomFloats(count);
        std::vector<float> result(count);
        std::vector<float> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = (a[i] - b[i]) * c[i];
        }

        SubMul(a.data(), b.data(), c.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "SubMul failed for count " << count;
    }
}

// =============================================================================
// Axpy Fusion: out[i] = alpha * x[i] + y[i] (BLAS-like)
// =============================================================================

TEST_F(KernelFusionTest, Axpy_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto x = GenerateRandomFloats(count);
        auto y = GenerateRandomFloats(count);
        float alpha = 2.5f;
        std::vector<float> result(count);
        std::vector<float> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = alpha * x[i] + y[i];
        }

        Axpy(alpha, x.data(), y.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "Axpy failed for count " << count;
    }
}

TEST_F(KernelFusionTest, Axpy_SpecialValues) {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> result(4);

    // alpha = 0 -> result = y
    Axpy(0.0f, x.data(), y.data(), result.data(), 4);
    EXPECT_TRUE(ArraysApproxEqual(result.data(), y.data(), 4));

    // alpha = 1 -> result = x + y
    std::vector<float> expected_add = {11.0f, 22.0f, 33.0f, 44.0f};
    Axpy(1.0f, x.data(), y.data(), result.data(), 4);
    EXPECT_TRUE(ArraysApproxEqual(result.data(), expected_add.data(), 4));

    // alpha = -1 -> result = y - x
    std::vector<float> expected_sub = {9.0f, 18.0f, 27.0f, 36.0f};
    Axpy(-1.0f, x.data(), y.data(), result.data(), 4);
    EXPECT_TRUE(ArraysApproxEqual(result.data(), expected_sub.data(), 4));
}

// =============================================================================
// Axpby Fusion: out[i] = alpha * x[i] + beta * y[i]
// =============================================================================

TEST_F(KernelFusionTest, Axpby_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto x = GenerateRandomFloats(count);
        auto y = GenerateRandomFloats(count);
        float alpha = 2.5f;
        float beta = -1.5f;
        std::vector<float> result(count);
        std::vector<float> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = alpha * x[i] + beta * y[i];
        }

        Axpby(alpha, x.data(), beta, y.data(), result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "Axpby failed for count " << count;
    }
}

// =============================================================================
// ScaleAdd Fusion: out[i] = scale * a[i] + offset
// =============================================================================

TEST_F(KernelFusionTest, ScaleAdd_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count);
        float scale = 2.0f;
        float offset = 10.0f;
        std::vector<float> result(count);
        std::vector<float> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = scale * a[i] + offset;
        }

        ScaleAdd(a.data(), scale, offset, result.data(), count);

        ASSERT_TRUE(ArraysApproxEqual(result.data(), expected.data(), count))
            << "ScaleAdd failed for count " << count;
    }
}

// =============================================================================
// SumSq (sum of squares): returns sum(a[i]^2)
// =============================================================================

TEST_F(KernelFusionTest, SumSq_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count, -10.0f, 10.0f);  // Smaller range to avoid overflow
        float expected = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            expected += a[i] * a[i];
        }

        float result = SumSq(a.data(), count);

        ASSERT_TRUE(ApproxEqual(result, expected))
            << "SumSq failed for count " << count << " expected=" << expected << " got=" << result;
    }
}

// =============================================================================
// SumAbsDiff (L1 distance): returns sum(|a[i] - b[i]|)
// =============================================================================

TEST_F(KernelFusionTest, SumAbsDiff_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count);
        auto b = GenerateRandomFloats(count);
        float expected = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            expected += std::abs(a[i] - b[i]);
        }

        float result = SumAbsDiff(a.data(), b.data(), count);

        ASSERT_TRUE(ApproxEqual(result, expected)) << "SumAbsDiff failed for count " << count
                                                   << " expected=" << expected << " got=" << result;
    }
}

// =============================================================================
// SumSqDiff (squared L2 distance): returns sum((a[i] - b[i])^2)
// =============================================================================

TEST_F(KernelFusionTest, SumSqDiff_Correctness) {
    for (size_t count : {16, 64, 256, 1024, 4096}) {
        auto a = GenerateRandomFloats(count, -10.0f, 10.0f);
        auto b = GenerateRandomFloats(count, -10.0f, 10.0f);
        float expected = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            float diff = a[i] - b[i];
            expected += diff * diff;
        }

        float result = SumSqDiff(a.data(), b.data(), count);

        ASSERT_TRUE(ApproxEqual(result, expected)) << "SumSqDiff failed for count " << count
                                                   << " expected=" << expected << " got=" << result;
    }
}

// =============================================================================
// Performance Tests - Verify fusion provides speedup
// =============================================================================

TEST_F(KernelFusionTest, AddMul_Performance) {
    constexpr size_t kCount = 262144;  // 256K elements
    constexpr int kIterations = 1000;

    auto a = GenerateRandomFloats(kCount);
    auto b = GenerateRandomFloats(kCount);
    auto c = GenerateRandomFloats(kCount);
    std::vector<float> result(kCount);
    std::vector<float> temp(kCount);

    // Warm up
    for (int i = 0; i < 10; ++i) {
        AddMul(a.data(), b.data(), c.data(), result.data(), kCount);
    }

    // Benchmark fused kernel
    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        AddMul(a.data(), b.data(), c.data(), result.data(), kCount);
    }
    auto end_fused = std::chrono::high_resolution_clock::now();

    // Benchmark unfused (Add then Mul)
    auto start_unfused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        Add(a.data(), b.data(), temp.data(), kCount);
        Mul(temp.data(), c.data(), result.data(), kCount);
    }
    auto end_unfused = std::chrono::high_resolution_clock::now();

    double time_fused = std::chrono::duration<double, std::micro>(end_fused - start_fused).count();
    double time_unfused =
        std::chrono::duration<double, std::micro>(end_unfused - start_unfused).count();
    double speedup = time_unfused / time_fused;

    std::cout << "AddMul fusion performance (" << kCount << " elements):\n"
              << "  Unfused (Add+Mul): " << time_unfused / kIterations << " us/iter\n"
              << "  Fused (AddMul):    " << time_fused / kIterations << " us/iter\n"
              << "  Speedup:           " << speedup << "x\n";

    // Fused should be faster (fewer memory round-trips)
    EXPECT_GE(speedup, 1.1) << "Fused AddMul should be faster than unfused Add+Mul";
}

TEST_F(KernelFusionTest, Axpy_Performance) {
    constexpr size_t kCount = 262144;
    constexpr int kIterations = 1000;

    auto x = GenerateRandomFloats(kCount);
    auto y = GenerateRandomFloats(kCount);
    float alpha = 2.5f;
    std::vector<float> result(kCount);
    std::vector<float> temp(kCount);

    // Warm up
    for (int i = 0; i < 10; ++i) {
        Axpy(alpha, x.data(), y.data(), result.data(), kCount);
    }

    // Benchmark fused Axpy
    auto start_fused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        Axpy(alpha, x.data(), y.data(), result.data(), kCount);
    }
    auto end_fused = std::chrono::high_resolution_clock::now();

    // Benchmark unfused (manual scale then Add)
    auto start_unfused = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        // Manual scale: temp[i] = alpha * x[i]
        for (size_t j = 0; j < kCount; ++j) {
            temp[j] = alpha * x[j];
        }
        Add(temp.data(), y.data(), result.data(), kCount);
    }
    auto end_unfused = std::chrono::high_resolution_clock::now();

    double time_fused = std::chrono::duration<double, std::micro>(end_fused - start_fused).count();
    double time_unfused =
        std::chrono::duration<double, std::micro>(end_unfused - start_unfused).count();
    double speedup = time_unfused / time_fused;

    std::cout << "Axpy fusion performance (" << kCount << " elements):\n"
              << "  Unfused (Scale+Add): " << time_unfused / kIterations << " us/iter\n"
              << "  Fused (Axpy):        " << time_fused / kIterations << " us/iter\n"
              << "  Speedup:             " << speedup << "x\n";

    EXPECT_GE(speedup, 1.1) << "Fused Axpy should be faster than unfused Scale+Add";
}

// =============================================================================
// Double precision tests
// =============================================================================

TEST_F(KernelFusionTest, AddMul_Float64_Correctness) {
    for (size_t count : {16, 256, 4096}) {
        auto a = GenerateRandomDoubles(count);
        auto b = GenerateRandomDoubles(count);
        auto c = GenerateRandomDoubles(count);
        std::vector<double> result(count);
        std::vector<double> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = (a[i] + b[i]) * c[i];
        }

        AddMul(a.data(), b.data(), c.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_NEAR(result[i], expected[i], std::abs(expected[i]) * 1e-10 + 1e-10)
                << "AddMul double failed at index " << i << " for count " << count;
        }
    }
}

TEST_F(KernelFusionTest, Axpy_Float64_Correctness) {
    for (size_t count : {16, 256, 4096}) {
        auto x = GenerateRandomDoubles(count);
        auto y = GenerateRandomDoubles(count);
        double alpha = 2.5;
        std::vector<double> result(count);
        std::vector<double> expected(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = alpha * x[i] + y[i];
        }

        Axpy(alpha, x.data(), y.data(), result.data(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_NEAR(result[i], expected[i], std::abs(expected[i]) * 1e-10 + 1e-10)
                << "Axpy double failed at index " << i << " for count " << count;
        }
    }
}

}  // namespace
}  // namespace simd
}  // namespace bud
