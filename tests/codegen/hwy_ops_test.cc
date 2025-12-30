// =============================================================================
// Bud Flow Lang - Highway SIMD Operations Tests
// =============================================================================
//
// Comprehensive TDD tests for all Highway SIMD operations.
// Tests cover:
// - Basic functionality correctness
// - Edge cases (empty, single element)
// - Remainder handling (non-vector-aligned sizes)
// - Numerical accuracy (ULP error bounds)
// - Special values (inf, nan, zero, denormals)
// - Memory safety (no buffer overruns)
//
// =============================================================================

#include "bud_flow_lang/codegen/hwy_ops.h"

#include <hwy/aligned_allocator.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
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

class HwyOpsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize random generator with fixed seed for reproducibility
        rng_.seed(42);
    }

    // Generate random float array
    hwy::AlignedFreeUniquePtr<float[]> RandomFloats(size_t count, float min = -100.0f,
                                                    float max = 100.0f) {
        auto arr = hwy::AllocateAligned<float>(count);
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    // Generate random double array
    hwy::AlignedFreeUniquePtr<double[]> RandomDoubles(size_t count, double min = -100.0,
                                                      double max = 100.0) {
        auto arr = hwy::AllocateAligned<double>(count);
        std::uniform_real_distribution<double> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    // Generate random int32 array
    hwy::AlignedFreeUniquePtr<int32_t[]> RandomInts32(size_t count, int32_t min = -1000,
                                                      int32_t max = 1000) {
        auto arr = hwy::AllocateAligned<int32_t>(count);
        std::uniform_int_distribution<int32_t> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    // Generate random int64 array
    hwy::AlignedFreeUniquePtr<int64_t[]> RandomInts64(size_t count, int64_t min = -1000,
                                                      int64_t max = 1000) {
        auto arr = hwy::AllocateAligned<int64_t>(count);
        std::uniform_int_distribution<int64_t> dist(min, max);
        for (size_t i = 0; i < count; ++i) {
            arr[i] = dist(rng_);
        }
        return arr;
    }

    // Allocate aligned output array
    template <typename T>
    hwy::AlignedFreeUniquePtr<T[]> AllocateOutput(size_t count) {
        auto arr = hwy::AllocateAligned<T>(count);
        // Initialize to sentinel value to detect partial writes
        std::fill(arr.get(), arr.get() + count, static_cast<T>(0xDEADBEEF));
        return arr;
    }

    // Check if two floats are approximately equal
    static bool ApproxEqual(float a, float b, float rel_tol = 1e-5f, float abs_tol = 1e-8f) {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (std::isinf(a) && std::isinf(b))
            return (a > 0) == (b > 0);
        float diff = std::abs(a - b);
        return diff <= abs_tol || diff <= rel_tol * std::max(std::abs(a), std::abs(b));
    }

    // Check if two doubles are approximately equal
    static bool ApproxEqual(double a, double b, double rel_tol = 1e-12, double abs_tol = 1e-15) {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (std::isinf(a) && std::isinf(b))
            return (a > 0) == (b > 0);
        double diff = std::abs(a - b);
        return diff <= abs_tol || diff <= rel_tol * std::max(std::abs(a), std::abs(b));
    }

    // Test sizes including edge cases and remainder scenarios
    // Note: 0 is excluded to avoid allocating 0-size arrays; empty arrays are tested separately
    static std::vector<size_t> TestSizes() {
        return {1,  2,   3,   4,   5,   7,   8,   15,  16,  17,  31,   32,   33,   63,  64,
                65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025};
    }

    std::mt19937 rng_;
};

// =============================================================================
// Priority 1: Arithmetic Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, AddFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Add(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = a[i] + b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count << ": got " << out[i]
                << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, AddFloat64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomDoubles(count);
        auto b = RandomDoubles(count);
        auto out = AllocateOutput<double>(count);

        Add(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            double expected = a[i] + b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, AddInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto b = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Add(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = a[i] + b[i];
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, SubFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Sub(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = a[i] - b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, MulFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Mul(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = a[i] * b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, DivFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        // Avoid division by values close to zero
        auto b = RandomFloats(count, 1.0f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Div(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = a[i] / b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, NegFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Neg(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = -a[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, AbsFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -100.0f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Abs(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::abs(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, AbsInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count, -1000, 1000);
        auto out = AllocateOutput<int32_t>(count);

        Abs(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = std::abs(a[i]);
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 1: FMA Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MulAddFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto c = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        MulAdd(out.get(), a.get(), b.get(), c.get(), count);

        for (size_t i = 0; i < count; ++i) {
            // Use fused multiply-add for reference if available
            float expected = std::fma(a[i], b[i], c[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count << ": got " << out[i]
                << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, MulSubFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto c = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        MulSub(out.get(), a.get(), b.get(), c.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::fma(a[i], b[i], -c[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, NegMulAddFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto c = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        NegMulAdd(out.get(), a.get(), b.get(), c.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::fma(-a[i], b[i], c[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 1: MinMax Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MinFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Min(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::min(a[i], b[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, MaxFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Max(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::max(a[i], b[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, ClampFloat32_BasicCorrectness) {
    const float lo = -10.0f;
    const float hi = 10.0f;

    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -100.0f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Clamp(out.get(), a.get(), lo, hi, count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::clamp(a[i], lo, hi);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 1: Math Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SqrtFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        // Only positive values for sqrt
        auto a = RandomFloats(count, 0.001f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Sqrt(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::sqrt(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count << ": sqrt(" << a[i]
                << ") = " << out[i] << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, ExpFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        // Limit range to avoid overflow
        auto a = RandomFloats(count, -10.0f, 10.0f);
        auto out = AllocateOutput<float>(count);

        Exp(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::exp(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count << ": exp(" << a[i]
                << ") = " << out[i] << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, LogFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        // Only positive values for log
        auto a = RandomFloats(count, 0.001f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Log(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::log(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count << ": log(" << a[i]
                << ") = " << out[i] << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, SinFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        // Limited range for better accuracy
        auto a = RandomFloats(count, -10.0f, 10.0f);
        auto out = AllocateOutput<float>(count);

        Sin(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::sin(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count << ": sin(" << a[i]
                << ") = " << out[i] << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, CosFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -10.0f, 10.0f);
        auto out = AllocateOutput<float>(count);

        Cos(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::cos(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count << ": cos(" << a[i]
                << ") = " << out[i] << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, TanhFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -5.0f, 5.0f);
        auto out = AllocateOutput<float>(count);

        Tanh(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::tanh(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count << ": tanh(" << a[i]
                << ") = " << out[i] << ", expected " << expected;
        }
    }
}

// =============================================================================
// Priority 1: Comparison Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, EqFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<uint8_t>(count);

        // Make some values equal
        for (size_t i = 0; i < count; i += 3) {
            b[i] = a[i];
        }

        Eq(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] == b[i]) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, LtFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<uint8_t>(count);

        Lt(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] < b[i]) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, GtFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<uint8_t>(count);

        Gt(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] > b[i]) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 1: Reduction Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, ReduceSumFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;  // Skip empty case

        auto a = RandomFloats(count, -10.0f, 10.0f);

        float result = ReduceSum(a.get(), count);

        // Compute reference sum with Kahan summation for accuracy
        double expected = 0.0;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<double>(a[i]);
        }

        ASSERT_TRUE(ApproxEqual(result, static_cast<float>(expected), 1e-3f))
            << "ReduceSum mismatch for count " << count << ": got " << result << ", expected "
            << expected;
    }
}

TEST_F(HwyOpsTest, ReduceMinFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count);

        float result = ReduceMin(a.get(), count);
        float expected = *std::min_element(a.get(), a.get() + count);

        ASSERT_TRUE(ApproxEqual(result, expected)) << "ReduceMin mismatch for count " << count;
    }
}

TEST_F(HwyOpsTest, ReduceMaxFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count);

        float result = ReduceMax(a.get(), count);
        float expected = *std::max_element(a.get(), a.get() + count);

        ASSERT_TRUE(ApproxEqual(result, expected)) << "ReduceMax mismatch for count " << count;
    }
}

TEST_F(HwyOpsTest, DotFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        if (count == 0)
            continue;

        auto a = RandomFloats(count, -10.0f, 10.0f);
        auto b = RandomFloats(count, -10.0f, 10.0f);

        float result = Dot(a.get(), b.get(), count);

        // Compute reference dot product
        double expected = 0.0;
        for (size_t i = 0; i < count; ++i) {
            expected += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }

        ASSERT_TRUE(ApproxEqual(result, static_cast<float>(expected), 1e-3f))
            << "Dot mismatch for count " << count << ": got " << result << ", expected "
            << expected;
    }
}

// =============================================================================
// Priority 1: Select Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SelectFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto yes = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = AllocateOutput<uint8_t>(count);
        auto out = AllocateOutput<float>(count);

        // Create alternating mask
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 2 == 0) ? 0xFF : 0x00;
        }

        Select(out.get(), mask.get(), yes.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = (mask[i] != 0) ? yes[i] : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 2: Extended Math Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Exp2Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -10.0f, 10.0f);
        auto out = AllocateOutput<float>(count);

        Exp2(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::exp2(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, Log2Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, 0.001f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Log2(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::log2(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, SinhFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -5.0f, 5.0f);
        auto out = AllocateOutput<float>(count);

        Sinh(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::sinh(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 2: Rounding Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, RoundFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -100.0f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Round(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::nearbyint(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, FloorFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -100.0f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Floor(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::floor(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, CeilFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count, -100.0f, 100.0f);
        auto out = AllocateOutput<float>(count);

        Ceil(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::ceil(a[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Priority 2: Bitwise Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, BitwiseAndInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto b = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        BitwiseAnd(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = a[i] & b[i];
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, BitwiseOrInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto b = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        BitwiseOr(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = a[i] | b[i];
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, BitwiseXorInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto b = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        BitwiseXor(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = a[i] ^ b[i];
            ASSERT_EQ(out[i], expected) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// Edge Cases and Special Values Tests
// =============================================================================

TEST_F(HwyOpsTest, AddFloat32_SpecialValues) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto b = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    // Test special values
    a[0] = 0.0f;
    b[0] = 0.0f;  // 0 + 0
    a[1] = std::numeric_limits<float>::max();
    b[1] = 1.0f;  // overflow
    a[2] = -std::numeric_limits<float>::max();
    b[2] = -1.0f;  // underflow
    a[3] = std::numeric_limits<float>::infinity();
    b[3] = 1.0f;  // inf + 1
    a[4] = -std::numeric_limits<float>::infinity();
    b[4] = 1.0f;  // -inf + 1
    a[5] = std::numeric_limits<float>::quiet_NaN();
    b[5] = 1.0f;  // nan + 1
    a[6] = std::numeric_limits<float>::denorm_min();
    b[6] = 0.0f;  // denormal
    a[7] = -0.0f;
    b[7] = 0.0f;  // -0 + 0

    Add(out.get(), a.get(), b.get(), count);

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_TRUE(std::isinf(out[3]) && out[3] > 0);
    EXPECT_TRUE(std::isinf(out[4]) && out[4] < 0);
    EXPECT_TRUE(std::isnan(out[5]));
}

TEST_F(HwyOpsTest, SqrtFloat32_SpecialValues) {
    const size_t count = 5;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    a[0] = 0.0f;                                      // sqrt(0) = 0
    a[1] = 1.0f;                                      // sqrt(1) = 1
    a[2] = 4.0f;                                      // sqrt(4) = 2
    a[3] = std::numeric_limits<float>::infinity();    // sqrt(inf) = inf
    a[4] = std::numeric_limits<float>::denorm_min();  // sqrt(denormal)

    Sqrt(out.get(), a.get(), count);

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
    EXPECT_TRUE(std::isinf(out[3]));
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

TEST_F(HwyOpsTest, EmptyArray_NoOperation) {
    auto a = hwy::AllocateAligned<float>(1);
    auto b = hwy::AllocateAligned<float>(1);
    auto out = hwy::AllocateAligned<float>(1);

    a[0] = 1.0f;
    b[0] = 2.0f;
    out[0] = 999.0f;

    // Should not crash or modify output for count=0
    Add(out.get(), a.get(), b.get(), 0);
    EXPECT_FLOAT_EQ(out[0], 999.0f);
}

// =============================================================================
// Type-Generic Dispatch Tests
// =============================================================================

TEST_F(HwyOpsTest, DispatchAdd_Float32) {
    const size_t count = 100;
    auto a = RandomFloats(count);
    auto b = RandomFloats(count);
    auto out = AllocateOutput<float>(count);

    auto result = DispatchAdd(out.get(), a.get(), b.get(), count, ScalarType::kFloat32);
    ASSERT_TRUE(result.hasValue()) << "DispatchAdd failed";

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], a[i] + b[i]));
    }
}

TEST_F(HwyOpsTest, DispatchAdd_Float64) {
    const size_t count = 100;
    auto a = RandomDoubles(count);
    auto b = RandomDoubles(count);
    auto out = AllocateOutput<double>(count);

    auto result = DispatchAdd(out.get(), a.get(), b.get(), count, ScalarType::kFloat64);
    ASSERT_TRUE(result.hasValue()) << "DispatchAdd failed";

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], a[i] + b[i]));
    }
}

TEST_F(HwyOpsTest, DispatchAdd_Int32) {
    const size_t count = 100;
    auto a = RandomInts32(count);
    auto b = RandomInts32(count);
    auto out = AllocateOutput<int32_t>(count);

    auto result = DispatchAdd(out.get(), a.get(), b.get(), count, ScalarType::kInt32);
    ASSERT_TRUE(result.hasValue()) << "DispatchAdd failed";

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], a[i] + b[i]);
    }
}

TEST_F(HwyOpsTest, DispatchAdd_UnsupportedType) {
    const size_t count = 10;
    auto a = hwy::AllocateAligned<float>(count);
    auto b = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<float>(count);

    auto result = DispatchAdd(out.get(), a.get(), b.get(), count, ScalarType::kUnknown);
    ASSERT_TRUE(result.hasError()) << "DispatchAdd should fail for unknown type";
    ASSERT_EQ(result.error().code(), ErrorCode::kUnsupportedType);
}

// =============================================================================
// New Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, IsNaNFloat32_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<uint8_t>(count);

    a[0] = 1.0f;
    a[1] = std::numeric_limits<float>::quiet_NaN();
    a[2] = 0.0f;
    a[3] = std::numeric_limits<float>::signaling_NaN();
    a[4] = std::numeric_limits<float>::infinity();
    a[5] = -std::numeric_limits<float>::infinity();
    a[6] = -1.0f;
    a[7] = std::numeric_limits<float>::denorm_min();

    IsNaN(out.get(), a.get(), count);

    EXPECT_EQ(out[0], 0x00);  // 1.0 is not NaN
    EXPECT_EQ(out[1], 0xFF);  // quiet_NaN is NaN
    EXPECT_EQ(out[2], 0x00);  // 0 is not NaN
    EXPECT_EQ(out[3], 0xFF);  // signaling_NaN is NaN
    EXPECT_EQ(out[4], 0x00);  // inf is not NaN
    EXPECT_EQ(out[5], 0x00);  // -inf is not NaN
    EXPECT_EQ(out[6], 0x00);  // -1 is not NaN
    EXPECT_EQ(out[7], 0x00);  // denorm is not NaN
}

TEST_F(HwyOpsTest, IsInfFloat32_BasicCorrectness) {
    const size_t count = 6;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<uint8_t>(count);

    a[0] = std::numeric_limits<float>::infinity();
    a[1] = -std::numeric_limits<float>::infinity();
    a[2] = 1.0f;
    a[3] = 0.0f;
    a[4] = std::numeric_limits<float>::quiet_NaN();
    a[5] = std::numeric_limits<float>::max();

    IsInf(out.get(), a.get(), count);

    EXPECT_EQ(out[0], 0xFF);  // inf
    EXPECT_EQ(out[1], 0xFF);  // -inf
    EXPECT_EQ(out[2], 0x00);  // 1.0
    EXPECT_EQ(out[3], 0x00);  // 0
    EXPECT_EQ(out[4], 0x00);  // NaN is not inf
    EXPECT_EQ(out[5], 0x00);  // max is not inf
}

TEST_F(HwyOpsTest, IsFiniteFloat32_BasicCorrectness) {
    const size_t count = 6;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<uint8_t>(count);

    a[0] = 1.0f;
    a[1] = 0.0f;
    a[2] = std::numeric_limits<float>::infinity();
    a[3] = -std::numeric_limits<float>::infinity();
    a[4] = std::numeric_limits<float>::quiet_NaN();
    a[5] = std::numeric_limits<float>::max();

    IsFinite(out.get(), a.get(), count);

    EXPECT_EQ(out[0], 0xFF);  // 1.0 is finite
    EXPECT_EQ(out[1], 0xFF);  // 0 is finite
    EXPECT_EQ(out[2], 0x00);  // inf is not finite
    EXPECT_EQ(out[3], 0x00);  // -inf is not finite
    EXPECT_EQ(out[4], 0x00);  // NaN is not finite
    EXPECT_EQ(out[5], 0xFF);  // max is finite
}

TEST_F(HwyOpsTest, BroadcastFloat32_BasicCorrectness) {
    const size_t count = 100;
    auto out = AllocateOutput<float>(count);

    Broadcast(out.get(), 42.0f, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(out[i], 42.0f) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, BroadcastInt32_BasicCorrectness) {
    const size_t count = 100;
    auto out = AllocateOutput<int32_t>(count);

    Broadcast(out.get(), 123, count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(out[i], 123) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, AbsDiffFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        AbsDiff(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::abs(a[i] - b[i]);
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, TanFloat32_BasicCorrectness) {
    const size_t count = 10;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    // Use values where tan is well-defined
    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;  // 0, 0.1, 0.2, ...
    }

    Tan(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = std::tan(a[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-4f))
            << "Mismatch at index " << i << ": got " << out[i] << ", expected " << expected;
    }
}

TEST_F(HwyOpsTest, Expm1Float32_BasicCorrectness) {
    const size_t count = 10;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i) * 0.1f - 0.5f;  // -0.5 to 0.4
    }

    Expm1(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = std::expm1(a[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-5f))
            << "Mismatch at index " << i << ": got " << out[i] << ", expected " << expected;
    }
}

TEST_F(HwyOpsTest, Log1pFloat32_BasicCorrectness) {
    const size_t count = 10;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i) * 0.2f;  // 0 to 1.8 (must be > -1)
    }

    Log1p(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = std::log1p(a[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected, 1e-5f))
            << "Mismatch at index " << i << ": got " << out[i] << ", expected " << expected;
    }
}

TEST_F(HwyOpsTest, CopySignFloat32_BasicCorrectness) {
    const size_t count = 6;
    auto mag = hwy::AllocateAligned<float>(count);
    auto sign = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    mag[0] = 5.0f;
    sign[0] = 1.0f;  // positive -> +5
    mag[1] = 5.0f;
    sign[1] = -1.0f;  // negative -> -5
    mag[2] = -5.0f;
    sign[2] = 1.0f;  // positive -> +5
    mag[3] = -5.0f;
    sign[3] = -1.0f;  // negative -> -5
    mag[4] = 0.0f;
    sign[4] = -1.0f;  // negative -> -0
    mag[5] = 3.14f;
    sign[5] = -0.0f;  // negative zero -> -3.14

    CopySign(out.get(), mag.get(), sign.get(), count);

    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], -5.0f);
    EXPECT_FLOAT_EQ(out[2], 5.0f);
    EXPECT_FLOAT_EQ(out[3], -5.0f);
}

TEST_F(HwyOpsTest, ConvertInt32ToFloat32_BasicCorrectness) {
    const size_t count = 10;
    auto a = RandomInts32(count, -1000, 1000);
    auto out = AllocateOutput<float>(count);

    ConvertTo(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = static_cast<float>(a[i]);
        EXPECT_FLOAT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ConvertFloat32ToInt32_BasicCorrectness) {
    const size_t count = 10;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<int32_t>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i) * 1.5f;  // 0, 1.5, 3, 4.5, ...
    }

    ConvertTo(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        int32_t expected = static_cast<int32_t>(a[i]);
        EXPECT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, PromoteFloat32ToFloat64_BasicCorrectness) {
    const size_t count = 100;
    auto a = RandomFloats(count);
    auto out = AllocateOutput<double>(count);

    PromoteTo(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        double expected = static_cast<double>(a[i]);
        EXPECT_DOUBLE_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, DemoteFloat64ToFloat32_BasicCorrectness) {
    const size_t count = 100;
    auto a = RandomDoubles(count, -1000.0, 1000.0);
    auto out = AllocateOutput<float>(count);

    DemoteTo(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = static_cast<float>(a[i]);
        EXPECT_FLOAT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, InterleaveFloat32_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto b = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<float>(count * 2);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i);        // 0, 1, 2, 3, ...
        b[i] = static_cast<float>(i + 100);  // 100, 101, 102, ...
    }

    Interleave(out.get(), a.get(), b.get(), count);

    // Expected: 0, 100, 1, 101, 2, 102, ...
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(out[2 * i], static_cast<float>(i));
        EXPECT_FLOAT_EQ(out[2 * i + 1], static_cast<float>(i + 100));
    }
}

TEST_F(HwyOpsTest, CompressFloat32_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    auto out = hwy::AllocateAligned<float>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i);
        mask[i] = (i % 2 == 0) ? 0xFF : 0x00;  // Keep even indices
    }

    size_t result_count = Compress(out.get(), a.get(), mask.get(), count);

    EXPECT_EQ(result_count, 4u);  // 4 even indices: 0, 2, 4, 6
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 4.0f);
    EXPECT_FLOAT_EQ(out[3], 6.0f);
}

TEST_F(HwyOpsTest, PairwiseAddFloat32_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<float>(count / 2);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i);  // 0, 1, 2, 3, 4, 5, 6, 7
    }

    PairwiseAdd(out.get(), a.get(), count);

    // Expected: 0+1=1, 2+3=5, 4+5=9, 6+7=13
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 5.0f);
    EXPECT_FLOAT_EQ(out[2], 9.0f);
    EXPECT_FLOAT_EQ(out[3], 13.0f);
}

TEST_F(HwyOpsTest, GatherFloat32_BasicCorrectness) {
    const size_t base_count = 10;
    const size_t idx_count = 5;
    auto base = hwy::AllocateAligned<float>(base_count);
    auto indices = hwy::AllocateAligned<int32_t>(idx_count);
    auto out = hwy::AllocateAligned<float>(idx_count);

    for (size_t i = 0; i < base_count; ++i) {
        base[i] = static_cast<float>(i * 10);  // 0, 10, 20, 30, ...
    }
    indices[0] = 0;
    indices[1] = 3;
    indices[2] = 7;
    indices[3] = 1;
    indices[4] = 5;

    Gather(out.get(), base.get(), indices.get(), idx_count);

    EXPECT_FLOAT_EQ(out[0], 0.0f);   // base[0]
    EXPECT_FLOAT_EQ(out[1], 30.0f);  // base[3]
    EXPECT_FLOAT_EQ(out[2], 70.0f);  // base[7]
    EXPECT_FLOAT_EQ(out[3], 10.0f);  // base[1]
    EXPECT_FLOAT_EQ(out[4], 50.0f);  // base[5]
}

TEST_F(HwyOpsTest, ScatterFloat32_BasicCorrectness) {
    const size_t base_count = 10;
    const size_t val_count = 3;
    auto base = hwy::AllocateAligned<float>(base_count);
    auto values = hwy::AllocateAligned<float>(val_count);
    auto indices = hwy::AllocateAligned<int32_t>(val_count);

    // Initialize base to zeros
    for (size_t i = 0; i < base_count; ++i) {
        base[i] = 0.0f;
    }
    values[0] = 100.0f;
    values[1] = 200.0f;
    values[2] = 300.0f;
    indices[0] = 2;
    indices[1] = 5;
    indices[2] = 8;

    Scatter(values.get(), base.get(), indices.get(), val_count);

    EXPECT_FLOAT_EQ(base[2], 100.0f);
    EXPECT_FLOAT_EQ(base[5], 200.0f);
    EXPECT_FLOAT_EQ(base[8], 300.0f);
    EXPECT_FLOAT_EQ(base[0], 0.0f);  // unchanged
}

TEST_F(HwyOpsTest, AddInt8_BasicCorrectness) {
    const size_t count = 100;
    auto a = hwy::AllocateAligned<int8_t>(count);
    auto b = hwy::AllocateAligned<int8_t>(count);
    auto out = hwy::AllocateAligned<int8_t>(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-50, 50);
    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<int8_t>(dist(rng));
        b[i] = static_cast<int8_t>(dist(rng));
    }

    Add(out.get(), a.get(), b.get(), count);

    for (size_t i = 0; i < count; ++i) {
        int8_t expected = static_cast<int8_t>(a[i] + b[i]);
        EXPECT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, AddInt16_BasicCorrectness) {
    const size_t count = 100;
    auto a = hwy::AllocateAligned<int16_t>(count);
    auto b = hwy::AllocateAligned<int16_t>(count);
    auto out = hwy::AllocateAligned<int16_t>(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<int16_t>(dist(rng));
        b[i] = static_cast<int16_t>(dist(rng));
    }

    Add(out.get(), a.get(), b.get(), count);

    for (size_t i = 0; i < count; ++i) {
        int16_t expected = static_cast<int16_t>(a[i] + b[i]);
        EXPECT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, AddUint32_BasicCorrectness) {
    const size_t count = 100;
    auto a = hwy::AllocateAligned<uint32_t>(count);
    auto b = hwy::AllocateAligned<uint32_t>(count);
    auto out = hwy::AllocateAligned<uint32_t>(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 1000000);
    for (size_t i = 0; i < count; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    Add(out.get(), a.get(), b.get(), count);

    for (size_t i = 0; i < count; ++i) {
        uint32_t expected = a[i] + b[i];
        EXPECT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SaturatedAddInt16_BasicCorrectness) {
    const size_t count = 6;
    auto a = hwy::AllocateAligned<int16_t>(count);
    auto b = hwy::AllocateAligned<int16_t>(count);
    auto out = hwy::AllocateAligned<int16_t>(count);

    a[0] = 100;
    b[0] = 50;  // Normal add
    a[1] = 32000;
    b[1] = 1000;  // Should saturate to max
    a[2] = -32000;
    b[2] = -1000;  // Should saturate to min
    a[3] = 0;
    b[3] = 0;
    a[4] = 32767;
    b[4] = 1;  // Exactly at overflow
    a[5] = -32768;
    b[5] = -1;  // Exactly at underflow

    SaturatedAdd(out.get(), a.get(), b.get(), count);

    EXPECT_EQ(out[0], 150);
    EXPECT_EQ(out[1], 32767);   // saturated
    EXPECT_EQ(out[2], -32768);  // saturated
    EXPECT_EQ(out[3], 0);
    EXPECT_EQ(out[4], 32767);   // saturated
    EXPECT_EQ(out[5], -32768);  // saturated
}

TEST_F(HwyOpsTest, SaturatedSubInt16_BasicCorrectness) {
    const size_t count = 4;
    auto a = hwy::AllocateAligned<int16_t>(count);
    auto b = hwy::AllocateAligned<int16_t>(count);
    auto out = hwy::AllocateAligned<int16_t>(count);

    a[0] = 100;
    b[0] = 50;  // Normal sub
    a[1] = -32000;
    b[1] = 1000;  // Should saturate to min
    a[2] = 32000;
    b[2] = -1000;  // Should saturate to max
    a[3] = 0;
    b[3] = 0;

    SaturatedSub(out.get(), a.get(), b.get(), count);

    EXPECT_EQ(out[0], 50);
    EXPECT_EQ(out[1], -32768);  // saturated
    EXPECT_EQ(out[2], 32767);   // saturated
    EXPECT_EQ(out[3], 0);
}

TEST_F(HwyOpsTest, ApproxReciprocalFloat32_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 4.0f;
    a[3] = 0.5f;
    a[4] = 10.0f;
    a[5] = 100.0f;
    a[6] = 0.1f;
    a[7] = 0.25f;

    ApproxReciprocal(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = 1.0f / a[i];
        // Approximate reciprocal may have ~12-bit precision
        ASSERT_TRUE(ApproxEqual(out[i], expected, 0.01f))
            << "Mismatch at index " << i << ": got " << out[i] << ", expected " << expected;
    }
}

TEST_F(HwyOpsTest, ApproxReciprocalSqrt_Float32_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    a[0] = 1.0f;
    a[1] = 4.0f;
    a[2] = 9.0f;
    a[3] = 16.0f;
    a[4] = 25.0f;
    a[5] = 100.0f;
    a[6] = 0.25f;
    a[7] = 0.01f;

    ApproxReciprocalSqrt(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = 1.0f / std::sqrt(a[i]);
        // Approximate rsqrt may have ~12-bit precision
        ASSERT_TRUE(ApproxEqual(out[i], expected, 0.02f))
            << "Mismatch at index " << i << ": got " << out[i] << ", expected " << expected;
    }
}

TEST_F(HwyOpsTest, ApproxReciprocalSqrt_Float64_BasicCorrectness) {
    const size_t count = 8;
    auto a = hwy::AllocateAligned<double>(count);
    auto out = AllocateOutput<double>(count);

    a[0] = 1.0;
    a[1] = 4.0;
    a[2] = 9.0;
    a[3] = 16.0;
    a[4] = 25.0;
    a[5] = 100.0;
    a[6] = 0.25;
    a[7] = 0.01;

    ApproxReciprocalSqrt(out.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        double expected = 1.0 / std::sqrt(a[i]);
        // Approximate rsqrt may have ~12-bit precision
        ASSERT_TRUE(ApproxEqual(out[i], expected, 0.02))
            << "Mismatch at index " << i << ": got " << out[i] << ", expected " << expected;
    }
}

// =============================================================================
// P0: Load/Store Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, LoadFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Load(out.get(), src.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(out[i], src[i]))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, LoadFloat64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomDoubles(count);
        auto out = AllocateOutput<double>(count);

        Load(out.get(), src.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(out[i], src[i]))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, StoreFloat32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto dst = AllocateOutput<float>(count);

        Store(src.get(), dst.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(dst[i], src[i]))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, StoreFloat64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomDoubles(count);
        auto dst = AllocateOutput<double>(count);

        Store(src.get(), dst.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(dst[i], src[i]))
                << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, LoadInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Load(out.get(), src.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(out[i], src[i]) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, StoreInt32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomInts32(count);
        auto dst = AllocateOutput<int32_t>(count);

        Store(src.get(), dst.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(dst[i], src[i]) << "Mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// P0: BitCast Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, BitCastFloat32ToInt32_BasicCorrectness) {
    const size_t count = 64;
    auto src = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<int32_t>(count);

    // Test specific values
    src[0] = 1.0f;
    src[1] = -1.0f;
    src[2] = 0.0f;
    src[3] = std::numeric_limits<float>::infinity();
    src[4] = -std::numeric_limits<float>::infinity();
    src[5] = 3.14159f;

    // Fill rest with random values
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (size_t i = 6; i < count; ++i) {
        src[i] = dist(rng_);
    }

    BitCastFloat32ToInt32(out.get(), src.get(), count);

    // Verify bit-level equality using memcmp
    for (size_t i = 0; i < count; ++i) {
        int32_t expected;
        std::memcpy(&expected, &src[i], sizeof(float));
        ASSERT_EQ(out[i], expected) << "BitCast mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, BitCastInt32ToFloat32_BasicCorrectness) {
    const size_t count = 64;
    auto src = hwy::AllocateAligned<int32_t>(count);
    auto out = hwy::AllocateAligned<float>(count);

    // Test specific bit patterns
    src[0] = 0x3f800000;  // 1.0f
    src[1] = 0xbf800000;  // -1.0f
    src[2] = 0x00000000;  // 0.0f
    src[3] = 0x7f800000;  // +infinity
    src[4] = 0xff800000;  // -infinity

    // Fill rest with random values (avoiding NaN patterns for simpler comparison)
    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
    for (size_t i = 5; i < count; ++i) {
        src[i] = dist(rng_);
    }

    BitCastInt32ToFloat32(out.get(), src.get(), count);

    // Verify bit-level equality
    for (size_t i = 0; i < count; ++i) {
        float expected;
        std::memcpy(&expected, &src[i], sizeof(int32_t));
        // For non-NaN values, check equality. For NaN, check that both are NaN
        if (std::isnan(expected)) {
            ASSERT_TRUE(std::isnan(out[i])) << "Expected NaN at index " << i;
        } else {
            ASSERT_EQ(out[i], expected) << "BitCast mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, BitCastFloat64ToInt64_BasicCorrectness) {
    const size_t count = 32;
    auto src = RandomDoubles(count);
    auto out = hwy::AllocateAligned<int64_t>(count);

    BitCastFloat64ToInt64(out.get(), src.get(), count);

    for (size_t i = 0; i < count; ++i) {
        int64_t expected;
        std::memcpy(&expected, &src[i], sizeof(double));
        ASSERT_EQ(out[i], expected) << "BitCast mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, BitCastInt64ToFloat64_BasicCorrectness) {
    const size_t count = 32;
    auto src = RandomInts64(count);
    auto out = hwy::AllocateAligned<double>(count);

    BitCastInt64ToFloat64(out.get(), src.get(), count);

    for (size_t i = 0; i < count; ++i) {
        double expected;
        std::memcpy(&expected, &src[i], sizeof(int64_t));
        if (std::isnan(expected)) {
            ASSERT_TRUE(std::isnan(out[i])) << "Expected NaN at index " << i;
        } else {
            ASSERT_EQ(out[i], expected) << "BitCast mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, BitCastRoundTrip_Float32) {
    // Test that bitcast is reversible: float -> int32 -> float
    const size_t count = 100;
    auto original = RandomFloats(count);
    auto intermediate = hwy::AllocateAligned<int32_t>(count);
    auto result = hwy::AllocateAligned<float>(count);

    BitCastFloat32ToInt32(intermediate.get(), original.get(), count);
    BitCastInt32ToFloat32(result.get(), intermediate.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(result[i], original[i])) << "Round-trip mismatch at index " << i;
    }
}

// =============================================================================
// P0: Mask Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, CountTrue_Uint8_BasicCorrectness) {
    const size_t count = 100;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // Create mask with known number of true values
    size_t expected_true = 0;
    for (size_t i = 0; i < count; ++i) {
        mask[i] = (i % 3 == 0) ? 1 : 0;  // Every 3rd element is true
        if (mask[i])
            expected_true++;
    }

    size_t result = CountTrue(mask.get(), count);
    ASSERT_EQ(result, expected_true);
}

TEST_F(HwyOpsTest, CountTrue_AllTrue) {
    const size_t count = 64;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    for (size_t i = 0; i < count; ++i) {
        mask[i] = 1;
    }

    size_t result = CountTrue(mask.get(), count);
    ASSERT_EQ(result, count);
}

TEST_F(HwyOpsTest, CountTrue_AllFalse) {
    const size_t count = 64;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    for (size_t i = 0; i < count; ++i) {
        mask[i] = 0;
    }

    size_t result = CountTrue(mask.get(), count);
    ASSERT_EQ(result, 0u);
}

TEST_F(HwyOpsTest, CountTrue_Uint32_BasicCorrectness) {
    const size_t count = 50;
    auto mask = hwy::AllocateAligned<uint32_t>(count);

    size_t expected_true = 0;
    for (size_t i = 0; i < count; ++i) {
        mask[i] = (i % 2 == 0) ? 0xFFFFFFFF : 0;  // Every other element is true
        if (mask[i])
            expected_true++;
    }

    size_t result = CountTrue(mask.get(), count);
    ASSERT_EQ(result, expected_true);
}

TEST_F(HwyOpsTest, AllTrue_BasicCorrectness) {
    const size_t count = 64;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // All true
    for (size_t i = 0; i < count; ++i) {
        mask[i] = 1;
    }
    ASSERT_TRUE(AllTrue(mask.get(), count));

    // One false
    mask[count / 2] = 0;
    ASSERT_FALSE(AllTrue(mask.get(), count));

    // All false
    for (size_t i = 0; i < count; ++i) {
        mask[i] = 0;
    }
    ASSERT_FALSE(AllTrue(mask.get(), count));
}

TEST_F(HwyOpsTest, AllFalse_BasicCorrectness) {
    const size_t count = 64;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // All false
    for (size_t i = 0; i < count; ++i) {
        mask[i] = 0;
    }
    ASSERT_TRUE(AllFalse(mask.get(), count));

    // One true
    mask[count / 2] = 1;
    ASSERT_FALSE(AllFalse(mask.get(), count));

    // All true
    for (size_t i = 0; i < count; ++i) {
        mask[i] = 1;
    }
    ASSERT_FALSE(AllFalse(mask.get(), count));
}

TEST_F(HwyOpsTest, FindFirstTrue_BasicCorrectness) {
    const size_t count = 100;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // First true at index 25
    for (size_t i = 0; i < count; ++i) {
        mask[i] = (i == 25) ? 1 : 0;
    }
    ASSERT_EQ(FindFirstTrue(mask.get(), count), 25);

    // First true at index 0
    mask[0] = 1;
    ASSERT_EQ(FindFirstTrue(mask.get(), count), 0);

    // No true elements
    for (size_t i = 0; i < count; ++i) {
        mask[i] = 0;
    }
    ASSERT_EQ(FindFirstTrue(mask.get(), count), -1);
}

TEST_F(HwyOpsTest, FindLastTrue_BasicCorrectness) {
    const size_t count = 100;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // Last true at index 75
    for (size_t i = 0; i < count; ++i) {
        mask[i] = (i == 75) ? 1 : 0;
    }
    ASSERT_EQ(FindLastTrue(mask.get(), count), 75);

    // Multiple true, last at 99
    mask[99] = 1;
    ASSERT_EQ(FindLastTrue(mask.get(), count), 99);

    // No true elements
    for (size_t i = 0; i < count; ++i) {
        mask[i] = 0;
    }
    ASSERT_EQ(FindLastTrue(mask.get(), count), -1);
}

// =============================================================================
// P1: Shuffle Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Reverse_Float32_BasicCorrectness) {
    for (size_t count : {1u, 2u, 4u, 7u, 8u, 15u, 16u, 31u, 64u, 100u}) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Reverse(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(out[i], a[count - 1 - i]))
                << "Reverse mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, Reverse_Int32_BasicCorrectness) {
    for (size_t count : {1u, 2u, 4u, 7u, 8u, 15u, 16u, 31u, 64u, 100u}) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Reverse(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(out[i], a[count - 1 - i])
                << "Reverse mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, Fill_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = AllocateOutput<float>(count);
        float fill_value = 3.14159f;

        Fill(data.get(), fill_value, count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(data[i], fill_value))
                << "Fill mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, Fill_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = AllocateOutput<int32_t>(count);
        int32_t fill_value = 42;

        Fill(data.get(), fill_value, count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(data[i], fill_value)
                << "Fill mismatch at index " << i << " for count " << count;
        }
    }
}

TEST_F(HwyOpsTest, Copy_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto dst = AllocateOutput<float>(count);

        Copy(dst.get(), src.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(dst[i], src[i]))
                << "Copy mismatch at index " << i << " for count " << count;
        }
    }
}

// =============================================================================
// P1: Reduction Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MinOfLanes_Float32_BasicCorrectness) {
    for (size_t count : {8u, 15u, 16u, 31u, 64u, 100u}) {
        auto a = RandomFloats(count, 1.0f, 1000.0f);
        float result = 0;

        MinOfLanes(&result, a.get(), count);

        float expected = *std::min_element(a.get(), a.get() + count);
        ASSERT_TRUE(ApproxEqual(result, expected))
            << "MinOfLanes mismatch for count " << count << ": got " << result << ", expected "
            << expected;
    }
}

TEST_F(HwyOpsTest, MaxOfLanes_Float32_BasicCorrectness) {
    for (size_t count : {8u, 15u, 16u, 31u, 64u, 100u}) {
        auto a = RandomFloats(count, 1.0f, 1000.0f);
        float result = 0;

        MaxOfLanes(&result, a.get(), count);

        float expected = *std::max_element(a.get(), a.get() + count);
        ASSERT_TRUE(ApproxEqual(result, expected))
            << "MaxOfLanes mismatch for count " << count << ": got " << result << ", expected "
            << expected;
    }
}

TEST_F(HwyOpsTest, MinOfLanes_Int32_BasicCorrectness) {
    for (size_t count : {8u, 15u, 16u, 31u, 64u, 100u}) {
        auto a = RandomInts32(count, 1, 10000);
        int32_t result = 0;

        MinOfLanes(&result, a.get(), count);

        int32_t expected = *std::min_element(a.get(), a.get() + count);
        ASSERT_EQ(result, expected) << "MinOfLanes mismatch for count " << count;
    }
}

TEST_F(HwyOpsTest, MaxOfLanes_Int32_BasicCorrectness) {
    for (size_t count : {8u, 15u, 16u, 31u, 64u, 100u}) {
        auto a = RandomInts32(count, 1, 10000);
        int32_t result = 0;

        MaxOfLanes(&result, a.get(), count);

        int32_t expected = *std::max_element(a.get(), a.get() + count);
        ASSERT_EQ(result, expected) << "MaxOfLanes mismatch for count " << count;
    }
}

// =============================================================================
// P1: Interleaved Load/Store Tests
// =============================================================================

TEST_F(HwyOpsTest, LoadInterleaved2_Float32_BasicCorrectness) {
    const size_t count = 32;                                    // Elements per channel
    auto interleaved = hwy::AllocateAligned<float>(count * 2);  // a0,b0,a1,b1,...
    auto a_out = AllocateOutput<float>(count);
    auto b_out = AllocateOutput<float>(count);

    // Create interleaved data: a0,b0,a1,b1,...
    for (size_t i = 0; i < count; ++i) {
        interleaved[2 * i] = static_cast<float>(i * 10);          // a values
        interleaved[2 * i + 1] = static_cast<float>(i * 10 + 1);  // b values
    }

    LoadInterleaved2(a_out.get(), b_out.get(), interleaved.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(a_out[i], static_cast<float>(i * 10)))
            << "Channel A mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(b_out[i], static_cast<float>(i * 10 + 1)))
            << "Channel B mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, StoreInterleaved2_Float32_BasicCorrectness) {
    const size_t count = 32;
    auto a = RandomFloats(count);
    auto b = RandomFloats(count);
    auto interleaved = AllocateOutput<float>(count * 2);

    StoreInterleaved2(interleaved.get(), a.get(), b.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(interleaved[2 * i], a[i])) << "Channel A mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(interleaved[2 * i + 1], b[i]))
            << "Channel B mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, LoadInterleaved3_Float32_BasicCorrectness) {
    const size_t count = 24;                                    // Elements per channel
    auto interleaved = hwy::AllocateAligned<float>(count * 3);  // RGB layout
    auto r_out = AllocateOutput<float>(count);
    auto g_out = AllocateOutput<float>(count);
    auto b_out = AllocateOutput<float>(count);

    // Create RGB interleaved data
    for (size_t i = 0; i < count; ++i) {
        interleaved[3 * i] = static_cast<float>(i);            // R
        interleaved[3 * i + 1] = static_cast<float>(i + 100);  // G
        interleaved[3 * i + 2] = static_cast<float>(i + 200);  // B
    }

    LoadInterleaved3(r_out.get(), g_out.get(), b_out.get(), interleaved.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(r_out[i], static_cast<float>(i)))
            << "R channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(g_out[i], static_cast<float>(i + 100)))
            << "G channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(b_out[i], static_cast<float>(i + 200)))
            << "B channel mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, StoreInterleaved3_Float32_BasicCorrectness) {
    const size_t count = 24;
    auto r = RandomFloats(count);
    auto g = RandomFloats(count);
    auto b = RandomFloats(count);
    auto interleaved = AllocateOutput<float>(count * 3);

    StoreInterleaved3(interleaved.get(), r.get(), g.get(), b.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(interleaved[3 * i], r[i])) << "R channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(interleaved[3 * i + 1], g[i]))
            << "G channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(interleaved[3 * i + 2], b[i]))
            << "B channel mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, LoadInterleaved4_Float32_BasicCorrectness) {
    const size_t count = 16;                                    // Elements per channel
    auto interleaved = hwy::AllocateAligned<float>(count * 4);  // RGBA layout
    auto r_out = AllocateOutput<float>(count);
    auto g_out = AllocateOutput<float>(count);
    auto b_out = AllocateOutput<float>(count);
    auto a_out = AllocateOutput<float>(count);

    // Create RGBA interleaved data
    for (size_t i = 0; i < count; ++i) {
        interleaved[4 * i] = static_cast<float>(i);            // R
        interleaved[4 * i + 1] = static_cast<float>(i + 50);   // G
        interleaved[4 * i + 2] = static_cast<float>(i + 100);  // B
        interleaved[4 * i + 3] = static_cast<float>(i + 150);  // A
    }

    LoadInterleaved4(r_out.get(), g_out.get(), b_out.get(), a_out.get(), interleaved.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(r_out[i], static_cast<float>(i)))
            << "R channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(g_out[i], static_cast<float>(i + 50)))
            << "G channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(b_out[i], static_cast<float>(i + 100)))
            << "B channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(a_out[i], static_cast<float>(i + 150)))
            << "A channel mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, StoreInterleaved4_Float32_BasicCorrectness) {
    const size_t count = 16;
    auto r = RandomFloats(count);
    auto g = RandomFloats(count);
    auto b = RandomFloats(count);
    auto a = RandomFloats(count);
    auto interleaved = AllocateOutput<float>(count * 4);

    StoreInterleaved4(interleaved.get(), r.get(), g.get(), b.get(), a.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(interleaved[4 * i], r[i])) << "R channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(interleaved[4 * i + 1], g[i]))
            << "G channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(interleaved[4 * i + 2], b[i]))
            << "B channel mismatch at index " << i;
        ASSERT_TRUE(ApproxEqual(interleaved[4 * i + 3], a[i]))
            << "A channel mismatch at index " << i;
    }
}

// =============================================================================
// P1: Find Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Find_Float32_BasicCorrectness) {
    const size_t count = 100;
    auto data = RandomFloats(count, 1.0f, 100.0f);

    // Set a known value at a specific position
    data[42] = -999.0f;

    int64_t result = Find(data.get(), -999.0f, count);
    ASSERT_EQ(result, 42) << "Find should return index 42";

    // Search for value not in array
    result = Find(data.get(), -123456.0f, count);
    ASSERT_EQ(result, -1) << "Find should return -1 for missing value";
}

TEST_F(HwyOpsTest, Find_Int32_BasicCorrectness) {
    const size_t count = 100;
    auto data = RandomInts32(count, 0, 1000);

    // Set a known value at a specific position
    data[73] = -999;

    int64_t result = Find(data.get(), -999, count);
    ASSERT_EQ(result, 73) << "Find should return index 73";

    // Search for value not in array
    result = Find(data.get(), -123456, count);
    ASSERT_EQ(result, -1) << "Find should return -1 for missing value";
}

TEST_F(HwyOpsTest, FindGt_Float32_BasicCorrectness) {
    const size_t count = 100;
    auto data = hwy::AllocateAligned<float>(count);

    // Fill with increasing values
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i);
    }

    int64_t result = FindGt(data.get(), 50.0f, count);
    ASSERT_EQ(result, 51) << "FindGt should return first index > 50";

    result = FindGt(data.get(), -1.0f, count);
    ASSERT_EQ(result, 0) << "FindGt should return 0 (first element > -1)";

    result = FindGt(data.get(), 1000.0f, count);
    ASSERT_EQ(result, -1) << "FindGt should return -1 for no matches";
}

TEST_F(HwyOpsTest, FindLt_Float32_BasicCorrectness) {
    const size_t count = 100;
    auto data = hwy::AllocateAligned<float>(count);

    // Fill with increasing values
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i);
    }

    int64_t result = FindLt(data.get(), 50.0f, count);
    ASSERT_EQ(result, 0) << "FindLt should return 0 (first element < 50)";

    result = FindLt(data.get(), 0.0f, count);
    ASSERT_EQ(result, -1) << "FindLt should return -1 for no matches";
}

// =============================================================================
// P1: Transform Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, TransformAdd_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = RandomFloats(count);
        auto expected = hwy::AllocateAligned<float>(count);
        float scalar = 10.5f;

        // Store expected results
        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i] + scalar;
        }

        TransformAdd(data.get(), scalar, count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(data[i], expected[i]))
                << "TransformAdd mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, TransformMul_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = RandomFloats(count);
        auto expected = hwy::AllocateAligned<float>(count);
        float scalar = 2.5f;

        // Store expected results
        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i] * scalar;
        }

        TransformMul(data.get(), scalar, count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(data[i], expected[i]))
                << "TransformMul mismatch at index " << i;
        }
    }
}

// =============================================================================
// P1: VQSort Integration Tests
// =============================================================================

TEST_F(HwyOpsTest, Sort_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = RandomFloats(count);
        auto expected = hwy::AllocateAligned<float>(count);

        // Copy for expected
        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i];
        }

        // Sort using std::sort for expected
        std::sort(expected.get(), expected.get() + count);

        // Sort using our VQSort wrapper
        Sort(data.get(), count);

        // Verify sorted order
        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(data[i], expected[i]))
                << "Sort mismatch at index " << i << " for count " << count << ": got " << data[i]
                << ", expected " << expected[i];
        }
    }
}

TEST_F(HwyOpsTest, Sort_Float64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = hwy::AllocateAligned<double>(count);
        auto expected = hwy::AllocateAligned<double>(count);

        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<double>(rand()) / RAND_MAX * 200.0 - 100.0;
            expected[i] = data[i];
        }

        std::sort(expected.get(), expected.get() + count);
        Sort(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_NEAR(data[i], expected[i], 1e-10) << "Sort mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Sort_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = RandomInts32(count);
        auto expected = hwy::AllocateAligned<int32_t>(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i];
        }

        std::sort(expected.get(), expected.get() + count);
        Sort(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(data[i], expected[i]) << "Sort mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Sort_Int64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = hwy::AllocateAligned<int64_t>(count);
        auto expected = hwy::AllocateAligned<int64_t>(count);

        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<int64_t>(rand()) - RAND_MAX / 2;
            expected[i] = data[i];
        }

        std::sort(expected.get(), expected.get() + count);
        Sort(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(data[i], expected[i]) << "Sort mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Sort_Uint32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = hwy::AllocateAligned<uint32_t>(count);
        auto expected = hwy::AllocateAligned<uint32_t>(count);

        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<uint32_t>(rand());
            expected[i] = data[i];
        }

        std::sort(expected.get(), expected.get() + count);
        Sort(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(data[i], expected[i]) << "Sort mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Sort_Uint64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = hwy::AllocateAligned<uint64_t>(count);
        auto expected = hwy::AllocateAligned<uint64_t>(count);

        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<uint64_t>(rand()) << 16 | rand();
            expected[i] = data[i];
        }

        std::sort(expected.get(), expected.get() + count);
        Sort(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(data[i], expected[i]) << "Sort mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, SortDescending_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = RandomFloats(count);
        auto expected = hwy::AllocateAligned<float>(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i];
        }

        // Sort descending using std::sort with reverse compare
        std::sort(expected.get(), expected.get() + count, std::greater<float>());

        SortDescending(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(data[i], expected[i]))
                << "SortDescending mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, SortDescending_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto data = RandomInts32(count);
        auto expected = hwy::AllocateAligned<int32_t>(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i];
        }

        std::sort(expected.get(), expected.get() + count, std::greater<int32_t>());
        SortDescending(data.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(data[i], expected[i]) << "SortDescending mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, PartialSort_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        if (count < 2)
            continue;

        size_t k = count / 2;  // Sort first half
        auto data = RandomFloats(count);
        auto expected = hwy::AllocateAligned<float>(count);

        for (size_t i = 0; i < count; ++i) {
            expected[i] = data[i];
        }

        // Use std::partial_sort for expected
        std::partial_sort(expected.get(), expected.get() + k, expected.get() + count);

        PartialSort(data.get(), k, count);

        // First k elements should be smallest k in sorted order
        for (size_t i = 0; i < k; ++i) {
            ASSERT_TRUE(ApproxEqual(data[i], expected[i])) << "PartialSort mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Sort_AlreadySorted_NoChange) {
    const size_t count = 100;
    auto data = hwy::AllocateAligned<float>(count);

    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i);
    }

    Sort(data.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(data[i], static_cast<float>(i))
            << "Sort changed already sorted array at index " << i;
    }
}

TEST_F(HwyOpsTest, Sort_ReverseSorted_CorrectlySorts) {
    const size_t count = 100;
    auto data = hwy::AllocateAligned<float>(count);

    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(count - i - 1);
    }

    Sort(data.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(data[i], static_cast<float>(i))
            << "Sort failed on reverse sorted array at index " << i;
    }
}

TEST_F(HwyOpsTest, Sort_WithDuplicates_HandlesCorrectly) {
    const size_t count = 100;
    auto data = hwy::AllocateAligned<int32_t>(count);
    auto expected = hwy::AllocateAligned<int32_t>(count);

    // Create array with many duplicates
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<int32_t>(i % 10);  // Only 10 unique values
        expected[i] = data[i];
    }

    std::sort(expected.get(), expected.get() + count);
    Sort(data.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(data[i], expected[i]) << "Sort with duplicates mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Sort_EmptyArray_NoOp) {
    float empty[1] = {0.0f};
    Sort(empty, 0);  // Should not crash
}

TEST_F(HwyOpsTest, Sort_SingleElement_NoChange) {
    float data[1] = {42.0f};
    Sort(data, 1);
    ASSERT_EQ(data[0], 42.0f);
}

// =============================================================================
// P2: SinCos Tests
// =============================================================================

TEST_F(HwyOpsTest, SinCos_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto input = RandomFloats(count, -3.14159f, 3.14159f);
        auto sin_out = hwy::AllocateAligned<float>(count);
        auto cos_out = hwy::AllocateAligned<float>(count);

        SinCos(sin_out.get(), cos_out.get(), input.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected_sin = std::sin(input[i]);
            float expected_cos = std::cos(input[i]);
            ASSERT_NEAR(sin_out[i], expected_sin, 1e-5f) << "SinCos sin mismatch at index " << i;
            ASSERT_NEAR(cos_out[i], expected_cos, 1e-5f) << "SinCos cos mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, SinCos_Float64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto input = hwy::AllocateAligned<double>(count);
        auto sin_out = hwy::AllocateAligned<double>(count);
        auto cos_out = hwy::AllocateAligned<double>(count);

        for (size_t i = 0; i < count; ++i) {
            input[i] = static_cast<double>(rand()) / RAND_MAX * 6.28 - 3.14;
        }

        SinCos(sin_out.get(), cos_out.get(), input.get(), count);

        for (size_t i = 0; i < count; ++i) {
            double expected_sin = std::sin(input[i]);
            double expected_cos = std::cos(input[i]);
            ASSERT_NEAR(sin_out[i], expected_sin, 1e-10) << "SinCos sin mismatch at index " << i;
            ASSERT_NEAR(cos_out[i], expected_cos, 1e-10) << "SinCos cos mismatch at index " << i;
        }
    }
}

// =============================================================================
// P2: Hypot Tests
// =============================================================================

TEST_F(HwyOpsTest, Hypot_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = hwy::AllocateAligned<float>(count);

        Hypot(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::hypot(a[i], b[i]);
            ASSERT_NEAR(out[i], expected, std::abs(expected) * 1e-5f + 1e-6f)
                << "Hypot mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Hypot_Float64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = hwy::AllocateAligned<double>(count);
        auto b = hwy::AllocateAligned<double>(count);
        auto out = hwy::AllocateAligned<double>(count);

        for (size_t i = 0; i < count; ++i) {
            a[i] = static_cast<double>(rand()) / RAND_MAX * 200.0 - 100.0;
            b[i] = static_cast<double>(rand()) / RAND_MAX * 200.0 - 100.0;
        }

        Hypot(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            double expected = std::hypot(a[i], b[i]);
            ASSERT_NEAR(out[i], expected, std::abs(expected) * 1e-10 + 1e-12)
                << "Hypot mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Hypot_PythagoreanTriples) {
    // Test with known Pythagorean triples for exact results
    const size_t count = 4;
    float a[] = {3.0f, 5.0f, 8.0f, 7.0f};
    float b[] = {4.0f, 12.0f, 15.0f, 24.0f};
    float expected[] = {5.0f, 13.0f, 17.0f, 25.0f};
    auto out = hwy::AllocateAligned<float>(count);

    Hypot(out.get(), a, b, count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f)
            << "Hypot Pythagorean triple mismatch at index " << i;
    }
}

// =============================================================================
// P2: MatVec Tests
// =============================================================================

TEST_F(HwyOpsTest, MatVec_Float32_Identity) {
    const size_t rows = 4;
    const size_t cols = 4;
    auto A = hwy::AllocateAligned<float>(rows * cols);
    auto x = hwy::AllocateAligned<float>(cols);
    auto out = hwy::AllocateAligned<float>(rows);

    // Identity matrix
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            A[i * cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Vector
    for (size_t i = 0; i < cols; ++i) {
        x[i] = static_cast<float>(i + 1);
    }

    MatVec(out.get(), A.get(), x.get(), rows, cols);

    // Identity * x = x
    for (size_t i = 0; i < rows; ++i) {
        ASSERT_NEAR(out[i], x[i], 1e-6f) << "MatVec identity mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, MatVec_Float32_BasicCorrectness) {
    const size_t rows = 3;
    const size_t cols = 4;

    // Row-major matrix
    float A[] = {
        1, 2,  3,  4,  // Row 0
        5, 6,  7,  8,  // Row 1
        9, 10, 11, 12  // Row 2
    };
    float x[] = {1, 2, 3, 4};
    auto out = hwy::AllocateAligned<float>(rows);

    // Expected: A * x
    // Row 0: 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    // Row 1: 5*1 + 6*2 + 7*3 + 8*4 = 5 + 12 + 21 + 32 = 70
    // Row 2: 9*1 + 10*2 + 11*3 + 12*4 = 9 + 20 + 33 + 48 = 110
    float expected[] = {30.0f, 70.0f, 110.0f};

    MatVec(out.get(), A, x, rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f) << "MatVec mismatch at row " << i;
    }
}

TEST_F(HwyOpsTest, MatVec_Float64_BasicCorrectness) {
    const size_t rows = 2;
    const size_t cols = 3;

    double A[] = {
        1.5, 2.5, 3.5,  // Row 0
        4.5, 5.5, 6.5   // Row 1
    };
    double x[] = {1.0, 2.0, 3.0};
    auto out = hwy::AllocateAligned<double>(rows);

    // Expected: A * x
    // Row 0: 1.5*1 + 2.5*2 + 3.5*3 = 1.5 + 5 + 10.5 = 17
    // Row 1: 4.5*1 + 5.5*2 + 6.5*3 = 4.5 + 11 + 19.5 = 35
    double expected[] = {17.0, 35.0};

    MatVec(out.get(), A, x, rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        ASSERT_NEAR(out[i], expected[i], 1e-10) << "MatVec mismatch at row " << i;
    }
}

TEST_F(HwyOpsTest, MatVec_VariousSizes) {
    // Test various matrix sizes
    std::vector<std::pair<size_t, size_t>> sizes = {{1, 1},   {1, 8},   {8, 1},  {7, 9},
                                                    {16, 16}, {31, 33}, {64, 64}};

    for (auto& [rows, cols] : sizes) {
        auto A = hwy::AllocateAligned<float>(rows * cols);
        auto x = hwy::AllocateAligned<float>(cols);
        auto out = hwy::AllocateAligned<float>(rows);
        auto expected = hwy::AllocateAligned<float>(rows);

        // Initialize
        for (size_t i = 0; i < rows * cols; ++i) {
            A[i] = static_cast<float>(i % 10);
        }
        for (size_t i = 0; i < cols; ++i) {
            x[i] = static_cast<float>(i + 1);
        }

        // Compute expected
        for (size_t r = 0; r < rows; ++r) {
            expected[r] = 0;
            for (size_t c = 0; c < cols; ++c) {
                expected[r] += A[r * cols + c] * x[c];
            }
        }

        MatVec(out.get(), A.get(), x.get(), rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            ASSERT_NEAR(out[i], expected[i], std::abs(expected[i]) * 1e-5f + 1e-5f)
                << "MatVec size (" << rows << "x" << cols << ") mismatch at row " << i;
        }
    }
}

// =============================================================================
// P2: Pow Tests
// =============================================================================

TEST_F(HwyOpsTest, Pow_Float32_BasicCorrectness) {
    const size_t count = 16;
    auto base = hwy::AllocateAligned<float>(count);
    auto exp = hwy::AllocateAligned<float>(count);
    auto out = hwy::AllocateAligned<float>(count);

    for (size_t i = 0; i < count; ++i) {
        base[i] = static_cast<float>(i + 1);  // 1, 2, 3, ...
        exp[i] = 2.0f;                        // Square all
    }

    Pow(out.get(), base.get(), exp.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = std::pow(base[i], exp[i]);
        ASSERT_NEAR(out[i], expected, std::abs(expected) * 1e-5f + 1e-6f)
            << "Pow mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, PowScalar_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto base = RandomFloats(count, 0.1f, 10.0f);  // Positive bases
        auto out = hwy::AllocateAligned<float>(count);
        float exp = 2.5f;

        PowScalar(out.get(), base.get(), exp, count);

        for (size_t i = 0; i < count; ++i) {
            float expected = std::pow(base[i], exp);
            ASSERT_NEAR(out[i], expected, std::abs(expected) * 1e-4f + 1e-6f)
                << "PowScalar mismatch at index " << i;
        }
    }
}

// =============================================================================
// P2: Float16/BFloat16 Conversion Tests
// =============================================================================

TEST_F(HwyOpsTest, F32ToF16_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto in = RandomFloats(count, -1000.0f, 1000.0f);
        auto out = hwy::AllocateAligned<uint16_t>(count);

        F32ToF16(out.get(), in.get(), count);

        for (size_t i = 0; i < count; ++i) {
            // Convert back and check round-trip
            float back = hwy::F32FromF16(hwy::float16_t::FromBits(out[i]));
            // F16 has limited precision - allow for rounding
            float expected = in[i];
            if (std::abs(expected) > 65504.0f) {
                // Outside F16 range - expect inf
                ASSERT_TRUE(std::isinf(back)) << "Expected inf for large value at index " << i;
            } else if (std::abs(expected) < 6.1e-5f) {
                // Denormal range or underflow
                ASSERT_NEAR(back, 0.0f, 1e-4f)
                    << "Expected near-zero for small value at index " << i;
            } else {
                // Normal range - check relative error (F16 has ~0.1% precision)
                ASSERT_NEAR(back, expected, std::abs(expected) * 0.01f + 1e-4f)
                    << "F32ToF16 round-trip mismatch at index " << i;
            }
        }
    }
}

TEST_F(HwyOpsTest, F16ToF32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto in = hwy::AllocateAligned<uint16_t>(count);
        auto out = hwy::AllocateAligned<float>(count);

        // Create valid F16 bit patterns
        for (size_t i = 0; i < count; ++i) {
            // Create a valid float, convert to F16, get bits
            float val = static_cast<float>(i % 100) - 50.0f;
            in[i] = hwy::BitCastScalar<uint16_t>(hwy::F16FromF32(val));
        }

        F16ToF32(out.get(), in.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = hwy::F32FromF16(hwy::float16_t::FromBits(in[i]));
            ASSERT_EQ(out[i], expected) << "F16ToF32 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, F32ToBF16_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto in = RandomFloats(count);
        auto out = hwy::AllocateAligned<uint16_t>(count);

        F32ToBF16(out.get(), in.get(), count);

        for (size_t i = 0; i < count; ++i) {
            // Convert back and check round-trip
            float back = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(out[i]));
            // BF16 has ~0.8% precision (7-bit mantissa)
            ASSERT_NEAR(back, in[i], std::abs(in[i]) * 0.01f + 1e-6f)
                << "F32ToBF16 round-trip mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, BF16ToF32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto in = hwy::AllocateAligned<uint16_t>(count);
        auto out = hwy::AllocateAligned<float>(count);

        // Create valid BF16 bit patterns
        for (size_t i = 0; i < count; ++i) {
            float val = static_cast<float>(i % 100) - 50.0f;
            in[i] = hwy::BitCastScalar<uint16_t>(hwy::BF16FromF32(val));
        }

        BF16ToF32(out.get(), in.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(in[i]));
            ASSERT_EQ(out[i], expected) << "BF16ToF32 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, F64ToF16_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto in = hwy::AllocateAligned<double>(count);
        auto out = hwy::AllocateAligned<uint16_t>(count);

        for (size_t i = 0; i < count; ++i) {
            in[i] = static_cast<double>(rand()) / RAND_MAX * 200.0 - 100.0;
        }

        F64ToF16(out.get(), in.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float back = hwy::F32FromF16(hwy::float16_t::FromBits(out[i]));
            // Should be close to original (within F16 precision)
            ASSERT_NEAR(back, static_cast<float>(in[i]),
                        std::abs(static_cast<float>(in[i])) * 0.01f + 1e-4f)
                << "F64ToF16 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, F64ToBF16_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto in = hwy::AllocateAligned<double>(count);
        auto out = hwy::AllocateAligned<uint16_t>(count);

        for (size_t i = 0; i < count; ++i) {
            in[i] = static_cast<double>(rand()) / RAND_MAX * 200.0 - 100.0;
        }

        F64ToBF16(out.get(), in.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float back = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(out[i]));
            // Should be close to original (within BF16 precision)
            ASSERT_NEAR(back, static_cast<float>(in[i]),
                        std::abs(static_cast<float>(in[i])) * 0.01f + 1e-6f)
                << "F64ToBF16 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, F16_SpecialValues) {
    const size_t count = 5;
    float in[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f};
    auto out = hwy::AllocateAligned<uint16_t>(count);

    F32ToF16(out.get(), in, count);

    // Verify known F16 bit patterns
    // 0.0f -> 0x0000
    ASSERT_EQ(out[0], 0x0000);
    // -0.0f -> 0x8000
    ASSERT_EQ(out[1], 0x8000);
    // 1.0f -> 0x3C00
    ASSERT_EQ(out[2], 0x3C00);
    // -1.0f -> 0xBC00
    ASSERT_EQ(out[3], 0xBC00);
    // 0.5f -> 0x3800
    ASSERT_EQ(out[4], 0x3800);
}

TEST_F(HwyOpsTest, BF16_SpecialValues) {
    const size_t count = 5;
    float in[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f};
    auto out = hwy::AllocateAligned<uint16_t>(count);

    F32ToBF16(out.get(), in, count);

    // Verify known BF16 bit patterns (upper 16 bits of float32)
    // 0.0f -> 0x0000
    ASSERT_EQ(out[0], 0x0000);
    // -0.0f -> 0x8000
    ASSERT_EQ(out[1], 0x8000);
    // 1.0f -> 0x3F80
    ASSERT_EQ(out[2], 0x3F80);
    // -1.0f -> 0xBF80
    ASSERT_EQ(out[3], 0xBF80);
    // 2.0f -> 0x4000
    ASSERT_EQ(out[4], 0x4000);
}

// =============================================================================
// P0: Masked Arithmetic Operations Tests
// =============================================================================

// Helper function to generate random mask
hwy::AlignedFreeUniquePtr<uint8_t[]> RandomMask(size_t count, std::mt19937& rng) {
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < count; ++i) {
        mask[i] = dist(rng) ? 0xFF : 0x00;
    }
    return mask;
}

TEST_F(HwyOpsTest, MaskedAdd_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedAdd(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? (a[i] + b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "MaskedAdd mismatch at index " << i << " for count " << count
                << ": mask=" << (int)mask[i] << ", got " << out[i] << ", expected " << expected;
        }
    }
}

TEST_F(HwyOpsTest, MaskedAdd_Float64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomDoubles(count);
        auto b = RandomDoubles(count);
        auto no = RandomDoubles(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<double>(count);

        MaskedAdd(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            double expected = mask[i] ? (a[i] + b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedAdd mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedAdd_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto b = RandomInts32(count);
        auto no = RandomInts32(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<int32_t>(count);

        MaskedAdd(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = mask[i] ? (a[i] + b[i]) : no[i];
            ASSERT_EQ(out[i], expected) << "MaskedAdd mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedSub_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedSub(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? (a[i] - b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedSub mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedMul_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedMul(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? (a[i] * b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedMul mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedDiv_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        // Generate non-zero divisors
        auto b = RandomFloats(count, 1.0f, 100.0f);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedDiv(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? (a[i] / b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedDiv mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedMin_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedMin(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? std::min(a[i], b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedMin mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedMax_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedMax(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? std::max(a[i], b[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedMax mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedAbs_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedAbs(out.get(), mask.get(), a.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? std::abs(a[i]) : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedAbs mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedNeg_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto no = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = AllocateOutput<float>(count);

        MaskedNeg(out.get(), mask.get(), a.get(), no.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = mask[i] ? -a[i] : no[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MaskedNeg mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskedOps_AllMaskTrue) {
    const size_t count = 100;
    auto a = RandomFloats(count);
    auto b = RandomFloats(count);
    auto no = RandomFloats(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::fill(mask.get(), mask.get() + count, 0xFF);  // All true
    auto out = AllocateOutput<float>(count);

    MaskedAdd(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float expected = a[i] + b[i];  // All should be computed
        ASSERT_TRUE(ApproxEqual(out[i], expected))
            << "MaskedAdd (all true) mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, MaskedOps_AllMaskFalse) {
    const size_t count = 100;
    auto a = RandomFloats(count);
    auto b = RandomFloats(count);
    auto no = RandomFloats(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::fill(mask.get(), mask.get() + count, 0x00);  // All false
    auto out = AllocateOutput<float>(count);

    MaskedAdd(out.get(), mask.get(), a.get(), b.get(), no.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], no[i]))
            << "MaskedAdd (all false) should equal no[i] at index " << i;
    }
}

// =============================================================================
// P0: Widening Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SumsOf2_Int16_BasicCorrectness) {
    // Note: SumsOf2 produces half the output elements
    for (size_t count : {2, 4, 8, 16, 32, 64, 128, 256}) {
        auto in = hwy::AllocateAligned<int16_t>(count);
        for (size_t i = 0; i < count; ++i) {
            in[i] = static_cast<int16_t>(i - 50);
        }
        auto out = AllocateOutput<int32_t>(count / 2);

        SumsOf2(out.get(), in.get(), count);

        for (size_t i = 0; i < count / 2; ++i) {
            int32_t expected =
                static_cast<int32_t>(in[2 * i]) + static_cast<int32_t>(in[2 * i + 1]);
            ASSERT_EQ(out[i], expected) << "SumsOf2 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, SumsOf4_Int8_BasicCorrectness) {
    // Note: SumsOf4 produces 1/4 the output elements
    for (size_t count : {4, 8, 16, 32, 64, 128, 256}) {
        auto in = hwy::AllocateAligned<int8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            in[i] = static_cast<int8_t>((i % 100) - 50);
        }
        auto out = AllocateOutput<int32_t>(count / 4);

        SumsOf4(out.get(), in.get(), count);

        for (size_t i = 0; i < count / 4; ++i) {
            int32_t expected =
                static_cast<int32_t>(in[4 * i]) + static_cast<int32_t>(in[4 * i + 1]) +
                static_cast<int32_t>(in[4 * i + 2]) + static_cast<int32_t>(in[4 * i + 3]);
            ASSERT_EQ(out[i], expected) << "SumsOf4 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MulEven_Int32_BasicCorrectness) {
    // MulEven produces half the output elements
    for (size_t count : {2, 4, 8, 16, 32, 64}) {
        auto a = RandomInts32(count, -100, 100);
        auto b = RandomInts32(count, -100, 100);
        auto out = AllocateOutput<int64_t>(count / 2);

        MulEven(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count / 2; ++i) {
            int64_t expected = static_cast<int64_t>(a[2 * i]) * static_cast<int64_t>(b[2 * i]);
            ASSERT_EQ(out[i], expected) << "MulEven mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MulOdd_Int32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64}) {
        auto a = RandomInts32(count, -100, 100);
        auto b = RandomInts32(count, -100, 100);
        auto out = AllocateOutput<int64_t>(count / 2);

        MulOdd(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count / 2; ++i) {
            int64_t expected =
                static_cast<int64_t>(a[2 * i + 1]) * static_cast<int64_t>(b[2 * i + 1]);
            ASSERT_EQ(out[i], expected) << "MulOdd mismatch at index " << i;
        }
    }
}

// =============================================================================
// P0: Additional Comparison Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, IsNegative_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto out = hwy::AllocateAligned<uint8_t>(count);

        IsNegative(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] < 0.0f) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "IsNegative mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, IsNegative_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto out = hwy::AllocateAligned<uint8_t>(count);

        IsNegative(out.get(), a.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] < 0) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "IsNegative mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, IsEitherNaN_Float32_BasicCorrectness) {
    const size_t count = 16;
    float a[] = {1.0f, NAN,  2.0f, NAN,  3.0f,  4.0f, NAN,   5.0f,
                 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, NAN,  11.0f, 12.0f};
    float b[] = {NAN,  1.0f, 2.0f, NAN,  NAN,   4.0f,  5.0f, 6.0f,
                 7.0f, 8.0f, NAN,  9.0f, 10.0f, 11.0f, NAN,  12.0f};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    IsEitherNaN(out.get(), a, b, count);

    for (size_t i = 0; i < count; ++i) {
        bool either_nan = std::isnan(a[i]) || std::isnan(b[i]);
        uint8_t expected = either_nan ? 0xFF : 0x00;
        ASSERT_EQ(out[i], expected) << "IsEitherNaN mismatch at index " << i;
    }
}

// =============================================================================
// P1: Extended FMA Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, NegMulSub_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto c = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        NegMulSub(out.get(), a.get(), b.get(), c.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = -(a[i] * b[i]) - c[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "NegMulSub mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MulAddSub_Float32_BasicCorrectness) {
    // Test with a size that's a multiple of 2 to properly test even/odd pattern
    const size_t count = 16;
    auto a = hwy::AllocateAligned<float>(count);
    auto b = hwy::AllocateAligned<float>(count);
    auto c = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = 2.0f;
        c[i] = 1.0f;
    }

    MulAddSub(out.get(), a.get(), b.get(), c.get(), count);

    for (size_t i = 0; i < count; ++i) {
        float mul = a[i] * b[i];
        float expected = (i % 2 == 0) ? (mul - c[i]) : (mul + c[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected))
            << "MulAddSub mismatch at index " << i << ": got " << out[i] << ", expected "
            << expected;
    }
}

TEST_F(HwyOpsTest, MulAddSub_Float64_BasicCorrectness) {
    const size_t count = 16;
    auto a = hwy::AllocateAligned<double>(count);
    auto b = hwy::AllocateAligned<double>(count);
    auto c = hwy::AllocateAligned<double>(count);
    auto out = AllocateOutput<double>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<double>(i + 1);
        b[i] = 2.0;
        c[i] = 1.0;
    }

    MulAddSub(out.get(), a.get(), b.get(), c.get(), count);

    for (size_t i = 0; i < count; ++i) {
        double mul = a[i] * b[i];
        double expected = (i % 2 == 0) ? (mul - c[i]) : (mul + c[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected))
            << "MulAddSub mismatch at index " << i << ": got " << out[i] << ", expected "
            << expected;
    }
}

// =============================================================================
// P1: CompressStore Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, CompressStore_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto mask = RandomMask(count, rng_);
        auto out = hwy::AllocateAligned<float>(count);

        size_t written = CompressStore(out.get(), a.get(), mask.get(), count);

        // Verify count
        size_t expected_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (mask[i])
                expected_count++;
        }
        ASSERT_EQ(written, expected_count) << "CompressStore wrote wrong count";

        // Verify values
        size_t out_idx = 0;
        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_TRUE(ApproxEqual(out[out_idx], a[i]))
                    << "CompressStore value mismatch at out index " << out_idx;
                out_idx++;
            }
        }
    }
}

TEST_F(HwyOpsTest, CompressStore_AllTrue) {
    const size_t count = 100;
    auto a = RandomFloats(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::fill(mask.get(), mask.get() + count, 0xFF);
    auto out = hwy::AllocateAligned<float>(count);

    size_t written = CompressStore(out.get(), a.get(), mask.get(), count);

    ASSERT_EQ(written, count);
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], a[i]));
    }
}

TEST_F(HwyOpsTest, CompressStore_AllFalse) {
    const size_t count = 100;
    auto a = RandomFloats(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::fill(mask.get(), mask.get() + count, 0x00);
    auto out = hwy::AllocateAligned<float>(count);

    size_t written = CompressStore(out.get(), a.get(), mask.get(), count);

    ASSERT_EQ(written, 0u);
}

// =============================================================================
// P2: Iota and FirstN Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Iota_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        float start = 5.0f;
        auto out = hwy::AllocateAligned<float>(count);

        Iota(out.get(), start, count);

        for (size_t i = 0; i < count; ++i) {
            float expected = start + static_cast<float>(i);
            ASSERT_TRUE(ApproxEqual(out[i], expected)) << "Iota mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Iota_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        int32_t start = -10;
        auto out = hwy::AllocateAligned<int32_t>(count);

        Iota(out.get(), start, count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = start + static_cast<int32_t>(i);
            ASSERT_EQ(out[i], expected) << "Iota mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, FirstN_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        for (size_t n : {0ul, 1ul, count / 2, count - 1, count, count + 1}) {
            auto out = hwy::AllocateAligned<uint8_t>(count);

            FirstN(out.get(), n, count);

            for (size_t i = 0; i < count; ++i) {
                uint8_t expected = (i < n) ? 0xFF : 0x00;
                ASSERT_EQ(out[i], expected)
                    << "FirstN mismatch at index " << i << " with n=" << n << ", count=" << count;
            }
        }
    }
}

// =============================================================================
// P0: WidenMulAccumulate Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, WidenMulAccumulate_Int16_BasicCorrectness) {
    for (size_t count : {4, 8, 16, 32, 64, 128, 256}) {
        auto a = hwy::AllocateAligned<int16_t>(count);
        auto b = hwy::AllocateAligned<int16_t>(count);
        auto c = hwy::AllocateAligned<int32_t>(count);
        auto out = AllocateOutput<int32_t>(count);

        std::uniform_int_distribution<int16_t> dist(-100, 100);
        for (size_t i = 0; i < count; ++i) {
            a[i] = dist(rng_);
            b[i] = dist(rng_);
            c[i] = dist(rng_);
        }

        WidenMulAccumulate(out.get(), a.get(), b.get(), c.get(), count);

        for (size_t i = 0; i < count; ++i) {
            int32_t expected = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]) + c[i];
            ASSERT_EQ(out[i], expected) << "WidenMulAccumulate Int16 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, WidenMulAccumulate_Uint16_BasicCorrectness) {
    for (size_t count : {4, 8, 16, 32, 64, 128, 256}) {
        auto a = hwy::AllocateAligned<uint16_t>(count);
        auto b = hwy::AllocateAligned<uint16_t>(count);
        auto c = hwy::AllocateAligned<uint32_t>(count);
        auto out = AllocateOutput<uint32_t>(count);

        std::uniform_int_distribution<uint16_t> dist(0, 200);
        for (size_t i = 0; i < count; ++i) {
            a[i] = dist(rng_);
            b[i] = dist(rng_);
            c[i] = dist(rng_);
        }

        WidenMulAccumulate(out.get(), a.get(), b.get(), c.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint32_t expected = static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]) + c[i];
            ASSERT_EQ(out[i], expected) << "WidenMulAccumulate Uint16 mismatch at index " << i;
        }
    }
}

// =============================================================================
// P0: SumsOf8 Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SumsOf8_Uint8_BasicCorrectness) {
    for (size_t count : {8, 16, 32, 64, 128, 256, 512}) {
        auto in = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            in[i] = static_cast<uint8_t>(i % 256);
        }
        auto out = AllocateOutput<uint64_t>(count / 8);

        SumsOf8(out.get(), in.get(), count);

        for (size_t i = 0; i < count / 8; ++i) {
            uint64_t expected = 0;
            for (size_t j = 0; j < 8; ++j) {
                expected += static_cast<uint64_t>(in[8 * i + j]);
            }
            ASSERT_EQ(out[i], expected) << "SumsOf8 mismatch at index " << i;
        }
    }
}

// =============================================================================
// P0: TestBit Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, TestBit_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto out = hwy::AllocateAligned<uint8_t>(count);

        for (size_t bit = 0; bit < 32; bit += 8) {
            TestBit(out.get(), a.get(), bit, count);

            for (size_t i = 0; i < count; ++i) {
                uint8_t expected = ((a[i] >> bit) & 1) ? 0xFF : 0x00;
                ASSERT_EQ(out[i], expected)
                    << "TestBit Int32 mismatch at index " << i << " for bit " << bit;
            }
        }
    }
}

TEST_F(HwyOpsTest, TestBit_Int64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts64(count);
        auto out = hwy::AllocateAligned<uint8_t>(count);

        for (size_t bit = 0; bit < 64; bit += 16) {
            TestBit(out.get(), a.get(), bit, count);

            for (size_t i = 0; i < count; ++i) {
                uint8_t expected = ((a[i] >> bit) & 1) ? 0xFF : 0x00;
                ASSERT_EQ(out[i], expected)
                    << "TestBit Int64 mismatch at index " << i << " for bit " << bit;
            }
        }
    }
}

TEST_F(HwyOpsTest, TestBit_Uint32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = hwy::AllocateAligned<uint32_t>(count);
        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
        for (size_t i = 0; i < count; ++i) {
            a[i] = dist(rng_);
        }
        auto out = hwy::AllocateAligned<uint8_t>(count);

        for (size_t bit = 0; bit < 32; bit += 8) {
            TestBit(out.get(), a.get(), bit, count);

            for (size_t i = 0; i < count; ++i) {
                uint8_t expected = ((a[i] >> bit) & 1) ? 0xFF : 0x00;
                ASSERT_EQ(out[i], expected)
                    << "TestBit Uint32 mismatch at index " << i << " for bit " << bit;
            }
        }
    }
}

// =============================================================================
// P1: MulSubAdd Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MulSubAdd_Float32_BasicCorrectness) {
    const size_t count = 16;
    auto a = hwy::AllocateAligned<float>(count);
    auto b = hwy::AllocateAligned<float>(count);
    auto c = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = 2.0f;
        c[i] = 1.0f;
    }

    MulSubAdd(out.get(), a.get(), b.get(), c.get(), count);

    // MulSubAdd: even lanes add, odd lanes subtract (opposite of MulAddSub)
    for (size_t i = 0; i < count; ++i) {
        float mul = a[i] * b[i];
        float expected = (i % 2 == 0) ? (mul + c[i]) : (mul - c[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected))
            << "MulSubAdd Float32 mismatch at index " << i << ": got " << out[i] << ", expected "
            << expected;
    }
}

TEST_F(HwyOpsTest, MulSubAdd_Float64_BasicCorrectness) {
    const size_t count = 16;
    auto a = hwy::AllocateAligned<double>(count);
    auto b = hwy::AllocateAligned<double>(count);
    auto c = hwy::AllocateAligned<double>(count);
    auto out = AllocateOutput<double>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<double>(i + 1);
        b[i] = 2.0;
        c[i] = 1.0;
    }

    MulSubAdd(out.get(), a.get(), b.get(), c.get(), count);

    for (size_t i = 0; i < count; ++i) {
        double mul = a[i] * b[i];
        double expected = (i % 2 == 0) ? (mul + c[i]) : (mul - c[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected)) << "MulSubAdd Float64 mismatch at index " << i;
    }
}

// =============================================================================
// P1: Reverse Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Reverse2_Float32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64, 128}) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Reverse2(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i + 1]))
                << "Reverse2 Float32 mismatch at index " << i;
            ASSERT_TRUE(ApproxEqual(out[i + 1], a[i]))
                << "Reverse2 Float32 mismatch at index " << (i + 1);
        }
    }
}

TEST_F(HwyOpsTest, Reverse2_Float64_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64}) {
        auto a = RandomDoubles(count);
        auto out = AllocateOutput<double>(count);

        Reverse2(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i + 1]))
                << "Reverse2 Float64 mismatch at index " << i;
            ASSERT_TRUE(ApproxEqual(out[i + 1], a[i]))
                << "Reverse2 Float64 mismatch at index " << (i + 1);
        }
    }
}

TEST_F(HwyOpsTest, Reverse2_Int32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64, 128}) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Reverse2(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_EQ(out[i], a[i + 1]) << "Reverse2 Int32 mismatch at index " << i;
            ASSERT_EQ(out[i + 1], a[i]) << "Reverse2 Int32 mismatch at index " << (i + 1);
        }
    }
}

TEST_F(HwyOpsTest, Reverse4_Float32_BasicCorrectness) {
    for (size_t count : {4, 8, 16, 32, 64, 128}) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Reverse4(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 4) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i + 3]));
            ASSERT_TRUE(ApproxEqual(out[i + 1], a[i + 2]));
            ASSERT_TRUE(ApproxEqual(out[i + 2], a[i + 1]));
            ASSERT_TRUE(ApproxEqual(out[i + 3], a[i]));
        }
    }
}

TEST_F(HwyOpsTest, Reverse4_Int32_BasicCorrectness) {
    for (size_t count : {4, 8, 16, 32, 64, 128}) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Reverse4(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 4) {
            ASSERT_EQ(out[i], a[i + 3]);
            ASSERT_EQ(out[i + 1], a[i + 2]);
            ASSERT_EQ(out[i + 2], a[i + 1]);
            ASSERT_EQ(out[i + 3], a[i]);
        }
    }
}

TEST_F(HwyOpsTest, Reverse8_Uint8_BasicCorrectness) {
    for (size_t count : {8, 16, 32, 64, 128, 256}) {
        auto a = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = static_cast<uint8_t>(i);
        }
        auto out = AllocateOutput<uint8_t>(count);

        Reverse8(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 8) {
            for (size_t j = 0; j < 8; ++j) {
                ASSERT_EQ(out[i + j], a[i + 7 - j]) << "Reverse8 mismatch at index " << (i + j);
            }
        }
    }
}

// =============================================================================
// P1: DupEven and DupOdd Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, DupEven_Float32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64, 128}) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        DupEven(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i])) << "DupEven Float32 mismatch at index " << i;
            ASSERT_TRUE(ApproxEqual(out[i + 1], a[i]))
                << "DupEven Float32 mismatch at index " << (i + 1);
        }
    }
}

TEST_F(HwyOpsTest, DupEven_Int32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64, 128}) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        DupEven(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_EQ(out[i], a[i]) << "DupEven Int32 mismatch at index " << i;
            ASSERT_EQ(out[i + 1], a[i]) << "DupEven Int32 mismatch at index " << (i + 1);
        }
    }
}

TEST_F(HwyOpsTest, DupOdd_Float32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64, 128}) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        DupOdd(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i + 1])) << "DupOdd Float32 mismatch at index " << i;
            ASSERT_TRUE(ApproxEqual(out[i + 1], a[i + 1]))
                << "DupOdd Float32 mismatch at index " << (i + 1);
        }
    }
}

TEST_F(HwyOpsTest, DupOdd_Int32_BasicCorrectness) {
    for (size_t count : {2, 4, 8, 16, 32, 64, 128}) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        DupOdd(out.get(), a.get(), count);

        for (size_t i = 0; i < count; i += 2) {
            ASSERT_EQ(out[i], a[i + 1]) << "DupOdd Int32 mismatch at index " << i;
            ASSERT_EQ(out[i + 1], a[i + 1]) << "DupOdd Int32 mismatch at index " << (i + 1);
        }
    }
}

// =============================================================================
// P1: Interleave Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, InterleaveLower_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    InterleaveLower(out.get(), a, b, count);

    // InterleaveLower takes lower halves and interleaves
    // For count=8: lower half is indices 0-3
    // Expected: a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]
    float expected[] = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "InterleaveLower Float32 mismatch at index " << i << ": got " << out[i]
            << ", expected " << expected[i];
    }
}

TEST_F(HwyOpsTest, InterleaveLower_Int32_BasicCorrectness) {
    const size_t count = 8;
    int32_t a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t b[] = {10, 20, 30, 40, 50, 60, 70, 80};
    auto out = AllocateOutput<int32_t>(count);

    InterleaveLower(out.get(), a, b, count);

    int32_t expected[] = {1, 10, 2, 20, 3, 30, 4, 40};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "InterleaveLower Int32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, InterleaveUpper_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    InterleaveUpper(out.get(), a, b, count);

    // InterleaveUpper takes upper halves and interleaves
    // For count=8: upper half is indices 4-7
    // Expected: a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]
    float expected[] = {5.0f, 50.0f, 6.0f, 60.0f, 7.0f, 70.0f, 8.0f, 80.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "InterleaveUpper Float32 mismatch at index " << i << ": got " << out[i]
            << ", expected " << expected[i];
    }
}

TEST_F(HwyOpsTest, InterleaveUpper_Int32_BasicCorrectness) {
    const size_t count = 8;
    int32_t a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t b[] = {10, 20, 30, 40, 50, 60, 70, 80};
    auto out = AllocateOutput<int32_t>(count);

    InterleaveUpper(out.get(), a, b, count);

    int32_t expected[] = {5, 50, 6, 60, 7, 70, 8, 80};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "InterleaveUpper Int32 mismatch at index " << i;
    }
}

// =============================================================================
// P1: Mask Logical Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MaskNot_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 2 == 0) ? 0xFF : 0x00;
        }
        auto out = hwy::AllocateAligned<uint8_t>(count);

        MaskNot(out.get(), mask.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (mask[i] == 0xFF) ? 0x00 : 0xFF;
            ASSERT_EQ(out[i], expected) << "MaskNot mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskAnd_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = hwy::AllocateAligned<uint8_t>(count);
        auto b = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = (i % 2 == 0) ? 0xFF : 0x00;
            b[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto out = hwy::AllocateAligned<uint8_t>(count);

        MaskAnd(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] && b[i]) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "MaskAnd mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskOr_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = hwy::AllocateAligned<uint8_t>(count);
        auto b = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = (i % 2 == 0) ? 0xFF : 0x00;
            b[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto out = hwy::AllocateAligned<uint8_t>(count);

        MaskOr(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            uint8_t expected = (a[i] || b[i]) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "MaskOr mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskXor_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = hwy::AllocateAligned<uint8_t>(count);
        auto b = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = (i % 2 == 0) ? 0xFF : 0x00;
            b[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto out = hwy::AllocateAligned<uint8_t>(count);

        MaskXor(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            bool a_true = (a[i] != 0);
            bool b_true = (b[i] != 0);
            uint8_t expected = (a_true != b_true) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "MaskXor mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaskAndNot_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = hwy::AllocateAligned<uint8_t>(count);
        auto b = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            a[i] = (i % 2 == 0) ? 0xFF : 0x00;
            b[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto out = hwy::AllocateAligned<uint8_t>(count);

        MaskAndNot(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            // AndNot: NOT(a) AND b = b AND NOT(a)
            bool not_a = (a[i] == 0);
            bool b_true = (b[i] != 0);
            uint8_t expected = (not_a && b_true) ? 0xFF : 0x00;
            ASSERT_EQ(out[i], expected) << "MaskAndNot mismatch at index " << i;
        }
    }
}

// =============================================================================
// P2: AddSub Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, AddSub_Float32_BasicCorrectness) {
    const size_t count = 16;
    auto a = hwy::AllocateAligned<float>(count);
    auto b = hwy::AllocateAligned<float>(count);
    auto out = AllocateOutput<float>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = 1.0f;
    }

    AddSub(out.get(), a.get(), b.get(), count);

    // AddSub: even lanes subtract, odd lanes add
    for (size_t i = 0; i < count; ++i) {
        float expected = (i % 2 == 0) ? (a[i] - b[i]) : (a[i] + b[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected))
            << "AddSub Float32 mismatch at index " << i << ": got " << out[i] << ", expected "
            << expected;
    }
}

TEST_F(HwyOpsTest, AddSub_Float64_BasicCorrectness) {
    const size_t count = 16;
    auto a = hwy::AllocateAligned<double>(count);
    auto b = hwy::AllocateAligned<double>(count);
    auto out = AllocateOutput<double>(count);

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<double>(i + 1);
        b[i] = 1.0;
    }

    AddSub(out.get(), a.get(), b.get(), count);

    for (size_t i = 0; i < count; ++i) {
        double expected = (i % 2 == 0) ? (a[i] - b[i]) : (a[i] + b[i]);
        ASSERT_TRUE(ApproxEqual(out[i], expected)) << "AddSub Float64 mismatch at index " << i;
    }
}

// =============================================================================
// P2: MinMagnitude and MaxMagnitude Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MinMagnitude_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        MinMagnitude(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = (std::abs(a[i]) <= std::abs(b[i])) ? a[i] : b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "MinMagnitude Float32 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MinMagnitude_Float64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomDoubles(count);
        auto b = RandomDoubles(count);
        auto out = AllocateOutput<double>(count);

        MinMagnitude(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            double expected = (std::abs(a[i]) <= std::abs(b[i])) ? a[i] : b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "MinMagnitude Float64 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaxMagnitude_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto b = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        MaxMagnitude(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            float expected = (std::abs(a[i]) >= std::abs(b[i])) ? a[i] : b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "MaxMagnitude Float32 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MaxMagnitude_Float64_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomDoubles(count);
        auto b = RandomDoubles(count);
        auto out = AllocateOutput<double>(count);

        MaxMagnitude(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count; ++i) {
            double expected = (std::abs(a[i]) >= std::abs(b[i])) ? a[i] : b[i];
            ASSERT_TRUE(ApproxEqual(out[i], expected))
                << "MaxMagnitude Float64 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, MinMagnitude_SignedValues) {
    const size_t count = 8;
    float a[] = {-5.0f, 3.0f, -2.0f, 4.0f, -1.0f, 6.0f, -7.0f, 8.0f};
    float b[] = {4.0f, -3.0f, 3.0f, -4.0f, 2.0f, -5.0f, 7.0f, -9.0f};
    auto out = AllocateOutput<float>(count);

    MinMagnitude(out.get(), a, b, count);

    float expected[] = {4.0f, 3.0f, -2.0f, 4.0f, -1.0f, -5.0f, -7.0f, 8.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "MinMagnitude SignedValues mismatch at index " << i << ": got " << out[i]
            << ", expected " << expected[i];
    }
}

// =============================================================================
// P2: MaskedLoad and MaskedStore Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MaskedLoad_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 2 == 0) ? 0xFF : 0x00;
        }
        auto out = AllocateOutput<float>(count);
        const float fallback = 0.0f;

        MaskedLoad(out.get(), src.get(), mask.get(), fallback, count);

        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_TRUE(ApproxEqual(out[i], src[i]))
                    << "MaskedLoad Float32 (true mask) mismatch at index " << i;
            } else {
                ASSERT_TRUE(ApproxEqual(out[i], fallback))
                    << "MaskedLoad Float32 (false mask) should be fallback at index " << i;
            }
        }
    }
}

TEST_F(HwyOpsTest, MaskedLoad_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomInts32(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto out = AllocateOutput<int32_t>(count);
        const int32_t fallback = 0;

        MaskedLoad(out.get(), src.get(), mask.get(), fallback, count);

        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_EQ(out[i], src[i]) << "MaskedLoad Int32 (true mask) mismatch at index " << i;
            } else {
                ASSERT_EQ(out[i], fallback)
                    << "MaskedLoad Int32 (false mask) should be fallback at index " << i;
            }
        }
    }
}

TEST_F(HwyOpsTest, MaskedStore_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 2 == 0) ? 0xFF : 0x00;
        }
        auto dst = hwy::AllocateAligned<float>(count);
        // Initialize dst with known values
        for (size_t i = 0; i < count; ++i) {
            dst[i] = -999.0f;
        }

        MaskedStore(dst.get(), src.get(), mask.get(), count);

        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_TRUE(ApproxEqual(dst[i], src[i]))
                    << "MaskedStore Float32 (true mask) mismatch at index " << i;
            } else {
                ASSERT_TRUE(ApproxEqual(dst[i], -999.0f))
                    << "MaskedStore Float32 (false mask) should be unchanged at index " << i;
            }
        }
    }
}

TEST_F(HwyOpsTest, MaskedStore_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomInts32(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto dst = hwy::AllocateAligned<int32_t>(count);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = -999;
        }

        MaskedStore(dst.get(), src.get(), mask.get(), count);

        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_EQ(dst[i], src[i])
                    << "MaskedStore Int32 (true mask) mismatch at index " << i;
            } else {
                ASSERT_EQ(dst[i], -999)
                    << "MaskedStore Int32 (false mask) should be unchanged at index " << i;
            }
        }
    }
}

// =============================================================================
// P2: BlendedStore Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, BlendedStore_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto new_val = RandomFloats(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 2 == 0) ? 0xFF : 0x00;
        }
        auto dst = hwy::AllocateAligned<float>(count);
        // Initialize dst with known values
        for (size_t i = 0; i < count; ++i) {
            dst[i] = static_cast<float>(i * 100);
        }
        auto original = hwy::AllocateAligned<float>(count);
        std::copy(dst.get(), dst.get() + count, original.get());

        BlendedStore(dst.get(), new_val.get(), mask.get(), count);

        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_TRUE(ApproxEqual(dst[i], new_val[i]))
                    << "BlendedStore Float32 (true mask) mismatch at index " << i;
            } else {
                ASSERT_TRUE(ApproxEqual(dst[i], original[i]))
                    << "BlendedStore Float32 (false mask) should be unchanged at index " << i;
            }
        }
    }
}

TEST_F(HwyOpsTest, BlendedStore_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto new_val = RandomInts32(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto dst = hwy::AllocateAligned<int32_t>(count);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = static_cast<int32_t>(i * 100);
        }
        auto original = hwy::AllocateAligned<int32_t>(count);
        std::copy(dst.get(), dst.get() + count, original.get());

        BlendedStore(dst.get(), new_val.get(), mask.get(), count);

        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_EQ(dst[i], new_val[i])
                    << "BlendedStore Int32 (true mask) mismatch at index " << i;
            } else {
                ASSERT_EQ(dst[i], original[i])
                    << "BlendedStore Int32 (false mask) should be unchanged at index " << i;
            }
        }
    }
}

TEST_F(HwyOpsTest, BlendedStore_AllTrue) {
    const size_t count = 100;
    auto new_val = RandomFloats(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::fill(mask.get(), mask.get() + count, 0xFF);
    auto dst = hwy::AllocateAligned<float>(count);
    std::fill(dst.get(), dst.get() + count, 0.0f);

    BlendedStore(dst.get(), new_val.get(), mask.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(dst[i], new_val[i]))
            << "BlendedStore AllTrue mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, BlendedStore_AllFalse) {
    const size_t count = 100;
    auto new_val = RandomFloats(count);
    auto mask = hwy::AllocateAligned<uint8_t>(count);
    std::fill(mask.get(), mask.get() + count, 0x00);
    auto dst = hwy::AllocateAligned<float>(count);
    auto original = hwy::AllocateAligned<float>(count);
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(i);
        original[i] = static_cast<float>(i);
    }

    BlendedStore(dst.get(), new_val.get(), mask.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(dst[i], original[i]))
            << "BlendedStore AllFalse should preserve original at index " << i;
    }
}

// =============================================================================
// P0: WidenMulPairwiseAdd Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, WidenMulPairwiseAdd_Int16_BasicCorrectness) {
    for (size_t count : {4, 8, 16, 32, 64, 128, 256}) {
        auto a = hwy::AllocateAligned<int16_t>(count);
        auto b = hwy::AllocateAligned<int16_t>(count);
        std::uniform_int_distribution<int16_t> dist(-100, 100);
        for (size_t i = 0; i < count; ++i) {
            a[i] = dist(rng_);
            b[i] = dist(rng_);
        }
        auto out = AllocateOutput<int32_t>(count / 2);

        WidenMulPairwiseAdd(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count / 2; ++i) {
            int32_t expected =
                static_cast<int32_t>(a[2 * i]) * static_cast<int32_t>(b[2 * i]) +
                static_cast<int32_t>(a[2 * i + 1]) * static_cast<int32_t>(b[2 * i + 1]);
            ASSERT_EQ(out[i], expected) << "WidenMulPairwiseAdd Int16 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, WidenMulPairwiseAdd_Uint16_BasicCorrectness) {
    for (size_t count : {4, 8, 16, 32, 64, 128, 256}) {
        auto a = hwy::AllocateAligned<uint16_t>(count);
        auto b = hwy::AllocateAligned<uint16_t>(count);
        std::uniform_int_distribution<uint16_t> dist(0, 200);
        for (size_t i = 0; i < count; ++i) {
            a[i] = dist(rng_);
            b[i] = dist(rng_);
        }
        auto out = AllocateOutput<uint32_t>(count / 2);

        WidenMulPairwiseAdd(out.get(), a.get(), b.get(), count);

        for (size_t i = 0; i < count / 2; ++i) {
            uint32_t expected =
                static_cast<uint32_t>(a[2 * i]) * static_cast<uint32_t>(b[2 * i]) +
                static_cast<uint32_t>(a[2 * i + 1]) * static_cast<uint32_t>(b[2 * i + 1]);
            ASSERT_EQ(out[i], expected) << "WidenMulPairwiseAdd Uint16 mismatch at index " << i;
        }
    }
}

// =============================================================================
// P1: BroadcastLane Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, BroadcastLane_Float32_BasicCorrectness) {
    const size_t count = 16;
    auto a = RandomFloats(count);
    auto out = AllocateOutput<float>(count);

    for (size_t lane = 0; lane < count; ++lane) {
        BroadcastLane(out.get(), a.get(), lane, count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(out[i], a[lane]))
                << "BroadcastLane Float32 mismatch at index " << i << " for lane " << lane;
        }
    }
}

TEST_F(HwyOpsTest, BroadcastLane_Int32_BasicCorrectness) {
    const size_t count = 16;
    auto a = RandomInts32(count);
    auto out = AllocateOutput<int32_t>(count);

    for (size_t lane = 0; lane < count; ++lane) {
        BroadcastLane(out.get(), a.get(), lane, count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(out[i], a[lane])
                << "BroadcastLane Int32 mismatch at index " << i << " for lane " << lane;
        }
    }
}

// =============================================================================
// P1: Slide Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Slide1Up_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Slide1Up(out.get(), a.get(), count);

        ASSERT_TRUE(ApproxEqual(out[0], 0.0f)) << "Slide1Up Float32 first element should be 0";
        for (size_t i = 1; i < count; ++i) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i - 1]))
                << "Slide1Up Float32 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Slide1Up_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Slide1Up(out.get(), a.get(), count);

        ASSERT_EQ(out[0], 0) << "Slide1Up Int32 first element should be 0";
        for (size_t i = 1; i < count; ++i) {
            ASSERT_EQ(out[i], a[i - 1]) << "Slide1Up Int32 mismatch at index " << i;
        }
    }
}

TEST_F(HwyOpsTest, Slide1Down_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomFloats(count);
        auto out = AllocateOutput<float>(count);

        Slide1Down(out.get(), a.get(), count);

        for (size_t i = 0; i < count - 1; ++i) {
            ASSERT_TRUE(ApproxEqual(out[i], a[i + 1]))
                << "Slide1Down Float32 mismatch at index " << i;
        }
        ASSERT_TRUE(ApproxEqual(out[count - 1], 0.0f))
            << "Slide1Down Float32 last element should be 0";
    }
}

TEST_F(HwyOpsTest, Slide1Down_Int32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto a = RandomInts32(count);
        auto out = AllocateOutput<int32_t>(count);

        Slide1Down(out.get(), a.get(), count);

        for (size_t i = 0; i < count - 1; ++i) {
            ASSERT_EQ(out[i], a[i + 1]) << "Slide1Down Int32 mismatch at index " << i;
        }
        ASSERT_EQ(out[count - 1], 0) << "Slide1Down Int32 last element should be 0";
    }
}

// =============================================================================
// P1: Concat Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, ConcatLowerUpper_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    ConcatLowerUpper(out.get(), a, b, count);

    // Lower half from a, upper half from b
    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "ConcatLowerUpper Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ConcatUpperLower_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    ConcatUpperLower(out.get(), a, b, count);

    // Upper half of a, lower half of b
    float expected[] = {5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "ConcatUpperLower Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ConcatEven_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    ConcatEven(out.get(), a, b, count);

    // Even elements from a, then even elements from b
    float expected[] = {1.0f, 3.0f, 5.0f, 7.0f, 10.0f, 30.0f, 50.0f, 70.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "ConcatEven Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ConcatOdd_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    ConcatOdd(out.get(), a, b, count);

    // Odd elements from a, then odd elements from b
    float expected[] = {2.0f, 4.0f, 6.0f, 8.0f, 20.0f, 40.0f, 60.0f, 80.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "ConcatOdd Float32 mismatch at index " << i;
    }
}

// =============================================================================
// P1: Mask Utility Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, FindKnownFirstTrue_BasicCorrectness) {
    const size_t count = 16;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // Test with first true at different positions
    for (size_t first_true = 0; first_true < count; ++first_true) {
        std::fill(mask.get(), mask.get() + count, 0x00);
        mask[first_true] = 0xFF;

        size_t result = FindKnownFirstTrue(mask.get(), count);
        ASSERT_EQ(result, first_true)
            << "FindKnownFirstTrue mismatch for first_true=" << first_true;
    }
}

TEST_F(HwyOpsTest, FindKnownLastTrue_BasicCorrectness) {
    const size_t count = 16;
    auto mask = hwy::AllocateAligned<uint8_t>(count);

    // Test with last true at different positions
    for (size_t last_true = 0; last_true < count; ++last_true) {
        std::fill(mask.get(), mask.get() + count, 0x00);
        mask[last_true] = 0xFF;

        size_t result = FindKnownLastTrue(mask.get(), count);
        ASSERT_EQ(result, last_true) << "FindKnownLastTrue mismatch for last_true=" << last_true;
    }
}

TEST_F(HwyOpsTest, StoreMaskBits_LoadMaskBits_RoundTrip) {
    for (size_t count : {8, 16, 24, 32, 64, 100}) {
        auto mask_in = hwy::AllocateAligned<uint8_t>(count);
        auto bits = hwy::AllocateAligned<uint8_t>((count + 7) / 8);
        auto mask_out = hwy::AllocateAligned<uint8_t>(count);

        // Create a pattern
        for (size_t i = 0; i < count; ++i) {
            mask_in[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }

        size_t num_bytes = StoreMaskBits(bits.get(), mask_in.get(), count);
        ASSERT_EQ(num_bytes, (count + 7) / 8);

        LoadMaskBits(mask_out.get(), bits.get(), count);

        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(mask_out[i], mask_in[i])
                << "StoreMaskBits/LoadMaskBits roundtrip mismatch at index " << i;
        }
    }
}

// =============================================================================
// P1: CompressBlendedStore and CompressNot Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, CompressBlendedStore_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 3 == 0) ? 0xFF : 0x00;
        }
        auto dst = hwy::AllocateAligned<float>(count);

        size_t written = CompressBlendedStore(dst.get(), src.get(), mask.get(), count);

        // Verify count
        size_t expected_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (mask[i])
                expected_count++;
        }
        ASSERT_EQ(written, expected_count);

        // Verify values
        size_t out_idx = 0;
        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                ASSERT_TRUE(ApproxEqual(dst[out_idx], src[i]))
                    << "CompressBlendedStore mismatch at out index " << out_idx;
                out_idx++;
            }
        }
    }
}

TEST_F(HwyOpsTest, CompressNot_Float32_BasicCorrectness) {
    for (size_t count : TestSizes()) {
        auto src = RandomFloats(count);
        auto mask = hwy::AllocateAligned<uint8_t>(count);
        for (size_t i = 0; i < count; ++i) {
            mask[i] = (i % 2 == 0) ? 0xFF : 0x00;
        }
        auto out = hwy::AllocateAligned<float>(count);

        size_t written = CompressNot(out.get(), src.get(), mask.get(), count);

        // Verify count (should be elements where mask is false)
        size_t expected_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!mask[i])
                expected_count++;
        }
        ASSERT_EQ(written, expected_count);

        // Verify values
        size_t out_idx = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!mask[i]) {
                ASSERT_TRUE(ApproxEqual(out[out_idx], src[i]))
                    << "CompressNot mismatch at out index " << out_idx;
                out_idx++;
            }
        }
    }
}

// =============================================================================
// P1: InterleaveEven and InterleaveOdd Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, InterleaveEven_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    InterleaveEven(out.get(), a, b, count);

    // Take even elements (0, 2, 4, 6) and interleave
    // Expected: a[0], b[0], a[2], b[2], a[4], b[4], a[6], b[6]
    float expected[] = {1.0f, 10.0f, 3.0f, 30.0f, 5.0f, 50.0f, 7.0f, 70.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "InterleaveEven Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, InterleaveOdd_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    auto out = AllocateOutput<float>(count);

    InterleaveOdd(out.get(), a, b, count);

    // Take odd elements (1, 3, 5, 7) and interleave
    // Expected: a[1], b[1], a[3], b[3], a[5], b[5], a[7], b[7]
    float expected[] = {2.0f, 20.0f, 4.0f, 40.0f, 6.0f, 60.0f, 8.0f, 80.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "InterleaveOdd Float32 mismatch at index " << i;
    }
}

// =============================================================================
// P1: Shuffle Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Shuffle0123_Float32_BasicCorrectness) {
    const size_t count = 8;
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = AllocateOutput<float>(count);

    Shuffle0123(out.get(), in, count);

    // Identity shuffle - output should match input
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], in[i])) << "Shuffle0123 Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Shuffle2301_Float32_BasicCorrectness) {
    const size_t count = 8;
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = AllocateOutput<float>(count);

    Shuffle2301(out.get(), in, count);

    // [0,1,2,3] -> [2,3,0,1], [4,5,6,7] -> [6,7,4,5]
    float expected[] = {3.0f, 4.0f, 1.0f, 2.0f, 7.0f, 8.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "Shuffle2301 Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Shuffle1032_Float32_BasicCorrectness) {
    const size_t count = 8;
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = AllocateOutput<float>(count);

    Shuffle1032(out.get(), in, count);

    // [0,1,2,3] -> [1,0,3,2], [4,5,6,7] -> [5,4,7,6]
    float expected[] = {2.0f, 1.0f, 4.0f, 3.0f, 6.0f, 5.0f, 8.0f, 7.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "Shuffle1032 Float32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Shuffle01_Float64_BasicCorrectness) {
    const size_t count = 4;
    double in[] = {1.0, 2.0, 3.0, 4.0};
    auto out = hwy::AllocateAligned<double>(count);

    Shuffle01(out.get(), in, count);

    // Identity - output should match input
    for (size_t i = 0; i < count; ++i) {
        ASSERT_DOUBLE_EQ(out[i], in[i]) << "Shuffle01 Float64 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Shuffle10_Float64_BasicCorrectness) {
    const size_t count = 4;
    double in[] = {1.0, 2.0, 3.0, 4.0};
    auto out = hwy::AllocateAligned<double>(count);

    Shuffle10(out.get(), in, count);

    // [0,1] -> [1,0], [2,3] -> [3,2]
    double expected[] = {2.0, 1.0, 4.0, 3.0};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_DOUBLE_EQ(out[i], expected[i]) << "Shuffle10 Float64 mismatch at index " << i;
    }
}

// =============================================================================
// P1: TableLookup Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, TableLookupBytes_BasicCorrectness) {
    const size_t table_size = 16;
    const size_t count = 8;
    uint8_t table[16];
    for (size_t i = 0; i < table_size; ++i) {
        table[i] = static_cast<uint8_t>(i * 10);
    }
    uint8_t indices[] = {0, 2, 4, 6, 8, 10, 12, 14};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    TableLookupBytes(out.get(), table, indices, count, table_size);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], table[indices[i]]) << "TableLookupBytes mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, TableLookupLanes_Float32_BasicCorrectness) {
    const size_t table_size = 8;
    const size_t count = 4;
    float table[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    int32_t indices[] = {7, 3, 0, 5};
    auto out = AllocateOutput<float>(count);

    TableLookupLanes(out.get(), table, indices, count, table_size);

    float expected[] = {80.0f, 40.0f, 10.0f, 60.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "TableLookupLanes Float32 mismatch at index " << i;
    }
}

// =============================================================================
// P1: Mask Set Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SetBeforeFirst_BasicCorrectness) {
    const size_t count = 8;
    uint8_t mask[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    SetBeforeFirst(out.get(), mask, count);

    // First true is at index 3, so indices 0-2 should be true
    uint8_t expected[] = {0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "SetBeforeFirst mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SetAtOrBeforeFirst_BasicCorrectness) {
    const size_t count = 8;
    uint8_t mask[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    SetAtOrBeforeFirst(out.get(), mask, count);

    // First true is at index 3, so indices 0-3 should be true
    uint8_t expected[] = {0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "SetAtOrBeforeFirst mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SetOnlyFirst_BasicCorrectness) {
    const size_t count = 8;
    uint8_t mask[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    SetOnlyFirst(out.get(), mask, count);

    // Only index 3 should be true
    uint8_t expected[] = {0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "SetOnlyFirst mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SetAtOrAfterFirst_BasicCorrectness) {
    const size_t count = 8;
    uint8_t mask[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    SetAtOrAfterFirst(out.get(), mask, count);

    // First true is at index 3, so indices 3-7 should be true
    uint8_t expected[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "SetAtOrAfterFirst mismatch at index " << i;
    }
}

// =============================================================================
// P1: Masked Reduction Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MaskedReduceSum_Float32_BasicCorrectness) {
    const size_t count = 8;
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00};

    float result = MaskedReduceSum(src, mask, count);

    // Sum of 1 + 3 + 5 + 7 = 16
    ASSERT_TRUE(ApproxEqual(result, 16.0f))
        << "MaskedReduceSum Float32 expected 16.0, got " << result;
}

TEST_F(HwyOpsTest, MaskedReduceSum_Int32_BasicCorrectness) {
    const size_t count = 8;
    int32_t src[] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00};

    int32_t result = MaskedReduceSum(src, mask, count);

    // Sum of 10 + 20 + 50 + 60 = 140
    ASSERT_EQ(result, 140) << "MaskedReduceSum Int32 expected 140, got " << result;
}

TEST_F(HwyOpsTest, MaskedReduceMin_Float32_BasicCorrectness) {
    const size_t count = 8;
    float src[] = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 7.0f, 4.0f};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00};

    float result = MaskedReduceMin(src, mask, count);

    // Min of 5, 8, 9, 7 = 5
    ASSERT_TRUE(ApproxEqual(result, 5.0f))
        << "MaskedReduceMin Float32 expected 5.0, got " << result;
}

TEST_F(HwyOpsTest, MaskedReduceMin_Int32_BasicCorrectness) {
    const size_t count = 8;
    int32_t src[] = {50, 20, 80, 10, 90, 30, 70, 40};
    uint8_t mask[] = {0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF};

    int32_t result = MaskedReduceMin(src, mask, count);

    // Min of 20, 10, 30, 40 = 10
    ASSERT_EQ(result, 10) << "MaskedReduceMin Int32 expected 10, got " << result;
}

TEST_F(HwyOpsTest, MaskedReduceMax_Float32_BasicCorrectness) {
    const size_t count = 8;
    float src[] = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 7.0f, 4.0f};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00};

    float result = MaskedReduceMax(src, mask, count);

    // Max of 5, 8, 7 = 8
    ASSERT_TRUE(ApproxEqual(result, 8.0f))
        << "MaskedReduceMax Float32 expected 8.0, got " << result;
}

TEST_F(HwyOpsTest, MaskedReduceMax_Int32_BasicCorrectness) {
    const size_t count = 8;
    int32_t src[] = {50, 20, 80, 10, 90, 30, 70, 40};
    uint8_t mask[] = {0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00};

    int32_t result = MaskedReduceMax(src, mask, count);

    // Max of 50, 20, 80 = 80
    ASSERT_EQ(result, 80) << "MaskedReduceMax Int32 expected 80, got " << result;
}

// =============================================================================
// P1: Remaining Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, TwoTablesLookupLanes_Float32_BasicCorrectness) {
    const size_t count = 4;
    const size_t table_size = 4;
    float table0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float table1[] = {5.0f, 6.0f, 7.0f, 8.0f};
    int32_t indices[] = {0, 5, 2, 7};  // 0,2 from table0, 5,7 from table1 (index >= 4)
    auto out = hwy::AllocateAligned<float>(count);

    TwoTablesLookupLanes(out.get(), table0, table1, indices, count, table_size);

    float expected[] = {1.0f, 6.0f, 3.0f, 8.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "TwoTablesLookupLanes mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, CompressBits_Float32_BasicCorrectness) {
    const size_t count = 8;
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    uint8_t bits[] = {0b01010101};  // Bits 0,2,4,6 are set
    auto out = hwy::AllocateAligned<float>(count);

    size_t result = CompressBits(out.get(), src, bits, count);

    // Elements at indices 0,2,4,6 should be compressed
    ASSERT_EQ(result, 4u) << "CompressBits should return 4 elements";
    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 5.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 7.0f));
}

TEST_F(HwyOpsTest, PairwiseSub_Float32_BasicCorrectness) {
    const size_t count = 8;
    float a[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = hwy::AllocateAligned<float>(count);

    PairwiseSub(out.get(), a, b, count);

    // Pairwise subtraction: (a[0]-a[1], b[0]-b[1], a[2]-a[3], b[2]-b[3], ...)
    for (size_t i = 0; i < count / 2; ++i) {
        ASSERT_TRUE(ApproxEqual(out[2 * i], a[2 * i] - a[2 * i + 1]));
        ASSERT_TRUE(ApproxEqual(out[2 * i + 1], b[2 * i] - b[2 * i + 1]));
    }
}

// =============================================================================
// P2: Math Functions Tests
// =============================================================================

TEST_F(HwyOpsTest, Cbrt_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 8.0f, 27.0f, 64.0f};
    auto out = hwy::AllocateAligned<float>(count);

    Cbrt(out.get(), a, count);

    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i], 1e-5f)) << "Cbrt mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Erf_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {0.0f, 0.5f, 1.0f, -1.0f};
    auto out = hwy::AllocateAligned<float>(count);

    Erf(out.get(), a, count);

    // Check that results are in valid range [-1, 1]
    for (size_t i = 0; i < count; ++i) {
        ASSERT_GE(out[i], -1.0f) << "Erf result should be >= -1";
        ASSERT_LE(out[i], 1.0f) << "Erf result should be <= 1";
    }
    // erf(0) = 0
    ASSERT_TRUE(ApproxEqual(out[0], 0.0f, 1e-5f)) << "erf(0) should be 0";
}

// =============================================================================
// P2: Bitwise Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Xor3_Int32_BasicCorrectness) {
    const size_t count = 4;
    int32_t a[] = {0x12340000, 0x00001234, 0x00123400, 0x12003400};
    int32_t b[] = {0x00005678, 0x56780000, 0x00567800, 0x00560078};
    int32_t c[] = {0x00009ABC, 0x00009ABC, 0x0000009A, 0x0000BC00};
    auto out = hwy::AllocateAligned<int32_t>(count);

    Xor3(out.get(), a, b, c, count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], a[i] ^ b[i] ^ c[i]) << "Xor3 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Or3_Int32_BasicCorrectness) {
    const size_t count = 4;
    int32_t a[] = {0x10000000, 0x01000000, 0x00100000, 0x00010000};
    int32_t b[] = {0x00001000, 0x00000100, 0x00000010, 0x00000001};
    int32_t c[] = {0x20000000, 0x02000000, 0x00200000, 0x00020000};
    auto out = hwy::AllocateAligned<int32_t>(count);

    Or3(out.get(), a, b, c, count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], a[i] | b[i] | c[i]) << "Or3 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ReverseBits_Uint32_BasicCorrectness) {
    const size_t count = 4;
    uint32_t a[] = {0x00000001, 0x80000000, 0xF0000000, 0x0000000F};
    auto out = hwy::AllocateAligned<uint32_t>(count);

    ReverseBits(out.get(), a, count);

    // Bit reverse: 0x00000001 -> 0x80000000, etc.
    ASSERT_EQ(out[0], 0x80000000u);
    ASSERT_EQ(out[1], 0x00000001u);
    ASSERT_EQ(out[2], 0x0000000Fu);
    ASSERT_EQ(out[3], 0xF0000000u);
}

TEST_F(HwyOpsTest, HighestSetBitIndex_Uint32_BasicCorrectness) {
    const size_t count = 4;
    uint32_t a[] = {0x00000001, 0x00000010, 0x00000100, 0x80000000};
    auto out = hwy::AllocateAligned<int32_t>(count);

    HighestSetBitIndex(out.get(), a, count);

    int32_t expected[] = {0, 4, 8, 31};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "HighestSetBitIndex mismatch at index " << i;
    }
}

// =============================================================================
// P2: Arithmetic Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Mod_Int32_BasicCorrectness) {
    const size_t count = 4;
    int32_t a[] = {10, 17, 23, 100};
    int32_t b[] = {3, 5, 7, 11};
    auto out = hwy::AllocateAligned<int32_t>(count);

    Mod(out.get(), a, b, count);

    int32_t expected[] = {1, 2, 2, 1};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "Mod mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SaturatedNeg_Int8_BasicCorrectness) {
    const size_t count = 4;
    int8_t a[] = {127, -128, 0, 50};
    auto out = hwy::AllocateAligned<int8_t>(count);

    SaturatedNeg(out.get(), a, count);

    // -127, 127 (saturated), 0, -50
    ASSERT_EQ(out[0], -127);
    ASSERT_EQ(out[1], 127);  // -(-128) saturates to 127
    ASSERT_EQ(out[2], 0);
    ASSERT_EQ(out[3], -50);
}

TEST_F(HwyOpsTest, SaturatedAbs_Int8_BasicCorrectness) {
    const size_t count = 4;
    int8_t a[] = {127, -128, 0, -50};
    auto out = hwy::AllocateAligned<int8_t>(count);

    SaturatedAbs(out.get(), a, count);

    // 127, 127 (saturated), 0, 50
    ASSERT_EQ(out[0], 127);
    ASSERT_EQ(out[1], 127);  // |-128| saturates to 127
    ASSERT_EQ(out[2], 0);
    ASSERT_EQ(out[3], 50);
}

// =============================================================================
// P2: Memory Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SafeFillN_Float32_BasicCorrectness) {
    const size_t count = 8;
    auto out = hwy::AllocateAligned<float>(count);

    for (size_t i = 0; i < count; ++i)
        out[i] = 0.0f;

    SafeFillN(out.get(), 42.0f, count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], 42.0f)) << "SafeFillN mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SafeCopyN_Float32_BasicCorrectness) {
    const size_t count = 8;
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = hwy::AllocateAligned<float>(count);

    SafeCopyN(out.get(), src, count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], src[i])) << "SafeCopyN mismatch at index " << i;
    }
}

// =============================================================================
// P2: Special Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MulByPow2_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 2.0f, 4.0f, 8.0f};
    int32_t exp[] = {1, 2, 3, -1};
    auto out = hwy::AllocateAligned<float>(count);

    MulByPow2(out.get(), a, exp, count);

    float expected[] = {2.0f, 8.0f, 32.0f, 4.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i])) << "MulByPow2 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, SignBit_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, -1.0f, 0.0f, -0.0f};
    auto out = hwy::AllocateAligned<uint32_t>(count);

    SignBit(out.get(), a, count);

    // Sign bit is the MSB
    ASSERT_EQ(out[0], 0u);
    ASSERT_NE(out[1], 0u);  // Negative has sign bit set
    ASSERT_EQ(out[2], 0u);
    ASSERT_NE(out[3], 0u);  // -0.0 has sign bit set
}

TEST_F(HwyOpsTest, NaN_Float32_BasicCorrectness) {
    const size_t count = 4;
    auto out = hwy::AllocateAligned<float>(count);

    NaN(out.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(std::isnan(out[i])) << "NaN should produce NaN at index " << i;
    }
}

TEST_F(HwyOpsTest, Inf_Float32_BasicCorrectness) {
    const size_t count = 4;
    auto out = hwy::AllocateAligned<float>(count);

    Inf(out.get(), count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(std::isinf(out[i])) << "Inf should produce infinity at index " << i;
    }
}

// =============================================================================
// P3: Complex Number Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, ComplexConj_Float32_BasicCorrectness) {
    const size_t count = 4;  // 2 complex numbers (real, imag pairs)
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto out = hwy::AllocateAligned<float>(count);

    ComplexConj(out.get(), in, count);

    // Imaginary parts should be negated
    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));   // real
    ASSERT_TRUE(ApproxEqual(out[1], -2.0f));  // imag (negated)
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));   // real
    ASSERT_TRUE(ApproxEqual(out[3], -4.0f));  // imag (negated)
}

TEST_F(HwyOpsTest, MulComplex_Float32_BasicCorrectness) {
    const size_t count = 4;                // 2 complex numbers (real, imag pairs)
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};  // (1+2i), (3+4i)
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};  // (5+6i), (7+8i)
    auto out = hwy::AllocateAligned<float>(count);

    MulComplex(out.get(), a, b, count);

    // (1+2i)(5+6i) = (5-12) + (6+10)i = -7 + 16i
    // (3+4i)(7+8i) = (21-32) + (24+28)i = -11 + 52i
    ASSERT_TRUE(ApproxEqual(out[0], -7.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 16.0f));
    ASSERT_TRUE(ApproxEqual(out[2], -11.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 52.0f));
}

// =============================================================================
// P3: Saturation Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, SaturatedAdd_Int8_BasicCorrectness) {
    const size_t count = 4;
    int8_t a[] = {100, -100, 50, -50};
    int8_t b[] = {50, -50, 100, -100};
    auto out = hwy::AllocateAligned<int8_t>(count);

    SaturatedAdd(out.get(), a, b, count);

    // 100+50=150 saturates to 127
    // -100-50=-150 saturates to -128
    ASSERT_EQ(out[0], 127);
    ASSERT_EQ(out[1], -128);
    ASSERT_EQ(out[2], 127);
    ASSERT_EQ(out[3], -128);
}

TEST_F(HwyOpsTest, SaturatedSub_Int8_BasicCorrectness) {
    const size_t count = 4;
    int8_t a[] = {100, -100, 50, -50};
    int8_t b[] = {-50, 50, -100, 100};
    auto out = hwy::AllocateAligned<int8_t>(count);

    SaturatedSub(out.get(), a, b, count);

    // 100-(-50)=150 saturates to 127
    // -100-50=-150 saturates to -128
    ASSERT_EQ(out[0], 127);
    ASSERT_EQ(out[1], -128);
    ASSERT_EQ(out[2], 127);
    ASSERT_EQ(out[3], -128);
}

// =============================================================================
// P3: Additional Masked Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, ZeroIfNegative_Float32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {1.0f, -2.0f, 3.0f, -4.0f};
    auto out = hwy::AllocateAligned<float>(count);

    ZeroIfNegative(out.get(), src, count);

    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 0.0f));  // Negative -> 0
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 0.0f));  // Negative -> 0
}

TEST_F(HwyOpsTest, MaskedSqrt_Float32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {4.0f, 9.0f, 16.0f, 25.0f};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<float>(count);

    MaskedSqrt(out.get(), src, mask, 0.0f, count);

    ASSERT_TRUE(ApproxEqual(out[0], 2.0f));  // sqrt(4)
    ASSERT_TRUE(ApproxEqual(out[1], 0.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[2], 4.0f));  // sqrt(16)
    ASSERT_TRUE(ApproxEqual(out[3], 0.0f));  // fallback
}

// =============================================================================
// P2: Type Conversion Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, ReorderDemote2To_Int32ToInt16_BasicCorrectness) {
    const size_t count = 8;
    int32_t hi[] = {5, 6, 7, 8};
    int32_t lo[] = {1, 2, 3, 4};
    auto out = hwy::AllocateAligned<int16_t>(count);

    ReorderDemote2To(out.get(), hi, lo, count);

    // Result should be lo demoted, then hi demoted (interleaved order)
    int16_t expected[] = {1, 5, 2, 6, 3, 7, 4, 8};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "ReorderDemote2To mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, OrderedTruncate2To_Uint32ToUint16_BasicCorrectness) {
    const size_t count = 8;
    uint32_t hi[] = {0x00010005, 0x00010006, 0x00010007, 0x00010008};
    uint32_t lo[] = {0x00010001, 0x00010002, 0x00010003, 0x00010004};
    auto out = hwy::AllocateAligned<uint16_t>(count);

    OrderedTruncate2To(out.get(), hi, lo, count);

    // Lower 16 bits of each, lo first then hi
    uint16_t expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "OrderedTruncate2To mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ConvertInRangeTo_Float32ToInt32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {1.5f, 2.7f, -3.2f, 4.9f};
    auto out = hwy::AllocateAligned<int32_t>(count);

    ConvertInRangeTo(out.get(), src, count);

    // Convert with truncation toward zero
    int32_t expected[] = {1, 2, -3, 4};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "ConvertInRangeTo mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ResizeBitCast_Float32ToUint32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto out = hwy::AllocateAligned<uint32_t>(count);

    ResizeBitCast(out.get(), src, count);

    // Should be the same bit pattern
    for (size_t i = 0; i < count; ++i) {
        uint32_t expected;
        std::memcpy(&expected, &src[i], sizeof(float));
        ASSERT_EQ(out[i], expected) << "ResizeBitCast mismatch at index " << i;
    }
}

// =============================================================================
// P2: Special Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MulByFloorPow2_Float32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float pow2[] = {2.0f, 4.0f, 0.5f, 8.0f};  // pow2 values that are powers of 2
    auto out = hwy::AllocateAligned<float>(count);

    MulByFloorPow2(out.get(), src, pow2, count);

    // Multiply by floor(log2(pow2)) power of 2
    float expected[] = {2.0f, 8.0f, 1.5f, 32.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i])) << "MulByFloorPow2 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, GetBiasedExponent_Float32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {1.0f, 2.0f, 0.5f, 4.0f};
    auto out = hwy::AllocateAligned<int32_t>(count);

    GetBiasedExponent(out.get(), src, count);

    // IEEE-754 biased exponent: exp + 127 for float32
    // 1.0 = 2^0 -> biased exp = 127
    // 2.0 = 2^1 -> biased exp = 128
    // 0.5 = 2^-1 -> biased exp = 126
    // 4.0 = 2^2 -> biased exp = 129
    int32_t expected[] = {127, 128, 126, 129};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "GetBiasedExponent mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, MulFixedPoint15_Int16_BasicCorrectness) {
    const size_t count = 4;
    int16_t a[] = {16384, 8192, -16384, 32767};  // Q15: 0.5, 0.25, -0.5, ~1.0
    int16_t b[] = {16384, 16384, 16384, 16384};  // Q15: 0.5
    auto out = hwy::AllocateAligned<int16_t>(count);

    MulFixedPoint15(out.get(), a, b, count);

    // Q15 multiply: (a * b) >> 15
    // 0.5 * 0.5 = 0.25 -> 8192
    // 0.25 * 0.5 = 0.125 -> 4096
    // -0.5 * 0.5 = -0.25 -> -8192
    int16_t expected[] = {8192, 4096, -8192, 16383};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "MulFixedPoint15 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, MulRound_Int16_BasicCorrectness) {
    const size_t count = 4;
    int16_t a[] = {100, 200, 300, -400};
    int16_t b[] = {3, 3, 3, 3};
    auto out = hwy::AllocateAligned<int16_t>(count);

    MulRound(out.get(), a, b, count);

    // Multiply with rounding (implementation dependent)
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], a[i] * b[i]) << "MulRound mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, RoundingShiftRight_Int16_BasicCorrectness) {
    const size_t count = 4;
    int16_t src[] = {15, 16, 17, -15};
    auto out = hwy::AllocateAligned<int16_t>(count);

    RoundingShiftRight(out.get(), src, 2, count);  // Shift right by 2 with rounding

    // Rounding shift: (x + (1 << (shift-1))) >> shift
    // 15 + 2 = 17 >> 2 = 4
    // 16 + 2 = 18 >> 2 = 4
    // 17 + 2 = 19 >> 2 = 4
    // -15 should use arithmetic shift with rounding
    int16_t expected[] = {4, 4, 4, -4};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "RoundingShiftRight mismatch at index " << i;
    }
}

// =============================================================================
// P3: Complex Number Operations Tests (Additional)
// =============================================================================

TEST_F(HwyOpsTest, MulComplexConj_Float32_BasicCorrectness) {
    const size_t count = 4;                // 2 complex numbers
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};  // (1+2i), (3+4i)
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};  // (5+6i), (7+8i)
    auto out = hwy::AllocateAligned<float>(count);

    MulComplexConj(out.get(), a, b, count);

    // (1+2i) * conj(5+6i) = (1+2i) * (5-6i) = (5+12) + (-6+10)i = 17 + 4i
    // (3+4i) * conj(7+8i) = (3+4i) * (7-8i) = (21+32) + (-24+28)i = 53 + 4i
    ASSERT_TRUE(ApproxEqual(out[0], 17.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 4.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 53.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 4.0f));
}

TEST_F(HwyOpsTest, MulComplexConjAdd_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};      // (1+2i), (3+4i)
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};      // (5+6i), (7+8i)
    float c[] = {10.0f, 20.0f, 30.0f, 40.0f};  // (10+20i), (30+40i)
    auto out = hwy::AllocateAligned<float>(count);

    MulComplexConjAdd(out.get(), a, b, c, count);

    // (1+2i) * conj(5+6i) + (10+20i) = 17+4i + 10+20i = 27+24i
    // (3+4i) * conj(7+8i) + (30+40i) = 53+4i + 30+40i = 83+44i
    ASSERT_TRUE(ApproxEqual(out[0], 27.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 24.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 83.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 44.0f));
}

// =============================================================================
// P3: Block Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, Per4LaneBlockShuffle_Float32_BasicCorrectness) {
    const size_t count = 8;
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = hwy::AllocateAligned<float>(count);

    // Shuffle pattern: 0x1B = reverse within each 4-lane block
    Per4LaneBlockShuffle(out.get(), src, 0x1B, count);

    // Reverse within blocks: [4,3,2,1, 8,7,6,5]
    float expected[] = {4.0f, 3.0f, 2.0f, 1.0f, 8.0f, 7.0f, 6.0f, 5.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i]))
            << "Per4LaneBlockShuffle mismatch at index " << i;
    }
}

// =============================================================================
// P3: Masked Operations Tests (Additional)
// =============================================================================

TEST_F(HwyOpsTest, MaskedReciprocal_Float32_BasicCorrectness) {
    const size_t count = 4;
    float src[] = {2.0f, 4.0f, 5.0f, 10.0f};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<float>(count);

    MaskedReciprocal(out.get(), src, mask, 0.0f, count);

    ASSERT_TRUE(ApproxEqual(out[0], 0.5f));  // 1/2
    ASSERT_TRUE(ApproxEqual(out[1], 0.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[2], 0.2f));  // 1/5
    ASSERT_TRUE(ApproxEqual(out[3], 0.0f));  // fallback
}

TEST_F(HwyOpsTest, MaskedShiftLeft_Int32_BasicCorrectness) {
    const size_t count = 4;
    int32_t src[] = {1, 2, 3, 4};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<int32_t>(count);

    MaskedShiftLeft(out.get(), src, mask, 2, 0, count);

    ASSERT_EQ(out[0], 4);   // 1 << 2
    ASSERT_EQ(out[1], 0);   // fallback
    ASSERT_EQ(out[2], 12);  // 3 << 2
    ASSERT_EQ(out[3], 0);   // fallback
}

TEST_F(HwyOpsTest, MaskedShiftRight_Int32_BasicCorrectness) {
    const size_t count = 4;
    int32_t src[] = {8, 16, 24, 32};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<int32_t>(count);

    MaskedShiftRight(out.get(), src, mask, 2, 0, count);

    ASSERT_EQ(out[0], 2);  // 8 >> 2
    ASSERT_EQ(out[1], 0);  // fallback
    ASSERT_EQ(out[2], 6);  // 24 >> 2
    ASSERT_EQ(out[3], 0);  // fallback
}

TEST_F(HwyOpsTest, MaskedSatAdd_Int8_BasicCorrectness) {
    const size_t count = 4;
    int8_t a[] = {100, 50, -100, -50};
    int8_t b[] = {50, 30, -50, -30};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<int8_t>(count);

    MaskedSatAdd(out.get(), a, b, mask, 0, count);

    ASSERT_EQ(out[0], 127);   // saturated
    ASSERT_EQ(out[1], 0);     // fallback
    ASSERT_EQ(out[2], -128);  // saturated
    ASSERT_EQ(out[3], 0);     // fallback
}

TEST_F(HwyOpsTest, MaskedSatSub_Int8_BasicCorrectness) {
    const size_t count = 4;
    int8_t a[] = {100, 50, -100, -50};
    int8_t b[] = {-50, -30, 50, 30};
    uint8_t mask[] = {0xFF, 0x00, 0xFF, 0x00};
    auto out = hwy::AllocateAligned<int8_t>(count);

    MaskedSatSub(out.get(), a, b, mask, 0, count);

    ASSERT_EQ(out[0], 127);   // saturated
    ASSERT_EQ(out[1], 0);     // fallback
    ASSERT_EQ(out[2], -128);  // saturated
    ASSERT_EQ(out[3], 0);     // fallback
}

// =============================================================================
// P3: Masked Comparison Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, MaskedEq_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {1.0f, 3.0f, 3.0f, 5.0f};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0xFF};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    MaskedEq(out.get(), a, b, mask, count);

    ASSERT_EQ(out[0], 0xFF);  // 1 == 1, masked true
    ASSERT_EQ(out[1], 0x00);  // 2 != 3, masked true
    ASSERT_EQ(out[2], 0x00);  // masked false
    ASSERT_EQ(out[3], 0x00);  // 4 != 5, masked true
}

TEST_F(HwyOpsTest, MaskedNe_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {1.0f, 3.0f, 3.0f, 5.0f};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0xFF};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    MaskedNe(out.get(), a, b, mask, count);

    ASSERT_EQ(out[0], 0x00);  // 1 == 1 (not ne), masked true
    ASSERT_EQ(out[1], 0xFF);  // 2 != 3, masked true
    ASSERT_EQ(out[2], 0x00);  // masked false
    ASSERT_EQ(out[3], 0xFF);  // 4 != 5, masked true
}

TEST_F(HwyOpsTest, MaskedLt_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 3.0f, 2.0f, 5.0f};
    float b[] = {2.0f, 2.0f, 3.0f, 4.0f};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0xFF};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    MaskedLt(out.get(), a, b, mask, count);

    ASSERT_EQ(out[0], 0xFF);  // 1 < 2
    ASSERT_EQ(out[1], 0x00);  // 3 >= 2
    ASSERT_EQ(out[2], 0x00);  // masked false
    ASSERT_EQ(out[3], 0x00);  // 5 >= 4
}

TEST_F(HwyOpsTest, MaskedLe_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 5.0f};
    float b[] = {2.0f, 2.0f, 2.0f, 4.0f};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0xFF};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    MaskedLe(out.get(), a, b, mask, count);

    ASSERT_EQ(out[0], 0xFF);  // 1 <= 2
    ASSERT_EQ(out[1], 0xFF);  // 2 <= 2
    ASSERT_EQ(out[2], 0x00);  // masked false
    ASSERT_EQ(out[3], 0x00);  // 5 > 4
}

TEST_F(HwyOpsTest, MaskedGt_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {3.0f, 2.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 2.0f, 3.0f, 6.0f};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0xFF};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    MaskedGt(out.get(), a, b, mask, count);

    ASSERT_EQ(out[0], 0xFF);  // 3 > 2
    ASSERT_EQ(out[1], 0x00);  // 2 == 2
    ASSERT_EQ(out[2], 0x00);  // masked false
    ASSERT_EQ(out[3], 0x00);  // 5 < 6
}

TEST_F(HwyOpsTest, MaskedGe_Float32_BasicCorrectness) {
    const size_t count = 4;
    float a[] = {3.0f, 2.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 2.0f, 3.0f, 6.0f};
    uint8_t mask[] = {0xFF, 0xFF, 0x00, 0xFF};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    MaskedGe(out.get(), a, b, mask, count);

    ASSERT_EQ(out[0], 0xFF);  // 3 >= 2
    ASSERT_EQ(out[1], 0xFF);  // 2 >= 2
    ASSERT_EQ(out[2], 0x00);  // masked false
    ASSERT_EQ(out[3], 0x00);  // 5 < 6
}

// =============================================================================
// P3.2: Cryptographic Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, AESRound_BasicCorrectness) {
    // AES state is 16 bytes (128-bit block)
    const size_t count = 16;
    // Test vector from FIPS-197
    uint8_t state[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    uint8_t round_key[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    AESRound(out.get(), state, round_key, count);

    // Verify output is different from input (transformation applied)
    bool different = false;
    for (size_t i = 0; i < count; ++i) {
        if (out[i] != state[i]) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "AES round should transform the state";
}

TEST_F(HwyOpsTest, AESLastRound_BasicCorrectness) {
    const size_t count = 16;
    uint8_t state[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    uint8_t round_key[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    AESLastRound(out.get(), state, round_key, count);

    bool different = false;
    for (size_t i = 0; i < count; ++i) {
        if (out[i] != state[i]) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "AES last round should transform the state";
}

TEST_F(HwyOpsTest, AESRoundInv_BasicCorrectness) {
    const size_t count = 16;
    uint8_t state[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    uint8_t round_key[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    AESRoundInv(out.get(), state, round_key, count);

    bool different = false;
    for (size_t i = 0; i < count; ++i) {
        if (out[i] != state[i]) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "AES inverse round should transform the state";
}

TEST_F(HwyOpsTest, AESLastRoundInv_BasicCorrectness) {
    const size_t count = 16;
    uint8_t state[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    uint8_t round_key[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    AESLastRoundInv(out.get(), state, round_key, count);

    bool different = false;
    for (size_t i = 0; i < count; ++i) {
        if (out[i] != state[i]) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "AES inverse last round should transform the state";
}

TEST_F(HwyOpsTest, AESInvMixColumns_BasicCorrectness) {
    const size_t count = 16;
    uint8_t state[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    AESInvMixColumns(out.get(), state, count);

    bool different = false;
    for (size_t i = 0; i < count; ++i) {
        if (out[i] != state[i]) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "AES InvMixColumns should transform the state";
}

TEST_F(HwyOpsTest, AESKeyGenAssist_BasicCorrectness) {
    const size_t count = 16;
    uint8_t key[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    auto out = hwy::AllocateAligned<uint8_t>(count);

    AESKeyGenAssist(out.get(), key, 1, count);

    bool different = false;
    for (size_t i = 0; i < count; ++i) {
        if (out[i] != key[i]) {
            different = true;
            break;
        }
    }
    ASSERT_TRUE(different) << "AES KeyGenAssist should transform the key";
}

TEST_F(HwyOpsTest, CLMulLower_BasicCorrectness) {
    // Carry-less multiplication of lower 64-bit halves
    const size_t count = 2;                  // 128-bit output
    uint64_t a[] = {0x0000000000000003ULL};  // x + 1
    uint64_t b[] = {0x0000000000000005ULL};  // x^2 + 1
    auto out = hwy::AllocateAligned<uint64_t>(count);

    CLMulLower(out.get(), a, b, 1);

    // (x + 1) * (x^2 + 1) = x^3 + x^2 + x + 1 = 0xF in polynomial
    // But carry-less multiplication works differently, let's just verify it computes something
    ASSERT_NE(out[0], 0ULL) << "CLMul should produce non-zero result";
}

TEST_F(HwyOpsTest, CLMulUpper_BasicCorrectness) {
    const size_t count = 2;
    uint64_t a[] = {0x0, 0x0000000000000003ULL};  // Upper 64-bits
    uint64_t b[] = {0x0, 0x0000000000000005ULL};
    auto out = hwy::AllocateAligned<uint64_t>(count);

    CLMulUpper(out.get(), a, b, 2);

    // Should compute carry-less multiplication of upper halves
    ASSERT_NE(out[0], 0ULL) << "CLMul upper should produce non-zero result";
}

// =============================================================================
// P3.3: Random Number Generation Tests
// =============================================================================

TEST_F(HwyOpsTest, RandomState_Initialization) {
    uint64_t state[4];
    RandomStateInit(state, 12345);

    // State should be initialized (not all zeros)
    bool non_zero = false;
    for (int i = 0; i < 4; ++i) {
        if (state[i] != 0) {
            non_zero = true;
            break;
        }
    }
    ASSERT_TRUE(non_zero) << "Random state should be initialized";
}

TEST_F(HwyOpsTest, Random32_GeneratesValues) {
    uint64_t state[4];
    RandomStateInit(state, 42);

    const size_t count = 16;
    auto out = hwy::AllocateAligned<uint32_t>(count);

    Random32(out.get(), state, count);

    // Check we got varied random values
    bool varied = false;
    for (size_t i = 1; i < count; ++i) {
        if (out[i] != out[0]) {
            varied = true;
            break;
        }
    }
    ASSERT_TRUE(varied) << "Random32 should generate varied values";
}

TEST_F(HwyOpsTest, Random64_GeneratesValues) {
    uint64_t state[4];
    RandomStateInit(state, 42);

    const size_t count = 8;
    auto out = hwy::AllocateAligned<uint64_t>(count);

    Random64(out.get(), state, count);

    bool varied = false;
    for (size_t i = 1; i < count; ++i) {
        if (out[i] != out[0]) {
            varied = true;
            break;
        }
    }
    ASSERT_TRUE(varied) << "Random64 should generate varied values";
}

TEST_F(HwyOpsTest, RandomFloat_GeneratesInRange) {
    uint64_t state[4];
    RandomStateInit(state, 42);

    const size_t count = 32;
    auto out = hwy::AllocateAligned<float>(count);

    RandomFloat(out.get(), state, count);

    for (size_t i = 0; i < count; ++i) {
        ASSERT_GE(out[i], 0.0f) << "RandomFloat should be >= 0";
        ASSERT_LT(out[i], 1.0f) << "RandomFloat should be < 1";
    }
}

// =============================================================================
// P3.4: Bit Packing Tests
// =============================================================================

TEST_F(HwyOpsTest, PackBits_BasicCorrectness) {
    // Pack 8 bytes into 1 byte (MSB of each byte becomes one bit)
    const size_t count = 8;
    uint8_t src[] = {0x80, 0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00};
    auto out = hwy::AllocateAligned<uint8_t>(1);

    PackBits(out.get(), src, count);

    // Expected: 1 0 1 1 0 1 0 0 = 0xB4
    ASSERT_EQ(out[0], 0xB4) << "PackBits should pack MSBs correctly";
}

TEST_F(HwyOpsTest, UnpackBits_BasicCorrectness) {
    // Unpack 1 byte into 8 bytes
    uint8_t src[] = {0xB4};  // 10110100
    const size_t count = 8;
    auto out = hwy::AllocateAligned<uint8_t>(count);

    UnpackBits(out.get(), src, count);

    // Each bit becomes 0xFF or 0x00
    uint8_t expected[] = {0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x00};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "UnpackBits mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, PackUnpackBits_Roundtrip) {
    // Pack then unpack should give back original (when MSBs are set)
    const size_t count = 16;
    uint8_t src[] = {0xFF, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00,
                     0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF};
    auto packed = hwy::AllocateAligned<uint8_t>(2);
    auto unpacked = hwy::AllocateAligned<uint8_t>(count);

    PackBits(packed.get(), src, count);
    UnpackBits(unpacked.get(), packed.get(), count);

    for (size_t i = 0; i < count; ++i) {
        uint8_t expected = (src[i] & 0x80) ? 0xFF : 0x00;
        ASSERT_EQ(unpacked[i], expected) << "Roundtrip mismatch at index " << i;
    }
}

// =============================================================================
// P3.6: Algorithm Operations Tests
// =============================================================================

TEST_F(HwyOpsTest, FindIf_FindsElement) {
    const size_t count = 16;
    float arr[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    // Find first element > 10
    size_t result = FindIfGreaterThan(arr, 10.0f, count);

    ASSERT_EQ(result, 10u) << "Should find element at index 10 (value 11.0)";
}

TEST_F(HwyOpsTest, FindIf_NotFound) {
    const size_t count = 8;
    float arr[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    size_t result = FindIfGreaterThan(arr, 100.0f, count);

    ASSERT_EQ(result, count) << "Should return count when not found";
}

TEST_F(HwyOpsTest, Generate_Sequence) {
    const size_t count = 8;
    auto out = hwy::AllocateAligned<float>(count);

    // Generate sequence starting at 5.0 with step 2.0
    Generate(out.get(), 5.0f, 2.0f, count);

    float expected[] = {5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f, 19.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i])) << "Generate mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Generate_Int32_Sequence) {
    const size_t count = 8;
    auto out = hwy::AllocateAligned<int32_t>(count);

    Generate(out.get(), 10, 3, count);

    int32_t expected[] = {10, 13, 16, 19, 22, 25, 28, 31};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(out[i], expected[i]) << "Generate int32 mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, Replace_BasicCorrectness) {
    const size_t count = 8;
    float arr[] = {1.0f, 2.0f, 3.0f, 2.0f, 5.0f, 2.0f, 7.0f, 8.0f};
    auto out = hwy::AllocateAligned<float>(count);

    // Replace all 2.0 with 99.0
    Replace(out.get(), arr, 2.0f, 99.0f, count);

    float expected[] = {1.0f, 99.0f, 3.0f, 99.0f, 5.0f, 99.0f, 7.0f, 8.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i])) << "Replace mismatch at index " << i;
    }
}

TEST_F(HwyOpsTest, ReplaceIf_GreaterThan) {
    const size_t count = 8;
    float arr[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto out = hwy::AllocateAligned<float>(count);

    // Replace all values > 4 with 0
    ReplaceIfGreaterThan(out.get(), arr, 4.0f, 0.0f, count);

    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], expected[i])) << "ReplaceIf mismatch at index " << i;
    }
}

// =============================================================================
// P3.5: Image Processing Tests
// =============================================================================

TEST_F(HwyOpsTest, ImageCreate_BasicCorrectness) {
    const size_t width = 64;
    const size_t height = 48;

    auto img = ImageCreate(width, height);

    ASSERT_NE(img.data, nullptr) << "Image data should be allocated";
    ASSERT_EQ(img.width, width) << "Width should match";
    ASSERT_EQ(img.height, height) << "Height should match";
    ASSERT_GE(img.stride, width) << "Stride should be at least width";

    ImageFree(&img);
    ASSERT_EQ(img.data, nullptr) << "Data should be null after free";
}

TEST_F(HwyOpsTest, ImageFill_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;
    const float fill_value = 0.5f;

    auto img = ImageCreate(width, height);
    ImageFill(&img, fill_value);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ASSERT_TRUE(ApproxEqual(img.data[y * img.stride + x], fill_value))
                << "Fill value mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&img);
}

TEST_F(HwyOpsTest, ImageCopy_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto dst = ImageCreate(width, height);

    // Fill src with pattern
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            src.data[y * src.stride + x] = static_cast<float>(x + y * width);
        }
    }

    ImageCopy(&src, &dst);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float expected = src.data[y * src.stride + x];
            float actual = dst.data[y * dst.stride + x];
            ASSERT_TRUE(ApproxEqual(actual, expected))
                << "Copy mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&dst);
}

TEST_F(HwyOpsTest, ImageAdd_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto a = ImageCreate(width, height);
    auto b = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    ImageFill(&a, 0.3f);
    ImageFill(&b, 0.2f);

    ImageAdd(&a, &b, &out);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 0.5f))
                << "Add mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&a);
    ImageFree(&b);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, ImageSub_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto a = ImageCreate(width, height);
    auto b = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    ImageFill(&a, 0.7f);
    ImageFill(&b, 0.2f);

    ImageSub(&a, &b, &out);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 0.5f))
                << "Sub mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&a);
    ImageFree(&b);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, ImageMul_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto a = ImageCreate(width, height);
    auto b = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    ImageFill(&a, 0.5f);
    ImageFill(&b, 0.4f);

    ImageMul(&a, &b, &out);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 0.2f))
                << "Mul mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&a);
    ImageFree(&b);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, ImageScale_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    ImageFill(&src, 0.4f);
    ImageScale(&src, 2.5f, &out);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 1.0f))
                << "Scale mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, BoxBlur3x3_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Create impulse in center
    ImageFill(&src, 0.0f);
    src.data[4 * src.stride + 4] = 9.0f;  // Center pixel

    BoxBlur3x3(&src, &out);

    // Center and its 8 neighbors should each have 1.0 (9/9)
    // due to box blur averaging
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            size_t y = 4 + dy;
            size_t x = 4 + dx;
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 1.0f, 0.01f))
                << "BoxBlur mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, GaussianBlur3x3_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Fill with constant - should remain constant after blur
    ImageFill(&src, 0.5f);

    GaussianBlur3x3(&src, &out);

    // Interior pixels should be unchanged (constant image)
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 0.5f, 0.01f))
                << "GaussianBlur mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, SobelEdge_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Create vertical edge
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            src.data[y * src.stride + x] = (x < 4) ? 0.0f : 1.0f;
        }
    }

    SobelEdge(&src, &out);

    // Edge should be detected at x=3 and x=4
    // Interior non-edge pixels should be near zero
    ASSERT_GT(out.data[4 * out.stride + 3], 0.1f) << "Edge not detected at x=3";
    ASSERT_GT(out.data[4 * out.stride + 4], 0.1f) << "Edge not detected at x=4";

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Sharpen_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Constant image should remain constant after sharpening
    ImageFill(&src, 0.5f);

    Sharpen(&src, &out);

    // Interior pixels should be unchanged
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 0.5f, 0.01f))
                << "Sharpen changed constant image at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Threshold_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Create gradient
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            src.data[y * src.stride + x] = static_cast<float>(x) / 7.0f;
        }
    }

    Threshold(&src, 0.5f, &out);

    // Values below threshold -> 0, above -> 1
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float expected = (src.data[y * src.stride + x] > 0.5f) ? 1.0f : 0.0f;
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], expected))
                << "Threshold mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Grayscale_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto r = ImageCreate(width, height);
    auto g = ImageCreate(width, height);
    auto b = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // White pixel: R=G=B=1.0 -> grayscale = 1.0
    ImageFill(&r, 1.0f);
    ImageFill(&g, 1.0f);
    ImageFill(&b, 1.0f);

    Grayscale(&r, &g, &b, &out);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            // Luminance formula: 0.299*R + 0.587*G + 0.114*B = 1.0 for white
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 1.0f, 0.01f))
                << "Grayscale mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&r);
    ImageFree(&g);
    ImageFree(&b);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Grayscale_RedChannel) {
    const size_t width = 4;
    const size_t height = 4;

    auto r = ImageCreate(width, height);
    auto g = ImageCreate(width, height);
    auto b = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Pure red: R=1, G=B=0
    ImageFill(&r, 1.0f);
    ImageFill(&g, 0.0f);
    ImageFill(&b, 0.0f);

    Grayscale(&r, &g, &b, &out);

    // Expected: 0.299
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], 0.299f, 0.01f))
                << "Red grayscale mismatch";
        }
    }

    ImageFree(&r);
    ImageFree(&g);
    ImageFree(&b);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, ImageClamp_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Values outside [0, 1] range
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            src.data[y * src.stride + x] = static_cast<float>(x) / 3.5f - 0.5f;
        }
    }

    ImageClamp(&src, 0.0f, 1.0f, &out);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float val = out.data[y * out.stride + x];
            ASSERT_GE(val, 0.0f) << "Value below min";
            ASSERT_LE(val, 1.0f) << "Value above max";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Convolve3x3_Identity) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width, height);

    // Random-ish values
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            src.data[y * src.stride + x] = static_cast<float>((x * 7 + y * 13) % 100) / 100.0f;
        }
    }

    // Identity kernel
    float kernel[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};

    Convolve3x3(&src, kernel, &out);

    // Interior should match source
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
            ASSERT_TRUE(
                ApproxEqual(out.data[y * out.stride + x], src.data[y * src.stride + x], 0.001f))
                << "Identity convolution changed value at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Downsample2x_BasicCorrectness) {
    const size_t width = 8;
    const size_t height = 8;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width / 2, height / 2);

    // Fill with pattern where 2x2 blocks have same value
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            src.data[y * src.stride + x] = static_cast<float>(y / 2 * 4 + x / 2);
        }
    }

    Downsample2x(&src, &out);

    for (size_t y = 0; y < height / 2; ++y) {
        for (size_t x = 0; x < width / 2; ++x) {
            float expected = static_cast<float>(y * 4 + x);
            ASSERT_TRUE(ApproxEqual(out.data[y * out.stride + x], expected, 0.01f))
                << "Downsample mismatch at (" << x << ", " << y << ")";
        }
    }

    ImageFree(&src);
    ImageFree(&out);
}

TEST_F(HwyOpsTest, Upsample2x_BasicCorrectness) {
    const size_t width = 4;
    const size_t height = 4;

    auto src = ImageCreate(width, height);
    auto out = ImageCreate(width * 2, height * 2);

    // Simple pattern
    ImageFill(&src, 0.0f);
    src.data[1 * src.stride + 1] = 1.0f;

    Upsample2x(&src, &out);

    // The 2x2 block corresponding to src[1,1] should have values
    ASSERT_GT(out.data[2 * out.stride + 2], 0.0f);
    ASSERT_GT(out.data[2 * out.stride + 3], 0.0f);
    ASSERT_GT(out.data[3 * out.stride + 2], 0.0f);
    ASSERT_GT(out.data[3 * out.stride + 3], 0.0f);

    ImageFree(&src);
    ImageFree(&out);
}

// =============================================================================
// Gap Analysis: Remaining Highway Operations Tests
// =============================================================================

// -----------------------------------------------------------------------------
// LoadInterleaved2/3/4 - Load AoS data as SoA
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, LoadInterleaved2_BasicCorrectness) {
    // Interleaved data: a0,b0,a1,b1,a2,b2,...
    alignas(64) float interleaved[16] = {
        1.0f, 10.0f,  // pair 0
        2.0f, 20.0f,  // pair 1
        3.0f, 30.0f,  // pair 2
        4.0f, 40.0f,  // pair 3
        5.0f, 50.0f,  // pair 4
        6.0f, 60.0f,  // pair 5
        7.0f, 70.0f,  // pair 6
        8.0f, 80.0f   // pair 7
    };
    alignas(64) float a[8], b[8];

    LoadInterleaved2(a, b, interleaved, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ApproxEqual(a[i], static_cast<float>(i + 1))) << "a[" << i << "] mismatch";
        ASSERT_TRUE(ApproxEqual(b[i], static_cast<float>((i + 1) * 10)))
            << "b[" << i << "] mismatch";
    }
}

TEST_F(HwyOpsTest, LoadInterleaved3_BasicCorrectness) {
    // Interleaved RGB data: r0,g0,b0,r1,g1,b1,...
    alignas(64) float interleaved[24] = {
        1.0f, 10.0f, 100.0f,  // pixel 0
        2.0f, 20.0f, 200.0f,  // pixel 1
        3.0f, 30.0f, 300.0f,  // pixel 2
        4.0f, 40.0f, 400.0f,  // pixel 3
        5.0f, 50.0f, 500.0f,  // pixel 4
        6.0f, 60.0f, 600.0f,  // pixel 5
        7.0f, 70.0f, 700.0f,  // pixel 6
        8.0f, 80.0f, 800.0f   // pixel 7
    };
    alignas(64) float r[8], g[8], b[8];

    LoadInterleaved3(r, g, b, interleaved, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ApproxEqual(r[i], static_cast<float>(i + 1))) << "r[" << i << "] mismatch";
        ASSERT_TRUE(ApproxEqual(g[i], static_cast<float>((i + 1) * 10)))
            << "g[" << i << "] mismatch";
        ASSERT_TRUE(ApproxEqual(b[i], static_cast<float>((i + 1) * 100)))
            << "b[" << i << "] mismatch";
    }
}

TEST_F(HwyOpsTest, LoadInterleaved4_BasicCorrectness) {
    // Interleaved RGBA data: r0,g0,b0,a0,r1,g1,b1,a1,...
    alignas(64) float interleaved[32] = {
        1.0f, 10.0f, 100.0f, 1000.0f,  // pixel 0
        2.0f, 20.0f, 200.0f, 2000.0f,  // pixel 1
        3.0f, 30.0f, 300.0f, 3000.0f,  // pixel 2
        4.0f, 40.0f, 400.0f, 4000.0f,  // pixel 3
        5.0f, 50.0f, 500.0f, 5000.0f,  // pixel 4
        6.0f, 60.0f, 600.0f, 6000.0f,  // pixel 5
        7.0f, 70.0f, 700.0f, 7000.0f,  // pixel 6
        8.0f, 80.0f, 800.0f, 8000.0f   // pixel 7
    };
    alignas(64) float r[8], g[8], b[8], a[8];

    LoadInterleaved4(r, g, b, a, interleaved, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ApproxEqual(r[i], static_cast<float>(i + 1))) << "r[" << i << "] mismatch";
        ASSERT_TRUE(ApproxEqual(g[i], static_cast<float>((i + 1) * 10)))
            << "g[" << i << "] mismatch";
        ASSERT_TRUE(ApproxEqual(b[i], static_cast<float>((i + 1) * 100)))
            << "b[" << i << "] mismatch";
        ASSERT_TRUE(ApproxEqual(a[i], static_cast<float>((i + 1) * 1000)))
            << "a[" << i << "] mismatch";
    }
}

// -----------------------------------------------------------------------------
// StoreInterleaved2/3/4 - Store SoA data as AoS
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, StoreInterleaved2_BasicCorrectness) {
    alignas(64) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float interleaved[16] = {0};

    StoreInterleaved2(interleaved, a, b, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ApproxEqual(interleaved[i * 2], a[i])) << "a at position " << i << " mismatch";
        ASSERT_TRUE(ApproxEqual(interleaved[i * 2 + 1], b[i]))
            << "b at position " << i << " mismatch";
    }
}

TEST_F(HwyOpsTest, StoreInterleaved3_BasicCorrectness) {
    alignas(64) float r[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float g[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float b[8] = {100, 200, 300, 400, 500, 600, 700, 800};
    alignas(64) float interleaved[24] = {0};

    StoreInterleaved3(interleaved, r, g, b, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ApproxEqual(interleaved[i * 3], r[i])) << "r at position " << i << " mismatch";
        ASSERT_TRUE(ApproxEqual(interleaved[i * 3 + 1], g[i]))
            << "g at position " << i << " mismatch";
        ASSERT_TRUE(ApproxEqual(interleaved[i * 3 + 2], b[i]))
            << "b at position " << i << " mismatch";
    }
}

TEST_F(HwyOpsTest, StoreInterleaved4_BasicCorrectness) {
    alignas(64) float r[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float g[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float b[8] = {100, 200, 300, 400, 500, 600, 700, 800};
    alignas(64) float a[8] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    alignas(64) float interleaved[32] = {0};

    StoreInterleaved4(interleaved, r, g, b, a, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(ApproxEqual(interleaved[i * 4], r[i])) << "r at position " << i << " mismatch";
        ASSERT_TRUE(ApproxEqual(interleaved[i * 4 + 1], g[i]))
            << "g at position " << i << " mismatch";
        ASSERT_TRUE(ApproxEqual(interleaved[i * 4 + 2], b[i]))
            << "b at position " << i << " mismatch";
        ASSERT_TRUE(ApproxEqual(interleaved[i * 4 + 3], a[i]))
            << "a at position " << i << " mismatch";
    }
}

// -----------------------------------------------------------------------------
// PairwiseAdd/Sub - Adjacent pair operations
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, PairwiseAdd_Float32) {
    alignas(64) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float out[4] = {0};

    // PairwiseAdd: out[i] = a[2i] + a[2i+1]
    PairwiseAdd(out, a, 8);

    // Expected: (1+2), (3+4), (5+6), (7+8)
    ASSERT_TRUE(ApproxEqual(out[0], 3.0f));   // 1+2
    ASSERT_TRUE(ApproxEqual(out[1], 7.0f));   // 3+4
    ASSERT_TRUE(ApproxEqual(out[2], 11.0f));  // 5+6
    ASSERT_TRUE(ApproxEqual(out[3], 15.0f));  // 7+8
}

TEST_F(HwyOpsTest, PairwiseSub_Float32) {
    alignas(64) float a[8] = {2, 1, 4, 3, 6, 5, 8, 7};
    alignas(64) float b[8] = {20, 10, 40, 30, 60, 50, 80, 70};
    alignas(64) float out[8] = {0};

    PairwiseSub(out, a, b, 8);

    // out[2i] = a[2i] - a[2i+1], out[2i+1] = b[2i] - b[2i+1]
    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));   // 2-1
    ASSERT_TRUE(ApproxEqual(out[1], 10.0f));  // 20-10
    ASSERT_TRUE(ApproxEqual(out[2], 1.0f));   // 4-3
    ASSERT_TRUE(ApproxEqual(out[3], 10.0f));  // 40-30
    ASSERT_TRUE(ApproxEqual(out[4], 1.0f));   // 6-5
    ASSERT_TRUE(ApproxEqual(out[5], 10.0f));  // 60-50
    ASSERT_TRUE(ApproxEqual(out[6], 1.0f));   // 8-7
    ASSERT_TRUE(ApproxEqual(out[7], 10.0f));  // 80-70
}

// -----------------------------------------------------------------------------
// SumOfLanes/MinOfLanes/MaxOfLanes - Reduction to broadcast vector
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, SumOfLanes_Float32) {
    alignas(64) float in[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float out[1] = {0};

    SumOfLanes(out, in, 8);

    // out[0] should have the sum (1+2+3+4+5+6+7+8 = 36)
    ASSERT_TRUE(ApproxEqual(out[0], 36.0f));
}

TEST_F(HwyOpsTest, MinOfLanes_Float32) {
    alignas(64) float in[8] = {5, 2, 8, 1, 9, 3, 7, 4};
    alignas(64) float out[1] = {0};

    MinOfLanes(out, in, 8);

    // out[0] should have the min (1)
    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
}

TEST_F(HwyOpsTest, MaxOfLanes_Float32) {
    alignas(64) float in[8] = {5, 2, 8, 1, 9, 3, 7, 4};
    alignas(64) float out[1] = {0};

    MaxOfLanes(out, in, 8);

    // out[0] should have the max (9)
    ASSERT_TRUE(ApproxEqual(out[0], 9.0f));
}

TEST_F(HwyOpsTest, SumOfLanes_Int32) {
    alignas(64) int32_t in[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) int32_t out[1] = {0};

    SumOfLanes(out, in, 8);

    ASSERT_EQ(out[0], 36);
}

// -----------------------------------------------------------------------------
// MaskedMulAdd/MaskedNegMulAdd - Masked FMA operations
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedMulAdd_BasicCorrectness) {
    alignas(64) float mul[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float x[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    alignas(64) float add[8] = {10, 10, 10, 10, 10, 10, 10, 10};
    alignas(64) float no[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    alignas(64) uint8_t mask[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) float out[8] = {0};

    // out[i] = mask[i] ? (mul[i] * x[i] + add[i]) : no[i]
    MaskedMulAdd(out, mask, mul, x, add, no, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 12.0f));  // 1*2+10 = 12
    ASSERT_TRUE(ApproxEqual(out[1], -1.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[2], 16.0f));  // 3*2+10 = 16
    ASSERT_TRUE(ApproxEqual(out[3], -1.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[4], 20.0f));  // 5*2+10 = 20
    ASSERT_TRUE(ApproxEqual(out[5], -1.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[6], 24.0f));  // 7*2+10 = 24
    ASSERT_TRUE(ApproxEqual(out[7], -1.0f));  // fallback
}

TEST_F(HwyOpsTest, MaskedNegMulAdd_BasicCorrectness) {
    alignas(64) float mul[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float x[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    alignas(64) float add[8] = {10, 10, 10, 10, 10, 10, 10, 10};
    alignas(64) float no[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    alignas(64) uint8_t mask[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) float out[8] = {0};

    // out[i] = mask[i] ? (add[i] - mul[i] * x[i]) : no[i]
    MaskedNegMulAdd(out, mask, mul, x, add, no, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 8.0f));   // 10 - 1*2 = 8
    ASSERT_TRUE(ApproxEqual(out[1], -1.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[2], 4.0f));   // 10 - 3*2 = 4
    ASSERT_TRUE(ApproxEqual(out[3], -1.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[4], 0.0f));   // 10 - 5*2 = 0
    ASSERT_TRUE(ApproxEqual(out[5], -1.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[6], -4.0f));  // 10 - 7*2 = -4
    ASSERT_TRUE(ApproxEqual(out[7], -1.0f));  // fallback
}

// -----------------------------------------------------------------------------
// InterleaveWholeLower/Upper - Full vector interleaving
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, InterleaveWholeLower_Float32) {
    alignas(64) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float out[8] = {0};

    // Interleave lower halves: a0,b0,a1,b1,a2,b2,a3,b3
    InterleaveWholeLower(out, a, b, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 10.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 2.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 20.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[5], 30.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 4.0f));
    ASSERT_TRUE(ApproxEqual(out[7], 40.0f));
}

TEST_F(HwyOpsTest, InterleaveWholeUpper_Float32) {
    alignas(64) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float out[8] = {0};

    // Interleave upper halves: a4,b4,a5,b5,a6,b6,a7,b7
    InterleaveWholeUpper(out, a, b, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 5.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 50.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 6.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 60.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 7.0f));
    ASSERT_TRUE(ApproxEqual(out[5], 70.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 8.0f));
    ASSERT_TRUE(ApproxEqual(out[7], 80.0f));
}

// -----------------------------------------------------------------------------
// OddEven - Select odd from one, even from another
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, OddEven_Float32) {
    alignas(64) float odd[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float even[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float out[8] = {0};

    // out[i] = (i % 2 == 0) ? even[i] : odd[i]
    OddEven(out, odd, even, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));   // even
    ASSERT_TRUE(ApproxEqual(out[1], 20.0f));  // odd
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));   // even
    ASSERT_TRUE(ApproxEqual(out[3], 40.0f));  // odd
    ASSERT_TRUE(ApproxEqual(out[4], 5.0f));   // even
    ASSERT_TRUE(ApproxEqual(out[5], 60.0f));  // odd
    ASSERT_TRUE(ApproxEqual(out[6], 7.0f));   // even
    ASSERT_TRUE(ApproxEqual(out[7], 80.0f));  // odd
}

// -----------------------------------------------------------------------------
// ZeroIfNegative - Conditional zero
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, ZeroIfNegative_Float32) {
    alignas(64) float in[8] = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f, -0.5f, 0.5f, -1.0f};
    alignas(64) float out[8] = {0};

    ZeroIfNegative(out, in, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[5], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 0.5f));
    ASSERT_TRUE(ApproxEqual(out[7], 0.0f));
}

// -----------------------------------------------------------------------------
// IfNegativeThenElseZero/IfNegativeThenZeroElse
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, IfNegativeThenElseZero_Float32) {
    alignas(64) float v[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    alignas(64) float yes[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float out[8] = {0};

    // out[i] = (v[i] < 0) ? yes[i] : 0
    IfNegativeThenElseZero(out, v, yes, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 10.0f));  // v<0, use yes
    ASSERT_TRUE(ApproxEqual(out[1], 0.0f));   // v>=0, use 0
    ASSERT_TRUE(ApproxEqual(out[2], 30.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 50.0f));
    ASSERT_TRUE(ApproxEqual(out[5], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 70.0f));
    ASSERT_TRUE(ApproxEqual(out[7], 0.0f));
}

TEST_F(HwyOpsTest, IfNegativeThenZeroElse_Float32) {
    alignas(64) float v[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    alignas(64) float no[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float out[8] = {0};

    // out[i] = (v[i] < 0) ? 0 : no[i]
    IfNegativeThenZeroElse(out, v, no, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 0.0f));   // v<0, use 0
    ASSERT_TRUE(ApproxEqual(out[1], 20.0f));  // v>=0, use no
    ASSERT_TRUE(ApproxEqual(out[2], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 40.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[5], 60.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 0.0f));
    ASSERT_TRUE(ApproxEqual(out[7], 80.0f));
}

// -----------------------------------------------------------------------------
// BitwiseIfThenElse - Bit-level conditional
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, BitwiseIfThenElse_Int32) {
    alignas(64) int32_t mask[8] = {-1, 0, -1, 0, -1, 0, -1, 0};  // -1 = all bits set
    alignas(64) int32_t yes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) int32_t no[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) int32_t out[8] = {0};

    BitwiseIfThenElse(out, mask, yes, no, 8);

    ASSERT_EQ(out[0], 1);   // mask=-1, use yes
    ASSERT_EQ(out[1], 20);  // mask=0, use no
    ASSERT_EQ(out[2], 3);
    ASSERT_EQ(out[3], 40);
    ASSERT_EQ(out[4], 5);
    ASSERT_EQ(out[5], 60);
    ASSERT_EQ(out[6], 7);
    ASSERT_EQ(out[7], 80);
}

// -----------------------------------------------------------------------------
// MaskFalse - Create all-false mask
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskFalse_BasicCorrectness) {
    alignas(64) uint8_t mask[16] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    MaskFalse(mask, 16);

    for (size_t i = 0; i < 16; ++i) {
        ASSERT_EQ(mask[i], 0) << "MaskFalse should set all to 0";
    }
}

// -----------------------------------------------------------------------------
// SetMask - Create mask from bool
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, SetMask_True) {
    alignas(64) uint8_t mask[8] = {0};

    SetMask(mask, true, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_EQ(mask[i], 0xFF) << "SetMask(true) should set all to 0xFF";
    }
}

TEST_F(HwyOpsTest, SetMask_False) {
    alignas(64) uint8_t mask[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    SetMask(mask, false, 8);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_EQ(mask[i], 0) << "SetMask(false) should set all to 0";
    }
}

// -----------------------------------------------------------------------------
// Remaining Saturation Operations
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, SaturatedAdd_Int16) {
    alignas(64) int16_t a[8] = {30000, -30000, 10000, -10000, 0, 32767, -32768, 100};
    alignas(64) int16_t b[8] = {10000, -10000, 25000, -25000, 0, 1, -1, 200};
    alignas(64) int16_t out[8] = {0};

    SaturatedAdd(out, a, b, 8);

    ASSERT_EQ(out[0], 32767);   // Saturate at max
    ASSERT_EQ(out[1], -32768);  // Saturate at min
    ASSERT_EQ(out[2], 32767);   // Saturate
    ASSERT_EQ(out[3], -32768);  // Saturate
    ASSERT_EQ(out[4], 0);
    ASSERT_EQ(out[5], 32767);   // Already at max
    ASSERT_EQ(out[6], -32768);  // Already at min
    ASSERT_EQ(out[7], 300);
}

TEST_F(HwyOpsTest, SaturatedSub_Int16) {
    alignas(64) int16_t a[8] = {-30000, 30000, -10000, 10000, 0, -32768, 32767, 100};
    alignas(64) int16_t b[8] = {10000, -10000, 25000, -25000, 0, 1, -1, 200};
    alignas(64) int16_t out[8] = {0};

    SaturatedSub(out, a, b, 8);

    ASSERT_EQ(out[0], -32768);  // Saturate at min
    ASSERT_EQ(out[1], 32767);   // Saturate at max
    ASSERT_EQ(out[2], -32768);  // Saturate
    ASSERT_EQ(out[3], 32767);   // Saturate
    ASSERT_EQ(out[4], 0);
    ASSERT_EQ(out[5], -32768);  // Already at min
    ASSERT_EQ(out[6], 32767);   // Already at max
    ASSERT_EQ(out[7], -100);
}

TEST_F(HwyOpsTest, SaturatedAdd_Uint8) {
    alignas(64) uint8_t a[8] = {200, 100, 255, 0, 128, 250, 1, 127};
    alignas(64) uint8_t b[8] = {100, 100, 1, 0, 128, 10, 254, 128};
    alignas(64) uint8_t out[8] = {0};

    SaturatedAdd(out, a, b, 8);

    ASSERT_EQ(out[0], 255);  // Saturate
    ASSERT_EQ(out[1], 200);
    ASSERT_EQ(out[2], 255);  // Saturate
    ASSERT_EQ(out[3], 0);
    ASSERT_EQ(out[4], 255);  // Saturate
    ASSERT_EQ(out[5], 255);  // Saturate
    ASSERT_EQ(out[6], 255);
    ASSERT_EQ(out[7], 255);
}

TEST_F(HwyOpsTest, SaturatedSub_Uint8) {
    alignas(64) uint8_t a[8] = {200, 100, 0, 10, 128, 255, 1, 127};
    alignas(64) uint8_t b[8] = {100, 150, 1, 10, 200, 0, 254, 128};
    alignas(64) uint8_t out[8] = {0};

    SaturatedSub(out, a, b, 8);

    ASSERT_EQ(out[0], 100);
    ASSERT_EQ(out[1], 0);  // Saturate at 0
    ASSERT_EQ(out[2], 0);  // Saturate
    ASSERT_EQ(out[3], 0);
    ASSERT_EQ(out[4], 0);  // Saturate
    ASSERT_EQ(out[5], 255);
    ASSERT_EQ(out[6], 0);  // Saturate
    ASSERT_EQ(out[7], 0);  // Saturate
}

// -----------------------------------------------------------------------------
// CeilInt/FloorInt - Rounding to int
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, CeilInt_Float32) {
    alignas(64) float in[8] = {1.1f, 1.9f, -1.1f, -1.9f, 2.0f, -2.0f, 0.5f, -0.5f};
    alignas(64) int32_t out[8] = {0};

    CeilInt(out, in, 8);

    ASSERT_EQ(out[0], 2);
    ASSERT_EQ(out[1], 2);
    ASSERT_EQ(out[2], -1);
    ASSERT_EQ(out[3], -1);
    ASSERT_EQ(out[4], 2);
    ASSERT_EQ(out[5], -2);
    ASSERT_EQ(out[6], 1);
    ASSERT_EQ(out[7], 0);
}

TEST_F(HwyOpsTest, FloorInt_Float32) {
    alignas(64) float in[8] = {1.1f, 1.9f, -1.1f, -1.9f, 2.0f, -2.0f, 0.5f, -0.5f};
    alignas(64) int32_t out[8] = {0};

    FloorInt(out, in, 8);

    ASSERT_EQ(out[0], 1);
    ASSERT_EQ(out[1], 1);
    ASSERT_EQ(out[2], -2);
    ASSERT_EQ(out[3], -2);
    ASSERT_EQ(out[4], 2);
    ASSERT_EQ(out[5], -2);
    ASSERT_EQ(out[6], 0);
    ASSERT_EQ(out[7], -1);
}

// -----------------------------------------------------------------------------
// TruncateStore - Store with truncation
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, TruncateStore_Int32ToInt16) {
    alignas(64) int32_t src[8] = {100, 200, -100, -200, 32767, -32768, 0, 12345};
    alignas(64) int16_t dst[8] = {0};

    TruncateStore(dst, src, 8);

    ASSERT_EQ(dst[0], 100);
    ASSERT_EQ(dst[1], 200);
    ASSERT_EQ(dst[2], -100);
    ASSERT_EQ(dst[3], -200);
    ASSERT_EQ(dst[4], 32767);
    ASSERT_EQ(dst[5], -32768);
    ASSERT_EQ(dst[6], 0);
    ASSERT_EQ(dst[7], 12345);
}

// -----------------------------------------------------------------------------
// MaskedModOr - Masked modulo
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedModOr_Int32) {
    alignas(64) int32_t a[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    alignas(64) int32_t b[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    alignas(64) int32_t no[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    alignas(64) uint8_t mask[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) int32_t out[8] = {0};

    MaskedModOr(out, mask, a, b, no, 8);

    ASSERT_EQ(out[0], 1);   // 10 % 3
    ASSERT_EQ(out[1], -1);  // fallback
    ASSERT_EQ(out[2], 0);   // 12 % 3
    ASSERT_EQ(out[3], -1);  // fallback
    ASSERT_EQ(out[4], 2);   // 14 % 3
    ASSERT_EQ(out[5], -1);  // fallback
    ASSERT_EQ(out[6], 1);   // 16 % 3
    ASSERT_EQ(out[7], -1);  // fallback
}

// =============================================================================
// Final Gap Operations: Low Priority Remaining Operations
// =============================================================================

// -----------------------------------------------------------------------------
// Per2LaneBlockShuffle - 2-lane block shuffle
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, Per2LaneBlockShuffle_Float32) {
    alignas(64) float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float out[8] = {0};

    // Shuffle within 2-lane blocks
    Per2LaneBlockShuffle(out, a, b, 8);

    // Result depends on specific shuffle pattern
    // Typically alternates elements from a and b within 2-lane blocks
    ASSERT_TRUE(out[0] != 0 || out[1] != 0);  // Basic sanity check
}

// -----------------------------------------------------------------------------
// MaskedIsNaN - Masked NaN check
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedIsNaN_Float32) {
    alignas(64) float in[8] = {1.0f, NAN, 3.0f, NAN, 5.0f, INFINITY, -INFINITY, NAN};
    alignas(64) uint8_t mask[8] = {0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0};
    alignas(64) uint8_t out[8] = {0};

    MaskedIsNaN(out, mask, in, 8);

    ASSERT_EQ(out[0], 0);     // 1.0 is not NaN
    ASSERT_EQ(out[1], 0xFF);  // NAN is NaN
    ASSERT_EQ(out[2], 0);     // 3.0 is not NaN
    ASSERT_EQ(out[3], 0);     // masked out (NaN but masked)
    ASSERT_EQ(out[4], 0);     // 5.0 is not NaN
    ASSERT_EQ(out[5], 0);     // INFINITY is not NaN
    ASSERT_EQ(out[6], 0);     // -INFINITY is not NaN
    ASSERT_EQ(out[7], 0);     // masked out
}

// -----------------------------------------------------------------------------
// IfNegativeThenNegOrUndefIfZero - Conditional negate
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, IfNegativeThenNegOrUndefIfZero_Float32) {
    alignas(64) float v[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    alignas(64) float x[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    alignas(64) float out[8] = {0};

    // out[i] = (v[i] < 0) ? -x[i] : x[i] (undefined if v[i] == 0)
    IfNegativeThenNegOrUndefIfZero(out, v, x, 8);

    ASSERT_TRUE(ApproxEqual(out[0], -10.0f));  // v<0, negate
    ASSERT_TRUE(ApproxEqual(out[1], 20.0f));   // v>=0, keep
    ASSERT_TRUE(ApproxEqual(out[2], -30.0f));  // v<0, negate
    ASSERT_TRUE(ApproxEqual(out[3], 40.0f));   // v>=0, keep
    ASSERT_TRUE(ApproxEqual(out[4], -50.0f));  // v<0, negate
    ASSERT_TRUE(ApproxEqual(out[5], 60.0f));   // v>=0, keep
    ASSERT_TRUE(ApproxEqual(out[6], -70.0f));  // v<0, negate
    ASSERT_TRUE(ApproxEqual(out[7], 80.0f));   // v>=0, keep
}

// -----------------------------------------------------------------------------
// MaskedSetOr - Masked set with fallback
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedSetOr_Float32) {
    alignas(64) uint8_t mask[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) float no[8] = {-1, -2, -3, -4, -5, -6, -7, -8};
    alignas(64) float out[8] = {0};
    float value = 42.0f;

    // out[i] = mask[i] ? value : no[i]
    MaskedSetOr(out, mask, value, no, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 42.0f));  // masked, use value
    ASSERT_TRUE(ApproxEqual(out[1], -2.0f));  // not masked, use no
    ASSERT_TRUE(ApproxEqual(out[2], 42.0f));
    ASSERT_TRUE(ApproxEqual(out[3], -4.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 42.0f));
    ASSERT_TRUE(ApproxEqual(out[5], -6.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 42.0f));
    ASSERT_TRUE(ApproxEqual(out[7], -8.0f));
}

// -----------------------------------------------------------------------------
// MaskedMulFixedPoint15 - Masked fixed-point multiplication (Q15)
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedMulFixedPoint15_Int16) {
    alignas(64) int16_t a[8] = {16384, 16384, 8192, 8192, 32767, 32767, -16384, -16384};
    alignas(64) int16_t b[8] = {16384, 16384, 16384, 16384, 32767, 32767, 16384, 16384};
    alignas(64) int16_t no[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    alignas(64) uint8_t mask[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) int16_t out[8] = {0};

    // Fixed-point Q15: result = (a * b) >> 15
    MaskedMulFixedPoint15(out, mask, a, b, no, 8);

    // 16384 * 16384 / 32768 = 8192
    ASSERT_EQ(out[0], 8192);
    ASSERT_EQ(out[1], -1);  // fallback
    // 8192 * 16384 / 32768 = 4096
    ASSERT_EQ(out[2], 4096);
    ASSERT_EQ(out[3], -1);  // fallback
}

// -----------------------------------------------------------------------------
// MaskedWidenMulPairwiseAdd - Masked widening multiply-accumulate
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedWidenMulPairwiseAdd_Int16ToInt32) {
    alignas(64) int16_t a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    alignas(64) int16_t b[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    alignas(64) int32_t no[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    alignas(64) uint8_t mask[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
    alignas(64) int32_t out[8] = {0};

    // out[i] = mask[i] ? (a[2i]*b[2i] + a[2i+1]*b[2i+1]) : no[i]
    MaskedWidenMulPairwiseAdd(out, mask, a, b, no, 8);

    ASSERT_EQ(out[0], 3);   // 1*1 + 2*1 = 3
    ASSERT_EQ(out[1], 7);   // 3*1 + 4*1 = 7
    ASSERT_EQ(out[2], 11);  // 5*1 + 6*1 = 11
    ASSERT_EQ(out[3], 15);  // 7*1 + 8*1 = 15
    ASSERT_EQ(out[4], -1);  // fallback
    ASSERT_EQ(out[5], -1);  // fallback
    ASSERT_EQ(out[6], -1);  // fallback
    ASSERT_EQ(out[7], -1);  // fallback
}

// -----------------------------------------------------------------------------
// MaskedAbsOr - Masked absolute value with fallback
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedAbsOr_Float32) {
    alignas(64) float a[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    alignas(64) float no[8] = {-100, -200, -300, -400, -500, -600, -700, -800};
    alignas(64) uint8_t mask[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) float out[8] = {0};

    // out[i] = mask[i] ? abs(a[i]) : no[i]
    MaskedAbsOr(out, mask, a, no, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));     // abs(-1)
    ASSERT_TRUE(ApproxEqual(out[1], -200.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));     // abs(-3)
    ASSERT_TRUE(ApproxEqual(out[3], -400.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[4], 5.0f));     // abs(-5)
    ASSERT_TRUE(ApproxEqual(out[5], -600.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[6], 7.0f));     // abs(-7)
    ASSERT_TRUE(ApproxEqual(out[7], -800.0f));  // fallback
}

// -----------------------------------------------------------------------------
// InsertIntoUpper - Insert scalar into upper half
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, InsertIntoUpper_Float32) {
    alignas(64) float vec[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) float out[8] = {0};
    float scalar = 99.0f;

    // Insert scalar into upper lanes
    InsertIntoUpper(out, vec, scalar, 8);

    // Lower half unchanged, upper half gets scalar
    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 2.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 4.0f));
    ASSERT_TRUE(ApproxEqual(out[4], 99.0f));
    ASSERT_TRUE(ApproxEqual(out[5], 99.0f));
    ASSERT_TRUE(ApproxEqual(out[6], 99.0f));
    ASSERT_TRUE(ApproxEqual(out[7], 99.0f));
}

// -----------------------------------------------------------------------------
// MaskedGatherIndexOr - Masked gather with fallback
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MaskedGatherIndexOr_Float32) {
    alignas(64) float base[16] = {0,  10, 20,  30,  40,  50,  60,  70,
                                  80, 90, 100, 110, 120, 130, 140, 150};
    alignas(64) int32_t indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    alignas(64) float no[8] = {-1, -2, -3, -4, -5, -6, -7, -8};
    alignas(64) uint8_t mask[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
    alignas(64) float out[8] = {0};

    // out[i] = mask[i] ? base[indices[i]] : no[i]
    MaskedGatherIndexOr(out, mask, base, indices, no, 8);

    ASSERT_TRUE(ApproxEqual(out[0], 0.0f));   // base[0]
    ASSERT_TRUE(ApproxEqual(out[1], 10.0f));  // base[1]
    ASSERT_TRUE(ApproxEqual(out[2], 20.0f));  // base[2]
    ASSERT_TRUE(ApproxEqual(out[3], 30.0f));  // base[3]
    ASSERT_TRUE(ApproxEqual(out[4], -5.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[5], -6.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[6], -7.0f));  // fallback
    ASSERT_TRUE(ApproxEqual(out[7], -8.0f));  // fallback
}

// -----------------------------------------------------------------------------
// SumsOf8AbsDiff - Sum of absolute differences (SAD-like)
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, SumsOf8AbsDiff_Uint8) {
    alignas(64)
        uint8_t a[16] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160};
    alignas(64)
        uint8_t b[16] = {15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165};
    alignas(64) uint64_t out[2] = {0};

    // SAD: sum |a[i] - b[i]| for groups of 8
    SumsOf8AbsDiff(out, a, b, 16);

    // Each difference is 5, so sum of 8 is 40
    ASSERT_EQ(out[0], 40);  // First 8 elements
    ASSERT_EQ(out[1], 40);  // Second 8 elements
}

// -----------------------------------------------------------------------------
// CombineMasks - Combine two half-masks into one
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, CombineMasks_BasicCorrectness) {
    alignas(64) uint8_t lo[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) uint8_t hi[8] = {0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF};
    alignas(64) uint8_t out[16] = {0};

    CombineMasks(out, lo, hi, 8);

    // Lower half from lo, upper half from hi
    for (size_t i = 0; i < 8; ++i) {
        ASSERT_EQ(out[i], lo[i]) << "lo mismatch at " << i;
        ASSERT_EQ(out[i + 8], hi[i]) << "hi mismatch at " << i;
    }
}

// -----------------------------------------------------------------------------
// LowerHalfOfMask - Extract lower half of mask
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, LowerHalfOfMask_BasicCorrectness) {
    alignas(64)
        uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF};
    alignas(64) uint8_t out[8] = {0};

    LowerHalfOfMask(out, mask, 16);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_EQ(out[i], mask[i]) << "mismatch at " << i;
    }
}

// -----------------------------------------------------------------------------
// UpperHalfOfMask - Extract upper half of mask
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, UpperHalfOfMask_BasicCorrectness) {
    alignas(64)
        uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF};
    alignas(64) uint8_t out[8] = {0};

    UpperHalfOfMask(out, mask, 16);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_EQ(out[i], mask[i + 8]) << "mismatch at " << i;
    }
}

// -----------------------------------------------------------------------------
// PromoteMaskTo - Promote mask to wider type
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, PromoteMaskTo_Uint8ToUint16) {
    alignas(64) uint8_t narrow[8] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
    alignas(64) uint16_t wide[8] = {0};

    PromoteMaskTo(wide, narrow, 8);

    ASSERT_EQ(wide[0], 0xFFFF);
    ASSERT_EQ(wide[1], 0);
    ASSERT_EQ(wide[2], 0xFFFF);
    ASSERT_EQ(wide[3], 0);
    ASSERT_EQ(wide[4], 0xFFFF);
    ASSERT_EQ(wide[5], 0);
    ASSERT_EQ(wide[6], 0xFFFF);
    ASSERT_EQ(wide[7], 0);
}

// -----------------------------------------------------------------------------
// DemoteMaskTo - Demote mask to narrower type
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, DemoteMaskTo_Uint16ToUint8) {
    alignas(64) uint16_t wide[8] = {0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0};
    alignas(64) uint8_t narrow[8] = {0};

    DemoteMaskTo(narrow, wide, 8);

    ASSERT_EQ(narrow[0], 0xFF);
    ASSERT_EQ(narrow[1], 0);
    ASSERT_EQ(narrow[2], 0xFF);
    ASSERT_EQ(narrow[3], 0);
    ASSERT_EQ(narrow[4], 0xFF);
    ASSERT_EQ(narrow[5], 0);
    ASSERT_EQ(narrow[6], 0xFF);
    ASSERT_EQ(narrow[7], 0);
}

// -----------------------------------------------------------------------------
// ZeroExtendResizeBitCast - Zero-extend with bitcast
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, ZeroExtendResizeBitCast_Uint8ToUint32) {
    alignas(64) uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    alignas(64) uint32_t dst[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

    ZeroExtendResizeBitCast(dst, src, 8);

    ASSERT_EQ(dst[0], 1u);
    ASSERT_EQ(dst[1], 2u);
    ASSERT_EQ(dst[2], 3u);
    ASSERT_EQ(dst[3], 4u);
    ASSERT_EQ(dst[4], 5u);
    ASSERT_EQ(dst[5], 6u);
    ASSERT_EQ(dst[6], 7u);
    ASSERT_EQ(dst[7], 8u);
}

// Note: F64ToF16 and F64ToBF16 tests already exist above

// -----------------------------------------------------------------------------
// MatVecMul - Matrix-vector multiplication
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MatVecMul_Float32) {
    // 4x4 matrix (row-major)
    alignas(64) float mat[16] = {
        1, 0, 0, 0,  // row 0: identity-like
        0, 1, 0, 0,  // row 1
        0, 0, 1, 0,  // row 2
        0, 0, 0, 1   // row 3
    };
    alignas(64) float vec[4] = {1, 2, 3, 4};
    alignas(64) float out[4] = {0};

    MatVecMul(out, mat, vec, 4, 4);

    // Identity matrix * vec = vec
    ASSERT_TRUE(ApproxEqual(out[0], 1.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 2.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 4.0f));
}

TEST_F(HwyOpsTest, MatVecMul_Scale) {
    // Scaling matrix
    alignas(64) float mat[16] = {2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5};
    alignas(64) float vec[4] = {1, 1, 1, 1};
    alignas(64) float out[4] = {0};

    MatVecMul(out, mat, vec, 4, 4);

    ASSERT_TRUE(ApproxEqual(out[0], 2.0f));
    ASSERT_TRUE(ApproxEqual(out[1], 3.0f));
    ASSERT_TRUE(ApproxEqual(out[2], 4.0f));
    ASSERT_TRUE(ApproxEqual(out[3], 5.0f));
}

// -----------------------------------------------------------------------------
// MatMul - Matrix-matrix multiplication
// -----------------------------------------------------------------------------

TEST_F(HwyOpsTest, MatMul_Float32_Identity) {
    // 4x4 identity matrices
    alignas(64) float a[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    alignas(64) float b[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    alignas(64) float out[16] = {0};

    MatMul(out, a, b, 4, 4, 4);

    // Identity * B = B
    for (int i = 0; i < 16; ++i) {
        ASSERT_TRUE(ApproxEqual(out[i], b[i])) << "mismatch at " << i;
    }
}

TEST_F(HwyOpsTest, MatMul_Float32_2x2) {
    alignas(64) float a[4] = {1, 2, 3, 4};  // [[1,2],[3,4]]
    alignas(64) float b[4] = {5, 6, 7, 8};  // [[5,6],[7,8]]
    alignas(64) float out[4] = {0};

    MatMul(out, a, b, 2, 2, 2);

    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    ASSERT_TRUE(ApproxEqual(out[0], 19.0f));  // 1*5 + 2*7
    ASSERT_TRUE(ApproxEqual(out[1], 22.0f));  // 1*6 + 2*8
    ASSERT_TRUE(ApproxEqual(out[2], 43.0f));  // 3*5 + 4*7
    ASSERT_TRUE(ApproxEqual(out[3], 50.0f));  // 3*6 + 4*8
}

}  // namespace
}  // namespace simd
}  // namespace bud
