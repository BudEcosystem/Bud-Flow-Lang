// =============================================================================
// Bud Flow Lang - Copy-and-Patch JIT Tests
// =============================================================================

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/stencil.h"
#include "bud_flow_lang/type_system.h"

#include <hwy/aligned_allocator.h>

#include <cmath>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace jit {
namespace {

// =============================================================================
// Test Environment (initialized once for all tests)
// =============================================================================

class JitTestEnvironment : public ::testing::Environment {
  public:
    void SetUp() override {
        auto result = initializeCompiler();
        if (!result.hasValue()) {
            std::cerr << "Failed to initialize JIT compiler\n";
            std::exit(1);
        }
    }

    void TearDown() override { shutdownCompiler(); }
};

// Register the environment
static ::testing::Environment* const jit_env =
    ::testing::AddGlobalTestEnvironment(new JitTestEnvironment());

// =============================================================================
// Test Fixture
// =============================================================================

class CopyPatchTest : public ::testing::Test {
  protected:
    // Helper to allocate aligned memory for SIMD operations
    template <typename T>
    std::vector<T, hwy::AlignedAllocator<T>> allocateAligned(size_t count) {
        return std::vector<T, hwy::AlignedAllocator<T>>(count);
    }
};

// =============================================================================
// Stencil Registry Tests
// =============================================================================

TEST_F(CopyPatchTest, StencilRegistryHasEntries) {
    // Verify stencils were registered
    size_t count = stencilCount();
    EXPECT_GT(count, 0) << "No stencils registered";
}

TEST_F(CopyPatchTest, StencilRegistryFindAdd) {
    // Should find add stencil for float32
    const Stencil* stencil = findStencil(ir::OpCode::kAdd, ScalarType::kFloat32);
    EXPECT_NE(stencil, nullptr) << "Add stencil not found";
    if (stencil) {
        EXPECT_EQ(stencil->op, ir::OpCode::kAdd);
        EXPECT_EQ(stencil->dtype, ScalarType::kFloat32);
        EXPECT_GT(stencil->code.size(), 0) << "Stencil has no code";
    }
}

TEST_F(CopyPatchTest, StencilRegistryFindMul) {
    const Stencil* stencil = findStencil(ir::OpCode::kMul, ScalarType::kFloat32);
    EXPECT_NE(stencil, nullptr) << "Mul stencil not found";
}

TEST_F(CopyPatchTest, StencilRegistryFindFma) {
    const Stencil* stencil = findStencil(ir::OpCode::kFma, ScalarType::kFloat32);
    EXPECT_NE(stencil, nullptr) << "FMA stencil not found";
}

TEST_F(CopyPatchTest, StencilRegistryFindUnary) {
    // Test unary operations
    EXPECT_NE(findStencil(ir::OpCode::kNeg, ScalarType::kFloat32), nullptr);
    EXPECT_NE(findStencil(ir::OpCode::kAbs, ScalarType::kFloat32), nullptr);
    EXPECT_NE(findStencil(ir::OpCode::kSqrt, ScalarType::kFloat32), nullptr);
    EXPECT_NE(findStencil(ir::OpCode::kExp, ScalarType::kFloat32), nullptr);
}

TEST_F(CopyPatchTest, StencilRegistryHasStencil) {
    EXPECT_TRUE(hasStencil(ir::OpCode::kAdd, ScalarType::kFloat32));
    EXPECT_TRUE(hasStencil(ir::OpCode::kMul, ScalarType::kFloat32));
    // Should not have stencil for unsupported op/dtype combo
    // (This depends on implementation - adjust if needed)
}

// =============================================================================
// Highway Function Pointer Tests
// =============================================================================

TEST_F(CopyPatchTest, GetHwyFunctionPtrAdd) {
    void* ptr = getHwyFunctionPtr(ir::OpCode::kAdd, ScalarType::kFloat32);
    EXPECT_NE(ptr, nullptr) << "No Highway function pointer for Add";
}

TEST_F(CopyPatchTest, GetHwyFunctionPtrMul) {
    void* ptr = getHwyFunctionPtr(ir::OpCode::kMul, ScalarType::kFloat32);
    EXPECT_NE(ptr, nullptr) << "No Highway function pointer for Mul";
}

TEST_F(CopyPatchTest, GetHwyFunctionPtrSqrt) {
    void* ptr = getHwyFunctionPtr(ir::OpCode::kSqrt, ScalarType::kFloat32);
    EXPECT_NE(ptr, nullptr) << "No Highway function pointer for Sqrt";
}

// =============================================================================
// JIT Binary Operation Tests
// =============================================================================

TEST_F(CopyPatchTest, JitBinaryAddFloat32) {
    constexpr size_t N = 256;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);

    // Initialize test data
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
        c[i] = 0.0f;
    }

    // Execute JIT add
    auto result =
        executeJitBinaryOp(ir::OpCode::kAdd, ScalarType::kFloat32, c.data(), a.data(), b.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT add failed: " << result.error().message();

    // Verify results
    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] + b[i];
        EXPECT_FLOAT_EQ(c[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(CopyPatchTest, JitBinaryMulFloat32) {
    constexpr size_t N = 256;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    auto result =
        executeJitBinaryOp(ir::OpCode::kMul, ScalarType::kFloat32, c.data(), a.data(), b.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT mul failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] * b[i];
        EXPECT_FLOAT_EQ(c[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(CopyPatchTest, JitBinarySubFloat32) {
    constexpr size_t N = 128;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i * 3);
        b[i] = static_cast<float>(i);
        c[i] = 0.0f;
    }

    auto result =
        executeJitBinaryOp(ir::OpCode::kSub, ScalarType::kFloat32, c.data(), a.data(), b.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT sub failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] - b[i];
        EXPECT_FLOAT_EQ(c[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(CopyPatchTest, JitBinaryDivFloat32) {
    constexpr size_t N = 128;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i + 1) * 4.0f;
        b[i] = 2.0f;  // Avoid division by zero
        c[i] = 0.0f;
    }

    auto result =
        executeJitBinaryOp(ir::OpCode::kDiv, ScalarType::kFloat32, c.data(), a.data(), b.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT div failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] / b[i];
        EXPECT_FLOAT_EQ(c[i], expected) << "Mismatch at index " << i;
    }
}

// =============================================================================
// JIT Unary Operation Tests
// =============================================================================

TEST_F(CopyPatchTest, JitUnaryNegFloat32) {
    constexpr size_t N = 256;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) - 128.0f;  // Mix of positive and negative
        b[i] = 0.0f;
    }

    auto result = executeJitUnaryOp(ir::OpCode::kNeg, ScalarType::kFloat32, b.data(), a.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT neg failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = -a[i];
        EXPECT_FLOAT_EQ(b[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(CopyPatchTest, JitUnaryAbsFloat32) {
    constexpr size_t N = 256;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) - 128.0f;
        b[i] = 0.0f;
    }

    auto result = executeJitUnaryOp(ir::OpCode::kAbs, ScalarType::kFloat32, b.data(), a.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT abs failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = std::abs(a[i]);
        EXPECT_FLOAT_EQ(b[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(CopyPatchTest, JitUnarySqrtFloat32) {
    constexpr size_t N = 128;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i + 1);  // Positive values for sqrt
        b[i] = 0.0f;
    }

    auto result = executeJitUnaryOp(ir::OpCode::kSqrt, ScalarType::kFloat32, b.data(), a.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT sqrt failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = std::sqrt(a[i]);
        EXPECT_NEAR(b[i], expected, 1e-5f) << "Mismatch at index " << i;
    }
}

// =============================================================================
// JIT FMA Operation Tests
// =============================================================================

TEST_F(CopyPatchTest, JitFmaFloat32) {
    constexpr size_t N = 256;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);
    auto d = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f;
        c[i] = 1.0f;
        d[i] = 0.0f;
    }

    auto result = executeJitFmaOp(ScalarType::kFloat32, d.data(), a.data(), b.data(), c.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT FMA failed";

    for (size_t i = 0; i < N; ++i) {
        float expected = a[i] * b[i] + c[i];  // FMA: a * b + c
        EXPECT_FLOAT_EQ(d[i], expected) << "Mismatch at index " << i;
    }
}

// =============================================================================
// JIT Statistics Tests
// =============================================================================

TEST_F(CopyPatchTest, JitStatsAfterCompilation) {
    // Perform some operations to trigger compilation
    constexpr size_t N = 64;
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 1.0f;
        c[i] = 0.0f;
    }

    // Execute an operation
    auto result =
        executeJitBinaryOp(ir::OpCode::kAdd, ScalarType::kFloat32, c.data(), a.data(), b.data(), N);
    ASSERT_TRUE(result.hasValue());

    // Check stats
    JitStats stats = getJitStats();
    EXPECT_GT(stats.stencil_count, 0) << "No stencils registered";
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(CopyPatchTest, JitWithSmallArrays) {
    // Test with very small arrays (edge cases for vectorization)
    for (size_t n : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17}) {
        auto a = allocateAligned<float>(n);
        auto b = allocateAligned<float>(n);
        auto c = allocateAligned<float>(n);

        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = 1.0f;
        }

        auto result = executeJitBinaryOp(ir::OpCode::kAdd, ScalarType::kFloat32, c.data(), a.data(),
                                         b.data(), n);
        ASSERT_TRUE(result.hasValue()) << "Failed for size " << n;

        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << "Mismatch at index " << i << " for size " << n;
        }
    }
}

TEST_F(CopyPatchTest, JitWithLargeArrays) {
    // Test with large arrays
    constexpr size_t N = 1024 * 1024;  // 1M elements
    auto a = allocateAligned<float>(N);
    auto b = allocateAligned<float>(N);
    auto c = allocateAligned<float>(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i % 1000);
        b[i] = 1.0f;
    }

    auto result =
        executeJitBinaryOp(ir::OpCode::kAdd, ScalarType::kFloat32, c.data(), a.data(), b.data(), N);
    ASSERT_TRUE(result.hasValue()) << "JIT failed for large array";

    // Spot check results
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
    }
    for (size_t i = N - 100; i < N; ++i) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
    }
}

}  // namespace
}  // namespace jit
}  // namespace bud
