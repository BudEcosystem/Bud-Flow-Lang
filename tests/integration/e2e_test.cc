// =============================================================================
// Bud Flow Lang - End-to-End Integration Tests
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <gtest/gtest.h>

namespace bud {
namespace {

class E2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize runtime for each test
        RuntimeConfig config;
        config.enable_debug_output = false;
        auto result = initialize(config);
        ASSERT_TRUE(result.hasValue() || isInitialized());
    }
};

TEST_F(E2ETest, InitializeRuntime) {
    EXPECT_TRUE(isInitialized());
}

TEST_F(E2ETest, HardwareDetection) {
    const auto& hw = getHardwareInfo();

    // Should detect something
    EXPECT_GT(hw.simd_width, 0u);
    EXPECT_GT(hw.physical_cores, 0);
}

TEST_F(E2ETest, SimpleComputation) {
    // Create input
    auto input_result = Bunch::arange(1000, 0.0f, 0.01f);
    ASSERT_TRUE(input_result.hasValue());

    Bunch input = *input_result;

    // Compute sum
    float sum = input.sum();
    EXPECT_GT(sum, 0.0f);
}

TEST_F(E2ETest, ChainedOperations) {
    // Create data
    auto a_result = Bunch::fill(100, 2.0f);
    auto b_result = Bunch::fill(100, 3.0f);

    ASSERT_TRUE(a_result.hasValue());
    ASSERT_TRUE(b_result.hasValue());

    Bunch a = *a_result;
    Bunch b = *b_result;

    // Dot product: sum(a * b) = 100 * 2 * 3 = 600
    float dot = a.dot(b);
    EXPECT_FLOAT_EQ(dot, 600.0f);
}

TEST_F(E2ETest, LargeArray) {
    // Test with 1M elements
    const size_t n = 1000000;

    auto result = Bunch::ones(n);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    EXPECT_EQ(b.size(), n);
    EXPECT_FLOAT_EQ(b.sum(), static_cast<float>(n));
}

}  // namespace
}  // namespace bud
