// =============================================================================
// Bud Flow Lang - Bunch Tests
// =============================================================================

#include "bud_flow_lang/bunch.h"

#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace {

TEST(BunchTest, CreateZeros) {
    auto result = Bunch::zeros(100);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    EXPECT_EQ(b.size(), 100u);
    EXPECT_EQ(b.dtype(), ScalarType::kFloat32);
}

TEST(BunchTest, CreateOnes) {
    auto result = Bunch::ones(100);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    EXPECT_EQ(b.size(), 100u);

    auto data = b.as<float>();
    for (size_t i = 0; i < b.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(BunchTest, CreateFromData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto result = Bunch::fromData(data.data(), data.size());
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    EXPECT_EQ(b.size(), 5u);

    auto view = b.as<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(view[i], data[i]);
    }
}

TEST(BunchTest, CreateFill) {
    auto result = Bunch::fill(100, 42.0f);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    auto data = b.as<float>();
    for (size_t i = 0; i < b.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 42.0f);
    }
}

TEST(BunchTest, CreateArange) {
    auto result = Bunch::arange(5, 0.0f, 1.0f);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    auto data = b.as<float>();

    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 3.0f);
    EXPECT_FLOAT_EQ(data[4], 4.0f);
}

TEST(BunchTest, Sum) {
    auto result = Bunch::arange(100, 0.0f, 1.0f);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    float sum = b.sum();

    // Sum of 0..99 = 99*100/2 = 4950
    EXPECT_FLOAT_EQ(sum, 4950.0f);
}

TEST(BunchTest, MinMax) {
    std::vector<float> data = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f};

    auto result = Bunch::fromData(data.data(), data.size());
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    EXPECT_FLOAT_EQ(b.min(), 1.0f);
    EXPECT_FLOAT_EQ(b.max(), 9.0f);
}

TEST(BunchTest, Mean) {
    auto result = Bunch::fill(100, 5.0f);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    EXPECT_FLOAT_EQ(b.mean(), 5.0f);
}

TEST(BunchTest, DotProduct) {
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> b_data = {4.0f, 5.0f, 6.0f};

    auto a_result = Bunch::fromData(a_data.data(), a_data.size());
    auto b_result = Bunch::fromData(b_data.data(), b_data.size());

    ASSERT_TRUE(a_result.hasValue());
    ASSERT_TRUE(b_result.hasValue());

    Bunch a = *a_result;
    Bunch b = *b_result;

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(a.dot(b), 32.0f);
}

TEST(BunchTest, CopyTo) {
    auto result = Bunch::arange(5, 0.0f, 1.0f);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    std::vector<float> dest(5);

    auto copy_result = b.copyTo(dest.data(), 5);
    EXPECT_TRUE(copy_result.hasValue());

    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(dest[i], static_cast<float>(i));
    }
}

TEST(BunchTest, ToString) {
    auto result = Bunch::zeros(100);
    ASSERT_TRUE(result.hasValue());

    Bunch b = *result;
    std::string str = b.toString();

    EXPECT_NE(str.find("Bunch"), std::string::npos);
    EXPECT_NE(str.find("100"), std::string::npos);
}

}  // namespace
}  // namespace bud
