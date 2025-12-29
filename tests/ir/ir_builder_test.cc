// =============================================================================
// Bud Flow Lang - IR Builder Tests
// =============================================================================

#include "bud_flow_lang/ir.h"

#include <gtest/gtest.h>

namespace bud {
namespace ir {
namespace {

TEST(IRBuilderTest, CreateConstant) {
    Arena arena;
    IRBuilder builder(arena);

    auto id = builder.constant(3.14f);
    ASSERT_TRUE(id.isValid());

    auto* node = builder.getNode(id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->opCode(), OpCode::kConstantScalar);
    EXPECT_FLOAT_EQ(node->floatAttr("value"), 3.14);
}

TEST(IRBuilderTest, CreateBinaryOp) {
    Arena arena;
    IRBuilder builder(arena);

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto sum = builder.add(a, b);

    ASSERT_TRUE(sum.isValid());

    auto* node = builder.getNode(sum);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->opCode(), OpCode::kAdd);
    EXPECT_EQ(node->numOperands(), 2u);
}

TEST(IRBuilderTest, CreateUnaryOp) {
    Arena arena;
    IRBuilder builder(arena);

    auto x = builder.constant(4.0f);
    auto sqrt_x = builder.sqrt(x);

    ASSERT_TRUE(sqrt_x.isValid());

    auto* node = builder.getNode(sqrt_x);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->opCode(), OpCode::kSqrt);
    EXPECT_EQ(node->numOperands(), 1u);
}

TEST(IRBuilderTest, CreateFMA) {
    Arena arena;
    IRBuilder builder(arena);

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);
    auto fma = builder.fma(a, b, c);

    ASSERT_TRUE(fma.isValid());

    auto* node = builder.getNode(fma);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->opCode(), OpCode::kFma);
    EXPECT_EQ(node->numOperands(), 3u);
}

TEST(IRBuilderTest, SSAProperty) {
    Arena arena;
    IRBuilder builder(arena);

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto sum = builder.add(a, b);

    // Check SSA: operands must be defined before use
    auto result = builder.validate();
    EXPECT_TRUE(result.hasValue());
}

TEST(IRBuilderTest, Dump) {
    Arena arena;
    IRBuilder builder(arena);

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto sum = builder.add(a, b);
    auto result = builder.sqrt(sum);

    std::string dump = builder.dump();
    EXPECT_FALSE(dump.empty());
    EXPECT_NE(dump.find("constant"), std::string::npos);
    EXPECT_NE(dump.find("add"), std::string::npos);
    EXPECT_NE(dump.find("sqrt"), std::string::npos);
}

TEST(IRModuleTest, Construction) {
    IRModule module("test_module");
    EXPECT_EQ(module.name(), "test_module");
}

TEST(IRModuleTest, BuildAndOptimize) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto sum = builder.add(a, b);
    module.setOutput(sum);

    auto result = module.optimize(1);
    EXPECT_TRUE(result.hasValue());
}

}  // namespace
}  // namespace ir
}  // namespace bud
