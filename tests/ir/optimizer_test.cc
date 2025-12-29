// =============================================================================
// Bud Flow Lang - IR Optimizer Tests
// =============================================================================

#include "bud_flow_lang/ir.h"

#include <gtest/gtest.h>

namespace bud {
namespace ir {
namespace {

TEST(OptimizerTest, OptimizationLevelZero) {
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(1.0f);
    module.setOutput(x);

    auto result = module.optimize(0);
    EXPECT_TRUE(result.hasValue());
}

TEST(OptimizerTest, OptimizationLevelTwo) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(2.0f);
    auto b = builder.constant(3.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(1.0f);
    auto add = builder.add(mul, c);
    module.setOutput(add);

    auto result = module.optimize(2);
    EXPECT_TRUE(result.hasValue());
}

TEST(OptimizerTest, InvalidOptimizationLevel) {
    IRModule module("test");
    auto result = module.optimize(10);
    EXPECT_TRUE(result.hasError());
}

}  // namespace
}  // namespace ir
}  // namespace bud
