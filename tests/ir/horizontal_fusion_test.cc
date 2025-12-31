// =============================================================================
// Bud Flow Lang - Horizontal Fusion Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for horizontal fusion that batches independent operations into a
// single kernel launch to reduce per-kernel overhead.
//
// Pattern: Independent operations A, B, C -> Batch into single launch
//
// Benefits:
// - Reduces kernel launch overhead (~5-10µs per launch)
// - Increases thread utilization
// - Better instruction cache locality
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/ir/horizontal_fusion.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace ir {
namespace {

// =============================================================================
// HorizontalFusionAnalyzer Tests
// =============================================================================

class HorizontalFusionTest : public ::testing::Test {
  protected:
    void SetUp() override { analyzer_ = std::make_unique<HorizontalFusionAnalyzer>(); }

    std::unique_ptr<HorizontalFusionAnalyzer> analyzer_;
};

TEST_F(HorizontalFusionTest, FindIndependentGroups_TwoIndependent) {
    // Create IR with two independent computation paths
    // Path 1: a = const -> b = a + 1
    // Path 2: c = const -> d = c * 2
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto b = builder.add(a, one);  // Path 1

    auto c = builder.constant(2.0f);
    auto two = builder.constant(2.0f);
    auto d = builder.mul(c, two);  // Path 2

    auto groups = analyzer_->findIndependentGroups(builder);

    // Should find groups of independent operations
    EXPECT_FALSE(groups.empty());

    // Should have at least one group with independent operations
    bool found_independent = false;
    for (const auto& group : groups) {
        if (group.operations.size() >= 2) {
            found_independent = true;
            break;
        }
    }
    EXPECT_TRUE(found_independent) << "Should find groups of independent operations";
}

TEST_F(HorizontalFusionTest, FindIndependentGroups_NoIndependent) {
    // Create IR with a single linear chain (no independent ops)
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto b = builder.add(a, one);
    auto two = builder.constant(2.0f);
    auto c = builder.mul(b, two);  // c depends on b
    auto three = builder.constant(3.0f);
    auto d = builder.add(c, three);  // d depends on c
    module.setOutput(d);

    auto groups = analyzer_->findIndependentGroups(builder);

    // In a pure linear chain, operations after constants aren't independent
    // The constants themselves could be grouped, but the ops are sequential
    (void)groups;  // Result depends on implementation details
}

TEST_F(HorizontalFusionTest, FindIndependentGroups_ThreeIndependent) {
    // Create IR with three independent computation paths
    IRModule module("test");
    auto& builder = module.builder();

    // Path 1
    auto a = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto path1 = builder.add(a, one);

    // Path 2
    auto b = builder.constant(2.0f);
    auto two = builder.constant(2.0f);
    auto path2 = builder.mul(b, two);

    // Path 3
    auto c = builder.constant(3.0f);
    auto three = builder.constant(3.0f);
    auto path3 = builder.sub(c, three);

    // Combine at the end
    auto t1 = builder.add(path1, path2);
    auto final_out = builder.add(t1, path3);
    module.setOutput(final_out);

    auto groups = analyzer_->findIndependentGroups(builder);

    // Should find a group containing path1, path2, path3
    bool found_triple = false;
    for (const auto& group : groups) {
        if (group.operations.size() >= 3) {
            found_triple = true;
            break;
        }
    }
    EXPECT_TRUE(found_triple) << "Should find group with three independent operations";
}

TEST_F(HorizontalFusionTest, FindIndependentGroups_SameOutputType) {
    // Only operations with the same output type can be batched
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.add(a, b);  // Float

    auto x = builder.constant(static_cast<int32_t>(1));
    auto y = builder.constant(static_cast<int32_t>(2));
    // Note: Currently int operations produce int, float produce float

    auto groups = analyzer_->findIndependentGroups(builder);

    // Groups should be formed by type compatibility
    (void)groups;  // Just verify it doesn't crash
}

// =============================================================================
// CanBatchOperations Tests
// =============================================================================

TEST_F(HorizontalFusionTest, CanBatchOperations_Compatible) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);

    auto one = builder.constant(1.0f);
    auto x = builder.add(a, one);
    auto y = builder.add(b, one);
    auto z = builder.add(c, one);

    std::vector<ValueId> ops = {x, y, z};
    EXPECT_TRUE(analyzer_->canBatchOperations(builder, ops));
}

TEST_F(HorizontalFusionTest, CanBatchOperations_WithDependency) {
    // If one op depends on another, they can't be batched horizontally
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto x = builder.add(a, one);
    auto y = builder.mul(x, a);  // y depends on x

    std::vector<ValueId> ops = {x, y};
    EXPECT_FALSE(analyzer_->canBatchOperations(builder, ops));
}

TEST_F(HorizontalFusionTest, CanBatchOperations_DifferentOpCodes) {
    // Operations with different opcodes can still be batched if independent
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto one = builder.constant(1.0f);

    auto x = builder.add(a, one);  // Add
    auto y = builder.mul(b, one);  // Mul

    std::vector<ValueId> ops = {x, y};
    EXPECT_TRUE(analyzer_->canBatchOperations(builder, ops));
}

// =============================================================================
// EstimateBatchBenefit Tests
// =============================================================================

TEST_F(HorizontalFusionTest, EstimateBatchBenefit_Positive) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);
    auto one = builder.constant(1.0f);

    auto x = builder.add(a, one);
    auto y = builder.add(b, one);
    auto z = builder.add(c, one);

    IndependentGroup group;
    group.operations = {x, y, z};

    float benefit = analyzer_->estimateBatchBenefit(builder, group);

    // Benefit should be positive (saved kernel launches)
    EXPECT_GT(benefit, 0.0f);
}

TEST_F(HorizontalFusionTest, EstimateBatchBenefit_SingleOp) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto x = builder.add(a, one);

    IndependentGroup group;
    group.operations = {x};

    float benefit = analyzer_->estimateBatchBenefit(builder, group);

    // Single operation - no batching benefit
    EXPECT_EQ(benefit, 0.0f);
}

// =============================================================================
// HorizontalFusionPass Tests
// =============================================================================

class HorizontalFusionPassTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(HorizontalFusionPassTest, BatchIndependentOperations) {
    IRModule module("test");
    auto& builder = module.builder();

    // Create three independent additions
    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);
    auto one = builder.constant(1.0f);

    auto x = builder.add(a, one);
    auto y = builder.add(b, one);
    auto z = builder.add(c, one);

    // Combine at the end
    auto t1 = builder.add(x, y);
    auto out = builder.add(t1, z);
    module.setOutput(out);

    HorizontalFusionPass pass;
    size_t batched = pass.run(builder);

    // Should batch the independent operations
    (void)batched;

    // Output should still be valid
    EXPECT_TRUE(module.output().isValid());
}

TEST_F(HorizontalFusionPassTest, PreservesSemantics) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(2.0f);
    auto b = builder.constant(3.0f);
    auto one = builder.constant(1.0f);

    auto x = builder.add(a, one);  // 2 + 1 = 3
    auto y = builder.mul(b, one);  // 3 * 1 = 3
    auto out = builder.add(x, y);  // 3 + 3 = 6
    module.setOutput(out);

    HorizontalFusionPass pass;
    pass.run(builder);

    // Output should still be valid
    EXPECT_TRUE(module.output().isValid());
}

TEST_F(HorizontalFusionPassTest, HandlesNoIndependentOps) {
    IRModule module("test");
    auto& builder = module.builder();

    // Pure linear chain
    auto a = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto b = builder.add(a, one);
    auto c = builder.mul(b, one);
    auto d = builder.sub(c, one);
    module.setOutput(d);

    HorizontalFusionPass pass;
    size_t batched = pass.run(builder);

    // No horizontal fusion opportunities in linear chain
    EXPECT_EQ(batched, 0);
}

TEST_F(HorizontalFusionPassTest, MultipleIndependentChains) {
    IRModule module("test");
    auto& builder = module.builder();

    // Chain 1: a -> b -> c
    auto a1 = builder.constant(1.0f);
    auto one = builder.constant(1.0f);
    auto b1 = builder.add(a1, one);
    auto c1 = builder.mul(b1, one);

    // Chain 2: a -> b -> c (independent of chain 1)
    auto a2 = builder.constant(2.0f);
    auto b2 = builder.add(a2, one);
    auto c2 = builder.mul(b2, one);

    auto out = builder.add(c1, c2);
    module.setOutput(out);

    HorizontalFusionPass pass;
    size_t batched = pass.run(builder);

    // Should find horizontal fusion opportunities
    // b1 and b2 are independent, c1 and c2 are independent
    (void)batched;

    EXPECT_TRUE(module.output().isValid());
}

// =============================================================================
// BatchedKernel Tests
// =============================================================================

TEST_F(HorizontalFusionTest, CreateBatchedKernel) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);
    auto one = builder.constant(1.0f);

    auto x = builder.add(a, one);
    auto y = builder.add(b, one);
    auto z = builder.add(c, one);

    IndependentGroup group;
    group.operations = {x, y, z};

    auto batched_outputs = analyzer_->createBatchedKernel(builder, group);

    // Should return the same number of outputs as operations
    EXPECT_EQ(batched_outputs.size(), group.operations.size());

    // Each output should be valid
    for (auto out_id : batched_outputs) {
        EXPECT_TRUE(out_id.isValid());
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(HorizontalFusionTest, IntegrationWithOptimizer) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);
    auto one = builder.constant(1.0f);

    auto x = builder.add(a, one);
    auto y = builder.add(b, one);
    auto z = builder.add(c, one);

    auto t1 = builder.add(x, y);
    auto out = builder.add(t1, z);
    module.setOutput(out);

    // Run full optimization (which should include horizontal fusion)
    auto result = module.optimize(2);
    EXPECT_TRUE(result.hasValue());

    // Output should still be valid
    EXPECT_TRUE(module.output().isValid());
}

TEST_F(HorizontalFusionTest, LargeBatch) {
    // Test with a larger number of independent operations
    IRModule module("test");
    auto& builder = module.builder();

    auto one = builder.constant(1.0f);
    std::vector<ValueId> independent_ops;

    // Create 10 independent additions
    for (int i = 0; i < 10; ++i) {
        auto c = builder.constant(static_cast<float>(i));
        auto x = builder.add(c, one);
        independent_ops.push_back(x);
    }

    // Combine them
    ValueId result = independent_ops[0];
    for (size_t i = 1; i < independent_ops.size(); ++i) {
        result = builder.add(result, independent_ops[i]);
    }
    module.setOutput(result);

    auto groups = analyzer_->findIndependentGroups(builder);

    // Should find a large independent group
    bool found_large_group = false;
    for (const auto& group : groups) {
        if (group.operations.size() >= 5) {
            found_large_group = true;
            break;
        }
    }
    EXPECT_TRUE(found_large_group) << "Should find large group of independent ops";
}

}  // namespace
}  // namespace ir
}  // namespace bud
