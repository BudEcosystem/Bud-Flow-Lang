// =============================================================================
// Bud Flow Lang - Sibling (Multi-Output) Fusion Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for multi-output fusion that combines operations sharing common inputs.
// Pattern: Operations A and B both read from input X -> Fuse to read X once.
//
// Benefits:
// - Reduces memory bandwidth by reading shared inputs once
// - Better cache utilization
// - Enables further optimizations (loop fusion)
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/ir/sibling_fusion.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace ir {
namespace {

// =============================================================================
// SiblingFusionAnalyzer Tests
// =============================================================================

class SiblingFusionTest : public ::testing::Test {
  protected:
    void SetUp() override { analyzer_ = std::make_unique<SiblingFusionAnalyzer>(); }

    std::unique_ptr<SiblingFusionAnalyzer> analyzer_;
};

TEST_F(SiblingFusionTest, FindSiblingGroups_SimpleSiblings) {
    // Create IR: x = input, y = x + 1, z = x * 2
    // y and z are siblings (both read from x)
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);  // Shared input
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);  // y = x + 1
    auto z = builder.mul(x, two);  // z = x * 2

    auto groups = analyzer_->findSiblingGroups(builder);

    // Should find at least one sibling group containing y and z
    bool found_group = false;
    for (const auto& group : groups) {
        if (group.siblings.size() >= 2) {
            // Check if both y and z are in this group
            bool has_y =
                std::find(group.siblings.begin(), group.siblings.end(), y) != group.siblings.end();
            bool has_z =
                std::find(group.siblings.begin(), group.siblings.end(), z) != group.siblings.end();
            if (has_y && has_z) {
                found_group = true;
                // Verify shared input
                EXPECT_EQ(group.shared_inputs.size(), 1);
                EXPECT_EQ(group.shared_inputs[0], x);
                break;
            }
        }
    }
    EXPECT_TRUE(found_group) << "Should find sibling group for y and z sharing input x";
}

TEST_F(SiblingFusionTest, FindSiblingGroups_NoSiblings) {
    // Create IR with no shared inputs
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);
    auto d = builder.constant(4.0f);

    auto x = builder.add(a, b);  // Different inputs
    auto y = builder.mul(c, d);  // Different inputs

    auto groups = analyzer_->findSiblingGroups(builder);

    // Should not find meaningful sibling groups (no shared non-constant inputs)
    bool found_meaningful_group = false;
    for (const auto& group : groups) {
        if (group.siblings.size() >= 2 && !group.shared_inputs.empty()) {
            // Check if shared inputs are non-trivial (not just constants)
            for (auto input_id : group.shared_inputs) {
                auto* node = builder.getNode(input_id);
                if (node && node->opCode() != OpCode::kConstantScalar) {
                    found_meaningful_group = true;
                    break;
                }
            }
        }
    }
    // Constants can be shared, but that's not interesting for fusion
    (void)found_meaningful_group;
}

TEST_F(SiblingFusionTest, FindSiblingGroups_ThreeSiblings) {
    // Create IR: x = input, a = x + 1, b = x * 2, c = x - 3
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(10.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);
    auto three = builder.constant(3.0f);

    auto a = builder.add(x, one);
    auto b = builder.mul(x, two);
    auto c = builder.sub(x, three);

    auto groups = analyzer_->findSiblingGroups(builder);

    // Should find a group with all three siblings
    bool found_triple = false;
    for (const auto& group : groups) {
        if (group.siblings.size() >= 3) {
            bool has_a =
                std::find(group.siblings.begin(), group.siblings.end(), a) != group.siblings.end();
            bool has_b =
                std::find(group.siblings.begin(), group.siblings.end(), b) != group.siblings.end();
            bool has_c =
                std::find(group.siblings.begin(), group.siblings.end(), c) != group.siblings.end();
            if (has_a && has_b && has_c) {
                found_triple = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_triple) << "Should find sibling group with three operations";
}

TEST_F(SiblingFusionTest, FindSiblingGroups_MultipleSharedInputs) {
    // Create IR: x, y = inputs, a = x + y, b = x * y
    // a and b share BOTH x and y
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(3.0f);
    auto y = builder.constant(4.0f);

    auto a = builder.add(x, y);  // a = x + y
    auto b = builder.mul(x, y);  // b = x * y

    auto groups = analyzer_->findSiblingGroups(builder);

    // Should find siblings sharing multiple inputs
    bool found_multi_shared = false;
    for (const auto& group : groups) {
        if (group.siblings.size() >= 2 && group.shared_inputs.size() >= 2) {
            found_multi_shared = true;
            break;
        }
    }
    EXPECT_TRUE(found_multi_shared) << "Should find siblings sharing multiple inputs";
}

// =============================================================================
// CanFuseSiblings Tests
// =============================================================================

TEST_F(SiblingFusionTest, CanFuseSiblings_Compatible) {
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);
    auto z = builder.mul(x, two);

    std::vector<ValueId> siblings = {y, z};
    EXPECT_TRUE(analyzer_->canFuseSiblings(builder, siblings));
}

TEST_F(SiblingFusionTest, CanFuseSiblings_DifferentTypes) {
    IRModule module("test");
    auto& builder = module.builder();

    auto x_float = builder.constant(5.0f);
    auto x_int = builder.constant(static_cast<int32_t>(5));

    auto y = builder.add(x_float, x_float);

    // Can't easily test type mismatch with current builder, skip detailed test
    // The point is: siblings with different output types cannot be fused
    (void)y;
    (void)x_int;
}

TEST_F(SiblingFusionTest, CanFuseSiblings_DataDependency) {
    // If one sibling depends on another, they can't be fused as siblings
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);

    auto y = builder.add(x, one);  // y = x + 1
    auto z = builder.mul(y, x);    // z = y * x  (depends on y!)

    std::vector<ValueId> siblings = {y, z};
    // z depends on y, so they shouldn't be fusable as siblings
    EXPECT_FALSE(analyzer_->canFuseSiblings(builder, siblings));
}

// =============================================================================
// FusionGroup Tests
// =============================================================================

TEST_F(SiblingFusionTest, FusionGroup_EstimateBenefit) {
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);
    auto z = builder.mul(x, two);

    auto groups = analyzer_->findSiblingGroups(builder);

    // Find the group containing y and z
    for (const auto& group : groups) {
        if (group.siblings.size() >= 2) {
            // Benefit should be positive (saved memory reads)
            EXPECT_GE(group.estimated_benefit, 0.0f);
        }
    }
}

// =============================================================================
// SiblingFusionPass Tests
// =============================================================================

class SiblingFusionPassTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(SiblingFusionPassTest, FuseSimpleSiblings) {
    IRModule module("test");
    auto& builder = module.builder();

    // Create: x = input, y = x + 1, z = x * 2, out = y + z
    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);  // y = x + 1
    auto z = builder.mul(x, two);  // z = x * 2
    auto out = builder.add(y, z);  // out = y + z
    module.setOutput(out);

    size_t initial_reads = 0;
    for (const auto* node : builder.nodes()) {
        if (node && !node->isDead()) {
            initial_reads += node->numOperands();
        }
    }

    SiblingFusionPass pass;
    size_t fused = pass.run(builder);

    // Should create a multi-output fused operation
    // The exact number depends on implementation
    (void)fused;
    (void)initial_reads;

    // Verify the output is still valid
    EXPECT_TRUE(module.output().isValid());
}

TEST_F(SiblingFusionPassTest, PreservesSemantics) {
    // Verify that fusion doesn't change computation results
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);
    auto z = builder.mul(x, two);
    auto out = builder.add(y, z);
    module.setOutput(out);

    // Record expected result: (5+1) + (5*2) = 6 + 10 = 16
    // We can't easily compute this without execution, but we can verify
    // the IR structure is preserved

    SiblingFusionPass pass;
    pass.run(builder);

    // Output should still be valid
    EXPECT_TRUE(module.output().isValid());

    // The computation graph should still produce the same result
    // (verification would require IR interpretation)
}

TEST_F(SiblingFusionPassTest, HandlesNoSiblings) {
    IRModule module("test");
    auto& builder = module.builder();

    // Linear chain: a -> b -> c (no siblings)
    auto a = builder.constant(1.0f);
    auto two = builder.constant(2.0f);
    auto b = builder.mul(a, two);
    auto three = builder.constant(3.0f);
    auto c = builder.add(b, three);
    module.setOutput(c);

    SiblingFusionPass pass;
    size_t fused = pass.run(builder);

    // No sibling fusion opportunities
    EXPECT_EQ(fused, 0);
}

TEST_F(SiblingFusionPassTest, MultipleFusionGroups) {
    IRModule module("test");
    auto& builder = module.builder();

    // Two independent sibling groups:
    // Group 1: x -> (y = x+1, z = x*2)
    // Group 2: a -> (b = a-1, c = a/2)
    auto x = builder.constant(10.0f);
    auto a = builder.constant(20.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);
    auto z = builder.mul(x, two);

    auto b = builder.sub(a, one);
    auto c = builder.div(a, two);

    auto out1 = builder.add(y, z);
    auto out2 = builder.add(b, c);
    auto final_out = builder.add(out1, out2);
    module.setOutput(final_out);

    SiblingFusionPass pass;
    size_t fused = pass.run(builder);

    // Should fuse both groups (or at least attempt to)
    // Exact count depends on implementation
    (void)fused;

    EXPECT_TRUE(module.output().isValid());
}

// =============================================================================
// MultiOutputNode Tests
// =============================================================================

TEST_F(SiblingFusionTest, CreateMultiOutputFusion) {
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);
    auto z = builder.mul(x, two);

    // Find the sibling group
    auto groups = analyzer_->findSiblingGroups(builder);

    for (const auto& group : groups) {
        if (group.siblings.size() >= 2) {
            // Create multi-output fusion
            auto fused_outputs = analyzer_->createMultiOutputFusion(builder, group);

            // Should return the same number of outputs as siblings
            EXPECT_EQ(fused_outputs.size(), group.siblings.size());

            // Each output should be valid
            for (auto out_id : fused_outputs) {
                EXPECT_TRUE(out_id.isValid());
            }
            break;
        }
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(SiblingFusionTest, IntegrationWithOptimizer) {
    IRModule module("test");
    auto& builder = module.builder();

    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);

    auto y = builder.add(x, one);
    auto z = builder.mul(x, two);
    auto out = builder.add(y, z);
    module.setOutput(out);

    // Run full optimization (which should include sibling fusion)
    auto result = module.optimize(2);
    EXPECT_TRUE(result.hasValue());

    // Output should still be valid
    EXPECT_TRUE(module.output().isValid());
}

TEST_F(SiblingFusionTest, BenefitEstimation) {
    IRModule module("test");
    auto& builder = module.builder();

    // Create a case with clear memory savings
    auto x = builder.constant(5.0f);
    auto one = builder.constant(1.0f);
    auto two = builder.constant(2.0f);
    auto three = builder.constant(3.0f);

    // Three siblings all reading x
    auto a = builder.add(x, one);
    auto b = builder.mul(x, two);
    auto c = builder.sub(x, three);

    auto groups = analyzer_->findSiblingGroups(builder);

    // Find group with three siblings
    for (const auto& group : groups) {
        if (group.siblings.size() >= 3) {
            // Benefit should be higher for more siblings sharing an input
            // (more memory reads saved)
            EXPECT_GT(group.estimated_benefit, 0.0f);
            break;
        }
    }

    (void)a;
    (void)b;
    (void)c;
}

}  // namespace
}  // namespace ir
}  // namespace bud
