// =============================================================================
// Bud Flow Lang - Fusion Cost Model Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for the analytical cost model that predicts kernel fusion benefits.
// This model replaces heuristic-based fusion with data-driven decisions.
//
// Key metrics:
// - Memory cost: bytes / bandwidth
// - Compute cost: elements * op_latency / SIMD_width
// - Fusion benefit: unfused_time - fused_time
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/ir/fusion_cost_model.h"
#include "bud_flow_lang/memory/cache_config.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace ir {
namespace {

// =============================================================================
// CacheInfo Tests
// =============================================================================

class CacheInfoTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create default cache info from detected config
        cache_info_ = CacheInfo::fromCacheConfig(memory::CacheConfig::detect());
    }

    CacheInfo cache_info_;
};

TEST_F(CacheInfoTest, DefaultConstruction) {
    CacheInfo info;
    EXPECT_GT(info.l1Size(), 0);
    EXPECT_GT(info.l2Size(), 0);
    EXPECT_GT(info.lineSize(), 0);
    EXPECT_GT(info.memoryBandwidth(), 0.0f);
}

TEST_F(CacheInfoTest, FromCacheConfig) {
    memory::CacheConfig config(32 * 1024, 256 * 1024, 8 * 1024 * 1024, 64);
    CacheInfo info = CacheInfo::fromCacheConfig(config);

    EXPECT_EQ(info.l1Size(), 32 * 1024);
    EXPECT_EQ(info.l2Size(), 256 * 1024);
    EXPECT_EQ(info.l3Size(), 8 * 1024 * 1024);
    EXPECT_EQ(info.lineSize(), 64);
}

TEST_F(CacheInfoTest, EstimatedBandwidth) {
    // Bandwidth should be reasonable (10-100 GB/s for modern CPUs)
    float bandwidth = cache_info_.memoryBandwidth();
    EXPECT_GE(bandwidth, 10.0f * 1e9);   // At least 10 GB/s
    EXPECT_LE(bandwidth, 500.0f * 1e9);  // At most 500 GB/s (high-end)
}

// =============================================================================
// FusionCostModel - Memory Cost Tests
// =============================================================================

class FusionCostModelTest : public ::testing::Test {
  protected:
    void SetUp() override {
        model_ = std::make_unique<FusionCostModel>();
        cache_info_ = CacheInfo::fromCacheConfig(memory::CacheConfig::detect());
    }

    std::unique_ptr<FusionCostModel> model_;
    CacheInfo cache_info_;
};

TEST_F(FusionCostModelTest, MemoryReadCost_Cached) {
    // Reading cached data should be faster than uncached
    size_t bytes = 4096;  // 4KB - should fit in L1

    float cached_cost = model_->memoryReadCost(bytes, true);
    float uncached_cost = model_->memoryReadCost(bytes, false);

    EXPECT_GT(cached_cost, 0.0f);
    EXPECT_GT(uncached_cost, 0.0f);
    EXPECT_LT(cached_cost, uncached_cost);  // Cached should be faster
}

TEST_F(FusionCostModelTest, MemoryReadCost_ScalesWithSize) {
    // Larger reads should cost more
    float cost_1k = model_->memoryReadCost(1024, false);
    float cost_4k = model_->memoryReadCost(4096, false);
    float cost_64k = model_->memoryReadCost(65536, false);

    EXPECT_LT(cost_1k, cost_4k);
    EXPECT_LT(cost_4k, cost_64k);

    // Should be monotonically increasing (not strictly linear due to latency)
    // The ratio should be > 1 and generally increase with size
    EXPECT_GT(cost_4k / cost_1k, 1.0f);
    EXPECT_GT(cost_64k / cost_4k, 1.0f);

    // For large sizes, scaling should approach linear
    float cost_1m = model_->memoryReadCost(1024 * 1024, false);
    float cost_4m = model_->memoryReadCost(4 * 1024 * 1024, false);
    EXPECT_NEAR(cost_4m / cost_1m, 4.0f, 1.0f);  // Large sizes scale more linearly
}

TEST_F(FusionCostModelTest, MemoryWriteCost) {
    size_t bytes = 4096;

    float write_cost = model_->memoryWriteCost(bytes);

    EXPECT_GT(write_cost, 0.0f);
    // Writes are typically similar or slightly more expensive than reads
    float read_cost = model_->memoryReadCost(bytes, false);
    EXPECT_GE(write_cost, read_cost * 0.8f);
}

// =============================================================================
// FusionCostModel - Compute Cost Tests
// =============================================================================

TEST_F(FusionCostModelTest, ComputeCost_Add) {
    // Add is a simple operation
    float cost = model_->computeCost(OpCode::kAdd, 1024);
    EXPECT_GT(cost, 0.0f);
}

TEST_F(FusionCostModelTest, ComputeCost_Mul) {
    // Mul is similar cost to Add on modern CPUs
    float add_cost = model_->computeCost(OpCode::kAdd, 1024);
    float mul_cost = model_->computeCost(OpCode::kMul, 1024);

    EXPECT_GT(mul_cost, 0.0f);
    // Should be within 2x of add
    EXPECT_LT(std::abs(mul_cost - add_cost) / add_cost, 1.0f);
}

TEST_F(FusionCostModelTest, ComputeCost_FMA) {
    // FMA (a*b+c) should be similar to Mul or Add (fused single instruction)
    float add_cost = model_->computeCost(OpCode::kAdd, 1024);
    float fma_cost = model_->computeCost(OpCode::kFma, 1024);

    EXPECT_GT(fma_cost, 0.0f);
    // FMA replaces Mul+Add, so should be less than 2x Add
    EXPECT_LT(fma_cost, add_cost * 2.0f);
}

TEST_F(FusionCostModelTest, ComputeCost_Exp) {
    // Transcendentals are more expensive
    float add_cost = model_->computeCost(OpCode::kAdd, 1024);
    float exp_cost = model_->computeCost(OpCode::kExp, 1024);

    EXPECT_GT(exp_cost, 0.0f);
    EXPECT_GT(exp_cost, add_cost);  // Exp is more expensive than Add
}

TEST_F(FusionCostModelTest, ComputeCost_ScalesWithElements) {
    float cost_1k = model_->computeCost(OpCode::kMul, 1024);
    float cost_4k = model_->computeCost(OpCode::kMul, 4096);

    EXPECT_NEAR(cost_4k / cost_1k, 4.0f, 0.5f);  // Should scale linearly
}

// =============================================================================
// FusionCostModel - Predict Time Tests
// =============================================================================

TEST_F(FusionCostModelTest, PredictTime_SimpleAdd) {
    // Create a simple Add node
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto add = builder.add(a, b);

    const IRNode* node = builder.getNode(add);
    ASSERT_NE(node, nullptr);

    float time = model_->predictTime(node, cache_info_);
    EXPECT_GT(time, 0.0f);
}

TEST_F(FusionCostModelTest, PredictTime_MulAdd) {
    // Create Mul followed by Add (should be fusable to FMA)
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add = builder.add(mul, c);

    const IRNode* mul_node = builder.getNode(mul);
    const IRNode* add_node = builder.getNode(add);

    float mul_time = model_->predictTime(mul_node, cache_info_);
    float add_time = model_->predictTime(add_node, cache_info_);

    EXPECT_GT(mul_time, 0.0f);
    EXPECT_GT(add_time, 0.0f);
}

// =============================================================================
// FusionCostModel - Fusion Benefit Tests
// =============================================================================

TEST_F(FusionCostModelTest, CalculateBenefit_MulAdd) {
    // Mul+Add -> FMA should have positive benefit
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add = builder.add(mul, c);

    IRNode* mul_node = builder.getNode(mul);
    IRNode* add_node = builder.getNode(add);

    float benefit = model_->calculateBenefit(mul_node, add_node, cache_info_);

    // Fusion should provide benefit (positive value)
    EXPECT_GT(benefit, 0.0f);
}

TEST_F(FusionCostModelTest, CalculateBenefit_IndependentOps) {
    // Two independent operations with no data dependency shouldn't fuse
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto add1 = builder.add(a, b);

    auto c = builder.constant(3.0f);
    auto d = builder.constant(4.0f);
    auto add2 = builder.add(c, d);

    IRNode* add1_node = builder.getNode(add1);
    IRNode* add2_node = builder.getNode(add2);

    // No producer-consumer relationship, benefit should be 0 or negative
    float benefit = model_->calculateBenefit(add1_node, add2_node, cache_info_);
    EXPECT_LE(benefit, 0.0f);
}

// =============================================================================
// FusionCostModel - Predict Fused Time Tests
// =============================================================================

TEST_F(FusionCostModelTest, PredictFusedTime_MulAdd) {
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add = builder.add(mul, c);

    std::vector<IRNode*> nodes = {builder.getNode(mul), builder.getNode(add)};

    float fused_time = model_->predictFusedTime(nodes, cache_info_);
    EXPECT_GT(fused_time, 0.0f);

    // Fused time should be less than sum of individual times
    float mul_time = model_->predictTime(nodes[0], cache_info_);
    float add_time = model_->predictTime(nodes[1], cache_info_);
    float unfused_time = mul_time + add_time;

    EXPECT_LT(fused_time, unfused_time);
}

TEST_F(FusionCostModelTest, PredictFusedTime_LongerChain) {
    // a*b+c*d+e should benefit from fusion
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul1 = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto d = builder.constant(4.0f);
    auto mul2 = builder.mul(c, d);
    auto add = builder.add(mul1, mul2);

    std::vector<IRNode*> nodes = {builder.getNode(mul1), builder.getNode(mul2),
                                  builder.getNode(add)};

    float fused_time = model_->predictFusedTime(nodes, cache_info_);

    // Calculate unfused time
    float unfused_time = 0.0f;
    for (auto* node : nodes) {
        unfused_time += model_->predictTime(node, cache_info_);
    }

    // Fused should be faster
    EXPECT_LT(fused_time, unfused_time);
}

// =============================================================================
// FusionCostModel - Latency Tables Tests
// =============================================================================

TEST_F(FusionCostModelTest, GetOpLatency_BasicOps) {
    // Basic ops should have reasonable latency values
    float add_latency = model_->getOpLatency(OpCode::kAdd);
    float mul_latency = model_->getOpLatency(OpCode::kMul);
    float div_latency = model_->getOpLatency(OpCode::kDiv);

    EXPECT_GT(add_latency, 0.0f);
    EXPECT_GT(mul_latency, 0.0f);
    EXPECT_GT(div_latency, 0.0f);

    // Div is typically more expensive than Mul
    EXPECT_GT(div_latency, mul_latency);
}

TEST_F(FusionCostModelTest, GetOpLatency_Transcendentals) {
    float sin_latency = model_->getOpLatency(OpCode::kSin);
    float exp_latency = model_->getOpLatency(OpCode::kExp);
    float add_latency = model_->getOpLatency(OpCode::kAdd);

    // Transcendentals are much more expensive
    EXPECT_GT(sin_latency, add_latency * 2.0f);
    EXPECT_GT(exp_latency, add_latency * 2.0f);
}

TEST_F(FusionCostModelTest, GetOpThroughput_BasicOps) {
    // Get throughput (ops per cycle) for basic operations
    float add_throughput = model_->getOpThroughput(OpCode::kAdd);
    float mul_throughput = model_->getOpThroughput(OpCode::kMul);

    EXPECT_GT(add_throughput, 0.0f);
    EXPECT_GT(mul_throughput, 0.0f);
}

// =============================================================================
// PriorityFusionPass Tests
// =============================================================================

class PriorityFusionPassTest : public ::testing::Test {
  protected:
    void SetUp() override { cost_model_ = std::make_unique<FusionCostModel>(); }

    std::unique_ptr<FusionCostModel> cost_model_;
};

TEST_F(PriorityFusionPassTest, FuseMulAdd) {
    // Mul+Add should be fused to FMA
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(2.0f);
    auto b = builder.constant(3.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(1.0f);
    auto add = builder.add(mul, c);
    module.setOutput(add);

    // Run priority fusion pass
    PriorityFusionPass pass(*cost_model_);
    size_t fused = pass.run(builder);

    EXPECT_GE(fused, 1);  // At least one fusion

    // Check that FMA was created
    bool found_fma = false;
    for (const auto* node : builder.nodes()) {
        if (node && !node->isDead() && node->opCode() == OpCode::kFma) {
            found_fma = true;
            break;
        }
    }
    EXPECT_TRUE(found_fma);
}

TEST_F(PriorityFusionPassTest, PrioritizesByBenefit) {
    // Create multiple fusion opportunities, verify highest benefit is chosen first
    IRModule module("test");
    auto& builder = module.builder();

    // Chain 1: simple mul+add
    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul1 = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add1 = builder.add(mul1, c);

    // Chain 2: another mul+add
    auto d = builder.constant(4.0f);
    auto e = builder.constant(5.0f);
    auto mul2 = builder.mul(d, e);
    auto f = builder.constant(6.0f);
    auto add2 = builder.add(mul2, f);

    auto final_add = builder.add(add1, add2);
    module.setOutput(final_add);

    PriorityFusionPass pass(*cost_model_);
    size_t fused = pass.run(builder);

    // Both chains should be fused
    EXPECT_GE(fused, 2);
}

TEST_F(PriorityFusionPassTest, RespectsDataDependencies) {
    // Verify fusion respects SSA data dependencies
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add = builder.add(mul, c);
    module.setOutput(add);

    size_t initial_count = builder.liveNodeCount();

    PriorityFusionPass pass(*cost_model_);
    pass.run(builder);
    builder.compactNodes();

    // After fusion, we should have fewer nodes (eliminated mul+add, added fma)
    size_t final_count = builder.liveNodeCount();
    EXPECT_LE(final_count, initial_count);
}

TEST_F(PriorityFusionPassTest, HandlesMultipleUses) {
    // If mul result is used by multiple nodes, don't fuse
    IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(2.0f);
    auto b = builder.constant(3.0f);
    auto mul = builder.mul(a, b);  // Used by both add and sub

    auto c = builder.constant(1.0f);
    auto add = builder.add(mul, c);  // mul+c
    auto sub = builder.sub(mul, c);  // mul-c

    auto final_add = builder.add(add, sub);
    module.setOutput(final_add);

    size_t mul_uses = builder.useCount(mul);
    EXPECT_EQ(mul_uses, 2);  // mul is used twice

    // Should NOT fuse mul into add since mul has multiple uses
    PriorityFusionPass pass(*cost_model_);
    size_t fused = pass.run(builder);

    // No fusion should happen because mul has multiple uses
    EXPECT_EQ(fused, 0);
}

// =============================================================================
// FusionOpportunity Tests
// =============================================================================

TEST_F(FusionCostModelTest, FindFusionOpportunities) {
    IRModule module("test");
    auto& builder = module.builder();

    // Create fusable patterns
    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add = builder.add(mul, c);
    module.setOutput(add);

    std::vector<FusionOpportunity> opportunities =
        model_->findFusionOpportunities(builder, cache_info_);

    EXPECT_GE(opportunities.size(), 1);

    // Verify the opportunity has positive benefit
    if (!opportunities.empty()) {
        EXPECT_GT(opportunities[0].benefit, 0.0f);
        EXPECT_EQ(opportunities[0].producer.id, mul.id);
        EXPECT_EQ(opportunities[0].consumer.id, add.id);
    }
}

TEST_F(FusionCostModelTest, SortOpportunitiesByBenefit) {
    IRModule module("test");
    auto& builder = module.builder();

    // Create multiple fusion opportunities
    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto mul1 = builder.mul(a, b);
    auto c = builder.constant(3.0f);
    auto add1 = builder.add(mul1, c);

    auto d = builder.constant(4.0f);
    auto sub1 = builder.sub(add1, d);
    auto e = builder.constant(5.0f);
    auto mul2 = builder.mul(sub1, e);
    module.setOutput(mul2);

    std::vector<FusionOpportunity> opportunities =
        model_->findFusionOpportunities(builder, cache_info_);

    // Opportunities should be sorted by benefit (highest first)
    for (size_t i = 1; i < opportunities.size(); ++i) {
        EXPECT_GE(opportunities[i - 1].benefit, opportunities[i].benefit);
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(FusionCostModelTest, IntegrationWithOptimizer) {
    // Test that cost model integrates with the optimizer
    IRModule module("test");
    auto& builder = module.builder();

    // Build a computation graph
    auto a = builder.constant(2.0f);
    auto b = builder.constant(3.0f);
    auto mul = builder.mul(a, b);
    auto c = builder.constant(1.0f);
    auto add = builder.add(mul, c);
    module.setOutput(add);

    // Run optimization with level 2 (includes fusion)
    auto result = module.optimize(2);
    EXPECT_TRUE(result.hasValue());

    // Verify FMA was created by checking IR
    bool has_fma = false;
    for (const auto* node : builder.nodes()) {
        if (node && !node->isDead() && node->opCode() == OpCode::kFma) {
            has_fma = true;
            break;
        }
    }
    EXPECT_TRUE(has_fma);
}

}  // namespace
}  // namespace ir
}  // namespace bud
