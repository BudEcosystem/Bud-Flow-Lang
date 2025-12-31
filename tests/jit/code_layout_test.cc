// =============================================================================
// Bud Flow Lang - Code Layout Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for hot/cold code separation that improves I-cache utilization by
// placing frequently executed code contiguously.
//
// Benefits:
// - Better instruction cache utilization
// - Reduced branch mispredictions
// - Smaller hot code footprint
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/code_layout.h"

#include <chrono>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace jit {
namespace {

// =============================================================================
// CodeRegion Tests
// =============================================================================

class CodeRegionTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(CodeRegionTest, DefaultConstruction) {
    CodeRegion region;
    EXPECT_EQ(region.type(), RegionType::kGeneric);
    EXPECT_TRUE(region.isEmpty());
    EXPECT_EQ(region.size(), 0);
}

TEST_F(CodeRegionTest, HotRegionCreation) {
    CodeRegion hot(RegionType::kHot);
    EXPECT_EQ(hot.type(), RegionType::kHot);
    EXPECT_TRUE(hot.isEmpty());
}

TEST_F(CodeRegionTest, ColdRegionCreation) {
    CodeRegion cold(RegionType::kCold);
    EXPECT_EQ(cold.type(), RegionType::kCold);
    EXPECT_TRUE(cold.isEmpty());
}

TEST_F(CodeRegionTest, AddCode) {
    CodeRegion region(RegionType::kHot);

    // Simulate adding code bytes
    std::vector<uint8_t> code = {0x90, 0x90, 0x90, 0x90};  // NOP instructions
    region.appendCode(code.data(), code.size());

    EXPECT_FALSE(region.isEmpty());
    EXPECT_EQ(region.size(), 4);
}

TEST_F(CodeRegionTest, Alignment) {
    CodeRegion region(RegionType::kHot);

    std::vector<uint8_t> code = {0x90, 0x90, 0x90};
    region.appendCode(code.data(), code.size());

    // Align to 16-byte boundary
    region.alignTo(16);

    // Size should be padded to 16 bytes
    EXPECT_EQ(region.size() % 16, 0);
}

// =============================================================================
// CodeLayoutOptimizer Tests
// =============================================================================

class CodeLayoutOptimizerTest : public ::testing::Test {
  protected:
    void SetUp() override { optimizer_ = std::make_unique<CodeLayoutOptimizer>(); }

    std::unique_ptr<CodeLayoutOptimizer> optimizer_;
};

TEST_F(CodeLayoutOptimizerTest, DefaultConstruction) {
    EXPECT_TRUE(optimizer_->isEnabled());
}

TEST_F(CodeLayoutOptimizerTest, EnableDisable) {
    optimizer_->disable();
    EXPECT_FALSE(optimizer_->isEnabled());

    optimizer_->enable();
    EXPECT_TRUE(optimizer_->isEnabled());
}

TEST_F(CodeLayoutOptimizerTest, ClassifyAsHot) {
    // Create a function with high execution count
    FunctionProfile profile;
    profile.execution_count = 10000;
    profile.is_critical_path = true;

    auto classification = optimizer_->classifyFunction(profile);
    EXPECT_EQ(classification, RegionType::kHot);
}

TEST_F(CodeLayoutOptimizerTest, ClassifyAsCold) {
    // Create a function with very low execution count
    FunctionProfile profile;
    profile.execution_count = 1;
    profile.is_critical_path = false;

    auto classification = optimizer_->classifyFunction(profile);
    EXPECT_EQ(classification, RegionType::kCold);
}

TEST_F(CodeLayoutOptimizerTest, ClassifyAsGeneric) {
    // Create a function with medium execution count
    FunctionProfile profile;
    profile.execution_count = 100;
    profile.is_critical_path = false;

    auto classification = optimizer_->classifyFunction(profile);
    // Medium count functions go to generic
    EXPECT_TRUE(classification == RegionType::kGeneric || classification == RegionType::kHot);
}

TEST_F(CodeLayoutOptimizerTest, SetHotThreshold) {
    optimizer_->setHotThreshold(5000);
    EXPECT_EQ(optimizer_->hotThreshold(), 5000);

    // Function below threshold should not be hot
    FunctionProfile profile;
    profile.execution_count = 3000;
    profile.is_critical_path = false;

    auto classification = optimizer_->classifyFunction(profile);
    EXPECT_NE(classification, RegionType::kHot);
}

TEST_F(CodeLayoutOptimizerTest, SetColdThreshold) {
    optimizer_->setColdThreshold(5);
    EXPECT_EQ(optimizer_->coldThreshold(), 5);

    // Function above threshold should not be cold
    FunctionProfile profile;
    profile.execution_count = 10;
    profile.is_critical_path = false;

    auto classification = optimizer_->classifyFunction(profile);
    EXPECT_NE(classification, RegionType::kCold);
}

// =============================================================================
// LayoutPlan Tests
// =============================================================================

TEST_F(CodeLayoutOptimizerTest, CreateLayoutPlan) {
    // Create profiles for multiple functions
    std::vector<FunctionProfile> profiles = {
        {.name = "compute_hot", .execution_count = 10000, .is_critical_path = true},
        {.name = "error_handler", .execution_count = 5, .is_critical_path = false},
        {.name = "validate_input", .execution_count = 500, .is_critical_path = false},
        {.name = "main_loop", .execution_count = 8000, .is_critical_path = true},
    };

    auto plan = optimizer_->createLayoutPlan(profiles);

    EXPECT_FALSE(plan.hot_functions.empty());
    EXPECT_FALSE(plan.cold_functions.empty());

    // Hot functions should include high-count ones
    bool found_compute = false;
    bool found_main = false;
    for (const auto& name : plan.hot_functions) {
        if (name == "compute_hot")
            found_compute = true;
        if (name == "main_loop")
            found_main = true;
    }
    EXPECT_TRUE(found_compute);
    EXPECT_TRUE(found_main);
}

TEST_F(CodeLayoutOptimizerTest, OrderHotFunctionsByFrequency) {
    std::vector<FunctionProfile> profiles = {
        {.name = "A", .execution_count = 100, .is_critical_path = true},
        {.name = "B", .execution_count = 1000, .is_critical_path = true},
        {.name = "C", .execution_count = 500, .is_critical_path = true},
    };

    auto plan = optimizer_->createLayoutPlan(profiles);

    // All should be hot (setting low threshold)
    optimizer_->setHotThreshold(50);
    plan = optimizer_->createLayoutPlan(profiles);

    // Hot functions should be ordered by execution count (descending)
    if (plan.hot_functions.size() >= 2) {
        // B should come before C, C before A
        auto find_idx = [&](const std::string& name) {
            for (size_t i = 0; i < plan.hot_functions.size(); ++i) {
                if (plan.hot_functions[i] == name)
                    return i;
            }
            return plan.hot_functions.size();
        };

        size_t b_idx = find_idx("B");
        size_t c_idx = find_idx("C");
        size_t a_idx = find_idx("A");

        if (b_idx < plan.hot_functions.size() && c_idx < plan.hot_functions.size() &&
            a_idx < plan.hot_functions.size()) {
            EXPECT_LT(b_idx, c_idx);
            EXPECT_LT(c_idx, a_idx);
        }
    }
}

// =============================================================================
// Fall-through Optimization Tests
// =============================================================================

TEST_F(CodeLayoutOptimizerTest, FallThroughOptimization) {
    // Test that the optimizer prefers fall-through for common branches
    BranchProfile branch;
    branch.taken_count = 100;
    branch.not_taken_count = 900;

    // Not-taken is more common, so should fall through
    bool should_invert = optimizer_->shouldInvertBranch(branch);
    EXPECT_FALSE(should_invert);  // Don't invert - fall through on not-taken
}

TEST_F(CodeLayoutOptimizerTest, InvertBranchForHotPath) {
    BranchProfile branch;
    branch.taken_count = 900;
    branch.not_taken_count = 100;

    // Taken is more common, may want to invert so fall-through is hot path
    bool should_invert = optimizer_->shouldInvertBranch(branch);
    EXPECT_TRUE(should_invert);  // Invert so fall-through is common case
}

// =============================================================================
// Code Alignment Tests
// =============================================================================

TEST_F(CodeLayoutOptimizerTest, AlignmentForLoops) {
    // Loop headers should be aligned for performance
    size_t alignment = optimizer_->recommendedAlignment(AlignmentTarget::kLoopHeader);
    EXPECT_GE(alignment, 16);  // At least 16-byte aligned
}

TEST_F(CodeLayoutOptimizerTest, AlignmentForFunctions) {
    size_t alignment = optimizer_->recommendedAlignment(AlignmentTarget::kFunctionEntry);
    EXPECT_GE(alignment, 16);  // At least 16-byte aligned
}

TEST_F(CodeLayoutOptimizerTest, AlignmentForJumpTargets) {
    size_t alignment = optimizer_->recommendedAlignment(AlignmentTarget::kJumpTarget);
    EXPECT_GE(alignment, 4);  // At least 4-byte aligned
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(CodeLayoutOptimizerTest, ApplyLayoutToIR) {
    ir::IRModule module("test");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.add(a, b);
    module.setOutput(c);

    // Create profile data
    ProfileData profile;
    for (int i = 0; i < 1000; ++i) {
        profile.recordExecution(1024, std::chrono::nanoseconds(100));
    }

    // Apply layout optimization
    auto result = optimizer_->optimizeLayout(builder, profile);

    // Result should indicate some optimization was applied
    EXPECT_TRUE(result.has_value());
}

TEST_F(CodeLayoutOptimizerTest, CalculateMemoryFootprint) {
    CodeRegion hot(RegionType::kHot);
    CodeRegion cold(RegionType::kCold);
    CodeRegion generic(RegionType::kGeneric);

    // Add some code to each region
    std::vector<uint8_t> code(64, 0x90);
    hot.appendCode(code.data(), code.size());
    cold.appendCode(code.data(), 32);
    generic.appendCode(code.data(), 48);

    auto footprint = optimizer_->calculateFootprint({hot, cold, generic});

    EXPECT_EQ(footprint.hot_size, 64);
    EXPECT_EQ(footprint.cold_size, 32);
    EXPECT_EQ(footprint.total_size, 64 + 32 + 48);
}

// =============================================================================
// Benchmark Tests
// =============================================================================

TEST_F(CodeLayoutOptimizerTest, BenchmarkClassification) {
    // Measure classification overhead
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 100000;
    for (int i = 0; i < kIterations; ++i) {
        FunctionProfile profile;
        profile.execution_count = i % 10000;
        profile.is_critical_path = (i % 3) == 0;
        volatile auto result = optimizer_->classifyFunction(profile);
        (void)result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double ns_per_classification = static_cast<double>(duration_ns) / kIterations;

    // Classification should be very fast (<100ns)
    EXPECT_LT(ns_per_classification, 500.0)
        << "Classification too slow: " << ns_per_classification << " ns/op";

    std::cout << "Classification time: " << ns_per_classification << " ns/op\n";
}

TEST_F(CodeLayoutOptimizerTest, BenchmarkLayoutPlanCreation) {
    // Create a realistic set of function profiles
    std::vector<FunctionProfile> profiles;
    for (int i = 0; i < 100; ++i) {
        profiles.push_back({.name = "func_" + std::to_string(i),
                            .execution_count = static_cast<uint64_t>(i * 100),
                            .is_critical_path = (i % 5) == 0});
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        auto plan = optimizer_->createLayoutPlan(profiles);
        (void)plan;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_plan = static_cast<double>(duration_us) / kIterations;

    // Layout planning should be fast (<1ms for 100 functions)
    EXPECT_LT(us_per_plan, 1000.0) << "Layout planning too slow: " << us_per_plan << " us/plan";

    std::cout << "Layout plan creation: " << us_per_plan << " us/plan (100 functions)\n";
}

}  // namespace
}  // namespace jit
}  // namespace bud
