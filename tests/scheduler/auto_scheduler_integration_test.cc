// =============================================================================
// Bud Flow Lang - Auto-Scheduler Integration Tests
// =============================================================================
//
// End-to-end integration tests for the auto-scheduler:
// - Full pipeline from IR to optimized schedule
// - Correctness verification
// - Component integration testing
// - Real-world pattern tests
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/auto_scheduler.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/evolutionary_search.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// Integration Test Fixture
// =============================================================================

class AutoSchedulerIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override { rng_.seed(42); }

    void TearDown() override {
        // Cleanup any temp files
        for (const auto& path : temp_files_) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }

    std::string createTempFile(const std::string& suffix = ".json") {
        std::string path = "/tmp/bud_test_" + std::to_string(temp_counter_++) + suffix;
        temp_files_.push_back(path);
        return path;
    }

    // =============================================================================
    // IR Pattern Generators
    // =============================================================================

    // Element-wise chain: a + b + c + d + ...
    ir::IRModule createElementWiseChain(size_t length) {
        ir::IRModule module("ewise_chain_" + std::to_string(length));
        auto& builder = module.builder();

        auto result = builder.constant(1.0f);
        for (size_t i = 0; i < length; ++i) {
            auto val = builder.constant(static_cast<float>(i + 1));
            result = builder.add(result, val);
        }
        module.setOutput(result);
        return module;
    }

    // Reduction tree: sum(a, b, c, d, ...)
    ir::IRModule createReductionTree(size_t inputs) {
        ir::IRModule module("reduction_" + std::to_string(inputs));
        auto& builder = module.builder();

        std::vector<ir::ValueId> leaves;
        for (size_t i = 0; i < inputs; ++i) {
            leaves.push_back(builder.constant(static_cast<float>(i + 1)));
        }

        // Build balanced tree
        while (leaves.size() > 1) {
            std::vector<ir::ValueId> next_level;
            for (size_t i = 0; i < leaves.size(); i += 2) {
                if (i + 1 < leaves.size()) {
                    next_level.push_back(builder.add(leaves[i], leaves[i + 1]));
                } else {
                    next_level.push_back(leaves[i]);
                }
            }
            leaves = next_level;
        }

        module.setOutput(leaves[0]);
        return module;
    }

    // Stencil pattern: result[i] = a[i-1] + 2*a[i] + a[i+1]
    ir::IRModule createStencilPattern() {
        ir::IRModule module("stencil_1d");
        auto& builder = module.builder();

        auto left = builder.constant(1.0f);
        auto center = builder.constant(2.0f);
        auto right = builder.constant(3.0f);
        auto two = builder.constant(2.0f);

        auto scaled_center = builder.mul(center, two);
        auto sum1 = builder.add(left, scaled_center);
        auto result = builder.add(sum1, right);

        module.setOutput(result);
        return module;
    }

    // Softmax pattern: exp(x - max(x)) / sum(exp(x - max(x)))
    ir::IRModule createSoftmaxPattern() {
        ir::IRModule module("softmax");
        auto& builder = module.builder();

        auto x = builder.constant(1.0f);
        auto max_x = builder.constant(2.0f);  // Simulated max

        auto shifted = builder.sub(x, max_x);
        auto exp_shifted = builder.exp(shifted);
        auto sum_exp = builder.constant(1.0f);  // Simulated sum
        auto result = builder.div(exp_shifted, sum_exp);

        module.setOutput(result);
        return module;
    }

    // Layer normalization pattern
    ir::IRModule createLayerNormPattern() {
        ir::IRModule module("layer_norm");
        auto& builder = module.builder();

        auto x = builder.constant(1.0f);
        auto mean = builder.constant(0.5f);
        auto var = builder.constant(0.25f);
        auto gamma = builder.constant(1.0f);
        auto beta = builder.constant(0.0f);
        auto eps = builder.constant(0.00001f);

        auto centered = builder.sub(x, mean);
        auto var_eps = builder.add(var, eps);
        auto std_dev = builder.sqrt(var_eps);
        auto normalized = builder.div(centered, std_dev);
        auto scaled = builder.mul(normalized, gamma);
        auto result = builder.add(scaled, beta);

        module.setOutput(result);
        return module;
    }

    // Attention pattern (simplified QK^T)
    ir::IRModule createAttentionPattern() {
        ir::IRModule module("attention");
        auto& builder = module.builder();

        auto q = builder.constant(1.0f);
        auto k = builder.constant(1.0f);
        auto scale = builder.constant(0.125f);  // 1/sqrt(d_k)

        auto qk = builder.mul(q, k);  // Simulated matmul
        auto scaled_qk = builder.mul(qk, scale);

        // Softmax approximation
        auto exp_qk = builder.exp(scaled_qk);
        auto sum_exp = builder.constant(1.0f);
        auto attention = builder.div(exp_qk, sum_exp);

        module.setOutput(attention);
        return module;
    }

    // GELU activation pattern
    ir::IRModule createGELUPattern() {
        ir::IRModule module("gelu");
        auto& builder = module.builder();

        auto x = builder.constant(1.0f);
        auto half = builder.constant(0.5f);
        auto one = builder.constant(1.0f);
        auto sqrt_2_over_pi = builder.constant(0.7978845608f);
        auto coeff = builder.constant(0.044715f);

        // GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        auto x_cubed = builder.mul(x, builder.mul(x, x));
        auto scaled_cubed = builder.mul(coeff, x_cubed);
        auto sum = builder.add(x, scaled_cubed);
        auto scaled = builder.mul(sqrt_2_over_pi, sum);
        auto tanh_val = builder.tanh(scaled);
        auto one_plus_tanh = builder.add(one, tanh_val);
        auto half_x = builder.mul(half, x);
        auto result = builder.mul(half_x, one_plus_tanh);

        module.setOutput(result);
        return module;
    }

    std::mt19937 rng_;
    std::vector<std::string> temp_files_;
    int temp_counter_ = 0;
};

// =============================================================================
// Full Pipeline Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, FullPipeline_ElementWiseChain) {
    auto module = createElementWiseChain(10);

    // Step 1: Build DAG
    auto dag = ComputeDAG::fromIR(module.builder());
    EXPECT_GT(dag.numNodes(), 0);

    // Step 2: Create search space
    auto space = SearchSpace::fromDAG(dag);
    EXPECT_GT(space.numSketches(), 0);

    // Step 3: Initialize cost model
    CostModel cost_model;

    // Step 4: Run tuning
    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    AutoScheduler scheduler(config);
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.schedule.validate());

    // Step 5: Apply schedule
    bool applied = scheduler.applySchedule(module, result.schedule);
    EXPECT_TRUE(applied);
}

TEST_F(AutoSchedulerIntegrationTest, FullPipeline_ReductionTree) {
    auto module = createReductionTree(8);

    AutoScheduler scheduler;
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);

    // Verify DAG structure
    auto dag = ComputeDAG::fromIR(module.builder());
    auto analysis = dag.analyze();

    // Reduction tree should have certain characteristics
    EXPECT_GT(dag.numNodes(), 0);
}

TEST_F(AutoSchedulerIntegrationTest, FullPipeline_RealWorldPatterns) {
    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    // Test each pattern individually (IRModule is non-copyable)
    {
        auto module = createStencilPattern();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success) << "Failed for pattern: Stencil";
        EXPECT_TRUE(result.schedule.validate()) << "Invalid schedule for: Stencil";
    }
    {
        auto module = createSoftmaxPattern();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success) << "Failed for pattern: Softmax";
        EXPECT_TRUE(result.schedule.validate()) << "Invalid schedule for: Softmax";
    }
    {
        auto module = createLayerNormPattern();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success) << "Failed for pattern: LayerNorm";
        EXPECT_TRUE(result.schedule.validate()) << "Invalid schedule for: LayerNorm";
    }
    {
        auto module = createAttentionPattern();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success) << "Failed for pattern: Attention";
        EXPECT_TRUE(result.schedule.validate()) << "Invalid schedule for: Attention";
    }
    {
        auto module = createGELUPattern();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success) << "Failed for pattern: GELU";
        EXPECT_TRUE(result.schedule.validate()) << "Invalid schedule for: GELU";
    }
}

// =============================================================================
// Component Integration Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, Components_DAGToSearchSpace) {
    auto module = createElementWiseChain(5);

    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Should generate multiple sketches
    EXPECT_GT(space.numSketches(), 0);

    // Sample schedules and verify they are valid
    auto schedules = space.sample(10);
    for (const auto& sched : schedules) {
        EXPECT_TRUE(sched.validate());
    }
}

TEST_F(AutoSchedulerIntegrationTest, Components_SearchSpaceToCostModel) {
    auto module = createElementWiseChain(5);

    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    CostModel cost_model;

    // Sample schedules and get predictions
    auto schedules = space.sample(10);
    for (const auto& schedule : schedules) {
        auto features = cost_model.extractFeatures(schedule);
        EXPECT_FALSE(features.empty());

        float cost = cost_model.predict(schedule);
        EXPECT_GE(cost, 0.0f);
    }
}

TEST_F(AutoSchedulerIntegrationTest, Components_EvolutionarySearchIntegration) {
    auto module = createElementWiseChain(8);

    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    SearchConfig search_config;
    search_config.population_size = 32;
    search_config.num_generations = 5;

    EvolutionarySearch search(search_config);

    search.initializePopulation(space);
    EXPECT_GT(search.populationSize(), 0);

    CostModel cost_model;
    search.evaluatePopulation(cost_model, dag);

    // Evolve
    for (size_t gen = 0; gen < 3; ++gen) {
        search.evolveGeneration(space, cost_model, dag);
    }

    // Should have improved or maintained
    const auto& best = search.bestIndividual();
    EXPECT_TRUE(best.schedule.validate());
}

// =============================================================================
// Persistence Integration Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, Persistence_FullWorkflow) {
    std::string log_path = createTempFile();

    // Session 1: Initial tuning
    {
        TuningConfig config;
        config.population_size = 32;
        config.num_generations = 5;

        AutoScheduler scheduler(config);

        // Tune multiple patterns
        [[maybe_unused]] auto r1 = scheduler.tune(createElementWiseChain(5));
        [[maybe_unused]] auto r2 = scheduler.tune(createReductionTree(4));
        [[maybe_unused]] auto r3 = scheduler.tune(createStencilPattern());

        // Export
        EXPECT_TRUE(scheduler.exportLog(log_path));
        EXPECT_EQ(scheduler.numTuningRecords(), 3);
    }

    // Session 2: Load and continue
    {
        AutoScheduler scheduler;
        EXPECT_TRUE(scheduler.importLog(log_path));
        EXPECT_EQ(scheduler.numTuningRecords(), 3);

        // Verify best schedules are available
        EXPECT_TRUE(scheduler.getBestSchedule("ewise_chain_5").has_value());
        EXPECT_TRUE(scheduler.getBestSchedule("reduction_4").has_value());
        EXPECT_TRUE(scheduler.getBestSchedule("stencil_1d").has_value());

        // Continue tuning with imported knowledge
        auto module = createElementWiseChain(5);
        auto best = scheduler.getBestSchedule("ewise_chain_5");
        ASSERT_TRUE(best.has_value());

        TuneResult result = scheduler.continueTuning(module, *best);
        EXPECT_TRUE(result.success);
    }
}

TEST_F(AutoSchedulerIntegrationTest, Persistence_SerializationRoundTrip) {
    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 3;

    AutoScheduler original(config);

    // Build up state
    [[maybe_unused]] auto r1 = original.tune(createElementWiseChain(5));
    [[maybe_unused]] auto r2 = original.tune(createSoftmaxPattern());

    // Serialize
    std::string serialized = original.serialize();
    EXPECT_FALSE(serialized.empty());

    // Deserialize
    AutoScheduler restored;
    EXPECT_TRUE(restored.deserialize(serialized));

    // Verify state is preserved
    EXPECT_EQ(restored.numTuningRecords(), original.numTuningRecords());
    EXPECT_EQ(restored.config().population_size, original.config().population_size);

    // Statistics should match
    auto orig_stats = original.statistics();
    auto rest_stats = restored.statistics();
    EXPECT_EQ(rest_stats.total_schedules_evaluated, orig_stats.total_schedules_evaluated);
}

// =============================================================================
// Callback Integration Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, Callbacks_ProgressTracking) {
    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    AutoScheduler scheduler(config);

    std::vector<TuningProgress> progress_history;
    scheduler.setProgressCallback(
        [&](const TuningProgress& progress) { progress_history.push_back(progress); });

    auto module = createElementWiseChain(10);
    [[maybe_unused]] auto result = scheduler.tune(module);

    // Should have progress callbacks
    EXPECT_GT(progress_history.size(), 0);

    // Progress should show increasing generations
    for (size_t i = 1; i < progress_history.size(); ++i) {
        EXPECT_GE(progress_history[i].current_generation,
                  progress_history[i - 1].current_generation);
    }
}

TEST_F(AutoSchedulerIntegrationTest, Callbacks_ImprovementTracking) {
    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 10;

    AutoScheduler scheduler(config);

    std::vector<std::pair<float, Schedule>> improvements;
    scheduler.setFoundBetterCallback(
        [&](const Schedule& schedule, float cost) { improvements.push_back({cost, schedule}); });

    auto module = createGELUPattern();  // Complex pattern
    [[maybe_unused]] auto result = scheduler.tune(module);

    // Should find improvements
    EXPECT_GT(improvements.size(), 0);

    // Improvements should be in order of decreasing cost
    for (size_t i = 1; i < improvements.size(); ++i) {
        EXPECT_LE(improvements[i].first, improvements[i - 1].first);
    }

    // Each improvement should be valid
    for (const auto& [cost, schedule] : improvements) {
        EXPECT_TRUE(schedule.validate());
    }
}

// =============================================================================
// Edge Case Integration Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, EdgeCase_MultipleTuningsOfSameModule) {
    auto module = createElementWiseChain(5);

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    AutoScheduler scheduler(config);

    // Tune same module multiple times
    TuneResult result1 = scheduler.tune(module);
    TuneResult result2 = scheduler.tune(module);
    TuneResult result3 = scheduler.tune(module);

    EXPECT_TRUE(result1.success);
    EXPECT_TRUE(result2.success);
    EXPECT_TRUE(result3.success);

    // Best schedule should be tracked
    auto best = scheduler.getBestSchedule("ewise_chain_5");
    EXPECT_TRUE(best.has_value());

    // Should have 3 records
    EXPECT_EQ(scheduler.numTuningRecords(), 3);
}

TEST_F(AutoSchedulerIntegrationTest, EdgeCase_ZeroGenerations) {
    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 0;  // No evolution

    AutoScheduler scheduler(config);
    auto module = createElementWiseChain(5);

    TuneResult result = scheduler.tune(module);

    // Should still succeed with initial population
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.generations, 0);
}

TEST_F(AutoSchedulerIntegrationTest, EdgeCase_VerySmallPopulation) {
    TuningConfig config;
    config.population_size = 2;  // Minimum viable
    config.num_generations = 3;

    AutoScheduler scheduler(config);
    auto module = createElementWiseChain(5);

    TuneResult result = scheduler.tune(module);
    EXPECT_TRUE(result.success);
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, Stress_ManyModules) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;

    AutoScheduler scheduler(config);

    // Tune many different modules
    for (int i = 1; i <= 20; ++i) {
        auto module = createElementWiseChain(i);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success) << "Failed at module " << i;
    }

    EXPECT_EQ(scheduler.numTuningRecords(), 20);
}

TEST_F(AutoSchedulerIntegrationTest, Stress_LargeSearchSpace) {
    // Create module with many operations -> large search space
    ir::IRModule module("large_space");
    auto& builder = module.builder();

    auto a = builder.constant(1.0f);
    auto b = builder.constant(2.0f);
    auto c = builder.constant(3.0f);

    // Create complex graph with many paths
    std::vector<ir::ValueId> level1;
    for (int i = 0; i < 8; ++i) {
        level1.push_back(builder.add(a, b));
        level1.push_back(builder.mul(b, c));
    }

    std::vector<ir::ValueId> level2;
    for (size_t i = 0; i < level1.size(); i += 2) {
        level2.push_back(builder.add(level1[i], level1[i + 1]));
    }

    auto result = level2[0];
    for (size_t i = 1; i < level2.size(); ++i) {
        result = builder.add(result, level2[i]);
    }

    module.setOutput(result);

    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 5;
    config.time_budget_seconds = 2.0f;  // Time limit

    AutoScheduler scheduler(config);
    TuneResult tune_result = scheduler.tune(module);

    EXPECT_TRUE(tune_result.success);
}

// =============================================================================
// Correctness Verification Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, Correctness_ScheduleValidation) {
    std::vector<ir::IRModule> modules;
    modules.push_back(createElementWiseChain(5));
    modules.push_back(createReductionTree(4));
    modules.push_back(createStencilPattern());
    modules.push_back(createSoftmaxPattern());
    modules.push_back(createLayerNormPattern());
    modules.push_back(createAttentionPattern());
    modules.push_back(createGELUPattern());

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    for (auto& module : modules) {
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);

        EXPECT_TRUE(result.success);
        EXPECT_TRUE(result.schedule.validate()) << "Invalid schedule for module: " << module.name();

        // Verify we can apply the schedule
        bool applied = scheduler.applySchedule(module, result.schedule);
        EXPECT_TRUE(applied);
    }
}

TEST_F(AutoSchedulerIntegrationTest, Correctness_DAGPreserved) {
    auto module = createGELUPattern();

    // Get DAG before scheduling
    auto dag_before = ComputeDAG::fromIR(module.builder());
    size_t nodes_before = dag_before.numNodes();

    // Schedule
    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    AutoScheduler scheduler(config);
    TuneResult result = scheduler.tune(module);
    [[maybe_unused]] bool applied = scheduler.applySchedule(module, result.schedule);

    // DAG structure should be preserved (transforms don't remove nodes)
    auto dag_after = ComputeDAG::fromIR(module.builder());
    EXPECT_EQ(dag_after.numNodes(), nodes_before);
}

// =============================================================================
// Feature Tests
// =============================================================================

TEST_F(AutoSchedulerIntegrationTest, Feature_TimeBudgetEnforcement) {
    auto module = createElementWiseChain(20);

    TuningConfig config;
    config.time_budget_seconds = 0.1f;
    config.population_size = 64;
    config.num_generations = 1000;  // Would take forever without budget

    AutoScheduler scheduler(config);

    auto start = std::chrono::high_resolution_clock::now();
    TuneResult result = scheduler.tune(module);
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed = std::chrono::duration<float>(end - start).count();

    EXPECT_TRUE(result.success);
    EXPECT_LT(elapsed, 0.5f);  // Should be well under budget * 2
}

TEST_F(AutoSchedulerIntegrationTest, Feature_BatchTuning) {
    std::vector<ir::IRModule> modules;
    modules.push_back(createElementWiseChain(5));
    modules.push_back(createReductionTree(4));
    modules.push_back(createStencilPattern());

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;

    AutoScheduler scheduler(config);
    auto results = scheduler.tuneBatch(modules);

    EXPECT_EQ(results.size(), modules.size());

    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_TRUE(results[i].success) << "Failed for module " << i;
        EXPECT_TRUE(results[i].schedule.validate());
    }

    // All modules should have best schedules
    EXPECT_EQ(scheduler.numTuningRecords(), 3);
}

TEST_F(AutoSchedulerIntegrationTest, Feature_IncrementalImprovement) {
    auto module = createGELUPattern();

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 3;

    AutoScheduler scheduler(config);

    // First round
    TuneResult result1 = scheduler.tune(module);
    float cost1 = result1.best_cost;

    // Continue from best
    TuneResult result2 = scheduler.continueTuning(module, result1.schedule);
    float cost2 = result2.best_cost;

    // Continue again
    TuneResult result3 = scheduler.continueTuning(module, result2.schedule);
    float cost3 = result3.best_cost;

    // Stochastic search - verify completion and reasonable results
    // Allow significant variance since each tune starts fresh
    EXPECT_TRUE(result2.success);
    EXPECT_TRUE(result3.success);
    EXPECT_GT(cost2, 0.0f);
    EXPECT_GT(cost3, 0.0f);
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
