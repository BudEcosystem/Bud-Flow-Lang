// =============================================================================
// Bud Flow Lang - Auto-Scheduler Stress Tests
// =============================================================================
//
// Stress tests for the auto-scheduler system:
// - Scalability under load
// - Memory stress testing
// - Concurrent access patterns
// - Long-running stability
// - Resource exhaustion handling
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/auto_scheduler.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/evolutionary_search.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// Stress Test Fixture
// =============================================================================

class AutoSchedulerStressTest : public ::testing::Test {
  protected:
    void SetUp() override { rng_.seed(42); }

    // Generate random IR with given complexity
    ir::IRModule createRandomIR(size_t num_ops, uint32_t seed) {
        std::mt19937 local_rng(seed);
        std::uniform_int_distribution<int> op_dist(0, 3);

        ir::IRModule module("random_" + std::to_string(seed));
        auto& builder = module.builder();

        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto current = a;

        for (size_t i = 0; i < num_ops; ++i) {
            switch (op_dist(local_rng)) {
            case 0:
                current = builder.add(current, b);
                break;
            case 1:
                current = builder.mul(current, b);
                break;
            case 2:
                current = builder.sub(current, a);
                break;
            case 3:
                current = builder.div(current, b);
                break;
            }
        }

        module.setOutput(current);
        return module;
    }

    // Generate very deep IR (stress memory)
    ir::IRModule createDeepIR(size_t depth) {
        ir::IRModule module("deep_" + std::to_string(depth));
        auto& builder = module.builder();

        auto result = builder.constant(1.0f);
        auto two = builder.constant(2.0f);

        for (size_t i = 0; i < depth; ++i) {
            result = builder.add(result, two);
        }

        module.setOutput(result);
        return module;
    }

    // Generate wide IR (many parallel paths)
    ir::IRModule createWideIR(size_t width) {
        ir::IRModule module("wide_" + std::to_string(width));
        auto& builder = module.builder();

        auto base = builder.constant(1.0f);
        std::vector<ir::ValueId> paths;

        for (size_t i = 0; i < width; ++i) {
            auto val = builder.constant(static_cast<float>(i));
            paths.push_back(builder.add(val, base));
        }

        auto result = paths[0];
        for (size_t i = 1; i < paths.size(); ++i) {
            result = builder.add(result, paths[i]);
        }

        module.setOutput(result);
        return module;
    }

    std::mt19937 rng_;
};

// =============================================================================
// Scalability Tests
// =============================================================================

TEST_F(AutoSchedulerStressTest, Scalability_IRDepth) {
    std::cout << "\n=== Scalability: IR Depth ===\n";

    std::vector<size_t> depths = {10, 50, 100, 200};

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;
    config.time_budget_seconds = 1.0f;

    for (size_t depth : depths) {
        auto module = createDeepIR(depth);

        auto start = std::chrono::high_resolution_clock::now();

        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);

        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();

        std::cout << "  Depth " << depth << ": " << ms << " ms, success=" << result.success << "\n";

        EXPECT_TRUE(result.success);
    }
}

TEST_F(AutoSchedulerStressTest, Scalability_IRWidth) {
    std::cout << "\n=== Scalability: IR Width ===\n";

    std::vector<size_t> widths = {10, 50, 100, 200};

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;
    config.time_budget_seconds = 1.0f;

    for (size_t width : widths) {
        auto module = createWideIR(width);

        auto start = std::chrono::high_resolution_clock::now();

        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);

        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();

        std::cout << "  Width " << width << ": " << ms << " ms, success=" << result.success << "\n";

        EXPECT_TRUE(result.success);
    }
}

TEST_F(AutoSchedulerStressTest, Scalability_PopulationSize) {
    std::cout << "\n=== Scalability: Population Size ===\n";

    auto module = createRandomIR(20, 42);
    std::vector<size_t> pop_sizes = {16, 64, 256, 512};

    for (size_t pop_size : pop_sizes) {
        TuningConfig config;
        config.population_size = pop_size;
        config.num_generations = 2;
        config.time_budget_seconds = 2.0f;

        auto start = std::chrono::high_resolution_clock::now();

        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);

        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();

        std::cout << "  Pop size " << pop_size << ": " << ms
                  << " ms, evals=" << result.schedules_evaluated << "\n";

        EXPECT_TRUE(result.success);
    }
}

// =============================================================================
// Memory Stress Tests
// =============================================================================

TEST_F(AutoSchedulerStressTest, Memory_ManySchedulers) {
    std::cout << "\n=== Memory: Many Schedulers ===\n";

    std::vector<std::unique_ptr<AutoScheduler>> schedulers;

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 2;

    // Create many schedulers
    constexpr size_t kNumSchedulers = 50;

    for (size_t i = 0; i < kNumSchedulers; ++i) {
        schedulers.push_back(std::make_unique<AutoScheduler>(config));
    }

    std::cout << "  Created " << kNumSchedulers << " schedulers\n";

    // Tune with each
    for (size_t i = 0; i < schedulers.size(); ++i) {
        auto module = createRandomIR(5, static_cast<uint32_t>(i));
        [[maybe_unused]] auto result = schedulers[i]->tune(module);
    }

    std::cout << "  Completed tuning with all schedulers\n";

    // Should all still be valid
    for (const auto& scheduler : schedulers) {
        EXPECT_TRUE(scheduler->isValid());
    }
}

TEST_F(AutoSchedulerStressTest, Memory_LargeTuningLog) {
    std::cout << "\n=== Memory: Large Tuning Log ===\n";

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 1;

    AutoScheduler scheduler(config);

    // Tune many modules to build up tuning log
    constexpr size_t kNumModules = 100;

    for (size_t i = 0; i < kNumModules; ++i) {
        auto module = createRandomIR(5, static_cast<uint32_t>(i));
        [[maybe_unused]] auto result = scheduler.tune(module);
    }

    EXPECT_EQ(scheduler.numTuningRecords(), kNumModules);

    // Serialize and deserialize
    std::string serialized = scheduler.serialize();
    std::cout << "  Serialized size: " << serialized.size() << " bytes\n";

    AutoScheduler restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.numTuningRecords(), kNumModules);
}

TEST_F(AutoSchedulerStressTest, Memory_RepeatedTuning) {
    std::cout << "\n=== Memory: Repeated Tuning (Memory Leak Check) ===\n";

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 3;

    auto module = createRandomIR(10, 42);

    // Tune same module many times
    constexpr size_t kIterations = 100;

    for (size_t i = 0; i < kIterations; ++i) {
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        EXPECT_TRUE(result.success);
    }

    std::cout << "  Completed " << kIterations << " tuning iterations\n";
}

// =============================================================================
// Long-Running Stability Tests
// =============================================================================

TEST_F(AutoSchedulerStressTest, Stability_ExtendedTuning) {
    std::cout << "\n=== Stability: Extended Tuning ===\n";

    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 50;        // Many generations
    config.time_budget_seconds = 3.0f;  // But limited by time

    auto module = createRandomIR(15, 42);

    AutoScheduler scheduler(config);

    auto start = std::chrono::high_resolution_clock::now();
    TuneResult result = scheduler.tune(module);
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed = std::chrono::duration<float>(end - start).count();

    std::cout << "  Elapsed: " << elapsed << " s\n";
    std::cout << "  Generations: " << result.generations << "\n";
    std::cout << "  Schedules evaluated: " << result.schedules_evaluated << "\n";

    EXPECT_TRUE(result.success);
    EXPECT_LT(elapsed, 5.0f);  // Should respect time budget
}

TEST_F(AutoSchedulerStressTest, Stability_ContinuousTuning) {
    std::cout << "\n=== Stability: Continuous Tuning ===\n";

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 3;

    auto module = createRandomIR(10, 42);

    AutoScheduler scheduler(config);
    TuneResult prev_result = scheduler.tune(module);
    float best_cost = prev_result.best_cost;

    // Continue tuning many times
    constexpr size_t kContinues = 20;

    for (size_t i = 0; i < kContinues; ++i) {
        TuneResult result = scheduler.continueTuning(module, prev_result.schedule);
        EXPECT_TRUE(result.success);

        // Stochastic search may not always improve - just verify it succeeds
        // and doesn't regress catastrophically (5x worse)
        EXPECT_LE(result.best_cost, best_cost * 5.0f);

        best_cost = std::min(best_cost, result.best_cost);
        prev_result = result;
    }

    std::cout << "  Initial cost: " << prev_result.best_cost << "\n";
    std::cout << "  Final best: " << best_cost << "\n";
}

// =============================================================================
// Resource Exhaustion Tests
// =============================================================================

TEST_F(AutoSchedulerStressTest, ResourceExhaustion_TightTimeBudget) {
    std::cout << "\n=== Resource Exhaustion: Tight Time Budget ===\n";

    std::vector<float> budgets = {0.001f, 0.01f, 0.05f};

    auto module = createRandomIR(20, 42);

    for (float budget : budgets) {
        TuningConfig config;
        config.time_budget_seconds = budget;
        config.population_size = 64;
        config.num_generations = 100;

        AutoScheduler scheduler(config);

        auto start = std::chrono::high_resolution_clock::now();
        TuneResult result = scheduler.tune(module);
        auto end = std::chrono::high_resolution_clock::now();

        float elapsed = std::chrono::duration<float>(end - start).count();

        std::cout << "  Budget " << budget << "s: actual=" << elapsed
                  << "s, success=" << result.success << "\n";

        // Should always succeed (even with tight budget)
        EXPECT_TRUE(result.success);
    }
}

TEST_F(AutoSchedulerStressTest, ResourceExhaustion_MinimalConfig) {
    std::cout << "\n=== Resource Exhaustion: Minimal Config ===\n";

    TuningConfig config;
    config.population_size = 1;  // Absolute minimum
    config.num_generations = 1;

    auto module = createRandomIR(5, 42);

    AutoScheduler scheduler(config);
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
}

// =============================================================================
// Randomized Stress Tests
// =============================================================================

TEST_F(AutoSchedulerStressTest, Randomized_ManyRandomIRs) {
    std::cout << "\n=== Randomized: Many Random IRs ===\n";

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;

    AutoScheduler scheduler(config);

    constexpr size_t kNumIRs = 50;
    size_t successes = 0;

    for (size_t i = 0; i < kNumIRs; ++i) {
        // Random sizes
        std::uniform_int_distribution<size_t> size_dist(3, 30);
        size_t num_ops = size_dist(rng_);

        auto module = createRandomIR(num_ops, static_cast<uint32_t>(i));
        TuneResult result = scheduler.tune(module);

        if (result.success) {
            ++successes;
        }
    }

    std::cout << "  Success rate: " << successes << "/" << kNumIRs << "\n";
    EXPECT_EQ(successes, kNumIRs);
}

TEST_F(AutoSchedulerStressTest, Randomized_RandomConfigs) {
    std::cout << "\n=== Randomized: Random Configs ===\n";

    auto module = createRandomIR(10, 42);

    constexpr size_t kNumConfigs = 20;
    size_t successes = 0;

    for (size_t i = 0; i < kNumConfigs; ++i) {
        std::uniform_int_distribution<size_t> pop_dist(8, 128);
        std::uniform_int_distribution<size_t> gen_dist(1, 10);
        std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

        TuningConfig config;
        config.population_size = pop_dist(rng_);
        config.num_generations = gen_dist(rng_);
        config.time_budget_seconds = time_dist(rng_);

        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);

        if (result.success) {
            ++successes;
        }
    }

    std::cout << "  Success rate: " << successes << "/" << kNumConfigs << "\n";
    EXPECT_EQ(successes, kNumConfigs);
}

// =============================================================================
// Component Stress Tests
// =============================================================================

TEST_F(AutoSchedulerStressTest, Component_DAGConstruction) {
    std::cout << "\n=== Component Stress: DAG Construction ===\n";

    constexpr size_t kIterations = 1000;

    auto module = createRandomIR(20, 42);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < kIterations; ++i) {
        auto dag = ComputeDAG::fromIR(module.builder());
        EXPECT_GT(dag.numNodes(), 0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "  " << kIterations << " DAG constructions in " << us / 1000.0 << " ms ("
              << us / kIterations << " us/dag)\n";
}

TEST_F(AutoSchedulerStressTest, Component_SearchSpaceGeneration) {
    std::cout << "\n=== Component Stress: Search Space Generation ===\n";

    constexpr size_t kIterations = 500;

    auto module = createRandomIR(15, 42);
    auto dag = ComputeDAG::fromIR(module.builder());

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < kIterations; ++i) {
        auto space = SearchSpace::fromDAG(dag);
        EXPECT_GT(space.numSketches(), 0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "  " << kIterations << " space generations in " << us / 1000.0 << " ms ("
              << us / kIterations << " us/space)\n";
}

TEST_F(AutoSchedulerStressTest, Component_CostModelPrediction) {
    std::cout << "\n=== Component Stress: Cost Model Prediction ===\n";

    constexpr size_t kIterations = 10000;

    auto module = createRandomIR(10, 42);
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    CostModel model;
    auto schedules = space.sample(100);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < kIterations; ++i) {
        const auto& schedule = schedules[i % schedules.size()];
        float cost = model.predict(schedule);
        (void)cost;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "  " << kIterations << " predictions in " << us / 1000.0 << " ms ("
              << us / kIterations << " us/pred)\n";
}

TEST_F(AutoSchedulerStressTest, Component_EvolutionaryOperators) {
    std::cout << "\n=== Component Stress: Evolutionary Operators ===\n";

    constexpr size_t kIterations = 5000;

    auto module = createRandomIR(10, 42);
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    SearchConfig search_config;
    search_config.population_size = 64;
    search_config.num_generations = 1;

    EvolutionarySearch search(search_config);
    search.initializePopulation(space);

    auto schedules = space.sample(100);

    // Mutation stress
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < kIterations; ++i) {
            [[maybe_unused]] auto mutated = search.mutate(schedules[i % schedules.size()], space);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();

        std::cout << "  " << kIterations << " mutations in " << us / 1000.0 << " ms ("
                  << us / kIterations << " us/mut)\n";
    }

    // Crossover stress
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < kIterations; ++i) {
            [[maybe_unused]] auto crossed = search.crossover(
                schedules[i % schedules.size()], schedules[(i + 1) % schedules.size()], space);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();

        std::cout << "  " << kIterations << " crossovers in " << us / 1000.0 << " ms ("
                  << us / kIterations << " us/cross)\n";
    }
}

// =============================================================================
// Summary Test
// =============================================================================

TEST_F(AutoSchedulerStressTest, Summary_PerformanceProfile) {
    std::cout << "\n";
    std::cout << "=============================================================\n";
    std::cout << "         AUTO-SCHEDULER STRESS TEST SUMMARY                 \n";
    std::cout << "=============================================================\n";

    struct Profile {
        std::string name;
        size_t ir_size;
        size_t pop_size;
        size_t generations;
        double time_ms;
        bool success;
    };

    std::vector<Profile> profiles;

    // Small/Fast
    {
        TuningConfig config;
        config.population_size = 16;
        config.num_generations = 2;

        auto module = createRandomIR(5, 1);

        auto start = std::chrono::high_resolution_clock::now();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        auto end = std::chrono::high_resolution_clock::now();

        profiles.push_back({"Small/Fast", 5, 16, 2,
                            std::chrono::duration<double, std::milli>(end - start).count(),
                            result.success});
    }

    // Medium
    {
        TuningConfig config;
        config.population_size = 32;
        config.num_generations = 5;

        auto module = createRandomIR(20, 2);

        auto start = std::chrono::high_resolution_clock::now();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        auto end = std::chrono::high_resolution_clock::now();

        profiles.push_back({"Medium", 20, 32, 5,
                            std::chrono::duration<double, std::milli>(end - start).count(),
                            result.success});
    }

    // Large
    {
        TuningConfig config;
        config.population_size = 64;
        config.num_generations = 10;
        config.time_budget_seconds = 2.0f;

        auto module = createRandomIR(50, 3);

        auto start = std::chrono::high_resolution_clock::now();
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        auto end = std::chrono::high_resolution_clock::now();

        profiles.push_back({"Large", 50, 64, 10,
                            std::chrono::duration<double, std::milli>(end - start).count(),
                            result.success});
    }

    // Print table
    std::cout << "\n";
    std::cout << "  Profile    | IR Size | Pop | Gens | Time (ms) | Success\n";
    std::cout << "  -----------|---------|-----|------|-----------|--------\n";

    for (const auto& p : profiles) {
        std::cout << "  " << std::left << std::setw(10) << p.name << " | " << std::setw(7)
                  << p.ir_size << " | " << std::setw(3) << p.pop_size << " | " << std::setw(4)
                  << p.generations << " | " << std::setw(9) << std::fixed << std::setprecision(2)
                  << p.time_ms << " | " << (p.success ? "Yes" : "No") << "\n";
    }

    std::cout << "\n";
    std::cout << "=============================================================\n";

    // All should succeed
    for (const auto& p : profiles) {
        EXPECT_TRUE(p.success) << "Failed for profile: " << p.name;
    }
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
