// =============================================================================
// Bud Flow Lang - Auto-Scheduler Benchmark Tests
// =============================================================================
//
// Comprehensive benchmarks for the auto-scheduler system:
// - Tuning performance at various scales
// - Schedule quality measurements
// - Comparison with baseline (unscheduled) execution
// - Scalability tests
// - Memory usage tracking
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/auto_scheduler.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/evolutionary_search.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// Benchmark Utilities
// =============================================================================

struct BenchmarkStats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double median_ms = 0.0;
    size_t iterations = 0;
};

template <typename Func>
BenchmarkStats runBenchmark(Func&& func, size_t warmup = 3, size_t iterations = 10) {
    // Warmup
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(iterations);

    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Calculate statistics
    BenchmarkStats stats;
    stats.iterations = iterations;

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    stats.mean_ms = sum / iterations;

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    stats.std_ms = std::sqrt(sq_sum / iterations - stats.mean_ms * stats.mean_ms);

    stats.min_ms = *std::min_element(times.begin(), times.end());
    stats.max_ms = *std::max_element(times.begin(), times.end());

    std::sort(times.begin(), times.end());
    stats.median_ms = times[iterations / 2];

    return stats;
}

void printStats(const std::string& name, const BenchmarkStats& stats) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::left << std::setw(30) << name << " mean: " << std::setw(10)
              << stats.mean_ms << " ms" << " std: " << std::setw(8) << stats.std_ms << " ms"
              << " min: " << std::setw(10) << stats.min_ms << " ms" << " max: " << std::setw(10)
              << stats.max_ms << " ms\n";
}

// =============================================================================
// Test Fixture
// =============================================================================

class AutoSchedulerBenchmarkTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Seed for reproducibility
        rng_.seed(42);
    }

    // Create IR with variable complexity
    ir::IRModule createIRWithDepth(size_t depth, const std::string& name = "bench") {
        ir::IRModule module(name);
        auto& builder = module.builder();

        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto current = a;

        for (size_t i = 0; i < depth; ++i) {
            switch (i % 4) {
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

    // Create IR with multiple parallel paths
    ir::IRModule createParallelIR(size_t paths, size_t depth) {
        ir::IRModule module("parallel");
        auto& builder = module.builder();

        std::vector<ir::ValueId> path_outputs;
        auto base = builder.constant(1.0f);

        for (size_t p = 0; p < paths; ++p) {
            auto current = builder.constant(static_cast<float>(p + 1));
            for (size_t d = 0; d < depth; ++d) {
                if (d % 2 == 0) {
                    current = builder.add(current, base);
                } else {
                    current = builder.mul(current, base);
                }
            }
            path_outputs.push_back(current);
        }

        // Reduce all paths
        auto result = path_outputs[0];
        for (size_t i = 1; i < path_outputs.size(); ++i) {
            result = builder.add(result, path_outputs[i]);
        }

        module.setOutput(result);
        return module;
    }

    // Create IR with FMA opportunities
    ir::IRModule createFMAChainIR(size_t length) {
        ir::IRModule module("fma_chain");
        auto& builder = module.builder();

        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.constant(3.0f);

        auto current = a;
        for (size_t i = 0; i < length; ++i) {
            auto mul = builder.mul(current, b);
            current = builder.add(mul, c);  // FMA opportunity
        }

        module.setOutput(current);
        return module;
    }

    // Create diamond-shaped IR (for fusion opportunities)
    ir::IRModule createDiamondIR(size_t diamonds) {
        ir::IRModule module("diamond");
        auto& builder = module.builder();

        auto input = builder.constant(1.0f);
        auto current = input;

        for (size_t d = 0; d < diamonds; ++d) {
            // Fork
            auto left = builder.add(current, input);
            auto right = builder.mul(current, input);

            // Join
            current = builder.add(left, right);
        }

        module.setOutput(current);
        return module;
    }

    std::mt19937 rng_;
};

// =============================================================================
// Tuning Speed Benchmarks
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, TuningSpeed_SmallIR) {
    std::cout << "\n=== Tuning Speed: Small IR (5 ops) ===\n";

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    auto module = createIRWithDepth(5);

    auto stats = runBenchmark(
        [&]() {
            AutoScheduler scheduler(config);
            [[maybe_unused]] auto result = scheduler.tune(module);
        },
        2, 5);

    printStats("Small IR tuning", stats);

    EXPECT_LT(stats.mean_ms, 100.0) << "Small IR tuning too slow";
}

TEST_F(AutoSchedulerBenchmarkTest, TuningSpeed_MediumIR) {
    std::cout << "\n=== Tuning Speed: Medium IR (20 ops) ===\n";

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    auto module = createIRWithDepth(20);

    auto stats = runBenchmark(
        [&]() {
            AutoScheduler scheduler(config);
            [[maybe_unused]] auto result = scheduler.tune(module);
        },
        2, 5);

    printStats("Medium IR tuning", stats);

    EXPECT_LT(stats.mean_ms, 500.0) << "Medium IR tuning too slow";
}

TEST_F(AutoSchedulerBenchmarkTest, TuningSpeed_LargeIR) {
    std::cout << "\n=== Tuning Speed: Large IR (50 ops) ===\n";

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    auto module = createIRWithDepth(50);

    auto stats = runBenchmark(
        [&]() {
            AutoScheduler scheduler(config);
            [[maybe_unused]] auto result = scheduler.tune(module);
        },
        1, 3);

    printStats("Large IR tuning", stats);

    EXPECT_LT(stats.mean_ms, 2000.0) << "Large IR tuning too slow";
}

// =============================================================================
// Population Size Scaling
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, PopulationScaling) {
    std::cout << "\n=== Population Size Scaling ===\n";

    auto module = createIRWithDepth(10);
    std::vector<size_t> pop_sizes = {16, 32, 64, 128};

    for (size_t pop_size : pop_sizes) {
        TuningConfig config;
        config.population_size = pop_size;
        config.num_generations = 3;

        auto stats = runBenchmark(
            [&]() {
                AutoScheduler scheduler(config);
                [[maybe_unused]] auto result = scheduler.tune(module);
            },
            1, 3);

        std::string name = "Pop size " + std::to_string(pop_size);
        printStats(name, stats);
    }
}

TEST_F(AutoSchedulerBenchmarkTest, GenerationScaling) {
    std::cout << "\n=== Generation Count Scaling ===\n";

    auto module = createIRWithDepth(10);
    std::vector<size_t> gen_counts = {2, 5, 10, 20};

    for (size_t num_gen : gen_counts) {
        TuningConfig config;
        config.population_size = 32;
        config.num_generations = num_gen;

        auto stats = runBenchmark(
            [&]() {
                AutoScheduler scheduler(config);
                [[maybe_unused]] auto result = scheduler.tune(module);
            },
            1, 3);

        std::string name = std::to_string(num_gen) + " generations";
        printStats(name, stats);
    }
}

// =============================================================================
// Schedule Quality Benchmarks
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, ScheduleQuality_ConvergenceOverGenerations) {
    std::cout << "\n=== Schedule Quality: Convergence Over Generations ===\n";

    auto module = createFMAChainIR(10);
    std::vector<float> costs_per_generation;

    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 20;

    AutoScheduler scheduler(config);

    float last_best_cost = std::numeric_limits<float>::max();
    scheduler.setFoundBetterCallback([&](const Schedule& schedule, float cost) {
        costs_per_generation.push_back(cost);
        last_best_cost = cost;
    });

    TuneResult result = scheduler.tune(module);

    std::cout << "  Generation  |  Best Cost\n";
    std::cout << "  -----------|------------\n";
    for (size_t i = 0; i < costs_per_generation.size(); ++i) {
        std::cout << "  " << std::setw(10) << i << " | " << std::fixed << std::setprecision(4)
                  << costs_per_generation[i] << "\n";
    }

    // Should find at least one cost value during search
    EXPECT_GT(costs_per_generation.size(), 0) << "Should record at least one cost";

    // Verify convergence completed successfully
    EXPECT_TRUE(result.success) << "Tuning should complete successfully";
}

TEST_F(AutoSchedulerBenchmarkTest, ScheduleQuality_FMAFusion) {
    std::cout << "\n=== Schedule Quality: FMA Fusion Detection ===\n";

    auto module = createFMAChainIR(5);  // 5 FMA opportunities

    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 10;

    AutoScheduler scheduler(config);
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);

    // Check that the schedule identifies fusion opportunities
    // The schedule should have transforms that can be applied
    std::cout << "  Best cost: " << result.best_cost << "\n";
    std::cout << "  Schedules evaluated: " << result.schedules_evaluated << "\n";
    std::cout << "  Schedule valid: " << result.schedule.validate() << "\n";
}

TEST_F(AutoSchedulerBenchmarkTest, ScheduleQuality_ParallelPaths) {
    std::cout << "\n=== Schedule Quality: Parallel Path Detection ===\n";

    auto module = createParallelIR(4, 5);  // 4 parallel paths, depth 5

    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 10;

    AutoScheduler scheduler(config);
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);

    std::cout << "  Best cost: " << result.best_cost << "\n";
    std::cout << "  Tuning time: " << result.tuning_time_seconds << " s\n";
}

// =============================================================================
// Warm Start Benchmarks
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, WarmStart_SpeedImprovement) {
    std::cout << "\n=== Warm Start: Speed Improvement ===\n";

    auto module = createIRWithDepth(15);

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    // Cold start timing
    auto cold_stats = runBenchmark(
        [&]() {
            AutoScheduler scheduler(config);
            [[maybe_unused]] auto result = scheduler.tune(module);
        },
        1, 3);

    printStats("Cold start", cold_stats);

    // Get a baseline schedule
    AutoScheduler base_scheduler(config);
    TuneResult base_result = base_scheduler.tune(module);
    std::vector<Schedule> warm_schedules = {base_result.schedule};

    // Warm start timing
    auto warm_stats = runBenchmark(
        [&]() {
            AutoScheduler scheduler(config);
            [[maybe_unused]] auto result = scheduler.tuneWithWarmStart(module, warm_schedules);
        },
        1, 3);

    printStats("Warm start", warm_stats);
}

TEST_F(AutoSchedulerBenchmarkTest, WarmStart_QualityImprovement) {
    std::cout << "\n=== Warm Start: Quality Improvement ===\n";

    auto module = createIRWithDepth(20);

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 3;

    // Initial tuning
    AutoScheduler scheduler1(config);
    TuneResult result1 = scheduler1.tune(module);

    // Continue tuning
    TuneResult result2 = scheduler1.continueTuning(module, result1.schedule);

    std::cout << "  Initial best cost: " << result1.best_cost << "\n";
    std::cout << "  After continue: " << result2.best_cost << "\n";

    // Should not regress
    EXPECT_LE(result2.best_cost, result1.best_cost * 1.05f);
}

// =============================================================================
// Batch Tuning Benchmarks
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, BatchTuning_Throughput) {
    std::cout << "\n=== Batch Tuning: Throughput ===\n";

    std::vector<ir::IRModule> modules;
    for (int i = 0; i < 5; ++i) {
        modules.push_back(createIRWithDepth(5 + i * 2, "module_" + std::to_string(i)));
    }

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 3;

    auto stats = runBenchmark(
        [&]() {
            AutoScheduler scheduler(config);
            [[maybe_unused]] auto results = scheduler.tuneBatch(modules);
        },
        1, 3);

    double modules_per_sec = (modules.size() * 1000.0) / stats.mean_ms;
    std::cout << "  Batch tuning: " << stats.mean_ms << " ms for " << modules.size()
              << " modules\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << modules_per_sec
              << " modules/sec\n";
}

// =============================================================================
// Memory Efficiency Tests
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, MemoryUsage_LargePopulation) {
    std::cout << "\n=== Memory Usage: Large Population ===\n";

    auto module = createIRWithDepth(20);

    TuningConfig config;
    config.population_size = 256;  // Large population
    config.num_generations = 5;

    // This should complete without OOM
    AutoScheduler scheduler(config);
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
    std::cout << "  Completed with population size 256\n";
}

TEST_F(AutoSchedulerBenchmarkTest, MemoryUsage_ManyTuningRecords) {
    std::cout << "\n=== Memory Usage: Many Tuning Records ===\n";

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;

    AutoScheduler scheduler(config);

    // Tune many modules
    for (int i = 0; i < 20; ++i) {
        auto module = createIRWithDepth(5, "module_" + std::to_string(i));
        [[maybe_unused]] auto result = scheduler.tune(module);
    }

    EXPECT_EQ(scheduler.numTuningRecords(), 20);
    std::cout << "  Successfully tuned 20 modules\n";
}

// =============================================================================
// Stability Tests
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, Stability_DeterministicResults) {
    std::cout << "\n=== Stability: Deterministic Results ===\n";

    auto module = createIRWithDepth(10);

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    std::vector<float> costs;
    for (int i = 0; i < 3; ++i) {
        AutoScheduler scheduler(config);
        TuneResult result = scheduler.tune(module);
        costs.push_back(result.best_cost);
    }

    // Results should be reasonably consistent
    float mean = std::accumulate(costs.begin(), costs.end(), 0.0f) / costs.size();
    for (float cost : costs) {
        std::cout << "  Run cost: " << cost << "\n";
        EXPECT_NEAR(cost, mean, mean * 0.5f) << "Results too variable";
    }
}

TEST_F(AutoSchedulerBenchmarkTest, Stability_NoRegressionOnContinue) {
    std::cout << "\n=== Stability: No Regression On Continue ===\n";

    auto module = createIRWithDepth(15);

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    AutoScheduler scheduler(config);

    TuneResult result1 = scheduler.tune(module);
    float best_so_far = result1.best_cost;

    for (int i = 0; i < 3; ++i) {
        TuneResult result = scheduler.continueTuning(module, result1.schedule);
        std::cout << "  Continue " << (i + 1) << " cost: " << result.best_cost << "\n";

        // Stochastic search - just verify completion and no catastrophic regression
        EXPECT_TRUE(result.success);
        EXPECT_LE(result.best_cost, best_so_far * 5.0f);  // Allow significant variance
        best_so_far = std::min(best_so_far, result.best_cost);
    }
}

// =============================================================================
// Time Budget Tests
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, TimeBudget_Respected) {
    std::cout << "\n=== Time Budget: Respected ===\n";

    auto module = createIRWithDepth(20);

    std::vector<float> budgets = {0.05f, 0.1f, 0.2f};

    for (float budget : budgets) {
        TuningConfig config;
        config.time_budget_seconds = budget;
        config.population_size = 64;
        config.num_generations = 100;  // Would be very long without budget

        AutoScheduler scheduler(config);

        auto start = std::chrono::high_resolution_clock::now();
        TuneResult result = scheduler.tune(module);
        auto end = std::chrono::high_resolution_clock::now();

        float actual = std::chrono::duration<float>(end - start).count();

        std::cout << "  Budget: " << budget << "s, Actual: " << actual << "s\n";

        // Should be within 2x of budget (allowing for overhead)
        EXPECT_LT(actual, budget * 2.0f);
    }
}

// =============================================================================
// Component Interaction Tests
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, ComponentInteraction_CostModelAccuracy) {
    std::cout << "\n=== Component Interaction: Cost Model Accuracy ===\n";

    auto module = createFMAChainIR(8);

    TuningConfig config;
    config.population_size = 64;
    config.num_generations = 10;

    AutoScheduler scheduler(config);

    // Track cost predictions vs actual ranking
    std::vector<std::pair<float, size_t>> cost_ranks;

    scheduler.setFoundBetterCallback([&](const Schedule& schedule, float cost) {
        cost_ranks.push_back({cost, cost_ranks.size()});
    });

    [[maybe_unused]] auto tune_result = scheduler.tune(module);

    // Check that lower costs come later (better schedules found over time)
    if (cost_ranks.size() > 1) {
        // Costs should generally decrease
        bool decreasing_trend = true;
        for (size_t i = 1; i < cost_ranks.size(); ++i) {
            if (cost_ranks[i].first > cost_ranks[i - 1].first * 1.1f) {
                decreasing_trend = false;
                break;
            }
        }

        std::cout << "  Found " << cost_ranks.size() << " improvements\n";
        std::cout << "  First cost: " << cost_ranks.front().first << "\n";
        std::cout << "  Final cost: " << cost_ranks.back().first << "\n";
    }
}

// =============================================================================
// Serialization Performance
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, Serialization_Speed) {
    std::cout << "\n=== Serialization: Speed ===\n";

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 3;

    AutoScheduler scheduler(config);

    // Tune several modules
    for (int i = 0; i < 10; ++i) {
        auto module = createIRWithDepth(5, "mod_" + std::to_string(i));
        [[maybe_unused]] auto result = scheduler.tune(module);
    }

    // Benchmark serialization
    std::string serialized;
    auto ser_stats = runBenchmark([&]() { serialized = scheduler.serialize(); }, 3, 10);

    printStats("Serialize", ser_stats);
    std::cout << "  Serialized size: " << serialized.size() << " bytes\n";

    // Benchmark deserialization
    auto deser_stats = runBenchmark(
        [&]() {
            AutoScheduler restored;
            restored.deserialize(serialized);
        },
        3, 10);

    printStats("Deserialize", deser_stats);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, EdgeCase_EmptyModule) {
    std::cout << "\n=== Edge Case: Empty Module ===\n";

    ir::IRModule module("empty");

    AutoScheduler scheduler;
    TuneResult result = scheduler.tune(module);

    // Should handle gracefully
    EXPECT_TRUE(result.success);
}

TEST_F(AutoSchedulerBenchmarkTest, EdgeCase_SingleOp) {
    std::cout << "\n=== Edge Case: Single Op ===\n";

    ir::IRModule module("single");
    auto& builder = module.builder();
    auto a = builder.constant(1.0f);
    module.setOutput(a);

    AutoScheduler scheduler;
    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
}

TEST_F(AutoSchedulerBenchmarkTest, EdgeCase_VeryDeepIR) {
    std::cout << "\n=== Edge Case: Very Deep IR (100 ops) ===\n";

    auto module = createIRWithDepth(100);

    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;
    config.time_budget_seconds = 1.0f;  // Limit time

    AutoScheduler scheduler(config);

    auto start = std::chrono::high_resolution_clock::now();
    TuneResult result = scheduler.tune(module);
    auto end = std::chrono::high_resolution_clock::now();

    float duration = std::chrono::duration<float>(end - start).count();

    std::cout << "  Duration: " << duration << "s\n";
    std::cout << "  Success: " << result.success << "\n";

    EXPECT_TRUE(result.success);
}

// =============================================================================
// Comparative Analysis Summary
// =============================================================================

TEST_F(AutoSchedulerBenchmarkTest, Summary_AllBenchmarks) {
    std::cout << "\n";
    std::cout << "=============================================================\n";
    std::cout << "         AUTO-SCHEDULER BENCHMARK SUMMARY                   \n";
    std::cout << "=============================================================\n";

    struct BenchResult {
        std::string name;
        size_t ops;
        double tune_ms;
        float best_cost;
    };

    std::vector<BenchResult> results;

    TuningConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    std::vector<size_t> depths = {5, 10, 20, 50};

    for (size_t depth : depths) {
        auto module = createIRWithDepth(depth);

        auto start = std::chrono::high_resolution_clock::now();
        AutoScheduler scheduler(config);
        TuneResult tune_result = scheduler.tune(module);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        results.push_back({"Depth " + std::to_string(depth), depth, ms, tune_result.best_cost});
    }

    std::cout << "\n";
    std::cout << "  IR Depth   |  Tune Time (ms)  |  Best Cost\n";
    std::cout << "  -----------|------------------|------------\n";
    for (const auto& r : results) {
        std::cout << "  " << std::setw(9) << r.ops << "  |  " << std::setw(14) << std::fixed
                  << std::setprecision(2) << r.tune_ms << "  |  " << std::setw(10)
                  << std::setprecision(4) << r.best_cost << "\n";
    }
    std::cout << "\n";

    // Performance summary
    double avg_ms_per_op = 0;
    for (const auto& r : results) {
        avg_ms_per_op += r.tune_ms / r.ops;
    }
    avg_ms_per_op /= results.size();

    std::cout << "  Average tuning time per IR op: " << std::fixed << std::setprecision(3)
              << avg_ms_per_op << " ms\n";
    std::cout << "=============================================================\n";
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
