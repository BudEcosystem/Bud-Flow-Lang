// =============================================================================
// Bud Flow Lang - Auto-Scheduler Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for AutoScheduler which provides the main entry point for automatic
// schedule optimization.
//
// Features:
// - One-shot tuning
// - Incremental tuning
// - Schedule application
// - Tuning log export/import
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/auto_scheduler.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/evolutionary_search.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <chrono>
#include <filesystem>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// AutoScheduler Construction Tests
// =============================================================================

class AutoSchedulerTest : public ::testing::Test {
  protected:
    void SetUp() override {}

    ir::IRModule createSimpleIR() {
        ir::IRModule module("simple");
        auto& builder = module.builder();
        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.add(a, b);
        module.setOutput(c);
        return module;
    }

    ir::IRModule createComplexIR() {
        ir::IRModule module("complex");
        auto& builder = module.builder();
        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.constant(3.0f);

        auto x = builder.add(a, b);
        auto y = builder.mul(x, c);
        auto z = builder.sub(y, a);
        auto w = builder.div(z, b);
        module.setOutput(w);
        return module;
    }

    ir::IRModule createFusableIR() {
        ir::IRModule module("fusable");
        auto& builder = module.builder();
        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.constant(3.0f);

        auto x = builder.mul(a, b);
        auto y = builder.add(x, c);  // FMA opportunity
        auto z = builder.mul(y, a);
        auto w = builder.add(z, b);  // Another FMA
        module.setOutput(w);
        return module;
    }
};

TEST_F(AutoSchedulerTest, DefaultConstruction) {
    AutoScheduler scheduler;
    EXPECT_TRUE(scheduler.isValid());
}

TEST_F(AutoSchedulerTest, ConstructWithConfig) {
    TuningConfig config;
    config.time_budget_seconds = 10.0f;
    config.population_size = 128;
    config.num_generations = 10;

    AutoScheduler scheduler(config);
    EXPECT_TRUE(scheduler.isValid());
    EXPECT_EQ(scheduler.config().population_size, 128);
}

// =============================================================================
// Tuning Tests
// =============================================================================

TEST_F(AutoSchedulerTest, Tune_Simple) {
    AutoScheduler scheduler;
    auto module = createSimpleIR();

    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.schedule.validate());
}

TEST_F(AutoSchedulerTest, Tune_Complex) {
    AutoScheduler scheduler;
    auto module = createComplexIR();

    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.schedule.validate());
}

TEST_F(AutoSchedulerTest, Tune_WithConfig) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 3;

    AutoScheduler scheduler(config);
    auto module = createComplexIR();

    TuneResult result = scheduler.tune(module);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.generations, 1);
}

TEST_F(AutoSchedulerTest, Tune_TimeBudget) {
    TuningConfig config;
    config.time_budget_seconds = 0.1f;  // Short budget
    config.population_size = 32;
    config.num_generations = 100;  // Will be limited by time

    AutoScheduler scheduler(config);
    auto module = createComplexIR();

    auto start = std::chrono::high_resolution_clock::now();
    TuneResult result = scheduler.tune(module);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_s = std::chrono::duration<float>(end - start).count();

    // Should respect time budget (with some slack)
    EXPECT_LT(duration_s, config.time_budget_seconds * 2.0f);
    EXPECT_TRUE(result.success);
}

// =============================================================================
// Incremental Tuning Tests
// =============================================================================

TEST_F(AutoSchedulerTest, ContinueTuning) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 3;

    AutoScheduler scheduler(config);
    auto module = createComplexIR();

    // Initial tuning
    TuneResult result1 = scheduler.tune(module);
    EXPECT_TRUE(result1.success);

    // Continue tuning from previous result
    TuneResult result2 = scheduler.continueTuning(module, result1.schedule);
    EXPECT_TRUE(result2.success);

    // Cost should not get worse
    EXPECT_LE(result2.best_cost, result1.best_cost * 1.1f);
}

TEST_F(AutoSchedulerTest, WarmStart) {
    AutoScheduler scheduler;
    auto module = createComplexIR();

    // Create a baseline schedule
    Schedule baseline;
    Var i("i"), outer, inner;
    baseline.split(i, 32, outer, inner);
    baseline.vectorize(inner, 8);

    TuneResult result = scheduler.tuneWithWarmStart(module, {baseline});

    EXPECT_TRUE(result.success);
}

// =============================================================================
// Schedule Application Tests
// =============================================================================

TEST_F(AutoSchedulerTest, ApplySchedule) {
    AutoScheduler scheduler;
    auto module = createComplexIR();

    TuneResult result = scheduler.tune(module);
    EXPECT_TRUE(result.success);

    // Apply schedule to module
    bool success = scheduler.applySchedule(module, result.schedule);
    EXPECT_TRUE(success);
}

TEST_F(AutoSchedulerTest, ApplySchedule_PreservesSemantics) {
    AutoScheduler scheduler;
    auto module = createSimpleIR();

    TuneResult result = scheduler.tune(module);
    bool success = scheduler.applySchedule(module, result.schedule);

    // The optimized IR should still be valid
    // (Full semantic verification would require execution)
    EXPECT_TRUE(success);
}

// =============================================================================
// Tuning Log Tests
// =============================================================================

TEST_F(AutoSchedulerTest, ExportLog) {
    AutoScheduler scheduler;
    auto module = createComplexIR();

    scheduler.tune(module);

    std::string path = "/tmp/bud_tuning_log_test.json";
    EXPECT_TRUE(scheduler.exportLog(path));

    // File should exist
    EXPECT_TRUE(std::filesystem::exists(path));

    // Cleanup
    std::filesystem::remove(path);
}

TEST_F(AutoSchedulerTest, ImportLog) {
    AutoScheduler scheduler1;
    auto module = createComplexIR();

    scheduler1.tune(module);

    std::string path = "/tmp/bud_tuning_log_test.json";
    scheduler1.exportLog(path);

    // Create new scheduler and import log
    AutoScheduler scheduler2;
    EXPECT_TRUE(scheduler2.importLog(path));

    // Should have same number of tuning records
    EXPECT_EQ(scheduler2.numTuningRecords(), scheduler1.numTuningRecords());

    std::filesystem::remove(path);
}

TEST_F(AutoSchedulerTest, LogPersistence) {
    std::string path = "/tmp/bud_tuning_log_test.json";

    // First session
    {
        AutoScheduler scheduler;
        auto module = createComplexIR();
        scheduler.tune(module);
        scheduler.exportLog(path);
    }

    // Second session
    {
        AutoScheduler scheduler;
        EXPECT_TRUE(scheduler.importLog(path));
        EXPECT_GT(scheduler.numTuningRecords(), 0);
    }

    std::filesystem::remove(path);
}

// =============================================================================
// Multi-Module Tests
// =============================================================================

TEST_F(AutoSchedulerTest, TuneMultipleModules) {
    AutoScheduler scheduler;

    auto module1 = createSimpleIR();
    auto module2 = createComplexIR();

    TuneResult result1 = scheduler.tune(module1);
    TuneResult result2 = scheduler.tune(module2);

    EXPECT_TRUE(result1.success);
    EXPECT_TRUE(result2.success);
}

TEST_F(AutoSchedulerTest, TuneBatch) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;

    AutoScheduler scheduler(config);

    std::vector<ir::IRModule> modules;
    modules.push_back(createSimpleIR());
    modules.push_back(createComplexIR());
    modules.push_back(createFusableIR());

    auto results = scheduler.tuneBatch(modules);

    EXPECT_EQ(results.size(), modules.size());
    for (const auto& result : results) {
        EXPECT_TRUE(result.success);
    }
}

// =============================================================================
// Callback Tests
// =============================================================================

TEST_F(AutoSchedulerTest, ProgressCallback) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 3;

    AutoScheduler scheduler(config);

    int callback_count = 0;
    scheduler.setProgressCallback(
        [&callback_count](const TuningProgress& progress) { callback_count++; });

    auto module = createComplexIR();
    scheduler.tune(module);

    EXPECT_GT(callback_count, 0);
}

TEST_F(AutoSchedulerTest, FoundBetterCallback) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 5;

    AutoScheduler scheduler(config);

    int improvement_count = 0;
    scheduler.setFoundBetterCallback(
        [&improvement_count](const Schedule& schedule, float cost) { improvement_count++; });

    auto module = createComplexIR();
    scheduler.tune(module);

    // Should find at least one improvement (the initial best)
    EXPECT_GE(improvement_count, 1);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(AutoSchedulerTest, Statistics) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 3;

    AutoScheduler scheduler(config);
    auto module = createComplexIR();

    scheduler.tune(module);

    auto stats = scheduler.statistics();
    EXPECT_GE(stats.total_schedules_evaluated, 1);
    EXPECT_GE(stats.tuning_time_seconds, 0.0f);
}

TEST_F(AutoSchedulerTest, BestScheduleForModule) {
    AutoScheduler scheduler;
    auto module = createComplexIR();

    scheduler.tune(module);

    auto best = scheduler.getBestSchedule("complex");
    EXPECT_TRUE(best.has_value());
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(AutoSchedulerTest, SerializeDeserialize) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 2;

    AutoScheduler scheduler(config);
    auto module = createComplexIR();
    scheduler.tune(module);

    std::string serialized = scheduler.serialize();
    EXPECT_FALSE(serialized.empty());

    AutoScheduler restored;
    EXPECT_TRUE(restored.deserialize(serialized));
}

// =============================================================================
// Benchmark Tests
// =============================================================================

TEST_F(AutoSchedulerTest, BenchmarkTuning) {
    TuningConfig config;
    config.population_size = 16;
    config.num_generations = 5;

    AutoScheduler scheduler(config);
    auto module = createComplexIR();

    auto start = std::chrono::high_resolution_clock::now();

    TuneResult result = scheduler.tune(module);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Tuning time (" << config.num_generations << " generations): " << duration_ms
              << " ms\n";
    std::cout << "Best cost: " << result.best_cost << "\n";
    std::cout << "Schedules evaluated: " << result.schedules_evaluated << "\n";

    EXPECT_TRUE(result.success);
}

TEST_F(AutoSchedulerTest, BenchmarkApplySchedule) {
    AutoScheduler scheduler;
    auto module = createComplexIR();

    TuneResult result = scheduler.tune(module);

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        bool success = scheduler.applySchedule(module, result.schedule);
        (void)success;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_apply = static_cast<double>(duration_us) / kIterations;

    std::cout << "Apply schedule time: " << us_per_apply << " us/apply\n";

    EXPECT_LT(us_per_apply, 100.0) << "Apply schedule too slow";
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
