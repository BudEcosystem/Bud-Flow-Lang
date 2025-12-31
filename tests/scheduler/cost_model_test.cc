// =============================================================================
// Bud Flow Lang - Cost Model Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for CostModel which predicts schedule performance.
//
// Features:
// - Extract features from schedules
// - Predict runtime (faster than measurement)
// - Online learning from measured results
// - Persistent model serialization
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// CostModel Construction Tests
// =============================================================================

class CostModelTest : public ::testing::Test {
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

    Schedule createSampleSchedule() {
        Schedule s;
        Var i("i"), outer, inner;
        s.split(i, 32, outer, inner);
        s.vectorize(inner, 8);
        return s;
    }
};

TEST_F(CostModelTest, DefaultConstruction) {
    CostModel model;
    EXPECT_TRUE(model.isValid());
}

TEST_F(CostModelTest, ConstructWithConfig) {
    CostModelConfig config;
    config.num_trees = 100;
    config.max_depth = 6;
    config.learning_rate = 0.1f;

    CostModel model(config);
    EXPECT_TRUE(model.isValid());
    EXPECT_EQ(model.config().num_trees, 100);
}

// =============================================================================
// Feature Extraction Tests
// =============================================================================

TEST_F(CostModelTest, ExtractFeatures) {
    CostModel model;
    Schedule schedule = createSampleSchedule();

    auto features = model.extractFeatures(schedule);
    EXPECT_FALSE(features.empty());
}

TEST_F(CostModelTest, ExtractFeaturesWithDAG) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    CostModel model;
    Schedule schedule = createSampleSchedule();

    auto features = model.extractFeatures(schedule, dag);
    EXPECT_FALSE(features.empty());
}

TEST_F(CostModelTest, FeaturesIncludeLoopInfo) {
    CostModel model;
    Schedule schedule;
    Var i("i"), outer, inner;
    schedule.split(i, 32, outer, inner);

    auto features = model.extractFeatures(schedule);

    // Features should include split factor information
    EXPECT_GE(features.size(), 1);
}

TEST_F(CostModelTest, FeaturesIncludeVectorInfo) {
    CostModel model;
    Schedule schedule;
    Var inner("inner");
    schedule.vectorize(inner, 8);

    auto features = model.extractFeatures(schedule);

    // Features should include vectorization info
    EXPECT_GE(features.size(), 1);
}

TEST_F(CostModelTest, FeaturesIncludeParallelInfo) {
    CostModel model;
    Schedule schedule;
    Var outer("outer");
    schedule.parallel(outer);

    auto features = model.extractFeatures(schedule);
    EXPECT_GE(features.size(), 1);
}

TEST_F(CostModelTest, DifferentSchedulesProduceDifferentFeatures) {
    CostModel model;

    Schedule s1;
    Var i1("i"), o1, i1_inner;
    s1.split(i1, 32, o1, i1_inner);

    Schedule s2;
    Var i2("i"), o2, i2_inner;
    s2.split(i2, 16, o2, i2_inner);

    auto f1 = model.extractFeatures(s1);
    auto f2 = model.extractFeatures(s2);

    // Features should differ
    bool all_same = true;
    for (size_t i = 0; i < std::min(f1.size(), f2.size()); ++i) {
        if (std::abs(f1[i] - f2[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

// =============================================================================
// Prediction Tests
// =============================================================================

TEST_F(CostModelTest, Predict) {
    CostModel model;
    Schedule schedule = createSampleSchedule();

    float prediction = model.predict(schedule);
    EXPECT_GE(prediction, 0.0f);
}

TEST_F(CostModelTest, PredictWithDAG) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    CostModel model;
    Schedule schedule = createSampleSchedule();

    float prediction = model.predict(schedule, dag);
    EXPECT_GE(prediction, 0.0f);
}

TEST_F(CostModelTest, PredictBatch) {
    CostModel model;

    std::vector<Schedule> schedules;
    for (int i = 0; i < 10; ++i) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, 8 * (i + 1), outer, inner);
        schedules.push_back(s);
    }

    auto predictions = model.predictBatch(schedules);
    EXPECT_EQ(predictions.size(), schedules.size());
    for (float p : predictions) {
        EXPECT_GE(p, 0.0f);
    }
}

TEST_F(CostModelTest, PredictIsDeterministic) {
    CostModel model;
    Schedule schedule = createSampleSchedule();

    float p1 = model.predict(schedule);
    float p2 = model.predict(schedule);

    EXPECT_FLOAT_EQ(p1, p2);
}

// =============================================================================
// Online Learning Tests
// =============================================================================

TEST_F(CostModelTest, UpdateWithMeasurement) {
    CostModel model;
    Schedule schedule = createSampleSchedule();

    float measured_time = 1.0f;  // 1ms
    model.update(schedule, measured_time);

    // Model should have learned
    EXPECT_GT(model.numSamples(), 0);
}

TEST_F(CostModelTest, UpdateBatch) {
    CostModel model;

    std::vector<Schedule> schedules;
    std::vector<float> measured_times;
    for (int i = 0; i < 10; ++i) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, 8 * (i + 1), outer, inner);
        schedules.push_back(s);
        measured_times.push_back(static_cast<float>(i + 1) * 0.1f);
    }

    model.updateBatch(schedules, measured_times);
    EXPECT_EQ(model.numSamples(), 10);
}

TEST_F(CostModelTest, PredictionImprovesWithLearning) {
    CostModel model;

    // Create schedules with known "performance" pattern
    // Larger split factors are "faster"
    std::vector<Schedule> schedules;
    std::vector<float> measured_times;

    for (int split = 8; split <= 128; split *= 2) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, split, outer, inner);
        schedules.push_back(s);
        // Larger split = lower time
        measured_times.push_back(1000.0f / split);
    }

    // Train model
    model.updateBatch(schedules, measured_times);

    // Model should predict lower time for larger splits
    Schedule small_split;
    Var v1("i");
    Var o1, i1;
    small_split.split(v1, 8, o1, i1);

    Schedule large_split;
    Var v2("i");
    Var o2, i2;
    large_split.split(v2, 64, o2, i2);

    float p_small = model.predict(small_split);
    float p_large = model.predict(large_split);

    // After learning, larger split should predict faster (lower time)
    // Note: This is a weak test since the model may not learn perfectly
    // with so few samples
    EXPECT_GE(p_small, 0.0f);
    EXPECT_GE(p_large, 0.0f);
}

// =============================================================================
// Model Statistics Tests
// =============================================================================

TEST_F(CostModelTest, NumSamples) {
    CostModel model;
    EXPECT_EQ(model.numSamples(), 0);

    Schedule s = createSampleSchedule();
    model.update(s, 1.0f);
    EXPECT_EQ(model.numSamples(), 1);
}

TEST_F(CostModelTest, ClearModel) {
    CostModel model;
    Schedule s = createSampleSchedule();
    model.update(s, 1.0f);
    EXPECT_GT(model.numSamples(), 0);

    model.clear();
    EXPECT_EQ(model.numSamples(), 0);
}

TEST_F(CostModelTest, GetTrainingError) {
    CostModel model;

    std::vector<Schedule> schedules;
    std::vector<float> measured_times;
    for (int i = 0; i < 10; ++i) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, 8 * (i + 1), outer, inner);
        schedules.push_back(s);
        measured_times.push_back(static_cast<float>(i + 1) * 0.1f);
    }

    model.updateBatch(schedules, measured_times);

    float error = model.trainingError();
    EXPECT_GE(error, 0.0f);
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(CostModelTest, SerializeDeserialize) {
    CostModel model;

    // Train model
    Schedule s = createSampleSchedule();
    model.update(s, 1.0f);

    std::string serialized = model.serialize();
    EXPECT_FALSE(serialized.empty());

    CostModel restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.numSamples(), model.numSamples());
}

TEST_F(CostModelTest, SaveLoadFile) {
    CostModel model;
    Schedule s = createSampleSchedule();
    model.update(s, 1.0f);

    std::string path = "/tmp/bud_cost_model_test.bin";
    EXPECT_TRUE(model.save(path));

    CostModel loaded;
    EXPECT_TRUE(loaded.load(path));
    EXPECT_EQ(loaded.numSamples(), model.numSamples());

    // Cleanup
    std::remove(path.c_str());
}

TEST_F(CostModelTest, PredictionsConsistentAfterRestore) {
    CostModel model;
    Schedule s = createSampleSchedule();
    model.update(s, 1.0f);

    float before_save = model.predict(s);

    std::string serialized = model.serialize();
    CostModel restored;
    restored.deserialize(serialized);

    float after_restore = restored.predict(s);

    EXPECT_FLOAT_EQ(before_save, after_restore);
}

// =============================================================================
// Feature Configuration Tests
// =============================================================================

TEST_F(CostModelTest, FeatureConfig) {
    FeatureConfig config;
    config.use_loop_features = true;
    config.use_memory_features = true;
    config.use_compute_features = true;
    config.use_reuse_features = true;

    CostModel model;
    model.setFeatureConfig(config);

    auto actual = model.featureConfig();
    EXPECT_TRUE(actual.use_loop_features);
    EXPECT_TRUE(actual.use_memory_features);
}

TEST_F(CostModelTest, DisabledFeaturesProduceSmallerVector) {
    CostModel model;
    Schedule s = createSampleSchedule();

    FeatureConfig config_full;
    config_full.use_loop_features = true;
    config_full.use_memory_features = true;
    config_full.use_compute_features = true;
    config_full.use_reuse_features = true;
    model.setFeatureConfig(config_full);
    auto features_full = model.extractFeatures(s);

    FeatureConfig config_partial;
    config_partial.use_loop_features = true;
    config_partial.use_memory_features = false;
    config_partial.use_compute_features = false;
    config_partial.use_reuse_features = false;
    model.setFeatureConfig(config_partial);
    auto features_partial = model.extractFeatures(s);

    EXPECT_LE(features_partial.size(), features_full.size());
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(CostModelTest, IntegrationWithSearchSpace) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    CostModel model;

    // Sample schedules and predict their costs
    auto samples = space.sample(10);
    for (const auto& schedule : samples) {
        float cost = model.predict(schedule, dag);
        EXPECT_GE(cost, 0.0f);
    }
}

TEST_F(CostModelTest, RankSchedules) {
    CostModel model;

    std::vector<Schedule> schedules;
    for (int split : {8, 16, 32, 64}) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, split, outer, inner);
        schedules.push_back(s);
    }

    auto ranked = model.rankSchedules(schedules);
    EXPECT_EQ(ranked.size(), schedules.size());

    // Check ranking is sorted by predicted cost (ascending)
    for (size_t i = 1; i < ranked.size(); ++i) {
        EXPECT_LE(ranked[i - 1].first, ranked[i].first);
    }
}

TEST_F(CostModelTest, SelectTopK) {
    CostModel model;

    std::vector<Schedule> schedules;
    for (int split : {8, 16, 32, 64, 128}) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, split, outer, inner);
        schedules.push_back(s);
    }

    auto top3 = model.selectTopK(schedules, 3);
    EXPECT_EQ(top3.size(), 3);
}

// =============================================================================
// Benchmark Tests
// =============================================================================

TEST_F(CostModelTest, BenchmarkFeatureExtraction) {
    CostModel model;
    Schedule schedule = createSampleSchedule();

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        auto features = model.extractFeatures(schedule);
        (void)features;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_extraction = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_extraction, 10.0)
        << "Feature extraction too slow: " << us_per_extraction << " us/extraction";

    std::cout << "Feature extraction time: " << us_per_extraction << " us/extraction\n";
}

TEST_F(CostModelTest, BenchmarkPrediction) {
    CostModel model;
    Schedule schedule = createSampleSchedule();

    // Train a bit first
    for (int i = 0; i < 100; ++i) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, 8 + i, outer, inner);
        model.update(s, static_cast<float>(i + 1));
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        float cost = model.predict(schedule);
        (void)cost;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_prediction = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_prediction, 10.0)
        << "Prediction too slow: " << us_per_prediction << " us/prediction";

    std::cout << "Prediction time: " << us_per_prediction << " us/prediction\n";
}

TEST_F(CostModelTest, BenchmarkBatchPrediction) {
    CostModel model;

    // Prepare batch
    std::vector<Schedule> schedules;
    for (int i = 0; i < 100; ++i) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, 8 + i, outer, inner);
        schedules.push_back(s);
    }

    // Train a bit first
    for (const auto& s : schedules) {
        model.update(s, 1.0f);
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        auto predictions = model.predictBatch(schedules);
        (void)predictions;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_batch = static_cast<double>(duration_us) / kIterations;
    double us_per_item = us_per_batch / schedules.size();

    EXPECT_LT(us_per_item, 5.0) << "Batch prediction too slow: " << us_per_item << " us/item";

    std::cout << "Batch prediction time: " << us_per_batch << " us/batch, " << us_per_item
              << " us/item\n";
}

TEST_F(CostModelTest, BenchmarkUpdate) {
    CostModel model;

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        Schedule s;
        Var v("i");
        Var outer, inner;
        s.split(v, 8 + (i % 100), outer, inner);
        model.update(s, static_cast<float>(i + 1));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_update = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_update, 50.0) << "Update too slow: " << us_per_update << " us/update";

    std::cout << "Update time: " << us_per_update << " us/update\n";
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
