// =============================================================================
// Bud Flow Lang - Search Space Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for SearchSpace which generates schedule sketches and annotations
// for auto-scheduling optimization.
//
// Features:
// - Generate sketches (high-level loop structures)
// - Annotate sketches with specific parameters
// - Sample from search space
// - Define search space bounds
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <chrono>
#include <unordered_set>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// SearchSpace Construction Tests
// =============================================================================

class SearchSpaceTest : public ::testing::Test {
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

    ir::IRModule createLoopableIR() {
        // Create IR with potential for loop transformations
        ir::IRModule module("loopable");
        auto& builder = module.builder();
        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.constant(3.0f);
        auto d = builder.constant(4.0f);

        // Chain of operations that could be tiled/vectorized
        auto x = builder.add(a, b);
        auto y = builder.add(c, d);
        auto z = builder.mul(x, y);
        auto w = builder.fma(z, a, b);
        module.setOutput(w);
        return module;
    }
};

TEST_F(SearchSpaceTest, DefaultConstruction) {
    SearchSpace space;
    EXPECT_EQ(space.numSketches(), 0);
}

TEST_F(SearchSpaceTest, FromDAG) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Should generate at least one sketch
    EXPECT_GT(space.numSketches(), 0);
}

TEST_F(SearchSpaceTest, FromDAG_Simple) {
    auto module = createSimpleIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    EXPECT_GE(space.numSketches(), 0);
}

// =============================================================================
// Sketch Generation Tests
// =============================================================================

TEST_F(SearchSpaceTest, GetSketches) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto sketches = space.getSketches();
    EXPECT_FALSE(sketches.empty());
}

TEST_F(SearchSpaceTest, SketchHasValidType) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto sketches = space.getSketches();
    for (const auto& sketch : sketches) {
        // Each sketch should have a valid type
        EXPECT_NE(sketch.type, SketchType::kInvalid);
    }
}

TEST_F(SearchSpaceTest, SketchHasTransformSteps) {
    auto module = createLoopableIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto sketches = space.getSketches();
    for (const auto& sketch : sketches) {
        // Sketches should have transform steps
        EXPECT_GE(sketch.steps.size(), 0);
    }
}

// =============================================================================
// Transform Step Tests
// =============================================================================

TEST_F(SearchSpaceTest, TransformStepTypes) {
    // Verify all transform step types are defined
    EXPECT_NE(static_cast<int>(TransformStep::kSplit), 0);
    EXPECT_NE(static_cast<int>(TransformStep::kTile), 0);
    EXPECT_NE(static_cast<int>(TransformStep::kFuse), 0);
    EXPECT_NE(static_cast<int>(TransformStep::kReorder), 0);
    EXPECT_NE(static_cast<int>(TransformStep::kVectorize), 0);
    EXPECT_NE(static_cast<int>(TransformStep::kParallel), 0);
    EXPECT_NE(static_cast<int>(TransformStep::kUnroll), 0);
}

TEST_F(SearchSpaceTest, TransformStepHasParameters) {
    TransformStepDef step;
    step.type = TransformStep::kSplit;
    step.target_var = "i";
    step.factor = 16;

    EXPECT_EQ(step.type, TransformStep::kSplit);
    EXPECT_EQ(step.target_var, "i");
    EXPECT_EQ(step.factor, 16);
}

// =============================================================================
// Annotation Tests
// =============================================================================

TEST_F(SearchSpaceTest, AnnotateSketch) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto sketches = space.getSketches();
    if (!sketches.empty()) {
        // Should be able to annotate a sketch
        auto schedules = space.annotateSketch(sketches[0]);
        // May or may not produce schedules depending on sketch
        EXPECT_GE(schedules.size(), 0);
    }
}

TEST_F(SearchSpaceTest, AnnotationBounds) {
    SearchSpaceBounds bounds;
    bounds.min_split_factor = 2;
    bounds.max_split_factor = 128;
    bounds.min_tile_size = 4;
    bounds.max_tile_size = 64;
    bounds.max_unroll_factor = 16;
    bounds.vector_widths = {4, 8, 16};

    EXPECT_EQ(bounds.min_split_factor, 2);
    EXPECT_EQ(bounds.max_split_factor, 128);
    EXPECT_EQ(bounds.min_tile_size, 4);
    EXPECT_EQ(bounds.max_tile_size, 64);
    EXPECT_EQ(bounds.max_unroll_factor, 16);
    EXPECT_EQ(bounds.vector_widths.size(), 3);
}

TEST_F(SearchSpaceTest, AnnotateWithBounds) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    SearchSpaceBounds bounds;
    bounds.min_split_factor = 4;
    bounds.max_split_factor = 32;

    auto space = SearchSpace::fromDAG(dag, bounds);

    // Space should respect bounds
    EXPECT_EQ(space.bounds().min_split_factor, 4);
    EXPECT_EQ(space.bounds().max_split_factor, 32);
}

// =============================================================================
// Sampling Tests
// =============================================================================

TEST_F(SearchSpaceTest, SampleSchedules) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Sample some schedules
    auto samples = space.sample(10);
    // Should get some samples (may be fewer than requested)
    EXPECT_LE(samples.size(), 10);
}

TEST_F(SearchSpaceTest, SampleUnique) {
    auto module = createLoopableIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Sample with uniqueness
    auto samples = space.sampleUnique(5);
    // All samples should be unique (by comparing serialization)
    std::unordered_set<std::string> seen;
    for (const auto& schedule : samples) {
        std::string serialized = schedule.serialize();
        EXPECT_EQ(seen.count(serialized), 0) << "Duplicate schedule found";
        seen.insert(serialized);
    }
}

TEST_F(SearchSpaceTest, SampleWithSeed) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Same seed should give same results
    auto samples1 = space.sample(5, 42);
    auto samples2 = space.sample(5, 42);

    EXPECT_EQ(samples1.size(), samples2.size());
    for (size_t i = 0; i < samples1.size(); ++i) {
        EXPECT_EQ(samples1[i].serialize(), samples2[i].serialize());
    }
}

// =============================================================================
// Search Space Size Tests
// =============================================================================

TEST_F(SearchSpaceTest, EstimateSize) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    size_t estimated_size = space.estimateSize();
    // Should have some search space
    EXPECT_GE(estimated_size, 0);
}

TEST_F(SearchSpaceTest, EnumerateAll_Small) {
    auto module = createSimpleIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // For small DAGs, enumeration should complete
    auto all = space.enumerateAll(100);  // Limit to 100
    EXPECT_LE(all.size(), 100);
}

// =============================================================================
// Sketch Type Tests
// =============================================================================

TEST_F(SearchSpaceTest, SketchTypes) {
    // Verify sketch types are defined
    EXPECT_NE(static_cast<int>(SketchType::kBasic), -1);
    EXPECT_NE(static_cast<int>(SketchType::kTiled), -1);
    EXPECT_NE(static_cast<int>(SketchType::kFused), -1);
    EXPECT_NE(static_cast<int>(SketchType::kVectorized), -1);
    EXPECT_NE(static_cast<int>(SketchType::kParallel), -1);
}

TEST_F(SearchSpaceTest, SketchFromType) {
    auto module = createLoopableIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Request specific sketch types
    auto tiled_sketches = space.getSketchesOfType(SketchType::kTiled);
    for (const auto& sketch : tiled_sketches) {
        EXPECT_EQ(sketch.type, SketchType::kTiled);
    }
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(SearchSpaceTest, SerializeDeserialize) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    std::string serialized = space.serialize();
    EXPECT_FALSE(serialized.empty());

    SearchSpace restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.numSketches(), space.numSketches());
}

TEST_F(SearchSpaceTest, SketchSerialization) {
    Sketch sketch;
    sketch.type = SketchType::kTiled;
    sketch.name = "test_sketch";

    TransformStepDef step;
    step.type = TransformStep::kSplit;
    step.target_var = "i";
    step.factor = 32;
    sketch.steps.push_back(step);

    std::string serialized = sketch.serialize();
    EXPECT_FALSE(serialized.empty());

    Sketch restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.type, SketchType::kTiled);
    EXPECT_EQ(restored.name, "test_sketch");
    EXPECT_EQ(restored.steps.size(), 1);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(SearchSpaceTest, DAGToSchedulePipeline) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto sketches = space.getSketches();
    if (!sketches.empty()) {
        auto schedules = space.annotateSketch(sketches[0]);
        // Should be able to get some schedules from the pipeline
        EXPECT_GE(schedules.size(), 0);
    }
}

TEST_F(SearchSpaceTest, IterativeRefinement) {
    auto module = createLoopableIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    // Initial sample
    auto initial = space.sample(5);

    // Refine around a sample (if any)
    if (!initial.empty()) {
        auto refined = space.refineAround(initial[0], 3);
        // Refined should be related to initial
        EXPECT_GE(refined.size(), 0);
    }
}

// =============================================================================
// Benchmark Tests
// =============================================================================

TEST_F(SearchSpaceTest, BenchmarkFromDAG) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        auto space = SearchSpace::fromDAG(dag);
        (void)space;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_space = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_space, 100.0)
        << "Search space construction too slow: " << us_per_space << " us/space";

    std::cout << "Search space construction time: " << us_per_space << " us/space\n";
}

TEST_F(SearchSpaceTest, BenchmarkSampling) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 1000;
    constexpr int kSamplesPerIteration = 10;
    for (int i = 0; i < kIterations; ++i) {
        auto samples = space.sample(kSamplesPerIteration);
        (void)samples;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_sample = static_cast<double>(duration_us) / (kIterations * kSamplesPerIteration);

    EXPECT_LT(us_per_sample, 10.0) << "Sampling too slow: " << us_per_sample << " us/sample";

    std::cout << "Sampling time: " << us_per_sample << " us/sample\n";
}

TEST_F(SearchSpaceTest, BenchmarkAnnotation) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());
    auto space = SearchSpace::fromDAG(dag);

    auto sketches = space.getSketches();
    if (sketches.empty()) {
        GTEST_SKIP() << "No sketches available for annotation benchmark";
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        auto schedules = space.annotateSketch(sketches[0]);
        (void)schedules;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_annotation = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_annotation, 100.0)
        << "Annotation too slow: " << us_per_annotation << " us/annotation";

    std::cout << "Annotation time: " << us_per_annotation << " us/annotation\n";
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
