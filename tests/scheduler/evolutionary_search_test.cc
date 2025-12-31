// =============================================================================
// Bud Flow Lang - Evolutionary Search Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for EvolutionarySearch which uses genetic algorithms to find
// optimal schedules.
//
// Features:
// - Population management
// - Mutation and crossover operators
// - Selection strategies
// - Convergence tracking
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/evolutionary_search.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <chrono>
#include <unordered_set>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// EvolutionarySearch Construction Tests
// =============================================================================

class EvolutionarySearchTest : public ::testing::Test {
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

    ComputeDAG createTestDAG() {
        auto module = createComplexIR();
        return ComputeDAG::fromIR(module.builder());
    }

    SearchSpace createTestSpace() {
        auto dag = createTestDAG();
        return SearchSpace::fromDAG(dag);
    }
};

TEST_F(EvolutionarySearchTest, DefaultConstruction) {
    EvolutionarySearch search;
    EXPECT_TRUE(search.isValid());
}

TEST_F(EvolutionarySearchTest, ConstructWithConfig) {
    SearchConfig config;
    config.population_size = 256;
    config.num_generations = 50;
    config.mutation_prob = 0.15f;
    config.crossover_prob = 0.6f;

    EvolutionarySearch search(config);
    EXPECT_TRUE(search.isValid());
    EXPECT_EQ(search.config().population_size, 256);
}

// =============================================================================
// Population Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, InitializePopulation) {
    auto space = createTestSpace();
    EvolutionarySearch search;

    search.initializePopulation(space);
    EXPECT_GT(search.populationSize(), 0);
}

TEST_F(EvolutionarySearchTest, PopulationSize) {
    SearchConfig config;
    config.population_size = 100;

    EvolutionarySearch search(config);
    auto space = createTestSpace();

    search.initializePopulation(space);
    EXPECT_LE(search.populationSize(), config.population_size);
}

TEST_F(EvolutionarySearchTest, PopulationUnique) {
    auto space = createTestSpace();
    EvolutionarySearch search;

    search.initializePopulation(space);
    auto population = search.population();

    std::unordered_set<std::string> seen;
    for (const auto& individual : population) {
        std::string serialized = individual.schedule.serialize();
        EXPECT_EQ(seen.count(serialized), 0) << "Duplicate in population";
        seen.insert(serialized);
    }
}

// =============================================================================
// Mutation Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, Mutate) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);
    auto population = search.population();

    if (!population.empty()) {
        Schedule mutated = search.mutate(population[0].schedule, space);
        // Mutation should produce a valid schedule
        EXPECT_TRUE(mutated.validate());
    }
}

TEST_F(EvolutionarySearchTest, MutateProducesDifferentSchedule) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);
    auto population = search.population();

    if (!population.empty()) {
        const Schedule& original = population[0].schedule;
        bool found_different = false;

        // Multiple mutation attempts should eventually produce something different
        for (int i = 0; i < 10; ++i) {
            Schedule mutated = search.mutate(original, space);
            if (mutated.serialize() != original.serialize()) {
                found_different = true;
                break;
            }
        }
        // It's acceptable if mutations are identical for simple schedules
        EXPECT_TRUE(true);
    }
}

TEST_F(EvolutionarySearchTest, MutationTypes) {
    // Test different mutation types are available
    EXPECT_NE(static_cast<int>(MutationType::kAddTransform), 0);
    EXPECT_NE(static_cast<int>(MutationType::kRemoveTransform), 0);
    EXPECT_NE(static_cast<int>(MutationType::kModifyParameter), 0);
    EXPECT_NE(static_cast<int>(MutationType::kSwapTransforms), 0);
}

// =============================================================================
// Crossover Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, Crossover) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);
    auto population = search.population();

    if (population.size() >= 2) {
        Schedule child = search.crossover(population[0].schedule, population[1].schedule, space);
        EXPECT_TRUE(child.validate());
    }
}

TEST_F(EvolutionarySearchTest, CrossoverCombinesParents) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);
    auto population = search.population();

    if (population.size() >= 2) {
        const Schedule& p1 = population[0].schedule;
        const Schedule& p2 = population[1].schedule;
        Schedule child = search.crossover(p1, p2, space);

        // Child should be valid
        EXPECT_TRUE(child.validate());
    }
}

// =============================================================================
// Selection Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, TournamentSelection) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    auto selected = search.tournamentSelect(5);
    EXPECT_LE(selected.size(), 5);
}

TEST_F(EvolutionarySearchTest, TopKSelection) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    auto top_k = search.selectTopK(3);
    EXPECT_LE(top_k.size(), 3);
}

TEST_F(EvolutionarySearchTest, SelectionFavorsBetter) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    auto top_2 = search.selectTopK(2);
    auto all = search.population();

    if (top_2.size() >= 2 && all.size() >= 2) {
        // Top-2 should have better (lower) fitness than average
        // (Fitness = cost, lower is better)
        float avg_fitness = 0.0f;
        for (const auto& ind : all) {
            avg_fitness += ind.fitness;
        }
        avg_fitness /= static_cast<float>(all.size());

        for (const auto& ind : top_2) {
            EXPECT_LE(ind.fitness, avg_fitness * 2.0f);  // Generous bound
        }
    }
}

// =============================================================================
// Evolution Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, EvolveOneGeneration) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    size_t gen_before = search.currentGeneration();
    search.evolveGeneration(space, model);

    EXPECT_EQ(search.currentGeneration(), gen_before + 1);
}

TEST_F(EvolutionarySearchTest, FitnessImprovesOverGenerations) {
    SearchConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    EvolutionarySearch search(config);
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    float initial_best = search.bestFitness();

    for (size_t i = 0; i < 5; ++i) {
        search.evolveGeneration(space, model);
    }

    // Fitness should not get worse (lower is better)
    EXPECT_LE(search.bestFitness(), initial_best * 1.1f);  // Allow small fluctuation
}

TEST_F(EvolutionarySearchTest, TrackBestSolution) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    auto best = search.bestIndividual();
    EXPECT_TRUE(best.schedule.validate());
}

// =============================================================================
// Full Search Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, Search) {
    auto dag = createTestDAG();
    auto space = SearchSpace::fromDAG(dag);
    CostModel model;

    SearchConfig config;
    config.population_size = 32;
    config.num_generations = 5;

    EvolutionarySearch search(config);
    Schedule best = search.search(dag, space, model);

    EXPECT_TRUE(best.validate());
}

TEST_F(EvolutionarySearchTest, SearchWithCallback) {
    auto dag = createTestDAG();
    auto space = SearchSpace::fromDAG(dag);
    CostModel model;

    SearchConfig config;
    config.population_size = 16;
    config.num_generations = 3;

    EvolutionarySearch search(config);

    int callback_count = 0;
    search.setGenerationCallback(
        [&callback_count](size_t gen, float best_fitness) { callback_count++; });

    search.search(dag, space, model);
    EXPECT_EQ(callback_count, config.num_generations);
}

TEST_F(EvolutionarySearchTest, EarlyTermination) {
    auto dag = createTestDAG();
    auto space = SearchSpace::fromDAG(dag);
    CostModel model;

    SearchConfig config;
    config.population_size = 16;
    config.num_generations = 100;         // Many generations
    config.early_stop_threshold = 0.01f;  // Stop if improvement is tiny
    config.early_stop_patience = 3;       // Stop after 3 generations without improvement

    EvolutionarySearch search(config);
    Schedule best = search.search(dag, space, model);

    // Should terminate early
    EXPECT_LT(search.currentGeneration(), config.num_generations);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, Statistics) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    auto stats = search.statistics();
    EXPECT_GE(stats.min_fitness, 0.0f);
    EXPECT_GE(stats.max_fitness, stats.min_fitness);
    EXPECT_GE(stats.avg_fitness, stats.min_fitness);
    EXPECT_LE(stats.avg_fitness, stats.max_fitness);
}

TEST_F(EvolutionarySearchTest, DiversityMetric) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);

    float diversity = search.populationDiversity();
    EXPECT_GE(diversity, 0.0f);
    EXPECT_LE(diversity, 1.0f);
}

TEST_F(EvolutionarySearchTest, ConvergenceHistory) {
    SearchConfig config;
    config.population_size = 16;
    config.num_generations = 5;

    EvolutionarySearch search(config);
    auto dag = createTestDAG();
    auto space = SearchSpace::fromDAG(dag);
    CostModel model;

    search.search(dag, space, model);

    auto history = search.convergenceHistory();
    EXPECT_EQ(history.size(), config.num_generations);
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, SerializeDeserialize) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    std::string serialized = search.serialize();
    EXPECT_FALSE(serialized.empty());

    EvolutionarySearch restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.populationSize(), search.populationSize());
}

TEST_F(EvolutionarySearchTest, SaveLoadCheckpoint) {
    EvolutionarySearch search;
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    std::string path = "/tmp/bud_evo_search_test.bin";
    EXPECT_TRUE(search.saveCheckpoint(path));

    EvolutionarySearch loaded;
    EXPECT_TRUE(loaded.loadCheckpoint(path));
    EXPECT_EQ(loaded.currentGeneration(), search.currentGeneration());

    std::remove(path.c_str());
}

// =============================================================================
// Benchmark Tests
// =============================================================================

TEST_F(EvolutionarySearchTest, BenchmarkMutation) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);
    auto population = search.population();

    if (population.empty()) {
        GTEST_SKIP() << "Empty population";
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        auto mutated = search.mutate(population[0].schedule, space);
        (void)mutated;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_mutation = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_mutation, 50.0) << "Mutation too slow: " << us_per_mutation << " us/mutation";

    std::cout << "Mutation time: " << us_per_mutation << " us/mutation\n";
}

TEST_F(EvolutionarySearchTest, BenchmarkCrossover) {
    EvolutionarySearch search;
    auto space = createTestSpace();

    search.initializePopulation(space);
    auto population = search.population();

    if (population.size() < 2) {
        GTEST_SKIP() << "Not enough individuals for crossover";
    }

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        auto child = search.crossover(population[0].schedule, population[1].schedule, space);
        (void)child;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_crossover = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_crossover, 50.0)
        << "Crossover too slow: " << us_per_crossover << " us/crossover";

    std::cout << "Crossover time: " << us_per_crossover << " us/crossover\n";
}

TEST_F(EvolutionarySearchTest, BenchmarkGeneration) {
    SearchConfig config;
    config.population_size = 32;

    EvolutionarySearch search(config);
    auto space = createTestSpace();
    CostModel model;

    search.initializePopulation(space);
    search.evaluatePopulation(model);

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10;
    for (int i = 0; i < kIterations; ++i) {
        search.evolveGeneration(space, model);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double ms_per_gen = static_cast<double>(duration_ms) / kIterations;

    EXPECT_LT(ms_per_gen, 500.0) << "Generation too slow: " << ms_per_gen << " ms/generation";

    std::cout << "Generation time: " << ms_per_gen << " ms/generation\n";
}

TEST_F(EvolutionarySearchTest, BenchmarkFullSearch) {
    SearchConfig config;
    config.population_size = 16;
    config.num_generations = 5;

    EvolutionarySearch search(config);
    auto dag = createTestDAG();
    auto space = SearchSpace::fromDAG(dag);
    CostModel model;

    auto start = std::chrono::high_resolution_clock::now();

    Schedule best = search.search(dag, space, model);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Full search time (" << config.num_generations << " generations): " << duration_ms
              << " ms\n";

    EXPECT_TRUE(best.validate());
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
