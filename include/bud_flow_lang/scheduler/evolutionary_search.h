#pragma once

// =============================================================================
// Bud Flow Lang - Evolutionary Search
// =============================================================================
//
// Genetic algorithm-based search for optimal schedules.
// Inspired by TVM's Ansor (OSDI 2020) evolutionary search.
//
// Features:
// - Population-based optimization
// - Mutation and crossover operators
// - Tournament and elitist selection
// - Early termination and convergence tracking
//

#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// Search Configuration
// =============================================================================

/// Configuration for evolutionary search
struct SearchConfig {
    // Population parameters
    size_t population_size = 512;
    size_t num_generations = 20;

    // Genetic operators
    float mutation_prob = 0.1f;
    float crossover_prob = 0.5f;

    // Selection
    size_t tournament_size = 4;
    size_t elite_count = 8;  // Best individuals to preserve

    // Early termination
    float early_stop_threshold = 0.001f;  // Stop if improvement below this
    size_t early_stop_patience = 5;       // Generations without improvement

    // Diversity
    bool maintain_diversity = true;
    float diversity_weight = 0.1f;

    // Random seed (0 = random)
    uint64_t seed = 0;
};

// =============================================================================
// Mutation Types
// =============================================================================

/// Types of mutations for schedules
enum class MutationType : uint8_t {
    kNone = 0,
    kAddTransform = 1,      // Add a new transformation
    kRemoveTransform = 2,   // Remove an existing transformation
    kModifyParameter = 3,   // Change a parameter (tile size, etc.)
    kSwapTransforms = 4,    // Swap order of two transforms
    kReplaceTransform = 5,  // Replace one transform with another
};

// =============================================================================
// Individual
// =============================================================================

/// An individual in the population
struct Individual {
    Schedule schedule;
    float fitness = std::numeric_limits<float>::max();
    bool evaluated = false;

    // For diversity tracking
    size_t hash = 0;
};

// =============================================================================
// Population Statistics
// =============================================================================

/// Statistics about the population
struct PopulationStats {
    float min_fitness = 0.0f;
    float max_fitness = 0.0f;
    float avg_fitness = 0.0f;
    float std_fitness = 0.0f;
    float diversity = 0.0f;
};

// =============================================================================
// EvolutionarySearch
// =============================================================================

/// Genetic algorithm search for optimal schedules
class EvolutionarySearch {
  public:
    using GenerationCallback = std::function<void(size_t generation, float best_fitness)>;

    EvolutionarySearch();
    explicit EvolutionarySearch(const SearchConfig& config);
    ~EvolutionarySearch();

    // Move-only
    EvolutionarySearch(EvolutionarySearch&&) noexcept;
    EvolutionarySearch& operator=(EvolutionarySearch&&) noexcept;
    EvolutionarySearch(const EvolutionarySearch&) = delete;
    EvolutionarySearch& operator=(const EvolutionarySearch&) = delete;

    // -------------------------------------------------------------------------
    // Validity
    // -------------------------------------------------------------------------

    [[nodiscard]] bool isValid() const { return valid_; }
    [[nodiscard]] const SearchConfig& config() const { return config_; }

    // -------------------------------------------------------------------------
    // Population Management
    // -------------------------------------------------------------------------

    /// Initialize population from search space
    void initializePopulation(const SearchSpace& space);

    /// Get current population
    [[nodiscard]] const std::vector<Individual>& population() const { return population_; }

    /// Get population size
    [[nodiscard]] size_t populationSize() const { return population_.size(); }

    /// Evaluate all individuals in population
    void evaluatePopulation(const CostModel& model);

    /// Evaluate population with DAG context
    void evaluatePopulation(const CostModel& model, const ComputeDAG& dag);

    // -------------------------------------------------------------------------
    // Genetic Operators
    // -------------------------------------------------------------------------

    /// Mutate a schedule
    [[nodiscard]] Schedule mutate(const Schedule& schedule, const SearchSpace& space) const;

    /// Crossover two schedules
    [[nodiscard]] Schedule crossover(const Schedule& parent1, const Schedule& parent2,
                                     const SearchSpace& space) const;

    // -------------------------------------------------------------------------
    // Selection
    // -------------------------------------------------------------------------

    /// Tournament selection
    [[nodiscard]] std::vector<Individual> tournamentSelect(size_t count) const;

    /// Select top-K individuals
    [[nodiscard]] std::vector<Individual> selectTopK(size_t k) const;

    // -------------------------------------------------------------------------
    // Evolution
    // -------------------------------------------------------------------------

    /// Evolve one generation
    void evolveGeneration(const SearchSpace& space, const CostModel& model);

    /// Evolve with DAG context
    void evolveGeneration(const SearchSpace& space, const CostModel& model, const ComputeDAG& dag);

    /// Get current generation number
    [[nodiscard]] size_t currentGeneration() const { return current_generation_; }

    /// Get best fitness (lowest cost)
    [[nodiscard]] float bestFitness() const;

    /// Get best individual
    [[nodiscard]] const Individual& bestIndividual() const;

    // -------------------------------------------------------------------------
    // Full Search
    // -------------------------------------------------------------------------

    /// Run full evolutionary search
    [[nodiscard]] Schedule search(const ComputeDAG& dag, const SearchSpace& space,
                                  const CostModel& model);

    /// Set callback for generation completion
    void setGenerationCallback(GenerationCallback callback);

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    /// Get population statistics
    [[nodiscard]] PopulationStats statistics() const;

    /// Get population diversity (0-1)
    [[nodiscard]] float populationDiversity() const;

    /// Get convergence history (best fitness per generation)
    [[nodiscard]] const std::vector<float>& convergenceHistory() const {
        return convergence_history_;
    }

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    [[nodiscard]] std::string serialize() const;
    bool deserialize(const std::string& data);
    bool saveCheckpoint(const std::string& path) const;
    bool loadCheckpoint(const std::string& path);

  private:
    SearchConfig config_;
    bool valid_ = true;

    // Population
    std::vector<Individual> population_;
    size_t current_generation_ = 0;

    // Best solution tracking
    Individual best_ever_;
    std::vector<float> convergence_history_;

    // Callbacks
    GenerationCallback generation_callback_;

    // Random number generator
    mutable std::mt19937_64 rng_;

    // Helper methods
    void sortPopulationByFitness();
    [[nodiscard]] size_t selectParentIndex() const;
    [[nodiscard]] MutationType selectMutationType() const;
    void applyMutation(Schedule& schedule, MutationType type, const SearchSpace& space) const;
    [[nodiscard]] size_t computeScheduleHash(const Schedule& schedule) const;
    void ensureDiversity();
    [[nodiscard]] bool shouldTerminateEarly() const;
};

}  // namespace scheduler
}  // namespace bud
