// =============================================================================
// Bud Flow Lang - Evolutionary Search Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/evolutionary_search.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_set>

namespace bud {
namespace scheduler {

// =============================================================================
// EvolutionarySearch Implementation
// =============================================================================

EvolutionarySearch::EvolutionarySearch() : config_{}, rng_(std::random_device{}()) {}

EvolutionarySearch::EvolutionarySearch(const SearchConfig& config)
    : config_(config), rng_(config.seed == 0 ? std::random_device{}() : config.seed) {}

EvolutionarySearch::~EvolutionarySearch() = default;

EvolutionarySearch::EvolutionarySearch(EvolutionarySearch&&) noexcept = default;
EvolutionarySearch& EvolutionarySearch::operator=(EvolutionarySearch&&) noexcept = default;

// =============================================================================
// Population Management
// =============================================================================

void EvolutionarySearch::initializePopulation(const SearchSpace& space) {
    population_.clear();
    current_generation_ = 0;
    convergence_history_.clear();

    // Sample unique schedules from search space
    auto samples = space.sampleUnique(config_.population_size, rng_());

    for (auto& schedule : samples) {
        Individual ind;
        ind.schedule = std::move(schedule);
        ind.fitness = std::numeric_limits<float>::max();
        ind.evaluated = false;
        ind.hash = computeScheduleHash(ind.schedule);
        population_.push_back(std::move(ind));
    }

    spdlog::debug("Initialized population with {} individuals", population_.size());
}

void EvolutionarySearch::evaluatePopulation(const CostModel& model) {
    for (auto& ind : population_) {
        if (!ind.evaluated) {
            ind.fitness = model.predict(ind.schedule);
            ind.evaluated = true;
        }
    }

    sortPopulationByFitness();

    // Update best ever
    if (!population_.empty() && population_[0].fitness < best_ever_.fitness) {
        best_ever_ = population_[0];
    }
}

void EvolutionarySearch::evaluatePopulation(const CostModel& model, const ComputeDAG& dag) {
    for (auto& ind : population_) {
        if (!ind.evaluated) {
            ind.fitness = model.predict(ind.schedule, dag);
            ind.evaluated = true;
        }
    }

    sortPopulationByFitness();

    if (!population_.empty() && population_[0].fitness < best_ever_.fitness) {
        best_ever_ = population_[0];
    }
}

void EvolutionarySearch::sortPopulationByFitness() {
    std::sort(population_.begin(), population_.end(),
              [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
}

// =============================================================================
// Genetic Operators
// =============================================================================

Schedule EvolutionarySearch::mutate(const Schedule& schedule, const SearchSpace& space) const {
    auto cloned = schedule.clone();
    MutationType type = selectMutationType();
    applyMutation(*cloned, type, space);
    return *cloned;
}

MutationType EvolutionarySearch::selectMutationType() const {
    std::uniform_int_distribution<int> dist(1, 5);
    return static_cast<MutationType>(dist(rng_));
}

void EvolutionarySearch::applyMutation(Schedule& schedule, MutationType type,
                                       const SearchSpace& space) const {

    const auto& bounds = space.bounds();
    std::uniform_int_distribution<size_t> factor_dist(bounds.min_split_factor,
                                                      bounds.max_split_factor);
    std::uniform_int_distribution<size_t> vw_dist(0, bounds.vector_widths.size() - 1);

    switch (type) {
    case MutationType::kAddTransform: {
        // Add a random transformation
        std::uniform_int_distribution<int> transform_type(0, 3);
        int t = transform_type(rng_);

        Var v("mutate_" + std::to_string(rng_()));

        if (t == 0) {
            // Add split
            Var outer, inner;
            schedule.split(v, factor_dist(rng_), outer, inner);
        } else if (t == 1) {
            // Add vectorize
            size_t width = bounds.vector_widths[vw_dist(rng_)];
            schedule.vectorize(v, width);
        } else if (t == 2) {
            // Add parallel
            schedule.parallel(v);
        } else {
            // Add unroll
            std::uniform_int_distribution<size_t> unroll_dist(2, bounds.max_unroll_factor);
            schedule.unroll(v, unroll_dist(rng_));
        }
        break;
    }

    case MutationType::kRemoveTransform: {
        // Remove last transformation (simplified)
        // In a full implementation, we'd modify the transforms list
        break;
    }

    case MutationType::kModifyParameter: {
        // Add a new transformation with different parameter
        Var v("mod_" + std::to_string(rng_()));
        Var outer, inner;
        schedule.split(v, factor_dist(rng_), outer, inner);
        break;
    }

    case MutationType::kSwapTransforms: {
        // In a full implementation, swap order of transforms
        break;
    }

    case MutationType::kReplaceTransform: {
        // Add a different transformation
        Var v("replace_" + std::to_string(rng_()));
        size_t width = bounds.vector_widths[vw_dist(rng_)];
        schedule.vectorize(v, width);
        break;
    }

    default:
        break;
    }
}

Schedule EvolutionarySearch::crossover(const Schedule& parent1, const Schedule& parent2,
                                       const SearchSpace& space) const {

    // Single-point crossover: take some transforms from each parent
    auto child = parent1.clone();

    const auto& p2_transforms = parent2.transforms();
    if (!p2_transforms.empty()) {
        // Add some transforms from parent2
        std::uniform_int_distribution<size_t> count_dist(1, p2_transforms.size());
        size_t num_to_add = count_dist(rng_);

        for (size_t i = 0; i < num_to_add; ++i) {
            std::uniform_int_distribution<size_t> idx_dist(0, p2_transforms.size() - 1);
            const auto& t = p2_transforms[idx_dist(rng_)];

            // Apply transform to child
            Var v(t.vars.empty() ? "cross_" + std::to_string(i) : t.vars[0].name());

            switch (t.type) {
            case TransformType::kSplit: {
                Var outer, inner;
                child->split(v, t.factor, outer, inner);
                break;
            }
            case TransformType::kVectorize:
                child->vectorize(v, t.vector_width);
                break;
            case TransformType::kParallel:
                child->parallel(v);
                break;
            case TransformType::kUnroll:
                child->unroll(v, t.factor);
                break;
            default:
                break;
            }
        }
    }

    return *child;
}

// =============================================================================
// Selection
// =============================================================================

std::vector<Individual> EvolutionarySearch::tournamentSelect(size_t count) const {
    std::vector<Individual> selected;
    selected.reserve(count);

    if (population_.empty()) {
        return selected;
    }

    std::uniform_int_distribution<size_t> dist(0, population_.size() - 1);

    for (size_t i = 0; i < count; ++i) {
        // Tournament: select best from random subset
        size_t best_idx = dist(rng_);
        float best_fitness = population_[best_idx].fitness;

        for (size_t j = 1; j < config_.tournament_size && j < population_.size(); ++j) {
            size_t idx = dist(rng_);
            if (population_[idx].fitness < best_fitness) {
                best_idx = idx;
                best_fitness = population_[idx].fitness;
            }
        }

        selected.push_back(population_[best_idx]);
    }

    return selected;
}

std::vector<Individual> EvolutionarySearch::selectTopK(size_t k) const {
    std::vector<Individual> top_k;
    k = std::min(k, population_.size());
    top_k.reserve(k);

    // Population should already be sorted by fitness
    for (size_t i = 0; i < k; ++i) {
        top_k.push_back(population_[i]);
    }

    return top_k;
}

size_t EvolutionarySearch::selectParentIndex() const {
    // Tournament selection for a single parent
    std::uniform_int_distribution<size_t> dist(0, population_.size() - 1);

    size_t best_idx = dist(rng_);
    float best_fitness = population_[best_idx].fitness;

    for (size_t j = 1; j < config_.tournament_size; ++j) {
        size_t idx = dist(rng_);
        if (population_[idx].fitness < best_fitness) {
            best_idx = idx;
            best_fitness = population_[idx].fitness;
        }
    }

    return best_idx;
}

// =============================================================================
// Evolution
// =============================================================================

void EvolutionarySearch::evolveGeneration(const SearchSpace& space, const CostModel& model) {
    if (population_.empty()) {
        return;
    }

    // Record best fitness before evolution
    float gen_best = bestFitness();

    std::vector<Individual> new_population;
    new_population.reserve(config_.population_size);

    // Elitism: preserve best individuals
    for (size_t i = 0; i < std::min(config_.elite_count, population_.size()); ++i) {
        new_population.push_back(population_[i]);
    }

    // Generate offspring
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    while (new_population.size() < config_.population_size) {
        size_t p1_idx = selectParentIndex();

        Individual child;

        if (prob_dist(rng_) < config_.crossover_prob && population_.size() > 1) {
            // Crossover
            size_t p2_idx = selectParentIndex();
            while (p2_idx == p1_idx) {
                p2_idx = selectParentIndex();
            }
            child.schedule =
                crossover(population_[p1_idx].schedule, population_[p2_idx].schedule, space);
        } else {
            // Clone parent
            child.schedule = *population_[p1_idx].schedule.clone();
        }

        // Mutation
        if (prob_dist(rng_) < config_.mutation_prob) {
            child.schedule = mutate(child.schedule, space);
        }

        child.fitness = std::numeric_limits<float>::max();
        child.evaluated = false;
        child.hash = computeScheduleHash(child.schedule);

        new_population.push_back(std::move(child));
    }

    population_ = std::move(new_population);

    // Evaluate new population
    evaluatePopulation(model);

    // Maintain diversity if configured
    if (config_.maintain_diversity) {
        ensureDiversity();
    }

    // Update convergence history
    convergence_history_.push_back(bestFitness());
    ++current_generation_;

    // Call callback if set
    if (generation_callback_) {
        generation_callback_(current_generation_, bestFitness());
    }

    spdlog::debug("Generation {}: best fitness = {}", current_generation_, bestFitness());
}

void EvolutionarySearch::evolveGeneration(const SearchSpace& space, const CostModel& model,
                                          const ComputeDAG& dag) {

    if (population_.empty()) {
        return;
    }

    std::vector<Individual> new_population;
    new_population.reserve(config_.population_size);

    // Elitism
    for (size_t i = 0; i < std::min(config_.elite_count, population_.size()); ++i) {
        new_population.push_back(population_[i]);
    }

    // Generate offspring
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    while (new_population.size() < config_.population_size) {
        size_t p1_idx = selectParentIndex();

        Individual child;

        if (prob_dist(rng_) < config_.crossover_prob && population_.size() > 1) {
            size_t p2_idx = selectParentIndex();
            while (p2_idx == p1_idx) {
                p2_idx = selectParentIndex();
            }
            child.schedule =
                crossover(population_[p1_idx].schedule, population_[p2_idx].schedule, space);
        } else {
            child.schedule = *population_[p1_idx].schedule.clone();
        }

        if (prob_dist(rng_) < config_.mutation_prob) {
            child.schedule = mutate(child.schedule, space);
        }

        child.fitness = std::numeric_limits<float>::max();
        child.evaluated = false;
        child.hash = computeScheduleHash(child.schedule);

        new_population.push_back(std::move(child));
    }

    population_ = std::move(new_population);
    evaluatePopulation(model, dag);

    if (config_.maintain_diversity) {
        ensureDiversity();
    }

    convergence_history_.push_back(bestFitness());
    ++current_generation_;

    if (generation_callback_) {
        generation_callback_(current_generation_, bestFitness());
    }
}

float EvolutionarySearch::bestFitness() const {
    if (population_.empty()) {
        return std::numeric_limits<float>::max();
    }
    return population_[0].fitness;
}

const Individual& EvolutionarySearch::bestIndividual() const {
    static Individual empty;
    if (population_.empty()) {
        return empty;
    }
    return population_[0];
}

// =============================================================================
// Full Search
// =============================================================================

Schedule EvolutionarySearch::search(const ComputeDAG& dag, const SearchSpace& space,
                                    const CostModel& model) {

    // Initialize
    initializePopulation(space);
    evaluatePopulation(model, dag);

    // Evolve for specified generations
    for (size_t gen = 0; gen < config_.num_generations; ++gen) {
        evolveGeneration(space, model, dag);

        // Check for early termination
        if (shouldTerminateEarly()) {
            spdlog::info("Early termination at generation {}", current_generation_);
            break;
        }
    }

    return bestIndividual().schedule;
}

void EvolutionarySearch::setGenerationCallback(GenerationCallback callback) {
    generation_callback_ = std::move(callback);
}

bool EvolutionarySearch::shouldTerminateEarly() const {
    if (convergence_history_.size() < config_.early_stop_patience) {
        return false;
    }

    // Check if improvement is below threshold
    size_t n = convergence_history_.size();
    float recent_best = convergence_history_[n - 1];
    float old_best = convergence_history_[n - config_.early_stop_patience];

    float improvement = (old_best - recent_best) / (old_best + 1e-10f);

    return improvement < config_.early_stop_threshold;
}

// =============================================================================
// Statistics
// =============================================================================

PopulationStats EvolutionarySearch::statistics() const {
    PopulationStats stats;

    if (population_.empty()) {
        return stats;
    }

    // Compute min, max, avg
    float sum = 0.0f;
    stats.min_fitness = std::numeric_limits<float>::max();
    stats.max_fitness = std::numeric_limits<float>::lowest();

    for (const auto& ind : population_) {
        sum += ind.fitness;
        stats.min_fitness = std::min(stats.min_fitness, ind.fitness);
        stats.max_fitness = std::max(stats.max_fitness, ind.fitness);
    }

    stats.avg_fitness = sum / static_cast<float>(population_.size());

    // Compute standard deviation
    float sq_sum = 0.0f;
    for (const auto& ind : population_) {
        float diff = ind.fitness - stats.avg_fitness;
        sq_sum += diff * diff;
    }
    stats.std_fitness = std::sqrt(sq_sum / static_cast<float>(population_.size()));

    stats.diversity = populationDiversity();

    return stats;
}

float EvolutionarySearch::populationDiversity() const {
    if (population_.size() < 2) {
        return 0.0f;
    }

    // Measure diversity based on unique hashes
    std::unordered_set<size_t> unique_hashes;
    for (const auto& ind : population_) {
        unique_hashes.insert(ind.hash);
    }

    return static_cast<float>(unique_hashes.size()) / static_cast<float>(population_.size());
}

size_t EvolutionarySearch::computeScheduleHash(const Schedule& schedule) const {
    std::string serialized = schedule.serialize();
    return std::hash<std::string>{}(serialized);
}

void EvolutionarySearch::ensureDiversity() {
    // Remove duplicates (keep first occurrence which has better fitness after sorting)
    std::unordered_set<size_t> seen;
    std::vector<Individual> unique_population;

    for (auto& ind : population_) {
        if (seen.count(ind.hash) == 0) {
            seen.insert(ind.hash);
            unique_population.push_back(std::move(ind));
        }
    }

    population_ = std::move(unique_population);
}

// =============================================================================
// Serialization
// =============================================================================

std::string EvolutionarySearch::serialize() const {
    nlohmann::json j;

    // Config
    nlohmann::json cj;
    cj["population_size"] = config_.population_size;
    cj["num_generations"] = config_.num_generations;
    cj["mutation_prob"] = config_.mutation_prob;
    cj["crossover_prob"] = config_.crossover_prob;
    cj["tournament_size"] = config_.tournament_size;
    cj["elite_count"] = config_.elite_count;
    cj["early_stop_threshold"] = config_.early_stop_threshold;
    cj["early_stop_patience"] = config_.early_stop_patience;
    cj["maintain_diversity"] = config_.maintain_diversity;
    cj["diversity_weight"] = config_.diversity_weight;
    j["config"] = cj;

    // State
    j["current_generation"] = current_generation_;
    j["convergence_history"] = convergence_history_;

    // Population
    nlohmann::json pop_array = nlohmann::json::array();
    for (const auto& ind : population_) {
        nlohmann::json ij;
        ij["schedule"] = ind.schedule.serialize();
        ij["fitness"] = ind.fitness;
        ij["evaluated"] = ind.evaluated;
        ij["hash"] = ind.hash;
        pop_array.push_back(ij);
    }
    j["population"] = pop_array;

    // Best ever
    nlohmann::json bj;
    bj["schedule"] = best_ever_.schedule.serialize();
    bj["fitness"] = best_ever_.fitness;
    j["best_ever"] = bj;

    return j.dump();
}

bool EvolutionarySearch::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        // Config
        auto cj = j["config"];
        config_.population_size = cj["population_size"].get<size_t>();
        config_.num_generations = cj["num_generations"].get<size_t>();
        config_.mutation_prob = cj["mutation_prob"].get<float>();
        config_.crossover_prob = cj["crossover_prob"].get<float>();
        config_.tournament_size = cj["tournament_size"].get<size_t>();
        config_.elite_count = cj["elite_count"].get<size_t>();
        config_.early_stop_threshold = cj["early_stop_threshold"].get<float>();
        config_.early_stop_patience = cj["early_stop_patience"].get<size_t>();
        config_.maintain_diversity = cj["maintain_diversity"].get<bool>();
        config_.diversity_weight = cj["diversity_weight"].get<float>();

        // State
        current_generation_ = j["current_generation"].get<size_t>();
        convergence_history_ = j["convergence_history"].get<std::vector<float>>();

        // Population
        population_.clear();
        for (const auto& ij : j["population"]) {
            Individual ind;
            ind.schedule.deserialize(ij["schedule"].get<std::string>());
            ind.fitness = ij["fitness"].get<float>();
            ind.evaluated = ij["evaluated"].get<bool>();
            ind.hash = ij["hash"].get<size_t>();
            population_.push_back(std::move(ind));
        }

        // Best ever
        auto bj = j["best_ever"];
        best_ever_.schedule.deserialize(bj["schedule"].get<std::string>());
        best_ever_.fitness = bj["fitness"].get<float>();

        valid_ = true;
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize EvolutionarySearch: {}", e.what());
        return false;
    }
}

bool EvolutionarySearch::saveCheckpoint(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        spdlog::warn("Failed to open file for saving: {}", path);
        return false;
    }

    std::string data = serialize();
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    return file.good();
}

bool EvolutionarySearch::loadCheckpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::warn("Failed to open file for loading: {}", path);
        return false;
    }

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return deserialize(data);
}

}  // namespace scheduler
}  // namespace bud
