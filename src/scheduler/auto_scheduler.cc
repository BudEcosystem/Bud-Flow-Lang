// =============================================================================
// Bud Flow Lang - Auto-Scheduler Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/auto_scheduler.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>

namespace bud {
namespace scheduler {

// =============================================================================
// AutoScheduler Implementation
// =============================================================================

AutoScheduler::AutoScheduler() : config_{}, cost_model_(std::make_unique<CostModel>()) {}

AutoScheduler::AutoScheduler(const TuningConfig& config)
    : config_(config), cost_model_(std::make_unique<CostModel>()) {}

AutoScheduler::~AutoScheduler() = default;

AutoScheduler::AutoScheduler(AutoScheduler&&) noexcept = default;
AutoScheduler& AutoScheduler::operator=(AutoScheduler&&) noexcept = default;

// =============================================================================
// Tuning
// =============================================================================

TuneResult AutoScheduler::tune(const ir::IRModule& module) {
    return tuneInternal(module, {});
}

TuneResult AutoScheduler::continueTuning(const ir::IRModule& module, const Schedule& baseline) {
    return tuneInternal(module, {baseline});
}

TuneResult AutoScheduler::tuneWithWarmStart(const ir::IRModule& module,
                                            const std::vector<Schedule>& warm_schedules) {
    return tuneInternal(module, warm_schedules);
}

TuneResult AutoScheduler::tuneInternal(const ir::IRModule& module,
                                       const std::vector<Schedule>& warm_schedules) {

    TuneResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Build compute DAG
    auto dag = ComputeDAG::fromIR(module.builder());
    if (dag.numNodes() == 0) {
        result.success = true;  // Empty DAG is valid
        return result;
    }

    // Build search space
    auto space = SearchSpace::fromDAG(dag);
    if (space.numSketches() == 0) {
        result.success = true;
        return result;
    }

    // Configure evolutionary search
    SearchConfig search_config;
    search_config.population_size = config_.population_size;
    search_config.num_generations = config_.num_generations;
    search_config.mutation_prob = config_.mutation_prob;
    search_config.crossover_prob = config_.crossover_prob;
    search_config.early_stop_threshold = config_.early_stop_threshold;
    search_config.early_stop_patience = config_.early_stop_patience;

    EvolutionarySearch search(search_config);

    // Initialize with warm start schedules if provided
    search.initializePopulation(space);

    // Add warm start schedules to population
    if (!warm_schedules.empty()) {
        // In a full implementation, we'd inject these into the population
    }

    // Evaluate initial population
    search.evaluatePopulation(*cost_model_, dag);

    // Track best so far
    float best_cost = search.bestFitness();
    Schedule best_schedule = search.bestIndividual().schedule;

    // Report initial progress
    if (found_better_callback_) {
        found_better_callback_(best_schedule, best_cost);
    }

    // Evolution loop
    for (size_t gen = 0; gen < config_.num_generations; ++gen) {
        // Check time budget
        if (config_.time_budget_seconds > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - start_time).count();
            if (elapsed >= config_.time_budget_seconds) {
                if (config_.verbose) {
                    spdlog::info("Time budget reached after {} generations", gen);
                }
                break;
            }
        }

        // Evolve one generation
        search.evolveGeneration(space, *cost_model_, dag);
        result.generations = gen + 1;

        // Update statistics
        result.schedules_evaluated += config_.population_size;
        stats_.total_schedules_evaluated += config_.population_size;

        // Check for improvement
        float current_best = search.bestFitness();
        if (current_best < best_cost) {
            best_cost = current_best;
            best_schedule = search.bestIndividual().schedule;

            if (found_better_callback_) {
                found_better_callback_(best_schedule, best_cost);
            }
        }

        // Report progress
        if (progress_callback_) {
            TuningProgress progress;
            progress.current_generation = gen + 1;
            progress.total_generations = config_.num_generations;
            progress.best_cost = best_cost;

            auto now = std::chrono::high_resolution_clock::now();
            progress.elapsed_seconds = std::chrono::duration<float>(now - start_time).count();

            if (gen > 0) {
                float avg_time_per_gen = progress.elapsed_seconds / (gen + 1);
                progress.estimated_remaining_seconds =
                    avg_time_per_gen * (config_.num_generations - gen - 1);
            }

            progress.population_diversity = search.populationDiversity();
            reportProgress(progress);
        }

        // Log progress
        if (config_.verbose && (gen + 1) % config_.log_interval == 0) {
            spdlog::info("Generation {}/{}: best cost = {}", gen + 1, config_.num_generations,
                         best_cost);
        }
    }

    // Finalize result
    auto end_time = std::chrono::high_resolution_clock::now();
    result.tuning_time_seconds = std::chrono::duration<float>(end_time - start_time).count();
    result.success = true;
    result.schedule = best_schedule;
    result.best_cost = best_cost;

    // Update statistics
    stats_.tuning_time_seconds += result.tuning_time_seconds;
    stats_.num_modules_tuned++;

    // Record best schedule
    updateBestSchedule(module.name(), best_schedule, best_cost);

    return result;
}

std::vector<TuneResult> AutoScheduler::tuneBatch(const std::vector<ir::IRModule>& modules) {

    std::vector<TuneResult> results;
    results.reserve(modules.size());

    for (const auto& module : modules) {
        results.push_back(tune(module));
    }

    return results;
}

// =============================================================================
// Schedule Application
// =============================================================================

bool AutoScheduler::applySchedule(ir::IRModule& module, const Schedule& schedule) {
    // In a full implementation, we would:
    // 1. Parse the schedule transforms
    // 2. Apply loop transformations to the IR
    // 3. Insert prefetch hints
    // 4. Mark parallel loops

    // For now, we validate the schedule and return success
    if (!schedule.validate()) {
        return false;
    }

    // Apply transformations (placeholder)
    // The actual implementation would modify the module's IR
    // based on the schedule's transforms

    return true;
}

// =============================================================================
// Callbacks
// =============================================================================

void AutoScheduler::setProgressCallback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

void AutoScheduler::setFoundBetterCallback(FoundBetterCallback callback) {
    found_better_callback_ = std::move(callback);
}

void AutoScheduler::reportProgress(const TuningProgress& progress) {
    if (progress_callback_) {
        progress_callback_(progress);
    }
}

// =============================================================================
// Tuning Log
// =============================================================================

bool AutoScheduler::exportLog(const std::string& path) const {
    nlohmann::json j;

    // Export config
    nlohmann::json cj;
    cj["population_size"] = config_.population_size;
    cj["num_generations"] = config_.num_generations;
    cj["mutation_prob"] = config_.mutation_prob;
    cj["crossover_prob"] = config_.crossover_prob;
    cj["time_budget_seconds"] = config_.time_budget_seconds;
    j["config"] = cj;

    // Export tuning records
    nlohmann::json records_array = nlohmann::json::array();
    for (const auto& record : tuning_records_) {
        nlohmann::json rj;
        rj["module_name"] = record.module_name;
        rj["schedule"] = record.schedule.serialize();
        rj["cost"] = record.cost;
        rj["timestamp"] =
            std::chrono::duration_cast<std::chrono::seconds>(record.timestamp.time_since_epoch())
                .count();
        records_array.push_back(rj);
    }
    j["tuning_records"] = records_array;

    // Export statistics
    nlohmann::json sj;
    sj["total_schedules_evaluated"] = stats_.total_schedules_evaluated;
    sj["tuning_time_seconds"] = stats_.tuning_time_seconds;
    sj["num_modules_tuned"] = stats_.num_modules_tuned;
    j["statistics"] = sj;

    // Write to file
    std::ofstream file(path);
    if (!file) {
        spdlog::warn("Failed to open file for export: {}", path);
        return false;
    }

    file << j.dump(2);
    return file.good();
}

bool AutoScheduler::importLog(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        spdlog::warn("Failed to open file for import: {}", path);
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        // Import config (optional, may want to keep current config)
        if (j.contains("config")) {
            auto cj = j["config"];
            config_.population_size = cj["population_size"].get<size_t>();
            config_.num_generations = cj["num_generations"].get<size_t>();
            config_.mutation_prob = cj["mutation_prob"].get<float>();
            config_.crossover_prob = cj["crossover_prob"].get<float>();
            config_.time_budget_seconds = cj["time_budget_seconds"].get<float>();
        }

        // Import tuning records
        tuning_records_.clear();
        best_schedule_index_.clear();

        if (j.contains("tuning_records")) {
            for (const auto& rj : j["tuning_records"]) {
                TuningRecord record;
                record.module_name = rj["module_name"].get<std::string>();
                record.schedule.deserialize(rj["schedule"].get<std::string>());
                record.cost = rj["cost"].get<float>();

                auto ts_seconds = rj["timestamp"].get<int64_t>();
                record.timestamp =
                    std::chrono::system_clock::time_point(std::chrono::seconds(ts_seconds));

                size_t idx = tuning_records_.size();
                tuning_records_.push_back(record);

                // Update best schedule index
                auto it = best_schedule_index_.find(record.module_name);
                if (it == best_schedule_index_.end() ||
                    record.cost < tuning_records_[it->second].cost) {
                    best_schedule_index_[record.module_name] = idx;
                }
            }
        }

        // Import statistics
        if (j.contains("statistics")) {
            auto sj = j["statistics"];
            stats_.total_schedules_evaluated = sj["total_schedules_evaluated"].get<size_t>();
            stats_.tuning_time_seconds = sj["tuning_time_seconds"].get<float>();
            stats_.num_modules_tuned = sj["num_modules_tuned"].get<size_t>();
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to import tuning log: {}", e.what());
        return false;
    }
}

std::optional<Schedule> AutoScheduler::getBestSchedule(const std::string& module_name) const {
    auto it = best_schedule_index_.find(module_name);
    if (it == best_schedule_index_.end()) {
        return std::nullopt;
    }
    return tuning_records_[it->second].schedule;
}

void AutoScheduler::updateBestSchedule(const std::string& module_name, const Schedule& schedule,
                                       float cost) {

    TuningRecord record;
    record.module_name = module_name;
    record.schedule = schedule;
    record.cost = cost;
    record.timestamp = std::chrono::system_clock::now();

    size_t idx = tuning_records_.size();
    tuning_records_.push_back(record);

    auto it = best_schedule_index_.find(module_name);
    if (it == best_schedule_index_.end() || cost < tuning_records_[it->second].cost) {
        best_schedule_index_[module_name] = idx;
    }
}

// =============================================================================
// Statistics
// =============================================================================

TuningStatistics AutoScheduler::statistics() const {
    return stats_;
}

// =============================================================================
// Serialization
// =============================================================================

std::string AutoScheduler::serialize() const {
    nlohmann::json j;

    // Config
    nlohmann::json cj;
    cj["population_size"] = config_.population_size;
    cj["num_generations"] = config_.num_generations;
    cj["mutation_prob"] = config_.mutation_prob;
    cj["crossover_prob"] = config_.crossover_prob;
    cj["time_budget_seconds"] = config_.time_budget_seconds;
    cj["early_stop_threshold"] = config_.early_stop_threshold;
    cj["early_stop_patience"] = config_.early_stop_patience;
    cj["use_learned_model"] = config_.use_learned_model;
    cj["model_warmup_samples"] = config_.model_warmup_samples;
    cj["verbose"] = config_.verbose;
    cj["log_interval"] = config_.log_interval;
    j["config"] = cj;

    // Statistics
    nlohmann::json sj;
    sj["total_schedules_evaluated"] = stats_.total_schedules_evaluated;
    sj["tuning_time_seconds"] = stats_.tuning_time_seconds;
    sj["num_modules_tuned"] = stats_.num_modules_tuned;
    sj["average_improvement"] = stats_.average_improvement;
    sj["cache_hits"] = stats_.cache_hits;
    j["statistics"] = sj;

    // Tuning records
    nlohmann::json records_array = nlohmann::json::array();
    for (const auto& record : tuning_records_) {
        nlohmann::json rj;
        rj["module_name"] = record.module_name;
        rj["schedule"] = record.schedule.serialize();
        rj["cost"] = record.cost;
        rj["timestamp"] =
            std::chrono::duration_cast<std::chrono::seconds>(record.timestamp.time_since_epoch())
                .count();
        records_array.push_back(rj);
    }
    j["tuning_records"] = records_array;

    // Best schedule index
    nlohmann::json idx_obj;
    for (const auto& [name, idx] : best_schedule_index_) {
        idx_obj[name] = idx;
    }
    j["best_schedule_index"] = idx_obj;

    return j.dump();
}

bool AutoScheduler::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        // Config
        auto cj = j["config"];
        config_.population_size = cj["population_size"].get<size_t>();
        config_.num_generations = cj["num_generations"].get<size_t>();
        config_.mutation_prob = cj["mutation_prob"].get<float>();
        config_.crossover_prob = cj["crossover_prob"].get<float>();
        config_.time_budget_seconds = cj["time_budget_seconds"].get<float>();
        config_.early_stop_threshold = cj["early_stop_threshold"].get<float>();
        config_.early_stop_patience = cj["early_stop_patience"].get<size_t>();
        config_.use_learned_model = cj["use_learned_model"].get<bool>();
        config_.model_warmup_samples = cj["model_warmup_samples"].get<size_t>();
        config_.verbose = cj["verbose"].get<bool>();
        config_.log_interval = cj["log_interval"].get<size_t>();

        // Statistics
        auto sj = j["statistics"];
        stats_.total_schedules_evaluated = sj["total_schedules_evaluated"].get<size_t>();
        stats_.tuning_time_seconds = sj["tuning_time_seconds"].get<float>();
        stats_.num_modules_tuned = sj["num_modules_tuned"].get<size_t>();
        stats_.average_improvement = sj["average_improvement"].get<float>();
        stats_.cache_hits = sj["cache_hits"].get<size_t>();

        // Tuning records
        tuning_records_.clear();
        for (const auto& rj : j["tuning_records"]) {
            TuningRecord record;
            record.module_name = rj["module_name"].get<std::string>();
            record.schedule.deserialize(rj["schedule"].get<std::string>());
            record.cost = rj["cost"].get<float>();

            auto ts_seconds = rj["timestamp"].get<int64_t>();
            record.timestamp =
                std::chrono::system_clock::time_point(std::chrono::seconds(ts_seconds));

            tuning_records_.push_back(record);
        }

        // Best schedule index
        best_schedule_index_.clear();
        for (const auto& [name, idx] : j["best_schedule_index"].items()) {
            best_schedule_index_[name] = idx.get<size_t>();
        }

        valid_ = true;
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize AutoScheduler: {}", e.what());
        return false;
    }
}

}  // namespace scheduler
}  // namespace bud
