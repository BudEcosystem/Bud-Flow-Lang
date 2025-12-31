#pragma once

// =============================================================================
// Bud Flow Lang - Auto-Scheduler
// =============================================================================
//
// Main entry point for automatic schedule optimization.
// Combines all scheduler components into a unified interface.
//
// Features:
// - One-shot tuning
// - Incremental tuning with warm start
// - Schedule application
// - Tuning log persistence
// - Multi-module batch tuning
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/evolutionary_search.h"
#include "bud_flow_lang/scheduler/schedule.h"
#include "bud_flow_lang/scheduler/search_space.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// Tuning Configuration
// =============================================================================

/// Configuration for auto-tuning
struct TuningConfig {
    // Search parameters
    size_t population_size = 64;
    size_t num_generations = 10;
    float mutation_prob = 0.1f;
    float crossover_prob = 0.5f;

    // Time budget (0 = no limit)
    float time_budget_seconds = 0.0f;

    // Early termination
    float early_stop_threshold = 0.001f;
    size_t early_stop_patience = 5;

    // Cost model
    bool use_learned_model = true;
    size_t model_warmup_samples = 100;

    // Verbosity
    bool verbose = false;
    size_t log_interval = 1;  // Log every N generations
};

// =============================================================================
// Tuning Result
// =============================================================================

/// Result of a tuning operation
struct TuneResult {
    bool success = false;
    Schedule schedule;
    float best_cost = std::numeric_limits<float>::max();
    size_t generations = 0;
    size_t schedules_evaluated = 0;
    float tuning_time_seconds = 0.0f;
    std::string error_message;
};

// =============================================================================
// Tuning Progress
// =============================================================================

/// Progress information during tuning
struct TuningProgress {
    size_t current_generation = 0;
    size_t total_generations = 0;
    float best_cost = 0.0f;
    float elapsed_seconds = 0.0f;
    float estimated_remaining_seconds = 0.0f;
    float population_diversity = 0.0f;
};

// =============================================================================
// Tuning Statistics
// =============================================================================

/// Statistics about tuning operations
struct TuningStatistics {
    size_t total_schedules_evaluated = 0;
    float tuning_time_seconds = 0.0f;
    size_t num_modules_tuned = 0;
    float average_improvement = 0.0f;
    size_t cache_hits = 0;
};

// =============================================================================
// AutoScheduler
// =============================================================================

/// Main auto-scheduler class
class AutoScheduler {
  public:
    using ProgressCallback = std::function<void(const TuningProgress&)>;
    using FoundBetterCallback = std::function<void(const Schedule&, float cost)>;

    AutoScheduler();
    explicit AutoScheduler(const TuningConfig& config);
    ~AutoScheduler();

    // Move-only
    AutoScheduler(AutoScheduler&&) noexcept;
    AutoScheduler& operator=(AutoScheduler&&) noexcept;
    AutoScheduler(const AutoScheduler&) = delete;
    AutoScheduler& operator=(const AutoScheduler&) = delete;

    // -------------------------------------------------------------------------
    // Validity
    // -------------------------------------------------------------------------

    [[nodiscard]] bool isValid() const { return valid_; }
    [[nodiscard]] const TuningConfig& config() const { return config_; }

    // -------------------------------------------------------------------------
    // Tuning
    // -------------------------------------------------------------------------

    /// Tune an IR module
    [[nodiscard]] TuneResult tune(const ir::IRModule& module);

    /// Continue tuning from a previous result
    [[nodiscard]] TuneResult continueTuning(const ir::IRModule& module, const Schedule& baseline);

    /// Tune with warm start from existing schedules
    [[nodiscard]] TuneResult tuneWithWarmStart(const ir::IRModule& module,
                                               const std::vector<Schedule>& warm_schedules);

    /// Tune multiple modules in batch
    [[nodiscard]] std::vector<TuneResult> tuneBatch(const std::vector<ir::IRModule>& modules);

    // -------------------------------------------------------------------------
    // Schedule Application
    // -------------------------------------------------------------------------

    /// Apply a schedule to an IR module (modifies in place, returns success)
    [[nodiscard]] bool applySchedule(ir::IRModule& module, const Schedule& schedule);

    // -------------------------------------------------------------------------
    // Callbacks
    // -------------------------------------------------------------------------

    /// Set progress callback
    void setProgressCallback(ProgressCallback callback);

    /// Set "found better" callback
    void setFoundBetterCallback(FoundBetterCallback callback);

    // -------------------------------------------------------------------------
    // Tuning Log
    // -------------------------------------------------------------------------

    /// Export tuning log to file
    bool exportLog(const std::string& path) const;

    /// Import tuning log from file
    bool importLog(const std::string& path);

    /// Get number of tuning records
    [[nodiscard]] size_t numTuningRecords() const { return tuning_records_.size(); }

    /// Get best schedule for a module (by name)
    [[nodiscard]] std::optional<Schedule> getBestSchedule(const std::string& module_name) const;

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------

    [[nodiscard]] TuningStatistics statistics() const;

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    [[nodiscard]] std::string serialize() const;
    bool deserialize(const std::string& data);

  private:
    TuningConfig config_;
    bool valid_ = true;

    // Components
    std::unique_ptr<CostModel> cost_model_;

    // Callbacks
    ProgressCallback progress_callback_;
    FoundBetterCallback found_better_callback_;

    // Tuning records
    struct TuningRecord {
        std::string module_name;
        Schedule schedule;
        float cost = std::numeric_limits<float>::max();
        std::chrono::system_clock::time_point timestamp;
    };
    std::vector<TuningRecord> tuning_records_;
    std::unordered_map<std::string, size_t> best_schedule_index_;

    // Statistics
    TuningStatistics stats_;

    // Internal helpers
    TuneResult tuneInternal(const ir::IRModule& module,
                            const std::vector<Schedule>& warm_schedules);

    void updateBestSchedule(const std::string& module_name, const Schedule& schedule, float cost);

    void reportProgress(const TuningProgress& progress);
};

}  // namespace scheduler
}  // namespace bud
