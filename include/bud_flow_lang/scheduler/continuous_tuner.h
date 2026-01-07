#pragma once

// =============================================================================
// Bud Flow Lang - Continuous Auto-Tuner
// =============================================================================
//
// Runtime-adaptive kernel selection using Thompson Sampling (Bayesian approach)
// for multi-armed bandit optimization. Continuously learns from execution times
// to select the best kernel variant for each workload.
//
// Features:
// - Thompson Sampling for exploration/exploitation balance
// - Online learning from actual execution times
// - Integration with CostModel for performance prediction
// - Profile-guided kernel specialization
// - Warm start from historical data
//
// Based on:
// - Thompson Sampling (1933) - Bayesian multi-armed bandit
// - AutoTVM/Ansor (Chen et al., 2018) - ML-based tuning
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/pgo_specializer.h"
#include "bud_flow_lang/scheduler/cost_model.h"
#include "bud_flow_lang/scheduler/schedule.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// Kernel Variant - Represents a compiled kernel implementation
// =============================================================================

struct KernelVariant {
    uint32_t id = 0;                // Unique variant ID
    Schedule schedule;              // The schedule used to generate this variant
    std::function<void()> execute;  // Compiled kernel execution function
    std::string name;               // Human-readable name
    bool is_specialized = false;    // Whether this is a size-specialized variant

    // Performance statistics
    mutable uint64_t total_executions = 0;
    mutable uint64_t total_time_ns = 0;
    mutable float avg_time_ns = 0.0f;
    mutable float min_time_ns = std::numeric_limits<float>::max();
    mutable float max_time_ns = 0.0f;
};

// =============================================================================
// Continuous Tuner Configuration
// =============================================================================

struct ContinuousTunerConfig {
    // Thompson Sampling parameters
    float prior_alpha = 1.0f;        // Beta distribution prior alpha
    float prior_beta = 1.0f;         // Beta distribution prior beta
    float exploration_bonus = 0.1f;  // Bonus for less-explored variants

    // Tuning behavior
    size_t warmup_executions = 10;        // Min executions before adaptive selection
    size_t min_samples_per_variant = 5;   // Min samples before including in selection
    float improvement_threshold = 0.05f;  // Min improvement to switch variants (5%)

    // Decay for online learning (exponential moving average)
    float time_decay = 0.99f;  // Weight for historical observations

    // Retuning triggers
    float retune_threshold = 0.2f;  // Performance regression to trigger retune (20%)
    size_t retune_interval = 1000;  // Executions between retune checks

    // Memory limits
    size_t max_variants_per_kernel = 8;  // Maximum variants to maintain
    size_t max_profiled_sizes = 16;      // Maximum sizes to profile

    // Background tuning (variant generation, not recording)
    bool enable_background_tuning = true;  // Tune new variants in background
    size_t background_tune_batch = 4;      // Variants to tune per batch
};

// =============================================================================
// Thompson Sampling State - Per-variant Bayesian statistics
// =============================================================================

struct ThompsonState {
    // Beta distribution parameters (successes, failures scaled by performance)
    float alpha = 1.0f;  // Prior + weighted successes
    float beta = 1.0f;   // Prior + weighted failures

    // Performance tracking
    float sum_rewards = 0.0f;  // Cumulative reward (1/time)
    float sum_squared = 0.0f;  // For variance estimation
    uint64_t num_samples = 0;

    // Derived statistics
    [[nodiscard]] float mean() const { return alpha / (alpha + beta); }

    [[nodiscard]] float variance() const {
        float ab = alpha + beta;
        return (alpha * beta) / (ab * ab * (ab + 1.0f));
    }

    // Sample from posterior Beta distribution
    [[nodiscard]] float sample(std::mt19937& rng) const;

    // Update with new observation (reward = 1/execution_time normalized)
    void update(float reward);
};

// =============================================================================
// Kernel Profile - Runtime profile for a specific kernel
// =============================================================================

struct KernelProfile {
    std::string kernel_name;
    std::vector<KernelVariant> variants;
    std::vector<ThompsonState> thompson_states;  // Parallel to variants
    size_t current_best = 0;                     // Index of current best variant
    jit::ProfileData profile_data;               // PGO profile data

    // Selection history
    std::vector<uint32_t> selection_history;  // Recent selections for debugging
    uint64_t total_executions = 0;
    float cumulative_time_ns = 0.0f;

    // Timestamps
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_tuned_at;
    std::chrono::steady_clock::time_point last_retune_check;
};

// =============================================================================
// Tuning Event - For logging and debugging
// =============================================================================

enum class TuningEventType : uint8_t {
    kVariantSelected,  // A variant was selected for execution
    kVariantRecorded,  // Execution time recorded
    kBestChanged,      // Best variant changed
    kRetuneTriggered,  // Retuning was triggered
    kVariantAdded,     // New variant added
    kVariantEvicted,   // Variant removed (LRU)
};

struct TuningEvent {
    TuningEventType type;
    std::string kernel_name;
    uint32_t variant_id = 0;
    float time_ns = 0.0f;
    float improvement = 0.0f;
    std::chrono::steady_clock::time_point timestamp;
};

// =============================================================================
// Continuous Auto-Tuner Statistics
// =============================================================================

struct ContinuousTunerStats {
    // Selection statistics
    uint64_t total_selections = 0;
    uint64_t exploitation_count = 0;  // Selected known-best
    uint64_t exploration_count = 0;   // Explored new variant

    // Learning statistics
    uint64_t total_observations = 0;
    uint64_t variants_created = 0;
    uint64_t variants_evicted = 0;

    // Performance
    float avg_selection_overhead_ns = 0.0f;
    float cumulative_speedup = 0.0f;  // vs baseline (first variant)

    // Retune statistics
    uint64_t retune_triggers = 0;
    uint64_t successful_retunes = 0;
};

// =============================================================================
// ContinuousAutoTuner
// =============================================================================

/// Runtime-adaptive kernel selector using Thompson Sampling
class ContinuousAutoTuner {
  public:
    using ExecutionCallback = std::function<void(const KernelVariant&, float time_ns)>;

    ContinuousAutoTuner();
    explicit ContinuousAutoTuner(const ContinuousTunerConfig& config);
    ~ContinuousAutoTuner();

    // Move-only
    ContinuousAutoTuner(ContinuousAutoTuner&&) noexcept;
    ContinuousAutoTuner& operator=(ContinuousAutoTuner&&) noexcept;
    ContinuousAutoTuner(const ContinuousAutoTuner&) = delete;
    ContinuousAutoTuner& operator=(const ContinuousAutoTuner&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    [[nodiscard]] const ContinuousTunerConfig& config() const { return config_; }
    void setConfig(const ContinuousTunerConfig& config);

    /// Enable/disable continuous tuning
    void setEnabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool isEnabled() const { return enabled_; }

    // =========================================================================
    // Kernel Registration
    // =========================================================================

    /// Register a kernel with initial variant
    bool registerKernel(const std::string& name, KernelVariant initial_variant);

    /// Register a kernel with multiple variants
    bool registerKernel(const std::string& name, std::vector<KernelVariant> variants);

    /// Add a new variant to an existing kernel
    bool addVariant(const std::string& kernel_name, KernelVariant variant);

    /// Check if kernel is registered
    [[nodiscard]] bool hasKernel(const std::string& name) const;

    /// Get number of variants for a kernel
    [[nodiscard]] size_t numVariants(const std::string& kernel_name) const;

    // =========================================================================
    // Variant Selection (Core API)
    // =========================================================================

    /// Select best variant using Thompson Sampling
    /// Returns variant index, or 0 if kernel not found
    [[nodiscard]] size_t selectVariant(const std::string& kernel_name);

    /// Select variant for specific input size
    [[nodiscard]] size_t selectVariant(const std::string& kernel_name, size_t input_size);

    /// Get the selected variant (const access)
    [[nodiscard]] const KernelVariant* getVariant(const std::string& kernel_name,
                                                  size_t variant_idx) const;

    /// Get current best variant for a kernel
    [[nodiscard]] const KernelVariant* getBestVariant(const std::string& kernel_name) const;

    // =========================================================================
    // Execution Recording (Feedback Loop)
    // =========================================================================

    /// Record execution time for a variant (nanoseconds)
    void recordExecution(const std::string& kernel_name, size_t variant_idx, float time_ns);

    /// Record execution with input size
    void recordExecution(const std::string& kernel_name, size_t variant_idx, size_t input_size,
                         float time_ns);

    /// Batch record multiple executions
    void recordExecutionBatch(const std::string& kernel_name,
                              const std::vector<size_t>& variant_indices,
                              const std::vector<float>& times_ns);

    // =========================================================================
    // Integrated Dispatch (Select + Execute + Record)
    // =========================================================================

    /// Execute kernel with automatic variant selection and timing
    template <typename Func>
    float dispatch(const std::string& kernel_name, Func&& execute_func) {
        size_t variant_idx = selectVariant(kernel_name);

        auto start = std::chrono::steady_clock::now();
        execute_func(variant_idx);
        auto end = std::chrono::steady_clock::now();

        float time_ns = std::chrono::duration<float, std::nano>(end - start).count();
        recordExecution(kernel_name, variant_idx, time_ns);

        return time_ns;
    }

    /// Execute with size context
    template <typename Func>
    float dispatch(const std::string& kernel_name, size_t input_size, Func&& execute_func) {
        size_t variant_idx = selectVariant(kernel_name, input_size);

        auto start = std::chrono::steady_clock::now();
        execute_func(variant_idx);
        auto end = std::chrono::steady_clock::now();

        float time_ns = std::chrono::duration<float, std::nano>(end - start).count();
        recordExecution(kernel_name, variant_idx, input_size, time_ns);

        return time_ns;
    }

    // =========================================================================
    // Cost Model Integration
    // =========================================================================

    /// Set the cost model for performance prediction
    void setCostModel(std::shared_ptr<CostModel> model);

    /// Get the cost model
    [[nodiscard]] std::shared_ptr<CostModel> costModel() const { return cost_model_; }

    /// Update cost model with recorded observations
    void syncCostModel();

    // =========================================================================
    // Retuning
    // =========================================================================

    /// Check if kernel needs retuning
    [[nodiscard]] bool needsRetune(const std::string& kernel_name) const;

    /// Force retune for a kernel
    void triggerRetune(const std::string& kernel_name);

    /// Set retune callback
    void setRetuneCallback(std::function<void(const std::string&)> callback);

    // =========================================================================
    // Statistics and Debugging
    // =========================================================================

    /// Get overall statistics
    [[nodiscard]] ContinuousTunerStats statistics() const;

    /// Get kernel-specific profile
    [[nodiscard]] std::optional<KernelProfile> getKernelProfile(const std::string& name) const;

    /// Get recent tuning events
    [[nodiscard]] std::vector<TuningEvent> recentEvents(size_t count = 100) const;

    /// Set execution callback for monitoring
    void setExecutionCallback(ExecutionCallback callback);

    /// Reset all statistics
    void resetStatistics();

    /// Reset specific kernel profile
    void resetKernel(const std::string& kernel_name);

    // =========================================================================
    // Persistence
    // =========================================================================

    /// Serialize tuner state to string
    [[nodiscard]] std::string serialize() const;

    /// Deserialize tuner state from string
    bool deserialize(const std::string& data);

    /// Save state to file
    bool save(const std::string& path) const;

    /// Load state from file
    bool load(const std::string& path);

    // =========================================================================
    // Warm Start
    // =========================================================================

    /// Import historical execution data for warm start
    void importHistory(const std::string& kernel_name,
                       const std::vector<std::pair<size_t, float>>& variant_times);

    /// Export execution history
    [[nodiscard]] std::vector<std::pair<size_t, float>>
    exportHistory(const std::string& kernel_name) const;

  private:
    ContinuousTunerConfig config_;
    bool enabled_ = true;

    // Kernel profiles indexed by name
    mutable std::mutex mutex_;
    std::unordered_map<std::string, KernelProfile> kernels_;

    // Random number generator for Thompson Sampling
    mutable std::mt19937 rng_;

    // Cost model integration
    std::shared_ptr<CostModel> cost_model_;

    // Callbacks
    ExecutionCallback execution_callback_;
    std::function<void(const std::string&)> retune_callback_;

    // Event log (circular buffer)
    static constexpr size_t kMaxEvents = 1000;
    std::vector<TuningEvent> events_;
    size_t event_head_ = 0;

    // Statistics
    mutable std::atomic<uint64_t> total_selections_{0};
    mutable std::atomic<uint64_t> total_observations_{0};
    ContinuousTunerStats stats_;

    // Internal helpers
    void logEvent(TuningEvent event);
    void updateThompsonState(ThompsonState& state, float time_ns);
    void checkRetune(KernelProfile& profile);
    void evictVariant(KernelProfile& profile);
    float normalizeReward(float time_ns, const KernelProfile& profile) const;
};

// =============================================================================
// Global Continuous Tuner Instance
// =============================================================================

/// Get the global continuous tuner instance
ContinuousAutoTuner& getGlobalTuner();

/// Set the global continuous tuner config
void configureGlobalTuner(const ContinuousTunerConfig& config);

}  // namespace scheduler
}  // namespace bud
