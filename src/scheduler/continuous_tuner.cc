// =============================================================================
// Bud Flow Lang - Continuous Auto-Tuner Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/continuous_tuner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace bud {
namespace scheduler {

// =============================================================================
// Thompson Sampling State
// =============================================================================

float ThompsonState::sample(std::mt19937& rng) const {
    // Sample from Beta(alpha, beta) distribution
    // Using the gamma distribution method:
    // X ~ Gamma(alpha, 1), Y ~ Gamma(beta, 1)
    // X / (X + Y) ~ Beta(alpha, beta)

    std::gamma_distribution<float> gamma_alpha(alpha, 1.0f);
    std::gamma_distribution<float> gamma_beta(beta, 1.0f);

    float x = gamma_alpha(rng);
    float y = gamma_beta(rng);

    // Avoid division by zero
    if (x + y < 1e-10f) {
        return 0.5f;
    }

    return x / (x + y);
}

void ThompsonState::update(float reward) {
    // Update running statistics
    num_samples++;
    sum_rewards += reward;
    sum_squared += reward * reward;

    // Update Beta distribution parameters
    // We scale rewards to [0, 1] where 1 = best observed, 0 = worst
    // Higher reward -> increase alpha (success rate)
    // Lower reward -> increase beta (failure rate)

    // Additive update based on reward
    alpha += reward;
    beta += (1.0f - reward);
}

// =============================================================================
// ContinuousAutoTuner - Constructor / Destructor
// =============================================================================

ContinuousAutoTuner::ContinuousAutoTuner() : ContinuousAutoTuner(ContinuousTunerConfig{}) {}

ContinuousAutoTuner::ContinuousAutoTuner(const ContinuousTunerConfig& config) : config_(config) {
    // Seed RNG with random device
    std::random_device rd;
    rng_.seed(rd());

    // Pre-allocate event buffer
    events_.reserve(kMaxEvents);
}

ContinuousAutoTuner::~ContinuousAutoTuner() = default;

ContinuousAutoTuner::ContinuousAutoTuner(ContinuousAutoTuner&& other) noexcept
    : config_(std::move(other.config_)),
      enabled_(other.enabled_),
      kernels_(std::move(other.kernels_)),
      rng_(std::move(other.rng_)),
      cost_model_(std::move(other.cost_model_)),
      execution_callback_(std::move(other.execution_callback_)),
      retune_callback_(std::move(other.retune_callback_)),
      events_(std::move(other.events_)),
      event_head_(other.event_head_),
      stats_(other.stats_) {
    // Atomics need explicit load/store
    total_selections_.store(other.total_selections_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    total_observations_.store(other.total_observations_.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
}

ContinuousAutoTuner& ContinuousAutoTuner::operator=(ContinuousAutoTuner&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock(mutex_);

        config_ = std::move(other.config_);
        enabled_ = other.enabled_;
        kernels_ = std::move(other.kernels_);
        rng_ = std::move(other.rng_);
        cost_model_ = std::move(other.cost_model_);
        execution_callback_ = std::move(other.execution_callback_);
        retune_callback_ = std::move(other.retune_callback_);
        events_ = std::move(other.events_);
        event_head_ = other.event_head_;
        total_selections_.store(other.total_selections_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        total_observations_.store(other.total_observations_.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
        stats_ = other.stats_;
    }
    return *this;
}

// =============================================================================
// Configuration
// =============================================================================

void ContinuousAutoTuner::setConfig(const ContinuousTunerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

// =============================================================================
// Kernel Registration
// =============================================================================

bool ContinuousAutoTuner::registerKernel(const std::string& name, KernelVariant initial_variant) {
    return registerKernel(name, std::vector<KernelVariant>{std::move(initial_variant)});
}

bool ContinuousAutoTuner::registerKernel(const std::string& name,
                                         std::vector<KernelVariant> variants) {
    if (variants.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (kernels_.count(name) > 0) {
        // Kernel already registered
        return false;
    }

    KernelProfile profile;
    profile.kernel_name = name;
    profile.variants = std::move(variants);
    profile.current_best = 0;
    profile.created_at = std::chrono::steady_clock::now();
    profile.last_tuned_at = profile.created_at;
    profile.last_retune_check = profile.created_at;

    // Initialize Thompson states with prior
    profile.thompson_states.resize(profile.variants.size());
    for (auto& state : profile.thompson_states) {
        state.alpha = config_.prior_alpha;
        state.beta = config_.prior_beta;
    }

    // Assign IDs
    for (size_t i = 0; i < profile.variants.size(); ++i) {
        profile.variants[i].id = static_cast<uint32_t>(i);
    }

    size_t num_variants = profile.variants.size();
    kernels_[name] = std::move(profile);
    stats_.variants_created += num_variants;

    return true;
}

bool ContinuousAutoTuner::addVariant(const std::string& kernel_name, KernelVariant variant) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return false;
    }

    auto& profile = it->second;

    // Check variant limit
    if (profile.variants.size() >= config_.max_variants_per_kernel) {
        evictVariant(profile);
    }

    variant.id = static_cast<uint32_t>(profile.variants.size());
    profile.variants.push_back(std::move(variant));

    ThompsonState state;
    state.alpha = config_.prior_alpha;
    state.beta = config_.prior_beta;
    profile.thompson_states.push_back(state);

    stats_.variants_created++;

    logEvent({TuningEventType::kVariantAdded, kernel_name, variant.id, 0.0f, 0.0f,
              std::chrono::steady_clock::now()});

    return true;
}

bool ContinuousAutoTuner::hasKernel(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.count(name) > 0;
}

size_t ContinuousAutoTuner::numVariants(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return 0;
    }
    return it->second.variants.size();
}

// =============================================================================
// Variant Selection (Thompson Sampling)
// =============================================================================

size_t ContinuousAutoTuner::selectVariant(const std::string& kernel_name) {
    if (!enabled_) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return 0;
    }

    auto& profile = it->second;
    profile.total_executions++;
    total_selections_.fetch_add(1, std::memory_order_relaxed);

    // During warmup, round-robin through variants
    if (profile.total_executions <= config_.warmup_executions * profile.variants.size()) {
        size_t idx = (profile.total_executions - 1) % profile.variants.size();
        logEvent({TuningEventType::kVariantSelected, kernel_name, profile.variants[idx].id, 0.0f,
                  0.0f, std::chrono::steady_clock::now()});
        return idx;
    }

    // Thompson Sampling: sample from each variant's posterior and pick highest
    float best_sample = -1.0f;
    size_t best_idx = 0;

    for (size_t i = 0; i < profile.thompson_states.size(); ++i) {
        const auto& state = profile.thompson_states[i];

        // Skip variants with too few samples (unless all are under-sampled)
        if (state.num_samples < config_.min_samples_per_variant) {
            // Give exploration bonus to under-sampled variants
            float bonus =
                config_.exploration_bonus * (config_.min_samples_per_variant - state.num_samples);
            float sample = state.sample(rng_) + bonus;
            if (sample > best_sample) {
                best_sample = sample;
                best_idx = i;
            }
        } else {
            float sample = state.sample(rng_);
            if (sample > best_sample) {
                best_sample = sample;
                best_idx = i;
            }
        }
    }

    // Track exploration vs exploitation
    if (best_idx == profile.current_best) {
        stats_.exploitation_count++;
    } else {
        stats_.exploration_count++;
    }

    profile.selection_history.push_back(static_cast<uint32_t>(best_idx));
    if (profile.selection_history.size() > 1000) {
        profile.selection_history.erase(profile.selection_history.begin(),
                                        profile.selection_history.begin() + 500);
    }

    logEvent({TuningEventType::kVariantSelected, kernel_name, profile.variants[best_idx].id, 0.0f,
              0.0f, std::chrono::steady_clock::now()});

    return best_idx;
}

size_t ContinuousAutoTuner::selectVariant(const std::string& kernel_name, size_t input_size) {
    // For now, size-aware selection just uses the standard selection
    // In the future, this could maintain per-size Thompson states
    (void)input_size;
    return selectVariant(kernel_name);
}

const KernelVariant* ContinuousAutoTuner::getVariant(const std::string& kernel_name,
                                                     size_t variant_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end() || variant_idx >= it->second.variants.size()) {
        return nullptr;
    }

    return &it->second.variants[variant_idx];
}

const KernelVariant* ContinuousAutoTuner::getBestVariant(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end() || it->second.variants.empty()) {
        return nullptr;
    }

    return &it->second.variants[it->second.current_best];
}

// =============================================================================
// Execution Recording
// =============================================================================

void ContinuousAutoTuner::recordExecution(const std::string& kernel_name, size_t variant_idx,
                                          float time_ns) {
    recordExecution(kernel_name, variant_idx, 0, time_ns);
}

void ContinuousAutoTuner::recordExecution(const std::string& kernel_name, size_t variant_idx,
                                          size_t input_size, float time_ns) {
    if (!enabled_ || time_ns <= 0.0f) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end() || variant_idx >= it->second.variants.size()) {
        return;
    }

    auto& profile = it->second;
    auto& variant = profile.variants[variant_idx];
    auto& state = profile.thompson_states[variant_idx];

    // Update variant statistics
    variant.total_executions++;
    variant.total_time_ns += static_cast<uint64_t>(time_ns);
    variant.min_time_ns = std::min(variant.min_time_ns, time_ns);
    variant.max_time_ns = std::max(variant.max_time_ns, time_ns);

    // Exponential moving average for avg_time
    if (variant.avg_time_ns == 0.0f) {
        variant.avg_time_ns = time_ns;
    } else {
        variant.avg_time_ns =
            config_.time_decay * variant.avg_time_ns + (1.0f - config_.time_decay) * time_ns;
    }

    // Update profile totals
    profile.cumulative_time_ns += time_ns;

    // Record in PGO profile data
    profile.profile_data.recordExecution(input_size,
                                         std::chrono::nanoseconds(static_cast<int64_t>(time_ns)));

    // Normalize reward (inverse time, scaled to [0, 1])
    float reward = normalizeReward(time_ns, profile);

    // Update Thompson state
    updateThompsonState(state, reward);

    total_observations_.fetch_add(1, std::memory_order_relaxed);
    stats_.total_observations++;

    // Check if this variant is now the best
    size_t old_best = profile.current_best;
    float best_mean = profile.thompson_states[old_best].mean();

    if (variant_idx != old_best &&
        state.mean() > best_mean * (1.0f + config_.improvement_threshold)) {
        profile.current_best = variant_idx;
        float improvement = (state.mean() - best_mean) / best_mean;

        logEvent({TuningEventType::kBestChanged, kernel_name, static_cast<uint32_t>(variant_idx),
                  time_ns, improvement, std::chrono::steady_clock::now()});
    }

    // Check if retuning is needed
    checkRetune(profile);

    // Callback if set
    if (execution_callback_) {
        execution_callback_(variant, time_ns);
    }

    logEvent({TuningEventType::kVariantRecorded, kernel_name, static_cast<uint32_t>(variant_idx),
              time_ns, 0.0f, std::chrono::steady_clock::now()});
}

void ContinuousAutoTuner::recordExecutionBatch(const std::string& kernel_name,
                                               const std::vector<size_t>& variant_indices,
                                               const std::vector<float>& times_ns) {
    if (variant_indices.size() != times_ns.size()) {
        return;
    }

    for (size_t i = 0; i < variant_indices.size(); ++i) {
        recordExecution(kernel_name, variant_indices[i], times_ns[i]);
    }
}

// =============================================================================
// Cost Model Integration
// =============================================================================

void ContinuousAutoTuner::setCostModel(std::shared_ptr<CostModel> model) {
    std::lock_guard<std::mutex> lock(mutex_);
    cost_model_ = std::move(model);
}

void ContinuousAutoTuner::syncCostModel() {
    if (!cost_model_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Update cost model with observations from all kernels
    for (auto& [name, profile] : kernels_) {
        for (size_t i = 0; i < profile.variants.size(); ++i) {
            const auto& variant = profile.variants[i];
            if (variant.total_executions > 0) {
                cost_model_->update(variant.schedule, variant.avg_time_ns);
            }
        }
    }
}

// =============================================================================
// Retuning
// =============================================================================

bool ContinuousAutoTuner::needsRetune(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return false;
    }

    const auto& profile = it->second;

    // Check if enough executions since last check
    if (profile.total_executions < config_.retune_interval) {
        return false;
    }

    // Check for performance regression
    const auto& best = profile.variants[profile.current_best];
    if (best.total_executions < config_.warmup_executions) {
        return false;
    }

    // Compare recent performance to historical
    float recent_time = best.avg_time_ns;
    float historical_time =
        static_cast<float>(best.total_time_ns) / static_cast<float>(best.total_executions);

    if (recent_time > historical_time * (1.0f + config_.retune_threshold)) {
        return true;
    }

    return false;
}

void ContinuousAutoTuner::triggerRetune(const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return;
    }

    it->second.last_retune_check = std::chrono::steady_clock::now();
    stats_.retune_triggers++;

    logEvent({TuningEventType::kRetuneTriggered, kernel_name, 0, 0.0f, 0.0f,
              std::chrono::steady_clock::now()});

    if (retune_callback_) {
        retune_callback_(kernel_name);
    }
}

void ContinuousAutoTuner::setRetuneCallback(std::function<void(const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    retune_callback_ = std::move(callback);
}

// =============================================================================
// Statistics and Debugging
// =============================================================================

ContinuousTunerStats ContinuousAutoTuner::statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);

    ContinuousTunerStats stats = stats_;
    stats.total_selections = total_selections_.load(std::memory_order_relaxed);
    stats.total_observations = total_observations_.load(std::memory_order_relaxed);

    return stats;
}

std::optional<KernelProfile> ContinuousAutoTuner::getKernelProfile(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(name);
    if (it == kernels_.end()) {
        return std::nullopt;
    }

    return it->second;
}

std::vector<TuningEvent> ContinuousAutoTuner::recentEvents(size_t count) const {
    std::lock_guard<std::mutex> lock(mutex_);

    count = std::min(count, events_.size());
    std::vector<TuningEvent> result;
    result.reserve(count);

    // Return most recent events
    size_t start = events_.size() > count ? events_.size() - count : 0;
    for (size_t i = start; i < events_.size(); ++i) {
        result.push_back(events_[i]);
    }

    return result;
}

void ContinuousAutoTuner::setExecutionCallback(ExecutionCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    execution_callback_ = std::move(callback);
}

void ContinuousAutoTuner::resetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);

    stats_ = ContinuousTunerStats{};
    total_selections_.store(0, std::memory_order_relaxed);
    total_observations_.store(0, std::memory_order_relaxed);
    events_.clear();
    event_head_ = 0;

    for (auto& [name, profile] : kernels_) {
        profile.total_executions = 0;
        profile.cumulative_time_ns = 0.0f;
        profile.selection_history.clear();
        profile.profile_data.reset();

        for (auto& variant : profile.variants) {
            variant.total_executions = 0;
            variant.total_time_ns = 0;
            variant.avg_time_ns = 0.0f;
            variant.min_time_ns = std::numeric_limits<float>::max();
            variant.max_time_ns = 0.0f;
        }

        for (auto& state : profile.thompson_states) {
            state = ThompsonState{};
            state.alpha = config_.prior_alpha;
            state.beta = config_.prior_beta;
        }
    }
}

void ContinuousAutoTuner::resetKernel(const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return;
    }

    auto& profile = it->second;
    profile.total_executions = 0;
    profile.cumulative_time_ns = 0.0f;
    profile.current_best = 0;
    profile.selection_history.clear();
    profile.profile_data.reset();

    for (auto& variant : profile.variants) {
        variant.total_executions = 0;
        variant.total_time_ns = 0;
        variant.avg_time_ns = 0.0f;
        variant.min_time_ns = std::numeric_limits<float>::max();
        variant.max_time_ns = 0.0f;
    }

    for (auto& state : profile.thompson_states) {
        state = ThompsonState{};
        state.alpha = config_.prior_alpha;
        state.beta = config_.prior_beta;
    }
}

// =============================================================================
// Persistence
// =============================================================================

std::string ContinuousAutoTuner::serialize() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream ss;
    ss << "ContinuousAutoTuner v1\n";
    ss << kernels_.size() << "\n";

    for (const auto& [name, profile] : kernels_) {
        ss << name << "\n";
        ss << profile.variants.size() << "\n";

        for (size_t i = 0; i < profile.variants.size(); ++i) {
            const auto& variant = profile.variants[i];
            const auto& state = profile.thompson_states[i];

            ss << variant.name << "\n";
            ss << variant.total_executions << " " << variant.total_time_ns << " "
               << variant.avg_time_ns << "\n";
            ss << state.alpha << " " << state.beta << " " << state.num_samples << " "
               << state.sum_rewards << "\n";
        }

        ss << profile.current_best << "\n";
    }

    return ss.str();
}

bool ContinuousAutoTuner::deserialize(const std::string& data) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::istringstream ss(data);
    std::string header;
    std::getline(ss, header);

    if (header != "ContinuousAutoTuner v1") {
        return false;
    }

    size_t num_kernels = 0;
    ss >> num_kernels;
    ss.ignore();

    for (size_t k = 0; k < num_kernels; ++k) {
        std::string name;
        std::getline(ss, name);

        size_t num_variants = 0;
        ss >> num_variants;
        ss.ignore();

        auto it = kernels_.find(name);
        if (it == kernels_.end()) {
            // Skip this kernel if not registered
            for (size_t v = 0; v < num_variants; ++v) {
                std::string line;
                std::getline(ss, line);  // variant name
                std::getline(ss, line);  // stats
                std::getline(ss, line);  // thompson
            }
            std::getline(ss, header);  // current_best
            continue;
        }

        auto& profile = it->second;

        for (size_t v = 0; v < num_variants && v < profile.variants.size(); ++v) {
            auto& variant = profile.variants[v];
            auto& state = profile.thompson_states[v];

            std::string variant_name;
            std::getline(ss, variant_name);

            ss >> variant.total_executions >> variant.total_time_ns >> variant.avg_time_ns;
            ss >> state.alpha >> state.beta >> state.num_samples >> state.sum_rewards;
            ss.ignore();
        }

        ss >> profile.current_best;
        ss.ignore();
    }

    return true;
}

bool ContinuousAutoTuner::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        return false;
    }

    file << serialize();
    return file.good();
}

bool ContinuousAutoTuner::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return deserialize(ss.str());
}

// =============================================================================
// Warm Start
// =============================================================================

void ContinuousAutoTuner::importHistory(
    const std::string& kernel_name, const std::vector<std::pair<size_t, float>>& variant_times) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return;
    }

    auto& profile = it->second;

    for (const auto& [variant_idx, time_ns] : variant_times) {
        if (variant_idx >= profile.variants.size()) {
            continue;
        }

        auto& variant = profile.variants[variant_idx];
        auto& state = profile.thompson_states[variant_idx];

        variant.total_executions++;
        variant.total_time_ns += static_cast<uint64_t>(time_ns);

        if (variant.avg_time_ns == 0.0f) {
            variant.avg_time_ns = time_ns;
        } else {
            variant.avg_time_ns =
                config_.time_decay * variant.avg_time_ns + (1.0f - config_.time_decay) * time_ns;
        }

        float reward = normalizeReward(time_ns, profile);
        state.update(reward);
    }
}

std::vector<std::pair<size_t, float>>
ContinuousAutoTuner::exportHistory(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::pair<size_t, float>> result;

    auto it = kernels_.find(kernel_name);
    if (it == kernels_.end()) {
        return result;
    }

    const auto& profile = it->second;

    for (size_t i = 0; i < profile.variants.size(); ++i) {
        const auto& variant = profile.variants[i];
        if (variant.total_executions > 0) {
            result.emplace_back(i, variant.avg_time_ns);
        }
    }

    return result;
}

// =============================================================================
// Internal Helpers
// =============================================================================

void ContinuousAutoTuner::logEvent(TuningEvent event) {
    if (events_.size() < kMaxEvents) {
        events_.push_back(std::move(event));
    } else {
        events_[event_head_] = std::move(event);
        event_head_ = (event_head_ + 1) % kMaxEvents;
    }
}

void ContinuousAutoTuner::updateThompsonState(ThompsonState& state, float reward) {
    state.update(reward);
}

void ContinuousAutoTuner::checkRetune(KernelProfile& profile) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - profile.last_retune_check)
            .count();

    // Check every retune_interval executions or every 10 seconds
    if (profile.total_executions % config_.retune_interval == 0 || elapsed > 10000) {
        profile.last_retune_check = now;

        // Check for performance regression (inline, no locking since caller holds lock)
        const auto& best = profile.variants[profile.current_best];
        if (best.total_executions >= config_.warmup_executions) {
            float recent_time = best.avg_time_ns;
            float historical_time =
                static_cast<float>(best.total_time_ns) / static_cast<float>(best.total_executions);

            if (recent_time > historical_time * (1.0f + config_.retune_threshold)) {
                stats_.retune_triggers++;

                logEvent({TuningEventType::kRetuneTriggered, profile.kernel_name, 0, 0.0f, 0.0f,
                          std::chrono::steady_clock::now()});

                // Call callback without holding lock (release first)
                if (retune_callback_) {
                    // Note: we can't release the lock here in a scoped guard,
                    // so just call the callback with the lock held.
                    // For production, consider using a deferred callback queue.
                    retune_callback_(profile.kernel_name);
                }
            }
        }
    }
}

void ContinuousAutoTuner::evictVariant(KernelProfile& profile) {
    if (profile.variants.size() <= 1) {
        return;
    }

    // Find the variant with lowest Thompson mean (excluding current best)
    size_t evict_idx = 0;
    float lowest_mean = std::numeric_limits<float>::max();

    for (size_t i = 0; i < profile.thompson_states.size(); ++i) {
        if (i == profile.current_best) {
            continue;
        }

        float mean = profile.thompson_states[i].mean();
        if (mean < lowest_mean) {
            lowest_mean = mean;
            evict_idx = i;
        }
    }

    logEvent({TuningEventType::kVariantEvicted, profile.kernel_name, profile.variants[evict_idx].id,
              0.0f, 0.0f, std::chrono::steady_clock::now()});

    profile.variants.erase(profile.variants.begin() + evict_idx);
    profile.thompson_states.erase(profile.thompson_states.begin() + evict_idx);

    // Adjust current_best if needed
    if (profile.current_best >= profile.variants.size()) {
        profile.current_best = profile.variants.size() - 1;
    } else if (profile.current_best > evict_idx) {
        profile.current_best--;
    }

    stats_.variants_evicted++;
}

float ContinuousAutoTuner::normalizeReward(float time_ns, const KernelProfile& profile) const {
    // Find best (min) and worst (max) times across variants
    float min_time = std::numeric_limits<float>::max();
    float max_time = 0.0f;

    for (const auto& variant : profile.variants) {
        if (variant.total_executions > 0) {
            min_time = std::min(min_time, variant.min_time_ns);
            max_time = std::max(max_time, variant.max_time_ns);
        }
    }

    if (min_time >= max_time || max_time == 0.0f) {
        return 0.5f;  // No differentiation possible
    }

    // Reward = 1 - normalized_time (lower time = higher reward)
    float normalized = (time_ns - min_time) / (max_time - min_time);
    normalized = std::clamp(normalized, 0.0f, 1.0f);

    return 1.0f - normalized;
}

// =============================================================================
// Global Instance
// =============================================================================

namespace {
std::unique_ptr<ContinuousAutoTuner> g_tuner;
std::once_flag g_tuner_init_flag;
}  // namespace

ContinuousAutoTuner& getGlobalTuner() {
    std::call_once(g_tuner_init_flag, []() { g_tuner = std::make_unique<ContinuousAutoTuner>(); });
    return *g_tuner;
}

void configureGlobalTuner(const ContinuousTunerConfig& config) {
    getGlobalTuner().setConfig(config);
}

}  // namespace scheduler
}  // namespace bud
