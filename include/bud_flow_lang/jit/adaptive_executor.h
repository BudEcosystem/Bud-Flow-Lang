#pragma once

// =============================================================================
// Bud Flow Lang - Adaptive Executor
// =============================================================================
//
// Integrates the tiered execution system with Thompson Sampling-based
// adaptive optimization. This is the main entry point for adaptive JIT
// compilation that:
//
// 1. Wraps TieredExecutor for actual code execution
// 2. Uses AdaptiveKernelCache for context-keyed version management
// 3. Implements Thompson-based tier promotion (not static thresholds)
// 4. Supports profile persistence for warm start optimization
//
// Based on:
// - JVM HotSpot tiered compilation
// - V8 TurboFan speculative optimization
// - .NET 8 Dynamic PGO
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/adaptive_kernel_cache.h"

#include <chrono>
#include <functional>
#include <memory>

namespace bud {
namespace jit {

// Forward declarations
class AdaptiveExecutor;

// =============================================================================
// Execution Result with Profiling Data
// =============================================================================

struct ExecutionResult {
    bool success = false;
    float execution_time_ns = 0.0f;
    ExecutionTier tier_used = ExecutionTier::kInterpreter;
    size_t version_idx = 0;
    bool was_promoted = false;
    std::string error_message;
};

// =============================================================================
// Adaptive Executor Configuration
// =============================================================================

struct AdaptiveExecutorConfig {
    // Use adaptive cache configuration
    AdaptiveCacheConfig cache_config;

    // Execution settings
    bool enable_timing = true;          // Record execution times
    bool enable_promotion = true;       // Enable automatic tier promotion
    bool enable_specialization = true;  // Enable size specialization

    // Fast path for small arrays (bypasses adaptive overhead)
    // Arrays smaller than this threshold use direct Highway dispatch
    // without Thompson Sampling, timing, or context lookup.
    // Set to 0 to disable fast path (always use adaptive)
    size_t small_array_threshold = 10000;  // Elements

    // Profile persistence
    bool enable_persistence = true;  // Save/load profiles
    std::string profile_dir = "~/.bud_flow/profiles";
    std::chrono::seconds auto_save_interval{300};  // 5 minutes

    // Logging
    bool verbose = false;  // Log promotion decisions
};

// =============================================================================
// Adaptive Executor
// =============================================================================

class AdaptiveExecutor {
  public:
    explicit AdaptiveExecutor(const AdaptiveExecutorConfig& config = {});
    ~AdaptiveExecutor();

    AdaptiveExecutor(AdaptiveExecutor&&) noexcept;
    AdaptiveExecutor& operator=(AdaptiveExecutor&&) noexcept;
    AdaptiveExecutor(const AdaptiveExecutor&) = delete;
    AdaptiveExecutor& operator=(const AdaptiveExecutor&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    [[nodiscard]] const AdaptiveExecutorConfig& config() const { return config_; }
    void setConfig(const AdaptiveExecutorConfig& config);

    /// Enable/disable adaptive optimization
    void setEnabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool isEnabled() const { return enabled_; }

    /// Set small array threshold (fast path bypasses adaptive overhead)
    void setSmallArrayThreshold(size_t threshold) { config_.small_array_threshold = threshold; }
    [[nodiscard]] size_t smallArrayThreshold() const { return config_.small_array_threshold; }

    // =========================================================================
    // Binary Operation Execution (Primary API)
    // =========================================================================

    /// Execute a binary operation with adaptive optimization
    ExecutionResult executeBinaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                    const void* input_a, const void* input_b, size_t count);

    /// Execute a unary operation with adaptive optimization
    ExecutionResult executeUnaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input,
                                   size_t count);

    /// Execute FMA operation with adaptive optimization
    ExecutionResult executeFmaOp(ScalarType dtype, void* output, const void* input_a,
                                 const void* input_b, const void* input_c, size_t count);

    // =========================================================================
    // Generic Dispatch (for custom kernels)
    // =========================================================================

    /// Execute a kernel with adaptive variant selection
    template <typename Func>
    ExecutionResult dispatch(uint64_t kernel_hash, ScalarType dtype, size_t input_size,
                             Func&& execute_func) {
        if (!enabled_) {
            auto start = std::chrono::steady_clock::now();
            execute_func(0);  // Default to variant 0
            auto end = std::chrono::steady_clock::now();

            ExecutionResult result;
            result.success = true;
            result.execution_time_ns = std::chrono::duration<float, std::nano>(end - start).count();
            result.tier_used = ExecutionTier::kInterpreter;
            return result;
        }

        // Create context key
        auto key = AdaptiveKernelCache::makeKey(kernel_hash, dtype, input_size);

        // Select version using Thompson Sampling
        auto decision = cache_.selectVersion(key, input_size);

        // Execute
        auto start = std::chrono::steady_clock::now();
        execute_func(decision.version_idx);
        auto end = std::chrono::steady_clock::now();

        float time_ns = std::chrono::duration<float, std::nano>(end - start).count();

        // Record execution for learning
        cache_.recordExecution(key, decision.version_idx, input_size, time_ns);

        // Handle promotion if needed
        ExecutionResult result;
        result.success = true;
        result.execution_time_ns = time_ns;
        result.tier_used = decision.tier;
        result.version_idx = decision.version_idx;

        if (decision.should_promote && config_.enable_promotion) {
            handlePromotion(key, decision);
            result.was_promoted = true;
        }

        return result;
    }

    // =========================================================================
    // Direct Cache Access (for advanced use cases)
    // =========================================================================

    /// Get the underlying adaptive kernel cache
    [[nodiscard]] AdaptiveKernelCache& cache() { return cache_; }
    [[nodiscard]] const AdaptiveKernelCache& cache() const { return cache_; }

    // =========================================================================
    // Statistics
    // =========================================================================

    struct ExecutorStats {
        // Execution counts
        uint64_t total_executions = 0;
        uint64_t tier0_executions = 0;
        uint64_t tier1_executions = 0;
        uint64_t tier2_executions = 0;
        uint64_t fast_path_executions = 0;  // Small arrays bypassing adaptive overhead

        // Timing
        float total_execution_time_ns = 0.0f;
        float avg_execution_time_ns = 0.0f;

        // Optimization
        uint64_t promotions_performed = 0;
        uint64_t specializations_performed = 0;

        // Cache stats
        AdaptiveKernelCache::CacheStats cache_stats;
    };

    [[nodiscard]] ExecutorStats statistics() const;

    /// Reset all statistics and cache
    void reset();

    // =========================================================================
    // Profile Persistence
    // =========================================================================

    /// Save current profiles to disk
    bool saveProfiles() const;

    /// Load profiles from disk
    bool loadProfiles();

    /// Set profile directory
    void setProfileDir(const std::string& dir);

  private:
    AdaptiveExecutorConfig config_;
    AdaptiveKernelCache cache_;
    bool enabled_ = true;

    // Statistics
    mutable std::atomic<uint64_t> total_executions_{0};
    mutable std::atomic<uint64_t> tier0_executions_{0};
    mutable std::atomic<uint64_t> tier1_executions_{0};
    mutable std::atomic<uint64_t> tier2_executions_{0};
    mutable std::atomic<uint64_t> fast_path_executions_{0};
    mutable std::atomic<uint64_t> promotions_performed_{0};
    mutable std::atomic<uint64_t> specializations_performed_{0};
    mutable std::atomic<float> total_execution_time_ns_{0.0f};

    // Internal helpers
    void handlePromotion(const ContextKey& key, const TierDecision& decision);
    ContextKey makeOperationKey(ir::OpCode op, ScalarType dtype, size_t count) const;
    void recordTierExecution(ExecutionTier tier);

    // Profile path management
    [[nodiscard]] std::string getProfilePath() const;
    static std::string expandPath(const std::string& path);
};

// =============================================================================
// Global Adaptive Executor
// =============================================================================

/// Get the global adaptive executor instance
AdaptiveExecutor& getGlobalExecutor();

/// Configure the global executor
void configureGlobalExecutor(const AdaptiveExecutorConfig& config);

/// Initialize adaptive execution (loads profiles)
void initializeAdaptiveExecution();

/// Shutdown adaptive execution (saves profiles)
void shutdownAdaptiveExecution();

}  // namespace jit
}  // namespace bud
