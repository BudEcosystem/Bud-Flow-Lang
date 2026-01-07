// =============================================================================
// Bud Flow Lang - Runtime Executor with Tiered Compilation
// =============================================================================
//
// Implements a tiered execution system based on CPython PEP 744:
//
// Tier 0: Interpreter (direct Highway dispatch)
//   - No compilation overhead
//   - Used for cold code paths
//   - Tracks call counts for promotion
//
// Tier 1: Copy-and-Patch JIT
//   - Sub-millisecond compilation
//   - Promotes after TIER1_THRESHOLD calls
//   - Patches pre-compiled stencils together
//
// Tier 2: Fused Kernels
//   - Maximum performance for hot paths
//   - Promotes after TIER2_THRESHOLD calls
//   - Uses fused Highway kernels (6-30x speedup)
//
// Based on:
// - CPython PEP 744: https://peps.python.org/pep-0744/
// - Weld Lazy Evaluation: https://www.vldb.org/pvldb/vol11/p1002-palkar.pdf
//

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/codegen/fused_kernel.h"
#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/adaptive_executor.h"
#include "bud_flow_lang/jit/stencil.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/prefetch.h"
#include "bud_flow_lang/memory/tiled_executor.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace bud {

// =============================================================================
// Tiered Compilation Constants
// =============================================================================

namespace {

// Call count thresholds for tier promotion
constexpr size_t kTier1Threshold = 10;   // Promote to copy-patch JIT
constexpr size_t kTier2Threshold = 100;  // Promote to fused kernels

// Maximum entries in call count cache
constexpr size_t kMaxCallCountEntries = 10000;

// Memory optimization thresholds
constexpr size_t kTilingThreshold = 1024;    // Elements: Use tiled execution for large arrays
constexpr size_t kPrefetchThreshold = 4096;  // Elements: Enable prefetching for very large arrays

}  // namespace

// =============================================================================
// Execution Tier Enum
// =============================================================================

enum class ExecutionTier : uint8_t {
    kInterpreter = 0,  // Tier 0: Direct Highway dispatch
    kCopyPatch = 1,    // Tier 1: Copy-and-patch JIT
    kFusedKernel = 2,  // Tier 2: Fused Highway kernels
};

const char* tierName(ExecutionTier tier) {
    switch (tier) {
    case ExecutionTier::kInterpreter:
        return "Interpreter";
    case ExecutionTier::kCopyPatch:
        return "CopyPatch";
    case ExecutionTier::kFusedKernel:
        return "FusedKernel";
    default:
        return "Unknown";
    }
}

// =============================================================================
// Call Counter (Thread-Safe)
// =============================================================================

class CallCounter {
  public:
    // Get current tier and increment call count
    ExecutionTier getAndIncrement(uint64_t key) {
        std::shared_lock<std::shared_mutex> read_lock(mutex_);

        auto it = counts_.find(key);
        if (it == counts_.end()) {
            read_lock.unlock();
            std::unique_lock<std::shared_mutex> write_lock(mutex_);

            // Double-check after acquiring write lock
            it = counts_.find(key);
            if (it == counts_.end()) {
                // Check if we need to evict entries
                if (counts_.size() >= kMaxCallCountEntries) {
                    evictOldEntries();
                }

                counts_[key] = {1, ExecutionTier::kInterpreter};
                return ExecutionTier::kInterpreter;
            }
        }

        // Atomically increment and check for promotion
        auto& entry = counts_[key];
        size_t new_count = ++entry.count;
        ExecutionTier current_tier = entry.tier;

        // Check for tier promotion
        if (current_tier == ExecutionTier::kInterpreter && new_count >= kTier1Threshold) {
            entry.tier = ExecutionTier::kCopyPatch;
            spdlog::debug("Promoting key {} to Tier 1 (CopyPatch) after {} calls", key, new_count);
            return ExecutionTier::kCopyPatch;
        }

        if (current_tier == ExecutionTier::kCopyPatch && new_count >= kTier2Threshold) {
            entry.tier = ExecutionTier::kFusedKernel;
            spdlog::debug("Promoting key {} to Tier 2 (FusedKernel) after {} calls", key,
                          new_count);
            return ExecutionTier::kFusedKernel;
        }

        return current_tier;
    }

    // Get call count for a key
    size_t getCount(uint64_t key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = counts_.find(key);
        return it != counts_.end() ? it->second.count : 0;
    }

    // Get current tier for a key
    ExecutionTier getTier(uint64_t key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = counts_.find(key);
        return it != counts_.end() ? it->second.tier : ExecutionTier::kInterpreter;
    }

    // Reset all counters
    void reset() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        counts_.clear();
    }

    // Get statistics
    struct Stats {
        size_t total_entries = 0;
        size_t tier0_entries = 0;
        size_t tier1_entries = 0;
        size_t tier2_entries = 0;
    };

    Stats getStats() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        Stats stats;
        stats.total_entries = counts_.size();

        for (const auto& [key, entry] : counts_) {
            switch (entry.tier) {
            case ExecutionTier::kInterpreter:
                ++stats.tier0_entries;
                break;
            case ExecutionTier::kCopyPatch:
                ++stats.tier1_entries;
                break;
            case ExecutionTier::kFusedKernel:
                ++stats.tier2_entries;
                break;
            }
        }

        return stats;
    }

  private:
    struct CountEntry {
        size_t count = 0;  // Use regular size_t (protected by mutex)
        ExecutionTier tier = ExecutionTier::kInterpreter;
    };

    void evictOldEntries() {
        // Simple eviction: remove entries in Tier 0 with lowest counts
        std::vector<std::pair<uint64_t, size_t>> tier0_entries;

        for (const auto& [key, entry] : counts_) {
            if (entry.tier == ExecutionTier::kInterpreter) {
                tier0_entries.emplace_back(key, entry.count);
            }
        }

        // Sort by count (ascending) and remove bottom 25%
        std::sort(tier0_entries.begin(), tier0_entries.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        size_t to_remove = std::max(size_t{1}, tier0_entries.size() / 4);
        for (size_t i = 0; i < to_remove && i < tier0_entries.size(); ++i) {
            counts_.erase(tier0_entries[i].first);
        }

        spdlog::debug("Evicted {} call count entries", to_remove);
    }

    mutable std::shared_mutex mutex_;
    std::unordered_map<uint64_t, CountEntry> counts_;
};

// =============================================================================
// Profile-Guided Optimization (PGO) Infrastructure
// =============================================================================
//
// Tracks detailed runtime metrics to guide optimization decisions:
// - Call frequency and patterns
// - Common array sizes for kernel specialization
// - Execution time for hotspot identification
// - Operation combinations for fusion opportunities
//

class KernelProfiler {
  public:
    // Record a kernel execution for profiling
    void recordExecution(uint64_t key, ir::OpCode op, size_t element_count,
                         std::chrono::nanoseconds exec_time) {
        std::unique_lock<std::shared_mutex> lock(mutex_);

        auto& profile = profiles_[key];
        profile.call_count++;
        profile.total_elements += element_count;
        profile.total_time_ns += exec_time.count();
        profile.op = op;

        // Track common sizes for specialization
        updateSizeHistogram(profile, element_count);

        // Track operation sequences for fusion opportunities
        recordOperationSequence(op);
    }

    // Get profiling data for a kernel
    struct KernelProfile {
        ir::OpCode op = ir::OpCode::kAdd;
        uint64_t call_count = 0;
        uint64_t total_elements = 0;
        uint64_t total_time_ns = 0;
        std::array<uint32_t, 8> size_histogram{};  // Counts for common sizes
        std::array<size_t, 8> common_sizes{};      // The actual sizes

        // Derived metrics
        [[nodiscard]] double avgElementsPerCall() const {
            return call_count > 0 ? static_cast<double>(total_elements) / call_count : 0.0;
        }

        [[nodiscard]] double avgTimePerElement() const {
            return total_elements > 0 ? static_cast<double>(total_time_ns) / total_elements : 0.0;
        }

        [[nodiscard]] size_t mostCommonSize() const {
            size_t max_idx = 0;
            for (size_t i = 1; i < size_histogram.size(); ++i) {
                if (size_histogram[i] > size_histogram[max_idx]) {
                    max_idx = i;
                }
            }
            return common_sizes[max_idx];
        }

        [[nodiscard]] bool shouldSpecialize() const {
            // Specialize if:
            // 1. Called frequently (>100 times)
            // 2. One size dominates (>50% of calls)
            if (call_count < 100)
                return false;

            uint32_t max_count = *std::max_element(size_histogram.begin(), size_histogram.end());
            return max_count > call_count / 2;
        }
    };

    [[nodiscard]] KernelProfile getProfile(uint64_t key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = profiles_.find(key);
        if (it != profiles_.end()) {
            return it->second;
        }
        return {};
    }

    // Get fusion opportunities based on operation sequences
    struct FusionOpportunity {
        ir::OpCode op1;
        ir::OpCode op2;
        uint64_t sequence_count;
    };

    [[nodiscard]] std::vector<FusionOpportunity> getFusionOpportunities() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<FusionOpportunity> opportunities;

        for (const auto& [pair, count] : op_sequences_) {
            if (count >= 10) {  // Only report significant patterns
                opportunities.push_back({static_cast<ir::OpCode>(pair >> 16),
                                         static_cast<ir::OpCode>(pair & 0xFFFF), count});
            }
        }

        // Sort by count descending
        std::sort(opportunities.begin(), opportunities.end(),
                  [](const auto& a, const auto& b) { return a.sequence_count > b.sequence_count; });

        return opportunities;
    }

    // Get hot kernels (most frequently called)
    [[nodiscard]] std::vector<std::pair<uint64_t, KernelProfile>>
    getHotKernels(size_t limit = 10) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<std::pair<uint64_t, KernelProfile>> hot;

        for (const auto& [key, profile] : profiles_) {
            hot.emplace_back(key, profile);
        }

        // Sort by call count descending
        std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) {
            return a.second.call_count > b.second.call_count;
        });

        if (hot.size() > limit) {
            hot.resize(limit);
        }

        return hot;
    }

    // Get statistics summary
    struct ProfileStats {
        size_t total_profiles = 0;
        uint64_t total_calls = 0;
        uint64_t total_elements = 0;
        uint64_t total_time_ns = 0;
        size_t specializable_kernels = 0;
        size_t fusion_opportunities = 0;
    };

    [[nodiscard]] ProfileStats getStats() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        ProfileStats stats;

        stats.total_profiles = profiles_.size();

        for (const auto& [key, profile] : profiles_) {
            stats.total_calls += profile.call_count;
            stats.total_elements += profile.total_elements;
            stats.total_time_ns += profile.total_time_ns;
            if (profile.shouldSpecialize()) {
                ++stats.specializable_kernels;
            }
        }

        for (const auto& [pair, count] : op_sequences_) {
            if (count >= 10) {
                ++stats.fusion_opportunities;
            }
        }

        return stats;
    }

    // Reset all profiling data
    void reset() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        profiles_.clear();
        op_sequences_.clear();
        last_op_ = ir::OpCode::kAdd;
    }

    // Dump profiling report
    void dumpReport() const {
        auto stats = getStats();
        spdlog::info("=== PGO Profile Report ===");
        spdlog::info("Total kernels profiled: {}", stats.total_profiles);
        spdlog::info("Total calls: {}", stats.total_calls);
        spdlog::info("Total elements processed: {}", stats.total_elements);
        spdlog::info("Total execution time: {} ms", stats.total_time_ns / 1000000.0);
        spdlog::info("Kernels eligible for specialization: {}", stats.specializable_kernels);
        spdlog::info("Fusion opportunities detected: {}", stats.fusion_opportunities);

        // Report hot kernels
        auto hot = getHotKernels(5);
        spdlog::info("Top 5 hot kernels:");
        for (const auto& [key, profile] : hot) {
            spdlog::info("  Key {}: {} calls, {} elements/call, {} ns/element", key,
                         profile.call_count, profile.avgElementsPerCall(),
                         profile.avgTimePerElement());
        }

        // Report fusion opportunities
        auto fusions = getFusionOpportunities();
        if (!fusions.empty()) {
            spdlog::info("Top fusion opportunities:");
            for (size_t i = 0; i < std::min(size_t{5}, fusions.size()); ++i) {
                spdlog::info("  {} -> {}: {} sequences", ir::opCodeName(fusions[i].op1),
                             ir::opCodeName(fusions[i].op2), fusions[i].sequence_count);
            }
        }
    }

  private:
    void updateSizeHistogram(KernelProfile& profile, size_t element_count) {
        // Find if this size is already tracked
        for (size_t i = 0; i < profile.common_sizes.size(); ++i) {
            if (profile.common_sizes[i] == element_count) {
                profile.size_histogram[i]++;
                return;
            }
            if (profile.common_sizes[i] == 0) {
                // Empty slot - use it
                profile.common_sizes[i] = element_count;
                profile.size_histogram[i] = 1;
                return;
            }
        }

        // No room - replace lowest count if this is more common
        size_t min_idx = 0;
        for (size_t i = 1; i < profile.size_histogram.size(); ++i) {
            if (profile.size_histogram[i] < profile.size_histogram[min_idx]) {
                min_idx = i;
            }
        }

        // Only replace if we've seen this new size a few times
        if (profile.size_histogram[min_idx] < 3) {
            profile.common_sizes[min_idx] = element_count;
            profile.size_histogram[min_idx] = 1;
        }
    }

    void recordOperationSequence(ir::OpCode op) {
        // Track consecutive operation pairs for fusion detection
        uint32_t pair = (static_cast<uint32_t>(last_op_) << 16) | static_cast<uint32_t>(op);
        op_sequences_[pair]++;
        last_op_ = op;
    }

    mutable std::shared_mutex mutex_;
    std::unordered_map<uint64_t, KernelProfile> profiles_;
    std::unordered_map<uint32_t, uint64_t> op_sequences_;  // Tracks op1->op2 sequences
    ir::OpCode last_op_ = ir::OpCode::kAdd;
};

// Global profiler instance
static std::unique_ptr<KernelProfiler> g_profiler;

// Public API for PGO
void enableProfiling() {
    if (!g_profiler) {
        g_profiler = std::make_unique<KernelProfiler>();
        spdlog::info("PGO profiling enabled");
    }
}

void disableProfiling() {
    g_profiler.reset();
    spdlog::info("PGO profiling disabled");
}

bool isProfilingEnabled() {
    return g_profiler != nullptr;
}

void dumpProfilingReport() {
    if (g_profiler) {
        g_profiler->dumpReport();
    } else {
        spdlog::warn("Profiling not enabled - no report available");
    }
}

void resetProfilingData() {
    if (g_profiler) {
        g_profiler->reset();
        spdlog::info("PGO profiling data reset");
    }
}

// =============================================================================
// Tiered Executor
// =============================================================================

class TieredExecutor {
  public:
    // Execute a binary operation using the appropriate tier
    Result<void> executeBinaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input_a,
                                 const void* input_b, size_t count) {
        // If adaptive mode is enabled, use AdaptiveExecutor with Thompson Sampling
        if (adaptive_enabled_ && adaptive_executor_) {
            auto result =
                adaptive_executor_->executeBinaryOp(op, dtype, output, input_a, input_b, count);
            if (result.success) {
                return {};
            } else {
                return Error(ErrorCode::kRuntimeError, result.error_message);
            }
        }

        // Fallback: Use static threshold-based tier promotion
        // Generate a key for this operation signature
        uint64_t key = operationKey(op, dtype, count);

        // Get current tier and increment call count
        ExecutionTier tier = call_counter_.getAndIncrement(key);

        // Execute based on tier
        switch (tier) {
        case ExecutionTier::kInterpreter:
            return executeTier0BinaryOp(op, dtype, output, input_a, input_b, count);

        case ExecutionTier::kCopyPatch:
            return executeTier1BinaryOp(op, dtype, output, input_a, input_b, count);

        case ExecutionTier::kFusedKernel:
            // For single binary ops, Tier 2 uses JIT (same as Tier 1)
            // Tier 2 optimization is for fused chains
            return executeTier1BinaryOp(op, dtype, output, input_a, input_b, count);
        }

        return {};
    }

    // Execute a unary operation
    Result<void> executeUnaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input,
                                size_t count) {
        // If adaptive mode is enabled, use AdaptiveExecutor with Thompson Sampling
        if (adaptive_enabled_ && adaptive_executor_) {
            auto result = adaptive_executor_->executeUnaryOp(op, dtype, output, input, count);
            if (result.success) {
                return {};
            } else {
                return Error(ErrorCode::kRuntimeError, result.error_message);
            }
        }

        // Fallback: Use static threshold-based tier promotion
        uint64_t key = operationKey(op, dtype, count);
        ExecutionTier tier = call_counter_.getAndIncrement(key);

        switch (tier) {
        case ExecutionTier::kInterpreter:
            return executeTier0UnaryOp(op, dtype, output, input, count);

        case ExecutionTier::kCopyPatch:
        case ExecutionTier::kFusedKernel:
            return executeTier1UnaryOp(op, dtype, output, input, count);
        }

        return {};
    }

    // Execute FMA operation
    Result<void> executeFmaOp(ScalarType dtype, void* output, const void* input_a,
                              const void* input_b, const void* input_c, size_t count) {
        // If adaptive mode is enabled, use AdaptiveExecutor with Thompson Sampling
        if (adaptive_enabled_ && adaptive_executor_) {
            auto result =
                adaptive_executor_->executeFmaOp(dtype, output, input_a, input_b, input_c, count);
            if (result.success) {
                return {};
            } else {
                return Error(ErrorCode::kRuntimeError, result.error_message);
            }
        }

        // Fallback: Use static threshold-based tier promotion
        uint64_t key = operationKey(ir::OpCode::kFma, dtype, count);
        ExecutionTier tier = call_counter_.getAndIncrement(key);

        switch (tier) {
        case ExecutionTier::kInterpreter:
            return executeTier0FmaOp(dtype, output, input_a, input_b, input_c, count);

        case ExecutionTier::kCopyPatch:
        case ExecutionTier::kFusedKernel:
            return jit::executeJitFmaOp(dtype, output, input_a, input_b, input_c, count);
        }

        return {};
    }

    // Execute a fused operation chain (Tier 2 optimization)
    Result<void> executeFusedChain(const std::string& pattern, void* output,
                                   const std::vector<const void*>& inputs, size_t count) {
        // Dispatch to appropriate fused kernel based on pattern
        if (pattern == "dot_product" && inputs.size() >= 2) {
            float* result = static_cast<float*>(output);

            // For large arrays, we'll use the fused kernel directly
            // TiledExecutor integration is done at the public API level
            *result = simd::FusedDotProduct(static_cast<const float*>(inputs[0]),
                                            static_cast<const float*>(inputs[1]), count);
            return {};
        }

        if (pattern == "squared_distance" && inputs.size() >= 2) {
            float* result = static_cast<float*>(output);
            *result = simd::FusedSquaredDistance(static_cast<const float*>(inputs[0]),
                                                 static_cast<const float*>(inputs[1]), count);
            return {};
        }

        if (pattern == "norm_squared" && inputs.size() >= 1) {
            float* result = static_cast<float*>(output);
            *result = simd::FusedNormSquared(static_cast<const float*>(inputs[0]), count);
            return {};
        }

        if (pattern == "softmax" && inputs.size() >= 1) {
            simd::FusedSoftmax(static_cast<float*>(output), static_cast<const float*>(inputs[0]),
                               count);
            return {};
        }

        if (pattern == "sigmoid" && inputs.size() >= 1) {
            simd::FusedSigmoid(static_cast<float*>(output), static_cast<const float*>(inputs[0]),
                               count);
            return {};
        }

        if (pattern == "relu" && inputs.size() >= 1) {
            simd::FusedRelu(static_cast<float*>(output), static_cast<const float*>(inputs[0]),
                            count);
            return {};
        }

        // Fallback: pattern not recognized
        return Error(ErrorCode::kNotSupported, "Fusion pattern not supported: " + pattern);
    }

    // Get call counter statistics
    CallCounter::Stats getStats() const { return call_counter_.getStats(); }

    // Reset call counters
    void reset() {
        call_counter_.reset();
        if (adaptive_executor_) {
            adaptive_executor_->reset();
        }
    }

    // ==========================================================================
    // Adaptive Execution Mode
    // ==========================================================================

    /// Enable adaptive execution with Thompson Sampling
    void setAdaptiveEnabled(bool enabled) {
        if (enabled && !adaptive_executor_) {
            // Create adaptive executor on first enable
            jit::AdaptiveExecutorConfig config;
            config.enable_persistence = true;
            config.verbose = false;
            adaptive_executor_ = std::make_unique<jit::AdaptiveExecutor>(config);
            spdlog::info("Adaptive execution enabled with Thompson Sampling");
        }
        adaptive_enabled_ = enabled;
    }

    [[nodiscard]] bool isAdaptiveEnabled() const { return adaptive_enabled_; }

    /// Get the adaptive executor (for statistics, etc.)
    [[nodiscard]] jit::AdaptiveExecutor* adaptiveExecutor() { return adaptive_executor_.get(); }
    [[nodiscard]] const jit::AdaptiveExecutor* adaptiveExecutor() const {
        return adaptive_executor_.get();
    }

  private:
    // Generate a unique key for an operation signature
    static uint64_t operationKey(ir::OpCode op, ScalarType dtype, size_t count) {
        // Combine op, dtype, and count bucket into a single key
        // Use count buckets to group similar sizes
        size_t count_bucket = count < 256     ? 0
                              : count < 1024  ? 1
                              : count < 4096  ? 2
                              : count < 16384 ? 3
                                              : 4;

        return (static_cast<uint64_t>(op) << 48) | (static_cast<uint64_t>(dtype) << 40) |
               (count_bucket << 32) | (count & 0xFFFFFFFF);
    }

    // Tier 0: Direct Highway dispatch
    Result<void> executeTier0BinaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                      const void* input_a, const void* input_b, size_t count) {
        // Get Highway function pointer
        void* func_ptr = jit::getHwyFunctionPtr(op, dtype);
        if (!func_ptr) {
            return Error(ErrorCode::kNotSupported, "No Highway implementation for operation");
        }

        // Call the function directly
        using BinaryOpFunc = void (*)(void*, const void*, const void*, size_t);
        auto func = reinterpret_cast<BinaryOpFunc>(func_ptr);
        func(output, input_a, input_b, count);

        return {};
    }

    Result<void> executeTier0UnaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                     const void* input, size_t count) {
        void* func_ptr = jit::getHwyFunctionPtr(op, dtype);
        if (!func_ptr) {
            return Error(ErrorCode::kNotSupported, "No Highway implementation for operation");
        }

        using UnaryOpFunc = void (*)(void*, const void*, size_t);
        auto func = reinterpret_cast<UnaryOpFunc>(func_ptr);
        func(output, input, count);

        return {};
    }

    Result<void> executeTier0FmaOp(ScalarType dtype, void* output, const void* input_a,
                                   const void* input_b, const void* input_c, size_t count) {
        void* func_ptr = jit::getHwyFunctionPtr(ir::OpCode::kFma, dtype);
        if (!func_ptr) {
            return Error(ErrorCode::kNotSupported, "No Highway implementation for FMA");
        }

        using FmaOpFunc = void (*)(void*, const void*, const void*, const void*, size_t);
        auto func = reinterpret_cast<FmaOpFunc>(func_ptr);
        func(output, input_a, input_b, input_c, count);

        return {};
    }

    // Tier 1: Copy-and-patch JIT
    Result<void> executeTier1BinaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                      const void* input_a, const void* input_b, size_t count) {
        return jit::executeJitBinaryOp(op, dtype, output, input_a, input_b, count);
    }

    Result<void> executeTier1UnaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                     const void* input, size_t count) {
        return jit::executeJitUnaryOp(op, dtype, output, input, count);
    }

    CallCounter call_counter_;

    // Adaptive execution with Thompson Sampling
    std::unique_ptr<jit::AdaptiveExecutor> adaptive_executor_;
    bool adaptive_enabled_ = false;
};

// =============================================================================
// Runtime State (Thread-Safe)
// =============================================================================

namespace {

std::mutex g_init_mutex;
std::atomic<bool> g_initialized{false};
RuntimeConfig g_config;
HardwareInfo g_hardware_info;

// Tiered executor instance
std::unique_ptr<TieredExecutor> g_executor;

// Memory optimization components
std::unique_ptr<memory::CacheConfig> g_cache_config;
std::unique_ptr<memory::TiledExecutor> g_tiled_executor;
bool g_tiling_enabled = true;
bool g_prefetch_enabled = true;

// CompilationStats uses read-write lock for better concurrency
std::shared_mutex g_stats_mutex;
CompilationStats g_compilation_stats;

}  // namespace

// =============================================================================
// Initialization
// =============================================================================

Result<void> initialize(const RuntimeConfig& config) {
    // Double-checked locking with mutex for thread safety
    if (g_initialized.load(std::memory_order_acquire)) {
        spdlog::warn("Bud Flow Lang runtime already initialized");
        return {};
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Check again under lock
    if (g_initialized.load(std::memory_order_relaxed)) {
        return {};
    }

    g_config = config;

    // Initialize logging
    if (config.enable_debug_output) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    spdlog::info("Bud Flow Lang v{} initializing...", Version::string());

    // Detect hardware
    g_hardware_info = getHardwareInfo();

    spdlog::info("  SIMD width: {} bytes", g_hardware_info.simd_width);
    spdlog::info("  Cores: {} physical, {} logical", g_hardware_info.physical_cores,
                 g_hardware_info.logical_cores);

    // Initialize JIT compiler
    auto jit_result = jit::initializeCompiler();
    if (!jit_result) {
        spdlog::error("Failed to initialize JIT compiler: {}", jit_result.error().message());
        return jit_result;
    }

    // Get JIT stats for logging
    auto jit_stats = jit::getJitStats();
    spdlog::info("  JIT stencils: {}", jit_stats.stencil_count);
    spdlog::info("  JIT memory: {} KB available", jit_stats.memory_remaining / 1024);

    // Create tiered executor
    g_executor = std::make_unique<TieredExecutor>();

    spdlog::info("  Tiered compilation: enabled");
    spdlog::info("    Tier 0->1 threshold: {} calls", kTier1Threshold);
    spdlog::info("    Tier 1->2 threshold: {} calls", kTier2Threshold);

    // Initialize memory optimization components
    g_cache_config = std::make_unique<memory::CacheConfig>(memory::CacheConfig::detect());
    g_tiled_executor = std::make_unique<memory::TiledExecutor>(*g_cache_config);

    spdlog::info("  Memory optimization: enabled");
    spdlog::info("    L1 cache: {} KB", g_cache_config->l1Size() / 1024);
    spdlog::info("    L2 cache: {} KB", g_cache_config->l2Size() / 1024);
    spdlog::info("    L3 cache: {} KB", g_cache_config->l3Size() / 1024);
    spdlog::info("    Cache line: {} bytes", g_cache_config->lineSize());
    spdlog::info("    Tiling threshold: {} elements", kTilingThreshold);
    spdlog::info("    Prefetch threshold: {} elements", kPrefetchThreshold);

    // Mark as initialized with release semantics
    g_initialized.store(true, std::memory_order_release);

    spdlog::info("Bud Flow Lang initialized successfully");
    return {};
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (!g_initialized.load(std::memory_order_relaxed)) {
        return;  // Not initialized
    }

    spdlog::info("Bud Flow Lang shutting down...");

    // Log final statistics
    if (g_executor) {
        auto stats = g_executor->getStats();
        spdlog::info("  Tiered execution stats:");
        spdlog::info("    Total entries: {}", stats.total_entries);
        spdlog::info("    Tier 0 (Interpreter): {}", stats.tier0_entries);
        spdlog::info("    Tier 1 (CopyPatch): {}", stats.tier1_entries);
        spdlog::info("    Tier 2 (Fused): {}", stats.tier2_entries);
    }

    // Cleanup tiered executor
    g_executor.reset();

    // Cleanup memory optimization components
    g_tiled_executor.reset();
    g_cache_config.reset();

    // Shutdown JIT compiler
    jit::shutdownCompiler();

    g_initialized.store(false, std::memory_order_release);

    spdlog::info("Bud Flow Lang shutdown complete");
}

bool isInitialized() {
    return g_initialized.load(std::memory_order_acquire);
}

// =============================================================================
// Statistics (Thread-Safe)
// =============================================================================

CompilationStats getCompilationStats() {
    std::shared_lock<std::shared_mutex> lock(g_stats_mutex);

    // Merge JIT stats into compilation stats
    auto jit_stats = jit::getJitStats();
    g_compilation_stats.total_compilations = jit_stats.total_compilations;
    g_compilation_stats.code_cache_bytes = jit_stats.memory_used;
    g_compilation_stats.cache_hits = jit_stats.cache_size;

    if (jit_stats.total_compilations > 0) {
        g_compilation_stats.avg_compile_time_ms =
            static_cast<double>(jit_stats.total_compile_time_us) /
            static_cast<double>(jit_stats.total_compilations) / 1000.0;
        g_compilation_stats.total_compile_time_ms =
            static_cast<double>(jit_stats.total_compile_time_us) / 1000.0;
    }

    return g_compilation_stats;
}

void resetCompilationStats() {
    std::unique_lock<std::shared_mutex> lock(g_stats_mutex);
    g_compilation_stats = {};
    if (g_executor) {
        g_executor->reset();
    }
}

// =============================================================================
// Tiered Execution Public API
// =============================================================================

// Get the tiered executor (for internal use)
TieredExecutor* getTieredExecutor() {
    return g_executor.get();
}

// Execute a binary operation using tiered compilation
Result<void> executeBinaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input_a,
                             const void* input_b, size_t count) {
    if (!g_executor) {
        return Error(ErrorCode::kRuntimeError, "Runtime not initialized");
    }
    return g_executor->executeBinaryOp(op, dtype, output, input_a, input_b, count);
}

// Execute a unary operation using tiered compilation
Result<void> executeUnaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input,
                            size_t count) {
    if (!g_executor) {
        return Error(ErrorCode::kRuntimeError, "Runtime not initialized");
    }
    return g_executor->executeUnaryOp(op, dtype, output, input, count);
}

// Execute FMA operation using tiered compilation
Result<void> executeFmaOp(ScalarType dtype, void* output, const void* input_a, const void* input_b,
                          const void* input_c, size_t count) {
    if (!g_executor) {
        return Error(ErrorCode::kRuntimeError, "Runtime not initialized");
    }
    return g_executor->executeFmaOp(dtype, output, input_a, input_b, input_c, count);
}

// Execute a fused operation pattern (Tier 2)
Result<void> executeFusedPattern(const std::string& pattern, void* output,
                                 const std::vector<const void*>& inputs, size_t count) {
    if (!g_executor) {
        return Error(ErrorCode::kRuntimeError, "Runtime not initialized");
    }
    return g_executor->executeFusedChain(pattern, output, inputs, count);
}

// =============================================================================
// Tiered Execution Statistics
// =============================================================================

TieredStats getTieredStats() {
    TieredStats stats;
    if (g_executor) {
        auto counter_stats = g_executor->getStats();
        stats.total_entries = counter_stats.total_entries;
        stats.tier0_entries = counter_stats.tier0_entries;
        stats.tier1_entries = counter_stats.tier1_entries;
        stats.tier2_entries = counter_stats.tier2_entries;
    }
    return stats;
}

// =============================================================================
// Adaptive Execution Public API
// =============================================================================

void setAdaptiveExecutionEnabled(bool enabled) {
    if (!g_executor) {
        spdlog::warn("Cannot enable adaptive execution: runtime not initialized");
        return;
    }
    g_executor->setAdaptiveEnabled(enabled);
}

bool isAdaptiveExecutionEnabled() {
    return g_executor && g_executor->isAdaptiveEnabled();
}

AdaptiveExecutorStats getAdaptiveExecutorStats() {
    AdaptiveExecutorStats stats;
    if (g_executor && g_executor->adaptiveExecutor()) {
        auto executor_stats = g_executor->adaptiveExecutor()->statistics();
        stats.total_executions = executor_stats.total_executions;
        stats.tier0_executions = executor_stats.tier0_executions;
        stats.tier1_executions = executor_stats.tier1_executions;
        stats.tier2_executions = executor_stats.tier2_executions;
        stats.fast_path_executions = executor_stats.fast_path_executions;
        stats.promotions_performed = executor_stats.promotions_performed;
        stats.specializations_performed = executor_stats.specializations_performed;
        stats.avg_execution_time_ns = executor_stats.avg_execution_time_ns;
        stats.total_contexts = executor_stats.cache_stats.total_contexts;
        stats.total_versions = executor_stats.cache_stats.total_versions;
    }
    return stats;
}

void setSmallArrayThreshold(size_t threshold) {
    if (g_executor && g_executor->adaptiveExecutor()) {
        g_executor->adaptiveExecutor()->setSmallArrayThreshold(threshold);
        spdlog::debug("Small array threshold set to {} elements", threshold);
    }
}

size_t getSmallArrayThreshold() {
    if (g_executor && g_executor->adaptiveExecutor()) {
        return g_executor->adaptiveExecutor()->smallArrayThreshold();
    }
    return 10000;  // Default
}

bool saveAdaptiveProfiles() {
    if (g_executor && g_executor->adaptiveExecutor()) {
        return g_executor->adaptiveExecutor()->saveProfiles();
    }
    return false;
}

bool loadAdaptiveProfiles() {
    if (g_executor && g_executor->adaptiveExecutor()) {
        return g_executor->adaptiveExecutor()->loadProfiles();
    }
    return false;
}

// =============================================================================
// Memory Optimization Public API
// =============================================================================

void setTilingEnabled(bool enabled) {
    g_tiling_enabled = enabled;
    spdlog::debug("Tiling {}", enabled ? "enabled" : "disabled");
}

bool isTilingEnabled() {
    return g_tiling_enabled;
}

void setPrefetchEnabled(bool enabled) {
    g_prefetch_enabled = enabled;
    if (g_tiled_executor) {
        g_tiled_executor->setPrefetchEnabled(enabled);
    }
    spdlog::debug("Prefetching {}", enabled ? "enabled" : "disabled");
}

bool isPrefetchEnabled() {
    return g_prefetch_enabled;
}

const memory::CacheConfig* getCacheConfig() {
    return g_cache_config.get();
}

memory::TiledExecutor* getTiledExecutor() {
    return g_tiled_executor.get();
}

// =============================================================================
// Flow Implementation
// =============================================================================

struct Flow::Impl {
    std::string name;
    CompileHint hint;
    std::unique_ptr<ir::IRModule> module;
    size_t call_count = 0;
    ExecutionTier current_tier = ExecutionTier::kInterpreter;
};

Flow::Flow(std::string_view name) : impl_(std::make_unique<Impl>()) {
    impl_->name = std::string(name);
    impl_->module = std::make_unique<ir::IRModule>(impl_->name);
}

Flow::~Flow() = default;

Flow& Flow::hint(const CompileHint& hint) {
    impl_->hint = hint;
    return *this;
}

// =============================================================================
// IR Execution Engine
// =============================================================================

namespace {

// Execute an IR module using the tiered system
Result<Bunch> executeIRModule(ir::IRModule& module, const std::vector<Bunch*>& inputs) {
    if (!g_executor) {
        return Error(ErrorCode::kRuntimeError, "Runtime not initialized");
    }

    // Get execution order from IR
    auto& builder = module.builder();
    ir::ValueId output_id = module.output();

    if (!output_id.isValid()) {
        return Error(ErrorCode::kInvalidInput, "No output defined in IR module");
    }

    // Simple execution: traverse IR and execute each operation
    // This is a placeholder for the full IR interpreter
    // In a complete implementation, this would:
    // 1. Analyze fusion opportunities
    // 2. Generate fused execution plan
    // 3. Execute using appropriate tier

    // For now, just execute the output node
    const ir::IRNode* output_node = builder.getNode(output_id);
    if (!output_node) {
        return Error(ErrorCode::kInvalidInput, "Output node not found");
    }

    // Determine output size from inputs
    size_t output_size = 0;
    if (!inputs.empty()) {
        output_size = inputs[0]->size();
    }

    if (output_size == 0) {
        return Error(ErrorCode::kInvalidInput, "Cannot determine output size");
    }

    // Create output bunch
    auto result = Bunch::zeros(output_size, ScalarType::kFloat32);
    if (!result) {
        return result.error();
    }

    // Return the result (actual computation would happen here based on IR)
    return *result;
}

}  // namespace

// =============================================================================
// Lazy Evaluator Integration
// =============================================================================

namespace {

// Hash an IR subgraph for caching
uint64_t hashIRSubgraph(const ir::IRBuilder& builder, ir::ValueId output) {
    // FNV-1a hash
    uint64_t hash = 0xcbf29ce484222325ULL;

    std::function<void(ir::ValueId)> visit = [&](ir::ValueId id) {
        const ir::IRNode* node = builder.getNode(id);
        if (!node || node->isDead())
            return;

        // Hash opcode
        hash ^= static_cast<uint64_t>(node->opCode());
        hash *= 0x100000001b3ULL;

        // Hash type
        hash ^= static_cast<uint64_t>(node->type().scalarType());
        hash *= 0x100000001b3ULL;

        // Hash operand count
        hash ^= node->numOperands();
        hash *= 0x100000001b3ULL;

        // Recursively hash operands
        for (const auto& operand : node->operands()) {
            visit(operand);
        }
    };

    visit(output);
    return hash;
}

}  // namespace

// =============================================================================
// Advanced Tiered Execution: IR-Level Fusion
// =============================================================================

Result<void> executeWithFusion(ir::IRModule& module) {
    if (!g_executor) {
        return Error(ErrorCode::kRuntimeError, "Runtime not initialized");
    }

    auto& builder = module.builder();
    ir::ValueId output_id = module.output();

    if (!output_id.isValid()) {
        return Error(ErrorCode::kInvalidInput, "No output defined in IR module");
    }

    // Compute hash for tier tracking
    uint64_t ir_hash = hashIRSubgraph(builder, output_id);

    // Run optimization if not already done
    auto opt_result = module.optimize(g_config.jit_optimization_level);
    if (!opt_result) {
        spdlog::warn("IR optimization failed: {}", opt_result.error().message());
        // Continue with unoptimized IR
    }

    // Analyze fusion opportunities
    auto fusions = ir::analyzeFusionOpportunities(module);

    if (!fusions.empty()) {
        spdlog::debug("Found {} fusion opportunities in IR {:016x}", fusions.size(), ir_hash);
        for (const auto& [pattern, speedup] : fusions) {
            spdlog::debug("  - {} ({:.1f}x estimated speedup)", pattern, speedup);
        }
    }

    // For now, this is a framework placeholder
    // Full implementation would:
    // 1. Match fusion patterns to FusedKernel implementations
    // 2. Execute fused patterns using executeFusedChain
    // 3. Fall back to individual operations for non-fused portions

    return {};
}

}  // namespace bud
