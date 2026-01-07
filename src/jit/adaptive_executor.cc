// =============================================================================
// Bud Flow Lang - Adaptive Executor Implementation
// =============================================================================

#include "bud_flow_lang/jit/adaptive_executor.h"

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/jit/stencil.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace bud {
namespace jit {

// =============================================================================
// AdaptiveExecutor - Constructor / Destructor
// =============================================================================

AdaptiveExecutor::AdaptiveExecutor(const AdaptiveExecutorConfig& config)
    : config_(config), cache_(config.cache_config) {
    // Load profiles if persistence is enabled
    if (config_.enable_persistence) {
        loadProfiles();
    }
}

AdaptiveExecutor::~AdaptiveExecutor() {
    // Save profiles on destruction if persistence is enabled
    if (config_.enable_persistence) {
        saveProfiles();
    }
}

AdaptiveExecutor::AdaptiveExecutor(AdaptiveExecutor&& other) noexcept
    : config_(std::move(other.config_)), cache_(std::move(other.cache_)), enabled_(other.enabled_) {
    total_executions_.store(other.total_executions_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    tier0_executions_.store(other.tier0_executions_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    tier1_executions_.store(other.tier1_executions_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    tier2_executions_.store(other.tier2_executions_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    fast_path_executions_.store(other.fast_path_executions_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
    promotions_performed_.store(other.promotions_performed_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
}

AdaptiveExecutor& AdaptiveExecutor::operator=(AdaptiveExecutor&& other) noexcept {
    if (this != &other) {
        config_ = std::move(other.config_);
        cache_ = std::move(other.cache_);
        enabled_ = other.enabled_;
        total_executions_.store(other.total_executions_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        tier0_executions_.store(other.tier0_executions_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        tier1_executions_.store(other.tier1_executions_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        tier2_executions_.store(other.tier2_executions_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        fast_path_executions_.store(other.fast_path_executions_.load(std::memory_order_relaxed),
                                    std::memory_order_relaxed);
        promotions_performed_.store(other.promotions_performed_.load(std::memory_order_relaxed),
                                    std::memory_order_relaxed);
    }
    return *this;
}

// =============================================================================
// Configuration
// =============================================================================

void AdaptiveExecutor::setConfig(const AdaptiveExecutorConfig& config) {
    config_ = config;
    cache_.setConfig(config.cache_config);
}

// =============================================================================
// Binary Operation Execution
// =============================================================================

ExecutionResult AdaptiveExecutor::executeBinaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                                  const void* input_a, const void* input_b,
                                                  size_t count) {
    total_executions_.fetch_add(1, std::memory_order_relaxed);

    // FAST PATH: For small arrays, bypass all adaptive overhead
    // Direct Highway call without Thompson Sampling, timing, or context lookup
    if (config_.small_array_threshold > 0 && count < config_.small_array_threshold) {
        fast_path_executions_.fetch_add(1, std::memory_order_relaxed);

        void* func_ptr = jit::getHwyFunctionPtr(op, dtype);
        if (func_ptr) {
            using BinaryOpFunc = void (*)(void*, const void*, const void*, size_t);
            auto func = reinterpret_cast<BinaryOpFunc>(func_ptr);
            func(output, input_a, input_b, count);

            ExecutionResult exec_result;
            exec_result.success = true;
            exec_result.tier_used = ExecutionTier::kInterpreter;
            return exec_result;
        }
        // Fall through to adaptive path if no Highway implementation
    }

    if (!enabled_) {
        // Direct execution without adaptive optimization
        auto start = std::chrono::steady_clock::now();
        auto result = jit::executeJitBinaryOp(op, dtype, output, input_a, input_b, count);
        auto end = std::chrono::steady_clock::now();

        ExecutionResult exec_result;
        exec_result.success = result.hasValue();
        exec_result.execution_time_ns =
            std::chrono::duration<float, std::nano>(end - start).count();
        exec_result.tier_used = ExecutionTier::kCopyPatch;

        if (!result) {
            exec_result.error_message = result.error().message();
        }

        return exec_result;
    }

    // Create context key for this operation
    auto key = makeOperationKey(op, dtype, count);

    // Select version using Thompson Sampling
    auto decision = cache_.selectVersion(key, count);

    // Execute based on selected tier
    auto start = std::chrono::steady_clock::now();
    Result<void> result;

    switch (decision.tier) {
    case ExecutionTier::kInterpreter: {
        // Tier 0: Direct Highway dispatch
        void* func_ptr = jit::getHwyFunctionPtr(op, dtype);
        if (func_ptr) {
            using BinaryOpFunc = void (*)(void*, const void*, const void*, size_t);
            auto func = reinterpret_cast<BinaryOpFunc>(func_ptr);
            func(output, input_a, input_b, count);
            result = {};
        } else {
            result = Error(ErrorCode::kNotSupported, "No Highway implementation");
        }
        recordTierExecution(ExecutionTier::kInterpreter);
        break;
    }

    case ExecutionTier::kCopyPatch:
    case ExecutionTier::kFusedKernel:
        // Tier 1/2: JIT compiled code
        result = jit::executeJitBinaryOp(op, dtype, output, input_a, input_b, count);
        recordTierExecution(decision.tier);
        break;
    }

    auto end = std::chrono::steady_clock::now();
    float time_ns = std::chrono::duration<float, std::nano>(end - start).count();

    // Record execution for learning
    cache_.recordExecution(key, decision.version_idx, count, time_ns);

    // Update total time
    float old_time = total_execution_time_ns_.load(std::memory_order_relaxed);
    total_execution_time_ns_.store(old_time + time_ns, std::memory_order_relaxed);

    // Build result
    ExecutionResult exec_result;
    exec_result.success = result.hasValue();
    exec_result.execution_time_ns = time_ns;
    exec_result.tier_used = decision.tier;
    exec_result.version_idx = decision.version_idx;

    if (!result) {
        exec_result.error_message = result.error().message();
    }

    // Handle promotion if needed
    if (decision.should_promote && config_.enable_promotion) {
        handlePromotion(key, decision);
        exec_result.was_promoted = true;
    }

    return exec_result;
}

ExecutionResult AdaptiveExecutor::executeUnaryOp(ir::OpCode op, ScalarType dtype, void* output,
                                                 const void* input, size_t count) {
    total_executions_.fetch_add(1, std::memory_order_relaxed);

    // FAST PATH: For small arrays, bypass all adaptive overhead
    if (config_.small_array_threshold > 0 && count < config_.small_array_threshold) {
        fast_path_executions_.fetch_add(1, std::memory_order_relaxed);

        void* func_ptr = jit::getHwyFunctionPtr(op, dtype);
        if (func_ptr) {
            using UnaryOpFunc = void (*)(void*, const void*, size_t);
            auto func = reinterpret_cast<UnaryOpFunc>(func_ptr);
            func(output, input, count);

            ExecutionResult exec_result;
            exec_result.success = true;
            exec_result.tier_used = ExecutionTier::kInterpreter;
            return exec_result;
        }
        // Fall through to adaptive path if no Highway implementation
    }

    if (!enabled_) {
        auto start = std::chrono::steady_clock::now();
        auto result = jit::executeJitUnaryOp(op, dtype, output, input, count);
        auto end = std::chrono::steady_clock::now();

        ExecutionResult exec_result;
        exec_result.success = result.hasValue();
        exec_result.execution_time_ns =
            std::chrono::duration<float, std::nano>(end - start).count();
        exec_result.tier_used = ExecutionTier::kCopyPatch;

        if (!result) {
            exec_result.error_message = result.error().message();
        }

        return exec_result;
    }

    // Create context key
    auto key = makeOperationKey(op, dtype, count);
    auto decision = cache_.selectVersion(key, count);

    auto start = std::chrono::steady_clock::now();
    Result<void> result;

    switch (decision.tier) {
    case ExecutionTier::kInterpreter: {
        void* func_ptr = jit::getHwyFunctionPtr(op, dtype);
        if (func_ptr) {
            using UnaryOpFunc = void (*)(void*, const void*, size_t);
            auto func = reinterpret_cast<UnaryOpFunc>(func_ptr);
            func(output, input, count);
            result = {};
        } else {
            result = Error(ErrorCode::kNotSupported, "No Highway implementation");
        }
        recordTierExecution(ExecutionTier::kInterpreter);
        break;
    }

    case ExecutionTier::kCopyPatch:
    case ExecutionTier::kFusedKernel:
        result = jit::executeJitUnaryOp(op, dtype, output, input, count);
        recordTierExecution(decision.tier);
        break;
    }

    auto end = std::chrono::steady_clock::now();
    float time_ns = std::chrono::duration<float, std::nano>(end - start).count();

    cache_.recordExecution(key, decision.version_idx, count, time_ns);

    ExecutionResult exec_result;
    exec_result.success = result.hasValue();
    exec_result.execution_time_ns = time_ns;
    exec_result.tier_used = decision.tier;
    exec_result.version_idx = decision.version_idx;

    if (!result) {
        exec_result.error_message = result.error().message();
    }

    if (decision.should_promote && config_.enable_promotion) {
        handlePromotion(key, decision);
        exec_result.was_promoted = true;
    }

    return exec_result;
}

ExecutionResult AdaptiveExecutor::executeFmaOp(ScalarType dtype, void* output, const void* input_a,
                                               const void* input_b, const void* input_c,
                                               size_t count) {
    total_executions_.fetch_add(1, std::memory_order_relaxed);

    // FAST PATH: For small arrays, bypass all adaptive overhead
    if (config_.small_array_threshold > 0 && count < config_.small_array_threshold) {
        fast_path_executions_.fetch_add(1, std::memory_order_relaxed);

        void* func_ptr = jit::getHwyFunctionPtr(ir::OpCode::kFma, dtype);
        if (func_ptr) {
            using FmaOpFunc = void (*)(void*, const void*, const void*, const void*, size_t);
            auto func = reinterpret_cast<FmaOpFunc>(func_ptr);
            func(output, input_a, input_b, input_c, count);

            ExecutionResult exec_result;
            exec_result.success = true;
            exec_result.tier_used = ExecutionTier::kInterpreter;
            return exec_result;
        }
        // Fall through to adaptive path if no Highway implementation
    }

    if (!enabled_) {
        auto start = std::chrono::steady_clock::now();
        auto result = jit::executeJitFmaOp(dtype, output, input_a, input_b, input_c, count);
        auto end = std::chrono::steady_clock::now();

        ExecutionResult exec_result;
        exec_result.success = result.hasValue();
        exec_result.execution_time_ns =
            std::chrono::duration<float, std::nano>(end - start).count();
        exec_result.tier_used = ExecutionTier::kCopyPatch;

        if (!result) {
            exec_result.error_message = result.error().message();
        }

        return exec_result;
    }

    // FMA always uses specialized key
    auto key = makeOperationKey(ir::OpCode::kFma, dtype, count);
    auto decision = cache_.selectVersion(key, count);

    auto start = std::chrono::steady_clock::now();
    Result<void> result;

    switch (decision.tier) {
    case ExecutionTier::kInterpreter: {
        void* func_ptr = jit::getHwyFunctionPtr(ir::OpCode::kFma, dtype);
        if (func_ptr) {
            using FmaOpFunc = void (*)(void*, const void*, const void*, const void*, size_t);
            auto func = reinterpret_cast<FmaOpFunc>(func_ptr);
            func(output, input_a, input_b, input_c, count);
            result = {};
        } else {
            result = Error(ErrorCode::kNotSupported, "No Highway implementation for FMA");
        }
        recordTierExecution(ExecutionTier::kInterpreter);
        break;
    }

    case ExecutionTier::kCopyPatch:
    case ExecutionTier::kFusedKernel:
        result = jit::executeJitFmaOp(dtype, output, input_a, input_b, input_c, count);
        recordTierExecution(decision.tier);
        break;
    }

    auto end = std::chrono::steady_clock::now();
    float time_ns = std::chrono::duration<float, std::nano>(end - start).count();

    cache_.recordExecution(key, decision.version_idx, count, time_ns);

    ExecutionResult exec_result;
    exec_result.success = result.hasValue();
    exec_result.execution_time_ns = time_ns;
    exec_result.tier_used = decision.tier;
    exec_result.version_idx = decision.version_idx;

    if (!result) {
        exec_result.error_message = result.error().message();
    }

    if (decision.should_promote && config_.enable_promotion) {
        handlePromotion(key, decision);
        exec_result.was_promoted = true;
    }

    return exec_result;
}

// =============================================================================
// Statistics
// =============================================================================

AdaptiveExecutor::ExecutorStats AdaptiveExecutor::statistics() const {
    ExecutorStats stats;

    stats.total_executions = total_executions_.load(std::memory_order_relaxed);
    stats.tier0_executions = tier0_executions_.load(std::memory_order_relaxed);
    stats.tier1_executions = tier1_executions_.load(std::memory_order_relaxed);
    stats.tier2_executions = tier2_executions_.load(std::memory_order_relaxed);
    stats.fast_path_executions = fast_path_executions_.load(std::memory_order_relaxed);
    stats.promotions_performed = promotions_performed_.load(std::memory_order_relaxed);
    stats.specializations_performed = specializations_performed_.load(std::memory_order_relaxed);
    stats.total_execution_time_ns = total_execution_time_ns_.load(std::memory_order_relaxed);

    if (stats.total_executions > 0) {
        stats.avg_execution_time_ns =
            stats.total_execution_time_ns / static_cast<float>(stats.total_executions);
    }

    stats.cache_stats = cache_.statistics();

    return stats;
}

void AdaptiveExecutor::reset() {
    cache_.clear();
    total_executions_.store(0, std::memory_order_relaxed);
    tier0_executions_.store(0, std::memory_order_relaxed);
    tier1_executions_.store(0, std::memory_order_relaxed);
    tier2_executions_.store(0, std::memory_order_relaxed);
    fast_path_executions_.store(0, std::memory_order_relaxed);
    promotions_performed_.store(0, std::memory_order_relaxed);
    specializations_performed_.store(0, std::memory_order_relaxed);
    total_execution_time_ns_.store(0.0f, std::memory_order_relaxed);
}

// =============================================================================
// Profile Persistence
// =============================================================================

bool AdaptiveExecutor::saveProfiles() const {
    std::string path = getProfilePath();
    if (path.empty()) {
        return false;
    }

    // Create directory if needed
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            spdlog::warn("Failed to create profile directory: {}", ec.message());
            return false;
        }
    }

    bool success = cache_.save(path);
    if (success) {
        spdlog::debug("Saved adaptive profiles to: {}", path);
    } else {
        spdlog::warn("Failed to save adaptive profiles to: {}", path);
    }

    return success;
}

bool AdaptiveExecutor::loadProfiles() {
    std::string path = getProfilePath();
    if (path.empty() || !std::filesystem::exists(path)) {
        spdlog::debug("No existing profiles found at: {}", path);
        return false;
    }

    bool success = cache_.load(path);
    if (success) {
        auto stats = cache_.statistics();
        spdlog::info("Loaded {} adaptive profiles from: {}", stats.total_contexts, path);
    } else {
        spdlog::warn("Failed to load adaptive profiles from: {}", path);
    }

    return success;
}

void AdaptiveExecutor::setProfileDir(const std::string& dir) {
    config_.profile_dir = dir;
}

std::string AdaptiveExecutor::getProfilePath() const {
    std::string dir = expandPath(config_.profile_dir);
    if (dir.empty()) {
        return "";
    }

    // Use hardware ID in filename for hardware-specific profiles
    uint32_t hw_id = AdaptiveKernelCache::getHardwareId();
    std::string filename = "profile_" + std::to_string(hw_id) + ".dat";

    return dir + "/" + filename;
}

std::string AdaptiveExecutor::expandPath(const std::string& path) {
    if (path.empty()) {
        return "";
    }

    std::string result = path;

    // Expand ~ to home directory
    if (result[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            result = std::string(home) + result.substr(1);
        } else {
            // Fallback to /tmp
            result = "/tmp" + result.substr(1);
        }
    }

    return result;
}

// =============================================================================
// Internal Helpers
// =============================================================================

void AdaptiveExecutor::handlePromotion(const ContextKey& key, const TierDecision& decision) {
    ExecutionTier new_tier = ExecutionTier::kInterpreter;

    if (decision.tier == ExecutionTier::kInterpreter) {
        new_tier = ExecutionTier::kCopyPatch;
    } else if (decision.tier == ExecutionTier::kCopyPatch) {
        new_tier = ExecutionTier::kFusedKernel;
    } else {
        return;  // Already at max tier
    }

    // Promote the context
    // Note: We don't pass code_ptr here because JIT compilation happens lazily
    // in the actual execution path
    bool promoted = cache_.promoteContext(key, new_tier, nullptr);

    if (promoted) {
        promotions_performed_.fetch_add(1, std::memory_order_relaxed);

        if (config_.verbose) {
            spdlog::info("Promoted kernel to {}: {}", tierName(new_tier), decision.reason);
        }
    }

    // Check for specialization opportunity
    if (decision.should_specialize && config_.enable_specialization) {
        auto dominant_size = cache_.getDominantSize(key);
        if (dominant_size) {
            specializations_performed_.fetch_add(1, std::memory_order_relaxed);

            if (config_.verbose) {
                spdlog::info("Detected dominant size {} for specialization", *dominant_size);
            }
        }
    }
}

ContextKey AdaptiveExecutor::makeOperationKey(ir::OpCode op, ScalarType dtype, size_t count) const {
    // Create a hash from operation type and dtype
    uint64_t ir_hash = 0xcbf29ce484222325ULL;  // FNV-1a basis
    ir_hash ^= static_cast<uint64_t>(op);
    ir_hash *= 0x100000001b3ULL;
    ir_hash ^= static_cast<uint64_t>(dtype);
    ir_hash *= 0x100000001b3ULL;

    return AdaptiveKernelCache::makeKey(ir_hash, dtype, count);
}

void AdaptiveExecutor::recordTierExecution(ExecutionTier tier) {
    switch (tier) {
    case ExecutionTier::kInterpreter:
        tier0_executions_.fetch_add(1, std::memory_order_relaxed);
        break;
    case ExecutionTier::kCopyPatch:
        tier1_executions_.fetch_add(1, std::memory_order_relaxed);
        break;
    case ExecutionTier::kFusedKernel:
        tier2_executions_.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

// =============================================================================
// Global Instance
// =============================================================================

namespace {
std::unique_ptr<AdaptiveExecutor> g_executor;
std::once_flag g_executor_init_flag;
bool g_executor_initialized = false;
}  // namespace

AdaptiveExecutor& getGlobalExecutor() {
    std::call_once(g_executor_init_flag,
                   []() { g_executor = std::make_unique<AdaptiveExecutor>(); });
    return *g_executor;
}

void configureGlobalExecutor(const AdaptiveExecutorConfig& config) {
    getGlobalExecutor().setConfig(config);
}

void initializeAdaptiveExecution() {
    if (g_executor_initialized) {
        return;
    }

    auto& executor = getGlobalExecutor();
    executor.loadProfiles();
    g_executor_initialized = true;

    spdlog::info("Adaptive JIT execution initialized");
}

void shutdownAdaptiveExecution() {
    if (!g_executor_initialized || !g_executor) {
        return;
    }

    // Save profiles before shutdown
    g_executor->saveProfiles();

    // Log final statistics
    auto stats = g_executor->statistics();
    spdlog::info("Adaptive JIT execution shutdown:");
    spdlog::info("  Total executions: {}", stats.total_executions);
    spdlog::info("  Tier 0 (Interpreter): {}", stats.tier0_executions);
    spdlog::info("  Tier 1 (CopyPatch): {}", stats.tier1_executions);
    spdlog::info("  Tier 2 (Fused): {}", stats.tier2_executions);
    spdlog::info("  Promotions: {}", stats.promotions_performed);
    spdlog::info("  Cached contexts: {}", stats.cache_stats.total_contexts);

    g_executor_initialized = false;
}

}  // namespace jit
}  // namespace bud
