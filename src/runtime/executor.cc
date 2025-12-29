// =============================================================================
// Bud Flow Lang - Runtime Executor
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/ir.h"

#include <spdlog/spdlog.h>

#include <atomic>

namespace bud {

// =============================================================================
// Runtime State
// =============================================================================

namespace {
std::atomic<bool> g_initialized{false};
RuntimeConfig g_config;
HardwareInfo g_hardware_info;
CompilationStats g_compilation_stats;
}  // namespace

// =============================================================================
// Initialization
// =============================================================================

Result<void> initialize(const RuntimeConfig& config) {
    if (g_initialized.exchange(true)) {
        spdlog::warn("Bud Flow Lang runtime already initialized");
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
    spdlog::info("  Cores: {} physical, {} logical",
                 g_hardware_info.physical_cores,
                 g_hardware_info.logical_cores);

    // Initialize JIT compiler
    // TODO: Initialize copy-patch compiler
    // TODO: Initialize code cache

    spdlog::info("Bud Flow Lang initialized successfully");
    return {};
}

void shutdown() {
    if (!g_initialized.exchange(false)) {
        return;  // Not initialized
    }

    spdlog::info("Bud Flow Lang shutting down...");

    // Cleanup JIT resources
    // TODO: Shutdown compiler
    // TODO: Clear code cache

    spdlog::info("Bud Flow Lang shutdown complete");
}

bool isInitialized() {
    return g_initialized.load();
}

// =============================================================================
// Statistics
// =============================================================================

CompilationStats getCompilationStats() {
    return g_compilation_stats;
}

void resetCompilationStats() {
    g_compilation_stats = {};
}

// =============================================================================
// Flow Implementation
// =============================================================================

struct Flow::Impl {
    std::string name;
    CompileHint hint;
    ir::IRModule* module = nullptr;
};

Flow::Flow(std::string_view name)
    : impl_(std::make_unique<Impl>()) {
    impl_->name = std::string(name);
}

Flow::~Flow() = default;

Flow& Flow::hint(const CompileHint& hint) {
    impl_->hint = hint;
    return *this;
}

}  // namespace bud
