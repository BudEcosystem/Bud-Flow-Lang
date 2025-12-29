#pragma once

// =============================================================================
// Bud Flow Lang - Main Header
// =============================================================================
//
// Python DSL for SIMD Programming with JIT Compilation
//
// Include this header for full API access.
//

#include "bud_flow_lang/arena.h"
#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/type_system.h"

namespace bud {

// =============================================================================
// Version Information
// =============================================================================

struct Version {
    static constexpr int major = BUD_VERSION_MAJOR;
    static constexpr int minor = BUD_VERSION_MINOR;
    static constexpr int patch = BUD_VERSION_PATCH;

    static std::string string() {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
};

// =============================================================================
// Runtime Initialization
// =============================================================================

struct RuntimeConfig {
    // JIT settings
    int jit_optimization_level = 2;        // 0=none, 1=basic, 2=full, 3=aggressive
    bool enable_async_compilation = true;  // Background tier-2 compilation
    size_t code_cache_size_mb = 64;        // JIT code cache size

    // Memory settings
    size_t arena_initial_size = 64 * 1024;  // 64 KB
    size_t memory_pool_size_mb = 256;       // For large allocations

    // Debug settings
    bool enable_ir_validation = false;  // Validate IR on every build (slow)
    bool enable_debug_output = false;   // Print compilation info
};

// Initialize the runtime (call once at program start)
Result<void> initialize(const RuntimeConfig& config = {});

// Shutdown the runtime (call at program end)
void shutdown();

// Check if runtime is initialized
bool isInitialized();

// =============================================================================
// Hardware Information
// =============================================================================

struct HardwareInfo {
    std::string cpu_name;
    std::string vendor;

    // SIMD capabilities
    bool has_sse4 = false;
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_neon = false;
    bool has_sve = false;
    bool has_sve2 = false;
    bool has_rvv = false;

    // Best available SIMD width in bytes
    size_t simd_width = 0;

    // Cache sizes
    size_t l1_cache_size = 0;
    size_t l2_cache_size = 0;
    size_t l3_cache_size = 0;

    // Core count
    int physical_cores = 0;
    int logical_cores = 0;
};

// Get hardware information
const HardwareInfo& getHardwareInfo();

// =============================================================================
// Compilation Hints (for users who want to tune)
// =============================================================================

struct CompileHint {
    // Unrolling
    int unroll_factor = 0;  // 0 = auto, N = unroll N times

    // Vectorization
    bool force_vectorize = false;
    bool disable_vectorize = false;

    // Tiling (for matrix operations)
    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;

    // Prefetching
    bool enable_prefetch = true;
    int prefetch_distance = 0;  // 0 = auto

    // Target ISA override
    std::string target_isa;  // empty = auto-detect
};

// =============================================================================
// Flow Function Decorator (C++ side)
// =============================================================================

// Mark a computation graph for JIT compilation
class Flow {
  public:
    explicit Flow(std::string_view name = "");
    ~Flow();

    // Configure compilation hints
    Flow& hint(const CompileHint& hint);

    // Build and execute
    template <typename Func>
    Result<Bunch> operator()(Func&& func, const Bunch& input) {
        return execute(std::forward<Func>(func), input);
    }

    template <typename Func>
    Result<Bunch> operator()(Func&& func, const Bunch& a, const Bunch& b) {
        return execute(std::forward<Func>(func), a, b);
    }

  private:
    template <typename Func, typename... Args>
    Result<Bunch> execute(Func&& func, Args&&... args);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Statistics and Profiling
// =============================================================================

struct CompilationStats {
    size_t total_compilations = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    double total_compile_time_ms = 0.0;
    double avg_compile_time_ms = 0.0;
    size_t code_cache_bytes = 0;
};

CompilationStats getCompilationStats();
void resetCompilationStats();

}  // namespace bud
