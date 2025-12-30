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

// Architecture family enum (matching Highway's supported architectures)
enum class ArchFamily : uint8_t {
    kUnknown = 0,
    kX86,     // x86 (32-bit and 64-bit)
    kArm,     // ARM (32-bit and 64-bit)
    kRiscV,   // RISC-V (32-bit and 64-bit)
    kWasm,    // WebAssembly
    kPpc,     // PowerPC
    kS390x,   // IBM z/Architecture
    kScalar,  // Scalar fallback (EMU128)
};

// Get architecture family name
inline std::string_view archFamilyName(ArchFamily family) {
    switch (family) {
    case ArchFamily::kX86:
        return "x86";
    case ArchFamily::kArm:
        return "ARM";
    case ArchFamily::kRiscV:
        return "RISC-V";
    case ArchFamily::kWasm:
        return "WebAssembly";
    case ArchFamily::kPpc:
        return "PowerPC";
    case ArchFamily::kS390x:
        return "S390x";
    case ArchFamily::kScalar:
        return "Scalar";
    default:
        return "Unknown";
    }
}

struct HardwareInfo {
    std::string cpu_name;
    std::string vendor;

    // Architecture information
    ArchFamily arch_family = ArchFamily::kUnknown;
    bool is_64bit = false;
    size_t pointer_size = sizeof(void*);

    // x86 SIMD capabilities (ordered by introduction date)
    bool has_sse2 = false;  // Baseline for x86-64
    bool has_sse3 = false;
    bool has_ssse3 = false;
    bool has_sse4 = false;  // SSE4.1 + SSE4.2
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;          // FMA3 (part of AVX2)
    bool has_avx512 = false;       // AVX-512F foundation
    bool has_avx512_bw = false;    // AVX-512 byte/word
    bool has_avx512_dq = false;    // AVX-512 doubleword/quadword
    bool has_avx512_vl = false;    // AVX-512 vector length extensions
    bool has_avx512_bf16 = false;  // AVX-512 BF16 (Cooper Lake+)
    bool has_avx512_fp16 = false;  // AVX-512 FP16 (Sapphire Rapids+)

    // ARM SIMD capabilities
    bool has_neon = false;          // NEON (AdvSIMD) - baseline for ARM64
    bool has_neon_fp16 = false;     // NEON half-precision support
    bool has_neon_bf16 = false;     // NEON BF16 support (ARMv8.6+)
    bool has_neon_dotprod = false;  // Dot product instructions
    bool has_sve = false;           // Scalable Vector Extension
    bool has_sve2 = false;          // SVE2

    // RISC-V Vector capabilities
    bool has_rvv = false;  // RISC-V Vector extension
    size_t rvv_vlen = 0;   // Vector register length in bits (128, 256, 512, etc.)

    // WebAssembly SIMD capabilities
    bool has_wasm_simd128 = false;  // WebAssembly SIMD 128-bit

    // PowerPC SIMD capabilities
    bool has_vsx = false;   // Vector Scalar Extension (POWER7+)
    bool has_vsx3 = false;  // VSX3 (POWER9+)

    // S390x (z/Architecture) SIMD capabilities
    bool has_z_vector = false;  // z/Architecture Vector Extension

    // Best available SIMD width in bytes
    size_t simd_width = 0;

    // Scalable vector support (SVE, RVV can have variable-width vectors)
    bool has_scalable_vectors = false;
    size_t min_scalable_vector_bits = 0;  // Minimum vector length for scalable
    size_t max_scalable_vector_bits = 0;  // Maximum/current vector length

    // Cache sizes
    size_t l1_cache_size = 0;
    size_t l2_cache_size = 0;
    size_t l3_cache_size = 0;

    // Core count
    int physical_cores = 0;
    int logical_cores = 0;

    // Helper methods
    [[nodiscard]] bool supportsFloat16() const { return has_avx512_fp16 || has_neon_fp16; }

    [[nodiscard]] bool supportsBFloat16() const { return has_avx512_bf16 || has_neon_bf16; }

    [[nodiscard]] bool supportsScalableVectors() const { return has_scalable_vectors; }

    [[nodiscard]] std::string archName() const {
        std::string name(archFamilyName(arch_family));
        name += is_64bit ? " (64-bit)" : " (32-bit)";
        return name;
    }

    [[nodiscard]] std::string simdCapabilitySummary() const;
};

// Get hardware information (thread-safe, cached)
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

// =============================================================================
// Tiered Execution API
// =============================================================================

// Tiered execution statistics
struct TieredStats {
    size_t total_entries = 0;
    size_t tier0_entries = 0;  // Interpreter (direct Highway dispatch)
    size_t tier1_entries = 0;  // Copy-and-patch JIT
    size_t tier2_entries = 0;  // Fused kernels
};

// Get tiered execution statistics
TieredStats getTieredStats();

// Execute a binary operation using tiered compilation
// Automatically promotes hot paths from Tier 0 -> Tier 1 -> Tier 2
Result<void> executeBinaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input_a,
                             const void* input_b, size_t count);

// Execute a unary operation using tiered compilation
Result<void> executeUnaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input,
                            size_t count);

// Execute FMA operation using tiered compilation
Result<void> executeFmaOp(ScalarType dtype, void* output, const void* input_a, const void* input_b,
                          const void* input_c, size_t count);

// Execute a fused operation pattern (Tier 2 optimization)
// Supported patterns: "dot_product", "squared_distance", "norm_squared",
//                     "softmax", "sigmoid", "relu"
Result<void> executeFusedPattern(const std::string& pattern, void* output,
                                 const std::vector<const void*>& inputs, size_t count);

// Execute an IR module with fusion analysis and tiered execution
Result<void> executeWithFusion(ir::IRModule& module);

// =============================================================================
// Memory Optimization API
// =============================================================================

// Forward declarations
namespace memory {
class CacheConfig;
class TiledExecutor;
}  // namespace memory

// Enable/disable cache-aware tiled execution for large arrays
void setTilingEnabled(bool enabled);
bool isTilingEnabled();

// Enable/disable software prefetching
void setPrefetchEnabled(bool enabled);
bool isPrefetchEnabled();

// Get cache configuration (L1/L2/L3 sizes, line size)
// Returns nullptr if runtime not initialized
const memory::CacheConfig* getCacheConfig();

// Get the tiled executor instance
// Returns nullptr if runtime not initialized
memory::TiledExecutor* getTiledExecutor();

}  // namespace bud
