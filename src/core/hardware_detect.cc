// =============================================================================
// Bud Flow Lang - Hardware Detection Implementation
// =============================================================================
//
// Comprehensive hardware detection supporting all Highway architectures:
// - x86/x86-64: SSE2/3/4, AVX/AVX2, AVX-512 variants
// - ARM32/ARM64: NEON, SVE/SVE2
// - RISC-V: RVV (Vector Extension)
// - WebAssembly: SIMD128
// - PowerPC: VSX
// - S390x: z/Architecture Vector Extension
// - Scalar fallback for unknown architectures
//

#include "bud_flow_lang/bud_flow_lang.h"

#include <hwy/targets.h>

#include <spdlog/spdlog.h>

#include <mutex>
#include <thread>

// =============================================================================
// Highway Target Compatibility Definitions
// =============================================================================
// Define fallbacks for Highway targets that may not exist in all versions.
// These ensure compilation succeeds even with older Highway versions.

#ifndef HWY_SSE2
    #define HWY_SSE2 0
#endif
#ifndef HWY_SSSE3
    #define HWY_SSSE3 0
#endif
#ifndef HWY_SSE4
    #define HWY_SSE4 0
#endif
#ifndef HWY_AVX2
    #define HWY_AVX2 0
#endif
#ifndef HWY_AVX3
    #define HWY_AVX3 0
#endif
#ifndef HWY_AVX3_DL
    #define HWY_AVX3_DL 0
#endif
#ifndef HWY_AVX3_ZEN4
    #define HWY_AVX3_ZEN4 0
#endif
#ifndef HWY_AVX3_SPR
    #define HWY_AVX3_SPR 0
#endif
#ifndef HWY_NEON
    #define HWY_NEON 0
#endif
#ifndef HWY_NEON_BF16
    #define HWY_NEON_BF16 0
#endif
#ifndef HWY_SVE
    #define HWY_SVE 0
#endif
#ifndef HWY_SVE2
    #define HWY_SVE2 0
#endif
#ifndef HWY_RVV
    #define HWY_RVV 0
#endif
#ifndef HWY_WASM
    #define HWY_WASM 0
#endif
#ifndef HWY_WASM_EMU256
    #define HWY_WASM_EMU256 0
#endif
#ifndef HWY_PPC8
    #define HWY_PPC8 0
#endif
#ifndef HWY_PPC9
    #define HWY_PPC9 0
#endif
#ifndef HWY_PPC10
    #define HWY_PPC10 0
#endif
#ifndef HWY_Z14
    #define HWY_Z14 0
#endif
#ifndef HWY_Z15
    #define HWY_Z15 0
#endif
#ifndef HWY_EMU128
    #define HWY_EMU128 0
#endif
#ifndef HWY_SCALAR
    #define HWY_SCALAR 0
#endif

namespace bud {

namespace {

// =============================================================================
// Architecture-Specific Detection
// =============================================================================

void detectX86Features(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kX86;

#if defined(BUD_ARCH_X86_64)
    info.is_64bit = true;
    // SSE2 is baseline for x86-64
    info.has_sse2 = true;
#elif defined(BUD_ARCH_X86_32)
    info.is_64bit = false;
    info.has_sse2 = (targets & HWY_SSE2) != 0;
#endif

    // Check for SSE3/SSSE3/SSE4
    info.has_sse3 = (targets & HWY_SSE2) != 0;  // Highway bundles SSE3 with SSE2
    info.has_ssse3 = (targets & HWY_SSSE3) != 0;
    info.has_sse4 = (targets & HWY_SSE4) != 0;

    // Check for AVX/AVX2/FMA
    info.has_avx = (targets & HWY_AVX2) != 0;  // AVX2 implies AVX
    info.has_avx2 = (targets & HWY_AVX2) != 0;
    info.has_fma = info.has_avx2;  // FMA is part of AVX2 in practice

    // Check for AVX-512 variants
    const int64_t avx512_any = HWY_AVX3 | HWY_AVX3_DL | HWY_AVX3_ZEN4 | HWY_AVX3_SPR;
    info.has_avx512 = (targets & avx512_any) != 0;

    // AVX-512 sub-features (detected via HWY target variants)
    if (info.has_avx512) {
        // AVX3 includes F, CD, BW, DQ, VL
        info.has_avx512_bw = true;
        info.has_avx512_dq = true;
        info.has_avx512_vl = true;

        // AVX3_SPR (Sapphire Rapids) includes FP16
        info.has_avx512_fp16 = (targets & HWY_AVX3_SPR) != 0;

        // BF16 is in AVX3_DL (Cooper Lake) and later
        info.has_avx512_bf16 = (targets & (HWY_AVX3_DL | HWY_AVX3_ZEN4 | HWY_AVX3_SPR)) != 0;
    }

    // Determine SIMD width
    if (info.has_avx512) {
        info.simd_width = 64;  // 512 bits
    } else if (info.has_avx2) {
        info.simd_width = 32;  // 256 bits
    } else if (info.has_sse4) {
        info.simd_width = 16;  // 128 bits
    } else if (info.has_sse2) {
        info.simd_width = 16;  // 128 bits
    } else {
        info.simd_width = 0;  // Scalar fallback
    }
}

void detectArmFeatures(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kArm;

#if defined(BUD_ARCH_ARM64)
    info.is_64bit = true;
    // NEON is baseline for ARM64
    info.has_neon = true;
#elif defined(BUD_ARCH_ARM32)
    info.is_64bit = false;
    info.has_neon = (targets & HWY_NEON) != 0;
#endif

    // Check for SVE/SVE2 (only on ARM64)
#if defined(BUD_ARCH_ARM64)
    info.has_sve = (targets & HWY_SVE) != 0;
    info.has_sve2 = (targets & HWY_SVE2) != 0;

    // SVE/SVE2 provides scalable vectors
    if (info.has_sve || info.has_sve2) {
        info.has_scalable_vectors = true;
        // SVE minimum vector length is 128 bits
        info.min_scalable_vector_bits = 128;
        // Maximum varies by implementation (128, 256, 512, 1024, 2048)
        // We'll query this at runtime if needed; assume 256 for now
        info.max_scalable_vector_bits = 256;
    }

    // Check for ARMv8 advanced features
    // Note: Highway doesn't expose these directly; we check via target support
    if (targets & HWY_NEON_BF16) {
        info.has_neon_bf16 = true;
    }
#endif

    // Half-precision support (ARMv8.2+)
    // Most modern ARM64 chips support this
#if defined(BUD_ARCH_ARM64)
    info.has_neon_fp16 = true;  // Common on ARM64
#endif

    // Determine SIMD width
    if (info.has_sve2 || info.has_sve) {
        // SVE is scalable; report minimum guaranteed width
        info.simd_width = 16;  // 128 bits minimum, but scalable
    } else if (info.has_neon) {
        info.simd_width = 16;  // 128 bits
    } else {
        info.simd_width = 0;  // Scalar fallback
    }
}

void detectRiscVFeatures(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kRiscV;

#if defined(BUD_ARCH_RISCV64)
    info.is_64bit = true;
#elif defined(BUD_ARCH_RISCV32)
    info.is_64bit = false;
#endif

    // Check for RVV (RISC-V Vector extension)
    info.has_rvv = (targets & HWY_RVV) != 0;

    if (info.has_rvv) {
        info.has_scalable_vectors = true;
        // RISC-V V extension minimum VLEN is 128 bits
        info.min_scalable_vector_bits = 128;
        // Default assumption; actual VLEN varies by implementation
        info.rvv_vlen = 128;
        info.max_scalable_vector_bits = info.rvv_vlen;
        info.simd_width = info.rvv_vlen / 8;  // Convert bits to bytes
    } else {
        info.simd_width = 0;  // Scalar fallback
    }
}

void detectWasmFeatures(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kWasm;
    info.is_64bit = false;  // WebAssembly is 32-bit address space
    info.pointer_size = 4;

    // Check for WASM SIMD128
    info.has_wasm_simd128 = (targets & HWY_WASM) != 0;

    if (info.has_wasm_simd128) {
        info.simd_width = 16;  // 128 bits
    } else {
        info.simd_width = 0;  // Scalar fallback
    }
}

void detectPpcFeatures(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kPpc;

#if defined(BUD_ARCH_PPC64)
    info.is_64bit = true;
#elif defined(BUD_ARCH_PPC32)
    info.is_64bit = false;
#endif

    // Check for VSX (Vector Scalar Extension)
    info.has_vsx = (targets & HWY_PPC8) != 0;
    info.has_vsx3 = (targets & HWY_PPC10) != 0;

    if (info.has_vsx) {
        info.simd_width = 16;  // 128 bits
    } else {
        info.simd_width = 0;  // Scalar fallback
    }
}

void detectS390xFeatures(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kS390x;
    info.is_64bit = true;  // S390x is always 64-bit

    // Check for z/Architecture Vector Extension
    info.has_z_vector = (targets & HWY_Z14) != 0 || (targets & HWY_Z15) != 0;

    if (info.has_z_vector) {
        info.simd_width = 16;  // 128 bits
    } else {
        info.simd_width = 0;  // Scalar fallback
    }
}

void detectScalarFallback(HardwareInfo& info, int64_t targets) {
    info.arch_family = ArchFamily::kScalar;

#if defined(BUD_64BIT)
    info.is_64bit = true;
#else
    info.is_64bit = false;
#endif

    // EMU128 provides 128-bit emulated SIMD
    if (targets & HWY_EMU128) {
        info.simd_width = 16;  // Emulated 128 bits
    } else if (targets & HWY_SCALAR) {
        info.simd_width = 0;  // Pure scalar
    } else {
        info.simd_width = 0;
    }
}

// =============================================================================
// Main Detection Function
// =============================================================================

HardwareInfo detectHardware() {
    HardwareInfo info;

    // Set pointer size
    info.pointer_size = sizeof(void*);
    info.is_64bit = (sizeof(void*) == 8);

    // Get Highway's detected targets
    const int64_t targets = hwy::SupportedTargets();

    // Detect features based on compile-time architecture
#if defined(BUD_ARCH_X86)
    detectX86Features(info, targets);
#elif defined(BUD_ARCH_ARM)
    detectArmFeatures(info, targets);
#elif defined(BUD_ARCH_RISCV)
    detectRiscVFeatures(info, targets);
#elif defined(BUD_ARCH_WASM)
    detectWasmFeatures(info, targets);
#elif defined(BUD_ARCH_PPC)
    detectPpcFeatures(info, targets);
#elif defined(BUD_ARCH_S390X)
    detectS390xFeatures(info, targets);
#else
    // Unknown architecture - use scalar fallback
    detectScalarFallback(info, targets);
#endif

    // Get core count
    unsigned int hw_concurrency = std::thread::hardware_concurrency();
    info.physical_cores = (hw_concurrency > 0) ? static_cast<int>(hw_concurrency) : 1;
    info.logical_cores = info.physical_cores;  // Simplified; hyperthreading detection is complex

    // TODO: Detect cache sizes via CPUID (x86), /proc/cpuinfo (Linux), or sysctlbyname (macOS)

    // Log detection results
    spdlog::info("Hardware detection: {} ({})", info.archName(),
                 info.simd_width > 0 ? std::to_string(info.simd_width * 8) + "-bit SIMD"
                                     : "scalar");
    spdlog::debug("  Pointer size: {} bytes", info.pointer_size);
    spdlog::debug("  Physical cores: {}", info.physical_cores);

#if defined(BUD_ARCH_X86)
    spdlog::debug("  x86: SSE2={} SSE4={} AVX2={} AVX512={} FP16={} BF16={}", info.has_sse2,
                  info.has_sse4, info.has_avx2, info.has_avx512, info.has_avx512_fp16,
                  info.has_avx512_bf16);
#elif defined(BUD_ARCH_ARM)
    spdlog::debug("  ARM: NEON={} SVE={} SVE2={} FP16={} BF16={}", info.has_neon, info.has_sve,
                  info.has_sve2, info.has_neon_fp16, info.has_neon_bf16);
#elif defined(BUD_ARCH_RISCV)
    spdlog::debug("  RISC-V: RVV={} VLEN={} bits", info.has_rvv, info.rvv_vlen);
#elif defined(BUD_ARCH_WASM)
    spdlog::debug("  WebAssembly: SIMD128={}", info.has_wasm_simd128);
#elif defined(BUD_ARCH_PPC)
    spdlog::debug("  PowerPC: VSX={} VSX3={}", info.has_vsx, info.has_vsx3);
#elif defined(BUD_ARCH_S390X)
    spdlog::debug("  S390x: Z_VECTOR={}", info.has_z_vector);
#endif

    return info;
}

// =============================================================================
// Thread-Safe Singleton Access
// =============================================================================

// Use std::call_once for thread-safe lazy initialization
std::once_flag g_hardware_init_flag;
HardwareInfo g_hardware_info;

void initializeHardwareInfo() {
    g_hardware_info = detectHardware();
}

}  // namespace

// =============================================================================
// Public API
// =============================================================================

const HardwareInfo& getHardwareInfo() {
    std::call_once(g_hardware_init_flag, initializeHardwareInfo);
    return g_hardware_info;
}

// =============================================================================
// HardwareInfo::simdCapabilitySummary Implementation
// =============================================================================

std::string HardwareInfo::simdCapabilitySummary() const {
    std::string summary;
    summary.reserve(256);

    summary += "Architecture: ";
    summary += archName();
    summary += "\n";

    summary += "SIMD Width: ";
    if (simd_width > 0) {
        summary += std::to_string(simd_width * 8);
        summary += " bits";
        if (has_scalable_vectors) {
            summary += " (scalable: ";
            summary += std::to_string(min_scalable_vector_bits);
            summary += "-";
            summary += std::to_string(max_scalable_vector_bits);
            summary += " bits)";
        }
    } else {
        summary += "scalar only";
    }
    summary += "\n";

    summary += "Features: ";
    std::string features;

    // x86 features
    if (has_sse2)
        features += "SSE2 ";
    if (has_ssse3)
        features += "SSSE3 ";
    if (has_sse4)
        features += "SSE4 ";
    if (has_avx)
        features += "AVX ";
    if (has_avx2)
        features += "AVX2 ";
    if (has_fma)
        features += "FMA ";
    if (has_avx512)
        features += "AVX512 ";
    if (has_avx512_fp16)
        features += "AVX512_FP16 ";
    if (has_avx512_bf16)
        features += "AVX512_BF16 ";

    // ARM features
    if (has_neon)
        features += "NEON ";
    if (has_neon_fp16)
        features += "NEON_FP16 ";
    if (has_neon_bf16)
        features += "NEON_BF16 ";
    if (has_sve)
        features += "SVE ";
    if (has_sve2)
        features += "SVE2 ";

    // RISC-V features
    if (has_rvv) {
        features += "RVV(VLEN=";
        features += std::to_string(rvv_vlen);
        features += ") ";
    }

    // WebAssembly features
    if (has_wasm_simd128)
        features += "WASM_SIMD128 ";

    // PowerPC features
    if (has_vsx)
        features += "VSX ";
    if (has_vsx3)
        features += "VSX3 ";

    // S390x features
    if (has_z_vector)
        features += "Z_VECTOR ";

    if (features.empty()) {
        features = "none (scalar fallback)";
    }

    summary += features;
    summary += "\n";

    summary += "Half-Precision Support: ";
    summary += supportsFloat16() ? "yes" : "no";
    summary += "\n";

    summary += "BFloat16 Support: ";
    summary += supportsBFloat16() ? "yes" : "no";
    summary += "\n";

    return summary;
}

}  // namespace bud
