// =============================================================================
// Bud Flow Lang - Hardware Detection Implementation
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <hwy/targets.h>
#include <spdlog/spdlog.h>

#include <thread>

namespace bud {

namespace {

HardwareInfo detectHardware() {
    HardwareInfo info;

    // Detect CPU features using Highway
    const auto targets = hwy::SupportedTargets();

    // Check x86 SIMD support
#if HWY_ARCH_X86
    info.has_sse4 = (targets & HWY_SSE4) != 0;
    info.has_avx2 = (targets & HWY_AVX2) != 0;
    info.has_avx512 = (targets & (HWY_AVX3 | HWY_AVX3_DL | HWY_AVX3_ZEN4 | HWY_AVX3_SPR)) != 0;

    if (info.has_avx512) {
        info.simd_width = 64;  // 512 bits = 64 bytes
    } else if (info.has_avx2) {
        info.simd_width = 32;  // 256 bits = 32 bytes
    } else if (info.has_sse4) {
        info.simd_width = 16;  // 128 bits = 16 bytes
    }
#endif

    // Check ARM SIMD support
#if HWY_ARCH_ARM
    info.has_neon = (targets & HWY_NEON) != 0;
    info.has_sve = (targets & HWY_SVE) != 0;
    info.has_sve2 = (targets & HWY_SVE2) != 0;

    if (info.has_sve2 || info.has_sve) {
        // SVE width is implementation-defined, assume 256-bit minimum
        info.simd_width = 32;
    } else if (info.has_neon) {
        info.simd_width = 16;
    }
#endif

    // Check RISC-V Vector support
#if HWY_ARCH_RISCV
    info.has_rvv = (targets & HWY_RVV) != 0;
    if (info.has_rvv) {
        info.simd_width = 16;  // Minimum VLEN = 128
    }
#endif

    // Get core count
    info.physical_cores = static_cast<int>(std::thread::hardware_concurrency());
    info.logical_cores = info.physical_cores;  // Simplified

    // TODO: Detect cache sizes via CPUID or /proc/cpuinfo

    spdlog::info("Hardware detection: SIMD width = {} bytes", info.simd_width);
    spdlog::info("  SSE4={}, AVX2={}, AVX512={}",
                 info.has_sse4, info.has_avx2, info.has_avx512);
    spdlog::info("  NEON={}, SVE={}, SVE2={}",
                 info.has_neon, info.has_sve, info.has_sve2);
    spdlog::info("  RVV={}", info.has_rvv);

    return info;
}

// Global hardware info (initialized once)
HardwareInfo g_hardware_info;
bool g_hardware_detected = false;

}  // namespace

const HardwareInfo& getHardwareInfo() {
    if (!g_hardware_detected) {
        g_hardware_info = detectHardware();
        g_hardware_detected = true;
    }
    return g_hardware_info;
}

}  // namespace bud
