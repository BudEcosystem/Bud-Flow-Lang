#pragma once

// =============================================================================
// Bud Flow Lang - Common Definitions
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

// Version information
#define BUD_VERSION_MAJOR 0
#define BUD_VERSION_MINOR 1
#define BUD_VERSION_PATCH 0

namespace bud {

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define BUD_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
    #define BUD_PLATFORM_MACOS 1
#elif defined(__linux__)
    #define BUD_PLATFORM_LINUX 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
    #define BUD_PLATFORM_BSD 1
#elif defined(__EMSCRIPTEN__)
    #define BUD_PLATFORM_WASM 1
#else
    #define BUD_PLATFORM_UNKNOWN 1
#endif

// =============================================================================
// Architecture Detection (Matching Highway's Supported Architectures)
// =============================================================================
// Highway supports: x86 (32/64), ARM (32/64), RISC-V (32/64), WASM, PPC, S390x, EMU128

// Pointer size detection (32-bit vs 64-bit)
#if defined(__LP64__) || defined(_LP64) || defined(_WIN64) || defined(__x86_64__) || \
    defined(__aarch64__) || defined(__ppc64__) || defined(__s390x__) ||              \
    (defined(__riscv) && __riscv_xlen == 64)
    #define BUD_POINTER_SIZE 8
    #define BUD_64BIT 1
#else
    #define BUD_POINTER_SIZE 4
    #define BUD_32BIT 1
#endif

// x86-64 (64-bit x86)
#if defined(__x86_64__) || defined(_M_X64)
    #define BUD_ARCH_X86_64 1
    #define BUD_ARCH_X86 1
    #define BUD_ARCH_NAME "x86_64"
// x86 (32-bit)
#elif defined(__i386__) || defined(_M_IX86) || defined(__i686__)
    #define BUD_ARCH_X86_32 1
    #define BUD_ARCH_X86 1
    #define BUD_ARCH_NAME "x86"
// ARM64 / AArch64
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define BUD_ARCH_ARM64 1
    #define BUD_ARCH_ARM 1
    #define BUD_ARCH_NAME "arm64"
// ARM 32-bit (ARMv7, ARMv8 in 32-bit mode)
#elif defined(__arm__) || defined(_M_ARM)
    #define BUD_ARCH_ARM32 1
    #define BUD_ARCH_ARM 1
    #define BUD_ARCH_NAME "arm32"
// RISC-V 64-bit
#elif defined(__riscv) && __riscv_xlen == 64
    #define BUD_ARCH_RISCV64 1
    #define BUD_ARCH_RISCV 1
    #define BUD_ARCH_NAME "riscv64"
// RISC-V 32-bit
#elif defined(__riscv) && __riscv_xlen == 32
    #define BUD_ARCH_RISCV32 1
    #define BUD_ARCH_RISCV 1
    #define BUD_ARCH_NAME "riscv32"
// WebAssembly
#elif defined(__wasm__) || defined(__EMSCRIPTEN__)
    #define BUD_ARCH_WASM 1
    #define BUD_ARCH_NAME "wasm"
// PowerPC 64-bit
#elif defined(__ppc64__) || defined(__powerpc64__) || defined(_ARCH_PPC64)
    #define BUD_ARCH_PPC64 1
    #define BUD_ARCH_PPC 1
    #define BUD_ARCH_NAME "ppc64"
// PowerPC 32-bit
#elif defined(__ppc__) || defined(__powerpc__) || defined(_ARCH_PPC)
    #define BUD_ARCH_PPC32 1
    #define BUD_ARCH_PPC 1
    #define BUD_ARCH_NAME "ppc32"
// IBM S390x (z/Architecture)
#elif defined(__s390x__)
    #define BUD_ARCH_S390X 1
    #define BUD_ARCH_NAME "s390x"
// Unknown architecture - will use scalar/emulated SIMD
#else
    #define BUD_ARCH_UNKNOWN 1
    #define BUD_ARCH_NAME "unknown"
#endif

// Convenience macro to check if architecture is known
#if !defined(BUD_ARCH_UNKNOWN)
    #define BUD_ARCH_KNOWN 1
#endif

// =============================================================================
// Compiler Attributes
// =============================================================================

#if defined(__GNUC__) || defined(__clang__)
    #define BUD_LIKELY(x) __builtin_expect(!!(x), 1)
    #define BUD_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define BUD_ALWAYS_INLINE __attribute__((always_inline)) inline
    #define BUD_NEVER_INLINE __attribute__((noinline))
    #define BUD_RESTRICT __restrict__
    #define BUD_PREFETCH(addr) __builtin_prefetch(addr)
    #define BUD_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned(ptr, align)
#elif defined(_MSC_VER)
    #define BUD_LIKELY(x) (x)
    #define BUD_UNLIKELY(x) (x)
    #define BUD_ALWAYS_INLINE __forceinline
    #define BUD_NEVER_INLINE __declspec(noinline)
    #define BUD_RESTRICT __restrict
    #define BUD_PREFETCH(addr) ((void)0)
    #define BUD_ASSUME_ALIGNED(ptr, align) (ptr)
#else
    #define BUD_LIKELY(x) (x)
    #define BUD_UNLIKELY(x) (x)
    #define BUD_ALWAYS_INLINE inline
    #define BUD_NEVER_INLINE
    #define BUD_RESTRICT
    #define BUD_PREFETCH(addr) ((void)0)
    #define BUD_ASSUME_ALIGNED(ptr, align) (ptr)
#endif

// =============================================================================
// Memory Alignment
// =============================================================================

// Highway requires 128-byte alignment for maximum compatibility
constexpr size_t kSimdAlignment = 128;
constexpr size_t kCacheLineSize = 64;

// Alignment helper
template <typename T>
constexpr T alignUp(T value, T alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
constexpr bool isAligned(T* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// =============================================================================
// Type Traits
// =============================================================================

// Supported scalar types for SIMD operations
template <typename T>
struct IsSimdScalar : std::false_type {};

template <>
struct IsSimdScalar<float> : std::true_type {};
template <>
struct IsSimdScalar<double> : std::true_type {};
template <>
struct IsSimdScalar<int8_t> : std::true_type {};
template <>
struct IsSimdScalar<int16_t> : std::true_type {};
template <>
struct IsSimdScalar<int32_t> : std::true_type {};
template <>
struct IsSimdScalar<int64_t> : std::true_type {};
template <>
struct IsSimdScalar<uint8_t> : std::true_type {};
template <>
struct IsSimdScalar<uint16_t> : std::true_type {};
template <>
struct IsSimdScalar<uint32_t> : std::true_type {};
template <>
struct IsSimdScalar<uint64_t> : std::true_type {};

template <typename T>
inline constexpr bool kIsSimdScalar = IsSimdScalar<T>::value;

// =============================================================================
// Debug Macros
// =============================================================================

#ifdef NDEBUG
    #define BUD_DEBUG_ONLY(x) ((void)0)
    #define BUD_ASSERT(cond) ((void)0)
#else
    #define BUD_DEBUG_ONLY(x) x
    #define BUD_ASSERT(cond)                                  \
        do {                                                  \
            if (BUD_UNLIKELY(!(cond))) {                      \
                bud::assertFailed(#cond, __FILE__, __LINE__); \
            }                                                 \
        } while (0)
#endif

// Assert failure handler (implemented in error.cc)
[[noreturn]] void assertFailed(const char* cond, const char* file, int line);

// =============================================================================
// Security-Critical Bounds Checks (Always Active, Even in Release)
// =============================================================================
//
// Use BUD_BOUNDS_CHECK for security-critical bounds checks that must remain
// active even in release builds. This prevents buffer overflows and other
// memory safety issues that could be exploited.
//
// Use BUD_ASSERT for general logic assertions that can be disabled in release.

// Bounds check failure handler (implemented in error.cc)
[[noreturn]] void boundsFailed(const char* cond, const char* file, int line);

// Security-critical bounds check - always enabled
#define BUD_BOUNDS_CHECK(cond)                            \
    do {                                                  \
        if (BUD_UNLIKELY(!(cond))) {                      \
            bud::boundsFailed(#cond, __FILE__, __LINE__); \
        }                                                 \
    } while (0)

// Overflow check - always enabled
#define BUD_OVERFLOW_CHECK(cond)                          \
    do {                                                  \
        if (BUD_UNLIKELY(!(cond))) {                      \
            bud::boundsFailed(#cond, __FILE__, __LINE__); \
        }                                                 \
    } while (0)

// =============================================================================
// Utility Types
// =============================================================================

// Non-copyable base class
class NonCopyable {
  protected:
    NonCopyable() = default;
    ~NonCopyable() = default;

    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};

// Non-movable base class
class NonMovable : public NonCopyable {
  protected:
    NonMovable() = default;
    ~NonMovable() = default;

    NonMovable(NonMovable&&) = delete;
    NonMovable& operator=(NonMovable&&) = delete;
};

}  // namespace bud
