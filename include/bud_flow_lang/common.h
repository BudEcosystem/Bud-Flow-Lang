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
#else
    #define BUD_PLATFORM_UNKNOWN 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define BUD_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define BUD_ARCH_ARM64 1
#elif defined(__riscv) && __riscv_xlen == 64
    #define BUD_ARCH_RISCV64 1
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
