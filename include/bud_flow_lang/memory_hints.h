// =============================================================================
// Bud Flow Lang - Memory Hints (Developer Tier)
// =============================================================================
//
// Memory hints allow developers to optimize memory access patterns:
//
// - Memory.aligned(data, alignment) - Declare alignment for better loads
// - Memory.prefetch(ptr, distance) - Prefetch data ahead of use
// - Memory.stream_store(ptr, data) - Non-temporal store (bypass cache)
// - Memory.layout(data, order) - Specify memory layout hints
//
// Usage:
//   from flow import Memory, Prefetch
//
//   # Declare alignment for optimal SIMD loads
//   aligned_data = Memory.aligned(data, 64)
//
//   # Prefetch ahead of loop
//   with Prefetch.ahead(distance=4):
//       for chunk in data.chunks(1024):
//           process(chunk)
//
//   # Non-temporal store for write-only patterns
//   Memory.stream_store(output, result)
//
// =============================================================================

#pragma once

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"

#include <cstddef>
#include <cstdint>

namespace bud {

// Forward declarations
class Bunch;

// =============================================================================
// AlignmentHint - Specify memory alignment
// =============================================================================

enum class AlignmentHint : size_t {
    kNone = 1,
    kSimd16 = 16,    // 128-bit SSE
    kSimd32 = 32,    // 256-bit AVX
    kSimd64 = 64,    // 512-bit AVX-512 / cache line
    kSimd128 = 128,  // Full cache line pair
};

// =============================================================================
// PrefetchHint - Prefetch locality hint
// =============================================================================

enum class PrefetchHint {
    kNone,     // No prefetch
    kRead,     // Prefetch for read
    kWrite,    // Prefetch for write
    kNonTemp,  // Non-temporal (will be evicted soon)
};

// =============================================================================
// MemoryLayout - Memory ordering hint
// =============================================================================

enum class MemoryLayout {
    kRowMajor,     // C-style, last index varies fastest
    kColumnMajor,  // Fortran-style, first index varies fastest
    kDefault,      // Use default for data type
};

// =============================================================================
// MemoryHints - Memory optimization hints container
// =============================================================================

struct MemoryHints {
    AlignmentHint alignment = AlignmentHint::kNone;
    PrefetchHint prefetch = PrefetchHint::kNone;
    size_t prefetch_distance = 4;  // Number of iterations ahead
    bool use_streaming_store = false;
    MemoryLayout layout = MemoryLayout::kDefault;
    bool is_read_only = false;
    bool is_write_only = false;
};

// =============================================================================
// Memory - Static utility class for memory operations
// =============================================================================

class Memory {
  public:
    // =========================================================================
    // Alignment
    // =========================================================================

    // Check if pointer is aligned to given boundary
    template <typename T>
    [[nodiscard]] static bool isAligned(const T* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }

    // Get alignment of pointer
    template <typename T>
    [[nodiscard]] static size_t getAlignment(const T* ptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr == 0)
            return 0;

        size_t alignment = 1;
        while ((addr & 1) == 0) {
            alignment *= 2;
            addr >>= 1;
            if (alignment >= 4096)
                break;  // Cap at page size
        }
        return alignment;
    }

    // =========================================================================
    // Prefetching
    // =========================================================================

    // Prefetch memory for read
    static void prefetchRead(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr, 0, 3);  // Read, high locality
#elif defined(_MSC_VER) && defined(_M_X64)
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
    }

    // Prefetch memory for write
    static void prefetchWrite(void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr, 1, 3);  // Write, high locality
#elif defined(_MSC_VER) && defined(_M_X64)
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
    }

    // Prefetch non-temporal (hint: will be used once)
    static void prefetchNonTemporal(const void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr, 0, 0);  // Read, no locality
#elif defined(_MSC_VER) && defined(_M_X64)
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA);
#endif
    }

    // =========================================================================
    // Cache Control
    // =========================================================================

    // Flush cache line containing address
    static void cacheLineFlush(void* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    #if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_clflush(ptr);
    #endif
#endif
    }

    // =========================================================================
    // Streaming Stores
    // =========================================================================

    // Check if streaming stores are beneficial
    // (Use for write-only patterns with large data)
    [[nodiscard]] static bool shouldUseStreamingStore(size_t data_bytes) {
        // Streaming stores are beneficial when data exceeds cache size
        // Assuming 32MB L3 cache as threshold
        constexpr size_t kStreamThreshold = 32 * 1024 * 1024;
        return data_bytes > kStreamThreshold;
    }

    // =========================================================================
    // Memory Traffic Estimation
    // =========================================================================

    // Estimate memory bandwidth requirement
    [[nodiscard]] static size_t estimateBandwidth(size_t elements, size_t element_size, bool read,
                                                  bool write) {
        size_t bytes = elements * element_size;
        size_t traffic = 0;
        if (read)
            traffic += bytes;
        if (write)
            traffic += bytes;
        return traffic;
    }
};

// =============================================================================
// PrefetchScope - RAII guard for prefetch configuration
// =============================================================================

class PrefetchScope {
  public:
    explicit PrefetchScope(size_t distance, PrefetchHint hint = PrefetchHint::kRead)
        : prev_distance_(s_current_distance_), prev_hint_(s_current_hint_) {
        s_current_distance_ = distance;
        s_current_hint_ = hint;
    }

    ~PrefetchScope() {
        s_current_distance_ = prev_distance_;
        s_current_hint_ = prev_hint_;
    }

    // Non-copyable
    PrefetchScope(const PrefetchScope&) = delete;
    PrefetchScope& operator=(const PrefetchScope&) = delete;

    // Get current prefetch settings
    [[nodiscard]] static size_t currentDistance() { return s_current_distance_; }
    [[nodiscard]] static PrefetchHint currentHint() { return s_current_hint_; }

  private:
    size_t prev_distance_;
    PrefetchHint prev_hint_;

    static inline size_t s_current_distance_ = 4;
    static inline PrefetchHint s_current_hint_ = PrefetchHint::kNone;
};

// =============================================================================
// AlignmentChecker - Validate alignment at runtime
// =============================================================================

class AlignmentChecker {
  public:
    template <typename T>
    [[nodiscard]] static Result<void> check(const T* ptr, size_t required_alignment) {
        if (!Memory::isAligned(ptr, required_alignment)) {
            return Error(ErrorCode::kInvalidInput,
                         "Pointer not aligned to " + std::to_string(required_alignment));
        }
        return {};
    }

    // Check Bunch alignment
    [[nodiscard]] static Result<void> check(const Bunch& bunch, size_t required_alignment);
};

}  // namespace bud
