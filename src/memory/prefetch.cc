/**
 * @file prefetch.cc
 * @brief Software prefetching implementation
 */

#include "bud_flow_lang/memory/prefetch.h"

#include <algorithm>

namespace bud {
namespace memory {

namespace {

// Default prefetch distance in cache lines
constexpr size_t kDefaultPrefetchLines = 8;

}  // namespace

Prefetcher::Prefetcher() : Prefetcher(CacheConfig::detect()) {}

Prefetcher::Prefetcher(const CacheConfig& config)
    : config_(config), distance_bytes_(kDefaultPrefetchLines * config.lineSize()), enabled_(true) {}

void Prefetcher::issuePrefetch(const void* addr, PrefetchHint hint) {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang intrinsic
    // __builtin_prefetch(addr, rw, locality)
    // rw: 0 = read, 1 = write
    // locality: 0 = no temporal locality (NTA), 3 = high temporal locality
    switch (hint) {
    case PrefetchHint::kT0:
        __builtin_prefetch(addr, 0, 3);  // All cache levels
        break;
    case PrefetchHint::kT1:
        __builtin_prefetch(addr, 0, 2);  // L2 and above
        break;
    case PrefetchHint::kT2:
        __builtin_prefetch(addr, 0, 1);  // L3 and above
        break;
    case PrefetchHint::kNTA:
        __builtin_prefetch(addr, 0, 0);  // Non-temporal
        break;
    }
#elif defined(_MSC_VER)
    // MSVC intrinsic
    switch (hint) {
    case PrefetchHint::kT0:
        _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
        break;
    case PrefetchHint::kT1:
        _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
        break;
    case PrefetchHint::kT2:
        _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T2);
        break;
    case PrefetchHint::kNTA:
        _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
        break;
    }
#else
    // No-op on unsupported platforms
    (void)addr;
    (void)hint;
#endif
}

void Prefetcher::prefetch(const void* addr, PrefetchHint hint) const {
    if (!enabled_)
        return;
    issuePrefetch(addr, hint);
}

void Prefetcher::prefetchBytes(const void* addr, size_t bytes, PrefetchHint hint) const {
    if (!enabled_)
        return;

    const size_t line_size = config_.lineSize();
    const char* ptr = static_cast<const char*>(addr);

    // Prefetch each cache line in the range
    for (size_t offset = 0; offset < bytes; offset += line_size) {
        issuePrefetch(ptr + offset, hint);
    }
}

size_t Prefetcher::optimalDistance(size_t element_size) const {
    return distance_bytes_ / element_size;
}

size_t Prefetcher::optimalDistanceBytes() const {
    return distance_bytes_;
}

void Prefetcher::setDistance(size_t distance_bytes) {
    // Align to cache line size
    const size_t line_size = config_.lineSize();
    distance_bytes_ = ((distance_bytes + line_size - 1) / line_size) * line_size;

    // Ensure reasonable bounds
    distance_bytes_ = std::max(distance_bytes_, line_size * 4);   // At least 4 lines
    distance_bytes_ = std::min(distance_bytes_, line_size * 64);  // At most 64 lines
}

}  // namespace memory
}  // namespace bud
