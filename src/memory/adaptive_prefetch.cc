// =============================================================================
// Bud Flow Lang - Adaptive Prefetching Implementation
// =============================================================================

#include "bud_flow_lang/memory/adaptive_prefetch.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

// Platform-specific prefetch intrinsics
#if defined(__x86_64__) || defined(_M_X64)
    #include <xmmintrin.h>  // _mm_prefetch
#elif defined(__aarch64__) || defined(_M_ARM64)
// ARM prefetch handled via __builtin_prefetch
#endif

namespace bud {
namespace memory {

// =============================================================================
// AdaptivePrefetcher Implementation
// =============================================================================

AdaptivePrefetcher::AdaptivePrefetcher()
    : config_(CacheConfig::detect()), base_prefetcher_(config_) {
    // Initialize distance based on cache config
    // Target: prefetch far enough ahead to hide memory latency
    // but not so far that we pollute the cache
    size_t l2_size = config_.l2Size();
    size_t line_size = config_.lineSize() > 0 ? config_.lineSize() : 64;

    // Start with a moderate distance
    distance_ = 8;

    // Adjust based on L2 cache size
    if (l2_size > 1024 * 1024) {
        // Large L2 - can be more aggressive
        distance_ = 16;
        max_distance_ = 64;
    } else if (l2_size > 256 * 1024) {
        // Medium L2
        distance_ = 8;
        max_distance_ = 32;
    } else {
        // Small L2 - be conservative
        distance_ = 4;
        max_distance_ = 16;
    }

    (void)line_size;
}

AdaptivePrefetcher::AdaptivePrefetcher(const CacheConfig& config)
    : config_(config), base_prefetcher_(config) {
    // Same logic as default constructor but with provided config
    size_t l2_size = config_.l2Size();

    distance_ = 8;

    if (l2_size > 1024 * 1024) {
        distance_ = 16;
        max_distance_ = 64;
    } else if (l2_size > 256 * 1024) {
        distance_ = 8;
        max_distance_ = 32;
    } else {
        distance_ = 4;
        max_distance_ = 16;
    }
}

void AdaptivePrefetcher::prefetch(const void* addr, PrefetchHint hint) const {
    if (!enabled_) {
        return;
    }
    issuePrefetch(addr, hint);
}

void AdaptivePrefetcher::prefetchBytes(const void* addr, size_t bytes, PrefetchHint hint) const {
    if (!enabled_ || bytes == 0) {
        return;
    }

    size_t line_size = config_.lineSize() > 0 ? config_.lineSize() : 64;
    const char* ptr = static_cast<const char*>(addr);

    // Prefetch cache lines covering the range
    for (size_t offset = 0; offset < bytes; offset += line_size) {
        issuePrefetch(ptr + offset, hint);
    }
}

void AdaptivePrefetcher::setMinDistance(size_t min_dist) {
    min_distance_ = min_dist;
    if (distance_ < min_distance_) {
        distance_ = min_distance_;
    }
}

void AdaptivePrefetcher::setMaxDistance(size_t max_dist) {
    max_distance_ = max_dist;
    if (distance_ > max_distance_) {
        distance_ = max_distance_;
    }
}

void AdaptivePrefetcher::recordLatency(uint64_t latency_ns) {
    // Exponential moving average
    float latency = static_cast<float>(latency_ns);

    if (latency_samples_ == 0) {
        avg_latency_ = latency;
    } else {
        // EMA: new_avg = (1 - alpha) * old_avg + alpha * new_value
        float alpha = adaptation_rate_;
        avg_latency_ = (1.0f - alpha) * avg_latency_ + alpha * latency;
    }
    ++latency_samples_;
}

float AdaptivePrefetcher::averageLatency() const {
    return avg_latency_;
}

void AdaptivePrefetcher::adjustDistance() {
    if (latency_samples_ < 5) {
        return;  // Not enough samples
    }

    // Calculate adjustment based on current vs target latency
    float target_latency = (kLowLatencyThreshold + kHighLatencyThreshold) / 2.0f;

    if (avg_latency_ > kHighLatencyThreshold) {
        // High latency - increase prefetch distance
        size_t increase = 1;
        if (aggressiveness_ == Aggressiveness::kAggressive) {
            increase = 2;
        }
        distance_ = std::min(distance_ + increase, max_distance_);
        spdlog::debug("Adaptive prefetch: increasing distance to {} (latency: {:.1f}ns)", distance_,
                      avg_latency_);
    } else if (avg_latency_ < kLowLatencyThreshold && distance_ > min_distance_) {
        // Low latency - might be able to decrease distance
        size_t decrease = 1;
        if (aggressiveness_ == Aggressiveness::kAggressive) {
            decrease = 2;
        }
        if (distance_ > decrease + min_distance_) {
            distance_ -= decrease;
        } else {
            distance_ = min_distance_;
        }
        spdlog::debug("Adaptive prefetch: decreasing distance to {} (latency: {:.1f}ns)", distance_,
                      avg_latency_);
    }

    (void)target_latency;

    // Reset samples after adjustment
    latency_samples_ = 0;
}

void AdaptivePrefetcher::primeHardwarePrefetcher(const void* data, size_t stride,
                                                 size_t count) const {
    // Access pattern to train HW prefetcher
    // Touch elements with the given stride to establish a pattern
    const char* ptr = static_cast<const char*>(data);

    // Touch a few elements to establish the pattern
    size_t prime_count = std::min(count, size_t(8));
    volatile char dummy = 0;

    for (size_t i = 0; i < prime_count; ++i) {
        dummy += ptr[i * stride];
    }

    (void)dummy;
}

bool AdaptivePrefetcher::isHardwarePrefetcherActive() const {
    // Heuristic: if average latency is very low, HW prefetcher is likely active
    // This is an approximation - true detection would require perf counters
    return avg_latency_ < kLowLatencyThreshold && latency_samples_ > 10;
}

void AdaptivePrefetcher::setAggressiveness(Aggressiveness level) {
    aggressiveness_ = level;

    // Adjust bounds based on aggressiveness
    switch (level) {
    case Aggressiveness::kConservative:
        adaptation_rate_ = 0.05f;
        max_distance_ = 32;
        break;
    case Aggressiveness::kModerate:
        adaptation_rate_ = 0.1f;
        max_distance_ = 64;
        break;
    case Aggressiveness::kAggressive:
        adaptation_rate_ = 0.2f;
        max_distance_ = 128;
        break;
    }
}

void AdaptivePrefetcher::setAdaptationRate(float rate) {
    adaptation_rate_ = std::clamp(rate, 0.01f, 1.0f);
}

void AdaptivePrefetcher::issuePrefetch(const void* addr, PrefetchHint hint) {
#if defined(__x86_64__) || defined(_M_X64)
    const char* ptr = static_cast<const char*>(addr);
    switch (hint) {
    case PrefetchHint::kT0:
        _mm_prefetch(ptr, _MM_HINT_T0);
        break;
    case PrefetchHint::kT1:
        _mm_prefetch(ptr, _MM_HINT_T1);
        break;
    case PrefetchHint::kT2:
        _mm_prefetch(ptr, _MM_HINT_T2);
        break;
    case PrefetchHint::kNTA:
        _mm_prefetch(ptr, _MM_HINT_NTA);
        break;
    }
#else
    // Fallback using GCC/Clang builtin
    int rw = 0;        // Read
    int locality = 3;  // Default to T0
    switch (hint) {
    case PrefetchHint::kT0:
        locality = 3;
        break;
    case PrefetchHint::kT1:
        locality = 2;
        break;
    case PrefetchHint::kT2:
        locality = 1;
        break;
    case PrefetchHint::kNTA:
        locality = 0;
        break;
    }
    __builtin_prefetch(addr, rw, locality);
#endif
}

}  // namespace memory
}  // namespace bud
