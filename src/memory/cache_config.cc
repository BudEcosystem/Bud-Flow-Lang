/**
 * @file cache_config.cc
 * @brief Cache configuration detection implementation
 */

#include "bud_flow_lang/memory/cache_config.h"

#include <algorithm>
#include <fstream>
#include <string>

#if defined(__linux__)
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

namespace bud {
namespace memory {

namespace {

// Default cache sizes if detection fails
constexpr size_t kDefaultL1Size = 32 * 1024;        // 32 KB
constexpr size_t kDefaultL2Size = 256 * 1024;       // 256 KB
constexpr size_t kDefaultL3Size = 8 * 1024 * 1024;  // 8 MB
constexpr size_t kDefaultLineSize = 64;             // 64 bytes

#if defined(__linux__)
// Read cache size from sysfs
size_t readCacheSizeFromSysfs(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }

    std::string line;
    if (!std::getline(file, line)) {
        return 0;
    }

    // Parse size with optional K/M suffix
    size_t size = 0;
    try {
        size_t pos = 0;
        size = std::stoull(line, &pos);
        if (pos < line.size()) {
            char suffix = line[pos];
            if (suffix == 'K' || suffix == 'k') {
                size *= 1024;
            } else if (suffix == 'M' || suffix == 'm') {
                size *= 1024 * 1024;
            }
        }
    } catch (...) {
        return 0;
    }

    return size;
}

// Detect cache configuration on Linux
void detectLinuxCaches(size_t& l1_size, size_t& l2_size, size_t& l3_size, size_t& line_size) {
    // Try sysfs first (most reliable on modern Linux)
    const std::string base = "/sys/devices/system/cpu/cpu0/cache/";

    // Check each cache index
    for (int i = 0; i < 4; ++i) {
        std::string index = base + "index" + std::to_string(i) + "/";

        // Read cache level
        std::ifstream level_file(index + "level");
        if (!level_file.is_open())
            continue;

        int level = 0;
        level_file >> level;

        // Read cache type
        std::ifstream type_file(index + "type");
        std::string type;
        if (type_file.is_open()) {
            std::getline(type_file, type);
        }

        // Read cache size
        size_t size = readCacheSizeFromSysfs(index + "size");

        // Read coherency line size
        std::ifstream line_file(index + "coherency_line_size");
        size_t this_line_size = 0;
        if (line_file.is_open()) {
            line_file >> this_line_size;
            if (this_line_size > 0) {
                line_size = this_line_size;
            }
        }

        // Assign to appropriate level
        // L1 data cache (not instruction cache)
        if (level == 1 && (type == "Data" || type.empty())) {
            l1_size = size;
        } else if (level == 2) {
            l2_size = size;
        } else if (level == 3) {
            l3_size = size;
        }
    }

    // Fallback to sysconf if sysfs didn't work
    if (l1_size == 0) {
        long val = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (val > 0)
            l1_size = static_cast<size_t>(val);
    }
    if (l2_size == 0) {
        long val = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (val > 0)
            l2_size = static_cast<size_t>(val);
    }
    if (l3_size == 0) {
        long val = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (val > 0)
            l3_size = static_cast<size_t>(val);
    }
    if (line_size == 0) {
        long val = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (val > 0)
            line_size = static_cast<size_t>(val);
    }
}

#elif defined(__APPLE__)
// Detect cache configuration on macOS
void detectMacOSCaches(size_t& l1_size, size_t& l2_size, size_t& l3_size, size_t& line_size) {
    size_t size;
    size_t len = sizeof(size);

    // L1 data cache
    if (sysctlbyname("hw.l1dcachesize", &size, &len, nullptr, 0) == 0) {
        l1_size = size;
    }

    // L2 cache
    len = sizeof(size);
    if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0) {
        l2_size = size;
    }

    // L3 cache
    len = sizeof(size);
    if (sysctlbyname("hw.l3cachesize", &size, &len, nullptr, 0) == 0) {
        l3_size = size;
    }

    // Cache line size
    len = sizeof(size);
    if (sysctlbyname("hw.cachelinesize", &size, &len, nullptr, 0) == 0) {
        line_size = size;
    }
}

#elif defined(_WIN32)
// Detect cache configuration on Windows
void detectWindowsCaches(size_t& l1_size, size_t& l2_size, size_t& l3_size, size_t& line_size) {
    DWORD buffer_size = 0;
    GetLogicalProcessorInformation(nullptr, &buffer_size);

    if (buffer_size == 0)
        return;

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

    if (!GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
        return;
    }

    for (const auto& info : buffer) {
        if (info.Relationship == RelationCache) {
            const CACHE_DESCRIPTOR& cache = info.Cache;
            switch (cache.Level) {
            case 1:
                if (cache.Type == CacheData || cache.Type == CacheUnified) {
                    l1_size = cache.Size;
                    line_size = cache.LineSize;
                }
                break;
            case 2:
                l2_size = cache.Size;
                break;
            case 3:
                l3_size = cache.Size;
                break;
            }
        }
    }
}
#endif

}  // namespace

CacheConfig::CacheConfig()
    : l1_size_(kDefaultL1Size),
      l2_size_(kDefaultL2Size),
      l3_size_(kDefaultL3Size),
      line_size_(kDefaultLineSize) {}

CacheConfig::CacheConfig(size_t l1_size, size_t l2_size, size_t l3_size, size_t line_size)
    : l1_size_(l1_size), l2_size_(l2_size), l3_size_(l3_size), line_size_(line_size) {}

CacheConfig CacheConfig::detect() {
    size_t l1_size = 0;
    size_t l2_size = 0;
    size_t l3_size = 0;
    size_t line_size = 0;

#if defined(__linux__)
    detectLinuxCaches(l1_size, l2_size, l3_size, line_size);
#elif defined(__APPLE__)
    detectMacOSCaches(l1_size, l2_size, l3_size, line_size);
#elif defined(_WIN32)
    detectWindowsCaches(l1_size, l2_size, l3_size, line_size);
#endif

    // Apply defaults for any undetected values
    if (l1_size == 0)
        l1_size = kDefaultL1Size;
    if (l2_size == 0)
        l2_size = kDefaultL2Size;
    if (l3_size == 0)
        l3_size = kDefaultL3Size;
    if (line_size == 0)
        line_size = kDefaultLineSize;

    return CacheConfig(l1_size, l2_size, l3_size, line_size);
}

size_t CacheConfig::alignToSimd(size_t elements, size_t element_size) const {
    // Align to SIMD width (128 bytes / element_size)
    const size_t simd_elements = kSimdAlignment / element_size;
    return (elements / simd_elements) * simd_elements;
}

size_t CacheConfig::cacheSizeForLevel(CacheLevel level) const {
    switch (level) {
    case CacheLevel::kL1:
        return l1_size_;
    case CacheLevel::kL2:
        return l2_size_;
    case CacheLevel::kL3:
        return l3_size_;
    default:
        return l1_size_;
    }
}

size_t CacheConfig::optimalTileSize(size_t element_size, size_t num_arrays,
                                    float utilization_factor) const {
    return optimalTileSizeForLevel(element_size, CacheLevel::kL1, num_arrays, utilization_factor);
}

size_t CacheConfig::optimalTileSizeForLevel(size_t element_size, CacheLevel level,
                                            size_t num_arrays, float utilization_factor) const {
    // Calculate how many bytes we can use for arrays
    size_t cache_size = cacheSizeForLevel(level);
    size_t usable_bytes = static_cast<size_t>(static_cast<float>(cache_size) * utilization_factor);

    // Divide among arrays
    size_t bytes_per_array = usable_bytes / num_arrays;

    // Convert to elements
    size_t elements = bytes_per_array / element_size;

    // Align to SIMD width
    elements = alignToSimd(elements, element_size);

    // Ensure at least one SIMD vector
    const size_t min_elements = kSimdAlignment / element_size;
    return std::max(elements, min_elements);
}

size_t CacheConfig::prefetchDistance(size_t element_size) const {
    // Prefetch 8-16 cache lines ahead
    // This is tuned for modern CPUs with out-of-order execution
    constexpr size_t kPrefetchLines = 8;
    size_t distance_bytes = kPrefetchLines * line_size_;

    // Convert to elements
    return distance_bytes / element_size;
}

}  // namespace memory
}  // namespace bud
