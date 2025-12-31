// =============================================================================
// Bud Flow Lang - Persistent Kernel Cache Implementation
// =============================================================================

#include "bud_flow_lang/jit/persistent_cache.h"

#include "bud_flow_lang/ir.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>

namespace bud {
namespace jit {

// =============================================================================
// CacheKey Implementation
// =============================================================================

namespace {

// Compiler version for cache invalidation
constexpr uint32_t kCompilerVersion = 1;

// FNV-1a hash combining
uint64_t hashCombine(uint64_t h, uint64_t value) {
    h ^= value;
    h *= 0x100000001b3ULL;
    return h;
}

// Get current timestamp in microseconds
uint64_t getCurrentTimestamp() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}

}  // namespace

CacheKey::CacheKey()
    : ir_hash_(0),
      dtype_(ScalarType::kUnknown),
      size_class_(SizeClass::kSmall),
      target_features_(detectTargetFeatures()),
      compiler_version_(kCompilerVersion) {}

CacheKey CacheKey::fromIR(const ir::IRModule& module) {
    return fromIR(module.builder());
}

CacheKey CacheKey::fromIR(const ir::IRBuilder& builder) {
    CacheKey key;

    // Compute IR hash
    uint64_t h = 0;
    ScalarType detected_dtype = ScalarType::kUnknown;
    size_t max_elements = 0;

    for (const auto* node : builder.nodes()) {
        if (!node)
            continue;

        // Hash node structure
        h = hashCombine(h, node->id().id);
        h = hashCombine(h, static_cast<uint64_t>(node->opCode()));
        h = hashCombine(h, static_cast<uint64_t>(node->type().scalarType()));
        h = hashCombine(h, node->type().elementCount());

        // Track dtype
        if (detected_dtype == ScalarType::kUnknown) {
            detected_dtype = node->type().scalarType();
        }

        // Track max elements for size class
        max_elements = std::max(max_elements, node->type().elementCount());

        // Hash operands
        for (size_t i = 0; i < node->numOperands(); ++i) {
            h = hashCombine(h, node->operand(i).id);
        }

        // Hash constant values
        if (node->hasAttr("value")) {
            double val = node->floatAttr("value");
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            h = hashCombine(h, bits);
        }
    }

    key.ir_hash_ = h;
    key.dtype_ = detected_dtype;

    // Determine size class
    if (max_elements < 1024) {
        key.size_class_ = SizeClass::kSmall;
    } else if (max_elements < 65536) {
        key.size_class_ = SizeClass::kMedium;
    } else {
        key.size_class_ = SizeClass::kLarge;
    }

    return key;
}

uint32_t CacheKey::detectTargetFeatures() {
    uint32_t features = 0;

    // Feature bits
    constexpr uint32_t kSSE2 = 1 << 0;
    constexpr uint32_t kSSE4_1 = 1 << 1;
    constexpr uint32_t kSSE4_2 = 1 << 2;
    constexpr uint32_t kAVX = 1 << 3;
    constexpr uint32_t kAVX2 = 1 << 4;
    constexpr uint32_t kAVX512F = 1 << 5;
    constexpr uint32_t kNEON = 1 << 6;

#if defined(__SSE2__)
    features |= kSSE2;
#endif
#if defined(__SSE4_1__)
    features |= kSSE4_1;
#endif
#if defined(__SSE4_2__)
    features |= kSSE4_2;
#endif
#if defined(__AVX__)
    features |= kAVX;
#endif
#if defined(__AVX2__)
    features |= kAVX2;
#endif
#if defined(__AVX512F__)
    features |= kAVX512F;
#endif
#if defined(__ARM_NEON)
    features |= kNEON;
#endif

    return features;
}

std::string CacheKey::toString() const {
    char buf[128];
    snprintf(buf, sizeof(buf), "0x%016lx_d%d_s%d_f%08x_v%d", static_cast<unsigned long>(ir_hash_),
             static_cast<int>(dtype_), static_cast<int>(size_class_), target_features_,
             compiler_version_);
    return std::string(buf);
}

size_t CacheKey::hash() const {
    size_t h = std::hash<uint64_t>{}(ir_hash_);
    h ^= std::hash<int>{}(static_cast<int>(dtype_)) << 1;
    h ^= std::hash<int>{}(static_cast<int>(size_class_)) << 2;
    h ^= std::hash<uint32_t>{}(target_features_) << 3;
    h ^= std::hash<uint32_t>{}(compiler_version_) << 4;
    return h;
}

bool CacheKey::operator==(const CacheKey& other) const {
    return ir_hash_ == other.ir_hash_ && dtype_ == other.dtype_ &&
           size_class_ == other.size_class_ && target_features_ == other.target_features_ &&
           compiler_version_ == other.compiler_version_;
}

// =============================================================================
// PersistentKernelCache Implementation
// =============================================================================

PersistentKernelCache::PersistentKernelCache(const std::filesystem::path& cache_dir,
                                             size_t max_size_bytes)
    : cache_dir_(cache_dir), max_size_bytes_(max_size_bytes) {
    std::error_code ec;

    // Create cache directory if it doesn't exist
    if (!std::filesystem::exists(cache_dir_, ec)) {
        if (!std::filesystem::create_directories(cache_dir_, ec)) {
            spdlog::warn("Failed to create cache directory: {}", cache_dir_.string());
            return;
        }
    }

    // Load existing index
    loadIndex();
    initialized_ = true;

    spdlog::debug("PersistentKernelCache initialized: {} entries, {} bytes", index_.size(),
                  current_size_bytes_);
}

PersistentKernelCache::~PersistentKernelCache() {
    if (initialized_) {
        saveIndex();
    }
}

std::filesystem::path PersistentKernelCache::keyToPath(const CacheKey& key) const {
    return cache_dir_ / (key.toString() + ".bin");
}

std::optional<std::vector<uint8_t>> PersistentKernelCache::load(const CacheKey& key) {
    std::shared_lock lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end()) {
        ++misses_;
        return std::nullopt;
    }

    // Read file
    std::ifstream file(it->second.path, std::ios::binary);
    if (!file) {
        ++misses_;
        return std::nullopt;
    }

    // File format: [header_size:4][key_data:header_size][kernel_data:rest]
    uint32_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file || header_size > 1024) {  // Sanity check
        ++misses_;
        return std::nullopt;
    }

    // Skip header (we already have the key)
    file.seekg(sizeof(uint32_t) + header_size);

    // Get remaining size
    auto current_pos = file.tellg();
    file.seekg(0, std::ios::end);
    size_t data_size = static_cast<size_t>(file.tellg()) - static_cast<size_t>(current_pos);
    file.seekg(current_pos);

    // Read kernel data
    std::vector<uint8_t> data(data_size);
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data_size))) {
        ++misses_;
        return std::nullopt;
    }

    ++hits_;

    // Update LRU (need exclusive lock)
    lock.unlock();
    touchEntry(key);

    return data;
}

bool PersistentKernelCache::save(const CacheKey& key, const std::vector<uint8_t>& kernel_data) {
    if (!initialized_) {
        return false;
    }

    std::unique_lock lock(mutex_);

    // Check if we need to evict
    evictIfNeeded(kernel_data.size());

    // Get path for this key
    std::filesystem::path path = keyToPath(key);

    // Write to file with header format: [header_size:4][key_data][kernel_data]
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        spdlog::warn("Failed to open cache file for writing: {}", path.string());
        return false;
    }

    // Create header with key data
    struct KeyHeader {
        uint64_t ir_hash;
        uint8_t dtype;
        uint8_t size_class;
        uint32_t target_features;
        uint32_t compiler_version;
    };

    KeyHeader header;
    header.ir_hash = key.irHash();
    header.dtype = static_cast<uint8_t>(key.dtype());
    header.size_class = static_cast<uint8_t>(key.sizeClass());
    header.target_features = key.targetFeatures();
    header.compiler_version = key.compilerVersion();

    uint32_t header_size = sizeof(KeyHeader);
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(kernel_data.data()),
               static_cast<std::streamsize>(kernel_data.size()));

    if (!file) {
        spdlog::warn("Failed to write to cache file: {}", path.string());
        return false;
    }

    file.close();

    // Update index
    IndexEntry entry;
    entry.path = path;
    entry.size_bytes = kernel_data.size();
    entry.last_access_time = getCurrentTimestamp();

    // Remove old entry size if exists
    auto it = index_.find(key);
    if (it != index_.end()) {
        current_size_bytes_ -= it->second.size_bytes;
    }

    index_[key] = entry;
    current_size_bytes_ += kernel_data.size();

    spdlog::debug("Saved kernel to cache: {} ({} bytes)", key.toString(), kernel_data.size());

    return true;
}

void PersistentKernelCache::touchEntry(const CacheKey& key) {
    std::unique_lock lock(mutex_);

    auto it = index_.find(key);
    if (it != index_.end()) {
        it->second.last_access_time = getCurrentTimestamp();
    }
}

void PersistentKernelCache::evictIfNeeded(size_t new_entry_size) {
    // Already holding lock from caller

    while (current_size_bytes_ + new_entry_size > max_size_bytes_ && !index_.empty()) {
        // Find LRU entry
        auto min_it = index_.begin();
        for (auto it = index_.begin(); it != index_.end(); ++it) {
            if (it->second.last_access_time < min_it->second.last_access_time) {
                min_it = it;
            }
        }

        // Remove file
        std::error_code ec;
        std::filesystem::remove(min_it->second.path, ec);

        // Update stats
        current_size_bytes_ -= min_it->second.size_bytes;

        spdlog::debug("Evicted cache entry: {}", min_it->first.toString());

        // Remove from index
        index_.erase(min_it);
    }
}

void PersistentKernelCache::loadIndex() {
    // Scan cache directory for existing entries
    std::error_code ec;

    if (!std::filesystem::exists(cache_dir_, ec)) {
        return;
    }

    // Header structure for reading
    struct KeyHeader {
        uint64_t ir_hash;
        uint8_t dtype;
        uint8_t size_class;
        uint32_t target_features;
        uint32_t compiler_version;
    };

    for (const auto& entry : std::filesystem::directory_iterator(cache_dir_, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string filename = entry.path().filename().string();
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".bin") {
            continue;
        }

        // Read header from file to reconstruct key
        std::ifstream file(entry.path(), std::ios::binary);
        if (!file) {
            continue;
        }

        uint32_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        if (!file || header_size != sizeof(KeyHeader)) {
            continue;  // Invalid or incompatible format
        }

        KeyHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file) {
            continue;
        }

        // Reconstruct CacheKey
        CacheKey key;
        key.setIrHash(header.ir_hash);
        key.setDtype(static_cast<ScalarType>(header.dtype));
        key.setSizeClass(static_cast<SizeClass>(header.size_class));
        key.setTargetFeatures(header.target_features);
        key.setCompilerVersion(header.compiler_version);

        // Check if key matches current target features and compiler version
        // Skip entries that were compiled for different targets
        if (header.target_features != CacheKey().targetFeatures() ||
            header.compiler_version != CacheKey().compilerVersion()) {
            // Optionally delete outdated entries
            spdlog::debug("Skipping outdated cache entry: {}", entry.path().string());
            continue;
        }

        // Add to index
        IndexEntry idx_entry;
        idx_entry.path = entry.path();
        idx_entry.size_bytes = entry.file_size();
        idx_entry.last_access_time = 0;  // Old entries have low priority

        index_[key] = idx_entry;
        current_size_bytes_ += idx_entry.size_bytes;
    }

    spdlog::debug("Loaded {} entries ({} bytes) from cache directory", index_.size(),
                  current_size_bytes_);
}

void PersistentKernelCache::saveIndex() {
    // In a production implementation, we would save an index file
    // For now, the directory structure itself is the index
}

void PersistentKernelCache::clear() {
    std::unique_lock lock(mutex_);

    std::error_code ec;

    // Remove all cached files
    for (const auto& [key, entry] : index_) {
        std::filesystem::remove(entry.path, ec);
    }

    index_.clear();
    current_size_bytes_ = 0;
    hits_ = 0;
    misses_ = 0;

    spdlog::debug("Cleared persistent cache");
}

CacheStats PersistentKernelCache::stats() const {
    std::shared_lock lock(mutex_);

    CacheStats s;
    s.total_entries = index_.size();
    s.total_bytes = current_size_bytes_;
    s.hits = hits_;
    s.misses = misses_;

    size_t total = hits_ + misses_;
    if (total > 0) {
        s.hit_rate = static_cast<double>(hits_) / total;
    }

    return s;
}

bool PersistentKernelCache::contains(const CacheKey& key) const {
    std::shared_lock lock(mutex_);
    return index_.find(key) != index_.end();
}

// =============================================================================
// Global Cache Instance
// =============================================================================

namespace {
std::unique_ptr<PersistentKernelCache> g_persistent_cache;
std::once_flag g_cache_init_flag;

std::filesystem::path getDefaultCacheDir() {
    // Try XDG_CACHE_HOME first, then HOME/.cache
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache && *xdg_cache) {
        return std::filesystem::path(xdg_cache) / "bud_flow_lang";
    }

    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::filesystem::path(home) / ".cache" / "bud_flow_lang";
    }

    // Fallback to temp directory
    return std::filesystem::temp_directory_path() / "bud_flow_lang_cache";
}
}  // namespace

PersistentKernelCache& getPersistentCache() {
    std::call_once(g_cache_init_flag, []() {
        g_persistent_cache = std::make_unique<PersistentKernelCache>(getDefaultCacheDir());
    });
    return *g_persistent_cache;
}

void initPersistentCache(const std::filesystem::path& cache_dir, size_t max_size_bytes) {
    // Note: This should only be called once at startup
    g_persistent_cache = std::make_unique<PersistentKernelCache>(cache_dir, max_size_bytes);
}

}  // namespace jit
}  // namespace bud
