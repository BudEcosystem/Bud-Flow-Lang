// =============================================================================
// Bud Flow Lang - Adaptive Kernel Cache Implementation
// =============================================================================

#include "bud_flow_lang/jit/adaptive_kernel_cache.h"

#include "bud_flow_lang/bud_flow_lang.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>

namespace bud {
namespace jit {

// =============================================================================
// Tier Name
// =============================================================================

const char* tierName(ExecutionTier tier) {
    switch (tier) {
    case ExecutionTier::kInterpreter:
        return "Interpreter";
    case ExecutionTier::kCopyPatch:
        return "CopyPatch";
    case ExecutionTier::kFusedKernel:
        return "FusedKernel";
    default:
        return "Unknown";
    }
}

// =============================================================================
// Context Entry Methods
// =============================================================================

void ContextEntry::recordSize(size_t input_size) {
    // Find bucket for this size using log2 bucketing
    size_t bucket_idx = 0;
    if (input_size > 0) {
        // Log2 bucketing: 0-31, 32-63, 64-127, etc.
        size_t log2_size = 63 - __builtin_clzll(input_size | 1);
        bucket_idx = std::min(log2_size / 2, kMaxSizeBuckets - 1);
    }

    // Update histogram
    size_histogram[bucket_idx]++;

    // Track the actual size for this bucket (most recent wins)
    // This allows dominantSize() to return the actual size, not just bucket index
    bucket_sizes[bucket_idx] = input_size;
}

std::optional<size_t> ContextEntry::dominantSize(float threshold) const {
    if (total_executions == 0) {
        return std::nullopt;
    }

    // Find the bucket with most executions
    size_t max_bucket = 0;
    uint32_t max_count = 0;
    for (size_t i = 0; i < kMaxSizeBuckets; ++i) {
        if (size_histogram[i] > max_count) {
            max_count = size_histogram[i];
            max_bucket = i;
        }
    }

    // Check if it exceeds threshold
    float ratio = static_cast<float>(max_count) / static_cast<float>(total_executions);
    if (ratio >= threshold && bucket_sizes[max_bucket] > 0) {
        return bucket_sizes[max_bucket];
    }

    return std::nullopt;
}

bool ContextEntry::shouldSpecialize(size_t min_executions) const {
    if (total_executions < min_executions) {
        return false;
    }

    // Check if one size dominates (>50% of calls)
    return dominantSize(0.5f).has_value();
}

// =============================================================================
// AdaptiveKernelCache - Constructor / Destructor
// =============================================================================

AdaptiveKernelCache::AdaptiveKernelCache(const AdaptiveCacheConfig& config) : config_(config) {
    // Seed RNG
    std::random_device rd;
    rng_.seed(rd());
}

AdaptiveKernelCache::~AdaptiveKernelCache() = default;

AdaptiveKernelCache::AdaptiveKernelCache(AdaptiveKernelCache&& other) noexcept
    : config_(std::move(other.config_)),
      entries_(std::move(other.entries_)),
      rng_(std::move(other.rng_)) {
    total_selections_.store(other.total_selections_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    total_executions_.store(other.total_executions_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
}

AdaptiveKernelCache& AdaptiveKernelCache::operator=(AdaptiveKernelCache&& other) noexcept {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        config_ = std::move(other.config_);
        entries_ = std::move(other.entries_);
        rng_ = std::move(other.rng_);
        total_selections_.store(other.total_selections_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        total_executions_.store(other.total_executions_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
    }
    return *this;
}

// =============================================================================
// Configuration
// =============================================================================

void AdaptiveKernelCache::setConfig(const AdaptiveCacheConfig& config) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    config_ = config;
}

// =============================================================================
// Context Management
// =============================================================================

ContextEntry& AdaptiveKernelCache::getOrCreateContext(const ContextKey& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        it->second.last_accessed_at = std::chrono::steady_clock::now();
        return it->second;
    }

    // Check if we need to evict
    evictContextIfNeeded();

    // Create new entry
    ContextEntry entry;
    entry.key = key;
    entry.created_at = std::chrono::steady_clock::now();
    entry.last_accessed_at = entry.created_at;

    // Add default interpreter version
    KernelVersion default_version;
    default_version.id = 0;
    default_version.tier = ExecutionTier::kInterpreter;
    default_version.thompson_state.alpha = config_.prior_alpha;
    default_version.thompson_state.beta = config_.prior_beta;
    default_version.created_at = entry.created_at;
    entry.versions.push_back(std::move(default_version));

    auto [inserted_it, success] = entries_.emplace(key, std::move(entry));
    return inserted_it->second;
}

bool AdaptiveKernelCache::hasContext(const ContextKey& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return entries_.count(key) > 0;
}

const ContextEntry* AdaptiveKernelCache::getContext(const ContextKey& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return &it->second;
    }
    return nullptr;
}

// =============================================================================
// Version Management
// =============================================================================

size_t AdaptiveKernelCache::addVersion(const ContextKey& key, KernelVersion version) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        // Create context first (release and re-acquire lock via getOrCreateContext)
        lock.unlock();
        getOrCreateContext(key);
        lock.lock();
        it = entries_.find(key);
    }

    auto& entry = it->second;

    // Check version limit
    evictVersionIfNeeded(entry);

    // Assign ID and add
    version.id = static_cast<uint32_t>(entry.versions.size());
    version.thompson_state.alpha = config_.prior_alpha;
    version.thompson_state.beta = config_.prior_beta;
    version.created_at = std::chrono::steady_clock::now();

    entry.versions.push_back(std::move(version));
    return entry.versions.size() - 1;
}

TierDecision AdaptiveKernelCache::selectVersion(const ContextKey& key, size_t input_size) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    TierDecision decision;

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        // Create default entry
        lock.unlock();
        (void)getOrCreateContext(key);
        lock.lock();

        decision.tier = ExecutionTier::kInterpreter;
        decision.version_idx = 0;
        decision.call_count = 0;
        return decision;
    }

    auto& entry = it->second;
    entry.total_executions++;
    entry.last_accessed_at = std::chrono::steady_clock::now();

    // Record size for specialization analysis
    if (input_size > 0) {
        entry.recordSize(input_size);
    }

    total_selections_.fetch_add(1, std::memory_order_relaxed);

    // During warmup, round-robin through versions
    size_t valid_versions = 0;
    for (const auto& v : entry.versions) {
        if (v.canHandle(input_size)) {
            valid_versions++;
        }
    }

    if (entry.total_executions <= config_.warmup_executions * valid_versions) {
        // Round-robin warmup
        size_t idx = (entry.total_executions - 1) % entry.versions.size();
        const auto& version = entry.versions[idx];

        decision.tier = version.tier;
        decision.version_idx = idx;
        decision.call_count = entry.total_executions;
        decision.reason = "warmup round-robin";
        return decision;
    }

    // Thompson Sampling: sample from each version's posterior
    float best_sample = -1.0f;
    size_t best_idx = 0;

    for (size_t i = 0; i < entry.versions.size(); ++i) {
        const auto& version = entry.versions[i];

        // Skip versions that can't handle this size
        if (!version.canHandle(input_size)) {
            continue;
        }

        const auto& state = version.thompson_state;

        // Give exploration bonus to under-sampled versions
        float sample = state.sample(rng_);
        if (state.num_samples < config_.min_samples_per_version) {
            float bonus = config_.exploration_bonus *
                          static_cast<float>(config_.min_samples_per_version - state.num_samples);
            sample += bonus;
        }

        if (sample > best_sample) {
            best_sample = sample;
            best_idx = i;
        }
    }

    const auto& selected = entry.versions[best_idx];

    decision.tier = selected.tier;
    decision.version_idx = best_idx;
    decision.call_count = entry.total_executions;
    decision.thompson_variance = selected.thompson_state.variance();

    if (best_idx == entry.best_version) {
        decision.reason = "exploitation";
    } else {
        decision.reason = "exploration";
    }

    // Check for promotion opportunity
    if (selected.tier == ExecutionTier::kInterpreter &&
        entry.total_executions >= config_.min_calls_tier1 &&
        selected.thompson_state.variance() < config_.variance_threshold) {
        decision.should_promote = true;
        decision.reason = "ready for tier1 promotion";
    } else if (selected.tier == ExecutionTier::kCopyPatch &&
               entry.total_executions >= config_.min_calls_tier2 &&
               selected.thompson_state.variance() < config_.variance_threshold) {
        decision.should_promote = true;
        decision.reason = "ready for tier2 promotion";
    }

    // Check for specialization opportunity
    if (!selected.is_size_specialized &&
        entry.shouldSpecialize(config_.min_executions_specialize)) {
        decision.should_specialize = true;
    }

    return decision;
}

const KernelVersion* AdaptiveKernelCache::getVersion(const ContextKey& key,
                                                     size_t version_idx) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it == entries_.end() || version_idx >= it->second.versions.size()) {
        return nullptr;
    }

    return &it->second.versions[version_idx];
}

void AdaptiveKernelCache::recordExecution(const ContextKey& key, size_t version_idx,
                                          size_t input_size, float time_ns) {
    if (time_ns <= 0.0f) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it == entries_.end() || version_idx >= it->second.versions.size()) {
        return;
    }

    auto& entry = it->second;
    auto& version = entry.versions[version_idx];

    // Update version statistics
    version.executions++;
    version.total_time_ns += static_cast<uint64_t>(time_ns);
    version.min_time_ns = std::min(version.min_time_ns, time_ns);
    version.max_time_ns = std::max(version.max_time_ns, time_ns);
    version.last_executed_at = std::chrono::steady_clock::now();

    // Exponential moving average for avg_time
    if (version.avg_time_ns == 0.0f) {
        version.avg_time_ns = time_ns;
    } else {
        version.avg_time_ns =
            config_.time_decay * version.avg_time_ns + (1.0f - config_.time_decay) * time_ns;
    }

    // Update entry totals
    entry.cumulative_time_ns += time_ns;

    // Normalize reward and update Thompson state
    float reward = normalizeReward(time_ns, entry);
    version.thompson_state.update(reward);

    total_executions_.fetch_add(1, std::memory_order_relaxed);

    // Check if this version is now the best
    if (version_idx != entry.best_version) {
        float current_mean = version.thompson_state.mean();
        float best_mean = entry.versions[entry.best_version].thompson_state.mean();

        if (current_mean > best_mean * 1.05f) {  // 5% improvement threshold
            entry.best_version = version_idx;
        }
    }
}

// =============================================================================
// Tier Promotion
// =============================================================================

bool AdaptiveKernelCache::shouldPromote(const ContextKey& key) const {
    return getPromotionDecision(key).should_promote;
}

TierDecision AdaptiveKernelCache::getPromotionDecision(const ContextKey& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    TierDecision decision;

    auto it = entries_.find(key);
    if (it == entries_.end() || it->second.versions.empty()) {
        decision.reason = "context not found";
        return decision;
    }

    const auto& entry = it->second;
    const auto& best = entry.versions[entry.best_version];

    decision.tier = best.tier;
    decision.version_idx = entry.best_version;
    decision.call_count = entry.total_executions;
    decision.thompson_variance = best.thompson_state.variance();

    // Check promotion criteria
    if (best.tier == ExecutionTier::kInterpreter) {
        // Tier 0 -> Tier 1 promotion
        if (entry.total_executions >= config_.min_calls_tier1 &&
            best.thompson_state.variance() < config_.variance_threshold) {
            decision.should_promote = true;
            decision.reason = "confident variance, sufficient calls for tier1";
        } else if (entry.total_executions < config_.min_calls_tier1) {
            decision.reason = "insufficient calls for tier1";
        } else {
            decision.reason = "high variance, need more exploration";
        }
    } else if (best.tier == ExecutionTier::kCopyPatch) {
        // Tier 1 -> Tier 2 promotion
        if (entry.total_executions >= config_.min_calls_tier2 &&
            best.thompson_state.variance() < config_.variance_threshold) {
            // Also check for specialization opportunity
            if (entry.shouldSpecialize(config_.min_executions_specialize)) {
                decision.should_promote = true;
                decision.should_specialize = true;
                decision.reason = "ready for tier2 with size specialization";
            } else {
                decision.should_promote = true;
                decision.reason = "confident variance, sufficient calls for tier2";
            }
        } else if (entry.total_executions < config_.min_calls_tier2) {
            decision.reason = "insufficient calls for tier2";
        } else {
            decision.reason = "high variance, need more exploration";
        }
    } else {
        decision.reason = "already at maximum tier";
    }

    return decision;
}

bool AdaptiveKernelCache::promoteContext(const ContextKey& key, ExecutionTier new_tier,
                                         void* code_ptr) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }

    auto& entry = it->second;

    // Create a new version with the higher tier
    KernelVersion new_version;
    new_version.id = static_cast<uint32_t>(entry.versions.size());
    new_version.tier = new_tier;
    new_version.code_ptr = code_ptr;
    new_version.thompson_state.alpha = config_.prior_alpha;
    new_version.thompson_state.beta = config_.prior_beta;
    new_version.created_at = std::chrono::steady_clock::now();

    entry.versions.push_back(std::move(new_version));

    return true;
}

// =============================================================================
// Specialization
// =============================================================================

bool AdaptiveKernelCache::shouldSpecialize(const ContextKey& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }

    return it->second.shouldSpecialize(config_.min_executions_specialize);
}

std::optional<size_t> AdaptiveKernelCache::getDominantSize(const ContextKey& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return std::nullopt;
    }

    return it->second.dominantSize(config_.dominant_size_ratio);
}

// =============================================================================
// Statistics
// =============================================================================

AdaptiveKernelCache::CacheStats AdaptiveKernelCache::statistics() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    CacheStats stats;
    stats.total_contexts = entries_.size();
    stats.total_selections = total_selections_.load(std::memory_order_relaxed);
    stats.total_executions = total_executions_.load(std::memory_order_relaxed);

    for (const auto& [key, entry] : entries_) {
        stats.total_versions += entry.versions.size();

        // Count by tier (use best version's tier)
        if (!entry.versions.empty()) {
            ExecutionTier tier = entry.versions[entry.best_version].tier;
            switch (tier) {
            case ExecutionTier::kInterpreter:
                stats.tier0_contexts++;
                break;
            case ExecutionTier::kCopyPatch:
                stats.tier1_contexts++;
                break;
            case ExecutionTier::kFusedKernel:
                stats.tier2_contexts++;
                break;
            }
        }

        // Count specialized versions
        for (const auto& version : entry.versions) {
            if (version.is_size_specialized) {
                stats.specialized_versions++;
            }
        }
    }

    return stats;
}

void AdaptiveKernelCache::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    entries_.clear();
    total_selections_.store(0, std::memory_order_relaxed);
    total_executions_.store(0, std::memory_order_relaxed);
}

// =============================================================================
// Persistence
// =============================================================================

std::string AdaptiveKernelCache::serialize() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::ostringstream ss;
    ss << "AdaptiveKernelCache v1\n";
    ss << entries_.size() << "\n";

    for (const auto& [key, entry] : entries_) {
        // Serialize key
        ss << key.ir_hash << " " << static_cast<int>(key.dtype) << " " << key.size_bucket << " "
           << key.hardware_id << "\n";

        // Serialize versions
        ss << entry.versions.size() << "\n";
        for (const auto& version : entry.versions) {
            ss << version.id << " " << static_cast<int>(version.tier) << " " << version.executions
               << " " << version.avg_time_ns << " " << version.thompson_state.alpha << " "
               << version.thompson_state.beta << " " << version.thompson_state.num_samples << " "
               << version.is_size_specialized << " " << version.min_size << " " << version.max_size
               << "\n";
        }

        // Serialize entry metadata
        ss << entry.best_version << " " << entry.total_executions << " " << entry.cumulative_time_ns
           << "\n";
    }

    return ss.str();
}

bool AdaptiveKernelCache::deserialize(const std::string& data) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    std::istringstream ss(data);
    std::string header;
    std::getline(ss, header);

    if (header != "AdaptiveKernelCache v1") {
        return false;
    }

    size_t num_entries = 0;
    ss >> num_entries;
    ss.ignore();

    entries_.clear();

    for (size_t e = 0; e < num_entries; ++e) {
        ContextKey key;
        int dtype_int = 0;
        ss >> key.ir_hash >> dtype_int >> key.size_bucket >> key.hardware_id;
        key.dtype = static_cast<ScalarType>(dtype_int);

        ContextEntry entry;
        entry.key = key;
        entry.created_at = std::chrono::steady_clock::now();
        entry.last_accessed_at = entry.created_at;

        size_t num_versions = 0;
        ss >> num_versions;

        for (size_t v = 0; v < num_versions; ++v) {
            KernelVersion version;
            int tier_int = 0;
            int is_specialized = 0;

            ss >> version.id >> tier_int >> version.executions >> version.avg_time_ns >>
                version.thompson_state.alpha >> version.thompson_state.beta >>
                version.thompson_state.num_samples >> is_specialized >> version.min_size >>
                version.max_size;

            version.tier = static_cast<ExecutionTier>(tier_int);
            version.is_size_specialized = (is_specialized != 0);
            version.created_at = entry.created_at;

            entry.versions.push_back(std::move(version));
        }

        ss >> entry.best_version >> entry.total_executions >> entry.cumulative_time_ns;

        entries_[key] = std::move(entry);
    }

    return true;
}

bool AdaptiveKernelCache::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        return false;
    }
    file << serialize();
    return file.good();
}

bool AdaptiveKernelCache::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return deserialize(ss.str());
}

// =============================================================================
// Hardware Detection
// =============================================================================

uint32_t AdaptiveKernelCache::getHardwareId() {
    // Create a fingerprint from hardware info
    const auto& hw = getHardwareInfo();

    uint32_t id = 0;
    id |= (hw.simd_width & 0xFF);                                 // Bits 0-7: SIMD width
    id |= ((hw.physical_cores & 0xFF) << 8);                      // Bits 8-15: Physical cores
    id |= ((static_cast<uint32_t>(hw.arch_family) & 0xF) << 16);  // Bits 16-19: Architecture

    // Add L1 cache size (normalized to KB)
    if (hw.l1_cache_size > 0) {
        id |= (((hw.l1_cache_size / 1024) & 0x7F) << 20);  // Bits 20-26: L1 size (KB)
    }

    return id;
}

ContextKey AdaptiveKernelCache::makeKey(uint64_t ir_hash, ScalarType dtype, size_t input_size) {
    ContextKey key;
    key.ir_hash = ir_hash;
    key.dtype = dtype;
    key.hardware_id = getHardwareId();

    // Size bucket: 0 = <256, 1 = <1K, 2 = <4K, 3 = <16K, 4 = >=16K
    if (input_size < 256) {
        key.size_bucket = 0;
    } else if (input_size < 1024) {
        key.size_bucket = 1;
    } else if (input_size < 4096) {
        key.size_bucket = 2;
    } else if (input_size < 16384) {
        key.size_bucket = 3;
    } else {
        key.size_bucket = 4;
    }

    return key;
}

// =============================================================================
// Internal Helpers
// =============================================================================

void AdaptiveKernelCache::evictContextIfNeeded() {
    if (entries_.size() < config_.max_context_entries) {
        return;
    }

    // Find LRU context in tier 0
    auto lru_it = entries_.end();
    auto oldest_time = std::chrono::steady_clock::time_point::max();

    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const auto& entry = it->second;

        // Only evict tier 0 contexts
        if (!entry.versions.empty() &&
            entry.versions[entry.best_version].tier == ExecutionTier::kInterpreter) {
            if (entry.last_accessed_at < oldest_time) {
                oldest_time = entry.last_accessed_at;
                lru_it = it;
            }
        }
    }

    if (lru_it != entries_.end()) {
        entries_.erase(lru_it);
    }
}

void AdaptiveKernelCache::evictVersionIfNeeded(ContextEntry& entry) {
    if (entry.versions.size() < config_.max_versions_per_context) {
        return;
    }

    // Find version with lowest Thompson mean (excluding best)
    size_t evict_idx = 0;
    float lowest_mean = std::numeric_limits<float>::max();

    for (size_t i = 0; i < entry.versions.size(); ++i) {
        if (i == entry.best_version) {
            continue;
        }

        float mean = entry.versions[i].thompson_state.mean();
        if (mean < lowest_mean) {
            lowest_mean = mean;
            evict_idx = i;
        }
    }

    entry.versions.erase(entry.versions.begin() + static_cast<ptrdiff_t>(evict_idx));

    // Adjust best_version if needed
    if (entry.best_version >= entry.versions.size()) {
        entry.best_version = entry.versions.size() - 1;
    } else if (entry.best_version > evict_idx) {
        entry.best_version--;
    }
}

float AdaptiveKernelCache::normalizeReward(float time_ns, const ContextEntry& entry) const {
    // Find best (min) and worst (max) times across versions
    float min_time = std::numeric_limits<float>::max();
    float max_time = 0.0f;

    for (const auto& version : entry.versions) {
        if (version.executions > 0) {
            min_time = std::min(min_time, version.min_time_ns);
            max_time = std::max(max_time, version.max_time_ns);
        }
    }

    if (min_time >= max_time || max_time == 0.0f) {
        return 0.5f;  // No differentiation possible
    }

    // Reward = 1 - normalized_time (lower time = higher reward)
    float normalized = (time_ns - min_time) / (max_time - min_time);
    normalized = std::clamp(normalized, 0.0f, 1.0f);

    return 1.0f - normalized;
}

size_t AdaptiveKernelCache::getSizeBucket(size_t input_size) const {
    if (input_size < 256)
        return 0;
    if (input_size < 1024)
        return 1;
    if (input_size < 4096)
        return 2;
    if (input_size < 16384)
        return 3;
    return 4;
}

// =============================================================================
// Global Instance
// =============================================================================

namespace {
std::unique_ptr<AdaptiveKernelCache> g_cache;
std::once_flag g_cache_init_flag;
}  // namespace

AdaptiveKernelCache& getGlobalCache() {
    std::call_once(g_cache_init_flag, []() { g_cache = std::make_unique<AdaptiveKernelCache>(); });
    return *g_cache;
}

void configureGlobalCache(const AdaptiveCacheConfig& config) {
    getGlobalCache().setConfig(config);
}

}  // namespace jit
}  // namespace bud
