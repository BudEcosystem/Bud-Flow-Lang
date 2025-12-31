// =============================================================================
// Bud Flow Lang - PGO Specializer Implementation
// =============================================================================

#include "bud_flow_lang/jit/pgo_specializer.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace bud {
namespace jit {

// =============================================================================
// ProfileData Implementation
// =============================================================================

void ProfileData::recordExecution(size_t element_count, std::chrono::nanoseconds exec_time) {
    ++total_calls_;
    total_elements_ += element_count;
    total_time_ns_ += static_cast<uint64_t>(exec_time.count());

    updateSizeStats(element_count, static_cast<uint64_t>(exec_time.count()));
}

void ProfileData::updateSizeStats(size_t size, uint64_t time_ns) {
    // Try to find existing entry for this size
    for (size_t i = 0; i < num_tracked_sizes_; ++i) {
        if (size_stats_[i].size == size) {
            size_stats_[i].count++;
            size_stats_[i].total_time_ns += time_ns;
            return;
        }
    }

    // Add new entry if room
    if (num_tracked_sizes_ < kMaxTrackedSizes) {
        size_stats_[num_tracked_sizes_] = {size, 1, time_ns};
        ++num_tracked_sizes_;
        return;
    }

    // No room - replace entry with lowest count
    size_t min_idx = 0;
    for (size_t i = 1; i < kMaxTrackedSizes; ++i) {
        if (size_stats_[i].count < size_stats_[min_idx].count) {
            min_idx = i;
        }
    }

    // Only replace if this is a more common size
    if (size_stats_[min_idx].count < 5) {
        size_stats_[min_idx] = {size, 1, time_ns};
    }
}

std::vector<SizeStats> ProfileData::commonSizes() const {
    std::vector<SizeStats> result;
    result.reserve(num_tracked_sizes_);

    for (size_t i = 0; i < num_tracked_sizes_; ++i) {
        if (size_stats_[i].count > 0) {
            result.push_back(size_stats_[i]);
        }
    }

    // Sort by count (descending)
    std::sort(result.begin(), result.end(),
              [](const SizeStats& a, const SizeStats& b) { return a.count > b.count; });

    return result;
}

bool ProfileData::shouldSpecialize() const {
    return total_calls_ >= kMinCallsForSpecialization;
}

std::optional<size_t> ProfileData::dominantSize() const {
    if (total_calls_ == 0) {
        return std::nullopt;
    }

    auto sizes = commonSizes();
    if (sizes.empty()) {
        return std::nullopt;
    }

    // Check if top size accounts for >80% of calls
    float ratio = static_cast<float>(sizes[0].count) / total_calls_;
    if (ratio >= 0.8f) {
        return sizes[0].size;
    }

    return std::nullopt;
}

float ProfileData::averageTimePerElement() const {
    if (total_elements_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(total_time_ns_) / total_elements_;
}

std::string ProfileData::serialize() const {
    nlohmann::json j;

    j["total_calls"] = total_calls_;
    j["total_elements"] = total_elements_;
    j["total_time_ns"] = total_time_ns_;

    nlohmann::json sizes_array = nlohmann::json::array();
    for (size_t i = 0; i < num_tracked_sizes_; ++i) {
        nlohmann::json entry;
        entry["size"] = size_stats_[i].size;
        entry["count"] = size_stats_[i].count;
        entry["time_ns"] = size_stats_[i].total_time_ns;
        sizes_array.push_back(entry);
    }
    j["sizes"] = sizes_array;

    return j.dump();
}

bool ProfileData::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        total_calls_ = j["total_calls"].get<uint64_t>();
        total_elements_ = j["total_elements"].get<uint64_t>();
        total_time_ns_ = j["total_time_ns"].get<uint64_t>();

        auto sizes_array = j["sizes"];
        num_tracked_sizes_ = std::min(sizes_array.size(), kMaxTrackedSizes);

        for (size_t i = 0; i < num_tracked_sizes_; ++i) {
            size_stats_[i].size = sizes_array[i]["size"].get<size_t>();
            size_stats_[i].count = sizes_array[i]["count"].get<uint64_t>();
            size_stats_[i].total_time_ns = sizes_array[i]["time_ns"].get<uint64_t>();
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize profile data: {}", e.what());
        return false;
    }
}

void ProfileData::reset() {
    total_calls_ = 0;
    total_elements_ = 0;
    total_time_ns_ = 0;
    num_tracked_sizes_ = 0;
    size_stats_.fill(SizeStats{});
}

// =============================================================================
// PGOSpecializer Implementation
// =============================================================================

PGOSpecializer::PGOSpecializer() = default;

std::optional<SpecializedIR> PGOSpecializer::specializeForSize(const ir::IRBuilder& builder,
                                                               size_t size) {
    if (!enabled_ || size == 0) {
        return std::nullopt;
    }

    // Create a specialized IR for this size
    SpecializedIR specialized(size);

    spdlog::debug("Created specialization for size {}", size);

    return specialized;
}

std::vector<SpecializedIR> PGOSpecializer::specializeForSizes(const ir::IRBuilder& builder,
                                                              const std::vector<size_t>& sizes) {
    std::vector<SpecializedIR> result;
    result.reserve(sizes.size());

    for (size_t size : sizes) {
        auto spec = specializeForSize(builder, size);
        if (spec.has_value()) {
            result.push_back(std::move(*spec));
        }
    }

    return result;
}

const SpecializedIR*
PGOSpecializer::selectSpecialization(const std::vector<SpecializedIR>& specializations,
                                     size_t size) const {
    if (specializations.empty()) {
        return nullptr;
    }

    const SpecializedIR* best = nullptr;
    size_t best_diff = SIZE_MAX;

    for (const auto& spec : specializations) {
        size_t spec_size = spec.specializedSize();

        // Calculate difference
        size_t diff = (size > spec_size) ? (size - spec_size) : (spec_size - size);

        // Check if within acceptable ratio
        size_t max_size = std::max(size, spec_size);
        size_t min_size = std::min(size, spec_size);

        if (max_size > min_size * kMaxSpecializationRatio) {
            continue;  // Too different
        }

        if (diff < best_diff) {
            best_diff = diff;
            best = &spec;
        }
    }

    return best;
}

std::optional<PGOResult> PGOSpecializer::applyProfile(const ir::IRBuilder& builder,
                                                      const ProfileData& profile) {
    if (!enabled_ || !profile.shouldSpecialize()) {
        return std::nullopt;
    }

    PGOResult result;
    result.success = true;

    // Get common sizes and create specializations
    auto common_sizes = profile.commonSizes();

    // Take top N most common sizes
    constexpr size_t kMaxSpecializations = 4;
    size_t num_to_specialize = std::min(common_sizes.size(), kMaxSpecializations);

    std::vector<size_t> sizes_to_specialize;
    sizes_to_specialize.reserve(num_to_specialize);

    for (size_t i = 0; i < num_to_specialize; ++i) {
        // Only specialize if this size is significant (>5% of calls)
        float ratio = static_cast<float>(common_sizes[i].count) / profile.totalCalls();
        if (ratio >= 0.05f) {
            sizes_to_specialize.push_back(common_sizes[i].size);
        }
    }

    result.specializations = specializeForSizes(builder, sizes_to_specialize);

    // Identify and optimize hot paths
    result.optimized_paths = identifyHotPaths(builder, profile);

    spdlog::debug("PGO applied: {} specializations, {} hot paths", result.specializations.size(),
                  result.optimized_paths.size());

    return result;
}

std::vector<HotPath> PGOSpecializer::identifyHotPaths(const ir::IRBuilder& builder,
                                                      const ProfileData& profile) const {
    std::vector<HotPath> hot_paths;

    // Simple heuristic: linear sequence of operations
    HotPath main_path;

    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        // Only include compute operations
        switch (node->opCode()) {
        case ir::OpCode::kAdd:
        case ir::OpCode::kSub:
        case ir::OpCode::kMul:
        case ir::OpCode::kDiv:
        case ir::OpCode::kFma:
        case ir::OpCode::kSqrt:
        case ir::OpCode::kExp:
        case ir::OpCode::kLog:
            main_path.operations.push_back(node->id());
            break;
        default:
            break;
        }
    }

    if (!main_path.operations.empty()) {
        main_path.execution_count = profile.totalCalls();
        main_path.avg_time_ns = profile.averageTimePerElement();
        hot_paths.push_back(std::move(main_path));
    }

    return hot_paths;
}

std::optional<ir::IRBuilder> PGOSpecializer::optimizeHotPaths(const ir::IRBuilder& builder,
                                                              const ProfileData& profile) {
    if (!enabled_ || !profile.shouldSpecialize()) {
        return std::nullopt;
    }

    // For now, return a copy of the builder
    // In a full implementation, we would:
    // 1. Inline constants for hot paths
    // 2. Unroll loops
    // 3. Specialize branches

    // Create a new builder as a "optimized" version
    // This is a placeholder - real optimization would modify the IR

    return std::nullopt;  // Indicate no optimization was done (placeholder)
}

std::optional<SpecializedIR> PGOSpecializer::getOrCreateSpecialization(const ir::IRBuilder& builder,
                                                                       size_t size) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check cache first
    auto it = cache_.find(size);
    if (it != cache_.end()) {
        ++cache_hits_;
        return it->second;
    }

    ++cache_misses_;

    // Create new specialization
    auto spec = specializeForSize(builder, size);
    if (!spec.has_value()) {
        return std::nullopt;
    }

    // Add to cache
    evictIfNeeded();
    cache_[size] = *spec;

    return spec;
}

SpecializationCacheStats PGOSpecializer::cacheStats() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    SpecializationCacheStats stats;
    stats.total_entries = cache_.size();
    stats.hits = cache_hits_;
    stats.misses = cache_misses_;
    stats.evictions = cache_evictions_;

    return stats;
}

void PGOSpecializer::setMaxCacheSize(size_t max_entries) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    max_cache_size_ = max_entries;

    // Evict if necessary
    while (cache_.size() > max_cache_size_) {
        evictIfNeeded();
    }
}

void PGOSpecializer::clearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
    cache_evictions_ = 0;
}

void PGOSpecializer::evictIfNeeded() {
    // Already holding lock
    if (cache_.size() < max_cache_size_) {
        return;
    }

    // Simple eviction: remove smallest size (LRU would be better)
    if (!cache_.empty()) {
        auto min_it = cache_.begin();
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->first < min_it->first) {
                min_it = it;
            }
        }
        cache_.erase(min_it);
        ++cache_evictions_;
    }
}

bool PGOSpecializer::loadProfile(const std::string& data) {
    ProfileData profile;
    if (!profile.deserialize(data)) {
        return false;
    }

    current_profile_ = std::move(profile);
    return true;
}

std::optional<ProfileData> PGOSpecializer::currentProfile() const {
    return current_profile_;
}

void PGOSpecializer::setProfile(const ProfileData& profile) {
    current_profile_ = profile;
}

}  // namespace jit
}  // namespace bud
