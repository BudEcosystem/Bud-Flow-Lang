// =============================================================================
// Bud Flow Lang - Schedule Primitives Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/schedule.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace bud {
namespace scheduler {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

std::string transformTypeToString(TransformType type) {
    switch (type) {
    case TransformType::kSplit:
        return "split";
    case TransformType::kTile:
        return "tile";
    case TransformType::kFuse:
        return "fuse";
    case TransformType::kReorder:
        return "reorder";
    case TransformType::kVectorize:
        return "vectorize";
    case TransformType::kParallel:
        return "parallel";
    case TransformType::kUnroll:
        return "unroll";
    case TransformType::kComputeAt:
        return "compute_at";
    case TransformType::kComputeInline:
        return "compute_inline";
    case TransformType::kCacheRead:
        return "cache_read";
    case TransformType::kCacheWrite:
        return "cache_write";
    default:
        return "unknown";
    }
}

TransformType stringToTransformType(const std::string& str) {
    if (str == "split")
        return TransformType::kSplit;
    if (str == "tile")
        return TransformType::kTile;
    if (str == "fuse")
        return TransformType::kFuse;
    if (str == "reorder")
        return TransformType::kReorder;
    if (str == "vectorize")
        return TransformType::kVectorize;
    if (str == "parallel")
        return TransformType::kParallel;
    if (str == "unroll")
        return TransformType::kUnroll;
    if (str == "compute_at")
        return TransformType::kComputeAt;
    if (str == "compute_inline")
        return TransformType::kComputeInline;
    if (str == "cache_read")
        return TransformType::kCacheRead;
    if (str == "cache_write")
        return TransformType::kCacheWrite;
    return TransformType::kSplit;  // Default
}

std::string cacheTypeToString(CacheType type) {
    switch (type) {
    case CacheType::kGlobal:
        return "global";
    case CacheType::kShared:
        return "shared";
    case CacheType::kLocal:
        return "local";
    case CacheType::kRegister:
        return "register";
    default:
        return "local";
    }
}

CacheType stringToCacheType(const std::string& str) {
    if (str == "global")
        return CacheType::kGlobal;
    if (str == "shared")
        return CacheType::kShared;
    if (str == "local")
        return CacheType::kLocal;
    if (str == "register")
        return CacheType::kRegister;
    return CacheType::kLocal;
}

}  // namespace

// =============================================================================
// Schedule Implementation
// =============================================================================

std::string Schedule::generateVarName(const std::string& base, const std::string& suffix) {
    return base + "_" + suffix + "_" + std::to_string(var_counter_++);
}

Schedule& Schedule::split(const Var& var, size_t factor, Var& outer, Var& inner) {
    // Generate names for outer and inner variables
    std::string outer_name = generateVarName(var.name(), "outer");
    std::string inner_name = generateVarName(var.name(), "inner");

    outer = Var(outer_name);
    inner = Var(inner_name);

    // Record the transformation
    Transform t;
    t.type = TransformType::kSplit;
    t.vars = {var, outer, inner};
    t.factor = factor;

    transforms_.push_back(t);

    spdlog::debug("Split {} by {} into {}, {}", var.name(), factor, outer.name(), inner.name());

    return *this;
}

Schedule& Schedule::tile(const Var& x, const Var& y, size_t tile_x, size_t tile_y, Var& x_outer,
                         Var& x_inner, Var& y_outer, Var& y_inner) {
    // Generate names
    std::string xo_name = generateVarName(x.name(), "outer");
    std::string xi_name = generateVarName(x.name(), "inner");
    std::string yo_name = generateVarName(y.name(), "outer");
    std::string yi_name = generateVarName(y.name(), "inner");

    x_outer = Var(xo_name);
    x_inner = Var(xi_name);
    y_outer = Var(yo_name);
    y_inner = Var(yi_name);

    // Record the transformation
    Transform t;
    t.type = TransformType::kTile;
    t.vars = {x, y, x_outer, x_inner, y_outer, y_inner};
    t.tile_x = tile_x;
    t.tile_y = tile_y;

    transforms_.push_back(t);

    spdlog::debug("Tile {},{} by {}x{}", x.name(), y.name(), tile_x, tile_y);

    return *this;
}

Schedule& Schedule::fuse(const Var& outer, const Var& inner, Var& fused) {
    std::string fused_name = generateVarName(outer.name() + "_" + inner.name(), "fused");
    fused = Var(fused_name);

    Transform t;
    t.type = TransformType::kFuse;
    t.vars = {outer, inner, fused};

    transforms_.push_back(t);

    spdlog::debug("Fuse {},{} into {}", outer.name(), inner.name(), fused.name());

    return *this;
}

Schedule& Schedule::reorder(const std::vector<Var>& new_order) {
    Transform t;
    t.type = TransformType::kReorder;
    t.vars = new_order;

    transforms_.push_back(t);

    if (spdlog::should_log(spdlog::level::debug)) {
        std::string order_str;
        for (const auto& v : new_order) {
            if (!order_str.empty())
                order_str += ",";
            order_str += v.name();
        }
        spdlog::debug("Reorder to: {}", order_str);
    }

    return *this;
}

Schedule& Schedule::vectorize(const Var& var, size_t width) {
    Transform t;
    t.type = TransformType::kVectorize;
    t.vars = {var};
    t.vector_width = width;

    transforms_.push_back(t);

    spdlog::debug("Vectorize {} width={}", var.name(), width);

    return *this;
}

Schedule& Schedule::parallel(const Var& var) {
    Transform t;
    t.type = TransformType::kParallel;
    t.vars = {var};

    transforms_.push_back(t);

    spdlog::debug("Parallel {}", var.name());

    return *this;
}

Schedule& Schedule::unroll(const Var& var, size_t factor) {
    Transform t;
    t.type = TransformType::kUnroll;
    t.vars = {var};
    t.factor = factor;

    transforms_.push_back(t);

    spdlog::debug("Unroll {} factor={}", var.name(), factor);

    return *this;
}

Schedule& Schedule::computeAt(const Stage& producer, const Stage& consumer, const Var& var) {
    Transform t;
    t.type = TransformType::kComputeAt;
    t.vars = {var};
    t.producer_stage = producer.name();
    t.consumer_stage = consumer.name();

    transforms_.push_back(t);

    spdlog::debug("ComputeAt {} at {} var {}", producer.name(), consumer.name(), var.name());

    return *this;
}

Schedule& Schedule::computeInline(const Stage& producer) {
    Transform t;
    t.type = TransformType::kComputeInline;
    t.producer_stage = producer.name();

    transforms_.push_back(t);

    spdlog::debug("ComputeInline {}", producer.name());

    return *this;
}

Schedule& Schedule::cacheRead(const Stage& stage, CacheType type) {
    Transform t;
    t.type = TransformType::kCacheRead;
    t.producer_stage = stage.name();
    t.cache_type = type;

    transforms_.push_back(t);

    spdlog::debug("CacheRead {} type={}", stage.name(), cacheTypeToString(type));

    return *this;
}

Schedule& Schedule::cacheWrite(const Stage& stage, CacheType type) {
    Transform t;
    t.type = TransformType::kCacheWrite;
    t.producer_stage = stage.name();
    t.cache_type = type;

    transforms_.push_back(t);

    spdlog::debug("CacheWrite {} type={}", stage.name(), cacheTypeToString(type));

    return *this;
}

std::unique_ptr<Schedule> Schedule::clone() const {
    auto cloned = std::make_unique<Schedule>();
    cloned->transforms_ = transforms_;
    cloned->var_counter_ = var_counter_;
    return cloned;
}

std::string Schedule::serialize() const {
    nlohmann::json j;

    j["var_counter"] = var_counter_;

    nlohmann::json transforms_array = nlohmann::json::array();
    for (const auto& t : transforms_) {
        nlohmann::json tj;
        tj["type"] = transformTypeToString(t.type);
        tj["factor"] = t.factor;
        tj["tile_x"] = t.tile_x;
        tj["tile_y"] = t.tile_y;
        tj["vector_width"] = t.vector_width;
        tj["producer_stage"] = t.producer_stage;
        tj["consumer_stage"] = t.consumer_stage;
        tj["cache_type"] = cacheTypeToString(t.cache_type);

        nlohmann::json vars_array = nlohmann::json::array();
        for (const auto& v : t.vars) {
            vars_array.push_back(v.name());
        }
        tj["vars"] = vars_array;

        transforms_array.push_back(tj);
    }
    j["transforms"] = transforms_array;

    return j.dump();
}

bool Schedule::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        var_counter_ = j["var_counter"].get<size_t>();
        transforms_.clear();

        for (const auto& tj : j["transforms"]) {
            Transform t;
            t.type = stringToTransformType(tj["type"].get<std::string>());
            t.factor = tj["factor"].get<size_t>();
            t.tile_x = tj["tile_x"].get<size_t>();
            t.tile_y = tj["tile_y"].get<size_t>();
            t.vector_width = tj["vector_width"].get<size_t>();
            t.producer_stage = tj["producer_stage"].get<std::string>();
            t.consumer_stage = tj["consumer_stage"].get<std::string>();
            t.cache_type = stringToCacheType(tj["cache_type"].get<std::string>());

            for (const auto& v : tj["vars"]) {
                t.vars.push_back(Var(v.get<std::string>()));
            }

            transforms_.push_back(t);
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize schedule: {}", e.what());
        return false;
    }
}

bool Schedule::validate() const {
    // Basic validation:
    // 1. All transforms have valid types
    // 2. Split/tile have non-zero factors
    // 3. Variables are valid

    for (const auto& t : transforms_) {
        switch (t.type) {
        case TransformType::kSplit:
            if (t.factor == 0) {
                spdlog::warn("Split transform has zero factor");
                return false;
            }
            if (t.vars.size() < 3) {
                spdlog::warn("Split transform missing vars");
                return false;
            }
            break;

        case TransformType::kTile:
            if (t.tile_x == 0 || t.tile_y == 0) {
                spdlog::warn("Tile transform has zero dimensions");
                return false;
            }
            break;

        case TransformType::kReorder:
            if (t.vars.empty()) {
                spdlog::warn("Reorder transform has no vars");
                return false;
            }
            break;

        default:
            break;
        }
    }

    return true;
}

}  // namespace scheduler
}  // namespace bud
