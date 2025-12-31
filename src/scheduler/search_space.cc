// =============================================================================
// Bud Flow Lang - Search Space Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/search_space.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <unordered_set>

namespace bud {
namespace scheduler {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

std::string transformStepToString(TransformStep type) {
    switch (type) {
    case TransformStep::kNone:
        return "none";
    case TransformStep::kSplit:
        return "split";
    case TransformStep::kTile:
        return "tile";
    case TransformStep::kFuse:
        return "fuse";
    case TransformStep::kReorder:
        return "reorder";
    case TransformStep::kVectorize:
        return "vectorize";
    case TransformStep::kParallel:
        return "parallel";
    case TransformStep::kUnroll:
        return "unroll";
    case TransformStep::kComputeAt:
        return "compute_at";
    case TransformStep::kCacheRead:
        return "cache_read";
    case TransformStep::kCacheWrite:
        return "cache_write";
    default:
        return "unknown";
    }
}

TransformStep stringToTransformStep(const std::string& str) {
    if (str == "split")
        return TransformStep::kSplit;
    if (str == "tile")
        return TransformStep::kTile;
    if (str == "fuse")
        return TransformStep::kFuse;
    if (str == "reorder")
        return TransformStep::kReorder;
    if (str == "vectorize")
        return TransformStep::kVectorize;
    if (str == "parallel")
        return TransformStep::kParallel;
    if (str == "unroll")
        return TransformStep::kUnroll;
    if (str == "compute_at")
        return TransformStep::kComputeAt;
    if (str == "cache_read")
        return TransformStep::kCacheRead;
    if (str == "cache_write")
        return TransformStep::kCacheWrite;
    return TransformStep::kNone;
}

std::string sketchTypeToString(SketchType type) {
    switch (type) {
    case SketchType::kInvalid:
        return "invalid";
    case SketchType::kBasic:
        return "basic";
    case SketchType::kTiled:
        return "tiled";
    case SketchType::kFused:
        return "fused";
    case SketchType::kVectorized:
        return "vectorized";
    case SketchType::kParallel:
        return "parallel";
    case SketchType::kCombined:
        return "combined";
    default:
        return "unknown";
    }
}

SketchType stringToSketchType(const std::string& str) {
    if (str == "basic")
        return SketchType::kBasic;
    if (str == "tiled")
        return SketchType::kTiled;
    if (str == "fused")
        return SketchType::kFused;
    if (str == "vectorized")
        return SketchType::kVectorized;
    if (str == "parallel")
        return SketchType::kParallel;
    if (str == "combined")
        return SketchType::kCombined;
    return SketchType::kInvalid;
}

}  // namespace

// =============================================================================
// TransformStepDef Implementation
// =============================================================================

std::string TransformStepDef::serialize() const {
    nlohmann::json j;
    j["type"] = transformStepToString(type);
    j["target_var"] = target_var;
    j["secondary_var"] = secondary_var;
    j["factor"] = factor;
    j["tile_x"] = tile_x;
    j["tile_y"] = tile_y;
    j["vector_width"] = vector_width;
    j["stage_name"] = stage_name;
    return j.dump();
}

bool TransformStepDef::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);
        type = stringToTransformStep(j["type"].get<std::string>());
        target_var = j["target_var"].get<std::string>();
        secondary_var = j["secondary_var"].get<std::string>();
        factor = j["factor"].get<size_t>();
        tile_x = j["tile_x"].get<size_t>();
        tile_y = j["tile_y"].get<size_t>();
        vector_width = j["vector_width"].get<size_t>();
        stage_name = j["stage_name"].get<std::string>();
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize TransformStepDef: {}", e.what());
        return false;
    }
}

// =============================================================================
// Sketch Implementation
// =============================================================================

std::string Sketch::serialize() const {
    nlohmann::json j;
    j["type"] = sketchTypeToString(type);
    j["name"] = name;

    nlohmann::json steps_array = nlohmann::json::array();
    for (const auto& step : steps) {
        nlohmann::json sj;
        sj["type"] = transformStepToString(step.type);
        sj["target_var"] = step.target_var;
        sj["secondary_var"] = step.secondary_var;
        sj["factor"] = step.factor;
        sj["tile_x"] = step.tile_x;
        sj["tile_y"] = step.tile_y;
        sj["vector_width"] = step.vector_width;
        sj["stage_name"] = step.stage_name;
        steps_array.push_back(sj);
    }
    j["steps"] = steps_array;

    nlohmann::json nodes_array = nlohmann::json::array();
    for (const auto& node_id : affected_nodes) {
        nodes_array.push_back(node_id.value());
    }
    j["affected_nodes"] = nodes_array;

    return j.dump();
}

bool Sketch::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);
        type = stringToSketchType(j["type"].get<std::string>());
        name = j["name"].get<std::string>();

        steps.clear();
        for (const auto& sj : j["steps"]) {
            TransformStepDef step;
            step.type = stringToTransformStep(sj["type"].get<std::string>());
            step.target_var = sj["target_var"].get<std::string>();
            step.secondary_var = sj["secondary_var"].get<std::string>();
            step.factor = sj["factor"].get<size_t>();
            step.tile_x = sj["tile_x"].get<size_t>();
            step.tile_y = sj["tile_y"].get<size_t>();
            step.vector_width = sj["vector_width"].get<size_t>();
            step.stage_name = sj["stage_name"].get<std::string>();
            steps.push_back(step);
        }

        affected_nodes.clear();
        for (const auto& node : j["affected_nodes"]) {
            affected_nodes.push_back(DAGNodeId(node.get<uint32_t>()));
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize Sketch: {}", e.what());
        return false;
    }
}

// =============================================================================
// SearchSpace Implementation
// =============================================================================

SearchSpace SearchSpace::fromDAG(const ComputeDAG& dag, const SearchSpaceBounds& bounds) {
    SearchSpace space;
    space.bounds_ = bounds;
    space.dag_ = &dag;

    // Generate sketches based on DAG structure
    space.generateBasicSketch();
    space.generateTiledSketches();
    space.generateFusedSketches();
    space.generateVectorizedSketches();
    space.generateParallelSketches();
    space.generateCombinedSketches();

    spdlog::debug("Generated {} sketches from DAG with {} nodes", space.sketches_.size(),
                  dag.numNodes());

    return space;
}

void SearchSpace::generateBasicSketch() {
    Sketch basic;
    basic.type = SketchType::kBasic;
    basic.name = "basic";
    // No transform steps - uses default schedule
    sketches_.push_back(basic);
}

void SearchSpace::generateTiledSketches() {
    if (!dag_ || dag_->numNodes() == 0) {
        return;
    }

    // Generate tiled sketches with different tile sizes
    auto tile_sizes = getTileSizes();
    for (const auto& [tile_x, tile_y] : tile_sizes) {
        Sketch sketch;
        sketch.type = SketchType::kTiled;
        sketch.name = "tiled_" + std::to_string(tile_x) + "x" + std::to_string(tile_y);

        TransformStepDef step;
        step.type = TransformStep::kTile;
        step.target_var = "i";
        step.secondary_var = "j";
        step.tile_x = tile_x;
        step.tile_y = tile_y;
        sketch.steps.push_back(step);

        sketches_.push_back(sketch);
    }
}

void SearchSpace::generateFusedSketches() {
    if (!dag_ || dag_->numEdges() == 0) {
        return;
    }

    // Create a fused sketch for producer-consumer pairs
    Sketch sketch;
    sketch.type = SketchType::kFused;
    sketch.name = "fused";

    TransformStepDef step;
    step.type = TransformStep::kFuse;
    step.target_var = "outer";
    step.secondary_var = "inner";
    sketch.steps.push_back(step);

    sketches_.push_back(sketch);
}

void SearchSpace::generateVectorizedSketches() {
    if (!dag_ || dag_->numNodes() == 0) {
        return;
    }

    // Generate vectorized sketches with different vector widths
    for (size_t width : bounds_.vector_widths) {
        Sketch sketch;
        sketch.type = SketchType::kVectorized;
        sketch.name = "vectorized_" + std::to_string(width);

        TransformStepDef step;
        step.type = TransformStep::kVectorize;
        step.target_var = "inner";
        step.vector_width = width;
        sketch.steps.push_back(step);

        sketches_.push_back(sketch);
    }
}

void SearchSpace::generateParallelSketches() {
    if (!dag_ || dag_->numNodes() == 0) {
        return;
    }

    // Generate parallel sketch
    Sketch sketch;
    sketch.type = SketchType::kParallel;
    sketch.name = "parallel";

    TransformStepDef step;
    step.type = TransformStep::kParallel;
    step.target_var = "outer";
    sketch.steps.push_back(step);

    sketches_.push_back(sketch);
}

void SearchSpace::generateCombinedSketches() {
    if (!dag_ || dag_->numNodes() == 0) {
        return;
    }

    // Generate combined sketch: tile + vectorize + parallel
    Sketch sketch;
    sketch.type = SketchType::kCombined;
    sketch.name = "combined";

    // Split for parallelization
    TransformStepDef split_step;
    split_step.type = TransformStep::kSplit;
    split_step.target_var = "i";
    split_step.factor = 32;
    sketch.steps.push_back(split_step);

    // Parallel outer
    TransformStepDef parallel_step;
    parallel_step.type = TransformStep::kParallel;
    parallel_step.target_var = "i_outer";
    sketch.steps.push_back(parallel_step);

    // Vectorize inner
    TransformStepDef vec_step;
    vec_step.type = TransformStep::kVectorize;
    vec_step.target_var = "i_inner";
    vec_step.vector_width = 8;
    sketch.steps.push_back(vec_step);

    sketches_.push_back(sketch);
}

std::vector<size_t> SearchSpace::getSplitFactors() const {
    std::vector<size_t> factors;
    for (size_t f = bounds_.min_split_factor; f <= bounds_.max_split_factor; f *= 2) {
        factors.push_back(f);
    }
    return factors;
}

std::vector<std::pair<size_t, size_t>> SearchSpace::getTileSizes() const {
    std::vector<std::pair<size_t, size_t>> sizes;
    for (size_t x = bounds_.min_tile_size; x <= bounds_.max_tile_size; x *= 2) {
        for (size_t y = bounds_.min_tile_size; y <= bounds_.max_tile_size; y *= 2) {
            sizes.emplace_back(x, y);
        }
    }
    return sizes;
}

std::vector<Sketch> SearchSpace::getSketchesOfType(SketchType type) const {
    std::vector<Sketch> result;
    for (const auto& sketch : sketches_) {
        if (sketch.type == type) {
            result.push_back(sketch);
        }
    }
    return result;
}

std::vector<Schedule> SearchSpace::annotateSketch(const Sketch& sketch) const {
    std::vector<Schedule> schedules;

    // Convert sketch to base schedule
    Schedule base = sketchToSchedule(sketch);
    schedules.push_back(base);

    // Generate variations with different parameters
    if (sketch.type == SketchType::kTiled) {
        auto tile_sizes = getTileSizes();
        for (const auto& [tile_x, tile_y] : tile_sizes) {
            Schedule s;
            Var x("x"), y("y"), xo, xi, yo, yi;
            s.tile(x, y, tile_x, tile_y, xo, xi, yo, yi);
            schedules.push_back(s);
        }
    } else if (sketch.type == SketchType::kVectorized) {
        for (size_t width : bounds_.vector_widths) {
            Schedule s;
            Var inner("inner");
            s.vectorize(inner, width);
            schedules.push_back(s);
        }
    }

    return schedules;
}

Schedule SearchSpace::sketchToSchedule(const Sketch& sketch) const {
    Schedule schedule;

    for (const auto& step : sketch.steps) {
        Var target(step.target_var);

        switch (step.type) {
        case TransformStep::kSplit: {
            Var outer, inner;
            schedule.split(target, step.factor, outer, inner);
            break;
        }
        case TransformStep::kTile: {
            Var secondary(step.secondary_var);
            Var xo, xi, yo, yi;
            schedule.tile(target, secondary, step.tile_x, step.tile_y, xo, xi, yo, yi);
            break;
        }
        case TransformStep::kFuse: {
            Var secondary(step.secondary_var);
            Var fused;
            schedule.fuse(target, secondary, fused);
            break;
        }
        case TransformStep::kVectorize:
            schedule.vectorize(target, step.vector_width);
            break;
        case TransformStep::kParallel:
            schedule.parallel(target);
            break;
        case TransformStep::kUnroll:
            schedule.unroll(target, step.factor);
            break;
        default:
            break;
        }
    }

    return schedule;
}

std::vector<Schedule> SearchSpace::sample(size_t count, uint64_t seed) const {
    std::vector<Schedule> samples;
    if (sketches_.empty()) {
        return samples;
    }

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);

    for (size_t i = 0; i < count; ++i) {
        // Pick a random sketch
        std::uniform_int_distribution<size_t> sketch_dist(0, sketches_.size() - 1);
        const auto& sketch = sketches_[sketch_dist(rng)];

        // Convert to schedule with random parameters
        Schedule schedule = sketchToSchedule(sketch);

        // Optionally add more transformations
        if (sketch.type != SketchType::kBasic) {
            // Maybe add vectorization
            std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
            if (prob_dist(rng) > 0.5f && !bounds_.vector_widths.empty()) {
                std::uniform_int_distribution<size_t> vw_dist(0, bounds_.vector_widths.size() - 1);
                size_t width = bounds_.vector_widths[vw_dist(rng)];
                Var inner("inner_" + std::to_string(i));
                schedule.vectorize(inner, width);
            }
        }

        samples.push_back(schedule);
    }

    return samples;
}

std::vector<Schedule> SearchSpace::sampleUnique(size_t count, uint64_t seed) const {
    std::vector<Schedule> samples;
    std::unordered_set<std::string> seen;

    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);

    size_t max_attempts = count * 10;  // Limit attempts to avoid infinite loop
    size_t attempts = 0;

    while (samples.size() < count && attempts < max_attempts) {
        auto new_samples = sample(1, rng());
        if (!new_samples.empty()) {
            std::string serialized = new_samples[0].serialize();
            if (seen.count(serialized) == 0) {
                seen.insert(serialized);
                samples.push_back(new_samples[0]);
            }
        }
        ++attempts;
    }

    return samples;
}

std::vector<Schedule> SearchSpace::refineAround(const Schedule& base, size_t num_neighbors) const {
    std::vector<Schedule> neighbors;

    for (size_t i = 0; i < num_neighbors; ++i) {
        Schedule mutated = mutateSchedule(base);
        neighbors.push_back(mutated);
    }

    return neighbors;
}

Schedule SearchSpace::mutateSchedule(const Schedule& base) const {
    // Clone the base schedule
    auto cloned = base.clone();

    // Apply a random mutation
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> mutation_type(0, 3);

    switch (mutation_type(rng)) {
    case 0: {
        // Add split
        auto factors = getSplitFactors();
        if (!factors.empty()) {
            std::uniform_int_distribution<size_t> factor_dist(0, factors.size() - 1);
            Var outer, inner;
            Var v("mutate_var");
            cloned->split(v, factors[factor_dist(rng)], outer, inner);
        }
        break;
    }
    case 1: {
        // Add vectorize
        if (!bounds_.vector_widths.empty()) {
            std::uniform_int_distribution<size_t> vw_dist(0, bounds_.vector_widths.size() - 1);
            Var v("mutate_vec_var");
            cloned->vectorize(v, bounds_.vector_widths[vw_dist(rng)]);
        }
        break;
    }
    case 2: {
        // Add unroll
        std::uniform_int_distribution<size_t> unroll_dist(2, bounds_.max_unroll_factor);
        Var v("mutate_unroll_var");
        cloned->unroll(v, unroll_dist(rng));
        break;
    }
    case 3: {
        // Add parallel
        Var v("mutate_parallel_var");
        cloned->parallel(v);
        break;
    }
    }

    return *cloned;
}

size_t SearchSpace::estimateSize() const {
    if (sketches_.empty()) {
        return 0;
    }

    size_t total = 0;

    // Each sketch can be annotated with different parameters
    auto split_factors = getSplitFactors();
    auto tile_sizes = getTileSizes();

    for (const auto& sketch : sketches_) {
        switch (sketch.type) {
        case SketchType::kBasic:
            total += 1;
            break;
        case SketchType::kTiled:
            total += tile_sizes.size();
            break;
        case SketchType::kVectorized:
            total += bounds_.vector_widths.size();
            break;
        case SketchType::kFused:
        case SketchType::kParallel:
            total += 1;
            break;
        case SketchType::kCombined:
            total += split_factors.size() * bounds_.vector_widths.size();
            break;
        default:
            break;
        }
    }

    return total;
}

std::vector<Schedule> SearchSpace::enumerateAll(size_t limit) const {
    std::vector<Schedule> all;

    for (const auto& sketch : sketches_) {
        auto annotated = annotateSketch(sketch);
        for (const auto& schedule : annotated) {
            all.push_back(schedule);
            if (all.size() >= limit) {
                return all;
            }
        }
    }

    return all;
}

std::string SearchSpace::serialize() const {
    nlohmann::json j;

    // Serialize bounds
    nlohmann::json bj;
    bj["min_split_factor"] = bounds_.min_split_factor;
    bj["max_split_factor"] = bounds_.max_split_factor;
    bj["min_tile_size"] = bounds_.min_tile_size;
    bj["max_tile_size"] = bounds_.max_tile_size;
    bj["max_unroll_factor"] = bounds_.max_unroll_factor;
    bj["vector_widths"] = bounds_.vector_widths;
    bj["max_parallel_levels"] = bounds_.max_parallel_levels;
    bj["max_compute_at_depth"] = bounds_.max_compute_at_depth;
    j["bounds"] = bj;

    // Serialize sketches
    nlohmann::json sketches_array = nlohmann::json::array();
    for (const auto& sketch : sketches_) {
        nlohmann::json sj;
        sj["type"] = sketchTypeToString(sketch.type);
        sj["name"] = sketch.name;

        nlohmann::json steps_array = nlohmann::json::array();
        for (const auto& step : sketch.steps) {
            nlohmann::json stj;
            stj["type"] = transformStepToString(step.type);
            stj["target_var"] = step.target_var;
            stj["secondary_var"] = step.secondary_var;
            stj["factor"] = step.factor;
            stj["tile_x"] = step.tile_x;
            stj["tile_y"] = step.tile_y;
            stj["vector_width"] = step.vector_width;
            stj["stage_name"] = step.stage_name;
            steps_array.push_back(stj);
        }
        sj["steps"] = steps_array;

        nlohmann::json nodes_array = nlohmann::json::array();
        for (const auto& node_id : sketch.affected_nodes) {
            nodes_array.push_back(node_id.value());
        }
        sj["affected_nodes"] = nodes_array;

        sketches_array.push_back(sj);
    }
    j["sketches"] = sketches_array;

    return j.dump();
}

bool SearchSpace::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        // Deserialize bounds
        auto bj = j["bounds"];
        bounds_.min_split_factor = bj["min_split_factor"].get<size_t>();
        bounds_.max_split_factor = bj["max_split_factor"].get<size_t>();
        bounds_.min_tile_size = bj["min_tile_size"].get<size_t>();
        bounds_.max_tile_size = bj["max_tile_size"].get<size_t>();
        bounds_.max_unroll_factor = bj["max_unroll_factor"].get<size_t>();
        bounds_.vector_widths = bj["vector_widths"].get<std::vector<size_t>>();
        bounds_.max_parallel_levels = bj["max_parallel_levels"].get<size_t>();
        bounds_.max_compute_at_depth = bj["max_compute_at_depth"].get<size_t>();

        // Deserialize sketches
        sketches_.clear();
        for (const auto& sj : j["sketches"]) {
            Sketch sketch;
            sketch.type = stringToSketchType(sj["type"].get<std::string>());
            sketch.name = sj["name"].get<std::string>();

            for (const auto& stj : sj["steps"]) {
                TransformStepDef step;
                step.type = stringToTransformStep(stj["type"].get<std::string>());
                step.target_var = stj["target_var"].get<std::string>();
                step.secondary_var = stj["secondary_var"].get<std::string>();
                step.factor = stj["factor"].get<size_t>();
                step.tile_x = stj["tile_x"].get<size_t>();
                step.tile_y = stj["tile_y"].get<size_t>();
                step.vector_width = stj["vector_width"].get<size_t>();
                step.stage_name = stj["stage_name"].get<std::string>();
                sketch.steps.push_back(step);
            }

            for (const auto& node : sj["affected_nodes"]) {
                sketch.affected_nodes.push_back(DAGNodeId(node.get<uint32_t>()));
            }

            sketches_.push_back(sketch);
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize SearchSpace: {}", e.what());
        return false;
    }
}

}  // namespace scheduler
}  // namespace bud
