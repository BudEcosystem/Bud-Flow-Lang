// =============================================================================
// Bud Flow Lang - Code Layout Optimizer Implementation
// =============================================================================

#include "bud_flow_lang/jit/code_layout.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace bud {
namespace jit {

// =============================================================================
// CodeLayoutOptimizer Implementation
// =============================================================================

CodeLayoutOptimizer::CodeLayoutOptimizer() = default;

RegionType CodeLayoutOptimizer::classifyFunction(const FunctionProfile& profile) const {
    // Critical path functions are always hot
    if (profile.is_critical_path && profile.execution_count > 0) {
        return RegionType::kHot;
    }

    // Check against thresholds
    if (profile.execution_count >= hot_threshold_) {
        return RegionType::kHot;
    }

    if (profile.execution_count <= cold_threshold_) {
        return RegionType::kCold;
    }

    return RegionType::kGeneric;
}

LayoutPlan
CodeLayoutOptimizer::createLayoutPlan(const std::vector<FunctionProfile>& profiles) const {
    LayoutPlan plan;

    // Create a sorted copy of profiles
    std::vector<FunctionProfile> sorted_profiles = profiles;

    // Sort by execution count (descending) for hot function ordering
    std::sort(sorted_profiles.begin(), sorted_profiles.end(),
              [](const FunctionProfile& a, const FunctionProfile& b) {
                  return a.execution_count > b.execution_count;
              });

    // Classify and organize functions
    for (const auto& profile : sorted_profiles) {
        auto classification = classifyFunction(profile);

        switch (classification) {
        case RegionType::kHot:
            plan.hot_functions.push_back(profile.name);
            break;
        case RegionType::kCold:
            plan.cold_functions.push_back(profile.name);
            break;
        case RegionType::kGeneric:
            plan.generic_functions.push_back(profile.name);
            break;
        }
    }

    spdlog::debug("Layout plan: {} hot, {} cold, {} generic functions", plan.hot_functions.size(),
                  plan.cold_functions.size(), plan.generic_functions.size());

    return plan;
}

bool CodeLayoutOptimizer::shouldInvertBranch(const BranchProfile& profile) const {
    // If taken is more common, invert so fall-through is the hot path
    // This improves branch prediction and instruction cache utilization
    return profile.taken_count > profile.not_taken_count;
}

size_t CodeLayoutOptimizer::recommendedAlignment(AlignmentTarget target) const {
    switch (target) {
    case AlignmentTarget::kFunctionEntry:
        return kFunctionAlignment;
    case AlignmentTarget::kLoopHeader:
        return kLoopAlignment;
    case AlignmentTarget::kJumpTarget:
        return kJumpTargetAlignment;
    case AlignmentTarget::kBasicBlock:
        return kBasicBlockAlignment;
    default:
        return 4;  // Default to 4-byte alignment
    }
}

std::optional<LayoutResult> CodeLayoutOptimizer::optimizeLayout(const ir::IRBuilder& builder,
                                                                const ProfileData& profile) {
    if (!enabled_ || !profile.shouldSpecialize()) {
        return std::nullopt;
    }

    LayoutResult result;
    result.optimized = true;

    // Analyze the IR to identify hot/cold regions
    // For now, this is a placeholder - a full implementation would:
    // 1. Build a call graph from the IR
    // 2. Map profile data to IR nodes
    // 3. Classify each function/basic block
    // 4. Reorder nodes for optimal layout

    // Count compute operations as "hot"
    size_t hot_ops = 0;
    size_t cold_ops = 0;

    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            ++cold_ops;  // Dead code is cold
            continue;
        }

        // Compute operations are typically hot
        switch (node->opCode()) {
        case ir::OpCode::kAdd:
        case ir::OpCode::kSub:
        case ir::OpCode::kMul:
        case ir::OpCode::kDiv:
        case ir::OpCode::kFma:
        case ir::OpCode::kSqrt:
        case ir::OpCode::kExp:
        case ir::OpCode::kLog:
            ++hot_ops;
            break;
        default:
            ++cold_ops;
            break;
        }
    }

    result.hot_region_size = hot_ops * 32;    // Estimate ~32 bytes per op
    result.cold_region_size = cold_ops * 16;  // Cold ops tend to be smaller
    result.functions_reordered = (hot_ops > 0) ? 1 : 0;

    spdlog::debug("Layout optimization: {} hot ops, {} cold ops", hot_ops, cold_ops);

    return result;
}

MemoryFootprint
CodeLayoutOptimizer::calculateFootprint(const std::vector<CodeRegion>& regions) const {
    MemoryFootprint footprint;

    for (const auto& region : regions) {
        size_t size = region.size();
        footprint.total_size += size;

        switch (region.type()) {
        case RegionType::kHot:
            footprint.hot_size += size;
            break;
        case RegionType::kCold:
            footprint.cold_size += size;
            break;
        case RegionType::kGeneric:
            footprint.generic_size += size;
            break;
        }
    }

    return footprint;
}

}  // namespace jit
}  // namespace bud
