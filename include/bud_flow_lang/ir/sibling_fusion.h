#pragma once

// =============================================================================
// Bud Flow Lang - Sibling (Multi-Output) Fusion
// =============================================================================
//
// Detects and fuses operations that share common inputs (siblings).
// Pattern: Operations A and B both read from input X -> Fuse to read X once.
//
// Benefits:
// - Reduces memory bandwidth by reading shared inputs once
// - Better cache utilization
// - Enables further optimizations (loop fusion)
//

#include "bud_flow_lang/ir.h"

#include <vector>

namespace bud {
namespace ir {

// =============================================================================
// Sibling Fusion Group
// =============================================================================

/// Represents a group of sibling operations that share common inputs
struct SiblingGroup {
    /// Operations in this sibling group
    std::vector<ValueId> siblings;

    /// Inputs shared by all siblings
    std::vector<ValueId> shared_inputs;

    /// Estimated benefit of fusing this group (memory reads saved)
    float estimated_benefit = 0.0f;
};

// =============================================================================
// SiblingFusionAnalyzer
// =============================================================================

/// Analyzes IR to find sibling fusion opportunities
class SiblingFusionAnalyzer {
  public:
    SiblingFusionAnalyzer() = default;

    /// Find groups of sibling operations (ops sharing common inputs)
    /// @param builder The IR builder to analyze
    /// @return Vector of sibling groups, sorted by estimated benefit
    [[nodiscard]] std::vector<SiblingGroup> findSiblingGroups(const IRBuilder& builder) const;

    /// Check if a set of siblings can be fused together
    /// @param builder The IR builder
    /// @param siblings The sibling operation IDs to check
    /// @return true if siblings can be fused (no data dependencies, compatible types)
    [[nodiscard]] bool canFuseSiblings(const IRBuilder& builder,
                                       const std::vector<ValueId>& siblings) const;

    /// Create a multi-output fusion from a sibling group
    /// @param builder The IR builder to modify
    /// @param group The sibling group to fuse
    /// @return Vector of new output ValueIds (one per original sibling)
    [[nodiscard]] std::vector<ValueId> createMultiOutputFusion(IRBuilder& builder,
                                                               const SiblingGroup& group) const;

  private:
    /// Check if there's a data dependency between two operations
    [[nodiscard]] bool hasDataDependency(const IRBuilder& builder, ValueId op1, ValueId op2) const;

    /// Check if all operations have compatible types for fusion
    [[nodiscard]] bool areTypesCompatible(const IRBuilder& builder,
                                          const std::vector<ValueId>& siblings) const;

    /// Calculate the benefit of fusing a sibling group
    [[nodiscard]] float calculateBenefit(const IRBuilder& builder, const SiblingGroup& group) const;

    /// Find all operations that use a given value as input
    [[nodiscard]] std::vector<ValueId> findConsumers(const IRBuilder& builder, ValueId value) const;
};

// =============================================================================
// SiblingFusionPass
// =============================================================================

/// Optimization pass that performs sibling fusion
class SiblingFusionPass {
  public:
    SiblingFusionPass() = default;

    /// Run the sibling fusion pass
    /// @param builder The IR builder to optimize
    /// @return Number of fusion groups applied
    size_t run(IRBuilder& builder);

    /// Get the analyzer for manual inspection
    [[nodiscard]] const SiblingFusionAnalyzer& analyzer() const { return analyzer_; }

  private:
    SiblingFusionAnalyzer analyzer_;
};

}  // namespace ir
}  // namespace bud
