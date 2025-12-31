#pragma once

// =============================================================================
// Bud Flow Lang - Horizontal Fusion
// =============================================================================
//
// Batches independent operations into a single kernel launch to reduce
// per-kernel overhead (~5-10µs per launch).
//
// Pattern: Independent operations A, B, C -> Batch into single launch
//
// Benefits:
// - Reduces kernel launch overhead
// - Increases thread utilization
// - Better instruction cache locality
//

#include "bud_flow_lang/ir.h"

#include <vector>

namespace bud {
namespace ir {

// =============================================================================
// Independent Operation Group
// =============================================================================

/// Represents a group of independent operations that can be batched
struct IndependentGroup {
    /// Operations in this group (all mutually independent)
    std::vector<ValueId> operations;

    /// Estimated benefit of batching (launch overhead saved)
    float estimated_benefit = 0.0f;
};

// =============================================================================
// HorizontalFusionAnalyzer
// =============================================================================

/// Analyzes IR to find horizontal fusion opportunities
class HorizontalFusionAnalyzer {
  public:
    HorizontalFusionAnalyzer() = default;

    /// Find groups of independent operations that can be batched
    /// @param builder The IR builder to analyze
    /// @return Vector of independent groups, sorted by estimated benefit
    [[nodiscard]] std::vector<IndependentGroup>
    findIndependentGroups(const IRBuilder& builder) const;

    /// Check if a set of operations can be batched together
    /// @param builder The IR builder
    /// @param operations The operation IDs to check
    /// @return true if operations can be batched (all mutually independent)
    [[nodiscard]] bool canBatchOperations(const IRBuilder& builder,
                                          const std::vector<ValueId>& operations) const;

    /// Estimate the benefit of batching a group of operations
    /// @param builder The IR builder
    /// @param group The group to estimate
    /// @return Estimated benefit in microseconds saved
    [[nodiscard]] float estimateBatchBenefit(const IRBuilder& builder,
                                             const IndependentGroup& group) const;

    /// Create a batched kernel from a group of independent operations
    /// @param builder The IR builder to modify
    /// @param group The group to batch
    /// @return Vector of new output ValueIds (one per original operation)
    [[nodiscard]] std::vector<ValueId> createBatchedKernel(IRBuilder& builder,
                                                           const IndependentGroup& group) const;

  private:
    /// Check if there's any dependency between two operations (direct or transitive)
    [[nodiscard]] bool hasDependency(const IRBuilder& builder, ValueId op1, ValueId op2) const;

    /// Check if op1 transitively depends on op2
    [[nodiscard]] bool dependsOn(const IRBuilder& builder, ValueId op1, ValueId op2) const;

    /// Find all compute operations (non-constants, non-inputs)
    [[nodiscard]] std::vector<ValueId> findComputeOperations(const IRBuilder& builder) const;

    /// Compute the level (topological depth) of each operation
    [[nodiscard]] std::vector<size_t> computeLevels(const IRBuilder& builder) const;
};

// =============================================================================
// HorizontalFusionPass
// =============================================================================

/// Optimization pass that performs horizontal fusion
class HorizontalFusionPass {
  public:
    HorizontalFusionPass() = default;

    /// Run the horizontal fusion pass
    /// @param builder The IR builder to optimize
    /// @return Number of batched groups applied
    size_t run(IRBuilder& builder);

    /// Get the analyzer for manual inspection
    [[nodiscard]] const HorizontalFusionAnalyzer& analyzer() const { return analyzer_; }

  private:
    HorizontalFusionAnalyzer analyzer_;
};

}  // namespace ir
}  // namespace bud
