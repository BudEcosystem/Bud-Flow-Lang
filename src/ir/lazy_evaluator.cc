// =============================================================================
// Bud Flow Lang - Lazy Evaluator (Weld-style)
// =============================================================================
//
// Implements deferred execution for maximum fusion opportunities.
// Key insight from Weld: deferring execution allows cross-operation fusion
// that achieves 6-32x speedups over eager evaluation.
//
// Fusion Patterns Detected:
// 1. Element-wise chains: (x * 2 + 1).sqrt().exp() -> single vectorized loop
// 2. Reduction fusion: sum(x * y) -> fused dot product (no intermediate alloc)
// 3. Broadcast elimination: scalar + vector chains -> single broadcast
//

#include "bud_flow_lang/ir.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace bud {
namespace ir {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Check if operation is element-wise (can be fused in a single loop)
bool isElementwiseOp(OpCode op) {
    switch (op) {
    // Binary arithmetic
    case OpCode::kAdd:
    case OpCode::kSub:
    case OpCode::kMul:
    case OpCode::kDiv:
    case OpCode::kMod:
    case OpCode::kMin:
    case OpCode::kMax:
    case OpCode::kPow:
    // Fused operations
    case OpCode::kFma:
    case OpCode::kFnma:
    // Unary arithmetic
    case OpCode::kNeg:
    case OpCode::kAbs:
    case OpCode::kSqrt:
    case OpCode::kRsqrt:
    case OpCode::kRcp:
    // Transcendentals
    case OpCode::kExp:
    case OpCode::kLog:
    case OpCode::kSin:
    case OpCode::kCos:
    case OpCode::kTan:
    case OpCode::kTanh:
    case OpCode::kSigmoid:
    // Comparison
    case OpCode::kEq:
    case OpCode::kNe:
    case OpCode::kLt:
    case OpCode::kLe:
    case OpCode::kGt:
    case OpCode::kGe:
    // Logical
    case OpCode::kAnd:
    case OpCode::kOr:
    case OpCode::kNot:
    case OpCode::kXor:
    // Bitwise
    case OpCode::kBitAnd:
    case OpCode::kBitOr:
    case OpCode::kBitXor:
    case OpCode::kBitNot:
    case OpCode::kShl:
    case OpCode::kShr:
    // Cast
    case OpCode::kCast:
    // Select (masked)
    case OpCode::kSelect:
        return true;
    default:
        return false;
    }
}

// Check if operation is a reduction
bool isReductionOp(OpCode op) {
    switch (op) {
    case OpCode::kReduceSum:
    case OpCode::kReduceMax:
    case OpCode::kReduceMin:
    case OpCode::kReduceProd:
    case OpCode::kReduceAny:
    case OpCode::kReduceAll:
    case OpCode::kHorizontalSum:
    case OpCode::kHorizontalMax:
    case OpCode::kHorizontalMin:
        return true;
    default:
        return false;
    }
}

// Check if operation involves broadcasting
bool isBroadcastOp(OpCode op) {
    return op == OpCode::kBroadcast;
}

// Check if a node is a constant
bool isConstantNode(const IRNode* node) {
    if (!node)
        return false;
    return node->opCode() == OpCode::kConstantScalar || node->opCode() == OpCode::kConstantVector;
}

// Build use count map for the IR
std::unordered_map<uint32_t, size_t> buildUseCountMap(const IRBuilder& builder) {
    std::unordered_map<uint32_t, size_t> use_counts;
    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead())
            continue;
        for (const auto& operand : node->operands()) {
            use_counts[operand.id]++;
        }
    }
    return use_counts;
}

}  // namespace

// =============================================================================
// Lazy Expression Graph
// =============================================================================

class LazyGraph {
  public:
    // Add a pending operation
    void addOp(ValueId result, OpCode op, const std::vector<ValueId>& inputs) {
        pending_ops_[result.id] = {op, inputs};
    }

    // Check if a value is computed
    bool isComputed(ValueId id) const { return computed_.count(id.id) > 0; }

    // Mark a value as computed
    void markComputed(ValueId id) {
        computed_.insert(id.id);
        pending_ops_.erase(id.id);
    }

    // Get pending operations in topological order
    std::vector<ValueId> getExecutionOrder(ValueId output) const {
        std::vector<ValueId> order;
        std::unordered_set<uint32_t> visited;

        topologicalSort(output, visited, order);

        return order;
    }

    // Check if graph is empty (all computed)
    bool isEmpty() const { return pending_ops_.empty(); }

    // Clear all pending operations
    void clear() {
        pending_ops_.clear();
        computed_.clear();
    }

    // Get the operation for a value ID
    const std::pair<OpCode, std::vector<ValueId>>* getOp(ValueId id) const {
        auto it = pending_ops_.find(id.id);
        if (it != pending_ops_.end()) {
            return reinterpret_cast<const std::pair<OpCode, std::vector<ValueId>>*>(&it->second);
        }
        return nullptr;
    }

  private:
    struct PendingOp {
        OpCode op;
        std::vector<ValueId> inputs;
    };

    void topologicalSort(ValueId id, std::unordered_set<uint32_t>& visited,
                         std::vector<ValueId>& order) const {
        if (visited.count(id.id) || isComputed(id)) {
            return;
        }

        visited.insert(id.id);

        auto it = pending_ops_.find(id.id);
        if (it != pending_ops_.end()) {
            for (const auto& input : it->second.inputs) {
                topologicalSort(input, visited, order);
            }
        }

        order.push_back(id);
    }

    std::unordered_map<uint32_t, PendingOp> pending_ops_;
    std::unordered_set<uint32_t> computed_;
};

// =============================================================================
// Fusion Analysis
// =============================================================================

class FusionAnalyzer {
  public:
    // Identify fusion opportunities in the IR
    struct FusionGroup {
        std::vector<ValueId> ops;  // Operations to fuse
        std::string pattern;       // Fusion pattern name
        double estimated_speedup;  // Expected performance gain
    };

    std::vector<FusionGroup> analyze(const IRBuilder& builder, ValueId output) {
        std::vector<FusionGroup> groups;

        // Build use count map for single-use detection
        use_counts_ = buildUseCountMap(builder);

        // Pattern 1: Element-wise operation chains
        // e.g., (x * 2 + 1).sqrt().exp() -> single vectorized loop
        auto elementwise = findElementwiseChains(builder, output);
        for (auto& chain : elementwise) {
            if (chain.size() >= 2) {
                // Speedup scales with chain length (fewer memory round-trips)
                double speedup = 1.0 + 0.5 * (chain.size() - 1);
                groups.push_back({chain, "elementwise_fusion", speedup});
            }
        }

        // Pattern 2: Reduction fusion
        // e.g., sum(x * y) -> fused dot product
        auto reductions = findReductionFusions(builder, output);
        groups.insert(groups.end(), reductions.begin(), reductions.end());

        // Pattern 3: Broadcast elimination
        // e.g., (scalar + vector) chains -> single broadcast
        auto broadcasts = findBroadcastFusions(builder, output);
        groups.insert(groups.end(), broadcasts.begin(), broadcasts.end());

        return groups;
    }

  private:
    std::unordered_map<uint32_t, size_t> use_counts_;

    // Find chains of element-wise operations that can be fused into a single loop
    std::vector<std::vector<ValueId>> findElementwiseChains(const IRBuilder& builder,
                                                            ValueId output) {
        std::vector<std::vector<ValueId>> chains;
        std::unordered_set<uint32_t> visited;

        // Start from each node and try to build chains
        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;
            if (visited.count(node->id().id))
                continue;

            // Start a new chain if this is an element-wise op
            if (isElementwiseOp(node->opCode())) {
                std::vector<ValueId> chain;
                buildElementwiseChain(builder, node->id(), visited, chain);

                if (chain.size() >= 2) {
                    // Reverse to get execution order (inputs before outputs)
                    std::reverse(chain.begin(), chain.end());
                    chains.push_back(chain);
                }
            }
        }

        return chains;
    }

    // Recursively build an element-wise chain from a starting node
    void buildElementwiseChain(const IRBuilder& builder, ValueId id,
                               std::unordered_set<uint32_t>& visited, std::vector<ValueId>& chain) {
        if (visited.count(id.id))
            return;

        const auto* node = builder.getNode(id);
        if (!node || node->isDead())
            return;

        // Only include element-wise operations
        if (!isElementwiseOp(node->opCode()))
            return;

        visited.insert(id.id);
        chain.push_back(id);

        // Follow operands that are single-use element-wise ops
        for (const auto& operand : node->operands()) {
            const auto* op_node = builder.getNode(operand);
            if (!op_node || op_node->isDead())
                continue;

            // Only extend chain through single-use element-wise ops
            // Multi-use nodes break the chain (values needed elsewhere)
            if (isElementwiseOp(op_node->opCode()) && use_counts_[operand.id] == 1) {
                buildElementwiseChain(builder, operand, visited, chain);
            }
        }
    }

    // Find reduction patterns that can be fused
    std::vector<FusionGroup> findReductionFusions(const IRBuilder& builder, ValueId output) {
        std::vector<FusionGroup> fusions;

        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;

            // Look for reduction operations
            if (!isReductionOp(node->opCode()))
                continue;

            // Check if the reduction input is a single-use operation
            if (node->numOperands() < 1)
                continue;

            ValueId input_id = node->operand(0);
            const auto* input_node = builder.getNode(input_id);
            if (!input_node || input_node->isDead())
                continue;

            // Pattern: reduce_sum(mul(x, y)) -> dot product
            if (node->opCode() == OpCode::kReduceSum && input_node->opCode() == OpCode::kMul &&
                use_counts_[input_id.id] == 1) {

                FusionGroup group;
                group.ops = {input_id, node->id()};
                group.pattern = "dot_product";
                group.estimated_speedup = 2.0;  // No intermediate allocation

                spdlog::debug("Fusion: Found dot_product pattern at node %{}", node->id().id);
                fusions.push_back(group);
                continue;
            }

            // Pattern: reduce_sum(x * x) -> norm squared
            if (node->opCode() == OpCode::kReduceSum && input_node->opCode() == OpCode::kMul &&
                input_node->numOperands() == 2 &&
                input_node->operand(0) == input_node->operand(1) && use_counts_[input_id.id] == 1) {

                FusionGroup group;
                group.ops = {input_id, node->id()};
                group.pattern = "norm_squared";
                group.estimated_speedup = 2.5;

                spdlog::debug("Fusion: Found norm_squared pattern at node %{}", node->id().id);
                fusions.push_back(group);
                continue;
            }

            // Pattern: reduce_sum(sub(x, y) * sub(x, y)) -> squared distance
            if (node->opCode() == OpCode::kReduceSum && input_node->opCode() == OpCode::kMul &&
                use_counts_[input_id.id] == 1) {

                // Check if both mul operands are the same subtraction
                if (input_node->numOperands() == 2 &&
                    input_node->operand(0) == input_node->operand(1)) {

                    const auto* sub_node = builder.getNode(input_node->operand(0));
                    if (sub_node && sub_node->opCode() == OpCode::kSub) {
                        FusionGroup group;
                        group.ops = {input_node->operand(0), input_id, node->id()};
                        group.pattern = "squared_distance";
                        group.estimated_speedup = 3.0;

                        spdlog::debug("Fusion: Found squared_distance pattern at node %{}",
                                      node->id().id);
                        fusions.push_back(group);
                    }
                }
            }

            // Pattern: reduce_max/reduce_min of element-wise chain
            if ((node->opCode() == OpCode::kReduceMax || node->opCode() == OpCode::kReduceMin) &&
                isElementwiseOp(input_node->opCode()) && use_counts_[input_id.id] == 1) {

                FusionGroup group;
                group.ops = {input_id, node->id()};
                group.pattern = "fused_reduce_" +
                                std::string(node->opCode() == OpCode::kReduceMax ? "max" : "min");
                group.estimated_speedup = 1.5;

                spdlog::debug("Fusion: Found fused_reduce pattern at node %{}", node->id().id);
                fusions.push_back(group);
            }
        }

        return fusions;
    }

    // Find broadcast operations that can be eliminated or fused
    std::vector<FusionGroup> findBroadcastFusions(const IRBuilder& builder, ValueId output) {
        std::vector<FusionGroup> fusions;

        // Track broadcast operations and their consumers
        std::unordered_map<uint32_t, std::vector<ValueId>> broadcast_users;

        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;

            for (const auto& operand : node->operands()) {
                const auto* op_node = builder.getNode(operand);
                if (op_node && isBroadcastOp(op_node->opCode())) {
                    broadcast_users[operand.id].push_back(node->id());
                }
            }
        }

        // Look for chains of operations using the same broadcast
        for (const auto& [broadcast_id, users] : broadcast_users) {
            if (users.size() >= 2) {
                // Multiple operations use the same broadcast
                // Can potentially hoist the broadcast outside the chain

                FusionGroup group;
                group.ops.push_back(ValueId{broadcast_id});
                for (const auto& user : users) {
                    group.ops.push_back(user);
                }
                group.pattern = "broadcast_hoisting";
                group.estimated_speedup = 1.3;

                spdlog::debug("Fusion: Found broadcast_hoisting pattern with {} users",
                              users.size());
                fusions.push_back(group);
            }
        }

        // Look for consecutive broadcasts that can be merged
        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;
            if (!isBroadcastOp(node->opCode()))
                continue;

            // Check if input to this broadcast is also a broadcast
            if (node->numOperands() > 0) {
                const auto* input = builder.getNode(node->operand(0));
                if (input && isBroadcastOp(input->opCode()) && use_counts_[input->id().id] == 1) {

                    FusionGroup group;
                    group.ops = {input->id(), node->id()};
                    group.pattern = "broadcast_merge";
                    group.estimated_speedup = 1.5;

                    spdlog::debug("Fusion: Found broadcast_merge pattern at node %{}",
                                  node->id().id);
                    fusions.push_back(group);
                }
            }
        }

        // Look for scalar-vector operations that could use implicit broadcast
        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;

            // Check binary ops where one operand is a constant (implicit broadcast)
            if (isElementwiseOp(node->opCode()) && node->numOperands() == 2) {
                const auto* lhs = builder.getNode(node->operand(0));
                const auto* rhs = builder.getNode(node->operand(1));

                bool lhs_is_scalar = lhs && isConstantNode(lhs) && lhs->type().isScalar();
                bool rhs_is_scalar = rhs && isConstantNode(rhs) && rhs->type().isScalar();

                // If one operand is a scalar constant, no explicit broadcast needed
                if ((lhs_is_scalar || rhs_is_scalar) && !(lhs_is_scalar && rhs_is_scalar)) {
                    // Check if there's an explicit broadcast that can be eliminated
                    const IRNode* scalar_node = lhs_is_scalar ? lhs : rhs;
                    const IRNode* vector_node = lhs_is_scalar ? rhs : lhs;

                    if (vector_node && isBroadcastOp(vector_node->opCode())) {
                        FusionGroup group;
                        group.ops = {vector_node->id(), node->id()};
                        group.pattern = "implicit_broadcast";
                        group.estimated_speedup = 1.2;

                        spdlog::debug("Fusion: Found implicit_broadcast pattern at node %{}",
                                      node->id().id);
                        fusions.push_back(group);
                    }
                }
            }
        }

        return fusions;
    }
};

// =============================================================================
// Lazy Evaluator
// =============================================================================

class LazyEvaluator {
  public:
    LazyEvaluator() = default;

    // Defer an operation (don't execute immediately)
    void defer(ValueId result, OpCode op, const std::vector<ValueId>& inputs) {
        graph_.addOp(result, op, inputs);
        spdlog::trace("Deferred op {} -> %{}", opCodeName(op), result.id);
    }

    // Force evaluation of a value and all its dependencies
    Result<void> evaluate(IRBuilder& builder, ValueId output) {
        if (graph_.isComputed(output)) {
            return {};  // Already computed
        }

        // Get execution order
        auto order = graph_.getExecutionOrder(output);

        spdlog::debug("Evaluating {} deferred operations", order.size());

        // Analyze fusion opportunities
        FusionAnalyzer analyzer;
        auto fusions = analyzer.analyze(builder, output);

        if (!fusions.empty()) {
            spdlog::debug("Found {} fusion opportunities:", fusions.size());
            double total_speedup = 1.0;
            for (const auto& fusion : fusions) {
                spdlog::debug("  - {} ({} ops, {:.1f}x speedup)", fusion.pattern, fusion.ops.size(),
                              fusion.estimated_speedup);
                total_speedup *= fusion.estimated_speedup;
            }
            spdlog::debug("  Total estimated speedup: {:.1f}x", total_speedup);

            // Apply fusions by modifying the execution plan
            // This will be connected to the JIT compiler which generates
            // fused kernel code instead of separate operations
        }

        // Execute in order
        for (const auto& id : order) {
            // Mark as computed (actual execution delegated to runtime)
            graph_.markComputed(id);
        }

        return {};
    }

    // Check if there are pending operations
    bool hasPending() const { return !graph_.isEmpty(); }

    // Clear all deferred operations
    void reset() { graph_.clear(); }

    // Get fusion analysis for current pending operations
    std::vector<FusionAnalyzer::FusionGroup> analyzeFusions(const IRBuilder& builder,
                                                            ValueId output) const {
        FusionAnalyzer analyzer;
        return analyzer.analyze(builder, output);
    }

  private:
    LazyGraph graph_;
};

// =============================================================================
// Global Lazy Evaluator Instance
// =============================================================================

namespace {
thread_local LazyEvaluator g_lazy_evaluator;
}

LazyEvaluator& getLazyEvaluator() {
    return g_lazy_evaluator;
}

// =============================================================================
// Public Analysis Functions
// =============================================================================

// Analyze fusion opportunities for an IR module
std::vector<std::pair<std::string, double>> analyzeFusionOpportunities(const IRModule& module) {
    std::vector<std::pair<std::string, double>> results;

    FusionAnalyzer analyzer;
    auto fusions = analyzer.analyze(module.builder(), module.output());

    for (const auto& fusion : fusions) {
        results.emplace_back(fusion.pattern, fusion.estimated_speedup);
    }

    return results;
}

// Get estimated speedup from fusion for an IR module
double estimateFusionSpeedup(const IRModule& module) {
    FusionAnalyzer analyzer;
    auto fusions = analyzer.analyze(module.builder(), module.output());

    double total_speedup = 1.0;
    for (const auto& fusion : fusions) {
        total_speedup *= fusion.estimated_speedup;
    }

    return total_speedup;
}

}  // namespace ir
}  // namespace bud
