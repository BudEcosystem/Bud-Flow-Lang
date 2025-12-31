// =============================================================================
// Bud Flow Lang - Horizontal Fusion Implementation
// =============================================================================

#include "bud_flow_lang/ir/horizontal_fusion.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace bud {
namespace ir {

namespace {

// Estimated kernel launch overhead in microseconds
constexpr float kKernelLaunchOverhead = 5.0f;

// Check if an operation is a computational operation (not a constant or input)
bool isComputeOp(OpCode op) {
    switch (op) {
    case OpCode::kAdd:
    case OpCode::kSub:
    case OpCode::kMul:
    case OpCode::kDiv:
    case OpCode::kNeg:
    case OpCode::kAbs:
    case OpCode::kSqrt:
    case OpCode::kExp:
    case OpCode::kLog:
    case OpCode::kSin:
    case OpCode::kCos:
    case OpCode::kTan:
    case OpCode::kFma:
    case OpCode::kMin:
    case OpCode::kMax:
    case OpCode::kReduceSum:
    case OpCode::kReduceMax:
    case OpCode::kReduceMin:
        return true;
    default:
        return false;
    }
}

}  // namespace

// =============================================================================
// HorizontalFusionAnalyzer Implementation
// =============================================================================

std::vector<IndependentGroup>
HorizontalFusionAnalyzer::findIndependentGroups(const IRBuilder& builder) const {
    std::vector<IndependentGroup> groups;

    // Compute topological levels for all operations
    auto levels = computeLevels(builder);

    // Find all compute operations
    auto compute_ops = findComputeOperations(builder);

    // Group operations by level - operations at the same level are independent
    std::unordered_map<size_t, std::vector<ValueId>> level_groups;
    for (const auto& op : compute_ops) {
        if (op.id < levels.size()) {
            level_groups[levels[op.id]].push_back(op);
        }
    }

    // Convert level groups to IndependentGroups
    for (auto& [level, ops] : level_groups) {
        if (ops.size() >= 2) {
            // Verify operations are truly independent (same level doesn't guarantee it)
            std::vector<ValueId> verified_independent;

            for (const auto& op : ops) {
                bool is_independent = true;
                for (const auto& other : verified_independent) {
                    if (hasDependency(builder, op, other)) {
                        is_independent = false;
                        break;
                    }
                }
                if (is_independent) {
                    verified_independent.push_back(op);
                }
            }

            if (verified_independent.size() >= 2) {
                IndependentGroup group;
                group.operations = std::move(verified_independent);
                group.estimated_benefit = estimateBatchBenefit(builder, group);
                groups.push_back(std::move(group));
            }
        }
    }

    // Sort by estimated benefit (highest first)
    std::sort(groups.begin(), groups.end(),
              [](const IndependentGroup& a, const IndependentGroup& b) {
                  return a.estimated_benefit > b.estimated_benefit;
              });

    return groups;
}

bool HorizontalFusionAnalyzer::canBatchOperations(const IRBuilder& builder,
                                                  const std::vector<ValueId>& operations) const {
    if (operations.size() < 2) {
        return false;
    }

    // Check that all operations are mutually independent
    for (size_t i = 0; i < operations.size(); ++i) {
        for (size_t j = i + 1; j < operations.size(); ++j) {
            if (hasDependency(builder, operations[i], operations[j])) {
                return false;
            }
        }
    }

    return true;
}

float HorizontalFusionAnalyzer::estimateBatchBenefit(const IRBuilder& builder,
                                                     const IndependentGroup& group) const {
    if (group.operations.size() <= 1) {
        return 0.0f;
    }

    // Benefit = (num_operations - 1) * kernel_launch_overhead
    // Batching N operations into 1 saves (N-1) launches
    float saved_launches = static_cast<float>(group.operations.size() - 1);
    float benefit = saved_launches * kKernelLaunchOverhead;

    // Reduce benefit if operations have different types (less efficient batching)
    if (!group.operations.empty()) {
        const auto* first_node = builder.getNode(group.operations[0]);
        if (first_node) {
            ScalarType first_type = first_node->type().scalarType();
            for (size_t i = 1; i < group.operations.size(); ++i) {
                const auto* node = builder.getNode(group.operations[i]);
                if (node && node->type().scalarType() != first_type) {
                    benefit *= 0.5f;  // Type mismatch penalty
                    break;
                }
            }
        }
    }

    return benefit;
}

std::vector<ValueId>
HorizontalFusionAnalyzer::createBatchedKernel(IRBuilder& builder,
                                              const IndependentGroup& group) const {
    // For now, we don't modify the IR structure - batching is done at codegen time
    // The operations are marked as part of a batch group and the backend
    // will generate a single fused kernel

    std::vector<ValueId> outputs;
    for (const auto& op : group.operations) {
        outputs.push_back(op);
    }

    spdlog::debug("Created batched kernel with {} operations, benefit: {:.2f}us", outputs.size(),
                  group.estimated_benefit);

    return outputs;
}

bool HorizontalFusionAnalyzer::hasDependency(const IRBuilder& builder, ValueId op1,
                                             ValueId op2) const {
    // Check both directions
    return dependsOn(builder, op1, op2) || dependsOn(builder, op2, op1);
}

bool HorizontalFusionAnalyzer::dependsOn(const IRBuilder& builder, ValueId op1, ValueId op2) const {
    // BFS to check if op1 transitively depends on op2
    std::queue<ValueId> work_queue;
    std::unordered_set<uint32_t> visited;

    work_queue.push(op1);
    visited.insert(op1.id);

    while (!work_queue.empty()) {
        ValueId current = work_queue.front();
        work_queue.pop();

        const auto* node = builder.getNode(current);
        if (!node) {
            continue;
        }

        for (size_t i = 0; i < node->numOperands(); ++i) {
            ValueId operand = node->operand(i);

            if (operand.id == op2.id) {
                return true;  // Found dependency
            }

            if (visited.find(operand.id) == visited.end()) {
                visited.insert(operand.id);
                work_queue.push(operand);
            }
        }
    }

    return false;
}

std::vector<ValueId>
HorizontalFusionAnalyzer::findComputeOperations(const IRBuilder& builder) const {
    std::vector<ValueId> compute_ops;

    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        if (isComputeOp(node->opCode())) {
            compute_ops.push_back(node->id());
        }
    }

    return compute_ops;
}

std::vector<size_t> HorizontalFusionAnalyzer::computeLevels(const IRBuilder& builder) const {
    // Compute topological level for each node
    // Level = 1 + max(level of all operands)
    // Constants/inputs have level 0

    size_t num_nodes = builder.nodes().size();
    std::vector<size_t> levels(num_nodes + 1, 0);

    // Process nodes in order (they should be topologically sorted already)
    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        size_t max_operand_level = 0;
        for (size_t i = 0; i < node->numOperands(); ++i) {
            ValueId operand = node->operand(i);
            if (operand.id < levels.size()) {
                max_operand_level = std::max(max_operand_level, levels[operand.id]);
            }
        }

        if (node->id().id < levels.size()) {
            levels[node->id().id] = max_operand_level + 1;
        }
    }

    return levels;
}

// =============================================================================
// HorizontalFusionPass Implementation
// =============================================================================

size_t HorizontalFusionPass::run(IRBuilder& builder) {
    // Find all independent groups
    auto groups = analyzer_.findIndependentGroups(builder);

    if (groups.empty()) {
        return 0;
    }

    size_t batched_count = 0;
    std::unordered_set<uint32_t> processed_ops;

    for (const auto& group : groups) {
        // Skip if any operation in this group is already processed
        bool already_processed = false;
        for (const auto& op : group.operations) {
            if (processed_ops.find(op.id) != processed_ops.end()) {
                already_processed = true;
                break;
            }
        }

        if (already_processed) {
            continue;
        }

        // Verify the group can still be batched
        if (!analyzer_.canBatchOperations(builder, group.operations)) {
            continue;
        }

        // Skip groups with low benefit
        if (group.estimated_benefit <= 0.0f) {
            continue;
        }

        // Create the batched kernel
        auto outputs = analyzer_.createBatchedKernel(builder, group);

        if (!outputs.empty()) {
            // Mark all operations as processed
            for (const auto& op : group.operations) {
                processed_ops.insert(op.id);
            }
            ++batched_count;

            spdlog::debug("Batched {} operations, benefit: {:.2f}us", group.operations.size(),
                          group.estimated_benefit);
        }
    }

    return batched_count;
}

}  // namespace ir
}  // namespace bud
