// =============================================================================
// Bud Flow Lang - Sibling (Multi-Output) Fusion Implementation
// =============================================================================

#include "bud_flow_lang/ir/sibling_fusion.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace bud {
namespace ir {

namespace {

// Bytes per element for different scalar types
size_t bytesPerElement(ScalarType type) {
    switch (type) {
    case ScalarType::kFloat32:
        return 4;
    case ScalarType::kFloat64:
        return 8;
    case ScalarType::kInt32:
        return 4;
    case ScalarType::kInt64:
        return 8;
    case ScalarType::kBool:
        return 1;
    default:
        return 4;
    }
}

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
// SiblingFusionAnalyzer Implementation
// =============================================================================

std::vector<SiblingGroup> SiblingFusionAnalyzer::findSiblingGroups(const IRBuilder& builder) const {
    std::vector<SiblingGroup> groups;

    // Build a map: input_id -> list of operations that use this input
    std::unordered_map<uint32_t, std::vector<ValueId>> input_consumers;

    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        // Only consider computational operations
        if (!isComputeOp(node->opCode())) {
            continue;
        }

        // Track which inputs this operation uses
        for (size_t i = 0; i < node->numOperands(); ++i) {
            ValueId operand = node->operand(i);
            input_consumers[operand.id].push_back(node->id());
        }
    }

    // Find inputs that are used by multiple operations (potential sibling groups)
    std::unordered_set<uint64_t> processed_pairs;

    for (const auto& [input_id, consumers] : input_consumers) {
        if (consumers.size() < 2) {
            continue;
        }

        // Create a sibling group from all consumers of this input
        SiblingGroup group;
        group.shared_inputs.push_back(ValueId{input_id});

        // Filter to keep only operations that can be fused
        for (const auto& consumer : consumers) {
            const auto* node = builder.getNode(consumer);
            if (node && !node->isDead() && isComputeOp(node->opCode())) {
                group.siblings.push_back(consumer);
            }
        }

        if (group.siblings.size() >= 2) {
            // Check for additional shared inputs
            std::vector<ValueId> additional_shared;
            const auto* first_node = builder.getNode(group.siblings[0]);
            if (first_node) {
                for (size_t i = 0; i < first_node->numOperands(); ++i) {
                    ValueId operand = first_node->operand(i);
                    if (operand.id == input_id) {
                        continue;  // Already in shared_inputs
                    }

                    // Check if all siblings use this operand
                    bool all_use_it = true;
                    for (size_t j = 1; j < group.siblings.size(); ++j) {
                        const auto* sibling_node = builder.getNode(group.siblings[j]);
                        if (!sibling_node) {
                            all_use_it = false;
                            break;
                        }

                        bool found = false;
                        for (size_t k = 0; k < sibling_node->numOperands(); ++k) {
                            if (sibling_node->operand(k).id == operand.id) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            all_use_it = false;
                            break;
                        }
                    }

                    if (all_use_it) {
                        additional_shared.push_back(operand);
                    }
                }
            }

            // Add additional shared inputs
            for (const auto& shared : additional_shared) {
                group.shared_inputs.push_back(shared);
            }

            // Calculate benefit
            group.estimated_benefit = calculateBenefit(builder, group);

            // Check if this is a unique group (avoid duplicates)
            // Create a canonical key for the group
            std::vector<uint32_t> sorted_siblings;
            for (const auto& s : group.siblings) {
                sorted_siblings.push_back(s.id);
            }
            std::sort(sorted_siblings.begin(), sorted_siblings.end());

            uint64_t group_key = 0;
            for (auto id : sorted_siblings) {
                group_key = group_key * 31 + id;
            }

            if (processed_pairs.find(group_key) == processed_pairs.end()) {
                processed_pairs.insert(group_key);
                groups.push_back(std::move(group));
            }
        }
    }

    // Sort by estimated benefit (highest first)
    std::sort(groups.begin(), groups.end(), [](const SiblingGroup& a, const SiblingGroup& b) {
        return a.estimated_benefit > b.estimated_benefit;
    });

    return groups;
}

bool SiblingFusionAnalyzer::canFuseSiblings(const IRBuilder& builder,
                                            const std::vector<ValueId>& siblings) const {
    if (siblings.size() < 2) {
        return false;
    }

    // Check for data dependencies between siblings
    for (size_t i = 0; i < siblings.size(); ++i) {
        for (size_t j = i + 1; j < siblings.size(); ++j) {
            if (hasDataDependency(builder, siblings[i], siblings[j])) {
                return false;
            }
        }
    }

    // Check type compatibility
    if (!areTypesCompatible(builder, siblings)) {
        return false;
    }

    return true;
}

std::vector<ValueId>
SiblingFusionAnalyzer::createMultiOutputFusion(IRBuilder& builder,
                                               const SiblingGroup& group) const {
    // For now, we create a simple fusion by marking the group as fused
    // The actual code generation will handle this at the codegen level

    std::vector<ValueId> outputs;

    // For each sibling, we keep the original operation but mark them as part
    // of a fusion group. The actual fusion happens at code generation time.
    // This preserves the IR semantics while enabling the backend to optimize.

    for (const auto& sibling : group.siblings) {
        outputs.push_back(sibling);
    }

    spdlog::debug("Created multi-output fusion with {} outputs, {} shared inputs", outputs.size(),
                  group.shared_inputs.size());

    return outputs;
}

bool SiblingFusionAnalyzer::hasDataDependency(const IRBuilder& builder, ValueId op1,
                                              ValueId op2) const {
    // Check if op2 depends on op1 (op1 is an operand of op2)
    const auto* node2 = builder.getNode(op2);
    if (node2) {
        for (size_t i = 0; i < node2->numOperands(); ++i) {
            if (node2->operand(i).id == op1.id) {
                return true;
            }
        }
    }

    // Check if op1 depends on op2 (op2 is an operand of op1)
    const auto* node1 = builder.getNode(op1);
    if (node1) {
        for (size_t i = 0; i < node1->numOperands(); ++i) {
            if (node1->operand(i).id == op2.id) {
                return true;
            }
        }
    }

    return false;
}

bool SiblingFusionAnalyzer::areTypesCompatible(const IRBuilder& builder,
                                               const std::vector<ValueId>& siblings) const {
    if (siblings.empty()) {
        return true;
    }

    const auto* first_node = builder.getNode(siblings[0]);
    if (!first_node) {
        return false;
    }

    ScalarType first_type = first_node->type().scalarType();
    size_t first_elements = first_node->type().elementCount();

    for (size_t i = 1; i < siblings.size(); ++i) {
        const auto* node = builder.getNode(siblings[i]);
        if (!node) {
            return false;
        }

        // Check scalar type matches
        if (node->type().scalarType() != first_type) {
            return false;
        }

        // Check element count matches
        if (node->type().elementCount() != first_elements) {
            return false;
        }
    }

    return true;
}

float SiblingFusionAnalyzer::calculateBenefit(const IRBuilder& builder,
                                              const SiblingGroup& group) const {
    if (group.siblings.size() < 2 || group.shared_inputs.empty()) {
        return 0.0f;
    }

    // Calculate memory savings from sharing inputs
    float memory_saved = 0.0f;

    for (const auto& shared_input : group.shared_inputs) {
        const auto* input_node = builder.getNode(shared_input);
        if (!input_node) {
            continue;
        }

        size_t bytes =
            input_node->type().elementCount() * bytesPerElement(input_node->type().scalarType());

        // Without fusion: each sibling reads the input separately
        // With fusion: input is read once and used by all siblings
        // Savings = (num_siblings - 1) * input_bytes
        memory_saved += static_cast<float>(bytes) * (group.siblings.size() - 1);
    }

    // Scale by memory bandwidth cost (assume ~50 GB/s bandwidth)
    constexpr float kBandwidth = 50e9f;  // bytes/second
    float time_saved = memory_saved / kBandwidth;

    // Convert to a benefit score (microseconds saved)
    return time_saved * 1e6f;
}

std::vector<ValueId> SiblingFusionAnalyzer::findConsumers(const IRBuilder& builder,
                                                          ValueId value) const {
    std::vector<ValueId> consumers;

    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        for (size_t i = 0; i < node->numOperands(); ++i) {
            if (node->operand(i).id == value.id) {
                consumers.push_back(node->id());
                break;
            }
        }
    }

    return consumers;
}

// =============================================================================
// SiblingFusionPass Implementation
// =============================================================================

size_t SiblingFusionPass::run(IRBuilder& builder) {
    // Find all sibling groups
    auto groups = analyzer_.findSiblingGroups(builder);

    if (groups.empty()) {
        return 0;
    }

    size_t fused_count = 0;
    std::unordered_set<uint32_t> fused_ops;

    for (const auto& group : groups) {
        // Skip if any sibling is already fused
        bool already_fused = false;
        for (const auto& sibling : group.siblings) {
            if (fused_ops.find(sibling.id) != fused_ops.end()) {
                already_fused = true;
                break;
            }
        }

        if (already_fused) {
            continue;
        }

        // Check if this group can be fused
        if (!analyzer_.canFuseSiblings(builder, group.siblings)) {
            continue;
        }

        // Skip groups with low benefit
        if (group.estimated_benefit <= 0.0f) {
            continue;
        }

        // Mark siblings as fused (actual fusion happens at codegen)
        auto outputs = analyzer_.createMultiOutputFusion(builder, group);

        if (!outputs.empty()) {
            // Mark all siblings as part of a fusion group
            for (const auto& sibling : group.siblings) {
                fused_ops.insert(sibling.id);
            }
            ++fused_count;

            spdlog::debug("Fused sibling group: {} siblings, benefit: {:.2f}us",
                          group.siblings.size(), group.estimated_benefit);
        }
    }

    return fused_count;
}

}  // namespace ir
}  // namespace bud
