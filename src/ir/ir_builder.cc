// =============================================================================
// Bud Flow Lang - IR Builder Implementation
// =============================================================================

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/ir/fusion_cost_model.h"
#include "bud_flow_lang/ir/horizontal_fusion.h"
#include "bud_flow_lang/ir/sibling_fusion.h"

#include <fmt/format.h>

namespace bud {
namespace ir {

// =============================================================================
// IRBuilder Implementation
// =============================================================================

IRBuilder::IRBuilder(Arena& arena) : arena_(arena) {}

ValueId IRBuilder::createNode(OpCode op, TypeDesc type) {
    ValueId id{next_id_++};
    auto* node = arena_.create<IRNode>(op, type, id);
    if (!node) {
        // Allocation failed - revert id and return invalid
        --next_id_;
        return ValueId::invalid();
    }
    nodes_.push_back(node);
    return id;
}

ValueId IRBuilder::createBinaryOp(OpCode op, ValueId lhs, ValueId rhs) {
    auto* lhs_node = getNode(lhs);
    auto* rhs_node = getNode(rhs);

    if (!lhs_node || !rhs_node) {
        return ValueId::invalid();
    }

    auto type_result = TypeInferrer::inferBinaryOp(lhs_node->type(), rhs_node->type());
    if (!type_result) {
        return ValueId::invalid();
    }

    ValueId id = createNode(op, *type_result);
    if (!id.isValid()) {
        return ValueId::invalid();  // Allocation failed
    }
    auto* node = getNode(id);
    if (!node) {
        return ValueId::invalid();  // Should not happen, but safety check
    }
    node->addOperand(lhs);
    node->addOperand(rhs);
    return id;
}

ValueId IRBuilder::createUnaryOp(OpCode op, ValueId operand) {
    auto* operand_node = getNode(operand);
    if (!operand_node) {
        return ValueId::invalid();
    }

    auto type_result = TypeInferrer::inferUnaryOp(operand_node->type());
    if (!type_result) {
        return ValueId::invalid();
    }

    ValueId id = createNode(op, *type_result);
    if (!id.isValid()) {
        return ValueId::invalid();
    }
    auto* node = getNode(id);
    if (!node) {
        return ValueId::invalid();
    }
    node->addOperand(operand);
    return id;
}

// Constants
ValueId IRBuilder::constant(float value) {
    ValueId id = createNode(OpCode::kConstantScalar, TypeDesc::f32());
    if (!id.isValid()) {
        return ValueId::invalid();
    }
    auto* node = getNode(id);
    if (!node) {
        return ValueId::invalid();
    }
    node->setFloatAttr("value", static_cast<double>(value));
    return id;
}

ValueId IRBuilder::constant(double value) {
    ValueId id = createNode(OpCode::kConstantScalar, TypeDesc::f64());
    if (!id.isValid()) {
        return ValueId::invalid();
    }
    auto* node = getNode(id);
    if (!node) {
        return ValueId::invalid();
    }
    node->setFloatAttr("value", value);
    return id;
}

ValueId IRBuilder::constant(int32_t value) {
    ValueId id = createNode(OpCode::kConstantScalar, TypeDesc::i32());
    auto* node = getNode(id);
    node->setIntAttr("value", static_cast<int64_t>(value));
    return id;
}

ValueId IRBuilder::constant(int64_t value) {
    ValueId id = createNode(OpCode::kConstantScalar, TypeDesc::i64());
    auto* node = getNode(id);
    node->setIntAttr("value", value);
    return id;
}

ValueId IRBuilder::constantVector(const std::vector<float>& values) {
    TypeDesc type(ScalarType::kFloat32, Shape::vector(values.size()));
    ValueId id = createNode(OpCode::kConstantVector, type);
    // Store values as attribute (serialized)
    // TODO: Better storage for large vectors
    return id;
}

// Binary operations
ValueId IRBuilder::add(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kAdd, lhs, rhs);
}

ValueId IRBuilder::sub(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kSub, lhs, rhs);
}

ValueId IRBuilder::mul(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kMul, lhs, rhs);
}

ValueId IRBuilder::div(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kDiv, lhs, rhs);
}

ValueId IRBuilder::min(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kMin, lhs, rhs);
}

ValueId IRBuilder::max(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kMax, lhs, rhs);
}

// Fused operations
ValueId IRBuilder::fma(ValueId a, ValueId b, ValueId c) {
    auto* a_node = getNode(a);
    auto* b_node = getNode(b);
    auto* c_node = getNode(c);

    if (!a_node || !b_node || !c_node) {
        return ValueId::invalid();
    }

    // FMA result type is same as inputs (assuming all match)
    ValueId id = createNode(OpCode::kFma, a_node->type());
    auto* node = getNode(id);
    node->addOperand(a);
    node->addOperand(b);
    node->addOperand(c);
    return id;
}

// Unary operations
ValueId IRBuilder::neg(ValueId x) {
    return createUnaryOp(OpCode::kNeg, x);
}

ValueId IRBuilder::abs(ValueId x) {
    return createUnaryOp(OpCode::kAbs, x);
}

ValueId IRBuilder::sqrt(ValueId x) {
    return createUnaryOp(OpCode::kSqrt, x);
}

ValueId IRBuilder::rsqrt(ValueId x) {
    return createUnaryOp(OpCode::kRsqrt, x);
}

// Transcendentals
ValueId IRBuilder::exp(ValueId x) {
    return createUnaryOp(OpCode::kExp, x);
}

ValueId IRBuilder::log(ValueId x) {
    return createUnaryOp(OpCode::kLog, x);
}

ValueId IRBuilder::sin(ValueId x) {
    return createUnaryOp(OpCode::kSin, x);
}

ValueId IRBuilder::cos(ValueId x) {
    return createUnaryOp(OpCode::kCos, x);
}

ValueId IRBuilder::tanh(ValueId x) {
    return createUnaryOp(OpCode::kTanh, x);
}

// Reductions
ValueId IRBuilder::reduceSum(ValueId x, int axis) {
    auto* x_node = getNode(x);
    if (!x_node)
        return ValueId::invalid();

    auto type_result = TypeInferrer::inferReduction(x_node->type(), axis);
    if (!type_result)
        return ValueId::invalid();

    ValueId id = createNode(OpCode::kReduceSum, *type_result);
    auto* node = getNode(id);
    node->addOperand(x);
    node->setIntAttr("axis", axis);
    return id;
}

ValueId IRBuilder::reduceMax(ValueId x, int axis) {
    auto* x_node = getNode(x);
    if (!x_node)
        return ValueId::invalid();

    auto type_result = TypeInferrer::inferReduction(x_node->type(), axis);
    if (!type_result)
        return ValueId::invalid();

    ValueId id = createNode(OpCode::kReduceMax, *type_result);
    auto* node = getNode(id);
    node->addOperand(x);
    node->setIntAttr("axis", axis);
    return id;
}

ValueId IRBuilder::reduceMin(ValueId x, int axis) {
    auto* x_node = getNode(x);
    if (!x_node)
        return ValueId::invalid();

    auto type_result = TypeInferrer::inferReduction(x_node->type(), axis);
    if (!type_result)
        return ValueId::invalid();

    ValueId id = createNode(OpCode::kReduceMin, *type_result);
    auto* node = getNode(id);
    node->addOperand(x);
    node->setIntAttr("axis", axis);
    return id;
}

// Comparisons
ValueId IRBuilder::eq(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kEq, lhs, rhs);
}

ValueId IRBuilder::lt(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kLt, lhs, rhs);
}

ValueId IRBuilder::le(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kLe, lhs, rhs);
}

ValueId IRBuilder::gt(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kGt, lhs, rhs);
}

ValueId IRBuilder::ge(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kGe, lhs, rhs);
}

ValueId IRBuilder::ne(ValueId lhs, ValueId rhs) {
    return createBinaryOp(OpCode::kNe, lhs, rhs);
}

// Select
ValueId IRBuilder::select(ValueId mask, ValueId true_val, ValueId false_val) {
    auto* true_node = getNode(true_val);
    if (!true_node)
        return ValueId::invalid();

    ValueId id = createNode(OpCode::kSelect, true_node->type());
    auto* node = getNode(id);
    node->addOperand(mask);
    node->addOperand(true_val);
    node->addOperand(false_val);
    return id;
}

// Memory operations
ValueId IRBuilder::load(ValueId ptr, TypeDesc element_type) {
    ValueId id = createNode(OpCode::kLoad, element_type);
    auto* node = getNode(id);
    node->addOperand(ptr);
    return id;
}

ValueId IRBuilder::store(ValueId ptr, ValueId value) {
    auto* value_node = getNode(value);
    if (!value_node)
        return ValueId::invalid();

    ValueId id = createNode(OpCode::kStore, TypeDesc());
    auto* node = getNode(id);
    node->addOperand(ptr);
    node->addOperand(value);
    return id;
}

// Type conversions
ValueId IRBuilder::cast(ValueId x, ScalarType target_type) {
    auto* x_node = getNode(x);
    if (!x_node)
        return ValueId::invalid();

    auto type_result = TypeInferrer::inferCast(x_node->type(), target_type);
    if (!type_result)
        return ValueId::invalid();

    ValueId id = createNode(OpCode::kCast, *type_result);
    auto* node = getNode(id);
    node->addOperand(x);
    return id;
}

// Node access
IRNode* IRBuilder::getNode(ValueId id) {
    if (id.id >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[id.id];
}

const IRNode* IRBuilder::getNode(ValueId id) const {
    if (id.id >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[id.id];
}

// =============================================================================
// IR Mutation Methods
// =============================================================================

size_t IRBuilder::liveNodeCount() const {
    size_t count = 0;
    for (const auto* node : nodes_) {
        if (node && !node->isDead()) {
            ++count;
        }
    }
    return count;
}

size_t IRBuilder::replaceAllUses(ValueId old_id, ValueId new_id) {
    if (!old_id.isValid() || !new_id.isValid()) {
        return 0;
    }

    size_t replacement_count = 0;

    // Iterate through all nodes and replace operands
    for (auto* node : nodes_) {
        if (!node || node->isDead()) {
            continue;
        }

        // Replace operands that match old_id
        for (size_t i = 0; i < node->numOperands(); ++i) {
            if (node->operand(i) == old_id) {
                node->setOperand(i, new_id);
                ++replacement_count;
            }
        }
    }

    return replacement_count;
}

void IRBuilder::markDead(ValueId id) {
    auto* node = getNode(id);
    if (node) {
        node->markDead();
    }
}

bool IRBuilder::markDeadIfUnused(ValueId id) {
    auto* node = getNode(id);
    if (!node || node->isDead()) {
        return false;
    }

    // Check if any live node uses this value
    if (!hasUses(id)) {
        node->markDead();

        // Recursively mark operands as dead if they become unused
        for (const auto& operand : node->operands()) {
            markDeadIfUnused(operand);
        }

        return true;
    }

    return false;
}

size_t IRBuilder::compactNodes() {
    // Build mapping from old IDs to new IDs
    std::vector<ValueId> id_map(nodes_.size(), ValueId::invalid());
    std::vector<IRNode*> new_nodes;
    new_nodes.reserve(nodes_.size());

    uint32_t new_id = 0;
    for (size_t old_idx = 0; old_idx < nodes_.size(); ++old_idx) {
        IRNode* node = nodes_[old_idx];
        if (node && !node->isDead()) {
            id_map[old_idx] = ValueId{new_id};
            node->setId(ValueId{new_id});
            new_nodes.push_back(node);
            ++new_id;
        }
    }

    size_t removed_count = nodes_.size() - new_nodes.size();

    // Update all operand references in surviving nodes
    for (auto* node : new_nodes) {
        for (size_t i = 0; i < node->numOperands(); ++i) {
            ValueId old_operand = node->operand(i);
            if (old_operand.id < id_map.size() && id_map[old_operand.id].isValid()) {
                node->setOperand(i, id_map[old_operand.id]);
            }
        }
    }

    // Replace old nodes vector
    nodes_ = std::move(new_nodes);
    next_id_ = new_id;

    return removed_count;
}

bool IRBuilder::hasUses(ValueId id) const {
    if (!id.isValid()) {
        return false;
    }

    for (const auto* node : nodes_) {
        if (!node || node->isDead()) {
            continue;
        }

        // Don't count self-references
        if (node->id() == id) {
            continue;
        }

        for (const auto& operand : node->operands()) {
            if (operand == id) {
                return true;
            }
        }
    }

    return false;
}

size_t IRBuilder::useCount(ValueId id) const {
    if (!id.isValid()) {
        return 0;
    }

    size_t count = 0;
    for (const auto* node : nodes_) {
        if (!node || node->isDead()) {
            continue;
        }

        // Don't count self-references
        if (node->id() == id) {
            continue;
        }

        for (const auto& operand : node->operands()) {
            if (operand == id) {
                ++count;
            }
        }
    }

    return count;
}

std::vector<ValueId> IRBuilder::findUses(ValueId id) const {
    std::vector<ValueId> uses;

    if (!id.isValid()) {
        return uses;
    }

    for (const auto* node : nodes_) {
        if (!node || node->isDead()) {
            continue;
        }

        // Don't count self-references
        if (node->id() == id) {
            continue;
        }

        for (const auto& operand : node->operands()) {
            if (operand == id) {
                uses.push_back(node->id());
                break;  // Only count each node once
            }
        }
    }

    return uses;
}

void IRBuilder::replaceNode(ValueId id, OpCode new_op, const std::vector<ValueId>& new_operands) {
    auto* node = getNode(id);
    if (!node) {
        return;
    }

    // Clear existing operands and add new ones
    node->clearOperands();
    for (const auto& operand : new_operands) {
        node->addOperand(operand);
    }

    // Note: OpCode and type cannot be changed after construction
    // This is a simplified replacement that only changes operands
    // For full replacement, would need to allocate a new node
}

ValueId IRBuilder::cloneNode(ValueId source_id, const std::vector<ValueId>& new_operands) {
    const auto* source = getNode(source_id);
    if (!source) {
        return ValueId::invalid();
    }

    // Create a new node with the same opcode and type
    ValueId new_id = createNode(source->opCode(), source->type());
    if (!new_id.isValid()) {
        return ValueId::invalid();
    }

    auto* new_node = getNode(new_id);
    if (!new_node) {
        return ValueId::invalid();
    }

    // Add operands
    for (const auto& operand : new_operands) {
        new_node->addOperand(operand);
    }

    // Copy attributes (simplified - would need full attribute copying)
    // For now, just copy the most common attributes
    if (source->hasAttr("value")) {
        if (source->opCode() == OpCode::kConstantScalar) {
            // Try to determine if it's int or float from the type
            if (source->type().scalarType() == ScalarType::kFloat32 ||
                source->type().scalarType() == ScalarType::kFloat64) {
                new_node->setFloatAttr("value", source->floatAttr("value"));
            } else {
                new_node->setIntAttr("value", source->intAttr("value"));
            }
        }
    }
    if (source->hasAttr("axis")) {
        new_node->setIntAttr("axis", source->intAttr("axis"));
    }

    return new_id;
}

// Validation
Result<void> IRBuilder::validate() const {
    // Check all operands reference valid nodes
    for (const auto* node : nodes_) {
        for (const auto& operand : node->operands()) {
            if (operand.id >= nodes_.size()) {
                return Error(ErrorCode::kInvalidIR,
                             fmt::format("Node %{} references invalid operand %{}", node->id().id,
                                         operand.id));
            }
            // Check operand is defined before use (SSA property)
            if (operand.id >= node->id().id) {
                return Error(
                    ErrorCode::kInvalidIR,
                    fmt::format("Node %{} uses %{} before definition", node->id().id, operand.id));
            }
        }
    }
    return {};
}

// Debug output
std::string IRBuilder::dump() const {
    std::string result = "IR Module:\n";
    for (const auto* node : nodes_) {
        result += "  " + node->toString() + "\n";
    }
    return result;
}

// =============================================================================
// IRModule Implementation
// =============================================================================

IRModule::IRModule(std::string name) : name_(std::move(name)), builder_(arena_) {}

Result<void> IRModule::optimize(int level) {
    if (level < 0 || level > 3) {
        return Error(ErrorCode::kInvalidInput, "Optimization level must be 0-3");
    }

    // Level 0: No optimization
    if (level == 0) {
        return {};
    }

    // Level 1+: Basic optimizations (constant folding, dead code elimination)
    // TODO: Add constant folding pass
    // TODO: Add dead code elimination pass

    // Level 2+: Standard optimizations (fusion)
    if (level >= 2) {
        // Use cost model-based fusion pass (producer-consumer fusion)
        FusionCostModel cost_model;
        PriorityFusionPass fusion_pass(cost_model);
        fusion_pass.run(builder_);

        // Use sibling fusion pass (multi-output fusion for shared inputs)
        SiblingFusionPass sibling_pass;
        sibling_pass.run(builder_);

        // Use horizontal fusion pass (batch independent operations)
        HorizontalFusionPass horizontal_pass;
        horizontal_pass.run(builder_);

        // Compact nodes to remove dead code from fusion
        builder_.compactNodes();
    }

    // Level 3: Aggressive optimizations
    if (level >= 3) {
        // TODO: Add loop unrolling hints
        // TODO: Add vectorization hints
    }

    return {};
}

std::string IRModule::toJson() const {
    // TODO: Implement JSON serialization
    return "{}";
}

Result<IRModule> IRModule::fromJson(std::string_view json) {
    // TODO: Implement JSON deserialization
    (void)json;
    return Error(ErrorCode::kNotImplemented, "JSON deserialization not implemented");
}

}  // namespace ir
}  // namespace bud
