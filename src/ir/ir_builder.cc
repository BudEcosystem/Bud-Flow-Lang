// =============================================================================
// Bud Flow Lang - IR Builder Implementation
// =============================================================================

#include "bud_flow_lang/ir.h"

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
    auto* node = getNode(id);
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
    auto* node = getNode(id);
    node->addOperand(operand);
    return id;
}

// Constants
ValueId IRBuilder::constant(float value) {
    ValueId id = createNode(OpCode::kConstantScalar, TypeDesc::f32());
    auto* node = getNode(id);
    node->setFloatAttr("value", static_cast<double>(value));
    return id;
}

ValueId IRBuilder::constant(double value) {
    ValueId id = createNode(OpCode::kConstantScalar, TypeDesc::f64());
    auto* node = getNode(id);
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
    if (!x_node) return ValueId::invalid();

    auto type_result = TypeInferrer::inferReduction(x_node->type(), axis);
    if (!type_result) return ValueId::invalid();

    ValueId id = createNode(OpCode::kReduceSum, *type_result);
    auto* node = getNode(id);
    node->addOperand(x);
    node->setIntAttr("axis", axis);
    return id;
}

ValueId IRBuilder::reduceMax(ValueId x, int axis) {
    auto* x_node = getNode(x);
    if (!x_node) return ValueId::invalid();

    auto type_result = TypeInferrer::inferReduction(x_node->type(), axis);
    if (!type_result) return ValueId::invalid();

    ValueId id = createNode(OpCode::kReduceMax, *type_result);
    auto* node = getNode(id);
    node->addOperand(x);
    node->setIntAttr("axis", axis);
    return id;
}

ValueId IRBuilder::reduceMin(ValueId x, int axis) {
    auto* x_node = getNode(x);
    if (!x_node) return ValueId::invalid();

    auto type_result = TypeInferrer::inferReduction(x_node->type(), axis);
    if (!type_result) return ValueId::invalid();

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

// Select
ValueId IRBuilder::select(ValueId mask, ValueId true_val, ValueId false_val) {
    auto* true_node = getNode(true_val);
    if (!true_node) return ValueId::invalid();

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
    if (!value_node) return ValueId::invalid();

    ValueId id = createNode(OpCode::kStore, TypeDesc());
    auto* node = getNode(id);
    node->addOperand(ptr);
    node->addOperand(value);
    return id;
}

// Type conversions
ValueId IRBuilder::cast(ValueId x, ScalarType target_type) {
    auto* x_node = getNode(x);
    if (!x_node) return ValueId::invalid();

    auto type_result = TypeInferrer::inferCast(x_node->type(), target_type);
    if (!type_result) return ValueId::invalid();

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

// Validation
Result<void> IRBuilder::validate() const {
    // Check all operands reference valid nodes
    for (const auto* node : nodes_) {
        for (const auto& operand : node->operands()) {
            if (operand.id >= nodes_.size()) {
                return Error(ErrorCode::kInvalidIR,
                             fmt::format("Node %{} references invalid operand %{}",
                                         node->id().id, operand.id));
            }
            // Check operand is defined before use (SSA property)
            if (operand.id >= node->id().id) {
                return Error(ErrorCode::kInvalidIR,
                             fmt::format("Node %{} uses %{} before definition",
                                         node->id().id, operand.id));
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

IRModule::IRModule(std::string name)
    : name_(std::move(name))
    , builder_(arena_) {}

Result<void> IRModule::optimize(int level) {
    if (level < 0 || level > 3) {
        return Error(ErrorCode::kInvalidInput, "Optimization level must be 0-3");
    }

    // TODO: Implement optimization passes
    // Level 0: No optimization
    // Level 1: Basic optimizations (constant folding, dead code elimination)
    // Level 2: Standard optimizations (fusion, strength reduction)
    // Level 3: Aggressive optimizations (loop unrolling, vectorization hints)

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
