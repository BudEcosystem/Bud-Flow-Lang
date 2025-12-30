// =============================================================================
// Bud Flow Lang - IR Node Implementation
// =============================================================================

#include "bud_flow_lang/ir.h"

#include <fmt/format.h>

namespace bud {
namespace ir {

// =============================================================================
// OpCode to String
// =============================================================================

std::string_view opCodeName(OpCode op) {
    switch (op) {
    case OpCode::kConstantScalar:
        return "constant.scalar";
    case OpCode::kConstantVector:
        return "constant.vector";
    case OpCode::kLoad:
        return "load";
    case OpCode::kStore:
        return "store";
    case OpCode::kAlloc:
        return "alloc";
    case OpCode::kAdd:
        return "add";
    case OpCode::kSub:
        return "sub";
    case OpCode::kMul:
        return "mul";
    case OpCode::kDiv:
        return "div";
    case OpCode::kMod:
        return "mod";
    case OpCode::kMin:
        return "min";
    case OpCode::kMax:
        return "max";
    case OpCode::kPow:
        return "pow";
    case OpCode::kFma:
        return "fma";
    case OpCode::kFnma:
        return "fnma";
    case OpCode::kNeg:
        return "neg";
    case OpCode::kAbs:
        return "abs";
    case OpCode::kSqrt:
        return "sqrt";
    case OpCode::kRsqrt:
        return "rsqrt";
    case OpCode::kRcp:
        return "rcp";
    case OpCode::kExp:
        return "exp";
    case OpCode::kLog:
        return "log";
    case OpCode::kSin:
        return "sin";
    case OpCode::kCos:
        return "cos";
    case OpCode::kTan:
        return "tan";
    case OpCode::kTanh:
        return "tanh";
    case OpCode::kSigmoid:
        return "sigmoid";
    case OpCode::kEq:
        return "eq";
    case OpCode::kNe:
        return "ne";
    case OpCode::kLt:
        return "lt";
    case OpCode::kLe:
        return "le";
    case OpCode::kGt:
        return "gt";
    case OpCode::kGe:
        return "ge";
    case OpCode::kAnd:
        return "and";
    case OpCode::kOr:
        return "or";
    case OpCode::kNot:
        return "not";
    case OpCode::kXor:
        return "xor";
    case OpCode::kReduceSum:
        return "reduce.sum";
    case OpCode::kReduceMax:
        return "reduce.max";
    case OpCode::kReduceMin:
        return "reduce.min";
    case OpCode::kReduceProd:
        return "reduce.prod";
    case OpCode::kBroadcast:
        return "broadcast";
    case OpCode::kGather:
        return "gather";
    case OpCode::kScatter:
        return "scatter";
    case OpCode::kCast:
        return "cast";
    case OpCode::kSelect:
        return "select";
    case OpCode::kPhi:
        return "phi";
    case OpCode::kReturn:
        return "return";
    case OpCode::kCall:
        return "call";
    default:
        return "unknown";
    }
}

// =============================================================================
// IRNode Implementation
// =============================================================================

void IRNode::setIntAttr(std::string_view name, int64_t value) {
    for (auto& attr : attrs_) {
        if (attr.name == name) {
            attr.value = value;
            return;
        }
    }
    attrs_.push_back({std::string(name), value});
}

void IRNode::setFloatAttr(std::string_view name, double value) {
    for (auto& attr : attrs_) {
        if (attr.name == name) {
            attr.value = value;
            return;
        }
    }
    attrs_.push_back({std::string(name), value});
}

void IRNode::setStringAttr(std::string_view name, std::string value) {
    for (auto& attr : attrs_) {
        if (attr.name == name) {
            attr.value = std::move(value);
            return;
        }
    }
    attrs_.push_back({std::string(name), std::move(value)});
}

int64_t IRNode::intAttr(std::string_view name, int64_t default_val) const {
    for (const auto& attr : attrs_) {
        if (attr.name == name && std::holds_alternative<int64_t>(attr.value)) {
            return std::get<int64_t>(attr.value);
        }
    }
    return default_val;
}

double IRNode::floatAttr(std::string_view name, double default_val) const {
    for (const auto& attr : attrs_) {
        if (attr.name == name && std::holds_alternative<double>(attr.value)) {
            return std::get<double>(attr.value);
        }
    }
    return default_val;
}

std::string_view IRNode::stringAttr(std::string_view name) const {
    for (const auto& attr : attrs_) {
        if (attr.name == name && std::holds_alternative<std::string>(attr.value)) {
            return std::get<std::string>(attr.value);
        }
    }
    return "";
}

bool IRNode::hasAttr(std::string_view name) const {
    for (const auto& attr : attrs_) {
        if (attr.name == name) {
            return true;
        }
    }
    return false;
}

void IRNode::replaceOperand(ValueId old_val, ValueId new_val) {
    for (auto& operand : operands_) {
        if (operand == old_val) {
            operand = new_val;
        }
    }
}

std::string IRNode::toString() const {
    std::string result = fmt::format("%{} = {} : {}", id_.id, opCodeName(op_), type_.toString());

    if (!operands_.empty()) {
        result += " (";
        for (size_t i = 0; i < operands_.size(); ++i) {
            if (i > 0)
                result += ", ";
            result += fmt::format("%{}", operands_[i].id);
        }
        result += ")";
    }

    return result;
}

}  // namespace ir
}  // namespace bud
