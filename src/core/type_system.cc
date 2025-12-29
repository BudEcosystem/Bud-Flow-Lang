// =============================================================================
// Bud Flow Lang - Type System Implementation
// =============================================================================

#include "bud_flow_lang/type_system.h"

#include <fmt/format.h>

#include <algorithm>
#include <numeric>

namespace bud {

// =============================================================================
// Scalar Type Utilities
// =============================================================================

size_t scalarTypeSize(ScalarType type) {
    switch (type) {
    case ScalarType::kFloat16:
    case ScalarType::kBFloat16:
        return 2;
    case ScalarType::kFloat32:
        return 4;
    case ScalarType::kFloat64:
        return 8;
    case ScalarType::kInt8:
    case ScalarType::kUint8:
    case ScalarType::kBool:
        return 1;
    case ScalarType::kInt16:
    case ScalarType::kUint16:
        return 2;
    case ScalarType::kInt32:
    case ScalarType::kUint32:
        return 4;
    case ScalarType::kInt64:
    case ScalarType::kUint64:
        return 8;
    default:
        return 0;
    }
}

std::string_view scalarTypeName(ScalarType type) {
    switch (type) {
    case ScalarType::kFloat16:
        return "float16";
    case ScalarType::kFloat32:
        return "float32";
    case ScalarType::kFloat64:
        return "float64";
    case ScalarType::kBFloat16:
        return "bfloat16";
    case ScalarType::kInt8:
        return "int8";
    case ScalarType::kInt16:
        return "int16";
    case ScalarType::kInt32:
        return "int32";
    case ScalarType::kInt64:
        return "int64";
    case ScalarType::kUint8:
        return "uint8";
    case ScalarType::kUint16:
        return "uint16";
    case ScalarType::kUint32:
        return "uint32";
    case ScalarType::kUint64:
        return "uint64";
    case ScalarType::kBool:
        return "bool";
    default:
        return "unknown";
    }
}

bool isFloatingPoint(ScalarType type) {
    return type == ScalarType::kFloat16 || type == ScalarType::kFloat32 ||
           type == ScalarType::kFloat64 || type == ScalarType::kBFloat16;
}

bool isInteger(ScalarType type) {
    return type >= ScalarType::kInt8 && type <= ScalarType::kUint64;
}

bool isSigned(ScalarType type) {
    return isFloatingPoint(type) || (type >= ScalarType::kInt8 && type <= ScalarType::kInt64);
}

// =============================================================================
// Shape Implementation
// =============================================================================

Shape::Shape(std::initializer_list<size_t> dims) {
    BUD_ASSERT(dims.size() <= kMaxDims);
    rank_ = static_cast<uint8_t>(dims.size());
    std::copy(dims.begin(), dims.end(), dims_.begin());
}

Shape::Shape(const std::vector<size_t>& dims) {
    BUD_ASSERT(dims.size() <= kMaxDims);
    rank_ = static_cast<uint8_t>(dims.size());
    std::copy(dims.begin(), dims.end(), dims_.begin());
}

size_t Shape::totalElements() const {
    if (rank_ == 0) {
        return 1;  // Scalar
    }
    return std::accumulate(dims_.begin(), dims_.begin() + rank_, size_t{1},
                           std::multiplies<size_t>());
}

Result<Shape> Shape::broadcast(const Shape& a, const Shape& b) {
    // NumPy-style broadcasting rules
    size_t result_rank = std::max(a.rank(), b.rank());
    if (result_rank > kMaxDims) {
        return ErrorCode::kShapeMismatch;
    }

    Shape result;
    result.rank_ = static_cast<uint8_t>(result_rank);

    for (size_t i = 0; i < result_rank; ++i) {
        size_t a_idx = a.rank() > i ? a.rank() - 1 - i : SIZE_MAX;
        size_t b_idx = b.rank() > i ? b.rank() - 1 - i : SIZE_MAX;

        size_t a_dim = a_idx != SIZE_MAX ? a.dims_[a_idx] : 1;
        size_t b_dim = b_idx != SIZE_MAX ? b.dims_[b_idx] : 1;

        if (a_dim == b_dim) {
            result.dims_[result_rank - 1 - i] = a_dim;
        } else if (a_dim == 1) {
            result.dims_[result_rank - 1 - i] = b_dim;
        } else if (b_dim == 1) {
            result.dims_[result_rank - 1 - i] = a_dim;
        } else {
            return Error(ErrorCode::kShapeMismatch, fmt::format("Cannot broadcast shapes {} and {}",
                                                                a.toString(), b.toString()));
        }
    }

    return result;
}

bool Shape::isBroadcastableWith(const Shape& other) const {
    auto result = broadcast(*this, other);
    return result.hasValue();
}

bool Shape::operator==(const Shape& other) const {
    if (rank_ != other.rank_) {
        return false;
    }
    for (size_t i = 0; i < rank_; ++i) {
        if (dims_[i] != other.dims_[i]) {
            return false;
        }
    }
    return true;
}

std::string Shape::toString() const {
    if (rank_ == 0) {
        return "()";
    }

    std::string result = "(";
    for (size_t i = 0; i < rank_; ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(dims_[i]);
    }
    result += ")";
    return result;
}

// =============================================================================
// TypeDesc Implementation
// =============================================================================

TypeDesc::TypeDesc(ScalarType scalar, Shape shape)
    : scalar_type_(scalar), shape_(std::move(shape)) {}

size_t TypeDesc::byteSize() const {
    return scalarTypeSize(scalar_type_) * elementCount();
}

bool TypeDesc::operator==(const TypeDesc& other) const {
    return scalar_type_ == other.scalar_type_ && shape_ == other.shape_;
}

std::string TypeDesc::toString() const {
    return fmt::format("{}{}", scalarTypeName(scalar_type_), shape_.toString());
}

// =============================================================================
// Type Inference
// =============================================================================

Result<TypeDesc> TypeInferrer::inferBinaryOp(const TypeDesc& lhs, const TypeDesc& rhs) {
    // Types must match for now (no automatic promotion)
    if (lhs.scalarType() != rhs.scalarType()) {
        return Error(ErrorCode::kTypeMismatch, fmt::format("Binary op type mismatch: {} vs {}",
                                                           scalarTypeName(lhs.scalarType()),
                                                           scalarTypeName(rhs.scalarType())));
    }

    // Broadcast shapes
    auto shape_result = Shape::broadcast(lhs.shape(), rhs.shape());
    if (!shape_result) {
        return shape_result.error();
    }

    return TypeDesc(lhs.scalarType(), *shape_result);
}

Result<TypeDesc> TypeInferrer::inferUnaryOp(const TypeDesc& operand) {
    // Unary ops preserve type
    return operand;
}

Result<TypeDesc> TypeInferrer::inferReduction(const TypeDesc& operand, int axis) {
    if (axis == -1) {
        // Full reduction -> scalar
        return TypeDesc(operand.scalarType());
    }

    // Reduce along axis
    const auto& shape = operand.shape();
    if (axis < 0 || static_cast<size_t>(axis) >= shape.rank()) {
        return Error(ErrorCode::kInvalidOperand,
                     fmt::format("Invalid reduction axis {} for rank {}", axis, shape.rank()));
    }

    // Build new shape with axis removed
    std::vector<size_t> new_dims;
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (static_cast<int>(i) != axis) {
            new_dims.push_back(shape[i]);
        }
    }

    return TypeDesc(operand.scalarType(), Shape(new_dims));
}

Result<TypeDesc> TypeInferrer::inferCast(const TypeDesc& from, ScalarType to) {
    return TypeDesc(to, from.shape());
}

}  // namespace bud
