#pragma once

// =============================================================================
// Bud Flow Lang - Type System
// =============================================================================
//
// Tag-based type system matching Highway's design for zero-cost abstraction.
// Supports:
// - Scalar types: float32, float64, int8-64, uint8-64
// - Vector types: fixed-width and scalable
// - Shape inference for broadcasting
//

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace bud {

// =============================================================================
// Scalar Type Tags
// =============================================================================

enum class ScalarType : uint8_t {
    kUnknown = 0,

    // Floating point
    kFloat16 = 1,
    kFloat32 = 2,
    kFloat64 = 3,
    kBFloat16 = 4,

    // Signed integers
    kInt8 = 10,
    kInt16 = 11,
    kInt32 = 12,
    kInt64 = 13,

    // Unsigned integers
    kUint8 = 20,
    kUint16 = 21,
    kUint32 = 22,
    kUint64 = 23,

    // Boolean (stored as uint8)
    kBool = 30,
};

// Get size in bytes for a scalar type
size_t scalarTypeSize(ScalarType type);

// Get string representation
std::string_view scalarTypeName(ScalarType type);

// Check type properties
bool isFloatingPoint(ScalarType type);
bool isInteger(ScalarType type);
bool isSigned(ScalarType type);

// =============================================================================
// Shape Representation
// =============================================================================

// Maximum number of dimensions supported
constexpr size_t kMaxDims = 8;

class Shape {
public:
    Shape() = default;
    explicit Shape(std::initializer_list<size_t> dims);
    explicit Shape(const std::vector<size_t>& dims);

    // Accessors
    [[nodiscard]] size_t rank() const { return rank_; }
    [[nodiscard]] size_t operator[](size_t i) const { return dims_[i]; }
    [[nodiscard]] size_t totalElements() const;

    // Iterators
    [[nodiscard]] const size_t* begin() const { return dims_.data(); }
    [[nodiscard]] const size_t* end() const { return dims_.data() + rank_; }

    // Modification
    void setDim(size_t i, size_t value) { dims_[i] = value; }

    // Broadcasting
    [[nodiscard]] static Result<Shape> broadcast(const Shape& a, const Shape& b);
    [[nodiscard]] bool isBroadcastableWith(const Shape& other) const;

    // Comparison
    [[nodiscard]] bool operator==(const Shape& other) const;
    [[nodiscard]] bool operator!=(const Shape& other) const { return !(*this == other); }

    // String representation
    [[nodiscard]] std::string toString() const;

    // Common shapes
    static Shape scalar() { return Shape(); }
    static Shape vector(size_t n) { return Shape({n}); }
    static Shape matrix(size_t m, size_t n) { return Shape({m, n}); }

private:
    std::array<size_t, kMaxDims> dims_ = {};
    uint8_t rank_ = 0;
};

// =============================================================================
// Type Descriptor
// =============================================================================

class TypeDesc {
public:
    TypeDesc() = default;
    TypeDesc(ScalarType scalar, Shape shape = Shape::scalar());

    // Accessors
    [[nodiscard]] ScalarType scalarType() const { return scalar_type_; }
    [[nodiscard]] const Shape& shape() const { return shape_; }
    [[nodiscard]] size_t elementCount() const { return shape_.totalElements(); }
    [[nodiscard]] size_t byteSize() const;

    // Type checks
    [[nodiscard]] bool isScalar() const { return shape_.rank() == 0; }
    [[nodiscard]] bool isVector() const { return shape_.rank() == 1; }
    [[nodiscard]] bool isMatrix() const { return shape_.rank() == 2; }

    // Comparison
    [[nodiscard]] bool operator==(const TypeDesc& other) const;
    [[nodiscard]] bool operator!=(const TypeDesc& other) const {
        return !(*this == other);
    }

    // String representation
    [[nodiscard]] std::string toString() const;

    // Factory methods
    static TypeDesc f32() { return TypeDesc(ScalarType::kFloat32); }
    static TypeDesc f64() { return TypeDesc(ScalarType::kFloat64); }
    static TypeDesc i32() { return TypeDesc(ScalarType::kInt32); }
    static TypeDesc i64() { return TypeDesc(ScalarType::kInt64); }
    static TypeDesc f32Vector(size_t n) {
        return TypeDesc(ScalarType::kFloat32, Shape::vector(n));
    }

private:
    ScalarType scalar_type_ = ScalarType::kUnknown;
    Shape shape_;
};

// =============================================================================
// Type Inference
// =============================================================================

class TypeInferrer {
public:
    // Binary operation type inference
    static Result<TypeDesc> inferBinaryOp(const TypeDesc& lhs, const TypeDesc& rhs);

    // Unary operation type inference
    static Result<TypeDesc> inferUnaryOp(const TypeDesc& operand);

    // Reduction type inference (e.g., sum, max)
    static Result<TypeDesc> inferReduction(const TypeDesc& operand, int axis = -1);

    // Cast type inference
    static Result<TypeDesc> inferCast(const TypeDesc& from, ScalarType to);
};

}  // namespace bud
