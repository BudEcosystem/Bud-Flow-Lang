// =============================================================================
// Bud Flow Lang - IR Interpreter (Tier 0 Execution Engine)
// =============================================================================
//
// This module implements a simple IR interpreter that walks the IR graph
// and dispatches operations to Highway SIMD functions.
//
// Features:
// - Topological order traversal for correct dependencies
// - Temporary storage management for intermediate values
// - Direct Highway dispatch (no JIT compilation overhead)
// - Support for all IR operations
//
// This is the foundation for lazy evaluation - IR graphs built from
// Bunch operations can be executed here with fusion optimization.
//

#include "bud_flow_lang/codegen/hwy_ops.h"
#include "bud_flow_lang/ir.h"

#include <hwy/aligned_allocator.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bud {

// Forward declare Bunch (defined in bunch.h, but we avoid circular include)
class Bunch;

namespace ir {

// =============================================================================
// Value Storage - Manages temporary buffers for IR values
// =============================================================================

class ValueStorage {
  public:
    ValueStorage() = default;
    ~ValueStorage() {
        for (auto& [id, buffer] : buffers_) {
            if (buffer.data && buffer.owned) {
                hwy::FreeAlignedBytes(buffer.data, nullptr, nullptr);
            }
        }
    }

    // Non-copyable, non-movable (owns raw memory)
    ValueStorage(const ValueStorage&) = delete;
    ValueStorage& operator=(const ValueStorage&) = delete;

    struct Buffer {
        void* data = nullptr;
        size_t size = 0;   // Element count
        size_t bytes = 0;  // Byte size
        ScalarType dtype = ScalarType::kUnknown;
        bool owned = false;  // If true, we own the memory
    };

    // Allocate storage for a value
    Result<void*> allocate(ValueId id, size_t count, ScalarType dtype) {
        size_t element_size = scalarTypeSize(dtype);
        size_t bytes = count * element_size;

        void* data = hwy::AllocateAlignedBytes(bytes, nullptr, nullptr);
        if (!data) {
            return ErrorCode::kAllocationFailed;
        }

        buffers_[id.id] = {data, count, bytes, dtype, true};
        return data;
    }

    // Register external storage (not owned by us)
    void registerExternal(ValueId id, void* data, size_t count, ScalarType dtype) {
        size_t bytes = count * scalarTypeSize(dtype);
        buffers_[id.id] = {data, count, bytes, dtype, false};
    }

    // Get buffer for a value
    Buffer* get(ValueId id) {
        auto it = buffers_.find(id.id);
        return it != buffers_.end() ? &it->second : nullptr;
    }

    const Buffer* get(ValueId id) const {
        auto it = buffers_.find(id.id);
        return it != buffers_.end() ? &it->second : nullptr;
    }

    bool has(ValueId id) const { return buffers_.count(id.id) > 0; }

  private:
    std::unordered_map<uint32_t, Buffer> buffers_;
};

// =============================================================================
// Topological Sort Helper
// =============================================================================

namespace {

// Build topological order for execution
std::vector<ValueId> topologicalSort(const IRBuilder& builder, ValueId output) {
    std::vector<ValueId> order;
    std::unordered_set<uint32_t> visited;
    std::unordered_set<uint32_t> in_stack;  // For cycle detection

    std::function<bool(ValueId)> visit = [&](ValueId id) -> bool {
        if (!id.isValid())
            return true;
        if (visited.count(id.id))
            return true;

        // Cycle detection
        if (in_stack.count(id.id)) {
            spdlog::error("Cycle detected in IR at node %{}", id.id);
            return false;
        }

        in_stack.insert(id.id);

        const IRNode* node = builder.getNode(id);
        if (!node) {
            in_stack.erase(id.id);
            return true;  // External input, not in graph
        }

        if (node->isDead()) {
            in_stack.erase(id.id);
            visited.insert(id.id);
            return true;
        }

        // Visit operands first
        for (const auto& operand : node->operands()) {
            if (!visit(operand)) {
                return false;
            }
        }

        in_stack.erase(id.id);
        visited.insert(id.id);
        order.push_back(id);
        return true;
    };

    if (!visit(output)) {
        order.clear();  // Return empty on cycle
    }

    return order;
}

// Get the size of an operation result
size_t getResultSize(const IRNode* node, const ValueStorage& storage) {
    // For reductions, result is scalar
    switch (node->opCode()) {
    case OpCode::kReduceSum:
    case OpCode::kReduceMax:
    case OpCode::kReduceMin:
    case OpCode::kReduceProd:
    case OpCode::kHorizontalSum:
    case OpCode::kHorizontalMax:
    case OpCode::kHorizontalMin:
        return 1;
    default:
        break;
    }

    // For other ops, result size matches first operand
    if (node->numOperands() > 0) {
        auto* operand_buf = storage.get(node->operand(0));
        if (operand_buf) {
            return operand_buf->size;
        }
    }

    // Fallback: check type shape
    const auto& shape = node->type().shape();
    if (shape.rank() > 0) {
        size_t total = 1;
        for (size_t i = 0; i < shape.rank(); ++i) {
            total *= shape[i];
        }
        return total;
    }

    return 1;  // Scalar
}

}  // namespace

// =============================================================================
// Operation Execution Helpers
// =============================================================================

namespace {

// Type-specific binary operation execution for integers (no Div support)
template <typename T>
Result<void> executeBinaryOpTyped(OpCode op, T* out, const T* a, const T* b, size_t count) {
    switch (op) {
    case OpCode::kAdd:
        simd::Add(out, a, b, count);
        break;
    case OpCode::kSub:
        simd::Sub(out, a, b, count);
        break;
    case OpCode::kMul:
        simd::Mul(out, a, b, count);
        break;
    case OpCode::kDiv:
        // Integer division - scalar fallback
        for (size_t i = 0; i < count; ++i) {
            out[i] = a[i] / b[i];
        }
        break;
    case OpCode::kMod:
        // Integer modulo
        for (size_t i = 0; i < count; ++i) {
            out[i] = a[i] % b[i];
        }
        break;
    case OpCode::kMin:
        simd::Min(out, a, b, count);
        break;
    case OpCode::kMax:
        simd::Max(out, a, b, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported binary operation for this type: " + std::string(opCodeName(op)));
    }
    return {};
}

// Specialization for float32 (supports more ops)
template <>
Result<void> executeBinaryOpTyped<float>(OpCode op, float* out, const float* a, const float* b,
                                         size_t count) {
    switch (op) {
    case OpCode::kAdd:
        simd::Add(out, a, b, count);
        break;
    case OpCode::kSub:
        simd::Sub(out, a, b, count);
        break;
    case OpCode::kMul:
        simd::Mul(out, a, b, count);
        break;
    case OpCode::kDiv:
        simd::Div(out, a, b, count);
        break;
    case OpCode::kMin:
        simd::Min(out, a, b, count);
        break;
    case OpCode::kMax:
        simd::Max(out, a, b, count);
        break;
    case OpCode::kPow:
        simd::Pow(out, a, b, count);
        break;
    case OpCode::kMod:
        for (size_t i = 0; i < count; ++i) {
            out[i] = std::fmod(a[i], b[i]);
        }
        break;
    case OpCode::kHypot:
        simd::Hypot(out, a, b, count);
        break;
    case OpCode::kAtan2:
        simd::Atan2(out, a, b, count);
        break;
    case OpCode::kCopySign:
        simd::CopySign(out, a, b, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported binary operation: " + std::string(opCodeName(op)));
    }
    return {};
}

// Specialization for float64 (supports more ops)
template <>
Result<void> executeBinaryOpTyped<double>(OpCode op, double* out, const double* a, const double* b,
                                          size_t count) {
    switch (op) {
    case OpCode::kAdd:
        simd::Add(out, a, b, count);
        break;
    case OpCode::kSub:
        simd::Sub(out, a, b, count);
        break;
    case OpCode::kMul:
        simd::Mul(out, a, b, count);
        break;
    case OpCode::kDiv:
        simd::Div(out, a, b, count);
        break;
    case OpCode::kMin:
        simd::Min(out, a, b, count);
        break;
    case OpCode::kMax:
        simd::Max(out, a, b, count);
        break;
    case OpCode::kPow:
        simd::Pow(out, a, b, count);
        break;
    case OpCode::kMod:
        for (size_t i = 0; i < count; ++i) {
            out[i] = std::fmod(a[i], b[i]);
        }
        break;
    case OpCode::kHypot:
        simd::Hypot(out, a, b, count);
        break;
    case OpCode::kAtan2:
        simd::Atan2(out, a, b, count);
        break;
    case OpCode::kCopySign:
        simd::CopySign(out, a, b, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported binary operation for float64: " + std::string(opCodeName(op)));
    }
    return {};
}

// Execute a binary operation - dispatches by type
Result<void> executeBinaryOp(OpCode op, void* out, const void* a, const void* b, size_t count,
                             ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        return executeBinaryOpTyped<float>(op, static_cast<float*>(out),
                                           static_cast<const float*>(a),
                                           static_cast<const float*>(b), count);
    case ScalarType::kFloat64:
        return executeBinaryOpTyped<double>(op, static_cast<double*>(out),
                                            static_cast<const double*>(a),
                                            static_cast<const double*>(b), count);
    case ScalarType::kInt32:
        return executeBinaryOpTyped<int32_t>(op, static_cast<int32_t*>(out),
                                             static_cast<const int32_t*>(a),
                                             static_cast<const int32_t*>(b), count);
    case ScalarType::kInt64:
        return executeBinaryOpTyped<int64_t>(op, static_cast<int64_t*>(out),
                                             static_cast<const int64_t*>(a),
                                             static_cast<const int64_t*>(b), count);
    default:
        return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for binary operation");
    }
}

// Type-specific unary operation execution for integers
template <typename T>
Result<void> executeUnaryOpTyped(OpCode op, T* out, const T* in, size_t count) {
    switch (op) {
    case OpCode::kNeg:
        simd::Neg(out, in, count);
        break;
    case OpCode::kAbs:
        simd::Abs(out, in, count);
        break;
    case OpCode::kNot:
        // Logical NOT for integers: 0 -> 1, non-zero -> 0
        for (size_t i = 0; i < count; ++i) {
            out[i] = (in[i] == 0) ? 1 : 0;
        }
        break;
    default:
        return Error(ErrorCode::kNotSupported, "Unsupported unary operation for integer type: " +
                                                   std::string(opCodeName(op)));
    }
    return {};
}

// Specialization for float32 (full transcendental support)
template <>
Result<void> executeUnaryOpTyped<float>(OpCode op, float* out, const float* in, size_t count) {
    switch (op) {
    case OpCode::kNeg:
        simd::Neg(out, in, count);
        break;
    case OpCode::kAbs:
        simd::Abs(out, in, count);
        break;
    case OpCode::kSqrt:
        simd::Sqrt(out, in, count);
        break;
    case OpCode::kRsqrt:
        simd::Rsqrt(out, in, count);
        break;
    case OpCode::kExp:
        simd::Exp(out, in, count);
        break;
    case OpCode::kLog:
        simd::Log(out, in, count);
        break;
    case OpCode::kSin:
        simd::Sin(out, in, count);
        break;
    case OpCode::kCos:
        simd::Cos(out, in, count);
        break;
    case OpCode::kTanh:
        simd::Tanh(out, in, count);
        break;
    case OpCode::kTan:
        simd::Tan(out, in, count);
        break;
    case OpCode::kCeil:
        simd::Ceil(out, in, count);
        break;
    case OpCode::kFloor:
        simd::Floor(out, in, count);
        break;
    case OpCode::kRound:
        simd::Round(out, in, count);
        break;
    case OpCode::kTrunc:
        simd::Trunc(out, in, count);
        break;
    case OpCode::kSigmoid:
        for (size_t i = 0; i < count; ++i) {
            out[i] = 1.0f / (1.0f + std::exp(-in[i]));
        }
        break;
    case OpCode::kRcp:
        for (size_t i = 0; i < count; ++i) {
            out[i] = 1.0f / in[i];
        }
        break;
    case OpCode::kNot:
        for (size_t i = 0; i < count; ++i) {
            out[i] = (in[i] == 0.0f) ? 1.0f : 0.0f;
        }
        break;
    case OpCode::kReverse:
        simd::Reverse(out, in, count);
        break;
    case OpCode::kZeroIfNeg:
        simd::ZeroIfNegative(out, in, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported unary operation: " + std::string(opCodeName(op)));
    }
    return {};
}

// Specialization for float64 (transcendental support)
template <>
Result<void> executeUnaryOpTyped<double>(OpCode op, double* out, const double* in, size_t count) {
    switch (op) {
    case OpCode::kNeg:
        simd::Neg(out, in, count);
        break;
    case OpCode::kAbs:
        simd::Abs(out, in, count);
        break;
    case OpCode::kSqrt:
        simd::Sqrt(out, in, count);
        break;
    case OpCode::kRsqrt:
        simd::Rsqrt(out, in, count);
        break;
    case OpCode::kExp:
        simd::Exp(out, in, count);
        break;
    case OpCode::kLog:
        simd::Log(out, in, count);
        break;
    case OpCode::kSin:
        simd::Sin(out, in, count);
        break;
    case OpCode::kCos:
        simd::Cos(out, in, count);
        break;
    case OpCode::kTanh:
        simd::Tanh(out, in, count);
        break;
    case OpCode::kTan:
        simd::Tan(out, in, count);
        break;
    case OpCode::kCeil:
        simd::Ceil(out, in, count);
        break;
    case OpCode::kFloor:
        simd::Floor(out, in, count);
        break;
    case OpCode::kRound:
        simd::Round(out, in, count);
        break;
    case OpCode::kTrunc:
        simd::Trunc(out, in, count);
        break;
    case OpCode::kSigmoid:
        for (size_t i = 0; i < count; ++i) {
            out[i] = 1.0 / (1.0 + std::exp(-in[i]));
        }
        break;
    case OpCode::kRcp:
        for (size_t i = 0; i < count; ++i) {
            out[i] = 1.0 / in[i];
        }
        break;
    case OpCode::kNot:
        for (size_t i = 0; i < count; ++i) {
            out[i] = (in[i] == 0.0) ? 1.0 : 0.0;
        }
        break;
    case OpCode::kReverse:
        simd::Reverse(out, in, count);
        break;
    case OpCode::kZeroIfNeg:
        simd::ZeroIfNegative(out, in, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported unary operation for float64: " + std::string(opCodeName(op)));
    }
    return {};
}

// Execute a unary operation - dispatches by type
Result<void> executeUnaryOp(OpCode op, void* out, const void* in, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        return executeUnaryOpTyped<float>(op, static_cast<float*>(out),
                                          static_cast<const float*>(in), count);
    case ScalarType::kFloat64:
        return executeUnaryOpTyped<double>(op, static_cast<double*>(out),
                                           static_cast<const double*>(in), count);
    case ScalarType::kInt32:
        return executeUnaryOpTyped<int32_t>(op, static_cast<int32_t*>(out),
                                            static_cast<const int32_t*>(in), count);
    case ScalarType::kInt64:
        return executeUnaryOpTyped<int64_t>(op, static_cast<int64_t*>(out),
                                            static_cast<const int64_t*>(in), count);
    default:
        return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for unary operation");
    }
}

// Type-specific reduction operation
template <typename T>
Result<void> executeReductionOpTyped(OpCode op, T* out, const T* in, size_t count) {
    if (count == 0) {
        return Error(ErrorCode::kInvalidInput, "Empty reduction input");
    }

    switch (op) {
    case OpCode::kReduceSum:
    case OpCode::kHorizontalSum: {
        T sum = T{0};
        for (size_t i = 0; i < count; ++i) {
            sum += in[i];
        }
        *out = sum;
        break;
    }
    case OpCode::kReduceMax:
    case OpCode::kHorizontalMax: {
        T max_val = in[0];
        for (size_t i = 1; i < count; ++i) {
            if (in[i] > max_val)
                max_val = in[i];
        }
        *out = max_val;
        break;
    }
    case OpCode::kReduceMin:
    case OpCode::kHorizontalMin: {
        T min_val = in[0];
        for (size_t i = 1; i < count; ++i) {
            if (in[i] < min_val)
                min_val = in[i];
        }
        *out = min_val;
        break;
    }
    case OpCode::kReduceProd: {
        T prod = T{1};
        for (size_t i = 0; i < count; ++i) {
            prod *= in[i];
        }
        *out = prod;
        break;
    }
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported reduction: " + std::string(opCodeName(op)));
    }
    return {};
}

// Execute a reduction operation - dispatches by type
Result<void> executeReductionOp(OpCode op, void* out, const void* in, size_t count,
                                ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        return executeReductionOpTyped<float>(op, static_cast<float*>(out),
                                              static_cast<const float*>(in), count);
    case ScalarType::kFloat64:
        return executeReductionOpTyped<double>(op, static_cast<double*>(out),
                                               static_cast<const double*>(in), count);
    case ScalarType::kInt32:
        return executeReductionOpTyped<int32_t>(op, static_cast<int32_t*>(out),
                                                static_cast<const int32_t*>(in), count);
    case ScalarType::kInt64:
        return executeReductionOpTyped<int64_t>(op, static_cast<int64_t*>(out),
                                                static_cast<const int64_t*>(in), count);
    default:
        return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for reduction");
    }
}

// Execute FMA operation - dispatches by type
Result<void> executeFmaOp(void* out, const void* a, const void* b, const void* c, size_t count,
                          ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        simd::MulAdd(static_cast<float*>(out), static_cast<const float*>(a),
                     static_cast<const float*>(b), static_cast<const float*>(c), count);
        return {};
    case ScalarType::kFloat64:
        simd::MulAdd(static_cast<double*>(out), static_cast<const double*>(a),
                     static_cast<const double*>(b), static_cast<const double*>(c), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "FMA only supports float32/float64");
    }
}

// Execute a comparison operation (returns float mask: 0.0 = false, 1.0 = true)
// Comparisons always output to the same type as input for mask operations
Result<void> executeComparisonOp(OpCode op, void* out, const void* a, const void* b, size_t count,
                                 ScalarType dtype) {
    // Use a temp uint8_t buffer for SIMD comparison
    auto temp_mask = std::make_unique<uint8_t[]>(count);

    switch (dtype) {
    case ScalarType::kFloat32: {
        const float* a_f = static_cast<const float*>(a);
        const float* b_f = static_cast<const float*>(b);
        float* out_f = static_cast<float*>(out);

        switch (op) {
        case OpCode::kEq:
            simd::Eq(temp_mask.get(), a_f, b_f, count);
            break;
        case OpCode::kNe:
            simd::Ne(temp_mask.get(), a_f, b_f, count);
            break;
        case OpCode::kLt:
            simd::Lt(temp_mask.get(), a_f, b_f, count);
            break;
        case OpCode::kLe:
            simd::Le(temp_mask.get(), a_f, b_f, count);
            break;
        case OpCode::kGt:
            simd::Gt(temp_mask.get(), a_f, b_f, count);
            break;
        case OpCode::kGe:
            simd::Ge(temp_mask.get(), a_f, b_f, count);
            break;
        default:
            return Error(ErrorCode::kNotSupported,
                         "Unsupported comparison operation: " + std::string(opCodeName(op)));
        }
        // Convert to float mask
        for (size_t i = 0; i < count; ++i) {
            out_f[i] = (temp_mask[i] != 0) ? 1.0f : 0.0f;
        }
        return {};
    }
    case ScalarType::kFloat64: {
        const double* a_d = static_cast<const double*>(a);
        const double* b_d = static_cast<const double*>(b);
        double* out_d = static_cast<double*>(out);

        switch (op) {
        case OpCode::kEq:
            simd::Eq(temp_mask.get(), a_d, b_d, count);
            break;
        case OpCode::kNe:
            simd::Ne(temp_mask.get(), a_d, b_d, count);
            break;
        case OpCode::kLt:
            simd::Lt(temp_mask.get(), a_d, b_d, count);
            break;
        case OpCode::kLe:
            simd::Le(temp_mask.get(), a_d, b_d, count);
            break;
        case OpCode::kGt:
            simd::Gt(temp_mask.get(), a_d, b_d, count);
            break;
        case OpCode::kGe:
            simd::Ge(temp_mask.get(), a_d, b_d, count);
            break;
        default:
            return Error(ErrorCode::kNotSupported,
                         "Unsupported comparison operation: " + std::string(opCodeName(op)));
        }
        // Convert to double mask
        for (size_t i = 0; i < count; ++i) {
            out_d[i] = (temp_mask[i] != 0) ? 1.0 : 0.0;
        }
        return {};
    }
    case ScalarType::kInt32: {
        const int32_t* a_i = static_cast<const int32_t*>(a);
        const int32_t* b_i = static_cast<const int32_t*>(b);
        int32_t* out_i = static_cast<int32_t*>(out);

        switch (op) {
        case OpCode::kEq:
            simd::Eq(temp_mask.get(), a_i, b_i, count);
            break;
        case OpCode::kNe:
            simd::Ne(temp_mask.get(), a_i, b_i, count);
            break;
        case OpCode::kLt:
            simd::Lt(temp_mask.get(), a_i, b_i, count);
            break;
        case OpCode::kLe:
            simd::Le(temp_mask.get(), a_i, b_i, count);
            break;
        case OpCode::kGt:
            simd::Gt(temp_mask.get(), a_i, b_i, count);
            break;
        case OpCode::kGe:
            simd::Ge(temp_mask.get(), a_i, b_i, count);
            break;
        default:
            return Error(ErrorCode::kNotSupported,
                         "Unsupported comparison operation: " + std::string(opCodeName(op)));
        }
        // Convert to int32 mask (0 or 1)
        for (size_t i = 0; i < count; ++i) {
            out_i[i] = (temp_mask[i] != 0) ? 1 : 0;
        }
        return {};
    }
    default:
        return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for comparison");
    }
}

// Type-specific logical operation
template <typename T>
Result<void> executeLogicalOpTyped(OpCode op, T* out, const T* a, const T* b, size_t count) {
    const T zero = T{0};
    const T one = T{1};

    switch (op) {
    case OpCode::kAnd:
        for (size_t i = 0; i < count; ++i) {
            out[i] = ((a[i] != zero) && (b[i] != zero)) ? one : zero;
        }
        break;
    case OpCode::kOr:
        for (size_t i = 0; i < count; ++i) {
            out[i] = ((a[i] != zero) || (b[i] != zero)) ? one : zero;
        }
        break;
    case OpCode::kXor:
        for (size_t i = 0; i < count; ++i) {
            bool a_bool = (a[i] != zero);
            bool b_bool = (b[i] != zero);
            out[i] = (a_bool != b_bool) ? one : zero;
        }
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported logical operation: " + std::string(opCodeName(op)));
    }
    return {};
}

// Execute a binary logical operation - dispatches by type
Result<void> executeLogicalOp(OpCode op, void* out, const void* a, const void* b, size_t count,
                              ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        return executeLogicalOpTyped<float>(op, static_cast<float*>(out),
                                            static_cast<const float*>(a),
                                            static_cast<const float*>(b), count);
    case ScalarType::kFloat64:
        return executeLogicalOpTyped<double>(op, static_cast<double*>(out),
                                             static_cast<const double*>(a),
                                             static_cast<const double*>(b), count);
    case ScalarType::kInt32:
        return executeLogicalOpTyped<int32_t>(op, static_cast<int32_t*>(out),
                                              static_cast<const int32_t*>(a),
                                              static_cast<const int32_t*>(b), count);
    case ScalarType::kInt64:
        return executeLogicalOpTyped<int64_t>(op, static_cast<int64_t*>(out),
                                              static_cast<const int64_t*>(a),
                                              static_cast<const int64_t*>(b), count);
    default:
        return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for logical operation");
    }
}

// Check if operation is a binary op
bool isBinaryOp(OpCode op) {
    switch (op) {
    case OpCode::kAdd:
    case OpCode::kSub:
    case OpCode::kMul:
    case OpCode::kDiv:
    case OpCode::kMin:
    case OpCode::kMax:
    case OpCode::kPow:
    case OpCode::kMod:
    case OpCode::kHypot:     // sqrt(x² + y²)
    case OpCode::kAtan2:     // atan2(y, x)
    case OpCode::kCopySign:  // Copy sign from y to x
        return true;
    default:
        return false;
    }
}

// Check if operation is a unary op
bool isUnaryOp(OpCode op) {
    switch (op) {
    case OpCode::kNeg:
    case OpCode::kAbs:
    case OpCode::kSqrt:
    case OpCode::kRsqrt:
    case OpCode::kRcp:
    case OpCode::kExp:
    case OpCode::kLog:
    case OpCode::kSin:
    case OpCode::kCos:
    case OpCode::kTan:
    case OpCode::kTanh:
    case OpCode::kSigmoid:
    case OpCode::kCeil:
    case OpCode::kFloor:
    case OpCode::kRound:
    case OpCode::kTrunc:
    case OpCode::kNot:        // Logical NOT is unary
    case OpCode::kReverse:    // Reverse lanes
    case OpCode::kZeroIfNeg:  // Zero out negative elements
        return true;
    default:
        return false;
    }
}

// Check if operation is a comparison op
bool isComparisonOp(OpCode op) {
    switch (op) {
    case OpCode::kEq:
    case OpCode::kNe:
    case OpCode::kLt:
    case OpCode::kLe:
    case OpCode::kGt:
    case OpCode::kGe:
        return true;
    default:
        return false;
    }
}

// Check if operation is a binary logical op
bool isLogicalOp(OpCode op) {
    switch (op) {
    case OpCode::kAnd:
    case OpCode::kOr:
    case OpCode::kXor:
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

}  // namespace

// =============================================================================
// IR Interpreter Class
// =============================================================================

class IRInterpreter {
  public:
    IRInterpreter() = default;

    // Parameter input storage (for JAX-style parameter binding)
    struct ParamInput {
        void* data = nullptr;
        size_t count = 0;
        ScalarType dtype = ScalarType::kUnknown;
    };

    // Register input by parameter index (for kParameter nodes)
    void registerInputByIndex(size_t idx, void* data, size_t count, ScalarType dtype) {
        param_inputs_[idx] = {data, count, dtype};
    }

    // Execute an IR module and return the result
    Result<void> execute(const IRBuilder& builder, ValueId output, void* result_data,
                         size_t result_size, ScalarType result_dtype) {
        if (!output.isValid()) {
            spdlog::error("IR Interpreter: Invalid output ValueId");
            return Error(ErrorCode::kInvalidInput, "Invalid output ValueId");
        }

        spdlog::debug("IR Interpreter: executing with output {} (total nodes: {})", output.id,
                      builder.nodes().size());

        // Get topological order
        auto order = topologicalSort(builder, output);
        if (order.empty()) {
            spdlog::error(
                "IR Interpreter: topologicalSort returned empty for output {} (valid: {})",
                output.id, output.isValid());
            // Debug: dump IR nodes
            for (const auto* node : builder.nodes()) {
                if (node) {
                    spdlog::error("  Node {}: {} (dead: {})", node->id().id,
                                  opCodeName(node->opCode()), node->isDead());
                }
            }
            return Error(ErrorCode::kInvalidIR, "Empty or cyclic IR graph");
        }

        spdlog::debug("IR Interpreter: Executing {} nodes", order.size());

        // Execute nodes in order
        for (const auto& id : order) {
            const IRNode* node = builder.getNode(id);
            if (!node || node->isDead())
                continue;

            auto result = executeNode(builder, node);
            if (!result) {
                return result;
            }
        }

        // Copy result to output buffer
        auto* output_buf = storage_.get(output);
        if (!output_buf) {
            return Error(ErrorCode::kRuntimeError, "Output value not computed");
        }

        size_t copy_size = std::min(output_buf->bytes, result_size * scalarTypeSize(result_dtype));
        std::memcpy(result_data, output_buf->data, copy_size);

        spdlog::debug("IR Interpreter: Execution complete");
        return {};
    }

    // Register an external input (data not owned by interpreter)
    void registerInput(ValueId id, void* data, size_t count, ScalarType dtype) {
        storage_.registerExternal(id, data, count, dtype);
    }

  private:
    ValueStorage storage_;
    std::unordered_map<size_t, ParamInput> param_inputs_;  // Indexed by param_index

    // Execute a single node
    Result<void> executeNode(const IRBuilder& builder, const IRNode* node) {
        OpCode op = node->opCode();
        ValueId id = node->id();
        ScalarType dtype = node->type().scalarType();

        spdlog::trace("Executing node %{}: {}", id.id, opCodeName(op));

        // Handle constants
        if (op == OpCode::kConstantScalar) {
            auto alloc_result = storage_.allocate(id, 1, dtype);
            if (!alloc_result)
                return alloc_result.error();

            void* data = *alloc_result;
            if (dtype == ScalarType::kFloat32 || dtype == ScalarType::kFloat64) {
                double value = node->floatAttr("value");
                if (dtype == ScalarType::kFloat32) {
                    *static_cast<float*>(data) = static_cast<float>(value);
                } else {
                    *static_cast<double*>(data) = value;
                }
            } else {
                int64_t value = node->intAttr("value");
                if (dtype == ScalarType::kInt32) {
                    *static_cast<int32_t*>(data) = static_cast<int32_t>(value);
                } else if (dtype == ScalarType::kInt64) {
                    *static_cast<int64_t*>(data) = value;
                }
            }
            return {};
        }

        // Handle parameter nodes (JAX-style input binding)
        if (op == OpCode::kParameter) {
            size_t param_index = static_cast<size_t>(node->intAttr("param_index"));
            auto it = param_inputs_.find(param_index);
            if (it == param_inputs_.end()) {
                return Error(ErrorCode::kRuntimeError,
                             fmt::format("Parameter {} not provided", param_index));
            }
            storage_.registerExternal(id, it->second.data, it->second.count, it->second.dtype);
            return {};
        }

        // Handle constant vectors
        if (op == OpCode::kConstantVector) {
            size_t count = static_cast<size_t>(node->intAttr("count", 0));
            if (count == 0) {
                const auto& shape = node->type().shape();
                count = 1;
                for (size_t i = 0; i < shape.rank(); ++i) {
                    count *= shape[i];
                }
            }

            auto alloc_result = storage_.allocate(id, count, dtype);
            if (!alloc_result)
                return alloc_result.error();
            void* data = *alloc_result;

            // Parse stored values
            std::string_view values_str = node->stringAttr("values");
            if (!values_str.empty() && dtype == ScalarType::kFloat32) {
                float* out = static_cast<float*>(data);
                size_t idx = 0;
                std::string token;
                for (char c : values_str) {
                    if (c == ',') {
                        if (idx < count && !token.empty()) {
                            out[idx++] = std::stof(token);
                        }
                        token.clear();
                    } else {
                        token += c;
                    }
                }
                // Handle last token
                if (!token.empty() && idx < count) {
                    out[idx] = std::stof(token);
                }
            } else {
                std::memset(data, 0, count * scalarTypeSize(dtype));
            }
            return {};
        }

        // Get result size and allocate output
        size_t result_size = getResultSize(node, storage_);
        auto alloc_result = storage_.allocate(id, result_size, dtype);
        if (!alloc_result)
            return alloc_result.error();
        void* out_data = *alloc_result;

        // Binary operations
        if (isBinaryOp(op)) {
            if (node->numOperands() < 2) {
                return Error(ErrorCode::kInvalidIR, "Binary op missing operands");
            }

            auto* buf_a = storage_.get(node->operand(0));
            auto* buf_b = storage_.get(node->operand(1));

            if (!buf_a || !buf_b) {
                return Error(ErrorCode::kRuntimeError, "Operand buffer not found");
            }

            // Handle scalar broadcasting: if one operand is scalar (size=1), broadcast it
            void* data_a = buf_a->data;
            void* data_b = buf_b->data;
            std::vector<uint8_t> broadcast_buffer;

            if (buf_a->size == 1 && result_size > 1) {
                // Broadcast operand A
                size_t elem_size = scalarTypeSize(dtype);
                broadcast_buffer.resize(result_size * elem_size);
                for (size_t i = 0; i < result_size; ++i) {
                    std::memcpy(broadcast_buffer.data() + i * elem_size, buf_a->data, elem_size);
                }
                data_a = broadcast_buffer.data();
            } else if (buf_b->size == 1 && result_size > 1) {
                // Broadcast operand B
                size_t elem_size = scalarTypeSize(dtype);
                broadcast_buffer.resize(result_size * elem_size);
                for (size_t i = 0; i < result_size; ++i) {
                    std::memcpy(broadcast_buffer.data() + i * elem_size, buf_b->data, elem_size);
                }
                data_b = broadcast_buffer.data();
            }

            return executeBinaryOp(op, out_data, data_a, data_b, result_size, dtype);
        }

        // Unary operations
        if (isUnaryOp(op)) {
            if (node->numOperands() < 1) {
                return Error(ErrorCode::kInvalidIR, "Unary op missing operand");
            }

            auto* buf_in = storage_.get(node->operand(0));
            if (!buf_in) {
                return Error(ErrorCode::kRuntimeError, "Operand buffer not found");
            }

            return executeUnaryOp(op, out_data, buf_in->data, buf_in->size, dtype);
        }

        // Reduction operations
        if (isReductionOp(op)) {
            if (node->numOperands() < 1) {
                return Error(ErrorCode::kInvalidIR, "Reduction op missing operand");
            }

            auto* buf_in = storage_.get(node->operand(0));
            if (!buf_in) {
                return Error(ErrorCode::kRuntimeError, "Operand buffer not found");
            }

            return executeReductionOp(op, out_data, buf_in->data, buf_in->size, dtype);
        }

        // Comparison operations (return float mask 0.0/1.0)
        if (isComparisonOp(op)) {
            if (node->numOperands() < 2) {
                return Error(ErrorCode::kInvalidIR, "Comparison op missing operands");
            }

            auto* buf_a = storage_.get(node->operand(0));
            auto* buf_b = storage_.get(node->operand(1));

            if (!buf_a || !buf_b) {
                return Error(ErrorCode::kRuntimeError, "Operand buffer not found");
            }

            return executeComparisonOp(op, out_data, buf_a->data, buf_b->data, result_size, dtype);
        }

        // Logical operations (binary: And, Or, Xor)
        if (isLogicalOp(op)) {
            if (node->numOperands() < 2) {
                return Error(ErrorCode::kInvalidIR, "Logical op missing operands");
            }

            auto* buf_a = storage_.get(node->operand(0));
            auto* buf_b = storage_.get(node->operand(1));

            if (!buf_a || !buf_b) {
                return Error(ErrorCode::kRuntimeError, "Operand buffer not found");
            }

            return executeLogicalOp(op, out_data, buf_a->data, buf_b->data, result_size, dtype);
        }

        // FMA operation
        if (op == OpCode::kFma) {
            if (node->numOperands() < 3) {
                return Error(ErrorCode::kInvalidIR, "FMA op missing operands");
            }

            auto* buf_a = storage_.get(node->operand(0));
            auto* buf_b = storage_.get(node->operand(1));
            auto* buf_c = storage_.get(node->operand(2));

            if (!buf_a || !buf_b || !buf_c) {
                return Error(ErrorCode::kRuntimeError, "FMA operand buffer not found");
            }

            return executeFmaOp(out_data, buf_a->data, buf_b->data, buf_c->data, result_size,
                                dtype);
        }

        // Select operation (ternary) - multi-type support
        if (op == OpCode::kSelect) {
            if (node->numOperands() < 3) {
                return Error(ErrorCode::kInvalidIR, "Select op missing operands");
            }

            auto* buf_mask = storage_.get(node->operand(0));
            auto* buf_true = storage_.get(node->operand(1));
            auto* buf_false = storage_.get(node->operand(2));

            if (!buf_mask || !buf_true || !buf_false) {
                return Error(ErrorCode::kRuntimeError, "Select operand buffer not found");
            }

            // Select based on dtype
            switch (dtype) {
            case ScalarType::kFloat32: {
                const float* mask = static_cast<const float*>(buf_mask->data);
                const float* true_val = static_cast<const float*>(buf_true->data);
                const float* false_val = static_cast<const float*>(buf_false->data);
                float* out = static_cast<float*>(out_data);
                for (size_t i = 0; i < result_size; ++i) {
                    out[i] = (mask[i] != 0.0f) ? true_val[i] : false_val[i];
                }
                return {};
            }
            case ScalarType::kFloat64: {
                const double* mask = static_cast<const double*>(buf_mask->data);
                const double* true_val = static_cast<const double*>(buf_true->data);
                const double* false_val = static_cast<const double*>(buf_false->data);
                double* out = static_cast<double*>(out_data);
                for (size_t i = 0; i < result_size; ++i) {
                    out[i] = (mask[i] != 0.0) ? true_val[i] : false_val[i];
                }
                return {};
            }
            case ScalarType::kInt32: {
                const int32_t* mask = static_cast<const int32_t*>(buf_mask->data);
                const int32_t* true_val = static_cast<const int32_t*>(buf_true->data);
                const int32_t* false_val = static_cast<const int32_t*>(buf_false->data);
                int32_t* out = static_cast<int32_t*>(out_data);
                for (size_t i = 0; i < result_size; ++i) {
                    out[i] = (mask[i] != 0) ? true_val[i] : false_val[i];
                }
                return {};
            }
            case ScalarType::kInt64: {
                const int64_t* mask = static_cast<const int64_t*>(buf_mask->data);
                const int64_t* true_val = static_cast<const int64_t*>(buf_true->data);
                const int64_t* false_val = static_cast<const int64_t*>(buf_false->data);
                int64_t* out = static_cast<int64_t*>(out_data);
                for (size_t i = 0; i < result_size; ++i) {
                    out[i] = (mask[i] != 0) ? true_val[i] : false_val[i];
                }
                return {};
            }
            default:
                return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for Select");
            }
        }

        // Compress operation: pack elements where mask is non-zero - multi-type
        if (op == OpCode::kCompress) {
            if (node->numOperands() < 2) {
                return Error(ErrorCode::kInvalidIR, "Compress op missing operands");
            }

            auto* buf_data = storage_.get(node->operand(0));
            auto* buf_mask = storage_.get(node->operand(1));

            if (!buf_data || !buf_mask) {
                return Error(ErrorCode::kRuntimeError, "Compress operand buffer not found");
            }

            // Create uint8_t mask from typed input
            auto temp_mask = std::make_unique<uint8_t[]>(buf_data->size);

            switch (dtype) {
            case ScalarType::kFloat32: {
                const float* data = static_cast<const float*>(buf_data->data);
                const float* mask = static_cast<const float*>(buf_mask->data);
                for (size_t i = 0; i < buf_data->size; ++i) {
                    temp_mask[i] = (mask[i] != 0.0f) ? 0xFF : 0x00;
                }
                simd::Compress(static_cast<float*>(out_data), data, temp_mask.get(),
                               buf_data->size);
                return {};
            }
            case ScalarType::kFloat64: {
                const double* data = static_cast<const double*>(buf_data->data);
                const double* mask = static_cast<const double*>(buf_mask->data);
                for (size_t i = 0; i < buf_data->size; ++i) {
                    temp_mask[i] = (mask[i] != 0.0) ? 0xFF : 0x00;
                }
                simd::Compress(static_cast<double*>(out_data), data, temp_mask.get(),
                               buf_data->size);
                return {};
            }
            case ScalarType::kInt32: {
                const int32_t* data = static_cast<const int32_t*>(buf_data->data);
                const int32_t* mask = static_cast<const int32_t*>(buf_mask->data);
                for (size_t i = 0; i < buf_data->size; ++i) {
                    temp_mask[i] = (mask[i] != 0) ? 0xFF : 0x00;
                }
                simd::Compress(static_cast<int32_t*>(out_data), data, temp_mask.get(),
                               buf_data->size);
                return {};
            }
            default:
                return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for Compress");
            }
        }

        // Expand operation: scatter elements to positions where mask is non-zero - multi-type
        if (op == OpCode::kExpand) {
            if (node->numOperands() < 2) {
                return Error(ErrorCode::kInvalidIR, "Expand op missing operands");
            }

            auto* buf_data = storage_.get(node->operand(0));
            auto* buf_mask = storage_.get(node->operand(1));

            if (!buf_data || !buf_mask) {
                return Error(ErrorCode::kRuntimeError, "Expand operand buffer not found");
            }

            // Create uint8_t mask from typed input
            auto temp_mask = std::make_unique<uint8_t[]>(result_size);

            switch (dtype) {
            case ScalarType::kFloat32: {
                const float* mask = static_cast<const float*>(buf_mask->data);
                for (size_t i = 0; i < result_size; ++i) {
                    temp_mask[i] = (mask[i] != 0.0f) ? 0xFF : 0x00;
                }
                simd::Expand(static_cast<float*>(out_data),
                             static_cast<const float*>(buf_data->data), temp_mask.get(),
                             result_size);
                return {};
            }
            case ScalarType::kFloat64: {
                const double* mask = static_cast<const double*>(buf_mask->data);
                for (size_t i = 0; i < result_size; ++i) {
                    temp_mask[i] = (mask[i] != 0.0) ? 0xFF : 0x00;
                }
                simd::Expand(static_cast<double*>(out_data),
                             static_cast<const double*>(buf_data->data), temp_mask.get(),
                             result_size);
                return {};
            }
            case ScalarType::kInt32: {
                const int32_t* mask = static_cast<const int32_t*>(buf_mask->data);
                for (size_t i = 0; i < result_size; ++i) {
                    temp_mask[i] = (mask[i] != 0) ? 0xFF : 0x00;
                }
                simd::Expand(static_cast<int32_t*>(out_data),
                             static_cast<const int32_t*>(buf_data->data), temp_mask.get(),
                             result_size);
                return {};
            }
            default:
                return Error(ErrorCode::kUnsupportedType, "Unsupported dtype for Expand");
            }
        }

        // Load operation (identity for now - data is pre-loaded)
        if (op == OpCode::kLoad) {
            if (node->numOperands() < 1) {
                return Error(ErrorCode::kInvalidIR, "Load op missing pointer operand");
            }

            auto* buf_ptr = storage_.get(node->operand(0));
            if (!buf_ptr) {
                return Error(ErrorCode::kRuntimeError, "Load source buffer not found");
            }

            std::memcpy(out_data, buf_ptr->data, result_size * scalarTypeSize(dtype));
            return {};
        }

        return Error(ErrorCode::kNotSupported,
                     "Unsupported IR operation: " + std::string(opCodeName(op)));
    }
};

// =============================================================================
// Public API
// =============================================================================

Result<void> executeIR(const IRBuilder& builder, ValueId output, void* result_data,
                       size_t result_size, ScalarType result_dtype,
                       const std::vector<std::pair<ValueId, void*>>& inputs,
                       const std::vector<std::pair<ValueId, size_t>>& input_sizes,
                       const std::vector<std::pair<ValueId, ScalarType>>& input_types) {
    IRInterpreter interpreter;

    // First, register inputs by their ValueId (for backward compatibility)
    for (size_t i = 0; i < inputs.size(); ++i) {
        ValueId id = inputs[i].first;
        void* data = inputs[i].second;
        size_t size = (i < input_sizes.size()) ? input_sizes[i].second : result_size;
        ScalarType dtype = (i < input_types.size()) ? input_types[i].second : result_dtype;

        interpreter.registerInput(id, data, size, dtype);
    }

    // Map parameter nodes to inputs by their param_index attribute
    // This enables JAX-style parameter binding where kParameter nodes
    // are matched to inputs by their param_index (which is stable across optimization)
    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead())
            continue;

        if (node->opCode() == OpCode::kParameter) {
            size_t param_index = static_cast<size_t>(node->intAttr("param_index"));

            // Match by param_index (stable across optimization) instead of ValueId
            if (param_index < inputs.size()) {
                size_t size = (param_index < input_sizes.size()) ? input_sizes[param_index].second
                                                                 : result_size;
                ScalarType dtype = (param_index < input_types.size())
                                       ? input_types[param_index].second
                                       : result_dtype;
                interpreter.registerInputByIndex(param_index, inputs[param_index].second, size,
                                                 dtype);
            }
        }
    }

    return interpreter.execute(builder, output, result_data, result_size, result_dtype);
}

// Simplified overload for common case
Result<void> executeIR(const IRBuilder& builder, ValueId output, void* result_data,
                       size_t result_size, ScalarType result_dtype) {
    return executeIR(builder, output, result_data, result_size, result_dtype, {}, {}, {});
}

}  // namespace ir
}  // namespace bud
