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

#include <spdlog/spdlog.h>

#include <algorithm>
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

// Execute a binary operation
Result<void> executeBinaryOp(OpCode op, void* out, const void* a, const void* b, size_t count,
                             ScalarType dtype) {
    if (dtype != ScalarType::kFloat32) {
        // For now, only support float32
        return Error(ErrorCode::kUnsupportedType, "IR interpreter only supports float32 currently");
    }

    float* out_f = static_cast<float*>(out);
    const float* a_f = static_cast<const float*>(a);
    const float* b_f = static_cast<const float*>(b);

    switch (op) {
    case OpCode::kAdd:
        simd::Add(out_f, a_f, b_f, count);
        break;
    case OpCode::kSub:
        simd::Sub(out_f, a_f, b_f, count);
        break;
    case OpCode::kMul:
        simd::Mul(out_f, a_f, b_f, count);
        break;
    case OpCode::kDiv:
        simd::Div(out_f, a_f, b_f, count);
        break;
    case OpCode::kMin:
        simd::Min(out_f, a_f, b_f, count);
        break;
    case OpCode::kMax:
        simd::Max(out_f, a_f, b_f, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported binary operation: " + std::string(opCodeName(op)));
    }

    return {};
}

// Execute a unary operation
Result<void> executeUnaryOp(OpCode op, void* out, const void* in, size_t count, ScalarType dtype) {
    if (dtype != ScalarType::kFloat32) {
        return Error(ErrorCode::kUnsupportedType, "IR interpreter only supports float32 currently");
    }

    float* out_f = static_cast<float*>(out);
    const float* in_f = static_cast<const float*>(in);

    switch (op) {
    case OpCode::kNeg:
        simd::Neg(out_f, in_f, count);
        break;
    case OpCode::kAbs:
        simd::Abs(out_f, in_f, count);
        break;
    case OpCode::kSqrt:
        simd::Sqrt(out_f, in_f, count);
        break;
    case OpCode::kRsqrt:
        simd::Rsqrt(out_f, in_f, count);
        break;
    case OpCode::kExp:
        simd::Exp(out_f, in_f, count);
        break;
    case OpCode::kLog:
        simd::Log(out_f, in_f, count);
        break;
    case OpCode::kSin:
        simd::Sin(out_f, in_f, count);
        break;
    case OpCode::kCos:
        simd::Cos(out_f, in_f, count);
        break;
    case OpCode::kTanh:
        simd::Tanh(out_f, in_f, count);
        break;
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported unary operation: " + std::string(opCodeName(op)));
    }

    return {};
}

// Execute a reduction operation
Result<void> executeReductionOp(OpCode op, void* out, const void* in, size_t count,
                                ScalarType dtype) {
    if (dtype != ScalarType::kFloat32) {
        return Error(ErrorCode::kUnsupportedType, "IR interpreter only supports float32 currently");
    }

    float* out_f = static_cast<float*>(out);
    const float* in_f = static_cast<const float*>(in);

    switch (op) {
    case OpCode::kReduceSum:
    case OpCode::kHorizontalSum: {
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum += in_f[i];
        }
        *out_f = sum;
        break;
    }
    case OpCode::kReduceMax:
    case OpCode::kHorizontalMax: {
        float max_val = in_f[0];
        for (size_t i = 1; i < count; ++i) {
            if (in_f[i] > max_val)
                max_val = in_f[i];
        }
        *out_f = max_val;
        break;
    }
    case OpCode::kReduceMin:
    case OpCode::kHorizontalMin: {
        float min_val = in_f[0];
        for (size_t i = 1; i < count; ++i) {
            if (in_f[i] < min_val)
                min_val = in_f[i];
        }
        *out_f = min_val;
        break;
    }
    case OpCode::kReduceProd: {
        float prod = 1.0f;
        for (size_t i = 0; i < count; ++i) {
            prod *= in_f[i];
        }
        *out_f = prod;
        break;
    }
    default:
        return Error(ErrorCode::kNotSupported,
                     "Unsupported reduction: " + std::string(opCodeName(op)));
    }

    return {};
}

// Execute FMA operation
Result<void> executeFmaOp(void* out, const void* a, const void* b, const void* c, size_t count,
                          ScalarType dtype) {
    if (dtype != ScalarType::kFloat32) {
        return Error(ErrorCode::kUnsupportedType, "IR interpreter only supports float32 currently");
    }

    simd::MulAdd(static_cast<float*>(out), static_cast<const float*>(a),
                 static_cast<const float*>(b), static_cast<const float*>(c), count);
    return {};
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

    // Execute an IR module and return the result
    Result<void> execute(const IRBuilder& builder, ValueId output, void* result_data,
                         size_t result_size, ScalarType result_dtype) {
        if (!output.isValid()) {
            return Error(ErrorCode::kInvalidInput, "Invalid output ValueId");
        }

        // Get topological order
        auto order = topologicalSort(builder, output);
        if (order.empty()) {
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

        // Handle constant vectors
        if (op == OpCode::kConstantVector) {
            // TODO: Implement constant vector handling
            return Error(ErrorCode::kNotImplemented, "Constant vectors not yet implemented");
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

            return executeBinaryOp(op, out_data, buf_a->data, buf_b->data, result_size, dtype);
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

        // Select operation (ternary)
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

            if (dtype == ScalarType::kFloat32) {
                const float* mask = static_cast<const float*>(buf_mask->data);
                const float* true_val = static_cast<const float*>(buf_true->data);
                const float* false_val = static_cast<const float*>(buf_false->data);
                float* out = static_cast<float*>(out_data);

                for (size_t i = 0; i < result_size; ++i) {
                    out[i] = (mask[i] != 0.0f) ? true_val[i] : false_val[i];
                }
                return {};
            }

            return Error(ErrorCode::kUnsupportedType, "Select only supports float32");
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

    // Register inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
        ValueId id = inputs[i].first;
        void* data = inputs[i].second;
        size_t size = (i < input_sizes.size()) ? input_sizes[i].second : result_size;
        ScalarType dtype = (i < input_types.size()) ? input_types[i].second : result_dtype;

        interpreter.registerInput(id, data, size, dtype);
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
