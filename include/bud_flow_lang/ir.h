#pragma once

// =============================================================================
// Bud Flow Lang - Intermediate Representation
// =============================================================================
//
// Weld-style lazy evaluation IR for maximum fusion opportunities.
// Key features:
// - SSA form for easy optimization
// - Deferred execution model
// - Explicit SIMD operations
//

#include "bud_flow_lang/arena.h"
#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/type_system.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace bud {
namespace ir {

// Forward declarations
class IRNode;
class IRBuilder;
class IRModule;

// =============================================================================
// Operation Codes
// =============================================================================

enum class OpCode : uint16_t {
    // Constants
    kConstantScalar = 0,
    kConstantVector,

    // Memory
    kLoad,
    kStore,
    kAlloc,

    // Binary arithmetic
    kAdd,
    kSub,
    kMul,
    kDiv,
    kMod,
    kMin,
    kMax,
    kPow,

    // Fused operations
    kFma,   // a * b + c
    kFnma,  // -(a * b) + c

    // Unary arithmetic
    kNeg,
    kAbs,
    kSqrt,
    kRsqrt,  // 1/sqrt(x)
    kRcp,    // 1/x

    // Transcendentals
    kExp,
    kLog,
    kSin,
    kCos,
    kTan,
    kTanh,
    kSigmoid,

    // Comparison
    kEq,
    kNe,
    kLt,
    kLe,
    kGt,
    kGe,

    // Logical
    kAnd,
    kOr,
    kNot,
    kXor,

    // Bitwise
    kBitAnd,
    kBitOr,
    kBitXor,
    kBitNot,
    kShl,
    kShr,

    // Reductions
    kReduceSum,
    kReduceMax,
    kReduceMin,
    kReduceProd,
    kReduceAny,
    kReduceAll,

    // Horizontal operations
    kHorizontalSum,
    kHorizontalMax,
    kHorizontalMin,

    // Vector operations
    kBroadcast,
    kGather,
    kScatter,
    kPermute,
    kShuffle,
    kConcat,
    kSlice,

    // Type conversions
    kCast,
    kBitcast,
    kPromote,
    kDemote,

    // Control flow (for loops)
    kFor,
    kIf,
    kWhile,

    // Masking
    kSelect,    // mask ? a : b
    kCompress,  // extract elements where mask is true
    kExpand,    // scatter elements to positions where mask is true

    // Special
    kPhi,  // SSA phi node
    kReturn,
    kCall,
};

std::string_view opCodeName(OpCode op);

// =============================================================================
// IR Value ID
// =============================================================================

// Unique identifier for IR values (SSA names)
struct ValueId {
    uint32_t id = 0;

    bool operator==(const ValueId& other) const { return id == other.id; }
    bool operator!=(const ValueId& other) const { return id != other.id; }
    bool operator<(const ValueId& other) const { return id < other.id; }

    static ValueId invalid() { return {UINT32_MAX}; }
    [[nodiscard]] bool isValid() const { return id != UINT32_MAX; }
};

// =============================================================================
// IR Node
// =============================================================================

class IRNode {
  public:
    IRNode(OpCode op, TypeDesc type, ValueId id) : op_(op), type_(type), id_(id) {}

    // Accessors
    [[nodiscard]] OpCode opCode() const { return op_; }
    [[nodiscard]] const TypeDesc& type() const { return type_; }
    [[nodiscard]] ValueId id() const { return id_; }

    // Operands
    void addOperand(ValueId operand) { operands_.push_back(operand); }
    [[nodiscard]] size_t numOperands() const { return operands_.size(); }
    [[nodiscard]] ValueId operand(size_t i) const { return operands_[i]; }
    [[nodiscard]] const std::vector<ValueId>& operands() const { return operands_; }

    // Operand mutation (for optimization passes)
    void setOperand(size_t i, ValueId new_operand) {
        if (i < operands_.size()) {
            operands_[i] = new_operand;
        }
    }
    void replaceOperand(ValueId old_val, ValueId new_val);
    void clearOperands() { operands_.clear(); }

    // Dead code marking (for DCE pass)
    [[nodiscard]] bool isDead() const { return dead_; }
    void markDead() { dead_ = true; }
    void markLive() { dead_ = false; }

    // ID remapping (for compaction)
    void setId(ValueId new_id) { id_ = new_id; }

    // Type mutation (for type inference refinement)
    void setType(TypeDesc new_type) { type_ = new_type; }

    // Attributes (operation-specific data)
    void setIntAttr(std::string_view name, int64_t value);
    void setFloatAttr(std::string_view name, double value);
    void setStringAttr(std::string_view name, std::string value);

    [[nodiscard]] int64_t intAttr(std::string_view name, int64_t default_val = 0) const;
    [[nodiscard]] double floatAttr(std::string_view name, double default_val = 0.0) const;
    [[nodiscard]] std::string_view stringAttr(std::string_view name) const;
    [[nodiscard]] bool hasAttr(std::string_view name) const;

    // Debug
    [[nodiscard]] std::string toString() const;

  private:
    OpCode op_;
    TypeDesc type_;
    ValueId id_;
    std::vector<ValueId> operands_;
    bool dead_ = false;  // For dead code elimination

    // Attributes stored inline for common cases
    struct AttrValue {
        std::string name;
        std::variant<int64_t, double, std::string> value;
    };
    std::vector<AttrValue> attrs_;
};

// =============================================================================
// IR Builder
// =============================================================================

class IRBuilder {
  public:
    explicit IRBuilder(Arena& arena);

    // Create constants
    ValueId constant(float value);
    ValueId constant(double value);
    ValueId constant(int32_t value);
    ValueId constant(int64_t value);
    ValueId constantVector(const std::vector<float>& values);

    // Binary operations
    ValueId add(ValueId lhs, ValueId rhs);
    ValueId sub(ValueId lhs, ValueId rhs);
    ValueId mul(ValueId lhs, ValueId rhs);
    ValueId div(ValueId lhs, ValueId rhs);
    ValueId min(ValueId lhs, ValueId rhs);
    ValueId max(ValueId lhs, ValueId rhs);

    // Fused operations
    ValueId fma(ValueId a, ValueId b, ValueId c);  // a * b + c

    // Unary operations
    ValueId neg(ValueId x);
    ValueId abs(ValueId x);
    ValueId sqrt(ValueId x);
    ValueId rsqrt(ValueId x);  // 1/sqrt(x)

    // Transcendentals
    ValueId exp(ValueId x);
    ValueId log(ValueId x);
    ValueId sin(ValueId x);
    ValueId cos(ValueId x);
    ValueId tanh(ValueId x);

    // Reductions
    ValueId reduceSum(ValueId x, int axis = -1);
    ValueId reduceMax(ValueId x, int axis = -1);
    ValueId reduceMin(ValueId x, int axis = -1);

    // Comparisons
    ValueId eq(ValueId lhs, ValueId rhs);
    ValueId lt(ValueId lhs, ValueId rhs);
    ValueId le(ValueId lhs, ValueId rhs);
    ValueId gt(ValueId lhs, ValueId rhs);
    ValueId ge(ValueId lhs, ValueId rhs);
    ValueId ne(ValueId lhs, ValueId rhs);

    // Select (ternary)
    ValueId select(ValueId mask, ValueId true_val, ValueId false_val);

    // Memory operations
    ValueId load(ValueId ptr, TypeDesc element_type);
    ValueId store(ValueId ptr, ValueId value);

    // Type conversions
    ValueId cast(ValueId x, ScalarType target_type);

    // Get node by ID
    [[nodiscard]] IRNode* getNode(ValueId id);
    [[nodiscard]] const IRNode* getNode(ValueId id) const;

    // Get all nodes
    [[nodiscard]] const std::vector<IRNode*>& nodes() const { return nodes_; }
    [[nodiscard]] std::vector<IRNode*>& mutableNodes() { return nodes_; }

    // Get live (non-dead) node count
    [[nodiscard]] size_t liveNodeCount() const;

    // =========================================================================
    // IR Mutation Methods (for optimization passes)
    // =========================================================================

    // Replace all uses of old_id with new_id throughout the IR
    // Returns the number of replacements made
    size_t replaceAllUses(ValueId old_id, ValueId new_id);

    // Mark a node as dead (will be removed by compactNodes)
    void markDead(ValueId id);

    // Mark a node as dead if it has no uses (recursive dead code elimination)
    // Returns true if the node was marked dead
    bool markDeadIfUnused(ValueId id);

    // Remove all dead nodes and renumber remaining nodes
    // Updates all operand references to use new IDs
    // Returns the number of nodes removed
    size_t compactNodes();

    // Check if a value is used by any live node
    [[nodiscard]] bool hasUses(ValueId id) const;

    // Count how many times a value is used
    [[nodiscard]] size_t useCount(ValueId id) const;

    // Find all nodes that use a given value
    [[nodiscard]] std::vector<ValueId> findUses(ValueId id) const;

    // Replace a node with a new operation (for peephole optimization)
    // The replacement node reuses the same ID
    void replaceNode(ValueId id, OpCode new_op, const std::vector<ValueId>& new_operands);

    // Clone a node with new operands (for fusion)
    ValueId cloneNode(ValueId source_id, const std::vector<ValueId>& new_operands);

    // Validation
    [[nodiscard]] Result<void> validate() const;

    // Debug output
    [[nodiscard]] std::string dump() const;

  private:
    ValueId createNode(OpCode op, TypeDesc type);
    ValueId createBinaryOp(OpCode op, ValueId lhs, ValueId rhs);
    ValueId createUnaryOp(OpCode op, ValueId operand);

    Arena& arena_;
    std::vector<IRNode*> nodes_;
    uint32_t next_id_ = 0;
};

// =============================================================================
// IR Module
// =============================================================================

class IRModule {
  public:
    explicit IRModule(std::string name);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] Arena& arena() { return arena_; }
    [[nodiscard]] IRBuilder& builder() { return builder_; }
    [[nodiscard]] const IRBuilder& builder() const { return builder_; }

    // Entry point for the module (output values)
    void setOutput(ValueId output) { output_ = output; }
    [[nodiscard]] ValueId output() const { return output_; }

    // Optimization passes
    Result<void> optimize(int level = 2);

    // Serialization
    [[nodiscard]] std::string toJson() const;
    static Result<IRModule> fromJson(std::string_view json);

  private:
    std::string name_;
    Arena arena_;
    IRBuilder builder_;
    ValueId output_ = ValueId::invalid();
};

// =============================================================================
// Fusion Analysis (from lazy_evaluator.cc)
// =============================================================================

// Analyze fusion opportunities for an IR module
// Returns pairs of (pattern_name, estimated_speedup)
std::vector<std::pair<std::string, double>> analyzeFusionOpportunities(const IRModule& module);

// Get estimated speedup from fusion for an IR module
double estimateFusionSpeedup(const IRModule& module);

// =============================================================================
// IR Interpreter (from ir_interpreter.cc)
// =============================================================================

// Execute an IR graph and store result in provided buffer
// inputs: vector of (ValueId, data pointer) pairs for external inputs
// input_sizes: vector of (ValueId, element count) pairs
// input_types: vector of (ValueId, ScalarType) pairs
Result<void> executeIR(const IRBuilder& builder, ValueId output, void* result_data,
                       size_t result_size, ScalarType result_dtype,
                       const std::vector<std::pair<ValueId, void*>>& inputs,
                       const std::vector<std::pair<ValueId, size_t>>& input_sizes,
                       const std::vector<std::pair<ValueId, ScalarType>>& input_types);

// Simplified overload - no external inputs
Result<void> executeIR(const IRBuilder& builder, ValueId output, void* result_data,
                       size_t result_size, ScalarType result_dtype);

}  // namespace ir
}  // namespace bud
