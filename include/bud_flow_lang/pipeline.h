// =============================================================================
// Bud Flow Lang - Pipeline Class (Developer Tier)
// =============================================================================
//
// Pipeline provides explicit fusion control for complex operation chains.
// Operations added to a Pipeline are fused into a single kernel, eliminating
// intermediate arrays and reducing memory bandwidth.
//
// Usage:
//   from flow import Pipeline
//
//   # Create a pipeline of operations
//   pipeline = Pipeline()
//   pipeline.add(lambda x: x * 2)
//   pipeline.add(lambda x: x + 1)
//   pipeline.add(lambda x: x ** 0.5)
//
//   # Run the fused pipeline - all ops in one pass
//   result = pipeline.run(data)
//
// Benefits:
// - Eliminates intermediate arrays (memory bandwidth reduction)
// - Enables cross-operation optimization
// - Predictable fusion boundaries
// - Better cache utilization
//
// =============================================================================

#pragma once

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/ir.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bud {

namespace ir {
class IRModule;
class IRBuilder;
}  // namespace ir

// =============================================================================
// OperationKind - Types of operations in a pipeline
// =============================================================================

enum class OperationKind {
    kUnary,      // f(x) -> y
    kBinary,     // f(x, y) -> z
    kReduction,  // f(x) -> scalar
    kTransform,  // f(x, params) -> y
    kCustom,     // User-defined operation
};

// =============================================================================
// PipelineStage - Single operation in the pipeline
// =============================================================================

struct PipelineStage {
    std::string name;
    OperationKind kind = OperationKind::kUnary;

    // IR operation code (for built-in ops)
    ir::OpCode op_code = ir::OpCode::kConstantScalar;

    // For scalar operations
    float scalar_param = 0.0f;

    // Custom operation callback (for kCustom)
    std::function<Bunch(const Bunch&)> custom_fn;

    // Is this a reduction?
    bool is_reduction = false;
};

// =============================================================================
// Pipeline - Fusion engine for operation chains
// =============================================================================

class Pipeline {
  public:
    Pipeline();
    ~Pipeline();

    // Non-copyable, movable
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    // =========================================================================
    // Add Operations
    // =========================================================================

    // Add element-wise operation by name
    Pipeline& add(const std::string& op_name);

    // Add scalar operation: x op scalar
    Pipeline& addScalar(const std::string& op_name, float scalar);

    // Add unary function by IR opcode
    Pipeline& addOp(ir::OpCode op);

    // Add with custom name
    Pipeline& addNamed(const std::string& name, ir::OpCode op);

    // =========================================================================
    // Common Operations (convenience methods)
    // =========================================================================

    // Arithmetic
    Pipeline& multiply(float scalar);  // x * scalar
    Pipeline& add(float scalar);       // x + scalar
    Pipeline& subtract(float scalar);  // x - scalar
    Pipeline& divide(float scalar);    // x / scalar
    Pipeline& power(float exponent);   // x ^ exponent

    // Math functions
    Pipeline& sqrt();   // sqrt(x)
    Pipeline& rsqrt();  // 1/sqrt(x)
    Pipeline& abs();    // |x|
    Pipeline& neg();    // -x
    Pipeline& exp();    // e^x
    Pipeline& log();    // ln(x)
    Pipeline& sin();    // sin(x)
    Pipeline& cos();    // cos(x)
    Pipeline& tanh();   // tanh(x)

    // Clamping
    Pipeline& clamp(float lo, float hi);  // clamp(x, lo, hi)

    // =========================================================================
    // Pipeline Properties
    // =========================================================================

    // Get number of stages
    [[nodiscard]] size_t numStages() const { return stages_.size(); }

    // Is pipeline empty?
    [[nodiscard]] bool empty() const { return stages_.empty(); }

    // Clear all stages
    void clear();

    // Get stage at index
    [[nodiscard]] const PipelineStage& stageAt(size_t index) const;

    // =========================================================================
    // Execution
    // =========================================================================

    // Run pipeline on input data
    [[nodiscard]] Result<Bunch> run(const Bunch& input) const;

    // Run pipeline in-place (modifies input)
    [[nodiscard]] Result<void> runInPlace(Bunch& input) const;

    // =========================================================================
    // Optimization & Analysis
    // =========================================================================

    // Check if pipeline can be fused into single kernel
    [[nodiscard]] bool canFuse() const;

    // Get estimated speedup from fusion
    [[nodiscard]] float estimatedSpeedup() const;

    // Get memory traffic reduction ratio
    [[nodiscard]] float memoryReduction() const;

    // =========================================================================
    // IR Generation
    // =========================================================================

    // Compile pipeline to IR module
    [[nodiscard]] Result<std::unique_ptr<ir::IRModule>> compile() const;

    // =========================================================================
    // String Representation
    // =========================================================================

    [[nodiscard]] std::string toString() const;

  private:
    std::vector<PipelineStage> stages_;

    // Helpers
    static ir::OpCode opCodeFromName(const std::string& name);
    [[nodiscard]] Result<Bunch> executeUnfused(const Bunch& input) const;
    [[nodiscard]] Result<Bunch> executeFused(const Bunch& input) const;
};

// =============================================================================
// PipelineBuilder - Fluent API for building pipelines
// =============================================================================

class PipelineBuilder {
  public:
    PipelineBuilder() = default;

    // Fluent API
    PipelineBuilder& multiply(float scalar) {
        pipeline_.multiply(scalar);
        return *this;
    }
    PipelineBuilder& add(float scalar) {
        pipeline_.add(scalar);
        return *this;
    }
    PipelineBuilder& subtract(float scalar) {
        pipeline_.subtract(scalar);
        return *this;
    }
    PipelineBuilder& divide(float scalar) {
        pipeline_.divide(scalar);
        return *this;
    }
    PipelineBuilder& sqrt() {
        pipeline_.sqrt();
        return *this;
    }
    PipelineBuilder& exp() {
        pipeline_.exp();
        return *this;
    }
    PipelineBuilder& log() {
        pipeline_.log();
        return *this;
    }
    PipelineBuilder& sin() {
        pipeline_.sin();
        return *this;
    }
    PipelineBuilder& cos() {
        pipeline_.cos();
        return *this;
    }
    PipelineBuilder& tanh() {
        pipeline_.tanh();
        return *this;
    }
    PipelineBuilder& abs() {
        pipeline_.abs();
        return *this;
    }
    PipelineBuilder& neg() {
        pipeline_.neg();
        return *this;
    }
    PipelineBuilder& clamp(float lo, float hi) {
        pipeline_.clamp(lo, hi);
        return *this;
    }

    // Build and return pipeline
    Pipeline build() { return std::move(pipeline_); }

    // Run directly
    [[nodiscard]] Result<Bunch> run(const Bunch& input) { return pipeline_.run(input); }

  private:
    Pipeline pipeline_;
};

// Convenience function to create builder
[[nodiscard]] inline PipelineBuilder pipeline() {
    return PipelineBuilder();
}

}  // namespace bud
