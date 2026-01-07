// =============================================================================
// Bud Flow Lang - Pipeline Implementation
// =============================================================================

#include "bud_flow_lang/pipeline.h"

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/codegen/hwy_ops.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

namespace bud {

// =============================================================================
// Pipeline Implementation
// =============================================================================

Pipeline::Pipeline() = default;
Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

// Operation name to OpCode mapping
ir::OpCode Pipeline::opCodeFromName(const std::string& name) {
    static const std::unordered_map<std::string, ir::OpCode> name_to_op = {
        {"add", ir::OpCode::kAdd},   {"sub", ir::OpCode::kSub},     {"mul", ir::OpCode::kMul},
        {"div", ir::OpCode::kDiv},   {"neg", ir::OpCode::kNeg},     {"abs", ir::OpCode::kAbs},
        {"sqrt", ir::OpCode::kSqrt}, {"rsqrt", ir::OpCode::kRsqrt}, {"exp", ir::OpCode::kExp},
        {"log", ir::OpCode::kLog},   {"sin", ir::OpCode::kSin},     {"cos", ir::OpCode::kCos},
        {"tanh", ir::OpCode::kTanh}, {"min", ir::OpCode::kMin},     {"max", ir::OpCode::kMax},
        {"pow", ir::OpCode::kPow},
    };

    auto it = name_to_op.find(name);
    return it != name_to_op.end() ? it->second : ir::OpCode::kConstantScalar;
}

Pipeline& Pipeline::add(const std::string& op_name) {
    PipelineStage stage;
    stage.name = op_name;
    stage.kind = OperationKind::kUnary;
    stage.op_code = opCodeFromName(op_name);
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::addScalar(const std::string& op_name, float scalar) {
    PipelineStage stage;
    stage.name = op_name;
    stage.kind = OperationKind::kTransform;
    stage.op_code = opCodeFromName(op_name);
    stage.scalar_param = scalar;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::addOp(ir::OpCode op) {
    PipelineStage stage;
    stage.kind = OperationKind::kUnary;
    stage.op_code = op;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::addNamed(const std::string& name, ir::OpCode op) {
    PipelineStage stage;
    stage.name = name;
    stage.kind = OperationKind::kUnary;
    stage.op_code = op;
    stages_.push_back(std::move(stage));
    return *this;
}

// Convenience methods
Pipeline& Pipeline::multiply(float scalar) {
    PipelineStage stage;
    stage.name = "multiply";
    stage.kind = OperationKind::kTransform;
    stage.op_code = ir::OpCode::kMul;
    stage.scalar_param = scalar;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::add(float scalar) {
    PipelineStage stage;
    stage.name = "add";
    stage.kind = OperationKind::kTransform;
    stage.op_code = ir::OpCode::kAdd;
    stage.scalar_param = scalar;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::subtract(float scalar) {
    PipelineStage stage;
    stage.name = "subtract";
    stage.kind = OperationKind::kTransform;
    stage.op_code = ir::OpCode::kSub;
    stage.scalar_param = scalar;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::divide(float scalar) {
    PipelineStage stage;
    stage.name = "divide";
    stage.kind = OperationKind::kTransform;
    stage.op_code = ir::OpCode::kDiv;
    stage.scalar_param = scalar;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::power(float exponent) {
    // x^n = exp(n * log(x))
    PipelineStage log_stage;
    log_stage.name = "log";
    log_stage.op_code = ir::OpCode::kLog;
    stages_.push_back(std::move(log_stage));

    PipelineStage mul_stage;
    mul_stage.name = "multiply";
    mul_stage.kind = OperationKind::kTransform;
    mul_stage.op_code = ir::OpCode::kMul;
    mul_stage.scalar_param = exponent;
    stages_.push_back(std::move(mul_stage));

    PipelineStage exp_stage;
    exp_stage.name = "exp";
    exp_stage.op_code = ir::OpCode::kExp;
    stages_.push_back(std::move(exp_stage));

    return *this;
}

Pipeline& Pipeline::sqrt() {
    PipelineStage stage;
    stage.name = "sqrt";
    stage.op_code = ir::OpCode::kSqrt;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::rsqrt() {
    PipelineStage stage;
    stage.name = "rsqrt";
    stage.op_code = ir::OpCode::kRsqrt;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::abs() {
    PipelineStage stage;
    stage.name = "abs";
    stage.op_code = ir::OpCode::kAbs;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::neg() {
    PipelineStage stage;
    stage.name = "neg";
    stage.op_code = ir::OpCode::kNeg;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::exp() {
    PipelineStage stage;
    stage.name = "exp";
    stage.op_code = ir::OpCode::kExp;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::log() {
    PipelineStage stage;
    stage.name = "log";
    stage.op_code = ir::OpCode::kLog;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::sin() {
    PipelineStage stage;
    stage.name = "sin";
    stage.op_code = ir::OpCode::kSin;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::cos() {
    PipelineStage stage;
    stage.name = "cos";
    stage.op_code = ir::OpCode::kCos;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::tanh() {
    PipelineStage stage;
    stage.name = "tanh";
    stage.op_code = ir::OpCode::kTanh;
    stages_.push_back(std::move(stage));
    return *this;
}

Pipeline& Pipeline::clamp(float lo, float hi) {
    // Clamp is min(max(x, lo), hi)
    // Implemented as two stages for now
    PipelineStage max_stage;
    max_stage.name = "max_lo";
    max_stage.kind = OperationKind::kTransform;
    max_stage.op_code = ir::OpCode::kMax;
    max_stage.scalar_param = lo;
    stages_.push_back(std::move(max_stage));

    PipelineStage min_stage;
    min_stage.name = "min_hi";
    min_stage.kind = OperationKind::kTransform;
    min_stage.op_code = ir::OpCode::kMin;
    min_stage.scalar_param = hi;
    stages_.push_back(std::move(min_stage));

    return *this;
}

void Pipeline::clear() {
    stages_.clear();
}

const PipelineStage& Pipeline::stageAt(size_t index) const {
    return stages_.at(index);
}

bool Pipeline::canFuse() const {
    // Can fuse if all operations are element-wise (no reductions in middle)
    for (size_t i = 0; i + 1 < stages_.size(); ++i) {
        if (stages_[i].is_reduction) {
            return false;  // Reduction in middle blocks fusion
        }
    }
    return true;
}

float Pipeline::estimatedSpeedup() const {
    if (stages_.empty())
        return 1.0f;

    // Estimate based on memory bandwidth savings
    // Each fused op saves one read and one write
    size_t n = stages_.size();

    // Unfused: n reads + n writes per op = 2n memory ops
    // Fused: 1 read + 1 write = 2 memory ops
    // Speedup = 2n / 2 = n (theoretical max)

    // Practical speedup is lower due to compute bound
    // Assume 50% of theoretical
    return std::min(static_cast<float>(n) * 0.5f, static_cast<float>(n));
}

float Pipeline::memoryReduction() const {
    if (stages_.size() <= 1)
        return 1.0f;

    // Unfused: n-1 intermediate arrays
    // Fused: 0 intermediate arrays
    return static_cast<float>(stages_.size());
}

Result<Bunch> Pipeline::run(const Bunch& input) const {
    if (stages_.empty()) {
        // No-op pipeline, return copy of input
        return Bunch::fromData(static_cast<const float*>(input.data()), input.size());
    }

    if (canFuse()) {
        return executeFused(input);
    } else {
        return executeUnfused(input);
    }
}

Result<void> Pipeline::runInPlace(Bunch& input) const {
    auto result = run(input);
    if (!result) {
        return Error(result.error().code(), std::string(result.error().message()));
    }

    // Copy result back to input
    const float* src = static_cast<const float*>(result->data());
    float* dst = static_cast<float*>(input.mutableData());
    std::copy(src, src + input.size(), dst);

    return {};
}

Result<Bunch> Pipeline::executeUnfused(const Bunch& input) const {
    // Execute each stage separately (creates intermediate arrays)
    auto current = Bunch::fromData(static_cast<const float*>(input.data()), input.size());
    if (!current) {
        return Error(ErrorCode::kAllocationFailed, "Failed to copy input");
    }

    for (const auto& stage : stages_) {
        // Apply operation based on opcode
        switch (stage.op_code) {
        case ir::OpCode::kAbs:
            current = current->abs();
            break;
        case ir::OpCode::kNeg:
            current = -(*current);
            break;
        case ir::OpCode::kSqrt:
            current = current->sqrt();
            break;
        case ir::OpCode::kRsqrt:
            current = current->rsqrt();
            break;
        case ir::OpCode::kExp:
            current = current->exp();
            break;
        case ir::OpCode::kLog:
            current = current->log();
            break;
        case ir::OpCode::kSin:
            current = current->sin();
            break;
        case ir::OpCode::kCos:
            current = current->cos();
            break;
        case ir::OpCode::kTanh:
            current = current->tanh();
            break;
        case ir::OpCode::kAdd:
            current = *current + stage.scalar_param;
            break;
        case ir::OpCode::kSub:
            current = *current - stage.scalar_param;
            break;
        case ir::OpCode::kMul:
            current = *current * stage.scalar_param;
            break;
        case ir::OpCode::kDiv:
            current = *current / stage.scalar_param;
            break;
        default:
            // Skip unknown ops
            break;
        }
    }

    return current;
}

Result<Bunch> Pipeline::executeFused(const Bunch& input) const {
    // Build IR graph from pipeline stages
    Arena arena;
    ir::IRBuilder builder(arena);

    const size_t count = input.size();

    // Create parameter node for input (index 0)
    TypeDesc input_type(ScalarType::kFloat32, Shape::vector(count));
    ir::ValueId current = builder.parameter(0, input_type);

    // Build IR for each stage
    for (const auto& stage : stages_) {
        ir::ValueId next = ir::ValueId::invalid();

        switch (stage.op_code) {
        // Unary operations
        case ir::OpCode::kAbs:
            next = builder.abs(current);
            break;
        case ir::OpCode::kNeg:
            next = builder.neg(current);
            break;
        case ir::OpCode::kSqrt:
            next = builder.sqrt(current);
            break;
        case ir::OpCode::kRsqrt:
            next = builder.rsqrt(current);
            break;
        case ir::OpCode::kExp:
            next = builder.exp(current);
            break;
        case ir::OpCode::kLog:
            next = builder.log(current);
            break;
        case ir::OpCode::kSin:
            next = builder.sin(current);
            break;
        case ir::OpCode::kCos:
            next = builder.cos(current);
            break;
        case ir::OpCode::kTanh:
            next = builder.tanh(current);
            break;

        // Binary operations with scalar
        case ir::OpCode::kAdd: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.add(current, scalar);
            break;
        }
        case ir::OpCode::kSub: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.sub(current, scalar);
            break;
        }
        case ir::OpCode::kMul: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.mul(current, scalar);
            break;
        }
        case ir::OpCode::kDiv: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.div(current, scalar);
            break;
        }
        case ir::OpCode::kMin: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.min(current, scalar);
            break;
        }
        case ir::OpCode::kMax: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.max(current, scalar);
            break;
        }

        default:
            // Fallback to unfused for unsupported ops
            return executeUnfused(input);
        }

        if (!next.isValid()) {
            return Error(ErrorCode::kInternalError, "Failed to build IR node");
        }
        current = next;
    }

    // Allocate output buffer
    auto output = Bunch::zeros(count, ScalarType::kFloat32);
    if (!output) {
        return Error(ErrorCode::kAllocationFailed, "Failed to allocate output buffer");
    }

    // Execute the IR graph
    std::vector<std::pair<ir::ValueId, void*>> inputs = {
        {ir::ValueId{0}, const_cast<void*>(input.data())}};
    std::vector<std::pair<ir::ValueId, size_t>> input_sizes = {{ir::ValueId{0}, count}};
    std::vector<std::pair<ir::ValueId, ScalarType>> input_types = {
        {ir::ValueId{0}, ScalarType::kFloat32}};

    auto result = ir::executeIR(builder, current, output->mutableData(), count,
                                ScalarType::kFloat32, inputs, input_sizes, input_types);

    if (!result) {
        return Error(result.error().code(), std::string(result.error().message()));
    }

    return output;
}

Result<std::unique_ptr<ir::IRModule>> Pipeline::compile() const {
    if (stages_.empty()) {
        return Error(ErrorCode::kInvalidInput, "Cannot compile empty pipeline");
    }

    // Create a new IR module for this pipeline
    auto module = std::make_unique<ir::IRModule>("pipeline");
    auto& builder = module->builder();

    // Create parameter node for input (index 0)
    // Use a placeholder size of 1 - actual size is substituted at runtime
    TypeDesc input_type(ScalarType::kFloat32, Shape::vector(1));
    ir::ValueId current = builder.parameter(0, input_type);

    // Build IR for each stage
    for (const auto& stage : stages_) {
        ir::ValueId next = ir::ValueId::invalid();

        switch (stage.op_code) {
        // Unary operations
        case ir::OpCode::kAbs:
            next = builder.abs(current);
            break;
        case ir::OpCode::kNeg:
            next = builder.neg(current);
            break;
        case ir::OpCode::kSqrt:
            next = builder.sqrt(current);
            break;
        case ir::OpCode::kRsqrt:
            next = builder.rsqrt(current);
            break;
        case ir::OpCode::kExp:
            next = builder.exp(current);
            break;
        case ir::OpCode::kLog:
            next = builder.log(current);
            break;
        case ir::OpCode::kSin:
            next = builder.sin(current);
            break;
        case ir::OpCode::kCos:
            next = builder.cos(current);
            break;
        case ir::OpCode::kTanh:
            next = builder.tanh(current);
            break;

        // Binary operations with scalar
        case ir::OpCode::kAdd: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.add(current, scalar);
            break;
        }
        case ir::OpCode::kSub: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.sub(current, scalar);
            break;
        }
        case ir::OpCode::kMul: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.mul(current, scalar);
            break;
        }
        case ir::OpCode::kDiv: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.div(current, scalar);
            break;
        }
        case ir::OpCode::kMin: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.min(current, scalar);
            break;
        }
        case ir::OpCode::kMax: {
            ir::ValueId scalar = builder.constant(stage.scalar_param);
            next = builder.max(current, scalar);
            break;
        }

        default:
            return Error(ErrorCode::kNotImplemented,
                         "Unsupported operation in pipeline: " + std::string(stage.name));
        }

        if (!next.isValid()) {
            return Error(ErrorCode::kInternalError,
                         "Failed to build IR node for: " + std::string(stage.name));
        }
        current = next;
    }

    // Set the output of the module
    module->setOutput(current);

    return module;
}

std::string Pipeline::toString() const {
    std::ostringstream ss;
    ss << "Pipeline([";
    for (size_t i = 0; i < stages_.size(); ++i) {
        if (i > 0)
            ss << " -> ";
        ss << stages_[i].name;
        if (stages_[i].kind == OperationKind::kTransform) {
            ss << "(" << stages_[i].scalar_param << ")";
        }
    }
    ss << "])";
    return ss.str();
}

}  // namespace bud
