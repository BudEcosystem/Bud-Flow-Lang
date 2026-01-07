// =============================================================================
// Bud Flow Lang - Tracing Context Implementation
// =============================================================================

#include "tracing_context.h"

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/codegen/hwy_ops.h"
#include "bud_flow_lang/ir.h"

#include <hwy/aligned_allocator.h>

#include <spdlog/spdlog.h>

namespace bud {

// =============================================================================
// Thread-Local Context Storage
// =============================================================================

namespace {
thread_local TracingContext* g_current_context = nullptr;
}  // namespace

// =============================================================================
// TracingContext Implementation
// =============================================================================

TracingContext::TracingContext()
    : module_(std::make_unique<ir::IRModule>("traced_kernel")), prev_context_(g_current_context) {
    // Set this as the current context
    setCurrent(this);
    spdlog::debug("TracingContext created, tracing enabled");
}

TracingContext::~TracingContext() {
    // Restore previous context
    setCurrent(prev_context_);
    spdlog::debug("TracingContext destroyed, tracing {}", prev_context_ ? "continues" : "disabled");
}

TracingContext* TracingContext::current() {
    return g_current_context;
}

void TracingContext::setCurrent(TracingContext* ctx) {
    g_current_context = ctx;
}

Bunch TracingContext::createTracerInput(size_t param_index, ScalarType dtype, const Shape& shape) {
    // Determine element count from shape
    size_t count = 1;
    for (size_t i = 0; i < shape.rank(); ++i) {
        count *= shape[i];
    }

    // Create a parameter node to represent the input (JAX-style input binding)
    // Parameters are substituted with actual input data at execution time
    TypeDesc type(dtype, shape);
    ir::ValueId input_id = module_->builder().parameter(param_index, type);

    if (!input_id.isValid()) {
        spdlog::error("Failed to create parameter node for input {}", param_index);
        return Bunch();
    }

    // Store the input ID for later mapping
    input_ids_.push_back(input_id);

    spdlog::debug("Created parameter input {} (param {}): {} elements, dtype {}", input_id.id,
                  param_index, count, scalarTypeName(dtype));

    // Create a Bunch that references this IR value
    // The Bunch will be in "tracer" mode - operations on it record to IR
    auto result = Bunch::zeros(count, dtype);
    if (!result) {
        spdlog::error("Failed to create tracer Bunch");
        return Bunch();
    }

    // Set the Bunch to tracer mode by setting its IR module and value
    result->setTracingState(module_.get(), input_id);

    return std::move(*result);
}

ir::ValueId TracingContext::extractOutput(const Bunch& result) const {
    // Get the IR value from the result Bunch
    auto value_id = result.tracingValueId();
    if (!value_id.isValid()) {
        spdlog::warn("extractOutput: Bunch is not a tracer, returning invalid ValueId");
    }
    return value_id;
}

// =============================================================================
// CompiledKernel Implementation
// =============================================================================

CompiledKernel::CompiledKernel(std::unique_ptr<ir::IRModule> module, ir::ValueId output_id,
                               std::vector<ir::ValueId> input_ids)
    : module_(std::move(module)), output_id_(output_id), input_ids_(std::move(input_ids)) {
    spdlog::debug("CompiledKernel created with {} inputs, output {}", input_ids_.size(),
                  output_id_.id);
}

Result<Bunch> CompiledKernel::execute(const std::vector<const Bunch*>& inputs) {
    if (inputs.size() != input_ids_.size()) {
        return Error(ErrorCode::kInvalidInput,
                     fmt::format("Expected {} inputs, got {}", input_ids_.size(), inputs.size()));
    }

    // Validate inputs and determine output size
    size_t output_size = 0;
    ScalarType output_dtype = ScalarType::kFloat32;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i] || !inputs[i]->isValid()) {
            return Error(ErrorCode::kInvalidInput, fmt::format("Input {} is invalid", i));
        }
        if (i == 0) {
            output_size = inputs[i]->size();
            output_dtype = inputs[i]->dtype();
        }
    }

    // Allocate output buffer
    auto result = Bunch::zeros(output_size, output_dtype);
    if (!result) {
        return Error(ErrorCode::kAllocationFailed, "Failed to allocate output buffer");
    }

    // Build input mapping for IR interpreter
    std::vector<std::pair<ir::ValueId, void*>> input_data;
    std::vector<std::pair<ir::ValueId, size_t>> input_sizes;
    std::vector<std::pair<ir::ValueId, ScalarType>> input_types;

    for (size_t i = 0; i < inputs.size(); ++i) {
        input_data.emplace_back(input_ids_[i], const_cast<void*>(inputs[i]->data()));
        input_sizes.emplace_back(input_ids_[i], inputs[i]->size());
        input_types.emplace_back(input_ids_[i], inputs[i]->dtype());
    }

    // Execute the IR
    auto exec_result =
        ir::executeIR(module_->builder(), output_id_, result->mutableData(), output_size,
                      output_dtype, input_data, input_sizes, input_types);

    if (!exec_result) {
        return Error(exec_result.error().code(), std::string("IR execution failed: ") +
                                                     std::string(exec_result.error().message()));
    }

    spdlog::debug("CompiledKernel executed successfully, {} elements", output_size);
    return std::move(*result);
}

// =============================================================================
// KernelCache Implementation
// =============================================================================

CompiledKernel* KernelCache::find(const KernelSignature& sig) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(sig);
    if (it != cache_.end()) {
        spdlog::debug("KernelCache hit");
        return it->second.get();
    }
    spdlog::debug("KernelCache miss");
    return nullptr;
}

CompiledKernel* KernelCache::insert(const KernelSignature& sig,
                                    std::unique_ptr<CompiledKernel> kernel) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = cache_.emplace(sig, std::move(kernel));
    if (inserted) {
        spdlog::debug("KernelCache: inserted new kernel (cache size: {})", cache_.size());
    }
    return it->second.get();
}

void KernelCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    spdlog::debug("KernelCache cleared");
}

size_t KernelCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

// =============================================================================
// Optimization Pipeline
// =============================================================================

Result<void> runOptimizationPipeline(ir::IRModule& module, int opt_level) {
    if (opt_level <= 0) {
        spdlog::debug("Optimization disabled (level 0)");
        return {};
    }

    spdlog::debug("Running optimization pipeline at level {}", opt_level);

    // Use the built-in optimize method which already handles all passes
    auto result = module.optimize(opt_level);
    if (!result) {
        spdlog::error("Optimization failed: {}", result.error().message());
        return result;
    }

    spdlog::debug("Optimization complete");
    return {};
}

}  // namespace bud
