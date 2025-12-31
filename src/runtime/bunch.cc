// =============================================================================
// Bud Flow Lang - Bunch Implementation
// =============================================================================

#include "bud_flow_lang/bunch.h"

#include "bud_flow_lang/codegen/hwy_ops.h"
#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/numa_allocator.h"
#include "bud_flow_lang/memory/tiled_executor.h"

// Include TracingContext for checking if we're in a tracing context
// Forward declare to avoid circular dependency in header
namespace bud {
class TracingContext;
}

#include <hwy/aligned_allocator.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstring>
#include <limits>

// Forward declaration of TracingContext access (defined in tracing_context.cc)
namespace bud {
TracingContext* getActiveTracingContext();
}  // namespace bud

namespace bud {

// =============================================================================
// Memory Constants
// =============================================================================

namespace {

// Threshold for NUMA-aware allocation (1 MB)
constexpr size_t kNumaAllocationThreshold = 1024 * 1024;

// Threshold for tiled execution (1024 elements)
constexpr size_t kTilingThreshold = 1024;

}  // namespace

// =============================================================================
// Allocation Type Tracking
// =============================================================================

enum class AllocationType : uint8_t {
    kNone = 0,  // No allocation
    kHighway,   // Highway aligned allocation (standard)
    kNuma,      // NUMA-aware allocation (large arrays)
    kExternal,  // External memory (not owned)
};

// =============================================================================
// BunchImpl - Internal Implementation
// =============================================================================

class BunchImpl {
  public:
    BunchImpl() = default;

    ~BunchImpl() {
        if (data_ && owns_data_) {
            switch (allocation_type_) {
            case AllocationType::kHighway:
                hwy::FreeAlignedBytes(data_, nullptr, nullptr);
                break;
            case AllocationType::kNuma:
                memory::NumaAllocator::global().deallocate(data_, allocated_bytes_);
                break;
            case AllocationType::kNone:
            case AllocationType::kExternal:
                // Nothing to free
                break;
            }
        }
    }

    // Allocate storage
    Result<void> allocate(size_t count, ScalarType dtype) {
        dtype_ = dtype;
        count_ = count;
        shape_ = Shape::vector(count);

        size_t bytes = count * scalarTypeSize(dtype);
        allocated_bytes_ = bytes;

        // Use NUMA-aware allocation for large arrays (>1MB)
        if (bytes >= kNumaAllocationThreshold) {
            data_ = memory::NumaAllocator::global().allocateAligned(bytes, kSimdAlignment);
            if (data_) {
                allocation_type_ = AllocationType::kNuma;
                spdlog::debug("Bunch allocated {} bytes using NUMA allocator", bytes);
            } else {
                // Fall back to Highway allocation
                data_ = hwy::AllocateAlignedBytes(bytes, nullptr, nullptr);
                allocation_type_ = AllocationType::kHighway;
            }
        } else {
            // Use Highway aligned allocation for smaller arrays
            data_ = hwy::AllocateAlignedBytes(bytes, nullptr, nullptr);
            allocation_type_ = AllocationType::kHighway;
        }

        if (!data_) {
            allocation_type_ = AllocationType::kNone;
            return ErrorCode::kAllocationFailed;
        }

        owns_data_ = true;
        std::memset(data_, 0, bytes);
        return {};
    }

    // Set data from external source (copies)
    template <typename T>
    Result<void> setData(const T* src, size_t count) {
        if (count != count_) {
            return ErrorCode::kShapeMismatch;
        }

        std::memcpy(data_, src, count * sizeof(T));
        return {};
    }

    // Accessors
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t count() const { return count_; }
    ScalarType dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    TypeDesc type() const { return TypeDesc(dtype_, shape_); }

    // IR for lazy evaluation
    void setIR(ir::IRModule* module, ir::ValueId value) {
        ir_module_ = module;
        ir_value_ = value;
        is_lazy_ = true;
    }

    bool isLazy() const { return is_lazy_; }

    // Get IR module and value for lazy ops that need to chain
    ir::IRModule* irModule() { return ir_module_; }
    ir::ValueId irValue() const { return ir_value_; }

    // Tracing state (for @flow.kernel)
    void setTracing(ir::IRModule* module, ir::ValueId value) {
        tracing_module_ = module;
        tracing_value_ = value;
        is_tracing_ = true;
    }

    bool isTracing() const { return is_tracing_; }
    ir::IRModule* tracingModule() { return tracing_module_; }
    ir::ValueId tracingValue() const { return tracing_value_; }

    // Evaluate lazy expression by executing IR
    Result<void> materialize() {
        if (!is_lazy_) {
            return {};  // Already materialized
        }

        spdlog::debug("Materializing lazy Bunch (value %{})", ir_value_.id);

        if (!ir_module_) {
            spdlog::error("Lazy Bunch has no IR module");
            is_lazy_ = false;
            return Error(ErrorCode::kRuntimeError, "Lazy Bunch has no IR module");
        }

        if (!ir_value_.isValid()) {
            spdlog::error("Lazy Bunch has invalid IR value");
            is_lazy_ = false;
            return Error(ErrorCode::kRuntimeError, "Lazy Bunch has invalid IR value");
        }

        // Ensure we have storage allocated
        if (!data_) {
            // Get output type from IR
            const ir::IRNode* output_node = ir_module_->builder().getNode(ir_value_);
            if (!output_node) {
                is_lazy_ = false;
                return Error(ErrorCode::kRuntimeError, "IR output node not found");
            }

            // Determine size from type shape or default
            size_t size = 1;
            const auto& shape = output_node->type().shape();
            if (shape.rank() > 0) {
                for (size_t i = 0; i < shape.rank(); ++i) {
                    size *= shape[i];
                }
            }

            // Allocate storage
            auto alloc_result = allocate(size, output_node->type().scalarType());
            if (!alloc_result) {
                is_lazy_ = false;
                return alloc_result;
            }
        }

        // Execute the IR using the interpreter
        auto result = ir::executeIR(ir_module_->builder(), ir_value_, data_, count_, dtype_);

        is_lazy_ = false;  // Mark as materialized regardless of result

        if (!result) {
            spdlog::error("IR execution failed: {}", result.error().message());
            return result;
        }

        spdlog::debug("Lazy Bunch materialized successfully ({} elements)", count_);
        return {};
    }

  private:
    void* data_ = nullptr;
    size_t count_ = 0;
    size_t allocated_bytes_ = 0;
    ScalarType dtype_ = ScalarType::kUnknown;
    Shape shape_;
    bool owns_data_ = false;
    AllocationType allocation_type_ = AllocationType::kNone;

    // Lazy evaluation state
    bool is_lazy_ = false;
    ir::IRModule* ir_module_ = nullptr;
    ir::ValueId ir_value_;

    // Tracing state (for @flow.kernel)
    bool is_tracing_ = false;
    ir::IRModule* tracing_module_ = nullptr;
    ir::ValueId tracing_value_;
};

// =============================================================================
// Bunch Implementation
// =============================================================================

Bunch::Bunch() : impl_(std::make_shared<BunchImpl>()) {}

Bunch::~Bunch() = default;

Bunch::Bunch(const Bunch& other) = default;
Bunch& Bunch::operator=(const Bunch& other) = default;
Bunch::Bunch(Bunch&& other) noexcept = default;
Bunch& Bunch::operator=(Bunch&& other) noexcept = default;

Bunch::Bunch(std::shared_ptr<BunchImpl> impl) : impl_(std::move(impl)) {}

// Factory methods
Result<Bunch> Bunch::fromData(const float* data, size_t count) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kFloat32));
    BUD_TRY(impl->setData(data, count));
    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::fromData(const double* data, size_t count) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kFloat64));
    BUD_TRY(impl->setData(data, count));
    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::fromData(const int32_t* data, size_t count) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kInt32));
    BUD_TRY(impl->setData(data, count));
    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::zeros(size_t count, ScalarType type) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, type));
    // Already zeroed by allocate
    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::ones(size_t count, ScalarType type) {
    if (type != ScalarType::kFloat32) {
        return ErrorCode::kUnsupportedType;
    }

    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, type));

    // Use ISA-aware SIMD dispatch
    auto result = simd::DispatchFill(impl->data(), 1.0f, count, type);
    if (!result) {
        spdlog::error("Bunch::ones dispatch failed");
        return result.error();
    }

    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::fill(size_t count, float value) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kFloat32));

    // Use ISA-aware SIMD dispatch
    auto result = simd::DispatchFill(impl->data(), value, count, ScalarType::kFloat32);
    if (!result) {
        spdlog::error("Bunch::fill dispatch failed");
        return result.error();
    }

    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::arange(size_t count, float start, float step) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kFloat32));

    // Use ISA-aware SIMD dispatch
    auto result = simd::DispatchArange(impl->data(), start, step, count, ScalarType::kFloat32);
    if (!result) {
        spdlog::error("Bunch::arange dispatch failed");
        return result.error();
    }

    return Bunch(std::move(impl));
}

// Properties
size_t Bunch::size() const {
    return impl_->count();
}
ScalarType Bunch::dtype() const {
    return impl_->dtype();
}
const Shape& Bunch::shape() const {
    return impl_->shape();
}
TypeDesc Bunch::type() const {
    return impl_->type();
}

const void* Bunch::data() const {
    // Materialize if lazy
    if (impl_->isLazy()) {
        auto result = const_cast<BunchImpl*>(impl_.get())->materialize();
        if (!result) {
            spdlog::error("Failed to materialize Bunch: {}", result.error().toString());
            return nullptr;
        }
    }
    return impl_->data();
}

void* Bunch::mutableData() {
    if (impl_->isLazy()) {
        auto result = impl_->materialize();
        if (!result) {
            return nullptr;
        }
    }
    return impl_->data();
}

Result<void> Bunch::copyTo(float* dest, size_t count) const {
    if (count != size()) {
        return ErrorCode::kShapeMismatch;
    }
    if (dtype() != ScalarType::kFloat32) {
        return ErrorCode::kTypeMismatch;
    }

    std::memcpy(dest, data(), count * sizeof(float));
    return {};
}

bool Bunch::isValid() const {
    return impl_ && (impl_->data() != nullptr || impl_->isLazy() || impl_->isTracing());
}

// =============================================================================
// Tracing Support
// =============================================================================

bool Bunch::isTracing() const {
    return impl_ && impl_->isTracing();
}

ir::ValueId Bunch::tracingValueId() const {
    if (impl_ && impl_->isTracing()) {
        return impl_->tracingValue();
    }
    return ir::ValueId::invalid();
}

void Bunch::setTracingState(ir::IRModule* module, ir::ValueId value_id) {
    if (impl_) {
        impl_->setTracing(module, value_id);
    }
}

Bunch Bunch::fromTracingValue(ir::IRModule* module, ir::ValueId value_id, size_t count,
                              ScalarType dtype) {
    auto impl = std::make_shared<BunchImpl>();
    // Allocate storage (will be filled when materialized)
    auto alloc_result = impl->allocate(count, dtype);
    if (!alloc_result) {
        spdlog::error("fromTracingValue: allocation failed");
        return Bunch();
    }
    // Set tracing state
    impl->setTracing(module, value_id);
    return Bunch(std::move(impl));
}

// Helper function to get/create ValueId for a Bunch in a tracing context
static ir::ValueId getOrCreateValueId(const Bunch& bunch, ir::IRModule* module) {
    // If already tracing, return the existing ValueId
    if (bunch.isTracing()) {
        return bunch.tracingValueId();
    }

    // Otherwise, this is a concrete value that needs to be added as a constant
    if (bunch.dtype() == ScalarType::kFloat32) {
        const float* data = static_cast<const float*>(bunch.data());
        std::vector<float> values(data, data + bunch.size());
        return module->builder().constantVector(values);
    }

    // For other types, create a float constant (with conversion if needed)
    spdlog::warn("getOrCreateValueId: non-float32 type, using zeros as placeholder");
    std::vector<float> zeros(bunch.size(), 0.0f);
    return module->builder().constantVector(zeros);
}

// =============================================================================
// Binary Arithmetic Operators (using Highway SIMD dispatch)
// =============================================================================

Bunch Bunch::operator+(const Bunch& other) const {
    if (size() != other.size()) {
        spdlog::error("Bunch::operator+ size mismatch: {} vs {}", size(), other.size());
        return *this;
    }
    if (dtype() != other.dtype()) {
        spdlog::error("Bunch::operator+ type mismatch");
        return *this;
    }

    // Check if either operand is tracing - if so, record to IR
    if (isTracing() || other.isTracing()) {
        // Get the tracing module from whichever operand is tracing
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module) {
            spdlog::error("Bunch::operator+ tracing but no IR module");
            return *this;
        }

        // Get or create ValueIds for both operands
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);

        // Record the add operation in IR
        ir::ValueId result_id = module->builder().add(lhs_id, rhs_id);

        spdlog::debug("Tracing operator+: {} + {} -> {}", lhs_id.id, rhs_id.id, result_id.id);

        // Return a tracer Bunch that references the result IR value
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    // Eager execution path
    auto result = Bunch::zeros(size(), dtype());
    if (!result) {
        spdlog::error("Bunch::operator+ allocation failed");
        return *this;
    }

    auto dispatch_result =
        simd::DispatchAdd(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::operator+ dispatch failed: {}", dispatch_result.error().toString());
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::operator-(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::operator- shape/type mismatch");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().sub(lhs_id, rhs_id);
        spdlog::debug("Tracing operator-: {} - {} -> {}", lhs_id.id, rhs_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchSub(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::operator- dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::operator*(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::operator* shape/type mismatch");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().mul(lhs_id, rhs_id);
        spdlog::debug("Tracing operator*: {} * {} -> {}", lhs_id.id, rhs_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchMul(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::operator* dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::operator/(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::operator/ shape/type mismatch");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().div(lhs_id, rhs_id);
        spdlog::debug("Tracing operator/: {} / {} -> {}", lhs_id.id, rhs_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchDiv(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::operator/ dispatch failed");
        return *this;
    }

    return std::move(*result);
}

// =============================================================================
// Scalar Arithmetic Operators
// =============================================================================

Bunch Bunch::operator+(float scalar) const {
    // Create a scalar bunch and use element-wise addition
    auto scalar_bunch = Bunch::fill(size(), scalar);
    if (!scalar_bunch)
        return *this;
    return *this + *scalar_bunch;
}

Bunch Bunch::operator-(float scalar) const {
    auto scalar_bunch = Bunch::fill(size(), scalar);
    if (!scalar_bunch)
        return *this;
    return *this - *scalar_bunch;
}

Bunch Bunch::operator*(float scalar) const {
    auto scalar_bunch = Bunch::fill(size(), scalar);
    if (!scalar_bunch)
        return *this;
    return *this * *scalar_bunch;
}

Bunch Bunch::operator/(float scalar) const {
    auto scalar_bunch = Bunch::fill(size(), scalar);
    if (!scalar_bunch)
        return *this;
    return *this / *scalar_bunch;
}

// =============================================================================
// Unary Operators (using Highway SIMD dispatch)
// =============================================================================

Bunch Bunch::operator-() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().neg(x_id);
        spdlog::debug("Tracing operator- (neg): -{} -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchNeg(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::operator- (unary) dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::abs() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().abs(x_id);
        spdlog::debug("Tracing abs: abs({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchAbs(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::abs dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::sqrt() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().sqrt(x_id);
        spdlog::debug("Tracing sqrt: sqrt({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchSqrt(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::sqrt dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::rsqrt() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().rsqrt(x_id);
        spdlog::debug("Tracing rsqrt: rsqrt({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    // rsqrt = 1 / sqrt
    auto sqrt_result = sqrt();
    if (!sqrt_result.isValid())
        return *this;

    // Use div with ones to compute 1/sqrt
    auto ones = Bunch::ones(size(), dtype());
    if (!ones)
        return *this;

    return *ones / sqrt_result;
}

Bunch Bunch::exp() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().exp(x_id);
        spdlog::debug("Tracing exp: exp({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchExp(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::exp dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::log() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().log(x_id);
        spdlog::debug("Tracing log: log({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchLog(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::log dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::sin() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().sin(x_id);
        spdlog::debug("Tracing sin: sin({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchSin(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::sin dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::cos() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().cos(x_id);
        spdlog::debug("Tracing cos: cos({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchCos(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::cos dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::tanh() const {
    // Tracing path
    if (isTracing()) {
        ir::IRModule* module = impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId x_id = getOrCreateValueId(*this, module);
        ir::ValueId result_id = module->builder().tanh(x_id);
        spdlog::debug("Tracing tanh: tanh({}) -> {}", x_id.id, result_id.id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchTanh(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::tanh dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::tan() const {
    // Tracing path - currently not supported in IR, fall through to direct execution
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchTan(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::tan dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::sigmoid() const {
    // Tracing path - currently not supported in IR, fall through to direct execution
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchSigmoid(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::sigmoid dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::floor() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchFloor(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::floor dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::ceil() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchCeil(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::ceil dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::round() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchRound(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::round dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::trunc() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchTrunc(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::trunc dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::isnan() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchIsNaN(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::isnan dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::isinf() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchIsInf(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::isinf dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::isfinite() const {
    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result = simd::DispatchIsFinite(result->mutableData(), data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::isfinite dispatch failed");
        return *this;
    }

    return std::move(*result);
}

// Reductions with null pointer safety - using SIMD dispatch
float Bunch::sum() const {
    if (dtype() != ScalarType::kFloat32) {
        spdlog::warn("Bunch::sum() called on non-float32 type");
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (size() == 0) {
        return 0.0f;  // Sum of empty array is 0
    }
    const void* ptr = data();
    if (!ptr) {
        spdlog::error("Bunch::sum() data is null (materialization may have failed)");
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Use ISA-aware SIMD dispatch (includes HWY_EMU128 scalar fallback)
    auto result = simd::DispatchReduceSum(ptr, size(), dtype());
    if (!result) {
        spdlog::error("Bunch::sum() dispatch failed: {}", result.error().toString());
        return std::numeric_limits<float>::quiet_NaN();
    }
    return *result;
}

float Bunch::max() const {
    if (dtype() != ScalarType::kFloat32) {
        spdlog::warn("Bunch::max() called on non-float32 type");
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (size() == 0) {
        return -std::numeric_limits<float>::infinity();  // Max of empty is -inf
    }
    const void* ptr = data();
    if (!ptr) {
        spdlog::error("Bunch::max() data is null");
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Use ISA-aware SIMD dispatch (includes HWY_EMU128 scalar fallback)
    auto result = simd::DispatchReduceMax(ptr, size(), dtype());
    if (!result) {
        spdlog::error("Bunch::max() dispatch failed: {}", result.error().toString());
        return std::numeric_limits<float>::quiet_NaN();
    }
    return *result;
}

float Bunch::min() const {
    if (dtype() != ScalarType::kFloat32) {
        spdlog::warn("Bunch::min() called on non-float32 type");
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (size() == 0) {
        return std::numeric_limits<float>::infinity();  // Min of empty is +inf
    }
    const void* ptr = data();
    if (!ptr) {
        spdlog::error("Bunch::min() data is null");
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Use ISA-aware SIMD dispatch (includes HWY_EMU128 scalar fallback)
    auto result = simd::DispatchReduceMin(ptr, size(), dtype());
    if (!result) {
        spdlog::error("Bunch::min() dispatch failed: {}", result.error().toString());
        return std::numeric_limits<float>::quiet_NaN();
    }
    return *result;
}

float Bunch::mean() const {
    if (size() == 0) {
        return std::numeric_limits<float>::quiet_NaN();  // Mean of empty is undefined
    }
    float s = sum();
    if (std::isnan(s)) {
        return s;  // Propagate NaN
    }
    return s / static_cast<float>(size());
}

float Bunch::dot(const Bunch& other) const {
    if (dtype() != ScalarType::kFloat32 || other.dtype() != ScalarType::kFloat32) {
        spdlog::warn("Bunch::dot() called on non-float32 types");
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (size() != other.size()) {
        spdlog::warn("Bunch::dot() size mismatch: {} vs {}", size(), other.size());
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (size() == 0) {
        return 0.0f;  // Dot product of empty arrays is 0
    }
    const float* a = static_cast<const float*>(data());
    const float* b = static_cast<const float*>(other.data());
    if (!a || !b) {
        spdlog::error("Bunch::dot() data is null");
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Use TiledExecutor for large arrays (cache-optimized)
    if (size() >= kTilingThreshold) {
        memory::CacheConfig cache_config = memory::CacheConfig::detect();
        memory::TiledExecutor tiled_executor(cache_config);
        float result = tiled_executor.dotProduct(a, b, size());
        spdlog::debug("Bunch::dot() used tiled execution for {} elements", size());
        return result;
    }

    // Scalar fallback for small arrays
    float total = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// Comparisons (produce float masks: 0.0f for false, 1.0f for true) - using SIMD dispatch
Bunch Bunch::eq(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::eq shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::eq only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().eq(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    // Use ISA-aware SIMD dispatch
    auto dispatch_result =
        simd::DispatchEq(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::eq dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::lt(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::lt shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::lt only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().lt(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchLt(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::lt dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::le(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::le shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::le only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().le(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchLe(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::le dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::gt(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::gt shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::gt only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().gt(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchGt(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::gt dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::ge(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::ge shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::ge only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().ge(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchGe(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::ge dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::ne(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::ne shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::ne only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().ne(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchNe(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::ne dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::minimum(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::minimum shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::minimum only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().min(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchMin(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::minimum dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::maximum(const Bunch& other) const {
    if (size() != other.size() || dtype() != other.dtype()) {
        spdlog::error("Bunch::maximum shape/type mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::maximum only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing()) {
        ir::IRModule* module = isTracing() ? impl_->tracingModule() : other.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId lhs_id = getOrCreateValueId(*this, module);
        ir::ValueId rhs_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().max(lhs_id, rhs_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    auto dispatch_result =
        simd::DispatchMax(result->mutableData(), data(), other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::maximum dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Bunch Bunch::where(const Bunch& mask, const Bunch& other) const {
    if (size() != other.size() || size() != mask.size()) {
        spdlog::error("Bunch::where size mismatch");
        return *this;
    }
    if (dtype() != ScalarType::kFloat32 || other.dtype() != ScalarType::kFloat32) {
        spdlog::error("Bunch::where only supports float32");
        return *this;
    }

    // Tracing path
    if (isTracing() || other.isTracing() || mask.isTracing()) {
        ir::IRModule* module = isTracing()         ? impl_->tracingModule()
                               : other.isTracing() ? other.impl_->tracingModule()
                                                   : mask.impl_->tracingModule();
        if (!module)
            return *this;
        ir::ValueId mask_id = getOrCreateValueId(mask, module);
        ir::ValueId a_id = getOrCreateValueId(*this, module);
        ir::ValueId b_id = getOrCreateValueId(other, module);
        ir::ValueId result_id = module->builder().select(mask_id, a_id, b_id);
        return Bunch::fromTracingValue(module, result_id, size(), dtype());
    }

    auto result = Bunch::zeros(size(), dtype());
    if (!result)
        return *this;

    // Use ISA-aware SIMD dispatch
    // DispatchWhere: mask selects from 'a' (this) when non-zero, else 'b' (other)
    auto dispatch_result = simd::DispatchWhere(result->mutableData(), mask.data(), data(),
                                               other.data(), size(), dtype());
    if (!dispatch_result) {
        spdlog::error("Bunch::where dispatch failed");
        return *this;
    }

    return std::move(*result);
}

Result<void> Bunch::eval() {
    return impl_->materialize();
}

std::string Bunch::toString() const {
    return fmt::format("Bunch(shape={}, dtype={})", shape().toString(), scalarTypeName(dtype()));
}

// =============================================================================
// Fused Operations (using Highway SIMD)
// =============================================================================

// Fused multiply-add: a * b + c
Bunch fma(const Bunch& a, const Bunch& b, const Bunch& c) {
    if (a.size() != b.size() || a.size() != c.size()) {
        spdlog::error("fma: size mismatch");
        return a;
    }
    if (a.dtype() != ScalarType::kFloat32 || b.dtype() != ScalarType::kFloat32 ||
        c.dtype() != ScalarType::kFloat32) {
        spdlog::error("fma: only float32 supported");
        return a;
    }

    auto result = Bunch::zeros(a.size(), a.dtype());
    if (!result)
        return a;

    // Use Highway MulAdd for FMA
    simd::MulAdd(static_cast<float*>(result->mutableData()), static_cast<const float*>(a.data()),
                 static_cast<const float*>(b.data()), static_cast<const float*>(c.data()),
                 a.size());

    return std::move(*result);
}

// Fused multiply-subtract: a * b - c
Bunch fms(const Bunch& a, const Bunch& b, const Bunch& c) {
    if (a.size() != b.size() || a.size() != c.size()) {
        spdlog::error("fms: size mismatch");
        return a;
    }
    if (a.dtype() != ScalarType::kFloat32 || b.dtype() != ScalarType::kFloat32 ||
        c.dtype() != ScalarType::kFloat32) {
        spdlog::error("fms: only float32 supported");
        return a;
    }

    auto result = Bunch::zeros(a.size(), a.dtype());
    if (!result)
        return a;

    // Use Highway MulSub
    simd::MulSub(static_cast<float*>(result->mutableData()), static_cast<const float*>(a.data()),
                 static_cast<const float*>(b.data()), static_cast<const float*>(c.data()),
                 a.size());

    return std::move(*result);
}

// Fused negative multiply-add: c - a * b
Bunch fnma(const Bunch& a, const Bunch& b, const Bunch& c) {
    if (a.size() != b.size() || a.size() != c.size()) {
        spdlog::error("fnma: size mismatch");
        return a;
    }
    if (a.dtype() != ScalarType::kFloat32 || b.dtype() != ScalarType::kFloat32 ||
        c.dtype() != ScalarType::kFloat32) {
        spdlog::error("fnma: only float32 supported");
        return a;
    }

    auto result = Bunch::zeros(a.size(), a.dtype());
    if (!result)
        return a;

    // Use Highway NegMulAdd
    simd::NegMulAdd(static_cast<float*>(result->mutableData()), static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()), static_cast<const float*>(c.data()),
                    a.size());

    return std::move(*result);
}

// Clamp values to [lo, hi] range
Bunch clamp(const Bunch& x, float lo, float hi) {
    if (x.dtype() != ScalarType::kFloat32) {
        spdlog::error("clamp: only float32 supported");
        return x;
    }

    auto result = Bunch::zeros(x.size(), x.dtype());
    if (!result)
        return x;

    // Use Highway Clamp
    simd::Clamp(static_cast<float*>(result->mutableData()), static_cast<const float*>(x.data()), lo,
                hi, x.size());

    return std::move(*result);
}

// Linear interpolation: a + t * (b - a) - using SIMD dispatch
Bunch lerp(const Bunch& a, const Bunch& b, float t) {
    if (a.size() != b.size()) {
        spdlog::error("lerp: size mismatch");
        return a;
    }
    if (a.dtype() != ScalarType::kFloat32 || b.dtype() != ScalarType::kFloat32) {
        spdlog::error("lerp: only float32 supported");
        return a;
    }

    auto result = Bunch::zeros(a.size(), a.dtype());
    if (!result)
        return a;

    // Use ISA-aware SIMD dispatch: lerp(a, b, t) = (1-t) * a + t * b
    auto dispatch_result =
        simd::DispatchLerp(result->mutableData(), a.data(), b.data(), t, a.size(), a.dtype());
    if (!dispatch_result) {
        spdlog::error("lerp dispatch failed");
        return a;
    }

    return std::move(*result);
}

// =============================================================================
// Lazy Evaluation Mode Control
// =============================================================================

namespace {
// Thread-local flag for lazy evaluation mode
thread_local bool g_lazy_mode = false;
}  // namespace

void setLazyMode(bool enabled) {
    g_lazy_mode = enabled;
    spdlog::debug("Lazy mode {}", enabled ? "enabled" : "disabled");
}

bool isLazyMode() {
    return g_lazy_mode;
}

}  // namespace bud
