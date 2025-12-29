// =============================================================================
// Bud Flow Lang - Bunch Implementation
// =============================================================================

#include "bud_flow_lang/bunch.h"

#include "bud_flow_lang/ir.h"

#include <hwy/aligned_allocator.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstring>

namespace bud {

// =============================================================================
// BunchImpl - Internal Implementation
// =============================================================================

class BunchImpl {
  public:
    BunchImpl() = default;

    ~BunchImpl() {
        if (data_ && owns_data_) {
            hwy::FreeAlignedBytes(data_, nullptr, nullptr);
        }
    }

    // Allocate storage
    Result<void> allocate(size_t count, ScalarType dtype) {
        dtype_ = dtype;
        count_ = count;
        shape_ = Shape::vector(count);

        size_t bytes = count * scalarTypeSize(dtype);
        // Highway handles SIMD alignment internally (HWY_ALIGNMENT)
        data_ = hwy::AllocateAlignedBytes(bytes, nullptr, nullptr);

        if (!data_) {
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

    // Evaluate lazy expression
    Result<void> materialize() {
        if (!is_lazy_) {
            return {};  // Already materialized
        }

        // TODO: Compile and execute IR
        spdlog::debug("Materializing lazy Bunch");

        is_lazy_ = false;
        return {};
    }

  private:
    void* data_ = nullptr;
    size_t count_ = 0;
    ScalarType dtype_ = ScalarType::kUnknown;
    Shape shape_;
    bool owns_data_ = false;

    // Lazy evaluation state
    bool is_lazy_ = false;
    ir::IRModule* ir_module_ = nullptr;
    ir::ValueId ir_value_;
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

    float* ptr = static_cast<float*>(impl->data());
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = 1.0f;
    }

    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::fill(size_t count, float value) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kFloat32));

    float* ptr = static_cast<float*>(impl->data());
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = value;
    }

    return Bunch(std::move(impl));
}

Result<Bunch> Bunch::arange(size_t count, float start, float step) {
    auto impl = std::make_shared<BunchImpl>();
    BUD_TRY(impl->allocate(count, ScalarType::kFloat32));

    float* ptr = static_cast<float*>(impl->data());
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = start + static_cast<float>(i) * step;
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
    return impl_ && (impl_->data() != nullptr || impl_->isLazy());
}

// Arithmetic operators (placeholder - would create lazy IR)
Bunch Bunch::operator+(const Bunch& other) const {
    // TODO: Create lazy IR expression
    spdlog::debug("Bunch::operator+ (lazy)");
    return *this;  // Placeholder
}

Bunch Bunch::operator-(const Bunch& other) const {
    spdlog::debug("Bunch::operator- (lazy)");
    return *this;
}

Bunch Bunch::operator*(const Bunch& other) const {
    spdlog::debug("Bunch::operator* (lazy)");
    return *this;
}

Bunch Bunch::operator/(const Bunch& other) const {
    spdlog::debug("Bunch::operator/ (lazy)");
    return *this;
}

Bunch Bunch::operator+(float scalar) const {
    spdlog::debug("Bunch::operator+(scalar) (lazy)");
    return *this;
}

Bunch Bunch::operator-(float scalar) const {
    return *this;
}

Bunch Bunch::operator*(float scalar) const {
    return *this;
}

Bunch Bunch::operator/(float scalar) const {
    return *this;
}

Bunch Bunch::operator-() const {
    return *this;
}

Bunch Bunch::abs() const {
    return *this;
}
Bunch Bunch::sqrt() const {
    return *this;
}
Bunch Bunch::rsqrt() const {
    return *this;
}
Bunch Bunch::exp() const {
    return *this;
}
Bunch Bunch::log() const {
    return *this;
}
Bunch Bunch::sin() const {
    return *this;
}
Bunch Bunch::cos() const {
    return *this;
}
Bunch Bunch::tanh() const {
    return *this;
}

// Reductions (placeholder)
float Bunch::sum() const {
    if (dtype() != ScalarType::kFloat32)
        return 0.0f;
    const float* ptr = static_cast<const float*>(data());
    float total = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        total += ptr[i];
    }
    return total;
}

float Bunch::max() const {
    if (dtype() != ScalarType::kFloat32 || size() == 0)
        return 0.0f;
    const float* ptr = static_cast<const float*>(data());
    float m = ptr[0];
    for (size_t i = 1; i < size(); ++i) {
        if (ptr[i] > m)
            m = ptr[i];
    }
    return m;
}

float Bunch::min() const {
    if (dtype() != ScalarType::kFloat32 || size() == 0)
        return 0.0f;
    const float* ptr = static_cast<const float*>(data());
    float m = ptr[0];
    for (size_t i = 1; i < size(); ++i) {
        if (ptr[i] < m)
            m = ptr[i];
    }
    return m;
}

float Bunch::mean() const {
    if (size() == 0)
        return 0.0f;
    return sum() / static_cast<float>(size());
}

float Bunch::dot(const Bunch& other) const {
    if (size() != other.size() || dtype() != ScalarType::kFloat32)
        return 0.0f;
    const float* a = static_cast<const float*>(data());
    const float* b = static_cast<const float*>(other.data());
    float total = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        total += a[i] * b[i];
    }
    return total;
}

// Comparisons
Bunch Bunch::eq(const Bunch& other) const {
    return *this;
}
Bunch Bunch::lt(const Bunch& other) const {
    return *this;
}
Bunch Bunch::le(const Bunch& other) const {
    return *this;
}
Bunch Bunch::gt(const Bunch& other) const {
    return *this;
}
Bunch Bunch::ge(const Bunch& other) const {
    return *this;
}

Bunch Bunch::where(const Bunch& mask, const Bunch& other) const {
    return *this;
}

Result<void> Bunch::eval() {
    return impl_->materialize();
}

std::string Bunch::toString() const {
    return fmt::format("Bunch(shape={}, dtype={})", shape().toString(), scalarTypeName(dtype()));
}

// =============================================================================
// Fused Operations
// =============================================================================

Bunch fma(const Bunch& a, const Bunch& b, const Bunch& c) {
    spdlog::debug("fma (lazy)");
    return a;  // Placeholder
}

Bunch fms(const Bunch& a, const Bunch& b, const Bunch& c) {
    return a;
}

Bunch fnma(const Bunch& a, const Bunch& b, const Bunch& c) {
    return a;
}

Bunch clamp(const Bunch& x, float lo, float hi) {
    return x;
}

Bunch lerp(const Bunch& a, const Bunch& b, float t) {
    return a;
}

}  // namespace bud
