#pragma once

// =============================================================================
// Bud Flow Lang - Bunch Abstraction
// =============================================================================
//
// The core abstraction that hides SIMD complexity from users.
// A "Bunch" represents a collection of values that can be processed
// in parallel using SIMD operations.
//
// Design goals:
// - Zero-overhead abstraction (compiles to optimal SIMD)
// - Length-agnostic (works with any SIMD width)
// - Safe by default (bounds checking in debug)
//

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/type_system.h"

#include <cstddef>
#include <memory>
#include <span>

namespace bud {

// Forward declarations
class BunchImpl;
class MemoryPool;

// =============================================================================
// Bunch - User-Facing Handle
// =============================================================================

class Bunch {
public:
    // Construction
    Bunch();
    ~Bunch();

    Bunch(const Bunch& other);
    Bunch& operator=(const Bunch& other);
    Bunch(Bunch&& other) noexcept;
    Bunch& operator=(Bunch&& other) noexcept;

    // Factory methods
    static Result<Bunch> fromData(const float* data, size_t count);
    static Result<Bunch> fromData(const double* data, size_t count);
    static Result<Bunch> fromData(const int32_t* data, size_t count);

    static Result<Bunch> zeros(size_t count, ScalarType type = ScalarType::kFloat32);
    static Result<Bunch> ones(size_t count, ScalarType type = ScalarType::kFloat32);
    static Result<Bunch> fill(size_t count, float value);
    static Result<Bunch> arange(size_t count, float start = 0.0f, float step = 1.0f);

    // Properties
    [[nodiscard]] size_t size() const;
    [[nodiscard]] ScalarType dtype() const;
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] TypeDesc type() const;
    [[nodiscard]] bool empty() const { return size() == 0; }

    // Data access (for advanced users)
    [[nodiscard]] const void* data() const;
    [[nodiscard]] void* mutableData();

    // Type-safe data access
    template <typename T>
    [[nodiscard]] std::span<const T> as() const {
        static_assert(kIsSimdScalar<T>, "T must be a SIMD-compatible scalar type");
        return std::span<const T>(static_cast<const T*>(data()), size());
    }

    template <typename T>
    [[nodiscard]] std::span<T> asMutable() {
        static_assert(kIsSimdScalar<T>, "T must be a SIMD-compatible scalar type");
        return std::span<T>(static_cast<T*>(mutableData()), size());
    }

    // Copy data out
    Result<void> copyTo(float* dest, size_t count) const;

    // Validation
    [[nodiscard]] bool isValid() const;

    // Arithmetic operators (lazy - returns IR expression)
    Bunch operator+(const Bunch& other) const;
    Bunch operator-(const Bunch& other) const;
    Bunch operator*(const Bunch& other) const;
    Bunch operator/(const Bunch& other) const;

    // Scalar operations
    Bunch operator+(float scalar) const;
    Bunch operator-(float scalar) const;
    Bunch operator*(float scalar) const;
    Bunch operator/(float scalar) const;

    // Unary operations
    Bunch operator-() const;
    [[nodiscard]] Bunch abs() const;
    [[nodiscard]] Bunch sqrt() const;
    [[nodiscard]] Bunch rsqrt() const;

    // Transcendentals
    [[nodiscard]] Bunch exp() const;
    [[nodiscard]] Bunch log() const;
    [[nodiscard]] Bunch sin() const;
    [[nodiscard]] Bunch cos() const;
    [[nodiscard]] Bunch tanh() const;

    // Reductions
    [[nodiscard]] float sum() const;
    [[nodiscard]] float max() const;
    [[nodiscard]] float min() const;
    [[nodiscard]] float mean() const;
    [[nodiscard]] float dot(const Bunch& other) const;

    // Comparison (returns mask Bunch)
    [[nodiscard]] Bunch eq(const Bunch& other) const;
    [[nodiscard]] Bunch lt(const Bunch& other) const;
    [[nodiscard]] Bunch le(const Bunch& other) const;
    [[nodiscard]] Bunch gt(const Bunch& other) const;
    [[nodiscard]] Bunch ge(const Bunch& other) const;

    // Masked operations
    [[nodiscard]] Bunch where(const Bunch& mask, const Bunch& other) const;

    // Force evaluation (execute lazy IR)
    Result<void> eval();

    // Debug
    [[nodiscard]] std::string toString() const;

private:
    explicit Bunch(std::shared_ptr<BunchImpl> impl);
    friend class BunchImpl;
    friend class Runtime;

    std::shared_ptr<BunchImpl> impl_;
};

// =============================================================================
// Free Function Operators
// =============================================================================

inline Bunch operator+(float scalar, const Bunch& bunch) {
    return bunch + scalar;
}

inline Bunch operator*(float scalar, const Bunch& bunch) {
    return bunch * scalar;
}

// =============================================================================
// Free Function Math Operations
// =============================================================================

inline Bunch abs(const Bunch& x) { return x.abs(); }
inline Bunch sqrt(const Bunch& x) { return x.sqrt(); }
inline Bunch rsqrt(const Bunch& x) { return x.rsqrt(); }
inline Bunch exp(const Bunch& x) { return x.exp(); }
inline Bunch log(const Bunch& x) { return x.log(); }
inline Bunch sin(const Bunch& x) { return x.sin(); }
inline Bunch cos(const Bunch& x) { return x.cos(); }
inline Bunch tanh(const Bunch& x) { return x.tanh(); }

inline float sum(const Bunch& x) { return x.sum(); }
inline float max(const Bunch& x) { return x.max(); }
inline float min(const Bunch& x) { return x.min(); }
inline float mean(const Bunch& x) { return x.mean(); }
inline float dot(const Bunch& a, const Bunch& b) { return a.dot(b); }

// =============================================================================
// Fused Operations
// =============================================================================

// a * b + c (fused multiply-add)
Bunch fma(const Bunch& a, const Bunch& b, const Bunch& c);

// a * b - c
Bunch fms(const Bunch& a, const Bunch& b, const Bunch& c);

// -(a * b) + c
Bunch fnma(const Bunch& a, const Bunch& b, const Bunch& c);

// Clamp to range
Bunch clamp(const Bunch& x, float lo, float hi);

// Linear interpolation: a + t * (b - a)
Bunch lerp(const Bunch& a, const Bunch& b, float t);

}  // namespace bud
