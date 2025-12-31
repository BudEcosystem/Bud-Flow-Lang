// =============================================================================
// Bud Flow Lang - Expert Tier Highway Type Wrappers
// =============================================================================
//
// This module provides Python-accessible wrappers for Highway SIMD operations.
// These are runtime wrappers that use Highway's dynamic dispatch internally.
//
// For Python bindings, we use type-erased wrappers that work at runtime,
// rather than exposing Highway's compile-time template types directly.
//
// =============================================================================

#pragma once

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/type_system.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bud {
namespace highway {

// =============================================================================
// SimdInfo - Runtime SIMD information
// =============================================================================

// Get the number of float32 lanes available on this CPU
[[nodiscard]] size_t getSimdLanesF32();

// Get the number of float64 lanes available on this CPU
[[nodiscard]] size_t getSimdLanesF64();

// Get the number of int32 lanes available on this CPU
[[nodiscard]] size_t getSimdLanesI32();

// Get a string describing the current SIMD target
[[nodiscard]] const char* getSimdTarget();

// =============================================================================
// VecBuffer - Buffer for holding SIMD vector data
// =============================================================================
//
// This is a simple wrapper around aligned memory for holding SIMD data.
// It's used to pass vector data between Python and Highway operations.
//

template <typename T>
class VecBuffer {
  public:
    // Create a buffer with given number of lanes
    explicit VecBuffer(size_t lanes);
    ~VecBuffer();

    // Non-copyable, movable
    VecBuffer(const VecBuffer&) = delete;
    VecBuffer& operator=(const VecBuffer&) = delete;
    VecBuffer(VecBuffer&&) noexcept;
    VecBuffer& operator=(VecBuffer&&) noexcept;

    // Access
    [[nodiscard]] T* data() { return data_; }
    [[nodiscard]] const T* data() const { return data_; }
    [[nodiscard]] size_t lanes() const { return lanes_; }

    // Element access
    [[nodiscard]] T& operator[](size_t i) { return data_[i]; }
    [[nodiscard]] const T& operator[](size_t i) const { return data_[i]; }

  private:
    T* data_ = nullptr;
    size_t lanes_ = 0;
};

// =============================================================================
// TagInfo - Runtime tag information (replacement for ScalableTag)
// =============================================================================

template <typename T>
struct TagInfo {
    // Get number of lanes for this type on current CPU
    [[nodiscard]] static size_t lanes();

    // Get scalar type
    [[nodiscard]] static ScalarType dtype();

    // Get type name
    [[nodiscard]] static const char* typeName();
};

// Template specializations
template <>
size_t TagInfo<float>::lanes();
template <>
size_t TagInfo<double>::lanes();
template <>
size_t TagInfo<int32_t>::lanes();
template <>
size_t TagInfo<int64_t>::lanes();
template <>
size_t TagInfo<uint8_t>::lanes();

template <>
ScalarType TagInfo<float>::dtype();
template <>
ScalarType TagInfo<double>::dtype();
template <>
ScalarType TagInfo<int32_t>::dtype();
template <>
ScalarType TagInfo<int64_t>::dtype();
template <>
ScalarType TagInfo<uint8_t>::dtype();

template <>
const char* TagInfo<float>::typeName();
template <>
const char* TagInfo<double>::typeName();
template <>
const char* TagInfo<int32_t>::typeName();
template <>
const char* TagInfo<int64_t>::typeName();
template <>
const char* TagInfo<uint8_t>::typeName();

// =============================================================================
// Highway Operations - Runtime wrappers for Highway SIMD operations
// =============================================================================

namespace ops {

// Arithmetic (element-wise, operating on arrays)
void Add(float* out, const float* a, const float* b, size_t count);
void Sub(float* out, const float* a, const float* b, size_t count);
void Mul(float* out, const float* a, const float* b, size_t count);
void Div(float* out, const float* a, const float* b, size_t count);
void Neg(float* out, const float* a, size_t count);
void Abs(float* out, const float* a, size_t count);

// FMA
void MulAdd(float* out, const float* a, const float* b, const float* c, size_t count);
void MulSub(float* out, const float* a, const float* b, const float* c, size_t count);

// Math
void Sqrt(float* out, const float* a, size_t count);
void Rsqrt(float* out, const float* a, size_t count);
void Exp(float* out, const float* a, size_t count);
void Log(float* out, const float* a, size_t count);
void Sin(float* out, const float* a, size_t count);
void Cos(float* out, const float* a, size_t count);
void Tanh(float* out, const float* a, size_t count);

// MinMax
void Min(float* out, const float* a, const float* b, size_t count);
void Max(float* out, const float* a, const float* b, size_t count);
void Clamp(float* out, const float* a, float lo, float hi, size_t count);

// Reductions
float ReduceSum(const float* a, size_t count);
float ReduceMin(const float* a, size_t count);
float ReduceMax(const float* a, size_t count);
float DotProduct(const float* a, const float* b, size_t count);

// Comparison (output is uint8_t mask: 0 or 0xFF)
void Eq(uint8_t* out, const float* a, const float* b, size_t count);
void Lt(uint8_t* out, const float* a, const float* b, size_t count);
void Le(uint8_t* out, const float* a, const float* b, size_t count);
void Gt(uint8_t* out, const float* a, const float* b, size_t count);
void Ge(uint8_t* out, const float* a, const float* b, size_t count);

// Masked operations
void MaskedStore(float* out, const float* values, const uint8_t* mask, size_t count);
void MaskedLoad(float* out, const float* src, const uint8_t* mask, size_t count);

// Select (mask ? a : b)
void Select(float* out, const uint8_t* mask, const float* a, const float* b, size_t count);

// Shuffle
void Reverse(float* out, const float* a, size_t count);

}  // namespace ops

// =============================================================================
// Type aliases for convenience
// =============================================================================

using TagF32 = TagInfo<float>;
using TagF64 = TagInfo<double>;
using TagI32 = TagInfo<int32_t>;
using TagI64 = TagInfo<int64_t>;
using TagU8 = TagInfo<uint8_t>;

}  // namespace highway
}  // namespace bud
