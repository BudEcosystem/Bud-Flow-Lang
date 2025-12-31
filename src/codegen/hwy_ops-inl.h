// =============================================================================
// Bud Flow Lang - Highway SIMD Operations Implementation (Inline Header)
// =============================================================================
//
// This file is included multiple times with different HWY_TARGET values to
// generate code for each supported SIMD instruction set. DO NOT include this
// file directly - include hwy_ops.h instead.
//
// =============================================================================

// Highway toggle guard pattern - allows this file to be included multiple times
// foreach_target.h toggles HWY_TARGET_TOGGLE, we match that with our guard
#if defined(HIGHWAY_HWY_BUD_OPS_INL_H_) == defined(HWY_TARGET_TOGGLE)
    #ifdef HIGHWAY_HWY_BUD_OPS_INL_H_
        #undef HIGHWAY_HWY_BUD_OPS_INL_H_
    #else
        #define HIGHWAY_HWY_BUD_OPS_INL_H_
    #endif

    #include <hwy/contrib/math/math-inl.h>
    #include <hwy/highway.h>

    #include <algorithm>
    #include <cmath>
    #include <cstddef>
    #include <cstdint>
    #include <cstring>
    #include <limits>

// Highway requires functions used with HWY_EXPORT to be in the hwy namespace
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// =============================================================================
// Binary Operations Template
// =============================================================================

// Generic binary operation template for element-wise operations
template <class D, class Op>
HWY_ATTR void BinaryOpImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out,
                           const typename hn::TFromD<D>* HWY_RESTRICT a,
                           const typename hn::TFromD<D>* HWY_RESTRICT b, size_t count, Op op) {
    const size_t N = hn::Lanes(d);

    size_t i = 0;

    // Process full vectors
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = op(d, va, vb);
        hn::StoreU(vout, d, out + i);
    }

    // Handle remainder with partial vector load/store
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vb = hn::LoadN(d, b + i, remaining);
        const auto vout = op(d, va, vb);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Unary Operations Template
// =============================================================================

template <class D, class Op>
HWY_ATTR void UnaryOpImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out,
                          const typename hn::TFromD<D>* HWY_RESTRICT a, size_t count, Op op) {
    const size_t N = hn::Lanes(d);

    size_t i = 0;

    // Process full vectors
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = op(d, va);
        hn::StoreU(vout, d, out + i);
    }

    // Handle remainder
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = op(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Ternary Operations Template (for FMA)
// =============================================================================

template <class D, class Op>
HWY_ATTR void TernaryOpImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out,
                            const typename hn::TFromD<D>* HWY_RESTRICT a,
                            const typename hn::TFromD<D>* HWY_RESTRICT b,
                            const typename hn::TFromD<D>* HWY_RESTRICT c, size_t count, Op op) {
    const size_t N = hn::Lanes(d);

    size_t i = 0;

    // Process full vectors
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = op(d, va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }

    // Handle remainder
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vb = hn::LoadN(d, b + i, remaining);
        const auto vc = hn::LoadN(d, c + i, remaining);
        const auto vout = op(d, va, vb, vc);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Operation Functors
// =============================================================================

// Arithmetic operations
struct AddOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Add(a, b);
    }
};

struct SubOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Sub(a, b);
    }
};

struct MulOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Mul(a, b);
    }
};

struct DivOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Div(a, b);
    }
};

struct NegOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Neg(a);
    }
};

struct AbsOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Abs(a);
    }
};

// MinMax operations
struct MinOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Min(a, b);
    }
};

struct MaxOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Max(a, b);
    }
};

// Math operations
struct SqrtOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Sqrt(a);
    }
};

// FMA operations
struct MulAddOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b, V c) const {
        return hn::MulAdd(a, b, c);
    }
};

struct MulSubOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b, V c) const {
        return hn::MulSub(a, b, c);
    }
};

struct NegMulAddOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b, V c) const {
        return hn::NegMulAdd(a, b, c);
    }
};

// Bitwise operations
struct AndOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::And(a, b);
    }
};

struct OrOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Or(a, b);
    }
};

struct XorOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a, V b) const {
        return hn::Xor(a, b);
    }
};

struct NotOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Not(a);
    }
};

// Comparison operations
struct EqOp {
    template <class D, class V>
    HWY_INLINE auto operator()(D d, V a, V b) const {
        return hn::VecFromMask(d, hn::Eq(a, b));
    }
};

struct NeOp {
    template <class D, class V>
    HWY_INLINE auto operator()(D d, V a, V b) const {
        return hn::VecFromMask(d, hn::Ne(a, b));
    }
};

struct LtOp {
    template <class D, class V>
    HWY_INLINE auto operator()(D d, V a, V b) const {
        return hn::VecFromMask(d, hn::Lt(a, b));
    }
};

struct LeOp {
    template <class D, class V>
    HWY_INLINE auto operator()(D d, V a, V b) const {
        return hn::VecFromMask(d, hn::Le(a, b));
    }
};

struct GtOp {
    template <class D, class V>
    HWY_INLINE auto operator()(D d, V a, V b) const {
        return hn::VecFromMask(d, hn::Gt(a, b));
    }
};

struct GeOp {
    template <class D, class V>
    HWY_INLINE auto operator()(D d, V a, V b) const {
        return hn::VecFromMask(d, hn::Ge(a, b));
    }
};

// Rounding operations
struct RoundOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Round(a);
    }
};

struct FloorOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Floor(a);
    }
};

struct CeilOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Ceil(a);
    }
};

struct TruncOp {
    template <class D, class V>
    HWY_INLINE V operator()(D, V a) const {
        return hn::Trunc(a);
    }
};

// =============================================================================
// Priority 1: Arithmetic Operations
// =============================================================================

HWY_ATTR void AddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void AddFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                         const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void AddInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void AddInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void SubFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void SubFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                         const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void SubInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void SubInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void MulFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    BinaryOpImpl(d, out, a, b, count, MulOp{});
}

HWY_ATTR void MulFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                         const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    BinaryOpImpl(d, out, a, b, count, MulOp{});
}

HWY_ATTR void MulInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, MulOp{});
}

HWY_ATTR void MulInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, MulOp{});
}

HWY_ATTR void DivFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    BinaryOpImpl(d, out, a, b, count, DivOp{});
}

HWY_ATTR void DivFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                         const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    BinaryOpImpl(d, out, a, b, count, DivOp{});
}

HWY_ATTR void NegFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, NegOp{});
}

HWY_ATTR void NegFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, NegOp{});
}

HWY_ATTR void NegInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int32_t> d;
    UnaryOpImpl(d, out, a, count, NegOp{});
}

HWY_ATTR void NegInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int64_t> d;
    UnaryOpImpl(d, out, a, count, NegOp{});
}

HWY_ATTR void AbsFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, AbsOp{});
}

HWY_ATTR void AbsFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, AbsOp{});
}

HWY_ATTR void AbsInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int32_t> d;
    UnaryOpImpl(d, out, a, count, AbsOp{});
}

HWY_ATTR void AbsInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int64_t> d;
    UnaryOpImpl(d, out, a, count, AbsOp{});
}

// =============================================================================
// Priority 1: FMA Operations
// =============================================================================

HWY_ATTR void MulAddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                            const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                            size_t count) {
    const hn::ScalableTag<float> d;
    TernaryOpImpl(d, out, a, b, c, count, MulAddOp{});
}

HWY_ATTR void MulAddFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                            const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                            size_t count) {
    const hn::ScalableTag<double> d;
    TernaryOpImpl(d, out, a, b, c, count, MulAddOp{});
}

HWY_ATTR void MulSubFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                            const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                            size_t count) {
    const hn::ScalableTag<float> d;
    TernaryOpImpl(d, out, a, b, c, count, MulSubOp{});
}

HWY_ATTR void MulSubFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                            const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                            size_t count) {
    const hn::ScalableTag<double> d;
    TernaryOpImpl(d, out, a, b, c, count, MulSubOp{});
}

HWY_ATTR void NegMulAddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                               const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<float> d;
    TernaryOpImpl(d, out, a, b, c, count, NegMulAddOp{});
}

HWY_ATTR void NegMulAddFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                               const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<double> d;
    TernaryOpImpl(d, out, a, b, c, count, NegMulAddOp{});
}

// =============================================================================
// Priority 1: MinMax Operations
// =============================================================================

HWY_ATTR void MinFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MinFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                         const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MinInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MinInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MaxFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

HWY_ATTR void MaxFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                         const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

HWY_ATTR void MaxInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

HWY_ATTR void MaxInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

// Clamp: max(lo, min(hi, x))
HWY_ATTR void ClampFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, float lo, float hi,
                           size_t count) {
    const hn::ScalableTag<float> d;
    const auto vlo = hn::Set(d, lo);
    const auto vhi = hn::Set(d, hi);
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreU(clamped, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreN(clamped, d, out + i, remaining);
    }
}

HWY_ATTR void ClampFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, double lo,
                           double hi, size_t count) {
    const hn::ScalableTag<double> d;
    const auto vlo = hn::Set(d, lo);
    const auto vhi = hn::Set(d, hi);
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreU(clamped, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreN(clamped, d, out + i, remaining);
    }
}

HWY_ATTR void ClampInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, int32_t lo,
                         int32_t hi, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const auto vlo = hn::Set(d, lo);
    const auto vhi = hn::Set(d, hi);
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreU(clamped, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreN(clamped, d, out + i, remaining);
    }
}

HWY_ATTR void ClampInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, int64_t lo,
                         int64_t hi, size_t count) {
    const hn::ScalableTag<int64_t> d;
    const auto vlo = hn::Set(d, lo);
    const auto vhi = hn::Set(d, hi);
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreU(clamped, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto clamped = hn::Max(vlo, hn::Min(vhi, va));
        hn::StoreN(clamped, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 1: Math Operations
// =============================================================================

HWY_ATTR void SqrtFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, SqrtOp{});
}

HWY_ATTR void SqrtFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, SqrtOp{});
}

HWY_ATTR void RsqrtFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::ApproximateReciprocal(hn::Sqrt(va));
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::ApproximateReciprocal(hn::Sqrt(va));
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void RsqrtFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto one = hn::Set(d, 1.0);
        const auto vout = hn::Div(one, hn::Sqrt(va));
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto one = hn::Set(d, 1.0);
        const auto vout = hn::Div(one, hn::Sqrt(va));
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// Math functions using Highway contrib/math
HWY_ATTR void ExpFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Exp(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Exp(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void ExpFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Exp(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Exp(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void LogFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void LogFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void SinFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Sin(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Sin(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void SinFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Sin(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Sin(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void CosFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Cos(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Cos(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void CosFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Cos(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Cos(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void TanhFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Tanh(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Tanh(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void TanhFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Tanh(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Tanh(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 1: Comparison Operations
// These output to uint8_t: 0xFF for true, 0x00 for false
// =============================================================================

// Template for comparison operations with uint8_t output
template <typename T, typename CmpOp>
HWY_ATTR void CompareOpImpl(uint8_t* HWY_RESTRICT out, const T* HWY_RESTRICT a,
                            const T* HWY_RESTRICT b, size_t count, CmpOp cmp) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = cmp(a[i], b[i]) ? 0xFF : 0x00;
    }
}

HWY_ATTR void EqFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                        const float* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](float x, float y) { return x == y; });
}

HWY_ATTR void EqFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                        const double* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](double x, double y) { return x == y; });
}

HWY_ATTR void EqInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](int32_t x, int32_t y) { return x == y; });
}

HWY_ATTR void NeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                        const float* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](float x, float y) { return x != y; });
}

HWY_ATTR void NeFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                        const double* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](double x, double y) { return x != y; });
}

HWY_ATTR void NeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](int32_t x, int32_t y) { return x != y; });
}

HWY_ATTR void LtFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                        const float* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](float x, float y) { return x < y; });
}

HWY_ATTR void LtFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                        const double* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](double x, double y) { return x < y; });
}

HWY_ATTR void LtInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](int32_t x, int32_t y) { return x < y; });
}

HWY_ATTR void LeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                        const float* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](float x, float y) { return x <= y; });
}

HWY_ATTR void LeFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                        const double* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](double x, double y) { return x <= y; });
}

HWY_ATTR void LeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](int32_t x, int32_t y) { return x <= y; });
}

HWY_ATTR void GtFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                        const float* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](float x, float y) { return x > y; });
}

HWY_ATTR void GtFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                        const double* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](double x, double y) { return x > y; });
}

HWY_ATTR void GtInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](int32_t x, int32_t y) { return x > y; });
}

HWY_ATTR void GeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                        const float* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](float x, float y) { return x >= y; });
}

HWY_ATTR void GeFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                        const double* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](double x, double y) { return x >= y; });
}

HWY_ATTR void GeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    CompareOpImpl(out, a, b, count, [](int32_t x, int32_t y) { return x >= y; });
}

// =============================================================================
// Priority 1: Reduction Operations
// =============================================================================

HWY_ATTR float ReduceSumFloat32(const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        sum = hn::Add(sum, hn::LoadU(d, a + i));
    }

    float result = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        result += a[i];
    }
    return result;
}

HWY_ATTR double ReduceSumFloat64(const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        sum = hn::Add(sum, hn::LoadU(d, a + i));
    }

    double result = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        result += a[i];
    }
    return result;
}

HWY_ATTR int32_t ReduceSumInt32(const int32_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        sum = hn::Add(sum, hn::LoadU(d, a + i));
    }

    int32_t result = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        result += a[i];
    }
    return result;
}

HWY_ATTR int64_t ReduceSumInt64(const int64_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        sum = hn::Add(sum, hn::LoadU(d, a + i));
    }

    int64_t result = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        result += a[i];
    }
    return result;
}

// =============================================================================
// Optimized Reductions with Multiple Accumulators (2-4x faster)
// =============================================================================
// These functions use 4 independent accumulators to hide instruction latency.
// The key insight is that modern CPUs can execute multiple independent ADD/MUL
// operations per cycle, but data dependencies force sequential execution.
// By using 4 independent accumulators, we can keep the pipeline full.
//
// Reference: https://en.algorithmica.org/hpc/simd/reduction/
// =============================================================================

HWY_ATTR float ReduceSumFastFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Use 4 independent accumulators to hide latency
    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;

    // Main loop: process 4 vectors per iteration
    const size_t stride = 4 * N;
    for (; i + stride <= count; i += stride) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
        sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
        sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
    }

    // Handle remaining vectors (1-3 vectors)
    for (; i + N <= count; i += N) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
    }

    // Combine accumulators and reduce to scalar
    const auto sum01 = hn::Add(sum0, sum1);
    const auto sum23 = hn::Add(sum2, sum3);
    const auto total = hn::Add(sum01, sum23);
    float result = hn::ReduceSum(d, total);

    // Handle scalar remainder
    for (; i < count; ++i) {
        result += a[i];
    }

    return result;
}

HWY_ATTR double ReduceSumFastFloat64(const double* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
        sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
        sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
    }

    for (; i + N <= count; i += N) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    double result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i];
    }

    return result;
}

HWY_ATTR int32_t ReduceSumFastInt32(const int32_t* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0;

    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
        sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
        sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
    }

    for (; i + N <= count; i += N) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    int32_t result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i];
    }

    return result;
}

HWY_ATTR int64_t ReduceSumFastInt64(const int64_t* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0;

    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
        sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
        sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
    }

    for (; i + N <= count; i += N) {
        sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    int64_t result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i];
    }

    return result;
}

HWY_ATTR float ReduceMinFastFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<float>::infinity();

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto inf = hn::Set(d, std::numeric_limits<float>::infinity());

    auto min0 = inf, min1 = inf, min2 = inf, min3 = inf;

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        min0 = hn::Min(min0, hn::LoadU(d, a + i));
        min1 = hn::Min(min1, hn::LoadU(d, a + i + N));
        min2 = hn::Min(min2, hn::LoadU(d, a + i + 2 * N));
        min3 = hn::Min(min3, hn::LoadU(d, a + i + 3 * N));
    }

    for (; i + N <= count; i += N) {
        min0 = hn::Min(min0, hn::LoadU(d, a + i));
    }

    const auto total = hn::Min(hn::Min(min0, min1), hn::Min(min2, min3));
    float result = hn::ReduceMin(d, total);

    for (; i < count; ++i) {
        result = std::min(result, a[i]);
    }

    return result;
}

HWY_ATTR float ReduceMaxFastFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return -std::numeric_limits<float>::infinity();

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto neg_inf = hn::Set(d, -std::numeric_limits<float>::infinity());

    auto max0 = neg_inf, max1 = neg_inf, max2 = neg_inf, max3 = neg_inf;

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        max0 = hn::Max(max0, hn::LoadU(d, a + i));
        max1 = hn::Max(max1, hn::LoadU(d, a + i + N));
        max2 = hn::Max(max2, hn::LoadU(d, a + i + 2 * N));
        max3 = hn::Max(max3, hn::LoadU(d, a + i + 3 * N));
    }

    for (; i + N <= count; i += N) {
        max0 = hn::Max(max0, hn::LoadU(d, a + i));
    }

    const auto total = hn::Max(hn::Max(max0, max1), hn::Max(max2, max3));
    float result = hn::ReduceMax(d, total);

    for (; i < count; ++i) {
        result = std::max(result, a[i]);
    }

    return result;
}

HWY_ATTR float DotFastFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                              size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        const auto a0 = hn::LoadU(d, a + i);
        const auto a1 = hn::LoadU(d, a + i + N);
        const auto a2 = hn::LoadU(d, a + i + 2 * N);
        const auto a3 = hn::LoadU(d, a + i + 3 * N);

        const auto b0 = hn::LoadU(d, b + i);
        const auto b1 = hn::LoadU(d, b + i + N);
        const auto b2 = hn::LoadU(d, b + i + 2 * N);
        const auto b3 = hn::LoadU(d, b + i + 3 * N);

        sum0 = hn::MulAdd(a0, b0, sum0);
        sum1 = hn::MulAdd(a1, b1, sum1);
        sum2 = hn::MulAdd(a2, b2, sum2);
        sum3 = hn::MulAdd(a3, b3, sum3);
    }

    for (; i + N <= count; i += N) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

HWY_ATTR double DotFastFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               size_t count) {
    if (count == 0)
        return 0.0;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
        sum1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N), sum1);
        sum2 = hn::MulAdd(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N), sum2);
        sum3 = hn::MulAdd(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N), sum3);
    }

    for (; i + N <= count; i += N) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    double result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

HWY_ATTR float MeanFastFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0f;
    return ReduceSumFastFloat32(a, count) / static_cast<float>(count);
}

HWY_ATTR float SumOfSquaresFastFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        const auto v0 = hn::LoadU(d, a + i);
        const auto v1 = hn::LoadU(d, a + i + N);
        const auto v2 = hn::LoadU(d, a + i + 2 * N);
        const auto v3 = hn::LoadU(d, a + i + 3 * N);

        sum0 = hn::MulAdd(v0, v0, sum0);
        sum1 = hn::MulAdd(v1, v1, sum1);
        sum2 = hn::MulAdd(v2, v2, sum2);
        sum3 = hn::MulAdd(v3, v3, sum3);
    }

    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, a + i);
        sum0 = hn::MulAdd(v, v, sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i] * a[i];
    }

    return result;
}

HWY_ATTR float NormL2FastFloat32(const float* HWY_RESTRICT a, size_t count) {
    return std::sqrt(SumOfSquaresFastFloat32(a, count));
}

HWY_ATTR float VarianceFastFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count < 2)
        return 0.0f;

    // Two-pass algorithm for numerical stability
    // Pass 1: compute mean
    const float mean = MeanFastFloat32(a, count);

    // Pass 2: compute sum of squared differences using 4 accumulators
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vmean = hn::Set(d, mean);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    const size_t stride = 4 * N;

    for (; i + stride <= count; i += stride) {
        const auto diff0 = hn::Sub(hn::LoadU(d, a + i), vmean);
        const auto diff1 = hn::Sub(hn::LoadU(d, a + i + N), vmean);
        const auto diff2 = hn::Sub(hn::LoadU(d, a + i + 2 * N), vmean);
        const auto diff3 = hn::Sub(hn::LoadU(d, a + i + 3 * N), vmean);

        sum0 = hn::MulAdd(diff0, diff0, sum0);
        sum1 = hn::MulAdd(diff1, diff1, sum1);
        sum2 = hn::MulAdd(diff2, diff2, sum2);
        sum3 = hn::MulAdd(diff3, diff3, sum3);
    }

    for (; i + N <= count; i += N) {
        const auto diff = hn::Sub(hn::LoadU(d, a + i), vmean);
        sum0 = hn::MulAdd(diff, diff, sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        const float diff = a[i] - mean;
        result += diff * diff;
    }

    return result / static_cast<float>(count);
}

HWY_ATTR float ReduceMinFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<float>::infinity();
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto min_val = hn::Set(d, std::numeric_limits<float>::infinity());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        min_val = hn::Min(min_val, hn::LoadU(d, a + i));
    }

    float result = hn::ReduceMin(d, min_val);
    for (; i < count; ++i) {
        result = std::min(result, a[i]);
    }
    return result;
}

HWY_ATTR double ReduceMinFloat64(const double* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<double>::infinity();
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    auto min_val = hn::Set(d, std::numeric_limits<double>::infinity());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        min_val = hn::Min(min_val, hn::LoadU(d, a + i));
    }

    double result = hn::ReduceMin(d, min_val);
    for (; i < count; ++i) {
        result = std::min(result, a[i]);
    }
    return result;
}

HWY_ATTR int32_t ReduceMinInt32(const int32_t* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<int32_t>::max();
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    auto min_val = hn::Set(d, std::numeric_limits<int32_t>::max());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        min_val = hn::Min(min_val, hn::LoadU(d, a + i));
    }

    int32_t result = hn::ReduceMin(d, min_val);
    for (; i < count; ++i) {
        result = std::min(result, a[i]);
    }
    return result;
}

HWY_ATTR int64_t ReduceMinInt64(const int64_t* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<int64_t>::max();
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    auto min_val = hn::Set(d, std::numeric_limits<int64_t>::max());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        min_val = hn::Min(min_val, hn::LoadU(d, a + i));
    }

    int64_t result = hn::ReduceMin(d, min_val);
    for (; i < count; ++i) {
        result = std::min(result, a[i]);
    }
    return result;
}

HWY_ATTR float ReduceMaxFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return -std::numeric_limits<float>::infinity();
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto max_val = hn::Set(d, -std::numeric_limits<float>::infinity());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        max_val = hn::Max(max_val, hn::LoadU(d, a + i));
    }

    float result = hn::ReduceMax(d, max_val);
    for (; i < count; ++i) {
        result = std::max(result, a[i]);
    }
    return result;
}

HWY_ATTR double ReduceMaxFloat64(const double* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return -std::numeric_limits<double>::infinity();
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    auto max_val = hn::Set(d, -std::numeric_limits<double>::infinity());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        max_val = hn::Max(max_val, hn::LoadU(d, a + i));
    }

    double result = hn::ReduceMax(d, max_val);
    for (; i < count; ++i) {
        result = std::max(result, a[i]);
    }
    return result;
}

HWY_ATTR int32_t ReduceMaxInt32(const int32_t* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<int32_t>::min();
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    auto max_val = hn::Set(d, std::numeric_limits<int32_t>::min());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        max_val = hn::Max(max_val, hn::LoadU(d, a + i));
    }

    int32_t result = hn::ReduceMax(d, max_val);
    for (; i < count; ++i) {
        result = std::max(result, a[i]);
    }
    return result;
}

HWY_ATTR int64_t ReduceMaxInt64(const int64_t* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return std::numeric_limits<int64_t>::min();
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    auto max_val = hn::Set(d, std::numeric_limits<int64_t>::min());

    size_t i = 0;
    for (; i + N <= count; i += N) {
        max_val = hn::Max(max_val, hn::LoadU(d, a + i));
    }

    int64_t result = hn::ReduceMax(d, max_val);
    for (; i < count; ++i) {
        result = std::max(result, a[i]);
    }
    return result;
}

// Dot product with multiple accumulators for better throughput
HWY_ATTR float DotFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Use 4 accumulators for better throughput
    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4 * N <= count; i += 4 * N) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
        sum1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N), sum1);
        sum2 = hn::MulAdd(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N), sum2);
        sum3 = hn::MulAdd(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N), sum3);
    }

    // Process remaining full vectors
    for (; i + N <= count; i += N) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
    }

    // Combine accumulators
    float result = hn::ReduceSum(d, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));

    // Handle remainder
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

HWY_ATTR double DotFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                           size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4 * N <= count; i += 4 * N) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
        sum1 = hn::MulAdd(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N), sum1);
        sum2 = hn::MulAdd(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N), sum2);
        sum3 = hn::MulAdd(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N), sum3);
    }

    for (; i + N <= count; i += N) {
        sum0 = hn::MulAdd(hn::LoadU(d, a + i), hn::LoadU(d, b + i), sum0);
    }

    double result = hn::ReduceSum(d, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));

    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// =============================================================================
// Priority 1: Select Operations
// These use uint8_t mask: 0xFF selects from 'a', 0x00 selects from 'b'
// =============================================================================

// Template for select operations with uint8_t mask
template <typename T>
HWY_ATTR void SelectOpImpl(T* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                           const T* HWY_RESTRICT a, const T* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? a[i] : b[i];
    }
}

HWY_ATTR void SelectFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                            const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                            size_t count) {
    SelectOpImpl(out, mask, a, b, count);
}

HWY_ATTR void SelectFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                            const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                            size_t count) {
    SelectOpImpl(out, mask, a, b, count);
}

HWY_ATTR void SelectInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                          const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                          size_t count) {
    SelectOpImpl(out, mask, a, b, count);
}

// =============================================================================
// Priority 2: Extended Math Operations
// =============================================================================

HWY_ATTR void Exp2Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Exp2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Exp2(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Exp2Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Exp2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Exp2(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Log2Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log2(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Log2Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log2(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Log10Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log10(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log10(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Log10Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log10(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log10(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void SinhFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Sinh(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Sinh(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void SinhFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Sinh(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Sinh(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void CoshFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto half = hn::Set(d, 0.5f);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // Cosh(x) = (Exp(x) + Exp(-x)) / 2
        const auto exp_x = hn::Exp(d, va);
        const auto exp_neg_x = hn::Exp(d, hn::Neg(va));
        const auto vout = hn::Mul(hn::Add(exp_x, exp_neg_x), half);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto exp_x = hn::Exp(d, va);
        const auto exp_neg_x = hn::Exp(d, hn::Neg(va));
        const auto vout = hn::Mul(hn::Add(exp_x, exp_neg_x), half);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void CoshFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto half = hn::Set(d, 0.5);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // Cosh(x) = (Exp(x) + Exp(-x)) / 2
        const auto exp_x = hn::Exp(d, va);
        const auto exp_neg_x = hn::Exp(d, hn::Neg(va));
        const auto vout = hn::Mul(hn::Add(exp_x, exp_neg_x), half);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto exp_x = hn::Exp(d, va);
        const auto exp_neg_x = hn::Exp(d, hn::Neg(va));
        const auto vout = hn::Mul(hn::Add(exp_x, exp_neg_x), half);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 2: Inverse Trigonometric Operations
// =============================================================================

HWY_ATTR void AsinFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Asin(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Asin(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AsinFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Asin(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Asin(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AcosFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Acos(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Acos(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AcosFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Acos(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Acos(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AtanFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Atan(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Atan(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AtanFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Atan(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Atan(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Atan2Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT y,
                           const float* HWY_RESTRICT x, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vy = hn::LoadU(d, y + i);
        const auto vx = hn::LoadU(d, x + i);
        const auto vout = hn::Atan2(d, vy, vx);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto vy = hn::LoadN(d, y + i, remaining);
        const auto vx = hn::LoadN(d, x + i, remaining);
        const auto vout = hn::Atan2(d, vy, vx);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Atan2Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT y,
                           const double* HWY_RESTRICT x, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vy = hn::LoadU(d, y + i);
        const auto vx = hn::LoadU(d, x + i);
        const auto vout = hn::Atan2(d, vy, vx);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto vy = hn::LoadN(d, y + i, remaining);
        const auto vx = hn::LoadN(d, x + i, remaining);
        const auto vout = hn::Atan2(d, vy, vx);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 2: Inverse Hyperbolic Operations
// =============================================================================

// asinh(x) = ln(x + sqrt(x² + 1))
HWY_ATTR void AsinhFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0f);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // asinh(x) = ln(x + sqrt(x² + 1))
        const auto x2 = hn::Mul(va, va);
        const auto x2_plus_1 = hn::Add(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_plus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto x2 = hn::Mul(va, va);
        const auto x2_plus_1 = hn::Add(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_plus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AsinhFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // asinh(x) = ln(x + sqrt(x² + 1))
        const auto x2 = hn::Mul(va, va);
        const auto x2_plus_1 = hn::Add(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_plus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto x2 = hn::Mul(va, va);
        const auto x2_plus_1 = hn::Add(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_plus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
HWY_ATTR void AcoshFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0f);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // acosh(x) = ln(x + sqrt(x² - 1))
        const auto x2 = hn::Mul(va, va);
        const auto x2_minus_1 = hn::Sub(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_minus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto x2 = hn::Mul(va, va);
        const auto x2_minus_1 = hn::Sub(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_minus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AcoshFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // acosh(x) = ln(x + sqrt(x² - 1))
        const auto x2 = hn::Mul(va, va);
        const auto x2_minus_1 = hn::Sub(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_minus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto x2 = hn::Mul(va, va);
        const auto x2_minus_1 = hn::Sub(x2, one);
        const auto sqrt_term = hn::Sqrt(x2_minus_1);
        const auto sum = hn::Add(va, sqrt_term);
        const auto vout = hn::Log(d, sum);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// atanh(x) = 0.5 * ln((1 + x) / (1 - x)) for |x| < 1
HWY_ATTR void AtanhFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0f);
    const auto half = hn::Set(d, 0.5f);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
        const auto one_plus_x = hn::Add(one, va);
        const auto one_minus_x = hn::Sub(one, va);
        const auto ratio = hn::Div(one_plus_x, one_minus_x);
        const auto log_ratio = hn::Log(d, ratio);
        const auto vout = hn::Mul(half, log_ratio);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto one_plus_x = hn::Add(one, va);
        const auto one_minus_x = hn::Sub(one, va);
        const auto ratio = hn::Div(one_plus_x, one_minus_x);
        const auto log_ratio = hn::Log(d, ratio);
        const auto vout = hn::Mul(half, log_ratio);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void AtanhFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    const auto half = hn::Set(d, 0.5);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
        const auto one_plus_x = hn::Add(one, va);
        const auto one_minus_x = hn::Sub(one, va);
        const auto ratio = hn::Div(one_plus_x, one_minus_x);
        const auto log_ratio = hn::Log(d, ratio);
        const auto vout = hn::Mul(half, log_ratio);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto one_plus_x = hn::Add(one, va);
        const auto one_minus_x = hn::Sub(one, va);
        const auto ratio = hn::Div(one_plus_x, one_minus_x);
        const auto log_ratio = hn::Log(d, ratio);
        const auto vout = hn::Mul(half, log_ratio);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 2: Rounding Operations
// =============================================================================

HWY_ATTR void RoundFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, RoundOp{});
}

HWY_ATTR void RoundFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, RoundOp{});
}

HWY_ATTR void FloorFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, FloorOp{});
}

HWY_ATTR void FloorFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, FloorOp{});
}

HWY_ATTR void CeilFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, CeilOp{});
}

HWY_ATTR void CeilFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, CeilOp{});
}

HWY_ATTR void TruncFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    UnaryOpImpl(d, out, a, count, TruncOp{});
}

HWY_ATTR void TruncFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    UnaryOpImpl(d, out, a, count, TruncOp{});
}

// =============================================================================
// Priority 2: Bitwise Operations
// =============================================================================

HWY_ATTR void BitwiseAndInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, AndOp{});
}

HWY_ATTR void BitwiseAndInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                              const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, AndOp{});
}

HWY_ATTR void BitwiseOrInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                             const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, OrOp{});
}

HWY_ATTR void BitwiseOrInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                             const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, OrOp{});
}

HWY_ATTR void BitwiseXorInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              const int32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int32_t> d;
    BinaryOpImpl(d, out, a, b, count, XorOp{});
}

HWY_ATTR void BitwiseXorInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                              const int64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int64_t> d;
    BinaryOpImpl(d, out, a, b, count, XorOp{});
}

HWY_ATTR void BitwiseNotInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              size_t count) {
    const hn::ScalableTag<int32_t> d;
    UnaryOpImpl(d, out, a, count, NotOp{});
}

HWY_ATTR void BitwiseNotInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                              size_t count) {
    const hn::ScalableTag<int64_t> d;
    UnaryOpImpl(d, out, a, count, NotOp{});
}

// =============================================================================
// Priority 3: Shift Operations
// =============================================================================

// Shift left by constant amount
template <int kBits, typename T>
HWY_ATTR void ShiftLeftImpl(T* HWY_RESTRICT out, const T* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::ShiftLeft<kBits>(va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::ShiftLeft<kBits>(va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// Shift right by constant amount (arithmetic for signed, logical for unsigned)
template <int kBits, typename T>
HWY_ATTR void ShiftRightImpl(T* HWY_RESTRICT out, const T* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::ShiftRight<kBits>(va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::ShiftRight<kBits>(va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// Variable shift - each element shifted by corresponding shift amount
HWY_ATTR void ShiftLeftVarInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                const int32_t* HWY_RESTRICT shift, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vshift = hn::LoadU(d, shift + i);
        const auto vout = hn::Shl(va, vshift);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vshift = hn::LoadN(d, shift + i, remaining);
        const auto vout = hn::Shl(va, vshift);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void ShiftLeftVarInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                                const int64_t* HWY_RESTRICT shift, size_t count) {
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vshift = hn::LoadU(d, shift + i);
        const auto vout = hn::Shl(va, vshift);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vshift = hn::LoadN(d, shift + i, remaining);
        const auto vout = hn::Shl(va, vshift);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void ShiftRightVarInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                 const int32_t* HWY_RESTRICT shift, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vshift = hn::LoadU(d, shift + i);
        const auto vout = hn::Shr(va, vshift);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vshift = hn::LoadN(d, shift + i, remaining);
        const auto vout = hn::Shr(va, vshift);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void ShiftRightVarInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                                 const int64_t* HWY_RESTRICT shift, size_t count) {
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vshift = hn::LoadU(d, shift + i);
        const auto vout = hn::Shr(va, vshift);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vshift = hn::LoadN(d, shift + i, remaining);
        const auto vout = hn::Shr(va, vshift);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// Unsigned shift variants
HWY_ATTR void ShiftLeftVarUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                                 const uint32_t* HWY_RESTRICT shift, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vshift = hn::LoadU(d, shift + i);
        const auto vout = hn::Shl(va, vshift);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vshift = hn::LoadN(d, shift + i, remaining);
        const auto vout = hn::Shl(va, vshift);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void ShiftRightVarUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                                  const uint32_t* HWY_RESTRICT shift, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vshift = hn::LoadU(d, shift + i);
        const auto vout = hn::Shr(va, vshift);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vshift = hn::LoadN(d, shift + i, remaining);
        const auto vout = hn::Shr(va, vshift);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 4: Special Value Checks
// =============================================================================

HWY_ATTR void IsNaNFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    // NaN != NaN is the standard way to detect NaN
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] != a[i]) ? 0xFF : 0x00;
    }
}

HWY_ATTR void IsNaNFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] != a[i]) ? 0xFF : 0x00;
    }
}

HWY_ATTR void IsFiniteFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float val = a[i];
        // Finite means not NaN and not Inf
        out[i] = (val == val && val != std::numeric_limits<float>::infinity() &&
                  val != -std::numeric_limits<float>::infinity())
                     ? 0xFF
                     : 0x00;
    }
}

HWY_ATTR void IsFiniteFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        double val = a[i];
        out[i] = (val == val && val != std::numeric_limits<double>::infinity() &&
                  val != -std::numeric_limits<double>::infinity())
                     ? 0xFF
                     : 0x00;
    }
}

HWY_ATTR void IsInfFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float val = a[i];
        out[i] = (val == std::numeric_limits<float>::infinity() ||
                  val == -std::numeric_limits<float>::infinity())
                     ? 0xFF
                     : 0x00;
    }
}

HWY_ATTR void IsInfFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        double val = a[i];
        out[i] = (val == std::numeric_limits<double>::infinity() ||
                  val == -std::numeric_limits<double>::infinity())
                     ? 0xFF
                     : 0x00;
    }
}

// =============================================================================
// Priority 5: Type Conversion Operations
// =============================================================================

// Float32 to Float64 (promote)
HWY_ATTR void PromoteFloat32ToFloat64(double* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                      size_t count) {
    // Scalar conversion for maximum portability
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(a[i]);
    }
}

// Float64 to Float32 (demote)
HWY_ATTR void DemoteFloat64ToFloat32(float* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                     size_t count) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float, decltype(dd)> df;
    const size_t Nd = hn::Lanes(dd);

    size_t i = 0;
    for (; i + Nd <= count; i += Nd) {
        const auto vd = hn::LoadU(dd, a + i);
        const auto vf = hn::DemoteTo(df, vd);
        hn::StoreU(vf, df, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<float>(a[i]);
    }
}

// Int32 to Float32
HWY_ATTR void ConvertInt32ToFloat32(float* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                    size_t count) {
    const hn::ScalableTag<int32_t> di;
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(di);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vi = hn::LoadU(di, a + i);
        const auto vf = hn::ConvertTo(df, vi);
        hn::StoreU(vf, df, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<float>(a[i]);
    }
}

// Float32 to Int32 (truncate toward zero)
HWY_ATTR void ConvertFloat32ToInt32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<int32_t> di;
    const size_t N = hn::Lanes(df);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vf = hn::LoadU(df, a + i);
        const auto vi = hn::ConvertTo(di, vf);
        hn::StoreU(vi, di, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[i]);
    }
}

// Int64 to Float64
HWY_ATTR void ConvertInt64ToFloat64(double* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                                    size_t count) {
    const hn::ScalableTag<int64_t> di;
    const hn::ScalableTag<double> dd;
    const size_t N = hn::Lanes(di);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vi = hn::LoadU(di, a + i);
        const auto vd = hn::ConvertTo(dd, vi);
        hn::StoreU(vd, dd, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<double>(a[i]);
    }
}

// Float64 to Int64
HWY_ATTR void ConvertFloat64ToInt64(int64_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                    size_t count) {
    const hn::ScalableTag<double> dd;
    const hn::ScalableTag<int64_t> di;
    const size_t N = hn::Lanes(dd);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vd = hn::LoadU(dd, a + i);
        const auto vi = hn::ConvertTo(di, vd);
        hn::StoreU(vi, di, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<int64_t>(a[i]);
    }
}

// Int16 to Int32 (promote)
HWY_ATTR void PromoteInt16ToInt32(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                  size_t count) {
    // Scalar conversion for maximum portability
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[i]);
    }
}

// Int32 to Int16 (demote with saturation)
HWY_ATTR void DemoteInt32ToInt16(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                 size_t count) {
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<int16_t, decltype(d32)> d16;
    const size_t N32 = hn::Lanes(d32);

    size_t i = 0;
    for (; i + N32 <= count; i += N32) {
        const auto v32 = hn::LoadU(d32, a + i);
        const auto v16 = hn::DemoteTo(d16, v32);
        hn::StoreU(v16, d16, out + i);
    }
    for (; i < count; ++i) {
        int32_t val = a[i];
        if (val > 32767)
            val = 32767;
        if (val < -32768)
            val = -32768;
        out[i] = static_cast<int16_t>(val);
    }
}

// Uint8 to Int32 (promote)
HWY_ATTR void PromoteUint8ToInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                                  size_t count) {
    // Use scalar conversion for maximum portability
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[i]);
    }
}

// =============================================================================
// Priority 6: Gather Operations
// =============================================================================

// Gather float32 using int32 indices
HWY_ATTR void GatherFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                            const int32_t* HWY_RESTRICT indices, size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<int32_t> di;
    const size_t N = hn::Lanes(df);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vidx = hn::LoadU(di, indices + i);
        const auto gathered = hn::GatherIndex(df, base, vidx);
        hn::StoreU(gathered, df, out + i);
    }
    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = base[indices[i]];
    }
}

HWY_ATTR void GatherFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT base,
                            const int64_t* HWY_RESTRICT indices, size_t count) {
    const hn::ScalableTag<double> dd;
    const hn::ScalableTag<int64_t> di;
    const size_t N = hn::Lanes(dd);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vidx = hn::LoadU(di, indices + i);
        const auto gathered = hn::GatherIndex(dd, base, vidx);
        hn::StoreU(gathered, dd, out + i);
    }
    for (; i < count; ++i) {
        out[i] = base[indices[i]];
    }
}

HWY_ATTR void GatherInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                          const int32_t* HWY_RESTRICT indices, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vidx = hn::LoadU(d, indices + i);
        const auto gathered = hn::GatherIndex(d, base, vidx);
        hn::StoreU(gathered, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = base[indices[i]];
    }
}

// =============================================================================
// Priority 7: Scatter Operations
// =============================================================================

HWY_ATTR void ScatterFloat32(const float* HWY_RESTRICT values, float* HWY_RESTRICT base,
                             const int32_t* HWY_RESTRICT indices, size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<int32_t> di;
    const size_t N = hn::Lanes(df);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vvals = hn::LoadU(df, values + i);
        const auto vidx = hn::LoadU(di, indices + i);
        hn::ScatterIndex(vvals, df, base, vidx);
    }
    for (; i < count; ++i) {
        base[indices[i]] = values[i];
    }
}

HWY_ATTR void ScatterFloat64(const double* HWY_RESTRICT values, double* HWY_RESTRICT base,
                             const int64_t* HWY_RESTRICT indices, size_t count) {
    const hn::ScalableTag<double> dd;
    const hn::ScalableTag<int64_t> di;
    const size_t N = hn::Lanes(dd);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vvals = hn::LoadU(dd, values + i);
        const auto vidx = hn::LoadU(di, indices + i);
        hn::ScatterIndex(vvals, dd, base, vidx);
    }
    for (; i < count; ++i) {
        base[indices[i]] = values[i];
    }
}

HWY_ATTR void ScatterInt32(const int32_t* HWY_RESTRICT values, int32_t* HWY_RESTRICT base,
                           const int32_t* HWY_RESTRICT indices, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vvals = hn::LoadU(d, values + i);
        const auto vidx = hn::LoadU(d, indices + i);
        hn::ScatterIndex(vvals, d, base, vidx);
    }
    for (; i < count; ++i) {
        base[indices[i]] = values[i];
    }
}

// =============================================================================
// Priority 8: Horizontal Reduction Operations
// =============================================================================

// Sum of all lanes in single vector (returns scalar)
HWY_ATTR float SumOfLanesFloat32(const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        sum = hn::Add(sum, hn::LoadU(d, a + i));
    }

    float result = hn::ReduceSum(d, sum);
    for (; i < count; ++i) {
        result += a[i];
    }
    return result;
}

// Pairwise horizontal add (adjacent pairs summed)
HWY_ATTR void PairwiseAddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                 size_t count) {
    // Each output element is sum of two adjacent input elements
    // out[i] = a[2*i] + a[2*i+1]
    size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = a[2 * i] + a[2 * i + 1];
    }
}

HWY_ATTR void PairwiseAddFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                 size_t count) {
    size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = a[2 * i] + a[2 * i + 1];
    }
}

HWY_ATTR void PairwiseAddInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                               size_t count) {
    size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = a[2 * i] + a[2 * i + 1];
    }
}

// =============================================================================
// Priority 9: Saturation Operations
// =============================================================================

HWY_ATTR void SaturatedAddInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::SaturatedAdd(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        int32_t sum = static_cast<int32_t>(a[i]) + static_cast<int32_t>(b[i]);
        if (sum > 32767)
            sum = 32767;
        if (sum < -32768)
            sum = -32768;
        out[i] = static_cast<int16_t>(sum);
    }
}

HWY_ATTR void SaturatedSubInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::SaturatedSub(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        if (diff > 32767)
            diff = 32767;
        if (diff < -32768)
            diff = -32768;
        out[i] = static_cast<int16_t>(diff);
    }
}

HWY_ATTR void SaturatedAddUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::SaturatedAdd(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        uint16_t sum = static_cast<uint16_t>(a[i]) + static_cast<uint16_t>(b[i]);
        out[i] = (sum > 255) ? 255 : static_cast<uint8_t>(sum);
    }
}

HWY_ATTR void SaturatedSubUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::SaturatedSub(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (a[i] > b[i]) ? (a[i] - b[i]) : 0;
    }
}

// =============================================================================
// Priority 10: Broadcast Operations
// =============================================================================

// Broadcast scalar to all elements
HWY_ATTR void BroadcastFloat32(float* HWY_RESTRICT out, float value, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto v = hn::Set(d, value);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(v, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

HWY_ATTR void BroadcastFloat64(double* HWY_RESTRICT out, double value, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto v = hn::Set(d, value);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(v, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

HWY_ATTR void BroadcastInt32(int32_t* HWY_RESTRICT out, int32_t value, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    const auto v = hn::Set(d, value);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(v, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

HWY_ATTR void BroadcastInt64(int64_t* HWY_RESTRICT out, int64_t value, size_t count) {
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    const auto v = hn::Set(d, value);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(v, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

// =============================================================================
// Priority 11: Compress/Expand Operations
// =============================================================================

// Compress: Pack elements where mask is true to contiguous output
HWY_ATTR size_t CompressFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[out_idx++] = a[i];
        }
    }
    return out_idx;
}

HWY_ATTR size_t CompressFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[out_idx++] = a[i];
        }
    }
    return out_idx;
}

HWY_ATTR size_t CompressInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[out_idx++] = a[i];
        }
    }
    return out_idx;
}

// Expand: Opposite of compress - scatter values to positions where mask is true
HWY_ATTR void ExpandFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                            const uint8_t* HWY_RESTRICT mask, size_t mask_count, float fill_value) {
    size_t a_idx = 0;
    for (size_t i = 0; i < mask_count; ++i) {
        if (mask[i]) {
            out[i] = a[a_idx++];
        } else {
            out[i] = fill_value;
        }
    }
}

// =============================================================================
// Priority 12: Interleave Operations
// =============================================================================

// Interleave two arrays: out[0]=a[0], out[1]=b[0], out[2]=a[1], out[3]=b[1], ...
// Note: Highway's InterleaveLower/Upper work at vector-lane granularity, which doesn't
// map to element-by-element array interleaving. Using scalar implementation for correctness.
HWY_ATTR void InterleaveFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[2 * i] = a[i];
        out[2 * i + 1] = b[i];
    }
}

HWY_ATTR void InterleaveFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[2 * i] = a[i];
        out[2 * i + 1] = b[i];
    }
}

// Deinterleave: opposite of interleave
HWY_ATTR void DeinterleaveFloat32(float* HWY_RESTRICT out_a, float* HWY_RESTRICT out_b,
                                  const float* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out_a[i] = interleaved[2 * i];
        out_b[i] = interleaved[2 * i + 1];
    }
}

// =============================================================================
// Priority 13: AbsDiff and Related Operations
// =============================================================================

HWY_ATTR void AbsDiffFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                             const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::AbsDiff(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        float diff = a[i] - b[i];
        out[i] = (diff >= 0) ? diff : -diff;
    }
}

HWY_ATTR void AbsDiffFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                             const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::AbsDiff(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        double diff = a[i] - b[i];
        out[i] = (diff >= 0) ? diff : -diff;
    }
}

// =============================================================================
// Priority 14: Reciprocal Operations
// =============================================================================

HWY_ATTR void ApproxReciprocalFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                      size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::ApproximateReciprocal(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = 1.0f / a[i];
    }
}

HWY_ATTR void ApproxReciprocalSqrtFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                          size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::ApproximateReciprocalSqrt(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = 1.0f / std::sqrt(a[i]);
    }
}

HWY_ATTR void ApproxReciprocalSqrtFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                          size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::ApproximateReciprocalSqrt(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = 1.0 / std::sqrt(a[i]);
    }
}

// =============================================================================
// Priority 15: CopySign and SignBit Operations
// =============================================================================

HWY_ATTR void CopySignFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT magnitude,
                              const float* HWY_RESTRICT sign, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vmag = hn::LoadU(d, magnitude + i);
        const auto vsign = hn::LoadU(d, sign + i);
        const auto vout = hn::CopySign(vmag, vsign);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        float mag = (magnitude[i] >= 0) ? magnitude[i] : -magnitude[i];
        out[i] = (sign[i] >= 0) ? mag : -mag;
    }
}

HWY_ATTR void CopySignFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT magnitude,
                              const double* HWY_RESTRICT sign, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vmag = hn::LoadU(d, magnitude + i);
        const auto vsign = hn::LoadU(d, sign + i);
        const auto vout = hn::CopySign(vmag, vsign);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        double mag = (magnitude[i] >= 0) ? magnitude[i] : -magnitude[i];
        out[i] = (sign[i] >= 0) ? mag : -mag;
    }
}

// =============================================================================
// Priority 16: Averaging Operations
// =============================================================================

HWY_ATTR void AverageRoundUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::AverageRound(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<uint8_t>(
            (static_cast<uint16_t>(a[i]) + static_cast<uint16_t>(b[i]) + 1) / 2);
    }
}

HWY_ATTR void AverageRoundUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                                 const uint16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint16_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::AverageRound(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = static_cast<uint16_t>(
            (static_cast<uint32_t>(a[i]) + static_cast<uint32_t>(b[i]) + 1) / 2);
    }
}

// =============================================================================
// Priority 17: PopCount and Bit Counting Operations
// =============================================================================

HWY_ATTR void PopCountUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::PopulationCount(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        uint8_t val = a[i];
        uint8_t cnt = 0;
        while (val) {
            cnt += val & 1;
            val >>= 1;
        }
        out[i] = cnt;
    }
}

HWY_ATTR void LeadingZeroCountUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                                     size_t count) {
    const hn::ScalableTag<uint32_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::LeadingZeroCount(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        uint32_t val = a[i];
        if (val == 0) {
            out[i] = 32;
        } else {
            out[i] = __builtin_clz(val);
        }
    }
}

HWY_ATTR void TrailingZeroCountUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                                      size_t count) {
    const hn::ScalableTag<uint32_t> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::TrailingZeroCount(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        uint32_t val = a[i];
        if (val == 0) {
            out[i] = 32;
        } else {
            out[i] = __builtin_ctz(val);
        }
    }
}

// Scalar fallback implementations for signed and uint64 types
HWY_ATTR void PopCountInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = __builtin_popcount(static_cast<uint32_t>(a[i]));
    }
}

HWY_ATTR void PopCountInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = __builtin_popcountll(static_cast<uint64_t>(a[i]));
    }
}

HWY_ATTR void PopCountUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = __builtin_popcount(a[i]);
    }
}

HWY_ATTR void PopCountUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = __builtin_popcountll(a[i]);
    }
}

HWY_ATTR void LeadingZeroCountInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t val = static_cast<uint32_t>(a[i]);
        out[i] = (val == 0) ? 32 : __builtin_clz(val);
    }
}

HWY_ATTR void LeadingZeroCountInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t val = static_cast<uint64_t>(a[i]);
        out[i] = (val == 0) ? 64 : __builtin_clzll(val);
    }
}

HWY_ATTR void LeadingZeroCountUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] == 0) ? 64 : __builtin_clzll(a[i]);
    }
}

HWY_ATTR void TrailingZeroCountInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t val = static_cast<uint32_t>(a[i]);
        out[i] = (val == 0) ? 32 : __builtin_ctz(val);
    }
}

HWY_ATTR void TrailingZeroCountInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t val = static_cast<uint64_t>(a[i]);
        out[i] = (val == 0) ? 64 : __builtin_ctzll(val);
    }
}

HWY_ATTR void TrailingZeroCountUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                                      size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] == 0) ? 64 : __builtin_ctzll(a[i]);
    }
}

// =============================================================================
// Priority 18: Integer Unsigned Arithmetic
// =============================================================================

HWY_ATTR void AddUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                        const uint32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void AddUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                        const uint64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint64_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void SubUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                        const uint32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void SubUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                        const uint64_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint64_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void MulUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                        const uint32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    BinaryOpImpl(d, out, a, b, count, MulOp{});
}

HWY_ATTR void MinUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                        const uint32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MaxUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                        const uint32_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

// =============================================================================
// Priority 19: 8-bit and 16-bit Integer Operations
// =============================================================================

HWY_ATTR void AddInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                      const int8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int8_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void AddInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                       const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void SubInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                      const int8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int8_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void SubInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                       const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void MulInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                       const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    BinaryOpImpl(d, out, a, b, count, MulOp{});
}

HWY_ATTR void MinInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                      const int8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int8_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MaxInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                      const int8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int8_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

HWY_ATTR void MinInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                       const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MaxInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                       const int16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<int16_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

HWY_ATTR void AddUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void AddUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                        const uint16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint16_t> d;
    BinaryOpImpl(d, out, a, b, count, AddOp{});
}

HWY_ATTR void SubUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void SubUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                        const uint16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint16_t> d;
    BinaryOpImpl(d, out, a, b, count, SubOp{});
}

HWY_ATTR void MinUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MaxUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

HWY_ATTR void MinUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                        const uint16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint16_t> d;
    BinaryOpImpl(d, out, a, b, count, MinOp{});
}

HWY_ATTR void MaxUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                        const uint16_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint16_t> d;
    BinaryOpImpl(d, out, a, b, count, MaxOp{});
}

// =============================================================================
// Priority 20: Power and Expm1/Log1p Operations
// =============================================================================

HWY_ATTR void Expm1Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Expm1(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Expm1(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Expm1Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Expm1(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Expm1(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Log1pFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log1p(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log1p(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void Log1pFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Log1p(d, va);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vout = hn::Log1p(d, va);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// Priority 21: Tan Function
// =============================================================================

HWY_ATTR void TanFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    // tan(x) = sin(x) / cos(x)
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto sin_val = hn::Sin(d, va);
        const auto cos_val = hn::Cos(d, va);
        const auto vout = hn::Div(sin_val, cos_val);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto sin_val = hn::Sin(d, va);
        const auto cos_val = hn::Cos(d, va);
        const auto vout = hn::Div(sin_val, cos_val);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void TanFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto sin_val = hn::Sin(d, va);
        const auto cos_val = hn::Cos(d, va);
        const auto vout = hn::Div(sin_val, cos_val);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto sin_val = hn::Sin(d, va);
        const auto cos_val = hn::Cos(d, va);
        const auto vout = hn::Div(sin_val, cos_val);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// P0: Load/Store Operations
// =============================================================================

template <class D>
HWY_ATTR void LoadImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out,
                       const typename hn::TFromD<D>* HWY_RESTRICT src, size_t count) {
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, src + i);
        hn::StoreU(v, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto v = hn::LoadN(d, src + i, remaining);
        hn::StoreN(v, d, out + i, remaining);
    }
}

HWY_ATTR void LoadFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<float> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<double> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<int32_t> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<int64_t> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT src,
                         size_t count) {
    const hn::ScalableTag<uint16_t> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src,
                         size_t count) {
    const hn::ScalableTag<uint32_t> d;
    LoadImpl(d, out, src, count);
}

HWY_ATTR void LoadUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src,
                         size_t count) {
    const hn::ScalableTag<uint64_t> d;
    LoadImpl(d, out, src, count);
}

template <class D>
HWY_ATTR void StoreImpl(D d, const typename hn::TFromD<D>* HWY_RESTRICT src,
                        typename hn::TFromD<D>* HWY_RESTRICT dst, size_t count) {
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, src + i);
        hn::StoreU(v, d, dst + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto v = hn::LoadN(d, src + i, remaining);
        hn::StoreN(v, d, dst + i, remaining);
    }
}

HWY_ATTR void StoreFloat32(const float* HWY_RESTRICT src, float* HWY_RESTRICT dst, size_t count) {
    const hn::ScalableTag<float> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreFloat64(const double* HWY_RESTRICT src, double* HWY_RESTRICT dst, size_t count) {
    const hn::ScalableTag<double> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreInt32(const int32_t* HWY_RESTRICT src, int32_t* HWY_RESTRICT dst, size_t count) {
    const hn::ScalableTag<int32_t> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreInt64(const int64_t* HWY_RESTRICT src, int64_t* HWY_RESTRICT dst, size_t count) {
    const hn::ScalableTag<int64_t> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreUint8(const uint8_t* HWY_RESTRICT src, uint8_t* HWY_RESTRICT dst, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreUint16(const uint16_t* HWY_RESTRICT src, uint16_t* HWY_RESTRICT dst,
                          size_t count) {
    const hn::ScalableTag<uint16_t> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreUint32(const uint32_t* HWY_RESTRICT src, uint32_t* HWY_RESTRICT dst,
                          size_t count) {
    const hn::ScalableTag<uint32_t> d;
    StoreImpl(d, src, dst, count);
}

HWY_ATTR void StoreUint64(const uint64_t* HWY_RESTRICT src, uint64_t* HWY_RESTRICT dst,
                          size_t count) {
    const hn::ScalableTag<uint64_t> d;
    StoreImpl(d, src, dst, count);
}

// =============================================================================
// P0: BitCast Operations
// =============================================================================

HWY_ATTR void BitCastFloat32ToInt32Impl(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                        size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<int32_t> di;
    const size_t N = hn::Lanes(df);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vf = hn::LoadU(df, src + i);
        const auto vi = hn::BitCast(di, vf);
        hn::StoreU(vi, di, out + i);
    }
    // Handle remainder with scalar
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(float));
    }
}

HWY_ATTR void BitCastInt32ToFloat32Impl(float* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                        size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<int32_t> di;
    const size_t N = hn::Lanes(di);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vi = hn::LoadU(di, src + i);
        const auto vf = hn::BitCast(df, vi);
        hn::StoreU(vf, df, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(int32_t));
    }
}

HWY_ATTR void BitCastFloat64ToInt64Impl(int64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                        size_t count) {
    const hn::ScalableTag<double> df;
    const hn::ScalableTag<int64_t> di;
    const size_t N = hn::Lanes(df);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vf = hn::LoadU(df, src + i);
        const auto vi = hn::BitCast(di, vf);
        hn::StoreU(vi, di, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(double));
    }
}

HWY_ATTR void BitCastInt64ToFloat64Impl(double* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                                        size_t count) {
    const hn::ScalableTag<double> df;
    const hn::ScalableTag<int64_t> di;
    const size_t N = hn::Lanes(di);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vi = hn::LoadU(di, src + i);
        const auto vf = hn::BitCast(df, vi);
        hn::StoreU(vf, df, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(int64_t));
    }
}

HWY_ATTR void BitCastFloat32ToUint32Impl(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                         size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<uint32_t> du;
    const size_t N = hn::Lanes(df);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vf = hn::LoadU(df, src + i);
        const auto vu = hn::BitCast(du, vf);
        hn::StoreU(vu, du, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(float));
    }
}

HWY_ATTR void BitCastUint32ToFloat32Impl(float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src,
                                         size_t count) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<uint32_t> du;
    const size_t N = hn::Lanes(du);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vu = hn::LoadU(du, src + i);
        const auto vf = hn::BitCast(df, vu);
        hn::StoreU(vf, df, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(uint32_t));
    }
}

HWY_ATTR void BitCastFloat64ToUint64Impl(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                         size_t count) {
    const hn::ScalableTag<double> df;
    const hn::ScalableTag<uint64_t> du;
    const size_t N = hn::Lanes(df);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vf = hn::LoadU(df, src + i);
        const auto vu = hn::BitCast(du, vf);
        hn::StoreU(vu, du, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(double));
    }
}

HWY_ATTR void BitCastUint64ToFloat64Impl(double* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src,
                                         size_t count) {
    const hn::ScalableTag<double> df;
    const hn::ScalableTag<uint64_t> du;
    const size_t N = hn::Lanes(du);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vu = hn::LoadU(du, src + i);
        const auto vf = hn::BitCast(df, vu);
        hn::StoreU(vf, df, out + i);
    }
    for (; i < count; ++i) {
        std::memcpy(&out[i], &src[i], sizeof(uint64_t));
    }
}

// =============================================================================
// P0: Mask Operations
// =============================================================================

HWY_ATTR size_t CountTrueUint8Impl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            ++result;
    }
    return result;
}

HWY_ATTR size_t CountTrueUint32Impl(const uint32_t* HWY_RESTRICT mask, size_t count) {
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            ++result;
    }
    return result;
}

HWY_ATTR size_t CountTrueUint64Impl(const uint64_t* HWY_RESTRICT mask, size_t count) {
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            ++result;
    }
    return result;
}

HWY_ATTR bool AllTrueUint8Impl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] == 0)
            return false;
    }
    return true;
}

HWY_ATTR bool AllTrueUint32Impl(const uint32_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] == 0)
            return false;
    }
    return true;
}

HWY_ATTR bool AllTrueUint64Impl(const uint64_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] == 0)
            return false;
    }
    return true;
}

HWY_ATTR bool AllFalseUint8Impl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            return false;
    }
    return true;
}

HWY_ATTR bool AllFalseUint32Impl(const uint32_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            return false;
    }
    return true;
}

HWY_ATTR bool AllFalseUint64Impl(const uint64_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            return false;
    }
    return true;
}

HWY_ATTR int64_t FindFirstTrueUint8Impl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindFirstTrueUint32Impl(const uint32_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindFirstTrueUint64Impl(const uint64_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] != 0)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindLastTrueUint8Impl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = count; i > 0; --i) {
        if (mask[i - 1] != 0)
            return static_cast<int64_t>(i - 1);
    }
    return -1;
}

HWY_ATTR int64_t FindLastTrueUint32Impl(const uint32_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = count; i > 0; --i) {
        if (mask[i - 1] != 0)
            return static_cast<int64_t>(i - 1);
    }
    return -1;
}

HWY_ATTR int64_t FindLastTrueUint64Impl(const uint64_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = count; i > 0; --i) {
        if (mask[i - 1] != 0)
            return static_cast<int64_t>(i - 1);
    }
    return -1;
}

// =============================================================================
// P1: Reverse Operations
// =============================================================================

template <class D>
HWY_ATTR void ReverseImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out,
                          const typename hn::TFromD<D>* HWY_RESTRICT a, size_t count) {
    // Simple scalar implementation - reversing a complete array
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[count - 1 - i];
    }
}

HWY_ATTR void ReverseFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<double> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int32_t> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int64_t> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<uint16_t> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<uint32_t> d;
    ReverseImpl(d, out, a, count);
}

HWY_ATTR void ReverseUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<uint64_t> d;
    ReverseImpl(d, out, a, count);
}

// =============================================================================
// P1: Fill Operations
// =============================================================================

template <class D>
HWY_ATTR void FillImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT data, typename hn::TFromD<D> value,
                       size_t count) {
    const size_t N = hn::Lanes(d);
    const auto v = hn::Set(d, value);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(v, d, data + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        hn::StoreN(v, d, data + i, remaining);
    }
}

HWY_ATTR void FillFloat32(float* HWY_RESTRICT data, float value, size_t count) {
    const hn::ScalableTag<float> d;
    FillImpl(d, data, value, count);
}

HWY_ATTR void FillFloat64(double* HWY_RESTRICT data, double value, size_t count) {
    const hn::ScalableTag<double> d;
    FillImpl(d, data, value, count);
}

HWY_ATTR void FillInt32(int32_t* HWY_RESTRICT data, int32_t value, size_t count) {
    const hn::ScalableTag<int32_t> d;
    FillImpl(d, data, value, count);
}

HWY_ATTR void FillInt64(int64_t* HWY_RESTRICT data, int64_t value, size_t count) {
    const hn::ScalableTag<int64_t> d;
    FillImpl(d, data, value, count);
}

HWY_ATTR void FillUint8(uint8_t* HWY_RESTRICT data, uint8_t value, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    FillImpl(d, data, value, count);
}

// =============================================================================
// P1: Copy Operations
// =============================================================================

HWY_ATTR void CopyFloat32(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<float> d;
    LoadImpl(d, dst, src, count);
}

HWY_ATTR void CopyFloat64(double* HWY_RESTRICT dst, const double* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<double> d;
    LoadImpl(d, dst, src, count);
}

HWY_ATTR void CopyInt32(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<int32_t> d;
    LoadImpl(d, dst, src, count);
}

HWY_ATTR void CopyInt64(int64_t* HWY_RESTRICT dst, const int64_t* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<int64_t> d;
    LoadImpl(d, dst, src, count);
}

HWY_ATTR void CopyUint8(uint8_t* HWY_RESTRICT dst, const uint8_t* HWY_RESTRICT src, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    LoadImpl(d, dst, src, count);
}

// =============================================================================
// P1: MinOfLanes/MaxOfLanes Operations
// =============================================================================

HWY_ATTR void MinOfLanesFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<float>::infinity();
        return;
    }
    float min_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] < min_val)
            min_val = a[i];
    }
    *out = min_val;
}

HWY_ATTR void MinOfLanesFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<double>::infinity();
        return;
    }
    double min_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] < min_val)
            min_val = a[i];
    }
    *out = min_val;
}

HWY_ATTR void MinOfLanesInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<int32_t>::max();
        return;
    }
    int32_t min_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] < min_val)
            min_val = a[i];
    }
    *out = min_val;
}

HWY_ATTR void MinOfLanesInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                              size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<int64_t>::max();
        return;
    }
    int64_t min_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] < min_val)
            min_val = a[i];
    }
    *out = min_val;
}

HWY_ATTR void MinOfLanesUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                               size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<uint32_t>::max();
        return;
    }
    uint32_t min_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] < min_val)
            min_val = a[i];
    }
    *out = min_val;
}

HWY_ATTR void MaxOfLanesFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                size_t count) {
    if (count == 0) {
        *out = -std::numeric_limits<float>::infinity();
        return;
    }
    float max_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] > max_val)
            max_val = a[i];
    }
    *out = max_val;
}

HWY_ATTR void MaxOfLanesFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                size_t count) {
    if (count == 0) {
        *out = -std::numeric_limits<double>::infinity();
        return;
    }
    double max_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] > max_val)
            max_val = a[i];
    }
    *out = max_val;
}

HWY_ATTR void MaxOfLanesInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<int32_t>::min();
        return;
    }
    int32_t max_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] > max_val)
            max_val = a[i];
    }
    *out = max_val;
}

HWY_ATTR void MaxOfLanesInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                              size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<int64_t>::min();
        return;
    }
    int64_t max_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] > max_val)
            max_val = a[i];
    }
    *out = max_val;
}

HWY_ATTR void MaxOfLanesUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                               size_t count) {
    if (count == 0) {
        *out = std::numeric_limits<uint32_t>::min();
        return;
    }
    uint32_t max_val = a[0];
    for (size_t i = 1; i < count; ++i) {
        if (a[i] > max_val)
            max_val = a[i];
    }
    *out = max_val;
}

// =============================================================================
// P1: Interleaved Load/Store Operations
// =============================================================================

HWY_ATTR void LoadInterleaved2Float32(float* HWY_RESTRICT a, float* HWY_RESTRICT b,
                                      const float* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[2 * i];
        b[i] = interleaved[2 * i + 1];
    }
}

HWY_ATTR void LoadInterleaved2Int32(int32_t* HWY_RESTRICT a, int32_t* HWY_RESTRICT b,
                                    const int32_t* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[2 * i];
        b[i] = interleaved[2 * i + 1];
    }
}

HWY_ATTR void LoadInterleaved2Uint8(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                                    const uint8_t* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[2 * i];
        b[i] = interleaved[2 * i + 1];
    }
}

HWY_ATTR void LoadInterleaved3Float32(float* HWY_RESTRICT a, float* HWY_RESTRICT b,
                                      float* HWY_RESTRICT c, const float* HWY_RESTRICT interleaved,
                                      size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[3 * i];
        b[i] = interleaved[3 * i + 1];
        c[i] = interleaved[3 * i + 2];
    }
}

HWY_ATTR void LoadInterleaved3Uint8(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                                    uint8_t* HWY_RESTRICT c,
                                    const uint8_t* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[3 * i];
        b[i] = interleaved[3 * i + 1];
        c[i] = interleaved[3 * i + 2];
    }
}

HWY_ATTR void LoadInterleaved4Float32(float* HWY_RESTRICT a, float* HWY_RESTRICT b,
                                      float* HWY_RESTRICT c, float* HWY_RESTRICT d,
                                      const float* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[4 * i];
        b[i] = interleaved[4 * i + 1];
        c[i] = interleaved[4 * i + 2];
        d[i] = interleaved[4 * i + 3];
    }
}

HWY_ATTR void LoadInterleaved4Uint8(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                                    uint8_t* HWY_RESTRICT c, uint8_t* HWY_RESTRICT d,
                                    const uint8_t* HWY_RESTRICT interleaved, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        a[i] = interleaved[4 * i];
        b[i] = interleaved[4 * i + 1];
        c[i] = interleaved[4 * i + 2];
        d[i] = interleaved[4 * i + 3];
    }
}

HWY_ATTR void StoreInterleaved2Float32(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                                       const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[2 * i] = a[i];
        interleaved[2 * i + 1] = b[i];
    }
}

HWY_ATTR void StoreInterleaved2Int32(int32_t* HWY_RESTRICT interleaved,
                                     const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[2 * i] = a[i];
        interleaved[2 * i + 1] = b[i];
    }
}

HWY_ATTR void StoreInterleaved2Uint8(uint8_t* HWY_RESTRICT interleaved,
                                     const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[2 * i] = a[i];
        interleaved[2 * i + 1] = b[i];
    }
}

HWY_ATTR void StoreInterleaved3Float32(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[3 * i] = a[i];
        interleaved[3 * i + 1] = b[i];
        interleaved[3 * i + 2] = c[i];
    }
}

HWY_ATTR void StoreInterleaved3Uint8(uint8_t* HWY_RESTRICT interleaved,
                                     const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
                                     const uint8_t* HWY_RESTRICT c, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[3 * i] = a[i];
        interleaved[3 * i + 1] = b[i];
        interleaved[3 * i + 2] = c[i];
    }
}

HWY_ATTR void StoreInterleaved4Float32(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                                       const float* HWY_RESTRICT d, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[4 * i] = a[i];
        interleaved[4 * i + 1] = b[i];
        interleaved[4 * i + 2] = c[i];
        interleaved[4 * i + 3] = d[i];
    }
}

HWY_ATTR void StoreInterleaved4Uint8(uint8_t* HWY_RESTRICT interleaved,
                                     const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
                                     const uint8_t* HWY_RESTRICT c, const uint8_t* HWY_RESTRICT d,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        interleaved[4 * i] = a[i];
        interleaved[4 * i + 1] = b[i];
        interleaved[4 * i + 2] = c[i];
        interleaved[4 * i + 3] = d[i];
    }
}

// =============================================================================
// P1: Find Operations
// =============================================================================

HWY_ATTR int64_t FindFloat32(const float* HWY_RESTRICT data, float value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] == value)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindInt32(const int32_t* HWY_RESTRICT data, int32_t value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] == value)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindUint8(const uint8_t* HWY_RESTRICT data, uint8_t value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] == value)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindGtFloat32(const float* HWY_RESTRICT data, float threshold, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] > threshold)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindGtInt32(const int32_t* HWY_RESTRICT data, int32_t threshold, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] > threshold)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindLtFloat32(const float* HWY_RESTRICT data, float threshold, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < threshold)
            return static_cast<int64_t>(i);
    }
    return -1;
}

HWY_ATTR int64_t FindLtInt32(const int32_t* HWY_RESTRICT data, int32_t threshold, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < threshold)
            return static_cast<int64_t>(i);
    }
    return -1;
}

// =============================================================================
// P1: Transform Operations
// =============================================================================

HWY_ATTR void TransformAddFloat32(float* HWY_RESTRICT data, float scalar, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        const auto vout = hn::Add(v, vs);
        hn::StoreU(vout, d, data + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto v = hn::LoadN(d, data + i, remaining);
        const auto vout = hn::Add(v, vs);
        hn::StoreN(vout, d, data + i, remaining);
    }
}

HWY_ATTR void TransformAddInt32(int32_t* HWY_RESTRICT data, int32_t scalar, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        const auto vout = hn::Add(v, vs);
        hn::StoreU(vout, d, data + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto v = hn::LoadN(d, data + i, remaining);
        const auto vout = hn::Add(v, vs);
        hn::StoreN(vout, d, data + i, remaining);
    }
}

HWY_ATTR void TransformMulFloat32(float* HWY_RESTRICT data, float scalar, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        const auto vout = hn::Mul(v, vs);
        hn::StoreU(vout, d, data + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto v = hn::LoadN(d, data + i, remaining);
        const auto vout = hn::Mul(v, vs);
        hn::StoreN(vout, d, data + i, remaining);
    }
}

HWY_ATTR void TransformMulInt32(int32_t* HWY_RESTRICT data, int32_t scalar, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    const auto vs = hn::Set(d, scalar);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, data + i);
        const auto vout = hn::Mul(v, vs);
        hn::StoreU(vout, d, data + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto v = hn::LoadN(d, data + i, remaining);
        const auto vout = hn::Mul(v, vs);
        hn::StoreN(vout, d, data + i, remaining);
    }
}

// =============================================================================
// P2: Hypot (sqrt(a^2 + b^2)) - SIMD implementation
// =============================================================================

HWY_ATTR void HypotFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                           const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        // hypot(a, b) = sqrt(a^2 + b^2)
        const auto va2 = hn::Mul(va, va);
        const auto vb2 = hn::Mul(vb, vb);
        const auto vsum = hn::Add(va2, vb2);
        const auto vout = hn::Sqrt(vsum);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vb = hn::LoadN(d, b + i, remaining);
        const auto va2 = hn::Mul(va, va);
        const auto vb2 = hn::Mul(vb, vb);
        const auto vsum = hn::Add(va2, vb2);
        const auto vout = hn::Sqrt(vsum);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void HypotFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                           const double* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto va2 = hn::Mul(va, va);
        const auto vb2 = hn::Mul(vb, vb);
        const auto vsum = hn::Add(va2, vb2);
        const auto vout = hn::Sqrt(vsum);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vb = hn::LoadN(d, b + i, remaining);
        const auto va2 = hn::Mul(va, va);
        const auto vb2 = hn::Mul(vb, vb);
        const auto vsum = hn::Add(va2, vb2);
        const auto vout = hn::Sqrt(vsum);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// P2: MatVec (Matrix-vector multiplication) - SIMD implementation
// =============================================================================

HWY_ATTR void MatVecFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT A,
                            const float* HWY_RESTRICT x, size_t rows, size_t cols) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t r = 0; r < rows; ++r) {
        const float* row = A + r * cols;

        // Use multiple accumulators for better pipelining
        auto acc0 = hn::Zero(d);
        auto acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d);
        auto acc3 = hn::Zero(d);

        size_t c = 0;
        // Process 4*N elements per iteration
        for (; c + 4 * N <= cols; c += 4 * N) {
            const auto a0 = hn::LoadU(d, row + c);
            const auto a1 = hn::LoadU(d, row + c + N);
            const auto a2 = hn::LoadU(d, row + c + 2 * N);
            const auto a3 = hn::LoadU(d, row + c + 3 * N);
            const auto x0 = hn::LoadU(d, x + c);
            const auto x1 = hn::LoadU(d, x + c + N);
            const auto x2 = hn::LoadU(d, x + c + 2 * N);
            const auto x3 = hn::LoadU(d, x + c + 3 * N);
            acc0 = hn::MulAdd(a0, x0, acc0);
            acc1 = hn::MulAdd(a1, x1, acc1);
            acc2 = hn::MulAdd(a2, x2, acc2);
            acc3 = hn::MulAdd(a3, x3, acc3);
        }

        // Process remaining full vectors
        for (; c + N <= cols; c += N) {
            const auto a0 = hn::LoadU(d, row + c);
            const auto x0 = hn::LoadU(d, x + c);
            acc0 = hn::MulAdd(a0, x0, acc0);
        }

        // Combine accumulators
        acc0 = hn::Add(acc0, acc1);
        acc2 = hn::Add(acc2, acc3);
        acc0 = hn::Add(acc0, acc2);

        // Handle remainder with scalar
        float result = hn::ReduceSum(d, acc0);
        for (; c < cols; ++c) {
            result += row[c] * x[c];
        }
        out[r] = result;
    }
}

HWY_ATTR void MatVecFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT A,
                            const double* HWY_RESTRICT x, size_t rows, size_t cols) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    for (size_t r = 0; r < rows; ++r) {
        const double* row = A + r * cols;

        auto acc0 = hn::Zero(d);
        auto acc1 = hn::Zero(d);
        auto acc2 = hn::Zero(d);
        auto acc3 = hn::Zero(d);

        size_t c = 0;
        for (; c + 4 * N <= cols; c += 4 * N) {
            const auto a0 = hn::LoadU(d, row + c);
            const auto a1 = hn::LoadU(d, row + c + N);
            const auto a2 = hn::LoadU(d, row + c + 2 * N);
            const auto a3 = hn::LoadU(d, row + c + 3 * N);
            const auto x0 = hn::LoadU(d, x + c);
            const auto x1 = hn::LoadU(d, x + c + N);
            const auto x2 = hn::LoadU(d, x + c + 2 * N);
            const auto x3 = hn::LoadU(d, x + c + 3 * N);
            acc0 = hn::MulAdd(a0, x0, acc0);
            acc1 = hn::MulAdd(a1, x1, acc1);
            acc2 = hn::MulAdd(a2, x2, acc2);
            acc3 = hn::MulAdd(a3, x3, acc3);
        }

        for (; c + N <= cols; c += N) {
            const auto a0 = hn::LoadU(d, row + c);
            const auto x0 = hn::LoadU(d, x + c);
            acc0 = hn::MulAdd(a0, x0, acc0);
        }

        acc0 = hn::Add(acc0, acc1);
        acc2 = hn::Add(acc2, acc3);
        acc0 = hn::Add(acc0, acc2);

        double result = hn::ReduceSum(d, acc0);
        for (; c < cols; ++c) {
            result += row[c] * x[c];
        }
        out[r] = result;
    }
}

// =============================================================================
// P2: Pow (Power function) - using exp(y * log(x))
// =============================================================================

HWY_ATTR void PowFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                         const float* HWY_RESTRICT exp, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vbase = hn::LoadU(d, base + i);
        const auto vexp = hn::LoadU(d, exp + i);
        // pow(x, y) = exp(y * log(x))
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto vbase = hn::LoadN(d, base + i, remaining);
        const auto vexp = hn::LoadN(d, exp + i, remaining);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void PowFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT base,
                         const double* HWY_RESTRICT exp, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vbase = hn::LoadU(d, base + i);
        const auto vexp = hn::LoadU(d, exp + i);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto vbase = hn::LoadN(d, base + i, remaining);
        const auto vexp = hn::LoadN(d, exp + i, remaining);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void PowScalarFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                               float scalar_exp, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vexp = hn::Set(d, scalar_exp);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vbase = hn::LoadU(d, base + i);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto vbase = hn::LoadN(d, base + i, remaining);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void PowScalarFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT base,
                               double scalar_exp, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto vexp = hn::Set(d, scalar_exp);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vbase = hn::LoadU(d, base + i);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto vbase = hn::LoadN(d, base + i, remaining);
        const auto vlog = hn::Log(d, vbase);
        const auto vprod = hn::Mul(vexp, vlog);
        const auto vout = hn::Exp(d, vprod);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// =============================================================================
// P0: Masked Arithmetic Operations
// =============================================================================

// Generic masked binary operation template
template <class D, class Op>
HWY_ATTR void
MaskedBinaryOpImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                   const typename hn::TFromD<D>* HWY_RESTRICT a,
                   const typename hn::TFromD<D>* HWY_RESTRICT b,
                   const typename hn::TFromD<D>* HWY_RESTRICT no, size_t count, Op op) {
    using T = typename hn::TFromD<D>;
    // Process element by element with mask check
    // This is a simple implementation; could be optimized with SIMD masking
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = op(a[i], b[i]);
        } else {
            out[i] = no[i];
        }
    }
}

// Generic masked unary operation template
template <class D, class Op>
HWY_ATTR void
MaskedUnaryOpImpl(D d, typename hn::TFromD<D>* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                  const typename hn::TFromD<D>* HWY_RESTRICT a,
                  const typename hn::TFromD<D>* HWY_RESTRICT no, size_t count, Op op) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = op(a[i]);
        } else {
            out[i] = no[i];
        }
    }
}

// MaskedAdd implementations
HWY_ATTR void MaskedAddFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](float x, float y) { return x + y; });
}

HWY_ATTR void MaskedAddFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               const double* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<double> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](double x, double y) { return x + y; });
}

HWY_ATTR void MaskedAddInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                             const int32_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](int32_t x, int32_t y) { return x + y; });
}

HWY_ATTR void MaskedAddInt64(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
                             const int64_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int64_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](int64_t x, int64_t y) { return x + y; });
}

// MaskedSub implementations
HWY_ATTR void MaskedSubFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](float x, float y) { return x - y; });
}

HWY_ATTR void MaskedSubFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               const double* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<double> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](double x, double y) { return x - y; });
}

HWY_ATTR void MaskedSubInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                             const int32_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](int32_t x, int32_t y) { return x - y; });
}

HWY_ATTR void MaskedSubInt64(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
                             const int64_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int64_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](int64_t x, int64_t y) { return x - y; });
}

// MaskedMul implementations
HWY_ATTR void MaskedMulFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](float x, float y) { return x * y; });
}

HWY_ATTR void MaskedMulFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               const double* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<double> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](double x, double y) { return x * y; });
}

HWY_ATTR void MaskedMulInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                             const int32_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](int32_t x, int32_t y) { return x * y; });
}

HWY_ATTR void MaskedMulInt64(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
                             const int64_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int64_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](int64_t x, int64_t y) { return x * y; });
}

// MaskedDiv implementations
HWY_ATTR void MaskedDivFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](float x, float y) { return x / y; });
}

HWY_ATTR void MaskedDivFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               const double* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<double> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count, [](double x, double y) { return x / y; });
}

// MaskedMin implementations
HWY_ATTR void MaskedMinFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count,
                       [](float x, float y) { return x < y ? x : y; });
}

HWY_ATTR void MaskedMinFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               const double* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<double> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count,
                       [](double x, double y) { return x < y ? x : y; });
}

HWY_ATTR void MaskedMinInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                             const int32_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count,
                       [](int32_t x, int32_t y) { return x < y ? x : y; });
}

// MaskedMax implementations
HWY_ATTR void MaskedMaxFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count,
                       [](float x, float y) { return x > y ? x : y; });
}

HWY_ATTR void MaskedMaxFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               const double* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<double> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count,
                       [](double x, double y) { return x > y ? x : y; });
}

HWY_ATTR void MaskedMaxInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                             const int32_t* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedBinaryOpImpl(d, out, mask, a, b, no, count,
                       [](int32_t x, int32_t y) { return x > y ? x : y; });
}

// MaskedAbs implementations
HWY_ATTR void MaskedAbsFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT no,
                               size_t count) {
    const hn::ScalableTag<float> d;
    MaskedUnaryOpImpl(d, out, mask, a, no, count, [](float x) { return x < 0 ? -x : x; });
}

HWY_ATTR void MaskedAbsFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT no,
                               size_t count) {
    const hn::ScalableTag<double> d;
    MaskedUnaryOpImpl(d, out, mask, a, no, count, [](double x) { return x < 0 ? -x : x; });
}

HWY_ATTR void MaskedAbsInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT no,
                             size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedUnaryOpImpl(d, out, mask, a, no, count, [](int32_t x) { return x < 0 ? -x : x; });
}

// MaskedNeg implementations
HWY_ATTR void MaskedNegFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const float* HWY_RESTRICT a, const float* HWY_RESTRICT no,
                               size_t count) {
    const hn::ScalableTag<float> d;
    MaskedUnaryOpImpl(d, out, mask, a, no, count, [](float x) { return -x; });
}

HWY_ATTR void MaskedNegFloat64(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const double* HWY_RESTRICT a, const double* HWY_RESTRICT no,
                               size_t count) {
    const hn::ScalableTag<double> d;
    MaskedUnaryOpImpl(d, out, mask, a, no, count, [](double x) { return -x; });
}

HWY_ATTR void MaskedNegInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                             const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT no,
                             size_t count) {
    const hn::ScalableTag<int32_t> d;
    MaskedUnaryOpImpl(d, out, mask, a, no, count, [](int32_t x) { return -x; });
}

// =============================================================================
// P0: Widening Operations
// =============================================================================

HWY_ATTR void SumsOf2Int16(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count) {
    const size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<int32_t>(a[2 * i]) + static_cast<int32_t>(a[2 * i + 1]);
    }
}

HWY_ATTR void SumsOf2Uint16(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                            size_t count) {
    const size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<uint32_t>(a[2 * i]) + static_cast<uint32_t>(a[2 * i + 1]);
    }
}

HWY_ATTR void SumsOf4Int8(int32_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count) {
    const size_t out_count = count / 4;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<int32_t>(a[4 * i]) + static_cast<int32_t>(a[4 * i + 1]) +
                 static_cast<int32_t>(a[4 * i + 2]) + static_cast<int32_t>(a[4 * i + 3]);
    }
}

HWY_ATTR void SumsOf4Uint8(uint32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                           size_t count) {
    const size_t out_count = count / 4;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<uint32_t>(a[4 * i]) + static_cast<uint32_t>(a[4 * i + 1]) +
                 static_cast<uint32_t>(a[4 * i + 2]) + static_cast<uint32_t>(a[4 * i + 3]);
    }
}

HWY_ATTR void MulEvenInt32(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                           const int32_t* HWY_RESTRICT b, size_t count) {
    const size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<int64_t>(a[2 * i]) * static_cast<int64_t>(b[2 * i]);
    }
}

HWY_ATTR void MulEvenUint32(uint64_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                            const uint32_t* HWY_RESTRICT b, size_t count) {
    const size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<uint64_t>(a[2 * i]) * static_cast<uint64_t>(b[2 * i]);
    }
}

HWY_ATTR void MulOddInt32(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                          const int32_t* HWY_RESTRICT b, size_t count) {
    const size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<int64_t>(a[2 * i + 1]) * static_cast<int64_t>(b[2 * i + 1]);
    }
}

HWY_ATTR void MulOddUint32(uint64_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                           const uint32_t* HWY_RESTRICT b, size_t count) {
    const size_t out_count = count / 2;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<uint64_t>(a[2 * i + 1]) * static_cast<uint64_t>(b[2 * i + 1]);
    }
}

// =============================================================================
// P0: Additional Comparison Operations
// =============================================================================

HWY_ATTR void IsNegativeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] < 0.0f) ? 0xFF : 0x00;
    }
}

HWY_ATTR void IsNegativeFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] < 0.0) ? 0xFF : 0x00;
    }
}

HWY_ATTR void IsNegativeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] < 0) ? 0xFF : 0x00;
    }
}

HWY_ATTR void IsEitherNaNFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                 const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // NaN != NaN is true for NaN values
        bool a_nan = (a[i] != a[i]);
        bool b_nan = (b[i] != b[i]);
        out[i] = (a_nan || b_nan) ? 0xFF : 0x00;
    }
}

HWY_ATTR void IsEitherNaNFloat64(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                 const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        bool a_nan = (a[i] != a[i]);
        bool b_nan = (b[i] != b[i]);
        out[i] = (a_nan || b_nan) ? 0xFF : 0x00;
    }
}

// =============================================================================
// P1: Extended FMA Operations
// =============================================================================

// NegMulSub: out[i] = -(a[i] * b[i]) - c[i]
HWY_ATTR void NegMulSubFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                               const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = hn::NegMulSub(va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vb = hn::LoadN(d, b + i, remaining);
        const auto vc = hn::LoadN(d, c + i, remaining);
        const auto vout = hn::NegMulSub(va, vb, vc);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

HWY_ATTR void NegMulSubFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                               const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = hn::NegMulSub(va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }
    const size_t remaining = count - i;
    if (remaining > 0) {
        const auto va = hn::LoadN(d, a + i, remaining);
        const auto vb = hn::LoadN(d, b + i, remaining);
        const auto vc = hn::LoadN(d, c + i, remaining);
        const auto vout = hn::NegMulSub(va, vb, vc);
        hn::StoreN(vout, d, out + i, remaining);
    }
}

// MulAddSub: Even lanes: a*b-c, Odd lanes: a*b+c
HWY_ATTR void MulAddSubFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                               const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = hn::MulAddSub(va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback for remainder
    for (; i < count; ++i) {
        float mul = a[i] * b[i];
        if (i % 2 == 0) {
            out[i] = mul - c[i];  // Even: subtract
        } else {
            out[i] = mul + c[i];  // Odd: add
        }
    }
}

HWY_ATTR void MulAddSubFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                               const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = hn::MulAddSub(va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback for remainder
    for (; i < count; ++i) {
        double mul = a[i] * b[i];
        if (i % 2 == 0) {
            out[i] = mul - c[i];  // Even: subtract
        } else {
            out[i] = mul + c[i];  // Odd: add
        }
    }
}

// =============================================================================
// P1: CompressStore Operations
// =============================================================================

HWY_ATTR size_t CompressStoreFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                     const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t written = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[written++] = a[i];
        }
    }
    return written;
}

HWY_ATTR size_t CompressStoreFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                     const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t written = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[written++] = a[i];
        }
    }
    return written;
}

HWY_ATTR size_t CompressStoreInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                   const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t written = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[written++] = a[i];
        }
    }
    return written;
}

// =============================================================================
// P2: Iota and FirstN Operations
// =============================================================================

HWY_ATTR void IotaFloat32(float* HWY_RESTRICT out, float start, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto increment = hn::Set(d, static_cast<float>(N));
    auto base = hn::Iota(d, start);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(base, d, out + i);
        base = hn::Add(base, increment);
    }

    // Handle remainder
    for (; i < count; ++i) {
        out[i] = start + static_cast<float>(i);
    }
}

HWY_ATTR void IotaFloat64(double* HWY_RESTRICT out, double start, size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto increment = hn::Set(d, static_cast<double>(N));
    auto base = hn::Iota(d, start);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(base, d, out + i);
        base = hn::Add(base, increment);
    }

    for (; i < count; ++i) {
        out[i] = start + static_cast<double>(i);
    }
}

HWY_ATTR void IotaInt32(int32_t* HWY_RESTRICT out, int32_t start, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    const auto increment = hn::Set(d, static_cast<int32_t>(N));
    auto base = hn::Iota(d, start);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(base, d, out + i);
        base = hn::Add(base, increment);
    }

    for (; i < count; ++i) {
        out[i] = start + static_cast<int32_t>(i);
    }
}

HWY_ATTR void IotaInt64(int64_t* HWY_RESTRICT out, int64_t start, size_t count) {
    const hn::ScalableTag<int64_t> d;
    const size_t N = hn::Lanes(d);
    const auto increment = hn::Set(d, static_cast<int64_t>(N));
    auto base = hn::Iota(d, start);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(base, d, out + i);
        base = hn::Add(base, increment);
    }

    for (; i < count; ++i) {
        out[i] = start + static_cast<int64_t>(i);
    }
}

HWY_ATTR void IotaUint32(uint32_t* HWY_RESTRICT out, uint32_t start, size_t count) {
    const hn::ScalableTag<uint32_t> d;
    const size_t N = hn::Lanes(d);
    const auto increment = hn::Set(d, static_cast<uint32_t>(N));
    auto base = hn::Iota(d, start);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(base, d, out + i);
        base = hn::Add(base, increment);
    }

    for (; i < count; ++i) {
        out[i] = start + static_cast<uint32_t>(i);
    }
}

HWY_ATTR void IotaUint64(uint64_t* HWY_RESTRICT out, uint64_t start, size_t count) {
    const hn::ScalableTag<uint64_t> d;
    const size_t N = hn::Lanes(d);
    const auto increment = hn::Set(d, static_cast<uint64_t>(N));
    auto base = hn::Iota(d, start);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(base, d, out + i);
        base = hn::Add(base, increment);
    }

    for (; i < count; ++i) {
        out[i] = start + static_cast<uint64_t>(i);
    }
}

HWY_ATTR void FirstNImpl(uint8_t* HWY_RESTRICT out, size_t n, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (i < n) ? 0xFF : 0x00;
    }
}

// =============================================================================
// P0: WidenMulAccumulate Operations
// =============================================================================

// WidenMulAccumulate: out[i] = a[i] * b[i] + c[i] with widening (int16 -> int32)
HWY_ATTR void WidenMulAccumulateInt16(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                      const int16_t* HWY_RESTRICT b, const int32_t* HWY_RESTRICT c,
                                      size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]) + c[i];
    }
}

// WidenMulAccumulate: out[i] = a[i] * b[i] + c[i] with widening (uint16 -> uint32)
HWY_ATTR void WidenMulAccumulateUint16(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                                       const uint16_t* HWY_RESTRICT b,
                                       const uint32_t* HWY_RESTRICT c, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]) + c[i];
    }
}

// =============================================================================
// P0: SumsOf8 Operation
// =============================================================================

// Sum of 8 adjacent elements with widening: out[i] = sum(a[8i:8i+8])
HWY_ATTR void SumsOf8Uint8(uint64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                           size_t count) {
    const size_t out_count = count / 8;
    for (size_t i = 0; i < out_count; ++i) {
        out[i] = static_cast<uint64_t>(a[8 * i]) + static_cast<uint64_t>(a[8 * i + 1]) +
                 static_cast<uint64_t>(a[8 * i + 2]) + static_cast<uint64_t>(a[8 * i + 3]) +
                 static_cast<uint64_t>(a[8 * i + 4]) + static_cast<uint64_t>(a[8 * i + 5]) +
                 static_cast<uint64_t>(a[8 * i + 6]) + static_cast<uint64_t>(a[8 * i + 7]);
    }
}

// =============================================================================
// P0: TestBit Operation
// =============================================================================

// TestBit: out[i] = (a[i] & (1 << bit)) ? 0xFF : 0x00
HWY_ATTR void TestBitInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t bit,
                           size_t count) {
    const int32_t mask = static_cast<int32_t>(1) << bit;
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] & mask) ? 0xFF : 0x00;
    }
}

HWY_ATTR void TestBitInt64(uint8_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t bit,
                           size_t count) {
    const int64_t mask = static_cast<int64_t>(1) << bit;
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] & mask) ? 0xFF : 0x00;
    }
}

HWY_ATTR void TestBitUint32(uint8_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t bit,
                            size_t count) {
    const uint32_t mask = static_cast<uint32_t>(1) << bit;
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] & mask) ? 0xFF : 0x00;
    }
}

HWY_ATTR void TestBitUint64(uint8_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t bit,
                            size_t count) {
    const uint64_t mask = static_cast<uint64_t>(1) << bit;
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] & mask) ? 0xFF : 0x00;
    }
}

// =============================================================================
// P1: MulSubAdd - Opposite of MulAddSub
// Even lanes: out[i] = a[i] * b[i] + c[i]
// Odd lanes: out[i] = a[i] * b[i] - c[i]
// =============================================================================

HWY_ATTR void MulSubAddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                               const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = hn::MulSubAdd(va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback for remainder
    for (; i < count; ++i) {
        float mul = a[i] * b[i];
        if (i % 2 == 0) {
            out[i] = mul + c[i];  // Even: add
        } else {
            out[i] = mul - c[i];  // Odd: subtract
        }
    }
}

HWY_ATTR void MulSubAddFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                               const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                               size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto vout = hn::MulSubAdd(va, vb, vc);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        double mul = a[i] * b[i];
        if (i % 2 == 0) {
            out[i] = mul + c[i];
        } else {
            out[i] = mul - c[i];
        }
    }
}

// =============================================================================
// P1: Reverse2, Reverse4, Reverse8 - Reverse within blocks
// =============================================================================

// Reverse2: Reverse adjacent pairs
HWY_ATTR void Reverse2Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Reverse2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback - swap adjacent pairs
    for (; i + 1 < count; i += 2) {
        out[i] = a[i + 1];
        out[i + 1] = a[i];
    }
    if (i < count) {
        out[i] = a[i];  // Odd element
    }
}

HWY_ATTR void Reverse2Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                              size_t count) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Reverse2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i + 1 < count; i += 2) {
        out[i] = a[i + 1];
        out[i + 1] = a[i];
    }
    if (i < count) {
        out[i] = a[i];
    }
}

HWY_ATTR void Reverse2Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Reverse2(d, va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i + 1 < count; i += 2) {
        out[i] = a[i + 1];
        out[i + 1] = a[i];
    }
    if (i < count) {
        out[i] = a[i];
    }
}

// Reverse4: Reverse in groups of 4
HWY_ATTR void Reverse4Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Reverse4(d, va);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback
    for (; i + 3 < count; i += 4) {
        out[i] = a[i + 3];
        out[i + 1] = a[i + 2];
        out[i + 2] = a[i + 1];
        out[i + 3] = a[i];
    }
    // Handle remaining
    for (; i < count; ++i) {
        out[i] = a[i];
    }
}

HWY_ATTR void Reverse4Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Reverse4(d, va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i + 3 < count; i += 4) {
        out[i] = a[i + 3];
        out[i + 1] = a[i + 2];
        out[i + 2] = a[i + 1];
        out[i + 3] = a[i];
    }
    for (; i < count; ++i) {
        out[i] = a[i];
    }
}

// Reverse8: Reverse in groups of 8 (for uint8)
HWY_ATTR void Reverse8Uint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                            size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::Reverse8(d, va);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback
    for (; i + 7 < count; i += 8) {
        for (size_t j = 0; j < 8; ++j) {
            out[i + j] = a[i + 7 - j];
        }
    }
    for (; i < count; ++i) {
        out[i] = a[i];
    }
}

// =============================================================================
// P1: DupEven, DupOdd - Duplicate even/odd lanes
// =============================================================================

// DupEven: out[2i] = out[2i+1] = a[2i]
HWY_ATTR void DupEvenFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::DupEven(va);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback
    for (; i + 1 < count; i += 2) {
        out[i] = a[i];
        out[i + 1] = a[i];
    }
    if (i < count) {
        out[i] = a[i];
    }
}

HWY_ATTR void DupEvenInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::DupEven(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i + 1 < count; i += 2) {
        out[i] = a[i];
        out[i + 1] = a[i];
    }
    if (i < count) {
        out[i] = a[i];
    }
}

// DupOdd: out[2i] = out[2i+1] = a[2i+1]
HWY_ATTR void DupOddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::DupOdd(va);
        hn::StoreU(vout, d, out + i);
    }
    // Scalar fallback
    for (; i + 1 < count; i += 2) {
        out[i] = a[i + 1];
        out[i + 1] = a[i + 1];
    }
    if (i < count) {
        out[i] = a[i];
    }
}

HWY_ATTR void DupOddInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vout = hn::DupOdd(va);
        hn::StoreU(vout, d, out + i);
    }
    for (; i + 1 < count; i += 2) {
        out[i] = a[i + 1];
        out[i + 1] = a[i + 1];
    }
    if (i < count) {
        out[i] = a[i];
    }
}

// =============================================================================
// P1: InterleaveLower, InterleaveUpper - Interleave halves
// =============================================================================

// InterleaveLower: Interleave lower halves of a and b
HWY_ATTR void InterleaveLowerFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                     const float* HWY_RESTRICT b, size_t count) {
    // Interleave: a0,b0,a1,b1,a2,b2,...
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = a[i];
        out[2 * i + 1] = b[i];
    }
}

HWY_ATTR void InterleaveLowerInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                   const int32_t* HWY_RESTRICT b, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = a[i];
        out[2 * i + 1] = b[i];
    }
}

// InterleaveUpper: Interleave upper halves of a and b
HWY_ATTR void InterleaveUpperFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                     const float* HWY_RESTRICT b, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = a[half + i];
        out[2 * i + 1] = b[half + i];
    }
}

HWY_ATTR void InterleaveUpperInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                   const int32_t* HWY_RESTRICT b, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = a[half + i];
        out[2 * i + 1] = b[half + i];
    }
}

// =============================================================================
// P1: Mask Logical Operations
// =============================================================================

// MaskNot: out[i] = ~mask[i]
HWY_ATTR void MaskNotUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                           size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vm = hn::LoadU(d, mask + i);
        const auto vout = hn::Not(vm);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = ~mask[i];
    }
}

// MaskAnd: out[i] = a[i] & b[i]
HWY_ATTR void MaskAndUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                           const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::And(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] & b[i];
    }
}

// MaskOr: out[i] = a[i] | b[i]
HWY_ATTR void MaskOrUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                          const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::Or(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] | b[i];
    }
}

// MaskXor: out[i] = a[i] ^ b[i]
HWY_ATTR void MaskXorUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                           const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::Xor(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] ^ b[i];
    }
}

// MaskAndNot: out[i] = ~a[i] & b[i]
HWY_ATTR void MaskAndNotUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                              const uint8_t* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vout = hn::AndNot(va, vb);
        hn::StoreU(vout, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (~a[i]) & b[i];
    }
}

// =============================================================================
// P2: AddSub - Alternating add/subtract (no FMA)
// Even lanes: subtract, Odd lanes: add
// =============================================================================

HWY_ATTR void AddSubFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                            const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (i % 2 == 0) {
            out[i] = a[i] - b[i];  // Even: subtract
        } else {
            out[i] = a[i] + b[i];  // Odd: add
        }
    }
}

HWY_ATTR void AddSubFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                            const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (i % 2 == 0) {
            out[i] = a[i] - b[i];
        } else {
            out[i] = a[i] + b[i];
        }
    }
}

// =============================================================================
// P2: MinMagnitude, MaxMagnitude - Compare by absolute value
// =============================================================================

HWY_ATTR void MinMagnitudeFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                  const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float abs_a = std::fabs(a[i]);
        float abs_b = std::fabs(b[i]);
        out[i] = (abs_a <= abs_b) ? a[i] : b[i];
    }
}

HWY_ATTR void MinMagnitudeFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                  const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        double abs_a = std::fabs(a[i]);
        double abs_b = std::fabs(b[i]);
        out[i] = (abs_a <= abs_b) ? a[i] : b[i];
    }
}

HWY_ATTR void MaxMagnitudeFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                  const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float abs_a = std::fabs(a[i]);
        float abs_b = std::fabs(b[i]);
        out[i] = (abs_a >= abs_b) ? a[i] : b[i];
    }
}

HWY_ATTR void MaxMagnitudeFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                  const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        double abs_a = std::fabs(a[i]);
        double abs_b = std::fabs(b[i]);
        out[i] = (abs_a >= abs_b) ? a[i] : b[i];
    }
}

// =============================================================================
// P2: MaskedLoad, MaskedStore, BlendedStore
// =============================================================================

// MaskedLoad: out[i] = mask[i] ? src[i] : fallback
HWY_ATTR void MaskedLoadFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                const uint8_t* HWY_RESTRICT mask, float fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? src[i] : fallback;
    }
}

HWY_ATTR void MaskedLoadFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                const uint8_t* HWY_RESTRICT mask, double fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? src[i] : fallback;
    }
}

HWY_ATTR void MaskedLoadInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                              const uint8_t* HWY_RESTRICT mask, int32_t fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? src[i] : fallback;
    }
}

// MaskedStore: dst[i] = mask[i] ? src[i] : dst[i] (only store where mask is true)
HWY_ATTR void MaskedStoreFloat32(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                 const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[i] = src[i];
        }
    }
}

HWY_ATTR void MaskedStoreFloat64(double* HWY_RESTRICT dst, const double* HWY_RESTRICT src,
                                 const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[i] = src[i];
        }
    }
}

HWY_ATTR void MaskedStoreInt32(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                               const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[i] = src[i];
        }
    }
}

// BlendedStore: dst[i] = mask[i] ? new_val[i] : dst[i]
HWY_ATTR void BlendedStoreFloat32(float* HWY_RESTRICT dst, const float* HWY_RESTRICT new_val,
                                  const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[i] = new_val[i];
        }
    }
}

HWY_ATTR void BlendedStoreFloat64(double* HWY_RESTRICT dst, const double* HWY_RESTRICT new_val,
                                  const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[i] = new_val[i];
        }
    }
}

HWY_ATTR void BlendedStoreInt32(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT new_val,
                                const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[i] = new_val[i];
        }
    }
}

// =============================================================================
// P0: WidenMulPairwiseAdd Operation
// =============================================================================

// WidenMulPairwiseAdd: Multiply pairs of elements, widen, and add adjacent results
// For int16: (a[2i]*b[2i]) + (a[2i+1]*b[2i+1]) -> int32
HWY_ATTR void WidenMulPairwiseAddInt16(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                       const int16_t* HWY_RESTRICT b, size_t count) {
    // count is the number of int16 input elements, output is count/2 int32 elements
    for (size_t i = 0; i < count / 2; ++i) {
        int32_t prod0 = static_cast<int32_t>(a[2 * i]) * static_cast<int32_t>(b[2 * i]);
        int32_t prod1 = static_cast<int32_t>(a[2 * i + 1]) * static_cast<int32_t>(b[2 * i + 1]);
        out[i] = prod0 + prod1;
    }
}

HWY_ATTR void WidenMulPairwiseAddUint16(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                                        const uint16_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        uint32_t prod0 = static_cast<uint32_t>(a[2 * i]) * static_cast<uint32_t>(b[2 * i]);
        uint32_t prod1 = static_cast<uint32_t>(a[2 * i + 1]) * static_cast<uint32_t>(b[2 * i + 1]);
        out[i] = prod0 + prod1;
    }
}

// =============================================================================
// P1: BroadcastLane Operation
// =============================================================================

// BroadcastLane: Broadcast a single lane to all lanes
HWY_ATTR void BroadcastLaneFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                   size_t lane, size_t count) {
    float val = a[lane];
    for (size_t i = 0; i < count; ++i) {
        out[i] = val;
    }
}

HWY_ATTR void BroadcastLaneFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                   size_t lane, size_t count) {
    double val = a[lane];
    for (size_t i = 0; i < count; ++i) {
        out[i] = val;
    }
}

HWY_ATTR void BroadcastLaneInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                 size_t lane, size_t count) {
    int32_t val = a[lane];
    for (size_t i = 0; i < count; ++i) {
        out[i] = val;
    }
}

// =============================================================================
// P1: Slide Operations
// =============================================================================

// Slide1Up: Shift elements up by 1, first element becomes 0
HWY_ATTR void Slide1UpFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return;
    out[0] = 0.0f;
    for (size_t i = 1; i < count; ++i) {
        out[i] = a[i - 1];
    }
}

HWY_ATTR void Slide1UpInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            size_t count) {
    if (count == 0)
        return;
    out[0] = 0;
    for (size_t i = 1; i < count; ++i) {
        out[i] = a[i - 1];
    }
}

// Slide1Down: Shift elements down by 1, last element becomes 0
HWY_ATTR void Slide1DownFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                size_t count) {
    if (count == 0)
        return;
    for (size_t i = 0; i < count - 1; ++i) {
        out[i] = a[i + 1];
    }
    out[count - 1] = 0.0f;
}

HWY_ATTR void Slide1DownInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              size_t count) {
    if (count == 0)
        return;
    for (size_t i = 0; i < count - 1; ++i) {
        out[i] = a[i + 1];
    }
    out[count - 1] = 0;
}

// =============================================================================
// P1: Concat Operations
// =============================================================================

// ConcatLowerUpper: Concatenate lower half of a with upper half of b
HWY_ATTR void ConcatLowerUpperFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                      const float* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    // Lower half from a
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[i];
    }
    // Upper half from b
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[half + i];
    }
}

HWY_ATTR void ConcatLowerUpperInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                    const int32_t* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[i];
    }
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[half + i];
    }
}

// ConcatUpperLower: Concatenate upper half of a with lower half of b
HWY_ATTR void ConcatUpperLowerFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                      const float* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    // Lower half from upper half of a
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[half + i];
    }
    // Upper half from lower half of b
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[i];
    }
}

HWY_ATTR void ConcatUpperLowerInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                    const int32_t* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[half + i];
    }
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[i];
    }
}

// ConcatEven: Take even lanes from a and b, concatenate
HWY_ATTR void ConcatEvenFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                const float* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    // First half: even elements from a
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[2 * i];
    }
    // Second half: even elements from b
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[2 * i];
    }
}

HWY_ATTR void ConcatEvenInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                              const int32_t* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[2 * i];
    }
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[2 * i];
    }
}

// ConcatOdd: Take odd lanes from a and b, concatenate
HWY_ATTR void ConcatOddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                               const float* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    // First half: odd elements from a
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[2 * i + 1];
    }
    // Second half: odd elements from b
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[2 * i + 1];
    }
}

HWY_ATTR void ConcatOddInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                             const int32_t* HWY_RESTRICT b, size_t count) {
    size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = a[2 * i + 1];
    }
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = b[2 * i + 1];
    }
}

// =============================================================================
// P1: Mask Utility Operations
// =============================================================================

// FindKnownFirstTrue: Find the index of the first true mask element
// Returns SIZE_MAX if no true element (caller should verify mask has at least one true)
HWY_ATTR size_t FindKnownFirstTrueImpl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            return i;
        }
    }
    return static_cast<size_t>(-1);  // Should not happen if mask has known true
}

// FindKnownLastTrue: Find the index of the last true mask element
HWY_ATTR size_t FindKnownLastTrueImpl(const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = count; i > 0; --i) {
        if (mask[i - 1]) {
            return i - 1;
        }
    }
    return static_cast<size_t>(-1);  // Should not happen if mask has known true
}

// StoreMaskBits: Store mask as packed bits (1 bit per lane)
HWY_ATTR size_t StoreMaskBitsImpl(uint8_t* HWY_RESTRICT bits_out, const uint8_t* HWY_RESTRICT mask,
                                  size_t count) {
    size_t num_bytes = (count + 7) / 8;
    for (size_t byte_idx = 0; byte_idx < num_bytes; ++byte_idx) {
        uint8_t byte_val = 0;
        for (size_t bit = 0; bit < 8 && (byte_idx * 8 + bit) < count; ++bit) {
            if (mask[byte_idx * 8 + bit]) {
                byte_val |= (1u << bit);
            }
        }
        bits_out[byte_idx] = byte_val;
    }
    return num_bytes;
}

// LoadMaskBits: Load packed bits as mask (1 bit per lane)
HWY_ATTR void LoadMaskBitsImpl(uint8_t* HWY_RESTRICT mask_out, const uint8_t* HWY_RESTRICT bits,
                               size_t count) {
    for (size_t i = 0; i < count; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        mask_out[i] = ((bits[byte_idx] >> bit_idx) & 1) ? 0xFF : 0x00;
    }
}

// =============================================================================
// P1: CompressBlendedStore and CompressNot Operations
// =============================================================================

// CompressBlendedStore: Compress and blend-store (only updates masked positions)
HWY_ATTR size_t CompressBlendedStoreFloat32(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                            const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[out_idx++] = src[i];
        }
    }
    return out_idx;
}

HWY_ATTR size_t CompressBlendedStoreInt32(int32_t* HWY_RESTRICT dst,
                                          const int32_t* HWY_RESTRICT src,
                                          const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            dst[out_idx++] = src[i];
        }
    }
    return out_idx;
}

// CompressNot: Compress where mask is false (inverse of Compress)
HWY_ATTR size_t CompressNotFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                   const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!mask[i]) {
            out[out_idx++] = src[i];
        }
    }
    return out_idx;
}

HWY_ATTR size_t CompressNotInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                 const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!mask[i]) {
            out[out_idx++] = src[i];
        }
    }
    return out_idx;
}

// =============================================================================
// P1: InterleaveEven and InterleaveOdd Operations
// =============================================================================

// InterleaveEven: Take even elements from a and b, interleave them
HWY_ATTR void InterleaveEvenFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        out[2 * i] = a[2 * i];
        out[2 * i + 1] = b[2 * i];
    }
}

HWY_ATTR void InterleaveEvenInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                  const int32_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        out[2 * i] = a[2 * i];
        out[2 * i + 1] = b[2 * i];
    }
}

// InterleaveOdd: Take odd elements from a and b, interleave them
HWY_ATTR void InterleaveOddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                   const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        out[2 * i] = a[2 * i + 1];
        out[2 * i + 1] = b[2 * i + 1];
    }
}

HWY_ATTR void InterleaveOddInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                 const int32_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        out[2 * i] = a[2 * i + 1];
        out[2 * i + 1] = b[2 * i + 1];
    }
}

// =============================================================================
// P1: Shuffle Operations (4-element block shuffles)
// =============================================================================

// Shuffle0123: Identity shuffle within 4-element blocks [0,1,2,3] -> [0,1,2,3]
HWY_ATTR void Shuffle0123Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                                 size_t count) {
    // Identity shuffle - copy as-is (useful as baseline)
    for (size_t i = 0; i < count; ++i) {
        out[i] = in[i];
    }
}

HWY_ATTR void Shuffle0123Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in,
                               size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = in[i];
    }
}

// Shuffle2301: Swap 2-element pairs within 4-element blocks [0,1,2,3] -> [2,3,0,1]
HWY_ATTR void Shuffle2301Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                                 size_t count) {
    size_t full_blocks = (count / 4) * 4;
    for (size_t i = 0; i < full_blocks; i += 4) {
        out[i + 0] = in[i + 2];
        out[i + 1] = in[i + 3];
        out[i + 2] = in[i + 0];
        out[i + 3] = in[i + 1];
    }
    // Handle remainder (partial block)
    for (size_t i = full_blocks; i < count; ++i) {
        out[i] = in[i];
    }
}

HWY_ATTR void Shuffle2301Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in,
                               size_t count) {
    size_t full_blocks = (count / 4) * 4;
    for (size_t i = 0; i < full_blocks; i += 4) {
        out[i + 0] = in[i + 2];
        out[i + 1] = in[i + 3];
        out[i + 2] = in[i + 0];
        out[i + 3] = in[i + 1];
    }
    for (size_t i = full_blocks; i < count; ++i) {
        out[i] = in[i];
    }
}

// Shuffle1032: Swap adjacent elements within 4-element blocks [0,1,2,3] -> [1,0,3,2]
HWY_ATTR void Shuffle1032Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                                 size_t count) {
    size_t full_blocks = (count / 4) * 4;
    for (size_t i = 0; i < full_blocks; i += 4) {
        out[i + 0] = in[i + 1];
        out[i + 1] = in[i + 0];
        out[i + 2] = in[i + 3];
        out[i + 3] = in[i + 2];
    }
    for (size_t i = full_blocks; i < count; ++i) {
        out[i] = in[i];
    }
}

HWY_ATTR void Shuffle1032Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in,
                               size_t count) {
    size_t full_blocks = (count / 4) * 4;
    for (size_t i = 0; i < full_blocks; i += 4) {
        out[i + 0] = in[i + 1];
        out[i + 1] = in[i + 0];
        out[i + 2] = in[i + 3];
        out[i + 3] = in[i + 2];
    }
    for (size_t i = full_blocks; i < count; ++i) {
        out[i] = in[i];
    }
}

// Shuffle01: Identity within 2-element blocks (for 64-bit lanes like float64)
// [0,1] -> [0,1]
HWY_ATTR void Shuffle01Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT in,
                               size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = in[i];
    }
}

HWY_ATTR void Shuffle01Int64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT in,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = in[i];
    }
}

// Shuffle10: Swap within 2-element blocks (for 64-bit lanes like float64)
// [0,1] -> [1,0]
HWY_ATTR void Shuffle10Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT in,
                               size_t count) {
    size_t full_blocks = (count / 2) * 2;
    for (size_t i = 0; i < full_blocks; i += 2) {
        out[i + 0] = in[i + 1];
        out[i + 1] = in[i + 0];
    }
    // Handle odd element
    if (count > full_blocks) {
        out[full_blocks] = in[full_blocks];
    }
}

HWY_ATTR void Shuffle10Int64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT in,
                             size_t count) {
    size_t full_blocks = (count / 2) * 2;
    for (size_t i = 0; i < full_blocks; i += 2) {
        out[i + 0] = in[i + 1];
        out[i + 1] = in[i + 0];
    }
    if (count > full_blocks) {
        out[full_blocks] = in[full_blocks];
    }
}

// =============================================================================
// P1: TableLookupBytes and TableLookupLanes Operations
// =============================================================================

// TableLookupBytes: Byte-level table lookup using indices
HWY_ATTR void TableLookupBytesUint8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT table,
                                    const uint8_t* HWY_RESTRICT indices, size_t count,
                                    size_t table_size) {
    for (size_t i = 0; i < count; ++i) {
        size_t idx = indices[i];
        out[i] = (idx < table_size) ? table[idx] : 0;
    }
}

// TableLookupLanes: Lane-level table lookup using indices
HWY_ATTR void TableLookupLanesFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT table,
                                      const int32_t* HWY_RESTRICT indices, size_t count,
                                      size_t table_size) {
    for (size_t i = 0; i < count; ++i) {
        int32_t idx = indices[i];
        out[i] = (idx >= 0 && static_cast<size_t>(idx) < table_size) ? table[idx] : 0.0f;
    }
}

HWY_ATTR void TableLookupLanesInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT table,
                                    const int32_t* HWY_RESTRICT indices, size_t count,
                                    size_t table_size) {
    for (size_t i = 0; i < count; ++i) {
        int32_t idx = indices[i];
        out[i] = (idx >= 0 && static_cast<size_t>(idx) < table_size) ? table[idx] : 0;
    }
}

// =============================================================================
// P1: Mask Set Operations
// =============================================================================

// SetBeforeFirst: Set mask true for all lanes before the first true lane
HWY_ATTR void SetBeforeFirstImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                 size_t count) {
    bool found_first = false;
    for (size_t i = 0; i < count; ++i) {
        if (!found_first && mask[i]) {
            found_first = true;
        }
        out[i] = (!found_first) ? 0xFF : 0x00;
    }
}

// SetAtOrBeforeFirst: Set mask true for all lanes at or before the first true lane
HWY_ATTR void SetAtOrBeforeFirstImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                     size_t count) {
    bool found_first = false;
    for (size_t i = 0; i < count; ++i) {
        if (!found_first && mask[i]) {
            out[i] = 0xFF;  // Include the first true lane
            found_first = true;
        } else {
            out[i] = (!found_first) ? 0xFF : 0x00;
        }
    }
}

// SetOnlyFirst: Set mask true only for the first true lane
HWY_ATTR void SetOnlyFirstImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               size_t count) {
    bool found_first = false;
    for (size_t i = 0; i < count; ++i) {
        if (!found_first && mask[i]) {
            out[i] = 0xFF;
            found_first = true;
        } else {
            out[i] = 0x00;
        }
    }
}

// SetAtOrAfterFirst: Set mask true for all lanes at or after the first true lane
HWY_ATTR void SetAtOrAfterFirstImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                    size_t count) {
    bool found_first = false;
    for (size_t i = 0; i < count; ++i) {
        if (!found_first && mask[i]) {
            found_first = true;
        }
        out[i] = found_first ? 0xFF : 0x00;
    }
}

// =============================================================================
// P1: Masked Reduction Operations
// =============================================================================

// MaskedReduceSum: Sum of elements where mask is true
HWY_ATTR float MaskedReduceSumFloat32(const float* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            sum += src[i];
        }
    }
    return sum;
}

HWY_ATTR double MaskedReduceSumFloat64(const double* HWY_RESTRICT src,
                                       const uint8_t* HWY_RESTRICT mask, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            sum += src[i];
        }
    }
    return sum;
}

HWY_ATTR int32_t MaskedReduceSumInt32(const int32_t* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    int32_t sum = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            sum += src[i];
        }
    }
    return sum;
}

// MaskedReduceMin: Minimum of elements where mask is true
HWY_ATTR float MaskedReduceMinFloat32(const float* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    float min_val = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] && src[i] < min_val) {
            min_val = src[i];
        }
    }
    return min_val;
}

HWY_ATTR double MaskedReduceMinFloat64(const double* HWY_RESTRICT src,
                                       const uint8_t* HWY_RESTRICT mask, size_t count) {
    double min_val = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] && src[i] < min_val) {
            min_val = src[i];
        }
    }
    return min_val;
}

HWY_ATTR int32_t MaskedReduceMinInt32(const int32_t* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    int32_t min_val = std::numeric_limits<int32_t>::max();
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] && src[i] < min_val) {
            min_val = src[i];
        }
    }
    return min_val;
}

// MaskedReduceMax: Maximum of elements where mask is true
HWY_ATTR float MaskedReduceMaxFloat32(const float* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] && src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

HWY_ATTR double MaskedReduceMaxFloat64(const double* HWY_RESTRICT src,
                                       const uint8_t* HWY_RESTRICT mask, size_t count) {
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] && src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

HWY_ATTR int32_t MaskedReduceMaxInt32(const int32_t* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    int32_t max_val = std::numeric_limits<int32_t>::min();
    for (size_t i = 0; i < count; ++i) {
        if (mask[i] && src[i] > max_val) {
            max_val = src[i];
        }
    }
    return max_val;
}

// =============================================================================
// Remaining P1 Operations
// =============================================================================

// TwoTablesLookupLanes: Lookup from two tables based on index high bit
HWY_ATTR void TwoTablesLookupLanesFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT table0,
                                          const float* HWY_RESTRICT table1,
                                          const int32_t* HWY_RESTRICT indices, size_t count,
                                          size_t table_size) {
    for (size_t i = 0; i < count; ++i) {
        int32_t idx = indices[i];
        if (idx >= 0 && static_cast<size_t>(idx) < table_size) {
            out[i] = table0[idx];
        } else if (idx >= static_cast<int32_t>(table_size) &&
                   static_cast<size_t>(idx) < 2 * table_size) {
            out[i] = table1[idx - table_size];
        } else {
            out[i] = 0.0f;
        }
    }
}

HWY_ATTR void TwoTablesLookupLanesInt32(int32_t* HWY_RESTRICT out,
                                        const int32_t* HWY_RESTRICT table0,
                                        const int32_t* HWY_RESTRICT table1,
                                        const int32_t* HWY_RESTRICT indices, size_t count,
                                        size_t table_size) {
    for (size_t i = 0; i < count; ++i) {
        int32_t idx = indices[i];
        if (idx >= 0 && static_cast<size_t>(idx) < table_size) {
            out[i] = table0[idx];
        } else if (idx >= static_cast<int32_t>(table_size) &&
                   static_cast<size_t>(idx) < 2 * table_size) {
            out[i] = table1[idx - table_size];
        } else {
            out[i] = 0;
        }
    }
}

// CompressBits: Compress using packed bits mask
HWY_ATTR size_t CompressBitsFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                    const uint8_t* HWY_RESTRICT bits, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        if ((bits[byte_idx] >> bit_idx) & 1) {
            out[out_idx++] = src[i];
        }
    }
    return out_idx;
}

HWY_ATTR size_t CompressBitsInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                  const uint8_t* HWY_RESTRICT bits, size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        if ((bits[byte_idx] >> bit_idx) & 1) {
            out[out_idx++] = src[i];
        }
    }
    return out_idx;
}

// CompressBitsStore: Compress using packed bits and store
HWY_ATTR size_t CompressBitsStoreFloat32(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                         const uint8_t* HWY_RESTRICT bits, size_t count) {
    return CompressBitsFloat32(dst, src, bits, count);
}

HWY_ATTR size_t CompressBitsStoreInt32(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                                       const uint8_t* HWY_RESTRICT bits, size_t count) {
    return CompressBitsInt32(dst, src, bits, count);
}

// LoadExpand: Load and expand based on mask
HWY_ATTR void LoadExpandFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t src_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = src[src_idx++];
        } else {
            out[i] = 0.0f;
        }
    }
}

HWY_ATTR void LoadExpandInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                              const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t src_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = src[src_idx++];
        } else {
            out[i] = 0;
        }
    }
}

// PairwiseSub: Subtract adjacent pairs
HWY_ATTR void PairwiseSubFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                 const float* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        out[2 * i] = a[2 * i] - a[2 * i + 1];
        out[2 * i + 1] = b[2 * i] - b[2 * i + 1];
    }
}

HWY_ATTR void PairwiseSubInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                               const int32_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 2; ++i) {
        out[2 * i] = a[2 * i] - a[2 * i + 1];
        out[2 * i + 1] = b[2 * i] - b[2 * i + 1];
    }
}

// SumsOfAdjQuadAbsDiff: Sum of absolute differences of adjacent quads (for uint8)
HWY_ATTR void SumsOfAdjQuadAbsDiffUint8(uint16_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                                        const uint8_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count / 4; ++i) {
        uint16_t sum = 0;
        for (size_t j = 0; j < 4; ++j) {
            int diff = static_cast<int>(a[4 * i + j]) - static_cast<int>(b[4 * i + j]);
            sum += static_cast<uint16_t>(diff < 0 ? -diff : diff);
        }
        out[i] = sum;
    }
}

// =============================================================================
// P2: Math Functions
// =============================================================================

// Cbrt: Cube root
HWY_ATTR void CbrtFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::cbrt(a[i]);
    }
}

HWY_ATTR void CbrtFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::cbrt(a[i]);
    }
}

// Erf: Error function
HWY_ATTR void ErfFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::erf(a[i]);
    }
}

HWY_ATTR void ErfFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::erf(a[i]);
    }
}

// Erfc: Complementary error function
HWY_ATTR void ErfcFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::erfc(a[i]);
    }
}

HWY_ATTR void ErfcFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::erfc(a[i]);
    }
}

// =============================================================================
// P2: Generation Operations
// =============================================================================

// IndicesFromVec: Create indices from mask vector
HWY_ATTR size_t IndicesFromVecImpl(int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                                   size_t count) {
    size_t idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            indices[idx++] = static_cast<int32_t>(i);
        }
    }
    return idx;
}

// IndicesFromNotVec: Create indices from negated mask vector
HWY_ATTR size_t IndicesFromNotVecImpl(int32_t* HWY_RESTRICT indices,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!mask[i]) {
            indices[idx++] = static_cast<int32_t>(i);
        }
    }
    return idx;
}

// =============================================================================
// P2: Type Conversions
// =============================================================================

// PromoteLowerTo: Promote lower half to wider type
HWY_ATTR void PromoteLowerToFloat64(double* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(a[i]);
    }
}

HWY_ATTR void PromoteLowerToInt32(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                  size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[i]);
    }
}

HWY_ATTR void PromoteLowerToInt64(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                                  size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int64_t>(a[i]);
    }
}

// PromoteUpperTo: Promote upper half to wider type
HWY_ATTR void PromoteUpperToFloat64(double* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    size_t count, size_t half) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(a[half + i]);
    }
}

HWY_ATTR void PromoteUpperToInt32(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                  size_t count, size_t half) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[half + i]);
    }
}

// PromoteEvenTo: Promote even lanes to wider type
HWY_ATTR void PromoteEvenToFloat64(double* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                   size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(a[2 * i]);
    }
}

HWY_ATTR void PromoteEvenToInt32(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                 size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[2 * i]);
    }
}

// PromoteOddTo: Promote odd lanes to wider type
HWY_ATTR void PromoteOddToFloat64(double* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                  size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<double>(a[2 * i + 1]);
    }
}

HWY_ATTR void PromoteOddToInt32(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(a[2 * i + 1]);
    }
}

// =============================================================================
// P2: Additional Arithmetic
// =============================================================================

// Mod: Integer modulo
HWY_ATTR void ModInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (b[i] != 0) ? (a[i] % b[i]) : 0;
    }
}

HWY_ATTR void ModInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (b[i] != 0) ? (a[i] % b[i]) : 0;
    }
}

// SaturatedNeg: Saturated negation
HWY_ATTR void SaturatedNegInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                               size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] == std::numeric_limits<int8_t>::min()) ? std::numeric_limits<int8_t>::max()
                                                              : -a[i];
    }
}

HWY_ATTR void SaturatedNegInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] == std::numeric_limits<int16_t>::min()) ? std::numeric_limits<int16_t>::max()
                                                               : -a[i];
    }
}

// SaturatedAbs: Saturated absolute value
HWY_ATTR void SaturatedAbsInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                               size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] == std::numeric_limits<int8_t>::min()) {
            out[i] = std::numeric_limits<int8_t>::max();
        } else {
            out[i] = (a[i] < 0) ? -a[i] : a[i];
        }
    }
}

HWY_ATTR void SaturatedAbsInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] == std::numeric_limits<int16_t>::min()) {
            out[i] = std::numeric_limits<int16_t>::max();
        } else {
            out[i] = (a[i] < 0) ? -a[i] : a[i];
        }
    }
}

// =============================================================================
// P2: Bitwise Operations
// =============================================================================

// Xor3: Three-way XOR
HWY_ATTR void Xor3Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                        const int32_t* HWY_RESTRICT b, const int32_t* HWY_RESTRICT c,
                        size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] ^ b[i] ^ c[i];
    }
}

HWY_ATTR void Xor3Int64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                        const int64_t* HWY_RESTRICT b, const int64_t* HWY_RESTRICT c,
                        size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] ^ b[i] ^ c[i];
    }
}

// Or3: Three-way OR
HWY_ATTR void Or3Int32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, const int32_t* HWY_RESTRICT c, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] | b[i] | c[i];
    }
}

HWY_ATTR void Or3Int64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                       const int64_t* HWY_RESTRICT b, const int64_t* HWY_RESTRICT c, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] | b[i] | c[i];
    }
}

// OrAnd: (a | b) & c
HWY_ATTR void OrAndInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                         const int32_t* HWY_RESTRICT b, const int32_t* HWY_RESTRICT c,
                         size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] | b[i]) & c[i];
    }
}

HWY_ATTR void OrAndInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                         const int64_t* HWY_RESTRICT b, const int64_t* HWY_RESTRICT c,
                         size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] | b[i]) & c[i];
    }
}

// ReverseBits: Reverse bits in each element
HWY_ATTR void ReverseBitsUint32(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t v = a[i];
        uint32_t r = 0;
        for (int j = 0; j < 32; ++j) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        out[i] = r;
    }
}

HWY_ATTR void ReverseBitsUint64(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                                size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t v = a[i];
        uint64_t r = 0;
        for (int j = 0; j < 64; ++j) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        out[i] = r;
    }
}

// HighestSetBitIndex: Index of highest set bit (or -1 if zero)
HWY_ATTR void HighestSetBitIndexUint32(int32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] == 0) {
            out[i] = -1;
        } else {
            out[i] = 31 - __builtin_clz(a[i]);
        }
    }
}

HWY_ATTR void HighestSetBitIndexUint64(int32_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] == 0) {
            out[i] = -1;
        } else {
            out[i] = 63 - __builtin_clzll(a[i]);
        }
    }
}

// =============================================================================
// P2: Memory Operations
// =============================================================================

// LoadDup128: Load and duplicate 128 bits
HWY_ATTR void LoadDup128Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                size_t count) {
    // 128 bits = 4 floats
    size_t num_blocks = count / 4;
    for (size_t block = 0; block < num_blocks; ++block) {
        for (size_t i = 0; i < 4; ++i) {
            out[block * 4 + i] = src[i % 4];
        }
    }
}

HWY_ATTR void LoadDup128Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                size_t count) {
    // 128 bits = 2 doubles
    size_t num_blocks = count / 2;
    for (size_t block = 0; block < num_blocks; ++block) {
        for (size_t i = 0; i < 2; ++i) {
            out[block * 2 + i] = src[i % 2];
        }
    }
}

// GatherOffset: Gather using byte offsets
HWY_ATTR void GatherOffsetFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                                  const int32_t* HWY_RESTRICT offsets, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const char* ptr = reinterpret_cast<const char*>(base) + offsets[i];
        out[i] = *reinterpret_cast<const float*>(ptr);
    }
}

HWY_ATTR void GatherOffsetInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                                const int32_t* HWY_RESTRICT offsets, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const char* ptr = reinterpret_cast<const char*>(base) + offsets[i];
        out[i] = *reinterpret_cast<const int32_t*>(ptr);
    }
}

// ScatterOffset: Scatter using byte offsets
HWY_ATTR void ScatterOffsetFloat32(float* HWY_RESTRICT base, const float* HWY_RESTRICT values,
                                   const int32_t* HWY_RESTRICT offsets, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        char* ptr = reinterpret_cast<char*>(base) + offsets[i];
        *reinterpret_cast<float*>(ptr) = values[i];
    }
}

HWY_ATTR void ScatterOffsetInt32(int32_t* HWY_RESTRICT base, const int32_t* HWY_RESTRICT values,
                                 const int32_t* HWY_RESTRICT offsets, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        char* ptr = reinterpret_cast<char*>(base) + offsets[i];
        *reinterpret_cast<int32_t*>(ptr) = values[i];
    }
}

// MaskedGatherIndex: Gather with mask using lane indices
HWY_ATTR void MaskedGatherIndexFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                                       const int32_t* HWY_RESTRICT indices,
                                       const uint8_t* HWY_RESTRICT mask, float fallback,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = base[indices[i]];
        } else {
            out[i] = fallback;
        }
    }
}

HWY_ATTR void MaskedGatherIndexInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                                     const int32_t* HWY_RESTRICT indices,
                                     const uint8_t* HWY_RESTRICT mask, int32_t fallback,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = base[indices[i]];
        } else {
            out[i] = fallback;
        }
    }
}

// MaskedScatterIndex: Scatter with mask using lane indices
HWY_ATTR void MaskedScatterIndexFloat32(float* HWY_RESTRICT base, const float* HWY_RESTRICT values,
                                        const int32_t* HWY_RESTRICT indices,
                                        const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            base[indices[i]] = values[i];
        }
    }
}

HWY_ATTR void MaskedScatterIndexInt32(int32_t* HWY_RESTRICT base,
                                      const int32_t* HWY_RESTRICT values,
                                      const int32_t* HWY_RESTRICT indices,
                                      const uint8_t* HWY_RESTRICT mask, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            base[indices[i]] = values[i];
        }
    }
}

// SafeFillN: Safely fill N elements
HWY_ATTR void SafeFillNFloat32(float* HWY_RESTRICT out, float value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = value;
    }
}

HWY_ATTR void SafeFillNInt32(int32_t* HWY_RESTRICT out, int32_t value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = value;
    }
}

// SafeCopyN: Safely copy N elements
HWY_ATTR void SafeCopyNFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                               size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = src[i];
    }
}

HWY_ATTR void SafeCopyNInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = src[i];
    }
}

// =============================================================================
// P2: Special Operations
// =============================================================================

// MulByPow2: Multiply by power of 2
HWY_ATTR void MulByPow2Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                               const int32_t* HWY_RESTRICT exp, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::ldexp(a[i], exp[i]);
    }
}

HWY_ATTR void MulByPow2Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                               const int32_t* HWY_RESTRICT exp, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::ldexp(a[i], exp[i]);
    }
}

// GetExponent: Extract exponent
HWY_ATTR void GetExponentFloat32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                 size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int exp;
        std::frexp(a[i], &exp);
        out[i] = exp - 1;  // frexp returns exponent such that 0.5 <= |mantissa| < 1
    }
}

HWY_ATTR void GetExponentFloat64(int32_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                 size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int exp;
        std::frexp(a[i], &exp);
        out[i] = exp - 1;
    }
}

// SignBit: Extract sign bit
HWY_ATTR void SignBitFloat32(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &a[i], sizeof(float));
        out[i] = bits >> 31;
    }
}

HWY_ATTR void SignBitFloat64(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t bits;
        std::memcpy(&bits, &a[i], sizeof(double));
        out[i] = bits >> 63;
    }
}

// NaN: Create NaN values
HWY_ATTR void NaNFloat32(float* HWY_RESTRICT out, size_t count) {
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 0; i < count; ++i) {
        out[i] = nan_val;
    }
}

HWY_ATTR void NaNFloat64(double* HWY_RESTRICT out, size_t count) {
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < count; ++i) {
        out[i] = nan_val;
    }
}

// Inf: Create infinity values
HWY_ATTR void InfFloat32(float* HWY_RESTRICT out, size_t count) {
    float inf_val = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        out[i] = inf_val;
    }
}

HWY_ATTR void InfFloat64(double* HWY_RESTRICT out, size_t count) {
    double inf_val = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        out[i] = inf_val;
    }
}

// =============================================================================
// P3: Complex Number Operations
// =============================================================================

// ComplexConj: Complex conjugate (negate imaginary parts)
HWY_ATTR void ComplexConjFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                 size_t count) {
    // Complex numbers stored as [real, imag, real, imag, ...]
    for (size_t i = 0; i < count; i += 2) {
        out[i] = a[i];           // real part
        out[i + 1] = -a[i + 1];  // imaginary part negated
    }
}

HWY_ATTR void ComplexConjFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                 size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        out[i] = a[i];
        out[i + 1] = -a[i + 1];
    }
}

// MulComplex: Complex multiplication
HWY_ATTR void MulComplexFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                const float* HWY_RESTRICT b, size_t count) {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    for (size_t i = 0; i < count; i += 2) {
        float ar = a[i], ai = a[i + 1];
        float br = b[i], bi = b[i + 1];
        out[i] = ar * br - ai * bi;      // real
        out[i + 1] = ar * bi + ai * br;  // imag
    }
}

HWY_ATTR void MulComplexFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        double ar = a[i], ai = a[i + 1];
        double br = b[i], bi = b[i + 1];
        out[i] = ar * br - ai * bi;
        out[i + 1] = ar * bi + ai * br;
    }
}

// MulComplexAdd: Complex multiply-add
HWY_ATTR void MulComplexAddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                   const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                                   size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        float ar = a[i], ai = a[i + 1];
        float br = b[i], bi = b[i + 1];
        out[i] = ar * br - ai * bi + c[i];
        out[i + 1] = ar * bi + ai * br + c[i + 1];
    }
}

// =============================================================================
// P3: Additional Saturation Operations
// =============================================================================

// SaturatedAdd for int8
HWY_ATTR void SaturatedAddInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                               const int8_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int16_t sum = static_cast<int16_t>(a[i]) + static_cast<int16_t>(b[i]);
        if (sum > 127)
            sum = 127;
        if (sum < -128)
            sum = -128;
        out[i] = static_cast<int8_t>(sum);
    }
}

// SaturatedSub for int8
HWY_ATTR void SaturatedSubInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                               const int8_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int16_t diff = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
        if (diff > 127)
            diff = 127;
        if (diff < -128)
            diff = -128;
        out[i] = static_cast<int8_t>(diff);
    }
}

// SaturatedAdd for uint16
HWY_ATTR void SaturatedAddUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                                 const uint16_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t sum = static_cast<uint32_t>(a[i]) + static_cast<uint32_t>(b[i]);
        if (sum > 65535)
            sum = 65535;
        out[i] = static_cast<uint16_t>(sum);
    }
}

// SaturatedSub for uint16
HWY_ATTR void SaturatedSubUint16(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                                 const uint16_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        if (diff < 0)
            diff = 0;
        out[i] = static_cast<uint16_t>(diff);
    }
}

// =============================================================================
// P3: Block Operations
// =============================================================================

// SlideUpBlocks: Slide up by blocks (128 bits)
HWY_ATTR void SlideUpBlocksFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                   size_t num_blocks, size_t count) {
    size_t shift = num_blocks * 4;  // 4 floats per 128-bit block
    for (size_t i = count; i > shift; --i) {
        out[i - 1] = a[i - 1 - shift];
    }
    for (size_t i = 0; i < shift && i < count; ++i) {
        out[i] = 0.0f;
    }
}

// SlideDownBlocks: Slide down by blocks (128 bits)
HWY_ATTR void SlideDownBlocksFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                     size_t num_blocks, size_t count) {
    size_t shift = num_blocks * 4;
    for (size_t i = 0; i + shift < count; ++i) {
        out[i] = a[i + shift];
    }
    for (size_t i = (count > shift) ? count - shift : 0; i < count; ++i) {
        out[i] = 0.0f;
    }
}

// CombineShiftRightLanes: Combine and shift right by lanes
HWY_ATTR void CombineShiftRightLanesFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT hi,
                                            const float* HWY_RESTRICT lo, size_t shift,
                                            size_t count) {
    // Result is like concatenating [hi, lo] and taking count elements starting at shift
    for (size_t i = 0; i < count; ++i) {
        size_t src_idx = i + shift;
        if (src_idx < count) {
            out[i] = lo[src_idx];
        } else {
            out[i] = hi[src_idx - count];
        }
    }
}

// =============================================================================
// P3: Additional Masked Operations
// =============================================================================

// MaskedSqrt
HWY_ATTR void MaskedSqrtFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT mask, float fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? std::sqrt(a[i]) : fallback;
    }
}

HWY_ATTR void MaskedSqrtFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                const uint8_t* HWY_RESTRICT mask, double fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? std::sqrt(a[i]) : fallback;
    }
}

// ZeroIfNegative: Set to zero if negative
HWY_ATTR void ZeroIfNegativeFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] < 0.0f) ? 0.0f : a[i];
    }
}

HWY_ATTR void ZeroIfNegativeFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] < 0.0) ? 0.0 : a[i];
    }
}

// =============================================================================
// P2: Additional Type Conversion Operations
// =============================================================================

// ReorderDemote2To: Demote two wider vectors to narrower, interleaving results
HWY_ATTR void ReorderDemote2ToInt32Int16(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT hi,
                                         const int32_t* HWY_RESTRICT lo, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = static_cast<int16_t>(lo[i]);
        out[2 * i + 1] = static_cast<int16_t>(hi[i]);
    }
}

HWY_ATTR void ReorderDemote2ToUint32Uint16(uint16_t* HWY_RESTRICT out,
                                           const uint32_t* HWY_RESTRICT hi,
                                           const uint32_t* HWY_RESTRICT lo, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = static_cast<uint16_t>(lo[i]);
        out[2 * i + 1] = static_cast<uint16_t>(hi[i]);
    }
}

// OrderedTruncate2To: Truncate two wider vectors, lo first then hi
HWY_ATTR void OrderedTruncate2ToInt32Int16(int16_t* HWY_RESTRICT out,
                                           const int32_t* HWY_RESTRICT hi,
                                           const int32_t* HWY_RESTRICT lo, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = static_cast<int16_t>(lo[i]);
    }
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = static_cast<int16_t>(hi[i]);
    }
}

HWY_ATTR void OrderedTruncate2ToUint32Uint16(uint16_t* HWY_RESTRICT out,
                                             const uint32_t* HWY_RESTRICT hi,
                                             const uint32_t* HWY_RESTRICT lo, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = static_cast<uint16_t>(lo[i]);
    }
    for (size_t i = 0; i < half; ++i) {
        out[half + i] = static_cast<uint16_t>(hi[i]);
    }
}

// ConvertInRangeTo: Convert float to int with truncation toward zero
HWY_ATTR void ConvertInRangeToFloat32Int32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                           size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(src[i]);
    }
}

HWY_ATTR void ConvertInRangeToFloat64Int64(int64_t* HWY_RESTRICT out,
                                           const double* HWY_RESTRICT src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int64_t>(src[i]);
    }
}

// ResizeBitCast: Bit cast between same-size types
HWY_ATTR void ResizeBitCastFloat32Uint32(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                         size_t count) {
    std::memcpy(out, src, count * sizeof(float));
}

HWY_ATTR void ResizeBitCastUint32Float32(float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src,
                                         size_t count) {
    std::memcpy(out, src, count * sizeof(uint32_t));
}

HWY_ATTR void ResizeBitCastFloat64Uint64(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                         size_t count) {
    std::memcpy(out, src, count * sizeof(double));
}

HWY_ATTR void ResizeBitCastUint64Float64(double* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src,
                                         size_t count) {
    std::memcpy(out, src, count * sizeof(uint64_t));
}

// =============================================================================
// P2: Additional Special Operations
// =============================================================================

// MulByFloorPow2: Multiply by floor of power of 2
HWY_ATTR void MulByFloorPow2Float32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    const float* HWY_RESTRICT pow2, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * pow2[i];
    }
}

HWY_ATTR void MulByFloorPow2Float64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                    const double* HWY_RESTRICT pow2, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * pow2[i];
    }
}

// GetBiasedExponent: Get the biased exponent from IEEE-754 float
HWY_ATTR void GetBiasedExponentFloat32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        out[i] = static_cast<int32_t>((bits >> 23) & 0xFF);
    }
}

HWY_ATTR void GetBiasedExponentFloat64(int32_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t bits;
        std::memcpy(&bits, &src[i], sizeof(double));
        out[i] = static_cast<int32_t>((bits >> 52) & 0x7FF);
    }
}

// MulFixedPoint15: Q15 fixed-point multiplication
HWY_ATTR void MulFixedPoint15Int16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                   const int16_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int32_t product = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        out[i] = static_cast<int16_t>(product >> 15);
    }
}

// MulRound: Multiply with rounding (standard multiplication for now)
HWY_ATTR void MulRoundInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                            const int16_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int16_t>(a[i] * b[i]);
    }
}

HWY_ATTR void MulRoundInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

// RoundingShiftRight: Shift right with rounding
HWY_ATTR void RoundingShiftRightInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT src,
                                      int shift, size_t count) {
    if (shift <= 0) {
        std::memcpy(out, src, count * sizeof(int16_t));
        return;
    }
    int16_t rounding = static_cast<int16_t>(1 << (shift - 1));
    for (size_t i = 0; i < count; ++i) {
        int16_t val = src[i];
        if (val >= 0) {
            out[i] = static_cast<int16_t>((val + rounding) >> shift);
        } else {
            // For negative numbers, use arithmetic right shift with rounding toward negative
            // infinity
            out[i] = static_cast<int16_t>(-(((-val) + rounding) >> shift));
        }
    }
}

HWY_ATTR void RoundingShiftRightInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                      int shift, size_t count) {
    if (shift <= 0) {
        std::memcpy(out, src, count * sizeof(int32_t));
        return;
    }
    int32_t rounding = 1 << (shift - 1);
    for (size_t i = 0; i < count; ++i) {
        int32_t val = src[i];
        if (val >= 0) {
            out[i] = (val + rounding) >> shift;
        } else {
            out[i] = -(((-val) + rounding) >> shift);
        }
    }
}

// =============================================================================
// P3: Additional Complex Number Operations
// =============================================================================

// MulComplexConj: Complex multiply with conjugate of second operand
HWY_ATTR void MulComplexConjFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                    const float* HWY_RESTRICT b, size_t count) {
    // (a + bi) * conj(c + di) = (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
    for (size_t i = 0; i < count; i += 2) {
        float ar = a[i], ai = a[i + 1];
        float br = b[i], bi = b[i + 1];
        out[i] = ar * br + ai * bi;      // real: ac + bd
        out[i + 1] = ai * br - ar * bi;  // imag: bc - ad
    }
}

HWY_ATTR void MulComplexConjFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                    const double* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        double ar = a[i], ai = a[i + 1];
        double br = b[i], bi = b[i + 1];
        out[i] = ar * br + ai * bi;
        out[i + 1] = ai * br - ar * bi;
    }
}

// MulComplexConjAdd: Complex multiply with conjugate and add
HWY_ATTR void MulComplexConjAddFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                                       size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        float ar = a[i], ai = a[i + 1];
        float br = b[i], bi = b[i + 1];
        out[i] = ar * br + ai * bi + c[i];
        out[i + 1] = ai * br - ar * bi + c[i + 1];
    }
}

HWY_ATTR void MulComplexConjAddFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                                       const double* HWY_RESTRICT b, const double* HWY_RESTRICT c,
                                       size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        double ar = a[i], ai = a[i + 1];
        double br = b[i], bi = b[i + 1];
        out[i] = ar * br + ai * bi + c[i];
        out[i + 1] = ai * br - ar * bi + c[i + 1];
    }
}

// =============================================================================
// P3: Per-Lane Block Shuffle
// =============================================================================

// Per4LaneBlockShuffle: Shuffle within each 4-lane block
HWY_ATTR void Per4LaneBlockShuffleFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                          uint8_t pattern, size_t count) {
    // Pattern is 2-bit indices: [idx3, idx2, idx1, idx0]
    // 0x1B = 00 01 10 11 = reverse (3,2,1,0)
    int idx0 = (pattern >> 0) & 0x3;
    int idx1 = (pattern >> 2) & 0x3;
    int idx2 = (pattern >> 4) & 0x3;
    int idx3 = (pattern >> 6) & 0x3;

    for (size_t block = 0; block + 4 <= count; block += 4) {
        out[block + 0] = src[block + idx0];
        out[block + 1] = src[block + idx1];
        out[block + 2] = src[block + idx2];
        out[block + 3] = src[block + idx3];
    }
}

HWY_ATTR void Per4LaneBlockShuffleInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                        uint8_t pattern, size_t count) {
    int idx0 = (pattern >> 0) & 0x3;
    int idx1 = (pattern >> 2) & 0x3;
    int idx2 = (pattern >> 4) & 0x3;
    int idx3 = (pattern >> 6) & 0x3;

    for (size_t block = 0; block + 4 <= count; block += 4) {
        out[block + 0] = src[block + idx0];
        out[block + 1] = src[block + idx1];
        out[block + 2] = src[block + idx2];
        out[block + 3] = src[block + idx3];
    }
}

// =============================================================================
// P3: Additional Masked Operations
// =============================================================================

// MaskedReciprocal
HWY_ATTR void MaskedReciprocalFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, float fallback,
                                      size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (1.0f / src[i]) : fallback;
    }
}

HWY_ATTR void MaskedReciprocalFloat64(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                                      const uint8_t* HWY_RESTRICT mask, double fallback,
                                      size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (1.0 / src[i]) : fallback;
    }
}

// MaskedShiftLeft
HWY_ATTR void MaskedShiftLeftInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                   const uint8_t* HWY_RESTRICT mask, int shift, int32_t fallback,
                                   size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (src[i] << shift) : fallback;
    }
}

HWY_ATTR void MaskedShiftLeftInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                                   const uint8_t* HWY_RESTRICT mask, int shift, int64_t fallback,
                                   size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (src[i] << shift) : fallback;
    }
}

// MaskedShiftRight
HWY_ATTR void MaskedShiftRightInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                    const uint8_t* HWY_RESTRICT mask, int shift, int32_t fallback,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (src[i] >> shift) : fallback;
    }
}

HWY_ATTR void MaskedShiftRightInt64(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                                    const uint8_t* HWY_RESTRICT mask, int shift, int64_t fallback,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (src[i] >> shift) : fallback;
    }
}

// MaskedSatAdd: Masked saturating add
HWY_ATTR void MaskedSatAddInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                               const int8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                               int8_t fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            int16_t sum = static_cast<int16_t>(a[i]) + static_cast<int16_t>(b[i]);
            if (sum > 127)
                sum = 127;
            if (sum < -128)
                sum = -128;
            out[i] = static_cast<int8_t>(sum);
        } else {
            out[i] = fallback;
        }
    }
}

HWY_ATTR void MaskedSatAddInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                const int16_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                                int16_t fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            int32_t sum = static_cast<int32_t>(a[i]) + static_cast<int32_t>(b[i]);
            if (sum > 32767)
                sum = 32767;
            if (sum < -32768)
                sum = -32768;
            out[i] = static_cast<int16_t>(sum);
        } else {
            out[i] = fallback;
        }
    }
}

// MaskedSatSub: Masked saturating subtract
HWY_ATTR void MaskedSatSubInt8(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                               const int8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                               int8_t fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            int16_t diff = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
            if (diff > 127)
                diff = 127;
            if (diff < -128)
                diff = -128;
            out[i] = static_cast<int8_t>(diff);
        } else {
            out[i] = fallback;
        }
    }
}

HWY_ATTR void MaskedSatSubInt16(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                                const int16_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                                int16_t fallback, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
            if (diff > 32767)
                diff = 32767;
            if (diff < -32768)
                diff = -32768;
            out[i] = static_cast<int16_t>(diff);
        } else {
            out[i] = fallback;
        }
    }
}

// =============================================================================
// P3: Masked Comparison Operations
// =============================================================================

// MaskedEq: Masked equality comparison
HWY_ATTR void MaskedEqFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] == b[i])) ? 0xFF : 0x00;
    }
}

HWY_ATTR void MaskedEqInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] == b[i])) ? 0xFF : 0x00;
    }
}

// MaskedNe: Masked not-equal comparison
HWY_ATTR void MaskedNeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] != b[i])) ? 0xFF : 0x00;
    }
}

HWY_ATTR void MaskedNeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] != b[i])) ? 0xFF : 0x00;
    }
}

// MaskedLt: Masked less-than comparison
HWY_ATTR void MaskedLtFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] < b[i])) ? 0xFF : 0x00;
    }
}

HWY_ATTR void MaskedLtInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] < b[i])) ? 0xFF : 0x00;
    }
}

// MaskedLe: Masked less-than-or-equal comparison
HWY_ATTR void MaskedLeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] <= b[i])) ? 0xFF : 0x00;
    }
}

HWY_ATTR void MaskedLeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] <= b[i])) ? 0xFF : 0x00;
    }
}

// MaskedGt: Masked greater-than comparison
HWY_ATTR void MaskedGtFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] > b[i])) ? 0xFF : 0x00;
    }
}

HWY_ATTR void MaskedGtInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] > b[i])) ? 0xFF : 0x00;
    }
}

// MaskedGe: Masked greater-than-or-equal comparison
HWY_ATTR void MaskedGeFloat32(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] >= b[i])) ? 0xFF : 0x00;
    }
}

HWY_ATTR void MaskedGeInt32(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                            const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] && (a[i] >= b[i])) ? 0xFF : 0x00;
    }
}

// =============================================================================
// P3.2: Cryptographic Operations
// =============================================================================

// AES S-Box
static constexpr uint8_t kAESSBox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

// AES Inverse S-Box
static constexpr uint8_t kAESInvSBox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

// GF(2^8) multiplication
HWY_INLINE uint8_t GFMul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < 8; ++i) {
        if (b & 1)
            p ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi)
            a ^= 0x1b;  // AES irreducible polynomial
        b >>= 1;
    }
    return p;
}

// AESRound: Perform one AES encryption round
HWY_ATTR void AESRoundImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                           const uint8_t* HWY_RESTRICT round_key, size_t count) {
    for (size_t block = 0; block + 16 <= count; block += 16) {
        uint8_t tmp[16];
        // SubBytes
        for (int i = 0; i < 16; ++i) {
            tmp[i] = kAESSBox[state[block + i]];
        }
        // ShiftRows
        uint8_t shifted[16];
        shifted[0] = tmp[0];
        shifted[4] = tmp[4];
        shifted[8] = tmp[8];
        shifted[12] = tmp[12];
        shifted[1] = tmp[5];
        shifted[5] = tmp[9];
        shifted[9] = tmp[13];
        shifted[13] = tmp[1];
        shifted[2] = tmp[10];
        shifted[6] = tmp[14];
        shifted[10] = tmp[2];
        shifted[14] = tmp[6];
        shifted[3] = tmp[15];
        shifted[7] = tmp[3];
        shifted[11] = tmp[7];
        shifted[15] = tmp[11];
        // MixColumns
        for (int col = 0; col < 4; ++col) {
            int c = col * 4;
            uint8_t a0 = shifted[c], a1 = shifted[c + 1], a2 = shifted[c + 2], a3 = shifted[c + 3];
            out[block + c] = GFMul(2, a0) ^ GFMul(3, a1) ^ a2 ^ a3;
            out[block + c + 1] = a0 ^ GFMul(2, a1) ^ GFMul(3, a2) ^ a3;
            out[block + c + 2] = a0 ^ a1 ^ GFMul(2, a2) ^ GFMul(3, a3);
            out[block + c + 3] = GFMul(3, a0) ^ a1 ^ a2 ^ GFMul(2, a3);
        }
        // AddRoundKey
        for (int i = 0; i < 16; ++i) {
            out[block + i] ^= round_key[i];
        }
    }
}

// AESLastRound: Perform final AES encryption round (no MixColumns)
HWY_ATTR void AESLastRoundImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                               const uint8_t* HWY_RESTRICT round_key, size_t count) {
    for (size_t block = 0; block + 16 <= count; block += 16) {
        uint8_t tmp[16];
        // SubBytes
        for (int i = 0; i < 16; ++i) {
            tmp[i] = kAESSBox[state[block + i]];
        }
        // ShiftRows
        out[block + 0] = tmp[0];
        out[block + 4] = tmp[4];
        out[block + 8] = tmp[8];
        out[block + 12] = tmp[12];
        out[block + 1] = tmp[5];
        out[block + 5] = tmp[9];
        out[block + 9] = tmp[13];
        out[block + 13] = tmp[1];
        out[block + 2] = tmp[10];
        out[block + 6] = tmp[14];
        out[block + 10] = tmp[2];
        out[block + 14] = tmp[6];
        out[block + 3] = tmp[15];
        out[block + 7] = tmp[3];
        out[block + 11] = tmp[7];
        out[block + 15] = tmp[11];
        // AddRoundKey (no MixColumns in last round)
        for (int i = 0; i < 16; ++i) {
            out[block + i] ^= round_key[i];
        }
    }
}

// AESRoundInv: Perform one AES decryption round
HWY_ATTR void AESRoundInvImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                              const uint8_t* HWY_RESTRICT round_key, size_t count) {
    for (size_t block = 0; block + 16 <= count; block += 16) {
        uint8_t tmp[16];
        // InvShiftRows
        tmp[0] = state[block + 0];
        tmp[4] = state[block + 4];
        tmp[8] = state[block + 8];
        tmp[12] = state[block + 12];
        tmp[1] = state[block + 13];
        tmp[5] = state[block + 1];
        tmp[9] = state[block + 5];
        tmp[13] = state[block + 9];
        tmp[2] = state[block + 10];
        tmp[6] = state[block + 14];
        tmp[10] = state[block + 2];
        tmp[14] = state[block + 6];
        tmp[3] = state[block + 7];
        tmp[7] = state[block + 11];
        tmp[11] = state[block + 15];
        tmp[15] = state[block + 3];
        // InvSubBytes
        for (int i = 0; i < 16; ++i) {
            tmp[i] = kAESInvSBox[tmp[i]];
        }
        // AddRoundKey
        for (int i = 0; i < 16; ++i) {
            tmp[i] ^= round_key[i];
        }
        // InvMixColumns
        for (int col = 0; col < 4; ++col) {
            int c = col * 4;
            uint8_t a0 = tmp[c], a1 = tmp[c + 1], a2 = tmp[c + 2], a3 = tmp[c + 3];
            out[block + c] = GFMul(0x0e, a0) ^ GFMul(0x0b, a1) ^ GFMul(0x0d, a2) ^ GFMul(0x09, a3);
            out[block + c + 1] =
                GFMul(0x09, a0) ^ GFMul(0x0e, a1) ^ GFMul(0x0b, a2) ^ GFMul(0x0d, a3);
            out[block + c + 2] =
                GFMul(0x0d, a0) ^ GFMul(0x09, a1) ^ GFMul(0x0e, a2) ^ GFMul(0x0b, a3);
            out[block + c + 3] =
                GFMul(0x0b, a0) ^ GFMul(0x0d, a1) ^ GFMul(0x09, a2) ^ GFMul(0x0e, a3);
        }
    }
}

// AESLastRoundInv: Perform final AES decryption round (no InvMixColumns)
HWY_ATTR void AESLastRoundInvImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                                  const uint8_t* HWY_RESTRICT round_key, size_t count) {
    for (size_t block = 0; block + 16 <= count; block += 16) {
        uint8_t tmp[16];
        // InvShiftRows
        tmp[0] = state[block + 0];
        tmp[4] = state[block + 4];
        tmp[8] = state[block + 8];
        tmp[12] = state[block + 12];
        tmp[1] = state[block + 13];
        tmp[5] = state[block + 1];
        tmp[9] = state[block + 5];
        tmp[13] = state[block + 9];
        tmp[2] = state[block + 10];
        tmp[6] = state[block + 14];
        tmp[10] = state[block + 2];
        tmp[14] = state[block + 6];
        tmp[3] = state[block + 7];
        tmp[7] = state[block + 11];
        tmp[11] = state[block + 15];
        tmp[15] = state[block + 3];
        // InvSubBytes
        for (int i = 0; i < 16; ++i) {
            out[block + i] = kAESInvSBox[tmp[i]];
        }
        // AddRoundKey
        for (int i = 0; i < 16; ++i) {
            out[block + i] ^= round_key[i];
        }
    }
}

// AESInvMixColumns: Apply InvMixColumns transformation
HWY_ATTR void AESInvMixColumnsImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                                   size_t count) {
    for (size_t block = 0; block + 16 <= count; block += 16) {
        for (int col = 0; col < 4; ++col) {
            int c = col * 4;
            uint8_t a0 = state[block + c], a1 = state[block + c + 1];
            uint8_t a2 = state[block + c + 2], a3 = state[block + c + 3];
            out[block + c] = GFMul(0x0e, a0) ^ GFMul(0x0b, a1) ^ GFMul(0x0d, a2) ^ GFMul(0x09, a3);
            out[block + c + 1] =
                GFMul(0x09, a0) ^ GFMul(0x0e, a1) ^ GFMul(0x0b, a2) ^ GFMul(0x0d, a3);
            out[block + c + 2] =
                GFMul(0x0d, a0) ^ GFMul(0x09, a1) ^ GFMul(0x0e, a2) ^ GFMul(0x0b, a3);
            out[block + c + 3] =
                GFMul(0x0b, a0) ^ GFMul(0x0d, a1) ^ GFMul(0x09, a2) ^ GFMul(0x0e, a3);
        }
    }
}

// AESKeyGenAssist: Key expansion assist
HWY_ATTR void AESKeyGenAssistImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT key,
                                  uint8_t rcon, size_t count) {
    for (size_t block = 0; block + 16 <= count; block += 16) {
        // Apply SubBytes to words 1 and 3
        uint8_t w1[4] = {kAESSBox[key[block + 4]], kAESSBox[key[block + 5]],
                         kAESSBox[key[block + 6]], kAESSBox[key[block + 7]]};
        uint8_t w3[4] = {kAESSBox[key[block + 12]], kAESSBox[key[block + 13]],
                         kAESSBox[key[block + 14]], kAESSBox[key[block + 15]]};
        // RotWord on w3 and XOR with Rcon
        out[block + 0] = key[block + 0];
        out[block + 1] = key[block + 1];
        out[block + 2] = key[block + 2];
        out[block + 3] = key[block + 3];
        out[block + 4] = w1[0];
        out[block + 5] = w1[1];
        out[block + 6] = w1[2];
        out[block + 7] = w1[3];
        out[block + 8] = key[block + 8];
        out[block + 9] = key[block + 9];
        out[block + 10] = key[block + 10];
        out[block + 11] = key[block + 11];
        out[block + 12] = w3[1] ^ rcon;  // RotWord
        out[block + 13] = w3[2];
        out[block + 14] = w3[3];
        out[block + 15] = w3[0];
    }
}

// CLMulLower: Carry-less multiplication of lower 64-bit halves
HWY_ATTR void CLMulLowerImpl(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                             const uint64_t* HWY_RESTRICT b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t x = a[i];
        uint64_t y = b[i];
        uint64_t lo = 0, hi = 0;
        // Carry-less multiplication
        for (int j = 0; j < 64; ++j) {
            if ((y >> j) & 1) {
                lo ^= x << j;
                if (j > 0)
                    hi ^= x >> (64 - j);
            }
        }
        out[i * 2] = lo;
        out[i * 2 + 1] = hi;
    }
}

// CLMulUpper: Carry-less multiplication of upper 64-bit halves
HWY_ATTR void CLMulUpperImpl(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                             const uint64_t* HWY_RESTRICT b, size_t count) {
    // For 128-bit inputs, upper half is at index 1
    for (size_t i = 0; i + 2 <= count; i += 2) {
        uint64_t x = a[i + 1];  // Upper 64 bits
        uint64_t y = b[i + 1];
        uint64_t lo = 0, hi = 0;
        for (int j = 0; j < 64; ++j) {
            if ((y >> j) & 1) {
                lo ^= x << j;
                if (j > 0)
                    hi ^= x >> (64 - j);
            }
        }
        out[i] = lo;
        out[i + 1] = hi;
    }
}

// =============================================================================
// P3.3: Random Number Generation
// =============================================================================

// Xoshiro256** state initialization
HWY_ATTR void RandomStateInitImpl(uint64_t* state, uint64_t seed) {
    // SplitMix64 to initialize state from seed
    uint64_t z = seed;
    for (int i = 0; i < 4; ++i) {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t result = z;
        result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
        result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
        state[i] = result ^ (result >> 31);
    }
}

// Xoshiro256** next
HWY_INLINE uint64_t Xoshiro256Next(uint64_t* state) {
    const uint64_t result = ((state[1] * 5) << 7 | (state[1] * 5) >> 57) * 9;
    const uint64_t t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = (state[3] << 45) | (state[3] >> 19);
    return result;
}

// Random32: Generate random 32-bit integers
HWY_ATTR void Random32Impl(uint32_t* HWY_RESTRICT out, uint64_t* state, size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        uint64_t r = Xoshiro256Next(state);
        out[i] = static_cast<uint32_t>(r);
        if (i + 1 < count) {
            out[i + 1] = static_cast<uint32_t>(r >> 32);
        }
    }
}

// Random64: Generate random 64-bit integers
HWY_ATTR void Random64Impl(uint64_t* HWY_RESTRICT out, uint64_t* state, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = Xoshiro256Next(state);
    }
}

// RandomFloat: Generate random floats in [0, 1)
HWY_ATTR void RandomFloatImpl(float* HWY_RESTRICT out, uint64_t* state, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t r = Xoshiro256Next(state);
        // Use upper 23 bits for mantissa
        uint32_t bits = static_cast<uint32_t>(r >> 41);
        out[i] = static_cast<float>(bits) * (1.0f / 8388608.0f);  // 2^23
    }
}

// =============================================================================
// P3.4: Bit Packing
// =============================================================================

// PackBits: Pack MSBs of bytes into bits
HWY_ATTR void PackBitsImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src,
                           size_t count) {
    size_t out_idx = 0;
    for (size_t i = 0; i < count; i += 8) {
        uint8_t packed = 0;
        for (size_t j = 0; j < 8 && (i + j) < count; ++j) {
            if (src[i + j] & 0x80) {
                packed |= (1 << (7 - j));
            }
        }
        out[out_idx++] = packed;
    }
}

// UnpackBits: Unpack bits to bytes (0xFF or 0x00)
HWY_ATTR void UnpackBitsImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src,
                             size_t count) {
    size_t src_idx = 0;
    for (size_t i = 0; i < count; i += 8) {
        uint8_t packed = src[src_idx++];
        for (size_t j = 0; j < 8 && (i + j) < count; ++j) {
            out[i + j] = (packed & (1 << (7 - j))) ? 0xFF : 0x00;
        }
    }
}

// =============================================================================
// P3.6: Algorithm Operations
// =============================================================================

// FindIfGreaterThan: Find first element > threshold
HWY_ATTR size_t FindIfGreaterThanFloat32(const float* HWY_RESTRICT arr, float threshold,
                                         size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vthresh = hn::Set(d, threshold);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, arr + i);
        const auto mask = hn::Gt(v, vthresh);
        if (!hn::AllFalse(d, mask)) {
            // Find first true in this vector
            for (size_t j = 0; j < N; ++j) {
                if (arr[i + j] > threshold)
                    return i + j;
            }
        }
    }
    // Check remainder
    for (; i < count; ++i) {
        if (arr[i] > threshold)
            return i;
    }
    return count;  // Not found
}

HWY_ATTR size_t FindIfGreaterThanInt32(const int32_t* HWY_RESTRICT arr, int32_t threshold,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (arr[i] > threshold)
            return i;
    }
    return count;
}

// Generate: Generate arithmetic sequence
HWY_ATTR void GenerateFloat32(float* HWY_RESTRICT out, float start, float step, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Create step vector
    auto vstep = hn::Set(d, step * static_cast<float>(N));
    auto vcur = hn::Iota(d, start);
    vcur = hn::MulAdd(hn::Iota(d, 0.0f), hn::Set(d, step), hn::Set(d, start));

    size_t i = 0;
    for (; i + N <= count; i += N) {
        hn::StoreU(vcur, d, out + i);
        vcur = hn::Add(vcur, vstep);
    }
    // Remainder
    float val = start + step * static_cast<float>(i);
    for (; i < count; ++i) {
        out[i] = val;
        val += step;
    }
}

HWY_ATTR void GenerateInt32(int32_t* HWY_RESTRICT out, int32_t start, int32_t step, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = start + static_cast<int32_t>(i) * step;
    }
}

// Replace: Replace matching values
HWY_ATTR void ReplaceFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, float old_val,
                             float new_val, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vold = hn::Set(d, old_val);
    const auto vnew = hn::Set(d, new_val);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, src + i);
        const auto mask = hn::Eq(v, vold);
        const auto result = hn::IfThenElse(mask, vnew, v);
        hn::StoreU(result, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (src[i] == old_val) ? new_val : src[i];
    }
}

HWY_ATTR void ReplaceInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                           int32_t old_val, int32_t new_val, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (src[i] == old_val) ? new_val : src[i];
    }
}

// ReplaceIfGreaterThan: Replace values > threshold
HWY_ATTR void ReplaceIfGreaterThanFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                                          float threshold, float new_val, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vthresh = hn::Set(d, threshold);
    const auto vnew = hn::Set(d, new_val);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, src + i);
        const auto mask = hn::Gt(v, vthresh);
        const auto result = hn::IfThenElse(mask, vnew, v);
        hn::StoreU(result, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (src[i] > threshold) ? new_val : src[i];
    }
}

HWY_ATTR void ReplaceIfGreaterThanInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                                        int32_t threshold, int32_t new_val, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (src[i] > threshold) ? new_val : src[i];
    }
}

// =============================================================================
// P3.5: Image Processing Operations
// =============================================================================

// Image structure - defined in header, implementations here use it
// Image memory layout: row-major with stride for alignment

// ImageFill: Fill image with constant value
HWY_ATTR void ImageFillImpl(float* HWY_RESTRICT data, size_t width, size_t height, size_t stride,
                            float value) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vval = hn::Set(d, value);

    for (size_t y = 0; y < height; ++y) {
        float* row = data + y * stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            hn::StoreU(vval, d, row + x);
        }
        for (; x < width; ++x) {
            row[x] = value;
        }
    }
}

// ImageCopy: Copy image data
HWY_ATTR void ImageCopyImpl(const float* HWY_RESTRICT src, size_t src_stride,
                            float* HWY_RESTRICT dst, size_t dst_stride, size_t width,
                            size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t y = 0; y < height; ++y) {
        const float* src_row = src + y * src_stride;
        float* dst_row = dst + y * dst_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            hn::StoreU(hn::LoadU(d, src_row + x), d, dst_row + x);
        }
        for (; x < width; ++x) {
            dst_row[x] = src_row[x];
        }
    }
}

// ImageAdd: Add two images
HWY_ATTR void ImageAddImpl(const float* HWY_RESTRICT a, size_t a_stride,
                           const float* HWY_RESTRICT b, size_t b_stride, float* HWY_RESTRICT out,
                           size_t out_stride, size_t width, size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t y = 0; y < height; ++y) {
        const float* a_row = a + y * a_stride;
        const float* b_row = b + y * b_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto va = hn::LoadU(d, a_row + x);
            auto vb = hn::LoadU(d, b_row + x);
            hn::StoreU(hn::Add(va, vb), d, out_row + x);
        }
        for (; x < width; ++x) {
            out_row[x] = a_row[x] + b_row[x];
        }
    }
}

// ImageSub: Subtract two images
HWY_ATTR void ImageSubImpl(const float* HWY_RESTRICT a, size_t a_stride,
                           const float* HWY_RESTRICT b, size_t b_stride, float* HWY_RESTRICT out,
                           size_t out_stride, size_t width, size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t y = 0; y < height; ++y) {
        const float* a_row = a + y * a_stride;
        const float* b_row = b + y * b_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto va = hn::LoadU(d, a_row + x);
            auto vb = hn::LoadU(d, b_row + x);
            hn::StoreU(hn::Sub(va, vb), d, out_row + x);
        }
        for (; x < width; ++x) {
            out_row[x] = a_row[x] - b_row[x];
        }
    }
}

// ImageMul: Multiply two images element-wise
HWY_ATTR void ImageMulImpl(const float* HWY_RESTRICT a, size_t a_stride,
                           const float* HWY_RESTRICT b, size_t b_stride, float* HWY_RESTRICT out,
                           size_t out_stride, size_t width, size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t y = 0; y < height; ++y) {
        const float* a_row = a + y * a_stride;
        const float* b_row = b + y * b_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto va = hn::LoadU(d, a_row + x);
            auto vb = hn::LoadU(d, b_row + x);
            hn::StoreU(hn::Mul(va, vb), d, out_row + x);
        }
        for (; x < width; ++x) {
            out_row[x] = a_row[x] * b_row[x];
        }
    }
}

// ImageScale: Scale image by constant
HWY_ATTR void ImageScaleImpl(const float* HWY_RESTRICT src, size_t src_stride, float scale,
                             float* HWY_RESTRICT out, size_t out_stride, size_t width,
                             size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vscale = hn::Set(d, scale);

    for (size_t y = 0; y < height; ++y) {
        const float* src_row = src + y * src_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto v = hn::LoadU(d, src_row + x);
            hn::StoreU(hn::Mul(v, vscale), d, out_row + x);
        }
        for (; x < width; ++x) {
            out_row[x] = src_row[x] * scale;
        }
    }
}

// ImageClamp: Clamp image values to range
HWY_ATTR void ImageClampImpl(const float* HWY_RESTRICT src, size_t src_stride, float min_val,
                             float max_val, float* HWY_RESTRICT out, size_t out_stride,
                             size_t width, size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vmin = hn::Set(d, min_val);
    const auto vmax = hn::Set(d, max_val);

    for (size_t y = 0; y < height; ++y) {
        const float* src_row = src + y * src_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto v = hn::LoadU(d, src_row + x);
            v = hn::Max(v, vmin);
            v = hn::Min(v, vmax);
            hn::StoreU(v, d, out_row + x);
        }
        for (; x < width; ++x) {
            float v = src_row[x];
            if (v < min_val)
                v = min_val;
            if (v > max_val)
                v = max_val;
            out_row[x] = v;
        }
    }
}

// Helper: Get pixel with mirror boundary handling
HWY_INLINE float GetPixelMirror(const float* data, size_t stride, int x, int y, int width,
                                int height) {
    if (x < 0)
        x = -x - 1;
    if (y < 0)
        y = -y - 1;
    if (x >= width)
        x = 2 * width - x - 1;
    if (y >= height)
        y = 2 * height - y - 1;
    return data[y * stride + x];
}

// Convolve3x3: General 3x3 convolution
HWY_ATTR void Convolve3x3Impl(const float* HWY_RESTRICT src, size_t src_stride,
                              const float* HWY_RESTRICT kernel, float* HWY_RESTRICT out,
                              size_t out_stride, size_t width, size_t height) {
    int w = static_cast<int>(width);
    int h = static_cast<int>(height);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    float pixel = GetPixelMirror(src, src_stride, x + kx, y + ky, w, h);
                    sum += pixel * kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }
            out[y * out_stride + x] = sum;
        }
    }
}

// BoxBlur3x3: 3x3 box blur (average filter)
HWY_ATTR void BoxBlur3x3Impl(const float* HWY_RESTRICT src, size_t src_stride,
                             float* HWY_RESTRICT out, size_t out_stride, size_t width,
                             size_t height) {
    const float kernel[9] = {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                             1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f};
    Convolve3x3Impl(src, src_stride, kernel, out, out_stride, width, height);
}

// GaussianBlur3x3: 3x3 Gaussian blur
HWY_ATTR void GaussianBlur3x3Impl(const float* HWY_RESTRICT src, size_t src_stride,
                                  float* HWY_RESTRICT out, size_t out_stride, size_t width,
                                  size_t height) {
    const float kernel[9] = {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f, 2.0f / 16.0f, 4.0f / 16.0f,
                             2.0f / 16.0f, 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f};
    Convolve3x3Impl(src, src_stride, kernel, out, out_stride, width, height);
}

// SobelEdge: Sobel edge detection (magnitude)
HWY_ATTR void SobelEdgeImpl(const float* HWY_RESTRICT src, size_t src_stride,
                            float* HWY_RESTRICT out, size_t out_stride, size_t width,
                            size_t height) {
    int w = static_cast<int>(width);
    int h = static_cast<int>(height);

    // Sobel kernels
    const float kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float ky[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky_off = -1; ky_off <= 1; ++ky_off) {
                for (int kx_off = -1; kx_off <= 1; ++kx_off) {
                    float pixel = GetPixelMirror(src, src_stride, x + kx_off, y + ky_off, w, h);
                    int kidx = (ky_off + 1) * 3 + (kx_off + 1);
                    gx += pixel * kx[kidx];
                    gy += pixel * ky[kidx];
                }
            }
            out[y * out_stride + x] = std::sqrt(gx * gx + gy * gy);
        }
    }
}

// Sharpen: Image sharpening using unsharp mask
HWY_ATTR void SharpenImpl(const float* HWY_RESTRICT src, size_t src_stride, float* HWY_RESTRICT out,
                          size_t out_stride, size_t width, size_t height) {
    // Sharpening kernel: emphasize center, subtract neighbors
    const float kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    Convolve3x3Impl(src, src_stride, kernel, out, out_stride, width, height);
}

// Threshold: Binary thresholding
HWY_ATTR void ThresholdImpl(const float* HWY_RESTRICT src, size_t src_stride, float threshold,
                            float* HWY_RESTRICT out, size_t out_stride, size_t width,
                            size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vthresh = hn::Set(d, threshold);
    const auto vzero = hn::Zero(d);
    const auto vone = hn::Set(d, 1.0f);

    for (size_t y = 0; y < height; ++y) {
        const float* src_row = src + y * src_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto v = hn::LoadU(d, src_row + x);
            auto mask = hn::Gt(v, vthresh);
            hn::StoreU(hn::IfThenElse(mask, vone, vzero), d, out_row + x);
        }
        for (; x < width; ++x) {
            out_row[x] = (src_row[x] > threshold) ? 1.0f : 0.0f;
        }
    }
}

// Grayscale: RGB to grayscale using luminance formula
HWY_ATTR void GrayscaleImpl(const float* HWY_RESTRICT r, size_t r_stride,
                            const float* HWY_RESTRICT g, size_t g_stride,
                            const float* HWY_RESTRICT b, size_t b_stride, float* HWY_RESTRICT out,
                            size_t out_stride, size_t width, size_t height) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vr_coef = hn::Set(d, 0.299f);
    const auto vg_coef = hn::Set(d, 0.587f);
    const auto vb_coef = hn::Set(d, 0.114f);

    for (size_t y = 0; y < height; ++y) {
        const float* r_row = r + y * r_stride;
        const float* g_row = g + y * g_stride;
        const float* b_row = b + y * b_stride;
        float* out_row = out + y * out_stride;
        size_t x = 0;
        for (; x + N <= width; x += N) {
            auto vr = hn::LoadU(d, r_row + x);
            auto vg = hn::LoadU(d, g_row + x);
            auto vb = hn::LoadU(d, b_row + x);
            auto gray = hn::MulAdd(vr, vr_coef, hn::MulAdd(vg, vg_coef, hn::Mul(vb, vb_coef)));
            hn::StoreU(gray, d, out_row + x);
        }
        for (; x < width; ++x) {
            out_row[x] = 0.299f * r_row[x] + 0.587f * g_row[x] + 0.114f * b_row[x];
        }
    }
}

// Downsample2x: Downsample image by factor of 2 (averaging)
HWY_ATTR void Downsample2xImpl(const float* HWY_RESTRICT src, size_t src_stride,
                               float* HWY_RESTRICT out, size_t out_stride, size_t out_width,
                               size_t out_height) {
    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            size_t sx = x * 2;
            size_t sy = y * 2;
            float sum = src[sy * src_stride + sx] + src[sy * src_stride + sx + 1] +
                        src[(sy + 1) * src_stride + sx] + src[(sy + 1) * src_stride + sx + 1];
            out[y * out_stride + x] = sum * 0.25f;
        }
    }
}

// Upsample2x: Upsample image by factor of 2 (bilinear interpolation)
HWY_ATTR void Upsample2xImpl(const float* HWY_RESTRICT src, size_t src_stride, size_t src_width,
                             size_t src_height, float* HWY_RESTRICT out, size_t out_stride) {
    for (size_t y = 0; y < src_height * 2; ++y) {
        for (size_t x = 0; x < src_width * 2; ++x) {
            size_t sx = x / 2;
            size_t sy = y / 2;
            size_t sx1 = (sx + 1 < src_width) ? sx + 1 : sx;
            size_t sy1 = (sy + 1 < src_height) ? sy + 1 : sy;

            float fx = (x % 2) * 0.5f;
            float fy = (y % 2) * 0.5f;

            float p00 = src[sy * src_stride + sx];
            float p10 = src[sy * src_stride + sx1];
            float p01 = src[sy1 * src_stride + sx];
            float p11 = src[sy1 * src_stride + sx1];

            float top = p00 * (1 - fx) + p10 * fx;
            float bot = p01 * (1 - fx) + p11 * fx;
            out[y * out_stride + x] = top * (1 - fy) + bot * fy;
        }
    }
}

// =============================================================================
// Gap Operations: Additional SIMD Operations
// =============================================================================

// OddEven: out[2i] = even[2i], out[2i+1] = odd[2i+1]
HWY_ATTR void OddEvenFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT odd,
                             const float* HWY_RESTRICT even, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (i % 2 == 0) ? even[i] : odd[i];
    }
}

HWY_ATTR void OddEvenInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT odd,
                           const int32_t* HWY_RESTRICT even, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (i % 2 == 0) ? even[i] : odd[i];
    }
}

// MaskedMulAdd: out[i] = mask[i] ? (mul[i] * x[i] + add[i]) : no[i]
HWY_ATTR void MaskedMulAddFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                  const float* HWY_RESTRICT mul, const float* HWY_RESTRICT x,
                                  const float* HWY_RESTRICT add, const float* HWY_RESTRICT no,
                                  size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (mul[i] * x[i] + add[i]) : no[i];
    }
}

// MaskedNegMulAdd: out[i] = mask[i] ? (add[i] - mul[i] * x[i]) : no[i]
HWY_ATTR void MaskedNegMulAddFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                     const float* HWY_RESTRICT mul, const float* HWY_RESTRICT x,
                                     const float* HWY_RESTRICT add, const float* HWY_RESTRICT no,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (add[i] - mul[i] * x[i]) : no[i];
    }
}

// InterleaveWholeLower: Interleave lower halves - a0,b0,a1,b1,...
HWY_ATTR void InterleaveWholeLowerFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                          const float* HWY_RESTRICT b, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = a[i];
        out[2 * i + 1] = b[i];
    }
}

// InterleaveWholeUpper: Interleave upper halves - a[half],b[half],a[half+1],b[half+1],...
HWY_ATTR void InterleaveWholeUpperFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                          const float* HWY_RESTRICT b, size_t count) {
    const size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[2 * i] = a[half + i];
        out[2 * i + 1] = b[half + i];
    }
}

// IfNegativeThenElseZero: out[i] = (v[i] < 0) ? yes[i] : 0
HWY_ATTR void IfNegativeThenElseZeroFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                                            const float* HWY_RESTRICT yes, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (v[i] < 0.0f) ? yes[i] : 0.0f;
    }
}

// IfNegativeThenZeroElse: out[i] = (v[i] < 0) ? 0 : no[i]
HWY_ATTR void IfNegativeThenZeroElseFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                                            const float* HWY_RESTRICT no, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (v[i] < 0.0f) ? 0.0f : no[i];
    }
}

// BitwiseIfThenElse: out[i] = (mask[i] & yes[i]) | (~mask[i] & no[i])
HWY_ATTR void BitwiseIfThenElseInt32(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT mask,
                                     const int32_t* HWY_RESTRICT yes,
                                     const int32_t* HWY_RESTRICT no, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = (mask[i] & yes[i]) | (~mask[i] & no[i]);
    }
}

// MaskFalse: Set all mask bytes to 0 (false)
HWY_ATTR void MaskFalseImpl(uint8_t* HWY_RESTRICT mask, size_t count) {
    std::memset(mask, 0, count);
}

// SetMask: Set all mask bytes to 0xFF (true) or 0 (false)
HWY_ATTR void SetMaskImpl(uint8_t* HWY_RESTRICT mask, bool value, size_t count) {
    std::memset(mask, value ? 0xFF : 0, count);
}

// CeilInt: out[i] = ceil(in[i]) as int32
HWY_ATTR void CeilIntFloat32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                             size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(std::ceil(in[i]));
    }
}

// FloorInt: out[i] = floor(in[i]) as int32
HWY_ATTR void FloorIntFloat32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT in,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int32_t>(std::floor(in[i]));
    }
}

// TruncateStore: Store int32 as int16 (simple truncation, not saturation)
HWY_ATTR void TruncateStoreInt32ToInt16(int16_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                                        size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<int16_t>(src[i]);
    }
}

// MaskedModOr: out[i] = mask[i] ? (a[i] % b[i]) : no[i]
HWY_ATTR void MaskedModOrInt32(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                               const int32_t* HWY_RESTRICT no, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (a[i] % b[i]) : no[i];
    }
}

// =============================================================================
// Final Gap Operations: Low Priority Remaining Operations
// =============================================================================

// Per2LaneBlockShuffle: Shuffle within 2-lane blocks
HWY_ATTR void Per2LaneBlockShuffleFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                                          const float* HWY_RESTRICT b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        // Interleave within 2-lane blocks: a0,b0,a1,b1,...
        auto lo = hn::InterleaveLower(d, va, vb);
        hn::StoreU(lo, d, out + i);
    }
    // Remainder
    for (; i < count; i += 2) {
        out[i] = a[i];
        if (i + 1 < count)
            out[i + 1] = b[i];
    }
}

// MaskedIsNaN: out[i] = mask[i] ? isnan(in[i]) : 0
HWY_ATTR void MaskedIsNaNFloat32(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                 const float* HWY_RESTRICT in, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? (std::isnan(in[i]) ? 0xFF : 0) : 0;
    }
}

// IfNegativeThenNegOrUndefIfZero: out[i] = (v[i] < 0) ? -x[i] : x[i]
HWY_ATTR void IfNegativeThenNegOrUndefIfZeroFloat32(float* HWY_RESTRICT out,
                                                    const float* HWY_RESTRICT v,
                                                    const float* HWY_RESTRICT x, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto vv = hn::LoadU(d, v + i);
        auto vx = hn::LoadU(d, x + i);
        auto neg_x = hn::Neg(vx);
        auto is_neg = hn::Lt(vv, hn::Zero(d));
        auto result = hn::IfThenElse(is_neg, neg_x, vx);
        hn::StoreU(result, d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (v[i] < 0) ? -x[i] : x[i];
    }
}

// MaskedSetOr: out[i] = mask[i] ? value : no[i]
HWY_ATTR void MaskedSetOrFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                 float value, const float* HWY_RESTRICT no, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    auto vval = hn::Set(d, value);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto vno = hn::LoadU(d, no + i);
        // Build mask from bytes
        hn::Mask<decltype(d)> m = hn::FirstN(d, 0);
        bool all_true = true, all_false = true;
        for (size_t j = 0; j < N && i + j < count; ++j) {
            if (mask[i + j])
                all_false = false;
            else
                all_true = false;
        }
        if (all_true) {
            hn::StoreU(vval, d, out + i);
        } else if (all_false) {
            hn::StoreU(vno, d, out + i);
        } else {
            for (size_t j = 0; j < N && i + j < count; ++j) {
                out[i + j] = mask[i + j] ? value : no[i + j];
            }
        }
    }
    for (; i < count; ++i) {
        out[i] = mask[i] ? value : no[i];
    }
}

// MaskedMulFixedPoint15: Q15 fixed-point multiply with mask
HWY_ATTR void MaskedMulFixedPoint15Int16(int16_t* HWY_RESTRICT out,
                                         const uint8_t* HWY_RESTRICT mask,
                                         const int16_t* HWY_RESTRICT a,
                                         const int16_t* HWY_RESTRICT b,
                                         const int16_t* HWY_RESTRICT no, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            // Q15: multiply and shift right by 15
            int32_t product = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            out[i] = static_cast<int16_t>(product >> 15);
        } else {
            out[i] = no[i];
        }
    }
}

// MaskedWidenMulPairwiseAdd: Widening multiply-add with mask
HWY_ATTR void MaskedWidenMulPairwiseAddInt16ToInt32(int32_t* HWY_RESTRICT out,
                                                    const uint8_t* HWY_RESTRICT mask,
                                                    const int16_t* HWY_RESTRICT a,
                                                    const int16_t* HWY_RESTRICT b,
                                                    const int32_t* HWY_RESTRICT no, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            // out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
            size_t idx = i * 2;
            int32_t p0 = static_cast<int32_t>(a[idx]) * static_cast<int32_t>(b[idx]);
            int32_t p1 = static_cast<int32_t>(a[idx + 1]) * static_cast<int32_t>(b[idx + 1]);
            out[i] = p0 + p1;
        } else {
            out[i] = no[i];
        }
    }
}

// MaskedAbsOr: out[i] = mask[i] ? abs(a[i]) : no[i]
HWY_ATTR void MaskedAbsOrFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                 const float* HWY_RESTRICT a, const float* HWY_RESTRICT no,
                                 size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vno = hn::LoadU(d, no + i);
        auto vabs = hn::Abs(va);
        // Scalar fallback for mask handling
        for (size_t j = 0; j < N && i + j < count; ++j) {
            out[i + j] = mask[i + j] ? std::abs(a[i + j]) : no[i + j];
        }
    }
    for (; i < count; ++i) {
        out[i] = mask[i] ? std::abs(a[i]) : no[i];
    }
}

// InsertIntoUpper: Copy lower half from vec, set upper half to scalar
HWY_ATTR void InsertIntoUpperFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT vec,
                                     float scalar, size_t count) {
    size_t half = count / 2;
    for (size_t i = 0; i < half; ++i) {
        out[i] = vec[i];
    }
    for (size_t i = half; i < count; ++i) {
        out[i] = scalar;
    }
}

// MaskedGatherIndexOr: Masked gather with fallback
HWY_ATTR void MaskedGatherIndexOrFloat32(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                         const float* HWY_RESTRICT base,
                                         const int32_t* HWY_RESTRICT indices,
                                         const float* HWY_RESTRICT no, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = mask[i] ? base[indices[i]] : no[i];
    }
}

// SumsOf8AbsDiff: Sum of absolute differences for groups of 8
HWY_ATTR void SumsOf8AbsDiffUint8(uint64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                                  const uint8_t* HWY_RESTRICT b, size_t count) {
    size_t num_groups = count / 8;
    for (size_t g = 0; g < num_groups; ++g) {
        uint64_t sum = 0;
        for (size_t i = 0; i < 8; ++i) {
            size_t idx = g * 8 + i;
            int diff = static_cast<int>(a[idx]) - static_cast<int>(b[idx]);
            sum += static_cast<uint64_t>(diff < 0 ? -diff : diff);
        }
        out[g] = sum;
    }
}

// CombineMasks: Combine lo and hi masks into out
HWY_ATTR void CombineMasksImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT lo,
                               const uint8_t* HWY_RESTRICT hi, size_t half_count) {
    std::memcpy(out, lo, half_count);
    std::memcpy(out + half_count, hi, half_count);
}

// LowerHalfOfMask: Extract lower half of mask
HWY_ATTR void LowerHalfOfMaskImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                  size_t count) {
    size_t half = count / 2;
    std::memcpy(out, mask, half);
}

// UpperHalfOfMask: Extract upper half of mask
HWY_ATTR void UpperHalfOfMaskImpl(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                                  size_t count) {
    size_t half = count / 2;
    std::memcpy(out, mask + half, half);
}

// PromoteMaskTo: Promote uint8 mask to uint16
HWY_ATTR void PromoteMaskToUint8ToUint16(uint16_t* HWY_RESTRICT wide,
                                         const uint8_t* HWY_RESTRICT narrow, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        wide[i] = narrow[i] ? 0xFFFF : 0;
    }
}

// DemoteMaskTo: Demote uint16 mask to uint8
HWY_ATTR void DemoteMaskToUint16ToUint8(uint8_t* HWY_RESTRICT narrow,
                                        const uint16_t* HWY_RESTRICT wide, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        narrow[i] = wide[i] ? 0xFF : 0;
    }
}

// ZeroExtendResizeBitCast: Zero-extend uint8 to uint32
HWY_ATTR void ZeroExtendResizeBitCastUint8ToUint32(uint32_t* HWY_RESTRICT dst,
                                                   const uint8_t* HWY_RESTRICT src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<uint32_t>(src[i]);
    }
}

// F64ToF16: Convert double to float16 (stored as uint16)
HWY_ATTR void F64ToF16Impl(uint16_t* HWY_RESTRICT dst, const double* HWY_RESTRICT src,
                           size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Convert double -> float -> float16
        float f = static_cast<float>(src[i]);
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));

        // Extract components
        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;  // Rebias for f16
        uint32_t frac = bits & 0x7FFFFF;

        uint16_t h;
        if (exp <= 0) {
            h = static_cast<uint16_t>(sign << 15);  // Zero or subnormal
        } else if (exp >= 31) {
            h = static_cast<uint16_t>((sign << 15) | 0x7C00);  // Infinity
        } else {
            h = static_cast<uint16_t>((sign << 15) | (exp << 10) | (frac >> 13));
        }
        dst[i] = h;
    }
}

// F64ToBF16: Convert double to bfloat16 (stored as uint16)
HWY_ATTR void F64ToBF16Impl(uint16_t* HWY_RESTRICT dst, const double* HWY_RESTRICT src,
                            size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Convert double -> float, then take upper 16 bits
        float f = static_cast<float>(src[i]);
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

// MatVecMul: Matrix-vector multiplication
HWY_ATTR void MatVecMulFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT mat,
                               const float* HWY_RESTRICT vec, size_t rows, size_t cols) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        size_t c = 0;

        // SIMD loop
        auto vsum = hn::Zero(d);
        for (; c + N <= cols; c += N) {
            auto vmat = hn::LoadU(d, mat + r * cols + c);
            auto vvec = hn::LoadU(d, vec + c);
            vsum = hn::MulAdd(vmat, vvec, vsum);
        }
        sum = hn::ReduceSum(d, vsum);

        // Remainder
        for (; c < cols; ++c) {
            sum += mat[r * cols + c] * vec[c];
        }
        out[r] = sum;
    }
}

// MatMul: Matrix-matrix multiplication (row-major)
HWY_ATTR void MatMulFloat32(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                            const float* HWY_RESTRICT b, size_t M, size_t K, size_t N) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            size_t k = 0;

            // SIMD accumulation
            auto vsum = hn::Zero(d);
            for (; k + lanes <= K; k += lanes) {
                auto va = hn::LoadU(d, a + i * K + k);
                // Gather b column - need scalar for non-contiguous access
                alignas(64) float b_col[16];
                for (size_t l = 0; l < lanes && k + l < K; ++l) {
                    b_col[l] = b[(k + l) * N + j];
                }
                auto vb = hn::LoadU(d, b_col);
                vsum = hn::MulAdd(va, vb, vsum);
            }
            sum = hn::ReduceSum(d, vsum);

            // Remainder
            for (; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

// =============================================================================
// Size-Specialized Kernels
// =============================================================================
// Different implementations optimized for different array sizes:
// - Small (<64): fully unrolled, minimal overhead
// - Medium (64-4096): 4x unrolled loop
// - Large (>4096): 8x unrolled + prefetching

// Size thresholds
constexpr size_t kSmallThreshold = 64;
constexpr size_t kMediumThreshold = 4096;

// Prefetch distance in bytes (typically 2-4 cache lines ahead)
constexpr size_t kPrefetchDistance = 256;

// -----------------------------------------------------------------------------
// AddSized - Size-specialized addition
// -----------------------------------------------------------------------------

// Small array kernel - minimal loop overhead
HWY_ATTR void AddSizedSmallFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                                   float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    // Single pass for small arrays
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }
    // Handle remainder
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Medium array kernel - 4x unrolled
HWY_ATTR void AddSizedMediumFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                                    float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const size_t stride = 4 * N;

    size_t i = 0;
    // 4x unrolled loop
    for (; i + stride <= count; i += stride) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);

        hn::StoreU(hn::Add(va0, vb0), d, out + i);
        hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
        hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
        hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
    }
    // Handle remaining vectors
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }
    // Handle scalar remainder
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Large array kernel - 8x unrolled + prefetching
HWY_ATTR void AddSizedLargeFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                                   float* HWY_RESTRICT out, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const size_t stride = 8 * N;
    const size_t prefetch_elems = kPrefetchDistance / sizeof(float);

    size_t i = 0;
    // 8x unrolled loop with prefetching
    for (; i + stride <= count; i += stride) {
        // Prefetch ahead (no fence needed - prefetch is just a hint)
        if (i + stride + prefetch_elems <= count) {
            __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
            __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
        }

        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);
        auto va4 = hn::LoadU(d, a + i + 4 * N);
        auto va5 = hn::LoadU(d, a + i + 5 * N);
        auto va6 = hn::LoadU(d, a + i + 6 * N);
        auto va7 = hn::LoadU(d, a + i + 7 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);
        auto vb4 = hn::LoadU(d, b + i + 4 * N);
        auto vb5 = hn::LoadU(d, b + i + 5 * N);
        auto vb6 = hn::LoadU(d, b + i + 6 * N);
        auto vb7 = hn::LoadU(d, b + i + 7 * N);

        hn::StoreU(hn::Add(va0, vb0), d, out + i);
        hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
        hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
        hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
        hn::StoreU(hn::Add(va4, vb4), d, out + i + 4 * N);
        hn::StoreU(hn::Add(va5, vb5), d, out + i + 5 * N);
        hn::StoreU(hn::Add(va6, vb6), d, out + i + 6 * N);
        hn::StoreU(hn::Add(va7, vb7), d, out + i + 7 * N);
    }
    // Handle remaining vectors with 4x unrolling
    const size_t stride4 = 4 * N;
    for (; i + stride4 <= count; i += stride4) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);

        hn::StoreU(hn::Add(va0, vb0), d, out + i);
        hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
        hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
        hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
    }
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Main dispatcher for AddSized float32
HWY_ATTR void AddSizedFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                              float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;
    if (count < kSmallThreshold) {
        AddSizedSmallFloat32(a, b, out, count);
    } else if (count <= kMediumThreshold) {
        AddSizedMediumFloat32(a, b, out, count);
    } else {
        AddSizedLargeFloat32(a, b, out, count);
    }
}

// AddSized for double
HWY_ATTR void AddSizedFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                              double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        // Small: single pass
        size_t i = 0;
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            hn::StoreU(hn::Add(va, vb), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    } else if (count <= kMediumThreshold) {
        // Medium: 4x unrolled
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            auto va0 = hn::LoadU(d, a + i);
            auto va1 = hn::LoadU(d, a + i + N);
            auto va2 = hn::LoadU(d, a + i + 2 * N);
            auto va3 = hn::LoadU(d, a + i + 3 * N);
            auto vb0 = hn::LoadU(d, b + i);
            auto vb1 = hn::LoadU(d, b + i + N);
            auto vb2 = hn::LoadU(d, b + i + 2 * N);
            auto vb3 = hn::LoadU(d, b + i + 3 * N);
            hn::StoreU(hn::Add(va0, vb0), d, out + i);
            hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
            hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
            hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    } else {
        // Large: 8x unrolled + prefetch
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(double);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
            }
            for (size_t j = 0; j < 8; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                hn::StoreU(hn::Add(va, vb), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    }
}

// AddSized for int32_t
HWY_ATTR void AddSizedInt32(const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                            int32_t* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<int32_t> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        // Small: single pass
        size_t i = 0;
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            hn::StoreU(hn::Add(va, vb), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    } else if (count <= kMediumThreshold) {
        // Medium: 4x unrolled
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            auto va0 = hn::LoadU(d, a + i);
            auto va1 = hn::LoadU(d, a + i + N);
            auto va2 = hn::LoadU(d, a + i + 2 * N);
            auto va3 = hn::LoadU(d, a + i + 3 * N);
            auto vb0 = hn::LoadU(d, b + i);
            auto vb1 = hn::LoadU(d, b + i + N);
            auto vb2 = hn::LoadU(d, b + i + 2 * N);
            auto vb3 = hn::LoadU(d, b + i + 3 * N);
            hn::StoreU(hn::Add(va0, vb0), d, out + i);
            hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
            hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
            hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    } else {
        // Large: 8x unrolled + prefetch
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(int32_t);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
            }
            for (size_t j = 0; j < 8; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                hn::StoreU(hn::Add(va, vb), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] + b[i];
        }
    }
}

// -----------------------------------------------------------------------------
// MulSized - Size-specialized multiplication
// -----------------------------------------------------------------------------

HWY_ATTR void MulSizedFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                              float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        // Small: single pass
        size_t i = 0;
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            hn::StoreU(hn::Mul(va, vb), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    } else if (count <= kMediumThreshold) {
        // Medium: 4x unrolled
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            auto va0 = hn::LoadU(d, a + i);
            auto va1 = hn::LoadU(d, a + i + N);
            auto va2 = hn::LoadU(d, a + i + 2 * N);
            auto va3 = hn::LoadU(d, a + i + 3 * N);
            auto vb0 = hn::LoadU(d, b + i);
            auto vb1 = hn::LoadU(d, b + i + N);
            auto vb2 = hn::LoadU(d, b + i + 2 * N);
            auto vb3 = hn::LoadU(d, b + i + 3 * N);
            hn::StoreU(hn::Mul(va0, vb0), d, out + i);
            hn::StoreU(hn::Mul(va1, vb1), d, out + i + N);
            hn::StoreU(hn::Mul(va2, vb2), d, out + i + 2 * N);
            hn::StoreU(hn::Mul(va3, vb3), d, out + i + 3 * N);
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Mul(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    } else {
        // Large: 8x unrolled + prefetch
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(float);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
            }
            for (size_t j = 0; j < 8; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                hn::StoreU(hn::Mul(va, vb), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Mul(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    }
}

HWY_ATTR void MulSizedFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                              double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        size_t i = 0;
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Mul(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    } else if (count <= kMediumThreshold) {
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            for (size_t j = 0; j < 4; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                hn::StoreU(hn::Mul(va, vb), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Mul(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    } else {
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(double);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
            }
            for (size_t j = 0; j < 8; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                hn::StoreU(hn::Mul(va, vb), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            hn::StoreU(hn::Mul(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i];
        }
    }
}

// -----------------------------------------------------------------------------
// FmaSized - Size-specialized fused multiply-add
// -----------------------------------------------------------------------------

HWY_ATTR void FmaSizedFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                              const float* HWY_RESTRICT c, float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        // Small: single pass with FMA
        size_t i = 0;
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            auto vc = hn::LoadU(d, c + i);
            hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    } else if (count <= kMediumThreshold) {
        // Medium: 4x unrolled FMA
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            auto va0 = hn::LoadU(d, a + i);
            auto va1 = hn::LoadU(d, a + i + N);
            auto va2 = hn::LoadU(d, a + i + 2 * N);
            auto va3 = hn::LoadU(d, a + i + 3 * N);
            auto vb0 = hn::LoadU(d, b + i);
            auto vb1 = hn::LoadU(d, b + i + N);
            auto vb2 = hn::LoadU(d, b + i + 2 * N);
            auto vb3 = hn::LoadU(d, b + i + 3 * N);
            auto vc0 = hn::LoadU(d, c + i);
            auto vc1 = hn::LoadU(d, c + i + N);
            auto vc2 = hn::LoadU(d, c + i + 2 * N);
            auto vc3 = hn::LoadU(d, c + i + 3 * N);
            hn::StoreU(hn::MulAdd(va0, vb0, vc0), d, out + i);
            hn::StoreU(hn::MulAdd(va1, vb1, vc1), d, out + i + N);
            hn::StoreU(hn::MulAdd(va2, vb2, vc2), d, out + i + 2 * N);
            hn::StoreU(hn::MulAdd(va3, vb3, vc3), d, out + i + 3 * N);
        }
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            auto vc = hn::LoadU(d, c + i);
            hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    } else {
        // Large: 8x unrolled + prefetch
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(float);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(c + i + stride + prefetch_elems, 0, 3);
            }
            for (size_t j = 0; j < 8; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                auto vc = hn::LoadU(d, c + i + j * N);
                hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            auto vc = hn::LoadU(d, c + i);
            hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

HWY_ATTR void FmaSizedFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                              const double* HWY_RESTRICT c, double* HWY_RESTRICT out,
                              size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        size_t i = 0;
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            auto vc = hn::LoadU(d, c + i);
            hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    } else if (count <= kMediumThreshold) {
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            for (size_t j = 0; j < 4; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                auto vc = hn::LoadU(d, c + i + j * N);
                hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            auto vc = hn::LoadU(d, c + i);
            hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    } else {
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(double);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(b + i + stride + prefetch_elems, 0, 3);
                __builtin_prefetch(c + i + stride + prefetch_elems, 0, 3);
            }
            for (size_t j = 0; j < 8; ++j) {
                auto va = hn::LoadU(d, a + i + j * N);
                auto vb = hn::LoadU(d, b + i + j * N);
                auto vc = hn::LoadU(d, c + i + j * N);
                hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i + j * N);
            }
        }
        for (; i + N <= count; i += N) {
            auto va = hn::LoadU(d, a + i);
            auto vb = hn::LoadU(d, b + i);
            auto vc = hn::LoadU(d, c + i);
            hn::StoreU(hn::MulAdd(va, vb, vc), d, out + i);
        }
        for (; i < count; ++i) {
            out[i] = a[i] * b[i] + c[i];
        }
    }
}

// -----------------------------------------------------------------------------
// ReduceSumSized - Size-specialized reduction
// -----------------------------------------------------------------------------

HWY_ATTR float ReduceSumSizedFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        // Small: single accumulator
        auto vsum = hn::Zero(d);
        size_t i = 0;
        for (; i + N <= count; i += N) {
            vsum = hn::Add(vsum, hn::LoadU(d, a + i));
        }
        float result = hn::ReduceSum(d, vsum);
        for (; i < count; ++i) {
            result += a[i];
        }
        return result;
    } else if (count <= kMediumThreshold) {
        // Medium: 4 accumulators (same as ReduceSumFast)
        auto sum0 = hn::Zero(d);
        auto sum1 = hn::Zero(d);
        auto sum2 = hn::Zero(d);
        auto sum3 = hn::Zero(d);
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
            sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
            sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
            sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
        }
        for (; i + N <= count; i += N) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        }
        const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
        float result = hn::ReduceSum(d, total);
        for (; i < count; ++i) {
            result += a[i];
        }
        return result;
    } else {
        // Large: 8 accumulators + prefetch
        auto sum0 = hn::Zero(d);
        auto sum1 = hn::Zero(d);
        auto sum2 = hn::Zero(d);
        auto sum3 = hn::Zero(d);
        auto sum4 = hn::Zero(d);
        auto sum5 = hn::Zero(d);
        auto sum6 = hn::Zero(d);
        auto sum7 = hn::Zero(d);
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(float);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
            }
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
            sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
            sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
            sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
            sum4 = hn::Add(sum4, hn::LoadU(d, a + i + 4 * N));
            sum5 = hn::Add(sum5, hn::LoadU(d, a + i + 5 * N));
            sum6 = hn::Add(sum6, hn::LoadU(d, a + i + 6 * N));
            sum7 = hn::Add(sum7, hn::LoadU(d, a + i + 7 * N));
        }
        // Handle remaining with 4x
        const size_t stride4 = 4 * N;
        for (; i + stride4 <= count; i += stride4) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
            sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
            sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
            sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
        }
        for (; i + N <= count; i += N) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        }
        // Combine all accumulators
        const auto total = hn::Add(hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)),
                                   hn::Add(hn::Add(sum4, sum5), hn::Add(sum6, sum7)));
        float result = hn::ReduceSum(d, total);
        for (; i < count; ++i) {
            result += a[i];
        }
        return result;
    }
}

HWY_ATTR double ReduceSumSizedFloat64(const double* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    if (count < kSmallThreshold) {
        auto vsum = hn::Zero(d);
        size_t i = 0;
        for (; i + N <= count; i += N) {
            vsum = hn::Add(vsum, hn::LoadU(d, a + i));
        }
        double result = hn::ReduceSum(d, vsum);
        for (; i < count; ++i) {
            result += a[i];
        }
        return result;
    } else if (count <= kMediumThreshold) {
        auto sum0 = hn::Zero(d);
        auto sum1 = hn::Zero(d);
        auto sum2 = hn::Zero(d);
        auto sum3 = hn::Zero(d);
        const size_t stride = 4 * N;
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
            sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
            sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
            sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
        }
        for (; i + N <= count; i += N) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        }
        const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
        double result = hn::ReduceSum(d, total);
        for (; i < count; ++i) {
            result += a[i];
        }
        return result;
    } else {
        auto sum0 = hn::Zero(d);
        auto sum1 = hn::Zero(d);
        auto sum2 = hn::Zero(d);
        auto sum3 = hn::Zero(d);
        auto sum4 = hn::Zero(d);
        auto sum5 = hn::Zero(d);
        auto sum6 = hn::Zero(d);
        auto sum7 = hn::Zero(d);
        const size_t stride = 8 * N;
        const size_t prefetch_elems = kPrefetchDistance / sizeof(double);
        size_t i = 0;
        for (; i + stride <= count; i += stride) {
            if (i + stride + prefetch_elems <= count) {
                __builtin_prefetch(a + i + stride + prefetch_elems, 0, 3);
            }
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
            sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
            sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
            sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
            sum4 = hn::Add(sum4, hn::LoadU(d, a + i + 4 * N));
            sum5 = hn::Add(sum5, hn::LoadU(d, a + i + 5 * N));
            sum6 = hn::Add(sum6, hn::LoadU(d, a + i + 6 * N));
            sum7 = hn::Add(sum7, hn::LoadU(d, a + i + 7 * N));
        }
        const size_t stride4 = 4 * N;
        for (; i + stride4 <= count; i += stride4) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
            sum1 = hn::Add(sum1, hn::LoadU(d, a + i + N));
            sum2 = hn::Add(sum2, hn::LoadU(d, a + i + 2 * N));
            sum3 = hn::Add(sum3, hn::LoadU(d, a + i + 3 * N));
        }
        for (; i + N <= count; i += N) {
            sum0 = hn::Add(sum0, hn::LoadU(d, a + i));
        }
        const auto total = hn::Add(hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)),
                                   hn::Add(hn::Add(sum4, sum5), hn::Add(sum6, sum7)));
        double result = hn::ReduceSum(d, total);
        for (; i < count; ++i) {
            result += a[i];
        }
        return result;
    }
}

// =============================================================================
// Fused Kernels
// =============================================================================
// Fused operations eliminate temporary arrays and improve cache efficiency

// AddMul: out[i] = (a[i] + b[i]) * c[i]
HWY_ATTR void AddMulFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                            const float* HWY_RESTRICT c, float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    // 4x unrolled loop for better instruction-level parallelism
    for (; i + 4 * N <= count; i += 4 * N) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);

        auto vc0 = hn::LoadU(d, c + i);
        auto vc1 = hn::LoadU(d, c + i + N);
        auto vc2 = hn::LoadU(d, c + i + 2 * N);
        auto vc3 = hn::LoadU(d, c + i + 3 * N);

        hn::StoreU(hn::Mul(hn::Add(va0, vb0), vc0), d, out + i);
        hn::StoreU(hn::Mul(hn::Add(va1, vb1), vc1), d, out + i + N);
        hn::StoreU(hn::Mul(hn::Add(va2, vb2), vc2), d, out + i + 2 * N);
        hn::StoreU(hn::Mul(hn::Add(va3, vb3), vc3), d, out + i + 3 * N);
    }
    // Cleanup loop
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        auto vc = hn::LoadU(d, c + i);
        hn::StoreU(hn::Mul(hn::Add(va, vb), vc), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (a[i] + b[i]) * c[i];
    }
}

HWY_ATTR void AddMulFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                            const double* HWY_RESTRICT c, double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    // 4x unrolled loop for better instruction-level parallelism
    for (; i + 4 * N <= count; i += 4 * N) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);

        auto vc0 = hn::LoadU(d, c + i);
        auto vc1 = hn::LoadU(d, c + i + N);
        auto vc2 = hn::LoadU(d, c + i + 2 * N);
        auto vc3 = hn::LoadU(d, c + i + 3 * N);

        hn::StoreU(hn::Mul(hn::Add(va0, vb0), vc0), d, out + i);
        hn::StoreU(hn::Mul(hn::Add(va1, vb1), vc1), d, out + i + N);
        hn::StoreU(hn::Mul(hn::Add(va2, vb2), vc2), d, out + i + 2 * N);
        hn::StoreU(hn::Mul(hn::Add(va3, vb3), vc3), d, out + i + 3 * N);
    }
    // Cleanup loop
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        auto vc = hn::LoadU(d, c + i);
        hn::StoreU(hn::Mul(hn::Add(va, vb), vc), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (a[i] + b[i]) * c[i];
    }
}

// SubMul: out[i] = (a[i] - b[i]) * c[i]
HWY_ATTR void SubMulFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                            const float* HWY_RESTRICT c, float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        auto vc = hn::LoadU(d, c + i);
        hn::StoreU(hn::Mul(hn::Sub(va, vb), vc), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (a[i] - b[i]) * c[i];
    }
}

HWY_ATTR void SubMulFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                            const double* HWY_RESTRICT c, double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        auto vc = hn::LoadU(d, c + i);
        hn::StoreU(hn::Mul(hn::Sub(va, vb), vc), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = (a[i] - b[i]) * c[i];
    }
}

// Axpy: out[i] = alpha * x[i] + y[i] (uses FMA for better precision)
HWY_ATTR void AxpyFloat32(float alpha, const float* HWY_RESTRICT x, const float* HWY_RESTRICT y,
                          float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto valpha = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto vx = hn::LoadU(d, x + i);
        auto vy = hn::LoadU(d, y + i);
        // FMA: alpha * x + y
        hn::StoreU(hn::MulAdd(valpha, vx, vy), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
}

HWY_ATTR void AxpyFloat64(double alpha, const double* HWY_RESTRICT x, const double* HWY_RESTRICT y,
                          double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto valpha = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto vx = hn::LoadU(d, x + i);
        auto vy = hn::LoadU(d, y + i);
        hn::StoreU(hn::MulAdd(valpha, vx, vy), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
}

// Axpby: out[i] = alpha * x[i] + beta * y[i]
HWY_ATTR void AxpbyFloat32(float alpha, const float* HWY_RESTRICT x, float beta,
                           const float* HWY_RESTRICT y, float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto valpha = hn::Set(d, alpha);
    const auto vbeta = hn::Set(d, beta);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto vx = hn::LoadU(d, x + i);
        auto vy = hn::LoadU(d, y + i);
        // alpha * x + beta * y = MulAdd(alpha, x, beta * y)
        hn::StoreU(hn::MulAdd(valpha, vx, hn::Mul(vbeta, vy)), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = alpha * x[i] + beta * y[i];
    }
}

HWY_ATTR void AxpbyFloat64(double alpha, const double* HWY_RESTRICT x, double beta,
                           const double* HWY_RESTRICT y, double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto valpha = hn::Set(d, alpha);
    const auto vbeta = hn::Set(d, beta);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto vx = hn::LoadU(d, x + i);
        auto vy = hn::LoadU(d, y + i);
        hn::StoreU(hn::MulAdd(valpha, vx, hn::Mul(vbeta, vy)), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = alpha * x[i] + beta * y[i];
    }
}

// ScaleAdd: out[i] = scale * a[i] + offset (affine transform)
HWY_ATTR void ScaleAddFloat32(const float* HWY_RESTRICT a, float scale, float offset,
                              float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto vscale = hn::Set(d, scale);
    const auto voffset = hn::Set(d, offset);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        hn::StoreU(hn::MulAdd(vscale, va, voffset), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = scale * a[i] + offset;
    }
}

HWY_ATTR void ScaleAddFloat64(const double* HWY_RESTRICT a, double scale, double offset,
                              double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto vscale = hn::Set(d, scale);
    const auto voffset = hn::Set(d, offset);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        hn::StoreU(hn::MulAdd(vscale, va, voffset), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = scale * a[i] + offset;
    }
}

// SumSq: returns sum(a[i]^2) - fused square and reduce
HWY_ATTR float SumSqFloat32(const float* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Use 4 accumulators to hide latency
    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    const size_t stride = 4 * N;
    size_t i = 0;
    for (; i + stride <= count; i += stride) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);
        sum0 = hn::MulAdd(va0, va0, sum0);
        sum1 = hn::MulAdd(va1, va1, sum1);
        sum2 = hn::MulAdd(va2, va2, sum2);
        sum3 = hn::MulAdd(va3, va3, sum3);
    }
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        sum0 = hn::MulAdd(va, va, sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i] * a[i];
    }
    return result;
}

HWY_ATTR double SumSqFloat64(const double* HWY_RESTRICT a, size_t count) {
    if (count == 0)
        return 0.0;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    const size_t stride = 4 * N;
    size_t i = 0;
    for (; i + stride <= count; i += stride) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);
        sum0 = hn::MulAdd(va0, va0, sum0);
        sum1 = hn::MulAdd(va1, va1, sum1);
        sum2 = hn::MulAdd(va2, va2, sum2);
        sum3 = hn::MulAdd(va3, va3, sum3);
    }
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        sum0 = hn::MulAdd(va, va, sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    double result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += a[i] * a[i];
    }
    return result;
}

// SumAbsDiff: returns sum(|a[i] - b[i]|) - L1 distance
HWY_ATTR float SumAbsDiffFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                                 size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    const size_t stride = 4 * N;
    size_t i = 0;
    for (; i + stride <= count; i += stride) {
        auto va0 = hn::LoadU(d, a + i);
        auto vb0 = hn::LoadU(d, b + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);
        sum0 = hn::Add(sum0, hn::Abs(hn::Sub(va0, vb0)));
        sum1 = hn::Add(sum1, hn::Abs(hn::Sub(va1, vb1)));
        sum2 = hn::Add(sum2, hn::Abs(hn::Sub(va2, vb2)));
        sum3 = hn::Add(sum3, hn::Abs(hn::Sub(va3, vb3)));
    }
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        sum0 = hn::Add(sum0, hn::Abs(hn::Sub(va, vb)));
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += std::abs(a[i] - b[i]);
    }
    return result;
}

HWY_ATTR double SumAbsDiffFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                                  size_t count) {
    if (count == 0)
        return 0.0;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    const size_t stride = 4 * N;
    size_t i = 0;
    for (; i + stride <= count; i += stride) {
        auto va0 = hn::LoadU(d, a + i);
        auto vb0 = hn::LoadU(d, b + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);
        sum0 = hn::Add(sum0, hn::Abs(hn::Sub(va0, vb0)));
        sum1 = hn::Add(sum1, hn::Abs(hn::Sub(va1, vb1)));
        sum2 = hn::Add(sum2, hn::Abs(hn::Sub(va2, vb2)));
        sum3 = hn::Add(sum3, hn::Abs(hn::Sub(va3, vb3)));
    }
    for (; i + N <= count; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        sum0 = hn::Add(sum0, hn::Abs(hn::Sub(va, vb)));
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    double result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        result += std::abs(a[i] - b[i]);
    }
    return result;
}

// SumSqDiff: returns sum((a[i] - b[i])^2) - squared L2 distance
HWY_ATTR float SumSqDiffFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                                size_t count) {
    if (count == 0)
        return 0.0f;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    const size_t stride = 4 * N;
    size_t i = 0;
    for (; i + stride <= count; i += stride) {
        auto diff0 = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        auto diff1 = hn::Sub(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N));
        auto diff2 = hn::Sub(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N));
        auto diff3 = hn::Sub(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N));
        sum0 = hn::MulAdd(diff0, diff0, sum0);
        sum1 = hn::MulAdd(diff1, diff1, sum1);
        sum2 = hn::MulAdd(diff2, diff2, sum2);
        sum3 = hn::MulAdd(diff3, diff3, sum3);
    }
    for (; i + N <= count; i += N) {
        auto diff = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        sum0 = hn::MulAdd(diff, diff, sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    float result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

HWY_ATTR double SumSqDiffFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                                 size_t count) {
    if (count == 0)
        return 0.0;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    const size_t stride = 4 * N;
    size_t i = 0;
    for (; i + stride <= count; i += stride) {
        auto diff0 = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        auto diff1 = hn::Sub(hn::LoadU(d, a + i + N), hn::LoadU(d, b + i + N));
        auto diff2 = hn::Sub(hn::LoadU(d, a + i + 2 * N), hn::LoadU(d, b + i + 2 * N));
        auto diff3 = hn::Sub(hn::LoadU(d, a + i + 3 * N), hn::LoadU(d, b + i + 3 * N));
        sum0 = hn::MulAdd(diff0, diff0, sum0);
        sum1 = hn::MulAdd(diff1, diff1, sum1);
        sum2 = hn::MulAdd(diff2, diff2, sum2);
        sum3 = hn::MulAdd(diff3, diff3, sum3);
    }
    for (; i + N <= count; i += N) {
        auto diff = hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        sum0 = hn::MulAdd(diff, diff, sum0);
    }

    const auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    double result = hn::ReduceSum(d, total);

    for (; i < count; ++i) {
        double diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

// =============================================================================
// Prefetch Utilities
// =============================================================================

// Add without prefetch - for benchmarking comparison
HWY_ATTR void AddNoPrefetchFloat32(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                                   float* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    // Simple 4x unrolled loop without prefetch
    for (; i + 4 * N <= count; i += 4 * N) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);

        hn::StoreU(hn::Add(va0, vb0), d, out + i);
        hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
        hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
        hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
    }
    for (; i + N <= count; i += N) {
        hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

HWY_ATTR void AddNoPrefetchFloat64(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                                   double* HWY_RESTRICT out, size_t count) {
    if (count == 0)
        return;

    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + 4 * N <= count; i += 4 * N) {
        auto va0 = hn::LoadU(d, a + i);
        auto va1 = hn::LoadU(d, a + i + N);
        auto va2 = hn::LoadU(d, a + i + 2 * N);
        auto va3 = hn::LoadU(d, a + i + 3 * N);

        auto vb0 = hn::LoadU(d, b + i);
        auto vb1 = hn::LoadU(d, b + i + N);
        auto vb2 = hn::LoadU(d, b + i + 2 * N);
        auto vb3 = hn::LoadU(d, b + i + 3 * N);

        hn::StoreU(hn::Add(va0, vb0), d, out + i);
        hn::StoreU(hn::Add(va1, vb1), d, out + i + N);
        hn::StoreU(hn::Add(va2, vb2), d, out + i + 2 * N);
        hn::StoreU(hn::Add(va3, vb3), d, out + i + 3 * N);
    }
    for (; i + N <= count; i += N) {
        hn::StoreU(hn::Add(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Process with prefetch for strided access patterns
HWY_ATTR void ProcessWithPrefetchFloat32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                                         size_t count, size_t stride) {
    if (count == 0)
        return;

    constexpr size_t prefetch_distance = 256 / sizeof(float);  // 64 elements

    for (size_t i = 0; i < count; ++i) {
        // Prefetch ahead for strided access
        size_t prefetch_idx = i + prefetch_distance;
        if (prefetch_idx < count) {
            __builtin_prefetch(in + prefetch_idx * stride, 0, 3);
        }

        // Simple copy with stride
        out[i] = in[i * stride];
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_BUD_OPS_INL_H_ toggle guard
