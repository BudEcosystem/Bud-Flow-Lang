// =============================================================================
// Bud Flow Lang - Highway Expert Tier Python Bindings
// =============================================================================
//
// Provides Python access to Highway SIMD operations for Expert-tier users.
//

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/codegen/hwy_ops.h"
#include "bud_flow_lang/highway/highway_types.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using namespace bud::highway;

void bind_highway(nb::module_& m) {
    // Create highway submodule
    auto highway = m.def_submodule("highway", "Expert-tier Highway SIMD operations");

    // =========================================================================
    // Scalar type constants
    // =========================================================================
    highway.attr("float32") = bud::ScalarType::kFloat32;
    highway.attr("float64") = bud::ScalarType::kFloat64;
    highway.attr("int32") = bud::ScalarType::kInt32;
    highway.attr("int64") = bud::ScalarType::kInt64;
    highway.attr("uint8") = bud::ScalarType::kUint8;

    // =========================================================================
    // SIMD Info functions
    // =========================================================================
    highway.def(
        "get_simd_lanes_f32", []() { return getSimdLanesF32(); },
        "Get number of float32 lanes available on this CPU");

    highway.def(
        "get_simd_lanes_f64", []() { return getSimdLanesF64(); },
        "Get number of float64 lanes available on this CPU");

    highway.def(
        "get_simd_lanes_i32", []() { return getSimdLanesI32(); },
        "Get number of int32 lanes available on this CPU");

    highway.def(
        "get_simd_target", []() -> std::string { return getSimdTarget(); },
        "Get the SIMD target being used (e.g., 'AVX2', 'AVX-512')");

    // =========================================================================
    // TagInfo classes for type information
    // =========================================================================
    nb::class_<TagF32>(highway, "TagF32", "Tag for float32 SIMD operations")
        .def(nb::init<>())
        .def_prop_ro_static(
            "lanes", [](nb::handle) { return getSimdLanesF32(); }, "Number of float32 lanes")
        .def_prop_ro_static(
            "dtype", [](nb::handle) { return bud::ScalarType::kFloat32; }, "Scalar type")
        .def_prop_ro_static("type_name", [](nb::handle) { return "float32"; }, "Type name");

    nb::class_<TagF64>(highway, "TagF64", "Tag for float64 SIMD operations")
        .def(nb::init<>())
        .def_prop_ro_static(
            "lanes", [](nb::handle) { return getSimdLanesF64(); }, "Number of float64 lanes")
        .def_prop_ro_static(
            "dtype", [](nb::handle) { return bud::ScalarType::kFloat64; }, "Scalar type")
        .def_prop_ro_static("type_name", [](nb::handle) { return "float64"; }, "Type name");

    nb::class_<TagI32>(highway, "TagI32", "Tag for int32 SIMD operations")
        .def(nb::init<>())
        .def_prop_ro_static(
            "lanes", [](nb::handle) { return getSimdLanesI32(); }, "Number of int32 lanes")
        .def_prop_ro_static(
            "dtype", [](nb::handle) { return bud::ScalarType::kInt32; }, "Scalar type")
        .def_prop_ro_static("type_name", [](nb::handle) { return "int32"; }, "Type name");

    // =========================================================================
    // SIMD Operations (using existing hwy_ops)
    // =========================================================================

    // These functions operate on Bunch objects using the existing hwy_ops
    highway.def(
        "add", [](const bud::Bunch& a, const bud::Bunch& b) { return a + b; }, nb::arg("a"),
        nb::arg("b"), "Element-wise addition");

    highway.def(
        "sub", [](const bud::Bunch& a, const bud::Bunch& b) { return a - b; }, nb::arg("a"),
        nb::arg("b"), "Element-wise subtraction");

    highway.def(
        "mul", [](const bud::Bunch& a, const bud::Bunch& b) { return a * b; }, nb::arg("a"),
        nb::arg("b"), "Element-wise multiplication");

    highway.def(
        "div", [](const bud::Bunch& a, const bud::Bunch& b) { return a / b; }, nb::arg("a"),
        nb::arg("b"), "Element-wise division");

    highway.def(
        "neg", [](const bud::Bunch& a) { return -a; }, nb::arg("a"), "Element-wise negation");

    highway.def(
        "abs", [](const bud::Bunch& a) { return a.abs(); }, nb::arg("a"),
        "Element-wise absolute value");

    highway.def(
        "sqrt", [](const bud::Bunch& a) { return a.sqrt(); }, nb::arg("a"),
        "Element-wise square root");

    highway.def(
        "rsqrt", [](const bud::Bunch& a) { return a.rsqrt(); }, nb::arg("a"),
        "Element-wise reciprocal square root");

    highway.def(
        "exp", [](const bud::Bunch& a) { return a.exp(); }, nb::arg("a"),
        "Element-wise exponential");

    highway.def(
        "log", [](const bud::Bunch& a) { return a.log(); }, nb::arg("a"),
        "Element-wise natural logarithm");

    highway.def(
        "sin", [](const bud::Bunch& a) { return a.sin(); }, nb::arg("a"), "Element-wise sine");

    highway.def(
        "cos", [](const bud::Bunch& a) { return a.cos(); }, nb::arg("a"), "Element-wise cosine");

    highway.def(
        "tanh", [](const bud::Bunch& a) { return a.tanh(); }, nb::arg("a"),
        "Element-wise hyperbolic tangent");

    // =========================================================================
    // Extended Math Operations
    // =========================================================================
    highway.def(
        "exp2",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchExp2(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise base-2 exponential (2^x)");

    highway.def(
        "log2",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchLog2(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise base-2 logarithm");

    highway.def(
        "log10",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchLog10(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise base-10 logarithm");

    highway.def(
        "sinh",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchSinh(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise hyperbolic sine");

    highway.def(
        "cosh",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchCosh(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise hyperbolic cosine");

    highway.def(
        "asinh",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchAsinh(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise inverse hyperbolic sine");

    highway.def(
        "acosh",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchAcosh(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise inverse hyperbolic cosine");

    highway.def(
        "atanh",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchAtanh(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise inverse hyperbolic tangent");

    // =========================================================================
    // Inverse Trigonometric Operations
    // =========================================================================
    highway.def(
        "asin",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchAsin(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise arc sine");

    highway.def(
        "acos",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchAcos(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise arc cosine");

    highway.def(
        "atan",
        [](const bud::Bunch& a) {
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchAtan(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Element-wise arc tangent");

    highway.def(
        "atan2",
        [](const bud::Bunch& y, const bud::Bunch& x) {
            if (y.size() != x.size()) {
                throw std::runtime_error("atan2: size mismatch");
            }
            auto result = bud::Bunch::zeros(y.size(), y.dtype());
            if (!result)
                return y;
            auto dispatch_result = bud::simd::DispatchAtan2(result->mutableData(), y.data(),
                                                            x.data(), y.size(), y.dtype());
            if (!dispatch_result)
                return y;
            return std::move(*result);
        },
        nb::arg("y"), nb::arg("x"), "Element-wise two-argument arc tangent (atan2(y, x))");

    // =========================================================================
    // FMA Operations
    // =========================================================================
    highway.def(
        "mul_add",
        [](const bud::Bunch& a, const bud::Bunch& b, const bud::Bunch& c) {
            return bud::fma(a, b, c);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("c"), "Fused multiply-add: a * b + c");

    highway.def(
        "mul_sub",
        [](const bud::Bunch& a, const bud::Bunch& b, const bud::Bunch& c) {
            if (a.size() != b.size() || a.size() != c.size()) {
                throw std::runtime_error("mul_sub: size mismatch");
            }
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result = bud::simd::DispatchMulSub(
                result->mutableData(), a.data(), b.data(), c.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("c"), "Fused multiply-subtract: a * b - c");

    highway.def(
        "neg_mul_add",
        [](const bud::Bunch& a, const bud::Bunch& b, const bud::Bunch& c) {
            if (a.size() != b.size() || a.size() != c.size()) {
                throw std::runtime_error("neg_mul_add: size mismatch");
            }
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result = bud::simd::DispatchNegMulAdd(
                result->mutableData(), a.data(), b.data(), c.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("c"), "Fused negative multiply-add: c - a * b");

    // Reductions
    highway.def(
        "reduce_sum", [](const bud::Bunch& a) { return a.sum(); }, nb::arg("a"),
        "Sum all elements");

    highway.def(
        "reduce_min", [](const bud::Bunch& a) { return a.min(); }, nb::arg("a"), "Minimum element");

    highway.def(
        "reduce_max", [](const bud::Bunch& a) { return a.max(); }, nb::arg("a"), "Maximum element");

    highway.def(
        "dot_product", [](const bud::Bunch& a, const bud::Bunch& b) { return a.dot(b); },
        nb::arg("a"), nb::arg("b"), "Dot product of two arrays");

    // Comparison
    highway.def(
        "eq", [](const bud::Bunch& a, const bud::Bunch& b) { return a.eq(b); }, nb::arg("a"),
        nb::arg("b"), "Element-wise equality comparison (returns mask Bunch)");

    highway.def(
        "lt", [](const bud::Bunch& a, const bud::Bunch& b) { return a.lt(b); }, nb::arg("a"),
        nb::arg("b"), "Element-wise less-than comparison");

    highway.def(
        "le", [](const bud::Bunch& a, const bud::Bunch& b) { return a.le(b); }, nb::arg("a"),
        nb::arg("b"), "Element-wise less-or-equal comparison");

    highway.def(
        "gt", [](const bud::Bunch& a, const bud::Bunch& b) { return a.gt(b); }, nb::arg("a"),
        nb::arg("b"), "Element-wise greater-than comparison");

    highway.def(
        "ge", [](const bud::Bunch& a, const bud::Bunch& b) { return a.ge(b); }, nb::arg("a"),
        nb::arg("b"), "Element-wise greater-or-equal comparison");

    // Conditional selection
    highway.def(
        "where",
        [](const bud::Bunch& mask, const bud::Bunch& if_true, const bud::Bunch& if_false) {
            return if_true.where(mask, if_false);
        },
        nb::arg("mask"), nb::arg("if_true"), nb::arg("if_false"),
        "Select elements: mask ? if_true : if_false");

    // Clamp
    highway.def(
        "clamp", [](const bud::Bunch& a, float lo, float hi) { return bud::clamp(a, lo, hi); },
        nb::arg("a"), nb::arg("lo"), nb::arg("hi"), "Clamp values to range [lo, hi]");

    // =========================================================================
    // Rounding Operations
    // =========================================================================
    highway.def(
        "round", [](const bud::Bunch& a) { return a.round(); }, nb::arg("a"),
        "Element-wise rounding to nearest integer");

    highway.def(
        "floor", [](const bud::Bunch& a) { return a.floor(); }, nb::arg("a"),
        "Element-wise floor (round down)");

    highway.def(
        "ceil", [](const bud::Bunch& a) { return a.ceil(); }, nb::arg("a"),
        "Element-wise ceiling (round up)");

    highway.def(
        "trunc", [](const bud::Bunch& a) { return a.trunc(); }, nb::arg("a"),
        "Element-wise truncation (round toward zero)");

    // =========================================================================
    // Special Value Checks
    // =========================================================================
    highway.def(
        "isnan", [](const bud::Bunch& a) { return a.isnan(); }, nb::arg("a"),
        "Element-wise NaN check (returns mask)");

    highway.def(
        "isinf", [](const bud::Bunch& a) { return a.isinf(); }, nb::arg("a"),
        "Element-wise infinity check (returns mask)");

    highway.def(
        "isfinite", [](const bud::Bunch& a) { return a.isfinite(); }, nb::arg("a"),
        "Element-wise finite check (returns mask)");

    // =========================================================================
    // Min/Max Operations
    // =========================================================================
    highway.def(
        "minimum", [](const bud::Bunch& a, const bud::Bunch& b) { return a.minimum(b); },
        nb::arg("a"), nb::arg("b"), "Element-wise minimum");

    highway.def(
        "maximum", [](const bud::Bunch& a, const bud::Bunch& b) { return a.maximum(b); },
        nb::arg("a"), nb::arg("b"), "Element-wise maximum");

    // =========================================================================
    // SIMD Introspection APIs
    // =========================================================================
    highway.def(
        "get_vector_bits",
        []() -> size_t {
            return getSimdLanesF32() * 32;  // lanes * bits per float
        },
        "Get SIMD vector width in bits (e.g., 256 for AVX2, 512 for AVX-512)");

    highway.def(
        "get_alignment",
        []() -> size_t {
            return getSimdLanesF32() * sizeof(float);  // Optimal alignment in bytes
        },
        "Get optimal memory alignment in bytes for SIMD operations");

    highway.def(
        "is_aligned",
        [](const bud::Bunch& bunch, size_t alignment) -> bool {
            if (alignment == 0) {
                alignment = getSimdLanesF32() * sizeof(float);
            }
            return (reinterpret_cast<uintptr_t>(bunch.data()) % alignment) == 0;
        },
        nb::arg("bunch"), nb::arg("alignment") = 0,
        "Check if bunch data is aligned (default: optimal SIMD alignment)");

    // =========================================================================
    // Bitwise Operations (Integer Types Only)
    // =========================================================================
    highway.def(
        "bitwise_and",
        [](const bud::Bunch& a, const bud::Bunch& b) {
            if (a.size() != b.size()) {
                throw std::runtime_error("bitwise_and: size mismatch");
            }
            if (a.dtype() != bud::ScalarType::kInt32 && a.dtype() != bud::ScalarType::kInt64) {
                throw std::runtime_error("bitwise_and: only int32/int64 supported");
            }
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result = bud::simd::DispatchBitwiseAnd(result->mutableData(), a.data(),
                                                                 b.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), nb::arg("b"), "Bitwise AND: a & b (integer types only)");

    highway.def(
        "bitwise_or",
        [](const bud::Bunch& a, const bud::Bunch& b) {
            if (a.size() != b.size()) {
                throw std::runtime_error("bitwise_or: size mismatch");
            }
            if (a.dtype() != bud::ScalarType::kInt32 && a.dtype() != bud::ScalarType::kInt64) {
                throw std::runtime_error("bitwise_or: only int32/int64 supported");
            }
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result = bud::simd::DispatchBitwiseOr(result->mutableData(), a.data(),
                                                                b.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), nb::arg("b"), "Bitwise OR: a | b (integer types only)");

    highway.def(
        "bitwise_xor",
        [](const bud::Bunch& a, const bud::Bunch& b) {
            if (a.size() != b.size()) {
                throw std::runtime_error("bitwise_xor: size mismatch");
            }
            if (a.dtype() != bud::ScalarType::kInt32 && a.dtype() != bud::ScalarType::kInt64) {
                throw std::runtime_error("bitwise_xor: only int32/int64 supported");
            }
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result = bud::simd::DispatchBitwiseXor(result->mutableData(), a.data(),
                                                                 b.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), nb::arg("b"), "Bitwise XOR: a ^ b (integer types only)");

    highway.def(
        "bitwise_not",
        [](const bud::Bunch& a) {
            if (a.dtype() != bud::ScalarType::kInt32 && a.dtype() != bud::ScalarType::kInt64) {
                throw std::runtime_error("bitwise_not: only int32/int64 supported");
            }
            auto result = bud::Bunch::zeros(a.size(), a.dtype());
            if (!result)
                return a;
            auto dispatch_result =
                bud::simd::DispatchBitwiseNot(result->mutableData(), a.data(), a.size(), a.dtype());
            if (!dispatch_result)
                return a;
            return std::move(*result);
        },
        nb::arg("a"), "Bitwise NOT: ~a (integer types only)");

    // =========================================================================
    // Raw pointer operations (for expert use)
    // =========================================================================
    highway.def(
        "data_ptr", [](const bud::Bunch& bunch) { return reinterpret_cast<size_t>(bunch.data()); },
        nb::arg("bunch"), "Get raw data pointer address (for expert-level operations)");

    highway.def(
        "mutable_data_ptr",
        [](bud::Bunch& bunch) { return reinterpret_cast<size_t>(bunch.mutableData()); },
        nb::arg("bunch"), "Get mutable raw data pointer address");
}
