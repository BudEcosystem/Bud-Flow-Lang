// =============================================================================
// Bud Flow Lang - Bunch Python Bindings
// =============================================================================
//
// Python bindings for the Bunch class with NumPy interoperability.
// Features:
// - NumPy buffer protocol support (zero-copy where possible)
// - All arithmetic and comparison operators
// - Math functions (sin, cos, exp, log, etc.)
// - Reduction operations (sum, mean, min, max, dot)
// - Fused operations (fma, clamp, lerp)
//

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/codegen/hwy_ops.h"
#include "bud_flow_lang/type_system.h"

#include <cstdint>
#include <cstring>
#include <sstream>

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Get NumPy dtype format string from ScalarType
const char* getNumpyDtypeFormat(bud::ScalarType dtype) {
    switch (dtype) {
    case bud::ScalarType::kFloat32:
        return "f";
    case bud::ScalarType::kFloat64:
        return "d";
    case bud::ScalarType::kInt8:
        return "b";
    case bud::ScalarType::kInt16:
        return "h";
    case bud::ScalarType::kInt32:
        return "i";
    case bud::ScalarType::kInt64:
        return "q";
    case bud::ScalarType::kUint8:
        return "B";
    case bud::ScalarType::kUint16:
        return "H";
    case bud::ScalarType::kUint32:
        return "I";
    case bud::ScalarType::kUint64:
        return "Q";
    case bud::ScalarType::kBool:
        return "?";
    default:
        return "f";  // Default to float32
    }
}

// Get Python type string for dtype property
const char* getDtypeString(bud::ScalarType dtype) {
    switch (dtype) {
    case bud::ScalarType::kFloat32:
        return "float32";
    case bud::ScalarType::kFloat64:
        return "float64";
    case bud::ScalarType::kInt8:
        return "int8";
    case bud::ScalarType::kInt16:
        return "int16";
    case bud::ScalarType::kInt32:
        return "int32";
    case bud::ScalarType::kInt64:
        return "int64";
    case bud::ScalarType::kUint8:
        return "uint8";
    case bud::ScalarType::kUint16:
        return "uint16";
    case bud::ScalarType::kUint32:
        return "uint32";
    case bud::ScalarType::kUint64:
        return "uint64";
    case bud::ScalarType::kFloat16:
        return "float16";
    case bud::ScalarType::kBFloat16:
        return "bfloat16";
    case bud::ScalarType::kBool:
        return "bool";
    default:
        return "unknown";
    }
}

}  // namespace

// =============================================================================
// Bunch Python Bindings
// =============================================================================

void bind_bunch(nb::module_& m) {
    // Expose ScalarType enum
    nb::enum_<bud::ScalarType>(m, "ScalarType", "Data types supported by Bunch")
        .value("Float16", bud::ScalarType::kFloat16)
        .value("Float32", bud::ScalarType::kFloat32)
        .value("Float64", bud::ScalarType::kFloat64)
        .value("BFloat16", bud::ScalarType::kBFloat16)
        .value("Int8", bud::ScalarType::kInt8)
        .value("Int16", bud::ScalarType::kInt16)
        .value("Int32", bud::ScalarType::kInt32)
        .value("Int64", bud::ScalarType::kInt64)
        .value("Uint8", bud::ScalarType::kUint8)
        .value("Uint16", bud::ScalarType::kUint16)
        .value("Uint32", bud::ScalarType::kUint32)
        .value("Uint64", bud::ScalarType::kUint64)
        .value("Bool", bud::ScalarType::kBool);

    // Main Bunch class binding
    nb::class_<bud::Bunch>(m, "Bunch",
                           "SIMD-optimized array for high-performance numerical computing")
        // =====================================================================
        // Factory methods
        // =====================================================================

        .def_static(
            "zeros",
            [](size_t count) {
                auto result = bud::Bunch::zeros(count);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("count"), "Create a Bunch of zeros")

        .def_static(
            "ones",
            [](size_t count) {
                auto result = bud::Bunch::ones(count);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("count"), "Create a Bunch of ones")

        .def_static(
            "fill",
            [](size_t count, float value) {
                auto result = bud::Bunch::fill(count, value);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("count"), nb::arg("value"), "Create a Bunch filled with value")

        .def_static(
            "arange",
            [](size_t count, float start, float step) {
                auto result = bud::Bunch::arange(count, start, step);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("count"), nb::arg("start") = 0.0f, nb::arg("step") = 1.0f,
            "Create a Bunch with values [start, start+step, start+2*step, ...]")

        .def_static(
            "from_list",
            [](const std::vector<float>& data) {
                auto result = bud::Bunch::fromData(data.data(), data.size());
                if (!result)
                    throw std::runtime_error("Failed to create Bunch from list");
                return std::move(*result);
            },
            nb::arg("data"), "Create a Bunch from a Python list of floats")

        // =====================================================================
        // NumPy interoperability
        // =====================================================================

        .def_static(
            "from_numpy",
            [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig> arr) {
                auto result = bud::Bunch::fromData(arr.data(), arr.shape(0));
                if (!result)
                    throw std::runtime_error("Failed to create Bunch from NumPy array: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("array"), "Create a Bunch from a 1D NumPy array (float32). Data is copied.")

        .def_static(
            "from_numpy",
            [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> arr) {
                auto result = bud::Bunch::fromData(arr.data(), arr.shape(0));
                if (!result)
                    throw std::runtime_error("Failed to create Bunch from NumPy array: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("array"), "Create a Bunch from a 1D NumPy array (float64). Data is copied.")

        .def_static(
            "from_numpy",
            [](nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> arr) {
                auto result = bud::Bunch::fromData(arr.data(), arr.shape(0));
                if (!result)
                    throw std::runtime_error("Failed to create Bunch from NumPy array: " +
                                             result.error().toString());
                return std::move(*result);
            },
            nb::arg("array"), "Create a Bunch from a 1D NumPy array (int32). Data is copied.")

        .def(
            "to_numpy",
            [](const bud::Bunch& self) {
                // Get data type and size
                size_t count = self.size();
                bud::ScalarType dtype = self.dtype();
                size_t elem_size = bud::scalarTypeSize(dtype);

                // Allocate NumPy array and copy data
                // For float32 (most common case)
                if (dtype == bud::ScalarType::kFloat32) {
                    auto result =
                        nb::ndarray<nb::numpy, float, nb::ndim<1>>(new float[count], {count});

                    // Copy data from Bunch to NumPy array
                    const float* src = static_cast<const float*>(self.data());
                    float* dst = result.data();
                    std::memcpy(dst, src, count * sizeof(float));

                    return nb::cast(result);
                }
                // For float64
                else if (dtype == bud::ScalarType::kFloat64) {
                    auto result =
                        nb::ndarray<nb::numpy, double, nb::ndim<1>>(new double[count], {count});

                    const double* src = static_cast<const double*>(self.data());
                    double* dst = result.data();
                    std::memcpy(dst, src, count * sizeof(double));

                    return nb::cast(result);
                }
                // For int32
                else if (dtype == bud::ScalarType::kInt32) {
                    auto result =
                        nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(new int32_t[count], {count});

                    const int32_t* src = static_cast<const int32_t*>(self.data());
                    int32_t* dst = result.data();
                    std::memcpy(dst, src, count * sizeof(int32_t));

                    return nb::cast(result);
                }
                // Default: cast to float32
                else {
                    auto result =
                        nb::ndarray<nb::numpy, float, nb::ndim<1>>(new float[count], {count});

                    const float* src = static_cast<const float*>(self.data());
                    float* dst = result.data();
                    std::memcpy(dst, src, count * sizeof(float));

                    return nb::cast(result);
                }
            },
            "Convert Bunch to a NumPy array. Data is copied.")

        // =====================================================================
        // Properties
        // =====================================================================

        .def_prop_ro("size", &bud::Bunch::size, "Number of elements")
        .def("__len__", &bud::Bunch::size)
        .def("is_valid", &bud::Bunch::isValid, "Check if the Bunch is valid")

        .def_prop_ro(
            "dtype", [](const bud::Bunch& self) { return getDtypeString(self.dtype()); },
            "Data type as string (e.g., 'float32')")

        .def_prop_ro("dtype_enum", &bud::Bunch::dtype, "Data type as ScalarType enum")

        .def_prop_ro(
            "shape",
            [](const bud::Bunch& self) {
                const auto& shape = self.shape();
                nb::tuple result = nb::make_tuple(self.size());
                return result;
            },
            "Shape as tuple (currently 1D only)")

        .def_prop_ro(
            "nbytes",
            [](const bud::Bunch& self) { return self.size() * bud::scalarTypeSize(self.dtype()); },
            "Total bytes consumed by the array data")

        .def_prop_ro(
            "itemsize", [](const bud::Bunch& self) { return bud::scalarTypeSize(self.dtype()); },
            "Size of each element in bytes")

        .def_prop_ro(
            "ndim", [](const bud::Bunch&) { return 1; }, "Number of dimensions (always 1)")

        // =====================================================================
        // Element access
        // =====================================================================

        .def(
            "__getitem__",
            [](const bud::Bunch& self, int64_t index) {
                int64_t size = static_cast<int64_t>(self.size());

                // Handle negative indices (Python style)
                if (index < 0) {
                    index += size;
                }

                // Bounds check
                if (index < 0 || index >= size) {
                    throw nb::index_error("Bunch index out of range");
                }

                // Return value based on dtype
                bud::ScalarType dtype = self.dtype();
                if (dtype == bud::ScalarType::kFloat32) {
                    return nb::cast(static_cast<const float*>(self.data())[index]);
                } else if (dtype == bud::ScalarType::kFloat64) {
                    return nb::cast(static_cast<const double*>(self.data())[index]);
                } else if (dtype == bud::ScalarType::kInt32) {
                    return nb::cast(static_cast<const int32_t*>(self.data())[index]);
                } else if (dtype == bud::ScalarType::kInt64) {
                    return nb::cast(static_cast<const int64_t*>(self.data())[index]);
                }
                // Default to float32
                return nb::cast(static_cast<const float*>(self.data())[index]);
            },
            nb::arg("index"), "Get element at index (supports negative indices)")

        // Slice support: x[start:stop:step]
        .def(
            "__getitem__",
            [](const bud::Bunch& self, nb::slice slice) {
                int64_t size = static_cast<int64_t>(self.size());

                // Extract slice indices using Python's slice.indices()
                Py_ssize_t start, stop, step, slicelength;
                if (PySlice_GetIndicesEx(slice.ptr(), size, &start, &stop, &step, &slicelength) <
                    0) {
                    throw nb::python_error();
                }

                if (slicelength == 0) {
                    // Return empty Bunch
                    auto result = bud::Bunch::zeros(0, self.dtype());
                    if (!result)
                        throw std::runtime_error("Failed to create empty Bunch");
                    return std::move(*result);
                }

                // Create result Bunch
                auto result = bud::Bunch::zeros(static_cast<size_t>(slicelength), self.dtype());
                if (!result)
                    throw std::runtime_error("Failed to create sliced Bunch");

                // Copy elements based on dtype
                if (self.dtype() == bud::ScalarType::kFloat32) {
                    const float* src = static_cast<const float*>(self.data());
                    float* dst = static_cast<float*>(result->mutableData());
                    for (Py_ssize_t i = 0; i < slicelength; ++i) {
                        dst[i] = src[start + i * step];
                    }
                } else if (self.dtype() == bud::ScalarType::kFloat64) {
                    const double* src = static_cast<const double*>(self.data());
                    double* dst = static_cast<double*>(result->mutableData());
                    for (Py_ssize_t i = 0; i < slicelength; ++i) {
                        dst[i] = src[start + i * step];
                    }
                } else if (self.dtype() == bud::ScalarType::kInt32) {
                    const int32_t* src = static_cast<const int32_t*>(self.data());
                    int32_t* dst = static_cast<int32_t*>(result->mutableData());
                    for (Py_ssize_t i = 0; i < slicelength; ++i) {
                        dst[i] = src[start + i * step];
                    }
                } else {
                    // Default: treat as float32
                    const float* src = static_cast<const float*>(self.data());
                    float* dst = static_cast<float*>(result->mutableData());
                    for (Py_ssize_t i = 0; i < slicelength; ++i) {
                        dst[i] = src[start + i * step];
                    }
                }

                return std::move(*result);
            },
            nb::arg("slice"), "Get a slice of elements (supports start:stop:step)")

        // =====================================================================
        // Arithmetic operators (element-wise)
        // =====================================================================

        .def("__add__", [](const bud::Bunch& a, const bud::Bunch& b) { return a + b; })
        .def("__sub__", [](const bud::Bunch& a, const bud::Bunch& b) { return a - b; })
        .def("__mul__", [](const bud::Bunch& a, const bud::Bunch& b) { return a * b; })
        .def("__truediv__", [](const bud::Bunch& a, const bud::Bunch& b) { return a / b; })

        // Scalar arithmetic
        .def("__add__", [](const bud::Bunch& a, float b) { return a + b; })
        .def("__radd__", [](const bud::Bunch& a, float b) { return a + b; })
        .def("__sub__", [](const bud::Bunch& a, float b) { return a - b; })
        .def("__rsub__",
             [](const bud::Bunch& a, float b) {
                 // b - a = -(a) + b
                 auto neg_a = -a;
                 return neg_a + b;
             })
        .def("__mul__", [](const bud::Bunch& a, float b) { return a * b; })
        .def("__rmul__", [](const bud::Bunch& a, float b) { return a * b; })
        .def("__truediv__", [](const bud::Bunch& a, float b) { return a / b; })
        .def("__rtruediv__",
             [](const bud::Bunch& a, float b) {
                 // b / a - need reciprocal
                 auto ones_bunch = bud::Bunch::fill(a.size(), b);
                 if (!ones_bunch)
                     throw std::runtime_error("Failed to create scalar Bunch");
                 return *ones_bunch / a;
             })

        // Power operations
        .def(
            "__pow__",
            [](const bud::Bunch& a, float exponent) {
                // x^n = exp(n * log(x))
                auto log_a = a.log();
                auto scaled = log_a * exponent;
                return scaled.exp();
            },
            nb::arg("exponent"), "Element-wise power (x^n)")

        // =====================================================================
        // In-place arithmetic operators
        // =====================================================================

        .def(
            "__iadd__",
            [](bud::Bunch& self, const bud::Bunch& other) -> bud::Bunch& {
                if (self.size() != other.size()) {
                    throw std::runtime_error("Size mismatch for in-place addition");
                }
                auto result = bud::simd::DispatchAdd(self.mutableData(), self.data(), other.data(),
                                                     self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place addition failed");
                return self;
            },
            nb::arg("other"), "In-place addition")

        .def(
            "__iadd__",
            [](bud::Bunch& self, float scalar) -> bud::Bunch& {
                auto scalar_bunch = bud::Bunch::fill(self.size(), scalar);
                if (!scalar_bunch)
                    throw std::runtime_error("Failed to create scalar Bunch");
                auto result =
                    bud::simd::DispatchAdd(self.mutableData(), self.data(), scalar_bunch->data(),
                                           self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place addition failed");
                return self;
            },
            nb::arg("scalar"), "In-place addition with scalar")

        .def(
            "__isub__",
            [](bud::Bunch& self, const bud::Bunch& other) -> bud::Bunch& {
                if (self.size() != other.size()) {
                    throw std::runtime_error("Size mismatch for in-place subtraction");
                }
                auto result = bud::simd::DispatchSub(self.mutableData(), self.data(), other.data(),
                                                     self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place subtraction failed");
                return self;
            },
            nb::arg("other"), "In-place subtraction")

        .def(
            "__isub__",
            [](bud::Bunch& self, float scalar) -> bud::Bunch& {
                auto scalar_bunch = bud::Bunch::fill(self.size(), scalar);
                if (!scalar_bunch)
                    throw std::runtime_error("Failed to create scalar Bunch");
                auto result =
                    bud::simd::DispatchSub(self.mutableData(), self.data(), scalar_bunch->data(),
                                           self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place subtraction failed");
                return self;
            },
            nb::arg("scalar"), "In-place subtraction with scalar")

        .def(
            "__imul__",
            [](bud::Bunch& self, const bud::Bunch& other) -> bud::Bunch& {
                if (self.size() != other.size()) {
                    throw std::runtime_error("Size mismatch for in-place multiplication");
                }
                auto result = bud::simd::DispatchMul(self.mutableData(), self.data(), other.data(),
                                                     self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place multiplication failed");
                return self;
            },
            nb::arg("other"), "In-place multiplication")

        .def(
            "__imul__",
            [](bud::Bunch& self, float scalar) -> bud::Bunch& {
                auto scalar_bunch = bud::Bunch::fill(self.size(), scalar);
                if (!scalar_bunch)
                    throw std::runtime_error("Failed to create scalar Bunch");
                auto result =
                    bud::simd::DispatchMul(self.mutableData(), self.data(), scalar_bunch->data(),
                                           self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place multiplication failed");
                return self;
            },
            nb::arg("scalar"), "In-place multiplication with scalar")

        .def(
            "__itruediv__",
            [](bud::Bunch& self, const bud::Bunch& other) -> bud::Bunch& {
                if (self.size() != other.size()) {
                    throw std::runtime_error("Size mismatch for in-place division");
                }
                auto result = bud::simd::DispatchDiv(self.mutableData(), self.data(), other.data(),
                                                     self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place division failed");
                return self;
            },
            nb::arg("other"), "In-place division")

        .def(
            "__itruediv__",
            [](bud::Bunch& self, float scalar) -> bud::Bunch& {
                auto scalar_bunch = bud::Bunch::fill(self.size(), scalar);
                if (!scalar_bunch)
                    throw std::runtime_error("Failed to create scalar Bunch");
                auto result =
                    bud::simd::DispatchDiv(self.mutableData(), self.data(), scalar_bunch->data(),
                                           self.size(), self.dtype());
                if (!result)
                    throw std::runtime_error("In-place division failed");
                return self;
            },
            nb::arg("scalar"), "In-place division with scalar")

        // =====================================================================
        // Unary operators
        // =====================================================================

        .def("__neg__", [](const bud::Bunch& a) { return -a; })
        .def(
            "__abs__", [](const bud::Bunch& a) { return a.abs(); }, "Element-wise absolute value")
        .def("__pos__", [](const bud::Bunch& a) { return a; })

        .def("abs", &bud::Bunch::abs, "Element-wise absolute value")
        .def("sqrt", &bud::Bunch::sqrt, "Element-wise square root")
        .def("rsqrt", &bud::Bunch::rsqrt, "Element-wise reciprocal square root (1/sqrt)")
        .def("exp", &bud::Bunch::exp, "Element-wise exponential")
        .def("log", &bud::Bunch::log, "Element-wise natural logarithm")
        .def("sin", &bud::Bunch::sin, "Element-wise sine")
        .def("cos", &bud::Bunch::cos, "Element-wise cosine")
        .def("tanh", &bud::Bunch::tanh, "Element-wise hyperbolic tangent")
        .def("tan", &bud::Bunch::tan, "Element-wise tangent")
        .def("sigmoid", &bud::Bunch::sigmoid, "Element-wise sigmoid (1/(1+exp(-x)))")
        .def("floor", &bud::Bunch::floor, "Element-wise floor")
        .def("ceil", &bud::Bunch::ceil, "Element-wise ceiling")
        .def("round", &bud::Bunch::round, "Element-wise rounding to nearest integer")
        .def("trunc", &bud::Bunch::trunc, "Element-wise truncation toward zero")
        .def("isnan", &bud::Bunch::isnan, "Element-wise NaN check (returns mask)")
        .def("isinf", &bud::Bunch::isinf, "Element-wise infinity check (returns mask)")
        .def("isfinite", &bud::Bunch::isfinite, "Element-wise finite check (returns mask)")

        // =====================================================================
        // Element-wise min/max
        // =====================================================================

        .def("minimum", &bud::Bunch::minimum, nb::arg("other"), "Element-wise minimum")
        .def("maximum", &bud::Bunch::maximum, nb::arg("other"), "Element-wise maximum")

        // =====================================================================
        // Comparison operators
        // =====================================================================

        .def(
            "__eq__", [](const bud::Bunch& a, const bud::Bunch& b) { return a.eq(b); },
            "Element-wise equality comparison")
        .def(
            "__lt__", [](const bud::Bunch& a, const bud::Bunch& b) { return a.lt(b); },
            "Element-wise less-than comparison")
        .def(
            "__le__", [](const bud::Bunch& a, const bud::Bunch& b) { return a.le(b); },
            "Element-wise less-than-or-equal comparison")
        .def(
            "__gt__", [](const bud::Bunch& a, const bud::Bunch& b) { return a.gt(b); },
            "Element-wise greater-than comparison")
        .def(
            "__ge__", [](const bud::Bunch& a, const bud::Bunch& b) { return a.ge(b); },
            "Element-wise greater-than-or-equal comparison")

        // Comparison with scalars
        .def("__eq__",
             [](const bud::Bunch& a, float b) {
                 auto b_bunch = bud::Bunch::fill(a.size(), b);
                 if (!b_bunch)
                     throw std::runtime_error("Failed to create scalar Bunch");
                 return a.eq(*b_bunch);
             })
        .def("__lt__",
             [](const bud::Bunch& a, float b) {
                 auto b_bunch = bud::Bunch::fill(a.size(), b);
                 if (!b_bunch)
                     throw std::runtime_error("Failed to create scalar Bunch");
                 return a.lt(*b_bunch);
             })
        .def("__le__",
             [](const bud::Bunch& a, float b) {
                 auto b_bunch = bud::Bunch::fill(a.size(), b);
                 if (!b_bunch)
                     throw std::runtime_error("Failed to create scalar Bunch");
                 return a.le(*b_bunch);
             })
        .def("__gt__",
             [](const bud::Bunch& a, float b) {
                 auto b_bunch = bud::Bunch::fill(a.size(), b);
                 if (!b_bunch)
                     throw std::runtime_error("Failed to create scalar Bunch");
                 return a.gt(*b_bunch);
             })
        .def("__ge__",
             [](const bud::Bunch& a, float b) {
                 auto b_bunch = bud::Bunch::fill(a.size(), b);
                 if (!b_bunch)
                     throw std::runtime_error("Failed to create scalar Bunch");
                 return a.ge(*b_bunch);
             })

        // Named comparison methods
        .def("eq", &bud::Bunch::eq, nb::arg("other"), "Element-wise equality")
        .def("lt", &bud::Bunch::lt, nb::arg("other"), "Element-wise less-than")
        .def("le", &bud::Bunch::le, nb::arg("other"), "Element-wise less-than-or-equal")
        .def("gt", &bud::Bunch::gt, nb::arg("other"), "Element-wise greater-than")
        .def("ge", &bud::Bunch::ge, nb::arg("other"), "Element-wise greater-than-or-equal")
        .def("ne", &bud::Bunch::ne, nb::arg("other"), "Element-wise not-equal")

        // =====================================================================
        // Masked operations
        // =====================================================================

        .def("where", &bud::Bunch::where, nb::arg("mask"), nb::arg("other"),
             "Select elements where mask is true, otherwise from other")

        // =====================================================================
        // Reductions
        // =====================================================================

        .def("sum", &bud::Bunch::sum, "Sum of all elements")
        .def("mean", &bud::Bunch::mean, "Mean of all elements")
        .def("min", &bud::Bunch::min, "Minimum element")
        .def("max", &bud::Bunch::max, "Maximum element")
        .def("dot", &bud::Bunch::dot, nb::arg("other"), "Dot product with another Bunch")

        // =====================================================================
        // Fused operations
        // =====================================================================

        .def(
            "fma",
            [](const bud::Bunch& a, const bud::Bunch& b, const bud::Bunch& c) {
                return bud::fma(a, b, c);
            },
            nb::arg("b"), nb::arg("c"), "Fused multiply-add: self * b + c")

        .def(
            "clamp",
            [](const bud::Bunch& self, float lo, float hi) { return bud::clamp(self, lo, hi); },
            nb::arg("lo"), nb::arg("hi"), "Clamp values to range [lo, hi]")

        .def(
            "lerp",
            [](const bud::Bunch& a, const bud::Bunch& b, float t) { return bud::lerp(a, b, t); },
            nb::arg("b"), nb::arg("t"), "Linear interpolation: self + t * (b - self)")

        // =====================================================================
        // Evaluation
        // =====================================================================

        .def(
            "eval",
            [](bud::Bunch& self) {
                auto result = self.eval();
                if (!result)
                    throw std::runtime_error("Evaluation failed: " + result.error().toString());
            },
            "Force evaluation of lazy operations (currently no-op in eager mode)")

        // =====================================================================
        // String representation
        // =====================================================================

        .def("__repr__", &bud::Bunch::toString)
        .def("__str__", &bud::Bunch::toString)

        // =====================================================================
        // Iteration support
        // =====================================================================
        // Note: For iteration, users should use to_numpy() and iterate over
        // the NumPy array. Direct iteration over Bunch is not supported in
        // nanobind 2.x due to API changes.

        // =====================================================================
        // Copy
        // =====================================================================

        .def(
            "copy",
            [](const bud::Bunch& self) {
                // Create a copy by getting data and creating new Bunch
                auto result =
                    bud::Bunch::fromData(static_cast<const float*>(self.data()), self.size());
                if (!result)
                    throw std::runtime_error("Failed to copy Bunch");
                return std::move(*result);
            },
            "Create a copy of this Bunch");

    // =========================================================================
    // Module-level free functions for math operations
    // =========================================================================

    m.def(
        "abs", [](const bud::Bunch& x) { return bud::abs(x); }, nb::arg("x"),
        "Element-wise absolute value");
    m.def(
        "sqrt", [](const bud::Bunch& x) { return bud::sqrt(x); }, nb::arg("x"),
        "Element-wise square root");
    m.def(
        "rsqrt", [](const bud::Bunch& x) { return bud::rsqrt(x); }, nb::arg("x"),
        "Element-wise reciprocal square root");
    m.def(
        "exp", [](const bud::Bunch& x) { return bud::exp(x); }, nb::arg("x"),
        "Element-wise exponential");
    m.def(
        "log", [](const bud::Bunch& x) { return bud::log(x); }, nb::arg("x"),
        "Element-wise natural logarithm");
    m.def(
        "sin", [](const bud::Bunch& x) { return bud::sin(x); }, nb::arg("x"), "Element-wise sine");
    m.def(
        "cos", [](const bud::Bunch& x) { return bud::cos(x); }, nb::arg("x"),
        "Element-wise cosine");
    m.def(
        "tanh", [](const bud::Bunch& x) { return bud::tanh(x); }, nb::arg("x"),
        "Element-wise hyperbolic tangent");
    m.def(
        "tan", [](const bud::Bunch& x) { return bud::tan(x); }, nb::arg("x"),
        "Element-wise tangent");
    m.def(
        "sigmoid", [](const bud::Bunch& x) { return bud::sigmoid(x); }, nb::arg("x"),
        "Element-wise sigmoid (1/(1+exp(-x)))");
    m.def(
        "floor", [](const bud::Bunch& x) { return bud::floor(x); }, nb::arg("x"),
        "Element-wise floor");
    m.def(
        "ceil", [](const bud::Bunch& x) { return bud::ceil(x); }, nb::arg("x"),
        "Element-wise ceiling");
    m.def(
        "round", [](const bud::Bunch& x) { return bud::round(x); }, nb::arg("x"),
        "Element-wise rounding");
    m.def(
        "trunc", [](const bud::Bunch& x) { return bud::trunc(x); }, nb::arg("x"),
        "Element-wise truncation toward zero");
    m.def(
        "isnan", [](const bud::Bunch& x) { return bud::isnan(x); }, nb::arg("x"),
        "Element-wise NaN check (returns mask)");
    m.def(
        "isinf", [](const bud::Bunch& x) { return bud::isinf(x); }, nb::arg("x"),
        "Element-wise infinity check (returns mask)");
    m.def(
        "isfinite", [](const bud::Bunch& x) { return bud::isfinite(x); }, nb::arg("x"),
        "Element-wise finite check (returns mask)");
    m.def(
        "minimum", [](const bud::Bunch& a, const bud::Bunch& b) { return bud::minimum(a, b); },
        nb::arg("a"), nb::arg("b"), "Element-wise minimum of two Bunches");
    m.def(
        "maximum", [](const bud::Bunch& a, const bud::Bunch& b) { return bud::maximum(a, b); },
        nb::arg("a"), nb::arg("b"), "Element-wise maximum of two Bunches");

    // Reductions as free functions
    m.def(
        "sum", [](const bud::Bunch& x) { return bud::sum(x); }, nb::arg("x"),
        "Sum of all elements");
    m.def("max", [](const bud::Bunch& x) { return bud::max(x); }, nb::arg("x"), "Maximum element");
    m.def("min", [](const bud::Bunch& x) { return bud::min(x); }, nb::arg("x"), "Minimum element");
    m.def(
        "mean", [](const bud::Bunch& x) { return bud::mean(x); }, nb::arg("x"),
        "Mean of all elements");
    m.def(
        "dot", [](const bud::Bunch& a, const bud::Bunch& b) { return bud::dot(a, b); },
        nb::arg("a"), nb::arg("b"), "Dot product of two Bunches");
}
