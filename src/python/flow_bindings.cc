// =============================================================================
// Bud Flow Lang - Flow Python Bindings
// =============================================================================
//
// Provides the main user-facing Python API:
// - flow() factory function for creating Bunch from Python data
// - @flow.kernel decorator for marking functions for JIT compilation
// - Factory functions: zeros(), ones(), arange(), linspace()
//

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/tiled_executor.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <optional>

#include "tracing_context.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Convert Python scalar type string to ScalarType enum
std::optional<bud::ScalarType> dtypeFromString(std::string_view dtype_str) {
    if (dtype_str == "float32" || dtype_str == "f32") {
        return bud::ScalarType::kFloat32;
    } else if (dtype_str == "float64" || dtype_str == "f64") {
        return bud::ScalarType::kFloat64;
    } else if (dtype_str == "int32" || dtype_str == "i32") {
        return bud::ScalarType::kInt32;
    } else if (dtype_str == "int64" || dtype_str == "i64") {
        return bud::ScalarType::kInt64;
    } else if (dtype_str == "int8" || dtype_str == "i8") {
        return bud::ScalarType::kInt8;
    } else if (dtype_str == "int16" || dtype_str == "i16") {
        return bud::ScalarType::kInt16;
    } else if (dtype_str == "uint8" || dtype_str == "u8") {
        return bud::ScalarType::kUint8;
    } else if (dtype_str == "uint16" || dtype_str == "u16") {
        return bud::ScalarType::kUint16;
    } else if (dtype_str == "uint32" || dtype_str == "u32") {
        return bud::ScalarType::kUint32;
    } else if (dtype_str == "uint64" || dtype_str == "u64") {
        return bud::ScalarType::kUint64;
    } else if (dtype_str == "float16" || dtype_str == "f16") {
        return bud::ScalarType::kFloat16;
    } else if (dtype_str == "bfloat16" || dtype_str == "bf16") {
        return bud::ScalarType::kBFloat16;
    }
    return std::nullopt;
}

// Convert ScalarType to Python dtype string
const char* dtypeToString(bud::ScalarType dtype) {
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
// Kernel Decorator Class
// =============================================================================

// The KernelOptions class stores options passed to @flow.kernel decorator
struct KernelOptions {
    std::string target_isa;          // Target ISA override (e.g., "avx2", "avx512")
    int opt_level = 2;               // Optimization level (0-3)
    bool enable_fusion = true;       // Enable operation fusion
    bool enable_prefetch = true;     // Enable prefetching
    int unroll_factor = 0;           // Loop unroll factor (0 = auto)
    bool force_vectorize = false;    // Force vectorization
    bool disable_vectorize = false;  // Disable vectorization
};

// KernelDecorator wraps a Python function with JIT compilation metadata
class KernelDecorator {
  public:
    KernelDecorator(nb::object func, KernelOptions options)
        : func_(std::move(func)),
          options_(std::move(options)),
          cache_(std::make_shared<bud::KernelCache>()) {}

    // Call operator - invokes the wrapped function with tracing
    nb::object operator()(nb::args args, nb::kwargs kwargs) const {
        // Build signature from Bunch arguments
        bud::KernelSignature sig;
        std::vector<bud::Bunch> bunch_copies;  // Keep copies to avoid dangling pointers

        // First pass: collect all Bunch inputs and reserve space
        for (size_t i = 0; i < args.size(); ++i) {
            if (nb::isinstance<bud::Bunch>(args[i])) {
                bunch_copies.push_back(nb::cast<bud::Bunch>(args[i]));
            }
        }

        // Build signature and input pointers (now safe since vector won't reallocate)
        std::vector<const bud::Bunch*> bunch_inputs;
        bunch_inputs.reserve(bunch_copies.size());
        for (const auto& b : bunch_copies) {
            sig.input_dtypes.push_back(b.dtype());
            sig.input_sizes.push_back(b.size());
            bunch_inputs.push_back(&b);
        }

        // Check cache for compiled kernel
        if (auto* cached = cache_->find(sig)) {
            spdlog::debug("KernelDecorator: cache hit, executing compiled kernel");
            auto result = cached->execute(bunch_inputs);
            if (!result) {
                throw std::runtime_error(std::string("Kernel execution failed: ") +
                                         std::string(result.error().message()));
            }
            return nb::cast(std::move(*result));
        }

        // Not cached - trace the function
        spdlog::debug("KernelDecorator: cache miss, tracing function");

        // Create tracing context
        bud::TracingContext ctx;

        // Create tracer inputs
        nb::list tracer_args;
        size_t bunch_idx = 0;
        for (size_t i = 0; i < args.size(); ++i) {
            if (nb::isinstance<bud::Bunch>(args[i])) {
                bud::Bunch b = nb::cast<bud::Bunch>(args[i]);
                bud::Bunch tracer = ctx.createTracerInput(bunch_idx++, b.dtype(), b.shape());
                tracer_args.append(nb::cast(tracer));
            } else {
                tracer_args.append(args[i]);  // Pass non-Bunch args through
            }
        }

        // Execute function with tracer inputs
        nb::object result;
        try {
            result = func_(*nb::tuple(tracer_args), **kwargs);
        } catch (const std::exception& e) {
            spdlog::error("KernelDecorator: tracing failed: {}", e.what());
            // Fall back to direct execution
            return func_(*args, **kwargs);
        }

        // Check if result is a Bunch
        if (!nb::isinstance<bud::Bunch>(result)) {
            spdlog::debug("KernelDecorator: result is not a Bunch, returning directly");
            return result;
        }

        const bud::Bunch& result_bunch = nb::cast<const bud::Bunch&>(result);

        // Extract output ValueId from traced result
        bud::ir::ValueId output_id = ctx.extractOutput(result_bunch);
        if (!output_id.isValid()) {
            spdlog::warn(
                "KernelDecorator: could not extract IR from result, falling back to eager");
            return func_(*args, **kwargs);
        }

        // Run optimization pipeline
        if (options_.enable_fusion) {
            auto opt_result = bud::runOptimizationPipeline(ctx.module(), options_.opt_level);
            if (!opt_result) {
                spdlog::warn("KernelDecorator: optimization failed, using unoptimized IR");
            }
        }

        // Create compiled kernel
        auto compiled = std::make_unique<bud::CompiledKernel>(
            std::make_unique<bud::ir::IRModule>(std::move(ctx.module())), output_id,
            std::vector<bud::ir::ValueId>(ctx.inputIds()));

        // Execute the compiled kernel
        auto exec_result = compiled->execute(bunch_inputs);
        if (!exec_result) {
            spdlog::error("KernelDecorator: compiled kernel execution failed: {}",
                          exec_result.error().message());
            // Fall back to direct execution
            return func_(*args, **kwargs);
        }

        // Cache the kernel for future calls
        cache_->insert(sig, std::move(compiled));

        return nb::cast(std::move(*exec_result));
    }

    // Get the wrapped function
    nb::object func() const { return func_; }

    // Get options
    const KernelOptions& options() const { return options_; }

    // Clear the kernel cache
    void clearCache() { cache_->clear(); }

    // Get cache size
    size_t cacheSize() const { return cache_->size(); }

    // Repr for debugging
    std::string repr() const {
        std::string name = "<unknown>";
        if (nb::hasattr(func_, "__name__")) {
            name = nb::cast<std::string>(func_.attr("__name__"));
        }
        return "<flow.kernel '" + name + "' opt=" + std::to_string(options_.opt_level) +
               " fusion=" + (options_.enable_fusion ? "on" : "off") +
               " cached=" + std::to_string(cache_->size()) +
               (options_.target_isa.empty() ? "" : " target=" + options_.target_isa) + ">";
    }

  private:
    nb::object func_;
    KernelOptions options_;
    std::shared_ptr<bud::KernelCache> cache_;
};

// =============================================================================
// Flow Module Bindings
// =============================================================================

void bind_flow(nb::module_& m) {
    // =========================================================================
    // KernelOptions class
    // =========================================================================

    nb::class_<KernelOptions>(
        m, "KernelOptions",
        "Options for @flow.kernel decorator controlling JIT compilation behavior")
        .def(nb::init<>())
        .def_rw("target_isa", &KernelOptions::target_isa,
                "Target ISA override (e.g., 'avx2', 'avx512', 'neon')")
        .def_rw("opt_level", &KernelOptions::opt_level, "Optimization level (0-3)")
        .def_rw("enable_fusion", &KernelOptions::enable_fusion, "Enable operation fusion")
        .def_rw("enable_prefetch", &KernelOptions::enable_prefetch, "Enable memory prefetching")
        .def_rw("unroll_factor", &KernelOptions::unroll_factor,
                "Loop unroll factor (0 = auto, N = unroll N times)")
        .def_rw("force_vectorize", &KernelOptions::force_vectorize,
                "Force vectorization even if deemed unprofitable")
        .def_rw("disable_vectorize", &KernelOptions::disable_vectorize, "Disable vectorization")
        .def("__repr__", [](const KernelOptions& o) {
            return "<KernelOptions opt=" + std::to_string(o.opt_level) +
                   " fusion=" + (o.enable_fusion ? "on" : "off") +
                   " prefetch=" + (o.enable_prefetch ? "on" : "off") +
                   (o.target_isa.empty() ? "" : " isa=" + o.target_isa) + ">";
        });

    // =========================================================================
    // KernelDecorator class
    // =========================================================================

    nb::class_<KernelDecorator>(m, "KernelDecorator",
                                "A decorated function marked for JIT compilation with tracing")
        .def(
            "__call__",
            [](const KernelDecorator& self, nb::args args, nb::kwargs kwargs) {
                return self(args, kwargs);
            },
            "Invoke the decorated kernel function (traces on first call, uses cache afterward)")
        .def_prop_ro("__wrapped__", &KernelDecorator::func, "The wrapped function")
        .def_prop_ro("options", &KernelDecorator::options, "Kernel compilation options")
        .def("clear_cache", &KernelDecorator::clearCache, "Clear the compiled kernel cache")
        .def_prop_ro("cache_size", &KernelDecorator::cacheSize,
                     "Number of compiled kernels in the cache")
        .def("__repr__", &KernelDecorator::repr);

    // =========================================================================
    // @flow.kernel decorator
    // =========================================================================

    // Usage: @flow.kernel or @flow.kernel(opt_level=3, target_isa='avx2')
    m.def(
        "kernel",
        [](nb::object func_or_none, std::optional<std::string> target_isa, int opt_level,
           bool enable_fusion, bool enable_prefetch, int unroll_factor, bool force_vectorize,
           bool disable_vectorize) -> nb::object {
            KernelOptions opts;
            opts.target_isa = target_isa.value_or("");
            opts.opt_level = opt_level;
            opts.enable_fusion = enable_fusion;
            opts.enable_prefetch = enable_prefetch;
            opts.unroll_factor = unroll_factor;
            opts.force_vectorize = force_vectorize;
            opts.disable_vectorize = disable_vectorize;

            // If called with a function directly: @flow.kernel
            if (!func_or_none.is_none() && PyCallable_Check(func_or_none.ptr())) {
                return nb::cast(KernelDecorator(func_or_none, opts));
            }

            // If called with options: @flow.kernel(opt_level=3)
            // Return a lambda that will receive the function
            return nb::cpp_function(
                [opts](nb::object func) { return KernelDecorator(func, opts); });
        },
        nb::arg("func") = nb::none(), nb::arg("target_isa") = nb::none(), nb::arg("opt_level") = 2,
        nb::arg("enable_fusion") = true, nb::arg("enable_prefetch") = true,
        nb::arg("unroll_factor") = 0, nb::arg("force_vectorize") = false,
        nb::arg("disable_vectorize") = false,
        R"doc(
        Mark a function for JIT compilation with SIMD optimization.

        Can be used as a simple decorator:
            @flow.kernel
            def my_func(a, b):
                return a + b * 2

        Or with options:
            @flow.kernel(opt_level=3, target_isa='avx2')
            def my_func(a, b):
                return a + b * 2

        Parameters
        ----------
        func : callable, optional
            The function to decorate (when used without parentheses)
        target_isa : str, optional
            Target ISA override (e.g., 'avx2', 'avx512', 'neon')
        opt_level : int
            Optimization level 0-3 (default: 2)
        enable_fusion : bool
            Enable operation fusion (default: True)
        enable_prefetch : bool
            Enable memory prefetching (default: True)
        unroll_factor : int
            Loop unroll factor, 0 for auto (default: 0)
        force_vectorize : bool
            Force vectorization (default: False)
        disable_vectorize : bool
            Disable vectorization (default: False)

        Returns
        -------
        KernelDecorator
            A decorated callable that will be JIT compiled on first execution
        )doc");

    // =========================================================================
    // flow() factory function
    // =========================================================================

    // Create Bunch from Python list of floats
    m.def(
        "flow",
        [](const std::vector<float>& data) {
            auto result = bud::Bunch::fromData(data.data(), data.size());
            if (!result) {
                throw std::runtime_error("Failed to create Bunch: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("data"),
        R"doc(
        Create a Bunch from a Python list of floats.

        Parameters
        ----------
        data : list[float]
            List of float values

        Returns
        -------
        Bunch
            A SIMD-optimized array

        Examples
        --------
        >>> x = flow.flow([1.0, 2.0, 3.0, 4.0])
        >>> print(x)
        Bunch[4]: [1.0, 2.0, 3.0, 4.0]
        )doc");

    // Create Bunch from NumPy array (float32)
    m.def(
        "flow",
        [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig> arr) {
            auto result = bud::Bunch::fromData(arr.data(), arr.shape(0));
            if (!result) {
                throw std::runtime_error("Failed to create Bunch from NumPy array: " +
                                         result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("array").noconvert(),
        R"doc(
        Create a Bunch from a 1D NumPy array (float32).

        Parameters
        ----------
        array : numpy.ndarray
            1D NumPy array of float32 values

        Returns
        -------
        Bunch
            A SIMD-optimized array

        Examples
        --------
        >>> import numpy as np
        >>> x = flow.flow(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        )doc");

    // Create Bunch from NumPy array (float64)
    m.def(
        "flow",
        [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> arr) {
            auto result = bud::Bunch::fromData(arr.data(), arr.shape(0));
            if (!result) {
                throw std::runtime_error("Failed to create Bunch from NumPy array: " +
                                         result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("array").noconvert(),
        R"doc(
        Create a Bunch from a 1D NumPy array (float64).

        Parameters
        ----------
        array : numpy.ndarray
            1D NumPy array of float64 values

        Returns
        -------
        Bunch
            A SIMD-optimized array
        )doc");

    // Create Bunch from NumPy array (int32)
    m.def(
        "flow",
        [](nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> arr) {
            auto result = bud::Bunch::fromData(arr.data(), arr.shape(0));
            if (!result) {
                throw std::runtime_error("Failed to create Bunch from NumPy array: " +
                                         result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("array").noconvert(),
        R"doc(
        Create a Bunch from a 1D NumPy array (int32).

        Parameters
        ----------
        array : numpy.ndarray
            1D NumPy array of int32 values

        Returns
        -------
        Bunch
            A SIMD-optimized array
        )doc");

    // =========================================================================
    // Factory functions
    // =========================================================================

    m.def(
        "zeros",
        [](size_t count, std::optional<std::string> dtype) {
            bud::ScalarType stype = bud::ScalarType::kFloat32;
            if (dtype.has_value()) {
                auto parsed = dtypeFromString(*dtype);
                if (!parsed) {
                    throw std::runtime_error("Unknown dtype: " + *dtype);
                }
                stype = *parsed;
            }
            auto result = bud::Bunch::zeros(count, stype);
            if (!result) {
                throw std::runtime_error("Failed to create zeros: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("count"), nb::arg("dtype") = nb::none(),
        R"doc(
        Create a Bunch filled with zeros.

        Parameters
        ----------
        count : int
            Number of elements
        dtype : str, optional
            Data type: 'float32', 'float64', 'int32', etc. (default: 'float32')

        Returns
        -------
        Bunch
            A SIMD-optimized array of zeros

        Examples
        --------
        >>> x = flow.zeros(1000)
        >>> y = flow.zeros(1000, dtype='float64')
        )doc");

    m.def(
        "ones",
        [](size_t count, std::optional<std::string> dtype) {
            bud::ScalarType stype = bud::ScalarType::kFloat32;
            if (dtype.has_value()) {
                auto parsed = dtypeFromString(*dtype);
                if (!parsed) {
                    throw std::runtime_error("Unknown dtype: " + *dtype);
                }
                stype = *parsed;
            }
            auto result = bud::Bunch::ones(count, stype);
            if (!result) {
                throw std::runtime_error("Failed to create ones: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("count"), nb::arg("dtype") = nb::none(),
        R"doc(
        Create a Bunch filled with ones.

        Parameters
        ----------
        count : int
            Number of elements
        dtype : str, optional
            Data type: 'float32', 'float64', 'int32', etc. (default: 'float32')

        Returns
        -------
        Bunch
            A SIMD-optimized array of ones

        Examples
        --------
        >>> x = flow.ones(1000)
        >>> y = flow.ones(1000, dtype='float64')
        )doc");

    m.def(
        "full",
        [](size_t count, float value) {
            auto result = bud::Bunch::fill(count, value);
            if (!result) {
                throw std::runtime_error("Failed to create full: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("count"), nb::arg("value"),
        R"doc(
        Create a Bunch filled with a constant value.

        Parameters
        ----------
        count : int
            Number of elements
        value : float
            Fill value

        Returns
        -------
        Bunch
            A SIMD-optimized array filled with the value

        Examples
        --------
        >>> x = flow.full(1000, 3.14159)
        )doc");

    m.def(
        "arange",
        [](size_t count, float start, float step) {
            auto result = bud::Bunch::arange(count, start, step);
            if (!result) {
                throw std::runtime_error("Failed to create arange: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("count"), nb::arg("start") = 0.0f, nb::arg("step") = 1.0f,
        R"doc(
        Create a Bunch with evenly spaced values.

        Values are: [start, start+step, start+2*step, ...]

        Parameters
        ----------
        count : int
            Number of elements
        start : float
            Starting value (default: 0.0)
        step : float
            Step between values (default: 1.0)

        Returns
        -------
        Bunch
            A SIMD-optimized array with evenly spaced values

        Examples
        --------
        >>> x = flow.arange(10)           # [0, 1, 2, ..., 9]
        >>> y = flow.arange(10, 1.0, 0.5) # [1.0, 1.5, 2.0, ..., 5.5]
        )doc");

    m.def(
        "linspace",
        [](float start, float stop, size_t count, bool endpoint) {
            if (count == 0) {
                throw std::runtime_error("count must be positive");
            }
            float step;
            if (count == 1) {
                step = 0.0f;
            } else if (endpoint) {
                step = (stop - start) / static_cast<float>(count - 1);
            } else {
                step = (stop - start) / static_cast<float>(count);
            }
            auto result = bud::Bunch::arange(count, start, step);
            if (!result) {
                throw std::runtime_error("Failed to create linspace: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("start"), nb::arg("stop"), nb::arg("count"), nb::arg("endpoint") = true,
        R"doc(
        Create a Bunch with evenly spaced values over an interval.

        Parameters
        ----------
        start : float
            Start of interval
        stop : float
            End of interval
        count : int
            Number of samples
        endpoint : bool
            If True, stop is the last sample; if False, stop is not included (default: True)

        Returns
        -------
        Bunch
            A SIMD-optimized array with evenly spaced values

        Examples
        --------
        >>> x = flow.linspace(0.0, 1.0, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> y = flow.linspace(0.0, 1.0, 5, endpoint=False)  # [0.0, 0.2, 0.4, 0.6, 0.8]
        )doc");

    // =========================================================================
    // Math operations as module-level functions
    // =========================================================================

    m.def(
        "fma",
        [](const bud::Bunch& a, const bud::Bunch& b, const bud::Bunch& c) {
            return bud::fma(a, b, c);
        },
        nb::arg("a"), nb::arg("b"), nb::arg("c"),
        R"doc(
        Fused multiply-add: a * b + c

        This is computed in a single SIMD operation for better accuracy and performance.

        Parameters
        ----------
        a, b, c : Bunch
            Input arrays (must have same size)

        Returns
        -------
        Bunch
            Result of a * b + c

        Examples
        --------
        >>> x = flow.arange(4)
        >>> y = flow.ones(4)
        >>> z = flow.full(4, 2.0)
        >>> result = flow.fma(x, y, z)  # [2, 3, 4, 5]
        )doc");

    m.def(
        "clamp", [](const bud::Bunch& x, float lo, float hi) { return bud::clamp(x, lo, hi); },
        nb::arg("x"), nb::arg("lo"), nb::arg("hi"),
        R"doc(
        Clamp values to a range [lo, hi].

        Parameters
        ----------
        x : Bunch
            Input array
        lo : float
            Lower bound
        hi : float
            Upper bound

        Returns
        -------
        Bunch
            Clamped values

        Examples
        --------
        >>> x = flow.arange(10)
        >>> y = flow.clamp(x, 2.0, 7.0)  # [2, 2, 2, 3, 4, 5, 6, 7, 7, 7]
        )doc");

    m.def(
        "lerp",
        [](const bud::Bunch& a, const bud::Bunch& b, float t) { return bud::lerp(a, b, t); },
        nb::arg("a"), nb::arg("b"), nb::arg("t"),
        R"doc(
        Linear interpolation: a + t * (b - a)

        Parameters
        ----------
        a : Bunch
            Start values
        b : Bunch
            End values
        t : float
            Interpolation factor (0.0 = a, 1.0 = b)

        Returns
        -------
        Bunch
            Interpolated values

        Examples
        --------
        >>> a = flow.zeros(4)
        >>> b = flow.ones(4)
        >>> y = flow.lerp(a, b, 0.5)  # [0.5, 0.5, 0.5, 0.5]
        )doc");

    // =========================================================================
    // Hardware info utilities
    // =========================================================================

    m.def(
        "get_hardware_info",
        []() {
            const auto& info = bud::getHardwareInfo();
            nb::dict d;
            d["cpu_name"] = info.cpu_name;
            d["vendor"] = info.vendor;
            d["arch_family"] = std::string(bud::archFamilyName(info.arch_family));
            d["is_64bit"] = info.is_64bit;
            d["simd_width"] = info.simd_width;
            d["physical_cores"] = info.physical_cores;
            d["logical_cores"] = info.logical_cores;

            // SIMD capabilities
            d["has_sse2"] = info.has_sse2;
            d["has_avx"] = info.has_avx;
            d["has_avx2"] = info.has_avx2;
            d["has_avx512"] = info.has_avx512;
            d["has_neon"] = info.has_neon;
            d["has_sve"] = info.has_sve;
            d["has_rvv"] = info.has_rvv;

            // Derived capabilities
            d["supports_float16"] = info.supportsFloat16();
            d["supports_bfloat16"] = info.supportsBFloat16();
            d["supports_scalable_vectors"] = info.supportsScalableVectors();

            // Cache info
            d["l1_cache_size"] = info.l1_cache_size;
            d["l2_cache_size"] = info.l2_cache_size;
            d["l3_cache_size"] = info.l3_cache_size;

            return d;
        },
        R"doc(
    Get detailed hardware information.

    Returns
    -------
    dict
        Dictionary with CPU and SIMD capabilities:
        - cpu_name: CPU model name
        - vendor: CPU vendor
        - arch_family: Architecture family (x86, ARM, RISC-V, etc.)
        - is_64bit: True if 64-bit architecture
        - simd_width: Best available SIMD width in bytes
        - has_avx2, has_avx512, has_neon, etc.: SIMD capability flags
        - supports_float16, supports_bfloat16: Half-precision support
        - l1_cache_size, l2_cache_size, l3_cache_size: Cache sizes

    Examples
    --------
    >>> info = flow.get_hardware_info()
    >>> print(f"SIMD width: {info['simd_width']} bytes")
    >>> print(f"AVX2: {info['has_avx2']}")
    )doc");

    m.def(
        "get_simd_capabilities",
        []() {
            const auto& info = bud::getHardwareInfo();
            return info.simdCapabilitySummary();
        },
        R"doc(
    Get a human-readable summary of SIMD capabilities.

    Returns
    -------
    str
        Summary string of available SIMD instruction sets
    )doc");

    // =========================================================================
    // Utility functions
    // =========================================================================

    m.def(
        "dtype_to_string",
        [](bud::ScalarType dtype) { return std::string(bud::scalarTypeName(dtype)); },
        nb::arg("dtype"), "Convert a ScalarType enum to its string representation");

    m.def(
        "string_to_dtype",
        [](const std::string& s) -> bud::ScalarType {
            auto result = dtypeFromString(s);
            if (!result) {
                throw std::runtime_error("Unknown dtype: " + s);
            }
            return *result;
        },
        nb::arg("dtype_str"), "Convert a string to ScalarType enum");

    // =========================================================================
    // Memory Optimization API
    // =========================================================================

    m.def(
        "detect_cache_config",
        []() {
            bud::memory::CacheConfig config = bud::memory::CacheConfig::detect();
            nb::dict d;
            d["l1_size"] = config.l1Size();
            d["l2_size"] = config.l2Size();
            d["l3_size"] = config.l3Size();
            d["line_size"] = config.lineSize();
            d["l1_size_kb"] = config.l1Size() / 1024;
            d["l2_size_kb"] = config.l2Size() / 1024;
            d["l3_size_kb"] = config.l3Size() / 1024;
            return d;
        },
        R"doc(
    Detect CPU cache configuration.

    Returns
    -------
    dict
        Dictionary with cache information:
        - l1_size: L1 cache size in bytes
        - l2_size: L2 cache size in bytes
        - l3_size: L3 cache size in bytes
        - line_size: Cache line size in bytes
        - l1_size_kb, l2_size_kb, l3_size_kb: Cache sizes in KB

    Examples
    --------
    >>> cache = flow.detect_cache_config()
    >>> print(f"L1: {cache['l1_size_kb']} KB")
    >>> print(f"L2: {cache['l2_size_kb']} KB")
    >>> print(f"Line size: {cache['line_size']} bytes")
    )doc");

    m.def(
        "optimal_tile_size",
        [](size_t element_size, size_t num_arrays) {
            bud::memory::CacheConfig config = bud::memory::CacheConfig::detect();
            return config.optimalTileSize(element_size, num_arrays);
        },
        nb::arg("element_size") = 4, nb::arg("num_arrays") = 2,
        R"doc(
    Calculate optimal tile size for cache efficiency.

    Parameters
    ----------
    element_size : int
        Size of each element in bytes (default: 4 for float32)
    num_arrays : int
        Number of arrays accessed together (default: 2)

    Returns
    -------
    int
        Optimal number of elements per tile

    Examples
    --------
    >>> tile_size = flow.optimal_tile_size(4, 3)  # For 3 float32 arrays
    >>> print(f"Process {tile_size} elements at a time")
    )doc");

    m.def(
        "set_tiling_enabled", [](bool enabled) { bud::setTilingEnabled(enabled); },
        nb::arg("enabled"),
        R"doc(
    Enable or disable cache-aware tiled execution.

    Tiling improves cache efficiency for large arrays by processing
    data in cache-sized chunks.

    Parameters
    ----------
    enabled : bool
        True to enable tiling, False to disable

    Examples
    --------
    >>> flow.set_tiling_enabled(True)   # Enable tiled execution
    >>> flow.set_tiling_enabled(False)  # Disable for debugging
    )doc");

    m.def(
        "is_tiling_enabled", []() { return bud::isTilingEnabled(); },
        R"doc(
    Check if cache-aware tiled execution is enabled.

    Returns
    -------
    bool
        True if tiling is enabled
    )doc");

    m.def(
        "set_prefetch_enabled", [](bool enabled) { bud::setPrefetchEnabled(enabled); },
        nb::arg("enabled"),
        R"doc(
    Enable or disable software prefetching.

    Prefetching loads data into cache before it's needed, reducing
    memory latency for large arrays.

    Parameters
    ----------
    enabled : bool
        True to enable prefetching, False to disable

    Examples
    --------
    >>> flow.set_prefetch_enabled(True)   # Enable prefetching
    >>> flow.set_prefetch_enabled(False)  # Disable for benchmarking
    )doc");

    m.def(
        "is_prefetch_enabled", []() { return bud::isPrefetchEnabled(); },
        R"doc(
    Check if software prefetching is enabled.

    Returns
    -------
    bool
        True if prefetching is enabled
    )doc");

    m.def(
        "get_memory_optimization_status",
        []() {
            nb::dict d;
            d["tiling_enabled"] = bud::isTilingEnabled();
            d["prefetch_enabled"] = bud::isPrefetchEnabled();

            // Get cache config
            auto* cache_config = bud::getCacheConfig();
            if (cache_config) {
                d["l1_size_kb"] = cache_config->l1Size() / 1024;
                d["l2_size_kb"] = cache_config->l2Size() / 1024;
                d["l3_size_kb"] = cache_config->l3Size() / 1024;
                d["line_size"] = cache_config->lineSize();
                d["optimal_tile_size_float32"] = cache_config->optimalTileSize(sizeof(float), 2);
            }

            return d;
        },
        R"doc(
    Get current memory optimization status.

    Returns
    -------
    dict
        Dictionary with:
        - tiling_enabled: Whether tiled execution is enabled
        - prefetch_enabled: Whether software prefetching is enabled
        - l1_size_kb, l2_size_kb, l3_size_kb: Cache sizes
        - line_size: Cache line size
        - optimal_tile_size_float32: Optimal tile size for float32

    Examples
    --------
    >>> status = flow.get_memory_optimization_status()
    >>> print(f"Tiling: {status['tiling_enabled']}")
    >>> print(f"Tile size: {status['optimal_tile_size_float32']}")
    )doc");

    // =========================================================================
    // JIT Control and PGO Query APIs (Developer Level)
    // =========================================================================

    m.def(
        "get_jit_stats",
        []() {
            nb::dict d;
            auto stats = bud::getCompilationStats();
            d["total_compilations"] = stats.total_compilations;
            d["cache_hits"] = stats.cache_hits;
            d["code_cache_bytes"] = stats.code_cache_bytes;
            d["avg_compile_time_ms"] = stats.avg_compile_time_ms;
            d["total_compile_time_ms"] = stats.total_compile_time_ms;
            return d;
        },
        R"doc(
    Get JIT compilation statistics.

    Returns
    -------
    dict
        Dictionary with:
        - total_compilations: Number of kernels compiled
        - cache_hits: Number of cached kernel lookups
        - code_cache_bytes: Bytes used by compiled code
        - avg_compile_time_ms: Average compilation time
        - total_compile_time_ms: Total compilation time

    Examples
    --------
    >>> stats = flow.get_jit_stats()
    >>> print(f"Compiled: {stats['total_compilations']} kernels")
    )doc");

    m.def(
        "get_tiered_stats",
        []() {
            nb::dict d;
            auto stats = bud::getTieredStats();
            d["total_entries"] = stats.total_entries;
            d["tier0_entries"] = stats.tier0_entries;  // Interpreter
            d["tier1_entries"] = stats.tier1_entries;  // Copy-patch JIT
            d["tier2_entries"] = stats.tier2_entries;  // Fused kernels

            // Calculate percentages
            if (stats.total_entries > 0) {
                d["tier0_pct"] = 100.0 * stats.tier0_entries / stats.total_entries;
                d["tier1_pct"] = 100.0 * stats.tier1_entries / stats.total_entries;
                d["tier2_pct"] = 100.0 * stats.tier2_entries / stats.total_entries;
            } else {
                d["tier0_pct"] = 0.0;
                d["tier1_pct"] = 0.0;
                d["tier2_pct"] = 0.0;
            }
            return d;
        },
        R"doc(
    Get tiered execution statistics.

    The JIT uses 3 execution tiers:
    - Tier 0: Interpreter (direct Highway dispatch)
    - Tier 1: Copy-and-patch JIT (compiled stencils)
    - Tier 2: Fused kernels (optimized operation chains)

    Returns
    -------
    dict
        Dictionary with tier entry counts and percentages

    Examples
    --------
    >>> stats = flow.get_tiered_stats()
    >>> print(f"Tier 0: {stats['tier0_pct']:.1f}%")
    >>> print(f"Tier 1: {stats['tier1_pct']:.1f}%")
    >>> print(f"Tier 2: {stats['tier2_pct']:.1f}%")
    )doc");

    m.def(
        "reset_jit_stats", []() { bud::resetCompilationStats(); },
        R"doc(
    Reset JIT compilation statistics and call counters.

    This clears all cached kernels and resets tier promotion tracking.
    Useful for benchmarking or debugging.

    Examples
    --------
    >>> flow.reset_jit_stats()
    >>> # Run your code
    >>> stats = flow.get_tiered_stats()
    )doc");

    m.def(
        "get_tier_thresholds",
        []() {
            nb::dict d;
            // These are the current thresholds (read-only for now)
            d["tier0_to_tier1"] = 10;   // kTier1Threshold
            d["tier1_to_tier2"] = 100;  // kTier2Threshold
            d["description"] = "Call count thresholds for tier promotion";
            return d;
        },
        R"doc(
    Get the tier promotion thresholds.

    Returns the call count thresholds used to promote operations
    from one tier to the next.

    Returns
    -------
    dict
        Dictionary with:
        - tier0_to_tier1: Calls before promoting to JIT
        - tier1_to_tier2: Calls before promoting to fused kernels

    Examples
    --------
    >>> thresholds = flow.get_tier_thresholds()
    >>> print(f"JIT after {thresholds['tier0_to_tier1']} calls")
    )doc");
}
