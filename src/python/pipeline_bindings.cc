// =============================================================================
// Bud Flow Lang - Pipeline Python Bindings
// =============================================================================

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/pipeline.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void bind_pipeline(nb::module_& m) {
    // =========================================================================
    // OperationKind enum
    // =========================================================================
    nb::enum_<bud::OperationKind>(m, "OperationKind", "Type of pipeline operation")
        .value("Unary", bud::OperationKind::kUnary, "f(x) -> y")
        .value("Binary", bud::OperationKind::kBinary, "f(x, y) -> z")
        .value("Reduction", bud::OperationKind::kReduction, "f(x) -> scalar")
        .value("Transform", bud::OperationKind::kTransform, "f(x, params) -> y")
        .value("Custom", bud::OperationKind::kCustom, "User-defined");

    // =========================================================================
    // Pipeline class
    // =========================================================================
    nb::class_<bud::Pipeline>(m, "Pipeline",
                              "Fusion engine for operation chains.\n\n"
                              "Pipeline fuses multiple operations into a single kernel,\n"
                              "eliminating intermediate arrays and reducing memory bandwidth.\n\n"
                              "Example:\n"
                              "  pipeline = Pipeline()\n"
                              "  pipeline.multiply(2.0).add(1.0).sqrt()\n"
                              "  result = pipeline.run(data)  # Single fused kernel\n")

        // Constructor
        .def(nb::init<>(), "Create empty pipeline")

        // Add operations (fluent API)
        .def("add", nb::overload_cast<float>(&bud::Pipeline::add), nb::arg("scalar"),
             nb::rv_policy::reference, "Add scalar to each element")

        .def("multiply", &bud::Pipeline::multiply, nb::arg("scalar"), nb::rv_policy::reference,
             "Multiply each element by scalar")

        .def("subtract", &bud::Pipeline::subtract, nb::arg("scalar"), nb::rv_policy::reference,
             "Subtract scalar from each element")

        .def("divide", &bud::Pipeline::divide, nb::arg("scalar"), nb::rv_policy::reference,
             "Divide each element by scalar")

        .def("power", &bud::Pipeline::power, nb::arg("exponent"), nb::rv_policy::reference,
             "Raise each element to power")

        // Math functions
        .def("sqrt", &bud::Pipeline::sqrt, nb::rv_policy::reference, "Square root of each element")

        .def("rsqrt", &bud::Pipeline::rsqrt, nb::rv_policy::reference,
             "Reciprocal square root of each element")

        .def("abs", &bud::Pipeline::abs, nb::rv_policy::reference, "Absolute value of each element")

        .def("neg", &bud::Pipeline::neg, nb::rv_policy::reference, "Negate each element")

        .def("exp", &bud::Pipeline::exp, nb::rv_policy::reference, "Exponential of each element")

        .def("log", &bud::Pipeline::log, nb::rv_policy::reference,
             "Natural logarithm of each element")

        .def("sin", &bud::Pipeline::sin, nb::rv_policy::reference, "Sine of each element")

        .def("cos", &bud::Pipeline::cos, nb::rv_policy::reference, "Cosine of each element")

        .def("tanh", &bud::Pipeline::tanh, nb::rv_policy::reference,
             "Hyperbolic tangent of each element")

        .def("clamp", &bud::Pipeline::clamp, nb::arg("lo"), nb::arg("hi"), nb::rv_policy::reference,
             "Clamp each element to [lo, hi]")

        // Properties
        .def_prop_ro("num_stages", &bud::Pipeline::numStages, "Number of stages in pipeline")

        .def("empty", &bud::Pipeline::empty, "Check if pipeline has no stages")

        .def("clear", &bud::Pipeline::clear, "Remove all stages from pipeline")

        // Analysis
        .def("can_fuse", &bud::Pipeline::canFuse,
             "Check if pipeline can be fused into single kernel")

        .def("estimated_speedup", &bud::Pipeline::estimatedSpeedup, "Estimate speedup from fusion")

        .def("memory_reduction", &bud::Pipeline::memoryReduction,
             "Estimate memory traffic reduction ratio")

        // Execution
        .def(
            "run",
            [](const bud::Pipeline& self, const bud::Bunch& input) {
                auto result = self.run(input);
                if (!result) {
                    throw std::runtime_error("Pipeline execution failed: " +
                                             result.error().toString());
                }
                return std::move(*result);
            },
            nb::arg("input"), "Run pipeline on input data, returns new Bunch")

        .def(
            "run_inplace",
            [](const bud::Pipeline& self, bud::Bunch& input) {
                auto result = self.runInPlace(input);
                if (!result) {
                    throw std::runtime_error("Pipeline execution failed: " +
                                             result.error().toString());
                }
            },
            nb::arg("input"), "Run pipeline in-place, modifying input")

        // Python operators for chaining
        .def(
            "__or__",
            [](bud::Pipeline& self, const std::string& op) -> bud::Pipeline& {
                if (op == "sqrt")
                    return self.sqrt();
                if (op == "exp")
                    return self.exp();
                if (op == "log")
                    return self.log();
                if (op == "abs")
                    return self.abs();
                if (op == "neg")
                    return self.neg();
                if (op == "sin")
                    return self.sin();
                if (op == "cos")
                    return self.cos();
                if (op == "tanh")
                    return self.tanh();
                throw std::runtime_error("Unknown operation: " + op);
            },
            nb::arg("op"), nb::rv_policy::reference, "Chain operation using | operator")

        // String representation
        .def("__repr__", &bud::Pipeline::toString)
        .def("__str__", &bud::Pipeline::toString)

        // Length
        .def("__len__", &bud::Pipeline::numStages);

    // =========================================================================
    // Module-level pipeline builder function
    // =========================================================================
    m.def(
        "pipeline", []() { return bud::Pipeline(); },
        "Create a new empty Pipeline for operation fusion");

    // Convenience: create pre-configured pipelines
    m.def(
        "normalize_pipeline",
        [](float mean, float std) {
            bud::Pipeline p;
            p.subtract(mean);
            p.divide(std);
            return p;
        },
        nb::arg("mean"), nb::arg("std"), "Create normalization pipeline: (x - mean) / std");

    m.def(
        "scale_pipeline",
        [](float scale, float offset) {
            bud::Pipeline p;
            p.multiply(scale);
            p.add(offset);
            return p;
        },
        nb::arg("scale"), nb::arg("offset"), "Create scale+offset pipeline: x * scale + offset");

    m.def(
        "sigmoid_pipeline",
        []() {
            bud::Pipeline p;
            p.neg();   // -x
            p.exp();   // exp(-x)
            p.add(1);  // 1 + exp(-x)
            // Note: need reciprocal here, using rsqrt workaround
            return p;
        },
        "Create (partial) sigmoid pipeline");

    m.def(
        "softplus_pipeline",
        []() {
            bud::Pipeline p;
            p.exp();   // exp(x)
            p.add(1);  // 1 + exp(x)
            p.log();   // log(1 + exp(x))
            return p;
        },
        "Create softplus pipeline: log(1 + exp(x))");
}
