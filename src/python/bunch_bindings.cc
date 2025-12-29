// =============================================================================
// Bud Flow Lang - Bunch Python Bindings
// =============================================================================

#include "bud_flow_lang/bunch.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bind_bunch(nb::module_& m) {
    nb::class_<bud::Bunch>(m, "Bunch")
        .def_static(
            "zeros",
            [](size_t count) {
                auto result = bud::Bunch::zeros(count);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch");
                return *result;
            },
            nb::arg("count"), "Create a Bunch of zeros")

        .def_static(
            "ones",
            [](size_t count) {
                auto result = bud::Bunch::ones(count);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch");
                return *result;
            },
            nb::arg("count"), "Create a Bunch of ones")

        .def_static(
            "fill",
            [](size_t count, float value) {
                auto result = bud::Bunch::fill(count, value);
                if (!result)
                    throw std::runtime_error("Failed to create Bunch");
                return *result;
            },
            nb::arg("count"), nb::arg("value"), "Create a Bunch filled with value")

        .def_prop_ro("size", &bud::Bunch::size, "Number of elements")
        .def("sum", &bud::Bunch::sum, "Sum of all elements")
        .def("mean", &bud::Bunch::mean, "Mean of all elements")
        .def("min", &bud::Bunch::min, "Minimum element")
        .def("max", &bud::Bunch::max, "Maximum element")
        .def("dot", &bud::Bunch::dot, nb::arg("other"), "Dot product")
        .def("__repr__", &bud::Bunch::toString);
}
