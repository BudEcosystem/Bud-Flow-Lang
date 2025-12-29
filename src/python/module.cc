// =============================================================================
// Bud Flow Lang - Python Module Entry Point
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bud_flow_lang/bud_flow_lang.h"

namespace nb = nanobind;

NB_MODULE(bud_flow_lang_py, m) {
    m.doc() = "Bud Flow Lang - Python DSL for SIMD Programming";

    // Version
    m.attr("__version__") = bud::Version::string();

    // Initialize/shutdown
    m.def("initialize", []() {
        auto result = bud::initialize();
        if (!result) {
            throw std::runtime_error(result.error().toString());
        }
    }, "Initialize the Bud Flow Lang runtime");

    m.def("shutdown", &bud::shutdown, "Shutdown the runtime");

    // Hardware info
    m.def("get_simd_width", []() {
        return bud::getHardwareInfo().simd_width;
    }, "Get available SIMD width in bytes");
}
