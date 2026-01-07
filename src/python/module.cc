// =============================================================================
// Bud Flow Lang - Python Module Entry Point
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <cstdlib>  // For atexit

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

// Global flag to track if we've registered the atexit handler
static bool g_atexit_registered = false;

// Cleanup function called at Python exit
static void cleanup_at_exit() {
    if (bud::isInitialized()) {
        bud::shutdown();
    }
}

// Forward declarations for binding functions defined in other files
void bind_bunch(nb::module_& m);
void bind_flow(nb::module_& m);
void bind_hints(nb::module_& m);
void bind_highway(nb::module_& m);
void bind_pattern(nb::module_& m);
void bind_pipeline(nb::module_& m);
void bind_memory(nb::module_& m);
void bind_stream(nb::module_& m);
void bind_continuous_tuner(nb::module_& m);

NB_MODULE(bud_flow_lang_py, m) {
    m.doc() = "Bud Flow Lang - High-performance SIMD array library for Python";

    // Version
    m.attr("__version__") = bud::Version::string();

    // Register class bindings
    bind_bunch(m);
    bind_flow(m);
    bind_hints(m);
    bind_highway(m);
    bind_pattern(m);
    bind_pipeline(m);
    bind_memory(m);
    bind_stream(m);
    bind_continuous_tuner(m);

    // Initialize/shutdown
    m.def(
        "initialize",
        []() {
            auto result = bud::initialize();
            if (!result) {
                throw std::runtime_error(result.error().toString());
            }
            // Register cleanup handler on first successful init
            if (!g_atexit_registered) {
                std::atexit(cleanup_at_exit);
                g_atexit_registered = true;
            }
        },
        "Initialize the Bud Flow Lang runtime");

    m.def("shutdown", &bud::shutdown, "Shutdown the runtime");

    // Hardware info
    m.def(
        "get_simd_width", []() { return bud::getHardwareInfo().simd_width; },
        "Get available SIMD width in bytes");

    m.def("is_initialized", &bud::isInitialized, "Check if the runtime is initialized");
}
