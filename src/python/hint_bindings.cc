// =============================================================================
// Bud Flow Lang - CompileHint Python Bindings
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_hints(nb::module_& m) {
    nb::class_<bud::CompileHint>(m, "CompileHint")
        .def(nb::init<>())
        .def_rw("unroll_factor", &bud::CompileHint::unroll_factor)
        .def_rw("force_vectorize", &bud::CompileHint::force_vectorize)
        .def_rw("enable_prefetch", &bud::CompileHint::enable_prefetch);
}
