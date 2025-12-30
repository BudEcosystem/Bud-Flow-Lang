// =============================================================================
// Bud Flow Lang - CompileHint Python Bindings
// =============================================================================

#include "bud_flow_lang/bud_flow_lang.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void bind_hints(nb::module_& m) {
    nb::class_<bud::CompileHint>(m, "CompileHint",
                                 "Compilation hints to control JIT optimization behavior")
        .def(nb::init<>())

        // Unrolling
        .def_rw("unroll_factor", &bud::CompileHint::unroll_factor,
                "Loop unroll factor (0 = auto, N = unroll N times)")

        // Vectorization
        .def_rw("force_vectorize", &bud::CompileHint::force_vectorize,
                "Force vectorization even if the compiler thinks it's not profitable")
        .def_rw("disable_vectorize", &bud::CompileHint::disable_vectorize,
                "Disable vectorization for this flow")

        // Tiling (for matrix operations)
        .def_rw("tile_m", &bud::CompileHint::tile_m,
                "Tile size for M dimension in matrix operations (0 = auto)")
        .def_rw("tile_n", &bud::CompileHint::tile_n,
                "Tile size for N dimension in matrix operations (0 = auto)")
        .def_rw("tile_k", &bud::CompileHint::tile_k,
                "Tile size for K dimension in matrix operations (0 = auto)")

        // Prefetching
        .def_rw("enable_prefetch", &bud::CompileHint::enable_prefetch, "Enable memory prefetching")
        .def_rw("prefetch_distance", &bud::CompileHint::prefetch_distance,
                "Prefetch distance in cache lines (0 = auto)")

        // Target ISA
        .def_rw("target_isa", &bud::CompileHint::target_isa,
                "Target ISA override (empty = auto-detect, e.g., 'AVX2', 'AVX512', 'NEON')")

        .def("__repr__", [](const bud::CompileHint& h) {
            return "<CompileHint unroll=" + std::to_string(h.unroll_factor) + " vec=" +
                   (h.force_vectorize ? "force" : (h.disable_vectorize ? "off" : "auto")) +
                   " prefetch=" + (h.enable_prefetch ? "on" : "off") +
                   (h.target_isa.empty() ? "" : " isa=" + h.target_isa) + ">";
        });
}
