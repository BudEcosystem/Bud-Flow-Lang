// =============================================================================
// Bud Flow Lang - Memory Hints Python Bindings
// =============================================================================

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/memory_hints.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void bind_memory(nb::module_& m) {
    // =========================================================================
    // AlignmentHint enum
    // =========================================================================
    nb::enum_<bud::AlignmentHint>(m, "AlignmentHint", "Memory alignment hints")
        .value("None", bud::AlignmentHint::kNone, "No alignment requirement")
        .value("SIMD16", bud::AlignmentHint::kSimd16, "16-byte (SSE)")
        .value("SIMD32", bud::AlignmentHint::kSimd32, "32-byte (AVX)")
        .value("SIMD64", bud::AlignmentHint::kSimd64, "64-byte (AVX-512/cache line)")
        .value("SIMD128", bud::AlignmentHint::kSimd128, "128-byte (cache line pair)");

    // =========================================================================
    // PrefetchHint enum
    // =========================================================================
    nb::enum_<bud::PrefetchHint>(m, "PrefetchHint", "Prefetch locality hints")
        .value("None", bud::PrefetchHint::kNone, "No prefetch")
        .value("Read", bud::PrefetchHint::kRead, "Prefetch for reading")
        .value("Write", bud::PrefetchHint::kWrite, "Prefetch for writing")
        .value("NonTemporal", bud::PrefetchHint::kNonTemp, "Non-temporal (use once)");

    // =========================================================================
    // MemoryLayout enum
    // =========================================================================
    nb::enum_<bud::MemoryLayout>(m, "MemoryLayout", "Memory ordering hints")
        .value("RowMajor", bud::MemoryLayout::kRowMajor, "C-style row-major")
        .value("ColumnMajor", bud::MemoryLayout::kColumnMajor, "Fortran-style column-major")
        .value("Default", bud::MemoryLayout::kDefault, "Use default for type");

    // =========================================================================
    // MemoryHints struct
    // =========================================================================
    nb::class_<bud::MemoryHints>(m, "MemoryHints", "Container for memory optimization hints")
        .def(nb::init<>())
        .def_rw("alignment", &bud::MemoryHints::alignment)
        .def_rw("prefetch", &bud::MemoryHints::prefetch)
        .def_rw("prefetch_distance", &bud::MemoryHints::prefetch_distance)
        .def_rw("use_streaming_store", &bud::MemoryHints::use_streaming_store)
        .def_rw("layout", &bud::MemoryHints::layout)
        .def_rw("is_read_only", &bud::MemoryHints::is_read_only)
        .def_rw("is_write_only", &bud::MemoryHints::is_write_only);

    // =========================================================================
    // Memory static class (exposed as module functions)
    // =========================================================================
    auto memory = m.def_submodule("Memory", "Memory optimization utilities");

    memory.def(
        "is_aligned", [](size_t ptr, size_t alignment) { return (ptr % alignment) == 0; },
        nb::arg("ptr"), nb::arg("alignment"),
        "Check if pointer address is aligned to given boundary");

    memory.def(
        "get_alignment",
        [](size_t ptr) {
            if (ptr == 0)
                return size_t(0);
            size_t alignment = 1;
            while ((ptr & 1) == 0) {
                alignment *= 2;
                ptr >>= 1;
                if (alignment >= 4096)
                    break;
            }
            return alignment;
        },
        nb::arg("ptr"), "Get alignment of pointer address");

    memory.def(
        "prefetch_read",
        [](size_t ptr) { bud::Memory::prefetchRead(reinterpret_cast<const void*>(ptr)); },
        nb::arg("ptr"), "Prefetch memory for reading");

    memory.def(
        "prefetch_write",
        [](size_t ptr) { bud::Memory::prefetchWrite(reinterpret_cast<void*>(ptr)); },
        nb::arg("ptr"), "Prefetch memory for writing");

    memory.def(
        "prefetch_non_temporal",
        [](size_t ptr) { bud::Memory::prefetchNonTemporal(reinterpret_cast<const void*>(ptr)); },
        nb::arg("ptr"), "Prefetch memory (non-temporal, use once)");

    memory.def("should_use_streaming_store", &bud::Memory::shouldUseStreamingStore,
               nb::arg("data_bytes"),
               "Check if streaming stores would be beneficial for given data size");

    memory.def("estimate_bandwidth", &bud::Memory::estimateBandwidth, nb::arg("elements"),
               nb::arg("element_size"), nb::arg("read") = true, nb::arg("write") = true,
               "Estimate memory bandwidth requirement in bytes");

    // Bunch-specific memory utilities
    memory.def(
        "check_alignment",
        [](const bud::Bunch& bunch, size_t required_alignment) {
            const void* ptr = bunch.data();
            if (!bud::Memory::isAligned(ptr, required_alignment)) {
                throw std::runtime_error("Bunch data not aligned to " +
                                         std::to_string(required_alignment) + " bytes");
            }
            return true;
        },
        nb::arg("bunch"), nb::arg("alignment"), "Check if Bunch data is aligned to given boundary");

    memory.def(
        "data_ptr", [](const bud::Bunch& bunch) { return reinterpret_cast<size_t>(bunch.data()); },
        nb::arg("bunch"), "Get raw data pointer address for expert-level operations");

    // =========================================================================
    // PrefetchScope - context manager for prefetch settings
    // =========================================================================
    nb::class_<bud::PrefetchScope>(m, "PrefetchScope",
                                   "Context manager for prefetch configuration.\n\n"
                                   "Usage:\n"
                                   "  with PrefetchScope(distance=4):\n"
                                   "      # Operations inside will use prefetch distance of 4\n"
                                   "      process(data)")

        .def(nb::init<size_t, bud::PrefetchHint>(), nb::arg("distance"),
             nb::arg("hint") = bud::PrefetchHint::kRead,
             "Create prefetch scope with given distance and hint")

        .def_static("current_distance", &bud::PrefetchScope::currentDistance,
                    "Get current prefetch distance")

        .def_static("current_hint", &bud::PrefetchScope::currentHint, "Get current prefetch hint")

        // Python context manager protocol
        .def("__enter__", [](bud::PrefetchScope& self) -> bud::PrefetchScope& { return self; })
        .def("__exit__", [](bud::PrefetchScope& self, nb::object, nb::object, nb::object) {
            // Destructor will restore previous settings
        });

    // =========================================================================
    // Prefetch convenience module
    // =========================================================================
    auto prefetch = m.def_submodule("Prefetch", "Prefetch utilities");

    prefetch.def(
        "ahead",
        [](size_t distance) { return new bud::PrefetchScope(distance, bud::PrefetchHint::kRead); },
        nb::arg("distance") = 4, nb::rv_policy::take_ownership,
        "Create prefetch scope with given distance ahead");

    prefetch.def(
        "for_write",
        [](size_t distance) { return new bud::PrefetchScope(distance, bud::PrefetchHint::kWrite); },
        nb::arg("distance") = 4, nb::rv_policy::take_ownership,
        "Create prefetch scope for write operations");

    prefetch.def(
        "non_temporal",
        [](size_t distance) {
            return new bud::PrefetchScope(distance, bud::PrefetchHint::kNonTemp);
        },
        nb::arg("distance") = 4, nb::rv_policy::take_ownership,
        "Create non-temporal prefetch scope");

    // =========================================================================
    // Convenience constants
    // =========================================================================
    m.attr("CACHE_LINE_SIZE") = size_t(64);
    m.attr("AVX_ALIGNMENT") = size_t(32);
    m.attr("AVX512_ALIGNMENT") = size_t(64);
    m.attr("SSE_ALIGNMENT") = size_t(16);
}
