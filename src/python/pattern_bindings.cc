// =============================================================================
// Bud Flow Lang - Pattern Python Bindings
// =============================================================================

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/pattern.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bind_pattern(nb::module_& m) {
    // =========================================================================
    // PatternType enum
    // =========================================================================
    nb::enum_<bud::PatternType>(m, "PatternType", "Type of access pattern")
        .value("Stride", bud::PatternType::kStride, "Every nth element")
        .value("Where", bud::PatternType::kWhere, "Elements matching predicate")
        .value("Block", bud::PatternType::kBlock, "Fixed-size blocks")
        .value("Indices", bud::PatternType::kIndices, "Specific indices")
        .value("Range", bud::PatternType::kRange, "Range with step")
        .value("Mask", bud::PatternType::kMask, "Boolean mask");

    // =========================================================================
    // Pattern class
    // =========================================================================
    nb::class_<bud::Pattern>(m, "Pattern",
                             "Access pattern for selecting and filtering array elements.\n\n"
                             "Patterns describe how to access data without executing the access.\n"
                             "They compile to efficient SIMD gather/scatter operations.\n\n"
                             "Examples:\n"
                             "  # Every other element\n"
                             "  every_other = Pattern.stride(2)\n"
                             "  \n"
                             "  # Specific indices\n"
                             "  selected = Pattern.indices([0, 5, 10, 15])\n"
                             "  \n"
                             "  # Block processing\n"
                             "  blocks = Pattern.block(64)\n")

        // Factory methods
        .def_static("stride", nb::overload_cast<size_t>(&bud::Pattern::stride), nb::arg("n"),
                    "Select every nth element (indices 0, n, 2n, ...)")

        .def_static("stride", nb::overload_cast<size_t, size_t>(&bud::Pattern::stride),
                    nb::arg("n"), nb::arg("offset"),
                    "Select every nth element starting from offset")

        .def_static("block", nb::overload_cast<size_t>(&bud::Pattern::block), nb::arg("size"),
                    "Divide data into fixed-size blocks for tiled processing")

        .def_static("block", nb::overload_cast<size_t, size_t>(&bud::Pattern::block),
                    nb::arg("size"), nb::arg("alignment"),
                    "Divide data into blocks with specific alignment")

        .def_static("indices", &bud::Pattern::indices, nb::arg("indices"),
                    "Select specific indices from the data")

        .def_static("range", &bud::Pattern::range, nb::arg("start"), nb::arg("stop"),
                    nb::arg("step") = 1, "Select range [start, stop) with optional step")

        .def_static("mask", &bud::Pattern::mask, nb::arg("mask"),
                    "Select elements where mask is True")

        // Properties
        .def_prop_ro("type", &bud::Pattern::type, "Pattern type")
        .def_prop_ro("stride_value", &bud::Pattern::strideValue,
                     "Stride value (for stride patterns)")
        .def_prop_ro("offset", &bud::Pattern::offset, "Offset value (for stride patterns)")
        .def_prop_ro("block_size", &bud::Pattern::blockSize, "Block size (for block patterns)")
        .def_prop_ro("alignment", &bud::Pattern::alignment, "Alignment (for block patterns)")

        // Operations
        .def("count_selected", &bud::Pattern::countSelected, nb::arg("data_size"),
             "Count how many elements would be selected from data of given size")

        .def("get_indices", &bud::Pattern::getSelectedIndices, nb::arg("data_size"),
             "Get the selected indices for data of given size")

        .def("requires_gather", &bud::Pattern::requiresGather,
             "Check if pattern requires gather (non-contiguous access)")

        .def("is_regular", &bud::Pattern::isRegular,
             "Check if pattern has regular/predictable stride")

        // Combine patterns
        .def("compose", &bud::Pattern::compose, nb::arg("other"),
             "Compose patterns: apply this pattern, then other")

        .def("intersect", &bud::Pattern::intersect, nb::arg("other"),
             "Intersect patterns: elements selected by both")

        .def("unite", &bud::Pattern::unite, nb::arg("other"),
             "Union patterns: elements selected by either")

        // String representation
        .def("__repr__", &bud::Pattern::toString)
        .def("__str__", &bud::Pattern::toString);

    // =========================================================================
    // BlockIterator class
    // =========================================================================
    nb::class_<bud::BlockIterator::Block>(m, "Block",
                                          "A block/tile of data for cache-friendly processing")
        .def_ro("start", &bud::BlockIterator::Block::start, "Start index")
        .def_ro("end", &bud::BlockIterator::Block::end, "End index (exclusive)")
        .def_ro("size", &bud::BlockIterator::Block::size, "Number of elements")
        .def_ro("is_last", &bud::BlockIterator::Block::is_last, "True if last block")
        .def("__repr__", [](const bud::BlockIterator::Block& b) {
            return "Block(start=" + std::to_string(b.start) + ", end=" + std::to_string(b.end) +
                   ", size=" + std::to_string(b.size) + ")";
        });

    nb::class_<bud::BlockIterator>(m, "BlockIterator",
                                   "Iterator over fixed-size blocks for tiled processing.\n\n"
                                   "Usage:\n"
                                   "  it = BlockIterator(data_size=1000, block_size=64)\n"
                                   "  while it.has_next():\n"
                                   "      block = it.next()\n"
                                   "      # Process data[block.start:block.end]")

        .def(nb::init<size_t, size_t, size_t>(), nb::arg("data_size"), nb::arg("block_size"),
             nb::arg("alignment") = 64, "Create block iterator")

        .def("has_next", &bud::BlockIterator::hasNext, "Check if there are more blocks")

        .def("next", &bud::BlockIterator::next, "Get next block")

        .def("reset", &bud::BlockIterator::reset, "Reset iterator to beginning")

        .def("num_blocks", &bud::BlockIterator::numBlocks, "Get total number of blocks")

        .def("block_at", &bud::BlockIterator::blockAt, nb::arg("index"),
             "Get block at specific index")

        // Python iterator protocol
        .def("__iter__",
             [](bud::BlockIterator& self) -> bud::BlockIterator& {
                 self.reset();
                 return self;
             })
        .def("__next__", [](bud::BlockIterator& self) -> bud::BlockIterator::Block {
            if (!self.hasNext()) {
                throw nb::stop_iteration();
            }
            return self.next();
        });

    // =========================================================================
    // PatternApplicator - utility functions
    // =========================================================================
    m.def(
        "select",
        [](const bud::Bunch& data, const bud::Pattern& pattern) {
            auto result = bud::PatternApplicator::select(data, pattern);
            if (!result) {
                throw std::runtime_error("Pattern select failed: " + result.error().toString());
            }
            return std::move(*result);
        },
        nb::arg("data"), nb::arg("pattern"), "Select elements from Bunch according to pattern");

    m.def(
        "scatter",
        [](bud::Bunch& dest, const bud::Bunch& values, const bud::Pattern& pattern) {
            auto result = bud::PatternApplicator::scatter(dest, values, pattern);
            if (!result) {
                throw std::runtime_error("Pattern scatter failed: " + result.error().toString());
            }
        },
        nb::arg("dest"), nb::arg("values"), nb::arg("pattern"),
        "Scatter values into Bunch according to pattern");
}
