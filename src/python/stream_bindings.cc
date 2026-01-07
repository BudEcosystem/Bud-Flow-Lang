// =============================================================================
// Bud Flow Lang - Stream Python Bindings
// =============================================================================

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/pipeline.h"
#include "bud_flow_lang/stream.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void bind_stream(nb::module_& m) {
    // =========================================================================
    // StreamConfig struct
    // =========================================================================
    nb::class_<bud::StreamConfig>(m, "StreamConfig", "Configuration for stream processing")
        .def(nb::init<>(), "Create default configuration")
        .def_rw("chunk_size", &bud::StreamConfig::chunk_size,
                "Chunk size for processing (0 = auto)")
        .def_rw("prefetch_count", &bud::StreamConfig::prefetch_count,
                "Number of chunks to prefetch")
        .def_rw("parallel_enabled", &bud::StreamConfig::parallel_enabled,
                "Enable parallel chunk processing")
        .def_rw("num_workers", &bud::StreamConfig::num_workers,
                "Number of worker threads (0 = auto)")
        .def_rw("double_buffer", &bud::StreamConfig::double_buffer, "Enable double-buffering")
        .def_rw("memory_limit_bytes", &bud::StreamConfig::memory_limit_bytes,
                "Memory limit (0 = no limit)");

    // =========================================================================
    // StreamStats struct
    // =========================================================================
    nb::class_<bud::StreamStats>(m, "StreamStats", "Statistics from stream processing")
        .def(nb::init<>())
        .def_ro("total_elements", &bud::StreamStats::total_elements)
        .def_ro("chunks_processed", &bud::StreamStats::chunks_processed)
        .def_ro("bytes_read", &bud::StreamStats::bytes_read)
        .def_ro("bytes_written", &bud::StreamStats::bytes_written)
        .def_ro("processing_time_ms", &bud::StreamStats::processing_time_ms)
        .def_ro("throughput_gb_s", &bud::StreamStats::throughput_gb_s);

    // =========================================================================
    // StreamChunk class
    // =========================================================================
    nb::class_<bud::StreamChunk>(m, "StreamChunk", "A chunk of data from the stream")
        .def(nb::init<>())
        .def_prop_ro(
            "data", [](const bud::StreamChunk& self) -> const bud::Bunch& { return self.data(); },
            nb::rv_policy::reference, "Get chunk data as Bunch")
        .def_prop_ro("size", &bud::StreamChunk::size, "Number of elements")
        .def_prop_ro("global_offset", &bud::StreamChunk::globalOffset, "Offset in original stream")
        .def("empty", &bud::StreamChunk::empty, "Check if chunk is empty")
        .def("at", &bud::StreamChunk::at, nb::arg("index"), "Get element at local index")
        .def("global_index", &bud::StreamChunk::globalIndex, nb::arg("local_index"),
             "Convert local index to global index");

    // =========================================================================
    // Stream class
    // =========================================================================
    nb::class_<bud::Stream>(m, "Stream",
                            "Lazy, chunked data processing abstraction.\n\n"
                            "Stream provides memory-efficient processing of large datasets\n"
                            "through lazy evaluation and automatic chunking.\n\n"
                            "Example:\n"
                            "  stream = Stream.from_bunch(data)\n"
                            "  result = stream.map('multiply', 2.0).map('sqrt').collect()\n")

        // Factory methods (static)
        .def_static("from_bunch", nb::overload_cast<const bud::Bunch&>(&bud::Stream::fromBunch),
                    nb::arg("data"), "Create stream from Bunch (copies data)")

        .def_static(
            "from_raw",
            [](nb::ndarray<float, nb::ndim<1>, nb::c_contig> arr) {
                return bud::Stream::fromRaw(arr.data(), arr.shape(0));
            },
            nb::arg("data"), "Create stream from numpy array")

        .def_static("zeros", &bud::Stream::zeros, nb::arg("count"), "Create stream of zeros")

        .def_static("range", &bud::Stream::range, nb::arg("start"), nb::arg("stop"),
                    nb::arg("step") = 1.0f, "Create stream from range [start, stop) with step")

        .def_static(
            "generate",
            [](size_t count, nb::callable gen) {
                return bud::Stream::generate(count, [gen](size_t i) -> float {
                    nb::gil_scoped_acquire guard;
                    return nb::cast<float>(gen(i));
                });
            },
            nb::arg("count"), nb::arg("generator"), "Create stream from generator function")

        // Configuration
        .def("chunk_size", &bud::Stream::chunkSize, nb::arg("size"), nb::rv_policy::reference,
             "Set chunk size for processing")

        .def("parallel", &bud::Stream::parallel, nb::arg("enabled") = true,
             nb::arg("num_workers") = 0, nb::rv_policy::reference, "Enable parallel processing")

        .def("memory_limit", &bud::Stream::memoryLimit, nb::arg("bytes"), nb::rv_policy::reference,
             "Set memory limit for processing")

        .def_prop_ro("config", &bud::Stream::config, "Get current configuration")

        // Lazy operations
        .def(
            "map",
            [](const bud::Stream& self, const bud::Pipeline& pipeline) {
                // Create a temporary pipeline copy via rebuild
                bud::Pipeline p;
                for (size_t i = 0; i < pipeline.numStages(); ++i) {
                    const auto& stage = pipeline.stageAt(i);
                    if (stage.scalar_param != 0.0f) {
                        // Use op_code directly to preserve operation type
                        switch (stage.op_code) {
                        case bud::ir::OpCode::kMul:
                            p.multiply(stage.scalar_param);
                            break;
                        case bud::ir::OpCode::kAdd:
                            p.add(stage.scalar_param);
                            break;
                        case bud::ir::OpCode::kSub:
                            p.subtract(stage.scalar_param);
                            break;
                        case bud::ir::OpCode::kDiv:
                            p.divide(stage.scalar_param);
                            break;
                        default:
                            p.addOp(stage.op_code);
                        }
                    } else {
                        p.addOp(stage.op_code);
                    }
                }
                return self.map(std::move(p));
            },
            nb::arg("pipeline"), "Apply Pipeline transformation to each chunk")

        .def(
            "map_op",
            [](const bud::Stream& self, const std::string& op, float scalar) {
                return self.map(op, scalar);
            },
            nb::arg("op"), nb::arg("scalar") = 0.0f, "Apply named operation to each element")

        .def(
            "filter",
            [](const bud::Stream& self, nb::callable pred) {
                return self.filter([pred](size_t i) -> bool {
                    nb::gil_scoped_acquire guard;
                    return nb::cast<bool>(pred(i));
                });
            },
            nb::arg("predicate"), "Filter elements by index predicate")

        .def("take", &bud::Stream::take, nb::arg("n"), "Take first N elements")

        .def("skip", &bud::Stream::skip, nb::arg("n"), "Skip first N elements")

        .def("chain", &bud::Stream::chain, nb::arg("other"), "Chain with another stream")

        // Terminal operations
        .def(
            "collect",
            [](const bud::Stream& self) {
                auto result = self.collect();
                if (!result) {
                    throw std::runtime_error("Stream collect failed: " + result.error().toString());
                }
                return std::move(*result);
            },
            "Collect all results into a Bunch")

        .def(
            "reduce",
            [](const bud::Stream& self, float initial, nb::callable reducer) {
                auto result = self.reduce(initial, [reducer](float a, float b) -> float {
                    nb::gil_scoped_acquire guard;
                    return nb::cast<float>(reducer(a, b));
                });
                if (!result) {
                    throw std::runtime_error("Stream reduce failed");
                }
                return *result;
            },
            nb::arg("initial"), nb::arg("reducer"), "Reduce all elements")

        .def(
            "sum",
            [](const bud::Stream& self) {
                auto result = self.sum();
                if (!result) {
                    throw std::runtime_error("Stream sum failed");
                }
                return *result;
            },
            "Sum all elements")

        .def(
            "min",
            [](const bud::Stream& self) {
                auto result = self.min();
                if (!result) {
                    throw std::runtime_error("Stream min failed");
                }
                return *result;
            },
            "Get minimum element")

        .def(
            "max",
            [](const bud::Stream& self) {
                auto result = self.max();
                if (!result) {
                    throw std::runtime_error("Stream max failed");
                }
                return *result;
            },
            "Get maximum element")

        .def("count", &bud::Stream::count, "Count elements (after filtering)")

        .def(
            "any",
            [](const bud::Stream& self, nb::callable pred) {
                return self.any([pred](float val) -> bool {
                    nb::gil_scoped_acquire guard;
                    return nb::cast<bool>(pred(val));
                });
            },
            nb::arg("predicate"), "Check if any element matches predicate")

        .def(
            "all",
            [](const bud::Stream& self, nb::callable pred) {
                return self.all([pred](float val) -> bool {
                    nb::gil_scoped_acquire guard;
                    return nb::cast<bool>(pred(val));
                });
            },
            nb::arg("predicate"), "Check if all elements match predicate")

        .def(
            "for_each",
            [](const bud::Stream& self, nb::callable action) {
                auto result = self.forEach([action](const bud::StreamChunk& chunk) {
                    nb::gil_scoped_acquire guard;
                    action(chunk);
                });
                if (!result) {
                    throw std::runtime_error("Stream forEach failed");
                }
            },
            nb::arg("action"), "Execute action for each chunk")

        // Properties
        .def_prop_ro("size", &bud::Stream::size, "Total element count (if known)")

        .def("has_known_size", &bud::Stream::hasKnownSize, "Check if size is known")

        .def_prop_ro("dtype", &bud::Stream::dtype, "Data type")

        .def_prop_ro("stats", &bud::Stream::stats, "Processing statistics")

        // Pipeline integration
        .def(
            "to_pipeline",
            [](const bud::Stream& self) {
                auto result = self.toPipeline();
                if (!result) {
                    throw std::runtime_error("Failed to create pipeline");
                }
                return std::move(*result);
            },
            "Create Pipeline from stream operations")

        // Iteration
        .def(
            "__iter__", [](const bud::Stream& self) { return self.begin(); },
            "Get iterator over chunks")

        // String representation
        .def("__repr__", &bud::Stream::toString)
        .def("__str__", &bud::Stream::toString)
        .def("__len__", &bud::Stream::size);

    // =========================================================================
    // ChunkIterator for Python iteration
    // =========================================================================
    nb::class_<bud::Stream::ChunkIterator>(m, "StreamChunkIterator")
        .def("__next__",
             [](bud::Stream::ChunkIterator& self) {
                 if (!self.hasNext()) {
                     throw nb::stop_iteration();
                 }
                 return self.next();
             })
        .def("__iter__",
             [](bud::Stream::ChunkIterator& self) -> bud::Stream::ChunkIterator& { return self; });

    // =========================================================================
    // Module-level stream functions
    // =========================================================================
    m.def("stream_from_bunch", nb::overload_cast<const bud::Bunch&>(&bud::Stream::fromBunch),
          nb::arg("data"), "Create stream from Bunch");

    m.def("stream_zeros", &bud::Stream::zeros, nb::arg("count"), "Create stream of zeros");

    m.def("stream_range", &bud::Stream::range, nb::arg("start"), nb::arg("stop"),
          nb::arg("step") = 1.0f, "Create stream from range");
}
