// =============================================================================
// Bud Flow Lang - Stream Class (Developer Tier)
// =============================================================================
//
// Stream provides lazy, chunked processing for memory-efficient operations
// on large datasets. Operations are queued and executed chunk-by-chunk,
// enabling processing of data larger than available memory.
//
// Usage:
//   from flow import Stream, Pipeline
//
//   # Create a stream from array data
//   stream = Stream.from_array(large_array)
//
//   # Apply operations lazily
//   stream = stream.map(lambda x: x * 2).filter(lambda x: x > 0)
//
//   # Execute with chunked processing
//   result = stream.collect()  # Gathers all results
//
//   # Or process chunk by chunk
//   for chunk in stream.chunks(1024):
//       process(chunk)
//
// Benefits:
// - Memory-efficient processing of large datasets
// - Lazy evaluation (operations only execute when needed)
// - Automatic chunking for cache-friendly execution
// - Integration with Pipeline for fused operations
// - Parallel chunk processing support
//
// =============================================================================

#pragma once

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/pipeline.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bud {

// Forward declarations
class Stream;
class StreamChunk;
class StreamIterator;

// =============================================================================
// StreamConfig - Configuration for stream processing
// =============================================================================

struct StreamConfig {
    // Chunk size for processing (0 = auto-detect based on cache)
    size_t chunk_size = 0;

    // Number of chunks to prefetch ahead
    size_t prefetch_count = 2;

    // Enable parallel chunk processing
    bool parallel_enabled = false;

    // Number of worker threads (0 = auto-detect)
    size_t num_workers = 0;

    // Enable double-buffering for overlapped I/O and compute
    bool double_buffer = true;

    // Memory limit for stream processing (0 = no limit)
    size_t memory_limit_bytes = 0;
};

// =============================================================================
// StreamStats - Statistics from stream processing
// =============================================================================

struct StreamStats {
    size_t total_elements = 0;
    size_t chunks_processed = 0;
    size_t bytes_read = 0;
    size_t bytes_written = 0;
    double processing_time_ms = 0.0;
    double throughput_gb_s = 0.0;
};

// =============================================================================
// StreamOp - A single operation in the stream pipeline
// =============================================================================

struct StreamOp {
    enum class Kind {
        kMap,       // Transform each element
        kFilter,    // Keep elements matching predicate
        kFlatMap,   // Transform and flatten
        kTake,      // Take first N elements
        kSkip,      // Skip first N elements
        kPipeline,  // Apply a Pipeline
    };

    Kind kind = Kind::kMap;

    // For kMap: transformation function (via Pipeline)
    std::shared_ptr<Pipeline> pipeline;

    // For kFilter: predicate (element index -> bool)
    std::function<bool(size_t)> filter_predicate;

    // For kTake/kSkip: count
    size_t count = 0;
};

// =============================================================================
// StreamChunk - A chunk of data from the stream
// =============================================================================

class StreamChunk {
  public:
    StreamChunk() = default;
    explicit StreamChunk(Bunch data);
    StreamChunk(Bunch data, size_t global_offset);

    // Accessors
    [[nodiscard]] const Bunch& data() const { return data_; }
    [[nodiscard]] Bunch& data() { return data_; }
    [[nodiscard]] size_t size() const { return data_.size(); }
    [[nodiscard]] size_t globalOffset() const { return global_offset_; }
    [[nodiscard]] bool empty() const { return data_.size() == 0; }

    // Get element at local index
    [[nodiscard]] float at(size_t index) const;

    // Get global index from local index
    [[nodiscard]] size_t globalIndex(size_t local_index) const {
        return global_offset_ + local_index;
    }

  private:
    Bunch data_;
    size_t global_offset_ = 0;
};

// =============================================================================
// Stream - Lazy, chunked data processing abstraction
// =============================================================================

class Stream {
  public:
    // =========================================================================
    // Factory Methods
    // =========================================================================

    // Create stream from a Bunch (copies data)
    [[nodiscard]] static Stream fromBunch(const Bunch& data);

    // Create stream from a Bunch (moves data)
    [[nodiscard]] static Stream fromBunch(Bunch&& data);

    // Create stream from raw data (does not copy, must outlive stream)
    [[nodiscard]] static Stream fromRaw(const float* data, size_t count);

    // Create stream of zeros
    [[nodiscard]] static Stream zeros(size_t count);

    // Create stream from range [start, stop) with step
    [[nodiscard]] static Stream range(float start, float stop, float step = 1.0f);

    // Create stream from generator function
    [[nodiscard]] static Stream generate(size_t count, std::function<float(size_t)> generator);

    // =========================================================================
    // Stream Configuration
    // =========================================================================

    // Set chunk size for processing
    Stream& chunkSize(size_t size);

    // Enable parallel processing
    Stream& parallel(bool enabled = true, size_t num_workers = 0);

    // Set memory limit
    Stream& memoryLimit(size_t bytes);

    // Get current configuration
    [[nodiscard]] const StreamConfig& config() const { return config_; }

    // =========================================================================
    // Lazy Operations (return new Stream)
    // =========================================================================

    // Apply transformation to each element
    [[nodiscard]] Stream map(Pipeline pipeline) const;

    // Apply scalar operation to each element
    [[nodiscard]] Stream map(const std::string& op, float scalar = 0.0f) const;

    // Filter elements based on predicate on indices
    [[nodiscard]] Stream filter(std::function<bool(size_t)> predicate) const;

    // Take first N elements
    [[nodiscard]] Stream take(size_t n) const;

    // Skip first N elements
    [[nodiscard]] Stream skip(size_t n) const;

    // Chain another stream operation
    [[nodiscard]] Stream chain(const Stream& other) const;

    // =========================================================================
    // Terminal Operations (execute the stream)
    // =========================================================================

    // Collect all results into a Bunch
    [[nodiscard]] Result<Bunch> collect() const;

    // Reduce all elements using a reduction operation
    [[nodiscard]] Result<float> reduce(float initial,
                                       std::function<float(float, float)> reducer) const;

    // Sum all elements
    [[nodiscard]] Result<float> sum() const;

    // Get minimum element
    [[nodiscard]] Result<float> min() const;

    // Get maximum element
    [[nodiscard]] Result<float> max() const;

    // Count elements (after filtering)
    [[nodiscard]] size_t count() const;

    // Check if any element matches predicate
    [[nodiscard]] bool any(std::function<bool(float)> predicate) const;

    // Check if all elements match predicate
    [[nodiscard]] bool all(std::function<bool(float)> predicate) const;

    // Execute operations and discard results (for side effects)
    Result<void> forEach(std::function<void(const StreamChunk&)> action) const;

    // =========================================================================
    // Chunk Iteration
    // =========================================================================

    // Get iterator over chunks
    class ChunkIterator {
      public:
        ChunkIterator(const Stream* stream, size_t position);

        [[nodiscard]] bool hasNext() const;
        [[nodiscard]] StreamChunk next();

        // Standard iterator interface
        [[nodiscard]] bool operator!=(const ChunkIterator& other) const;
        ChunkIterator& operator++();
        [[nodiscard]] StreamChunk operator*();

      private:
        const Stream* stream_;
        size_t position_;
        size_t chunk_size_;
    };

    [[nodiscard]] ChunkIterator begin() const;
    [[nodiscard]] ChunkIterator end() const;

    // =========================================================================
    // Properties
    // =========================================================================

    // Get total element count (if known), accounting for take/skip
    [[nodiscard]] size_t size() const;

    // Check if size is known
    [[nodiscard]] bool hasKnownSize() const { return size_known_; }

    // Get data type
    [[nodiscard]] ScalarType dtype() const { return dtype_; }

    // Get processing statistics
    [[nodiscard]] const StreamStats& stats() const { return stats_; }

    // =========================================================================
    // Pipeline Integration
    // =========================================================================

    // Create a Pipeline from this stream's operations
    [[nodiscard]] Result<Pipeline> toPipeline() const;

    // =========================================================================
    // String Representation
    // =========================================================================

    [[nodiscard]] std::string toString() const;

  private:
    // Private constructor
    Stream();

    // Source data
    std::shared_ptr<Bunch> owned_data_;  // Owned data (if fromBunch with copy/move)
    const float* raw_data_ = nullptr;    // Raw pointer to external data
    size_t total_size_ = 0;
    bool size_known_ = true;
    ScalarType dtype_ = ScalarType::kFloat32;

    // Generator function (if generated)
    std::function<float(size_t)> generator_;
    bool is_generated_ = false;

    // Configuration
    StreamConfig config_;

    // Queued operations
    std::vector<StreamOp> ops_;

    // Statistics
    mutable StreamStats stats_;

    // Helper methods
    [[nodiscard]] size_t effectiveChunkSize() const;
    [[nodiscard]] StreamChunk getChunk(size_t offset, size_t size) const;
    [[nodiscard]] StreamChunk applyOps(StreamChunk chunk) const;
};

// =============================================================================
// StreamBuilder - Fluent API for building streams
// =============================================================================

class StreamBuilder {
  public:
    StreamBuilder() = default;

    // Set source
    StreamBuilder& from(const Bunch& data);
    StreamBuilder& from(Bunch&& data);
    StreamBuilder& fromZeros(size_t count);
    StreamBuilder& fromRange(float start, float stop, float step = 1.0f);

    // Set configuration
    StreamBuilder& chunkSize(size_t size);
    StreamBuilder& parallel(bool enabled = true);
    StreamBuilder& memoryLimit(size_t bytes);

    // Build the stream
    [[nodiscard]] Stream build();

  private:
    std::unique_ptr<Stream> stream_;
};

// Convenience function to create builder
[[nodiscard]] inline StreamBuilder stream() {
    return StreamBuilder();
}

}  // namespace bud
