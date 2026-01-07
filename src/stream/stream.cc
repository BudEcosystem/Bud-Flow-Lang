// =============================================================================
// Bud Flow Lang - Stream Implementation
// =============================================================================

#include "bud_flow_lang/stream.h"

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>

namespace bud {

// =============================================================================
// StreamChunk Implementation
// =============================================================================

StreamChunk::StreamChunk(Bunch data) : data_(std::move(data)), global_offset_(0) {}

StreamChunk::StreamChunk(Bunch data, size_t global_offset)
    : data_(std::move(data)), global_offset_(global_offset) {}

float StreamChunk::at(size_t index) const {
    if (index >= data_.size()) {
        return 0.0f;  // Out of bounds
    }
    return static_cast<const float*>(data_.data())[index];
}

// =============================================================================
// Stream Private Constructor
// =============================================================================

Stream::Stream() = default;

// =============================================================================
// Stream Factory Methods
// =============================================================================

Stream Stream::fromBunch(const Bunch& data) {
    Stream s;
    s.owned_data_ = std::make_shared<Bunch>(data);
    s.raw_data_ = static_cast<const float*>(s.owned_data_->data());
    s.total_size_ = data.size();
    s.size_known_ = true;
    s.dtype_ = data.dtype();
    return s;
}

Stream Stream::fromBunch(Bunch&& data) {
    Stream s;
    s.owned_data_ = std::make_shared<Bunch>(std::move(data));
    s.raw_data_ = static_cast<const float*>(s.owned_data_->data());
    s.total_size_ = s.owned_data_->size();
    s.size_known_ = true;
    s.dtype_ = s.owned_data_->dtype();
    return s;
}

Stream Stream::fromRaw(const float* data, size_t count) {
    Stream s;
    s.raw_data_ = data;
    s.total_size_ = count;
    s.size_known_ = true;
    s.dtype_ = ScalarType::kFloat32;
    return s;
}

Stream Stream::zeros(size_t count) {
    Stream s;
    auto bunch = Bunch::zeros(count, ScalarType::kFloat32);
    if (bunch) {
        s.owned_data_ = std::make_shared<Bunch>(std::move(*bunch));
        s.raw_data_ = static_cast<const float*>(s.owned_data_->data());
        s.total_size_ = count;
        s.size_known_ = true;
        s.dtype_ = ScalarType::kFloat32;
    }
    return s;
}

Stream Stream::range(float start, float stop, float step) {
    Stream s;
    s.is_generated_ = true;
    s.generator_ = [start, step](size_t i) { return start + static_cast<float>(i) * step; };
    s.total_size_ = static_cast<size_t>(std::ceil((stop - start) / step));
    s.size_known_ = true;
    s.dtype_ = ScalarType::kFloat32;
    return s;
}

Stream Stream::generate(size_t count, std::function<float(size_t)> generator) {
    Stream s;
    s.is_generated_ = true;
    s.generator_ = std::move(generator);
    s.total_size_ = count;
    s.size_known_ = true;
    s.dtype_ = ScalarType::kFloat32;
    return s;
}

// =============================================================================
// Stream Configuration
// =============================================================================

Stream& Stream::chunkSize(size_t size) {
    config_.chunk_size = size;
    return *this;
}

Stream& Stream::parallel(bool enabled, size_t num_workers) {
    config_.parallel_enabled = enabled;
    config_.num_workers = num_workers;
    return *this;
}

Stream& Stream::memoryLimit(size_t bytes) {
    config_.memory_limit_bytes = bytes;
    return *this;
}

// =============================================================================
// Lazy Operations
// =============================================================================

Stream Stream::map(Pipeline pipeline) const {
    Stream s = *this;
    StreamOp op;
    op.kind = StreamOp::Kind::kPipeline;
    op.pipeline = std::make_shared<Pipeline>(std::move(pipeline));
    s.ops_.push_back(std::move(op));
    return s;
}

Stream Stream::map(const std::string& op_name, float scalar) const {
    Pipeline p;
    if (op_name == "add")
        p.add(scalar);
    else if (op_name == "multiply" || op_name == "mul")
        p.multiply(scalar);
    else if (op_name == "subtract" || op_name == "sub")
        p.subtract(scalar);
    else if (op_name == "divide" || op_name == "div")
        p.divide(scalar);
    else if (op_name == "sqrt")
        p.sqrt();
    else if (op_name == "exp")
        p.exp();
    else if (op_name == "log")
        p.log();
    else if (op_name == "abs")
        p.abs();
    else if (op_name == "neg")
        p.neg();
    else if (op_name == "sin")
        p.sin();
    else if (op_name == "cos")
        p.cos();
    else if (op_name == "tanh")
        p.tanh();
    else
        return *this;  // Unknown op, return unchanged

    return map(std::move(p));
}

Stream Stream::filter(std::function<bool(size_t)> predicate) const {
    Stream s = *this;
    StreamOp op;
    op.kind = StreamOp::Kind::kFilter;
    op.filter_predicate = std::move(predicate);
    s.ops_.push_back(std::move(op));
    // After filtering, size is unknown
    s.size_known_ = false;
    return s;
}

Stream Stream::take(size_t n) const {
    Stream s = *this;
    StreamOp op;
    op.kind = StreamOp::Kind::kTake;
    op.count = n;
    s.ops_.push_back(std::move(op));
    s.total_size_ = std::min(total_size_, n);
    return s;
}

Stream Stream::skip(size_t n) const {
    Stream s = *this;
    StreamOp op;
    op.kind = StreamOp::Kind::kSkip;
    op.count = n;
    s.ops_.push_back(std::move(op));
    // Don't modify total_size_ here - it's the source size
    // The skip is applied during collect() by adjusting position
    return s;
}

Stream Stream::chain(const Stream& other) const {
    // Create a new stream that combines both
    Stream s;
    s.is_generated_ = true;

    // Capture both streams' data
    auto self_copy = std::make_shared<Stream>(*this);
    auto other_copy = std::make_shared<Stream>(other);

    size_t self_size = this->total_size_;
    s.generator_ = [self_copy, other_copy, self_size](size_t i) {
        if (i < self_size) {
            auto chunk = self_copy->getChunk(i, 1);
            return chunk.at(0);
        }
        auto chunk = other_copy->getChunk(i - self_size, 1);
        return chunk.at(0);
    };

    s.total_size_ = this->total_size_ + other.total_size_;
    s.size_known_ = this->size_known_ && other.size_known_;
    s.dtype_ = ScalarType::kFloat32;
    return s;
}

// =============================================================================
// Terminal Operations
// =============================================================================

Result<Bunch> Stream::collect() const {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Determine output size
    size_t output_size = total_size_;

    // Apply take/skip ops to determine final size
    for (const auto& op : ops_) {
        if (op.kind == StreamOp::Kind::kTake) {
            output_size = std::min(output_size, op.count);
        } else if (op.kind == StreamOp::Kind::kSkip) {
            output_size = (op.count < output_size) ? (output_size - op.count) : 0;
        }
    }

    // Handle empty output
    if (output_size == 0) {
        // Return an empty-ish bunch (minimum size 1 to avoid assertion)
        auto result = Bunch::zeros(1, dtype_);
        if (!result) {
            return Error(ErrorCode::kAllocationFailed, "Failed to allocate result");
        }
        stats_.total_elements = 0;
        return result;
    }

    // Allocate result
    auto result = Bunch::zeros(output_size, dtype_);
    if (!result) {
        return Error(ErrorCode::kAllocationFailed, "Failed to allocate result");
    }

    float* out_ptr = static_cast<float*>(result->mutableData());
    size_t chunk_size = effectiveChunkSize();
    size_t out_idx = 0;

    // Process in chunks
    size_t position = 0;
    size_t skip_count = 0;
    size_t take_remaining = output_size;

    // Calculate initial skip
    for (const auto& op : ops_) {
        if (op.kind == StreamOp::Kind::kSkip) {
            skip_count += op.count;
        }
    }

    position = skip_count;
    size_t elements_to_process = total_size_ - skip_count;

    while (position < total_size_ && out_idx < output_size) {
        size_t remaining = total_size_ - position;
        size_t current_chunk_size = std::min({chunk_size, remaining, take_remaining});

        auto chunk = getChunk(position, current_chunk_size);
        chunk = applyOps(chunk);

        // Copy to output
        size_t copy_count = std::min(chunk.size(), output_size - out_idx);
        const float* src = static_cast<const float*>(chunk.data().data());
        for (size_t i = 0; i < copy_count; ++i) {
            out_ptr[out_idx++] = src[i];
        }

        position += current_chunk_size;
        take_remaining -= copy_count;

        // Update stats
        stats_.chunks_processed++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.total_elements = out_idx;
    stats_.processing_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    stats_.bytes_read = total_size_ * sizeof(float);
    stats_.bytes_written = out_idx * sizeof(float);
    if (stats_.processing_time_ms > 0) {
        stats_.throughput_gb_s =
            (stats_.bytes_read + stats_.bytes_written) / (stats_.processing_time_ms * 1e6);
    }

    return result;
}

Result<float> Stream::reduce(float initial, std::function<float(float, float)> reducer) const {
    float result = initial;
    size_t chunk_size = effectiveChunkSize();
    size_t position = 0;

    // Calculate skip amount
    size_t skip_count = 0;
    for (const auto& op : ops_) {
        if (op.kind == StreamOp::Kind::kSkip) {
            skip_count += op.count;
        }
    }
    position = skip_count;

    while (position < total_size_) {
        size_t remaining = total_size_ - position;
        size_t current_chunk_size = std::min(chunk_size, remaining);

        auto chunk = getChunk(position, current_chunk_size);
        chunk = applyOps(chunk);

        const float* data = static_cast<const float*>(chunk.data().data());
        for (size_t i = 0; i < chunk.size(); ++i) {
            result = reducer(result, data[i]);
        }

        position += current_chunk_size;
        stats_.chunks_processed++;
    }

    stats_.total_elements = total_size_ - skip_count;
    return result;
}

Result<float> Stream::sum() const {
    return reduce(0.0f, [](float a, float b) { return a + b; });
}

Result<float> Stream::min() const {
    return reduce(std::numeric_limits<float>::max(),
                  [](float a, float b) { return std::min(a, b); });
}

Result<float> Stream::max() const {
    return reduce(std::numeric_limits<float>::lowest(),
                  [](float a, float b) { return std::max(a, b); });
}

size_t Stream::size() const {
    // Calculate effective size after take/skip operations
    size_t result = total_size_;
    for (const auto& op : ops_) {
        if (op.kind == StreamOp::Kind::kTake) {
            result = std::min(result, op.count);
        } else if (op.kind == StreamOp::Kind::kSkip) {
            result = (op.count < result) ? (result - op.count) : 0;
        }
    }
    return result;
}

size_t Stream::count() const {
    // If no filter ops, return total size
    bool has_filter = false;
    for (const auto& op : ops_) {
        if (op.kind == StreamOp::Kind::kFilter) {
            has_filter = true;
            break;
        }
    }

    if (!has_filter) {
        size_t result = total_size_;
        for (const auto& op : ops_) {
            if (op.kind == StreamOp::Kind::kTake) {
                result = std::min(result, op.count);
            } else if (op.kind == StreamOp::Kind::kSkip) {
                result = (op.count < result) ? (result - op.count) : 0;
            }
        }
        return result;
    }

    // Need to process to count after filtering
    size_t count = 0;
    size_t chunk_size = effectiveChunkSize();
    size_t position = 0;

    while (position < total_size_) {
        size_t remaining = total_size_ - position;
        size_t current_chunk_size = std::min(chunk_size, remaining);

        auto chunk = getChunk(position, current_chunk_size);
        chunk = applyOps(chunk);
        count += chunk.size();

        position += current_chunk_size;
    }

    return count;
}

bool Stream::any(std::function<bool(float)> predicate) const {
    size_t chunk_size = effectiveChunkSize();
    size_t position = 0;

    while (position < total_size_) {
        size_t remaining = total_size_ - position;
        size_t current_chunk_size = std::min(chunk_size, remaining);

        auto chunk = getChunk(position, current_chunk_size);
        chunk = applyOps(chunk);

        const float* data = static_cast<const float*>(chunk.data().data());
        for (size_t i = 0; i < chunk.size(); ++i) {
            if (predicate(data[i])) {
                return true;
            }
        }

        position += current_chunk_size;
    }

    return false;
}

bool Stream::all(std::function<bool(float)> predicate) const {
    size_t chunk_size = effectiveChunkSize();
    size_t position = 0;

    while (position < total_size_) {
        size_t remaining = total_size_ - position;
        size_t current_chunk_size = std::min(chunk_size, remaining);

        auto chunk = getChunk(position, current_chunk_size);
        chunk = applyOps(chunk);

        const float* data = static_cast<const float*>(chunk.data().data());
        for (size_t i = 0; i < chunk.size(); ++i) {
            if (!predicate(data[i])) {
                return false;
            }
        }

        position += current_chunk_size;
    }

    return true;
}

Result<void> Stream::forEach(std::function<void(const StreamChunk&)> action) const {
    size_t chunk_size = effectiveChunkSize();
    size_t position = 0;

    while (position < total_size_) {
        size_t remaining = total_size_ - position;
        size_t current_chunk_size = std::min(chunk_size, remaining);

        auto chunk = getChunk(position, current_chunk_size);
        chunk = applyOps(chunk);
        action(chunk);

        position += current_chunk_size;
        stats_.chunks_processed++;
    }

    return {};
}

// =============================================================================
// Chunk Iteration
// =============================================================================

Stream::ChunkIterator::ChunkIterator(const Stream* stream, size_t position)
    : stream_(stream),
      position_(position),
      chunk_size_(stream ? stream->effectiveChunkSize() : 0) {}

bool Stream::ChunkIterator::hasNext() const {
    return stream_ && position_ < stream_->total_size_;
}

StreamChunk Stream::ChunkIterator::next() {
    if (!hasNext()) {
        return StreamChunk();
    }

    size_t remaining = stream_->total_size_ - position_;
    size_t current_chunk_size = std::min(chunk_size_, remaining);

    auto chunk = stream_->getChunk(position_, current_chunk_size);
    chunk = stream_->applyOps(chunk);

    position_ += current_chunk_size;
    return chunk;
}

bool Stream::ChunkIterator::operator!=(const ChunkIterator& other) const {
    return position_ != other.position_;
}

Stream::ChunkIterator& Stream::ChunkIterator::operator++() {
    if (hasNext()) {
        size_t remaining = stream_->total_size_ - position_;
        size_t current_chunk_size = std::min(chunk_size_, remaining);
        position_ += current_chunk_size;
    }
    return *this;
}

StreamChunk Stream::ChunkIterator::operator*() {
    if (!hasNext()) {
        return StreamChunk();
    }

    size_t remaining = stream_->total_size_ - position_;
    size_t current_chunk_size = std::min(chunk_size_, remaining);

    auto chunk = stream_->getChunk(position_, current_chunk_size);
    return stream_->applyOps(chunk);
}

Stream::ChunkIterator Stream::begin() const {
    return ChunkIterator(this, 0);
}

Stream::ChunkIterator Stream::end() const {
    return ChunkIterator(this, total_size_);
}

// =============================================================================
// Pipeline Integration
// =============================================================================

Result<Pipeline> Stream::toPipeline() const {
    Pipeline p;

    for (const auto& op : ops_) {
        if (op.kind == StreamOp::Kind::kPipeline && op.pipeline) {
            // Merge pipeline operations
            for (size_t i = 0; i < op.pipeline->numStages(); ++i) {
                const auto& stage = op.pipeline->stageAt(i);
                p.addOp(stage.op_code);
            }
        }
    }

    return p;
}

// =============================================================================
// String Representation
// =============================================================================

std::string Stream::toString() const {
    std::ostringstream ss;
    ss << "Stream(";
    ss << "size=" << total_size_;
    if (!size_known_) {
        ss << " (estimated)";
    }
    ss << ", ops=" << ops_.size();
    if (config_.chunk_size > 0) {
        ss << ", chunk=" << config_.chunk_size;
    }
    if (config_.parallel_enabled) {
        ss << ", parallel";
    }
    ss << ")";
    return ss.str();
}

// =============================================================================
// Helper Methods
// =============================================================================

size_t Stream::effectiveChunkSize() const {
    if (config_.chunk_size > 0) {
        return config_.chunk_size;
    }
    // Default: 64KB chunks (16K floats)
    constexpr size_t kDefaultChunkSize = 16 * 1024;
    return std::min(kDefaultChunkSize, total_size_);
}

StreamChunk Stream::getChunk(size_t offset, size_t size) const {
    if (offset >= total_size_ || size == 0) {
        return StreamChunk();
    }

    size_t actual_size = std::min(size, total_size_ - offset);
    if (actual_size == 0) {
        return StreamChunk();
    }

    if (is_generated_ && generator_) {
        // Generate data
        auto data = Bunch::zeros(actual_size, ScalarType::kFloat32);
        if (data) {
            float* ptr = static_cast<float*>(data->mutableData());
            for (size_t i = 0; i < actual_size; ++i) {
                ptr[i] = generator_(offset + i);
            }
            return StreamChunk(std::move(*data), offset);
        }
        return StreamChunk();
    }

    if (raw_data_) {
        // Create view of raw data (copy for safety)
        auto data = Bunch::zeros(actual_size, ScalarType::kFloat32);
        if (data) {
            float* ptr = static_cast<float*>(data->mutableData());
            std::copy(raw_data_ + offset, raw_data_ + offset + actual_size, ptr);
            return StreamChunk(std::move(*data), offset);
        }
    }

    return StreamChunk();
}

StreamChunk Stream::applyOps(StreamChunk chunk) const {
    for (const auto& op : ops_) {
        switch (op.kind) {
        case StreamOp::Kind::kPipeline:
            if (op.pipeline) {
                auto result = op.pipeline->run(chunk.data());
                if (result) {
                    chunk = StreamChunk(std::move(*result), chunk.globalOffset());
                }
            }
            break;

        case StreamOp::Kind::kFilter:
            if (op.filter_predicate) {
                // Filter elements based on global index
                std::vector<float> filtered;
                filtered.reserve(chunk.size());
                const float* src = static_cast<const float*>(chunk.data().data());
                for (size_t i = 0; i < chunk.size(); ++i) {
                    if (op.filter_predicate(chunk.globalIndex(i))) {
                        filtered.push_back(src[i]);
                    }
                }
                // Handle empty filter result
                if (filtered.empty()) {
                    // Return chunk with size 1 but mark as empty
                    auto data = Bunch::zeros(1, ScalarType::kFloat32);
                    if (data) {
                        chunk = StreamChunk(std::move(*data), chunk.globalOffset());
                    }
                } else {
                    auto data = Bunch::zeros(filtered.size(), ScalarType::kFloat32);
                    if (data) {
                        float* ptr = static_cast<float*>(data->mutableData());
                        std::copy(filtered.begin(), filtered.end(), ptr);
                        chunk = StreamChunk(std::move(*data), chunk.globalOffset());
                    }
                }
            }
            break;

        case StreamOp::Kind::kTake:
        case StreamOp::Kind::kSkip:
            // These are handled at the collect level
            break;

        case StreamOp::Kind::kMap:
        case StreamOp::Kind::kFlatMap:
            // Map handled via Pipeline
            break;
        }
    }

    return chunk;
}

// =============================================================================
// StreamBuilder Implementation
// =============================================================================

StreamBuilder& StreamBuilder::from(const Bunch& data) {
    stream_ = std::make_unique<Stream>(Stream::fromBunch(data));
    return *this;
}

StreamBuilder& StreamBuilder::from(Bunch&& data) {
    stream_ = std::make_unique<Stream>(Stream::fromBunch(std::move(data)));
    return *this;
}

StreamBuilder& StreamBuilder::fromZeros(size_t count) {
    stream_ = std::make_unique<Stream>(Stream::zeros(count));
    return *this;
}

StreamBuilder& StreamBuilder::fromRange(float start, float stop, float step) {
    stream_ = std::make_unique<Stream>(Stream::range(start, stop, step));
    return *this;
}

StreamBuilder& StreamBuilder::chunkSize(size_t size) {
    if (stream_) {
        stream_->chunkSize(size);
    }
    return *this;
}

StreamBuilder& StreamBuilder::parallel(bool enabled) {
    if (stream_) {
        stream_->parallel(enabled);
    }
    return *this;
}

StreamBuilder& StreamBuilder::memoryLimit(size_t bytes) {
    if (stream_) {
        stream_->memoryLimit(bytes);
    }
    return *this;
}

Stream StreamBuilder::build() {
    if (stream_) {
        return std::move(*stream_);
    }
    return Stream::zeros(0);
}

}  // namespace bud
