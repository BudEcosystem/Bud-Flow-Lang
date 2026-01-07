// =============================================================================
// Bud Flow Lang - Pattern Implementation
// =============================================================================

#include "bud_flow_lang/pattern.h"

#include "bud_flow_lang/bunch.h"

#include <algorithm>
#include <iterator>
#include <sstream>

namespace bud {

// =============================================================================
// Pattern Factory Methods
// =============================================================================

Pattern Pattern::stride(size_t n) {
    Pattern p;
    p.type_ = PatternType::kStride;
    p.stride_ = n;
    p.offset_ = 0;
    return p;
}

Pattern Pattern::stride(size_t n, size_t offset) {
    Pattern p;
    p.type_ = PatternType::kStride;
    p.stride_ = n;
    p.offset_ = offset;
    return p;
}

Pattern Pattern::block(size_t size) {
    Pattern p;
    p.type_ = PatternType::kBlock;
    p.block_size_ = size;
    p.alignment_ = 64;  // Default cache line alignment
    return p;
}

Pattern Pattern::block(size_t size, size_t alignment) {
    Pattern p;
    p.type_ = PatternType::kBlock;
    p.block_size_ = size;
    p.alignment_ = alignment;
    return p;
}

Pattern Pattern::indices(std::vector<size_t> idx) {
    Pattern p;
    p.type_ = PatternType::kIndices;
    p.indices_ = std::move(idx);
    return p;
}

Pattern Pattern::range(size_t start, size_t stop, size_t step) {
    Pattern p;
    p.type_ = PatternType::kRange;
    p.range_start_ = start;
    p.range_stop_ = stop;
    p.range_step_ = step;
    return p;
}

Pattern Pattern::mask(std::vector<bool> mask_values) {
    Pattern p;
    p.type_ = PatternType::kMask;
    p.mask_values_ = std::move(mask_values);
    return p;
}

// =============================================================================
// Pattern Operations
// =============================================================================

size_t Pattern::countSelected(size_t data_size) const {
    switch (type_) {
    case PatternType::kStride:
        if (offset_ >= data_size)
            return 0;
        return (data_size - offset_ + stride_ - 1) / stride_;

    case PatternType::kBlock:
        return data_size;  // Blocks don't reduce count

    case PatternType::kIndices:
        return std::count_if(indices_.begin(), indices_.end(),
                             [data_size](size_t i) { return i < data_size; });

    case PatternType::kRange: {
        if (range_start_ >= data_size || range_start_ >= range_stop_)
            return 0;
        size_t actual_stop = std::min(range_stop_, data_size);
        return (actual_stop - range_start_ + range_step_ - 1) / range_step_;
    }

    case PatternType::kMask: {
        size_t count = 0;
        size_t limit = std::min(mask_values_.size(), data_size);
        for (size_t i = 0; i < limit; ++i) {
            if (mask_values_[i])
                ++count;
        }
        return count;
    }

    case PatternType::kWhere:
        // Where patterns require the predicate to be evaluated
        // Return data_size as upper bound (actual count determined at runtime)
        return data_size;
    }
    return 0;
}

std::vector<size_t> Pattern::getSelectedIndices(size_t data_size) const {
    std::vector<size_t> result;

    switch (type_) {
    case PatternType::kStride:
        for (size_t i = offset_; i < data_size; i += stride_) {
            result.push_back(i);
        }
        break;

    case PatternType::kBlock:
        // Block patterns select all indices (in block order)
        for (size_t i = 0; i < data_size; ++i) {
            result.push_back(i);
        }
        break;

    case PatternType::kIndices:
        for (size_t idx : indices_) {
            if (idx < data_size) {
                result.push_back(idx);
            }
        }
        break;

    case PatternType::kRange:
        for (size_t i = range_start_; i < std::min(range_stop_, data_size); i += range_step_) {
            result.push_back(i);
        }
        break;

    case PatternType::kMask: {
        size_t limit = std::min(mask_values_.size(), data_size);
        for (size_t i = 0; i < limit; ++i) {
            if (mask_values_[i]) {
                result.push_back(i);
            }
        }
        break;
    }

    case PatternType::kWhere:
        // Where patterns need runtime evaluation
        break;
    }

    return result;
}

bool Pattern::requiresGather() const {
    switch (type_) {
    case PatternType::kStride:
        return stride_ > 1;  // Non-unit stride needs gather

    case PatternType::kBlock:
        return false;  // Blocks are contiguous

    case PatternType::kIndices:
        return true;  // Arbitrary indices always need gather

    case PatternType::kRange:
        return range_step_ > 1;

    case PatternType::kMask:
        return true;  // Masked access needs gather/scatter

    case PatternType::kWhere:
        return true;  // Predicated access needs gather/scatter
    }
    return false;
}

bool Pattern::isRegular() const {
    switch (type_) {
    case PatternType::kStride:
    case PatternType::kBlock:
    case PatternType::kRange:
        return true;  // Predictable stride patterns

    case PatternType::kIndices:
    case PatternType::kMask:
    case PatternType::kWhere:
        return false;  // Irregular access patterns
    }
    return false;
}

Pattern Pattern::compose(const Pattern& other) const {
    // Composing patterns: first apply this pattern, then other
    // Example: stride(2).compose(stride(3)) means:
    //   First: indices 0,2,4,6,8,10...
    //   Then apply stride(3): indices 0,6,12... (every 3rd of the first result)

    // Common case optimizations
    if (type_ == PatternType::kStride && other.type_ == PatternType::kStride) {
        // stride(a).compose(stride(b)) = stride(a*b, offset_*b)
        return Pattern::stride(stride_ * other.stride_, offset_ * other.stride_ + other.offset_);
    }

    if (type_ == PatternType::kRange && other.type_ == PatternType::kRange) {
        // Compose ranges
        size_t new_start = range_start_ + other.range_start_ * range_step_;
        size_t new_step = range_step_ * other.range_step_;
        size_t count = other.countSelected(countSelected(SIZE_MAX));
        size_t new_stop = new_start + count * new_step;
        return Pattern::range(new_start, new_stop, new_step);
    }

    // General case: convert to indices
    // Use a reasonable upper bound for data size
    constexpr size_t LARGE_SIZE = 1000000;
    auto first_indices = getSelectedIndices(LARGE_SIZE);
    auto composed_indices = other.getSelectedIndices(first_indices.size());

    std::vector<size_t> result;
    result.reserve(composed_indices.size());
    for (size_t idx : composed_indices) {
        if (idx < first_indices.size()) {
            result.push_back(first_indices[idx]);
        }
    }
    return Pattern::indices(std::move(result));
}

Pattern Pattern::intersect(const Pattern& other) const {
    // Intersection: elements selected by both patterns
    // Use a reasonable upper bound for data size
    constexpr size_t LARGE_SIZE = 1000000;

    auto indices_a = getSelectedIndices(LARGE_SIZE);
    auto indices_b = other.getSelectedIndices(LARGE_SIZE);

    // Sort for efficient intersection
    std::sort(indices_a.begin(), indices_a.end());
    std::sort(indices_b.begin(), indices_b.end());

    std::vector<size_t> result;
    std::set_intersection(indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(),
                          std::back_inserter(result));

    return Pattern::indices(std::move(result));
}

Pattern Pattern::unite(const Pattern& other) const {
    // Union: elements selected by either pattern
    // Use a reasonable upper bound for data size
    constexpr size_t LARGE_SIZE = 1000000;

    auto indices_a = getSelectedIndices(LARGE_SIZE);
    auto indices_b = other.getSelectedIndices(LARGE_SIZE);

    // Sort for efficient union
    std::sort(indices_a.begin(), indices_a.end());
    std::sort(indices_b.begin(), indices_b.end());

    std::vector<size_t> result;
    std::set_union(indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(),
                   std::back_inserter(result));

    return Pattern::indices(std::move(result));
}

std::string Pattern::toString() const {
    std::ostringstream ss;
    ss << "Pattern(";
    switch (type_) {
    case PatternType::kStride:
        ss << "stride=" << stride_;
        if (offset_ > 0)
            ss << ", offset=" << offset_;
        break;
    case PatternType::kBlock:
        ss << "block=" << block_size_;
        if (alignment_ != 64)
            ss << ", align=" << alignment_;
        break;
    case PatternType::kIndices:
        ss << "indices=[" << indices_.size() << " elements]";
        break;
    case PatternType::kRange:
        ss << "range=[" << range_start_ << ":" << range_stop_ << ":" << range_step_ << "]";
        break;
    case PatternType::kMask:
        ss << "mask=[" << mask_values_.size() << " elements]";
        break;
    case PatternType::kWhere:
        ss << "where=<predicate>";
        break;
    }
    ss << ")";
    return ss.str();
}

// =============================================================================
// BlockIterator Implementation
// =============================================================================

BlockIterator::BlockIterator(size_t data_size, size_t block_size, size_t alignment)
    : data_size_(data_size), block_size_(block_size), alignment_(alignment), current_start_(0) {}

BlockIterator::Block BlockIterator::next() {
    Block block;
    block.start = current_start_;
    block.end = std::min(current_start_ + block_size_, data_size_);
    block.size = block.end - block.start;
    block.is_last = (block.end >= data_size_);

    current_start_ = block.end;
    return block;
}

size_t BlockIterator::numBlocks() const {
    if (data_size_ == 0)
        return 0;
    return (data_size_ + block_size_ - 1) / block_size_;
}

BlockIterator::Block BlockIterator::blockAt(size_t index) const {
    Block block;
    block.start = index * block_size_;
    block.end = std::min(block.start + block_size_, data_size_);
    block.size = block.end - block.start;
    block.is_last = (block.end >= data_size_);
    return block;
}

// =============================================================================
// PatternApplicator Implementation
// =============================================================================

Result<Bunch> PatternApplicator::select(const Bunch& data, const Pattern& pattern) {
    size_t selected_count = pattern.countSelected(data.size());
    if (selected_count == 0) {
        return Bunch::zeros(0, data.dtype());
    }

    auto result = Bunch::zeros(selected_count, data.dtype());
    if (!result) {
        return Error(ErrorCode::kAllocationFailed, "Failed to allocate result Bunch");
    }

    // Get selected indices
    auto indices = pattern.getSelectedIndices(data.size());

    // Copy selected elements
    if (data.dtype() == ScalarType::kFloat32) {
        const float* src = static_cast<const float*>(data.data());
        float* dst = static_cast<float*>(result->mutableData());
        for (size_t i = 0; i < indices.size(); ++i) {
            dst[i] = src[indices[i]];
        }
    } else if (data.dtype() == ScalarType::kFloat64) {
        const double* src = static_cast<const double*>(data.data());
        double* dst = static_cast<double*>(result->mutableData());
        for (size_t i = 0; i < indices.size(); ++i) {
            dst[i] = src[indices[i]];
        }
    } else if (data.dtype() == ScalarType::kInt32) {
        const int32_t* src = static_cast<const int32_t*>(data.data());
        int32_t* dst = static_cast<int32_t*>(result->mutableData());
        for (size_t i = 0; i < indices.size(); ++i) {
            dst[i] = src[indices[i]];
        }
    }

    return result;
}

Result<void> PatternApplicator::scatter(Bunch& dest, const Bunch& values, const Pattern& pattern) {
    auto indices = pattern.getSelectedIndices(dest.size());
    if (indices.size() != values.size()) {
        return Error(ErrorCode::kInvalidInput, "Values size doesn't match pattern selection count");
    }

    // Scatter elements
    if (dest.dtype() == ScalarType::kFloat32) {
        const float* src = static_cast<const float*>(values.data());
        float* dst = static_cast<float*>(dest.mutableData());
        for (size_t i = 0; i < indices.size(); ++i) {
            dst[indices[i]] = src[i];
        }
    } else if (dest.dtype() == ScalarType::kFloat64) {
        const double* src = static_cast<const double*>(values.data());
        double* dst = static_cast<double*>(dest.mutableData());
        for (size_t i = 0; i < indices.size(); ++i) {
            dst[indices[i]] = src[i];
        }
    } else if (dest.dtype() == ScalarType::kInt32) {
        const int32_t* src = static_cast<const int32_t*>(values.data());
        int32_t* dst = static_cast<int32_t*>(dest.mutableData());
        for (size_t i = 0; i < indices.size(); ++i) {
            dst[indices[i]] = src[i];
        }
    }

    return {};
}

std::vector<int32_t> PatternApplicator::createGatherIndices(const Pattern& pattern,
                                                            size_t data_size) {
    auto indices = pattern.getSelectedIndices(data_size);
    std::vector<int32_t> gather_indices;
    gather_indices.reserve(indices.size());
    for (size_t idx : indices) {
        gather_indices.push_back(static_cast<int32_t>(idx));
    }
    return gather_indices;
}

std::vector<bool> PatternApplicator::createMask(const Pattern& pattern, size_t data_size) {
    std::vector<bool> mask(data_size, false);
    auto indices = pattern.getSelectedIndices(data_size);
    for (size_t idx : indices) {
        if (idx < data_size) {
            mask[idx] = true;
        }
    }
    return mask;
}

}  // namespace bud
