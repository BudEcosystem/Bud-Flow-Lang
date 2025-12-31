// =============================================================================
// Bud Flow Lang - Pattern Class (Developer Tier)
// =============================================================================
//
// Pattern provides semantic abstractions for selecting and filtering data:
//
// - Pattern.stride(n) - Select every nth element
// - Pattern.where(predicate) - Select elements matching a condition
// - Pattern.block(size) - Divide data into fixed-size blocks/tiles
// - Pattern.indices(list) - Select specific indices
//
// Patterns compile to efficient SIMD operations:
// - Strided patterns use GatherIndex
// - Conditional patterns use masked loads/stores
// - Block patterns enable cache-friendly tiling
//
// Usage:
//   from flow import Pattern, Bunch
//
//   # Select every other element
//   every_other = Pattern.stride(2)
//   result = data.select(every_other)
//
//   # Select elements greater than threshold
//   high_values = Pattern.where(lambda x: x > threshold)
//   result = data.select(high_values)
//
//   # Process in 64-element blocks
//   for block in data.iterate(Pattern.block(64)):
//       process(block)
//
// =============================================================================

#pragma once

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/type_system.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace bud {

// Forward declarations
class Bunch;

// =============================================================================
// PatternType - Enumeration of pattern kinds
// =============================================================================

enum class PatternType {
    kStride,   // Every nth element: 0, n, 2n, 3n, ...
    kWhere,    // Elements matching predicate
    kBlock,    // Fixed-size contiguous blocks
    kIndices,  // Specific index list
    kRange,    // Range [start, stop) with step
    kMask,     // Boolean mask array
};

// =============================================================================
// Pattern - User-facing pattern abstraction
// =============================================================================

class Pattern {
  public:
    // =========================================================================
    // Factory Methods
    // =========================================================================

    // Stride pattern: select every nth element
    // stride(2) selects indices 0, 2, 4, 6, ...
    [[nodiscard]] static Pattern stride(size_t n);

    // Stride with offset: select every nth element starting from offset
    // stride(2, 1) selects indices 1, 3, 5, 7, ...
    [[nodiscard]] static Pattern stride(size_t n, size_t offset);

    // Block pattern: divide into fixed-size contiguous blocks
    // block(64) creates 64-element blocks for tiled processing
    [[nodiscard]] static Pattern block(size_t size);

    // Block with alignment hint: blocks aligned to cache lines
    [[nodiscard]] static Pattern block(size_t size, size_t alignment);

    // Indices pattern: select specific indices
    [[nodiscard]] static Pattern indices(std::vector<size_t> idx);

    // Range pattern: [start, stop) with optional step
    [[nodiscard]] static Pattern range(size_t start, size_t stop, size_t step = 1);

    // Mask pattern: boolean mask array
    [[nodiscard]] static Pattern mask(std::vector<bool> mask_values);

    // =========================================================================
    // Pattern Properties
    // =========================================================================

    [[nodiscard]] PatternType type() const { return type_; }

    // Get stride (for stride patterns)
    [[nodiscard]] size_t strideValue() const { return stride_; }

    // Get offset (for stride patterns)
    [[nodiscard]] size_t offset() const { return offset_; }

    // Get block size (for block patterns)
    [[nodiscard]] size_t blockSize() const { return block_size_; }

    // Get alignment (for block patterns)
    [[nodiscard]] size_t alignment() const { return alignment_; }

    // Get indices (for indices patterns)
    [[nodiscard]] const std::vector<size_t>& indexList() const { return indices_; }

    // Get mask (for mask patterns)
    [[nodiscard]] const std::vector<bool>& maskValues() const { return mask_values_; }

    // Get range [start, stop, step]
    [[nodiscard]] size_t rangeStart() const { return range_start_; }
    [[nodiscard]] size_t rangeStop() const { return range_stop_; }
    [[nodiscard]] size_t rangeStep() const { return range_step_; }

    // =========================================================================
    // Pattern Operations
    // =========================================================================

    // Count number of elements selected from a data size
    [[nodiscard]] size_t countSelected(size_t data_size) const;

    // Get the selected indices for a given data size
    [[nodiscard]] std::vector<size_t> getSelectedIndices(size_t data_size) const;

    // Check if pattern requires gather/scatter (non-contiguous access)
    [[nodiscard]] bool requiresGather() const;

    // Check if pattern is regular (predictable stride)
    [[nodiscard]] bool isRegular() const;

    // =========================================================================
    // Combine Patterns
    // =========================================================================

    // Compose patterns: apply this pattern, then other pattern
    [[nodiscard]] Pattern compose(const Pattern& other) const;

    // Intersect patterns: elements selected by both
    [[nodiscard]] Pattern intersect(const Pattern& other) const;

    // Union patterns: elements selected by either
    [[nodiscard]] Pattern unite(const Pattern& other) const;

    // =========================================================================
    // String Representation
    // =========================================================================

    [[nodiscard]] std::string toString() const;

  private:
    PatternType type_ = PatternType::kStride;
    size_t stride_ = 1;
    size_t offset_ = 0;
    size_t block_size_ = 64;
    size_t alignment_ = 64;
    size_t range_start_ = 0;
    size_t range_stop_ = 0;
    size_t range_step_ = 1;
    std::vector<size_t> indices_;
    std::vector<bool> mask_values_;
};

// =============================================================================
// BlockIterator - Iterator over blocks for tiled processing
// =============================================================================

class BlockIterator {
  public:
    struct Block {
        size_t start;  // Start index in original data
        size_t end;    // End index (exclusive)
        size_t size;   // Number of elements in block
        bool is_last;  // True if this is the last (possibly partial) block
    };

    BlockIterator(size_t data_size, size_t block_size, size_t alignment = 64);

    // Iterator interface
    [[nodiscard]] bool hasNext() const { return current_start_ < data_size_; }
    [[nodiscard]] Block next();
    void reset() { current_start_ = 0; }

    // Get total number of blocks
    [[nodiscard]] size_t numBlocks() const;

    // Get block at index
    [[nodiscard]] Block blockAt(size_t index) const;

  private:
    size_t data_size_;
    size_t block_size_;
    size_t alignment_;
    size_t current_start_ = 0;
};

// =============================================================================
// PatternApplicator - Applies patterns to Bunch data
// =============================================================================

class PatternApplicator {
  public:
    // Apply pattern to select elements from a Bunch
    [[nodiscard]] static Result<Bunch> select(const Bunch& data, const Pattern& pattern);

    // Apply pattern to scatter values back to a Bunch
    [[nodiscard]] static Result<void> scatter(Bunch& dest, const Bunch& values,
                                              const Pattern& pattern);

    // Create gather indices for SIMD operations
    [[nodiscard]] static std::vector<int32_t> createGatherIndices(const Pattern& pattern,
                                                                  size_t data_size);

    // Create mask for SIMD operations
    [[nodiscard]] static std::vector<bool> createMask(const Pattern& pattern, size_t data_size);
};

}  // namespace bud
