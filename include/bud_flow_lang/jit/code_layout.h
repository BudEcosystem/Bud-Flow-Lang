#pragma once

// =============================================================================
// Bud Flow Lang - Hot/Cold Code Layout Optimizer
// =============================================================================
//
// Separates code into hot/cold regions for improved I-cache utilization:
// - Places frequently executed code contiguously
// - Moves error handling and cold paths to separate region
// - Optimizes branch layout for fall-through on common path
//
// Benefits:
// - Better instruction cache utilization
// - Reduced branch mispredictions
// - Smaller hot code footprint
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/pgo_specializer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace bud {
namespace jit {

// =============================================================================
// Region Types
// =============================================================================

/// Classification of code regions
enum class RegionType {
    kHot,     // Frequently executed code
    kCold,    // Rarely executed code (error handling, etc.)
    kGeneric  // Default/unclassified code
};

// =============================================================================
// Alignment Targets
// =============================================================================

/// Targets that may need specific alignment
enum class AlignmentTarget {
    kFunctionEntry,  // Function entry points
    kLoopHeader,     // Loop entry points
    kJumpTarget,     // Branch targets
    kBasicBlock      // Basic block starts
};

// =============================================================================
// CodeRegion
// =============================================================================

/// A region of generated code
class CodeRegion {
  public:
    CodeRegion() = default;
    explicit CodeRegion(RegionType type) : type_(type) {}

    /// Get the region type
    [[nodiscard]] RegionType type() const { return type_; }

    /// Check if region is empty
    [[nodiscard]] bool isEmpty() const { return code_.empty(); }

    /// Get the size of code in this region
    [[nodiscard]] size_t size() const { return code_.size(); }

    /// Get pointer to code data
    [[nodiscard]] const uint8_t* data() const { return code_.data(); }

    /// Append code to this region
    void appendCode(const uint8_t* data, size_t size) {
        code_.insert(code_.end(), data, data + size);
    }

    /// Align the region to a boundary (pads with NOPs)
    void alignTo(size_t alignment) {
        if (alignment == 0)
            return;
        size_t remainder = code_.size() % alignment;
        if (remainder != 0) {
            size_t padding = alignment - remainder;
            // Pad with NOP instructions (0x90 on x86)
            code_.insert(code_.end(), padding, 0x90);
        }
    }

    /// Clear the region
    void clear() { code_.clear(); }

  private:
    RegionType type_ = RegionType::kGeneric;
    std::vector<uint8_t> code_;
};

// =============================================================================
// FunctionProfile
// =============================================================================

/// Profile information for a function
struct FunctionProfile {
    std::string name;
    uint64_t execution_count = 0;
    bool is_critical_path = false;
    float avg_time_ns = 0.0f;
};

// =============================================================================
// BranchProfile
// =============================================================================

/// Profile information for a branch
struct BranchProfile {
    uint64_t taken_count = 0;
    uint64_t not_taken_count = 0;
};

// =============================================================================
// LayoutPlan
// =============================================================================

/// Plan for organizing code layout
struct LayoutPlan {
    std::vector<std::string> hot_functions;      // Hot functions (ordered by frequency)
    std::vector<std::string> cold_functions;     // Cold functions
    std::vector<std::string> generic_functions;  // Other functions
};

// =============================================================================
// LayoutResult
// =============================================================================

/// Result of applying layout optimization
struct LayoutResult {
    bool optimized = false;
    size_t hot_region_size = 0;
    size_t cold_region_size = 0;
    size_t functions_reordered = 0;
};

// =============================================================================
// MemoryFootprint
// =============================================================================

/// Memory footprint breakdown
struct MemoryFootprint {
    size_t hot_size = 0;
    size_t cold_size = 0;
    size_t generic_size = 0;
    size_t total_size = 0;
};

// =============================================================================
// CodeLayoutOptimizer
// =============================================================================

/// Optimizer for hot/cold code separation
class CodeLayoutOptimizer {
  public:
    CodeLayoutOptimizer();

    // -------------------------------------------------------------------------
    // Enable/Disable
    // -------------------------------------------------------------------------

    /// Check if optimization is enabled
    [[nodiscard]] bool isEnabled() const { return enabled_; }

    /// Enable optimization
    void enable() { enabled_ = true; }

    /// Disable optimization
    void disable() { enabled_ = false; }

    // -------------------------------------------------------------------------
    // Classification
    // -------------------------------------------------------------------------

    /// Classify a function based on its profile
    [[nodiscard]] RegionType classifyFunction(const FunctionProfile& profile) const;

    /// Get the hot threshold (execution count)
    [[nodiscard]] uint64_t hotThreshold() const { return hot_threshold_; }

    /// Set the hot threshold
    void setHotThreshold(uint64_t threshold) { hot_threshold_ = threshold; }

    /// Get the cold threshold (execution count)
    [[nodiscard]] uint64_t coldThreshold() const { return cold_threshold_; }

    /// Set the cold threshold
    void setColdThreshold(uint64_t threshold) { cold_threshold_ = threshold; }

    // -------------------------------------------------------------------------
    // Layout Planning
    // -------------------------------------------------------------------------

    /// Create a layout plan from function profiles
    [[nodiscard]] LayoutPlan createLayoutPlan(const std::vector<FunctionProfile>& profiles) const;

    // -------------------------------------------------------------------------
    // Branch Optimization
    // -------------------------------------------------------------------------

    /// Determine if a branch should be inverted for fall-through optimization
    [[nodiscard]] bool shouldInvertBranch(const BranchProfile& profile) const;

    // -------------------------------------------------------------------------
    // Alignment
    // -------------------------------------------------------------------------

    /// Get recommended alignment for a target type
    [[nodiscard]] size_t recommendedAlignment(AlignmentTarget target) const;

    // -------------------------------------------------------------------------
    // IR Optimization
    // -------------------------------------------------------------------------

    /// Apply layout optimization to IR
    [[nodiscard]] std::optional<LayoutResult> optimizeLayout(const ir::IRBuilder& builder,
                                                             const ProfileData& profile);

    // -------------------------------------------------------------------------
    // Memory Analysis
    // -------------------------------------------------------------------------

    /// Calculate memory footprint from regions
    [[nodiscard]] MemoryFootprint calculateFootprint(const std::vector<CodeRegion>& regions) const;

  private:
    bool enabled_ = true;

    // Classification thresholds
    uint64_t hot_threshold_ = 1000;  // Execution count for hot
    uint64_t cold_threshold_ = 10;   // Execution count for cold

    // Alignment preferences
    static constexpr size_t kFunctionAlignment = 16;
    static constexpr size_t kLoopAlignment = 32;
    static constexpr size_t kJumpTargetAlignment = 8;
    static constexpr size_t kBasicBlockAlignment = 4;
};

}  // namespace jit
}  // namespace bud
