#pragma once

// =============================================================================
// Bud Flow Lang - Schedule Primitives
// =============================================================================
//
// Schedule primitives for loop transformations, similar to Halide/TVM:
// - split: Split a loop dimension
// - tile: 2D cache blocking
// - fuse: Combine nested loops
// - reorder: Change loop order
// - vectorize: SIMD vectorization
// - parallel: Thread parallelization
// - unroll: Loop unrolling
// - compute_at: Producer placement
//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// Forward Declarations
// =============================================================================

class Schedule;
class Stage;

// =============================================================================
// Transform Types
// =============================================================================

/// Types of schedule transformations
enum class TransformType {
    kSplit,          // Split a loop into outer/inner
    kTile,           // 2D split for cache blocking
    kFuse,           // Combine nested loops
    kReorder,        // Change loop nesting order
    kVectorize,      // Apply SIMD vectorization
    kParallel,       // Parallelize across threads
    kUnroll,         // Unroll loop iterations
    kComputeAt,      // Control producer placement
    kComputeInline,  // Inline producer into consumer
    kCacheRead,      // Add read cache
    kCacheWrite      // Add write cache
};

// =============================================================================
// Cache Types
// =============================================================================

/// Types of caching
enum class CacheType {
    kGlobal,   // Global memory
    kShared,   // Shared/L2 cache
    kLocal,    // Local/L1 cache
    kRegister  // Register
};

// =============================================================================
// Var
// =============================================================================

/// Represents a loop variable (dimension)
class Var {
  public:
    Var() = default;
    explicit Var(const std::string& name) : name_(name), valid_(true) {}

    /// Get the variable name
    [[nodiscard]] const std::string& name() const { return name_; }

    /// Check if this is a valid variable
    [[nodiscard]] bool isValid() const { return valid_; }

    /// Equality comparison
    bool operator==(const Var& other) const { return name_ == other.name_; }
    bool operator!=(const Var& other) const { return !(*this == other); }

  private:
    std::string name_;
    bool valid_ = false;
};

// =============================================================================
// Stage
// =============================================================================

/// Represents a computation stage
class Stage {
  public:
    Stage() = default;
    explicit Stage(const std::string& name) : name_(name), valid_(true) {}

    /// Get the stage name
    [[nodiscard]] const std::string& name() const { return name_; }

    /// Check if this is a valid stage
    [[nodiscard]] bool isValid() const { return valid_; }

  private:
    std::string name_;
    bool valid_ = false;
};

// =============================================================================
// Transform
// =============================================================================

/// Represents a single schedule transformation
struct Transform {
    TransformType type;

    // Variables involved
    std::vector<Var> vars;

    // For split/unroll
    size_t factor = 0;

    // For tile
    size_t tile_x = 0;
    size_t tile_y = 0;

    // For vectorize
    size_t vector_width = 0;

    // For compute_at
    std::string producer_stage;
    std::string consumer_stage;

    // For cache operations
    CacheType cache_type = CacheType::kLocal;
};

// =============================================================================
// Schedule
// =============================================================================

/// Schedule containing a sequence of transformations
class Schedule {
  public:
    Schedule() = default;

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    /// Check if schedule is empty
    [[nodiscard]] bool isEmpty() const { return transforms_.empty(); }

    /// Get number of transformations
    [[nodiscard]] size_t numTransforms() const { return transforms_.size(); }

    /// Get all transformations
    [[nodiscard]] const std::vector<Transform>& transforms() const { return transforms_; }

    // -------------------------------------------------------------------------
    // Loop Splitting
    // -------------------------------------------------------------------------

    /// Split a loop into outer and inner loops
    Schedule& split(const Var& var, size_t factor, Var& outer, Var& inner);

    /// Tile two dimensions (2D split)
    Schedule& tile(const Var& x, const Var& y, size_t tile_x, size_t tile_y, Var& x_outer,
                   Var& x_inner, Var& y_outer, Var& y_inner);

    // -------------------------------------------------------------------------
    // Loop Manipulation
    // -------------------------------------------------------------------------

    /// Fuse two loops into one
    Schedule& fuse(const Var& outer, const Var& inner, Var& fused);

    /// Reorder loops
    Schedule& reorder(const std::vector<Var>& new_order);

    // -------------------------------------------------------------------------
    // Parallelization
    // -------------------------------------------------------------------------

    /// Vectorize a loop dimension
    Schedule& vectorize(const Var& var, size_t width = 0);

    /// Parallelize a loop dimension
    Schedule& parallel(const Var& var);

    /// Unroll a loop
    Schedule& unroll(const Var& var, size_t factor = 0);

    // -------------------------------------------------------------------------
    // Compute Placement
    // -------------------------------------------------------------------------

    /// Place producer computation at consumer's var
    Schedule& computeAt(const Stage& producer, const Stage& consumer, const Var& var);

    /// Inline producer into consumer
    Schedule& computeInline(const Stage& producer);

    // -------------------------------------------------------------------------
    // Caching
    // -------------------------------------------------------------------------

    /// Add read cache
    Schedule& cacheRead(const Stage& stage, CacheType type);

    /// Add write cache
    Schedule& cacheWrite(const Stage& stage, CacheType type);

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /// Clone this schedule
    [[nodiscard]] std::unique_ptr<Schedule> clone() const;

    /// Serialize to string
    [[nodiscard]] std::string serialize() const;

    /// Deserialize from string
    bool deserialize(const std::string& data);

    /// Validate the schedule
    [[nodiscard]] bool validate() const;

    /// Clear all transformations
    void clear() { transforms_.clear(); }

  private:
    std::vector<Transform> transforms_;

    // Counter for generating unique variable names
    size_t var_counter_ = 0;

    /// Generate a unique variable name
    std::string generateVarName(const std::string& base, const std::string& suffix);
};

}  // namespace scheduler
}  // namespace bud
