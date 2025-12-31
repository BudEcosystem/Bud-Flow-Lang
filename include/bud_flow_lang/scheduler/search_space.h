#pragma once

// =============================================================================
// Bud Flow Lang - Search Space
// =============================================================================
//
// Hierarchical search space for auto-scheduling. Inspired by TVM's Ansor.
//
// The search space consists of:
// 1. Sketches: High-level loop structures (e.g., tiled, fused, parallel)
// 2. Annotations: Specific parameters (tile sizes, unroll factors, etc.)
//
// Features:
// - Generate sketches from compute DAG
// - Annotate sketches with parameters
// - Sample from search space
// - Iterative refinement around promising schedules
//

#include "bud_flow_lang/scheduler/compute_dag.h"
#include "bud_flow_lang/scheduler/schedule.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// Transform Step
// =============================================================================

/// Types of transform steps in the search space
enum class TransformStep : uint8_t {
    kNone = 0,
    kSplit = 1,
    kTile = 2,
    kFuse = 3,
    kReorder = 4,
    kVectorize = 5,
    kParallel = 6,
    kUnroll = 7,
    kComputeAt = 8,
    kCacheRead = 9,
    kCacheWrite = 10,
};

/// Definition of a transform step with parameters
struct TransformStepDef {
    TransformStep type = TransformStep::kNone;
    std::string target_var;     // Variable to transform
    std::string secondary_var;  // For fuse, tile, etc.
    size_t factor = 1;          // Split factor, unroll factor
    size_t tile_x = 1;          // Tile size X
    size_t tile_y = 1;          // Tile size Y
    size_t vector_width = 0;    // For vectorize (0 = auto)
    std::string stage_name;     // For compute_at

    std::string serialize() const;
    bool deserialize(const std::string& data);
};

// =============================================================================
// Sketch Types
// =============================================================================

/// High-level sketch types
enum class SketchType : uint8_t {
    kInvalid = 0,
    kBasic = 1,       // No transformations
    kTiled = 2,       // Loop tiling applied
    kFused = 3,       // Operations fused
    kVectorized = 4,  // Vectorization applied
    kParallel = 5,    // Parallelization applied
    kCombined = 6,    // Multiple transformations
};

/// A sketch representing a high-level loop structure
struct Sketch {
    SketchType type = SketchType::kInvalid;
    std::string name;
    std::vector<TransformStepDef> steps;
    std::vector<DAGNodeId> affected_nodes;

    std::string serialize() const;
    bool deserialize(const std::string& data);
};

// =============================================================================
// Search Space Bounds
// =============================================================================

/// Bounds for search space parameters
struct SearchSpaceBounds {
    // Split/tile factors (powers of 2 by default)
    size_t min_split_factor = 2;
    size_t max_split_factor = 128;
    size_t min_tile_size = 4;
    size_t max_tile_size = 64;

    // Unroll bounds
    size_t max_unroll_factor = 16;

    // Vector widths to consider
    std::vector<size_t> vector_widths = {4, 8, 16};

    // Number of parallel levels
    size_t max_parallel_levels = 2;

    // Compute-at depth
    size_t max_compute_at_depth = 3;
};

// =============================================================================
// SearchSpace
// =============================================================================

/// Search space for schedule exploration
class SearchSpace {
  public:
    SearchSpace() = default;

    /// Build search space from compute DAG
    [[nodiscard]] static SearchSpace fromDAG(const ComputeDAG& dag,
                                             const SearchSpaceBounds& bounds = SearchSpaceBounds{});

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    /// Get number of sketches
    [[nodiscard]] size_t numSketches() const { return sketches_.size(); }

    /// Get bounds
    [[nodiscard]] const SearchSpaceBounds& bounds() const { return bounds_; }

    /// Get all sketches
    [[nodiscard]] const std::vector<Sketch>& getSketches() const { return sketches_; }

    /// Get sketches of a specific type
    [[nodiscard]] std::vector<Sketch> getSketchesOfType(SketchType type) const;

    // -------------------------------------------------------------------------
    // Annotation
    // -------------------------------------------------------------------------

    /// Annotate a sketch with specific parameters
    [[nodiscard]] std::vector<Schedule> annotateSketch(const Sketch& sketch) const;

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    /// Sample schedules from the search space
    [[nodiscard]] std::vector<Schedule> sample(size_t count, uint64_t seed = 0) const;

    /// Sample unique schedules
    [[nodiscard]] std::vector<Schedule> sampleUnique(size_t count, uint64_t seed = 0) const;

    /// Refine search around a given schedule
    [[nodiscard]] std::vector<Schedule> refineAround(const Schedule& base,
                                                     size_t num_neighbors) const;

    // -------------------------------------------------------------------------
    // Enumeration
    // -------------------------------------------------------------------------

    /// Estimate total size of search space
    [[nodiscard]] size_t estimateSize() const;

    /// Enumerate all schedules (up to limit)
    [[nodiscard]] std::vector<Schedule> enumerateAll(size_t limit = 1000) const;

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    /// Serialize to string
    [[nodiscard]] std::string serialize() const;

    /// Deserialize from string
    bool deserialize(const std::string& data);

  private:
    std::vector<Sketch> sketches_;
    SearchSpaceBounds bounds_;
    const ComputeDAG* dag_ = nullptr;  // Non-owning reference

    // Sketch generation
    void generateBasicSketch();
    void generateTiledSketches();
    void generateFusedSketches();
    void generateVectorizedSketches();
    void generateParallelSketches();
    void generateCombinedSketches();

    // Helper to generate split factors
    [[nodiscard]] std::vector<size_t> getSplitFactors() const;

    // Helper to generate tile sizes
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> getTileSizes() const;

    // Convert sketch to schedule
    [[nodiscard]] Schedule sketchToSchedule(const Sketch& sketch) const;

    // Mutate a schedule for refinement
    [[nodiscard]] Schedule mutateSchedule(const Schedule& base) const;
};

}  // namespace scheduler
}  // namespace bud
