#pragma once

// =============================================================================
// Bud Flow Lang - Compute DAG
// =============================================================================
//
// Represents computation as a directed acyclic graph for analysis
// and schedule search. Used by the auto-scheduler to understand
// the structure of computations.
//
// Features:
// - Build DAG from IR
// - Analyze access patterns
// - Compute FLOPs and memory traffic
// - Support subgraph extraction
//

#include "bud_flow_lang/ir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bud {
namespace scheduler {

// =============================================================================
// DAG Node ID
// =============================================================================

/// Unique identifier for DAG nodes
class DAGNodeId {
  public:
    DAGNodeId() : value_(kInvalid) {}
    explicit DAGNodeId(uint32_t value) : value_(value) {}

    [[nodiscard]] uint32_t value() const { return value_; }
    [[nodiscard]] bool isValid() const { return value_ != kInvalid; }

    static DAGNodeId Invalid() { return DAGNodeId(kInvalid); }

    bool operator==(const DAGNodeId& other) const { return value_ == other.value_; }
    bool operator!=(const DAGNodeId& other) const { return !(*this == other); }

  private:
    static constexpr uint32_t kInvalid = UINT32_MAX;
    uint32_t value_;
};

}  // namespace scheduler
}  // namespace bud

// Hash specialization for DAGNodeId
namespace std {
template <>
struct hash<bud::scheduler::DAGNodeId> {
    size_t operator()(const bud::scheduler::DAGNodeId& id) const {
        return hash<uint32_t>()(id.value());
    }
};

// Hash specialization for ir::ValueId
template <>
struct hash<bud::ir::ValueId> {
    size_t operator()(const bud::ir::ValueId& id) const { return hash<uint32_t>()(id.id); }
};
}  // namespace std

namespace bud {
namespace scheduler {

// =============================================================================
// DAG Node
// =============================================================================

/// A node in the compute DAG
struct DAGNode {
    DAGNodeId id;
    ir::OpCode op_code = ir::OpCode::kConstantScalar;

    std::vector<DAGNodeId> inputs;   // Input dependencies
    std::vector<DAGNodeId> outputs;  // Nodes that depend on this

    // Size information
    size_t element_count = 1;
    size_t bytes_per_element = 4;  // Default to float32 size
    std::vector<size_t> shape;

    // Original IR value ID (for mapping back)
    ir::ValueId ir_value_id;
};

// =============================================================================
// Access Pattern
// =============================================================================

/// Describes memory access pattern for a node
struct AccessPattern {
    DAGNodeId node_id;
    size_t stride = 1;  // Stride in elements
    bool is_contiguous = true;
    bool is_broadcast = false;
    size_t access_count = 1;  // Number of accesses
};

// =============================================================================
// Memory Traffic
// =============================================================================

/// Memory traffic statistics
struct MemoryTraffic {
    size_t reads = 0;         // Bytes read
    size_t writes = 0;        // Bytes written
    size_t intermediate = 0;  // Intermediate buffer bytes
};

// =============================================================================
// DAG Analysis Result
// =============================================================================

/// Result of analyzing a DAG
struct DAGAnalysis {
    bool is_valid = false;
    size_t depth = 0;         // Longest path length
    size_t width = 0;         // Maximum parallel operations
    size_t flops = 0;         // Total floating-point operations
    size_t memory_bytes = 0;  // Memory footprint

    // Optimization opportunities
    bool has_fusion_opportunity = false;
    bool has_parallel_opportunity = false;
    bool has_vectorization_opportunity = false;

    std::vector<DAGNodeId> critical_path;
};

// =============================================================================
// ComputeDAG
// =============================================================================

/// Directed acyclic graph representing computation
class ComputeDAG {
  public:
    ComputeDAG() = default;

    /// Build DAG from IR
    [[nodiscard]] static ComputeDAG fromIR(const ir::IRBuilder& builder);

    // -------------------------------------------------------------------------
    // Properties
    // -------------------------------------------------------------------------

    /// Get number of nodes
    [[nodiscard]] size_t numNodes() const { return nodes_.size(); }

    /// Get number of edges
    [[nodiscard]] size_t numEdges() const { return edge_count_; }

    /// Get all nodes
    [[nodiscard]] const std::vector<DAGNode>& nodes() const { return nodes_; }

    /// Get root nodes (outputs)
    [[nodiscard]] std::vector<DAGNodeId> roots() const;

    /// Get a specific node
    [[nodiscard]] const DAGNode& getNode(DAGNodeId id) const;

    // -------------------------------------------------------------------------
    // Analysis
    // -------------------------------------------------------------------------

    /// Analyze the DAG for optimization opportunities
    [[nodiscard]] DAGAnalysis analyze() const;

    /// Get access patterns for all nodes
    [[nodiscard]] std::vector<AccessPattern> getAccessPatterns() const;

    /// Compute total FLOPs
    [[nodiscard]] size_t getFlops() const;

    /// Compute memory traffic
    [[nodiscard]] MemoryTraffic getMemoryTraffic() const;

    /// Get topologically sorted node IDs
    [[nodiscard]] std::vector<DAGNodeId> topologicalSort() const;

    // -------------------------------------------------------------------------
    // Subgraph Operations
    // -------------------------------------------------------------------------

    /// Extract a subgraph containing specified nodes
    [[nodiscard]] ComputeDAG extractSubgraph(const std::vector<DAGNodeId>& node_ids) const;

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------

    /// Serialize to string
    [[nodiscard]] std::string serialize() const;

    /// Deserialize from string
    bool deserialize(const std::string& data);

  private:
    std::vector<DAGNode> nodes_;
    size_t edge_count_ = 0;

    // Mapping from IR value ID to DAG node ID
    std::unordered_map<ir::ValueId, DAGNodeId> ir_to_dag_;

    /// Add a node to the DAG
    DAGNodeId addNode(const DAGNode& node);

    /// Add an edge between nodes
    void addEdge(DAGNodeId from, DAGNodeId to);

    /// Calculate FLOPs for an operation
    static size_t flopsForOp(ir::OpCode op, size_t element_count);
};

}  // namespace scheduler
}  // namespace bud
