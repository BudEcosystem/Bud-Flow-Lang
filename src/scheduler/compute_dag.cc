// =============================================================================
// Bud Flow Lang - Compute DAG Implementation
// =============================================================================

#include "bud_flow_lang/scheduler/compute_dag.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <queue>
#include <stack>

namespace bud {
namespace scheduler {

// =============================================================================
// ComputeDAG Implementation
// =============================================================================

ComputeDAG ComputeDAG::fromIR(const ir::IRBuilder& builder) {
    ComputeDAG dag;

    // First pass: create nodes for all IR values
    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        DAGNode dag_node;
        dag_node.ir_value_id = node->id();
        dag_node.op_code = node->opCode();
        dag_node.element_count = 1;      // Scalar by default
        dag_node.bytes_per_element = 4;  // Default to float32
        dag_node.shape = {1};

        DAGNodeId id = dag.addNode(dag_node);
        dag.ir_to_dag_[node->id()] = id;
    }

    // Second pass: create edges based on operands
    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead()) {
            continue;
        }

        auto dag_node_id_it = dag.ir_to_dag_.find(node->id());
        if (dag_node_id_it == dag.ir_to_dag_.end()) {
            continue;
        }

        DAGNodeId dag_node_id = dag_node_id_it->second;

        // Add edges from operands to this node
        for (size_t i = 0; i < node->numOperands(); ++i) {
            ir::ValueId operand_id = node->operand(i);
            auto it = dag.ir_to_dag_.find(operand_id);
            if (it != dag.ir_to_dag_.end()) {
                dag.addEdge(it->second, dag_node_id);
            }
        }
    }

    spdlog::debug("Built DAG with {} nodes, {} edges", dag.numNodes(), dag.numEdges());

    return dag;
}

DAGNodeId ComputeDAG::addNode(const DAGNode& node) {
    DAGNodeId id(static_cast<uint32_t>(nodes_.size()));
    DAGNode new_node = node;
    new_node.id = id;
    nodes_.push_back(new_node);
    return id;
}

void ComputeDAG::addEdge(DAGNodeId from, DAGNodeId to) {
    if (!from.isValid() || !to.isValid()) {
        return;
    }

    uint32_t from_idx = from.value();
    uint32_t to_idx = to.value();

    if (from_idx >= nodes_.size() || to_idx >= nodes_.size()) {
        return;
    }

    // Add edge
    nodes_[from_idx].outputs.push_back(to);
    nodes_[to_idx].inputs.push_back(from);
    ++edge_count_;
}

std::vector<DAGNodeId> ComputeDAG::roots() const {
    std::vector<DAGNodeId> result;
    for (const auto& node : nodes_) {
        if (node.outputs.empty()) {
            result.push_back(node.id);
        }
    }
    return result;
}

const DAGNode& ComputeDAG::getNode(DAGNodeId id) const {
    static DAGNode invalid;
    if (!id.isValid() || id.value() >= nodes_.size()) {
        return invalid;
    }
    return nodes_[id.value()];
}

DAGAnalysis ComputeDAG::analyze() const {
    DAGAnalysis analysis;
    analysis.is_valid = !nodes_.empty();

    if (!analysis.is_valid) {
        return analysis;
    }

    // Calculate depth using topological sort with levels
    std::unordered_map<DAGNodeId, size_t> levels;
    size_t max_depth = 0;
    size_t max_width = 0;

    // BFS to assign levels
    std::queue<DAGNodeId> queue;

    // Find leaf nodes (no inputs)
    for (const auto& node : nodes_) {
        if (node.inputs.empty()) {
            queue.push(node.id);
            levels[node.id] = 0;
        }
    }

    // Count nodes at each level for width calculation
    std::unordered_map<size_t, size_t> level_counts;

    while (!queue.empty()) {
        DAGNodeId current_id = queue.front();
        queue.pop();

        const auto& node = getNode(current_id);
        size_t current_level = levels[current_id];
        max_depth = std::max(max_depth, current_level);
        level_counts[current_level]++;

        for (const auto& output_id : node.outputs) {
            // Calculate level as max of all input levels + 1
            size_t new_level = current_level + 1;
            if (levels.count(output_id) == 0 || levels[output_id] < new_level) {
                levels[output_id] = new_level;
                queue.push(output_id);
            }
        }
    }

    // Find max width
    for (const auto& [level, count] : level_counts) {
        max_width = std::max(max_width, count);
    }

    analysis.depth = max_depth + 1;
    analysis.width = max_width;
    analysis.flops = getFlops();

    // Identify optimization opportunities
    analysis.has_fusion_opportunity = (edge_count_ > 0);
    analysis.has_parallel_opportunity = (max_width > 1);
    analysis.has_vectorization_opportunity = (analysis.flops > 0);

    // Find critical path (longest path)
    // Use the nodes with maximum level
    for (const auto& [id, level] : levels) {
        if (level == max_depth) {
            analysis.critical_path.push_back(id);
        }
    }

    return analysis;
}

std::vector<AccessPattern> ComputeDAG::getAccessPatterns() const {
    std::vector<AccessPattern> patterns;

    for (const auto& node : nodes_) {
        AccessPattern pattern;
        pattern.node_id = node.id;
        pattern.stride = 1;
        pattern.is_contiguous = true;
        pattern.is_broadcast = (node.op_code == ir::OpCode::kBroadcast);
        pattern.access_count = node.element_count;

        patterns.push_back(pattern);
    }

    return patterns;
}

size_t ComputeDAG::getFlops() const {
    size_t total = 0;
    for (const auto& node : nodes_) {
        total += flopsForOp(node.op_code, node.element_count);
    }
    return total;
}

size_t ComputeDAG::flopsForOp(ir::OpCode op, size_t element_count) {
    switch (op) {
    case ir::OpCode::kAdd:
    case ir::OpCode::kSub:
    case ir::OpCode::kMul:
    case ir::OpCode::kNeg:
    case ir::OpCode::kAbs:
        return element_count;

    case ir::OpCode::kDiv:
    case ir::OpCode::kSqrt:
        return element_count * 4;  // More expensive

    case ir::OpCode::kFma:
        return element_count * 2;  // Mul + Add

    case ir::OpCode::kExp:
    case ir::OpCode::kLog:
    case ir::OpCode::kSin:
    case ir::OpCode::kCos:
        return element_count * 10;  // Transcendentals are expensive

    case ir::OpCode::kReduceSum:
    case ir::OpCode::kReduceMin:
    case ir::OpCode::kReduceMax:
    case ir::OpCode::kMin:
    case ir::OpCode::kMax:
        return element_count;  // One op per element

    default:
        return 0;  // No FLOPs (constants, loads, etc.)
    }
}

MemoryTraffic ComputeDAG::getMemoryTraffic() const {
    MemoryTraffic traffic;

    for (const auto& node : nodes_) {
        size_t bytes = node.element_count * node.bytes_per_element;

        // Each node reads from its inputs
        traffic.reads += node.inputs.size() * bytes;

        // Each node writes to its outputs (or is consumed)
        if (!node.outputs.empty()) {
            traffic.intermediate += bytes;
        } else {
            traffic.writes += bytes;
        }
    }

    return traffic;
}

std::vector<DAGNodeId> ComputeDAG::topologicalSort() const {
    std::vector<DAGNodeId> result;
    result.reserve(nodes_.size());

    // Kahn's algorithm
    std::unordered_map<DAGNodeId, size_t> in_degree;
    for (const auto& node : nodes_) {
        in_degree[node.id] = node.inputs.size();
    }

    std::queue<DAGNodeId> queue;
    for (const auto& node : nodes_) {
        if (in_degree[node.id] == 0) {
            queue.push(node.id);
        }
    }

    while (!queue.empty()) {
        DAGNodeId current = queue.front();
        queue.pop();
        result.push_back(current);

        const auto& node = getNode(current);
        for (const auto& output_id : node.outputs) {
            if (--in_degree[output_id] == 0) {
                queue.push(output_id);
            }
        }
    }

    return result;
}

ComputeDAG ComputeDAG::extractSubgraph(const std::vector<DAGNodeId>& node_ids) const {
    ComputeDAG subgraph;

    std::unordered_set<DAGNodeId> included(node_ids.begin(), node_ids.end());
    std::unordered_map<DAGNodeId, DAGNodeId> old_to_new;

    // Add nodes
    for (const auto& old_id : node_ids) {
        if (!old_id.isValid() || old_id.value() >= nodes_.size()) {
            continue;
        }

        const auto& old_node = nodes_[old_id.value()];
        DAGNode new_node = old_node;
        new_node.inputs.clear();
        new_node.outputs.clear();

        DAGNodeId new_id = subgraph.addNode(new_node);
        old_to_new[old_id] = new_id;
    }

    // Add edges (only between included nodes)
    for (const auto& old_id : node_ids) {
        if (!old_id.isValid() || old_id.value() >= nodes_.size()) {
            continue;
        }

        const auto& old_node = nodes_[old_id.value()];
        DAGNodeId new_id = old_to_new[old_id];

        for (const auto& input_id : old_node.inputs) {
            if (included.count(input_id) > 0) {
                subgraph.addEdge(old_to_new[input_id], new_id);
            }
        }
    }

    return subgraph;
}

std::string ComputeDAG::serialize() const {
    nlohmann::json j;

    j["edge_count"] = edge_count_;

    nlohmann::json nodes_array = nlohmann::json::array();
    for (const auto& node : nodes_) {
        nlohmann::json nj;
        nj["id"] = node.id.value();
        nj["op_code"] = static_cast<int>(node.op_code);
        nj["element_count"] = node.element_count;
        nj["bytes_per_element"] = node.bytes_per_element;
        nj["ir_value_id"] = node.ir_value_id.id;

        nlohmann::json inputs_array = nlohmann::json::array();
        for (const auto& input : node.inputs) {
            inputs_array.push_back(input.value());
        }
        nj["inputs"] = inputs_array;

        nlohmann::json outputs_array = nlohmann::json::array();
        for (const auto& output : node.outputs) {
            outputs_array.push_back(output.value());
        }
        nj["outputs"] = outputs_array;

        nodes_array.push_back(nj);
    }
    j["nodes"] = nodes_array;

    return j.dump();
}

bool ComputeDAG::deserialize(const std::string& data) {
    try {
        auto j = nlohmann::json::parse(data);

        edge_count_ = j["edge_count"].get<size_t>();
        nodes_.clear();
        ir_to_dag_.clear();

        for (const auto& nj : j["nodes"]) {
            DAGNode node;
            node.id = DAGNodeId(nj["id"].get<uint32_t>());
            node.op_code = static_cast<ir::OpCode>(nj["op_code"].get<int>());
            node.element_count = nj["element_count"].get<size_t>();
            node.bytes_per_element = nj["bytes_per_element"].get<size_t>();
            node.ir_value_id = ir::ValueId{nj["ir_value_id"].get<uint32_t>()};

            for (const auto& input : nj["inputs"]) {
                node.inputs.push_back(DAGNodeId(input.get<uint32_t>()));
            }

            for (const auto& output : nj["outputs"]) {
                node.outputs.push_back(DAGNodeId(output.get<uint32_t>()));
            }

            nodes_.push_back(node);
            ir_to_dag_[node.ir_value_id] = node.id;
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to deserialize DAG: {}", e.what());
        return false;
    }
}

}  // namespace scheduler
}  // namespace bud
