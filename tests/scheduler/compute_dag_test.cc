// =============================================================================
// Bud Flow Lang - Compute DAG Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for ComputeDAG which represents computation as a directed acyclic graph
// for analysis and schedule search.
//
// Features:
// - Build DAG from IR
// - Analyze access patterns
// - Identify optimization opportunities
// - Compute FLOPs and memory traffic
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/compute_dag.h"

#include <chrono>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// ComputeDAG Construction Tests
// =============================================================================

class ComputeDAGTest : public ::testing::Test {
  protected:
    void SetUp() override {}

    ir::IRModule createSimpleIR() {
        ir::IRModule module("simple");
        auto& builder = module.builder();
        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.add(a, b);
        module.setOutput(c);
        return module;
    }

    ir::IRModule createComplexIR() {
        ir::IRModule module("complex");
        auto& builder = module.builder();
        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto c = builder.constant(3.0f);

        auto x = builder.add(a, b);
        auto y = builder.mul(x, c);
        auto z = builder.sub(y, a);
        auto w = builder.div(z, b);
        module.setOutput(w);
        return module;
    }
};

TEST_F(ComputeDAGTest, DefaultConstruction) {
    ComputeDAG dag;
    EXPECT_EQ(dag.numNodes(), 0);
    EXPECT_EQ(dag.numEdges(), 0);
}

TEST_F(ComputeDAGTest, FromIR_Simple) {
    auto module = createSimpleIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    EXPECT_GT(dag.numNodes(), 0);
}

TEST_F(ComputeDAGTest, FromIR_Complex) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    // Complex IR has 4 operations: add, mul, sub, div
    EXPECT_GE(dag.numNodes(), 4);
}

TEST_F(ComputeDAGTest, HasRootNode) {
    auto module = createSimpleIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    // Should have at least one root (output)
    auto roots = dag.roots();
    EXPECT_FALSE(roots.empty());
}

// =============================================================================
// Node Analysis Tests
// =============================================================================

TEST_F(ComputeDAGTest, NodeOperations) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto nodes = dag.nodes();
    EXPECT_FALSE(nodes.empty());

    // Each node should have valid properties
    for (const auto& node : nodes) {
        EXPECT_TRUE(node.id != DAGNodeId::Invalid());
    }
}

TEST_F(ComputeDAGTest, NodeInputsOutputs) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    // At least one node should have inputs
    bool found_node_with_inputs = false;
    for (const auto& node : dag.nodes()) {
        if (!node.inputs.empty()) {
            found_node_with_inputs = true;
            break;
        }
    }
    EXPECT_TRUE(found_node_with_inputs);
}

// =============================================================================
// Access Pattern Tests
// =============================================================================

TEST_F(ComputeDAGTest, GetAccessPatterns) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto patterns = dag.getAccessPatterns();

    // Should have some access patterns
    EXPECT_FALSE(patterns.empty());
}

TEST_F(ComputeDAGTest, AccessPatternHasStride) {
    auto module = createSimpleIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto patterns = dag.getAccessPatterns();
    for (const auto& pattern : patterns) {
        // Stride should be defined
        EXPECT_GE(pattern.stride, 0);
    }
}

// =============================================================================
// FLOP Analysis Tests
// =============================================================================

TEST_F(ComputeDAGTest, GetFlops_Simple) {
    auto module = createSimpleIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    size_t flops = dag.getFlops();
    // Simple add = 1 FLOP
    EXPECT_GT(flops, 0);
}

TEST_F(ComputeDAGTest, GetFlops_Complex) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    size_t flops = dag.getFlops();
    // 4 operations = at least 4 FLOPs
    EXPECT_GE(flops, 4);
}

// =============================================================================
// Memory Traffic Analysis Tests
// =============================================================================

TEST_F(ComputeDAGTest, GetMemoryTraffic) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto traffic = dag.getMemoryTraffic();

    EXPECT_GE(traffic.reads, 0);
    EXPECT_GE(traffic.writes, 0);
}

// =============================================================================
// DAG Analysis Tests
// =============================================================================

TEST_F(ComputeDAGTest, Analyze) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto analysis = dag.analyze();

    EXPECT_TRUE(analysis.is_valid);
    EXPECT_GE(analysis.depth, 1);
}

TEST_F(ComputeDAGTest, AnalyzeIdentifiesOpportunities) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto analysis = dag.analyze();

    // Should identify some optimization opportunities
    // (fusion, parallelization, etc.)
    // The specific opportunities depend on the IR
}

// =============================================================================
// Topological Sort Tests
// =============================================================================

TEST_F(ComputeDAGTest, TopologicalSort) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto sorted = dag.topologicalSort();

    // Should have same number of nodes
    EXPECT_EQ(sorted.size(), dag.numNodes());

    // No node should appear before its dependencies
    std::unordered_set<DAGNodeId> seen;
    for (const auto& id : sorted) {
        const auto& node = dag.getNode(id);
        for (const auto& input : node.inputs) {
            EXPECT_TRUE(seen.count(input) > 0) << "Node appears before dependency";
        }
        seen.insert(id);
    }
}

// =============================================================================
// Subgraph Tests
// =============================================================================

TEST_F(ComputeDAGTest, ExtractSubgraph) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    // Extract subgraph containing first few nodes
    auto nodes = dag.nodes();
    std::vector<DAGNodeId> subset;
    for (size_t i = 0; i < std::min(size_t(2), nodes.size()); ++i) {
        subset.push_back(nodes[i].id);
    }

    auto subgraph = dag.extractSubgraph(subset);

    EXPECT_LE(subgraph.numNodes(), dag.numNodes());
}

// =============================================================================
// Serialization Tests
// =============================================================================

TEST_F(ComputeDAGTest, SerializeDeserialize) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    std::string serialized = dag.serialize();
    EXPECT_FALSE(serialized.empty());

    ComputeDAG restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.numNodes(), dag.numNodes());
}

// =============================================================================
// Benchmark Tests
// =============================================================================

TEST_F(ComputeDAGTest, BenchmarkFromIR) {
    auto module = createComplexIR();

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        auto dag = ComputeDAG::fromIR(module.builder());
        (void)dag;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_dag = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_dag, 100.0) << "DAG construction too slow: " << us_per_dag << " us/dag";

    std::cout << "DAG construction time: " << us_per_dag << " us/dag\n";
}

TEST_F(ComputeDAGTest, BenchmarkAnalysis) {
    auto module = createComplexIR();
    auto dag = ComputeDAG::fromIR(module.builder());

    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        auto analysis = dag.analyze();
        (void)analysis;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_analysis = static_cast<double>(duration_us) / kIterations;

    EXPECT_LT(us_per_analysis, 50.0)
        << "DAG analysis too slow: " << us_per_analysis << " us/analysis";

    std::cout << "DAG analysis time: " << us_per_analysis << " us/analysis\n";
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
