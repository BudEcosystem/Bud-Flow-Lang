#pragma once

// =============================================================================
// Bud Flow Lang - Fusion Cost Model
// =============================================================================
//
// Analytical cost model for predicting kernel fusion benefits.
// Replaces heuristic-based fusion with data-driven decisions based on:
// - Memory bandwidth costs
// - Compute operation latencies
// - Cache hierarchy effects
// - SIMD lane utilization
//
// References:
// - XLA's HloCostAnalysis
// - TVM's Ansor cost model
// - LLVM's TargetTransformInfo

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/memory/cache_config.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bud {
namespace ir {

// Forward declarations
class IRBuilder;
class IRNode;

// =============================================================================
// CacheInfo - Runtime cache configuration for cost estimation
// =============================================================================

class CacheInfo {
  public:
    CacheInfo();

    static CacheInfo fromCacheConfig(const memory::CacheConfig& config);

    // Cache sizes in bytes
    [[nodiscard]] size_t l1Size() const { return l1_size_; }
    [[nodiscard]] size_t l2Size() const { return l2_size_; }
    [[nodiscard]] size_t l3Size() const { return l3_size_; }
    [[nodiscard]] size_t lineSize() const { return line_size_; }

    // Estimated memory bandwidth in bytes per second
    [[nodiscard]] float memoryBandwidth() const { return memory_bandwidth_; }

    // Set memory bandwidth (for testing or known values)
    void setMemoryBandwidth(float bandwidth) { memory_bandwidth_ = bandwidth; }

    // Check if data of given size fits in cache level
    [[nodiscard]] bool fitsInL1(size_t bytes) const { return bytes <= l1_size_; }
    [[nodiscard]] bool fitsInL2(size_t bytes) const { return bytes <= l2_size_; }
    [[nodiscard]] bool fitsInL3(size_t bytes) const { return bytes <= l3_size_; }

  private:
    size_t l1_size_;
    size_t l2_size_;
    size_t l3_size_;
    size_t line_size_;
    float memory_bandwidth_;  // bytes/second
};

// =============================================================================
// FusionOpportunity - A potential fusion between producer and consumer
// =============================================================================

struct FusionOpportunity {
    ValueId producer;  // Producer node ID
    ValueId consumer;  // Consumer node ID
    float benefit;     // Estimated time saved (positive = good to fuse)
    OpCode fused_op;   // The fused operation (e.g., kFma)
    bool can_fuse;     // Whether fusion is legal

    // For sorting by benefit (highest first)
    bool operator<(const FusionOpportunity& other) const { return benefit > other.benefit; }
};

// =============================================================================
// FusionCostModel - Core cost estimation engine
// =============================================================================

class FusionCostModel {
  public:
    FusionCostModel();
    ~FusionCostModel();

    // =========================================================================
    // Memory Cost Estimation
    // =========================================================================

    // Estimate cost of reading data from memory
    // cached: true if data is likely in cache
    [[nodiscard]] float memoryReadCost(size_t bytes, bool cached) const;

    // Estimate cost of writing data to memory
    [[nodiscard]] float memoryWriteCost(size_t bytes) const;

    // =========================================================================
    // Compute Cost Estimation
    // =========================================================================

    // Estimate cost of an operation on N elements
    [[nodiscard]] float computeCost(OpCode op, size_t elements) const;

    // Get operation latency in cycles
    [[nodiscard]] float getOpLatency(OpCode op) const;

    // Get operation throughput (ops per cycle per lane)
    [[nodiscard]] float getOpThroughput(OpCode op) const;

    // =========================================================================
    // Node/Kernel Time Prediction
    // =========================================================================

    // Predict execution time for a single IR node
    [[nodiscard]] float predictTime(const IRNode* node, const CacheInfo& cache) const;

    // Predict execution time for a sequence of nodes if fused
    [[nodiscard]] float predictFusedTime(const std::vector<IRNode*>& nodes,
                                         const CacheInfo& cache) const;

    // =========================================================================
    // Fusion Benefit Calculation
    // =========================================================================

    // Calculate benefit of fusing producer into consumer
    // Returns: estimated time saved (positive = beneficial)
    [[nodiscard]] float calculateBenefit(IRNode* producer, IRNode* consumer,
                                         const CacheInfo& cache) const;

    // Find all fusion opportunities in the IR
    [[nodiscard]] std::vector<FusionOpportunity>
    findFusionOpportunities(IRBuilder& builder, const CacheInfo& cache) const;

    // =========================================================================
    // Configuration
    // =========================================================================

    // Set SIMD width (vector lanes) - detected automatically
    void setSimdWidth(size_t width) { simd_width_ = width; }
    [[nodiscard]] size_t simdWidth() const { return simd_width_; }

    // Set clock frequency for time estimation
    void setClockFrequency(float freq_hz) { clock_freq_ = freq_hz; }

  private:
    // Check if producer output is only used by consumer
    [[nodiscard]] bool isSingleUse(const IRBuilder& builder, ValueId producer_id) const;

    // Check if fusion pattern is supported
    [[nodiscard]] bool canFuse(OpCode producer_op, OpCode consumer_op) const;

    // Get the resulting fused opcode
    [[nodiscard]] OpCode getFusedOp(OpCode producer_op, OpCode consumer_op) const;

    // Estimate intermediate data size eliminated by fusion
    [[nodiscard]] size_t eliminatedMemoryTraffic(const IRNode* producer,
                                                 const IRNode* consumer) const;

    size_t simd_width_;    // SIMD lanes (e.g., 8 for AVX-256 float)
    float clock_freq_;     // CPU clock frequency in Hz
    float l1_bandwidth_;   // L1 cache bandwidth (bytes/cycle)
    float l2_bandwidth_;   // L2 cache bandwidth (bytes/cycle)
    float mem_bandwidth_;  // Main memory bandwidth (bytes/cycle)

    // Operation latency table (cycles)
    static const float kOpLatencies[];

    // Operation throughput table (ops per cycle per lane)
    static const float kOpThroughputs[];
};

// =============================================================================
// PriorityFusionPass - Cost model-driven fusion optimization pass
// =============================================================================

class PriorityFusionPass {
  public:
    explicit PriorityFusionPass(const FusionCostModel& cost_model);

    // Run the pass on the IR
    // Returns: number of fusions performed
    size_t run(IRBuilder& builder);

    // Set minimum benefit threshold for fusion
    void setMinBenefit(float min_benefit) { min_benefit_ = min_benefit; }

  private:
    const FusionCostModel& cost_model_;
    float min_benefit_;
    CacheInfo cache_info_;

    // Perform a single fusion
    bool performFusion(IRBuilder& builder, const FusionOpportunity& opportunity);
};

}  // namespace ir
}  // namespace bud
