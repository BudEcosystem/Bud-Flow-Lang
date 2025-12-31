// =============================================================================
// Bud Flow Lang - Fusion Cost Model Implementation
// =============================================================================
//
// Analytical cost model for predicting kernel fusion benefits.
// Based on memory bandwidth and compute latency modeling.

#include "bud_flow_lang/ir/fusion_cost_model.h"

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/memory/cache_config.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace bud {
namespace ir {

// =============================================================================
// Operation Latency Tables (in cycles)
// =============================================================================
// Based on Intel Skylake/Ice Lake latencies from Agner Fog's instruction tables
// These are approximate values for SIMD operations

namespace {

// Latency in cycles for each operation
constexpr float kLatencyAdd = 0.5f;  // Simple ALU
constexpr float kLatencySub = 0.5f;
constexpr float kLatencyMul = 0.5f;   // Modern CPUs: FMA unit
constexpr float kLatencyDiv = 11.0f;  // Division is expensive
constexpr float kLatencyFma = 0.5f;   // Fused multiply-add (same as mul)
constexpr float kLatencyFnma = 0.5f;
constexpr float kLatencyNeg = 0.5f;
constexpr float kLatencyAbs = 0.5f;
constexpr float kLatencySqrt = 6.0f;   // Approximate
constexpr float kLatencyRsqrt = 4.0f;  // Fast approximation
constexpr float kLatencyRcp = 4.0f;    // Fast reciprocal
constexpr float kLatencyExp = 15.0f;   // Transcendental (vector impl)
constexpr float kLatencyLog = 15.0f;
constexpr float kLatencySin = 20.0f;
constexpr float kLatencyCos = 20.0f;
constexpr float kLatencyTan = 25.0f;
constexpr float kLatencyTanh = 25.0f;
constexpr float kLatencySigmoid = 20.0f;
constexpr float kLatencyMin = 0.5f;
constexpr float kLatencyMax = 0.5f;
constexpr float kLatencyPow = 30.0f;     // Very expensive
constexpr float kLatencyCompare = 0.5f;  // All comparisons
constexpr float kLatencySelect = 0.5f;   // Blend/select
constexpr float kLatencyReduce = 3.0f;   // Horizontal reduction

// Throughput: operations per cycle per SIMD lane (reciprocal throughput)
constexpr float kThroughputAdd = 2.0f;  // 2 ops/cycle
constexpr float kThroughputMul = 2.0f;
constexpr float kThroughputDiv = 0.125f;  // 1/8 ops/cycle
constexpr float kThroughputFma = 2.0f;
constexpr float kThroughputSqrt = 0.25f;
constexpr float kThroughputTranscendental = 0.1f;

// Memory bandwidth constants (bytes per cycle, approximate)
constexpr float kL1BandwidthBytesPerCycle = 64.0f;  // 2 loads + 1 store @ 32B
constexpr float kL2BandwidthBytesPerCycle = 32.0f;  // Lower than L1
constexpr float kL3BandwidthBytesPerCycle = 16.0f;
constexpr float kMemBandwidthBytesPerCycle = 4.0f;  // ~50 GB/s @ 3GHz

// Cache latencies in cycles
constexpr float kL1Latency = 4.0f;
constexpr float kL2Latency = 12.0f;
constexpr float kL3Latency = 40.0f;
constexpr float kMemLatency = 200.0f;

// Default clock frequency (3 GHz)
constexpr float kDefaultClockFreq = 3.0e9f;

// Get latency for an opcode
float getLatencyForOp(OpCode op) {
    switch (op) {
    case OpCode::kAdd:
        return kLatencyAdd;
    case OpCode::kSub:
        return kLatencySub;
    case OpCode::kMul:
        return kLatencyMul;
    case OpCode::kDiv:
        return kLatencyDiv;
    case OpCode::kMod:
        return kLatencyDiv;  // Modulo uses division
    case OpCode::kFma:
        return kLatencyFma;
    case OpCode::kFnma:
        return kLatencyFnma;
    case OpCode::kNeg:
        return kLatencyNeg;
    case OpCode::kAbs:
        return kLatencyAbs;
    case OpCode::kSqrt:
        return kLatencySqrt;
    case OpCode::kRsqrt:
        return kLatencyRsqrt;
    case OpCode::kRcp:
        return kLatencyRcp;
    case OpCode::kExp:
        return kLatencyExp;
    case OpCode::kLog:
        return kLatencyLog;
    case OpCode::kSin:
        return kLatencySin;
    case OpCode::kCos:
        return kLatencyCos;
    case OpCode::kTan:
        return kLatencyTan;
    case OpCode::kTanh:
        return kLatencyTanh;
    case OpCode::kSigmoid:
        return kLatencySigmoid;
    case OpCode::kMin:
        return kLatencyMin;
    case OpCode::kMax:
        return kLatencyMax;
    case OpCode::kPow:
        return kLatencyPow;
    case OpCode::kEq:
    case OpCode::kNe:
    case OpCode::kLt:
    case OpCode::kLe:
    case OpCode::kGt:
    case OpCode::kGe:
        return kLatencyCompare;
    case OpCode::kSelect:
        return kLatencySelect;
    case OpCode::kReduceSum:
    case OpCode::kReduceMax:
    case OpCode::kReduceMin:
    case OpCode::kReduceProd:
        return kLatencyReduce;
    case OpCode::kAnd:
    case OpCode::kOr:
    case OpCode::kNot:
    case OpCode::kXor:
    case OpCode::kBitAnd:
    case OpCode::kBitOr:
    case OpCode::kBitXor:
    case OpCode::kBitNot:
    case OpCode::kShl:
    case OpCode::kShr:
        return 0.5f;  // Simple bit operations
    default:
        return 1.0f;  // Default fallback
    }
}

// Get throughput for an opcode
float getThroughputForOp(OpCode op) {
    switch (op) {
    case OpCode::kAdd:
    case OpCode::kSub:
        return kThroughputAdd;
    case OpCode::kMul:
        return kThroughputMul;
    case OpCode::kFma:
    case OpCode::kFnma:
        return kThroughputFma;
    case OpCode::kDiv:
    case OpCode::kMod:
        return kThroughputDiv;
    case OpCode::kSqrt:
    case OpCode::kRsqrt:
        return kThroughputSqrt;
    case OpCode::kExp:
    case OpCode::kLog:
    case OpCode::kSin:
    case OpCode::kCos:
    case OpCode::kTan:
    case OpCode::kTanh:
    case OpCode::kSigmoid:
    case OpCode::kPow:
        return kThroughputTranscendental;
    default:
        return 1.0f;  // Default: 1 op/cycle
    }
}

// Get element size from type
size_t getElementSize(const TypeDesc& type) {
    switch (type.scalarType()) {
    case ScalarType::kFloat32:
        return 4;
    case ScalarType::kFloat64:
        return 8;
    case ScalarType::kInt32:
        return 4;
    case ScalarType::kInt64:
        return 8;
    case ScalarType::kInt8:
        return 1;
    case ScalarType::kInt16:
        return 2;
    case ScalarType::kBool:
        return 1;
    default:
        return 4;  // Default to float32
    }
}

}  // namespace

// =============================================================================
// CacheInfo Implementation
// =============================================================================

CacheInfo::CacheInfo()
    : l1_size_(32 * 1024)  // 32 KB default
      ,
      l2_size_(256 * 1024)  // 256 KB default
      ,
      l3_size_(8 * 1024 * 1024)  // 8 MB default
      ,
      line_size_(64)  // 64 bytes default
      ,
      memory_bandwidth_(50.0e9f)  // 50 GB/s default
{}

CacheInfo CacheInfo::fromCacheConfig(const memory::CacheConfig& config) {
    CacheInfo info;
    info.l1_size_ = config.l1Size();
    info.l2_size_ = config.l2Size();
    info.l3_size_ = config.l3Size();
    info.line_size_ = config.lineSize();

    // Estimate bandwidth based on cache sizes and typical ratios
    // Modern desktop CPUs: ~50-100 GB/s
    // Modern server CPUs: ~100-200 GB/s
    // Estimate based on L3 size as a proxy for CPU class
    if (info.l3_size_ >= 30 * 1024 * 1024) {
        info.memory_bandwidth_ = 150.0e9f;  // Server class
    } else if (info.l3_size_ >= 12 * 1024 * 1024) {
        info.memory_bandwidth_ = 80.0e9f;  // High-end desktop
    } else {
        info.memory_bandwidth_ = 50.0e9f;  // Standard desktop
    }

    return info;
}

// =============================================================================
// FusionCostModel Implementation
// =============================================================================

FusionCostModel::FusionCostModel()
    : simd_width_(8)  // Default: 8 lanes (AVX-256 float32)
      ,
      clock_freq_(kDefaultClockFreq),
      l1_bandwidth_(kL1BandwidthBytesPerCycle),
      l2_bandwidth_(kL2BandwidthBytesPerCycle),
      mem_bandwidth_(kMemBandwidthBytesPerCycle) {
    // Detect SIMD width based on available CPU features
    // Default to AVX-256 (8 float32 lanes) for modern x86-64
#if defined(__AVX512F__)
    simd_width_ = 16;  // AVX-512: 16 float32 lanes
#elif defined(__AVX2__) || defined(__AVX__)
    simd_width_ = 8;  // AVX-256: 8 float32 lanes
#elif defined(__SSE4_2__) || defined(__SSE4_1__) || defined(__SSE2__)
    simd_width_ = 4;  // SSE: 4 float32 lanes
#elif defined(__ARM_NEON)
    simd_width_ = 4;  // NEON: 4 float32 lanes
#else
    simd_width_ = 1;  // Scalar fallback
#endif
}

FusionCostModel::~FusionCostModel() = default;

float FusionCostModel::memoryReadCost(size_t bytes, bool cached) const {
    if (bytes == 0)
        return 0.0f;

    // Cost in cycles
    float bandwidth = cached ? l1_bandwidth_ : mem_bandwidth_;
    float latency = cached ? kL1Latency : kMemLatency;

    // Total cost = latency + transfer time
    float transfer_cycles = static_cast<float>(bytes) / bandwidth;
    return latency + transfer_cycles;
}

float FusionCostModel::memoryWriteCost(size_t bytes) const {
    if (bytes == 0)
        return 0.0f;

    // Writes are typically buffered, so slightly lower latency
    // but similar bandwidth cost
    float transfer_cycles = static_cast<float>(bytes) / mem_bandwidth_;
    return kL1Latency + transfer_cycles;
}

float FusionCostModel::computeCost(OpCode op, size_t elements) const {
    if (elements == 0)
        return 0.0f;

    float latency = getLatencyForOp(op);
    float throughput = getThroughputForOp(op);

    // Number of vector iterations
    size_t vector_iters = (elements + simd_width_ - 1) / simd_width_;

    // Cost = latency + iterations / throughput
    float cost = latency + static_cast<float>(vector_iters) / throughput;
    return cost;
}

float FusionCostModel::getOpLatency(OpCode op) const {
    return getLatencyForOp(op);
}

float FusionCostModel::getOpThroughput(OpCode op) const {
    return getThroughputForOp(op);
}

float FusionCostModel::predictTime(const IRNode* node, const CacheInfo& cache) const {
    if (!node)
        return 0.0f;

    OpCode op = node->opCode();
    const TypeDesc& type = node->type();
    size_t elem_size = getElementSize(type);

    // Estimate number of elements from type
    size_t num_elements = type.elementCount();

    // Compute cost
    float compute = computeCost(op, num_elements);

    // Memory cost (reading operands)
    size_t read_bytes = node->numOperands() * num_elements * elem_size;
    bool cached = cache.fitsInL1(read_bytes);
    float memory_read = memoryReadCost(read_bytes, cached);

    // Memory cost (writing output)
    size_t write_bytes = num_elements * elem_size;
    float memory_write = memoryWriteCost(write_bytes);

    // Total time (cycles to seconds)
    float total_cycles = compute + memory_read + memory_write;
    return total_cycles / clock_freq_;
}

float FusionCostModel::predictFusedTime(const std::vector<IRNode*>& nodes,
                                        const CacheInfo& cache) const {
    if (nodes.empty())
        return 0.0f;

    // For fused kernels:
    // - Compute costs are additive
    // - Memory read costs: only for external inputs
    // - Memory write costs: only for final outputs
    // - Intermediate results stay in registers

    float total_compute = 0.0f;
    size_t total_elements = 0;
    size_t elem_size = 4;  // Default float32

    for (const auto* node : nodes) {
        if (!node)
            continue;

        OpCode op = node->opCode();
        const TypeDesc& type = node->type();
        elem_size = getElementSize(type);

        size_t num_elements = type.elementCount();
        total_elements = std::max(total_elements, num_elements);

        total_compute += computeCost(op, num_elements);
    }

    // External inputs: only from first node (producer)
    // In reality, we'd analyze the DAG to find true external inputs
    size_t external_input_bytes = 0;
    if (!nodes.empty() && nodes[0]) {
        external_input_bytes = nodes[0]->numOperands() * total_elements * elem_size;
    }

    // Output: only from last node (consumer)
    size_t output_bytes = total_elements * elem_size;

    bool cached = cache.fitsInL1(external_input_bytes);
    float memory_read = memoryReadCost(external_input_bytes, cached);
    float memory_write = memoryWriteCost(output_bytes);

    float total_cycles = total_compute + memory_read + memory_write;
    return total_cycles / clock_freq_;
}

float FusionCostModel::calculateBenefit(IRNode* producer, IRNode* consumer,
                                        const CacheInfo& cache) const {
    if (!producer || !consumer)
        return 0.0f;

    // Check if consumer uses producer's output
    bool uses_producer = false;
    for (size_t i = 0; i < consumer->numOperands(); ++i) {
        if (consumer->operand(i) == producer->id()) {
            uses_producer = true;
            break;
        }
    }

    if (!uses_producer) {
        return 0.0f;  // No data dependency, no fusion benefit
    }

    // Calculate unfused cost
    float producer_time = predictTime(producer, cache);
    float consumer_time = predictTime(consumer, cache);
    float unfused_time = producer_time + consumer_time;

    // Calculate fused cost
    std::vector<IRNode*> fused_nodes = {producer, consumer};
    float fused_time = predictFusedTime(fused_nodes, cache);

    // Benefit = time saved
    float benefit = unfused_time - fused_time;

    // Add bonus for eliminating intermediate memory traffic
    size_t eliminated_bytes = eliminatedMemoryTraffic(producer, consumer);
    float memory_savings =
        memoryWriteCost(eliminated_bytes) + memoryReadCost(eliminated_bytes, false);
    benefit += memory_savings / clock_freq_;

    return benefit;
}

size_t FusionCostModel::eliminatedMemoryTraffic(const IRNode* producer,
                                                const IRNode* consumer) const {
    if (!producer || !consumer)
        return 0;

    // The eliminated traffic is the size of producer's output
    const TypeDesc& type = producer->type();
    size_t elem_size = getElementSize(type);
    size_t num_elements = type.elementCount();

    // Producer writes, consumer reads = 2x the data size
    return 2 * num_elements * elem_size;
}

bool FusionCostModel::isSingleUse(const IRBuilder& builder, ValueId producer_id) const {
    return builder.useCount(producer_id) == 1;
}

bool FusionCostModel::canFuse(OpCode producer_op, OpCode consumer_op) const {
    // Supported fusion patterns:

    // Mul + Add -> FMA
    if (producer_op == OpCode::kMul && consumer_op == OpCode::kAdd) {
        return true;
    }

    // Mul + Sub -> FNMA (when sub is c - mul, not mul - c)
    if (producer_op == OpCode::kMul && consumer_op == OpCode::kSub) {
        return true;
    }

    // Add + Mul (for some patterns)
    if (producer_op == OpCode::kAdd && consumer_op == OpCode::kMul) {
        return true;
    }

    // Sub + Mul
    if (producer_op == OpCode::kSub && consumer_op == OpCode::kMul) {
        return true;
    }

    // Neg + Mul -> NegMul
    if (producer_op == OpCode::kNeg && consumer_op == OpCode::kMul) {
        return true;
    }

    // Future: more patterns like div+sqrt -> rsqrt, etc.

    return false;
}

OpCode FusionCostModel::getFusedOp(OpCode producer_op, OpCode consumer_op) const {
    // Mul + Add -> FMA
    if (producer_op == OpCode::kMul && consumer_op == OpCode::kAdd) {
        return OpCode::kFma;
    }

    // Mul + Sub (c - mul) -> FNMA
    if (producer_op == OpCode::kMul && consumer_op == OpCode::kSub) {
        return OpCode::kFnma;
    }

    // Default: no change
    return consumer_op;
}

std::vector<FusionOpportunity>
FusionCostModel::findFusionOpportunities(IRBuilder& builder, const CacheInfo& cache) const {

    std::vector<FusionOpportunity> opportunities;

    // Build use count map
    std::unordered_map<uint32_t, size_t> use_counts;
    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead())
            continue;
        for (const auto& operand : node->operands()) {
            use_counts[operand.id]++;
        }
    }

    // Find all producer-consumer pairs
    for (auto* consumer : builder.nodes()) {
        if (!consumer || consumer->isDead())
            continue;

        for (size_t i = 0; i < consumer->numOperands(); ++i) {
            ValueId producer_id = consumer->operand(i);
            IRNode* producer = builder.getNode(producer_id);

            if (!producer || producer->isDead())
                continue;

            // Check if fusion is legal (single use)
            if (use_counts[producer_id.id] != 1)
                continue;

            // Check if fusion pattern is supported
            if (!canFuse(producer->opCode(), consumer->opCode()))
                continue;

            // Calculate benefit
            float benefit = calculateBenefit(producer, consumer, cache);

            if (benefit > 0.0f) {
                FusionOpportunity opp;
                opp.producer = producer_id;
                opp.consumer = consumer->id();
                opp.benefit = benefit;
                opp.fused_op = getFusedOp(producer->opCode(), consumer->opCode());
                opp.can_fuse = true;
                opportunities.push_back(opp);
            }
        }
    }

    // Sort by benefit (highest first)
    std::sort(opportunities.begin(), opportunities.end());

    return opportunities;
}

// =============================================================================
// PriorityFusionPass Implementation
// =============================================================================

PriorityFusionPass::PriorityFusionPass(const FusionCostModel& cost_model)
    : cost_model_(cost_model),
      min_benefit_(0.0f),
      cache_info_(CacheInfo::fromCacheConfig(memory::CacheConfig::detect())) {}

size_t PriorityFusionPass::run(IRBuilder& builder) {
    size_t fused_count = 0;

    // Iteratively find and perform fusions
    // We iterate because fusion changes the IR and may enable new opportunities
    bool changed = true;
    while (changed) {
        changed = false;

        // Find all opportunities
        auto opportunities = cost_model_.findFusionOpportunities(builder, cache_info_);

        // Try to perform fusions in priority order
        for (const auto& opp : opportunities) {
            if (opp.benefit < min_benefit_)
                break;  // Sorted, so rest are worse

            if (performFusion(builder, opp)) {
                ++fused_count;
                changed = true;
                break;  // Restart analysis after each fusion
            }
        }
    }

    return fused_count;
}

bool PriorityFusionPass::performFusion(IRBuilder& builder, const FusionOpportunity& opp) {
    IRNode* producer = builder.getNode(opp.producer);
    IRNode* consumer = builder.getNode(opp.consumer);

    if (!producer || !consumer)
        return false;
    if (producer->isDead() || consumer->isDead())
        return false;

    // Verify still single use (IR may have changed)
    if (builder.useCount(opp.producer) != 1)
        return false;

    // Perform the fusion based on pattern
    OpCode prod_op = producer->opCode();
    OpCode cons_op = consumer->opCode();

    // Pattern: Mul + Add -> FMA
    if (prod_op == OpCode::kMul && cons_op == OpCode::kAdd) {
        // Find which operand of Add is the Mul result
        ValueId a, b, c;
        if (consumer->operand(0) == opp.producer) {
            // Add(Mul(a, b), c)
            a = producer->operand(0);
            b = producer->operand(1);
            c = consumer->operand(1);
        } else {
            // Add(c, Mul(a, b))
            a = producer->operand(0);
            b = producer->operand(1);
            c = consumer->operand(0);
        }

        // Create FMA node
        ValueId fma_id = builder.fma(a, b, c);
        if (!fma_id.isValid())
            return false;

        // Replace uses of consumer with FMA
        builder.replaceAllUses(consumer->id(), fma_id);

        // Mark old nodes as dead
        producer->markDead();
        consumer->markDead();

        return true;
    }

    // Pattern: Mul + Sub (c - Mul) -> FNMA
    if (prod_op == OpCode::kMul && cons_op == OpCode::kSub) {
        // Sub(c, Mul(a, b)) -> FNMA(a, b, c) = -(a*b) + c = c - a*b
        if (consumer->operand(1) == opp.producer) {
            ValueId a = producer->operand(0);
            ValueId b = producer->operand(1);
            ValueId c = consumer->operand(0);

            // Create FNMA node
            // Note: We need to create this manually as builder may not have fnma()
            auto fnma_type = consumer->type();
            auto* fnma_node = new IRNode(OpCode::kFnma, fnma_type,
                                         ValueId{static_cast<uint32_t>(builder.nodes().size())});
            fnma_node->addOperand(a);
            fnma_node->addOperand(b);
            fnma_node->addOperand(c);
            builder.mutableNodes().push_back(fnma_node);

            ValueId fnma_id = fnma_node->id();

            // Replace uses
            builder.replaceAllUses(consumer->id(), fnma_id);
            producer->markDead();
            consumer->markDead();

            return true;
        }
    }

    return false;
}

}  // namespace ir
}  // namespace bud
