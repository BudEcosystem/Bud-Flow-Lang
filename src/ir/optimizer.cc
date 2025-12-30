// =============================================================================
// Bud Flow Lang - IR Optimizer Implementation
// =============================================================================
//
// Optimization passes that actually mutate the IR using the new mutation
// methods: replaceAllUses, markDead, compactNodes.
//

#include "bud_flow_lang/ir.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bud {
namespace ir {

namespace {

// =============================================================================
// Helper Functions
// =============================================================================

// Check if a node is a constant
bool isConstant(const IRNode* node) {
    if (!node)
        return false;
    return node->opCode() == OpCode::kConstantScalar || node->opCode() == OpCode::kConstantVector;
}

// Get float constant value
double getConstantFloat(const IRNode* node) {
    if (!node || node->opCode() != OpCode::kConstantScalar) {
        return 0.0;
    }
    return node->floatAttr("value", 0.0);
}

// Get int constant value
int64_t getConstantInt(const IRNode* node) {
    if (!node || node->opCode() != OpCode::kConstantScalar) {
        return 0;
    }
    return node->intAttr("value", 0);
}

// Check if type is floating point
bool isFloatType(ScalarType type) {
    return type == ScalarType::kFloat32 || type == ScalarType::kFloat64 ||
           type == ScalarType::kFloat16 || type == ScalarType::kBFloat16;
}

// =============================================================================
// Constant Folding Pass
// =============================================================================

class ConstantFoldingPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t folded_count = 0;

        // Iterate through nodes and fold constants
        // Note: We iterate by index because we may add new nodes during folding
        size_t node_count = builder.nodes().size();
        for (size_t i = 0; i < node_count; ++i) {
            auto* node = builder.nodes()[i];
            if (!node || node->isDead())
                continue;

            OpCode op = node->opCode();
            TypeDesc result_type = node->type();

            // Binary operations with two constant operands
            if (node->numOperands() == 2) {
                auto* lhs = builder.getNode(node->operand(0));
                auto* rhs = builder.getNode(node->operand(1));

                if (isConstant(lhs) && isConstant(rhs) && isFloatType(result_type.scalarType())) {

                    double a = getConstantFloat(lhs);
                    double b = getConstantFloat(rhs);
                    double result = 0.0;
                    bool can_fold = true;

                    switch (op) {
                    case OpCode::kAdd:
                        result = a + b;
                        break;
                    case OpCode::kSub:
                        result = a - b;
                        break;
                    case OpCode::kMul:
                        result = a * b;
                        break;
                    case OpCode::kDiv:
                        if (b != 0.0)
                            result = a / b;
                        else
                            can_fold = false;
                        break;
                    case OpCode::kMin:
                        result = std::min(a, b);
                        break;
                    case OpCode::kMax:
                        result = std::max(a, b);
                        break;
                    case OpCode::kPow:
                        result = std::pow(a, b);
                        break;
                    default:
                        can_fold = false;
                        break;
                    }

                    if (can_fold) {
                        // Create new constant node
                        ValueId new_const;
                        if (result_type.scalarType() == ScalarType::kFloat32) {
                            new_const = builder.constant(static_cast<float>(result));
                        } else {
                            new_const = builder.constant(result);
                        }

                        if (new_const.isValid()) {
                            // Replace all uses of this node with the new constant
                            builder.replaceAllUses(node->id(), new_const);
                            node->markDead();
                            ++folded_count;

                            spdlog::debug("ConstFold: {} {} {} -> {} (node %{} -> %{})", a,
                                          opCodeName(op), b, result, node->id().id, new_const.id);
                        }
                    }
                }
            }

            // Unary operations with constant operand
            if (node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));

                if (isConstant(operand) && isFloatType(result_type.scalarType())) {
                    double a = getConstantFloat(operand);
                    double result = 0.0;
                    bool can_fold = true;

                    switch (op) {
                    case OpCode::kNeg:
                        result = -a;
                        break;
                    case OpCode::kAbs:
                        result = std::abs(a);
                        break;
                    case OpCode::kSqrt:
                        if (a >= 0)
                            result = std::sqrt(a);
                        else
                            can_fold = false;
                        break;
                    case OpCode::kRsqrt:
                        if (a > 0)
                            result = 1.0 / std::sqrt(a);
                        else
                            can_fold = false;
                        break;
                    case OpCode::kRcp:
                        if (a != 0)
                            result = 1.0 / a;
                        else
                            can_fold = false;
                        break;
                    case OpCode::kExp:
                        result = std::exp(a);
                        break;
                    case OpCode::kLog:
                        if (a > 0)
                            result = std::log(a);
                        else
                            can_fold = false;
                        break;
                    case OpCode::kSin:
                        result = std::sin(a);
                        break;
                    case OpCode::kCos:
                        result = std::cos(a);
                        break;
                    case OpCode::kTanh:
                        result = std::tanh(a);
                        break;
                    default:
                        can_fold = false;
                        break;
                    }

                    if (can_fold) {
                        ValueId new_const;
                        if (result_type.scalarType() == ScalarType::kFloat32) {
                            new_const = builder.constant(static_cast<float>(result));
                        } else {
                            new_const = builder.constant(result);
                        }

                        if (new_const.isValid()) {
                            builder.replaceAllUses(node->id(), new_const);
                            node->markDead();
                            ++folded_count;

                            spdlog::debug("ConstFold: {}({}) -> {} (node %{} -> %{})",
                                          opCodeName(op), a, result, node->id().id, new_const.id);
                        }
                    }
                }
            }
        }

        return folded_count;
    }
};

// =============================================================================
// Dead Code Elimination Pass
// =============================================================================

class DeadCodeEliminationPass {
  public:
    size_t run(IRBuilder& builder, ValueId output) {
        // Mark all nodes reachable from output as live
        std::unordered_set<uint32_t> live;
        markLive(builder, output, live);

        // Mark unreachable nodes as dead
        size_t dead_count = 0;
        for (auto* node : builder.mutableNodes()) {
            if (node && !node->isDead() && live.find(node->id().id) == live.end()) {
                spdlog::debug("DCE: Marking node %{} ({}) as dead", node->id().id,
                              opCodeName(node->opCode()));
                node->markDead();
                ++dead_count;
            }
        }

        // Compact nodes to remove dead ones
        if (dead_count > 0) {
            size_t compacted = builder.compactNodes();
            spdlog::debug("DCE: Removed {} dead nodes", compacted);
        }

        return dead_count;
    }

  private:
    void markLive(const IRBuilder& builder, ValueId id, std::unordered_set<uint32_t>& live) {
        if (!id.isValid() || live.count(id.id)) {
            return;
        }

        live.insert(id.id);
        const auto* node = builder.getNode(id);
        if (node && !node->isDead()) {
            for (const auto& operand : node->operands()) {
                markLive(builder, operand, live);
            }
        }
    }
};

// =============================================================================
// Operation Fusion Pass (Mul+Add -> FMA)
// =============================================================================

class FusionPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t fused_count = 0;

        // Build use-count map for detecting single-use operands
        std::unordered_map<uint32_t, size_t> use_counts;
        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;
            for (const auto& operand : node->operands()) {
                use_counts[operand.id]++;
            }
        }

        // Look for fusion opportunities
        size_t node_count = builder.nodes().size();
        for (size_t i = 0; i < node_count; ++i) {
            auto* node = builder.nodes()[i];
            if (!node || node->isDead())
                continue;

            // Pattern: Add where one operand is Mul with single use -> FMA
            if (node->opCode() == OpCode::kAdd && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);
                ValueId rhs_id = node->operand(1);
                auto* lhs = builder.getNode(lhs_id);
                auto* rhs = builder.getNode(rhs_id);

                // Check left operand: Add(Mul(a, b), c) -> FMA(a, b, c)
                if (lhs && !lhs->isDead() && lhs->opCode() == OpCode::kMul &&
                    lhs->numOperands() == 2 && use_counts[lhs_id.id] == 1) {

                    ValueId a = lhs->operand(0);
                    ValueId b = lhs->operand(1);
                    ValueId c = rhs_id;

                    // Create FMA node
                    ValueId fma_id = builder.fma(a, b, c);
                    if (fma_id.isValid()) {
                        builder.replaceAllUses(node->id(), fma_id);
                        node->markDead();
                        lhs->markDead();
                        ++fused_count;

                        spdlog::debug("Fusion: Add(Mul(%{}, %{}), %{}) -> FMA (node %{})", a.id,
                                      b.id, c.id, fma_id.id);
                        continue;
                    }
                }

                // Check right operand: Add(c, Mul(a, b)) -> FMA(a, b, c)
                if (rhs && !rhs->isDead() && rhs->opCode() == OpCode::kMul &&
                    rhs->numOperands() == 2 && use_counts[rhs_id.id] == 1) {

                    ValueId a = rhs->operand(0);
                    ValueId b = rhs->operand(1);
                    ValueId c = lhs_id;

                    ValueId fma_id = builder.fma(a, b, c);
                    if (fma_id.isValid()) {
                        builder.replaceAllUses(node->id(), fma_id);
                        node->markDead();
                        rhs->markDead();
                        ++fused_count;

                        spdlog::debug("Fusion: Add(%{}, Mul(%{}, %{})) -> FMA (node %{})", c.id,
                                      a.id, b.id, fma_id.id);
                        continue;
                    }
                }
            }

            // Pattern: Sub(c, Mul(a, b)) -> FNMA: -(a*b) + c = c - a*b
            // Note: This requires an FNMA opcode which negates the product
            // For now, we skip this pattern as it requires additional support
        }

        return fused_count;
    }
};

// =============================================================================
// Strength Reduction Pass
// =============================================================================

class StrengthReductionPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t reduced_count = 0;

        size_t node_count = builder.nodes().size();
        for (size_t i = 0; i < node_count; ++i) {
            auto* node = builder.nodes()[i];
            if (!node || node->isDead())
                continue;

            // x * 1 -> x (identity)
            // x * 0 -> 0
            // x + 0 -> x
            // x - 0 -> x
            // x / 1 -> x

            if (node->opCode() == OpCode::kMul && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);
                ValueId rhs_id = node->operand(1);
                auto* lhs = builder.getNode(lhs_id);
                auto* rhs = builder.getNode(rhs_id);

                // x * 1 -> x
                if (isConstant(rhs) && getConstantFloat(rhs) == 1.0) {
                    builder.replaceAllUses(node->id(), lhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: %{} * 1 -> %{}", lhs_id.id, lhs_id.id);
                    continue;
                }
                if (isConstant(lhs) && getConstantFloat(lhs) == 1.0) {
                    builder.replaceAllUses(node->id(), rhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: 1 * %{} -> %{}", rhs_id.id, rhs_id.id);
                    continue;
                }

                // x * 0 -> 0
                if (isConstant(rhs) && getConstantFloat(rhs) == 0.0) {
                    builder.replaceAllUses(node->id(), rhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: %{} * 0 -> 0", lhs_id.id);
                    continue;
                }
                if (isConstant(lhs) && getConstantFloat(lhs) == 0.0) {
                    builder.replaceAllUses(node->id(), lhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: 0 * %{} -> 0", rhs_id.id);
                    continue;
                }

                // x * 2 -> x + x (potentially faster on some architectures)
                // For now, skip this - it's not always faster
            }

            if (node->opCode() == OpCode::kAdd && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);
                ValueId rhs_id = node->operand(1);
                auto* lhs = builder.getNode(lhs_id);
                auto* rhs = builder.getNode(rhs_id);

                // x + 0 -> x
                if (isConstant(rhs) && getConstantFloat(rhs) == 0.0) {
                    builder.replaceAllUses(node->id(), lhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: %{} + 0 -> %{}", lhs_id.id, lhs_id.id);
                    continue;
                }
                if (isConstant(lhs) && getConstantFloat(lhs) == 0.0) {
                    builder.replaceAllUses(node->id(), rhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: 0 + %{} -> %{}", rhs_id.id, rhs_id.id);
                    continue;
                }
            }

            if (node->opCode() == OpCode::kSub && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);
                ValueId rhs_id = node->operand(1);
                auto* rhs = builder.getNode(rhs_id);

                // x - 0 -> x
                if (isConstant(rhs) && getConstantFloat(rhs) == 0.0) {
                    builder.replaceAllUses(node->id(), lhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: %{} - 0 -> %{}", lhs_id.id, lhs_id.id);
                    continue;
                }
            }

            if (node->opCode() == OpCode::kDiv && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);
                ValueId rhs_id = node->operand(1);
                auto* rhs = builder.getNode(rhs_id);

                // x / 1 -> x
                if (isConstant(rhs) && getConstantFloat(rhs) == 1.0) {
                    builder.replaceAllUses(node->id(), lhs_id);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: %{} / 1 -> %{}", lhs_id.id, lhs_id.id);
                    continue;
                }

                // x / c -> x * (1/c) for non-zero constant c
                // This can be faster as multiplication is cheaper than division
                if (isConstant(rhs)) {
                    double c = getConstantFloat(rhs);
                    if (c != 0.0 && c != 1.0) {
                        // Create reciprocal constant
                        ValueId recip_const;
                        if (node->type().scalarType() == ScalarType::kFloat32) {
                            recip_const = builder.constant(static_cast<float>(1.0 / c));
                        } else {
                            recip_const = builder.constant(1.0 / c);
                        }

                        if (recip_const.isValid()) {
                            // Create multiplication node
                            ValueId mul_id = builder.mul(lhs_id, recip_const);
                            if (mul_id.isValid()) {
                                builder.replaceAllUses(node->id(), mul_id);
                                node->markDead();
                                ++reduced_count;
                                spdlog::debug("StrengthReduction: %{} / {} -> %{} * {}", lhs_id.id,
                                              c, lhs_id.id, 1.0 / c);
                            }
                        }
                    }
                }
            }

            // Double negation: Neg(Neg(x)) -> x
            if (node->opCode() == OpCode::kNeg && node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));
                if (operand && !operand->isDead() && operand->opCode() == OpCode::kNeg) {
                    ValueId original = operand->operand(0);
                    builder.replaceAllUses(node->id(), original);
                    node->markDead();
                    ++reduced_count;
                    spdlog::debug("StrengthReduction: Neg(Neg(%{})) -> %{}", original.id,
                                  original.id);
                }
            }
        }

        return reduced_count;
    }
};

// =============================================================================
// Common Subexpression Elimination Pass
// =============================================================================

class CommonSubexpressionEliminationPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t eliminated_count = 0;

        // Build a hash map of (opcode, operands) -> node ID
        // For nodes with the same computation, keep only the first one
        std::unordered_map<std::string, ValueId> expr_map;

        for (auto* node : builder.mutableNodes()) {
            if (!node || node->isDead())
                continue;

            // Skip non-pure operations (loads, stores, etc.)
            OpCode op = node->opCode();
            if (op == OpCode::kLoad || op == OpCode::kStore || op == OpCode::kAlloc ||
                op == OpCode::kCall) {
                continue;
            }

            // Build expression key: "opcode:operand1,operand2,..."
            std::string key = std::string(opCodeName(op)) + ":";
            for (size_t i = 0; i < node->numOperands(); ++i) {
                if (i > 0)
                    key += ",";
                key += std::to_string(node->operand(i).id);
            }

            auto it = expr_map.find(key);
            if (it != expr_map.end()) {
                // Found duplicate - replace uses and mark dead
                builder.replaceAllUses(node->id(), it->second);
                node->markDead();
                ++eliminated_count;
                spdlog::debug("CSE: Node %{} is duplicate of %{} ({})", node->id().id,
                              it->second.id, key);
            } else {
                expr_map[key] = node->id();
            }
        }

        return eliminated_count;
    }
};

}  // namespace

// =============================================================================
// Public API
// =============================================================================

Result<void> runOptimizationPipeline(IRModule& module, int level) {
    if (level <= 0) {
        return {};  // No optimization
    }

    auto& builder = module.builder();
    auto output = module.output();

    bool changed = true;
    int iterations = 0;
    constexpr int kMaxIterations = 10;

    size_t total_folded = 0;
    size_t total_eliminated = 0;
    size_t total_fused = 0;
    size_t total_reduced = 0;
    size_t total_cse = 0;

    while (changed && iterations < kMaxIterations) {
        changed = false;
        ++iterations;

        // Level 1+: Basic optimizations
        if (level >= 1) {
            ConstantFoldingPass cf;
            size_t folded = cf.run(builder);
            total_folded += folded;
            changed |= (folded > 0);

            DeadCodeEliminationPass dce;
            size_t eliminated = dce.run(builder, output);
            total_eliminated += eliminated;
            changed |= (eliminated > 0);
        }

        // Level 2+: Standard optimizations
        if (level >= 2) {
            StrengthReductionPass sr;
            size_t reduced = sr.run(builder);
            total_reduced += reduced;
            changed |= (reduced > 0);

            FusionPass fusion;
            size_t fused = fusion.run(builder);
            total_fused += fused;
            changed |= (fused > 0);

            CommonSubexpressionEliminationPass cse;
            size_t cse_count = cse.run(builder);
            total_cse += cse_count;
            changed |= (cse_count > 0);
        }

        // Level 3: Aggressive optimizations
        if (level >= 3) {
            // Additional aggressive optimizations could go here:
            // - Loop unrolling hints
            // - Vectorization analysis
            // - Memory access pattern optimization
            // - Speculative optimization
        }
    }

    // Final compaction to clean up any remaining dead nodes
    builder.compactNodes();

    spdlog::info("Optimization completed in {} iterations: folded={}, eliminated={}, "
                 "reduced={}, fused={}, cse={}",
                 iterations, total_folded, total_eliminated, total_reduced, total_fused, total_cse);

    return {};
}

}  // namespace ir
}  // namespace bud
