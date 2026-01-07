// =============================================================================
// Bud Flow Lang - IR Optimizer Implementation
// =============================================================================
//
// Optimization passes that actually mutate the IR using the new mutation
// methods: replaceAllUses, markDead, compactNodes.
//

#include "bud_flow_lang/ir.h"

#include <spdlog/spdlog.h>

#include <algorithm>
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
            if (node->opCode() == OpCode::kSub && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);  // c
                ValueId rhs_id = node->operand(1);  // Mul(a, b)
                auto* rhs = builder.getNode(rhs_id);

                // Check: Sub(c, Mul(a, b)) -> FNMA(a, b, c)
                // FNMA computes: -(a*b) + c = c - a*b
                if (rhs && !rhs->isDead() && rhs->opCode() == OpCode::kMul &&
                    rhs->numOperands() == 2 && use_counts[rhs_id.id] == 1) {

                    ValueId a = rhs->operand(0);
                    ValueId b = rhs->operand(1);
                    ValueId c = lhs_id;

                    // Create FNMA node: -(a*b) + c
                    // We need to create this manually since builder may not have fnma()
                    // Use the three-operand pattern similar to FMA
                    auto fnma_type = node->type();
                    auto fnma_node =
                        new IRNode(OpCode::kFnma, fnma_type,
                                   ValueId{static_cast<uint32_t>(builder.nodes().size())});
                    fnma_node->addOperand(a);
                    fnma_node->addOperand(b);
                    fnma_node->addOperand(c);
                    builder.mutableNodes().push_back(fnma_node);
                    ValueId fnma_id = fnma_node->id();

                    if (fnma_id.isValid()) {
                        builder.replaceAllUses(node->id(), fnma_id);
                        node->markDead();
                        rhs->markDead();
                        ++fused_count;

                        spdlog::debug("Fusion: Sub(%{}, Mul(%{}, %{})) -> FNMA (node %{})", c.id,
                                      a.id, b.id, fnma_id.id);
                        continue;
                    }
                }
            }

            // Pattern: Sub(Mul(a, b), c) -> FMS: (a*b) - c
            // This is equivalent to FNMA(a, b, -c) or we can use FMA with negated c
            // For simplicity, we transform to: Neg(FNMA(a, b, Neg(Mul(a,b))))
            // Actually simpler: a*b - c = -(-(a*b) + c) = -FNMA(a, b, c) when c is not negated
            // Best approach: a*b - c = a*b + (-c) = FMA(a, b, -c)
            if (node->opCode() == OpCode::kSub && node->numOperands() == 2) {
                ValueId lhs_id = node->operand(0);  // Mul(a, b)
                ValueId rhs_id = node->operand(1);  // c
                auto* lhs = builder.getNode(lhs_id);

                // Check: Sub(Mul(a, b), c) -> FMA(a, b, -c)
                if (lhs && !lhs->isDead() && lhs->opCode() == OpCode::kMul &&
                    lhs->numOperands() == 2 && use_counts[lhs_id.id] == 1) {

                    ValueId a = lhs->operand(0);
                    ValueId b = lhs->operand(1);
                    ValueId c = rhs_id;

                    // Create Neg(c) first
                    ValueId neg_c = builder.neg(c);
                    if (neg_c.isValid()) {
                        // Create FMA(a, b, -c) = a*b + (-c) = a*b - c
                        ValueId fma_id = builder.fma(a, b, neg_c);
                        if (fma_id.isValid()) {
                            builder.replaceAllUses(node->id(), fma_id);
                            node->markDead();
                            lhs->markDead();
                            ++fused_count;

                            spdlog::debug(
                                "Fusion: Sub(Mul(%{}, %{}), %{}) -> FMA(a, b, Neg(c)) (node %{})",
                                a.id, b.id, c.id, fma_id.id);
                            continue;
                        }
                    }
                }
            }
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

// =============================================================================
// Loop Invariant Code Motion (LICM) Pass - Level 3
// =============================================================================
//
// Hoists computations that don't depend on loop variables outside of loops.
// Example: for(i) { result[i] = data[i] * scale * factor; }
//       -> temp = scale * factor; for(i) { result[i] = data[i] * temp; }
//
class LoopInvariantCodeMotionPass {
  public:
    size_t run(IRBuilder& builder, ValueId output) {
        size_t hoisted_count = 0;

        // Step 1: Find all loop nodes (kFor, kWhile)
        std::vector<IRNode*> loop_nodes;
        for (auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;
            if (node->opCode() == OpCode::kFor || node->opCode() == OpCode::kWhile) {
                loop_nodes.push_back(node);
            }
        }

        if (loop_nodes.empty()) {
            return 0;  // No loops to optimize
        }

        // Step 2: Build def-use chains - which nodes define each value
        std::unordered_map<uint32_t, IRNode*> definitions;
        for (auto* node : builder.nodes()) {
            if (node && !node->isDead()) {
                definitions[node->id().id] = node;
            }
        }

        // Step 3: For each loop, find invariant computations
        for (auto* loop : loop_nodes) {
            // Get loop body nodes (operands of the loop node represent body computations)
            std::unordered_set<uint32_t> loop_body_ids;
            collectLoopBodyNodes(builder, loop, loop_body_ids);

            // Find invariant nodes: nodes whose operands are all defined outside the loop
            // or are themselves loop-invariant
            std::unordered_set<uint32_t> invariant_ids;
            bool changed = true;
            constexpr size_t kMaxLICMIterations = 100;
            size_t licm_iterations = 0;
            while (changed && licm_iterations < kMaxLICMIterations) {
                ++licm_iterations;
                changed = false;
                for (uint32_t id : loop_body_ids) {
                    if (invariant_ids.count(id))
                        continue;

                    auto* node = definitions[id];
                    if (!node || node->isDead())
                        continue;

                    // Skip non-pure operations - they can't be hoisted
                    if (!isPureOp(node->opCode()))
                        continue;

                    // Check if all operands are either:
                    // 1. Defined outside the loop, OR
                    // 2. Already marked as invariant
                    bool all_invariant = true;
                    for (const auto& operand : node->operands()) {
                        bool outside_loop = (loop_body_ids.find(operand.id) == loop_body_ids.end());
                        bool is_invariant = (invariant_ids.find(operand.id) != invariant_ids.end());
                        if (!outside_loop && !is_invariant) {
                            all_invariant = false;
                            break;
                        }
                    }

                    if (all_invariant) {
                        invariant_ids.insert(id);
                        changed = true;
                    }
                }
            }

            // Step 4: Hoist invariant nodes by marking them to be computed before the loop
            // In SSA form, we can achieve this by reordering nodes in the IR
            for (uint32_t inv_id : invariant_ids) {
                auto* inv_node = definitions[inv_id];
                if (inv_node) {
                    // Mark node as hoisted (for code generation to handle)
                    inv_node->setIntAttr("hoisted", 1);
                    ++hoisted_count;
                    spdlog::debug("LICM: Hoisted node %{} ({}) out of loop %{}", inv_id,
                                  opCodeName(inv_node->opCode()), loop->id().id);
                }
            }
        }

        return hoisted_count;
    }

  private:
    // Collect all nodes that are part of the loop body
    void collectLoopBodyNodes(const IRBuilder& builder, const IRNode* loop,
                              std::unordered_set<uint32_t>& body_ids) {
        // Loop operands typically reference the body computation graph
        // We do a forward traversal from loop operands
        std::vector<ValueId> worklist;
        for (const auto& op : loop->operands()) {
            worklist.push_back(op);
        }

        while (!worklist.empty()) {
            ValueId id = worklist.back();
            worklist.pop_back();

            if (body_ids.count(id.id))
                continue;

            body_ids.insert(id.id);

            // Add users of this node to worklist
            for (const auto* node : builder.nodes()) {
                if (!node || node->isDead())
                    continue;
                for (const auto& operand : node->operands()) {
                    if (operand.id == id.id && body_ids.find(node->id().id) == body_ids.end()) {
                        worklist.push_back(node->id());
                    }
                }
            }
        }
    }

    // Check if operation is pure (no side effects)
    bool isPureOp(OpCode op) {
        switch (op) {
        case OpCode::kLoad:
        case OpCode::kStore:
        case OpCode::kAlloc:
        case OpCode::kCall:
        case OpCode::kScatter:
            return false;
        default:
            return true;
        }
    }
};

// =============================================================================
// Induction Variable Optimization Pass - Level 3
// =============================================================================
//
// Optimizes loop induction variables by:
// 1. Detecting i * stride patterns in loops
// 2. Replacing with accumulated increments: base += stride
//
// Example: for(i=0; i<n; i++) { ptr = base + i * 8; }
//       -> ptr = base; for(i=0; i<n; i++) { ...; ptr += 8; }
//
class InductionVariableOptimizationPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t optimized_count = 0;

        // Find multiplication patterns that could be induction variables
        // Pattern: Mul(loop_index, constant_stride)
        for (auto* node : builder.mutableNodes()) {
            if (!node || node->isDead())
                continue;

            if (node->opCode() != OpCode::kMul || node->numOperands() != 2)
                continue;

            auto* lhs = builder.getNode(node->operand(0));
            auto* rhs = builder.getNode(node->operand(1));

            // Check for pattern: var * constant where var is a loop counter
            IRNode* var_node = nullptr;
            IRNode* const_node = nullptr;

            if (isConstant(rhs) && isLoopInductionCandidate(lhs)) {
                var_node = lhs;
                const_node = rhs;
            } else if (isConstant(lhs) && isLoopInductionCandidate(rhs)) {
                var_node = rhs;
                const_node = lhs;
            }

            if (var_node && const_node) {
                // Mark this as an induction variable pattern for code generation
                double stride = getConstantFloat(const_node);
                node->setIntAttr("induction_var", 1);
                node->setFloatAttr("stride", stride);
                ++optimized_count;

                spdlog::debug("InductionVar: Marked %{} as induction variable with stride {}",
                              node->id().id, stride);
            }
        }

        // Also optimize: x * 2 -> x + x (shift-strength reduction for powers of 2)
        for (auto* node : builder.mutableNodes()) {
            if (!node || node->isDead())
                continue;

            if (node->opCode() != OpCode::kMul || node->numOperands() != 2)
                continue;

            auto* lhs = builder.getNode(node->operand(0));
            auto* rhs = builder.getNode(node->operand(1));

            ValueId var_id = ValueId::invalid();
            double const_val = 0.0;

            if (isConstant(rhs) && lhs && !lhs->isDead()) {
                var_id = node->operand(0);
                const_val = getConstantFloat(rhs);
            } else if (isConstant(lhs) && rhs && !rhs->isDead()) {
                var_id = node->operand(1);
                const_val = getConstantFloat(lhs);
            }

            // x * 2 -> x + x (avoids multiplication unit)
            if (var_id.isValid() && const_val == 2.0) {
                ValueId add_id = builder.add(var_id, var_id);
                if (add_id.isValid()) {
                    builder.replaceAllUses(node->id(), add_id);
                    node->markDead();
                    ++optimized_count;
                    spdlog::debug("InductionVar: %{} * 2 -> %{} + %{}", var_id.id, var_id.id,
                                  var_id.id);
                }
            }

            // x * 4 -> (x + x) + (x + x) or x << 2 (marked for codegen)
            if (var_id.isValid() && const_val == 4.0) {
                ValueId x2 = builder.add(var_id, var_id);
                if (x2.isValid()) {
                    ValueId x4 = builder.add(x2, x2);
                    if (x4.isValid()) {
                        builder.replaceAllUses(node->id(), x4);
                        node->markDead();
                        ++optimized_count;
                        spdlog::debug("InductionVar: %{} * 4 -> (%{} + %{}) + (...)", var_id.id,
                                      var_id.id, var_id.id);
                    }
                }
            }
        }

        return optimized_count;
    }

  private:
    bool isLoopInductionCandidate(const IRNode* node) {
        if (!node)
            return false;
        // A node is a loop induction candidate if it's a Phi node or
        // comes from a loop iteration counter (kFor body produces these)
        return node->opCode() == OpCode::kPhi || node->hasAttr("loop_index");
    }
};

// =============================================================================
// Escape Analysis Pass - Level 3
// =============================================================================
//
// Analyzes which allocations can be eliminated because they don't escape
// their defining scope. Non-escaping allocations can be:
// 1. Completely eliminated if unused
// 2. Scalar-replaced if only individual elements are accessed
// 3. Stack-allocated instead of heap-allocated
//
class EscapeAnalysisPass {
  public:
    size_t run(IRBuilder& builder, ValueId output) {
        size_t eliminated_count = 0;

        // Step 1: Find all allocation nodes
        std::vector<IRNode*> alloc_nodes;
        for (auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;
            if (node->opCode() == OpCode::kAlloc) {
                alloc_nodes.push_back(node);
            }
        }

        if (alloc_nodes.empty()) {
            return 0;  // No allocations to analyze
        }

        // Pre-build use map once for O(N+E) analysis instead of O(N²)
        UseMap use_map = buildUseMap(builder);

        // Step 2: For each allocation, check if it escapes
        for (auto* alloc : alloc_nodes) {
            EscapeState state = analyzeEscape(builder, alloc, output, use_map);

            switch (state) {
            case EscapeState::kNoEscape:
                // Allocation doesn't escape - can be completely eliminated
                // if it has no uses, or stack-allocated otherwise
                if (!hasNonTrivialUses(use_map, alloc->id())) {
                    alloc->markDead();
                    ++eliminated_count;
                    spdlog::debug("EscapeAnalysis: Eliminated unused allocation %{}",
                                  alloc->id().id);
                } else {
                    // Mark for stack allocation in codegen
                    alloc->setIntAttr("stack_allocate", 1);
                    spdlog::debug("EscapeAnalysis: Marked %{} for stack allocation",
                                  alloc->id().id);
                }
                break;

            case EscapeState::kArgEscape:
                // Only escapes through function arguments - may still be optimizable
                alloc->setIntAttr("arg_escape", 1);
                spdlog::debug("EscapeAnalysis: %{} escapes through args only", alloc->id().id);
                break;

            case EscapeState::kGlobalEscape:
                // Escapes globally - cannot optimize
                break;
            }
        }

        // Step 3: Scalar replacement for non-escaping allocations with known element access
        for (auto* alloc : alloc_nodes) {
            if (alloc->isDead())
                continue;
            if (!alloc->hasAttr("stack_allocate"))
                continue;

            size_t replaced = tryScalarReplacement(builder, alloc);
            eliminated_count += replaced;
        }

        return eliminated_count;
    }

  private:
    enum class EscapeState { kNoEscape, kArgEscape, kGlobalEscape };

    // Pre-built use map for O(N+E) escape analysis instead of O(N²)
    using UseMap = std::unordered_map<uint32_t, std::vector<const IRNode*>>;

    UseMap buildUseMap(const IRBuilder& builder) {
        UseMap use_map;
        for (const auto* node : builder.nodes()) {
            if (!node || node->isDead())
                continue;
            for (const auto& operand : node->operands()) {
                use_map[operand.id].push_back(node);
            }
        }
        return use_map;
    }

    EscapeState analyzeEscape(const IRBuilder& builder, const IRNode* alloc, ValueId output,
                              const UseMap& use_map) {
        // Track all uses of the allocation
        std::unordered_set<uint32_t> visited;
        std::vector<ValueId> worklist;
        worklist.push_back(alloc->id());

        EscapeState state = EscapeState::kNoEscape;

        // Safety limit to prevent infinite loops on pathological graphs
        constexpr size_t kMaxWorklistIterations = 10000;
        size_t iterations = 0;

        while (!worklist.empty() && iterations < kMaxWorklistIterations) {
            ++iterations;
            ValueId id = worklist.back();
            worklist.pop_back();

            if (visited.count(id.id))
                continue;
            visited.insert(id.id);

            // Use pre-built use map for O(uses) lookup instead of O(N) scan
            auto it = use_map.find(id.id);
            if (it == use_map.end())
                continue;

            for (const auto* node : it->second) {
                if (!node || node->isDead())
                    continue;

                // Check escape conditions
                switch (node->opCode()) {
                case OpCode::kStore:
                    // Storing TO the allocation is fine, storing the allocation
                    // pointer itself to another location causes escape
                    if (node->numOperands() > 1 && node->operand(1).id == id.id) {
                        // Value being stored is our allocation - it escapes
                        state = std::max(state, EscapeState::kGlobalEscape);
                    }
                    break;

                case OpCode::kCall:
                    // Passed to a function - escapes through args
                    state = std::max(state, EscapeState::kArgEscape);
                    break;

                case OpCode::kReturn:
                    // Returned from function - global escape
                    state = std::max(state, EscapeState::kGlobalEscape);
                    break;

                case OpCode::kLoad:
                    // Loading from allocation is fine - doesn't cause escape
                    break;

                default:
                    // Other uses propagate through the worklist
                    if (!visited.count(node->id().id)) {
                        worklist.push_back(node->id());
                    }
                    break;
                }
            }
        }

        // Conservative fallback if we hit the iteration limit
        if (iterations >= kMaxWorklistIterations) {
            spdlog::warn("EscapeAnalysis: Hit iteration limit for allocation %{}", alloc->id().id);
            return EscapeState::kGlobalEscape;
        }

        // Check if allocation reaches output
        if (visited.count(output.id)) {
            state = std::max(state, EscapeState::kGlobalEscape);
        }

        return state;
    }

    bool hasNonTrivialUses(const UseMap& use_map, ValueId alloc_id) {
        auto it = use_map.find(alloc_id.id);
        if (it == use_map.end())
            return false;
        // Check if any uses are still alive
        for (const auto* node : it->second) {
            if (node && !node->isDead()) {
                return true;
            }
        }
        return false;
    }

    // Try to replace allocation with scalar variables when access patterns are known
    size_t tryScalarReplacement(IRBuilder& builder, IRNode* alloc) {
        // Collect all loads and stores to this allocation
        std::vector<IRNode*> loads, stores;
        for (auto* node : builder.mutableNodes()) {
            if (!node || node->isDead())
                continue;

            if (node->opCode() == OpCode::kLoad && node->numOperands() >= 1 &&
                node->operand(0).id == alloc->id().id) {
                loads.push_back(node);
            }
            if (node->opCode() == OpCode::kStore && node->numOperands() >= 1 &&
                node->operand(0).id == alloc->id().id) {
                stores.push_back(node);
            }
        }

        // For now, only handle simple cases: single store followed by single load
        // This is common for temporary results in expressions
        if (stores.size() == 1 && loads.size() == 1) {
            auto* store = stores[0];
            auto* load = loads[0];

            // Get the value being stored
            if (store->numOperands() >= 2) {
                ValueId stored_value = store->operand(1);

                // Replace load with the stored value directly
                builder.replaceAllUses(load->id(), stored_value);
                load->markDead();
                store->markDead();
                alloc->markDead();

                spdlog::debug("EscapeAnalysis: Scalar-replaced allocation %{}", alloc->id().id);
                return 1;
            }
        }

        return 0;
    }
};

// =============================================================================
// Memory Access Coalescing Pass - Level 3
// =============================================================================
//
// Combines adjacent memory operations into wider loads/stores.
// Example: 4 separate float loads from consecutive addresses
//       -> 1 SIMD vector load (4x faster)
//
class MemoryCoalescingPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t coalesced_count = 0;

        // Group loads by base pointer
        std::unordered_map<uint32_t, std::vector<IRNode*>> loads_by_base;
        std::unordered_map<uint32_t, std::vector<IRNode*>> stores_by_base;

        for (auto* node : builder.mutableNodes()) {
            if (!node || node->isDead())
                continue;

            if (node->opCode() == OpCode::kLoad && node->numOperands() >= 1) {
                ValueId base = node->operand(0);
                loads_by_base[base.id].push_back(node);
            }
            if (node->opCode() == OpCode::kStore && node->numOperands() >= 1) {
                ValueId base = node->operand(0);
                stores_by_base[base.id].push_back(node);
            }
        }

        // Try to coalesce consecutive loads
        for (auto& [base_id, loads] : loads_by_base) {
            if (loads.size() < 2)
                continue;

            // Sort by offset attribute (if available)
            std::sort(loads.begin(), loads.end(), [](IRNode* a, IRNode* b) {
                return a->intAttr("offset", 0) < b->intAttr("offset", 0);
            });

            // Find consecutive sequences
            size_t i = 0;
            while (i < loads.size()) {
                // Find run of consecutive loads
                size_t run_start = i;
                size_t run_end = i + 1;

                int64_t expected_offset = loads[run_start]->intAttr("offset", 0);
                int64_t element_size = getElementSize(loads[run_start]->type().scalarType());

                while (run_end < loads.size()) {
                    int64_t actual_offset = loads[run_end]->intAttr("offset", 0);
                    expected_offset += element_size;
                    if (actual_offset != expected_offset) {
                        break;
                    }
                    ++run_end;
                }

                size_t run_length = run_end - run_start;

                // Coalesce runs of 4+ consecutive loads into vector loads
                if (run_length >= 4) {
                    // Mark first load as a vector load
                    auto* first_load = loads[run_start];
                    first_load->setIntAttr("vector_width", static_cast<int64_t>(run_length));
                    first_load->setIntAttr("coalesced", 1);

                    // Mark remaining loads as dead (will be extracted from vector)
                    for (size_t j = run_start + 1; j < run_end; ++j) {
                        loads[j]->setIntAttr("extract_from",
                                             static_cast<int64_t>(first_load->id().id));
                        loads[j]->setIntAttr("extract_index", static_cast<int64_t>(j - run_start));
                    }

                    coalesced_count += run_length - 1;
                    spdlog::debug("MemoryCoalescing: Coalesced {} loads from base %{}", run_length,
                                  base_id);
                }

                i = run_end;
            }
        }

        // Similar logic for stores
        for (auto& [base_id, stores] : stores_by_base) {
            if (stores.size() < 2)
                continue;

            std::sort(stores.begin(), stores.end(), [](IRNode* a, IRNode* b) {
                return a->intAttr("offset", 0) < b->intAttr("offset", 0);
            });

            size_t i = 0;
            while (i < stores.size()) {
                size_t run_start = i;
                size_t run_end = i + 1;

                int64_t expected_offset = stores[run_start]->intAttr("offset", 0);
                int64_t element_size = getElementSize(stores[run_start]->type().scalarType());

                while (run_end < stores.size()) {
                    int64_t actual_offset = stores[run_end]->intAttr("offset", 0);
                    expected_offset += element_size;
                    if (actual_offset != expected_offset) {
                        break;
                    }
                    ++run_end;
                }

                size_t run_length = run_end - run_start;

                if (run_length >= 4) {
                    auto* first_store = stores[run_start];
                    first_store->setIntAttr("vector_width", static_cast<int64_t>(run_length));
                    first_store->setIntAttr("coalesced", 1);

                    for (size_t j = run_start + 1; j < run_end; ++j) {
                        stores[j]->setIntAttr("combine_into",
                                              static_cast<int64_t>(first_store->id().id));
                        stores[j]->setIntAttr("combine_index", static_cast<int64_t>(j - run_start));
                    }

                    coalesced_count += run_length - 1;
                    spdlog::debug("MemoryCoalescing: Coalesced {} stores to base %{}", run_length,
                                  base_id);
                }

                i = run_end;
            }
        }

        return coalesced_count;
    }

  private:
    int64_t getElementSize(ScalarType type) {
        switch (type) {
        case ScalarType::kFloat16:
        case ScalarType::kBFloat16:
            return 2;
        case ScalarType::kFloat32:
        case ScalarType::kInt32:
            return 4;
        case ScalarType::kFloat64:
        case ScalarType::kInt64:
            return 8;
        default:
            return 4;
        }
    }
};

// =============================================================================
// Algebraic Simplification Pass - Level 3
// =============================================================================
//
// Additional algebraic identities beyond basic strength reduction:
// - a - a -> 0
// - a / a -> 1 (for non-zero a)
// - min(a, a) -> a
// - max(a, a) -> a
// - a & a -> a (bitwise)
// - a | a -> a (bitwise)
// - sqrt(x*x) -> abs(x)
// - exp(log(x)) -> x
// - log(exp(x)) -> x
//
class AlgebraicSimplificationPass {
  public:
    size_t run(IRBuilder& builder) {
        size_t simplified_count = 0;

        for (auto* node : builder.mutableNodes()) {
            if (!node || node->isDead())
                continue;

            // a - a -> 0
            if (node->opCode() == OpCode::kSub && node->numOperands() == 2) {
                if (node->operand(0).id == node->operand(1).id) {
                    ValueId zero;
                    if (node->type().scalarType() == ScalarType::kFloat32) {
                        zero = builder.constant(0.0f);
                    } else {
                        zero = builder.constant(0.0);
                    }
                    if (zero.isValid()) {
                        builder.replaceAllUses(node->id(), zero);
                        node->markDead();
                        ++simplified_count;
                        spdlog::debug("AlgSimp: %{} - %{} -> 0", node->operand(0).id,
                                      node->operand(1).id);
                        continue;
                    }
                }
            }

            // a / a -> 1 (assuming a != 0)
            if (node->opCode() == OpCode::kDiv && node->numOperands() == 2) {
                if (node->operand(0).id == node->operand(1).id) {
                    ValueId one;
                    if (node->type().scalarType() == ScalarType::kFloat32) {
                        one = builder.constant(1.0f);
                    } else {
                        one = builder.constant(1.0);
                    }
                    if (one.isValid()) {
                        builder.replaceAllUses(node->id(), one);
                        node->markDead();
                        ++simplified_count;
                        spdlog::debug("AlgSimp: %{} / %{} -> 1", node->operand(0).id,
                                      node->operand(1).id);
                        continue;
                    }
                }
            }

            // min(a, a) -> a, max(a, a) -> a
            if ((node->opCode() == OpCode::kMin || node->opCode() == OpCode::kMax) &&
                node->numOperands() == 2) {
                if (node->operand(0).id == node->operand(1).id) {
                    builder.replaceAllUses(node->id(), node->operand(0));
                    node->markDead();
                    ++simplified_count;
                    spdlog::debug("AlgSimp: {}(%{}, %{}) -> %{}", opCodeName(node->opCode()),
                                  node->operand(0).id, node->operand(1).id, node->operand(0).id);
                    continue;
                }
            }

            // sqrt(x*x) -> abs(x)
            if (node->opCode() == OpCode::kSqrt && node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));
                if (operand && operand->opCode() == OpCode::kMul && operand->numOperands() == 2) {
                    if (operand->operand(0).id == operand->operand(1).id) {
                        ValueId abs_id = builder.abs(operand->operand(0));
                        if (abs_id.isValid()) {
                            builder.replaceAllUses(node->id(), abs_id);
                            node->markDead();
                            ++simplified_count;
                            spdlog::debug("AlgSimp: sqrt(%{} * %{}) -> abs(%{})",
                                          operand->operand(0).id, operand->operand(1).id,
                                          operand->operand(0).id);
                            continue;
                        }
                    }
                }
            }

            // exp(log(x)) -> x
            if (node->opCode() == OpCode::kExp && node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));
                if (operand && operand->opCode() == OpCode::kLog && operand->numOperands() == 1) {
                    builder.replaceAllUses(node->id(), operand->operand(0));
                    node->markDead();
                    ++simplified_count;
                    spdlog::debug("AlgSimp: exp(log(%{})) -> %{}", operand->operand(0).id,
                                  operand->operand(0).id);
                    continue;
                }
            }

            // log(exp(x)) -> x
            if (node->opCode() == OpCode::kLog && node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));
                if (operand && operand->opCode() == OpCode::kExp && operand->numOperands() == 1) {
                    builder.replaceAllUses(node->id(), operand->operand(0));
                    node->markDead();
                    ++simplified_count;
                    spdlog::debug("AlgSimp: log(exp(%{})) -> %{}", operand->operand(0).id,
                                  operand->operand(0).id);
                    continue;
                }
            }

            // abs(abs(x)) -> abs(x)
            if (node->opCode() == OpCode::kAbs && node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));
                if (operand && operand->opCode() == OpCode::kAbs) {
                    builder.replaceAllUses(node->id(), node->operand(0));
                    node->markDead();
                    ++simplified_count;
                    spdlog::debug("AlgSimp: abs(abs(%{})) -> abs(%{})", operand->operand(0).id,
                                  operand->operand(0).id);
                    continue;
                }
            }

            // abs(neg(x)) -> abs(x)
            if (node->opCode() == OpCode::kAbs && node->numOperands() == 1) {
                auto* operand = builder.getNode(node->operand(0));
                if (operand && operand->opCode() == OpCode::kNeg && operand->numOperands() == 1) {
                    ValueId abs_id = builder.abs(operand->operand(0));
                    if (abs_id.isValid()) {
                        builder.replaceAllUses(node->id(), abs_id);
                        node->markDead();
                        ++simplified_count;
                        spdlog::debug("AlgSimp: abs(neg(%{})) -> abs(%{})", operand->operand(0).id,
                                      operand->operand(0).id);
                        continue;
                    }
                }
            }
        }

        return simplified_count;
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
    size_t total_alg_simplified = 0;
    size_t total_indvar = 0;
    size_t total_licm = 0;
    size_t total_escape = 0;
    size_t total_coalesced = 0;

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
            // Algebraic simplification (a-a=0, exp(log(x))=x, etc.)
            AlgebraicSimplificationPass algsimp;
            size_t alg_count = algsimp.run(builder);
            total_alg_simplified += alg_count;
            changed |= (alg_count > 0);

            // Induction variable optimization (x*2 -> x+x, etc.)
            InductionVariableOptimizationPass indvar;
            size_t indvar_count = indvar.run(builder);
            total_indvar += indvar_count;
            changed |= (indvar_count > 0);

            // Loop Invariant Code Motion
            LoopInvariantCodeMotionPass licm;
            size_t licm_count = licm.run(builder, output);
            total_licm += licm_count;
            changed |= (licm_count > 0);

            // Escape Analysis
            EscapeAnalysisPass escape;
            size_t escape_count = escape.run(builder, output);
            total_escape += escape_count;
            changed |= (escape_count > 0);

            // Memory Access Coalescing
            MemoryCoalescingPass coalesce;
            size_t coal_count = coalesce.run(builder);
            total_coalesced += coal_count;
            changed |= (coal_count > 0);
        }
    }

    // Final compaction to clean up any remaining dead nodes
    builder.compactNodes();

    if (level >= 3) {
        spdlog::info("Optimization completed in {} iterations: folded={}, eliminated={}, "
                     "reduced={}, fused={}, cse={}, alg_simplified={}, indvar={}, licm={}, "
                     "escape={}, coalesced={}",
                     iterations, total_folded, total_eliminated, total_reduced, total_fused,
                     total_cse, total_alg_simplified, total_indvar, total_licm, total_escape,
                     total_coalesced);
    } else {
        spdlog::info("Optimization completed in {} iterations: folded={}, eliminated={}, "
                     "reduced={}, fused={}, cse={}",
                     iterations, total_folded, total_eliminated, total_reduced, total_fused,
                     total_cse);
    }

    return {};
}

}  // namespace ir
}  // namespace bud
