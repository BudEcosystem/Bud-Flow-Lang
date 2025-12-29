// =============================================================================
// Bud Flow Lang - Stencil Registry (Copy-and-Patch JIT)
// =============================================================================
//
// Pre-compiled code stencils for copy-and-patch JIT compilation.
// Each stencil is a template with "holes" that get patched at runtime.
//

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/jit/stencil.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <unordered_map>

namespace bud {
namespace jit {

// =============================================================================
// Stencil Registry
// =============================================================================

class StencilRegistry {
  public:
    static StencilRegistry& instance() {
        static StencilRegistry registry;
        return registry;
    }

    // Register a stencil
    void registerStencil(Stencil stencil) {
        auto key = makeKey(stencil.op, stencil.dtype);
        stencils_[key] = std::move(stencil);
        spdlog::debug("Registered stencil: {} ({})", stencils_[key].name, key);
    }

    // Look up a stencil
    const Stencil* find(ir::OpCode op, ScalarType dtype) const {
        auto key = makeKey(op, dtype);
        auto it = stencils_.find(key);
        if (it != stencils_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    // Check if a stencil exists
    bool has(ir::OpCode op, ScalarType dtype) const { return find(op, dtype) != nullptr; }

    // Get all registered stencils
    size_t count() const { return stencils_.size(); }

  private:
    StencilRegistry() {
        // Initialize with built-in stencils
        initializeBuiltinStencils();
    }

    void initializeBuiltinStencils();

    static std::string makeKey(ir::OpCode op, ScalarType dtype) {
        return std::string(ir::opCodeName(op)) + "_" + std::string(scalarTypeName(dtype));
    }

    std::unordered_map<std::string, Stencil> stencils_;
};

// =============================================================================
// Built-in Stencil Initialization
// =============================================================================
//
// NOTE: Real implementation would generate these from Highway code
// compiled with -emit-llvm or similar, then extract the machine code.
// For now, we just register placeholder stencils.
//

void StencilRegistry::initializeBuiltinStencils() {
    // Placeholder stencils - real implementation would have actual machine code

    // Add operation (float32)
    {
        Stencil s;
        s.name = "add_f32";
        s.op = ir::OpCode::kAdd;
        s.dtype = ScalarType::kFloat32;
        // Placeholder - would be actual x86-64/ARM64 code
        s.code = {0x90};  // NOP placeholder
        s.holes = {
            {0, Stencil::Hole::kAbsAddress64, "input_a"},
            {0, Stencil::Hole::kAbsAddress64, "input_b"},
            {0, Stencil::Hole::kAbsAddress64, "output"},
            {0, Stencil::Hole::kImmediate64, "count"},
        };
        registerStencil(std::move(s));
    }

    // Multiply operation (float32)
    {
        Stencil s;
        s.name = "mul_f32";
        s.op = ir::OpCode::kMul;
        s.dtype = ScalarType::kFloat32;
        s.code = {0x90};
        s.holes = {
            {0, Stencil::Hole::kAbsAddress64, "input_a"},
            {0, Stencil::Hole::kAbsAddress64, "input_b"},
            {0, Stencil::Hole::kAbsAddress64, "output"},
            {0, Stencil::Hole::kImmediate64, "count"},
        };
        registerStencil(std::move(s));
    }

    // FMA operation (float32)
    {
        Stencil s;
        s.name = "fma_f32";
        s.op = ir::OpCode::kFma;
        s.dtype = ScalarType::kFloat32;
        s.code = {0x90};
        s.holes = {
            {0, Stencil::Hole::kAbsAddress64, "input_a"},
            {0, Stencil::Hole::kAbsAddress64, "input_b"},
            {0, Stencil::Hole::kAbsAddress64, "input_c"},
            {0, Stencil::Hole::kAbsAddress64, "output"},
            {0, Stencil::Hole::kImmediate64, "count"},
        };
        registerStencil(std::move(s));
    }

    // Exp operation (float32)
    {
        Stencil s;
        s.name = "exp_f32";
        s.op = ir::OpCode::kExp;
        s.dtype = ScalarType::kFloat32;
        s.code = {0x90};
        s.holes = {
            {0, Stencil::Hole::kAbsAddress64, "input"},
            {0, Stencil::Hole::kAbsAddress64, "output"},
            {0, Stencil::Hole::kImmediate64, "count"},
        };
        registerStencil(std::move(s));
    }

    // Reduce sum (float32)
    {
        Stencil s;
        s.name = "reduce_sum_f32";
        s.op = ir::OpCode::kReduceSum;
        s.dtype = ScalarType::kFloat32;
        s.code = {0x90};
        s.holes = {
            {0, Stencil::Hole::kAbsAddress64, "input"},
            {0, Stencil::Hole::kAbsAddress64, "output"},
            {0, Stencil::Hole::kImmediate64, "count"},
        };
        registerStencil(std::move(s));
    }

    spdlog::info("StencilRegistry: {} stencils registered", count());
}

// =============================================================================
// Public API
// =============================================================================

const Stencil* findStencil(ir::OpCode op, ScalarType dtype) {
    return StencilRegistry::instance().find(op, dtype);
}

bool hasStencil(ir::OpCode op, ScalarType dtype) {
    return StencilRegistry::instance().has(op, dtype);
}

}  // namespace jit
}  // namespace bud
