#pragma once

// =============================================================================
// Bud Flow Lang - Stencil Definitions (Copy-and-Patch JIT)
// =============================================================================

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/type_system.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bud {
namespace jit {

// =============================================================================
// Stencil Definition
// =============================================================================

struct Stencil {
    std::string name;
    ir::OpCode op;
    ScalarType dtype;

    // Machine code template
    std::vector<uint8_t> code;

    // Hole positions (offsets into code where patching is needed)
    struct Hole {
        size_t offset;  // Byte offset in code
        enum Kind {
            kImmediate32,   // 32-bit immediate value
            kImmediate64,   // 64-bit immediate value
            kRelAddress32,  // 32-bit relative address
            kAbsAddress64,  // 64-bit absolute address
        } kind;
        std::string name;  // Hole identifier (e.g., "input_ptr", "output_ptr")
    };
    std::vector<Hole> holes;

    // Size requirements
    size_t alignment = 16;
    size_t code_size() const { return code.size(); }
};

// =============================================================================
// Stencil Registry API
// =============================================================================

// Find a stencil for a given operation and data type
const Stencil* findStencil(ir::OpCode op, ScalarType dtype);

// Check if a stencil exists
bool hasStencil(ir::OpCode op, ScalarType dtype);

}  // namespace jit
}  // namespace bud
