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

// Get the total number of registered stencils
size_t stencilCount();

// Get a function pointer to the Highway implementation of an operation
// Returns nullptr if not available
void* getHwyFunctionPtr(ir::OpCode op, ScalarType dtype);

// =============================================================================
// JIT Compiler API
// =============================================================================

// Initialize the JIT compiler
Result<void> initializeCompiler();

// Shutdown the JIT compiler
void shutdownCompiler();

// JIT statistics
struct JitStats {
    size_t total_compilations = 0;
    uint64_t total_compile_time_us = 0;
    size_t memory_used = 0;
    size_t memory_remaining = 0;
    size_t cache_size = 0;
    size_t stencil_count = 0;
};

// Get JIT statistics
JitStats getJitStats();

// Execute a binary operation using JIT-compiled code
Result<void> executeJitBinaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input_a,
                                const void* input_b, size_t count);

// Execute a unary operation using JIT-compiled code
Result<void> executeJitUnaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input,
                               size_t count);

// Execute FMA operation using JIT-compiled code
Result<void> executeJitFmaOp(ScalarType dtype, void* output, const void* input_a,
                             const void* input_b, const void* input_c, size_t count);

}  // namespace jit
}  // namespace bud
