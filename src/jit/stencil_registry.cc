// =============================================================================
// Bud Flow Lang - Stencil Registry (Copy-and-Patch JIT)
// =============================================================================
//
// Pre-compiled code stencils for copy-and-patch JIT compilation.
// Each stencil is a template with "holes" that get patched at runtime.
//
// The stencils use a unified calling convention:
//   void stencil(void** args)
// where args contains:
//   - args[0]: output pointer
//   - args[1..n-1]: input pointers
//   - args[n]: count (size_t)
//   - args[n+1]: (optional) function pointer for dispatch
//
// Architecture-specific machine code is generated for:
//   - x86-64 (SSE/AVX/AVX-512)
//   - ARM64 (NEON/SVE)
//
// =============================================================================

#include "bud_flow_lang/codegen/hwy_ops.h"
#include "bud_flow_lang/common.h"
#include "bud_flow_lang/jit/stencil.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <unordered_map>

namespace bud {
namespace jit {

// =============================================================================
// Architecture Detection
// =============================================================================

enum class Architecture { kX86_64, kARM64, kUnknown };

static Architecture detectArchitecture() {
#if defined(__x86_64__) || defined(_M_X64)
    return Architecture::kX86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Architecture::kARM64;
#else
    return Architecture::kUnknown;
#endif
}

// =============================================================================
// x86-64 Stencil Generator
// =============================================================================
//
// x86-64 System V AMD64 ABI calling convention:
//   - Arguments: RDI, RSI, RDX, RCX, R8, R9, then stack
//   - Return: RAX
//   - Callee-saved: RBX, RBP, R12-R15
//   - Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
//
// Our stencil signature: void stencil(void** args)
//   - RDI = args pointer on entry
//
// =============================================================================

namespace x86_64 {

// Helper to emit bytes
class CodeEmitter {
  public:
    void emit(uint8_t byte) { code_.push_back(byte); }
    void emit(std::initializer_list<uint8_t> bytes) {
        for (auto b : bytes) {
            code_.push_back(b);
        }
    }

    // Emit 32-bit immediate
    void emit32(uint32_t value) {
        emit(value & 0xFF);
        emit((value >> 8) & 0xFF);
        emit((value >> 16) & 0xFF);
        emit((value >> 24) & 0xFF);
    }

    // Emit 64-bit immediate
    void emit64(uint64_t value) {
        emit(value & 0xFF);
        emit((value >> 8) & 0xFF);
        emit((value >> 16) & 0xFF);
        emit((value >> 24) & 0xFF);
        emit((value >> 32) & 0xFF);
        emit((value >> 40) & 0xFF);
        emit((value >> 48) & 0xFF);
        emit((value >> 56) & 0xFF);
    }

    // Mark a hole at current position
    void markHole(const std::string& name, Stencil::Hole::Kind kind) {
        holes_.push_back({code_.size(), kind, name});
    }

    std::vector<uint8_t>& code() { return code_; }
    std::vector<Stencil::Hole>& holes() { return holes_; }

  private:
    std::vector<uint8_t> code_;
    std::vector<Stencil::Hole> holes_;
};

// Generate prologue: save callee-saved registers we'll use
static void emitPrologue(CodeEmitter& e) {
    // push rbx
    e.emit(0x53);
    // push rbp
    e.emit(0x55);
    // push r12
    e.emit({0x41, 0x54});
    // push r13
    e.emit({0x41, 0x55});
    // mov rbp, rsp
    e.emit({0x48, 0x89, 0xE5});
    // mov rbx, rdi  ; save args pointer in rbx
    e.emit({0x48, 0x89, 0xFB});
}

// Generate epilogue: restore callee-saved registers and return
static void emitEpilogue(CodeEmitter& e) {
    // pop r13
    e.emit({0x41, 0x5D});
    // pop r12
    e.emit({0x41, 0x5C});
    // pop rbp
    e.emit(0x5D);
    // pop rbx
    e.emit(0x5B);
    // ret
    e.emit(0xC3);
}

// Load from args[index] into register
// mov reg, [rbx + index*8]
static void emitLoadArg(CodeEmitter& e, int reg, int index) {
    // REX prefix for 64-bit operation
    uint8_t rex = 0x48;
    if (reg >= 8) {
        rex |= 0x04;  // REX.R
        reg -= 8;
    }

    // MOV reg, [rbx + disp8/disp32]
    e.emit(rex);
    e.emit(0x8B);

    int disp = index * 8;
    if (disp == 0) {
        // [rbx + 0] - use mod=00, rm=011 (rbx)
        e.emit(static_cast<uint8_t>((reg << 3) | 0x03));
    } else if (disp <= 127 && disp >= -128) {
        // [rbx + disp8] - use mod=01
        e.emit(static_cast<uint8_t>(0x40 | (reg << 3) | 0x03));
        e.emit(static_cast<uint8_t>(disp));
    } else {
        // [rbx + disp32] - use mod=10
        e.emit(static_cast<uint8_t>(0x80 | (reg << 3) | 0x03));
        e.emit32(static_cast<uint32_t>(disp));
    }
}

// Call through RAX (assumes function pointer loaded into RAX)
static void emitCallRax(CodeEmitter& e) {
    // call rax
    e.emit({0xFF, 0xD0});
}

// Move immediate 64-bit value into register (for patching)
static void emitMovImm64(CodeEmitter& e, int reg, const std::string& hole_name) {
    // REX.W + B8+rd
    uint8_t rex = 0x48;
    if (reg >= 8) {
        rex |= 0x01;  // REX.B
        reg -= 8;
    }
    e.emit(rex);
    e.emit(static_cast<uint8_t>(0xB8 + reg));
    e.markHole(hole_name, Stencil::Hole::kAbsAddress64);
    e.emit64(0);  // Placeholder for patching
}

// =============================================================================
// Binary Operation Stencil Generator
// =============================================================================
// Generates stencil for: void op(out, a, b, count)
// Args: args[0]=out, args[1]=a, args[2]=b, args[3]=count, args[4]=func_ptr

static Stencil generateBinaryOpStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // Load arguments for the function call
    // System V AMD64: rdi=out, rsi=a, rdx=b, rcx=count, r8=dtype (if dispatch)
    emitLoadArg(e, 7, 0);  // rdi = args[0] (out)
    emitLoadArg(e, 6, 1);  // rsi = args[1] (a)
    emitLoadArg(e, 2, 2);  // rdx = args[2] (b)
    emitLoadArg(e, 1, 3);  // rcx = args[3] (count)

    // Load function pointer into rax
    emitLoadArg(e, 0, 4);  // rax = args[4] (func_ptr)

    emitCallRax(e);
    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},   {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"},  {0, Stencil::Hole::kImmediate64, "count"},
        {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

// =============================================================================
// Unary Operation Stencil Generator
// =============================================================================
// Generates stencil for: void op(out, a, count)
// Args: args[0]=out, args[1]=a, args[2]=count, args[3]=func_ptr

static Stencil generateUnaryOpStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // Load arguments
    // System V AMD64: rdi=out, rsi=a, rdx=count
    emitLoadArg(e, 7, 0);  // rdi = args[0] (out)
    emitLoadArg(e, 6, 1);  // rsi = args[1] (a)
    emitLoadArg(e, 2, 2);  // rdx = args[2] (count)

    // Load function pointer into rax
    emitLoadArg(e, 0, 3);  // rax = args[3] (func_ptr)

    emitCallRax(e);
    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input"},
        {0, Stencil::Hole::kImmediate64, "count"},
        {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

// =============================================================================
// Ternary Operation Stencil Generator (FMA)
// =============================================================================
// Generates stencil for: void op(out, a, b, c, count)
// Args: args[0]=out, args[1]=a, args[2]=b, args[3]=c, args[4]=count, args[5]=func_ptr

static Stencil generateTernaryOpStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // Load arguments
    // System V AMD64: rdi=out, rsi=a, rdx=b, rcx=c, r8=count
    emitLoadArg(e, 7, 0);  // rdi = args[0] (out)
    emitLoadArg(e, 6, 1);  // rsi = args[1] (a)
    emitLoadArg(e, 2, 2);  // rdx = args[2] (b)
    emitLoadArg(e, 1, 3);  // rcx = args[3] (c)
    emitLoadArg(e, 8, 4);  // r8 = args[4] (count)

    // Load function pointer into rax
    emitLoadArg(e, 0, 5);  // rax = args[5] (func_ptr)

    emitCallRax(e);
    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},  {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"}, {0, Stencil::Hole::kAbsAddress64, "input_c"},
        {0, Stencil::Hole::kImmediate64, "count"},    {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

// =============================================================================
// Reduction Operation Stencil Generator
// =============================================================================
// Generates stencil for: scalar_t op(a, count) -> writes to out
// Args: args[0]=out, args[1]=a, args[2]=count, args[3]=func_ptr

static Stencil generateReductionStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // For reductions, we need to call a function that returns the reduction value
    // Then store it to the output location
    // Load arguments: rdi=input, rsi=count
    emitLoadArg(e, 7, 1);  // rdi = args[1] (input)
    emitLoadArg(e, 6, 2);  // rsi = args[2] (count)

    // Load function pointer into rax
    emitLoadArg(e, 0, 3);  // rax = args[3] (func_ptr)

    emitCallRax(e);

    // Result is in rax (for int) or xmm0 (for float)
    // Load output pointer
    // mov r12, [rbx + 0]
    emitLoadArg(e, 12, 0);  // r12 = args[0] (output)

    // For float32, result is in xmm0
    // movss [r12], xmm0
    if (dtype == ScalarType::kFloat32) {
        e.emit({0xF3, 0x41, 0x0F, 0x11, 0x04, 0x24});
    } else if (dtype == ScalarType::kFloat64) {
        // movsd [r12], xmm0
        e.emit({0xF2, 0x41, 0x0F, 0x11, 0x04, 0x24});
    } else {
        // For integers: mov [r12], rax
        e.emit({0x49, 0x89, 0x04, 0x24});
    }

    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input"},
        {0, Stencil::Hole::kImmediate64, "count"},
        {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

// =============================================================================
// Inline Binary Operation Stencils (for small, hot operations)
// =============================================================================
// These stencils contain the actual SIMD code inline, not a function call.
// Uses SSE/AVX instructions for maximum performance.

static Stencil generateInlineAddF32Stencil() {
    CodeEmitter e;

    // This is a self-contained add loop using SSE
    // void add_f32(float* out, const float* a, const float* b, size_t count)
    // RDI=out, RSI=a, RDX=b, RCX=count

    emitPrologue(e);

    // Load arguments from args array
    emitLoadArg(e, 7, 0);  // rdi = out
    emitLoadArg(e, 6, 1);  // rsi = a
    emitLoadArg(e, 2, 2);  // rdx = b
    emitLoadArg(e, 1, 3);  // rcx = count

    // xor r8, r8  ; i = 0
    e.emit({0x4D, 0x31, 0xC0});

    // Main loop (processes 4 floats at a time with SSE)
    size_t loop_start = e.code().size();

    // cmp r8, rcx (0x4C 0x3B 0xC1 = CMP R8, RCX using opcode 0x3B)
    e.emit({0x4C, 0x3B, 0xC1});
    // jge end (will patch offset)
    e.emit({0x7D, 0x00});  // placeholder for offset
    size_t jge_offset = e.code().size() - 1;

    // Check if we have at least 4 elements left: rcx - r8 >= 4
    // lea rax, [rcx - 4]
    e.emit({0x48, 0x8D, 0x41, 0xFC});
    // cmp r8, rax (0x4C 0x3B 0xC0 = CMP R8, RAX)
    e.emit({0x4C, 0x3B, 0xC0});
    // jg scalar_loop (skip 24 bytes: 5+5+3+5+4+2)
    e.emit({0x7F, 0x18});

    // Vector loop: process 4 floats
    // movups xmm0, [rsi + r8*4] (0x42 = REX.X for R8 in SIB, no 0xF3 = packed not scalar)
    e.emit({0x42, 0x0F, 0x10, 0x04, 0x86});
    // movups xmm1, [rdx + r8*4]
    e.emit({0x42, 0x0F, 0x10, 0x0C, 0x82});
    // addps xmm0, xmm1
    e.emit({0x0F, 0x58, 0xC1});
    // movups [rdi + r8*4], xmm0
    e.emit({0x42, 0x0F, 0x11, 0x04, 0x87});
    // add r8, 4
    e.emit({0x49, 0x83, 0xC0, 0x04});
    // jmp loop_start
    int8_t rel_offset = static_cast<int8_t>(loop_start - e.code().size() - 2);
    e.emit({0xEB, static_cast<uint8_t>(rel_offset)});

    // Scalar loop for remainder
    size_t scalar_loop = e.code().size();

    // cmp r8, rcx (0x4C 0x3B 0xC1 = CMP R8, RCX)
    e.emit({0x4C, 0x3B, 0xC1});
    // jge end
    e.emit({0x7D, 0x14});

    // movss xmm0, [rsi + r8*4]
    e.emit({0xF3, 0x42, 0x0F, 0x10, 0x04, 0x86});
    // addss xmm0, [rdx + r8*4]
    e.emit({0xF3, 0x42, 0x0F, 0x58, 0x04, 0x82});
    // movss [rdi + r8*4], xmm0
    e.emit({0xF3, 0x42, 0x0F, 0x11, 0x04, 0x87});
    // inc r8
    e.emit({0x49, 0xFF, 0xC0});
    // jmp scalar_loop
    rel_offset = static_cast<int8_t>(scalar_loop - e.code().size() - 2);
    e.emit({0xEB, static_cast<uint8_t>(rel_offset)});

    // end:
    // Patch the initial jge
    e.code()[jge_offset] = static_cast<uint8_t>(e.code().size() - jge_offset - 1);

    emitEpilogue(e);

    Stencil s;
    s.name = "inline_add_f32";
    s.op = ir::OpCode::kAdd;
    s.dtype = ScalarType::kFloat32;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"},
        {0, Stencil::Hole::kImmediate64, "count"},
    };
    s.alignment = 16;

    return s;
}

static Stencil generateInlineMulF32Stencil() {
    CodeEmitter e;

    emitPrologue(e);

    // Load arguments
    emitLoadArg(e, 7, 0);  // rdi = out
    emitLoadArg(e, 6, 1);  // rsi = a
    emitLoadArg(e, 2, 2);  // rdx = b
    emitLoadArg(e, 1, 3);  // rcx = count

    // xor r8, r8  ; i = 0
    e.emit({0x4D, 0x31, 0xC0});

    size_t loop_start = e.code().size();

    // cmp r8, rcx (0x4C 0x3B 0xC1 = CMP R8, RCX)
    e.emit({0x4C, 0x3B, 0xC1});
    // jge end
    e.emit({0x7D, 0x00});
    size_t jge_offset = e.code().size() - 1;

    // Check for 4 elements
    e.emit({0x48, 0x8D, 0x41, 0xFC});  // lea rax, [rcx - 4]
    e.emit({0x4C, 0x3B, 0xC0});        // cmp r8, rax (CMP R8, RAX)
    e.emit({0x7F, 0x18});              // jg scalar (skip 24 bytes: 5+5+3+5+4+2)

    // Vector: mulps (no 0xF3 prefix = packed not scalar)
    e.emit({0x42, 0x0F, 0x10, 0x04, 0x86});  // movups xmm0, [rsi + r8*4]
    e.emit({0x42, 0x0F, 0x10, 0x0C, 0x82});  // movups xmm1, [rdx + r8*4]
    e.emit({0x0F, 0x59, 0xC1});              // mulps xmm0, xmm1
    e.emit({0x42, 0x0F, 0x11, 0x04, 0x87});  // movups [rdi + r8*4], xmm0
    e.emit({0x49, 0x83, 0xC0, 0x04});        // add r8, 4

    int8_t rel = static_cast<int8_t>(loop_start - e.code().size() - 2);
    e.emit({0xEB, static_cast<uint8_t>(rel)});

    size_t scalar_loop = e.code().size();
    e.emit({0x4C, 0x3B, 0xC1});  // cmp r8, rcx (CMP R8, RCX)
    e.emit({0x7D, 0x14});        // jge end

    e.emit({0xF3, 0x42, 0x0F, 0x10, 0x04, 0x86});  // movss xmm0, [rsi + r8*4]
    e.emit({0xF3, 0x42, 0x0F, 0x59, 0x04, 0x82});  // mulss xmm0, [rdx + r8*4]
    e.emit({0xF3, 0x42, 0x0F, 0x11, 0x04, 0x87});  // movss [rdi + r8*4], xmm0
    e.emit({0x49, 0xFF, 0xC0});                    // inc r8

    rel = static_cast<int8_t>(scalar_loop - e.code().size() - 2);
    e.emit({0xEB, static_cast<uint8_t>(rel)});

    e.code()[jge_offset] = static_cast<uint8_t>(e.code().size() - jge_offset - 1);

    emitEpilogue(e);

    Stencil s;
    s.name = "inline_mul_f32";
    s.op = ir::OpCode::kMul;
    s.dtype = ScalarType::kFloat32;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"},
        {0, Stencil::Hole::kImmediate64, "count"},
    };
    s.alignment = 16;

    return s;
}

// =============================================================================
// FMA Inline Stencil (a * b + c)
// =============================================================================

static Stencil generateInlineFmaF32Stencil() {
    CodeEmitter e;

    emitPrologue(e);

    // Load arguments: out, a, b, c, count
    emitLoadArg(e, 7, 0);  // rdi = out
    emitLoadArg(e, 6, 1);  // rsi = a
    emitLoadArg(e, 2, 2);  // rdx = b
    emitLoadArg(e, 1, 3);  // rcx = c
    emitLoadArg(e, 8, 4);  // r8 = count

    // xor r9, r9  ; i = 0
    e.emit({0x4D, 0x31, 0xC9});

    size_t loop_start = e.code().size();

    // cmp r9, r8
    e.emit({0x4D, 0x39, 0xC1});
    // jge end
    e.emit({0x7D, 0x00});
    size_t jge_offset = e.code().size() - 1;

    // Check for 4 elements
    e.emit({0x4D, 0x8D, 0x50, 0xFC});  // lea r10, [r8 - 4]
    e.emit({0x4D, 0x3B, 0xCA});        // cmp r9, r10 (using opcode 0x3B)
    e.emit({0x7F, 0x20});              // jg scalar (skip 32 bytes: 5+5+3+5+3+5+4+2)

    // Vector: fma = a * b + c using SSE (mul then add)
    // No 0xF3 prefix for packed operations
    e.emit({0x42, 0x0F, 0x10, 0x04, 0x8E});  // movups xmm0, [rsi + r9*4]
    e.emit({0x42, 0x0F, 0x10, 0x0C, 0x8A});  // movups xmm1, [rdx + r9*4]
    e.emit({0x0F, 0x59, 0xC1});              // mulps xmm0, xmm1
    e.emit({0x42, 0x0F, 0x10, 0x0C, 0x89});  // movups xmm1, [rcx + r9*4]
    e.emit({0x0F, 0x58, 0xC1});              // addps xmm0, xmm1
    e.emit({0x42, 0x0F, 0x11, 0x04, 0x8F});  // movups [rdi + r9*4], xmm0
    e.emit({0x49, 0x83, 0xC1, 0x04});        // add r9, 4

    int8_t rel = static_cast<int8_t>(loop_start - e.code().size() - 2);
    e.emit({0xEB, static_cast<uint8_t>(rel)});

    size_t scalar_loop = e.code().size();
    e.emit({0x4D, 0x39, 0xC1});  // cmp r9, r8
    e.emit({0x7D, 0x1A});        // jge end

    e.emit({0xF3, 0x42, 0x0F, 0x10, 0x04, 0x8E});  // movss xmm0, [rsi + r9*4]
    e.emit({0xF3, 0x42, 0x0F, 0x59, 0x04, 0x8A});  // mulss xmm0, [rdx + r9*4]
    e.emit({0xF3, 0x42, 0x0F, 0x58, 0x04, 0x89});  // addss xmm0, [rcx + r9*4]
    e.emit({0xF3, 0x42, 0x0F, 0x11, 0x04, 0x8F});  // movss [rdi + r9*4], xmm0
    e.emit({0x49, 0xFF, 0xC1});                    // inc r9

    rel = static_cast<int8_t>(scalar_loop - e.code().size() - 2);
    e.emit({0xEB, static_cast<uint8_t>(rel)});

    e.code()[jge_offset] = static_cast<uint8_t>(e.code().size() - jge_offset - 1);

    emitEpilogue(e);

    Stencil s;
    s.name = "inline_fma_f32";
    s.op = ir::OpCode::kFma;
    s.dtype = ScalarType::kFloat32;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},  {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"}, {0, Stencil::Hole::kAbsAddress64, "input_c"},
        {0, Stencil::Hole::kImmediate64, "count"},
    };
    s.alignment = 16;

    return s;
}

}  // namespace x86_64

// =============================================================================
// ARM64 Stencil Generator
// =============================================================================
//
// ARM64 (AAPCS64) calling convention:
//   - Arguments: X0-X7 (integers/pointers), V0-V7 (floats/vectors)
//   - Return: X0 or V0
//   - Callee-saved: X19-X29, LR (X30), V8-V15
//
// =============================================================================

namespace arm64 {

class CodeEmitter {
  public:
    void emit32(uint32_t insn) {
        code_.push_back(insn & 0xFF);
        code_.push_back((insn >> 8) & 0xFF);
        code_.push_back((insn >> 16) & 0xFF);
        code_.push_back((insn >> 24) & 0xFF);
    }

    std::vector<uint8_t>& code() { return code_; }
    std::vector<Stencil::Hole>& holes() { return holes_; }

  private:
    std::vector<uint8_t> code_;
    std::vector<Stencil::Hole> holes_;
};

// ARM64 instructions encoding helpers
static constexpr uint32_t arm_stp_pre(int rt, int rt2, int rn, int imm) {
    // STP Xt, Xt2, [Xn, #imm]!
    return 0xA9800000 | ((((imm / 8) & 0x7F) << 15)) | (rt2 << 10) | (rn << 5) | rt;
}

static constexpr uint32_t arm_ldp_post(int rt, int rt2, int rn, int imm) {
    // LDP Xt, Xt2, [Xn], #imm
    return 0xA8C00000 | ((((imm / 8) & 0x7F) << 15)) | (rt2 << 10) | (rn << 5) | rt;
}

static constexpr uint32_t arm_mov_sp(int rd, int rn) {
    // MOV Xd, Xn (alias for ADD)
    return 0x91000000 | (rn << 5) | rd;
}

static constexpr uint32_t arm_ldr_offset(int rt, int rn, int offset) {
    // LDR Xt, [Xn, #offset]
    return 0xF9400000 | (((offset / 8) & 0xFFF) << 10) | (rn << 5) | rt;
}

static constexpr uint32_t arm_blr(int rn) {
    // BLR Xn
    return 0xD63F0000 | (rn << 5);
}

static constexpr uint32_t arm_ret() {
    // RET
    return 0xD65F03C0;
}

static void emitPrologue(CodeEmitter& e) {
    // stp x29, x30, [sp, #-16]!
    e.emit32(arm_stp_pre(29, 30, 31, -16));
    // mov x29, sp
    e.emit32(arm_mov_sp(29, 31));
    // stp x19, x20, [sp, #-16]!
    e.emit32(arm_stp_pre(19, 20, 31, -16));
}

static void emitEpilogue(CodeEmitter& e) {
    // ldp x19, x20, [sp], #16
    e.emit32(arm_ldp_post(19, 20, 31, 16));
    // ldp x29, x30, [sp], #16
    e.emit32(arm_ldp_post(29, 30, 31, 16));
    // ret
    e.emit32(arm_ret());
}

// Generate binary operation stencil for ARM64
static Stencil generateBinaryOpStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // X0 contains args pointer
    // mov x19, x0  (save args pointer)
    e.emit32(0xAA0003F3);

    // Load arguments: x0=out, x1=a, x2=b, x3=count
    // ldr x0, [x19, #0]
    e.emit32(arm_ldr_offset(0, 19, 0));
    // ldr x1, [x19, #8]
    e.emit32(arm_ldr_offset(1, 19, 8));
    // ldr x2, [x19, #16]
    e.emit32(arm_ldr_offset(2, 19, 16));
    // ldr x3, [x19, #24]
    e.emit32(arm_ldr_offset(3, 19, 24));
    // ldr x4, [x19, #32]  (function pointer)
    e.emit32(arm_ldr_offset(4, 19, 32));

    // blr x4
    e.emit32(arm_blr(4));

    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},   {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"},  {0, Stencil::Hole::kImmediate64, "count"},
        {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

static Stencil generateUnaryOpStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // mov x19, x0
    e.emit32(0xAA0003F3);

    // Load: x0=out, x1=input, x2=count
    e.emit32(arm_ldr_offset(0, 19, 0));
    e.emit32(arm_ldr_offset(1, 19, 8));
    e.emit32(arm_ldr_offset(2, 19, 16));
    e.emit32(arm_ldr_offset(3, 19, 24));

    // blr x3
    e.emit32(arm_blr(3));

    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input"},
        {0, Stencil::Hole::kImmediate64, "count"},
        {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

static Stencil generateTernaryOpStencil(const std::string& name, ir::OpCode op, ScalarType dtype) {
    CodeEmitter e;

    emitPrologue(e);

    // mov x19, x0
    e.emit32(0xAA0003F3);

    // Load: x0=out, x1=a, x2=b, x3=c, x4=count
    e.emit32(arm_ldr_offset(0, 19, 0));
    e.emit32(arm_ldr_offset(1, 19, 8));
    e.emit32(arm_ldr_offset(2, 19, 16));
    e.emit32(arm_ldr_offset(3, 19, 24));
    e.emit32(arm_ldr_offset(4, 19, 32));
    e.emit32(arm_ldr_offset(5, 19, 40));

    // blr x5
    e.emit32(arm_blr(5));

    emitEpilogue(e);

    Stencil s;
    s.name = name;
    s.op = op;
    s.dtype = dtype;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},  {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"}, {0, Stencil::Hole::kAbsAddress64, "input_c"},
        {0, Stencil::Hole::kImmediate64, "count"},    {0, Stencil::Hole::kAbsAddress64, "func_ptr"},
    };
    s.alignment = 16;

    return s;
}

// =============================================================================
// ARM64 NEON Instruction Encodings
// =============================================================================

// NEON load/store
static constexpr uint32_t arm_ldr_q_post(int rt, int rn, int imm) {
    // LDR Qt, [Xn], #imm (post-index, 128-bit)
    return 0x3CC00400 | ((imm & 0x1FF) << 12) | (rn << 5) | rt;
}

static constexpr uint32_t arm_str_q_post(int rt, int rn, int imm) {
    // STR Qt, [Xn], #imm (post-index, 128-bit)
    return 0x3C800400 | ((imm & 0x1FF) << 12) | (rn << 5) | rt;
}

static constexpr uint32_t arm_ldr_s_post(int rt, int rn, int imm) {
    // LDR St, [Xn], #imm (post-index, 32-bit float)
    return 0xBC400400 | ((imm & 0x1FF) << 12) | (rn << 5) | rt;
}

static constexpr uint32_t arm_str_s_post(int rt, int rn, int imm) {
    // STR St, [Xn], #imm (post-index, 32-bit float)
    return 0xBC000400 | ((imm & 0x1FF) << 12) | (rn << 5) | rt;
}

// NEON arithmetic (4xfloat32)
static constexpr uint32_t arm_fadd_4s(int rd, int rn, int rm) {
    // FADD Vd.4S, Vn.4S, Vm.4S
    return 0x4E20D400 | (rm << 16) | (rn << 5) | rd;
}

static constexpr uint32_t arm_fmul_4s(int rd, int rn, int rm) {
    // FMUL Vd.4S, Vn.4S, Vm.4S
    return 0x6E20DC00 | (rm << 16) | (rn << 5) | rd;
}

static constexpr uint32_t arm_fmla_4s(int rd, int rn, int rm) {
    // FMLA Vd.4S, Vn.4S, Vm.4S (rd = rd + rn * rm)
    return 0x4E20CC00 | (rm << 16) | (rn << 5) | rd;
}

// Scalar NEON arithmetic
static constexpr uint32_t arm_fadd_s(int rd, int rn, int rm) {
    // FADD Sd, Sn, Sm
    return 0x1E202800 | (rm << 16) | (rn << 5) | rd;
}

static constexpr uint32_t arm_fmul_s(int rd, int rn, int rm) {
    // FMUL Sd, Sn, Sm
    return 0x1E200800 | (rm << 16) | (rn << 5) | rd;
}

// Control flow
static constexpr uint32_t arm_cmp(int rn, int rm) {
    // CMP Xn, Xm (alias for SUBS XZR, Xn, Xm)
    return 0xEB00001F | (rm << 16) | (rn << 5);
}

static constexpr uint32_t arm_sub_imm(int rd, int rn, int imm) {
    // SUB Xd, Xn, #imm
    return 0xD1000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd;
}

static constexpr uint32_t arm_add_imm(int rd, int rn, int imm) {
    // ADD Xd, Xn, #imm
    return 0x91000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd;
}

static constexpr uint32_t arm_b_cond(int cond, int imm19) {
    // B.<cond> label (PC-relative)
    return 0x54000000 | ((imm19 & 0x7FFFF) << 5) | cond;
}

static constexpr uint32_t arm_b(int imm26) {
    // B label (unconditional)
    return 0x14000000 | (imm26 & 0x3FFFFFF);
}

// Condition codes
static constexpr int ARM_COND_GE = 0xA;  // Greater than or equal
static constexpr int ARM_COND_LT = 0xB;  // Less than

// =============================================================================
// ARM64 Inline NEON Stencils
// =============================================================================

static Stencil generateInlineAddF32Stencil_ARM64() {
    CodeEmitter e;

    emitPrologue(e);

    // Load arguments from args array
    // X0 = args pointer on entry
    // mov x19, x0
    e.emit32(0xAA0003F3);

    // ldr x0, [x19, #0]   ; out
    e.emit32(arm_ldr_offset(0, 19, 0));
    // ldr x1, [x19, #8]   ; a
    e.emit32(arm_ldr_offset(1, 19, 8));
    // ldr x2, [x19, #16]  ; b
    e.emit32(arm_ldr_offset(2, 19, 16));
    // ldr x3, [x19, #24]  ; count
    e.emit32(arm_ldr_offset(3, 19, 24));

    // x4 = loop counter (starts at 0)
    // mov x4, #0
    e.emit32(0xD2800004);

    // Main loop: process 4 floats at a time
    size_t loop_start = e.code().size();

    // Check if we have at least 4 elements left: count - i >= 4
    // sub x5, x3, x4
    e.emit32(0xCB040065);
    // cmp x5, #4
    e.emit32(0xF1001085);
    // b.lt scalar_loop (skip 7 instructions = 28 bytes = 7 words)
    e.emit32(arm_b_cond(ARM_COND_LT, 8));

    // Vector loop body:
    // Calculate addresses: a + i*4, b + i*4, out + i*4
    // ldr q0, [x1, x4, lsl #2]
    e.emit32(0x3C647820);
    // ldr q1, [x2, x4, lsl #2]
    e.emit32(0x3C647841);
    // fadd v0.4s, v0.4s, v1.4s
    e.emit32(arm_fadd_4s(0, 0, 1));
    // str q0, [x0, x4, lsl #2]
    e.emit32(0x3C247800);
    // add x4, x4, #4
    e.emit32(arm_add_imm(4, 4, 4));
    // b loop_start
    int32_t loop_offset =
        (static_cast<int32_t>(loop_start) - static_cast<int32_t>(e.code().size())) / 4;
    e.emit32(arm_b(loop_offset));

    // Scalar loop for remainder
    size_t scalar_loop = e.code().size();

    // cmp x4, x3
    e.emit32(arm_cmp(4, 3));
    // b.ge end (skip 5 instructions)
    e.emit32(arm_b_cond(ARM_COND_GE, 6));

    // ldr s0, [x1, x4, lsl #2]
    e.emit32(0xBC647820);
    // ldr s1, [x2, x4, lsl #2]
    e.emit32(0xBC647841);
    // fadd s0, s0, s1
    e.emit32(arm_fadd_s(0, 0, 1));
    // str s0, [x0, x4, lsl #2]
    e.emit32(0xBC247800);
    // add x4, x4, #1
    e.emit32(arm_add_imm(4, 4, 1));
    // b scalar_loop
    int32_t scalar_offset =
        (static_cast<int32_t>(scalar_loop) - static_cast<int32_t>(e.code().size())) / 4;
    e.emit32(arm_b(scalar_offset));

    emitEpilogue(e);

    Stencil s;
    s.name = "inline_add_f32_arm64";
    s.op = ir::OpCode::kAdd;
    s.dtype = ScalarType::kFloat32;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"},
        {0, Stencil::Hole::kImmediate64, "count"},
    };
    s.alignment = 16;

    return s;
}

static Stencil generateInlineMulF32Stencil_ARM64() {
    CodeEmitter e;

    emitPrologue(e);

    // mov x19, x0
    e.emit32(0xAA0003F3);

    // Load arguments
    e.emit32(arm_ldr_offset(0, 19, 0));   // out
    e.emit32(arm_ldr_offset(1, 19, 8));   // a
    e.emit32(arm_ldr_offset(2, 19, 16));  // b
    e.emit32(arm_ldr_offset(3, 19, 24));  // count

    // mov x4, #0
    e.emit32(0xD2800004);

    size_t loop_start = e.code().size();

    // sub x5, x3, x4
    e.emit32(0xCB040065);
    // cmp x5, #4
    e.emit32(0xF1001085);
    // b.lt scalar_loop
    e.emit32(arm_b_cond(ARM_COND_LT, 8));

    // Vector mul
    e.emit32(0x3C647820);  // ldr q0, [x1, x4, lsl #2]
    e.emit32(0x3C647841);  // ldr q1, [x2, x4, lsl #2]
    e.emit32(arm_fmul_4s(0, 0, 1));
    e.emit32(0x3C247800);  // str q0, [x0, x4, lsl #2]
    e.emit32(arm_add_imm(4, 4, 4));
    int32_t loop_offset =
        (static_cast<int32_t>(loop_start) - static_cast<int32_t>(e.code().size())) / 4;
    e.emit32(arm_b(loop_offset));

    size_t scalar_loop = e.code().size();

    e.emit32(arm_cmp(4, 3));
    e.emit32(arm_b_cond(ARM_COND_GE, 6));

    e.emit32(0xBC647820);  // ldr s0, [x1, x4, lsl #2]
    e.emit32(0xBC647841);  // ldr s1, [x2, x4, lsl #2]
    e.emit32(arm_fmul_s(0, 0, 1));
    e.emit32(0xBC247800);  // str s0, [x0, x4, lsl #2]
    e.emit32(arm_add_imm(4, 4, 1));
    int32_t scalar_offset =
        (static_cast<int32_t>(scalar_loop) - static_cast<int32_t>(e.code().size())) / 4;
    e.emit32(arm_b(scalar_offset));

    emitEpilogue(e);

    Stencil s;
    s.name = "inline_mul_f32_arm64";
    s.op = ir::OpCode::kMul;
    s.dtype = ScalarType::kFloat32;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},
        {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"},
        {0, Stencil::Hole::kImmediate64, "count"},
    };
    s.alignment = 16;

    return s;
}

static Stencil generateInlineFmaF32Stencil_ARM64() {
    CodeEmitter e;

    emitPrologue(e);

    // mov x19, x0
    e.emit32(0xAA0003F3);

    // Load arguments: out, a, b, c, count
    e.emit32(arm_ldr_offset(0, 19, 0));   // out
    e.emit32(arm_ldr_offset(1, 19, 8));   // a
    e.emit32(arm_ldr_offset(2, 19, 16));  // b
    e.emit32(arm_ldr_offset(3, 19, 24));  // c
    e.emit32(arm_ldr_offset(4, 19, 32));  // count

    // mov x5, #0 (loop counter)
    e.emit32(0xD2800005);

    size_t loop_start = e.code().size();

    // sub x6, x4, x5
    e.emit32(0xCB050086);
    // cmp x6, #4
    e.emit32(0xF10010C6);
    // b.lt scalar_loop
    e.emit32(arm_b_cond(ARM_COND_LT, 9));

    // Vector FMA: out = a * b + c
    e.emit32(0x3C657820);            // ldr q0, [x1, x5, lsl #2] (a)
    e.emit32(0x3C657841);            // ldr q1, [x2, x5, lsl #2] (b)
    e.emit32(0x3C657862);            // ldr q2, [x3, x5, lsl #2] (c)
    e.emit32(arm_fmul_4s(0, 0, 1));  // v0 = a * b
    e.emit32(arm_fadd_4s(0, 0, 2));  // v0 = (a * b) + c
    e.emit32(0x3C257800);            // str q0, [x0, x5, lsl #2]
    e.emit32(arm_add_imm(5, 5, 4));
    int32_t loop_offset =
        (static_cast<int32_t>(loop_start) - static_cast<int32_t>(e.code().size())) / 4;
    e.emit32(arm_b(loop_offset));

    size_t scalar_loop = e.code().size();

    e.emit32(arm_cmp(5, 4));
    e.emit32(arm_b_cond(ARM_COND_GE, 7));

    e.emit32(0xBC657820);  // ldr s0, [x1, x5, lsl #2]
    e.emit32(0xBC657841);  // ldr s1, [x2, x5, lsl #2]
    e.emit32(0xBC657862);  // ldr s2, [x3, x5, lsl #2]
    e.emit32(arm_fmul_s(0, 0, 1));
    e.emit32(arm_fadd_s(0, 0, 2));
    e.emit32(0xBC257800);  // str s0, [x0, x5, lsl #2]
    e.emit32(arm_add_imm(5, 5, 1));
    int32_t scalar_offset =
        (static_cast<int32_t>(scalar_loop) - static_cast<int32_t>(e.code().size())) / 4;
    e.emit32(arm_b(scalar_offset));

    emitEpilogue(e);

    Stencil s;
    s.name = "inline_fma_f32_arm64";
    s.op = ir::OpCode::kFma;
    s.dtype = ScalarType::kFloat32;
    s.code = std::move(e.code());
    s.holes = {
        {0, Stencil::Hole::kAbsAddress64, "output"},  {0, Stencil::Hole::kAbsAddress64, "input_a"},
        {0, Stencil::Hole::kAbsAddress64, "input_b"}, {0, Stencil::Hole::kAbsAddress64, "input_c"},
        {0, Stencil::Hole::kImmediate64, "count"},
    };
    s.alignment = 16;

    return s;
}

}  // namespace arm64

// =============================================================================
// Stencil Registry
// =============================================================================

class StencilRegistry {
  public:
    static StencilRegistry& instance() {
        static StencilRegistry registry;
        return registry;
    }

    void registerStencil(Stencil stencil) {
        auto key = makeKey(stencil.op, stencil.dtype);
        stencils_[key] = std::move(stencil);
        spdlog::debug("Registered stencil: {} ({}) - {} bytes", stencils_[key].name, key,
                      stencils_[key].code.size());
    }

    const Stencil* find(ir::OpCode op, ScalarType dtype) const {
        auto key = makeKey(op, dtype);
        auto it = stencils_.find(key);
        if (it != stencils_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    bool has(ir::OpCode op, ScalarType dtype) const { return find(op, dtype) != nullptr; }

    size_t count() const { return stencils_.size(); }

    // Get all registered stencils
    const std::unordered_map<std::string, Stencil>& all() const { return stencils_; }

  private:
    StencilRegistry() { initializeBuiltinStencils(); }

    void initializeBuiltinStencils();

    static std::string makeKey(ir::OpCode op, ScalarType dtype) {
        return std::string(ir::opCodeName(op)) + "_" + std::string(scalarTypeName(dtype));
    }

    std::unordered_map<std::string, Stencil> stencils_;
};

// =============================================================================
// Built-in Stencil Initialization
// =============================================================================

void StencilRegistry::initializeBuiltinStencils() {
    Architecture arch = detectArchitecture();

    spdlog::info("StencilRegistry: Generating stencils for {} architecture",
                 arch == Architecture::kX86_64  ? "x86-64"
                 : arch == Architecture::kARM64 ? "ARM64"
                                                : "Unknown");

    if (arch == Architecture::kX86_64) {
        // Binary operations (call through function pointer - fallback)
        registerStencil(x86_64::generateBinaryOpStencil("add_f32_dispatch", ir::OpCode::kAdd,
                                                        ScalarType::kFloat32));
        registerStencil(
            x86_64::generateBinaryOpStencil("add_f64", ir::OpCode::kAdd, ScalarType::kFloat64));
        registerStencil(
            x86_64::generateBinaryOpStencil("sub_f32", ir::OpCode::kSub, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateBinaryOpStencil("sub_f64", ir::OpCode::kSub, ScalarType::kFloat64));
        registerStencil(x86_64::generateBinaryOpStencil("mul_f32_dispatch", ir::OpCode::kMul,
                                                        ScalarType::kFloat32));
        registerStencil(
            x86_64::generateBinaryOpStencil("mul_f64", ir::OpCode::kMul, ScalarType::kFloat64));
        registerStencil(
            x86_64::generateBinaryOpStencil("div_f32", ir::OpCode::kDiv, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateBinaryOpStencil("div_f64", ir::OpCode::kDiv, ScalarType::kFloat64));
        registerStencil(
            x86_64::generateBinaryOpStencil("min_f32", ir::OpCode::kMin, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateBinaryOpStencil("max_f32", ir::OpCode::kMax, ScalarType::kFloat32));

        // Unary operations
        registerStencil(
            x86_64::generateUnaryOpStencil("neg_f32", ir::OpCode::kNeg, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("abs_f32", ir::OpCode::kAbs, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("sqrt_f32", ir::OpCode::kSqrt, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("rsqrt_f32", ir::OpCode::kRsqrt, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("exp_f32", ir::OpCode::kExp, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("log_f32", ir::OpCode::kLog, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("sin_f32", ir::OpCode::kSin, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("cos_f32", ir::OpCode::kCos, ScalarType::kFloat32));
        registerStencil(
            x86_64::generateUnaryOpStencil("tanh_f32", ir::OpCode::kTanh, ScalarType::kFloat32));

        // FMA/Ternary operations (via function pointer)
        registerStencil(x86_64::generateTernaryOpStencil("fma_f32_dispatch", ir::OpCode::kFma,
                                                         ScalarType::kFloat32));
        registerStencil(
            x86_64::generateTernaryOpStencil("fnma_f32", ir::OpCode::kFnma, ScalarType::kFloat32));

        // Reductions
        registerStencil(x86_64::generateReductionStencil("reduce_sum_f32", ir::OpCode::kReduceSum,
                                                         ScalarType::kFloat32));
        registerStencil(x86_64::generateReductionStencil("reduce_max_f32", ir::OpCode::kReduceMax,
                                                         ScalarType::kFloat32));
        registerStencil(x86_64::generateReductionStencil("reduce_min_f32", ir::OpCode::kReduceMin,
                                                         ScalarType::kFloat32));

        // Inline SIMD stencils (fastest - override dispatch for common ops)
        // Registered last so they take precedence in the registry
        registerStencil(x86_64::generateInlineAddF32Stencil());
        registerStencil(x86_64::generateInlineMulF32Stencil());
        registerStencil(x86_64::generateInlineFmaF32Stencil());

    } else if (arch == Architecture::kARM64) {
        // Binary operations (fallback via function pointer dispatch)
        registerStencil(arm64::generateBinaryOpStencil("add_f32_dispatch", ir::OpCode::kAdd,
                                                       ScalarType::kFloat32));
        registerStencil(
            arm64::generateBinaryOpStencil("add_f64", ir::OpCode::kAdd, ScalarType::kFloat64));
        registerStencil(
            arm64::generateBinaryOpStencil("sub_f32", ir::OpCode::kSub, ScalarType::kFloat32));
        registerStencil(arm64::generateBinaryOpStencil("mul_f32_dispatch", ir::OpCode::kMul,
                                                       ScalarType::kFloat32));
        registerStencil(
            arm64::generateBinaryOpStencil("div_f32", ir::OpCode::kDiv, ScalarType::kFloat32));

        // Unary operations
        registerStencil(
            arm64::generateUnaryOpStencil("neg_f32", ir::OpCode::kNeg, ScalarType::kFloat32));
        registerStencil(
            arm64::generateUnaryOpStencil("abs_f32", ir::OpCode::kAbs, ScalarType::kFloat32));
        registerStencil(
            arm64::generateUnaryOpStencil("sqrt_f32", ir::OpCode::kSqrt, ScalarType::kFloat32));
        registerStencil(
            arm64::generateUnaryOpStencil("exp_f32", ir::OpCode::kExp, ScalarType::kFloat32));
        registerStencil(
            arm64::generateUnaryOpStencil("tanh_f32", ir::OpCode::kTanh, ScalarType::kFloat32));

        // FMA operations (via function pointer)
        registerStencil(arm64::generateTernaryOpStencil("fma_f32_dispatch", ir::OpCode::kFma,
                                                        ScalarType::kFloat32));

        // Inline NEON stencils (fastest - override dispatch for common ops)
        // Registered last so they take precedence in the registry
        registerStencil(arm64::generateInlineAddF32Stencil_ARM64());
        registerStencil(arm64::generateInlineMulF32Stencil_ARM64());
        registerStencil(arm64::generateInlineFmaF32Stencil_ARM64());
    } else {
        spdlog::warn("StencilRegistry: Unknown architecture, using fallback stencils");

        // Fallback: register placeholder stencils
        Stencil s;
        s.name = "fallback_ret";
        s.op = ir::OpCode::kAdd;
        s.dtype = ScalarType::kFloat32;
        s.code = {0xC3};  // RET on x86, will crash on other arches
        s.holes = {};
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

size_t stencilCount() {
    return StencilRegistry::instance().count();
}

// Get function pointers for Highway operations (used for dispatch stencils)
void* getHwyFunctionPtr(ir::OpCode op, ScalarType dtype) {
    // Return function pointers to Highway dispatch functions
    switch (op) {
    case ir::OpCode::kAdd:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, size_t)>(simd::Add));
        } else if (dtype == ScalarType::kFloat64) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(double*, const double*, const double*, size_t)>(simd::Add));
        }
        break;
    case ir::OpCode::kSub:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, size_t)>(simd::Sub));
        }
        break;
    case ir::OpCode::kMul:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, size_t)>(simd::Mul));
        }
        break;
    case ir::OpCode::kDiv:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, size_t)>(simd::Div));
        }
        break;
    case ir::OpCode::kNeg:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Neg));
        }
        break;
    case ir::OpCode::kAbs:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Abs));
        }
        break;
    case ir::OpCode::kSqrt:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Sqrt));
        }
        break;
    case ir::OpCode::kExp:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Exp));
        }
        break;
    case ir::OpCode::kLog:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Log));
        }
        break;
    case ir::OpCode::kSin:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Sin));
        }
        break;
    case ir::OpCode::kCos:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Cos));
        }
        break;
    case ir::OpCode::kTanh:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, size_t)>(simd::Tanh));
        }
        break;
    case ir::OpCode::kFma:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, const float*, size_t)>(
                    simd::MulAdd));
        }
        break;
    case ir::OpCode::kMin:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, size_t)>(simd::Min));
        }
        break;
    case ir::OpCode::kMax:
        if (dtype == ScalarType::kFloat32) {
            return reinterpret_cast<void*>(
                static_cast<void (*)(float*, const float*, const float*, size_t)>(simd::Max));
        }
        break;
    default:
        break;
    }
    return nullptr;
}

}  // namespace jit
}  // namespace bud
