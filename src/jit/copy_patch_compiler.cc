// =============================================================================
// Bud Flow Lang - Copy-and-Patch JIT Compiler
// =============================================================================
//
// Tier 1 JIT compiler using copy-and-patch technique.
// Key advantage: ~1ms compilation time (vs 100ms+ for LLVM)
//

#include "bud_flow_lang/jit/stencil.h"
#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/error.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <sys/mman.h>

namespace bud {
namespace jit {

// =============================================================================
// Executable Memory Management
// =============================================================================

class ExecutableMemory {
public:
    ExecutableMemory(size_t size) : size_(size) {
        // Allocate read-write-execute memory
        // NOTE: In production, would use W^X with separate RW and RX mappings
        ptr_ = mmap(nullptr, size,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            spdlog::error("Failed to allocate executable memory: {} bytes", size);
        }
    }

    ~ExecutableMemory() {
        if (ptr_) {
            munmap(ptr_, size_);
        }
    }

    ExecutableMemory(const ExecutableMemory&) = delete;
    ExecutableMemory& operator=(const ExecutableMemory&) = delete;

    ExecutableMemory(ExecutableMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), used_(other.used_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.used_ = 0;
    }

    void* allocate(size_t bytes, size_t alignment = 16) {
        // Align the current position
        size_t aligned = (used_ + alignment - 1) & ~(alignment - 1);

        if (aligned + bytes > size_) {
            return nullptr;  // Out of space
        }

        void* result = static_cast<char*>(ptr_) + aligned;
        used_ = aligned + bytes;
        return result;
    }

    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
    [[nodiscard]] size_t remaining() const { return size_ - used_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    size_t used_ = 0;
};

// =============================================================================
// Compiled Kernel
// =============================================================================

class CompiledKernel {
public:
    using KernelFn = void (*)(void** args);

    CompiledKernel() = default;
    explicit CompiledKernel(KernelFn fn, size_t code_size)
        : fn_(fn), code_size_(code_size) {}

    void operator()(void** args) const {
        if (fn_) {
            fn_(args);
        }
    }

    [[nodiscard]] bool valid() const { return fn_ != nullptr; }
    [[nodiscard]] size_t codeSize() const { return code_size_; }

private:
    KernelFn fn_ = nullptr;
    size_t code_size_ = 0;
};

// =============================================================================
// Copy-and-Patch Compiler
// =============================================================================

class CopyPatchCompiler {
public:
    CopyPatchCompiler()
        : exec_mem_(kDefaultMemorySize) {
        if (!exec_mem_.valid()) {
            spdlog::error("CopyPatchCompiler: Failed to initialize executable memory");
        }
    }

    Result<CompiledKernel> compile(const ir::IRBuilder& builder, ir::ValueId output) {
        if (!exec_mem_.valid()) {
            return Error(ErrorCode::kCompilationFailed,
                         "Executable memory not available");
        }

        spdlog::debug("CopyPatchCompiler: Compiling IR with {} nodes",
                      builder.nodes().size());

        // Calculate required code size
        size_t total_size = 0;
        for (const auto* node : builder.nodes()) {
            auto* stencil = findStencil(node->opCode(),
                                         node->type().scalarType());
            if (stencil) {
                total_size += stencil->code.size();
            } else {
                spdlog::warn("No stencil for op {}", ir::opCodeName(node->opCode()));
            }
        }

        // Add prologue/epilogue overhead
        total_size += 64;

        // Allocate space
        void* code_ptr = exec_mem_.allocate(total_size, 16);
        if (!code_ptr) {
            return Error(ErrorCode::kAllocationFailed,
                         "Not enough executable memory");
        }

        // Copy and patch stencils
        uint8_t* write_ptr = static_cast<uint8_t*>(code_ptr);

        // TODO: Implement actual copy-and-patch logic
        // 1. Copy each stencil's machine code
        // 2. Patch holes with actual addresses/values
        // 3. Link stencils together

        // For now, just create a stub that does nothing
        // (Real implementation would generate actual SIMD code)

        // x86-64 RET instruction
        *write_ptr++ = 0xC3;

        auto fn = reinterpret_cast<CompiledKernel::KernelFn>(code_ptr);
        return CompiledKernel(fn, total_size);
    }

    [[nodiscard]] size_t memoryRemaining() const {
        return exec_mem_.remaining();
    }

private:
    static constexpr size_t kDefaultMemorySize = 16 * 1024 * 1024;  // 16 MB
    ExecutableMemory exec_mem_;
};

// =============================================================================
// Global Compiler Instance
// =============================================================================

namespace {
CopyPatchCompiler* g_compiler = nullptr;
}

Result<void> initializeCompiler() {
    if (g_compiler) {
        return {};  // Already initialized
    }

    g_compiler = new CopyPatchCompiler();
    if (!g_compiler) {
        return ErrorCode::kAllocationFailed;
    }

    spdlog::info("CopyPatchCompiler initialized");
    return {};
}

void shutdownCompiler() {
    delete g_compiler;
    g_compiler = nullptr;
}

Result<CompiledKernel> compileKernel(const ir::IRBuilder& builder,
                                      ir::ValueId output) {
    if (!g_compiler) {
        auto init_result = initializeCompiler();
        if (!init_result) {
            return init_result.error();
        }
    }

    return g_compiler->compile(builder, output);
}

}  // namespace jit
}  // namespace bud
