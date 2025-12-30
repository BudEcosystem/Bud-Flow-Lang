// =============================================================================
// Bud Flow Lang - Copy-and-Patch JIT Compiler
// =============================================================================
//
// Tier 1 JIT compiler using copy-and-patch technique.
// Key advantage: ~1ms compilation time (vs 100ms+ for LLVM)
//
// The copy-and-patch approach:
// 1. Pre-compiled code stencils with "holes" for dynamic values
// 2. At compile time: copy stencils to executable memory
// 3. Patch holes with actual addresses and values
// 4. Link stencils together into complete kernels
//
// =============================================================================

#include "bud_flow_lang/error.h"
#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/stencil.h"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>

namespace bud {
namespace jit {

// =============================================================================
// Executable Memory Management
// =============================================================================

class ExecutableMemory {
  public:
    ExecutableMemory(size_t size) : size_(size) {
        // W^X Security: Allocate as read-write first (NOT executable)
        ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

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
        : ptr_(other.ptr_),
          size_(other.size_),
          used_(other.used_),
          is_executable_(other.is_executable_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.used_ = 0;
        other.is_executable_ = false;
    }

    void* allocate(size_t bytes, size_t alignment = 16) {
        // Cannot allocate after making executable
        if (is_executable_) {
            spdlog::error("ExecutableMemory: Cannot allocate after makeExecutable()");
            return nullptr;
        }

        // Align the current position
        size_t aligned = (used_ + alignment - 1) & ~(alignment - 1);

        if (aligned + bytes > size_) {
            return nullptr;  // Out of space
        }

        void* result = static_cast<char*>(ptr_) + aligned;
        used_ = aligned + bytes;
        return result;
    }

    // W^X Security: Transition from writable to executable
    [[nodiscard]] bool makeExecutable() {
        if (!ptr_ || is_executable_) {
            return is_executable_;
        }

        // Change protection from RW to RX (removes write permission)
        if (mprotect(ptr_, size_, PROT_READ | PROT_EXEC) != 0) {
            spdlog::error("ExecutableMemory: mprotect failed to make memory executable");
            return false;
        }

        // Flush instruction cache on architectures that require it (ARM, etc.)
#if defined(__aarch64__) || defined(__arm__)
        __builtin___clear_cache(static_cast<char*>(ptr_), static_cast<char*>(ptr_) + used_);
#elif defined(__riscv)
        __builtin___clear_cache(static_cast<char*>(ptr_), static_cast<char*>(ptr_) + used_);
#endif
        // x86/x86_64 has coherent instruction caches, no flush needed

        is_executable_ = true;
        spdlog::debug("ExecutableMemory: Made {} bytes executable", used_);
        return true;
    }

    // Reset memory for reuse (makes it writable again)
    [[nodiscard]] bool reset() {
        if (!ptr_)
            return false;

        if (is_executable_) {
            // Make writable again
            if (mprotect(ptr_, size_, PROT_READ | PROT_WRITE) != 0) {
                spdlog::error("ExecutableMemory: mprotect failed to reset memory");
                return false;
            }
            is_executable_ = false;
        }

        used_ = 0;
        return true;
    }

    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
    [[nodiscard]] bool isExecutable() const { return is_executable_; }
    [[nodiscard]] size_t remaining() const { return size_ - used_; }
    [[nodiscard]] size_t used() const { return used_; }
    [[nodiscard]] void* base() const { return ptr_; }

  private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    size_t used_ = 0;
    bool is_executable_ = false;
};

// =============================================================================
// Compiled Kernel
// =============================================================================

class CompiledKernel {
  public:
    using KernelFn = void (*)(void** args);

    CompiledKernel() = default;
    explicit CompiledKernel(KernelFn fn, size_t code_size, size_t num_ops)
        : fn_(fn), code_size_(code_size), num_ops_(num_ops) {}

    void operator()(void** args) const {
        if (fn_) {
            fn_(args);
        }
    }

    [[nodiscard]] bool valid() const { return fn_ != nullptr; }
    [[nodiscard]] size_t codeSize() const { return code_size_; }
    [[nodiscard]] size_t numOps() const { return num_ops_; }

  private:
    KernelFn fn_ = nullptr;
    size_t code_size_ = 0;
    size_t num_ops_ = 0;
};

// =============================================================================
// Copy-and-Patch Compiler
// =============================================================================

class CopyPatchCompiler {
  public:
    CopyPatchCompiler() : exec_mem_(kDefaultMemorySize) {
        if (!exec_mem_.valid()) {
            spdlog::error("CopyPatchCompiler: Failed to initialize executable memory");
        } else {
            spdlog::info("CopyPatchCompiler: Initialized with {} MB executable memory",
                         kDefaultMemorySize / (1024 * 1024));
        }
    }

    Result<CompiledKernel> compile(const ir::IRBuilder& builder, ir::ValueId output) {
        std::lock_guard<std::mutex> lock(compile_mutex_);

        auto start_time = std::chrono::high_resolution_clock::now();

        if (!exec_mem_.valid()) {
            return Error(ErrorCode::kCompilationFailed, "Executable memory not available");
        }

        const auto& nodes = builder.nodes();
        if (nodes.empty()) {
            return Error(ErrorCode::kInvalidInput, "Empty IR graph");
        }

        spdlog::debug("CopyPatchCompiler: Compiling IR with {} nodes, output=%{}", nodes.size(),
                      output.id);

        // Phase 1: Topological sort of nodes (already in SSA order)
        std::vector<const ir::IRNode*> schedule;
        schedule.reserve(nodes.size());
        for (const auto* node : nodes) {
            if (node && !node->isDead()) {
                schedule.push_back(node);
            }
        }

        if (schedule.empty()) {
            return Error(ErrorCode::kInvalidInput, "No live nodes in IR graph");
        }

        // Phase 2: Calculate required code size
        size_t total_code_size = 0;
        size_t stencil_count = 0;

        for (const auto* node : schedule) {
            auto* stencil = findStencil(node->opCode(), node->type().scalarType());
            if (stencil) {
                total_code_size += stencil->code.size();
                ++stencil_count;
            } else {
                spdlog::debug("No stencil for op {} (type: {})", ir::opCodeName(node->opCode()),
                              scalarTypeName(node->type().scalarType()));
            }
        }

        // Add prologue/epilogue overhead
        total_code_size += 64;

        // Phase 3: Allocate executable memory
        void* code_ptr = exec_mem_.allocate(total_code_size, 16);
        if (!code_ptr) {
            // Try to reset and allocate again
            if (exec_mem_.reset()) {
                code_ptr = exec_mem_.allocate(total_code_size, 16);
            }
            if (!code_ptr) {
                return Error(ErrorCode::kAllocationFailed, "Not enough executable memory");
            }
        }

        uint8_t* write_ptr = static_cast<uint8_t*>(code_ptr);
        uint8_t* code_start = write_ptr;

        // Phase 4: Generate kernel prologue
        // For now, we generate a simple wrapper that sets up args and calls stencils

        // x86-64 prologue: save registers
        *write_ptr++ = 0x55;  // push rbp
        *write_ptr++ = 0x48;  // mov rbp, rsp
        *write_ptr++ = 0x89;
        *write_ptr++ = 0xE5;
        *write_ptr++ = 0x53;  // push rbx
        *write_ptr++ = 0x48;  // mov rbx, rdi  ; save args
        *write_ptr++ = 0x89;
        *write_ptr++ = 0xFB;

        // Phase 5: Copy and patch stencils
        // For a simple linear IR, we execute stencils in order
        // Each stencil reads from args array and writes output

        size_t ops_compiled = 0;
        for (const auto* node : schedule) {
            auto* stencil = findStencil(node->opCode(), node->type().scalarType());
            if (!stencil) {
                continue;
            }

            // Copy stencil code
            std::memcpy(write_ptr, stencil->code.data(), stencil->code.size());

            // Patch holes
            // The stencil expects args to be passed in RDI
            // We maintain args pointer in RBX

            write_ptr += stencil->code.size();
            ++ops_compiled;
        }

        // Phase 6: Generate kernel epilogue
        *write_ptr++ = 0x5B;  // pop rbx
        *write_ptr++ = 0x5D;  // pop rbp
        *write_ptr++ = 0xC3;  // ret

        size_t actual_code_size = write_ptr - code_start;

        // Phase 7: Make code executable
        if (!exec_mem_.makeExecutable()) {
            return Error(ErrorCode::kCompilationFailed, "Failed to make code executable");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto compile_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        spdlog::debug("CopyPatchCompiler: Compiled {} ops in {} bytes ({} µs)", ops_compiled,
                      actual_code_size, compile_time.count());

        // Update statistics
        ++total_compilations_;
        total_compile_time_us_ += compile_time.count();

        auto fn = reinterpret_cast<CompiledKernel::KernelFn>(code_ptr);
        return CompiledKernel(fn, actual_code_size, ops_compiled);
    }

    // Compile a single operation (for simple cases)
    Result<CompiledKernel> compileSingleOp(ir::OpCode op, ScalarType dtype) {
        std::lock_guard<std::mutex> lock(compile_mutex_);

        auto* stencil = findStencil(op, dtype);
        if (!stencil) {
            return Error(
                ErrorCode::kNotSupported,
                fmt::format("No stencil for {} ({})", ir::opCodeName(op), scalarTypeName(dtype)));
        }

        // Allocate memory for stencil + wrapper
        size_t total_size = stencil->code.size() + 16;  // prologue/epilogue

        void* code_ptr = exec_mem_.allocate(total_size, stencil->alignment);
        if (!code_ptr) {
            if (exec_mem_.reset()) {
                code_ptr = exec_mem_.allocate(total_size, stencil->alignment);
            }
            if (!code_ptr) {
                return Error(ErrorCode::kAllocationFailed, "Not enough executable memory");
            }
        }

        uint8_t* write_ptr = static_cast<uint8_t*>(code_ptr);

        // Just copy the stencil - it already has prologue/epilogue
        std::memcpy(write_ptr, stencil->code.data(), stencil->code.size());

        if (!exec_mem_.makeExecutable()) {
            return Error(ErrorCode::kCompilationFailed, "Failed to make code executable");
        }

        auto fn = reinterpret_cast<CompiledKernel::KernelFn>(code_ptr);
        return CompiledKernel(fn, stencil->code.size(), 1);
    }

    [[nodiscard]] size_t memoryRemaining() const { return exec_mem_.remaining(); }
    [[nodiscard]] size_t memoryUsed() const { return exec_mem_.used(); }
    [[nodiscard]] size_t totalCompilations() const { return total_compilations_; }
    [[nodiscard]] uint64_t totalCompileTimeUs() const { return total_compile_time_us_; }

  private:
    static constexpr size_t kDefaultMemorySize = 16 * 1024 * 1024;  // 16 MB
    ExecutableMemory exec_mem_;
    mutable std::mutex compile_mutex_;  // Thread safety for compilation

    // Statistics
    size_t total_compilations_ = 0;
    uint64_t total_compile_time_us_ = 0;
};

// =============================================================================
// Kernel Cache
// =============================================================================

class KernelCache {
  public:
    static KernelCache& instance() {
        static KernelCache cache;
        return cache;
    }

    // Look up a cached kernel
    const CompiledKernel* find(uint64_t hash) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(hash);
        if (it != cache_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    // Insert a kernel into the cache
    void insert(uint64_t hash, CompiledKernel kernel) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[hash] = std::move(kernel);
    }

    // Clear the cache
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

  private:
    KernelCache() = default;

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, CompiledKernel> cache_;
};

// =============================================================================
// IR Hash for Cache Lookup
// =============================================================================

static uint64_t hashIR(const ir::IRBuilder& builder, ir::ValueId output) {
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t fnv_prime = 1099511628211ULL;

    for (const auto* node : builder.nodes()) {
        if (!node || node->isDead())
            continue;

        // Hash opcode
        hash ^= static_cast<uint64_t>(node->opCode());
        hash *= fnv_prime;

        // Hash type
        hash ^= static_cast<uint64_t>(node->type().scalarType());
        hash *= fnv_prime;

        // Hash operands
        for (size_t i = 0; i < node->numOperands(); ++i) {
            hash ^= node->operand(i).id;
            hash *= fnv_prime;
        }
    }

    // Include output ID
    hash ^= output.id;
    hash *= fnv_prime;

    return hash;
}

// =============================================================================
// Global Compiler Instance (Thread-Safe Singleton)
// =============================================================================

namespace {
std::once_flag g_compiler_init_flag;
std::atomic<CopyPatchCompiler*> g_compiler{nullptr};
std::mutex g_compiler_shutdown_mutex;
}  // namespace

Result<void> initializeCompiler() {
    std::call_once(g_compiler_init_flag, []() {
        g_compiler.store(new CopyPatchCompiler(), std::memory_order_release);
        spdlog::info("CopyPatchCompiler initialized with {} stencils available", stencilCount());
    });

    auto* compiler = g_compiler.load(std::memory_order_acquire);
    if (!compiler || !compiler->memoryRemaining()) {
        return Error(ErrorCode::kAllocationFailed, "Failed to initialize compiler");
    }

    return {};
}

void shutdownCompiler() {
    std::lock_guard<std::mutex> lock(g_compiler_shutdown_mutex);
    CopyPatchCompiler* compiler = g_compiler.exchange(nullptr, std::memory_order_acq_rel);
    if (compiler) {
        spdlog::info("CopyPatchCompiler shutdown: {} compilations, {} µs total",
                     compiler->totalCompilations(), compiler->totalCompileTimeUs());
        delete compiler;
    }
    // Note: g_compiler_init_flag cannot be reset, so re-initialization after shutdown
    // is not supported (by design - shutdown should only happen at program exit)
}

Result<CompiledKernel> compileKernel(const ir::IRBuilder& builder, ir::ValueId output) {
    // Ensure compiler is initialized
    auto init_result = initializeCompiler();
    if (!init_result) {
        return init_result.error();
    }

    // Check cache first
    uint64_t hash = hashIR(builder, output);
    if (auto* cached = KernelCache::instance().find(hash)) {
        spdlog::debug("Cache hit for kernel hash {:016x}", hash);
        return *cached;
    }

    // Compile
    auto* compiler = g_compiler.load(std::memory_order_acquire);
    if (!compiler) {
        return Error(ErrorCode::kCompilationFailed, "Compiler not available");
    }

    auto result = compiler->compile(builder, output);
    if (result) {
        // Cache the result
        KernelCache::instance().insert(hash, *result);
    }

    return result;
}

Result<CompiledKernel> compileSingleOp(ir::OpCode op, ScalarType dtype) {
    auto init_result = initializeCompiler();
    if (!init_result) {
        return init_result.error();
    }

    auto* compiler = g_compiler.load(std::memory_order_acquire);
    if (!compiler) {
        return Error(ErrorCode::kCompilationFailed, "Compiler not available");
    }

    return compiler->compileSingleOp(op, dtype);
}

// =============================================================================
// Execution Helpers
// =============================================================================

// Execute a compiled kernel with arguments
void executeKernel(const CompiledKernel& kernel, void** args) {
    if (kernel.valid()) {
        kernel(args);
    }
}

// Execute a single binary operation using JIT-compiled code
Result<void> executeJitBinaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input_a,
                                const void* input_b, size_t count) {
    auto* stencil = findStencil(op, dtype);
    if (!stencil) {
        return Error(ErrorCode::kNotSupported, "No stencil for operation");
    }

    // Get Highway function pointer
    void* func_ptr = getHwyFunctionPtr(op, dtype);
    if (!func_ptr) {
        return Error(ErrorCode::kNotSupported, "No Highway function for operation");
    }

    // For dispatch stencils, set up args array
    void* args[5] = {output, const_cast<void*>(input_a), const_cast<void*>(input_b),
                     reinterpret_cast<void*>(count), func_ptr};

    // Compile and execute
    auto kernel_result = compileSingleOp(op, dtype);
    if (!kernel_result) {
        return kernel_result.error();
    }

    kernel_result->operator()(args);
    return {};
}

// Execute a single unary operation using JIT-compiled code
Result<void> executeJitUnaryOp(ir::OpCode op, ScalarType dtype, void* output, const void* input,
                               size_t count) {
    auto* stencil = findStencil(op, dtype);
    if (!stencil) {
        return Error(ErrorCode::kNotSupported, "No stencil for operation");
    }

    void* func_ptr = getHwyFunctionPtr(op, dtype);
    if (!func_ptr) {
        return Error(ErrorCode::kNotSupported, "No Highway function for operation");
    }

    void* args[4] = {output, const_cast<void*>(input), reinterpret_cast<void*>(count), func_ptr};

    auto kernel_result = compileSingleOp(op, dtype);
    if (!kernel_result) {
        return kernel_result.error();
    }

    kernel_result->operator()(args);
    return {};
}

// Execute FMA operation
Result<void> executeJitFmaOp(ScalarType dtype, void* output, const void* input_a,
                             const void* input_b, const void* input_c, size_t count) {
    auto* stencil = findStencil(ir::OpCode::kFma, dtype);
    if (!stencil) {
        return Error(ErrorCode::kNotSupported, "No FMA stencil");
    }

    void* func_ptr = getHwyFunctionPtr(ir::OpCode::kFma, dtype);
    if (!func_ptr) {
        return Error(ErrorCode::kNotSupported, "No Highway FMA function");
    }

    void* args[6] = {output,
                     const_cast<void*>(input_a),
                     const_cast<void*>(input_b),
                     const_cast<void*>(input_c),
                     reinterpret_cast<void*>(count),
                     func_ptr};

    auto kernel_result = compileSingleOp(ir::OpCode::kFma, dtype);
    if (!kernel_result) {
        return kernel_result.error();
    }

    kernel_result->operator()(args);
    return {};
}

// =============================================================================
// Statistics API
// =============================================================================

JitStats getJitStats() {
    JitStats stats = {};

    auto* compiler = g_compiler.load(std::memory_order_acquire);
    if (compiler) {
        stats.total_compilations = compiler->totalCompilations();
        stats.total_compile_time_us = compiler->totalCompileTimeUs();
        stats.memory_used = compiler->memoryUsed();
        stats.memory_remaining = compiler->memoryRemaining();
    }

    stats.cache_size = KernelCache::instance().size();
    stats.stencil_count = stencilCount();

    return stats;
}

}  // namespace jit
}  // namespace bud
