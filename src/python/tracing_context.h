// =============================================================================
// Bud Flow Lang - Tracing Context for Python Kernel Compilation
// =============================================================================
//
// JAX-style function tracing infrastructure. TracingContext manages the IR
// recording during @flow.kernel execution, enabling lazy evaluation and
// operation fusion.
//
// Design:
// - Thread-local active context using RAII pattern
// - Creates tracer Bunches that record operations to IR
// - Extracts optimized IR for JIT compilation
//

#pragma once

#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/type_system.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace bud {

// Forward declarations
class Bunch;
class BunchImpl;

// =============================================================================
// TracingContext - Manages IR recording during kernel tracing
// =============================================================================

class TracingContext {
  public:
    TracingContext();
    ~TracingContext();

    // Non-copyable, non-movable (RAII resource management)
    TracingContext(const TracingContext&) = delete;
    TracingContext& operator=(const TracingContext&) = delete;
    TracingContext(TracingContext&&) = delete;
    TracingContext& operator=(TracingContext&&) = delete;

    // =========================================================================
    // IR Module Access
    // =========================================================================

    // Get the IR module being built during tracing
    [[nodiscard]] ir::IRModule& module() { return *module_; }
    [[nodiscard]] const ir::IRModule& module() const { return *module_; }

    // Get the IR builder for adding operations
    [[nodiscard]] ir::IRBuilder& builder() { return module_->builder(); }

    // =========================================================================
    // Tracer Bunch Creation
    // =========================================================================

    // Create a tracer Bunch representing a kernel input parameter
    // Records a parameter node in IR and returns a Bunch that will record ops
    [[nodiscard]] Bunch createTracerInput(size_t param_index, ScalarType dtype, const Shape& shape);

    // =========================================================================
    // IR Extraction
    // =========================================================================

    // Extract the output ValueId from a result Bunch
    // Returns invalid ValueId if the Bunch is not a tracer
    [[nodiscard]] ir::ValueId extractOutput(const Bunch& result) const;

    // Get all input parameter ValueIds in order
    [[nodiscard]] const std::vector<ir::ValueId>& inputIds() const { return input_ids_; }

    // Release ownership of the IR module (for transferring to CompiledKernel)
    [[nodiscard]] std::unique_ptr<ir::IRModule> releaseModule() { return std::move(module_); }

    // =========================================================================
    // Thread-Local Context Management
    // =========================================================================

    // Get the current active tracing context (nullptr if not tracing)
    [[nodiscard]] static TracingContext* current();

    // Check if currently in a tracing context
    [[nodiscard]] static bool isTracing() { return current() != nullptr; }

  private:
    // Set the current thread-local context (called by constructor/destructor)
    static void setCurrent(TracingContext* ctx);

    // The IR module being built
    std::unique_ptr<ir::IRModule> module_;

    // Input parameter ValueIds (in order of creation)
    std::vector<ir::ValueId> input_ids_;

    // Previous context (for nested tracing support, though rare)
    TracingContext* prev_context_ = nullptr;
};

// =============================================================================
// KernelSignature - Cache key for compiled kernels
// =============================================================================

struct KernelSignature {
    std::vector<ScalarType> input_dtypes;
    std::vector<size_t> input_sizes;

    bool operator==(const KernelSignature& other) const {
        return input_dtypes == other.input_dtypes && input_sizes == other.input_sizes;
    }
};

// Hash function for KernelSignature
struct KernelSignatureHash {
    size_t operator()(const KernelSignature& sig) const {
        size_t hash = 0;
        for (auto dtype : sig.input_dtypes) {
            hash ^=
                std::hash<int>{}(static_cast<int>(dtype)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        for (auto size : sig.input_sizes) {
            hash ^= std::hash<size_t>{}(size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// =============================================================================
// CompiledKernel - A traced and optimized kernel ready for execution
// =============================================================================

class CompiledKernel {
  public:
    CompiledKernel(std::unique_ptr<ir::IRModule> module, ir::ValueId output_id,
                   std::vector<ir::ValueId> input_ids);

    // Execute the kernel with actual input data
    // Returns the result Bunch or error
    [[nodiscard]] Result<Bunch> execute(const std::vector<const Bunch*>& inputs);

    // Get IR module for debugging/inspection
    [[nodiscard]] const ir::IRModule& module() const { return *module_; }
    [[nodiscard]] ir::ValueId outputId() const { return output_id_; }

  private:
    std::unique_ptr<ir::IRModule> module_;
    ir::ValueId output_id_;
    std::vector<ir::ValueId> input_ids_;
};

// =============================================================================
// KernelCache - Thread-safe cache for compiled kernels
// =============================================================================

class KernelCache {
  public:
    KernelCache() = default;

    // Find a cached kernel (returns nullptr if not found)
    [[nodiscard]] CompiledKernel* find(const KernelSignature& sig);

    // Insert a compiled kernel into the cache
    // Returns pointer to the cached kernel
    CompiledKernel* insert(const KernelSignature& sig, std::unique_ptr<CompiledKernel> kernel);

    // Clear all cached kernels
    void clear();

    // Get cache size
    [[nodiscard]] size_t size() const;

  private:
    mutable std::mutex mutex_;
    std::unordered_map<KernelSignature, std::unique_ptr<CompiledKernel>, KernelSignatureHash>
        cache_;
};

// =============================================================================
// Optimization Pipeline
// =============================================================================

// Run optimization passes on traced IR
// opt_level: 0 = none, 1 = basic, 2 = standard, 3 = aggressive
Result<void> runOptimizationPipeline(ir::IRModule& module, int opt_level);

}  // namespace bud
