# Bud Flow Lang: Comprehensive Implementation Plan

## State-of-the-Art JIT-Compiled Python DSL for SIMD Programming

**Version:** 2.0 (Critically Revised)
**Date:** December 2024
**Author:** SIMD Architecture Analysis

---

## Critical Revision Notes (v2.0)

This revision incorporates critical findings from SOTA research analysis:

### Key Changes from v1.0

1. **JIT Backend Revision**: Copy-and-Patch alone may be insufficient (only 2-9% improvement in CPython). Hybrid approach recommended: Copy-and-Patch for instant response + LLVM/MIR for peak performance.

2. **IR Design Enhancement**: Adopt Weld-style lazy evaluation IR for maximum fusion opportunities (6-32x speedups demonstrated across NumPy/Pandas/TensorFlow).

3. **Fusion Strategy Upgrade**: Move from rule-based fusion to lazy evaluation + deferred execution model.

4. **Memory Management**: Add arena allocator for SIMD-aligned allocations (50-100x allocation speedups).

5. **Alternative Considerations**: MIR (91% of GCC -O2), Cranelift (40% faster compilation than LLVM), MLIR dialects.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
   - 1.1 Objective
   - 1.2 Key Design Decisions
   - 1.3 Critical Concerns and Mitigations (NEW)
   - 1.4 Target Metrics
2. [First Principles Analysis](#2-first-principles-analysis)
   - 2.1 The Fundamental Problem
   - 2.2 First Principles Solution
   - 2.3 JIT Backend Analysis (NEW - Critical Revision)
   - 2.4 Lazy Evaluation Architecture (NEW - Weld-inspired)
   - 2.5 Why This Will Succeed
   - 2.6 Memory Management Strategy (NEW)
   - 2.7 Edge Cases and Handling (NEW)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Design](#4-component-design)
5. [Implementation Phases](#5-implementation-phases)
6. [Technical Specifications](#6-technical-specifications)
7. [Risk Analysis & Mitigations](#7-risk-analysis--mitigations)
8. [Appendices](#8-appendices)
   - 8.1 References (Updated with 15+ new sources)
   - 8.2 Glossary (Expanded)
   - 8.3 File Structure
   - 8.4 Version History (NEW)

---

## 1. Executive Summary

### 1.1 Objective

Create **Bud Flow Lang** - a Python DSL that makes SIMD programming accessible to beginners while providing expert-level control, achieving:

- **10-20x speedups** over NumPy for compute-bound operations
- **Sub-millisecond JIT compilation** via copy-and-patch technique
- **Zero-overhead abstractions** through Highway code generation
- **Cross-platform portability** across x86 (SSE4/AVX2/AVX-512), ARM (NEON/SVE/SVE2), RISC-V (RVV)

### 1.2 Key Design Decisions (Based on SOTA Research)

| Decision | Rationale | Source |
|----------|-----------|--------|
| **Hybrid JIT (Copy-and-Patch + LLVM)** | Copy-and-Patch for instant Tier 1, LLVM/MIR for peak Tier 2 | CPython PEP 744, MIR, Cranelift research |
| **Weld-style Lazy Evaluation** | Deferred execution enables cross-operation fusion (6-32x speedups) | Stanford Weld, VLDB 2018 |
| **Tiered Compilation** | Balance startup latency vs peak performance | .NET 8 Dynamic PGO |
| **Speculative Optimization + Deopt** | Aggressive optimization with safe fallbacks | V8 TurboFan |
| **Highway Backend** | Only portable SIMD library with scalable vector support | Google Highway |
| **nanobind** | 4x faster compilation, 5x smaller binaries vs pybind11 | nanobind benchmarks |
| **Arena Allocator** | 50-100x allocation speedups for SIMD-aligned memory | Game engine patterns |
| **Tag-based Type System** | Zero-cost abstraction matching Highway's design | Highway architecture |

### 1.3 Critical Concerns and Mitigations (NEW)

| Concern | Analysis | Mitigation |
|---------|----------|------------|
| **Copy-and-Patch limitations** | Only 2-9% improvement in CPython interpreter context | For SIMD stencils, pre-compiled SIMD code differs from interpreter micro-ops; validate with benchmarks |
| **Python AST limitations** | `inspect.getsource()` fails for lambdas, closures, dynamic code | Require file-defined functions; provide clear error messages |
| **LLVM compilation latency** | Can take 100+ seconds for complex functions (Numba reports) | Async compilation in background; return Tier 1 code immediately |
| **Numerical precision variance** | Different SIMD widths may produce slightly different results | Use Highway's consistent reduction algorithms; document tolerance expectations |
| **Memory alignment edge cases** | User-provided arrays may not be aligned | Detect alignment at runtime; use masked loads/stores for unaligned portions |

### 1.4 Target Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| First call latency | < 5ms | Copy-and-patch enables instant compilation |
| Peak performance | 90%+ of hand-tuned C++ | Highway generates optimal SIMD code |
| Binary size | < 5MB | nanobind + precompiled stencils |
| Supported targets | 10+ ISAs | Highway handles all portability |

---

## 2. First Principles Analysis

### 2.1 The Fundamental Problem

SIMD programming has three core challenges:

1. **Abstraction Gap**: Python developers think in terms of arrays and transformations, not registers and lanes
2. **Portability Gap**: Different CPUs have different SIMD capabilities (128-bit → 2048-bit vectors)
3. **Performance Gap**: Interpreted Python is 100-1000x slower than optimized SIMD C++

### 2.2 First Principles Solution

**Principle 1: Language follows mental model**
```
Python Mental Model          SIMD Reality              FlowSIMD Bridge
─────────────────────────────────────────────────────────────────────
"A collection of numbers"  → SIMD register(s)       → Bunch abstraction
"Keep positives"           → Mask + Compress        → Pattern abstraction
"Sum all elements"         → Horizontal reduction   → .sum() method
"Transform each element"   → Parallel operation     → Arithmetic operators
```

**Principle 2: Progressive disclosure over progressive complexity**
```
Tier 1 (Beginner):   flow(x) * 2 + 1           → Auto-vectorized, no SIMD knowledge
Tier 2 (Developer):  with hint.unroll(4): ...  → Performance tuning, no intrinsics
Tier 3 (Expert):     ScalableTag[float]()      → Full Highway access
```

**Principle 3: Compilation speed enables iteration**
```
Traditional JIT (LLVM):     500-2000ms per kernel → Unusable for interactive development
Copy-and-Patch:             1-5ms per kernel      → Feels instant
Optimization (async):       50-200ms (background) → Peak performance when needed
```

**Principle 4: Speculation enables optimization**
```
Observation:   Most Python code is monomorphic (same types at same locations)
Strategy:      Speculate types, guard assumptions, deoptimize on failure
Result:        90%+ of optimizations are safe, 10% trigger deopt path
```

### 2.3 JIT Backend Analysis (Critical Revision)

Based on SOTA research, the following JIT backends were evaluated:

#### 2.3.1 Copy-and-Patch (Baseline Choice)

**Pros:**
- 100x faster compilation than LLVM -O0
- Pre-compiled stencils for each SIMD operation
- Minimal runtime dependencies

**Concerns:**
- CPython's implementation only shows 2-9% runtime improvement
- However, for SIMD code, stencils are pre-compiled vectorized loops, not interpreter micro-ops
- **Decision:** Keep for Tier 1, but validate with benchmarks

#### 2.3.2 MIR (Medium Internal Representation)

**Pros:**
- 91% of GCC -O2 performance
- 100x smaller than LLVM codebase
- Fast compilation (~1ms per function)
- Supports x86_64, aarch64, ppc64le, s390x, riscv64

**Concerns:**
- Some real-world tests show only 1.06x faster than LLVM
- Less optimization than full LLVM

**Decision:** Consider as alternative Tier 2 backend for faster compilation

#### 2.3.3 Cranelift

**Pros:**
- 40% faster compilation than LLVM (125 vs 211 CPU-seconds)
- Written in Rust, 200K LOC (vs LLVM 20M LOC)
- Used in Wasmtime, Firefox

**Concerns:**
- Code approximately 2x slower than LLVM on some benchmarks
- Primarily designed for WebAssembly

**Decision:** Not suitable for peak SIMD performance; LLVM preferred for Tier 2

#### 2.3.4 Recommended Hybrid Approach

```
Tier 0: NumPy Interpreter (profiling)
    │
    │ 100 calls
    ▼
Tier 1: Copy-and-Patch JIT (~1-5ms)
    │   - Pre-compiled SIMD stencils
    │   - Immediate execution
    │
    │ 1000 calls + async compilation
    ▼
Tier 2: LLVM with Highway (~50-200ms, background)
    │   - Full optimization passes
    │   - Profile-guided optimization
    │   - Multi-target dispatch
```

### 2.4 Lazy Evaluation Architecture (NEW - Weld-inspired)

A critical insight from Stanford's Weld project: lazy evaluation enables cross-operation fusion that achieves 6-32x speedups.

#### 2.4.1 Lazy Object Model

```python
# Instead of immediate execution:
result = (flow(x) * 2 + 1).sqrt()  # Returns LazyExpr, not array

# Execution deferred until:
concrete = result.execute()  # Or when leaving @flow.kernel scope
```

#### 2.4.2 Benefits of Lazy Evaluation

1. **Cross-Operation Fusion**: Multiple operations fused into single pass
2. **Memory Traffic Reduction**: Intermediate results stay in registers
3. **Optimization Window**: Larger IR graph enables better optimization
4. **Adaptive Execution**: Choose execution strategy based on data size

#### 2.4.3 Fusion Opportunities

```python
# User code:
a = flow(x) * 2
b = a + 1
c = b.sqrt()
d = c.sum()

# Without lazy evaluation: 4 loops, 4 memory writes
# With lazy evaluation: 1 fused loop, 1 memory write
```

### 2.5 Why This Will Succeed

| Existing Solution | Limitation | FlowSIMD Advantage |
|-------------------|------------|-------------------|
| NumPy | No operation fusion, fixed SIMD width | JIT-fused operations, dynamic width |
| Numba | LLVM slow, no scalable vectors | Copy-and-patch fast, SVE/RVV support |
| JAX/XLA | Heavy runtime, GPU-focused | Lightweight, CPU-optimized |
| Cython | Manual type annotations | Automatic type inference |
| PyTorch | Deep learning focus | General-purpose SIMD |
| Taichi | GPU-focused, complex setup | Simpler CPU SIMD focus |
| Weld | Research project, not maintained | Production-grade implementation |

### 2.6 Memory Management Strategy (NEW)

#### 2.6.1 Arena Allocator for SIMD

Arena allocators provide 50-100x speedups for allocation-heavy workloads:

```cpp
class SIMDArena {
    static constexpr size_t ALIGNMENT = 64;  // AVX-512 alignment
    static constexpr size_t BLOCK_SIZE = 1 << 20;  // 1MB blocks

    struct Block {
        alignas(64) char data[BLOCK_SIZE];
        size_t offset = 0;
    };

    std::vector<std::unique_ptr<Block>> blocks_;

public:
    void* allocate(size_t size) {
        // Align size to SIMD boundary
        size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

        // Find or create block with space
        if (blocks_.empty() || blocks_.back()->offset + size > BLOCK_SIZE) {
            blocks_.push_back(std::make_unique<Block>());
        }

        Block& block = *blocks_.back();
        void* ptr = block.data + block.offset;
        block.offset += size;
        return ptr;
    }

    void reset() {
        for (auto& block : blocks_) {
            block->offset = 0;
        }
    }
};
```

#### 2.6.2 Integration with Kernel Execution

```python
@flow.kernel
def process(x):
    # Arena allocated for kernel scope
    # All intermediates use arena memory
    return (flow(x) * 2 + 1).sqrt()
    # Arena reset after return
```

### 2.7 Edge Cases and Handling (NEW)

#### 2.7.1 Python AST Limitations

| Limitation | Detection | Handling |
|------------|-----------|----------|
| Lambda functions | Check `func.__name__ == '<lambda>'` | Raise descriptive error with workaround |
| Dynamic code (exec/eval) | `inspect.getsource()` raises OSError | Suggest file-based definition |
| Closures with non-serializable captures | Type check captured variables | Warn about performance; fallback to interpreter |
| Interactive/REPL code | Check `__file__` attribute | Support via string-based parsing |

#### 2.7.2 Numerical Edge Cases

| Case | Risk | Mitigation |
|------|------|------------|
| Denormalized floats | 100x slower on some CPUs | Flush-to-zero mode with user opt-out |
| NaN/Inf propagation | Different behavior across ISAs | Use Highway's consistent handling |
| Reduction order sensitivity | Sum of 1M floats differs by 10^-4 | Document tolerance; offer Kahan summation |
| Integer overflow | Silent wraparound | Optional overflow checking mode |

#### 2.7.3 Memory Alignment Cases

```python
# Case 1: User array is aligned (fast path)
x = np.empty(1024, dtype=np.float32)  # Usually 16-byte aligned

# Case 2: User array is unaligned (slow path with masked load)
x = np.frombuffer(buffer, dtype=np.float32, offset=1)  # Unaligned

# Detection at runtime:
def get_alignment(arr):
    return arr.ctypes.data % 64  # Check AVX-512 alignment
```

#### 2.7.4 Deoptimization Loop Prevention

```python
class DeoptTracker:
    MAX_DEOPTS = 3
    COOLDOWN_CALLS = 1000

    def __init__(self):
        self.deopt_count = 0
        self.calls_since_deopt = 0

    def should_reoptimize(self):
        if self.deopt_count >= self.MAX_DEOPTS:
            # Too many deopts - stay at lower tier
            return False
        if self.calls_since_deopt < self.COOLDOWN_CALLS:
            # Cooldown period - gather more profile data
            return False
        return True
```

---

## 3. Architecture Overview

### 3.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BUD FLOW LANG RUNTIME                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Python    │     │   Flow IR    │     │   Highway    │                 │
│  │   Frontend  │────▶│   Compiler   │────▶│   Backend    │                 │
│  │             │     │              │     │              │                 │
│  │  • AST      │     │  • SSA Form  │     │  • C++ Gen   │                 │
│  │  • Types    │     │  • Optimize  │     │  • Multi-ISA │                 │
│  │  • Validate │     │  • Fusion    │     │  • Dispatch  │                 │
│  └─────────────┘     └──────────────┘     └──────────────┘                 │
│         │                   │                    │                          │
│         ▼                   ▼                    ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TIERED COMPILATION ENGINE                       │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  TIER 0: Interpreter          TIER 1: Baseline JIT                  │   │
│  │  ┌──────────────────┐         ┌──────────────────┐                  │   │
│  │  │ • Profiling      │  100    │ • Copy-and-Patch │                  │   │
│  │  │ • Type tracking  │ ─calls─▶│ • Pre-compiled   │                  │   │
│  │  │ • Branch counts  │         │   stencils       │                  │   │
│  │  └──────────────────┘         └──────────────────┘                  │   │
│  │                                       │                              │   │
│  │                                      1000                            │   │
│  │                                     calls                            │   │
│  │                                       │                              │   │
│  │                                       ▼                              │   │
│  │                        TIER 2: Optimizing JIT                        │   │
│  │                        ┌──────────────────┐                          │   │
│  │                        │ • PGO-guided     │                          │   │
│  │                        │ • Full Highway   │                          │   │
│  │                        │ • Multi-target   │                          │   │
│  │                        └──────────────────┘                          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      RUNTIME SERVICES                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Code Cache │  │  Dispatcher │  │  Profiler   │  │  Deoptim   │  │   │
│  │  │  (LRU)      │  │  (CPU Det.) │  │  (Counters) │  │  (Guards)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NATIVE CODE TARGETS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │  SSE4   │  │  AVX2   │  │ AVX-512 │  │  NEON   │  │  SVE2   │  ...     │
│  │ 128-bit │  │ 256-bit │  │ 512-bit │  │ 128-bit │  │ 256+bit │          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Pipeline

```
@flow.kernel
def process(x):                                 PYTHON SOURCE
    return (flow(x) * 2 + 1).sqrt()            ─────────────────
              │
              ▼
┌─────────────────────────────────┐
│  AST Extraction via inspect     │             FRONTEND
│  ─────────────────────────────  │            ─────────────────
│  FunctionDef                    │
│    └─ Return                    │
│         └─ Call: sqrt           │
│              └─ BinOp: +        │
│                   ├─ BinOp: *   │
│                   │    ├─ flow(x)
│                   │    └─ 2     │
│                   └─ 1          │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Flow IR (SSA Form)             │             IR GENERATION
│  ─────────────────────────────  │            ─────────────────
│  %0 = load x                    │
│  %1 = broadcast 2.0             │
│  %2 = mul %0, %1                │
│  %3 = broadcast 1.0             │
│  %4 = add %2, %3                │
│  %5 = sqrt %4                   │
│  return %5                      │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Optimized IR (Fused)           │             OPTIMIZATION
│  ─────────────────────────────  │            ─────────────────
│  %0 = load x                    │
│  %1 = fused_muladd_sqrt %0,     │  ◀── FMA + sqrt fusion
│         2.0, 1.0                │
│  return %1                      │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Highway C++ Code               │             CODE GENERATION
│  ─────────────────────────────  │            ─────────────────
│  template<class D>              │
│  void kernel(D d,               │
│              const float* in,   │
│              float* out,        │
│              size_t n) {        │
│    for (size_t i = 0; i < n;    │
│         i += Lanes(d)) {        │
│      auto v = Load(d, in + i);  │
│      v = Sqrt(MulAdd(v,         │
│        Set(d, 2.0f),            │
│        Set(d, 1.0f)));          │
│      Store(v, d, out + i);      │
│    }                            │
│  }                              │
│  HWY_EXPORT(kernel);            │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Compiled Native Code           │             COMPILATION
│  ─────────────────────────────  │            ─────────────────
│  AVX2: vmulps, vaddps, vsqrtps  │
│  AVX512: same with zmm regs     │
│  NEON: fmul, fadd, fsqrt        │
│  SVE: fmla, fsqrt (predicated)  │
└─────────────────────────────────┘
```

### 3.3 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER CODE                                         │
│  @flow.kernel                                                               │
│  def my_func(x):                                                            │
│      return flow(x) * 2                                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      KERNEL REGISTRY        │
                    │  ┌───────────────────────┐  │
                    │  │ Cache lookup by hash  │  │
                    │  │ if hit → return       │  │
                    │  │ if miss → compile     │  │
                    │  └───────────────────────┘  │
                    └──────────────┬──────────────┘
                                   │ miss
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  AST ANALYZER   │  │  TYPE INFERRER  │  │  IR BUILDER     │
    │                 │  │                 │  │                 │
    │ • Parse func    │  │ • Infer dtypes  │  │ • Build SSA IR  │
    │ • Extract ops   │  │ • Track shapes  │  │ • Add metadata  │
    │ • Validate      │  │ • Propagate     │  │ • Symbol table  │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     OPTIMIZATION PASSES    │
                    │  ┌─────────────────────┐  │
                    │  │ 1. Constant fold    │  │
                    │  │ 2. CSE              │  │
                    │  │ 3. Dead code elim   │  │
                    │  │ 4. Operation fusion │  │
                    │  │ 5. FMA detection    │  │
                    │  │ 6. Loop transforms  │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │  TIER 1: BASELINE   │               │  TIER 2: OPTIMIZING │
    │  (Copy-and-Patch)   │               │  (Full Highway)     │
    │                     │               │                     │
    │ • Select stencils   │               │ • Generate C++      │
    │ • Patch immediates  │               │ • Compile w/ Clang  │
    │ • Link together     │               │ • Multi-target      │
    │ • ~1-5ms            │               │ • ~50-200ms         │
    └──────────┬──────────┘               └──────────┬──────────┘
               │                                     │
               └──────────────┬──────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    DISPATCHER     │
                    │                   │
                    │ • CPU detection   │
                    │ • Target select   │
                    │ • Function cache  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  NATIVE EXECUTOR  │
                    │                   │
                    │ • Memory alloc    │
                    │ • Call kernel     │
                    │ • Return result   │
                    └───────────────────┘
```

---

## 4. Component Design

### 4.1 Python Frontend

#### 4.1.1 AST Analyzer

**Purpose:** Extract and validate Python AST from `@flow.kernel` decorated functions.

```python
# Core data structures
@dataclass
class FlowAST:
    """Analyzed AST node with SIMD semantics."""
    node_id: int
    op: FlowOp              # Operation type
    args: List['FlowAST']   # Child nodes
    dtype: Optional[DType]  # Inferred type
    shape: Optional[Shape]  # Inferred shape
    source_loc: SourceLoc   # For error messages

class ASTAnalyzer:
    """
    Extracts FlowSIMD operations from Python AST.

    Supported constructs:
    - Binary ops: +, -, *, /, **, //, %
    - Unary ops: -, abs()
    - Comparisons: <, <=, >, >=, ==, !=
    - Method calls: .sqrt(), .exp(), .log(), .sum(), etc.
    - Conditionals: .where(cond, else_=)
    - Reductions: .sum(), .min(), .max(), .mean()
    """

    SUPPORTED_BINOPS = {
        ast.Add: FlowOp.ADD,
        ast.Sub: FlowOp.SUB,
        ast.Mult: FlowOp.MUL,
        ast.Div: FlowOp.DIV,
        ast.Pow: FlowOp.POW,
        ast.FloorDiv: FlowOp.FLOORDIV,
        ast.Mod: FlowOp.MOD,
    }

    SUPPORTED_METHODS = {
        'sqrt': FlowOp.SQRT,
        'exp': FlowOp.EXP,
        'log': FlowOp.LOG,
        'sin': FlowOp.SIN,
        'cos': FlowOp.COS,
        'abs': FlowOp.ABS,
        'sum': FlowOp.REDUCE_SUM,
        'min': FlowOp.REDUCE_MIN,
        'max': FlowOp.REDUCE_MAX,
        'where': FlowOp.SELECT,
        'keep': FlowOp.COMPRESS,
        'reverse': FlowOp.REVERSE,
        # ... 50+ operations
    }

    def analyze(self, func: Callable) -> FlowAST:
        """Main entry point for AST analysis."""
        source = inspect.getsource(func)
        tree = ast.parse(source)

        # Validate function structure
        self._validate_kernel_structure(tree)

        # Extract operations
        return self._visit(tree.body[0])

    def _visit(self, node: ast.AST) -> FlowAST:
        """Recursive visitor dispatching to specific handlers."""
        handler = getattr(self, f'_visit_{node.__class__.__name__}', None)
        if handler is None:
            raise UnsupportedConstruct(node)
        return handler(node)
```

#### 4.1.2 Type Inferrer

**Purpose:** Infer NumPy dtypes and array shapes from usage patterns.

```python
@dataclass
class TypeInfo:
    dtype: np.dtype        # numpy.float32, numpy.int64, etc.
    shape: Tuple[int, ...] # Shape (None for dynamic dims)
    is_scalar: bool        # True if 0-dimensional
    alignment: int         # Memory alignment in bytes

class TypeInferrer:
    """
    Bidirectional type inference for FlowSIMD.

    Strategy:
    1. Forward pass: Propagate known types from inputs
    2. Backward pass: Infer inputs from output constraints
    3. Unification: Resolve type variables
    """

    # Type promotion rules (following NumPy)
    PROMOTION_TABLE = {
        (np.float32, np.float32): np.float32,
        (np.float32, np.float64): np.float64,
        (np.int32, np.float32): np.float32,
        (np.int32, np.int64): np.int64,
        # ...
    }

    def infer(self, ast: FlowAST,
              input_types: Dict[str, TypeInfo]) -> FlowAST:
        """Add type annotations to all nodes."""

        # Forward pass
        typed_ast = self._forward_pass(ast, input_types)

        # Check for type conflicts
        self._validate_types(typed_ast)

        return typed_ast

    def _infer_binop_type(self, op: FlowOp,
                          lhs: TypeInfo, rhs: TypeInfo) -> TypeInfo:
        """Infer type of binary operation."""

        # Scalar broadcast
        if lhs.is_scalar and not rhs.is_scalar:
            return TypeInfo(
                dtype=self._promote(lhs.dtype, rhs.dtype),
                shape=rhs.shape,
                is_scalar=False
            )

        # Element-wise
        if lhs.shape == rhs.shape:
            return TypeInfo(
                dtype=self._promote(lhs.dtype, rhs.dtype),
                shape=lhs.shape,
                is_scalar=False
            )

        raise ShapeMismatch(lhs.shape, rhs.shape)
```

### 4.2 Flow IR

#### 4.2.1 IR Node Types

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

class FlowOp(Enum):
    """All supported SIMD operations."""

    # Memory
    LOAD = auto()
    STORE = auto()
    GATHER = auto()
    SCATTER = auto()

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    ABS = auto()

    # FMA (Fused Multiply-Add)
    FMA = auto()      # a * b + c
    FMS = auto()      # a * b - c
    FNMA = auto()     # -(a * b) + c

    # Math
    SQRT = auto()
    RSQRT = auto()    # 1/sqrt(x)
    EXP = auto()
    LOG = auto()
    LOG2 = auto()
    POW = auto()
    SIN = auto()
    COS = auto()
    TAN = auto()

    # Rounding
    FLOOR = auto()
    CEIL = auto()
    ROUND = auto()
    TRUNC = auto()

    # Comparison (produce masks)
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Logical (on masks)
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    ANDNOT = auto()

    # Selection
    SELECT = auto()      # IfThenElse
    BLEND = auto()       # Masked blend
    COMPRESS = auto()    # Pack selected elements
    EXPAND = auto()      # Unpack elements

    # Reduction
    REDUCE_SUM = auto()
    REDUCE_PROD = auto()
    REDUCE_MIN = auto()
    REDUCE_MAX = auto()
    REDUCE_AND = auto()
    REDUCE_OR = auto()

    # Scan (prefix)
    SCAN_SUM = auto()
    SCAN_PROD = auto()
    SCAN_MIN = auto()
    SCAN_MAX = auto()

    # Shuffle/Permute
    REVERSE = auto()
    ROTATE = auto()
    SHUFFLE = auto()
    BROADCAST = auto()
    INTERLEAVE = auto()
    DEINTERLEAVE = auto()
    SLIDE_UP = auto()
    SLIDE_DOWN = auto()

    # Type conversion
    CONVERT = auto()
    BITCAST = auto()
    PROMOTE = auto()
    DEMOTE = auto()

    # Control
    LOOP = auto()
    BRANCH = auto()
    PHI = auto()

    # Special
    ZERO = auto()
    SET = auto()         # Broadcast scalar
    IOTA = auto()        # [0, 1, 2, ...]
    FIRST_N = auto()     # First N mask


@dataclass
class IRNode:
    """Single node in Flow IR (SSA form)."""

    id: int                        # Unique ID (%0, %1, ...)
    op: FlowOp                     # Operation type
    dtype: np.dtype                # Element type
    inputs: List['IRNode']         # Input nodes
    attrs: Dict[str, Any]          # Additional attributes

    # Metadata
    source_loc: Optional[SourceLoc] = None
    is_fused: bool = False         # Part of fused op

    def __hash__(self):
        return self.id


@dataclass
class FlowIR:
    """Complete IR for a kernel."""

    name: str
    nodes: List[IRNode]            # Topologically sorted
    inputs: List[IRNode]           # Input parameters
    outputs: List[IRNode]          # Return values

    # Type information
    input_types: List[TypeInfo]
    output_types: List[TypeInfo]

    # Profiling data (filled at runtime)
    profile: Optional['ProfileData'] = None

    def hash(self) -> str:
        """Content hash for caching."""
        content = [(n.op, n.dtype, n.attrs) for n in self.nodes]
        return hashlib.sha256(str(content).encode()).hexdigest()
```

#### 4.2.2 IR Builder

```python
class IRBuilder:
    """Builds Flow IR from analyzed AST."""

    def __init__(self):
        self.nodes: List[IRNode] = []
        self.node_id = 0
        self.symbol_table: Dict[str, IRNode] = {}

    def build(self, ast: FlowAST) -> FlowIR:
        """Convert typed AST to SSA IR."""

        # Create input nodes
        inputs = self._create_inputs(ast)

        # Build IR recursively
        output = self._build_node(ast)

        return FlowIR(
            name=ast.func_name,
            nodes=self.nodes,
            inputs=inputs,
            outputs=[output],
            input_types=ast.input_types,
            output_types=[output.dtype]
        )

    def _create_node(self, op: FlowOp, dtype: np.dtype,
                     inputs: List[IRNode], **attrs) -> IRNode:
        """Create a new IR node."""
        node = IRNode(
            id=self.node_id,
            op=op,
            dtype=dtype,
            inputs=inputs,
            attrs=attrs
        )
        self.node_id += 1
        self.nodes.append(node)
        return node

    def _build_node(self, ast: FlowAST) -> IRNode:
        """Recursively build IR nodes."""

        # Check for opportunities to create fused ops
        if self._can_fuse_fma(ast):
            return self._create_fma(ast)

        # Build children first (postorder)
        child_nodes = [self._build_node(c) for c in ast.args]

        # Create this node
        return self._create_node(
            op=ast.op,
            dtype=ast.dtype,
            inputs=child_nodes,
            **ast.attrs
        )

    def _can_fuse_fma(self, ast: FlowAST) -> bool:
        """Check if AST represents a * b + c pattern."""
        if ast.op != FlowOp.ADD:
            return False

        lhs, rhs = ast.args
        return (lhs.op == FlowOp.MUL or rhs.op == FlowOp.MUL)

    def _create_fma(self, ast: FlowAST) -> IRNode:
        """Create fused multiply-add node."""
        lhs, rhs = ast.args

        if lhs.op == FlowOp.MUL:
            # a * b + c
            a = self._build_node(lhs.args[0])
            b = self._build_node(lhs.args[1])
            c = self._build_node(rhs)
        else:
            # c + a * b
            a = self._build_node(rhs.args[0])
            b = self._build_node(rhs.args[1])
            c = self._build_node(lhs)

        return self._create_node(
            op=FlowOp.FMA,
            dtype=ast.dtype,
            inputs=[a, b, c],
            is_fused=True
        )
```

### 4.3 Optimization Passes

#### 4.3.1 Fusion Engine

```python
class FusionEngine:
    """
    Fuse operations to minimize memory traffic.

    Key insight: Memory bandwidth is the bottleneck for most SIMD code.
    Fusing operations keeps data in registers, avoiding store-load pairs.

    Fusable patterns:
    1. Element-wise chains: a + b * c - d
    2. FMA detection: a * b + c
    3. Comparison chains: (x > 0) & (x < 1)
    4. Unary chains: sqrt(abs(x))
    """

    # Operations that can be fused (element-wise)
    FUSABLE = {
        FlowOp.ADD, FlowOp.SUB, FlowOp.MUL, FlowOp.DIV,
        FlowOp.SQRT, FlowOp.EXP, FlowOp.LOG, FlowOp.ABS,
        FlowOp.SIN, FlowOp.COS, FlowOp.NEG,
        FlowOp.FLOOR, FlowOp.CEIL, FlowOp.ROUND,
        FlowOp.EQ, FlowOp.NE, FlowOp.LT, FlowOp.LE, FlowOp.GT, FlowOp.GE,
        FlowOp.AND, FlowOp.OR, FlowOp.XOR, FlowOp.NOT,
        FlowOp.SELECT, FlowOp.FMA, FlowOp.FMS,
    }

    # Operations that break fusion (need full vector results)
    BARRIERS = {
        FlowOp.REDUCE_SUM, FlowOp.REDUCE_MIN, FlowOp.REDUCE_MAX,
        FlowOp.SCAN_SUM, FlowOp.COMPRESS, FlowOp.EXPAND,
        FlowOp.GATHER, FlowOp.SCATTER,
        FlowOp.SHUFFLE, FlowOp.REVERSE,
    }

    def fuse(self, ir: FlowIR) -> FlowIR:
        """Apply fusion transformations."""

        # Find fusion groups
        groups = self._find_fusion_groups(ir)

        # Create fused operations
        fused_ir = self._apply_fusion(ir, groups)

        return fused_ir

    def _find_fusion_groups(self, ir: FlowIR) -> List[FusionGroup]:
        """Identify groups of operations that can be fused."""

        groups = []
        visited = set()

        for node in reversed(ir.nodes):
            if node.id in visited:
                continue

            if node.op in self.BARRIERS:
                # Barrier starts a new group
                groups.append(FusionGroup([node]))
                visited.add(node.id)
            else:
                # Try to grow a fusion group
                group = self._grow_group(node, ir, visited)
                groups.append(group)

        return groups

    def _grow_group(self, start: IRNode, ir: FlowIR,
                    visited: Set[int]) -> FusionGroup:
        """Grow a fusion group from a starting node."""

        group = [start]
        visited.add(start.id)

        # Follow data dependencies upward
        for inp in start.inputs:
            if inp.id not in visited and inp.op in self.FUSABLE:
                # Check register pressure
                if self._estimate_register_pressure(group + [inp]) <= 24:
                    group.append(inp)
                    visited.add(inp.id)

        return FusionGroup(group)

    def _estimate_register_pressure(self, nodes: List[IRNode]) -> int:
        """Estimate number of SIMD registers needed."""

        live = set()
        max_live = 0

        for node in nodes:
            # Inputs become live
            for inp in node.inputs:
                live.add(inp.id)

            # Update max
            max_live = max(max_live, len(live))

            # Output replaces one input
            if node.inputs:
                live.discard(node.inputs[0].id)
            live.add(node.id)

        return max_live
```

#### 4.3.2 Memory Optimization

```python
class MemoryOptimizer:
    """
    Optimize memory access patterns.

    Optimizations:
    1. Alignment: Ensure aligned loads/stores where possible
    2. Prefetching: Hide memory latency for streaming access
    3. Non-temporal stores: Bypass cache for write-only data
    4. Loop tiling: Improve cache utilization
    """

    def optimize(self, ir: FlowIR,
                 profile: Optional[ProfileData] = None) -> FlowIR:
        """Apply memory optimizations."""

        optimized = ir.clone()

        # 1. Alignment analysis
        optimized = self._optimize_alignment(optimized)

        # 2. Prefetch insertion (if profiling shows benefit)
        if profile and profile.memory_bound:
            optimized = self._insert_prefetch(optimized, profile)

        # 3. Streaming store detection
        optimized = self._detect_streaming_stores(optimized)

        # 4. Loop tiling for cache
        optimized = self._apply_tiling(optimized)

        return optimized

    def _optimize_alignment(self, ir: FlowIR) -> FlowIR:
        """
        Generate alignment-handling code.

        Pattern:
        1. Scalar prologue to reach alignment boundary
        2. Aligned vectorized loop
        3. Scalar epilogue for remainder
        """

        for node in ir.nodes:
            if node.op == FlowOp.LOAD:
                if not self._is_guaranteed_aligned(node):
                    # Mark for alignment handling
                    node.attrs['needs_alignment_check'] = True
                    node.attrs['alignment_target'] = 64  # AVX-512 alignment

        return ir

    def _detect_streaming_stores(self, ir: FlowIR) -> FlowIR:
        """
        Identify stores that should use non-temporal writes.

        Criteria:
        - Output is not read again in the kernel
        - Data is large enough to pollute cache
        - Access pattern is sequential
        """

        for node in ir.nodes:
            if node.op == FlowOp.STORE:
                ptr = node.attrs['ptr']

                # Check if this output is read later
                is_read_later = self._is_read_later(ptr, ir)

                if not is_read_later:
                    node.attrs['non_temporal'] = True

        return ir
```

### 4.4 Tiered Compilation System

#### 4.4.1 Tier 0: Interpreter with Profiling

```python
class Tier0Interpreter:
    """
    Bytecode interpreter with profiling instrumentation.

    Purpose:
    1. Execute kernels immediately (no compilation delay)
    2. Collect type/branch/loop profiling data
    3. Trigger promotion to higher tiers

    Based on:
    - PyPy's meta-tracing profiler
    - .NET's tiered compilation instrumentation
    """

    # Promotion thresholds
    TIER1_THRESHOLD = 100    # Calls before baseline JIT
    TIER2_THRESHOLD = 1000   # Calls before optimizing JIT

    def __init__(self, ir: FlowIR, runtime: 'FlowRuntime'):
        self.ir = ir
        self.runtime = runtime
        self.call_count = 0

        # Profiling data
        self.type_profile = TypeProfile()
        self.branch_profile = BranchProfile()
        self.loop_profile = LoopProfile()

    def execute(self, *args) -> np.ndarray:
        """Execute with profiling."""
        self.call_count += 1

        # Record type information
        for i, arg in enumerate(args):
            self.type_profile.record(i, type(arg),
                                     getattr(arg, 'dtype', None),
                                     getattr(arg, 'shape', None))

        # Check promotion thresholds
        if self.call_count == self.TIER1_THRESHOLD:
            self._schedule_tier1_compilation()
        elif self.call_count == self.TIER2_THRESHOLD:
            self._schedule_tier2_compilation()

        # Interpret (slow but collects profiling data)
        return self._interpret(*args)

    def _interpret(self, *args) -> np.ndarray:
        """Interpret IR nodes using NumPy as backend."""

        env = {}

        # Bind inputs
        for i, (node, arg) in enumerate(zip(self.ir.inputs, args)):
            env[node.id] = np.asarray(arg)

        # Execute nodes in order
        for node in self.ir.nodes:
            if node.op == FlowOp.ADD:
                env[node.id] = env[node.inputs[0].id] + env[node.inputs[1].id]
            elif node.op == FlowOp.MUL:
                env[node.id] = env[node.inputs[0].id] * env[node.inputs[1].id]
            elif node.op == FlowOp.SQRT:
                env[node.id] = np.sqrt(env[node.inputs[0].id])
            # ... handle all operations

        return env[self.ir.outputs[0].id]

    def _schedule_tier1_compilation(self):
        """Schedule background Tier 1 compilation."""
        self.runtime.compile_tier1_async(self.ir, self.type_profile)

    def _schedule_tier2_compilation(self):
        """Schedule background Tier 2 compilation."""
        self.runtime.compile_tier2_async(
            self.ir,
            self.type_profile,
            self.branch_profile,
            self.loop_profile
        )


class TypeProfile:
    """Collect type specialization data."""

    def __init__(self):
        self.type_counts: Dict[int, Counter] = defaultdict(Counter)
        self.dtype_counts: Dict[int, Counter] = defaultdict(Counter)
        self.shape_patterns: Dict[int, List[Tuple]] = defaultdict(list)

    def record(self, arg_idx: int, pytype, dtype, shape):
        """Record observed type information."""
        self.type_counts[arg_idx][pytype] += 1
        if dtype is not None:
            self.dtype_counts[arg_idx][dtype] += 1
        if shape is not None:
            self.shape_patterns[arg_idx].append(shape)

    def get_dominant_dtype(self, arg_idx: int) -> np.dtype:
        """Get most common dtype for specialization."""
        if not self.dtype_counts[arg_idx]:
            return np.float32  # Default
        return self.dtype_counts[arg_idx].most_common(1)[0][0]

    def is_monomorphic(self, arg_idx: int) -> bool:
        """Check if argument always has same type."""
        return len(self.dtype_counts[arg_idx]) == 1

    def get_alignment_hint(self, arg_idx: int) -> int:
        """Infer alignment from observed shapes."""
        shapes = self.shape_patterns.get(arg_idx, [])
        if not shapes:
            return 16  # Conservative

        # Check if all sizes are multiples of 64 bytes
        sizes = [np.prod(s) * 4 for s in shapes]  # Assume float32
        if all(s % 64 == 0 for s in sizes):
            return 64  # AVX-512 aligned
        if all(s % 32 == 0 for s in sizes):
            return 32  # AVX2 aligned
        return 16  # SSE aligned
```

#### 4.4.2 Tier 1: Copy-and-Patch JIT

```python
class Tier1CopyAndPatch:
    """
    Copy-and-patch JIT compiler for rapid baseline code generation.

    Based on:
    - CPython PEP 744 copy-and-patch implementation
    - OOPSLA 2021 paper "Copy-and-Patch Compilation"

    Performance characteristics:
    - Compilation: ~1-5ms (100x faster than LLVM -O0)
    - Code quality: ~15% better than LLVM -O0
    - Memory: Pre-compiled stencils ~500KB
    """

    def __init__(self):
        self.stencil_library = StencilLibrary.load()
        self.code_cache = CodeCache(max_size_mb=64)

    def compile(self, ir: FlowIR,
                type_profile: TypeProfile) -> CompiledKernel:
        """Generate native code via copy-and-patch."""

        # Get specialized dtype from profiling
        dtype = type_profile.get_dominant_dtype(0)

        # Select stencils for this dtype and CPU
        stencils = self._select_stencils(dtype)

        # Estimate code size
        code_size = sum(stencils[n.op].size for n in ir.nodes)
        code_size += 256  # Prologue/epilogue overhead

        # Allocate executable memory
        code_buffer = mmap.mmap(
            -1, code_size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
        )

        offset = 0

        # Emit prologue
        offset = self._emit_prologue(code_buffer, offset, ir)

        # Emit main loop
        offset = self._emit_loop_header(code_buffer, offset)

        for node in ir.nodes:
            # Get stencil for this operation
            stencil = stencils[node.op]

            # Copy stencil binary to code buffer
            code_buffer[offset:offset + stencil.size] = stencil.code

            # Patch holes with runtime values
            for hole in stencil.holes:
                patch_value = self._compute_patch(hole, node, offset, ir)
                self._apply_patch(code_buffer, offset + hole.offset,
                                 hole.size, patch_value)

            offset += stencil.size

        # Emit loop footer and epilogue
        offset = self._emit_loop_footer(code_buffer, offset)
        offset = self._emit_epilogue(code_buffer, offset)

        # Create callable kernel
        return CompiledKernel(
            code_buffer=code_buffer,
            code_size=offset,
            signature=ir.input_types,
            tier=1
        )

    def _select_stencils(self, dtype: np.dtype) -> Dict[FlowOp, Stencil]:
        """Select stencils matching dtype and detected CPU features."""

        cpu = detect_cpu_features()

        if cpu.has_avx512f:
            return self.stencil_library.get('avx512', dtype)
        elif cpu.has_avx2:
            return self.stencil_library.get('avx2', dtype)
        elif cpu.has_neon:
            return self.stencil_library.get('neon', dtype)
        else:
            return self.stencil_library.get('sse4', dtype)

    def _compute_patch(self, hole: Hole, node: IRNode,
                       offset: int, ir: FlowIR) -> int:
        """Compute value to patch into hole."""

        if hole.kind == HoleKind.IMMEDIATE:
            # Immediate value (scalar constant)
            return self._encode_float(node.attrs.get('value', 0.0))

        elif hole.kind == HoleKind.STACK_OFFSET:
            # Stack offset for variable
            var_idx = hole.variable_index
            return self._get_stack_offset(var_idx, ir)

        elif hole.kind == HoleKind.CALL_TARGET:
            # Relative call target
            target_offset = hole.target_offset
            return target_offset - (offset + hole.offset + hole.size)

        elif hole.kind == HoleKind.LOOP_COUNTER:
            # Loop iteration variable
            return self._get_loop_counter_offset()

        else:
            raise ValueError(f"Unknown hole kind: {hole.kind}")


@dataclass
class Stencil:
    """Pre-compiled binary code template."""

    code: bytes           # Binary code with holes
    size: int             # Total size in bytes
    holes: List[Hole]     # Locations to patch
    alignment: int        # Required alignment

    @classmethod
    def from_object(cls, obj_path: str, func_name: str) -> 'Stencil':
        """Extract stencil from compiled object file."""

        # Parse ELF/Mach-O object file
        with open(obj_path, 'rb') as f:
            obj = parse_object_file(f)

        # Find function
        func = obj.get_function(func_name)

        # Extract code and relocations
        code = func.code
        holes = []

        for reloc in func.relocations:
            holes.append(Hole(
                offset=reloc.offset,
                size=reloc.size,
                kind=cls._reloc_to_hole_kind(reloc.type)
            ))

        return cls(
            code=code,
            size=len(code),
            holes=holes,
            alignment=func.alignment
        )


class StencilLibrary:
    """
    Library of pre-compiled stencils.

    Generated at build time by compiling Highway C++ templates
    with Clang and extracting the binary code.
    """

    _instance = None

    @classmethod
    def load(cls) -> 'StencilLibrary':
        """Load stencil library (singleton)."""
        if cls._instance is None:
            cls._instance = cls._load_from_file()
        return cls._instance

    @classmethod
    def _load_from_file(cls) -> 'StencilLibrary':
        """Load pre-generated stencils."""

        # Find stencil data file
        stencil_path = pkg_resources.resource_filename(
            'bud_flow_lang', 'data/stencils.bin'
        )

        with open(stencil_path, 'rb') as f:
            data = pickle.load(f)

        library = cls()
        library.stencils = data['stencils']
        return library

    def get(self, target: str, dtype: np.dtype) -> Dict[FlowOp, Stencil]:
        """Get stencils for target ISA and dtype."""

        key = (target, str(dtype))
        return self.stencils.get(key, {})

    @classmethod
    def generate(cls, output_path: str):
        """
        Generate stencil library at build time.

        For each (target, dtype, operation) triple:
        1. Generate Highway C++ code
        2. Compile with Clang
        3. Extract binary code and relocations
        4. Save to stencil library
        """

        stencils = {}

        for target in ['sse4', 'avx2', 'avx512', 'neon', 'sve']:
            for dtype in [np.float32, np.float64, np.int32, np.int64]:
                for op in FlowOp:
                    # Generate Highway C++ for this operation
                    cpp_code = generate_stencil_cpp(op, dtype, target)

                    # Compile to object file
                    obj_path = compile_stencil(cpp_code, target)

                    # Extract stencil
                    stencil = Stencil.from_object(obj_path, f'stencil_{op.name}')

                    key = (target, str(dtype))
                    if key not in stencils:
                        stencils[key] = {}
                    stencils[key][op] = stencil

        # Save library
        with open(output_path, 'wb') as f:
            pickle.dump({'stencils': stencils}, f)
```

#### 4.4.3 Tier 2: Optimizing JIT with Highway

```python
class Tier2OptimizingJIT:
    """
    Optimizing JIT compiler with full Highway code generation.

    Based on:
    - .NET 8 Dynamic PGO (profile-guided optimization)
    - V8 TurboFan (speculative optimization + deoptimization)
    - Highway multi-target compilation

    Performance characteristics:
    - Compilation: ~50-200ms (async, doesn't block execution)
    - Code quality: ~90%+ of hand-tuned C++
    - Multi-target: Generates code for all enabled ISAs
    """

    def __init__(self):
        self.highway_codegen = HighwayCodeGenerator()
        self.compiler = HighwayCompiler()
        self.optimizer = PGOOptimizer()

    def compile(self, ir: FlowIR,
                type_profile: TypeProfile,
                branch_profile: BranchProfile,
                loop_profile: LoopProfile) -> CompiledKernel:
        """Generate highly optimized multi-target native code."""

        # Phase 1: Apply PGO optimizations
        optimized_ir = self.optimizer.apply_pgo(
            ir, type_profile, branch_profile, loop_profile
        )

        # Phase 2: Generate Highway C++ code
        cpp_code = self.highway_codegen.generate(optimized_ir, type_profile)

        # Phase 3: Compile for all supported targets
        binaries = self.compiler.compile_multi_target(cpp_code)

        # Phase 4: Create dispatch table
        return self._create_dispatched_kernel(binaries, ir)

    def _create_dispatched_kernel(self, binaries: Dict[str, bytes],
                                   ir: FlowIR) -> CompiledKernel:
        """Create kernel with runtime target dispatch."""

        # Load all compiled variants
        variants = {}
        for target, code in binaries.items():
            variants[target] = self._load_native(code)

        # Select best available target
        best_target = self._select_best_target(list(variants.keys()))

        return CompiledKernel(
            primary=variants[best_target],
            variants=variants,
            signature=ir.input_types,
            tier=2
        )


class PGOOptimizer:
    """
    Profile-Guided Optimization engine.

    Optimizations applied based on runtime profiles:
    1. Type specialization (monomorphic → remove type checks)
    2. Branch reordering (hot path first)
    3. Loop unrolling (based on iteration counts)
    4. Inlining (based on call frequency)
    """

    def apply_pgo(self, ir: FlowIR,
                  type_profile: TypeProfile,
                  branch_profile: BranchProfile,
                  loop_profile: LoopProfile) -> FlowIR:
        """Apply profile-guided optimizations."""

        optimized = ir.clone()

        # 1. Type specialization
        for i, inp in enumerate(optimized.inputs):
            if type_profile.is_monomorphic(i):
                dtype = type_profile.get_dominant_dtype(i)
                optimized = self._specialize_type(optimized, inp, dtype)

                # Add guard for deoptimization
                optimized = self._add_type_guard(optimized, inp, dtype)

        # 2. Branch reordering
        for node in optimized.nodes:
            if node.op == FlowOp.BRANCH:
                taken_prob = branch_profile.get_probability(node.id)
                if taken_prob > 0.9:
                    node.attrs['hot_path'] = 'taken'
                elif taken_prob < 0.1:
                    node.attrs['hot_path'] = 'not_taken'

        # 3. Loop unrolling
        for node in optimized.nodes:
            if node.op == FlowOp.LOOP:
                avg_iters = loop_profile.get_average(node.id)
                if avg_iters < 8:
                    node.attrs['unroll'] = 'full'
                elif avg_iters < 32:
                    node.attrs['unroll'] = 4
                else:
                    node.attrs['unroll'] = 8

        return optimized

    def _add_type_guard(self, ir: FlowIR,
                        inp: IRNode, expected_dtype: np.dtype) -> FlowIR:
        """
        Add type guard for speculative optimization.

        If guard fails at runtime, trigger deoptimization to Tier 1.
        """

        guard = IRNode(
            id=ir.next_id(),
            op=FlowOp.TYPE_GUARD,
            dtype=np.bool_,
            inputs=[inp],
            attrs={
                'expected_dtype': expected_dtype,
                'deopt_target': ir.tier1_entry_point
            }
        )

        # Insert guard at beginning
        ir.nodes.insert(0, guard)

        return ir


class HighwayCodeGenerator:
    """Generate Highway C++ code from optimized Flow IR."""

    # Operation mapping to Highway functions
    OP_MAP = {
        FlowOp.ADD: 'Add',
        FlowOp.SUB: 'Sub',
        FlowOp.MUL: 'Mul',
        FlowOp.DIV: 'Div',
        FlowOp.NEG: 'Neg',
        FlowOp.ABS: 'Abs',
        FlowOp.SQRT: 'Sqrt',
        FlowOp.FMA: 'MulAdd',
        FlowOp.FMS: 'MulSub',
        FlowOp.EXP: 'Exp',
        FlowOp.LOG: 'Log',
        FlowOp.SIN: 'Sin',
        FlowOp.COS: 'Cos',
        FlowOp.FLOOR: 'Floor',
        FlowOp.CEIL: 'Ceil',
        FlowOp.ROUND: 'Round',
        FlowOp.EQ: 'Eq',
        FlowOp.NE: 'Ne',
        FlowOp.LT: 'Lt',
        FlowOp.LE: 'Le',
        FlowOp.GT: 'Gt',
        FlowOp.GE: 'Ge',
        FlowOp.AND: 'And',
        FlowOp.OR: 'Or',
        FlowOp.XOR: 'Xor',
        FlowOp.NOT: 'Not',
        FlowOp.SELECT: 'IfThenElse',
        FlowOp.REDUCE_SUM: 'ReduceSum',
        FlowOp.REDUCE_MIN: 'MinOfLanes',
        FlowOp.REDUCE_MAX: 'MaxOfLanes',
        FlowOp.REVERSE: 'Reverse',
        FlowOp.COMPRESS: 'Compress',
    }

    def generate(self, ir: FlowIR,
                 type_profile: TypeProfile) -> str:
        """Generate complete Highway C++ source."""

        code = []

        # Header
        code.append(self._generate_header(ir))

        # Kernel function
        code.append(self._generate_kernel(ir, type_profile))

        # Export for dispatch
        code.append(self._generate_export(ir))

        # Footer
        code.append(self._generate_footer())

        return '\n'.join(code)

    def _generate_header(self, ir: FlowIR) -> str:
        return f'''
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "flowsimd_kernel_{ir.name}.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"

HWY_BEFORE_NAMESPACE();
namespace flowsimd {{
namespace HWY_NAMESPACE {{

namespace hn = hwy::HWY_NAMESPACE;
'''

    def _generate_kernel(self, ir: FlowIR,
                         type_profile: TypeProfile) -> str:
        """Generate the main kernel implementation."""

        dtype = type_profile.get_dominant_dtype(0)
        cpp_type = self._dtype_to_cpp(dtype)

        # Generate function signature
        inputs = ', '.join(
            f'const {cpp_type}* HWY_RESTRICT in{i}'
            for i in range(len(ir.inputs))
        )

        code = f'''
template<class D>
void {ir.name}_impl(D d, {inputs},
                    {cpp_type}* HWY_RESTRICT out,
                    size_t count) {{
    const size_t N = hn::Lanes(d);

    // Main vectorized loop
    size_t i = 0;
    for (; i + N <= count; i += N) {{
        {self._generate_loop_body(ir, 'i')}
    }}

    // Handle remainder with masking
    if (i < count) {{
        const auto mask = hn::FirstN(d, count - i);
        {self._generate_masked_body(ir, 'i')}
    }}
}}
'''
        return code

    def _generate_loop_body(self, ir: FlowIR, idx_var: str) -> str:
        """Generate vectorized loop body."""

        lines = []
        var_map = {}  # IRNode id -> C++ variable name

        for node in ir.nodes:
            if node.op == FlowOp.LOAD:
                var = f'v{node.id}'
                ptr = f'in{node.attrs["input_idx"]}'
                lines.append(f'const auto {var} = hn::Load(d, {ptr} + {idx_var});')
                var_map[node.id] = var

            elif node.op == FlowOp.STORE:
                inp_var = var_map[node.inputs[0].id]
                lines.append(f'hn::Store({inp_var}, d, out + {idx_var});')

            elif node.op in self.OP_MAP:
                var = f'v{node.id}'
                hwy_func = self.OP_MAP[node.op]

                if len(node.inputs) == 1:
                    inp = var_map[node.inputs[0].id]
                    lines.append(f'const auto {var} = hn::{hwy_func}({inp});')
                elif len(node.inputs) == 2:
                    lhs = var_map[node.inputs[0].id]
                    rhs = var_map[node.inputs[1].id]
                    lines.append(f'const auto {var} = hn::{hwy_func}({lhs}, {rhs});')
                elif len(node.inputs) == 3:
                    # FMA: MulAdd(a, b, c)
                    a = var_map[node.inputs[0].id]
                    b = var_map[node.inputs[1].id]
                    c = var_map[node.inputs[2].id]
                    lines.append(f'const auto {var} = hn::{hwy_func}({a}, {b}, {c});')

                var_map[node.id] = var

            elif node.op == FlowOp.SET:
                var = f'v{node.id}'
                value = node.attrs['value']
                lines.append(f'const auto {var} = hn::Set(d, {value}f);')
                var_map[node.id] = var

        return '\n        '.join(lines)

    def _generate_export(self, ir: FlowIR) -> str:
        """Generate Highway export for dynamic dispatch."""

        dtype = ir.input_types[0].dtype
        cpp_type = self._dtype_to_cpp(dtype)

        inputs = ', '.join(
            f'const {cpp_type}* in{i}'
            for i in range(len(ir.inputs))
        )

        args = ', '.join(
            f'in{i}' for i in range(len(ir.inputs))
        )

        return f'''
}}  // namespace HWY_NAMESPACE
}}  // namespace flowsimd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace flowsimd {{

HWY_EXPORT({ir.name}_impl);

void {ir.name}({inputs}, {cpp_type}* out, size_t count) {{
    return HWY_DYNAMIC_DISPATCH({ir.name}_impl)(
        hn::ScalableTag<{cpp_type}>(),
        {args}, out, count
    );
}}

}}  // namespace flowsimd
#endif
'''

    def _dtype_to_cpp(self, dtype: np.dtype) -> str:
        """Convert NumPy dtype to C++ type."""
        mapping = {
            np.float32: 'float',
            np.float64: 'double',
            np.int32: 'int32_t',
            np.int64: 'int64_t',
            np.uint32: 'uint32_t',
            np.uint64: 'uint64_t',
        }
        return mapping.get(dtype, 'float')


class HighwayCompiler:
    """Compile Highway C++ code to native binaries."""

    TARGET_FLAGS = {
        'sse4': ['-msse4.2'],
        'avx2': ['-mavx2', '-mfma', '-mbmi2'],
        'avx512': ['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl'],
        'avx512_vnni': ['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl',
                        '-mavx512vnni', '-mavx512vbmi', '-mavx512vbmi2'],
        'neon': ['-march=armv8-a+simd'],
        'sve': ['-march=armv8-a+sve'],
        'sve2': ['-march=armv9-a+sve2'],
    }

    def __init__(self):
        self.clang = self._find_clang()
        self.highway_include = self._find_highway()

    def compile_multi_target(self, cpp_code: str) -> Dict[str, bytes]:
        """Compile for all supported targets."""

        # Write source to temp file
        with tempfile.NamedTemporaryFile(suffix='.cc', delete=False) as f:
            f.write(cpp_code.encode())
            source_path = f.name

        binaries = {}

        try:
            # Detect which targets to compile
            targets = self._get_compile_targets()

            for target in targets:
                output_path = tempfile.mktemp(suffix='.so')

                cmd = [
                    self.clang,
                    '-shared', '-fPIC',
                    '-O3',
                    '-std=c++17',
                    f'-I{self.highway_include}',
                    *self.TARGET_FLAGS[target],
                    '-o', output_path,
                    source_path,
                    '-lhwy'
                ]

                result = subprocess.run(cmd, capture_output=True)

                if result.returncode == 0:
                    with open(output_path, 'rb') as f:
                        binaries[target] = f.read()
                    os.unlink(output_path)

        finally:
            os.unlink(source_path)

        return binaries

    def _get_compile_targets(self) -> List[str]:
        """Determine which targets to compile for."""

        cpu = detect_cpu_features()
        targets = []

        # Always include current CPU's best target
        if cpu.has_avx512vnni:
            targets.append('avx512_vnni')
        elif cpu.has_avx512f:
            targets.append('avx512')
        elif cpu.has_avx2:
            targets.append('avx2')
        elif cpu.has_sse42:
            targets.append('sse4')
        elif cpu.has_neon:
            if cpu.has_sve2:
                targets.append('sve2')
            elif cpu.has_sve:
                targets.append('sve')
            else:
                targets.append('neon')

        return targets
```

### 4.5 Python Binding Layer (nanobind)

```python
# src/flowsimd_bindings.cpp

"""
nanobind Python bindings for FlowSIMD.

Why nanobind over pybind11:
- 4x faster compile time
- 5x smaller binaries
- 10x lower runtime overhead
- Better NumPy ndarray integration
"""

BINDING_CODE = '''
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

// Forward declare Highway kernels
namespace flowsimd {
    void kernel_add(const float* a, const float* b, float* out, size_t n);
    void kernel_mul(const float* a, const float* b, float* out, size_t n);
    void kernel_sqrt(const float* in, float* out, size_t n);
    // ... more kernels
}

// NumPy array wrapper
template<typename T>
using ndarray = nb::ndarray<T, nb::ndim<1>, nb::c_contig>;

// Generic kernel wrapper
template<typename Func>
nb::ndarray<float> call_unary_kernel(
    Func func,
    ndarray<const float> input
) {
    size_t n = input.shape(0);

    // Allocate output
    float* out_data = new float[n];

    // Call Highway kernel
    func(input.data(), out_data, n);

    // Create output ndarray (takes ownership)
    nb::capsule deleter(out_data, [](void* p) noexcept {
        delete[] static_cast<float*>(p);
    });

    return nb::ndarray<float>(out_data, {n}, deleter);
}

NB_MODULE(_flowsimd_core, m) {
    m.doc() = "FlowSIMD native kernels";

    // CPU detection
    m.def("detect_cpu_features", &detect_cpu_features,
          "Detect available SIMD instruction sets");

    // Pre-compiled kernels
    m.def("kernel_sqrt", [](ndarray<const float> x) {
        return call_unary_kernel(flowsimd::kernel_sqrt, x);
    }, "x"_a, "Vectorized square root");

    m.def("kernel_add", [](ndarray<const float> a, ndarray<const float> b) {
        size_t n = a.shape(0);
        float* out = new float[n];
        flowsimd::kernel_add(a.data(), b.data(), out, n);
        nb::capsule del(out, [](void* p) { delete[] static_cast<float*>(p); });
        return nb::ndarray<float>(out, {n}, del);
    }, "a"_a, "b"_a, "Vectorized addition");

    // JIT kernel execution
    m.def("execute_jit_kernel", &execute_jit_kernel,
          "code"_a, "args"_a,
          "Execute JIT-compiled kernel");
}
'''
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-6)

**Goal:** Minimal viable DSL with interpreter backend.

#### Tasks:

1. **Core Infrastructure (Week 1-2)**
   - [ ] Project structure setup (Python package, CMake for native code)
   - [ ] Flow IR data structures (`FlowOp`, `IRNode`, `FlowIR`)
   - [ ] Type system (`TypeInfo`, `DType` enum)
   - [ ] Symbol table and scope management

2. **Python Frontend (Week 2-3)**
   - [ ] `@flow.kernel` decorator
   - [ ] AST extraction via `inspect` module
   - [ ] AST-to-IR conversion for basic operations (+, -, *, /, sqrt)
   - [ ] Type inference for NumPy arrays
   - [ ] Error messages with source locations

3. **Tier 0 Interpreter (Week 3-4)**
   - [ ] NumPy-based IR interpreter
   - [ ] Profiling infrastructure (call counts, type tracking)
   - [ ] Basic test suite

4. **Basic API (Week 4-5)**
   - [ ] `flow()` function for array wrapping
   - [ ] Arithmetic operators
   - [ ] Basic math functions (sqrt, exp, log)
   - [ ] Reduction operations (sum, min, max)

5. **Testing & Documentation (Week 5-6)**
   - [ ] Unit tests for all operations
   - [ ] Integration tests
   - [ ] Basic documentation
   - [ ] CI/CD setup

**Deliverable:** Working DSL that runs via NumPy interpreter, collecting profiling data.

```python
# End of Phase 1 capability:
@flow.kernel
def example(x):
    return (flow(x) * 2 + 1).sqrt()

# Works, but runs through NumPy (slow)
result = example(np.array([1, 2, 3, 4], dtype=np.float32))
```

---

### Phase 2: Baseline JIT (Weeks 7-12)

**Goal:** Copy-and-patch JIT for instant compilation.

#### Tasks:

1. **Stencil Generation (Week 7-8)**
   - [ ] Highway C++ templates for all basic operations
   - [ ] Build-time stencil compilation (CMake integration)
   - [ ] Stencil library format and loading
   - [ ] Stencil extraction from object files

2. **Copy-and-Patch Engine (Week 9-10)**
   - [ ] Executable memory allocation (mmap)
   - [ ] Stencil selection based on dtype and CPU
   - [ ] Hole patching for immediates and offsets
   - [ ] Loop prologue/epilogue generation
   - [ ] Calling convention handling

3. **Tier Promotion (Week 10-11)**
   - [ ] Call counting
   - [ ] Automatic promotion from Tier 0 to Tier 1
   - [ ] Code cache with LRU eviction
   - [ ] Thread-safe kernel replacement

4. **Testing & Benchmarking (Week 11-12)**
   - [ ] Correctness tests (compare to NumPy)
   - [ ] Performance benchmarks
   - [ ] Compilation time measurements
   - [ ] Memory usage profiling

**Deliverable:** Sub-5ms JIT compilation with 3-5x speedup over NumPy.

```python
# End of Phase 2 capability:
@flow.kernel
def example(x):
    return (flow(x) * 2 + 1).sqrt()

# First call: ~3ms (copy-and-patch)
# Subsequent calls: ~0.05ms (cached)
# Performance: 4x faster than NumPy
```

---

### Phase 3: Optimizing JIT (Weeks 13-20)

**Goal:** Full Highway code generation with PGO.

#### Tasks:

1. **Highway Code Generator (Week 13-15)**
   - [ ] IR-to-Highway C++ translation
   - [ ] Multi-target code generation (via HWY_FOREACH_TARGET)
   - [ ] Masked remainder handling
   - [ ] FMA detection and generation
   - [ ] Reduction code generation

2. **Optimization Passes (Week 15-17)**
   - [ ] Operation fusion engine
   - [ ] Common subexpression elimination
   - [ ] Dead code elimination
   - [ ] Loop unrolling (based on profile)
   - [ ] Memory access optimization

3. **Profile-Guided Optimization (Week 17-18)**
   - [ ] Type specialization from profiles
   - [ ] Branch probability annotation
   - [ ] Loop iteration hints
   - [ ] Speculative optimization with guards

4. **Compilation Infrastructure (Week 18-19)**
   - [ ] Clang-based Highway compilation
   - [ ] Async compilation (background thread)
   - [ ] Multi-target binary management
   - [ ] Runtime target dispatch

5. **Testing & Benchmarking (Week 19-20)**
   - [ ] Cross-platform validation (x86, ARM)
   - [ ] Performance comparison to hand-tuned C++
   - [ ] Compilation time vs optimization level trade-offs

**Deliverable:** ~90% of hand-tuned C++ performance with full Highway optimization.

```python
# End of Phase 3 capability:
@flow.kernel
def softmax(x):
    data = flow(x)
    shifted = data - data.max()
    exp_x = shifted.exp()
    return exp_x / exp_x.sum()

# First 100 calls: Tier 0 (NumPy) + profiling
# Calls 100-1000: Tier 1 (copy-and-patch), ~5x faster
# After 1000 calls: Tier 2 (Highway PGO), ~15x faster
```

---

### Phase 4: Advanced Features (Weeks 21-28)

**Goal:** Full DSL feature set for all user tiers.

#### Tasks:

1. **Beginner Tier API (Week 21-22)**
   - [ ] Pattern abstraction (comparisons → patterns)
   - [ ] .where(), .keep(), .drop() operations
   - [ ] Shape operations (.reverse(), .rotate())
   - [ ] User-friendly error messages

2. **Developer Tier API (Week 22-24)**
   - [ ] Performance hints (`hint.unroll()`, `hint.prefetch()`)
   - [ ] Explicit patterns (`Pattern.first()`, `Pattern.every()`)
   - [ ] Memory control (`load_aligned()`, `store_streaming()`)
   - [ ] Platform queries (`platform.simd_width()`)

3. **Expert Tier API (Week 24-26)**
   - [ ] Direct Highway bindings (`from flow.highway import ...`)
   - [ ] Explicit vector types (`Vec256`, `Mask128`)
   - [ ] Platform-specific intrinsics
   - [ ] Manual loop control

4. **Deoptimization System (Week 26-27)**
   - [ ] Type guards in generated code
   - [ ] Deoptimization entry points
   - [ ] Recompilation trigger on guard failure
   - [ ] Deoptimization loop prevention

5. **Documentation & Examples (Week 27-28)**
   - [ ] Complete API reference
   - [ ] Tutorial for each tier
   - [ ] Performance optimization guide
   - [ ] Example gallery

**Deliverable:** Complete three-tier DSL with progressive disclosure.

---

### Phase 5: Production Readiness (Weeks 29-36)

**Goal:** Stable, performant, well-documented release.

#### Tasks:

1. **Cross-Platform Support (Week 29-31)**
   - [ ] Windows support (MSVC compilation)
   - [ ] macOS support (Apple Silicon, x86)
   - [ ] ARM Linux (Raspberry Pi, AWS Graviton)
   - [ ] RISC-V support (experimental)

2. **Performance Optimization (Week 31-33)**
   - [ ] Roofline analysis integration
   - [ ] Auto-tuning for tile sizes
   - [ ] Memory prefetch optimization
   - [ ] Multi-threaded compilation

3. **Robustness (Week 33-35)**
   - [ ] Fuzzing for crash bugs
   - [ ] Memory leak detection
   - [ ] Thread safety audit
   - [ ] Error recovery mechanisms

4. **Release Engineering (Week 35-36)**
   - [ ] PyPI packaging
   - [ ] Conda packaging
   - [ ] Docker images
   - [ ] GitHub Actions CI/CD

**Deliverable:** Production-ready v1.0 release.

---

## 6. Technical Specifications

### 6.1 Supported Operations

| Category | Operations | Highway Mapping |
|----------|------------|-----------------|
| **Arithmetic** | +, -, *, /, //, %, ** | Add, Sub, Mul, Div, custom |
| **Math** | sqrt, exp, log, sin, cos, tan | Sqrt, Exp, Log, Sin, Cos |
| **Rounding** | floor, ceil, round, trunc | Floor, Ceil, Round, Trunc |
| **Comparison** | ==, !=, <, <=, >, >= | Eq, Ne, Lt, Le, Gt, Ge |
| **Logical** | &, \|, ^, ~ | And, Or, Xor, Not |
| **Selection** | where, keep, drop | IfThenElse, Compress |
| **Reduction** | sum, prod, min, max, mean | ReduceSum, MinOfLanes, MaxOfLanes |
| **Scan** | cumsum, cumprod | ScanSum (custom) |
| **Shape** | reverse, rotate, shuffle | Reverse, RotateRight, TableLookupLanes |

### 6.2 Supported Data Types

| FlowSIMD | NumPy | C++ | Highway Support |
|----------|-------|-----|-----------------|
| f32 | float32 | float | Full |
| f64 | float64 | double | Full |
| i8 | int8 | int8_t | Full |
| i16 | int16 | int16_t | Full |
| i32 | int32 | int32_t | Full |
| i64 | int64 | int64_t | Full |
| u8 | uint8 | uint8_t | Full |
| u16 | uint16 | uint16_t | Full |
| u32 | uint32 | uint32_t | Full |
| u64 | uint64 | uint64_t | Full |
| f16 | float16 | hwy::float16_t | Partial (HWY_HAVE_FLOAT16) |
| bf16 | bfloat16 | hwy::bfloat16_t | Partial (HWY_NATIVE_DOT_BF16) |

### 6.3 Target ISAs

| Target | Highway Name | Vector Width | Notes |
|--------|--------------|--------------|-------|
| SSE4.2 | HWY_SSE4 | 128-bit | Baseline x86-64 |
| AVX2+FMA | HWY_AVX2 | 256-bit | Most common x86 |
| AVX-512F | HWY_AVX3 | 512-bit | Server CPUs |
| AVX-512 VNNI | HWY_AVX3_DL | 512-bit | ML optimized |
| NEON | HWY_NEON | 128-bit | ARM baseline |
| SVE | HWY_SVE | 128-2048-bit | ARM scalable |
| SVE2 | HWY_SVE2 | 128-2048-bit | ARM v9 |
| RVV | HWY_RVV | 64-65536-bit | RISC-V scalable |

### 6.4 Performance Targets

| Benchmark | NumPy | FlowSIMD Target | Notes |
|-----------|-------|-----------------|-------|
| Vector add (1M f32) | 1.0x | 6-12x | Fused operations |
| FMA chain (1M f32) | 1.0x | 8-15x | Exploits FMA |
| Softmax (1M f32) | 1.0x | 10-18x | Fusion critical |
| Dot product (1M f32) | 1.0x | 5-10x | Reduction |
| 1D convolution (1M f32) | 1.0x | 8-20x | Neighbor access |

---

## 7. Risk Analysis & Mitigations

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Copy-and-patch complexity | Medium | High | Start with subset, incremental expansion |
| Highway API changes | Low | Medium | Pin to stable version, abstract interface |
| Cross-platform issues | Medium | Medium | CI on all platforms, fallback paths |
| Compilation time regression | Medium | Low | Continuous benchmarking, caching |
| Memory leaks in JIT | Medium | High | Extensive testing, sanitizers, fuzzing |

### 7.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Underestimated complexity | High | High | Buffer time in each phase |
| Dependency delays | Low | Medium | Early integration testing |
| Testing bottleneck | Medium | Medium | Automated test generation |

### 7.3 Adoption Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Learning curve too steep | Medium | High | Focus on Tier 1, extensive tutorials |
| NumPy compatibility issues | Medium | Medium | Comprehensive compatibility testing |
| Performance not compelling | Low | High | Continuous benchmarking, roofline analysis |

---

## 8. Appendices

### 8.1 References

**JIT Compilation:**
- [PEP 744 - JIT Compilation](https://peps.python.org/pep-0744/)
- [Copy-and-Patch OOPSLA 2021](https://dl.acm.org/doi/10.1145/3485513)
- [PyPy Meta-Tracing](https://aosabook.org/en/v2/pypy.html)
- [LuaJIT DynASM](https://luajit.org/dynasm.html)
- [MIR: A Lightweight JIT Compiler](https://developers.redhat.com/blog/2020/01/20/mir-a-lightweight-jit-compiler-project) - Red Hat
- [Cranelift](https://cranelift.dev/) - Fast, secure code generator
- [Cranelift vs LLVM Benchmarks](https://lwn.net/Articles/964735/) - LWN.net 2024

**Profile-Guided Optimization:**
- [.NET 8 Dynamic PGO](https://devblogs.microsoft.com/dotnet/bing-on-dotnet-8-the-impact-of-dynamic-pgo/)
- [V8 Speculative Optimization](https://v8.dev/blog/maglev)

**Lazy Evaluation & Fusion:**
- [Weld: A Common Runtime for High Performance Data Analytics](https://people.eecs.berkeley.edu/~matei/papers/2017/cidr_weld.pdf) - Stanford/Berkeley
- [Evaluating End-to-End Optimization in Weld](https://www.vldb.org/pvldb/vol11/p1002-palkar.pdf) - VLDB 2018
- [Operator Fusion in XLA](https://arxiv.org/pdf/2301.13062) - Analysis and Evaluation
- [TVM: End-to-End Optimizing Compiler](https://www.usenix.org/system/files/osdi18-chen.pdf) - OSDI 2018

**SIMD Libraries:**
- [Google Highway](https://github.com/google/highway)
- [MLIR Vector Dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
- [MLIR Introduction](https://mlir.llvm.org/) - Multi-Level Intermediate Representation

**Python Bindings:**
- [nanobind](https://nanobind.readthedocs.io/)
- [nanobind Benchmarks](https://nanobind.readthedocs.io/en/latest/benchmark.html)
- [cppyy](https://cppyy.readthedocs.io/) - Automatic Python-C++ bindings

**Related DSLs:**
- [Taichi Lang](https://www.taichi-lang.org/) - High-performance parallel programming in Python
- [Taichi Internal Designs](https://docs.taichi-lang.org/docs/internal)
- [Halide](https://halide-lang.org/) - Language for image and array processing

**Memory Management:**
- [Arena Allocators](https://www.rfleury.com/p/untangling-lifetimes-the-arena-allocator) - Ryan Fleury
- [Arena and Memory Pool Allocators](https://medium.com/@ramogh2404/arena-and-memory-pool-allocators-the-50-100x-performance-secret) - Performance analysis

### 8.2 Glossary

| Term | Definition |
|------|------------|
| **Copy-and-Patch** | JIT technique using pre-compiled binary templates |
| **Stencil** | Pre-compiled code template with holes for patching |
| **Tiered Compilation** | Multi-level JIT starting fast, optimizing hot code |
| **PGO** | Profile-Guided Optimization |
| **FMA** | Fused Multiply-Add (a*b+c in one instruction) |
| **OSR** | On-Stack Replacement (mid-execution code swap) |
| **Deoptimization** | Falling back from optimized to unoptimized code |
| **Guard** | Runtime check for speculative assumption |
| **Fusion** | Combining operations to reduce memory traffic |
| **Highway** | Google's portable SIMD C++ library |
| **ScalableTag** | Highway's type for variable-width vectors |
| **Lane** | Single element position in a SIMD register |
| **MIR** | Medium Internal Representation (lightweight JIT by Red Hat) |
| **Cranelift** | Rust-based JIT compiler (40% faster than LLVM compilation) |
| **MLIR** | Multi-Level Intermediate Representation (LLVM sub-project) |
| **Weld** | Stanford/Berkeley lazy evaluation runtime for data analytics |
| **Lazy Evaluation** | Deferring computation until result is needed |
| **Arena Allocator** | Fast memory allocator with bulk deallocation |
| **Taichi** | Python DSL for parallel programming with JIT compilation |
| **nanobind** | C++/Python bindings (3-10x faster than pybind11) |

### 8.3 File Structure

```
bud_flow_lang/
├── CMakeLists.txt              # Build configuration
├── setup.py                    # Python package setup
├── pyproject.toml              # Modern Python packaging
│
├── src/
│   ├── flowsimd/               # Python package
│   │   ├── __init__.py
│   │   ├── core.py             # Main API (@flow.kernel, flow())
│   │   ├── bunch.py            # Bunch abstraction
│   │   ├── pattern.py          # Pattern abstraction
│   │   ├── hints.py            # Developer hints
│   │   ├── highway.py          # Expert Highway bindings
│   │   │
│   │   ├── frontend/
│   │   │   ├── ast_analyzer.py
│   │   │   ├── type_inferrer.py
│   │   │   └── ir_builder.py
│   │   │
│   │   ├── ir/
│   │   │   ├── nodes.py
│   │   │   ├── types.py
│   │   │   └── printer.py
│   │   │
│   │   ├── optimizer/
│   │   │   ├── fusion.py
│   │   │   ├── memory.py
│   │   │   ├── pgo.py
│   │   │   └── passes.py
│   │   │
│   │   ├── jit/
│   │   │   ├── tier0_interpreter.py
│   │   │   ├── tier1_copy_patch.py
│   │   │   ├── tier2_highway.py
│   │   │   ├── stencil_library.py
│   │   │   └── code_cache.py
│   │   │
│   │   └── runtime/
│   │       ├── dispatcher.py
│   │       ├── profiler.py
│   │       └── deoptimizer.py
│   │
│   └── native/                 # C++ code
│       ├── bindings.cpp        # nanobind bindings
│       ├── cpu_detect.cpp      # CPU feature detection
│       ├── memory.cpp          # Memory management
│       │
│       ├── stencils/           # Stencil templates
│       │   ├── arithmetic.cc
│       │   ├── math.cc
│       │   ├── comparison.cc
│       │   └── reduction.cc
│       │
│       └── kernels/            # Pre-compiled kernels
│           ├── basic_ops.cc
│           └── reductions.cc
│
├── tests/
│   ├── test_frontend.py
│   ├── test_ir.py
│   ├── test_optimizer.py
│   ├── test_jit.py
│   ├── test_runtime.py
│   └── benchmarks/
│       ├── bench_vs_numpy.py
│       └── bench_compile_time.py
│
├── examples/
│   ├── beginner/
│   │   ├── hello_flow.py
│   │   ├── audio_normalize.py
│   │   └── softmax.py
│   ├── developer/
│   │   ├── optimized_dot.py
│   │   └── stencil_blur.py
│   └── expert/
│       ├── custom_highway.py
│       └── platform_specific.py
│
└── docs/
    ├── getting_started.md
    ├── api_reference.md
    ├── performance_guide.md
    └── architecture.md
```

---

### 8.4 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial comprehensive implementation plan |
| 2.0 | Dec 2024 | Critical revision with SOTA research validation |

#### v2.0 Key Changes:
1. Added JIT backend analysis (Copy-and-Patch vs MIR vs Cranelift vs LLVM)
2. Introduced Weld-style lazy evaluation architecture for 6-32x fusion speedups
3. Added arena allocator for SIMD-aligned memory management
4. Comprehensive edge case analysis (Python AST, numerical, memory alignment)
5. Deoptimization loop prevention strategy
6. Updated references with 15+ new SOTA sources
7. Added critical concerns and mitigations table

---

**Document Version:** 2.0 (Critically Revised)
**Last Updated:** December 2024
**Status:** Implementation Ready (Validated with SOTA Research)
