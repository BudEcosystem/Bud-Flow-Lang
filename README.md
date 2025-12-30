# Bud Flow Lang

A Domain-Specific Language (DSL) for portable high-performance SIMD programming, embedded within Python. Write simple array code, get automatic vectorization, JIT compilation, and near-native performance.

[![Build Status](https://github.com/BudEcosystem/Bud-Flow-Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/BudEcosystem/Bud-Flow-Lang/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## What is Bud Flow Lang?

Bud Flow Lang is **not just another array library** - it's a complete domain-specific language with its own:

- **Intermediate Representation (IR)** in SSA form for optimization
- **Multi-tier JIT Compiler** with sub-millisecond compilation
- **Automatic Operation Fusion** achieving 6-30x speedups
- **Portable SIMD Backend** supporting 10+ CPU architectures
- **Cache-Aware Memory System** with automatic tiling and prefetching

Think of it as **NumPy's simplicity meets compiler-level optimization**.

## Key Features

### Instant JIT Compilation

Our **Copy-and-Patch JIT compiler** achieves ~1ms compilation time (100x faster than LLVM):

```
Traditional JIT:  Source → IR → Optimization → Code Gen → Machine Code (100ms+)
Bud Flow Lang:    Source → IR → Stencil Patching → Machine Code (~1ms)
```

### Tiered Compilation System

Like modern JavaScript engines, we use progressive optimization:

| Tier | Name | Threshold | Description |
|------|------|-----------|-------------|
| 0 | Interpreter | - | Collects runtime profiles |
| 1 | Baseline JIT | 10 calls | Fast copy-and-patch compilation |
| 2 | Optimizing JIT | 100 calls | Profile-guided optimization |

### Automatic Operation Fusion

The DSL automatically fuses operations to minimize memory traffic:

```python
# What you write:
result = (a * b + c) * d

# What executes (single fused kernel):
# - One pass over memory instead of three
# - 3x fewer memory operations
# - Automatic FMA instruction usage
```

### Portable SIMD Across Architectures

Write once, run optimally everywhere:

| Architecture | SIMD Support |
|--------------|--------------|
| x86-64 | SSE4, AVX2, AVX-512, AVX-512BF16 |
| ARM64 | NEON, SVE, SVE2 |
| RISC-V | RVV (Vector Extension) |
| PowerPC | VSX, VSX3 |
| WebAssembly | SIMD128 |
| S390x | z/Architecture Vector |

## Quick Start

### Python DSL

```python
import bud_flow_lang_py as flow

# Initialize the DSL runtime
flow.initialize()

# Create arrays (SIMD-aligned automatically)
a = flow.ones(1_000_000)
b = flow.full(1_000_000, 2.0)
c = flow.full(1_000_000, 3.0)

# Operations are JIT-compiled and SIMD-vectorized
result = flow.fma(a, b, c)  # Fused multiply-add: a*b + c

# Reductions use multiple accumulators for ILP
total = result.sum()
dot_product = flow.dot(a, b)

# Check your hardware capabilities
info = flow.get_hardware_info()
print(f"SIMD Width: {info['simd_width']*8} bits")
print(f"AVX-512: {info['has_avx512']}")
```

### C++ API

```cpp
#include <bud_flow_lang/bud_flow_lang.h>

int main() {
    bud::initialize();

    // Create arrays with automatic SIMD alignment
    auto a = bud::Bunch::fill(1000000, 2.0f).value();
    auto b = bud::Bunch::fill(1000000, 3.0f).value();

    // JIT-compiled dot product
    float result = a.dot(b);

    // Get compilation statistics
    auto stats = bud::getCompilationStats();
    std::cout << "Compilations: " << stats.total_compilations << std::endl;
    std::cout << "Compile time: " << stats.total_compile_time_us << " us" << std::endl;

    bud::shutdown();
    return 0;
}
```

## JIT Compiler Architecture

### Copy-and-Patch Compilation

Instead of generating code from scratch, we use pre-compiled **stencils** (code templates) with holes that get patched at runtime:

```
Stencil Template:
  vmovups  ymm0, [HOLE_ADDR_A]    ; Load from address A
  vmovups  ymm1, [HOLE_ADDR_B]    ; Load from address B
  vaddps   ymm2, ymm0, ymm1       ; SIMD add
  vmovups  [HOLE_ADDR_OUT], ymm2  ; Store result

Runtime Patching:
  - HOLE_ADDR_A   → actual pointer to array a
  - HOLE_ADDR_B   → actual pointer to array b
  - HOLE_ADDR_OUT → actual pointer to output
```

**Benefits:**
- Sub-millisecond compilation (vs 100ms+ for LLVM -O0)
- W^X security model (allocate RW, then make RX)
- No runtime code generation dependencies

### Pre-Compiled Fused Kernels

Common patterns are pre-optimized:

| Kernel | Description | Speedup |
|--------|-------------|---------|
| `FusedDotProduct` | Dot product with 4 accumulators | 2-4x |
| `FusedSoftmax` | Numerically stable softmax | 3-5x |
| `FusedLayerNorm` | Layer normalization | 4-6x |
| `FusedGelu` | GELU activation | 2-3x |
| `FusedSquaredDistance` | L2 distance | 2-3x |

### Intermediate Representation

The DSL compiles to an SSA-form IR that enables:

- **Constant Folding**: Evaluate compile-time expressions
- **Dead Code Elimination**: Remove unused computations
- **Operation Fusion**: Merge compatible operations
- **Peephole Optimization**: Pattern-based micro-optimizations

## Memory Optimization

### Cache-Aware Execution

Automatic detection and optimization for your CPU's cache hierarchy:

```python
cache = flow.detect_cache_config()
# L1: 32 KB, L2: 256 KB, L3: 16 MB, Line: 64 bytes

# Large arrays automatically use tiled execution
# to maximize cache utilization
```

### Software Prefetching

Intelligent prefetch insertion reduces memory stalls:

```python
# Control prefetching behavior
flow.set_prefetch_enabled(True)
flow.set_tiling_enabled(True)

# Get optimal tile size for your operation
tile_size = flow.optimal_tile_size(
    element_size=4,    # float32
    num_arrays=3       # a, b, c in FMA
)
```

## Performance

### Benchmarks vs NumPy (1M elements)

| Operation | Bud Flow Lang | NumPy | Winner |
|-----------|---------------|-------|--------|
| FMA (a*b+c) | 0.48 ms | 0.67 ms | **Flow 1.4x** |
| Element-wise Add | 0.35 ms | 0.33 ms | Tie |
| Dot Product | 0.51 ms | 0.17 ms | NumPy* |
| Sum Reduction | 2.15 ms | 0.18 ms | NumPy* |

*NumPy uses multi-threaded BLAS; Bud Flow Lang is currently single-threaded.

### When to Use Bud Flow Lang

**Use Bud Flow Lang for:**
- Fused operations (FMA, chained arithmetic)
- Custom SIMD kernels
- Portable performance across architectures
- JIT compilation without LLVM dependency

**Use NumPy for:**
- Multi-threaded BLAS operations
- Rich ecosystem integration
- Established workflows

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/BudEcosystem/Bud-Flow-Lang.git
cd Bud-Flow-Lang

# Build with Python bindings
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUD_BUILD_PYTHON=ON
make -j$(nproc)

# Add to Python path
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "import bud_flow_lang_py as flow; flow.initialize(); print('Success!')"
```

### Google Colab

```python
# Install in Colab (takes ~2-3 minutes)
!apt-get update -qq && apt-get install -qq -y cmake g++
!git clone --depth 1 https://github.com/BudEcosystem/Bud-Flow-Lang.git
!cd Bud-Flow-Lang && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUD_BUILD_PYTHON=ON && \
    make -j4

import sys
sys.path.insert(0, '/content/Bud-Flow-Lang/build')
import bud_flow_lang_py as flow
flow.initialize()
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUD_BUILD_PYTHON` | OFF | Build Python bindings |
| `BUD_ENABLE_TESTS` | ON | Build test suite |
| `BUD_ENABLE_BENCHMARKS` | ON | Build benchmarks |
| `CMAKE_BUILD_TYPE` | Release | Build type |

## Project Structure

```
bud_flow_lang/
├── include/bud_flow_lang/     # Public C++ API
│   ├── ir.h                   # SSA-form IR
│   ├── bunch.h                # Array abstraction
│   ├── jit/                   # JIT compiler headers
│   │   ├── copy_patch_compiler.h
│   │   └── stencil.h
│   └── memory/                # Memory optimization
│       ├── cache_config.h
│       ├── prefetch.h
│       └── tiled_executor.h
├── src/
│   ├── ir/                    # IR builder & optimizer
│   ├── jit/                   # Copy-and-patch JIT
│   ├── codegen/               # Highway SIMD codegen
│   ├── runtime/               # Tiered executor
│   └── python/                # Python bindings
├── docs/                      # Documentation
│   ├── tutorials/             # Step-by-step guides
│   ├── api/                   # API reference
│   └── notebooks/             # Jupyter notebooks
└── examples/                  # Example programs
```

## Documentation

- **[Getting Started](docs/getting_started.md)** - 5-minute introduction
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Tutorials](docs/tutorials/)** - Step-by-step learning path
- **[API Reference](docs/api/)** - Complete function documentation
- **[Jupyter Notebooks](docs/notebooks/)** - Interactive examples for Colab

## Development

### Quality Gate

```bash
# Run full quality checks (required before commits)
./scripts/quality-gate.sh

# With auto-formatting
./scripts/quality-gate.sh --fix

# Quick mode (skip sanitizer builds)
./scripts/quality-gate.sh --quick
```

### Pre-commit Hooks

Automatically enforced on every commit:
- Code formatting (clang-format)
- Build verification
- Test suite execution
- Static analysis (clang-tidy)

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(jit): add AVX-512 stencils for float16 operations
fix(memory): resolve alignment issue in tiled executor
docs: add JIT compiler architecture section to README
perf(fusion): optimize dot product kernel with 8 accumulators
```

## Roadmap

- [ ] Multi-threaded parallel execution
- [ ] GPU backend (CUDA, Metal, Vulkan)
- [ ] Automatic differentiation
- [ ] Distributed execution
- [ ] Additional fusion patterns

## License

Copyright 2024-2025 Bud Ecosystem Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Author

**Jithin VG** - Bud Ecosystem Inc.

## Contributing

Contributions are welcome! Please ensure all quality checks pass before submitting a pull request.

## Links

- [Documentation](docs/index.md)
- [GitHub Repository](https://github.com/BudEcosystem/Bud-Flow-Lang)
- [Issue Tracker](https://github.com/BudEcosystem/Bud-Flow-Lang/issues)
- [Bud Ecosystem](https://budecosystem.com)
