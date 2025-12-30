# Bud Flow Lang

<p align="center">
  <strong>A Domain-Specific Language for Portable High-Performance SIMD Programming</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="getting_started.md">Getting Started</a> |
  <a href="tutorials/">Tutorials</a> |
  <a href="api/">API Reference</a> |
  <a href="notebooks/">Notebooks</a>
</p>

---

## What is Bud Flow Lang?

**Bud Flow Lang** is a Domain-Specific Language (DSL) embedded within Python for high-performance SIMD programming. It's not just an array library - it's a complete compiler infrastructure with:

- **SSA-Form Intermediate Representation** for optimization
- **Multi-Tier JIT Compiler** with sub-millisecond compilation
- **Automatic Operation Fusion** achieving 6-30x speedups
- **Portable SIMD Backend** supporting 10+ CPU architectures
- **Cache-Aware Memory System** with automatic tiling and prefetching

**Think of it as NumPy's simplicity meets compiler-level optimization.**

---

## The JIT Compiler

### Copy-and-Patch Architecture

Our JIT compiler achieves ~1ms compilation time (100x faster than LLVM) using pre-compiled code stencils:

```
Traditional JIT:  Source → IR → Optimization → Code Gen → Machine Code (100ms+)
Bud Flow Lang:    Source → IR → Stencil Patching → Machine Code (~1ms)
```

### Tiered Compilation

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

---

## Performance Highlights

| Operation | vs NumPy | Notes |
|-----------|----------|-------|
| **FMA (a*b+c)** | **1.4x faster** | True hardware fused multiply-add |
| Element-wise Add | ~1.0x | Competitive on large arrays |
| Chained Ops | 1.2-2x | Fusion eliminates temporaries |
| Dot Product | ~0.3x | NumPy uses multi-threaded BLAS |

*Bud Flow Lang excels at fused operations and custom kernels where it can optimize the entire computation.*

---

## Quick Example

```python
import bud_flow_lang_py as flow

# Initialize the DSL runtime
flow.initialize()

# Create SIMD-optimized arrays
a = flow.ones(1_000_000)
b = flow.full(1_000_000, 2.0)
c = flow.full(1_000_000, 3.0)

# Fused Multiply-Add: a*b + c (uses hardware FMA)
result = flow.fma(a, b, c)  # 1.4x faster than NumPy!

# JIT-compiled reductions with multiple accumulators
total = result.sum()
dot_product = flow.dot(a, b)

# Check your hardware capabilities
info = flow.get_hardware_info()
print(f"SIMD Width: {info['simd_width']*8} bits")
print(f"AVX2: {info['has_avx2']}, AVX-512: {info['has_avx512']}")

# Cache-aware memory optimization
cache = flow.detect_cache_config()
print(f"L1: {cache['l1_size_kb']} KB, L2: {cache['l2_size_kb']} KB")
```

---

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
python -c "import bud_flow_lang_py as flow; flow.initialize(); print('Ready!')"
```

### Google Colab

```python
# Run this cell in Colab to install (~2-3 minutes)
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

See the [Installation Guide](installation.md) for detailed instructions.

---

## Documentation

### Getting Started
- [Quick Start Guide](getting_started.md) - Get up and running in 5 minutes
- [Installation](installation.md) - Detailed installation instructions
- [Your First Program](tutorials/01_first_program.md) - Write your first Flow program

### Tutorials
- [Basic Operations](tutorials/02_basic_operations.md) - Arrays, arithmetic, reductions
- [Memory Optimization](tutorials/03_memory_optimization.md) - Tiling and cache efficiency
- [NumPy Interoperability](tutorials/04_numpy_interop.md) - Working with NumPy
- [Performance Tuning](tutorials/05_performance.md) - Getting maximum performance

### API Reference
- [Core Functions](api/core.md) - `initialize()`, `shutdown()`, hardware info
- [Array Creation](api/creation.md) - `ones()`, `zeros()`, `arange()`, etc.
- [Operations](api/operations.md) - Arithmetic, reductions, dot product

### Jupyter Notebooks
- [Introduction to Flow](notebooks/01_introduction.ipynb) - Interactive tutorial
- [NumPy Comparison](notebooks/02_numpy_comparison.ipynb) - Benchmarks and comparison

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python DSL API                          │
│  flow.ones() | flow.dot() | flow.fma() | flow.kernel()     │
├─────────────────────────────────────────────────────────────┤
│              SSA-Form Intermediate Representation            │
│  Constant Folding | Dead Code Elimination | Fusion          │
├─────────────────────────────────────────────────────────────┤
│                Multi-Tier JIT Compiler                       │
│  Tier 0: Profiling → Tier 1: Copy-Patch → Tier 2: Optimized │
├─────────────────────────────────────────────────────────────┤
│                Memory Optimization Layer                     │
│  Cache-Aware Tiling | Software Prefetch | SIMD Alignment    │
├─────────────────────────────────────────────────────────────┤
│              Google Highway Portable SIMD                    │
│  AVX2 | AVX-512 | NEON | SVE | RVV | VSX | WASM-SIMD       │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Support

| Architecture | SIMD Extensions | Status |
|--------------|-----------------|--------|
| x86-64 | SSE4, AVX2, AVX-512, AVX-512BF16 | Full Support |
| ARM64 | NEON, SVE, SVE2 | Full Support |
| RISC-V | RVV (Vector Extension) | Full Support |
| PowerPC | VSX, VSX3 | Full Support |
| WebAssembly | SIMD128 | Full Support |
| S390x | z/Architecture Vector | Full Support |

---

## Comparison with Other Libraries

| Feature | Bud Flow Lang | NumPy | JAX | PyTorch |
|---------|---------------|-------|-----|---------|
| DSL with JIT | **Yes** | No | XLA | TorchScript |
| Sub-ms compilation | **Yes (~1ms)** | N/A | No (100ms+) | No |
| Operation fusion | **Yes (Weld-style)** | No | Yes | Partial |
| Cache-aware tiling | **Yes** | No | Via XLA | No |
| Hardware FMA | **Yes** | Partial | Yes | Yes |
| Portable SIMD | **Highway (10+ archs)** | Platform-specific | Platform-specific | Platform-specific |
| Zero runtime deps | **Yes** | No | No | No |

---

## Pre-Compiled Fused Kernels

Common patterns are pre-optimized for maximum performance:

| Kernel | Description | Speedup |
|--------|-------------|---------|
| `FusedDotProduct` | Dot product with 4 accumulators | 2-4x |
| `FusedSoftmax` | Numerically stable softmax | 3-5x |
| `FusedLayerNorm` | Layer normalization | 4-6x |
| `FusedGelu` | GELU activation | 2-3x |
| `FusedSquaredDistance` | L2 distance | 2-3x |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## License

Bud Flow Lang is licensed under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.

Copyright 2024-2025 Bud Ecosystem Inc.

---

## Links

- [GitHub Repository](https://github.com/BudEcosystem/Bud-Flow-Lang)
- [Issue Tracker](https://github.com/BudEcosystem/Bud-Flow-Lang/issues)
- [Google Highway](https://github.com/google/highway) - The SIMD library we use
- [Bud Ecosystem](https://budecosystem.com)
