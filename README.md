# Bud Flow Lang

A high-performance SIMD array library with JIT compilation, designed to outperform NumPy and compete with JAX on CPU workloads.

[![Build Status](https://github.com/BudEcosystem/Bud-Flow-Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/BudEcosystem/Bud-Flow-Lang/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)](https://isocpp.org/)

## Performance Highlights

Benchmarked on Intel Core i7-10700KF @ 3.80GHz with AVX2 (256-bit SIMD):

| Operation | vs NumPy | vs JAX | Notes |
|-----------|----------|--------|-------|
| **SIN** | **2.93x faster** | **4.2x faster** | Highway SIMD transcendentals |
| **SUM** | **2.82x faster** | **3.4x faster** | Multi-accumulator reductions |
| **FMA** | **1.93x faster** | **2.3x faster** | Fused multiply-add |
| **EXP** | **1.55x faster** | **1.2x faster** | Vectorized exponential |
| **MUL** | **1.36x faster** | **1.5x faster** | SIMD multiplication |
| **ADD** | **1.29x faster** | **2.1x faster** | SIMD addition |

*Tested with arrays of 1K-1M elements. See [benchmarks/](benchmarks/) for full results.*

## Key Features

### Three-Tier API for All Skill Levels

```python
import bud_flow_lang_py as flow
flow.initialize()

# Beginner: NumPy-like simplicity
a = flow.Bunch.arange(1_000_000)
b = flow.Bunch.ones(1_000_000) * 2.0
result = a + b  # Automatic SIMD vectorization

# Developer: Enable optimizations
flow.set_tiling_enabled(True)      # Cache-aware tiling
flow.set_prefetch_enabled(True)    # Software prefetching
result = a.fma(b, c)               # Fused multiply-add

# Expert: Direct Highway SIMD access (coming soon)
# flow.highway.add(a_ptr, b_ptr, out_ptr, count)
```

### Sub-Millisecond JIT Compilation

Our **Copy-and-Patch JIT** compiles kernels in ~1ms (100x faster than LLVM):

```
Traditional:  Source -> IR -> Optimizer -> CodeGen -> Machine Code (100ms+)
Bud Flow:     Source -> IR -> Stencil Patch -> Machine Code (~1ms)
```

### Tiered Compilation System

Inspired by V8 and HotSpot JVMs:

| Tier | Threshold | Strategy | Use Case |
|------|-----------|----------|----------|
| 0 | Immediate | Highway Interpreter | Cold code |
| 1 | 10 calls | Copy-and-Patch JIT | Warm code |
| 2 | 100 calls | Fused Kernels | Hot code |

**Dynamic thresholds** adjust based on array size - large arrays JIT faster.

### JIT Optimizations

Five key optimizations implemented with TDD methodology:

1. **Multi-Accumulator Reductions** - 4 independent accumulators hide latency (2-4x speedup)
2. **Size-Specialized Kernels** - Optimized paths for small/medium/large arrays
3. **Kernel Fusion** - FMA, AXPY, and compound operations in single passes
4. **Software Prefetching** - Cache-line prefetch for large arrays
5. **Dynamic Tier Thresholds** - Array-size-aware JIT compilation triggers

### Portable SIMD via Google Highway

Write once, run optimally on any CPU:

| Architecture | SIMD Support |
|--------------|--------------|
| x86-64 | SSE4.2, AVX2, AVX-512, AVX-512BF16 |
| ARM64 | NEON, SVE, SVE2 |
| RISC-V | RVV (Vector Extension) |
| WebAssembly | SIMD128 |
| PowerPC | VSX, VSX3 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/BudEcosystem/Bud-Flow-Lang.git
cd Bud-Flow-Lang

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Build with Python bindings
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUD_BUILD_PYTHON=ON
make -j$(nproc)

# Test installation
python -c "import bud_flow_lang_py as flow; flow.initialize(); print('Success!')"
```

### Basic Usage

```python
import bud_flow_lang_py as flow

# Initialize runtime
flow.initialize()

# Create arrays (automatically SIMD-aligned)
a = flow.Bunch.arange(1_000_000)
b = flow.Bunch.ones(1_000_000) * 2.0
c = flow.Bunch.full(1_000_000, 3.0)

# Arithmetic operations
sum_ab = a + b
diff = a - b
prod = a * b
quot = a / b

# Fused multiply-add (single instruction)
fma_result = a.fma(b, c)  # a * b + c

# Reductions (multi-accumulator optimized)
total = a.sum()
average = a.mean()
minimum = a.min()
maximum = a.max()
dot_product = a.dot(b)

# Transcendental functions (SIMD vectorized)
sqrt_a = a.sqrt()
exp_a = a.exp()
log_a = a.log()
sin_a = a.sin()
cos_a = a.cos()
tanh_a = a.tanh()

# Get hardware info
info = flow.get_hardware_info()
print(f"SIMD Width: {info['simd_width'] * 8} bits")
print(f"AVX-512: {info['has_avx512']}")
print(f"Cores: {info['physical_cores']}")

# Cleanup
flow.shutdown()
```

### C++ API

```cpp
#include <bud_flow_lang/bud_flow_lang.h>

int main() {
    bud::RuntimeConfig config;
    config.enable_debug_output = false;
    bud::initialize(config);

    // Create arrays
    auto a = bud::Bunch::arange(1000000).value();
    auto b = bud::Bunch::fill(1000000, 2.0f).value();

    // Operations
    auto sum = a + b;
    float dot = a.dot(b);
    float total = a.sum();

    // Get JIT statistics
    auto stats = bud::jit::getJitStats();
    std::cout << "Compilations: " << stats.total_compilations << "\n";
    std::cout << "Compile time: " << stats.total_compile_time_us << " us\n";

    bud::shutdown();
    return 0;
}
```

## Architecture

```
bud_flow_lang/
├── include/bud_flow_lang/       # Public C++ headers
│   ├── bud_flow_lang.h          # Main entry point
│   ├── bunch.h                  # Array type (Bunch)
│   ├── ir.h                     # Intermediate representation
│   ├── codegen/
│   │   └── hwy_ops.h            # Highway SIMD operations
│   ├── jit/
│   │   ├── copy_patch_compiler.h
│   │   └── stencil.h            # JIT stencil definitions
│   └── memory/
│       ├── cache_config.h       # Cache detection
│       └── prefetch.h           # Prefetch hints
├── src/
│   ├── codegen/                 # Highway SIMD implementation
│   │   ├── hwy_ops.cc           # Operation wrappers
│   │   └── hwy_ops-inl.h        # SIMD kernels (multi-target)
│   ├── ir/                      # IR builder & optimizer
│   ├── jit/                     # Copy-and-patch JIT compiler
│   ├── runtime/                 # Tiered executor, Bunch
│   └── python/                  # Python bindings (nanobind)
├── tests/                       # 590+ unit tests
├── benchmarks/                  # Performance benchmarks
└── docs/                        # Documentation
```

## JIT Compiler Details

### Copy-and-Patch Compilation

Pre-compiled stencils with runtime patching:

```asm
; Stencil template (compiled once)
vmovups  ymm0, [HOLE_A]      ; Load from address A
vmovups  ymm1, [HOLE_B]      ; Load from address B
vaddps   ymm2, ymm0, ymm1    ; SIMD add
vmovups  [HOLE_OUT], ymm2    ; Store result

; Runtime: patch holes with actual addresses (~100ns)
```

### Multi-Accumulator Reductions

```cpp
// Standard reduction (latency-bound)
for (i = 0; i < n; i++)
    sum += a[i];  // 4-cycle dependency chain

// Multi-accumulator (4x throughput)
for (i = 0; i < n; i += 4*lanes) {
    sum0 += a[i];
    sum1 += a[i + lanes];
    sum2 += a[i + 2*lanes];
    sum3 += a[i + 3*lanes];  // Independent, pipelined
}
result = sum0 + sum1 + sum2 + sum3;
```

### Size-Specialized Kernels

| Array Size | Kernel Type | Optimization |
|------------|-------------|--------------|
| < 256 | Small | Fully unrolled, no loop overhead |
| 256-4K | Medium | 4x unrolled with SIMD |
| > 4K | Large | 8x unrolled + prefetching |

## Memory Optimization

### Cache-Aware Execution

```python
# Detect cache configuration
cache = flow.detect_cache_config()
print(f"L1: {cache['l1_size_kb']} KB")
print(f"L2: {cache['l2_size_kb']} KB")
print(f"L3: {cache['l3_size_kb']} KB")

# Enable cache optimizations
flow.set_tiling_enabled(True)
flow.set_prefetch_enabled(True)
```

### Dynamic Tier Thresholds

JIT compilation thresholds adapt to array size:

```
Small arrays (< 256):   50 calls to JIT (amortize compile cost)
Medium arrays (256-4K): 10 calls to JIT (standard)
Large arrays (> 4K):    3 calls to JIT (immediate benefit)
```

## Running Benchmarks

```bash
# Quick benchmark
cd bud_flow_lang
source .venv/bin/activate
python benchmarks/python/quick_bench.py

# Full benchmark suite
python benchmarks/python/run_benchmarks.py --sizes 1000,100000,1000000 --runs 20
```

## Running Tests

```bash
cd build

# Run all tests
./bud_tests

# Run specific test suite
./bud_tests --gtest_filter="DynamicThreshold*"
./bud_tests --gtest_filter="OptimizedReductions*"

# Run with verbose output
./bud_tests --gtest_filter="*" --gtest_print_time=1
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release) |
| `BUD_BUILD_PYTHON` | OFF | Build Python bindings |
| `BUD_ENABLE_TESTS` | ON | Build test suite |
| `BUD_ENABLE_BENCHMARKS` | ON | Build benchmarks |
| `BUD_ENABLE_SANITIZERS` | OFF | Enable ASan/UBSan |

## Requirements

- **C++ Compiler**: GCC 10+, Clang 12+, or MSVC 2019+
- **CMake**: 3.16+
- **Python**: 3.8+ (for bindings)
- **Dependencies**: Automatically fetched via CMake
  - Google Highway (SIMD)
  - nanobind (Python bindings)
  - Google Test (testing)
  - spdlog (logging)

## Documentation

- [Getting Started](docs/getting_started.md) - 5-minute introduction
- [Installation Guide](docs/installation.md) - Detailed setup
- [Python API Reference](docs/python_api.md) - Python bindings
- [Performance Guide](docs/performance_guide.md) - Optimization tips
- [Tutorials](docs/tutorials/) - Step-by-step guides

## Roadmap

- [x] Copy-and-Patch JIT compiler
- [x] Tiered compilation system
- [x] Multi-accumulator reductions
- [x] Size-specialized kernels
- [x] Kernel fusion (FMA, AXPY)
- [x] Software prefetching
- [x] Dynamic tier thresholds
- [ ] Multi-threaded execution
- [ ] GPU backend (CUDA/Metal)
- [ ] Automatic differentiation
- [ ] Graph-level optimization

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`./bud_tests`)
2. Code follows the style guide (clang-format)
3. New features have tests

## License

Copyright 2024-2025 Bud Ecosystem Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Author

**Jithin VG** - Bud Ecosystem Inc.

## Links

- [GitHub Repository](https://github.com/BudEcosystem/Bud-Flow-Lang)
- [Issue Tracker](https://github.com/BudEcosystem/Bud-Flow-Lang/issues)
- [Bud Ecosystem](https://budecosystem.com)
