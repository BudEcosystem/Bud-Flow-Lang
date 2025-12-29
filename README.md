# Bud Flow Lang

A high-performance Python DSL for SIMD programming with JIT compilation.

[![Build Status](https://github.com/BudEcosystem/Bud-Flow-Lang/actions/workflows/ci.yml/badge.svg)](https://github.com/BudEcosystem/Bud-Flow-Lang/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

Bud Flow Lang is a domain-specific language embedded in Python that enables high-performance SIMD (Single Instruction, Multiple Data) programming without requiring low-level expertise. It features:

- **Automatic Vectorization**: Write scalar-looking code, get SIMD performance
- **JIT Compilation**: Copy-and-patch compiler for sub-millisecond compilation
- **Portable SIMD**: Supports SSE4, AVX2, AVX-512, ARM NEON, SVE, and RISC-V Vector
- **Lazy Evaluation**: Automatic operation fusion for optimal performance
- **Type Safety**: Compile-time type checking with clear error messages

## Quick Start

```cpp
#include <bud_flow_lang/bud_flow_lang.h>

int main() {
    // Initialize runtime
    bud::initialize();

    // Create arrays
    auto a = bud::Bunch::fill(1000000, 2.0f).value();
    auto b = bud::Bunch::fill(1000000, 3.0f).value();

    // Compute dot product (automatically vectorized)
    float result = a.dot(b);

    std::cout << "Dot product: " << result << std::endl;

    bud::shutdown();
    return 0;
}
```

## Building

### Prerequisites

- CMake 3.16+
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Ninja (recommended) or Make

### Build Steps

```bash
# Clone the repository
git clone https://github.com/BudEcosystem/Bud-Flow-Lang.git
cd Bud-Flow-Lang

# Create build directory
mkdir build && cd build

# Configure
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
ninja

# Run tests
./bud_tests

# Run examples
./example_basic
./example_vectorize
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUD_ENABLE_TESTS` | ON | Build test suite |
| `BUD_ENABLE_BENCHMARKS` | ON | Build benchmarks |
| `BUD_ENABLE_PYTHON` | OFF | Build Python bindings |
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release/RelWithDebInfo) |

## Architecture

```
bud_flow_lang/
├── include/bud_flow_lang/   # Public API headers
│   ├── bud_flow_lang.h      # Main header
│   ├── bunch.h              # Bunch (array) abstraction
│   ├── ir.h                 # Intermediate representation
│   └── type_system.h        # Type system
├── src/
│   ├── core/                # Core infrastructure
│   ├── ir/                  # IR builder and optimizer
│   ├── jit/                 # JIT compiler (copy-and-patch)
│   └── runtime/             # Runtime support
├── tests/                   # Test suite
├── benchmarks/              # Performance benchmarks
└── examples/                # Usage examples
```

## Development

### Code Quality

This project enforces strict code quality standards:

```bash
# Run the full quality gate (required before commits)
./scripts/quality-gate.sh

# Run with auto-fix for formatting
./scripts/quality-gate.sh --fix

# Quick mode (skip sanitizer builds)
./scripts/quality-gate.sh --quick
```

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit:
- Code formatting (clang-format)
- Build verification
- Full test suite
- Static analysis
- Security scanning

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

Examples:
```
feat(jit): add copy-and-patch compiler for AVX2
fix(arena): resolve memory leak in block allocation
docs: update README with build instructions
test(ir): add unit tests for optimizer passes
```

## Performance

Bud Flow Lang achieves near-native performance through:

1. **Copy-and-Patch JIT**: ~1ms compilation time (vs 100ms+ for LLVM)
2. **Highway SIMD**: Portable, optimal code for each target architecture
3. **Operation Fusion**: Automatic fusion of element-wise operations
4. **Arena Allocation**: O(1) allocation for IR nodes

## License

Copyright 2024-2025 Bud Ecosystem Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Author

**Jithin VG** - Bud Ecosystem Inc.

## Contributing

Contributions are welcome! Please read our contributing guidelines and ensure all quality checks pass before submitting a pull request.

## Links

- [GitHub Repository](https://github.com/BudEcosystem/Bud-Flow-Lang)
- [Issue Tracker](https://github.com/BudEcosystem/Bud-Flow-Lang/issues)
- [Bud Ecosystem](https://budecosystem.com)
