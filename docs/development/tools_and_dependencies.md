# Bud Flow Lang: Complete Tools & Dependencies Guide

> Comprehensive list of all tools, libraries, and frameworks required to build Bud Flow Lang end-to-end.

---

## Table of Contents
1. [Build System & Toolchain](#1-build-system--toolchain)
2. [Compilers](#2-compilers)
3. [Core Libraries](#3-core-libraries)
4. [JIT Compilation Options](#4-jit-compilation-options)
5. [Python Bindings](#5-python-bindings)
6. [Testing Framework](#6-testing-framework)
7. [Benchmarking](#7-benchmarking)
8. [Static Analysis & Linting](#8-static-analysis--linting)
9. [Code Formatting](#9-code-formatting)
10. [Memory & Runtime Analysis](#10-memory--runtime-analysis)
11. [Fuzzing & Security](#11-fuzzing--security)
12. [Code Coverage](#12-code-coverage)
13. [Documentation](#13-documentation)
14. [Package Management](#14-package-management)
15. [CI/CD](#15-cicd)
16. [Development Environment](#16-development-environment)
17. [Complete Dependency Matrix](#17-complete-dependency-matrix)

---

## 1. Build System & Toolchain

### Primary: CMake + Ninja

| Tool | Version | Purpose |
|------|---------|---------|
| **CMake** | ≥3.20 | Meta-build system with modern C++20 support |
| **Ninja** | ≥1.11 | Fast build executor (10-20x faster than Make) |
| **ccache** | ≥4.8 | Compiler cache for faster rebuilds |

```cmake
# Recommended CMakePresets.json pattern
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy
```

### Why This Choice:
- CMake: Industry standard, excellent IDE integration, FetchContent for dependencies
- Ninja: Parallel builds, minimal overhead, used by LLVM/Chromium
- ccache: Critical for development cycles with template-heavy code

---

## 2. Compilers

### Supported Compilers

| Compiler | Minimum Version | Target Version | Notes |
|----------|----------------|----------------|-------|
| **Clang** | 14 | 18+ | Primary development compiler |
| **GCC** | 11 | 13+ | Secondary/CI testing |
| **MSVC** | 19.30 | Latest | Windows support |

### Compiler Flags (Development)
```cmake
# Clang/GCC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# For SIMD code generation analysis
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-missed")  # GCC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=loop-vectorize")  # Clang
```

---

## 3. Core Libraries

### SIMD Backend

| Library | Version | Purpose | Integration |
|---------|---------|---------|-------------|
| **Highway** | ≥1.0 | Portable SIMD primitives | **Core dependency** |

```cmake
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG master
)
FetchContent_MakeAvailable(highway)
target_link_libraries(bud_flow_lang PUBLIC hwy)
```

### JSON Handling

| Library | Purpose |
|---------|---------|
| **nlohmann/json** | Config files, IR serialization |

### Memory Management

| Library | Purpose |
|---------|---------|
| **Highway allocators** | `hwy::AllocateAligned<T>()` for SIMD-aligned memory |
| **Custom arena** | O(1) allocation for IR nodes (implement in-house) |

---

## 4. JIT Compilation Options

### Recommended: Hybrid Approach

For Bud Flow Lang, we recommend a **tiered JIT strategy**:

| Tier | Library | Use Case | Latency | Code Quality |
|------|---------|----------|---------|--------------|
| **Tier 0** | Interpreter | Cold code | N/A | N/A |
| **Tier 1** | Copy-and-Patch | Warm code | <1ms | ~70% of opt |
| **Tier 2** | MIR | Hot loops | ~10ms | ~91% of GCC -O2 |
| **Tier 3** | LLVM ORC JIT | Critical paths | ~100ms | 100% optimal |

### Library Details

#### MIR (Primary JIT Backend)
- **Repo**: https://github.com/vnmakarov/mir
- **Pros**: Lightweight, ~91% of GCC -O2 performance, fast compilation
- **Cons**: Limited to x86-64, aarch64, ppc64, s390x
- **Integration**: C API, easy to embed

```c
// MIR Example pattern
MIR_module_t m = MIR_new_module(ctx, "kernel");
MIR_item_t func = MIR_new_func(ctx, "vector_add", ...);
MIR_gen_init(ctx, 1);  // 1 = optimization level
void* code = MIR_gen(ctx, 0, func);
```

#### LLVM ORC JIT (Peak Performance)
- **Repo**: LLVM project
- **Pros**: State-of-the-art optimizations, all targets
- **Cons**: Large dependency (~100MB), slower compilation
- **Use**: Only for user-marked "hot" kernels

#### Copy-and-Patch (Fast Baseline)
- **Paper**: Xu & Kjolstad 2021
- **Pros**: 100x faster than LLVM -O0, ~900 lines Python + 500 lines C
- **Implementation**: Template stencils with hole-patching
- **Use**: First-tier warm-up before MIR/LLVM

#### AsmJit (Low-Level Alternative)
- **Repo**: https://asmjit.com/
- **Pros**: Lightweight, very low latency, fine-grained control
- **Cons**: Assembly-level only, no optimizations
- **Use**: Specialized micro-kernels

### JIT Selection Matrix

| Scenario | Recommended JIT |
|----------|----------------|
| Interactive REPL | Copy-and-Patch |
| Data science notebook | MIR |
| Production batch | LLVM ORC |
| Micro-benchmarks | AsmJit |

---

## 5. Python Bindings

### Primary: nanobind

| Library | Compile Speed | Binary Size | Performance |
|---------|---------------|-------------|-------------|
| **nanobind** | **4x faster** | **5x smaller** | Equal |
| pybind11 | Baseline | Baseline | Baseline |

```cmake
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG master
)
FetchContent_MakeAvailable(nanobind)
nanobind_add_module(bud_flow_lang_py src/python/bindings.cpp)
```

### Key nanobind Features:
- Stub generation for IDE autocomplete
- Native support for NumPy arrays
- Efficient type casters for Highway vectors
- C++17/20 support

---

## 6. Testing Framework

### Primary: GoogleTest + GoogleMock

| Component | Purpose |
|-----------|---------|
| **gtest** | Unit testing framework |
| **gmock** | Mocking framework |
| **gtest_discover_tests** | CMake CTest integration |

```cmake
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(bud_tests tests/main.cpp)
target_link_libraries(bud_tests GTest::gtest_main GTest::gmock)
gtest_discover_tests(bud_tests)
```

### Testing Categories

| Category | Tool | Purpose |
|----------|------|---------|
| Unit tests | gtest | Individual function verification |
| Property tests | rapidcheck | Fuzz-like property testing |
| SIMD correctness | Highway test utils | Cross-platform SIMD verification |
| JIT validation | Differential testing | Compare JIT vs interpreter |
| Integration | gtest | End-to-end pipeline testing |

### Differential Testing Pattern (Critical for JIT)
```cpp
// Always verify JIT output matches interpreter
void test_kernel(const Kernel& k, Input& input) {
    auto ref = run_interpreter(k, input);
    auto jit = run_jit(k, input);
    EXPECT_NEAR(ref, jit, tolerance);
}
```

---

## 7. Benchmarking

### Primary: nanobench

| Library | Overhead | Features |
|---------|----------|----------|
| **nanobench** | **80x lower than google benchmark** | Auto-warmup, PAPI integration |
| Google Benchmark | Baseline | Widely used, good docs |

```cmake
FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
)
FetchContent_MakeAvailable(nanobench)
```

### Benchmark Usage
```cpp
#include <nanobench.h>

ankerl::nanobench::Bench()
    .minEpochIterations(1000)
    .run("vector_add_highway", [&] {
        vector_add_highway(a, b, c, n);
        ankerl::nanobench::doNotOptimizeAway(c);
    });
```

### Performance Counter Integration

| Tool | Purpose | Integration |
|------|---------|-------------|
| **PAPI** | Hardware counters (cache misses, branch mispredicts) | nanobench built-in |
| **LIKWID** | Advanced performance analysis | Custom integration |
| **perf** | Linux profiling | CLI tool |

---

## 8. Static Analysis & Linting

### Multi-Tool Strategy

| Tool | Purpose | Priority |
|------|---------|----------|
| **clang-tidy** | Deep semantic analysis, modernization | **Required** |
| **cppcheck** | Fast bug detection, zero false positives | **Required** |
| **include-what-you-use** | Header optimization | Recommended |
| **PVS-Studio** | Commercial deep analysis | Optional |

### clang-tidy Configuration
```yaml
# .clang-tidy
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers

WarningsAsErrors: >
  bugprone-use-after-move,
  cppcoreguidelines-owning-memory,
  clang-analyzer-core.*

CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: camelCase
  - key: readability-identifier-naming.VariableCase
    value: lower_case
```

### CMake Integration
```cmake
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=-*,bugprone-*,performance-*")
set(CMAKE_CXX_CPPCHECK "cppcheck;--enable=all;--suppress=missingIncludeSystem")
set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "include-what-you-use")
```

---

## 9. Code Formatting

### Primary: clang-format

```yaml
# .clang-format
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
AlignAfterOpenBracket: Align
AllowShortFunctionsOnASingleLine: Inline
BreakBeforeBraces: Attach
PointerAlignment: Left
SpaceAfterCStyleCast: false
SpaceBeforeParens: ControlStatements
```

### Git Hooks (Pre-commit)
```bash
#!/bin/bash
# .git/hooks/pre-commit
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|h|hpp|cc)$')
if [ -n "$FILES" ]; then
    clang-format -i $FILES
    git add $FILES
fi
```

### EditorConfig
```ini
# .editorconfig
root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.{cpp,h,hpp}]
indent_size = 4

[CMakeLists.txt]
indent_size = 2
```

---

## 10. Memory & Runtime Analysis

### Sanitizers (MANDATORY in CI)

| Sanitizer | Flag | Detects |
|-----------|------|---------|
| **AddressSanitizer** | `-fsanitize=address` | Buffer overflows, use-after-free |
| **UndefinedBehaviorSanitizer** | `-fsanitize=undefined` | UB, integer overflow |
| **ThreadSanitizer** | `-fsanitize=thread` | Data races (cannot combine with ASan) |
| **MemorySanitizer** | `-fsanitize=memory` | Uninitialized reads (Clang only) |

```cmake
# CMake sanitizer targets
add_custom_target(build_asan
    COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g" ..
    COMMAND ${CMAKE_COMMAND} --build .
)
```

### Memory Profiling

| Tool | Use Case | Overhead |
|------|----------|----------|
| **Valgrind Memcheck** | Leak detection, invalid access | 20-30x slowdown |
| **Valgrind Massif** | Heap profiling over time | 10x slowdown |
| **heaptrack** | Lightweight heap profiling | 2-5x slowdown |
| **AddressSanitizer** | CI/development | 2x slowdown |

### Recommendation
- **Development**: heaptrack (faster than Valgrind)
- **CI**: AddressSanitizer (integrated, good speed)
- **Deep analysis**: Valgrind Massif + massif-visualizer

---

## 11. Fuzzing & Security

### Fuzzing Tools

| Tool | Strengths | Integration |
|------|-----------|-------------|
| **AFL++** | Best overall fuzzer | Requires compilation instrumentation |
| **libFuzzer** | Memory leak detection, Clang integration | Easy with `-fsanitize=fuzzer` |
| **Honggfuzz** | Multi-threading, Intel PT support | Alternative to AFL++ |

### libFuzzer Integration
```cpp
// fuzz_targets/fuzz_parser.cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        bud::parse_expression(std::string(data, data + size));
    } catch (...) {
        // Expected - parsing invalid input
    }
    return 0;
}
```

```cmake
add_executable(fuzz_parser fuzz_targets/fuzz_parser.cpp)
target_compile_options(fuzz_parser PRIVATE -fsanitize=fuzzer,address)
target_link_options(fuzz_parser PRIVATE -fsanitize=fuzzer,address)
```

### Security Hardening Flags
```cmake
# Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstack-protector-strong")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now")
```

---

## 12. Code Coverage

### Primary: llvm-cov (with gcov fallback)

| Tool | Best For |
|------|----------|
| **llvm-cov** | Clang builds, HTML reports |
| **gcov + lcov** | GCC builds, fallback |
| **gcovr** | Python-based gcov wrapper |

### CMake Coverage Target
```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-instr-generate -fcoverage-mapping")
endif()

add_custom_target(coverage
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMAND llvm-profdata merge -sparse default.profraw -o coverage.profdata
    COMMAND llvm-cov report ./bud_tests -instr-profile=coverage.profdata
    COMMAND llvm-cov show ./bud_tests -instr-profile=coverage.profdata -format=html -output-dir=coverage_html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
```

### Coverage Requirements
- **Target**: ≥80% line coverage for core modules
- **Critical paths** (JIT, type inference): ≥90% coverage

---

## 13. Documentation

### Multi-Tool Documentation Stack

| Tool | Purpose |
|------|---------|
| **Doxygen** | C++ API documentation extraction |
| **Sphinx** | Documentation website generation |
| **Breathe** | Doxygen-Sphinx bridge |
| **Exhale** | Automatic API documentation |

### Doxyfile Configuration
```
# Doxyfile
PROJECT_NAME = "Bud Flow Lang"
OUTPUT_DIRECTORY = docs/doxygen
GENERATE_XML = YES
XML_OUTPUT = xml
EXTRACT_ALL = YES
RECURSIVE = YES
INPUT = include src
```

### Sphinx conf.py
```python
extensions = [
    'breathe',
    'exhale',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

breathe_projects = {"bud_flow_lang": "../doxygen/xml"}
breathe_default_project = "bud_flow_lang"

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "doxygenStripFromPath": "..",
}
```

---

## 14. Package Management

### Primary: CMake FetchContent + CPM

| Manager | Use Case |
|---------|----------|
| **FetchContent** | Built-in CMake, simple dependencies |
| **CPM** | Enhanced FetchContent with caching |
| **vcpkg** | System-wide binary packages |
| **Conan** | Advanced dependency management |

### CPM Setup (Recommended)
```cmake
# Download CPM.cmake
file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
    ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

# Use CPM for dependencies
CPMAddPackage("gh:google/highway@1.0.7")
CPMAddPackage("gh:wjakob/nanobind@2.0.0")
CPMAddPackage("gh:google/googletest@1.14.0")
```

---

## 15. CI/CD

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    strategy:
      matrix:
        os: [ubuntu-24.04, macos-14, windows-2022]
        compiler: [clang, gcc]
        exclude:
          - os: windows-2022
            compiler: gcc

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          if [ "$RUNNER_OS" == "Linux" ]; then
            sudo apt-get update
            sudo apt-get install -y ninja-build ccache
          fi

      - name: Configure CMake
        run: cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

      - name: Build
        run: cmake --build build --parallel

      - name: Test
        run: ctest --test-dir build --output-on-failure

  sanitizers:
    runs-on: ubuntu-24.04
    strategy:
      matrix:
        sanitizer: [address, undefined, thread]

    steps:
      - uses: actions/checkout@v4
      - name: Build with sanitizer
        run: |
          cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=${{ matrix.sanitizer }} -g"
          cmake --build build
          ctest --test-dir build

  coverage:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Build with coverage
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Coverage
          cmake --build build
          cmake --build build --target coverage
      - uses: codecov/codecov-action@v4
        with:
          files: build/coverage.info

  static-analysis:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Run clang-tidy
        run: |
          cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
          clang-tidy -p build src/**/*.cpp

  fuzz:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Build fuzzers
        run: |
          cmake -B build -DBUILD_FUZZERS=ON
          cmake --build build
      - name: Run fuzzers (short)
        run: ./build/fuzz_parser -max_total_time=60
```

---

## 16. Development Environment

### IDE Support

| IDE | Configuration |
|-----|---------------|
| **VS Code** | `compile_commands.json` + clangd |
| **CLion** | Native CMake support |
| **Vim/Neovim** | clangd + coc.nvim |

### VS Code Settings
```json
// .vscode/settings.json
{
    "clangd.arguments": [
        "--background-index",
        "--clang-tidy",
        "--completion-style=detailed"
    ],
    "cmake.configureOnOpen": true,
    "cmake.generator": "Ninja"
}
```

### Required VS Code Extensions
- C/C++ Extension Pack
- CMake Tools
- clangd
- CodeLLDB (debugging)

---

## 17. Complete Dependency Matrix

### Required Dependencies

| Dependency | Version | Category | Source |
|------------|---------|----------|--------|
| Highway | ≥1.0 | SIMD | FetchContent |
| nanobind | ≥2.0 | Python bindings | FetchContent |
| nlohmann/json | ≥3.11 | Serialization | FetchContent |
| GoogleTest | ≥1.14 | Testing | FetchContent |

### Optional Dependencies

| Dependency | Version | Category | When Needed |
|------------|---------|----------|-------------|
| MIR | Latest | JIT | Tier 2 JIT backend |
| LLVM | ≥16 | JIT | Tier 3 JIT backend |
| AsmJit | ≥1.13 | JIT | Low-level kernels |
| PAPI | ≥7.0 | Profiling | Hardware counters |
| LIKWID | ≥5.3 | Profiling | Advanced analysis |

### Development Dependencies

| Dependency | Category |
|------------|----------|
| CMake ≥3.20 | Build |
| Ninja ≥1.11 | Build |
| Clang ≥14 | Compiler |
| clang-tidy | Linting |
| clang-format | Formatting |
| ccache | Build acceleration |
| Doxygen | Documentation |

---

## Quick Start Commands

```bash
# Clone and build
git clone https://github.com/your-org/bud_flow_lang.git
cd bud_flow_lang
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure

# Run with sanitizers
cmake -B build-asan -G Ninja -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g"
cmake --build build-asan && ctest --test-dir build-asan

# Generate coverage
cmake -B build-cov -G Ninja -DCMAKE_BUILD_TYPE=Coverage
cmake --build build-cov --target coverage

# Run static analysis
clang-tidy -p build src/**/*.cpp

# Format code
find src include -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Build Python bindings
cmake -B build -G Ninja -DBUILD_PYTHON=ON
cmake --build build
pip install ./build
```

---

## Sources & References

- [CMake Documentation](https://cmake.org/documentation/)
- [Google Highway](https://github.com/google/highway)
- [nanobind](https://github.com/wjakob/nanobind)
- [MIR JIT Compiler](https://github.com/vnmakarov/mir)
- [LLVM ORC JIT Tutorial](https://llvm.org/docs/tutorial/BuildingAJIT1.html)
- [AsmJit](https://asmjit.com/)
- [Copy-and-Patch Paper](https://fredrikbk.com/publications/copy-and-patch.pdf)
- [GoogleTest](https://google.github.io/googletest/)
- [nanobench](https://github.com/martinus/nanobench)
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
- [Include What You Use](https://include-what-you-use.org/)
- [heaptrack](https://github.com/KDE/heaptrack)
- [AFL++](https://github.com/AFLplusplus/AFLplusplus)
- [Codecov](https://codecov.io/)

---

*Document Version: 1.0*
*Last Updated: 2024*
