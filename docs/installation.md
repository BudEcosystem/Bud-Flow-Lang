# Installation Guide

This guide covers all methods to install Bud Flow Lang on various platforms.

---

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+), macOS 11+, Windows 10+ (WSL2 recommended)
- **CPU**: x86-64 with SSE4.2+ or ARM64 with NEON
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 500MB for build

### Build Requirements
- **C++ Compiler**: GCC 10+ or Clang 12+ (C++17 support required)
- **CMake**: 3.16 or higher
- **Python**: 3.8+ (for Python bindings)
- **NumPy**: 1.20+ (for Python interop)

---

## Quick Install (Google Colab)

The fastest way to try Bud Flow Lang is in Google Colab:

```python
# Cell 1: Install dependencies and build
!apt-get update && apt-get install -y cmake g++ python3-dev
!git clone https://github.com/anthropics/bud_flow_lang.git
!cd bud_flow_lang && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUD_BUILD_PYTHON=ON && make -j4

# Cell 2: Import and use
import sys
sys.path.insert(0, '/content/bud_flow_lang/build')
import bud_flow_lang_py as flow

flow.initialize()
print("Bud Flow Lang installed successfully!")
print(f"SIMD Width: {flow.get_hardware_info()['simd_width']} bytes")
```

---

## Building from Source

### Step 1: Clone the Repository

```bash
git clone https://github.com/anthropics/bud_flow_lang.git
cd bud_flow_lang
```

### Step 2: Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-numpy \
    python3-pip
```

#### Fedora/RHEL
```bash
sudo dnf install -y \
    gcc-c++ \
    cmake \
    git \
    python3-devel \
    python3-numpy
```

#### macOS
```bash
brew install cmake python numpy
```

#### Windows (WSL2 recommended)
```bash
# In WSL2 Ubuntu
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-numpy
```

### Step 3: Build

#### Standard Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### Build with Python Bindings (default)
```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUD_BUILD_PYTHON=ON \
    -DBUD_BUILD_TESTS=ON
make -j$(nproc)
```

#### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUD_BUILD_PYTHON` | ON | Build Python bindings |
| `BUD_BUILD_TESTS` | ON | Build test suite |
| `BUD_BUILD_BENCHMARKS` | OFF | Build benchmarks |
| `BUD_ENABLE_SANITIZERS` | OFF | Enable ASan/UBSan |

Example with all options:
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUD_BUILD_PYTHON=ON \
    -DBUD_BUILD_TESTS=ON \
    -DBUD_BUILD_BENCHMARKS=ON
```

### Step 4: Verify Installation

```bash
# Run C++ tests
ctest --output-on-failure

# Test Python bindings
python3 -c "
import sys
sys.path.insert(0, '.')
import bud_flow_lang_py as flow
flow.initialize()
a = flow.ones(100)
print(f'Sum of 100 ones: {a.sum()}')
print('Installation successful!')
"
```

---

## Installing the Python Module

### Option 1: Add to PYTHONPATH

```bash
# Add to your .bashrc or .zshrc
export PYTHONPATH="/path/to/bud_flow_lang/build:$PYTHONPATH"
```

### Option 2: Copy to site-packages

```bash
# Find your site-packages
python3 -c "import site; print(site.getsitepackages()[0])"

# Copy the module
cp build/bud_flow_lang_py.cpython-*.so /path/to/site-packages/
```

### Option 3: Create a symlink

```bash
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
ln -s $(pwd)/build/bud_flow_lang_py.cpython-*.so $SITE_PACKAGES/
```

---

## Platform-Specific Notes

### Intel CPUs (AVX2/AVX-512)

For best performance on Intel CPUs:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=native"
```

### AMD CPUs (Zen 3/4)

AMD Zen 3+ supports AVX2 well:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=znver3"
```

### Apple Silicon (M1/M2/M3)

For ARM64 with NEON:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64
```

### Raspberry Pi 4 (ARM64)

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=armv8-a+simd"
```

---

## Troubleshooting

### CMake can't find Python

```bash
cmake .. -DPython3_EXECUTABLE=$(which python3)
```

### Missing NumPy headers

```bash
pip3 install numpy
cmake .. -DPython3_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())")
```

### Linker errors with nanobind

Ensure you have a recent compiler:
```bash
# Check compiler version
g++ --version  # Need 10+
clang++ --version  # Need 12+
```

### Tests fail with "Illegal instruction"

Your CPU doesn't support the SIMD instructions. Build with:
```bash
cmake .. -DCMAKE_CXX_FLAGS="-march=x86-64-v2"  # Baseline with SSE4
```

### Import error in Python

```python
# Check if the module exists
import os
print(os.path.exists('/path/to/build/bud_flow_lang_py.cpython-*.so'))

# Check Python version match
import sys
print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
```

---

## Development Setup

For contributors:

```bash
# Clone with submodules
git clone --recursive https://github.com/anthropics/bud_flow_lang.git

# Build with debug symbols and sanitizers
mkdir build_debug && cd build_debug
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUD_ENABLE_SANITIZERS=ON \
    -DBUD_BUILD_TESTS=ON

make -j$(nproc)

# Run tests with sanitizers
ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure
```

---

## Verifying SIMD Support

After installation, verify your SIMD capabilities:

```python
import bud_flow_lang_py as flow

flow.initialize()
info = flow.get_hardware_info()

print("=== Hardware Info ===")
print(f"Architecture: {info['arch_family']}")
print(f"SIMD Width: {info['simd_width']} bytes ({info['simd_width']*8} bits)")
print(f"Physical Cores: {info['physical_cores']}")

print("\n=== x86 SIMD ===")
print(f"SSE2: {info['has_sse2']}")
print(f"AVX: {info['has_avx']}")
print(f"AVX2: {info['has_avx2']}")
print(f"AVX-512: {info['has_avx512']}")

print("\n=== ARM SIMD ===")
print(f"NEON: {info['has_neon']}")
print(f"SVE: {info['has_sve']}")

print("\n=== Cache Config ===")
cache = flow.detect_cache_config()
print(f"L1: {cache['l1_size_kb']}KB")
print(f"L2: {cache['l2_size_kb']}KB")
print(f"L3: {cache['l3_size_kb']}KB")
```

---

## Next Steps

- [Getting Started Guide](getting_started.md) - Your first Flow program
- [Basic Operations Tutorial](tutorials/02_basic_operations.md) - Learn the API
- [API Reference](api/core.md) - Complete API documentation
