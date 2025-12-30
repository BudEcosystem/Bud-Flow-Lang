# Bud Flow Lang Examples

This directory contains example programs demonstrating Bud Flow Lang features.

## Examples

### Getting Started

| File | Description |
|------|-------------|
| [hello_flow.py](hello_flow.py) | Your first Bud Flow Lang program |

### Benchmarking

| File | Description |
|------|-------------|
| [benchmark_suite.py](benchmark_suite.py) | Comprehensive performance benchmarks |
| [numpy_comparison.py](numpy_comparison.py) | Side-by-side comparison with NumPy |

### Applications

| File | Description |
|------|-------------|
| [vector_operations.py](vector_operations.py) | Common vector math operations |
| [machine_learning.py](machine_learning.py) | ML primitives (activations, losses, layers) |

## Running Examples

Make sure Bud Flow Lang is built first:

```bash
cd /path/to/bud_flow_lang
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Then run any example:

```bash
# Add build directory to Python path
export PYTHONPATH=/path/to/bud_flow_lang/build:$PYTHONPATH

# Run examples
python examples/hello_flow.py
python examples/benchmark_suite.py
python examples/numpy_comparison.py
python examples/vector_operations.py
python examples/machine_learning.py
```

## Example Output

### hello_flow.py

```
Bud Flow Lang initialized!

=== Hardware Information ===
Architecture: x86
SIMD Width: 256 bits
CPU Cores: 8 physical, 16 logical
SIMD Features: AVX-512, AVX2, AVX, SSE2

=== Array Creation ===
ones(10): [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
zeros(10): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
arange(10): [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
...
```

### benchmark_suite.py

```
=== Binary Operations ===

Size: 1,000,000 elements
  add                     0.3521 ms (0.0124 std)   2.84 Gelem/s  34.1 GB/s
  mul                     0.3489 ms (0.0098 std)   2.87 Gelem/s  34.4 GB/s
  fma                     0.4012 ms (0.0156 std)   2.49 Gelem/s  39.8 GB/s
...
```

## Creating Your Own Examples

Use this template:

```python
#!/usr/bin/env python3
"""
My Bud Flow Lang Example

Description of what this example demonstrates.

Run: python my_example.py
"""

import bud_flow_lang_py as flow


def main():
    # Initialize the runtime
    flow.initialize()

    # Print hardware info
    info = flow.get_hardware_info()
    print(f"Running on {info['arch_family']} with {info['simd_width']*8}-bit SIMD")

    # Your code here
    a = flow.ones(1000)
    b = flow.full(1000, 2.0)

    # Perform operations
    result = flow.fma(a, b, a)  # a * b + a

    print(f"Result sum: {result.sum()}")


if __name__ == "__main__":
    main()
```

## See Also

- [Documentation](../docs/index.md) - Full documentation
- [Tutorials](../docs/tutorials/) - Step-by-step tutorials
- [API Reference](../docs/api/) - Complete API documentation
- [Notebooks](../docs/notebooks/) - Jupyter notebooks for Google Colab
