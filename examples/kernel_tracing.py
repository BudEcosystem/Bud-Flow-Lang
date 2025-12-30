#!/usr/bin/env python3
"""Kernel tracing and JIT compilation example.

This example demonstrates how to use the @flow.kernel decorator to trace
Python functions and compile them to optimized SIMD code.

Run from project root: python3 examples/kernel_tracing.py
"""

import sys
import time
from pathlib import Path

# Add build directory to path if running from examples directory
build_dir = Path(__file__).parent.parent / "build"
if build_dir.exists():
    sys.path.insert(0, str(build_dir))

import bud_flow_lang_py as flow


def main():
    print("=== Bud Flow Lang Kernel Tracing ===\n")

    # Define a simple kernel using the decorator
    @flow.kernel
    def saxpy(x, y):
        """Simple addition: result = x + y"""
        return x + y

    @flow.kernel
    def polynomial(x, ones):
        """Evaluate x^2 + 2x + 1

        Note: We pass 'ones' as a parameter since flow.ones()
        cannot be called inside a traced kernel.
        """
        return x * x + x + x + ones

    @flow.kernel(opt_level=2, enable_fusion=True)
    def complex_kernel(a, b, c):
        """(a + b) * c with fusion enabled"""
        temp = a + b
        return temp * c

    @flow.kernel
    def scaled_sum(x, y, scale):
        """Compute (x + y) * scale"""
        return (x + y) * scale

    # Test data
    n = 1000
    x = flow.arange(n)
    y = flow.ones(n)

    print("1. Simple Addition Kernel")
    print("-" * 40)

    result = saxpy(x, y)
    print(f"x = [0, 1, 2, ..., {n-1}]")
    print(f"y = [1, 1, 1, ..., 1]")
    print(f"saxpy(x, y)[:5] = {list(result.to_numpy()[:5])}")
    print(f"saxpy(x, y)[-5:] = {list(result.to_numpy()[-5:])}")

    print("\n2. Polynomial Kernel")
    print("-" * 40)

    x_small = flow.arange(5)
    ones_small = flow.ones(5)
    result = polynomial(x_small, ones_small)
    print(f"x = {list(x_small.to_numpy())}")
    print(f"x^2 + 2x + 1 = {list(result.to_numpy())}")

    # Verify: 0^2+0+0+1=1, 1^2+1+1+1=4, 2^2+2+2+1=9, ...
    expected = [(i * i + 2 * i + 1) for i in range(5)]
    print(f"Expected: {expected}")

    print("\n3. Complex Kernel with Fusion")
    print("-" * 40)

    # Create arrays using factory functions
    a = flow.arange(5) + flow.ones(5)  # [1, 2, 3, 4, 5]
    b = flow.full(5, 0.5)               # [0.5, 0.5, 0.5, 0.5, 0.5]
    c = flow.full(5, 2.0)               # [2, 2, 2, 2, 2]

    result = complex_kernel(a, b, c)
    print(f"a = {list(a.to_numpy())}")
    print(f"b = {list(b.to_numpy())}")
    print(f"c = {list(c.to_numpy())}")
    print(f"(a + b) * c = {list(result.to_numpy())}")

    print("\n4. Scaled Sum Kernel")
    print("-" * 40)

    x_arr = flow.arange(5)
    y_arr = flow.full(5, 10.0)
    scale = flow.full(5, 0.5)

    result = scaled_sum(x_arr, y_arr, scale)
    print(f"x = {list(x_arr.to_numpy())}")
    print(f"y = {list(y_arr.to_numpy())}")
    print(f"scale = {list(scale.to_numpy())}")
    print(f"(x + y) * scale = {list(result.to_numpy())}")

    print("\n5. Performance Comparison")
    print("-" * 40)

    # Large array for benchmarking
    n_large = 100000
    x_large = flow.arange(n_large)
    y_large = flow.ones(n_large)

    # Warm up (JIT compilation happens here)
    _ = saxpy(x_large, y_large)

    # Benchmark compiled kernel
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        result = saxpy(x_large, y_large)
    elapsed = time.perf_counter() - start

    print(f"Array size: {n_large:,} elements")
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed * 1000:.2f} ms")
    print(f"Per iteration: {elapsed / iterations * 1000:.4f} ms")
    print(f"Throughput: {n_large * iterations / elapsed / 1e9:.2f} billion ops/sec")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
