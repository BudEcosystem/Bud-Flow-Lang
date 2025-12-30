#!/usr/bin/env python3
"""
Hello Bud Flow Lang - Your first program

This example demonstrates the basic usage of Bud Flow Lang:
- Initialization
- Hardware detection
- Array creation
- Basic operations
- Reductions

Run: python hello_flow.py
"""

import bud_flow_lang_py as flow


def main():
    # Step 1: Initialize the runtime
    flow.initialize()
    print("Bud Flow Lang initialized!")
    print()

    # Step 2: Check hardware capabilities
    info = flow.get_hardware_info()
    print("=== Hardware Information ===")
    print(f"Architecture: {info['arch_family']}")
    print(f"SIMD Width: {info['simd_width'] * 8} bits")
    print(f"CPU Cores: {info['physical_cores']} physical, {info['logical_cores']} logical")

    features = []
    if info['has_avx512']:
        features.append("AVX-512")
    if info['has_avx2']:
        features.append("AVX2")
    if info['has_avx']:
        features.append("AVX")
    if info['has_sse2']:
        features.append("SSE2")
    if info['has_neon']:
        features.append("NEON")
    if info['has_sve']:
        features.append("SVE")

    print(f"SIMD Features: {', '.join(features)}")
    print()

    # Step 3: Create arrays
    print("=== Array Creation ===")
    a = flow.ones(10)
    print(f"ones(10): {a.to_numpy()}")

    b = flow.zeros(10)
    print(f"zeros(10): {b.to_numpy()}")

    c = flow.arange(10)
    print(f"arange(10): {c.to_numpy()}")

    d = flow.full(10, 3.14)
    print(f"full(10, 3.14): {d.to_numpy()}")

    e = flow.linspace(0, 1, 5)
    print(f"linspace(0, 1, 5): {e.to_numpy()}")
    print()

    # Step 4: Perform operations
    print("=== Basic Operations ===")
    x = flow.arange(5)
    y = flow.full(5, 2.0)

    print(f"x = {x.to_numpy()}")
    print(f"y = {y.to_numpy()}")
    print(f"x + y = {(x + y).to_numpy()}")
    print(f"x - y = {(x - y).to_numpy()}")
    print(f"x * y = {(x * y).to_numpy()}")
    print(f"x / y = {(x / y).to_numpy()}")
    print()

    # Step 5: Reductions
    print("=== Reductions ===")
    arr = flow.arange(10)
    print(f"Array: {arr.to_numpy()}")
    print(f"Sum: {arr.sum()}")
    print(f"Min: {arr.min()}")
    print(f"Max: {arr.max()}")
    print(f"Mean: {arr.mean()}")
    print()

    # Step 6: Dot product
    print("=== Dot Product ===")
    a = flow.ones(100)
    b = flow.full(100, 2.0)
    dot = flow.dot(a, b)
    print(f"dot(ones(100), full(100, 2.0)) = {dot}")
    print()

    # Step 7: FMA (Fused Multiply-Add)
    print("=== Fused Multiply-Add ===")
    a = flow.ones(5)
    b = flow.full(5, 2.0)
    c = flow.full(5, 3.0)
    fma_result = flow.fma(a, b, c)
    print(f"a = {a.to_numpy()}")
    print(f"b = {b.to_numpy()}")
    print(f"c = {c.to_numpy()}")
    print(f"fma(a, b, c) = a*b + c = {fma_result.to_numpy()}")
    print()

    print("Success! Bud Flow Lang is working correctly.")


if __name__ == "__main__":
    main()
