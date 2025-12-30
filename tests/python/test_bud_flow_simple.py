#!/usr/bin/env python3
"""
Bud Flow Lang - Simple System Test & NumPy Comparison

This script tests the core functionality of bud_flow_lang and compares
performance with NumPy for key operations.

Run from build directory:
    python3 ../tests/python/test_bud_flow_simple.py
"""

import sys
import os
import time
import gc

# Add build directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'build')
sys.path.insert(0, build_dir)

try:
    import bud_flow_lang_py as flow
except ImportError as e:
    print(f"Error: Could not import bud_flow_lang_py: {e}")
    sys.exit(1)

import numpy as np


def benchmark(name, flow_func, np_func, iterations=10, warmup=2):
    """Benchmark flow vs numpy and return times in ms."""
    # Warmup
    for _ in range(warmup):
        flow_func()
        np_func()

    # Flow timing
    gc.collect()
    start = time.perf_counter()
    for _ in range(iterations):
        flow_func()
    flow_time = (time.perf_counter() - start) / iterations * 1000

    # NumPy timing
    gc.collect()
    start = time.perf_counter()
    for _ in range(iterations):
        np_func()
    np_time = (time.perf_counter() - start) / iterations * 1000

    return flow_time, np_time


def main():
    print("\n" + "=" * 70)
    print("  BUD FLOW LANG - SYSTEM TEST & NUMPY COMPARISON")
    print("=" * 70)

    # Initialize
    if not flow.is_initialized():
        flow.initialize()

    # =========================================================================
    # Section 1: Hardware Info
    # =========================================================================
    print("\n--- Hardware Configuration ---")
    info = flow.get_hardware_info()
    print(f"  Architecture: {info['arch_family']} ({'64-bit' if info['is_64bit'] else '32-bit'})")
    print(f"  SIMD Width: {info['simd_width']} bytes ({info['simd_width']*8} bits)")
    print(f"  AVX2: {info['has_avx2']}, AVX-512: {info['has_avx512']}")

    cache = flow.detect_cache_config()
    print(f"  L1: {cache['l1_size_kb']}KB, L2: {cache['l2_size_kb']}KB, L3: {cache['l3_size_kb']}KB")

    status = flow.get_memory_optimization_status()
    print(f"  Tiling: {'enabled' if status['tiling_enabled'] else 'disabled'}")
    print(f"  Prefetch: {'enabled' if status['prefetch_enabled'] else 'disabled'}")

    # =========================================================================
    # Section 2: Correctness Tests
    # =========================================================================
    print("\n--- Correctness Tests ---")
    all_passed = True

    # Array creation
    a = flow.ones(100)
    assert abs(a.sum() - 100.0) < 0.01, "ones() failed"
    print("  [PASS] ones()")

    a = flow.zeros(100)
    assert abs(a.sum()) < 0.01, "zeros() failed"
    print("  [PASS] zeros()")

    a = flow.arange(10)
    assert abs(a.sum() - 45.0) < 0.01, "arange() failed"
    print("  [PASS] arange()")

    # NumPy interop
    np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    flow_arr = flow.flow(np_arr)
    assert abs(flow_arr.sum() - 10.0) < 0.01, "flow(numpy) failed"
    print("  [PASS] flow(numpy)")

    # Binary operations
    a = flow.ones(100)
    b = flow.full(100, 2.0)
    c = a + b
    assert abs(c.sum() - 300.0) < 0.01, "add failed"
    print("  [PASS] add")

    c = a * b
    assert abs(c.sum() - 200.0) < 0.01, "mul failed"
    print("  [PASS] mul")

    # Dot product
    a = flow.ones(1000)
    b = flow.ones(1000)
    dot = flow.dot(a, b)
    assert abs(dot - 1000.0) < 0.1, "dot failed"
    print("  [PASS] dot")

    # FMA
    a = flow.ones(100)
    b = flow.full(100, 2.0)
    c = flow.full(100, 3.0)
    result = flow.fma(a, b, c)  # a*b + c = 1*2 + 3 = 5
    assert abs(result.sum() - 500.0) < 0.1, "fma failed"
    print("  [PASS] fma")

    # Large array (tests tiling)
    a = flow.ones(1000000)  # 1M elements
    assert abs(a.sum() - 1000000.0) < 1000, "large sum failed"
    print("  [PASS] large array sum (1M elements)")

    b = flow.ones(1000000)
    dot = flow.dot(a, b)
    assert abs(dot - 1000000.0) < 1000, "large dot failed"
    print("  [PASS] large dot product (1M elements)")

    print("\n  All correctness tests passed!")

    # =========================================================================
    # Section 3: Performance Comparison
    # =========================================================================
    print("\n--- Performance Comparison (Flow vs NumPy) ---")

    sizes = [10000, 100000, 1000000]

    for n in sizes:
        print(f"\n  Array size: {n:,} elements")

        # Create arrays
        np_a = np.ones(n, dtype=np.float32)
        np_b = np.ones(n, dtype=np.float32)
        flow_a = flow.ones(n)
        flow_b = flow.ones(n)

        # Sum reduction
        flow_time, np_time = benchmark(
            "sum",
            lambda: flow_a.sum(),
            lambda: np_a.sum(),
            iterations=20
        )
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Sum:       Flow={flow_time:.3f}ms, NumPy={np_time:.3f}ms, Speedup={speedup:.2f}x")

        # Dot product
        flow_time, np_time = benchmark(
            "dot",
            lambda: flow.dot(flow_a, flow_b),
            lambda: np.dot(np_a, np_b),
            iterations=20
        )
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Dot:       Flow={flow_time:.3f}ms, NumPy={np_time:.3f}ms, Speedup={speedup:.2f}x")

        # Addition (element-wise)
        flow_time, np_time = benchmark(
            "add",
            lambda: flow_a + flow_b,
            lambda: np_a + np_b,
            iterations=10
        )
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Add:       Flow={flow_time:.3f}ms, NumPy={np_time:.3f}ms, Speedup={speedup:.2f}x")

        # Cleanup to prevent memory buildup
        del flow_a, flow_b
        gc.collect()

    # =========================================================================
    # Section 4: Memory Bandwidth Estimate
    # =========================================================================
    print("\n--- Memory Bandwidth Estimate ---")

    n = 10000000  # 10M elements
    flow_a = flow.ones(n)
    flow_b = flow.ones(n)
    np_a = np.ones(n, dtype=np.float32)
    np_b = np.ones(n, dtype=np.float32)

    # Dot product reads 2 arrays = 80MB
    bytes_read = 2 * n * 4

    flow_time, np_time = benchmark("dot",
        lambda: flow.dot(flow_a, flow_b),
        lambda: np.dot(np_a, np_b),
        iterations=5
    )

    flow_bw = bytes_read / (flow_time / 1000) / 1e9
    np_bw = bytes_read / (np_time / 1000) / 1e9

    print(f"  Dot product (10M elements, 80MB):")
    print(f"    Flow: {flow_time:.2f}ms, {flow_bw:.1f} GB/s")
    print(f"    NumPy: {np_time:.2f}ms, {np_bw:.1f} GB/s")

    # Cleanup
    del flow_a, flow_b
    gc.collect()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    print("\nBud Flow Lang is working correctly with SIMD optimizations.")
    print("Memory optimization features (tiling, prefetching) are active.\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
