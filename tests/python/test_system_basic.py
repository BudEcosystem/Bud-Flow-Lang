#!/usr/bin/env python3
"""
Bud Flow Lang System Test - Basic Operations & NumPy Comparison

This script tests:
1. Basic SIMD operations through Python bindings
2. Memory optimization features (tiling, prefetching)
3. Performance comparison with NumPy

Run from build directory:
    python3 ../tests/python/test_system_basic.py
"""

import sys
import os
import time
import numpy as np

# Add build directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'build')
sys.path.insert(0, build_dir)

try:
    import bud_flow_lang_py as flow
except ImportError as e:
    print(f"Error: Could not import bud_flow_lang_py: {e}")
    print(f"Make sure you've built the project and are running from the correct directory")
    sys.exit(1)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def check_close(actual, expected, name, rtol=1e-4):
    """Check if values are close and report result."""
    if isinstance(actual, float) and isinstance(expected, float):
        close = abs(actual - expected) <= rtol * max(abs(actual), abs(expected), 1e-8) + 1e-8
    else:
        close = np.allclose(actual, expected, rtol=rtol)

    status = "PASS" if close else "FAIL"
    print(f"  [{status}] {name}")
    if not close:
        if hasattr(actual, '__len__') and len(actual) > 10:
            print(f"       Expected (first 5): {expected[:5]}")
            print(f"       Actual (first 5):   {actual[:5]}")
        else:
            print(f"       Expected: {expected}")
            print(f"       Actual:   {actual}")
    return close


def time_operation(name, func, iterations=10):
    """Time an operation and return average time in ms."""
    # Warmup
    func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()

    avg_ms = (end - start) / iterations * 1000
    return avg_ms


# =============================================================================
# Section 1: Hardware and Configuration
# =============================================================================

def test_hardware_info():
    """Test hardware detection and configuration."""
    print_header("1. HARDWARE DETECTION & CONFIGURATION")

    print_subheader("Hardware Info")
    info = flow.get_hardware_info()
    print(f"  CPU: {info['cpu_name']}")
    print(f"  Vendor: {info['vendor']}")
    print(f"  Architecture: {info['arch_family']} ({'64-bit' if info['is_64bit'] else '32-bit'})")
    print(f"  SIMD Width: {info['simd_width']} bytes")
    print(f"  Cores: {info['physical_cores']} physical, {info['logical_cores']} logical")

    print_subheader("SIMD Capabilities")
    caps = flow.get_simd_capabilities()
    print(f"  {caps}")

    print_subheader("x86 SIMD Flags")
    print(f"  SSE2:    {info['has_sse2']}")
    print(f"  AVX:     {info['has_avx']}")
    print(f"  AVX2:    {info['has_avx2']}")
    print(f"  AVX-512: {info['has_avx512']}")

    print_subheader("Cache Configuration")
    cache = flow.detect_cache_config()
    print(f"  L1 Cache: {cache['l1_size_kb']} KB")
    print(f"  L2 Cache: {cache['l2_size_kb']} KB")
    print(f"  L3 Cache: {cache['l3_size_kb']} KB")
    print(f"  Cache Line: {cache['line_size']} bytes")

    print_subheader("Memory Optimization")
    status = flow.get_memory_optimization_status()
    print(f"  Tiling Enabled: {status['tiling_enabled']}")
    print(f"  Prefetch Enabled: {status['prefetch_enabled']}")

    tile_size = flow.optimal_tile_size(4, 2)
    print(f"  Optimal Tile Size (float32, 2 arrays): {tile_size} elements")

    return True


# =============================================================================
# Section 2: Basic Operations
# =============================================================================

def test_basic_operations():
    """Test basic SIMD operations."""
    print_header("2. BASIC SIMD OPERATIONS")

    all_passed = True

    # Test array creation
    print_subheader("Array Creation")

    x = flow.zeros(10)
    all_passed &= check_close(x.sum(), 0.0, "zeros(10).sum() == 0")

    x = flow.ones(10)
    all_passed &= check_close(x.sum(), 10.0, "ones(10).sum() == 10")

    x = flow.full(10, 3.14)
    all_passed &= check_close(x.sum(), 31.4, "full(10, 3.14).sum() == 31.4")

    x = flow.arange(10)
    all_passed &= check_close(x.sum(), 45.0, "arange(10).sum() == 45")

    x = flow.linspace(0.0, 1.0, 5)
    arr = x.to_numpy()
    all_passed &= check_close(arr, [0.0, 0.25, 0.5, 0.75, 1.0], "linspace(0, 1, 5)")

    # Test from numpy
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = flow.flow(np_arr)
    all_passed &= check_close(x.sum(), 15.0, "flow(numpy).sum()")

    # Test from list
    x = flow.flow([1.0, 2.0, 3.0])
    all_passed &= check_close(x.sum(), 6.0, "flow([1,2,3]).sum()")

    print_subheader("Binary Operations")

    a = flow.arange(5)
    b = flow.ones(5)

    # Addition
    c = a + b
    all_passed &= check_close(c.to_numpy(), [1, 2, 3, 4, 5], "arange + ones")

    # Subtraction
    c = a - b
    all_passed &= check_close(c.to_numpy(), [-1, 0, 1, 2, 3], "arange - ones")

    # Multiplication
    c = a * b
    all_passed &= check_close(c.to_numpy(), [0, 1, 2, 3, 4], "arange * ones")

    # Division
    a = flow.flow([4.0, 6.0, 8.0, 10.0])
    b = flow.flow([2.0, 2.0, 2.0, 2.0])
    c = a / b
    all_passed &= check_close(c.to_numpy(), [2, 3, 4, 5], "division")

    print_subheader("Unary Operations")

    x = flow.flow([1.0, 4.0, 9.0, 16.0])
    all_passed &= check_close(x.sqrt().to_numpy(), [1, 2, 3, 4], "sqrt")

    x = flow.flow([-1.0, -2.0, 3.0, -4.0])
    all_passed &= check_close(x.abs().to_numpy(), [1, 2, 3, 4], "abs")

    x = flow.flow([1.0, 2.0, 3.0])
    all_passed &= check_close((-x).to_numpy(), [-1, -2, -3], "negation")

    print_subheader("Reduction Operations")

    x = flow.arange(10)
    all_passed &= check_close(x.sum(), 45.0, "sum")
    all_passed &= check_close(x.min(), 0.0, "min")
    all_passed &= check_close(x.max(), 9.0, "max")
    all_passed &= check_close(x.mean(), 4.5, "mean")

    print_subheader("Dot Product")

    a = flow.ones(100)
    b = flow.full(100, 2.0)
    all_passed &= check_close(a.dot(b), 200.0, "dot product (100 elements)")

    a = flow.ones(10000)
    b = flow.ones(10000)
    all_passed &= check_close(a.dot(b), 10000.0, "dot product (10K elements)", rtol=1e-3)

    print_subheader("FMA and Special Operations")

    a = flow.flow([1.0, 2.0, 3.0])
    b = flow.flow([2.0, 2.0, 2.0])
    c = flow.flow([10.0, 10.0, 10.0])
    result = flow.fma(a, b, c)  # a*b + c
    all_passed &= check_close(result.to_numpy(), [12, 14, 16], "fma")

    x = flow.flow([-5.0, 0.0, 5.0, 10.0])
    result = flow.clamp(x, 0.0, 7.0)
    all_passed &= check_close(result.to_numpy(), [0, 0, 5, 7], "clamp")

    a = flow.flow([0.0, 0.0, 0.0])
    b = flow.flow([10.0, 10.0, 10.0])
    result = flow.lerp(a, b, 0.3)
    all_passed &= check_close(result.to_numpy(), [3, 3, 3], "lerp")

    return all_passed


# =============================================================================
# Section 3: Large Array Operations (Tests Tiling/Prefetch)
# =============================================================================

def test_large_arrays():
    """Test operations on large arrays that benefit from tiling."""
    print_header("3. LARGE ARRAY OPERATIONS (Tiling/Prefetch)")

    all_passed = True

    sizes = [1000, 10000, 100000, 1000000]

    print_subheader("Sum Reduction (Verifying Correctness)")

    for n in sizes:
        x = flow.ones(n)
        result = x.sum()
        all_passed &= check_close(result, float(n), f"sum({n:,})", rtol=1e-2)

    print_subheader("Dot Product (Verifying Correctness)")

    for n in sizes:
        a = flow.ones(n)
        b = flow.full(n, 2.0)
        result = a.dot(b)
        expected = 2.0 * n
        all_passed &= check_close(result, expected, f"dot({n:,})", rtol=1e-2)

    print_subheader("Binary Operations on Large Arrays")

    n = 100000
    a = flow.arange(n)
    b = flow.ones(n)
    c = a + b

    # Check first and last elements
    arr = c.to_numpy()
    all_passed &= check_close(arr[0], 1.0, f"add[0]")
    all_passed &= check_close(arr[-1], float(n), f"add[-1]")

    print_subheader("Tiling Control")

    n = 10000
    a = flow.ones(n)
    b = flow.ones(n)

    # Test with tiling enabled
    flow.set_tiling_enabled(True)
    result_tiled = a.dot(b)

    # Test with tiling disabled
    flow.set_tiling_enabled(False)
    result_untiled = a.dot(b)

    # Re-enable
    flow.set_tiling_enabled(True)

    all_passed &= check_close(result_tiled, result_untiled, "tiled vs untiled results match")

    return all_passed


# =============================================================================
# Section 4: NumPy Comparison and Benchmarks
# =============================================================================

def test_numpy_comparison():
    """Compare bud flow results and performance with NumPy."""
    print_header("4. NUMPY COMPARISON & BENCHMARKS")

    all_passed = True

    print_subheader("Correctness Comparison")

    # Create matching arrays
    np.random.seed(42)
    np_arr = np.random.randn(1000).astype(np.float32)
    flow_arr = flow.flow(np_arr)

    # Sum
    np_sum = np_arr.sum()
    flow_sum = flow_arr.sum()
    all_passed &= check_close(flow_sum, np_sum, "sum vs numpy", rtol=1e-3)

    # Mean
    np_mean = np_arr.mean()
    flow_mean = flow_arr.mean()
    all_passed &= check_close(flow_mean, np_mean, "mean vs numpy", rtol=1e-3)

    # Min/Max
    np_min = float(np_arr.min())
    flow_min = flow_arr.min()
    all_passed &= check_close(flow_min, np_min, "min vs numpy", rtol=1e-3)

    np_max = float(np_arr.max())
    flow_max = flow_arr.max()
    all_passed &= check_close(flow_max, np_max, "max vs numpy", rtol=1e-3)

    # Dot product
    np_arr2 = np.random.randn(1000).astype(np.float32)
    flow_arr2 = flow.flow(np_arr2)

    np_dot = float(np.dot(np_arr, np_arr2))
    flow_dot = flow_arr.dot(flow_arr2)
    all_passed &= check_close(flow_dot, np_dot, "dot vs numpy", rtol=1e-2)

    # Element-wise operations
    np_result = np_arr + np_arr2
    flow_result = flow_arr + flow_arr2
    all_passed &= check_close(flow_result.to_numpy(), np_result, "add vs numpy", rtol=1e-5)

    np_result = np_arr * np_arr2
    flow_result = flow_arr * flow_arr2
    all_passed &= check_close(flow_result.to_numpy(), np_result, "mul vs numpy", rtol=1e-5)

    print_subheader("Performance Benchmarks")

    sizes = [10000, 100000, 1000000]

    for n in sizes:
        print(f"\n  Array size: {n:,} elements")

        # Create arrays
        np_a = np.ones(n, dtype=np.float32)
        np_b = np.ones(n, dtype=np.float32)
        flow_a = flow.ones(n)
        flow_b = flow.ones(n)

        # Benchmark dot product
        np_time = time_operation("numpy dot", lambda: np.dot(np_a, np_b))
        flow_time = time_operation("flow dot", lambda: flow_a.dot(flow_b))
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Dot Product: NumPy={np_time:.3f}ms, Flow={flow_time:.3f}ms, Speedup={speedup:.2f}x")

        # Benchmark sum
        np_time = time_operation("numpy sum", lambda: np_a.sum())
        flow_time = time_operation("flow sum", lambda: flow_a.sum())
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Sum:         NumPy={np_time:.3f}ms, Flow={flow_time:.3f}ms, Speedup={speedup:.2f}x")

        # Benchmark add
        np_time = time_operation("numpy add", lambda: np_a + np_b)
        flow_time = time_operation("flow add", lambda: flow_a + flow_b)
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Add:         NumPy={np_time:.3f}ms, Flow={flow_time:.3f}ms, Speedup={speedup:.2f}x")

        # Benchmark multiply
        np_time = time_operation("numpy mul", lambda: np_a * np_b)
        flow_time = time_operation("flow mul", lambda: flow_a * flow_b)
        speedup = np_time / flow_time if flow_time > 0 else 0
        print(f"    Multiply:    NumPy={np_time:.3f}ms, Flow={flow_time:.3f}ms, Speedup={speedup:.2f}x")

    return all_passed


# =============================================================================
# Section 5: Chained Operations Performance
# =============================================================================

def test_chained_operations():
    """Benchmark chained operations compared to NumPy."""
    print_header("5. CHAINED OPERATIONS PERFORMANCE")

    print_subheader("Complex Operation: (a + b) * (a - b)")

    sizes = [10000, 100000, 1000000]

    for n in sizes:
        print(f"\n  Array size: {n:,} elements")

        # Create arrays
        np_a = np.random.randn(n).astype(np.float32)
        np_b = np.random.randn(n).astype(np.float32)
        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)

        # Define operations
        def numpy_op():
            return (np_a + np_b) * (np_a - np_b)

        def flow_op():
            return (flow_a + flow_b) * (flow_a - flow_b)

        # Verify correctness
        np_result = numpy_op()
        flow_result = flow_op()

        if np.allclose(flow_result.to_numpy(), np_result, rtol=1e-4):
            print(f"    Correctness: PASS")
        else:
            print(f"    Correctness: FAIL")

        # Benchmark
        np_time = time_operation("numpy", numpy_op)
        flow_time = time_operation("flow", flow_op)
        speedup = np_time / flow_time if flow_time > 0 else 0

        print(f"    NumPy: {np_time:.3f}ms, Flow: {flow_time:.3f}ms, Speedup: {speedup:.2f}x")

    print_subheader("Polynomial: a*x^2 + b*x + c (manual)")

    n = 100000
    np_x = np.random.randn(n).astype(np.float32)
    np_a = np.full(n, 2.0, dtype=np.float32)
    np_b = np.full(n, 3.0, dtype=np.float32)
    np_c = np.full(n, 4.0, dtype=np.float32)

    flow_x = flow.flow(np_x)
    flow_a = flow.full(n, 2.0)
    flow_b = flow.full(n, 3.0)
    flow_c = flow.full(n, 4.0)

    def numpy_poly():
        return np_a * np_x * np_x + np_b * np_x + np_c

    def flow_poly():
        return flow_a * flow_x * flow_x + flow_b * flow_x + flow_c

    # Verify correctness
    np_result = numpy_poly()
    flow_result = flow_poly()

    if np.allclose(flow_result.to_numpy(), np_result, rtol=1e-3):
        print(f"\n  Polynomial Correctness: PASS")
    else:
        print(f"\n  Polynomial Correctness: FAIL")

    # Benchmark
    np_time = time_operation("numpy", numpy_poly)
    flow_time = time_operation("flow", flow_poly)
    speedup = np_time / flow_time if flow_time > 0 else 0

    print(f"  NumPy: {np_time:.3f}ms, Flow: {flow_time:.3f}ms, Speedup: {speedup:.2f}x")

    return True


# =============================================================================
# Section 6: Edge Cases
# =============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_header("6. EDGE CASES")

    all_passed = True

    print_subheader("Single Element")

    x = flow.ones(1)
    all_passed &= check_close(x.sum(), 1.0, "single element sum")
    all_passed &= check_close(x.min(), 1.0, "single element min")
    all_passed &= check_close(x.max(), 1.0, "single element max")

    print_subheader("Power of 2 Boundaries")

    for n in [255, 256, 257, 1023, 1024, 1025]:
        x = flow.ones(n)
        all_passed &= check_close(x.sum(), float(n), f"sum({n})")

    print_subheader("Prime Number Sizes")

    for n in [997, 10007]:
        x = flow.ones(n)
        result = x.sum()
        all_passed &= check_close(result, float(n), f"sum({n})", rtol=1e-3)

    print_subheader("Repeated Operations Stress Test")

    x = flow.ones(1000)
    for i in range(100):
        x = x + flow.ones(1000)
    all_passed &= check_close(x.sum(), 101000.0, "100 repeated additions", rtol=1e-2)

    return all_passed


# =============================================================================
# Section 7: Memory Bandwidth Analysis
# =============================================================================

def test_memory_bandwidth():
    """Analyze memory bandwidth for various operations."""
    print_header("7. MEMORY BANDWIDTH ANALYSIS")

    print_subheader("Measuring Memory Bandwidth")

    n = 10_000_000  # 10M elements = 40MB per array

    flow_a = flow.ones(n)
    flow_b = flow.ones(n)
    np_a = np.ones(n, dtype=np.float32)
    np_b = np.ones(n, dtype=np.float32)

    # Dot product: reads 2 arrays = 80MB
    flow_time = time_operation("flow dot", lambda: flow_a.dot(flow_b), iterations=5)
    np_time = time_operation("numpy dot", lambda: np.dot(np_a, np_b), iterations=5)

    bytes_read = 2 * n * 4  # 2 arrays * n elements * 4 bytes
    flow_bandwidth = bytes_read / (flow_time / 1000) / 1e9  # GB/s
    np_bandwidth = bytes_read / (np_time / 1000) / 1e9  # GB/s

    print(f"\n  Dot Product (10M elements):")
    print(f"    Flow: {flow_time:.2f}ms, Bandwidth: {flow_bandwidth:.1f} GB/s")
    print(f"    NumPy: {np_time:.2f}ms, Bandwidth: {np_bandwidth:.1f} GB/s")

    # Sum: reads 1 array = 40MB
    flow_time = time_operation("flow sum", lambda: flow_a.sum(), iterations=5)
    np_time = time_operation("numpy sum", lambda: np_a.sum(), iterations=5)

    bytes_read = n * 4
    flow_bandwidth = bytes_read / (flow_time / 1000) / 1e9
    np_bandwidth = bytes_read / (np_time / 1000) / 1e9

    print(f"\n  Sum Reduction (10M elements):")
    print(f"    Flow: {flow_time:.2f}ms, Bandwidth: {flow_bandwidth:.1f} GB/s")
    print(f"    NumPy: {np_time:.2f}ms, Bandwidth: {np_bandwidth:.1f} GB/s")

    # Add: reads 2 arrays, writes 1 = 120MB
    flow_time = time_operation("flow add", lambda: flow_a + flow_b, iterations=5)
    np_time = time_operation("numpy add", lambda: np_a + np_b, iterations=5)

    bytes_total = 3 * n * 4  # 2 reads + 1 write
    flow_bandwidth = bytes_total / (flow_time / 1000) / 1e9
    np_bandwidth = bytes_total / (np_time / 1000) / 1e9

    print(f"\n  Element-wise Add (10M elements):")
    print(f"    Flow: {flow_time:.2f}ms, Bandwidth: {flow_bandwidth:.1f} GB/s")
    print(f"    NumPy: {np_time:.2f}ms, Bandwidth: {np_bandwidth:.1f} GB/s")

    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all tests and report results."""
    print("\n")
    print("*" * 70)
    print("*  BUD FLOW LANG - SYSTEM TEST & NUMPY COMPARISON")
    print("*" * 70)

    results = {}

    test_functions = [
        ('hardware', test_hardware_info),
        ('basic_ops', test_basic_operations),
        ('large_arrays', test_large_arrays),
        ('numpy_comparison', test_numpy_comparison),
        ('chained_ops', test_chained_operations),
        ('edge_cases', test_edge_cases),
        ('memory_bandwidth', test_memory_bandwidth),
    ]

    for name, func in test_functions:
        try:
            results[name] = func()
        except Exception as e:
            print(f"ERROR in {name} test: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print_header("TEST SUMMARY")

    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {total_passed}/{total_tests} test sections passed")

    if total_passed == total_tests:
        print("\n*** ALL TESTS PASSED ***\n")
        return 0
    else:
        print("\n*** SOME TESTS FAILED ***\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
