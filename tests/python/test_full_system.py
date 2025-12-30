#!/usr/bin/env python3
"""
Comprehensive Bud Flow Lang System Test

This script tests:
1. Basic SIMD operations through Python bindings
2. JIT compilation with @flow.kernel decorator
3. Memory optimization features (tiling, prefetching)
4. Performance comparison with NumPy

Run from build directory:
    python3 ../tests/python/test_full_system.py
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
        close = abs(actual - expected) <= rtol * max(abs(actual), abs(expected)) + 1e-8
    else:
        close = np.allclose(actual, expected, rtol=rtol)

    status = "PASS" if close else "FAIL"
    print(f"  [{status}] {name}")
    if not close:
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
# Section 3: JIT Compilation with @flow.kernel
# =============================================================================

def test_jit_kernels():
    """Test JIT compilation with @flow.kernel decorator."""
    print_header("3. JIT COMPILATION (@flow.kernel)")

    all_passed = True

    print_subheader("Simple Kernel")

    @flow.kernel
    def add_one(x):
        return x + flow.ones(x.size)

    x = flow.arange(5)
    result = add_one(x)
    all_passed &= check_close(result.to_numpy(), [1, 2, 3, 4, 5], "add_one kernel")

    print_subheader("Binary Kernel")

    @flow.kernel
    def multiply_add(a, b):
        return a * b + a

    a = flow.flow([1.0, 2.0, 3.0])
    b = flow.flow([2.0, 2.0, 2.0])
    result = multiply_add(a, b)
    all_passed &= check_close(result.to_numpy(), [3, 6, 9], "multiply_add kernel")

    print_subheader("Chained Operations Kernel")

    @flow.kernel
    def complex_chain(x):
        # (x + 1) * 2 - x = x + 2
        return (x + flow.ones(x.size)) * flow.full(x.size, 2.0) - x

    x = flow.flow([1.0, 2.0, 3.0, 4.0])
    result = complex_chain(x)
    all_passed &= check_close(result.to_numpy(), [3, 4, 5, 6], "complex_chain kernel")

    print_subheader("Kernel with Options")

    @flow.kernel(opt_level=3, enable_fusion=True)
    def optimized_kernel(a, b):
        return (a + b) * (a - b)  # a^2 - b^2

    a = flow.flow([5.0, 6.0, 7.0])
    b = flow.flow([3.0, 4.0, 5.0])
    result = optimized_kernel(a, b)
    expected = [16, 20, 24]  # 25-9, 36-16, 49-25
    all_passed &= check_close(result.to_numpy(), expected, "optimized_kernel")

    print_subheader("Kernel Caching")

    @flow.kernel
    def cached_kernel(x):
        return x * x

    # First call - should trace and compile
    x = flow.arange(10)
    result1 = cached_kernel(x)
    cache_size_1 = cached_kernel.cache_size

    # Second call with same signature - should use cache
    result2 = cached_kernel(x)
    cache_size_2 = cached_kernel.cache_size

    all_passed &= check_close(result1.to_numpy(), result2.to_numpy(), "cache consistency")
    print(f"  [INFO] Cache size after 2 calls: {cache_size_2}")

    # Clear cache
    cached_kernel.clear_cache()
    print(f"  [INFO] Cache cleared, size: {cached_kernel.cache_size}")

    return all_passed


# =============================================================================
# Section 4: Large Array Operations (Tests Tiling/Prefetch)
# =============================================================================

def test_large_arrays():
    """Test operations on large arrays that benefit from tiling."""
    print_header("4. LARGE ARRAY OPERATIONS (Tiling/Prefetch)")

    all_passed = True

    sizes = [1000, 10000, 100000, 1000000]

    print_subheader("Sum Reduction (Verifying Correctness)")

    for n in sizes:
        x = flow.ones(n)
        result = x.sum()
        all_passed &= check_close(result, float(n), f"sum({n})", rtol=1e-2)

    print_subheader("Dot Product (Verifying Correctness)")

    for n in sizes:
        a = flow.ones(n)
        b = flow.full(n, 2.0)
        result = a.dot(b)
        expected = 2.0 * n
        all_passed &= check_close(result, expected, f"dot({n})", rtol=1e-2)

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
# Section 5: NumPy Comparison and Benchmarks
# =============================================================================

def test_numpy_comparison():
    """Compare bud flow results and performance with NumPy."""
    print_header("5. NUMPY COMPARISON & BENCHMARKS")

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
    np_min = np_arr.min()
    flow_min = flow_arr.min()
    all_passed &= check_close(flow_min, np_min, "min vs numpy", rtol=1e-3)

    np_max = np_arr.max()
    flow_max = flow_arr.max()
    all_passed &= check_close(flow_max, np_max, "max vs numpy", rtol=1e-3)

    # Dot product
    np_arr2 = np.random.randn(1000).astype(np.float32)
    flow_arr2 = flow.flow(np_arr2)

    np_dot = np.dot(np_arr, np_arr2)
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
# Section 6: JIT Kernel Performance vs NumPy
# =============================================================================

def test_jit_performance():
    """Benchmark JIT-compiled kernels vs NumPy."""
    print_header("6. JIT KERNEL PERFORMANCE")

    print_subheader("Complex Operation Benchmark")

    # Define a complex operation
    @flow.kernel(opt_level=3)
    def complex_op(a, b):
        # (a + b) * (a - b) + a * b
        return (a + b) * (a - b) + a * b

    def numpy_complex_op(a, b):
        return (a + b) * (a - b) + a * b

    sizes = [10000, 100000, 1000000]

    for n in sizes:
        print(f"\n  Array size: {n:,} elements")

        # Create arrays
        np_a = np.random.randn(n).astype(np.float32)
        np_b = np.random.randn(n).astype(np.float32)
        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)

        # Verify correctness
        np_result = numpy_complex_op(np_a, np_b)
        flow_result = complex_op(flow_a, flow_b)

        if np.allclose(flow_result.to_numpy(), np_result, rtol=1e-4):
            print(f"    Correctness: PASS")
        else:
            print(f"    Correctness: FAIL")

        # Benchmark
        np_time = time_operation("numpy", lambda: numpy_complex_op(np_a, np_b))
        flow_time = time_operation("flow", lambda: complex_op(flow_a, flow_b))
        speedup = np_time / flow_time if flow_time > 0 else 0

        print(f"    NumPy: {np_time:.3f}ms, Flow JIT: {flow_time:.3f}ms, Speedup: {speedup:.2f}x")

    print_subheader("Polynomial Evaluation Benchmark")

    # Horner's method: a*x^3 + b*x^2 + c*x + d
    @flow.kernel
    def poly_eval(x, a, b, c, d):
        # ((a*x + b)*x + c)*x + d
        return ((a * x + b) * x + c) * x + d

    def numpy_poly_eval(x, a, b, c, d):
        return ((a * x + b) * x + c) * x + d

    n = 100000
    np_x = np.random.randn(n).astype(np.float32)
    np_a = np.full(n, 1.0, dtype=np.float32)
    np_b = np.full(n, 2.0, dtype=np.float32)
    np_c = np.full(n, 3.0, dtype=np.float32)
    np_d = np.full(n, 4.0, dtype=np.float32)

    flow_x = flow.flow(np_x)
    flow_a = flow.full(n, 1.0)
    flow_b = flow.full(n, 2.0)
    flow_c = flow.full(n, 3.0)
    flow_d = flow.full(n, 4.0)

    # Verify correctness
    np_result = numpy_poly_eval(np_x, np_a, np_b, np_c, np_d)
    flow_result = poly_eval(flow_x, flow_a, flow_b, flow_c, flow_d)

    if np.allclose(flow_result.to_numpy(), np_result, rtol=1e-3):
        print(f"\n  Polynomial Eval Correctness: PASS")
    else:
        print(f"\n  Polynomial Eval Correctness: FAIL")

    # Benchmark
    np_time = time_operation("numpy", lambda: numpy_poly_eval(np_x, np_a, np_b, np_c, np_d))
    flow_time = time_operation("flow", lambda: poly_eval(flow_x, flow_a, flow_b, flow_c, flow_d))
    speedup = np_time / flow_time if flow_time > 0 else 0

    print(f"  NumPy: {np_time:.3f}ms, Flow JIT: {flow_time:.3f}ms, Speedup: {speedup:.2f}x")

    return True


# =============================================================================
# Section 7: Edge Cases and Stress Tests
# =============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_header("7. EDGE CASES & STRESS TESTS")

    all_passed = True

    print_subheader("Empty and Single Element")

    x = flow.ones(1)
    all_passed &= check_close(x.sum(), 1.0, "single element sum")
    all_passed &= check_close(x.min(), 1.0, "single element min")
    all_passed &= check_close(x.max(), 1.0, "single element max")

    print_subheader("Power of 2 Boundaries")

    for n in [255, 256, 257, 1023, 1024, 1025]:
        x = flow.ones(n)
        all_passed &= check_close(x.sum(), float(n), f"sum({n})")

    print_subheader("Prime Number Sizes")

    for n in [997, 10007, 100003]:
        x = flow.ones(n)
        result = x.sum()
        all_passed &= check_close(result, float(n), f"sum({n})", rtol=1e-3)

    print_subheader("Special Float Values")

    # Large values
    x = flow.full(100, 1e30)
    y = flow.full(100, 1e-30)
    result = x * y
    all_passed &= check_close(result.to_numpy()[0], 1.0, "large * small")

    # Very small values
    x = flow.full(100, 1e-20)
    result = x.sum()
    all_passed &= check_close(result, 100 * 1e-20, "sum of tiny values", rtol=1e-3)

    print_subheader("Stress: Repeated Operations")

    x = flow.ones(1000)
    for i in range(100):
        x = x + flow.ones(1000)
    all_passed &= check_close(x.sum(), 101000.0, "100 repeated additions", rtol=1e-2)

    print_subheader("Stress: Multiple Large Arrays")

    arrays = [flow.ones(100000) for _ in range(10)]
    total = sum(arr.sum() for arr in arrays)
    all_passed &= check_close(total, 1000000.0, "sum of 10 large arrays", rtol=1e-2)

    return all_passed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all tests and report results."""
    print("\n")
    print("*" * 70)
    print("*  BUD FLOW LANG - COMPREHENSIVE SYSTEM TEST")
    print("*" * 70)

    results = {}

    try:
        results['hardware'] = test_hardware_info()
    except Exception as e:
        print(f"ERROR in hardware test: {e}")
        results['hardware'] = False

    try:
        results['basic_ops'] = test_basic_operations()
    except Exception as e:
        print(f"ERROR in basic operations test: {e}")
        import traceback
        traceback.print_exc()
        results['basic_ops'] = False

    try:
        results['jit_kernels'] = test_jit_kernels()
    except Exception as e:
        print(f"ERROR in JIT kernels test: {e}")
        import traceback
        traceback.print_exc()
        results['jit_kernels'] = False

    try:
        results['large_arrays'] = test_large_arrays()
    except Exception as e:
        print(f"ERROR in large arrays test: {e}")
        import traceback
        traceback.print_exc()
        results['large_arrays'] = False

    try:
        results['numpy_comparison'] = test_numpy_comparison()
    except Exception as e:
        print(f"ERROR in numpy comparison test: {e}")
        import traceback
        traceback.print_exc()
        results['numpy_comparison'] = False

    try:
        results['jit_performance'] = test_jit_performance()
    except Exception as e:
        print(f"ERROR in JIT performance test: {e}")
        import traceback
        traceback.print_exc()
        results['jit_performance'] = False

    try:
        results['edge_cases'] = test_edge_cases()
    except Exception as e:
        print(f"ERROR in edge cases test: {e}")
        import traceback
        traceback.print_exc()
        results['edge_cases'] = False

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
