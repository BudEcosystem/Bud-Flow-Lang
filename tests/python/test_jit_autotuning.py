#!/usr/bin/env python3
"""
=============================================================================
JIT Kernel Auto-Tuning Test Suite
=============================================================================

This script tests the JIT compilation and auto-tuning system by:
1. Creating various kernel types (simple, complex, math-heavy)
2. Running them repeatedly with different input sizes
3. Verifying correctness against NumPy
4. Monitoring cache behavior and tuning metrics

Usage:
    python test_jit_autotuning.py [--iterations N] [--verbose]
"""

import sys
import os
import time
import argparse
import numpy as np

# Add build directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'build')
sys.path.insert(0, build_dir)

try:
    import bud_flow_lang_py as flow
except ImportError as e:
    print(f"Error: Could not import bud_flow_lang_py: {e}")
    print("Make sure you've built the project and are running from the build directory")
    sys.exit(1)


# =============================================================================
# Helper Functions
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{Colors.RESET}")


def print_pass(msg):
    """Print a pass message."""
    print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {msg}")


def print_fail(msg):
    """Print a fail message."""
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}")


def print_info(msg):
    """Print an info message."""
    print(f"  {Colors.BLUE}[INFO]{Colors.RESET} {msg}")


def check_close(actual, expected, name, rtol=1e-4, atol=1e-6):
    """Check if values are close and report result."""
    if isinstance(actual, float) and isinstance(expected, float):
        close = abs(actual - expected) <= rtol * max(abs(actual), abs(expected), 1.0) + atol
    else:
        # Use both rtol and atol for robust comparison
        # atol handles cases where expected values are very small
        close = np.allclose(actual, expected, rtol=rtol, atol=atol)

    if close:
        print_pass(name)
    else:
        print_fail(f"{name}: expected {expected}, got {actual}")
    return close


# =============================================================================
# Test Kernels
# =============================================================================

@flow.kernel
def simple_add(x, y):
    """Simple element-wise addition."""
    return x + y


@flow.kernel
def simple_mul(x, y):
    """Simple element-wise multiplication."""
    return x * y


@flow.kernel
def add_one(x):
    """Add 1 to each element."""
    return x + flow.ones(x.size)


@flow.kernel(opt_level=3, enable_fusion=True)
def fused_muladd(a, b, c):
    """FMA pattern: a*b + c - should be fused."""
    return a * b + c


@flow.kernel(opt_level=3, enable_fusion=True)
def complex_expression(a, b):
    """Complex expression with fusion opportunities: (a+b)*(a-b) = a^2 - b^2."""
    return (a + b) * (a - b)


@flow.kernel
def polynomial_eval(x, a, b, c, d):
    """Polynomial evaluation using Horner's method: ax^3 + bx^2 + cx + d."""
    return ((a * x + b) * x + c) * x + d


@flow.kernel
def chain_ops(x):
    """Chain of operations: ((x + 1) * 2 - 0.5) / 1.5."""
    return ((x + flow.ones(x.size)) * flow.full(x.size, 2.0) - flow.full(x.size, 0.5)) / flow.full(x.size, 1.5)


@flow.kernel
def transcendental_ops(x):
    """Transcendental operations: exp(log(x+1))."""
    return (x + flow.ones(x.size)).log().exp()


@flow.kernel
def trig_ops(x):
    """Trigonometric operations: sin(x)^2 + cos(x)^2 should = 1."""
    s = x.sin()
    c = x.cos()
    return s * s + c * c


@flow.kernel
def nested_fma(a, b, c, d):
    """Nested FMA: a*b + c*d."""
    return a * b + c * d


# =============================================================================
# Test Cases
# =============================================================================

def test_simple_kernels(iterations=10, verbose=False):
    """Test simple arithmetic kernels."""
    print_header("Simple Arithmetic Kernels")

    all_passed = True

    sizes = [100, 1000, 10000, 100000]

    for size in sizes:
        print_info(f"Testing with size={size:,}")

        # Create test data
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)

        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)

        # Test simple_add
        for i in range(iterations):
            result = simple_add(flow_a, flow_b)
        expected = np_a + np_b
        all_passed &= check_close(result.to_numpy(), expected, f"simple_add ({size})")

        # Test simple_mul
        for i in range(iterations):
            result = simple_mul(flow_a, flow_b)
        expected = np_a * np_b
        all_passed &= check_close(result.to_numpy(), expected, f"simple_mul ({size})")

        # Test add_one
        for i in range(iterations):
            result = add_one(flow_a)
        expected = np_a + 1.0
        all_passed &= check_close(result.to_numpy(), expected, f"add_one ({size})")

        if verbose:
            print_info(f"  Cache size for simple_add: {simple_add.cache_size}")
            print_info(f"  Cache size for simple_mul: {simple_mul.cache_size}")
            print_info(f"  Cache size for add_one: {add_one.cache_size}")

    return all_passed


def test_fused_operations(iterations=10, verbose=False):
    """Test operations that should be fused by the optimizer."""
    print_header("Fused Operations (FMA Pattern)")

    all_passed = True

    sizes = [1000, 10000, 100000]

    for size in sizes:
        print_info(f"Testing with size={size:,}")

        # Create test data with controlled seed for reproducibility
        np.random.seed(42 + size)
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)
        np_c = np.random.randn(size).astype(np.float32)

        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)
        flow_c = flow.flow(np_c)

        # Warmup and run
        for i in range(iterations):
            result = fused_muladd(flow_a, flow_b, flow_c)

        expected = np_a * np_b + np_c
        # Use slightly larger tolerance for floating point accumulation
        all_passed &= check_close(result.to_numpy(), expected, f"fused_muladd ({size})", rtol=1e-4)

        # Complex expression: (a+b)*(a-b) = a^2 - b^2
        for i in range(iterations):
            result = complex_expression(flow_a, flow_b)

        expected = (np_a + np_b) * (np_a - np_b)
        all_passed &= check_close(result.to_numpy(), expected, f"complex_expression ({size})", rtol=1e-4)

        if verbose:
            print_info(f"  fused_muladd cache size: {fused_muladd.cache_size}")
            print_info(f"  complex_expression cache size: {complex_expression.cache_size}")

    return all_passed


def test_polynomial_evaluation(iterations=10, verbose=False):
    """Test polynomial evaluation kernel."""
    print_header("Polynomial Evaluation")

    all_passed = True

    sizes = [1000, 10000, 100000]

    for size in sizes:
        print_info(f"Testing with size={size:,}")

        np_x = np.random.randn(size).astype(np.float32) * 0.5  # Smaller range for stability
        np_a = np.full(size, 1.0, dtype=np.float32)
        np_b = np.full(size, 2.0, dtype=np.float32)
        np_c = np.full(size, 3.0, dtype=np.float32)
        np_d = np.full(size, 4.0, dtype=np.float32)

        flow_x = flow.flow(np_x)
        flow_a = flow.full(size, 1.0)
        flow_b = flow.full(size, 2.0)
        flow_c = flow.full(size, 3.0)
        flow_d = flow.full(size, 4.0)

        # Run multiple times
        for i in range(iterations):
            result = polynomial_eval(flow_x, flow_a, flow_b, flow_c, flow_d)

        # Horner's method: ((a*x + b)*x + c)*x + d
        expected = ((np_a * np_x + np_b) * np_x + np_c) * np_x + np_d
        all_passed &= check_close(result.to_numpy(), expected, f"polynomial_eval ({size})", rtol=1e-3)

        if verbose:
            print_info(f"  polynomial_eval cache size: {polynomial_eval.cache_size}")

    return all_passed


def test_chain_operations(iterations=10, verbose=False):
    """Test chained arithmetic operations."""
    print_header("Chained Operations")

    all_passed = True

    sizes = [1000, 10000, 100000]

    for size in sizes:
        print_info(f"Testing with size={size:,}")

        np_x = np.random.randn(size).astype(np.float32)
        flow_x = flow.flow(np_x)

        # Run multiple times
        for i in range(iterations):
            result = chain_ops(flow_x)

        # ((x + 1) * 2 - 0.5) / 1.5
        expected = ((np_x + 1.0) * 2.0 - 0.5) / 1.5
        all_passed &= check_close(result.to_numpy(), expected, f"chain_ops ({size})", rtol=1e-3)

        if verbose:
            print_info(f"  chain_ops cache size: {chain_ops.cache_size}")

    return all_passed


def test_transcendental_operations(iterations=5, verbose=False):
    """Test transcendental operations (exp, log, sin, cos)."""
    print_header("Transcendental Operations")

    all_passed = True

    sizes = [1000, 10000]

    for size in sizes:
        print_info(f"Testing with size={size:,}")

        # Positive values for log
        np_x = np.abs(np.random.randn(size).astype(np.float32)) + 0.1
        flow_x = flow.flow(np_x)

        # exp(log(x+1)) should approximate x+1
        for i in range(iterations):
            result = transcendental_ops(flow_x)

        expected = np.exp(np.log(np_x + 1.0))
        all_passed &= check_close(result.to_numpy(), expected, f"transcendental_ops ({size})", rtol=1e-3)

        # sin^2 + cos^2 should = 1
        np_angles = np.random.randn(size).astype(np.float32) * np.pi
        flow_angles = flow.flow(np_angles)

        for i in range(iterations):
            result = trig_ops(flow_angles)

        expected = np.ones(size, dtype=np.float32)
        all_passed &= check_close(result.to_numpy(), expected, f"trig_identity ({size})", rtol=1e-3)

        if verbose:
            print_info(f"  transcendental_ops cache size: {transcendental_ops.cache_size}")
            print_info(f"  trig_ops cache size: {trig_ops.cache_size}")

    return all_passed


def test_cache_behavior(verbose=False):
    """Test kernel cache behavior - cache hits and signature-based caching."""
    print_header("Kernel Cache Behavior")

    all_passed = True

    # Clear cache to start fresh
    simple_add.clear_cache()
    print_info("Cache cleared")

    # First call with size 100 - should trace and compile
    np_a = np.ones(100, dtype=np.float32)
    np_b = np.ones(100, dtype=np.float32)
    flow_a = flow.flow(np_a)
    flow_b = flow.flow(np_b)

    result1 = simple_add(flow_a, flow_b)
    cache_size_after_first = simple_add.cache_size
    print_info(f"After first call (size=100): cache size = {cache_size_after_first}")
    all_passed &= (cache_size_after_first == 1)

    # Second call with same signature - should use cache
    result2 = simple_add(flow_a, flow_b)
    cache_size_after_second = simple_add.cache_size
    print_info(f"After second call (same size): cache size = {cache_size_after_second}")
    all_passed &= (cache_size_after_second == 1)  # No new entries
    all_passed &= check_close(result1.to_numpy(), result2.to_numpy(), "Cache consistency")

    # Third call with different size - should trace again
    np_a_large = np.ones(1000, dtype=np.float32)
    np_b_large = np.ones(1000, dtype=np.float32)
    flow_a_large = flow.flow(np_a_large)
    flow_b_large = flow.flow(np_b_large)

    result3 = simple_add(flow_a_large, flow_b_large)
    cache_size_after_third = simple_add.cache_size
    print_info(f"After third call (size=1000): cache size = {cache_size_after_third}")
    all_passed &= (cache_size_after_third == 2)  # New entry for different signature

    # Back to original size - should hit cache
    result4 = simple_add(flow_a, flow_b)
    cache_size_after_fourth = simple_add.cache_size
    print_info(f"After fourth call (size=100 again): cache size = {cache_size_after_fourth}")
    all_passed &= (cache_size_after_fourth == 2)  # No new entries

    if all_passed:
        print_pass("Cache behavior is correct - signature-based caching works")
    else:
        print_fail("Cache behavior issue detected")

    return all_passed


def test_tuning_performance(iterations=50, verbose=False):
    """Test performance tuning by running kernels many times."""
    print_header("Performance Tuning Test")

    sizes = [10000, 100000, 1000000]

    print_info(f"Running each kernel {iterations} times per size to warm up JIT")

    for size in sizes:
        print(f"\n  {Colors.BOLD}Size: {size:,}{Colors.RESET}")

        # Create test data
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)
        np_c = np.random.randn(size).astype(np.float32)

        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)
        flow_c = flow.flow(np_c)

        # Time simple_add
        start = time.perf_counter()
        for i in range(iterations):
            result = simple_add(flow_a, flow_b)
        flow_time = (time.perf_counter() - start) / iterations * 1000

        start = time.perf_counter()
        for i in range(iterations):
            np_result = np_a + np_b
        numpy_time = (time.perf_counter() - start) / iterations * 1000

        speedup = numpy_time / flow_time if flow_time > 0 else 0
        print(f"    simple_add:     Flow={flow_time:.4f}ms, NumPy={numpy_time:.4f}ms, Speedup={speedup:.2f}x")

        # Time fused_muladd
        start = time.perf_counter()
        for i in range(iterations):
            result = fused_muladd(flow_a, flow_b, flow_c)
        flow_time = (time.perf_counter() - start) / iterations * 1000

        start = time.perf_counter()
        for i in range(iterations):
            np_result = np_a * np_b + np_c
        numpy_time = (time.perf_counter() - start) / iterations * 1000

        speedup = numpy_time / flow_time if flow_time > 0 else 0
        print(f"    fused_muladd:   Flow={flow_time:.4f}ms, NumPy={numpy_time:.4f}ms, Speedup={speedup:.2f}x")

        # Time complex_expression
        start = time.perf_counter()
        for i in range(iterations):
            result = complex_expression(flow_a, flow_b)
        flow_time = (time.perf_counter() - start) / iterations * 1000

        start = time.perf_counter()
        for i in range(iterations):
            np_result = (np_a + np_b) * (np_a - np_b)
        numpy_time = (time.perf_counter() - start) / iterations * 1000

        speedup = numpy_time / flow_time if flow_time > 0 else 0
        print(f"    complex_expr:   Flow={flow_time:.4f}ms, NumPy={numpy_time:.4f}ms, Speedup={speedup:.2f}x")

    return True


def test_multisize_tuning(verbose=False):
    """Test that different sizes get tuned separately."""
    print_header("Multi-Size Tuning")

    all_passed = True

    # Clear cache
    simple_add.clear_cache()
    fused_muladd.clear_cache()

    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    print_info("Running kernels with varying sizes to build tuning profiles...")

    for size in sizes:
        np_a = np.ones(size, dtype=np.float32)
        np_b = np.ones(size, dtype=np.float32)
        np_c = np.ones(size, dtype=np.float32)

        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)
        flow_c = flow.flow(np_c)

        # Run multiple times per size
        for _ in range(5):
            result1 = simple_add(flow_a, flow_b)
            result2 = fused_muladd(flow_a, flow_b, flow_c)

        # Verify correctness
        expected1 = np_a + np_b
        expected2 = np_a * np_b + np_c

        all_passed &= check_close(result1.to_numpy(), expected1, f"simple_add (size={size})")
        all_passed &= check_close(result2.to_numpy(), expected2, f"fused_muladd (size={size})")

    print_info(f"simple_add cache entries: {simple_add.cache_size}")
    print_info(f"fused_muladd cache entries: {fused_muladd.cache_size}")

    # Each size should have its own cache entry
    all_passed &= (simple_add.cache_size == len(sizes))
    all_passed &= (fused_muladd.cache_size == len(sizes))

    if all_passed:
        print_pass("Multi-size tuning working correctly")
    else:
        print_fail("Multi-size tuning issue detected")

    return all_passed


def test_long_running(duration_seconds=30, verbose=False):
    """Long-running test to verify stability."""
    print_header(f"Long-Running Stability Test ({duration_seconds}s)")

    start_time = time.time()
    iteration = 0
    errors = 0

    sizes = [1000, 10000, 100000]

    print_info("Running continuous kernel executions...")

    while time.time() - start_time < duration_seconds:
        size = sizes[iteration % len(sizes)]

        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)

        flow_a = flow.flow(np_a)
        flow_b = flow.flow(np_b)

        # Test simple_add
        result = simple_add(flow_a, flow_b)
        expected = np_a + np_b
        if not np.allclose(result.to_numpy(), expected, rtol=1e-4):
            errors += 1

        # Test complex_expression
        result = complex_expression(flow_a, flow_b)
        expected = (np_a + np_b) * (np_a - np_b)
        if not np.allclose(result.to_numpy(), expected, rtol=1e-3):
            errors += 1

        iteration += 1

        # Progress update every 5 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            progress = elapsed / duration_seconds * 100
            print(f"\r  Progress: {progress:.0f}% ({iteration} iterations, {errors} errors)", end="", flush=True)

    print()  # New line after progress

    print_info(f"Completed {iteration} iterations with {errors} errors")

    if errors == 0:
        print_pass(f"Long-running test passed ({iteration} iterations)")
        return True
    else:
        print_fail(f"Long-running test failed ({errors} errors)")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='JIT Kernel Auto-Tuning Test Suite')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations per kernel test')
    parser.add_argument('--long-duration', type=int, default=30,
                        help='Duration of long-running test in seconds')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--skip-long', action='store_true',
                        help='Skip the long-running test')

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("*" * 70)
    print("*  BUD FLOW LANG - JIT KERNEL AUTO-TUNING TEST SUITE")
    print("*" * 70)
    print(Colors.RESET)

    # Get hardware info
    hw = flow.get_hardware_info()
    print_info(f"CPU: {hw.get('cpu_name', 'Unknown')}")
    print_info(f"SIMD Width: {hw.get('simd_width', 'N/A')} bytes")
    print_info(f"Cores: {hw.get('physical_cores', 'N/A')} physical, {hw.get('logical_cores', 'N/A')} logical")

    results = {}

    # Run tests
    tests = [
        ("Simple Kernels", lambda: test_simple_kernels(args.iterations, args.verbose)),
        ("Fused Operations", lambda: test_fused_operations(args.iterations, args.verbose)),
        ("Polynomial Eval", lambda: test_polynomial_evaluation(args.iterations, args.verbose)),
        ("Chain Operations", lambda: test_chain_operations(args.iterations, args.verbose)),
        ("Transcendental Ops", lambda: test_transcendental_operations(args.iterations // 2, args.verbose)),
        ("Cache Behavior", lambda: test_cache_behavior(args.verbose)),
        ("Multi-Size Tuning", lambda: test_multisize_tuning(args.verbose)),
        ("Performance Tuning", lambda: test_tuning_performance(args.iterations * 5, args.verbose)),
    ]

    if not args.skip_long:
        tests.append(("Long-Running", lambda: test_long_running(args.long_duration, args.verbose)))

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"{Colors.RED}ERROR in {name}: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print_header("TEST SUMMARY")

    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)

    for name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {total_passed}/{total_tests} test sections passed")

    # Final cache stats
    print(f"\n  {Colors.BOLD}Final Cache Statistics:{Colors.RESET}")
    print(f"    simple_add: {simple_add.cache_size} entries")
    print(f"    simple_mul: {simple_mul.cache_size} entries")
    print(f"    fused_muladd: {fused_muladd.cache_size} entries")
    print(f"    complex_expression: {complex_expression.cache_size} entries")
    print(f"    polynomial_eval: {polynomial_eval.cache_size} entries")

    if total_passed == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}*** ALL TESTS PASSED ***{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}*** SOME TESTS FAILED ***{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
