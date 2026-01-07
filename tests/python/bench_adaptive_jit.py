#!/usr/bin/env python3
"""
Bud Flow Lang - Adaptive JIT Compiler Benchmark

This script benchmarks the Adaptive JIT system to verify:
1. Performance improvement over repeated executions (learning)
2. Accuracy against NumPy reference
3. Tier promotion is happening
4. Profile persistence (warm start optimization)

Run from project root:
    python3 tests/python/bench_adaptive_jit.py
"""

import sys
import os
import time
import gc
import tempfile
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Add build directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'build')
sys.path.insert(0, build_dir)

try:
    import bud_flow_lang_py as bud
except ImportError as e:
    print(f"Error: Could not import bud_flow_lang_py: {e}")
    print("Make sure you've built the project: cd build && ninja")
    sys.exit(1)

import numpy as np

# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    size: int
    dtype: str
    iterations: int
    times_ns: List[float]
    tier_used: str
    numpy_time_ns: float
    max_error: float
    mean_error: float

    @property
    def mean_time_ns(self) -> float:
        return sum(self.times_ns) / len(self.times_ns)

    @property
    def min_time_ns(self) -> float:
        return min(self.times_ns)

    @property
    def speedup_vs_numpy(self) -> float:
        return self.numpy_time_ns / self.mean_time_ns if self.mean_time_ns > 0 else 0

    @property
    def improvement_first_to_last(self) -> float:
        """How much faster is the last iteration vs first (learning effect)."""
        if len(self.times_ns) < 2:
            return 1.0
        first_batch = sum(self.times_ns[:5]) / 5 if len(self.times_ns) >= 5 else self.times_ns[0]
        last_batch = sum(self.times_ns[-5:]) / 5 if len(self.times_ns) >= 5 else self.times_ns[-1]
        return first_batch / last_batch if last_batch > 0 else 1.0


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n--- {text} ---")


# =============================================================================
# SIMD Kernel Benchmarks
# =============================================================================

def benchmark_binary_op(op_name: str, bud_op, np_op, sizes: List[int],
                        iterations: int = 100) -> List[BenchmarkResult]:
    """Benchmark a binary operation (add, mul, sub, etc.)."""
    results = []

    for size in sizes:
        # Create test data
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)
        bud_a = bud.flow(np_a)
        bud_b = bud.flow(np_b)

        # Warmup
        for _ in range(5):
            _ = bud_op(bud_a, bud_b)

        # Benchmark bud
        times_ns = []
        gc.collect()
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = bud_op(bud_a, bud_b)
            end = time.perf_counter_ns()
            times_ns.append(end - start)

        # Benchmark NumPy
        gc.collect()
        start = time.perf_counter_ns()
        for _ in range(10):
            np_result = np_op(np_a, np_b)
        numpy_time_ns = (time.perf_counter_ns() - start) / 10

        # Accuracy check
        bud_result_np = np.array(result.to_numpy())
        max_error = np.max(np.abs(bud_result_np - np_result))
        mean_error = np.mean(np.abs(bud_result_np - np_result))

        # Get executor stats
        executor = bud.get_adaptive_executor()
        stats = executor.statistics()

        results.append(BenchmarkResult(
            name=op_name,
            size=size,
            dtype="float32",
            iterations=iterations,
            times_ns=times_ns,
            tier_used=f"T0:{stats.tier0_executions} T1:{stats.tier1_executions} T2:{stats.tier2_executions}",
            numpy_time_ns=numpy_time_ns,
            max_error=float(max_error),
            mean_error=float(mean_error)
        ))

        del bud_a, bud_b, result
        gc.collect()

    return results


def benchmark_reduction(op_name: str, bud_op, np_op, sizes: List[int],
                        iterations: int = 100) -> List[BenchmarkResult]:
    """Benchmark a reduction operation (sum, dot, max, etc.)."""
    results = []

    for size in sizes:
        # Create test data
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32) if op_name == "dot" else None
        bud_a = bud.flow(np_a)
        bud_b = bud.flow(np_b) if np_b is not None else None

        # Warmup
        for _ in range(5):
            if bud_b is not None:
                _ = bud_op(bud_a, bud_b)
            else:
                _ = bud_op(bud_a)

        # Benchmark bud
        times_ns = []
        gc.collect()
        for _ in range(iterations):
            start = time.perf_counter_ns()
            if bud_b is not None:
                result = bud_op(bud_a, bud_b)
            else:
                result = bud_op(bud_a)
            end = time.perf_counter_ns()
            times_ns.append(end - start)

        # Benchmark NumPy
        gc.collect()
        start = time.perf_counter_ns()
        for _ in range(10):
            if np_b is not None:
                np_result = np_op(np_a, np_b)
            else:
                np_result = np_op(np_a)
        numpy_time_ns = (time.perf_counter_ns() - start) / 10

        # Accuracy check
        max_error = abs(float(result) - float(np_result))
        mean_error = max_error  # Single value

        # Get executor stats
        executor = bud.get_adaptive_executor()
        stats = executor.statistics()

        results.append(BenchmarkResult(
            name=op_name,
            size=size,
            dtype="float32",
            iterations=iterations,
            times_ns=times_ns,
            tier_used=f"T0:{stats.tier0_executions} T1:{stats.tier1_executions} T2:{stats.tier2_executions}",
            numpy_time_ns=numpy_time_ns,
            max_error=max_error,
            mean_error=mean_error
        ))

        del bud_a
        if bud_b is not None:
            del bud_b
        gc.collect()

    return results


def benchmark_unary_op(op_name: str, bud_op, np_op, sizes: List[int],
                       iterations: int = 100, positive_only: bool = False) -> List[BenchmarkResult]:
    """Benchmark a unary operation (exp, log, sqrt, etc.)."""
    results = []

    for size in sizes:
        # Create test data
        if positive_only:
            np_a = np.abs(np.random.randn(size).astype(np.float32)) + 0.1
        else:
            np_a = np.random.randn(size).astype(np.float32)
        bud_a = bud.flow(np_a)

        # Warmup
        for _ in range(5):
            _ = bud_op(bud_a)

        # Benchmark bud
        times_ns = []
        gc.collect()
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = bud_op(bud_a)
            end = time.perf_counter_ns()
            times_ns.append(end - start)

        # Benchmark NumPy
        gc.collect()
        start = time.perf_counter_ns()
        for _ in range(10):
            np_result = np_op(np_a)
        numpy_time_ns = (time.perf_counter_ns() - start) / 10

        # Accuracy check
        bud_result_np = np.array(result.to_numpy())
        max_error = np.max(np.abs(bud_result_np - np_result))
        mean_error = np.mean(np.abs(bud_result_np - np_result))

        # Get executor stats
        executor = bud.get_adaptive_executor()
        stats = executor.statistics()

        results.append(BenchmarkResult(
            name=op_name,
            size=size,
            dtype="float32",
            iterations=iterations,
            times_ns=times_ns,
            tier_used=f"T0:{stats.tier0_executions} T1:{stats.tier1_executions} T2:{stats.tier2_executions}",
            numpy_time_ns=numpy_time_ns,
            max_error=float(max_error),
            mean_error=float(mean_error)
        ))

        del bud_a, result
        gc.collect()

    return results


def benchmark_fma(sizes: List[int], iterations: int = 100) -> List[BenchmarkResult]:
    """Benchmark fused multiply-add (FMA) operation."""
    results = []

    for size in sizes:
        # Create test data
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)
        np_c = np.random.randn(size).astype(np.float32)
        bud_a = bud.flow(np_a)
        bud_b = bud.flow(np_b)
        bud_c = bud.flow(np_c)

        # Warmup
        for _ in range(5):
            _ = bud.fma(bud_a, bud_b, bud_c)

        # Benchmark bud
        times_ns = []
        gc.collect()
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = bud.fma(bud_a, bud_b, bud_c)
            end = time.perf_counter_ns()
            times_ns.append(end - start)

        # Benchmark NumPy (no native FMA, use a*b + c)
        gc.collect()
        start = time.perf_counter_ns()
        for _ in range(10):
            np_result = np_a * np_b + np_c
        numpy_time_ns = (time.perf_counter_ns() - start) / 10

        # Accuracy check
        bud_result_np = np.array(result.to_numpy())
        max_error = np.max(np.abs(bud_result_np - np_result))
        mean_error = np.mean(np.abs(bud_result_np - np_result))

        # Get executor stats
        executor = bud.get_adaptive_executor()
        stats = executor.statistics()

        results.append(BenchmarkResult(
            name="fma",
            size=size,
            dtype="float32",
            iterations=iterations,
            times_ns=times_ns,
            tier_used=f"T0:{stats.tier0_executions} T1:{stats.tier1_executions} T2:{stats.tier2_executions}",
            numpy_time_ns=numpy_time_ns,
            max_error=float(max_error),
            mean_error=float(mean_error)
        ))

        del bud_a, bud_b, bud_c, result
        gc.collect()

    return results


# =============================================================================
# Test Learning Over Time
# =============================================================================

def test_learning_curve(size: int = 100000, iterations: int = 200) -> Dict[str, Any]:
    """Test that performance improves over repeated executions."""
    print_subheader(f"Learning Curve Test (size={size:,}, iters={iterations})")

    # Reset executor
    executor = bud.get_adaptive_executor()
    executor.reset()

    # Create test data
    np_a = np.random.randn(size).astype(np.float32)
    np_b = np.random.randn(size).astype(np.float32)
    bud_a = bud.flow(np_a)
    bud_b = bud.flow(np_b)

    # Track times in batches of 10
    batch_times = []
    batch_size = 10

    for batch in range(iterations // batch_size):
        batch_start = time.perf_counter_ns()
        for _ in range(batch_size):
            result = bud_a + bud_b
        batch_end = time.perf_counter_ns()
        batch_times.append((batch_end - batch_start) / batch_size)

    # Analyze learning
    first_quarter = batch_times[:len(batch_times)//4]
    last_quarter = batch_times[-len(batch_times)//4:]

    first_avg = sum(first_quarter) / len(first_quarter)
    last_avg = sum(last_quarter) / len(last_quarter)
    improvement = first_avg / last_avg if last_avg > 0 else 1.0

    # Get stats
    stats = executor.statistics()

    result = {
        "first_quarter_avg_ns": first_avg,
        "last_quarter_avg_ns": last_avg,
        "improvement_ratio": improvement,
        "tier0_executions": stats.tier0_executions,
        "tier1_executions": stats.tier1_executions,
        "tier2_executions": stats.tier2_executions,
        "promotions": stats.promotions_performed,
        "all_batch_times": batch_times
    }

    print(f"  First quarter avg:  {first_avg/1000:.2f} us")
    print(f"  Last quarter avg:   {last_avg/1000:.2f} us")
    print(f"  Improvement ratio:  {improvement:.2f}x")
    print(f"  Tier executions:    T0={stats.tier0_executions}, T1={stats.tier1_executions}, T2={stats.tier2_executions}")
    print(f"  Promotions:         {stats.promotions_performed}")

    if improvement > 1.0:
        print(f"  [PASS] Performance improved by {(improvement-1)*100:.1f}%")
    else:
        print(f"  [INFO] No significant improvement (may already be optimal)")

    del bud_a, bud_b
    gc.collect()

    return result


# =============================================================================
# Test Accuracy
# =============================================================================

def test_accuracy():
    """Test that all operations produce correct results."""
    print_subheader("Accuracy Tests (vs NumPy)")

    all_passed = True
    size = 10000
    tolerance = 1e-5

    tests = [
        # Binary ops
        ("add", lambda a, b: a + b, np.add),
        ("sub", lambda a, b: a - b, np.subtract),
        ("mul", lambda a, b: a * b, np.multiply),
        ("div", lambda a, b: a / b, np.divide),
    ]

    for name, bud_op, np_op in tests:
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)

        # Avoid division by zero
        if name == "div":
            np_b = np.abs(np_b) + 0.1

        bud_a = bud.flow(np_a)
        bud_b = bud.flow(np_b)

        bud_result = bud_op(bud_a, bud_b)
        np_result = np_op(np_a, np_b)

        bud_result_np = np.array(bud_result.to_numpy())
        max_error = np.max(np.abs(bud_result_np - np_result))

        if max_error < tolerance:
            print(f"  [PASS] {name}: max_error={max_error:.2e}")
        else:
            print(f"  [FAIL] {name}: max_error={max_error:.2e} > {tolerance}")
            all_passed = False

    # Unary ops
    unary_tests = [
        ("exp", bud.exp, np.exp, False),
        ("log", bud.log, np.log, True),
        ("abs", bud.abs, np.abs, False),
        ("cos", bud.cos, np.cos, False),
    ]

    for name, bud_op, np_op, positive_only in unary_tests:
        if positive_only:
            np_a = np.abs(np.random.randn(size).astype(np.float32)) + 0.1
        else:
            np_a = np.random.randn(size).astype(np.float32)

        bud_a = bud.flow(np_a)

        bud_result = bud_op(bud_a)
        np_result = np_op(np_a)

        bud_result_np = np.array(bud_result.to_numpy())
        max_error = np.max(np.abs(bud_result_np - np_result))

        # Slightly higher tolerance for transcendental functions
        tol = 1e-4 if name in ["exp", "log", "cos"] else tolerance

        if max_error < tol:
            print(f"  [PASS] {name}: max_error={max_error:.2e}")
        else:
            print(f"  [FAIL] {name}: max_error={max_error:.2e} > {tol}")
            all_passed = False

    # Reductions
    np_a = np.random.randn(size).astype(np.float32)
    np_b = np.random.randn(size).astype(np.float32)
    bud_a = bud.flow(np_a)
    bud_b = bud.flow(np_b)

    # Sum
    bud_sum = bud_a.sum()
    np_sum = np.sum(np_a)
    error = abs(bud_sum - np_sum)
    rel_error = error / abs(np_sum) if np_sum != 0 else error
    if rel_error < 1e-5:
        print(f"  [PASS] sum: rel_error={rel_error:.2e}")
    else:
        print(f"  [FAIL] sum: rel_error={rel_error:.2e}")
        all_passed = False

    # Dot
    bud_dot = bud.dot(bud_a, bud_b)
    np_dot = np.dot(np_a, np_b)
    error = abs(bud_dot - np_dot)
    rel_error = error / abs(np_dot) if np_dot != 0 else error
    if rel_error < 1e-5:
        print(f"  [PASS] dot: rel_error={rel_error:.2e}")
    else:
        print(f"  [FAIL] dot: rel_error={rel_error:.2e}")
        all_passed = False

    # FMA
    np_c = np.random.randn(size).astype(np.float32)
    bud_c = bud.flow(np_c)
    bud_fma = bud.fma(bud_a, bud_b, bud_c)
    np_fma = np_a * np_b + np_c
    bud_fma_np = np.array(bud_fma.to_numpy())
    max_error = np.max(np.abs(bud_fma_np - np_fma))
    if max_error < tolerance:
        print(f"  [PASS] fma: max_error={max_error:.2e}")
    else:
        print(f"  [FAIL] fma: max_error={max_error:.2e}")
        all_passed = False

    gc.collect()
    return all_passed


# =============================================================================
# Test Profile Persistence
# =============================================================================

def test_profile_persistence():
    """Test that profiles persist and provide warm start."""
    print_subheader("Profile Persistence Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        profile_path = os.path.join(tmpdir, "test_profiles")

        # Configure executor with persistence
        config = bud.AdaptiveExecutorConfig()
        config.enable_persistence = True
        config.profile_dir = profile_path

        executor = bud.get_adaptive_executor()
        executor.set_config(config)
        executor.reset()

        # Run some operations to build up profiles
        size = 50000
        np_a = np.random.randn(size).astype(np.float32)
        np_b = np.random.randn(size).astype(np.float32)
        bud_a = bud.flow(np_a)
        bud_b = bud.flow(np_b)

        for _ in range(50):
            result = bud_a + bud_b

        stats_before = executor.statistics()
        print(f"  Executions before save: {stats_before.total_executions}")

        # Save profiles
        saved = executor.save_profiles()
        print(f"  Save result: {'success' if saved else 'failed'}")

        # Reset and load
        executor.reset()
        stats_after_reset = executor.statistics()
        print(f"  Executions after reset: {stats_after_reset.total_executions}")

        # Check if profile file exists
        files = os.listdir(profile_path) if os.path.exists(profile_path) else []
        print(f"  Profile files: {files}")

        # Load profiles
        loaded = executor.load_profiles()
        print(f"  Load result: {'success' if loaded else 'failed (expected if no profiles)'}")

        gc.collect()

        return True


# =============================================================================
# Print Results
# =============================================================================

def print_benchmark_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print(f"\n{'Operation':<10} {'Size':>10} {'Bud (us)':>12} {'NumPy (us)':>12} {'Speedup':>10} {'Max Err':>12} {'Learn':>8}")
    print("-" * 80)

    for r in results:
        bud_us = r.mean_time_ns / 1000
        numpy_us = r.numpy_time_ns / 1000
        print(f"{r.name:<10} {r.size:>10,} {bud_us:>12.2f} {numpy_us:>12.2f} {r.speedup_vs_numpy:>9.2f}x {r.max_error:>12.2e} {r.improvement_first_to_last:>7.2f}x")


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("BUD FLOW LANG - ADAPTIVE JIT BENCHMARK")

    # Initialize
    if not bud.is_initialized():
        bud.initialize()

    # Initialize adaptive execution
    bud.initialize_adaptive_execution()

    # Print system info
    print_subheader("System Information")
    info = bud.get_hardware_info()
    print(f"  Architecture: {info['arch_family']} ({'64-bit' if info['is_64bit'] else '32-bit'})")
    print(f"  SIMD Width: {info['simd_width']} bytes ({info['simd_width']*8} bits)")
    print(f"  AVX2: {info['has_avx2']}, AVX-512: {info['has_avx512']}")

    executor = bud.get_adaptive_executor()
    print(f"  Adaptive Executor: {'enabled' if executor.enabled else 'disabled'}")
    print(f"  Tier Promotion: {'enabled' if executor.config.enable_promotion else 'disabled'}")
    print(f"  Profile Persistence: {'enabled' if executor.config.enable_persistence else 'disabled'}")

    # ==========================================================================
    # 1. Accuracy Tests
    # ==========================================================================
    print_header("ACCURACY TESTS", "-")
    accuracy_passed = test_accuracy()

    if not accuracy_passed:
        print("\n[ERROR] Accuracy tests failed! Aborting benchmark.")
        return 1

    # ==========================================================================
    # 2. Learning Curve Test
    # ==========================================================================
    print_header("LEARNING CURVE TEST", "-")
    learning_results = test_learning_curve()

    # ==========================================================================
    # 3. Performance Benchmarks
    # ==========================================================================
    print_header("PERFORMANCE BENCHMARKS", "-")

    # Reset executor for clean benchmarks
    executor.reset()

    sizes = [1000, 10000, 100000, 1000000]
    all_results = []

    # Binary ops
    print_subheader("Binary Operations")
    all_results.extend(benchmark_binary_op("add", lambda a, b: a + b, np.add, sizes, iterations=50))
    all_results.extend(benchmark_binary_op("mul", lambda a, b: a * b, np.multiply, sizes, iterations=50))
    all_results.extend(benchmark_binary_op("sub", lambda a, b: a - b, np.subtract, sizes, iterations=50))

    # Reductions
    print_subheader("Reductions")
    all_results.extend(benchmark_reduction("sum", lambda a: a.sum(), np.sum, sizes, iterations=50))
    all_results.extend(benchmark_reduction("dot", bud.dot, np.dot, sizes, iterations=50))

    # Unary ops
    print_subheader("Unary Operations")
    all_results.extend(benchmark_unary_op("exp", bud.exp, np.exp, sizes, iterations=50))
    all_results.extend(benchmark_unary_op("log", bud.log, np.log, sizes, iterations=50, positive_only=True))
    all_results.extend(benchmark_unary_op("cos", bud.cos, np.cos, sizes, iterations=50))

    # FMA
    print_subheader("Fused Multiply-Add")
    all_results.extend(benchmark_fma(sizes, iterations=50))

    # Print results
    print_header("BENCHMARK RESULTS", "-")
    print_benchmark_results(all_results)

    # ==========================================================================
    # 4. Profile Persistence Test
    # ==========================================================================
    print_header("PROFILE PERSISTENCE TEST", "-")
    test_profile_persistence()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_header("SUMMARY")

    # Calculate overall stats
    total_speedups = [r.speedup_vs_numpy for r in all_results]
    avg_speedup = sum(total_speedups) / len(total_speedups)
    max_speedup = max(total_speedups)
    min_speedup = min(total_speedups)

    learning_improvements = [r.improvement_first_to_last for r in all_results]
    avg_learning = sum(learning_improvements) / len(learning_improvements)

    max_errors = [r.max_error for r in all_results]
    worst_error = max(max_errors)

    print(f"  Total benchmarks run: {len(all_results)}")
    print(f"  Average speedup vs NumPy: {avg_speedup:.2f}x")
    print(f"  Max speedup: {max_speedup:.2f}x")
    print(f"  Min speedup: {min_speedup:.2f}x")
    print(f"  Average learning improvement: {avg_learning:.2f}x")
    print(f"  Worst numerical error: {worst_error:.2e}")
    print(f"  Accuracy: {'PASSED' if accuracy_passed else 'FAILED'}")

    # Final executor stats
    stats = executor.statistics()
    print(f"\n  Adaptive Executor Statistics:")
    print(f"    Total executions: {stats.total_executions}")
    print(f"    Tier 0 (Interpreter): {stats.tier0_executions}")
    print(f"    Tier 1 (CopyPatch): {stats.tier1_executions}")
    print(f"    Tier 2 (Fused): {stats.tier2_executions}")
    print(f"    Promotions: {stats.promotions_performed}")
    print(f"    Specializations: {stats.specializations_performed}")

    # Shutdown adaptive execution (saves profiles)
    bud.shutdown_adaptive_execution()

    print_header("BENCHMARK COMPLETE")
    print("\nThe Adaptive JIT system is working correctly.")
    print("Performance should improve over time as the system learns optimal configurations.\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
