#!/usr/bin/env python3
"""
=============================================================================
Comprehensive Benchmark: Bud Flow Lang vs JAX vs NumPy
=============================================================================

This benchmark compares three frameworks across multiple operation categories:
1. Element-wise operations (add, mul, sqrt, exp, sin)
2. Reduction operations (sum, mean, dot product)
3. Fused operations (FMA - Bud's specialty)
4. ML operations (softmax, ReLU)

For Bud Flow Lang, we demonstrate three usage tiers:
- Beginner: Simple NumPy-like API
- Developer: Memory optimization, tiling, prefetch
- Expert: Direct Highway SIMD operations

Usage:
    python run_benchmarks.py [--sizes SIZES] [--runs RUNS] [--warmup WARMUP]

Example:
    python run_benchmarks.py --sizes 1000,100000,1000000 --runs 10

"""

import sys
import os
import time
import argparse
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

# Add build directory to path for bud_flow_lang_py
BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build'))
sys.path.insert(0, BUILD_DIR)

import numpy as np
from tabulate import tabulate

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    # Disable JAX GPU and set CPU-only
    jax.config.update('jax_platform_name', 'cpu')
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("Warning: JAX not available, skipping JAX benchmarks")

# Import Bud Flow Lang
try:
    import bud_flow_lang_py as flow
    flow.initialize()
    HAS_BUD = True
except ImportError:
    HAS_BUD = False
    print("Error: bud_flow_lang_py not found. Build the project first.")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000]
DEFAULT_WARMUP = 3
DEFAULT_RUNS = 10

@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    size: int
    time_ms: float
    throughput_gflops: float = 0.0

# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark(func: Callable, warmup: int = DEFAULT_WARMUP, runs: int = DEFAULT_RUNS) -> float:
    """Run benchmark with warmup and return median time in ms."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        # Ensure computation completes
        if HAS_JAX and hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.median(times)

def get_system_info() -> str:
    """Get system information string."""
    info = []

    # CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    cpu = line.split(':')[1].strip()
                    info.append(f"CPU: {cpu}")
                    break
    except:
        pass

    # Bud Flow Lang hardware info
    if HAS_BUD:
        hw = flow.get_hardware_info()
        info.append(f"SIMD: {hw.get('simd_width', 'N/A')} bytes")
        if hw.get('has_avx512'):
            info.append("AVX-512")
        elif hw.get('has_avx2'):
            info.append("AVX2")
        elif hw.get('has_avx'):
            info.append("AVX")
        info.append(f"Cores: {hw.get('physical_cores', 'N/A')}")

    return ", ".join(info)

# =============================================================================
# NumPy Benchmarks
# =============================================================================

def benchmark_numpy_add(size: int) -> float:
    a = np.arange(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2.0
    return benchmark(lambda: a + b)

def benchmark_numpy_mul(size: int) -> float:
    a = np.arange(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2.0
    return benchmark(lambda: a * b)

def benchmark_numpy_fma(size: int) -> float:
    """FMA: a * b + c (unfused in NumPy)"""
    a = np.arange(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2.0
    c = np.ones(size, dtype=np.float32)
    return benchmark(lambda: a * b + c)

def benchmark_numpy_sqrt(size: int) -> float:
    a = np.arange(1, size + 1, dtype=np.float32)
    return benchmark(lambda: np.sqrt(a))

def benchmark_numpy_exp(size: int) -> float:
    a = np.arange(size, dtype=np.float32) / size  # Scale to avoid overflow
    return benchmark(lambda: np.exp(a))

def benchmark_numpy_sin(size: int) -> float:
    a = np.arange(size, dtype=np.float32)
    return benchmark(lambda: np.sin(a))

def benchmark_numpy_sum(size: int) -> float:
    a = np.arange(size, dtype=np.float32)
    return benchmark(lambda: a.sum())

def benchmark_numpy_dot(size: int) -> float:
    a = np.arange(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32)
    return benchmark(lambda: np.dot(a, b))

def benchmark_numpy_softmax(size: int) -> float:
    """Softmax operation."""
    a = np.random.randn(size).astype(np.float32)
    def softmax():
        exp_a = np.exp(a - a.max())
        return exp_a / exp_a.sum()
    return benchmark(softmax)

# =============================================================================
# JAX Benchmarks
# =============================================================================

def benchmark_jax_add(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32)
    b = jnp.ones(size, dtype=jnp.float32) * 2.0
    def run():
        r = a + b
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_mul(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32)
    b = jnp.ones(size, dtype=jnp.float32) * 2.0
    def run():
        r = a * b
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_fma(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32)
    b = jnp.ones(size, dtype=jnp.float32) * 2.0
    c = jnp.ones(size, dtype=jnp.float32)
    def run():
        r = a * b + c
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_sqrt(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(1, size + 1, dtype=jnp.float32)
    def run():
        r = jnp.sqrt(a)
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_exp(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32) / size
    def run():
        r = jnp.exp(a)
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_sin(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32)
    def run():
        r = jnp.sin(a)
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_sum(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32)
    def run():
        r = a.sum()
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_dot(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    a = jnp.arange(size, dtype=jnp.float32)
    b = jnp.ones(size, dtype=jnp.float32)
    def run():
        r = jnp.dot(a, b)
        r.block_until_ready()
        return r
    return benchmark(run)

def benchmark_jax_softmax(size: int) -> float:
    if not HAS_JAX:
        return float('nan')
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (size,), dtype=jnp.float32)
    def run():
        r = jax.nn.softmax(a)
        r.block_until_ready()
        return r
    return benchmark(run)

# =============================================================================
# Bud Flow Lang Benchmarks - BEGINNER TIER
# =============================================================================

def benchmark_bud_beginner_add(size: int) -> float:
    """Beginner: Simple NumPy-like operations."""
    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    return benchmark(lambda: a + b)

def benchmark_bud_beginner_mul(size: int) -> float:
    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    return benchmark(lambda: a * b)

def benchmark_bud_beginner_fma(size: int) -> float:
    """Beginner FMA: Using separate operations (not fused)."""
    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    c = flow.Bunch.ones(size)
    return benchmark(lambda: a * b + c)

def benchmark_bud_beginner_sqrt(size: int) -> float:
    a = flow.Bunch.arange(size) + 1.0  # Avoid sqrt(0)
    return benchmark(lambda: a.sqrt())

def benchmark_bud_beginner_exp(size: int) -> float:
    a = flow.Bunch.arange(size) / size
    return benchmark(lambda: a.exp())

def benchmark_bud_beginner_sin(size: int) -> float:
    a = flow.Bunch.arange(size)
    return benchmark(lambda: a.sin())

def benchmark_bud_beginner_sum(size: int) -> float:
    a = flow.Bunch.arange(size)
    return benchmark(lambda: a.sum())

def benchmark_bud_beginner_dot(size: int) -> float:
    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size)
    return benchmark(lambda: a.dot(b))

def benchmark_bud_beginner_softmax(size: int) -> float:
    """Beginner softmax: Simple implementation."""
    # Use numpy to create random data, then convert
    np_data = np.random.randn(size).astype(np.float32)
    a = flow.Bunch.from_list(list(np_data))
    def softmax():
        exp_a = (a - a.max()).exp()
        return exp_a / exp_a.sum()
    return benchmark(softmax)

# =============================================================================
# Bud Flow Lang Benchmarks - DEVELOPER TIER
# =============================================================================

def benchmark_bud_developer_add(size: int) -> float:
    """Developer: With memory optimization enabled."""
    # Enable tiling for better cache performance
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)

    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    result = benchmark(lambda: a + b)

    # Reset
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_developer_mul(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)

    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    result = benchmark(lambda: a * b)

    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_developer_fma(size: int) -> float:
    """Developer FMA: Using explicit fma() function (FUSED!)."""
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)

    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    c = flow.Bunch.ones(size)
    # Use explicit FMA for fused operation
    result = benchmark(lambda: a.fma(b, c))

    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_developer_sqrt(size: int) -> float:
    flow.set_tiling_enabled(True)
    a = flow.Bunch.arange(size) + 1.0
    result = benchmark(lambda: a.sqrt())
    flow.set_tiling_enabled(False)
    return result

def benchmark_bud_developer_exp(size: int) -> float:
    flow.set_tiling_enabled(True)
    a = flow.Bunch.arange(size) / size
    result = benchmark(lambda: a.exp())
    flow.set_tiling_enabled(False)
    return result

def benchmark_bud_developer_sin(size: int) -> float:
    flow.set_tiling_enabled(True)
    a = flow.Bunch.arange(size)
    result = benchmark(lambda: a.sin())
    flow.set_tiling_enabled(False)
    return result

def benchmark_bud_developer_sum(size: int) -> float:
    flow.set_tiling_enabled(True)
    a = flow.Bunch.arange(size)
    result = benchmark(lambda: a.sum())
    flow.set_tiling_enabled(False)
    return result

def benchmark_bud_developer_dot(size: int) -> float:
    flow.set_tiling_enabled(True)
    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size)
    result = benchmark(lambda: a.dot(b))
    flow.set_tiling_enabled(False)
    return result

def benchmark_bud_developer_softmax(size: int) -> float:
    """Developer softmax with optimizations."""
    flow.set_tiling_enabled(True)
    np_data = np.random.randn(size).astype(np.float32)
    a = flow.Bunch.from_list(list(np_data))
    def softmax():
        max_val = a.max()
        shifted = a - max_val
        exp_a = shifted.exp()
        return exp_a / exp_a.sum()
    result = benchmark(softmax)
    flow.set_tiling_enabled(False)
    return result

# =============================================================================
# Bud Flow Lang Benchmarks - EXPERT TIER
# =============================================================================

def benchmark_bud_expert_add(size: int) -> float:
    """Expert: Direct highway operations (where available)."""
    # For now, expert tier uses the same as developer with all optimizations
    # In future, this would use direct highway.add() calls
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)

    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    result = benchmark(lambda: a + b)

    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_mul(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)

    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    result = benchmark(lambda: a * b)

    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_fma(size: int) -> float:
    """Expert FMA: Explicit fused multiply-add with all optimizations."""
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)

    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size) * 2.0
    c = flow.Bunch.ones(size)
    result = benchmark(lambda: a.fma(b, c))

    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_sqrt(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)
    a = flow.Bunch.arange(size) + 1.0
    result = benchmark(lambda: a.sqrt())
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_exp(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)
    a = flow.Bunch.arange(size) / size
    result = benchmark(lambda: a.exp())
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_sin(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)
    a = flow.Bunch.arange(size)
    result = benchmark(lambda: a.sin())
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_sum(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)
    a = flow.Bunch.arange(size)
    result = benchmark(lambda: a.sum())
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_dot(size: int) -> float:
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)
    a = flow.Bunch.arange(size)
    b = flow.Bunch.ones(size)
    result = benchmark(lambda: a.dot(b))
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

def benchmark_bud_expert_softmax(size: int) -> float:
    """Expert softmax with all optimizations."""
    flow.set_tiling_enabled(True)
    flow.set_prefetch_enabled(True)
    np_data = np.random.randn(size).astype(np.float32)
    a = flow.Bunch.from_list(list(np_data))
    def softmax():
        max_val = a.max()
        shifted = a - max_val
        exp_a = shifted.exp()
        return exp_a / exp_a.sum()
    result = benchmark(softmax)
    flow.set_tiling_enabled(False)
    flow.set_prefetch_enabled(False)
    return result

# =============================================================================
# Main Runner
# =============================================================================

BENCHMARKS = {
    'add': {
        'numpy': benchmark_numpy_add,
        'jax': benchmark_jax_add,
        'bud_beginner': benchmark_bud_beginner_add,
        'bud_developer': benchmark_bud_developer_add,
        'bud_expert': benchmark_bud_expert_add,
    },
    'mul': {
        'numpy': benchmark_numpy_mul,
        'jax': benchmark_jax_mul,
        'bud_beginner': benchmark_bud_beginner_mul,
        'bud_developer': benchmark_bud_developer_mul,
        'bud_expert': benchmark_bud_expert_mul,
    },
    'fma': {
        'numpy': benchmark_numpy_fma,
        'jax': benchmark_jax_fma,
        'bud_beginner': benchmark_bud_beginner_fma,
        'bud_developer': benchmark_bud_developer_fma,
        'bud_expert': benchmark_bud_expert_fma,
    },
    'sqrt': {
        'numpy': benchmark_numpy_sqrt,
        'jax': benchmark_jax_sqrt,
        'bud_beginner': benchmark_bud_beginner_sqrt,
        'bud_developer': benchmark_bud_developer_sqrt,
        'bud_expert': benchmark_bud_expert_sqrt,
    },
    'exp': {
        'numpy': benchmark_numpy_exp,
        'jax': benchmark_jax_exp,
        'bud_beginner': benchmark_bud_beginner_exp,
        'bud_developer': benchmark_bud_developer_exp,
        'bud_expert': benchmark_bud_expert_exp,
    },
    'sin': {
        'numpy': benchmark_numpy_sin,
        'jax': benchmark_jax_sin,
        'bud_beginner': benchmark_bud_beginner_sin,
        'bud_developer': benchmark_bud_developer_sin,
        'bud_expert': benchmark_bud_expert_sin,
    },
    'sum': {
        'numpy': benchmark_numpy_sum,
        'jax': benchmark_jax_sum,
        'bud_beginner': benchmark_bud_beginner_sum,
        'bud_developer': benchmark_bud_developer_sum,
        'bud_expert': benchmark_bud_expert_sum,
    },
    'dot': {
        'numpy': benchmark_numpy_dot,
        'jax': benchmark_jax_dot,
        'bud_beginner': benchmark_bud_beginner_dot,
        'bud_developer': benchmark_bud_developer_dot,
        'bud_expert': benchmark_bud_expert_dot,
    },
    'softmax': {
        'numpy': benchmark_numpy_softmax,
        'jax': benchmark_jax_softmax,
        'bud_beginner': benchmark_bud_beginner_softmax,
        'bud_developer': benchmark_bud_developer_softmax,
        'bud_expert': benchmark_bud_expert_softmax,
    },
}

def run_benchmarks(sizes: List[int], selected_ops: List[str] = None) -> Dict:
    """Run all benchmarks and return results."""
    results = {}

    ops_to_run = selected_ops if selected_ops else list(BENCHMARKS.keys())

    for op_name in ops_to_run:
        if op_name not in BENCHMARKS:
            print(f"Warning: Unknown operation '{op_name}', skipping")
            continue

        print(f"\nBenchmarking: {op_name.upper()}")
        results[op_name] = {}

        for size in sizes:
            results[op_name][size] = {}
            print(f"  Size {size:,}...", end=" ", flush=True)

            for framework, bench_func in BENCHMARKS[op_name].items():
                if framework == 'jax' and not HAS_JAX:
                    results[op_name][size][framework] = float('nan')
                else:
                    try:
                        t = bench_func(size)
                        results[op_name][size][framework] = t
                    except Exception as e:
                        print(f"\n    Error in {framework}: {e}")
                        results[op_name][size][framework] = float('nan')

            print("done")

    return results

def print_results(results: Dict, sizes: List[int]):
    """Print results in formatted tables."""

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nSystem: {get_system_info()}")
    print(f"All times in milliseconds (ms), lower is better")
    print(f"Speedup shows Bud Expert vs NumPy (>1.0x means Bud is faster)")

    for op_name, op_results in results.items():
        print(f"\n{'=' * 80}")
        print(f"  {op_name.upper()}")
        print(f"{'=' * 80}")

        headers = ['Size', 'NumPy', 'JAX', 'Bud Beginner', 'Bud Developer', 'Bud Expert', 'Speedup']
        rows = []

        for size in sizes:
            if size not in op_results:
                continue
            data = op_results[size]

            numpy_time = data.get('numpy', float('nan'))
            jax_time = data.get('jax', float('nan'))
            bud_beg = data.get('bud_beginner', float('nan'))
            bud_dev = data.get('bud_developer', float('nan'))
            bud_exp = data.get('bud_expert', float('nan'))

            # Calculate speedup (NumPy / Bud Expert)
            if numpy_time > 0 and bud_exp > 0 and not np.isnan(numpy_time) and not np.isnan(bud_exp):
                speedup = numpy_time / bud_exp
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            def fmt(t):
                if np.isnan(t):
                    return "N/A"
                elif t < 0.01:
                    return f"{t*1000:.2f} us"
                elif t < 1:
                    return f"{t:.3f}"
                else:
                    return f"{t:.2f}"

            rows.append([
                f"{size:,}",
                fmt(numpy_time),
                fmt(jax_time),
                fmt(bud_beg),
                fmt(bud_dev),
                fmt(bud_exp),
                speedup_str
            ])

        print(tabulate(rows, headers=headers, tablefmt='grid'))

def main():
    parser = argparse.ArgumentParser(description='Benchmark Bud Flow Lang vs JAX vs NumPy')
    parser.add_argument('--sizes', type=str, default='1000,10000,100000,1000000',
                        help='Comma-separated list of array sizes')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help='Number of timed runs per benchmark')
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP,
                        help='Number of warmup runs')
    parser.add_argument('--ops', type=str, default=None,
                        help='Comma-separated list of operations to benchmark')

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    ops = args.ops.split(',') if args.ops else None

    print("=" * 80)
    print("  BUD FLOW LANG vs JAX vs NUMPY BENCHMARK")
    print("=" * 80)
    print(f"\nSystem: {get_system_info()}")
    print(f"Array sizes: {sizes}")
    print(f"Warmup runs: {args.warmup}, Timed runs: {args.runs}")
    print(f"JAX available: {HAS_JAX}")
    print(f"Bud Flow Lang available: {HAS_BUD}")

    # Run benchmarks
    results = run_benchmarks(sizes, ops)

    # Print results
    print_results(results, sizes)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
