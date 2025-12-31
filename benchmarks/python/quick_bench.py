#!/usr/bin/env python3
"""
Quick Performance Benchmark: Bud Flow Lang vs JAX vs NumPy
Compares the three frameworks across key operations.
"""

import sys
import os
import time
import numpy as np

# Add build directory to path
BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build'))
sys.path.insert(0, BUILD_DIR)

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_platform_name', 'cpu')
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("Warning: JAX not available")

# Import Bud Flow Lang
import bud_flow_lang_py as flow
flow.initialize()

# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark(func, warmup=5, runs=15):
    """Run benchmark with warmup and return median time in ms."""
    for _ in range(warmup):
        func()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        if HAS_JAX and hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.median(times)

def fmt_time(t):
    """Format time value."""
    if np.isnan(t):
        return "N/A"
    elif t < 0.001:
        return f"{t*1000:.2f}µs"
    elif t < 0.1:
        return f"{t*1000:.1f}µs"
    elif t < 1:
        return f"{t:.3f}ms"
    else:
        return f"{t:.2f}ms"

def fmt_speedup(numpy_t, bud_t):
    """Format speedup."""
    if np.isnan(numpy_t) or np.isnan(bud_t) or bud_t <= 0:
        return "N/A"
    return f"{numpy_t/bud_t:.2f}x"

# =============================================================================
# System Info
# =============================================================================

def get_system_info():
    """Get system information."""
    hw = flow.get_hardware_info()
    cpu = "Unknown CPU"
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    cpu = line.split(':')[1].strip()
                    break
    except:
        pass

    simd = "AVX-512" if hw.get('has_avx512') else ("AVX2" if hw.get('has_avx2') else "SSE")
    return f"{cpu} | {simd} ({hw['simd_width']*8}-bit)"

# =============================================================================
# Run Benchmarks
# =============================================================================

def run_all_benchmarks():
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    print("=" * 100)
    print("  PERFORMANCE BENCHMARK: Bud Flow Lang vs JAX vs NumPy")
    print("=" * 100)
    print(f"\nSystem: {get_system_info()}")
    print(f"Array sizes: {[f'{s:,}' for s in sizes]}")
    print(f"JAX available: {HAS_JAX}")
    print()

    operations = [
        ("ADD", "a + b"),
        ("MUL", "a * b"),
        ("FMA", "a*b + c"),
        ("SQRT", "sqrt(a)"),
        ("EXP", "exp(a)"),
        ("SIN", "sin(a)"),
        ("SUM", "sum(a)"),
        ("DOT", "dot(a,b)"),
    ]

    all_results = {}

    for op_name, op_desc in operations:
        print(f"\n{'='*100}")
        print(f"  {op_name} ({op_desc})")
        print(f"{'='*100}")
        print(f"{'Size':>12} | {'NumPy':>10} | {'JAX':>10} | {'Bud Begin':>10} | {'Bud Dev':>10} | {'Bud Expert':>10} | {'vs NumPy':>10}")
        print("-" * 100)

        for size in sizes:
            # Create test data
            np_a = np.arange(size, dtype=np.float32)
            np_b = np.ones(size, dtype=np.float32) * 2.0
            np_c = np.ones(size, dtype=np.float32)

            bud_a = flow.Bunch.arange(size)
            bud_b = flow.Bunch.ones(size) * 2.0
            bud_c = flow.Bunch.ones(size)

            if HAS_JAX:
                jax_a = jnp.array(np_a)
                jax_b = jnp.array(np_b)
                jax_c = jnp.array(np_c)

            # Run benchmarks based on operation
            if op_name == "ADD":
                np_time = benchmark(lambda: np_a + np_b)
                jax_time = benchmark(lambda: jax_a + jax_b) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a + bud_b)
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a + bud_b)
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a + bud_b)
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "MUL":
                np_time = benchmark(lambda: np_a * np_b)
                jax_time = benchmark(lambda: jax_a * jax_b) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a * bud_b)
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a * bud_b)
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a * bud_b)
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "FMA":
                np_time = benchmark(lambda: np_a * np_b + np_c)
                jax_time = benchmark(lambda: jax_a * jax_b + jax_c) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a * bud_b + bud_c)
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a.fma(bud_b, bud_c))  # Fused!
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a.fma(bud_b, bud_c))
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "SQRT":
                np_a_pos = np.abs(np_a) + 1.0
                bud_a_pos = bud_a.abs() + 1.0
                jax_a_pos = jnp.abs(jax_a) + 1.0 if HAS_JAX else None
                np_time = benchmark(lambda: np.sqrt(np_a_pos))
                jax_time = benchmark(lambda: jnp.sqrt(jax_a_pos)) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a_pos.sqrt())
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a_pos.sqrt())
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a_pos.sqrt())
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "EXP":
                np_a_scaled = np_a / size
                bud_a_scaled = bud_a / float(size)
                jax_a_scaled = jax_a / size if HAS_JAX else None
                np_time = benchmark(lambda: np.exp(np_a_scaled))
                jax_time = benchmark(lambda: jnp.exp(jax_a_scaled)) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a_scaled.exp())
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a_scaled.exp())
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a_scaled.exp())
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "SIN":
                np_time = benchmark(lambda: np.sin(np_a))
                jax_time = benchmark(lambda: jnp.sin(jax_a)) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a.sin())
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a.sin())
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a.sin())
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "SUM":
                np_time = benchmark(lambda: np_a.sum())
                jax_time = benchmark(lambda: jax_a.sum()) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a.sum())
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a.sum())
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a.sum())
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            elif op_name == "DOT":
                np_time = benchmark(lambda: np.dot(np_a, np_b))
                jax_time = benchmark(lambda: jnp.dot(jax_a, jax_b)) if HAS_JAX else float('nan')
                bud_beg = benchmark(lambda: bud_a.dot(bud_b))
                flow.set_tiling_enabled(True)
                bud_dev = benchmark(lambda: bud_a.dot(bud_b))
                flow.set_prefetch_enabled(True)
                bud_exp = benchmark(lambda: bud_a.dot(bud_b))
                flow.set_tiling_enabled(False)
                flow.set_prefetch_enabled(False)

            speedup = fmt_speedup(np_time, bud_exp)

            print(f"{size:>12,} | {fmt_time(np_time):>10} | {fmt_time(jax_time):>10} | "
                  f"{fmt_time(bud_beg):>10} | {fmt_time(bud_dev):>10} | {fmt_time(bud_exp):>10} | {speedup:>10}")

            # Store results
            if op_name not in all_results:
                all_results[op_name] = {}
            all_results[op_name][size] = {
                'numpy': np_time, 'jax': jax_time,
                'bud_beg': bud_beg, 'bud_dev': bud_dev, 'bud_exp': bud_exp
            }

    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY: Average Speedup vs NumPy (Bud Expert)")
    print("=" * 100)

    for op_name in all_results:
        speedups = []
        for size, data in all_results[op_name].items():
            if data['numpy'] > 0 and data['bud_exp'] > 0:
                speedups.append(data['numpy'] / data['bud_exp'])
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"  {op_name:>8}: {avg_speedup:.2f}x average speedup")

    print("\n" + "=" * 100)
    print("  BENCHMARK COMPLETE")
    print("=" * 100)

if __name__ == '__main__':
    run_all_benchmarks()
