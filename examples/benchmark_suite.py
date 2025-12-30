#!/usr/bin/env python3
"""
Bud Flow Lang Benchmark Suite

Comprehensive benchmarks for measuring SIMD performance:
- Element-wise operations
- Reductions
- Dot products
- FMA operations
- Memory bandwidth analysis

Run: python benchmark_suite.py
"""

import bud_flow_lang_py as flow
import time
import gc
import statistics


class Benchmark:
    """Benchmark utility with proper methodology."""

    def __init__(self, warmup=20, iterations=50):
        self.warmup = warmup
        self.iterations = iterations

    def run(self, func):
        """Run benchmark with warmup and statistics."""
        # Warmup phase
        for _ in range(self.warmup):
            func()

        # Force garbage collection
        gc.collect()

        # Measurement phase
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)

        return {
            'mean_ms': statistics.mean(times) * 1000,
            'std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
            'median_ms': statistics.median(times) * 1000,
        }


def print_result(name, stats, n_elements, bytes_accessed=None):
    """Print formatted benchmark result."""
    throughput = n_elements / (stats['mean_ms'] / 1000) / 1e9

    print(f"  {name:<20} {stats['mean_ms']:>8.4f} ms "
          f"({stats['std_ms']:.4f} std) "
          f"{throughput:>6.2f} Gelem/s", end="")

    if bytes_accessed:
        bandwidth = bytes_accessed / (stats['mean_ms'] / 1000) / 1e9
        print(f" {bandwidth:>5.1f} GB/s", end="")

    print()


def benchmark_creation(bench):
    """Benchmark array creation functions."""
    print("\n=== Array Creation ===")

    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        print(f"\nSize: {n:,} elements")

        # zeros
        stats = bench.run(lambda: flow.zeros(n))
        print_result("zeros", stats, n)

        # ones
        stats = bench.run(lambda: flow.ones(n))
        print_result("ones", stats, n)

        # arange
        stats = bench.run(lambda: flow.arange(n))
        print_result("arange", stats, n)

        # full
        stats = bench.run(lambda: flow.full(n, 3.14))
        print_result("full", stats, n)


def benchmark_binary_ops(bench):
    """Benchmark binary operations."""
    print("\n=== Binary Operations ===")

    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        print(f"\nSize: {n:,} elements")

        a = flow.ones(n)
        b = flow.full(n, 2.0)
        bytes_rw = n * 4 * 3  # 2 read + 1 write

        # Addition
        stats = bench.run(lambda: a + b)
        print_result("add", stats, n, bytes_rw)

        # Subtraction
        stats = bench.run(lambda: a - b)
        print_result("sub", stats, n, bytes_rw)

        # Multiplication
        stats = bench.run(lambda: a * b)
        print_result("mul", stats, n, bytes_rw)

        # Division
        stats = bench.run(lambda: a / b)
        print_result("div", stats, n, bytes_rw)

        del a, b


def benchmark_reductions(bench):
    """Benchmark reduction operations."""
    print("\n=== Reductions ===")

    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        print(f"\nSize: {n:,} elements")

        a = flow.arange(n)
        bytes_read = n * 4  # 1 read

        # Sum
        stats = bench.run(lambda: a.sum())
        print_result("sum", stats, n, bytes_read)

        # Min
        stats = bench.run(lambda: a.min())
        print_result("min", stats, n, bytes_read)

        # Max
        stats = bench.run(lambda: a.max())
        print_result("max", stats, n, bytes_read)

        # Mean
        stats = bench.run(lambda: a.mean())
        print_result("mean", stats, n, bytes_read)

        del a


def benchmark_dot_product(bench):
    """Benchmark dot product."""
    print("\n=== Dot Product ===")

    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        print(f"\nSize: {n:,} elements")

        a = flow.ones(n)
        b = flow.full(n, 2.0)
        bytes_read = n * 4 * 2  # 2 reads

        # Function form
        stats = bench.run(lambda: flow.dot(a, b))
        print_result("dot (function)", stats, n, bytes_read)

        # Method form
        stats = bench.run(lambda: a.dot(b))
        print_result("dot (method)", stats, n, bytes_read)

        del a, b


def benchmark_fma(bench):
    """Benchmark fused multiply-add."""
    print("\n=== Fused Multiply-Add ===")

    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        print(f"\nSize: {n:,} elements")

        a = flow.ones(n)
        b = flow.full(n, 2.0)
        c = flow.full(n, 3.0)
        bytes_rw = n * 4 * 4  # 3 read + 1 write

        # FMA
        stats = bench.run(lambda: flow.fma(a, b, c))
        print_result("fma", stats, n, bytes_rw)

        # Separate multiply-add for comparison
        stats = bench.run(lambda: a * b + c)
        print_result("mul + add", stats, n, bytes_rw)

        del a, b, c


def benchmark_unary_ops(bench):
    """Benchmark unary operations."""
    print("\n=== Unary Operations ===")

    n = 1_000_000
    print(f"\nSize: {n:,} elements")

    # Create test arrays
    a = flow.linspace(0.1, 10, n)
    b = flow.linspace(-3, 3, n)
    bytes_rw = n * 4 * 2  # 1 read + 1 write

    # sqrt
    stats = bench.run(lambda: flow.sqrt(a))
    print_result("sqrt", stats, n, bytes_rw)

    # exp
    stats = bench.run(lambda: flow.exp(b))
    print_result("exp", stats, n, bytes_rw)

    # log
    stats = bench.run(lambda: flow.log(a))
    print_result("log", stats, n, bytes_rw)

    # sin
    stats = bench.run(lambda: flow.sin(a))
    print_result("sin", stats, n, bytes_rw)

    # cos
    stats = bench.run(lambda: flow.cos(a))
    print_result("cos", stats, n, bytes_rw)

    # tanh
    stats = bench.run(lambda: flow.tanh(b))
    print_result("tanh", stats, n, bytes_rw)

    # abs
    stats = bench.run(lambda: flow.abs(b))
    print_result("abs", stats, n, bytes_rw)

    del a, b


def benchmark_scaling():
    """Analyze performance scaling with array size."""
    print("\n=== Scaling Analysis ===")

    bench = Benchmark(warmup=10, iterations=30)
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    print("\nDot Product scaling:")
    print(f"{'Size':>12} {'Time (ms)':>12} {'Throughput':>15} {'Bandwidth':>12}")
    print("-" * 55)

    for n in sizes:
        a = flow.ones(n)
        b = flow.ones(n)

        stats = bench.run(lambda: flow.dot(a, b))
        throughput = n / (stats['mean_ms'] / 1000) / 1e9
        bandwidth = (n * 4 * 2) / (stats['mean_ms'] / 1000) / 1e9

        print(f"{n:>12,} {stats['mean_ms']:>12.4f} {throughput:>12.2f} Gelem/s {bandwidth:>9.1f} GB/s")

        del a, b


def benchmark_cache_effects():
    """Analyze cache effects on performance."""
    print("\n=== Cache Effects Analysis ===")

    bench = Benchmark(warmup=10, iterations=30)
    cache = flow.detect_cache_config()

    print(f"\nYour cache configuration:")
    print(f"  L1: {cache['l1_size_kb']} KB")
    print(f"  L2: {cache['l2_size_kb']} KB")
    print(f"  L3: {cache['l3_size_kb']} KB")

    # Test sizes around cache boundaries
    test_sizes = [
        ("L1 (50%)", cache['l1_size'] // 8),        # 50% of L1, 2 arrays
        ("L1 (100%)", cache['l1_size'] // 4),       # 100% of L1, 2 arrays
        ("L2 (50%)", cache['l2_size'] // 8),        # 50% of L2
        ("L2 (100%)", cache['l2_size'] // 4),       # 100% of L2
        ("L3 (50%)", cache['l3_size'] // 8),        # 50% of L3
        ("L3 (100%)", cache['l3_size'] // 4),       # 100% of L3
        ("RAM", cache['l3_size']),                  # Exceeds L3
    ]

    print(f"\nDot product bandwidth at cache boundaries:")
    print(f"{'Level':>12} {'Elements':>12} {'Bandwidth':>12}")
    print("-" * 40)

    for name, n in test_sizes:
        a = flow.ones(n)
        b = flow.ones(n)

        stats = bench.run(lambda: flow.dot(a, b))
        bandwidth = (n * 4 * 2) / (stats['mean_ms'] / 1000) / 1e9

        print(f"{name:>12} {n:>12,} {bandwidth:>9.1f} GB/s")

        del a, b


def main():
    print("=" * 60)
    print("Bud Flow Lang Benchmark Suite")
    print("=" * 60)

    # Initialize
    flow.initialize()

    # Print system info
    info = flow.get_hardware_info()
    print(f"\nHardware: {info['arch_family']} with {info['simd_width']*8}-bit SIMD")
    print(f"Cores: {info['physical_cores']} physical")

    # Create benchmark runner
    bench = Benchmark(warmup=20, iterations=50)

    # Run benchmarks
    benchmark_creation(bench)
    benchmark_binary_ops(bench)
    benchmark_reductions(bench)
    benchmark_dot_product(bench)
    benchmark_fma(bench)
    benchmark_unary_ops(bench)
    benchmark_scaling()
    benchmark_cache_effects()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
