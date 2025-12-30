#!/usr/bin/env python3
"""
Bud Flow Lang vs NumPy Comparison

Side-by-side comparison of Bud Flow Lang and NumPy:
- API similarity
- Performance comparison
- Use case recommendations

Run: python numpy_comparison.py

Requirements: numpy
"""

import bud_flow_lang_py as flow
import numpy as np
import time
import gc


def benchmark(func, iterations=50, warmup=20):
    """Benchmark with proper methodology."""
    for _ in range(warmup):
        func()
    gc.collect()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }


def compare(name, flow_func, np_func, iterations=50):
    """Compare Flow vs NumPy and print results."""
    flow_stats = benchmark(flow_func, iterations)
    np_stats = benchmark(np_func, iterations)

    ratio = np_stats['mean_ms'] / flow_stats['mean_ms']
    winner = "Flow" if ratio > 1 else "NumPy"
    speedup = max(ratio, 1/ratio)

    print(f"  {name:<25}")
    print(f"    Flow:  {flow_stats['mean_ms']:>8.4f} ms ({flow_stats['std_ms']:.4f} std)")
    print(f"    NumPy: {np_stats['mean_ms']:>8.4f} ms ({np_stats['std_ms']:.4f} std)")
    print(f"    Winner: {winner} ({speedup:.2f}x faster)")
    print()

    return flow_stats['mean_ms'], np_stats['mean_ms'], ratio


def demo_api_similarity():
    """Demonstrate API similarity between Flow and NumPy."""
    print("\n" + "=" * 60)
    print("API Similarity Demo")
    print("=" * 60)

    print("\n--- Array Creation ---")

    # zeros
    print("\nZeros:")
    print(f"  NumPy: np.zeros(5)       -> {np.zeros(5, dtype=np.float32)}")
    print(f"  Flow:  flow.zeros(5)     -> {flow.zeros(5).to_numpy()}")

    # ones
    print("\nOnes:")
    print(f"  NumPy: np.ones(5)        -> {np.ones(5, dtype=np.float32)}")
    print(f"  Flow:  flow.ones(5)      -> {flow.ones(5).to_numpy()}")

    # arange
    print("\nArange:")
    print(f"  NumPy: np.arange(5)      -> {np.arange(5, dtype=np.float32)}")
    print(f"  Flow:  flow.arange(5)    -> {flow.arange(5).to_numpy()}")

    # linspace
    print("\nLinspace:")
    print(f"  NumPy: np.linspace(0,1,5) -> {np.linspace(0, 1, 5, dtype=np.float32)}")
    print(f"  Flow:  flow.linspace(0,1,5) -> {flow.linspace(0, 1, 5).to_numpy()}")

    # full
    print("\nFull:")
    print(f"  NumPy: np.full(5, 3.14)  -> {np.full(5, 3.14, dtype=np.float32)}")
    print(f"  Flow:  flow.full(5, 3.14) -> {flow.full(5, 3.14).to_numpy()}")

    print("\n--- Operations ---")

    np_a = np.arange(5, dtype=np.float32)
    np_b = np.full(5, 2.0, dtype=np.float32)
    flow_a = flow.arange(5)
    flow_b = flow.full(5, 2.0)

    print(f"\na = {np_a}")
    print(f"b = {np_b}")

    # Addition
    print("\nAddition (a + b):")
    print(f"  NumPy: {np_a + np_b}")
    print(f"  Flow:  {(flow_a + flow_b).to_numpy()}")

    # Multiplication
    print("\nMultiplication (a * b):")
    print(f"  NumPy: {np_a * np_b}")
    print(f"  Flow:  {(flow_a * flow_b).to_numpy()}")

    # Division
    np_a_pos = np.arange(1, 6, dtype=np.float32)
    flow_a_pos = flow.arange(5, 1.0, 1.0)  # 5 elements starting at 1, step 1
    print("\nDivision (a / b) where a = [1,2,3,4,5]:")
    print(f"  NumPy: {np_a_pos / np_b}")
    print(f"  Flow:  {(flow_a_pos / flow_b).to_numpy()}")

    print("\n--- Reductions ---")

    np_arr = np.arange(10, dtype=np.float32)
    flow_arr = flow.arange(10)
    print(f"\nArray: {np_arr}")

    print(f"\nSum:")
    print(f"  NumPy: np_arr.sum()  = {np_arr.sum()}")
    print(f"  Flow:  flow_arr.sum() = {flow_arr.sum()}")

    print(f"\nMean:")
    print(f"  NumPy: np_arr.mean() = {np_arr.mean()}")
    print(f"  Flow:  flow_arr.mean() = {flow_arr.mean()}")

    print(f"\nMin/Max:")
    print(f"  NumPy: min={np_arr.min()}, max={np_arr.max()}")
    print(f"  Flow:  min={flow_arr.min()}, max={flow_arr.max()}")

    print("\n--- Dot Product ---")

    print(f"\nDot product of [1,2,3] . [4,5,6]:")
    np_x = np.array([1, 2, 3], dtype=np.float32)
    np_y = np.array([4, 5, 6], dtype=np.float32)
    flow_x = flow.flow([1.0, 2.0, 3.0])
    flow_y = flow.flow([4.0, 5.0, 6.0])

    print(f"  NumPy: np.dot(x, y) = {np.dot(np_x, np_y)}")
    print(f"  Flow:  flow.dot(x, y) = {flow.dot(flow_x, flow_y)}")


def performance_comparison():
    """Compare performance between Flow and NumPy."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    n = 1_000_000
    print(f"\nArray size: {n:,} elements ({n * 4 / 1e6:.0f} MB)")

    # Create arrays
    np_a = np.ones(n, dtype=np.float32)
    np_b = np.ones(n, dtype=np.float32)
    np_c = np.ones(n, dtype=np.float32)

    flow_a = flow.ones(n)
    flow_b = flow.ones(n)
    flow_c = flow.ones(n)

    results = {}

    print("\n--- Arithmetic Operations ---")

    # Addition
    results['add'] = compare(
        "Element-wise Add",
        lambda: flow_a + flow_b,
        lambda: np_a + np_b
    )

    # Multiplication
    results['mul'] = compare(
        "Element-wise Multiply",
        lambda: flow_a * flow_b,
        lambda: np_a * np_b
    )

    # Division
    results['div'] = compare(
        "Element-wise Divide",
        lambda: flow_a / flow_b,
        lambda: np_a / np_b
    )

    print("\n--- Reductions ---")

    # Sum
    results['sum'] = compare(
        "Sum Reduction",
        lambda: flow_a.sum(),
        lambda: np_a.sum()
    )

    # Mean
    results['mean'] = compare(
        "Mean",
        lambda: flow_a.mean(),
        lambda: np_a.mean()
    )

    print("\n--- Dot Product ---")

    results['dot'] = compare(
        "Dot Product",
        lambda: flow.dot(flow_a, flow_b),
        lambda: np.dot(np_a, np_b)
    )

    print("\n--- FMA (Flow's Strength) ---")

    results['fma'] = compare(
        "FMA (a*b + c)",
        lambda: flow.fma(flow_a, flow_b, flow_c),
        lambda: np_a * np_b + np_c
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nOperation           | Flow Wins | NumPy Wins | Notes")
    print("-" * 70)

    flow_wins = []
    numpy_wins = []

    for op, (f, n, ratio) in results.items():
        if ratio > 1:
            flow_wins.append(op)
            winner = "Flow"
        else:
            numpy_wins.append(op)
            winner = "NumPy"
        print(f"{op:<20} | {'  X' if winner == 'Flow' else '   ':<9} | "
              f"{'  X' if winner == 'NumPy' else '   ':<10} | "
              f"{max(ratio, 1/ratio):.2f}x")

    print()
    print(f"Flow wins: {', '.join(flow_wins) if flow_wins else 'None'}")
    print(f"NumPy wins: {', '.join(numpy_wins) if numpy_wins else 'None'}")


def scaling_comparison():
    """Compare scaling behavior."""
    print("\n" + "=" * 60)
    print("Scaling Comparison")
    print("=" * 60)

    sizes = [10_000, 100_000, 1_000_000, 10_000_000]

    print("\nDot Product Scaling:")
    print(f"{'Size':>12} | {'Flow (ms)':>12} | {'NumPy (ms)':>12} | {'Ratio':>8}")
    print("-" * 52)

    for n in sizes:
        np_a = np.ones(n, dtype=np.float32)
        np_b = np.ones(n, dtype=np.float32)
        flow_a = flow.ones(n)
        flow_b = flow.ones(n)

        flow_stats = benchmark(lambda: flow.dot(flow_a, flow_b), iterations=30)
        np_stats = benchmark(lambda: np.dot(np_a, np_b), iterations=30)

        ratio = np_stats['mean_ms'] / flow_stats['mean_ms']

        print(f"{n:>12,} | {flow_stats['mean_ms']:>12.4f} | "
              f"{np_stats['mean_ms']:>12.4f} | {ratio:>8.2f}x")

        del np_a, np_b, flow_a, flow_b
        gc.collect()

    print("\nFMA Scaling (Flow's advantage):")
    print(f"{'Size':>12} | {'Flow (ms)':>12} | {'NumPy (ms)':>12} | {'Ratio':>8}")
    print("-" * 52)

    for n in sizes:
        np_a = np.ones(n, dtype=np.float32)
        np_b = np.ones(n, dtype=np.float32)
        np_c = np.ones(n, dtype=np.float32)
        flow_a = flow.ones(n)
        flow_b = flow.ones(n)
        flow_c = flow.ones(n)

        flow_stats = benchmark(lambda: flow.fma(flow_a, flow_b, flow_c), iterations=30)
        np_stats = benchmark(lambda: np_a * np_b + np_c, iterations=30)

        ratio = np_stats['mean_ms'] / flow_stats['mean_ms']

        print(f"{n:>12,} | {flow_stats['mean_ms']:>12.4f} | "
              f"{np_stats['mean_ms']:>12.4f} | {ratio:>8.2f}x")

        del np_a, np_b, np_c, flow_a, flow_b, flow_c
        gc.collect()


def recommendations():
    """Print recommendations for when to use each library."""
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    print("""
When to use Bud Flow Lang:
- FMA operations (a*b + c): True hardware FMA, 1.2-1.5x faster
- Custom fused operations: Fewer memory passes
- When you need explicit SIMD control
- Cache-aware tiled operations on large arrays

When to use NumPy:
- Reductions (sum, mean): Multi-threaded BLAS is hard to beat
- Dot products: Highly optimized BLAS implementation
- Rich ecosystem: Fancy indexing, broadcasting, many operations
- Interoperability: Matplotlib, Pandas, SciPy, etc.

Best of both worlds:
- Use NumPy for data loading and preparation
- Convert to Flow for heavy computation with FMA
- Convert back to NumPy for analysis and visualization

Example workflow:
    import numpy as np
    import bud_flow_lang_py as flow

    # Load with NumPy
    data = np.load('data.npy').astype(np.float32)

    # Heavy computation with Flow
    flow_data = flow.flow(data)
    result = flow.fma(flow_data, weights, bias)

    # Back to NumPy for analysis
    result_np = result.to_numpy()
    print(f"Mean: {np.mean(result_np)}")
""")


def main():
    print("=" * 60)
    print("Bud Flow Lang vs NumPy Comparison")
    print("=" * 60)

    # Check NumPy version
    print(f"\nNumPy version: {np.__version__}")

    # Initialize Flow
    flow.initialize()
    info = flow.get_hardware_info()
    print(f"Hardware: {info['arch_family']} with {info['simd_width']*8}-bit SIMD")

    # Run comparisons
    demo_api_similarity()
    performance_comparison()
    scaling_comparison()
    recommendations()

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
