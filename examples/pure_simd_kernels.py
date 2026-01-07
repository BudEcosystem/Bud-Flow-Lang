#!/usr/bin/env python3
"""
Pure SIMD Kernels - Maximum Performance Demonstrations
======================================================

These kernels are designed to show the maximum SIMD performance by:
- Using only vectorized operations (no Python loops)
- Leveraging FMA (Fused Multiply-Add) where possible
- Avoiding unnecessary data copies
- Using efficient reduction patterns

Run: python pure_simd_kernels.py
"""

import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, '/home/bud/Desktop/bud_simd/bud_flow_lang/build')
import bud_flow_lang_py as flow

flow.initialize()


@dataclass
class Result:
    name: str
    size: int
    numpy_us: float
    bud_us: float
    speedup: float
    gelem_s: float
    passed: bool


def bench(name: str, np_fn: Callable, bud_fn: Callable,
          np_args: tuple, bud_args: tuple, size: int,
          warmup: int = 20, iters: int = 100, rtol: float = 1e-4) -> Result:
    """Benchmark helper."""
    # Reference
    ref = np_fn(*np_args)

    # Warmup
    for _ in range(warmup):
        np_fn(*np_args)
        bud_fn(*bud_args)

    # NumPy timing
    start = time.perf_counter()
    for _ in range(iters):
        _ = np_fn(*np_args)
    np_time = (time.perf_counter() - start) / iters * 1e6

    # Bud timing
    start = time.perf_counter()
    for _ in range(iters):
        result = bud_fn(*bud_args)
    bud_time = (time.perf_counter() - start) / iters * 1e6

    # Convert and check
    if isinstance(result, flow.Bunch):
        result = result.to_numpy()
    if np.isscalar(ref):
        ref = np.array([ref])
        result = np.array([result])

    passed = np.allclose(result, ref, rtol=rtol, atol=1e-6)
    speedup = np_time / bud_time if bud_time > 0 else 0
    gelem_s = size / bud_time / 1000  # Billion elements per second

    return Result(name, size, np_time, bud_time, speedup, gelem_s, passed)


def print_results(results: list):
    """Print benchmark results table."""
    print(f"\n{'Kernel':<40} | {'Size':>10} | {'NumPy':>10} | {'Bud':>10} | "
          f"{'Speedup':>8} | {'Gelem/s':>8} | {'Status':>6}")
    print("-" * 110)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<40} | {r.size:>10,} | {r.numpy_us:>8.1f}us | {r.bud_us:>8.1f}us | "
              f"{r.speedup:>7.2f}x | {r.gelem_s:>7.2f} | {status:>6}")


# =============================================================================
# Pure SIMD Kernels (Vectorized - No Python Loops)
# =============================================================================

def run_pure_simd_benchmarks():
    """Run pure SIMD benchmarks at various sizes."""
    print("=" * 110)
    print("PURE SIMD KERNELS - MAXIMUM PERFORMANCE")
    print("=" * 110)

    hw = flow.get_hardware_info()
    print(f"\nHardware: {hw['arch_family']} | SIMD: {hw['simd_width']*8}-bit | "
          f"AVX2: {hw.get('has_avx2')}, AVX-512: {hw.get('has_avx512')}")

    # Test at multiple sizes
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    all_results = []

    for size in sizes:
        print(f"\n{'='*110}")
        print(f"ARRAY SIZE: {size:,} elements ({size * 4 / 1024:.1f} KB)")
        print(f"{'='*110}")

        results = []

        # Create test data
        x_np = np.random.randn(size).astype(np.float32)
        y_np = np.random.randn(size).astype(np.float32)
        z_np = np.random.randn(size).astype(np.float32)
        x_bud = flow.Bunch.from_numpy(x_np)
        y_bud = flow.Bunch.from_numpy(y_np)
        z_bud = flow.Bunch.from_numpy(z_np)

        # Scaled for numerical stability
        x_small_np = np.clip(x_np, -5, 5)
        x_small_bud = flow.Bunch.from_numpy(x_small_np)

        # Positive data
        x_pos_np = np.abs(x_np) + 0.1
        x_pos_bud = flow.Bunch.from_numpy(x_pos_np)

        # =====================================================================
        # 1. Element-wise Arithmetic
        # =====================================================================

        # ADD
        results.append(bench(
            "ADD: x + y",
            lambda x, y: x + y,
            lambda x, y: x + y,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        # MUL
        results.append(bench(
            "MUL: x * y",
            lambda x, y: x * y,
            lambda x, y: x * y,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        # FMA (Fused Multiply-Add)
        results.append(bench(
            "FMA: x * y + z",
            lambda x, y, z: x * y + z,
            lambda x, y, z: flow.fma(x, y, z),
            (x_np, y_np, z_np), (x_bud, y_bud, z_bud), size
        ))

        # DIV
        results.append(bench(
            "DIV: x / (|y| + 0.1)",
            lambda x, y: x / (np.abs(y) + 0.1),
            lambda x, y: x / (y.abs() + 0.1),
            (x_np, y_np), (x_bud, y_bud), size
        ))

        # =====================================================================
        # 2. Transcendental Functions
        # =====================================================================

        # EXP
        results.append(bench(
            "EXP: exp(x)",
            lambda x: np.exp(x),
            lambda x: x.exp(),
            (x_small_np,), (x_small_bud,), size
        ))

        # LOG
        results.append(bench(
            "LOG: log(|x| + 0.1)",
            lambda x: np.log(x),
            lambda x: x.log(),
            (x_pos_np,), (x_pos_bud,), size
        ))

        # SIN
        results.append(bench(
            "SIN: sin(x)",
            lambda x: np.sin(x),
            lambda x: x.sin(),
            (x_np,), (x_bud,), size
        ))

        # COS
        results.append(bench(
            "COS: cos(x)",
            lambda x: np.cos(x),
            lambda x: x.cos(),
            (x_np,), (x_bud,), size
        ))

        # TANH
        results.append(bench(
            "TANH: tanh(x)",
            lambda x: np.tanh(x),
            lambda x: x.tanh(),
            (x_small_np,), (x_small_bud,), size
        ))

        # SQRT
        results.append(bench(
            "SQRT: sqrt(|x| + 0.1)",
            lambda x: np.sqrt(x),
            lambda x: x.sqrt(),
            (x_pos_np,), (x_pos_bud,), size
        ))

        # =====================================================================
        # 3. Reductions
        # =====================================================================

        # SUM
        results.append(bench(
            "SUM: sum(x)",
            lambda x: np.sum(x),
            lambda x: x.sum(),
            (x_np,), (x_bud,), size
        ))

        # DOT
        results.append(bench(
            "DOT: dot(x, y)",
            lambda x, y: np.dot(x, y),
            lambda x, y: x.dot(y),
            (x_np, y_np), (x_bud, y_bud), size
        ))

        # MIN
        results.append(bench(
            "MIN: min(x)",
            lambda x: np.min(x),
            lambda x: x.min(),
            (x_np,), (x_bud,), size
        ))

        # MAX
        results.append(bench(
            "MAX: max(x)",
            lambda x: np.max(x),
            lambda x: x.max(),
            (x_np,), (x_bud,), size
        ))

        # MEAN
        results.append(bench(
            "MEAN: mean(x)",
            lambda x: np.mean(x),
            lambda x: x.mean(),
            (x_np,), (x_bud,), size
        ))

        # =====================================================================
        # 4. Compound Operations (Still Vectorized)
        # =====================================================================

        # Softmax (Stable)
        results.append(bench(
            "SOFTMAX: exp(x-max)/sum",
            lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))),
            lambda x: (lambda shifted: shifted.exp() / shifted.exp().sum())(x - x.max()),
            (x_small_np,), (x_small_bud,), size,
            rtol=1e-3
        ))

        # Sigmoid
        results.append(bench(
            "SIGMOID: 1/(1+exp(-x))",
            lambda x: 1.0 / (1.0 + np.exp(-x)),
            lambda x: flow.Bunch.ones(len(x)) / ((x * (-1.0)).exp() + 1.0),
            (x_small_np,), (x_small_bud,), size,
            rtol=1e-4
        ))

        # GELU Approximation
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        results.append(bench(
            "GELU: 0.5*x*(1+tanh(...))",
            lambda x: 0.5 * x * (1 + np.tanh(sqrt_2_pi * (x + 0.044715 * x**3))),
            lambda x: x * ((x + x * x * x * 0.044715) * sqrt_2_pi).tanh() * 0.5 + x * 0.5,
            (x_small_np,), (x_small_bud,), size,
            rtol=1e-3
        ))

        # Layer Norm (without gamma/beta)
        results.append(bench(
            "LAYERNORM: (x-mean)/std",
            lambda x: (x - np.mean(x)) / (np.std(x) + 1e-5),
            lambda x: (lambda centered: centered / ((centered * centered).sum() / len(x) + 1e-5) ** 0.5)(x - x.mean()),
            (x_np,), (x_bud,), size,
            rtol=1e-3
        ))

        # RMS Norm
        results.append(bench(
            "RMSNORM: x / sqrt(mean(x^2))",
            lambda x: x / (np.sqrt(np.mean(x**2)) + 1e-5),
            lambda x: x / (((x * x).sum() / len(x)) ** 0.5 + 1e-5),
            (x_np,), (x_bud,), size,
            rtol=1e-3
        ))

        # Swish/SiLU
        results.append(bench(
            "SWISH: x * sigmoid(x)",
            lambda x: x / (1.0 + np.exp(-x)),
            lambda x: x / ((x * (-1.0)).exp() + 1.0),
            (x_small_np,), (x_small_bud,), size,
            rtol=1e-4
        ))

        # Hard Swish
        results.append(bench(
            "HARDSWISH: x*clip(x+3,0,6)/6",
            lambda x: x * np.clip(x + 3, 0, 6) / 6,
            lambda x: x * flow.clamp(x + 3.0, 0.0, 6.0) / 6.0,
            (x_np,), (x_bud,), size
        ))

        # Leaky ReLU
        results.append(bench(
            "LEAKYRELU: max(0.01x, x)",
            lambda x: np.where(x > 0, x, 0.01 * x),
            lambda x: flow.clamp(x, 0.0, float('inf')) + flow.clamp(x, float('-inf'), 0.0) * 0.01,
            (x_np,), (x_bud,), size
        ))

        # Cosine Similarity
        results.append(bench(
            "COSINE_SIM: dot/(norm*norm)",
            lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)),
            lambda x, y: x.dot(y) / ((x.dot(x) ** 0.5) * (y.dot(y) ** 0.5)),
            (x_np, y_np), (x_bud, y_bud), size
        ))

        # Euclidean Distance
        results.append(bench(
            "EUCLIDEAN: sqrt(dot(x-y, x-y))",
            lambda x, y: np.sqrt(np.dot(x - y, x - y)),
            lambda x, y: ((x - y).dot(x - y)) ** 0.5,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        print_results(results)
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 110)
    print("OVERALL SUMMARY")
    print("=" * 110)

    passed = sum(1 for r in all_results if r.passed)
    avg_speedup = np.mean([r.speedup for r in all_results])
    median_speedup = np.median([r.speedup for r in all_results])
    max_speedup = max(r.speedup for r in all_results)
    max_kernel = [r for r in all_results if r.speedup == max_speedup][0]

    # Best per-kernel across sizes
    kernel_names = list(set(r.name for r in all_results))
    print("\nBest speedup per kernel (across all sizes):")
    kernel_best = []
    for name in sorted(kernel_names):
        best = max(r.speedup for r in all_results if r.name == name)
        kernel_best.append((name, best))

    kernel_best.sort(key=lambda x: x[1], reverse=True)
    for name, speedup in kernel_best[:15]:
        status = "FAST" if speedup > 1.5 else ("OK" if speedup >= 1.0 else "SLOW")
        print(f"  {name:<40}: {speedup:>6.2f}x [{status}]")

    # Summary stats
    print(f"\nTotal benchmarks: {len(all_results)}")
    print(f"Accuracy passed: {passed}/{len(all_results)} ({100*passed/len(all_results):.1f}%)")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Median speedup: {median_speedup:.2f}x")
    print(f"Max speedup: {max_speedup:.2f}x ({max_kernel.name} @ {max_kernel.size:,})")

    # Throughput summary
    best_throughput = max(r.gelem_s for r in all_results)
    best_tp_kernel = [r for r in all_results if r.gelem_s == best_throughput][0]
    print(f"\nPeak throughput: {best_throughput:.2f} Gelem/s ({best_tp_kernel.name} @ {best_tp_kernel.size:,})")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    run_pure_simd_benchmarks()
