#!/usr/bin/env python3
"""
JIT-Compiled SIMD Kernels with @flow.kernel Decorator
======================================================

This example demonstrates proper usage of the @flow.kernel decorator to trace
Python functions and compile them to optimized SIMD code.

IMPORTANT: The @flow.kernel decorator currently supports:
- Binary ops: +, -, *, /
- Unary ops: neg, abs, sqrt, exp, log, sin, cos, tanh
- Automatic FMA fusion: a * b + c -> fused multiply-add
- All inputs must be Bunch arrays (no scalar constants inside kernels)

Operations like ** (power) or scalar constant creation inside kernels
will fall back to eager mode (still works, but not JIT-optimized).

Run: python jit_compiled_kernels.py
"""

import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, List

sys.path.insert(0, '/home/bud/Desktop/bud_simd/bud_flow_lang/build')
import bud_flow_lang_py as flow

flow.initialize()


@dataclass
class BenchResult:
    name: str
    size: int
    numpy_us: float
    bud_us: float
    speedup: float
    passed: bool


def benchmark_kernel(name: str, np_fn: Callable, kernel_fn: Callable,
                     np_args: tuple, kernel_args: tuple, size: int,
                     warmup: int = 20, iters: int = 100, rtol: float = 1e-4) -> BenchResult:
    """Benchmark a JIT-compiled kernel against NumPy."""
    # Reference
    ref = np_fn(*np_args)

    # Warmup (first call compiles the kernel)
    for _ in range(warmup):
        np_fn(*np_args)
        kernel_fn(*kernel_args)

    # NumPy timing
    start = time.perf_counter()
    for _ in range(iters):
        _ = np_fn(*np_args)
    np_time = (time.perf_counter() - start) / iters * 1e6

    # Kernel timing (uses cached compiled kernel)
    start = time.perf_counter()
    for _ in range(iters):
        result = kernel_fn(*kernel_args)
    bud_time = (time.perf_counter() - start) / iters * 1e6

    # Convert and check
    if isinstance(result, flow.Bunch):
        result = result.to_numpy()
    if np.isscalar(ref):
        ref = np.array([ref])
        result = np.array([result]) if np.isscalar(result) else result

    passed = np.allclose(result, ref, rtol=rtol, atol=1e-6)
    speedup = np_time / bud_time if bud_time > 0 else 0

    return BenchResult(name, size, np_time, bud_time, speedup, passed)


def print_results(results: List[BenchResult]):
    """Print benchmark results table."""
    print(f"\n{'Kernel':<45} | {'Size':>10} | {'NumPy':>10} | {'Bud':>10} | "
          f"{'Speedup':>8} | {'Status':>6}")
    print("-" * 105)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<45} | {r.size:>10,} | {r.numpy_us:>8.1f}us | {r.bud_us:>8.1f}us | "
              f"{r.speedup:>7.2f}x | {status:>6}")


# =============================================================================
# JIT-COMPILED KERNELS USING @flow.kernel DECORATOR
# =============================================================================
# These kernels use ONLY operations that are fully JIT-compiled:
# - Binary: +, -, *, /
# - Unary: neg, abs, sqrt, exp, log, sin, cos, tanh
# - Automatic FMA fusion for a * b + c patterns
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Basic Arithmetic (Fully JIT-Compiled)
# -----------------------------------------------------------------------------

@flow.kernel
def kernel_add(x, y):
    """Element-wise addition: x + y"""
    return x + y


@flow.kernel
def kernel_sub(x, y):
    """Element-wise subtraction: x - y"""
    return x - y


@flow.kernel
def kernel_mul(x, y):
    """Element-wise multiplication: x * y"""
    return x * y


@flow.kernel
def kernel_div(x, y):
    """Element-wise division: x / y"""
    return x / y


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_fma(x, y, z):
    """Fused multiply-add: x * y + z (automatically fused to FMA)"""
    return x * y + z


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_fnma(x, y, z):
    """Fused negative multiply-add: z - x * y"""
    return z - x * y


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_axpy(a, x, y):
    """AXPY: a * x + y"""
    return a * x + y


# -----------------------------------------------------------------------------
# 2. Polynomial Kernels (Fully JIT-Compiled with FMA fusion)
# -----------------------------------------------------------------------------

@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_quadratic(x, a, b, c):
    """Quadratic: a*x*x + b*x + c (fuses to FMAs)"""
    return a * x * x + b * x + c


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_horner_quadratic(x, a, b, c):
    """Horner's method for quadratic: (a*x + b)*x + c"""
    return (a * x + b) * x + c


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_horner_cubic(x, a, b, c, d):
    """Horner's method for cubic: ((a*x + b)*x + c)*x + d"""
    return ((a * x + b) * x + c) * x + d


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_horner_quartic(x, a, b, c, d, e):
    """Horner's method for quartic: (((a*x + b)*x + c)*x + d)*x + e"""
    return (((a * x + b) * x + c) * x + d) * x + e


# -----------------------------------------------------------------------------
# 3. Transcendental Functions (Fully JIT-Compiled)
# -----------------------------------------------------------------------------

@flow.kernel
def kernel_exp(x):
    """Exponential: exp(x)"""
    return x.exp()


@flow.kernel
def kernel_log(x):
    """Natural logarithm: log(x)"""
    return x.log()


@flow.kernel
def kernel_sqrt(x):
    """Square root: sqrt(x)"""
    return x.sqrt()


@flow.kernel
def kernel_sin(x):
    """Sine: sin(x)"""
    return x.sin()


@flow.kernel
def kernel_cos(x):
    """Cosine: cos(x)"""
    return x.cos()


@flow.kernel
def kernel_tanh(x):
    """Hyperbolic tangent: tanh(x)"""
    return x.tanh()


@flow.kernel
def kernel_abs(x):
    """Absolute value: |x|"""
    return x.abs()


# -----------------------------------------------------------------------------
# 4. Combined Transcendental + Arithmetic (JIT Fused)
# -----------------------------------------------------------------------------

@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_exp_scaled(x, scale):
    """Scaled exponential: exp(x) * scale"""
    return x.exp() * scale


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_log_shifted(x, shift):
    """Shifted logarithm: log(x) + shift"""
    return x.log() + shift


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_sin_cos_sum(x):
    """sin(x) + cos(x)"""
    return x.sin() + x.cos()


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_tanh_squared(x):
    """tanh(x) * tanh(x)"""
    t = x.tanh()
    return t * t


# -----------------------------------------------------------------------------
# 5. Multi-Step Compound Operations (JIT Fused)
# -----------------------------------------------------------------------------

@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_affine(x, scale, bias):
    """Affine transform: x * scale + bias"""
    return x * scale + bias


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_linear_combo(a, b, wa, wb):
    """Linear combination: wa * a + wb * b"""
    return wa * a + wb * b


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_squared_diff(x, y):
    """Squared difference: (x - y) * (x - y)"""
    diff = x - y
    return diff * diff


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_normalize(x, mean, inv_std):
    """Normalization: (x - mean) * inv_std"""
    return (x - mean) * inv_std


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_normalize_affine(x, mean, inv_std, gamma, beta):
    """Affine normalization: (x - mean) * inv_std * gamma + beta"""
    return (x - mean) * inv_std * gamma + beta


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_lerp(a, b, t):
    """Linear interpolation: a + t * (b - a)"""
    return a + t * (b - a)


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_ema_step(prev, curr, alpha, one_minus_alpha):
    """EMA update: alpha * curr + one_minus_alpha * prev"""
    return alpha * curr + one_minus_alpha * prev


# -----------------------------------------------------------------------------
# 6. Deep Fusion Chains (Multiple FMA opportunities)
# -----------------------------------------------------------------------------

@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_chain_2fma(x, a1, b1, a2, b2):
    """Chain of 2 affine transforms: a2 * (a1 * x + b1) + b2"""
    t = a1 * x + b1
    return a2 * t + b2


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_chain_3fma(x, a1, b1, a2, b2, a3, b3):
    """Chain of 3 affine transforms"""
    t1 = a1 * x + b1
    t2 = a2 * t1 + b2
    return a3 * t2 + b3


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_bilinear(x, y, a, b, c, d):
    """Bilinear: a*x + b*y + c*x*y + d"""
    return a * x + b * y + c * x * y + d


# -----------------------------------------------------------------------------
# 7. Numerical Stability Patterns
# -----------------------------------------------------------------------------

@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_safe_div(x, y, epsilon):
    """Safe division: x / (y + epsilon)"""
    return x / (y + epsilon)


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_log_safe(x, epsilon):
    """Safe log: log(x + epsilon)"""
    return (x + epsilon).log()


@flow.kernel(opt_level=3, enable_fusion=True)
def kernel_sqrt_safe(x, epsilon):
    """Safe sqrt: sqrt(x + epsilon)"""
    return (x + epsilon).sqrt()


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_kernel_benchmarks():
    """Run comprehensive benchmarks for all JIT-compiled kernels."""
    print("=" * 105)
    print("JIT-COMPILED KERNELS BENCHMARK - @flow.kernel Decorator")
    print("=" * 105)

    hw = flow.get_hardware_info()
    print(f"\nHardware: {hw['arch_family']} | SIMD: {hw['simd_width']*8}-bit | "
          f"AVX2: {hw.get('has_avx2')}, AVX-512: {hw.get('has_avx512')}")

    sizes = [1_000, 10_000, 100_000, 1_000_000]
    all_results = []

    for size in sizes:
        print(f"\n{'=' * 105}")
        print(f"ARRAY SIZE: {size:,} elements ({size * 4 / 1024:.1f} KB)")
        print(f"{'=' * 105}")

        results = []

        # Create test data
        x_np = np.random.randn(size).astype(np.float32)
        y_np = np.random.randn(size).astype(np.float32)
        z_np = np.random.randn(size).astype(np.float32)

        x_bud = flow.Bunch.from_numpy(x_np)
        y_bud = flow.Bunch.from_numpy(y_np)
        z_bud = flow.Bunch.from_numpy(z_np)

        # Clipped for numerical stability with exp/tanh
        x_small_np = np.clip(x_np, -3, 3)
        x_small_bud = flow.Bunch.from_numpy(x_small_np)

        # Positive for log/sqrt
        x_pos_np = np.abs(x_np) + 0.1
        x_pos_bud = flow.Bunch.from_numpy(x_pos_np)
        y_pos_np = np.abs(y_np) + 0.1
        y_pos_bud = flow.Bunch.from_numpy(y_pos_np)

        # Coefficient arrays (all same value, broadcast-compatible)
        ones = flow.ones(size)
        halves = flow.full(size, 0.5)
        coef_a = flow.full(size, 1.5)
        coef_b = flow.full(size, 0.5)
        coef_c = flow.full(size, 0.25)
        coef_d = flow.full(size, 0.125)
        coef_e = flow.full(size, 0.0625)
        epsilon = flow.full(size, 1e-6)

        a_np = np.full(size, 1.5, dtype=np.float32)
        b_np = np.full(size, 0.5, dtype=np.float32)
        c_np = np.full(size, 0.25, dtype=np.float32)
        d_np = np.full(size, 0.125, dtype=np.float32)
        e_np = np.full(size, 0.0625, dtype=np.float32)
        eps_np = np.full(size, 1e-6, dtype=np.float32)

        # =====================================================================
        # 1. Basic Arithmetic
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel ADD",
            lambda x, y: x + y,
            kernel_add,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        results.append(benchmark_kernel(
            "@kernel SUB",
            lambda x, y: x - y,
            kernel_sub,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        results.append(benchmark_kernel(
            "@kernel MUL",
            lambda x, y: x * y,
            kernel_mul,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        results.append(benchmark_kernel(
            "@kernel DIV",
            lambda x, y: x / y,
            kernel_div,
            (x_np, y_pos_np), (x_bud, y_pos_bud), size
        ))

        results.append(benchmark_kernel(
            "@kernel FMA: x*y+z",
            lambda x, y, z: x * y + z,
            kernel_fma,
            (x_np, y_np, z_np), (x_bud, y_bud, z_bud), size
        ))

        results.append(benchmark_kernel(
            "@kernel FNMA: z-x*y",
            lambda x, y, z: z - x * y,
            kernel_fnma,
            (x_np, y_np, z_np), (x_bud, y_bud, z_bud), size
        ))

        # =====================================================================
        # 2. Polynomial Operations (Horner's Method)
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel HORNER_QUADRATIC: (ax+b)x+c",
            lambda x, a, b, c: (a * x + b) * x + c,
            kernel_horner_quadratic,
            (x_small_np, a_np, b_np, c_np), (x_small_bud, coef_a, coef_b, coef_c), size
        ))

        results.append(benchmark_kernel(
            "@kernel HORNER_CUBIC: ((ax+b)x+c)x+d",
            lambda x, a, b, c, d: ((a * x + b) * x + c) * x + d,
            kernel_horner_cubic,
            (x_small_np, a_np, b_np, c_np, d_np),
            (x_small_bud, coef_a, coef_b, coef_c, coef_d), size
        ))

        results.append(benchmark_kernel(
            "@kernel HORNER_QUARTIC: (((ax+b)x+c)x+d)x+e",
            lambda x, a, b, c, d, e: (((a * x + b) * x + c) * x + d) * x + e,
            kernel_horner_quartic,
            (x_small_np, a_np, b_np, c_np, d_np, e_np),
            (x_small_bud, coef_a, coef_b, coef_c, coef_d, coef_e), size
        ))

        # =====================================================================
        # 3. Transcendental Functions
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel EXP",
            lambda x: np.exp(x),
            kernel_exp,
            (x_small_np,), (x_small_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel LOG",
            lambda x: np.log(x),
            kernel_log,
            (x_pos_np,), (x_pos_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel SQRT",
            lambda x: np.sqrt(x),
            kernel_sqrt,
            (x_pos_np,), (x_pos_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel SIN",
            lambda x: np.sin(x),
            kernel_sin,
            (x_np,), (x_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel COS",
            lambda x: np.cos(x),
            kernel_cos,
            (x_np,), (x_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel TANH",
            lambda x: np.tanh(x),
            kernel_tanh,
            (x_small_np,), (x_small_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel ABS",
            lambda x: np.abs(x),
            kernel_abs,
            (x_np,), (x_bud,), size
        ))

        # =====================================================================
        # 4. Combined Transcendental + Arithmetic
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel EXP_SCALED: exp(x)*scale",
            lambda x, s: np.exp(x) * s,
            kernel_exp_scaled,
            (x_small_np, b_np), (x_small_bud, coef_b), size
        ))

        results.append(benchmark_kernel(
            "@kernel SIN_COS_SUM: sin(x)+cos(x)",
            lambda x: np.sin(x) + np.cos(x),
            kernel_sin_cos_sum,
            (x_np,), (x_bud,), size
        ))

        results.append(benchmark_kernel(
            "@kernel TANH_SQUARED: tanh(x)^2",
            lambda x: np.tanh(x) ** 2,
            kernel_tanh_squared,
            (x_small_np,), (x_small_bud,), size
        ))

        # =====================================================================
        # 5. Multi-Step Compound Operations
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel AFFINE: x*scale+bias",
            lambda x, s, b: x * s + b,
            kernel_affine,
            (x_np, a_np, b_np), (x_bud, coef_a, coef_b), size
        ))

        results.append(benchmark_kernel(
            "@kernel LINEAR_COMBO: wa*a+wb*b",
            lambda a, b, wa, wb: wa * a + wb * b,
            kernel_linear_combo,
            (x_np, y_np, a_np, b_np), (x_bud, y_bud, coef_a, coef_b), size
        ))

        results.append(benchmark_kernel(
            "@kernel SQUARED_DIFF: (x-y)^2",
            lambda x, y: (x - y) ** 2,
            kernel_squared_diff,
            (x_np, y_np), (x_bud, y_bud), size
        ))

        mean_np = np.full(size, float(np.mean(x_np)), dtype=np.float32)
        inv_std_np = np.full(size, 1.0 / (float(np.std(x_np)) + 1e-6), dtype=np.float32)
        mean_bud = flow.full(size, float(np.mean(x_np)))
        inv_std_bud = flow.full(size, 1.0 / (float(np.std(x_np)) + 1e-6))

        results.append(benchmark_kernel(
            "@kernel NORMALIZE: (x-mean)*inv_std",
            lambda x, m, s: (x - m) * s,
            kernel_normalize,
            (x_np, mean_np, inv_std_np), (x_bud, mean_bud, inv_std_bud), size
        ))

        gamma_np = np.full(size, 1.2, dtype=np.float32)
        beta_np = np.full(size, 0.1, dtype=np.float32)
        gamma_bud = flow.full(size, 1.2)
        beta_bud = flow.full(size, 0.1)

        results.append(benchmark_kernel(
            "@kernel NORMALIZE_AFFINE: (x-m)*s*g+b",
            lambda x, m, s, g, b: (x - m) * s * g + b,
            kernel_normalize_affine,
            (x_np, mean_np, inv_std_np, gamma_np, beta_np),
            (x_bud, mean_bud, inv_std_bud, gamma_bud, beta_bud), size
        ))

        t_np = np.full(size, 0.3, dtype=np.float32)
        t_bud = flow.full(size, 0.3)

        results.append(benchmark_kernel(
            "@kernel LERP: a+t*(b-a)",
            lambda a, b, t: a + t * (b - a),
            kernel_lerp,
            (x_np, y_np, t_np), (x_bud, y_bud, t_bud), size
        ))

        alpha = flow.full(size, 0.1)
        one_minus_alpha = flow.full(size, 0.9)
        alpha_np = np.full(size, 0.1, dtype=np.float32)
        one_minus_np = np.full(size, 0.9, dtype=np.float32)

        results.append(benchmark_kernel(
            "@kernel EMA_STEP: a*curr+(1-a)*prev",
            lambda prev, curr, a, oma: a * curr + oma * prev,
            kernel_ema_step,
            (x_np, y_np, alpha_np, one_minus_np),
            (x_bud, y_bud, alpha, one_minus_alpha), size
        ))

        # =====================================================================
        # 6. Deep Fusion Chains
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel CHAIN_2FMA: a2*(a1*x+b1)+b2",
            lambda x, a1, b1, a2, b2: a2 * (a1 * x + b1) + b2,
            kernel_chain_2fma,
            (x_np, a_np, b_np, c_np, d_np),
            (x_bud, coef_a, coef_b, coef_c, coef_d), size
        ))

        results.append(benchmark_kernel(
            "@kernel CHAIN_3FMA: a3*(a2*(a1*x+b1)+b2)+b3",
            lambda x, a1, b1, a2, b2, a3, b3: a3 * (a2 * (a1 * x + b1) + b2) + b3,
            kernel_chain_3fma,
            (x_np, a_np, b_np, c_np, d_np, e_np, eps_np),
            (x_bud, coef_a, coef_b, coef_c, coef_d, coef_e, epsilon), size
        ))

        results.append(benchmark_kernel(
            "@kernel BILINEAR: ax+by+cxy+d",
            lambda x, y, a, b, c, d: a * x + b * y + c * x * y + d,
            kernel_bilinear,
            (x_np, y_np, a_np, b_np, c_np, d_np),
            (x_bud, y_bud, coef_a, coef_b, coef_c, coef_d), size
        ))

        # =====================================================================
        # 7. Numerical Stability Patterns
        # =====================================================================

        results.append(benchmark_kernel(
            "@kernel SAFE_DIV: x/(y+eps)",
            lambda x, y, e: x / (y + e),
            kernel_safe_div,
            (x_np, y_np, eps_np), (x_bud, y_bud, epsilon), size
        ))

        results.append(benchmark_kernel(
            "@kernel LOG_SAFE: log(x+eps)",
            lambda x, e: np.log(x + e),
            kernel_log_safe,
            (x_pos_np, eps_np), (x_pos_bud, epsilon), size
        ))

        results.append(benchmark_kernel(
            "@kernel SQRT_SAFE: sqrt(x+eps)",
            lambda x, e: np.sqrt(x + e),
            kernel_sqrt_safe,
            (x_pos_np, eps_np), (x_pos_bud, epsilon), size
        ))

        print_results(results)
        all_results.extend(results)

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 105)
    print("OVERALL SUMMARY")
    print("=" * 105)

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
    for name, speedup in kernel_best[:20]:
        status = "FAST" if speedup > 1.5 else ("OK" if speedup >= 1.0 else "SLOW")
        print(f"  {name:<45}: {speedup:>6.2f}x [{status}]")

    print(f"\nTotal benchmarks: {len(all_results)}")
    print(f"Accuracy passed: {passed}/{len(all_results)} ({100*passed/len(all_results):.1f}%)")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Median speedup: {median_speedup:.2f}x")
    print(f"Max speedup: {max_speedup:.2f}x ({max_kernel.name} @ {max_kernel.size:,})")

    # Count fast kernels
    fast_count = sum(1 for _, s in kernel_best if s > 1.5)
    ok_count = sum(1 for _, s in kernel_best if 1.0 <= s <= 1.5)
    slow_count = sum(1 for _, s in kernel_best if s < 1.0)
    print(f"\nKernel performance distribution:")
    print(f"  FAST (>1.5x): {fast_count}/{len(kernel_best)}")
    print(f"  OK (1-1.5x): {ok_count}/{len(kernel_best)}")
    print(f"  SLOW (<1x): {slow_count}/{len(kernel_best)}")

    print("\n" + "=" * 105)


# =============================================================================
# KERNEL TRACING DEMONSTRATION
# =============================================================================

def demonstrate_kernel_tracing():
    """Demonstrate how @flow.kernel traces and compiles functions."""
    print("\n" + "=" * 105)
    print("KERNEL TRACING DEMONSTRATION")
    print("=" * 105)

    @flow.kernel(opt_level=3, enable_fusion=True)
    def traced_fma(a, b, c):
        """This function will be traced on first call."""
        temp = a * b
        return temp + c

    # Create test data
    n = 1000
    a = flow.arange(n)
    b = flow.ones(n)
    c = flow.full(n, 0.5)

    print("\n1. First call - triggers tracing and compilation:")
    start = time.perf_counter()
    result1 = traced_fma(a, b, c)
    first_call_time = (time.perf_counter() - start) * 1000
    print(f"   First call time: {first_call_time:.3f} ms (includes compilation)")

    print("\n2. Second call - uses cached compiled kernel:")
    start = time.perf_counter()
    result2 = traced_fma(a, b, c)
    second_call_time = (time.perf_counter() - start) * 1000
    print(f"   Second call time: {second_call_time:.4f} ms (cached)")

    print(f"\n3. Speedup from caching: {first_call_time/second_call_time:.1f}x")

    # Verify results
    expected = (np.arange(n) * 1.0 + 0.5).astype(np.float32)
    result_np = result2.to_numpy()
    matches = np.allclose(result_np, expected, rtol=1e-5)
    print(f"\n4. Result verification: {'PASS' if matches else 'FAIL'}")
    print(f"   Sample output: result[:5] = {list(result_np[:5])}")
    print(f"   Expected:      expected[:5] = {list(expected[:5])}")

    print("\n" + "-" * 60)
    print("Key Points about @flow.kernel:")
    print("-" * 60)
    print("- First call traces the function and compiles to SIMD")
    print("- Subsequent calls use the cached compiled kernel")
    print("- Supports: +, -, *, /, exp, log, sqrt, sin, cos, tanh, abs")
    print("- Automatic FMA fusion for a*b+c patterns")
    print("- All inputs must be Bunch arrays (no scalar constants inside)")
    print("=" * 105)


if __name__ == "__main__":
    demonstrate_kernel_tracing()
    run_kernel_benchmarks()
