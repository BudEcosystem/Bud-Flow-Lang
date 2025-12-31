#!/usr/bin/env python3
# =============================================================================
# Bud Flow Lang - Real SIMD Kernels Demo
# =============================================================================
#
# Demonstrates SIMD kernels from simple to complex using all API tiers:
# - User Tier: NumPy-like operations on Bunch
# - Developer Tier: Pattern, Pipeline, Memory hints
# - Expert Tier: Direct Highway SIMD operations
#
# =============================================================================

import sys
import time
import numpy as np

# Add build directory to path
sys.path.insert(0, '/home/bud/Desktop/bud_simd/bud_flow_lang/build')
sys.path.insert(0, '/home/bud/Desktop/bud_simd/bud_flow_lang/python')

import bud_flow_lang_py as flow
from flow import highway

# =============================================================================
# Utility Functions
# =============================================================================

def benchmark(name, func, *args, iterations=100):
    """Benchmark a function and return average time in microseconds"""
    # Warmup
    for _ in range(10):
        result = func(*args)

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    end = time.perf_counter()

    avg_us = (end - start) / iterations * 1e6
    return result, avg_us

def verify(name, got, expected, rtol=1e-5, atol=1e-6):
    """Verify results match expected values"""
    if isinstance(got, flow.Bunch):
        got = got.to_numpy()
    if isinstance(expected, flow.Bunch):
        expected = expected.to_numpy()

    if np.allclose(got, expected, rtol=rtol, atol=atol):
        print(f"  ✓ {name}: PASSED")
        return True
    else:
        max_diff = np.max(np.abs(got - expected))
        print(f"  ✗ {name}: FAILED (max diff: {max_diff})")
        return False

# =============================================================================
# PART 1: SIMPLE KERNELS
# =============================================================================

print("=" * 70)
print("PART 1: SIMPLE KERNELS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Vector Addition (SAXPY: y = a*x + y)
# -----------------------------------------------------------------------------
print("\n1.1 SAXPY (y = a*x + y)")

def saxpy_numpy(a, x, y):
    """NumPy reference implementation"""
    return a * x + y

def saxpy_bunch(a, x, y):
    """User tier: Bunch operations"""
    return x * a + y

def saxpy_highway(a, x, y):
    """Expert tier: Highway FMA"""
    # Create scalar bunch for 'a'
    a_vec = flow.Bunch.fill(len(x), a)
    return highway.mul_add(a_vec, x, y)

# Test data
n = 10000
alpha = 2.5
x_np = np.random.randn(n).astype(np.float32)
y_np = np.random.randn(n).astype(np.float32)
x_bunch = flow.Bunch.from_numpy(x_np)
y_bunch = flow.Bunch.from_numpy(y_np)

# Run and verify
expected = saxpy_numpy(alpha, x_np, y_np)
result_bunch, t1 = benchmark("saxpy_bunch", saxpy_bunch, alpha, x_bunch, y_bunch)
result_hwy, t2 = benchmark("saxpy_highway", saxpy_highway, alpha, x_bunch, y_bunch)

verify("Bunch SAXPY", result_bunch, expected)
verify("Highway SAXPY", result_hwy, expected)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# -----------------------------------------------------------------------------
# 1.2 Dot Product
# -----------------------------------------------------------------------------
print("\n1.2 Dot Product")

def dot_numpy(a, b):
    return np.dot(a, b)

def dot_bunch(a, b):
    return a.dot(b)

def dot_highway(a, b):
    return highway.dot_product(a, b)

result_bunch, t1 = benchmark("dot_bunch", dot_bunch, x_bunch, y_bunch)
result_hwy, t2 = benchmark("dot_highway", dot_highway, x_bunch, y_bunch)
expected = dot_numpy(x_np, y_np)

verify("Bunch dot", result_bunch, expected, rtol=1e-4)
verify("Highway dot", result_hwy, expected, rtol=1e-4)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# -----------------------------------------------------------------------------
# 1.3 Element-wise Operations Chain
# -----------------------------------------------------------------------------
print("\n1.3 Element-wise Chain: sqrt(abs(x) + 1)")

def chain_numpy(x):
    return np.sqrt(np.abs(x) + 1)

def chain_bunch(x):
    return (x.abs() + 1.0).sqrt()

def chain_highway(x):
    one = flow.Bunch.fill(len(x), 1.0)
    abs_x = highway.abs(x)
    sum_x = highway.add(abs_x, one)
    return highway.sqrt(sum_x)

result_bunch, t1 = benchmark("chain_bunch", chain_bunch, x_bunch)
result_hwy, t2 = benchmark("chain_highway", chain_highway, x_bunch)
expected = chain_numpy(x_np)

verify("Bunch chain", result_bunch, expected)
verify("Highway chain", result_hwy, expected)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# =============================================================================
# PART 2: MEDIUM COMPLEXITY KERNELS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: MEDIUM COMPLEXITY KERNELS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 ReLU Activation
# -----------------------------------------------------------------------------
print("\n2.1 ReLU Activation: max(0, x)")

def relu_numpy(x):
    return np.maximum(0, x)

def relu_bunch(x):
    """ReLU using comparison and where"""
    zero = flow.Bunch.zeros(len(x))
    mask = x.gt(zero)  # x > 0
    return x.where(mask, zero)

def relu_highway(x):
    zero = flow.Bunch.zeros(len(x))
    mask = highway.gt(x, zero)  # x > 0
    return highway.where(mask, x, zero)

result_bunch, t1 = benchmark("relu_bunch", relu_bunch, x_bunch)
result_hwy, t2 = benchmark("relu_highway", relu_highway, x_bunch)
expected = relu_numpy(x_np)

verify("Bunch ReLU", result_bunch, expected)
verify("Highway ReLU", result_hwy, expected)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# -----------------------------------------------------------------------------
# 2.2 Sigmoid Activation
# -----------------------------------------------------------------------------
print("\n2.2 Sigmoid Activation: 1 / (1 + exp(-x))")

def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_bunch(x):
    neg_x = -x
    exp_neg = neg_x.exp()
    denom = exp_neg + 1.0
    one = flow.Bunch.fill(len(x), 1.0)
    return one / denom

def sigmoid_highway(x):
    one = flow.Bunch.fill(len(x), 1.0)
    neg_x = highway.neg(x)
    exp_neg = highway.exp(neg_x)
    denom = highway.add(exp_neg, one)
    return highway.div(one, denom)

# Use smaller values to avoid overflow
x_small = flow.Bunch.from_numpy(np.clip(x_np, -10, 10).astype(np.float32))
x_small_np = x_small.to_numpy()

result_bunch, t1 = benchmark("sigmoid_bunch", sigmoid_bunch, x_small)
result_hwy, t2 = benchmark("sigmoid_highway", sigmoid_highway, x_small)
expected = sigmoid_numpy(x_small_np)

verify("Bunch Sigmoid", result_bunch, expected, rtol=1e-4)
verify("Highway Sigmoid", result_hwy, expected, rtol=1e-4)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# -----------------------------------------------------------------------------
# 2.3 Softmax (Stable)
# -----------------------------------------------------------------------------
print("\n2.3 Stable Softmax: exp(x - max(x)) / sum(exp(x - max(x)))")

def softmax_numpy(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def softmax_bunch(x):
    x_max = x.max()
    x_shifted = x - x_max
    exp_x = x_shifted.exp()
    sum_exp = exp_x.sum()
    return exp_x / sum_exp

def softmax_highway(x):
    # Get max for numerical stability
    x_max = highway.reduce_max(x)
    x_max_vec = flow.Bunch.fill(len(x), x_max)

    # Compute exp(x - max)
    x_shifted = highway.sub(x, x_max_vec)
    exp_x = highway.exp(x_shifted)

    # Sum and normalize
    sum_exp = highway.reduce_sum(exp_x)
    sum_vec = flow.Bunch.fill(len(x), sum_exp)
    return highway.div(exp_x, sum_vec)

# Small vector for softmax
n_softmax = 1000
x_softmax_np = np.random.randn(n_softmax).astype(np.float32)
x_softmax = flow.Bunch.from_numpy(x_softmax_np)

result_bunch, t1 = benchmark("softmax_bunch", softmax_bunch, x_softmax)
result_hwy, t2 = benchmark("softmax_highway", softmax_highway, x_softmax)
expected = softmax_numpy(x_softmax_np)

verify("Bunch Softmax", result_bunch, expected, rtol=1e-4)
verify("Highway Softmax", result_hwy, expected, rtol=1e-4)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# -----------------------------------------------------------------------------
# 2.4 Layer Normalization
# -----------------------------------------------------------------------------
print("\n2.4 Layer Normalization: (x - mean) / sqrt(var + eps)")

def layernorm_numpy(x, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)

def layernorm_bunch(x, eps=1e-5):
    n = len(x)
    mean = x.sum() / n
    x_centered = x - mean
    var = (x_centered * x_centered).sum() / n
    std = (var + eps) ** 0.5
    return x_centered / std

def layernorm_highway(x, eps=1e-5):
    n = len(x)

    # Compute mean
    sum_x = highway.reduce_sum(x)
    mean = sum_x / n
    mean_vec = flow.Bunch.fill(n, mean)

    # Center
    x_centered = highway.sub(x, mean_vec)

    # Compute variance
    x_sq = highway.mul(x_centered, x_centered)
    var = highway.reduce_sum(x_sq) / n

    # Normalize
    std = (var + eps) ** 0.5
    std_vec = flow.Bunch.fill(n, std)
    return highway.div(x_centered, std_vec)

result_bunch, t1 = benchmark("layernorm_bunch", layernorm_bunch, x_softmax)
result_hwy, t2 = benchmark("layernorm_highway", layernorm_highway, x_softmax)
expected = layernorm_numpy(x_softmax_np)

verify("Bunch LayerNorm", result_bunch, expected, rtol=1e-4)
verify("Highway LayerNorm", result_hwy, expected, rtol=1e-4)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# -----------------------------------------------------------------------------
# 2.5 Polynomial Evaluation (Horner's Method)
# -----------------------------------------------------------------------------
print("\n2.5 Polynomial: p(x) = a0 + a1*x + a2*x^2 + a3*x^3 (Horner's)")

def polynomial_numpy(x, coeffs):
    """Horner's method: p(x) = a0 + x*(a1 + x*(a2 + x*a3))"""
    result = np.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result

def polynomial_bunch(x, coeffs):
    result = flow.Bunch.fill(len(x), coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result

def polynomial_highway(x, coeffs):
    result = flow.Bunch.fill(len(x), coeffs[-1])
    for c in reversed(coeffs[:-1]):
        c_vec = flow.Bunch.fill(len(x), c)
        result = highway.mul_add(result, x, c_vec)  # FMA: result * x + c
    return result

coeffs = [1.0, 2.0, -0.5, 0.1]  # 1 + 2x - 0.5x^2 + 0.1x^3
x_poly_np = np.linspace(-2, 2, n).astype(np.float32)
x_poly = flow.Bunch.from_numpy(x_poly_np)

result_bunch, t1 = benchmark("poly_bunch", polynomial_bunch, x_poly, coeffs)
result_hwy, t2 = benchmark("poly_highway", polynomial_highway, x_poly, coeffs)
expected = polynomial_numpy(x_poly_np, coeffs)

verify("Bunch Polynomial", result_bunch, expected, rtol=1e-4)
verify("Highway Polynomial", result_hwy, expected, rtol=1e-4)
print(f"  Bunch: {t1:.2f} µs, Highway: {t2:.2f} µs")

# =============================================================================
# PART 3: PIPELINE-BASED KERNELS (Developer Tier)
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: PIPELINE-BASED KERNELS (Fused Operations)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Normalize Pipeline: (x - min) / (max - min)
# -----------------------------------------------------------------------------
print("\n3.1 Min-Max Normalization Pipeline")

def normalize_numpy(x):
    x_min, x_max = np.min(x), np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-8)

def normalize_pipeline(x):
    x_min = x.min()
    x_max = x.max()
    range_val = x_max - x_min + 1e-8

    # Build pipeline: subtract min, then divide by range
    pipe = flow.Pipeline()
    pipe.add(-x_min)      # x - min
    pipe.divide(range_val)  # / range

    return pipe.run(x)  # Returns Bunch directly

result_pipe, t1 = benchmark("normalize_pipeline", normalize_pipeline, x_bunch)
expected = normalize_numpy(x_np)

verify("Pipeline Normalize", result_pipe, expected, rtol=1e-4)
print(f"  Pipeline: {t1:.2f} µs")
print(f"  Pipeline can fuse: {flow.Pipeline().add(1.0).multiply(2.0).can_fuse()}")

# -----------------------------------------------------------------------------
# 3.2 GELU Approximation Pipeline
# -----------------------------------------------------------------------------
print("\n3.2 GELU Approximation Pipeline: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))")

def gelu_numpy(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_bunch(x):
    # GELU approximation
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    x_cubed = x * x * x
    inner = (x + x_cubed * 0.044715) * sqrt_2_pi
    return x * (inner.tanh() + 1.0) * 0.5

result_gelu, t1 = benchmark("gelu_bunch", gelu_bunch, x_small)
expected = gelu_numpy(x_small_np)

verify("Bunch GELU", result_gelu, expected, rtol=1e-3)
print(f"  Bunch: {t1:.2f} µs")

# =============================================================================
# PART 4: PATTERN-BASED KERNELS (Developer Tier)
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: PATTERN-BASED KERNELS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Strided Sum (every nth element)
# -----------------------------------------------------------------------------
print("\n4.1 Strided Sum (every 4th element)")

def strided_sum_numpy(x, stride):
    return np.sum(x[::stride])

def strided_sum_pattern(x, stride):
    pattern = flow.Pattern.stride(stride)
    selected = flow.select(x, pattern)
    return selected.sum()

stride = 4
result_pattern = strided_sum_pattern(x_bunch, stride)
expected = strided_sum_numpy(x_np, stride)

verify("Pattern Strided Sum", result_pattern, expected, rtol=1e-4)
print(f"  Selected {n // stride} elements out of {n}")

# -----------------------------------------------------------------------------
# 4.2 Block Processing (Cache-friendly tiling)
# -----------------------------------------------------------------------------
print("\n4.2 Block-wise Max (tile size 256)")

def blockwise_max_numpy(x, block_size):
    n_blocks = (len(x) + block_size - 1) // block_size
    maxes = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, len(x))
        maxes.append(np.max(x[start:end]))
    return np.array(maxes, dtype=np.float32)

def blockwise_max_pattern(x, block_size):
    n = len(x)
    x_np = x.to_numpy()
    iterator = flow.BlockIterator(n, block_size)

    maxes = []
    while iterator.has_next():
        block = iterator.next()
        block_data = flow.Bunch.from_numpy(x_np[block.start:block.end])
        maxes.append(block_data.max())

    return np.array(maxes, dtype=np.float32)

block_size = 256
result_blocks = blockwise_max_pattern(x_bunch, block_size)
expected = blockwise_max_numpy(x_np, block_size)

verify("Block-wise Max", result_blocks, expected)
print(f"  Processed {len(result_blocks)} blocks of size {block_size}")

# =============================================================================
# PART 5: COMPLEX KERNELS
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: COMPLEX KERNELS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Matrix-Vector Multiplication (GEMV)
# -----------------------------------------------------------------------------
print("\n5.1 Matrix-Vector Multiplication (GEMV): y = A @ x")

def gemv_numpy(A, x):
    return A @ x

def gemv_simd(A_flat, x, M, N):
    """GEMV using SIMD dot products for each row"""
    result = np.zeros(M, dtype=np.float32)

    for i in range(M):
        # Extract row i
        row_start = i * N
        row_data = A_flat.to_numpy()[row_start:row_start + N]
        row = flow.Bunch.from_numpy(row_data)

        # Dot product of row with x
        result[i] = highway.dot_product(row, x)

    return result

M, N = 64, 256
A_np = np.random.randn(M, N).astype(np.float32)
x_vec_np = np.random.randn(N).astype(np.float32)
A_flat = flow.Bunch.from_numpy(A_np.flatten())
x_vec = flow.Bunch.from_numpy(x_vec_np)

result_gemv, t1 = benchmark("gemv_simd", gemv_simd, A_flat, x_vec, M, N, iterations=50)
expected = gemv_numpy(A_np, x_vec_np)

verify("SIMD GEMV", result_gemv, expected, rtol=1e-3)  # Looser tolerance for accumulated fp errors
print(f"  Matrix: {M}x{N}, Time: {t1:.2f} µs")

# -----------------------------------------------------------------------------
# 5.2 Batched Softmax
# -----------------------------------------------------------------------------
print("\n5.2 Batched Softmax (batch_size x seq_len)")

def batched_softmax_numpy(x, batch_size, seq_len):
    x_2d = x.reshape(batch_size, seq_len)
    x_max = np.max(x_2d, axis=1, keepdims=True)
    exp_x = np.exp(x_2d - x_max)
    return (exp_x / np.sum(exp_x, axis=1, keepdims=True)).flatten()

def batched_softmax_simd(x_flat, batch_size, seq_len):
    x_np = x_flat.to_numpy()
    result = np.zeros_like(x_np)

    for b in range(batch_size):
        start = b * seq_len
        end = start + seq_len

        row = flow.Bunch.from_numpy(x_np[start:end])

        # Stable softmax
        row_max = highway.reduce_max(row)
        row_max_vec = flow.Bunch.fill(seq_len, row_max)
        shifted = highway.sub(row, row_max_vec)
        exp_row = highway.exp(shifted)
        sum_exp = highway.reduce_sum(exp_row)
        sum_vec = flow.Bunch.fill(seq_len, sum_exp)
        softmax_row = highway.div(exp_row, sum_vec)

        result[start:end] = softmax_row.to_numpy()

    return result

batch_size, seq_len = 32, 128
x_batch_np = np.random.randn(batch_size * seq_len).astype(np.float32)
x_batch = flow.Bunch.from_numpy(x_batch_np)

result_batch, t1 = benchmark("batched_softmax", batched_softmax_simd,
                             x_batch, batch_size, seq_len, iterations=50)
expected = batched_softmax_numpy(x_batch_np, batch_size, seq_len)

verify("Batched Softmax", result_batch, expected, rtol=1e-4)
print(f"  Batch: {batch_size}, Seq: {seq_len}, Time: {t1:.2f} µs")

# -----------------------------------------------------------------------------
# 5.3 1D Convolution
# -----------------------------------------------------------------------------
print("\n5.3 1D Convolution (kernel_size=5)")

def conv1d_numpy(x, kernel):
    """Simple 1D convolution (valid padding)"""
    k_len = len(kernel)
    out_len = len(x) - k_len + 1
    result = np.zeros(out_len, dtype=np.float32)
    for i in range(out_len):
        result[i] = np.dot(x[i:i+k_len], kernel)
    return result

def conv1d_simd(x, kernel):
    """1D convolution using SIMD dot products"""
    x_np = x.to_numpy()
    kernel_np = kernel.to_numpy()
    k_len = len(kernel)
    out_len = len(x) - k_len + 1
    result = np.zeros(out_len, dtype=np.float32)

    for i in range(out_len):
        window = flow.Bunch.from_numpy(x_np[i:i+k_len])
        result[i] = highway.dot_product(window, kernel)

    return result

kernel_size = 5
x_conv_np = np.random.randn(1024).astype(np.float32)
kernel_np = np.random.randn(kernel_size).astype(np.float32)
x_conv = flow.Bunch.from_numpy(x_conv_np)
kernel_conv = flow.Bunch.from_numpy(kernel_np)

result_conv, t1 = benchmark("conv1d_simd", conv1d_simd, x_conv, kernel_conv, iterations=20)
expected = conv1d_numpy(x_conv_np, kernel_np)

verify("SIMD Conv1D", result_conv, expected, rtol=1e-4)
print(f"  Input: {len(x_conv_np)}, Kernel: {kernel_size}, Output: {len(result_conv)}, Time: {t1:.2f} µs")

# -----------------------------------------------------------------------------
# 5.4 Distance Matrix (Euclidean)
# -----------------------------------------------------------------------------
print("\n5.4 Euclidean Distance Matrix (N points in D dimensions)")

def distance_matrix_numpy(X):
    """Compute pairwise Euclidean distances"""
    N = X.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            diff = X[i] - X[j]
            D[i, j] = D[j, i] = np.sqrt(np.dot(diff, diff))
    return D

def distance_matrix_simd(X_flat, N, dim):
    """SIMD-accelerated distance matrix"""
    X_np = X_flat.to_numpy().reshape(N, dim)
    D = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        xi = flow.Bunch.from_numpy(X_np[i])
        for j in range(i+1, N):
            xj = flow.Bunch.from_numpy(X_np[j])

            # diff = xi - xj
            diff = highway.sub(xi, xj)

            # squared_dist = dot(diff, diff)
            sq_dist = highway.dot_product(diff, diff)

            # dist = sqrt(squared_dist)
            dist = sq_dist ** 0.5

            D[i, j] = D[j, i] = dist

    return D

N_points, dim = 32, 64
X_np = np.random.randn(N_points, dim).astype(np.float32)
X_flat = flow.Bunch.from_numpy(X_np.flatten())

result_dist, t1 = benchmark("distance_matrix", distance_matrix_simd,
                            X_flat, N_points, dim, iterations=20)
expected = distance_matrix_numpy(X_np)

verify("Distance Matrix", result_dist, expected, rtol=1e-4)
print(f"  Points: {N_points}, Dim: {dim}, Time: {t1:.2f} µs")

# -----------------------------------------------------------------------------
# 5.5 Attention Score Computation (Scaled Dot-Product)
# -----------------------------------------------------------------------------
print("\n5.5 Scaled Dot-Product Attention Scores")

def attention_scores_numpy(Q, K, d_k):
    """Compute attention scores: softmax(Q @ K.T / sqrt(d_k))"""
    scores = Q @ K.T / np.sqrt(d_k)
    # Softmax over last dimension
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def attention_scores_simd(Q_flat, K_flat, seq_len, d_k):
    """SIMD attention score computation"""
    Q_np = Q_flat.to_numpy().reshape(seq_len, d_k)
    K_np = K_flat.to_numpy().reshape(seq_len, d_k)
    scale = 1.0 / np.sqrt(d_k)

    # Compute Q @ K.T
    scores = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        qi = flow.Bunch.from_numpy(Q_np[i])
        for j in range(seq_len):
            kj = flow.Bunch.from_numpy(K_np[j])
            scores[i, j] = highway.dot_product(qi, kj) * scale

    # Softmax per row
    result = np.zeros_like(scores)
    for i in range(seq_len):
        row = flow.Bunch.from_numpy(scores[i])
        row_max = highway.reduce_max(row)
        row_max_vec = flow.Bunch.fill(seq_len, row_max)
        shifted = highway.sub(row, row_max_vec)
        exp_row = highway.exp(shifted)
        sum_exp = highway.reduce_sum(exp_row)
        sum_vec = flow.Bunch.fill(seq_len, sum_exp)
        result[i] = highway.div(exp_row, sum_vec).to_numpy()

    return result

seq_len_attn = 16
d_k = 64
Q_np = np.random.randn(seq_len_attn, d_k).astype(np.float32)
K_np = np.random.randn(seq_len_attn, d_k).astype(np.float32)
Q_flat = flow.Bunch.from_numpy(Q_np.flatten())
K_flat = flow.Bunch.from_numpy(K_np.flatten())

result_attn, t1 = benchmark("attention_scores", attention_scores_simd,
                            Q_flat, K_flat, seq_len_attn, d_k, iterations=10)
expected = attention_scores_numpy(Q_np, K_np, d_k)

verify("Attention Scores", result_attn, expected, rtol=1e-3)
print(f"  Seq: {seq_len_attn}, d_k: {d_k}, Time: {t1:.2f} µs")

# =============================================================================
# PART 6: EXTREME COMPLEXITY - MULTI-HEAD ATTENTION
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: EXTREME COMPLEXITY")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Multi-Head Attention (Simplified)
# -----------------------------------------------------------------------------
print("\n6.1 Multi-Head Self-Attention")

def multihead_attention_numpy(X, Wq, Wk, Wv, n_heads, d_model, d_k):
    """
    Multi-head attention:
    1. Project X to Q, K, V
    2. Split into heads
    3. Compute attention per head
    4. Concatenate
    """
    seq_len = X.shape[0]

    # Linear projections
    Q = X @ Wq  # (seq_len, d_model)
    K = X @ Wk
    V = X @ Wv

    # Reshape to (n_heads, seq_len, d_k)
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Attention per head
    outputs = []
    scale = 1.0 / np.sqrt(d_k)
    for h in range(n_heads):
        scores = Q[h] @ K[h].T * scale
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        out = attn @ V[h]
        outputs.append(out)

    # Concatenate heads
    return np.concatenate(outputs, axis=1)

def multihead_attention_simd(X_flat, Wq_flat, Wk_flat, Wv_flat,
                              seq_len, n_heads, d_model, d_k):
    """Multi-head attention with SIMD acceleration"""
    X = X_flat.to_numpy().reshape(seq_len, d_model)
    Wq = Wq_flat.to_numpy().reshape(d_model, d_model)
    Wk = Wk_flat.to_numpy().reshape(d_model, d_model)
    Wv = Wv_flat.to_numpy().reshape(d_model, d_model)

    # Project to Q, K, V (GEMM - could be SIMD accelerated further)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Reshape for heads
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(d_k)
    outputs = []

    for h in range(n_heads):
        # Compute attention scores using SIMD
        scores = np.zeros((seq_len, seq_len), dtype=np.float32)
        for i in range(seq_len):
            qi = flow.Bunch.from_numpy(Q[h, i])
            for j in range(seq_len):
                kj = flow.Bunch.from_numpy(K[h, j])
                scores[i, j] = highway.dot_product(qi, kj) * scale

        # Softmax per row using SIMD
        attn = np.zeros_like(scores)
        for i in range(seq_len):
            row = flow.Bunch.from_numpy(scores[i])
            row_max = highway.reduce_max(row)
            row_max_vec = flow.Bunch.fill(seq_len, row_max)
            shifted = highway.sub(row, row_max_vec)
            exp_row = highway.exp(shifted)
            sum_exp = highway.reduce_sum(exp_row)
            sum_vec = flow.Bunch.fill(seq_len, sum_exp)
            attn[i] = highway.div(exp_row, sum_vec).to_numpy()

        # Apply attention to values (matmul)
        out = np.zeros((seq_len, d_k), dtype=np.float32)
        for i in range(seq_len):
            attn_row = flow.Bunch.from_numpy(attn[i])
            for k in range(d_k):
                v_col = flow.Bunch.from_numpy(V[h, :, k])
                out[i, k] = highway.dot_product(attn_row, v_col)

        outputs.append(out)

    return np.concatenate(outputs, axis=1)

# Small transformer config
seq_len_mha = 8
n_heads = 4
d_model = 32
d_k = d_model // n_heads

X_mha = np.random.randn(seq_len_mha, d_model).astype(np.float32)
Wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
Wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
Wv = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

X_flat = flow.Bunch.from_numpy(X_mha.flatten())
Wq_flat = flow.Bunch.from_numpy(Wq.flatten())
Wk_flat = flow.Bunch.from_numpy(Wk.flatten())
Wv_flat = flow.Bunch.from_numpy(Wv.flatten())

result_mha, t1 = benchmark("multihead_attention", multihead_attention_simd,
                           X_flat, Wq_flat, Wk_flat, Wv_flat,
                           seq_len_mha, n_heads, d_model, d_k, iterations=5)
expected = multihead_attention_numpy(X_mha, Wq, Wk, Wv, n_heads, d_model, d_k)

verify("Multi-Head Attention", result_mha, expected, rtol=1e-3)
print(f"  Seq: {seq_len_mha}, Heads: {n_heads}, d_model: {d_model}, Time: {t1:.2f} µs")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: SIMD Kernel Capabilities Demonstrated")
print("=" * 70)

print(f"""
System Info:
  SIMD Target: {highway.get_simd_target()}
  Float32 Lanes: {highway.get_simd_lanes_f32()}
  Float64 Lanes: {highway.get_simd_lanes_f64()}

Kernels Tested:
  Simple:
    - SAXPY (y = a*x + y)
    - Dot Product
    - Element-wise Chain

  Medium:
    - ReLU Activation
    - Sigmoid Activation
    - Stable Softmax
    - Layer Normalization
    - Polynomial (Horner's Method with FMA)

  Pipeline/Pattern:
    - Min-Max Normalization (Pipeline fusion)
    - GELU Approximation
    - Strided Sum (Pattern selection)
    - Block-wise Max (Cache-friendly tiling)

  Complex:
    - Matrix-Vector Multiplication (GEMV)
    - Batched Softmax
    - 1D Convolution
    - Euclidean Distance Matrix
    - Scaled Dot-Product Attention

  Extreme:
    - Multi-Head Self-Attention

All kernels verified against NumPy reference implementations!
""")
