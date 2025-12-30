#!/usr/bin/env python3
"""
Machine Learning Primitives with Bud Flow Lang

Common ML operations implemented efficiently:
- Activation functions
- Loss functions
- Layer operations
- Batch operations
- Feature processing

Run: python machine_learning.py
"""

import bud_flow_lang_py as flow
import math
import time


# ============================================================
# Activation Functions
# ============================================================

def relu(x):
    """
    ReLU activation: max(0, x)

    Args:
        x: Input array

    Returns:
        Element-wise max(0, x)
    """
    return flow.clamp(x, 0.0, float('inf'))


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: x if x > 0, else alpha * x

    Args:
        x: Input array
        alpha: Slope for negative values

    Returns:
        Leaky ReLU output
    """
    # leaky_relu(x) = max(x, alpha*x)
    # For positive x: max(x, alpha*x) = x (since alpha < 1)
    # For negative x: max(x, alpha*x) = alpha*x (since alpha > 0)
    positive = flow.clamp(x, 0.0, float('inf'))
    negative = flow.clamp(x, float('-inf'), 0.0) * alpha
    return positive + negative


def sigmoid(x):
    """
    Sigmoid activation: 1 / (1 + exp(-x))

    Args:
        x: Input array

    Returns:
        Sigmoid output in (0, 1)
    """
    neg_x = x * (-1.0)
    exp_neg_x = flow.exp(neg_x)
    return (exp_neg_x + 1.0) * (-1.0) + 1.0  # 1 - 1/(1+e^-x) form


def tanh_activation(x):
    """
    Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x: Input array

    Returns:
        Tanh output in (-1, 1)
    """
    return flow.tanh(x)


def softmax(x):
    """
    Softmax: exp(x) / sum(exp(x))

    For numerical stability, we compute exp(x - max(x)).

    Args:
        x: Input array (logits)

    Returns:
        Probability distribution (sums to 1)
    """
    x_max = x.max()
    shifted = x - x_max
    exp_x = flow.exp(shifted)
    sum_exp = exp_x.sum()
    return exp_x * (1.0 / sum_exp)


def gelu(x):
    """
    GELU activation (approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input array

    Returns:
        GELU output
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    x_cubed = x * x * x
    inner = (x + x_cubed * 0.044715) * sqrt_2_over_pi
    return x * 0.5 * (flow.tanh(inner) + 1.0)


# ============================================================
# Loss Functions
# ============================================================

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Scalar MSE loss
    """
    diff = predictions - targets
    squared = diff * diff
    return squared.mean()


def mae_loss(predictions, targets):
    """
    Mean Absolute Error loss.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Scalar MAE loss
    """
    diff = predictions - targets
    abs_diff = flow.abs(diff)
    return abs_diff.mean()


def binary_cross_entropy(predictions, targets, epsilon=1e-7):
    """
    Binary cross-entropy loss.

    Args:
        predictions: Predicted probabilities in (0, 1)
        targets: Binary targets (0 or 1)
        epsilon: Small constant for numerical stability

    Returns:
        Scalar BCE loss
    """
    # BCE = -mean(y * log(p) + (1-y) * log(1-p))
    # Clamp predictions to avoid log(0)
    p_clamped = flow.clamp(predictions, epsilon, 1.0 - epsilon)
    log_p = flow.log(p_clamped)
    log_1_minus_p = flow.log((p_clamped * (-1.0)) + 1.0)

    # y * log(p) + (1-y) * log(1-p)
    term1 = targets * log_p
    term2 = ((targets * (-1.0)) + 1.0) * log_1_minus_p
    loss_per_sample = term1 + term2

    return loss_per_sample.mean() * (-1.0)


def huber_loss(predictions, targets, delta=1.0):
    """
    Huber loss (smooth L1).

    Args:
        predictions: Model predictions
        targets: Ground truth values
        delta: Threshold where loss transitions from L2 to L1

    Returns:
        Scalar Huber loss
    """
    diff = predictions - targets
    abs_diff = flow.abs(diff)

    # Quadratic part: 0.5 * diff^2 (when |diff| <= delta)
    quadratic = diff * diff * 0.5

    # Linear part: delta * (|diff| - 0.5 * delta) (when |diff| > delta)
    linear = (abs_diff - delta * 0.5) * delta

    # Choose based on threshold (approximation using clamp)
    # When abs_diff <= delta, use quadratic; otherwise use linear
    # This is an approximation using element-wise operations
    mask_quadratic = flow.clamp(delta - abs_diff, 0.0, 1.0)  # 1 if quadratic, 0 otherwise
    mask_linear = flow.clamp(abs_diff - delta, 0.0, 1.0)

    result = quadratic * mask_quadratic + linear * mask_linear
    return result.mean()


# ============================================================
# Layer Operations
# ============================================================

def dense_layer(x, weights, bias):
    """
    Dense (fully connected) layer: y = Wx + b

    For 1D vectors, this is element-wise multiply + add (FMA).

    Args:
        x: Input features
        weights: Weight vector
        bias: Bias vector

    Returns:
        Output after linear transformation
    """
    return flow.fma(weights, x, bias)


def batch_norm_1d(x, gamma, beta, epsilon=1e-5):
    """
    Batch normalization for 1D data.

    Args:
        x: Input array
        gamma: Scale parameter
        beta: Shift parameter
        epsilon: Small constant for numerical stability

    Returns:
        Normalized output
    """
    mean = x.mean()
    variance = flow.dot(x - mean, x - mean) / x.size

    # Normalize
    x_norm = (x - mean) * (1.0 / (variance + epsilon) ** 0.5)

    # Scale and shift
    return flow.fma(x_norm, gamma, beta)


def layer_norm(x, gamma, beta, epsilon=1e-5):
    """
    Layer normalization.

    Args:
        x: Input array
        gamma: Scale parameter
        beta: Shift parameter
        epsilon: Small constant for numerical stability

    Returns:
        Normalized output
    """
    mean = x.mean()
    # Compute variance: E[(x - mean)^2]
    diff = x - mean
    variance = flow.dot(diff, diff) / x.size

    # Normalize
    x_norm = diff * (1.0 / (variance + epsilon) ** 0.5)

    # Scale and shift (using FMA)
    return flow.fma(x_norm, gamma, beta)


def dropout(x, keep_prob=0.5):
    """
    Dropout (inference mode - scales by keep_prob).

    Note: During training, you'd need random masks.
    This implements inference-time behavior.

    Args:
        x: Input array
        keep_prob: Probability of keeping a unit

    Returns:
        Scaled output
    """
    return x * keep_prob


# ============================================================
# Feature Processing
# ============================================================

def l2_normalize(x, epsilon=1e-12):
    """
    L2 normalization: x / ||x||_2

    Args:
        x: Input array
        epsilon: Small constant for numerical stability

    Returns:
        L2 normalized array
    """
    norm = (flow.dot(x, x) + epsilon) ** 0.5
    return x * (1.0 / norm)


def min_max_scale(x, feature_min=0.0, feature_max=1.0):
    """
    Min-max scaling to [feature_min, feature_max].

    Args:
        x: Input array
        feature_min: Minimum of output range
        feature_max: Maximum of output range

    Returns:
        Scaled array
    """
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min

    if x_range < 1e-10:
        # Constant array - return midpoint
        return flow.full(x.size, (feature_min + feature_max) / 2)

    # Scale to [0, 1]
    scaled = (x - x_min) * (1.0 / x_range)
    # Scale to [feature_min, feature_max]
    return scaled * (feature_max - feature_min) + feature_min


def standardize(x, epsilon=1e-10):
    """
    Standardization: (x - mean) / std

    Args:
        x: Input array
        epsilon: Small constant for numerical stability

    Returns:
        Standardized array with mean=0, std=1
    """
    mean = x.mean()
    centered = x - mean
    variance = flow.dot(centered, centered) / x.size
    std = (variance + epsilon) ** 0.5
    return centered * (1.0 / std)


# ============================================================
# Demos
# ============================================================

def demo_activations():
    """Demonstrate activation functions."""
    print("\n=== Activation Functions ===\n")

    x = flow.linspace(-3, 3, 7)
    print(f"Input: {x.to_numpy()}")
    print()

    # ReLU
    y = relu(x)
    print(f"ReLU: {y.to_numpy()}")

    # Leaky ReLU
    y = leaky_relu(x, alpha=0.1)
    print(f"Leaky ReLU (a=0.1): {y.to_numpy()}")

    # Tanh
    y = tanh_activation(x)
    print(f"Tanh: {y.to_numpy()}")

    # Softmax
    logits = flow.flow([1.0, 2.0, 3.0, 4.0])
    probs = softmax(logits)
    print(f"\nSoftmax([1,2,3,4]): {probs.to_numpy()}")
    print(f"Sum: {probs.sum():.6f} (should be 1.0)")


def demo_losses():
    """Demonstrate loss functions."""
    print("\n=== Loss Functions ===\n")

    predictions = flow.flow([0.5, 0.8, 0.3, 0.9])
    targets = flow.flow([0.0, 1.0, 0.0, 1.0])

    print(f"Predictions: {predictions.to_numpy()}")
    print(f"Targets: {targets.to_numpy()}")
    print()

    # MSE
    loss = mse_loss(predictions, targets)
    print(f"MSE Loss: {loss:.6f}")

    # MAE
    loss = mae_loss(predictions, targets)
    print(f"MAE Loss: {loss:.6f}")

    # Binary cross-entropy
    loss = binary_cross_entropy(predictions, targets)
    print(f"BCE Loss: {loss:.6f}")


def demo_layers():
    """Demonstrate layer operations."""
    print("\n=== Layer Operations ===\n")

    n = 100

    # Input features
    x = flow.linspace(-1, 1, n)

    # Weights and bias
    weights = flow.full(n, 0.1)
    bias = flow.full(n, 0.01)

    # Dense layer
    y = dense_layer(x, weights, bias)
    print(f"Dense layer output (first 5): {y.to_numpy()[:5]}")
    print(f"Dense layer output mean: {y.mean():.6f}")

    # Layer normalization
    gamma = flow.ones(n)
    beta = flow.zeros(n)
    y_norm = layer_norm(x, gamma, beta)
    print(f"\nLayer norm mean: {y_norm.mean():.6f} (should be ~0)")
    diff = y_norm - y_norm.mean()
    variance = flow.dot(diff, diff) / n
    print(f"Layer norm variance: {variance:.6f} (should be ~1)")


def demo_features():
    """Demonstrate feature processing."""
    print("\n=== Feature Processing ===\n")

    # Create sample data
    x = flow.flow([1.0, 5.0, 2.0, 8.0, 3.0])
    print(f"Original: {x.to_numpy()}")

    # L2 normalize
    y = l2_normalize(x)
    print(f"L2 normalized: {y.to_numpy()}")
    print(f"L2 norm: {flow.dot(y, y) ** 0.5:.6f} (should be 1.0)")

    # Min-max scale
    y = min_max_scale(x)
    print(f"\nMin-max scaled: {y.to_numpy()}")
    print(f"Range: [{y.min():.2f}, {y.max():.2f}]")

    # Standardize
    y = standardize(x)
    print(f"\nStandardized: {y.to_numpy()}")
    print(f"Mean: {y.mean():.6f} (should be ~0)")


def demo_performance():
    """Demonstrate performance on large arrays."""
    print("\n=== Performance ===\n")

    n = 1_000_000
    print(f"Array size: {n:,} elements")

    x = flow.linspace(-5, 5, n)
    weights = flow.full(n, 0.01)
    bias = flow.full(n, 0.001)
    gamma = flow.ones(n)
    beta = flow.zeros(n)

    # Warmup
    for _ in range(10):
        relu(x)
        softmax(x)
        dense_layer(x, weights, bias)
        layer_norm(x, gamma, beta)

    iterations = 50

    # Benchmark
    operations = [
        ("ReLU", lambda: relu(x)),
        ("Tanh", lambda: tanh_activation(x)),
        ("Dense Layer", lambda: dense_layer(x, weights, bias)),
        ("Layer Norm", lambda: layer_norm(x, gamma, beta)),
        ("L2 Normalize", lambda: l2_normalize(x)),
        ("Standardize", lambda: standardize(x)),
    ]

    print()
    for name, func in operations:
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        elapsed = (time.perf_counter() - start) / iterations * 1000
        throughput = n / (elapsed / 1000) / 1e9
        print(f"{name:<15}: {elapsed:>7.3f} ms ({throughput:.2f} Gelem/s)")


def demo_forward_pass():
    """Demonstrate a simple forward pass."""
    print("\n=== Simple Forward Pass ===\n")

    n = 1000

    # Input
    x = flow.linspace(-1, 1, n)
    print(f"Input shape: {x.size} elements")
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")

    # Layer 1: Dense + ReLU
    w1 = flow.full(n, 0.1)
    b1 = flow.full(n, 0.01)
    h1 = relu(dense_layer(x, w1, b1))
    print(f"\nAfter Layer 1 (Dense+ReLU):")
    print(f"  Range: [{h1.min():.4f}, {h1.max():.4f}]")
    print(f"  Mean: {h1.mean():.4f}")

    # Layer 2: Dense + Tanh
    w2 = flow.full(n, 0.05)
    b2 = flow.full(n, 0.0)
    h2 = tanh_activation(dense_layer(h1, w2, b2))
    print(f"\nAfter Layer 2 (Dense+Tanh):")
    print(f"  Range: [{h2.min():.4f}, {h2.max():.4f}]")
    print(f"  Mean: {h2.mean():.4f}")

    # Output: Mean pooling (simulate classification)
    output = h2.mean()
    print(f"\nOutput (mean pooling): {output:.6f}")


def main():
    print("=" * 60)
    print("Machine Learning Primitives with Bud Flow Lang")
    print("=" * 60)

    flow.initialize()

    info = flow.get_hardware_info()
    print(f"\nHardware: {info['arch_family']} with {info['simd_width']*8}-bit SIMD")

    demo_activations()
    demo_losses()
    demo_layers()
    demo_features()
    demo_performance()
    demo_forward_pass()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
