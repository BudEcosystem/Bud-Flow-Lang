// =============================================================================
// Bud Flow Lang - Fused Highway Kernels
// =============================================================================
//
// This module provides fused operation kernels that combine multiple operations
// into single-pass loops for maximum memory bandwidth efficiency.
//
// Based on Weld research showing 6-30x speedups from fusion:
// https://www.vldb.org/pvldb/vol11/p1002-palkar.pdf
//
// Key optimization patterns:
// - Element-wise fusion: a * 2 + 1 -> single loop
// - Reduction fusion: (a * b).sum() -> dot product
// - Broadcast fusion: Eliminate redundant broadcasts
//
// Note: These implementations use vectorized Highway ops where possible,
// with scalar fallbacks for complex transcendentals (exp, log, etc.)
// for portability across all Highway targets.
//
// =============================================================================

#include "bud_flow_lang/codegen/fused_kernel.h"

#include "bud_flow_lang/codegen/hwy_ops.h"

#include <hwy/highway.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace bud {
namespace simd {

// Use the ScalableTag for portable SIMD
namespace hn = hwy::HWY_NAMESPACE;

// =============================================================================
// Fused Element-wise Kernels
// =============================================================================

// Fused: out = a * scalar + b
void FusedMulScalarAdd(float* out, const float* a, float scalar, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto v_scalar = hn::Set(d, scalar);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto result = hn::MulAdd(va, v_scalar, vb);
        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = a[i] * scalar + b[i];
    }
}

// Fused: out = sqrt(a^2 + b^2)  (2D vector norm)
void FusedNorm2D(float* out, const float* a, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto sq_a = hn::Mul(va, va);
        const auto sq_b = hn::Mul(vb, vb);
        const auto sum_sq = hn::Add(sq_a, sq_b);
        const auto result = hn::Sqrt(sum_sq);
        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]);
    }
}

// Fused: out = 1 / (1 + exp(-x))  (sigmoid activation)
// Uses scalar implementation for portability
void FusedSigmoid(float* out, const float* x, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float xi = x[i];
        // Clamp to avoid overflow in exp
        if (xi > 88.0f) {
            out[i] = 1.0f;
        } else if (xi < -88.0f) {
            out[i] = 0.0f;
        } else {
            out[i] = 1.0f / (1.0f + std::exp(-xi));
        }
    }
}

// Fused: out = max(0, x)  (ReLU activation)
void FusedRelu(float* out, const float* x, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto v_zero = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vx = hn::LoadU(d, x + i);
        const auto result = hn::Max(vx, v_zero);
        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}

// Fused: out = sig_x * (1 - sig_x)  (sigmoid gradient)
void FusedSigmoidGrad(float* out, const float* sig_x, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto v_one = hn::Set(d, 1.0f);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vs = hn::LoadU(d, sig_x + i);
        const auto one_minus_s = hn::Sub(v_one, vs);
        const auto result = hn::Mul(vs, one_minus_s);
        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = sig_x[i] * (1.0f - sig_x[i]);
    }
}

// =============================================================================
// Fused Reduction Kernels
// =============================================================================

// Fused: sum(a * b)  (dot product with 4 accumulators for latency hiding)
float FusedDotProduct(const float* a, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Use 4 accumulators to hide latency
    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    // Unroll 4x for latency hiding
    for (; i + 4 * N <= count; i += 4 * N) {
        const auto a0 = hn::LoadU(d, a + i);
        const auto b0 = hn::LoadU(d, b + i);
        sum0 = hn::MulAdd(a0, b0, sum0);

        const auto a1 = hn::LoadU(d, a + i + N);
        const auto b1 = hn::LoadU(d, b + i + N);
        sum1 = hn::MulAdd(a1, b1, sum1);

        const auto a2 = hn::LoadU(d, a + i + 2 * N);
        const auto b2 = hn::LoadU(d, b + i + 2 * N);
        sum2 = hn::MulAdd(a2, b2, sum2);

        const auto a3 = hn::LoadU(d, a + i + 3 * N);
        const auto b3 = hn::LoadU(d, b + i + 3 * N);
        sum3 = hn::MulAdd(a3, b3, sum3);
    }

    // Process remaining full vectors
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        sum0 = hn::MulAdd(va, vb, sum0);
    }

    // Combine accumulators
    sum0 = hn::Add(sum0, sum1);
    sum2 = hn::Add(sum2, sum3);
    sum0 = hn::Add(sum0, sum2);

    float result = hn::ReduceSum(d, sum0);

    // Scalar remainder
    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// Fused: sum((a - b)^2)  (squared L2 distance)
float FusedSquaredDistance(const float* a, const float* b, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);

    size_t i = 0;
    // Unroll 2x
    for (; i + 2 * N <= count; i += 2 * N) {
        const auto va0 = hn::LoadU(d, a + i);
        const auto vb0 = hn::LoadU(d, b + i);
        const auto diff0 = hn::Sub(va0, vb0);
        sum0 = hn::MulAdd(diff0, diff0, sum0);

        const auto va1 = hn::LoadU(d, a + i + N);
        const auto vb1 = hn::LoadU(d, b + i + N);
        const auto diff1 = hn::Sub(va1, vb1);
        sum1 = hn::MulAdd(diff1, diff1, sum1);
    }

    // Process remaining full vectors
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto diff = hn::Sub(va, vb);
        sum0 = hn::MulAdd(diff, diff, sum0);
    }

    sum0 = hn::Add(sum0, sum1);
    float result = hn::ReduceSum(d, sum0);

    // Scalar remainder
    for (; i < count; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

// Fused: sum(a^2)  (squared L2 norm)
float FusedNormSquared(const float* a, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto sum = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        sum = hn::MulAdd(va, va, sum);
    }

    float result = hn::ReduceSum(d, sum);

    // Scalar remainder
    for (; i < count; ++i) {
        result += a[i] * a[i];
    }

    return result;
}

// Fused: sum(exp(x))  (used in softmax denominator)
// Uses scalar implementation for portability
float FusedSumExp(const float* x, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        result += std::exp(x[i]);
    }
    return result;
}

// =============================================================================
// Fused Transform-Reduce Kernels
// =============================================================================

// Fused: softmax = exp(x - max) / sum(exp(x - max))  (numerically stable version)
// Uses scalar implementation for portability
void FusedSoftmax(float* out, const float* x, size_t count) {
    if (count == 0)
        return;

    // Step 1: Find max for numerical stability
    float max_val = x[0];
    for (size_t i = 1; i < count; ++i) {
        if (x[i] > max_val)
            max_val = x[i];
    }

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }

    // Step 3: Divide by sum
    const float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < count; ++i) {
        out[i] *= inv_sum;
    }
}

// =============================================================================
// BLAS-like Fused Operations
// =============================================================================

// Fused: y = alpha * x + y  (axpy)
void FusedAxpy(float* y, float alpha, const float* x, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto v_alpha = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vx = hn::LoadU(d, x + i);
        const auto vy = hn::LoadU(d, y + i);
        const auto result = hn::MulAdd(vx, v_alpha, vy);
        hn::StoreU(result, d, y + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        y[i] += alpha * x[i];
    }
}

// =============================================================================
// Neural Network Fused Kernels
// =============================================================================

// Fused: out = x * sigmoid(1.702 * x)  (GELU activation - approximation)
// GELU(x) ≈ x * sigmoid(1.702 * x) per "Gaussian Error Linear Units" paper
void FusedGelu(float* out, const float* x, size_t count) {
    constexpr float kBeta = 1.702f;

    for (size_t i = 0; i < count; ++i) {
        const float xi = x[i];
        const float scaled = kBeta * xi;
        // Clamp to avoid overflow in exp
        float sigmoid_val;
        if (scaled > 88.0f) {
            sigmoid_val = 1.0f;
        } else if (scaled < -88.0f) {
            sigmoid_val = 0.0f;
        } else {
            sigmoid_val = 1.0f / (1.0f + std::exp(-scaled));
        }
        out[i] = xi * sigmoid_val;
    }
}

// Fused: Layer normalization: out = (x - mean) / sqrt(variance + epsilon) * gamma + beta
// Computes mean and variance in single pass, then normalizes
void FusedLayerNorm(float* out, const float* x, size_t count, float gamma, float beta,
                    float epsilon) {
    if (count == 0)
        return;

    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Step 1: Compute mean using vectorized reduction
    auto sum_vec = hn::Zero(d);
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vx = hn::LoadU(d, x + i);
        sum_vec = hn::Add(sum_vec, vx);
    }
    float sum = hn::ReduceSum(d, sum_vec);
    for (; i < count; ++i) {
        sum += x[i];
    }
    const float mean = sum / static_cast<float>(count);

    // Step 2: Compute variance using vectorized reduction
    auto var_sum = hn::Zero(d);
    const auto v_mean = hn::Set(d, mean);
    i = 0;
    for (; i + N <= count; i += N) {
        const auto vx = hn::LoadU(d, x + i);
        const auto diff = hn::Sub(vx, v_mean);
        var_sum = hn::MulAdd(diff, diff, var_sum);
    }
    float var = hn::ReduceSum(d, var_sum);
    for (; i < count; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(count);

    // Step 3: Normalize with gamma and beta
    const float inv_std = 1.0f / std::sqrt(var + epsilon);
    const auto v_inv_std = hn::Set(d, inv_std);
    const auto v_gamma = hn::Set(d, gamma);
    const auto v_beta = hn::Set(d, beta);

    i = 0;
    for (; i + N <= count; i += N) {
        const auto vx = hn::LoadU(d, x + i);
        const auto normalized = hn::Mul(hn::Sub(vx, v_mean), v_inv_std);
        const auto result = hn::MulAdd(normalized, v_gamma, v_beta);
        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = ((x[i] - mean) * inv_std) * gamma + beta;
    }
}

// Fused: Leaky ReLU: out = max(alpha * x, x)
void FusedLeakyRelu(float* out, const float* x, float alpha, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto v_alpha = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vx = hn::LoadU(d, x + i);
        const auto scaled = hn::Mul(vx, v_alpha);
        const auto result = hn::Max(vx, scaled);
        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = std::max(x[i], alpha * x[i]);
    }
}

// Fused: Swish/SiLU: out = x * sigmoid(x)
// SiLU(x) = x * sigmoid(x) - self-gated activation
void FusedSwish(float* out, const float* x, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float xi = x[i];
        float sigmoid_val;
        if (xi > 88.0f) {
            sigmoid_val = 1.0f;
        } else if (xi < -88.0f) {
            sigmoid_val = 0.0f;
        } else {
            sigmoid_val = 1.0f / (1.0f + std::exp(-xi));
        }
        out[i] = xi * sigmoid_val;
    }
}

// =============================================================================
// Multi-Operation Fused Chains
// =============================================================================

// Fused: out = ((a * b) + c) * d  (4-way fused)
void FusedMulAddMul(float* out, const float* a, const float* b, const float* c, const float* d,
                    size_t count) {
    const hn::ScalableTag<float> df;
    const size_t N = hn::Lanes(df);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(df, a + i);
        const auto vb = hn::LoadU(df, b + i);
        const auto vc = hn::LoadU(df, c + i);
        const auto vd = hn::LoadU(df, d + i);

        // (a * b) + c
        const auto temp = hn::MulAdd(va, vb, vc);
        // ((a * b) + c) * d
        const auto result = hn::Mul(temp, vd);

        hn::StoreU(result, df, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = (a[i] * b[i] + c[i]) * d[i];
    }
}

}  // namespace simd
}  // namespace bud
