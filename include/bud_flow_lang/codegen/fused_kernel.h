// =============================================================================
// Bud Flow Lang - Fused Highway Kernels Header
// =============================================================================
//
// Fused operation kernels that combine multiple operations into single-pass
// loops for maximum memory bandwidth efficiency.
//
// Based on Weld research showing 6-30x speedups from fusion.
//
// =============================================================================

#pragma once

#include <cstddef>

namespace bud {
namespace simd {

// =============================================================================
// Fused Element-wise Operations
// =============================================================================

// out = a * scalar + b
void FusedMulScalarAdd(float* out, const float* a, float scalar, const float* b, size_t count);

// out = sqrt(a^2 + b^2)  (2D vector norm)
void FusedNorm2D(float* out, const float* a, const float* b, size_t count);

// out = 1 / (1 + exp(-x))  (sigmoid activation)
void FusedSigmoid(float* out, const float* x, size_t count);

// out = max(0, x)  (ReLU activation)
void FusedRelu(float* out, const float* x, size_t count);

// out = sig_x * (1 - sig_x)  (sigmoid gradient, assumes sig_x is already sigmoid output)
void FusedSigmoidGrad(float* out, const float* sig_x, size_t count);

// =============================================================================
// Fused Reduction Operations
// =============================================================================

// sum(a * b)  (dot product)
float FusedDotProduct(const float* a, const float* b, size_t count);

// sum((a - b)^2)  (squared L2 distance)
float FusedSquaredDistance(const float* a, const float* b, size_t count);

// sum(a^2)  (squared L2 norm)
float FusedNormSquared(const float* a, size_t count);

// sum(exp(x))  (used in softmax denominator)
float FusedSumExp(const float* x, size_t count);

// =============================================================================
// Fused Transform-Reduce Operations
// =============================================================================

// Numerically stable softmax: out = exp(x - max) / sum(exp(x - max))
void FusedSoftmax(float* out, const float* x, size_t count);

// =============================================================================
// BLAS-like Fused Operations
// =============================================================================

// y = alpha * x + y  (axpy)
void FusedAxpy(float* y, float alpha, const float* x, size_t count);

// =============================================================================
// Neural Network Fused Kernels
// =============================================================================

// out = x * sigmoid(1.702 * x)  (GELU activation - fast approximation)
void FusedGelu(float* out, const float* x, size_t count);

// Layer normalization: out = (x - mean) / sqrt(var + eps) * gamma + beta
void FusedLayerNorm(float* out, const float* x, size_t count, float gamma, float beta,
                    float epsilon = 1e-5f);

// out = max(alpha * x, x)  (Leaky ReLU)
void FusedLeakyRelu(float* out, const float* x, float alpha, size_t count);

// out = x * sigmoid(x)  (Swish/SiLU activation)
void FusedSwish(float* out, const float* x, size_t count);

// =============================================================================
// Multi-Operation Fused Chains
// =============================================================================

// out = ((a * b) + c) * d  (4-way fused)
void FusedMulAddMul(float* out, const float* a, const float* b, const float* c, const float* d,
                    size_t count);

}  // namespace simd
}  // namespace bud
