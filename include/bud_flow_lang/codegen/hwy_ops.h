// =============================================================================
// Bud Flow Lang - Highway SIMD Operations Wrapper
// =============================================================================
//
// This module provides type-safe, portable SIMD operations using Google Highway.
// All operations are designed for:
// - Automatic multi-target dispatch (SSE4, AVX2, AVX-512, NEON, SVE, RVV)
// - Safe remainder handling for non-vector-aligned sizes
// - Multiple accumulator patterns for optimal reduction performance
// - FMA fusion for numerical accuracy and performance
//
// Usage:
//   #include "bud_flow_lang/codegen/hwy_ops.h"
//
//   float* out = ...;
//   float* in = ...;
//   size_t count = ...;
//
//   bud::simd::Add(out, in, in, count);  // out[i] = in[i] + in[i]
//
// =============================================================================

#pragma once

// Include Highway base for HWY_RESTRICT macro
#include "bud_flow_lang/error.h"
#include "bud_flow_lang/type_system.h"

#include <hwy/base.h>

#include <cstddef>
#include <cstdint>

namespace bud {
namespace simd {

// =============================================================================
// Configuration
// =============================================================================

// Memory alignment for optimal SIMD performance (128 bytes = cache line)
constexpr size_t kAlignment = 128;

// Number of accumulators for reduction operations (optimal for hiding latency)
constexpr size_t kNumAccumulators = 4;

// =============================================================================
// Alignment Utilities
// =============================================================================

// Check if pointer is aligned to kAlignment
template <typename T>
[[nodiscard]] inline bool isAligned(const T* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % kAlignment) == 0;
}

// Round up to alignment boundary
[[nodiscard]] inline size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// =============================================================================
// Priority 1: Arithmetic Operations
// =============================================================================

// Element-wise addition: out[i] = a[i] + b[i]
void Add(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count);
void Add(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count);
void Add(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count);
void Add(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count);

// Element-wise subtraction: out[i] = a[i] - b[i]
void Sub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count);
void Sub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count);
void Sub(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count);
void Sub(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count);

// Element-wise multiplication: out[i] = a[i] * b[i]
void Mul(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count);
void Mul(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count);
void Mul(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count);
void Mul(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count);

// Element-wise division: out[i] = a[i] / b[i]
void Div(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count);
void Div(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count);

// Element-wise negation: out[i] = -a[i]
void Neg(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Neg(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void Neg(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void Neg(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);

// Element-wise absolute value: out[i] = |a[i]|
void Abs(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Abs(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void Abs(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void Abs(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 1: FMA Operations (Fused Multiply-Add)
// =============================================================================

// Fused multiply-add: out[i] = a[i] * b[i] + c[i]
void MulAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            const float* HWY_RESTRICT c, size_t count);
void MulAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            const double* HWY_RESTRICT c, size_t count);

// Fused multiply-subtract: out[i] = a[i] * b[i] - c[i]
void MulSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            const float* HWY_RESTRICT c, size_t count);
void MulSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            const double* HWY_RESTRICT c, size_t count);

// Fused negative multiply-add: out[i] = c[i] - a[i] * b[i]
void NegMulAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count);
void NegMulAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count);

// Fused multiply add/sub alternating:
// Even lanes: out[i] = a[i] * b[i] - c[i]
// Odd lanes: out[i] = a[i] * b[i] + c[i]
void MulAddSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count);
void MulAddSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count);

// =============================================================================
// Priority 1: MinMax Operations
// =============================================================================

// Element-wise minimum: out[i] = min(a[i], b[i])
void Min(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count);
void Min(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count);
void Min(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count);
void Min(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count);

// Element-wise maximum: out[i] = max(a[i], b[i])
void Max(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count);
void Max(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count);
void Max(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count);
void Max(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count);

// Element-wise clamp: out[i] = clamp(a[i], lo, hi)
void Clamp(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, float lo, float hi, size_t count);
void Clamp(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, double lo, double hi,
           size_t count);
void Clamp(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, int32_t lo, int32_t hi,
           size_t count);
void Clamp(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, int64_t lo, int64_t hi,
           size_t count);

// =============================================================================
// Priority 1: Math Operations
// =============================================================================

// Square root: out[i] = sqrt(a[i])
void Sqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Sqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Reciprocal square root (fast): out[i] ~= 1/sqrt(a[i])
void Rsqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Rsqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Exponential: out[i] = exp(a[i])
void Exp(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Exp(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Natural logarithm: out[i] = log(a[i])
void Log(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Log(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Sine: out[i] = sin(a[i])
void Sin(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Sin(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Cosine: out[i] = cos(a[i])
void Cos(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Cos(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Hyperbolic tangent: out[i] = tanh(a[i])
void Tanh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Tanh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 1: Comparison Operations (Return mask as uint8)
// =============================================================================

// Equal: out[i] = (a[i] == b[i]) ? 0xFF : 0x00
void Eq(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count);
void Eq(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count);
void Eq(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count);

// Not equal: out[i] = (a[i] != b[i]) ? 0xFF : 0x00
void Ne(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count);
void Ne(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count);
void Ne(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count);

// Less than: out[i] = (a[i] < b[i]) ? 0xFF : 0x00
void Lt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count);
void Lt(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count);
void Lt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count);

// Less than or equal: out[i] = (a[i] <= b[i]) ? 0xFF : 0x00
void Le(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count);
void Le(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count);
void Le(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count);

// Greater than: out[i] = (a[i] > b[i]) ? 0xFF : 0x00
void Gt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count);
void Gt(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count);
void Gt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count);

// Greater than or equal: out[i] = (a[i] >= b[i]) ? 0xFF : 0x00
void Ge(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count);
void Ge(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count);
void Ge(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count);

// =============================================================================
// Priority 1: Reduction Operations (Using Multiple Accumulators)
// =============================================================================

// Sum all elements: return sum(a[0..count-1])
[[nodiscard]] float ReduceSum(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] double ReduceSum(const double* HWY_RESTRICT a, size_t count);
[[nodiscard]] int32_t ReduceSum(const int32_t* HWY_RESTRICT a, size_t count);
[[nodiscard]] int64_t ReduceSum(const int64_t* HWY_RESTRICT a, size_t count);

// Find minimum: return min(a[0..count-1])
[[nodiscard]] float ReduceMin(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] double ReduceMin(const double* HWY_RESTRICT a, size_t count);
[[nodiscard]] int32_t ReduceMin(const int32_t* HWY_RESTRICT a, size_t count);
[[nodiscard]] int64_t ReduceMin(const int64_t* HWY_RESTRICT a, size_t count);

// Find maximum: return max(a[0..count-1])
[[nodiscard]] float ReduceMax(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] double ReduceMax(const double* HWY_RESTRICT a, size_t count);
[[nodiscard]] int32_t ReduceMax(const int32_t* HWY_RESTRICT a, size_t count);
[[nodiscard]] int64_t ReduceMax(const int64_t* HWY_RESTRICT a, size_t count);

// Dot product: return sum(a[i] * b[i])
[[nodiscard]] float Dot(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count);
[[nodiscard]] double Dot(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b, size_t count);

// =============================================================================
// Optimized Reductions with Multiple Accumulators (2-4x faster)
// =============================================================================
// These functions use 4-8 independent accumulators to hide instruction latency.
// Reference: https://en.algorithmica.org/hpc/simd/reduction/

// Fast sum with 4 accumulators
[[nodiscard]] float ReduceSumFast(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] double ReduceSumFast(const double* HWY_RESTRICT a, size_t count);
[[nodiscard]] int32_t ReduceSumFast(const int32_t* HWY_RESTRICT a, size_t count);
[[nodiscard]] int64_t ReduceSumFast(const int64_t* HWY_RESTRICT a, size_t count);

// Fast min/max with 4 accumulators
[[nodiscard]] float ReduceMinFast(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] float ReduceMaxFast(const float* HWY_RESTRICT a, size_t count);

// Fast dot product with 4 accumulators
[[nodiscard]] float DotFast(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count);
[[nodiscard]] double DotFast(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                             size_t count);

// Statistical reductions with multi-accumulator optimization
[[nodiscard]] float MeanFast(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] float VarianceFast(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] float SumOfSquaresFast(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] float NormL2Fast(const float* HWY_RESTRICT a, size_t count);

// =============================================================================
// Size-Specialized Kernels
// =============================================================================
// These functions automatically select the optimal implementation based on array size:
// - Small (<64): fully unrolled, minimal overhead
// - Medium (64-4096): 4x unrolled loop
// - Large (>4096): 8x unrolled + prefetching

// Size thresholds (can be tuned per-platform)
constexpr size_t kSizeSmallThreshold = 64;
constexpr size_t kSizeMediumThreshold = 4096;

// Size-specialized addition: out[i] = a[i] + b[i]
void AddSized(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, float* HWY_RESTRICT out,
              size_t count);
void AddSized(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b, double* HWY_RESTRICT out,
              size_t count);
void AddSized(const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
              int32_t* HWY_RESTRICT out, size_t count);

// Size-specialized multiplication: out[i] = a[i] * b[i]
void MulSized(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, float* HWY_RESTRICT out,
              size_t count);
void MulSized(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b, double* HWY_RESTRICT out,
              size_t count);

// Size-specialized fused multiply-add: out[i] = a[i] * b[i] + c[i]
void FmaSized(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
              float* HWY_RESTRICT out, size_t count);
void FmaSized(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
              const double* HWY_RESTRICT c, double* HWY_RESTRICT out, size_t count);

// Size-specialized reduction: returns sum of elements
[[nodiscard]] float ReduceSumSized(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] double ReduceSumSized(const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// Fused Kernels
// =============================================================================
// Fused operations eliminate temporary arrays and improve cache efficiency

// AddMul: out[i] = (a[i] + b[i]) * c[i]
void AddMul(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
            float* HWY_RESTRICT out, size_t count);
void AddMul(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            const double* HWY_RESTRICT c, double* HWY_RESTRICT out, size_t count);

// Note: MulSub and NegMulAdd already declared above with (out, a, b, c) signature

// SubMul: out[i] = (a[i] - b[i]) * c[i]
void SubMul(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
            float* HWY_RESTRICT out, size_t count);
void SubMul(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            const double* HWY_RESTRICT c, double* HWY_RESTRICT out, size_t count);

// Axpy: out[i] = alpha * x[i] + y[i] (BLAS-style)
void Axpy(float alpha, const float* HWY_RESTRICT x, const float* HWY_RESTRICT y,
          float* HWY_RESTRICT out, size_t count);
void Axpy(double alpha, const double* HWY_RESTRICT x, const double* HWY_RESTRICT y,
          double* HWY_RESTRICT out, size_t count);

// Axpby: out[i] = alpha * x[i] + beta * y[i]
void Axpby(float alpha, const float* HWY_RESTRICT x, float beta, const float* HWY_RESTRICT y,
           float* HWY_RESTRICT out, size_t count);
void Axpby(double alpha, const double* HWY_RESTRICT x, double beta, const double* HWY_RESTRICT y,
           double* HWY_RESTRICT out, size_t count);

// ScaleAdd: out[i] = scale * a[i] + offset (affine transform)
void ScaleAdd(const float* HWY_RESTRICT a, float scale, float offset, float* HWY_RESTRICT out,
              size_t count);
void ScaleAdd(const double* HWY_RESTRICT a, double scale, double offset, double* HWY_RESTRICT out,
              size_t count);

// SumSq: returns sum(a[i]^2) - fused square and reduce
[[nodiscard]] float SumSq(const float* HWY_RESTRICT a, size_t count);
[[nodiscard]] double SumSq(const double* HWY_RESTRICT a, size_t count);

// SumAbsDiff: returns sum(|a[i] - b[i]|) - L1 distance
[[nodiscard]] float SumAbsDiff(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                               size_t count);
[[nodiscard]] double SumAbsDiff(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                                size_t count);

// SumSqDiff: returns sum((a[i] - b[i])^2) - squared L2 distance
[[nodiscard]] float SumSqDiff(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                              size_t count);
[[nodiscard]] double SumSqDiff(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                               size_t count);

// =============================================================================
// Prefetch Utilities
// =============================================================================
// Software prefetch hints for improved memory access performance

// Prefetch distance constants
constexpr size_t kPrefetchBytes = 256;  // Bytes to prefetch ahead (4 cache lines)

// Get optimal prefetch distance in elements for a given type
template <typename T>
constexpr size_t GetPrefetchDistance() {
    return kPrefetchBytes / sizeof(T);
}

// Prefetch for read with locality hint (0=no locality, 3=keep in cache)
template <typename T>
inline void PrefetchRead(const T* ptr, int locality = 3) {
    if (ptr != nullptr) {
        __builtin_prefetch(ptr, 0, locality);
    }
}

// Prefetch for write with locality hint
template <typename T>
inline void PrefetchWrite(T* ptr, int locality = 3) {
    if (ptr != nullptr) {
        __builtin_prefetch(ptr, 1, locality);
    }
}

// Prefetch stream for sequential access patterns
template <typename T>
class PrefetchStream {
  public:
    PrefetchStream(const T* base, size_t count)
        : base_(base), count_(count), prefetch_distance_(GetPrefetchDistance<T>()) {}

    void prefetchAhead(size_t current_idx) {
        size_t prefetch_idx = current_idx + prefetch_distance_;
        if (prefetch_idx < count_) {
            __builtin_prefetch(base_ + prefetch_idx, 0, 3);
        }
    }

  private:
    const T* base_;
    size_t count_;
    size_t prefetch_distance_;
};

// Add without prefetch (for benchmarking comparison)
void AddNoPrefetch(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                   float* HWY_RESTRICT out, size_t count);
void AddNoPrefetch(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                   double* HWY_RESTRICT out, size_t count);

// Process with prefetch (for strided access)
void ProcessWithPrefetch(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, size_t count,
                         size_t stride);

// =============================================================================
// Dynamic Tier Thresholds
// =============================================================================
// Runtime-configurable size tier thresholds for adaptive optimization

// Size tier enumeration
enum class SizeTier {
    Small,   // Fully unrolled, minimal overhead
    Medium,  // 4x unrolled
    Large    // 8x unrolled + prefetching
};

// Tier threshold configuration
struct TierThresholds {
    size_t small_to_medium;  // Threshold for small->medium transition
    size_t medium_to_large;  // Threshold for medium->large transition
};

// Cache information for threshold tuning
struct CacheInfo {
    size_t l1_data_cache;    // L1 data cache size in bytes
    size_t l2_cache;         // L2 cache size in bytes
    size_t cache_line_size;  // Cache line size in bytes
};

// Get default thresholds (compile-time constants)
TierThresholds GetDefaultThresholds();

// Get/set global thresholds
TierThresholds GetGlobalThresholds();
void SetGlobalThresholds(const TierThresholds& thresholds);

// Get the size tier for a given element count
SizeTier GetSizeTier(size_t count);

// Detect cache information from the system
CacheInfo DetectCacheInfo();

// Calculate optimal thresholds from cache info
TierThresholds CalculateThresholdsFromCache(const CacheInfo& info);

// Auto-tune thresholds by benchmarking
TierThresholds AutoTuneThresholds();

// Dynamic-threshold versions of size-specialized operations
void AddSizedDynamic(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                     float* HWY_RESTRICT out, size_t count);
void AddSizedDynamic(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
                     double* HWY_RESTRICT out, size_t count);

// =============================================================================
// Priority 1: Conditional Selection (Masked Operations)
// =============================================================================

// Select based on mask: out[i] = mask[i] ? yes[i] : no[i]
void Select(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
            const float* HWY_RESTRICT yes, const float* HWY_RESTRICT no, size_t count);
void Select(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
            const double* HWY_RESTRICT yes, const double* HWY_RESTRICT no, size_t count);
void Select(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
            const int32_t* HWY_RESTRICT yes, const int32_t* HWY_RESTRICT no, size_t count);

// =============================================================================
// Priority 2: Extended Math Operations
// =============================================================================

// Exponential base 2: out[i] = 2^a[i]
void Exp2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Exp2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Logarithm base 2: out[i] = log2(a[i])
void Log2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Log2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Logarithm base 10: out[i] = log10(a[i])
void Log10(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Log10(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Hyperbolic sine: out[i] = sinh(a[i])
void Sinh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Sinh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Hyperbolic cosine: out[i] = cosh(a[i])
void Cosh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Cosh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 2: Inverse Trigonometric Operations
// =============================================================================

// Arc sine: out[i] = asin(a[i])
void Asin(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Asin(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Arc cosine: out[i] = acos(a[i])
void Acos(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Acos(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Arc tangent: out[i] = atan(a[i])
void Atan(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Atan(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Arc tangent 2: out[i] = atan2(y[i], x[i])
void Atan2(float* HWY_RESTRICT out, const float* HWY_RESTRICT y, const float* HWY_RESTRICT x,
           size_t count);
void Atan2(double* HWY_RESTRICT out, const double* HWY_RESTRICT y, const double* HWY_RESTRICT x,
           size_t count);

// =============================================================================
// Priority 2: Rounding Operations
// =============================================================================

// Round to nearest: out[i] = round(a[i])
void Round(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Round(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Floor: out[i] = floor(a[i])
void Floor(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Floor(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Ceiling: out[i] = ceil(a[i])
void Ceil(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Ceil(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Truncate: out[i] = trunc(a[i])
void Trunc(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Trunc(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 2: Bitwise Operations (Integer Types)
// =============================================================================

// Bitwise AND: out[i] = a[i] & b[i]
void BitwiseAnd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const int32_t* HWY_RESTRICT b, size_t count);
void BitwiseAnd(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                const int64_t* HWY_RESTRICT b, size_t count);

// Bitwise OR: out[i] = a[i] | b[i]
void BitwiseOr(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT b, size_t count);
void BitwiseOr(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
               const int64_t* HWY_RESTRICT b, size_t count);

// Bitwise XOR: out[i] = a[i] ^ b[i]
void BitwiseXor(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const int32_t* HWY_RESTRICT b, size_t count);
void BitwiseXor(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                const int64_t* HWY_RESTRICT b, size_t count);

// Bitwise NOT: out[i] = ~a[i]
void BitwiseNot(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void BitwiseNot(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);

// Left shift by constant: out[i] = a[i] << shift
template <int kShift>
void ShiftLeft(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
template <int kShift>
void ShiftLeft(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);

// Right shift by constant: out[i] = a[i] >> shift
template <int kShift>
void ShiftRight(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
template <int kShift>
void ShiftRight(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// Type-Generic Dispatch (For Use with ScalarType Enum)
// =============================================================================

// Execute operation based on runtime type
Result<void> DispatchAdd(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchSub(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchMul(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchDiv(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchNeg(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAbs(void* out, const void* a, size_t count, ScalarType dtype);

Result<void> DispatchSqrt(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchExp(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchLog(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchSin(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchCos(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchTanh(void* out, const void* a, size_t count, ScalarType dtype);

// Extended Transcendental Dispatchers
Result<void> DispatchExp2(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchLog2(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchLog10(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchSinh(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchCosh(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAsinh(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAcosh(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAtanh(void* out, const void* a, size_t count, ScalarType dtype);

// Inverse Trigonometric Dispatchers
Result<void> DispatchAsin(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAcos(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAtan(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchAtan2(void* out, const void* y, const void* x, size_t count, ScalarType dtype);

// FMA Variant Dispatchers
Result<void> DispatchMulSub(void* out, const void* a, const void* b, const void* c, size_t count,
                            ScalarType dtype);
Result<void> DispatchNegMulAdd(void* out, const void* a, const void* b, const void* c, size_t count,
                               ScalarType dtype);

// Reduction Dispatchers (return scalar results)
[[nodiscard]] Result<float> DispatchReduceSum(const void* a, size_t count, ScalarType dtype);
[[nodiscard]] Result<float> DispatchReduceMin(const void* a, size_t count, ScalarType dtype);
[[nodiscard]] Result<float> DispatchReduceMax(const void* a, size_t count, ScalarType dtype);

// Comparison Dispatchers (produce float masks: 1.0f for true, 0.0f for false)
Result<void> DispatchEq(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchNe(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchLt(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchLe(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchGt(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchGe(void* out, const void* a, const void* b, size_t count, ScalarType dtype);

// Min/Max Dispatchers (element-wise)
Result<void> DispatchMin(void* out, const void* a, const void* b, size_t count, ScalarType dtype);
Result<void> DispatchMax(void* out, const void* a, const void* b, size_t count, ScalarType dtype);

// Conditional Dispatchers
Result<void> DispatchWhere(void* out, const void* mask, const void* a, const void* b, size_t count,
                           ScalarType dtype);

// Factory Dispatchers
Result<void> DispatchFill(void* out, float value, size_t count, ScalarType dtype);
Result<void> DispatchArange(void* out, float start, float step, size_t count, ScalarType dtype);

// Fused Operation Dispatchers
Result<void> DispatchLerp(void* out, const void* a, const void* b, float t, size_t count,
                          ScalarType dtype);

// Additional Transcendental Dispatchers
Result<void> DispatchTan(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchSigmoid(void* out, const void* a, size_t count, ScalarType dtype);

// Rounding Dispatchers
Result<void> DispatchFloor(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchCeil(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchRound(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchTrunc(void* out, const void* a, size_t count, ScalarType dtype);

// Special Value Check Dispatchers (output as float masks: 1.0f for true, 0.0f for false)
Result<void> DispatchIsNaN(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchIsInf(void* out, const void* a, size_t count, ScalarType dtype);
Result<void> DispatchIsFinite(void* out, const void* a, size_t count, ScalarType dtype);

// Bitwise Operation Dispatchers (integer types only)
Result<void> DispatchBitwiseAnd(void* out, const void* a, const void* b, size_t count,
                                ScalarType dtype);
Result<void> DispatchBitwiseOr(void* out, const void* a, const void* b, size_t count,
                               ScalarType dtype);
Result<void> DispatchBitwiseXor(void* out, const void* a, const void* b, size_t count,
                                ScalarType dtype);
Result<void> DispatchBitwiseNot(void* out, const void* a, size_t count, ScalarType dtype);

// =============================================================================
// Priority 3: Special Value Checks
// =============================================================================

// Check for NaN: out[i] = isnan(a[i]) ? 0xFF : 0x00
void IsNaN(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void IsNaN(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Check for infinity: out[i] = isinf(a[i]) ? 0xFF : 0x00
void IsInf(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void IsInf(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Check for finite: out[i] = isfinite(a[i]) ? 0xFF : 0x00
void IsFinite(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void IsFinite(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 3: Variable Shift Operations
// =============================================================================

// Left shift by variable amounts: out[i] = a[i] << shift[i]
void ShiftLeftVar(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                  const int32_t* HWY_RESTRICT shift, size_t count);
void ShiftLeftVar(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                  const int64_t* HWY_RESTRICT shift, size_t count);
void ShiftLeftVar(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                  const uint32_t* HWY_RESTRICT shift, size_t count);

// Right shift by variable amounts: out[i] = a[i] >> shift[i]
void ShiftRightVar(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                   const int32_t* HWY_RESTRICT shift, size_t count);
void ShiftRightVar(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                   const int64_t* HWY_RESTRICT shift, size_t count);
void ShiftRightVar(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                   const uint32_t* HWY_RESTRICT shift, size_t count);

// =============================================================================
// Priority 3: Type Conversion Operations
// =============================================================================

// Promote to wider type
void PromoteTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count);
void PromoteTo(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count);
void PromoteTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);

// Demote to narrower type
void DemoteTo(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void DemoteTo(float* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Convert between same-width types
void ConvertTo(float* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void ConvertTo(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 3: Gather/Scatter Operations
// =============================================================================

// Gather: out[i] = base[indices[i]]
void Gather(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
            const int32_t* HWY_RESTRICT indices, size_t count);
void Gather(double* HWY_RESTRICT out, const double* HWY_RESTRICT base,
            const int64_t* HWY_RESTRICT indices, size_t count);
void Gather(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
            const int32_t* HWY_RESTRICT indices, size_t count);

// Scatter: base[indices[i]] = values[i]
void Scatter(const float* HWY_RESTRICT values, float* HWY_RESTRICT base,
             const int32_t* HWY_RESTRICT indices, size_t count);
void Scatter(const double* HWY_RESTRICT values, double* HWY_RESTRICT base,
             const int64_t* HWY_RESTRICT indices, size_t count);
void Scatter(const int32_t* HWY_RESTRICT values, int32_t* HWY_RESTRICT base,
             const int32_t* HWY_RESTRICT indices, size_t count);

// =============================================================================
// Priority 3: Horizontal Reductions (Per-Vector)
// =============================================================================

// Sum of lanes within each vector
void SumOfLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void SumOfLanes(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void SumOfLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// Pairwise add: out[i] = a[2*i] + a[2*i+1]
void PairwiseAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void PairwiseAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void PairwiseAdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 3: Saturation Operations
// =============================================================================

// Saturating add: out[i] = saturate(a[i] + b[i])
void SaturatedAdd(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, size_t count);
void SaturatedAdd(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                  const uint8_t* HWY_RESTRICT b, size_t count);

// Saturating subtract: out[i] = saturate(a[i] - b[i])
void SaturatedSub(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, size_t count);
void SaturatedSub(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                  const uint8_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// Priority 3: Broadcast Operations
// =============================================================================

// Broadcast scalar to all elements: out[i] = value
void Broadcast(float* HWY_RESTRICT out, float value, size_t count);
void Broadcast(double* HWY_RESTRICT out, double value, size_t count);
void Broadcast(int32_t* HWY_RESTRICT out, int32_t value, size_t count);
void Broadcast(int64_t* HWY_RESTRICT out, int64_t value, size_t count);

// =============================================================================
// Priority 3: Compress/Expand Operations
// =============================================================================

// Compress: write elements where mask[i] is non-zero, returns count of written elements
size_t Compress(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT mask, size_t count);
size_t Compress(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT mask, size_t count);
size_t Compress(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT mask, size_t count);

// Expand: read elements into positions where mask[i] is non-zero
void Expand(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT mask,
            size_t count);
void Expand(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
            const uint8_t* HWY_RESTRICT mask, size_t count);
void Expand(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
            const uint8_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// Priority 3: Interleave/Deinterleave Operations
// =============================================================================

// Full interleave: out = {a[0], b[0], a[1], b[1], ...}, output size = count*2
void Interleave(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                size_t count);
void Interleave(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                const double* HWY_RESTRICT b, size_t count);

// Deinterleave: extract interleaved values back to separate arrays
void Deinterleave(float* HWY_RESTRICT out_a, float* HWY_RESTRICT out_b,
                  const float* HWY_RESTRICT interleaved, size_t count);
void Deinterleave(double* HWY_RESTRICT out_a, double* HWY_RESTRICT out_b,
                  const double* HWY_RESTRICT interleaved, size_t count);

// =============================================================================
// Priority 3: 8/16-bit Integer Operations
// =============================================================================

// 8-bit integer arithmetic
void Add(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count);
void Sub(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count);
void Mul(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count);
void Min(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count);
void Max(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count);

// 16-bit integer arithmetic
void Add(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count);
void Sub(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count);
void Mul(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count);
void Min(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count);
void Max(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count);

// Unsigned 8-bit arithmetic
void Add(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count);
void Sub(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count);
void Mul(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count);
void Min(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count);
void Max(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count);

// Unsigned 16-bit arithmetic
void Add(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count);
void Sub(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count);
void Mul(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count);
void Min(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count);
void Max(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count);

// Unsigned 32-bit arithmetic
void Add(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count);
void Sub(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count);
void Mul(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count);
void Min(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count);
void Max(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count);

// Unsigned 64-bit arithmetic
void Add(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count);
void Sub(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count);
void Mul(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count);
void Min(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count);
void Max(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count);

// =============================================================================
// Priority 3: Additional Math Operations
// =============================================================================

// Tangent: out[i] = tan(a[i])
void Tan(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Tan(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Exponential minus one: out[i] = exp(a[i]) - 1
void Expm1(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Expm1(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Log plus one: out[i] = log(1 + a[i])
void Log1p(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Log1p(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Absolute difference: out[i] = |a[i] - b[i]|
void AbsDiff(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
             size_t count);
void AbsDiff(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
             size_t count);
void AbsDiff(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
             const int32_t* HWY_RESTRICT b, size_t count);

// Copy sign: out[i] = |a[i]| * sign(b[i])
void CopySign(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              size_t count);
void CopySign(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
              size_t count);

// Approximate reciprocal: out[i] ~= 1/a[i]
void ApproxReciprocal(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);

// Approximate reciprocal square root: out[i] ~= 1/sqrt(a[i])
void ApproxReciprocalSqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void ApproxReciprocalSqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 3: Bit Counting Operations
// =============================================================================

// Population count: out[i] = popcount(a[i])
void PopCount(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void PopCount(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);
void PopCount(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void PopCount(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// Leading zero count: out[i] = clz(a[i])
void LeadingZeroCount(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void LeadingZeroCount(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);
void LeadingZeroCount(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void LeadingZeroCount(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// Trailing zero count: out[i] = ctz(a[i])
void TrailingZeroCount(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void TrailingZeroCount(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);
void TrailingZeroCount(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void TrailingZeroCount(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// Priority 3: Averaging Operations
// =============================================================================

// Averaging with rounding: out[i] = (a[i] + b[i] + 1) / 2
void AverageRound(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                  const uint8_t* HWY_RESTRICT b, size_t count);
void AverageRound(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                  const uint16_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P0: Memory Operations - Load/Store
// =============================================================================

// Load N elements from memory (aligned or unaligned)
// Returns vector with loaded elements, remaining lanes are zero
template <typename T>
void Load(T* HWY_RESTRICT out, const T* HWY_RESTRICT src, size_t count);
template <typename T>
void LoadU(T* HWY_RESTRICT out, const T* HWY_RESTRICT src, size_t count);

// Load exactly N elements, remaining lanes zeroed
template <typename T>
void LoadN(T* HWY_RESTRICT out, const T* HWY_RESTRICT src, size_t n, size_t max_n);

// Load N elements with fallback value for out-of-bounds
template <typename T>
void LoadNOr(T* HWY_RESTRICT out, T fallback, const T* HWY_RESTRICT src, size_t n, size_t max_n);

// Store N elements to memory (aligned or unaligned)
template <typename T>
void Store(const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, size_t count);
template <typename T>
void StoreU(const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, size_t count);

// Store exactly N elements
template <typename T>
void StoreN(const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, size_t n);

// Explicit instantiations for common types
void Load(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void Load(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);
void Load(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, size_t count);
void Load(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src, size_t count);
void Load(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count);
void Load(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT src, size_t count);
void Load(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src, size_t count);
void Load(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src, size_t count);

void Store(const float* HWY_RESTRICT src, float* HWY_RESTRICT dst, size_t count);
void Store(const double* HWY_RESTRICT src, double* HWY_RESTRICT dst, size_t count);
void Store(const int32_t* HWY_RESTRICT src, int32_t* HWY_RESTRICT dst, size_t count);
void Store(const int64_t* HWY_RESTRICT src, int64_t* HWY_RESTRICT dst, size_t count);
void Store(const uint8_t* HWY_RESTRICT src, uint8_t* HWY_RESTRICT dst, size_t count);
void Store(const uint16_t* HWY_RESTRICT src, uint16_t* HWY_RESTRICT dst, size_t count);
void Store(const uint32_t* HWY_RESTRICT src, uint32_t* HWY_RESTRICT dst, size_t count);
void Store(const uint64_t* HWY_RESTRICT src, uint64_t* HWY_RESTRICT dst, size_t count);

// =============================================================================
// P0: BitCast Operations - Type Reinterpretation
// =============================================================================

// Reinterpret bits of one type as another (no conversion, just reinterpretation)
// Important: Source and destination must have same size
void BitCastFloat32ToInt32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void BitCastInt32ToFloat32(float* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, size_t count);
void BitCastFloat64ToInt64(int64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);
void BitCastInt64ToFloat64(double* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src, size_t count);
void BitCastFloat32ToUint32(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                            size_t count);
void BitCastUint32ToFloat32(float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src,
                            size_t count);
void BitCastFloat64ToUint64(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                            size_t count);
void BitCastUint64ToFloat64(double* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src,
                            size_t count);

// =============================================================================
// P0: Mask Operations
// =============================================================================

// Count number of true (non-zero) elements in mask
size_t CountTrue(const uint8_t* HWY_RESTRICT mask, size_t count);
size_t CountTrue(const uint32_t* HWY_RESTRICT mask, size_t count);
size_t CountTrue(const uint64_t* HWY_RESTRICT mask, size_t count);

// Check if all elements in mask are true (non-zero)
bool AllTrue(const uint8_t* HWY_RESTRICT mask, size_t count);
bool AllTrue(const uint32_t* HWY_RESTRICT mask, size_t count);
bool AllTrue(const uint64_t* HWY_RESTRICT mask, size_t count);

// Check if all elements in mask are false (zero)
bool AllFalse(const uint8_t* HWY_RESTRICT mask, size_t count);
bool AllFalse(const uint32_t* HWY_RESTRICT mask, size_t count);
bool AllFalse(const uint64_t* HWY_RESTRICT mask, size_t count);

// Find index of first true element (-1 if none found)
int64_t FindFirstTrue(const uint8_t* HWY_RESTRICT mask, size_t count);
int64_t FindFirstTrue(const uint32_t* HWY_RESTRICT mask, size_t count);
int64_t FindFirstTrue(const uint64_t* HWY_RESTRICT mask, size_t count);

// Find index of last true element (-1 if none found)
int64_t FindLastTrue(const uint8_t* HWY_RESTRICT mask, size_t count);
int64_t FindLastTrue(const uint32_t* HWY_RESTRICT mask, size_t count);
int64_t FindLastTrue(const uint64_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// P1: Shuffle/Permute Operations
// =============================================================================

// Reverse all elements: out[i] = a[count-1-i]
void Reverse(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Reverse(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void Reverse(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void Reverse(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);
void Reverse(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count);
void Reverse(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, size_t count);
void Reverse(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void Reverse(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// Table lookup: out[i] = table[indices[i]]
void TableLookup(float* HWY_RESTRICT out, const float* HWY_RESTRICT table,
                 const int32_t* HWY_RESTRICT indices, size_t table_size, size_t count);
void TableLookup(double* HWY_RESTRICT out, const double* HWY_RESTRICT table,
                 const int64_t* HWY_RESTRICT indices, size_t table_size, size_t count);
void TableLookup(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT table,
                 const int32_t* HWY_RESTRICT indices, size_t table_size, size_t count);
void TableLookup(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT table,
                 const uint8_t* HWY_RESTRICT indices, size_t table_size, size_t count);

// Rotate elements left: out[i] = a[(i + shift) % count]
void RotateLeft(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t shift, size_t count);
void RotateLeft(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t shift,
                size_t count);

// Rotate elements right: out[i] = a[(i - shift + count) % count]
void RotateRight(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t shift, size_t count);
void RotateRight(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t shift,
                 size_t count);

// Slide elements up (shift in zeros from bottom)
void SlideUp(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t shift, size_t count);
void SlideUp(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t shift, size_t count);

// Slide elements down (shift in zeros from top)
void SlideDown(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t shift, size_t count);
void SlideDown(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t shift,
               size_t count);

// Concatenate lower halves of a and b
void ConcatLowerLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                      const float* HWY_RESTRICT b, size_t count);
void ConcatLowerLower(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count);

// Concatenate upper halves of a and b
void ConcatUpperUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                      const float* HWY_RESTRICT b, size_t count);
void ConcatUpperUpper(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count);

// Select odd elements from a, even from b: out[2i]=b[2i], out[2i+1]=a[2i+1]
void OddEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT odd, const float* HWY_RESTRICT even,
             size_t count);
void OddEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT odd,
             const int32_t* HWY_RESTRICT even, size_t count);

// =============================================================================
// P1: Multi-stream Interleaved Load/Store
// =============================================================================

// Load 2 interleaved streams: a0,b0,a1,b1,... -> a[], b[]
void LoadInterleaved2(float* HWY_RESTRICT a, float* HWY_RESTRICT b,
                      const float* HWY_RESTRICT interleaved, size_t count);
void LoadInterleaved2(int32_t* HWY_RESTRICT a, int32_t* HWY_RESTRICT b,
                      const int32_t* HWY_RESTRICT interleaved, size_t count);
void LoadInterleaved2(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                      const uint8_t* HWY_RESTRICT interleaved, size_t count);

// Load 3 interleaved streams: a0,b0,c0,a1,b1,c1,... -> a[], b[], c[]
void LoadInterleaved3(float* HWY_RESTRICT a, float* HWY_RESTRICT b, float* HWY_RESTRICT c,
                      const float* HWY_RESTRICT interleaved, size_t count);
void LoadInterleaved3(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b, uint8_t* HWY_RESTRICT c,
                      const uint8_t* HWY_RESTRICT interleaved, size_t count);

// Load 4 interleaved streams: a0,b0,c0,d0,... -> a[], b[], c[], d[]
void LoadInterleaved4(float* HWY_RESTRICT a, float* HWY_RESTRICT b, float* HWY_RESTRICT c,
                      float* HWY_RESTRICT d, const float* HWY_RESTRICT interleaved, size_t count);
void LoadInterleaved4(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b, uint8_t* HWY_RESTRICT c,
                      uint8_t* HWY_RESTRICT d, const uint8_t* HWY_RESTRICT interleaved,
                      size_t count);

// Store 2 streams as interleaved: a[], b[] -> a0,b0,a1,b1,...
void StoreInterleaved2(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, size_t count);
void StoreInterleaved2(int32_t* HWY_RESTRICT interleaved, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count);
void StoreInterleaved2(uint8_t* HWY_RESTRICT interleaved, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, size_t count);

// Store 3 streams as interleaved: a[], b[], c[] -> a0,b0,c0,a1,b1,c1,...
void StoreInterleaved3(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c, size_t count);
void StoreInterleaved3(uint8_t* HWY_RESTRICT interleaved, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT c, size_t count);

// Store 4 streams as interleaved
void StoreInterleaved4(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                       const float* HWY_RESTRICT d, size_t count);
void StoreInterleaved4(uint8_t* HWY_RESTRICT interleaved, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT c,
                       const uint8_t* HWY_RESTRICT d, size_t count);

// =============================================================================
// P1: Additional Reduction Operations
// =============================================================================

// Minimum of all lanes (returns single value broadcast to out)
void MinOfLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void MinOfLanes(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void MinOfLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void MinOfLanes(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);
void MinOfLanes(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);

// Maximum of all lanes (returns single value broadcast to out)
void MaxOfLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void MaxOfLanes(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void MaxOfLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);
void MaxOfLanes(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count);
void MaxOfLanes(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P1: Sorting Operations (VQSort Integration)
// =============================================================================

// Sort array in ascending order (uses vectorized quicksort)
void Sort(float* HWY_RESTRICT data, size_t count);
void Sort(double* HWY_RESTRICT data, size_t count);
void Sort(int32_t* HWY_RESTRICT data, size_t count);
void Sort(int64_t* HWY_RESTRICT data, size_t count);
void Sort(uint32_t* HWY_RESTRICT data, size_t count);
void Sort(uint64_t* HWY_RESTRICT data, size_t count);

// Sort array in descending order
void SortDescending(float* HWY_RESTRICT data, size_t count);
void SortDescending(double* HWY_RESTRICT data, size_t count);
void SortDescending(int32_t* HWY_RESTRICT data, size_t count);
void SortDescending(int64_t* HWY_RESTRICT data, size_t count);

// Partial sort: puts smallest k elements at beginning (not necessarily sorted among themselves)
void PartialSort(float* HWY_RESTRICT data, size_t k, size_t count);
void PartialSort(int32_t* HWY_RESTRICT data, size_t k, size_t count);

// =============================================================================
// P1: Find/Search Operations
// =============================================================================

// Find first occurrence of value, returns index or -1 if not found
int64_t Find(const float* HWY_RESTRICT data, float value, size_t count);
int64_t Find(const int32_t* HWY_RESTRICT data, int32_t value, size_t count);
int64_t Find(const uint8_t* HWY_RESTRICT data, uint8_t value, size_t count);

// Find first element satisfying predicate (greater than threshold)
int64_t FindGt(const float* HWY_RESTRICT data, float threshold, size_t count);
int64_t FindGt(const int32_t* HWY_RESTRICT data, int32_t threshold, size_t count);

// Find first element satisfying predicate (less than threshold)
int64_t FindLt(const float* HWY_RESTRICT data, float threshold, size_t count);
int64_t FindLt(const int32_t* HWY_RESTRICT data, int32_t threshold, size_t count);

// =============================================================================
// P1: Transform Operations
// =============================================================================

// In-place transform: data[i] = f(data[i]) where f is Add(scalar)
void TransformAdd(float* HWY_RESTRICT data, float scalar, size_t count);
void TransformAdd(int32_t* HWY_RESTRICT data, int32_t scalar, size_t count);

// In-place transform: data[i] = data[i] * scalar
void TransformMul(float* HWY_RESTRICT data, float scalar, size_t count);
void TransformMul(int32_t* HWY_RESTRICT data, int32_t scalar, size_t count);

// Fill array with value
void Fill(float* HWY_RESTRICT data, float value, size_t count);
void Fill(double* HWY_RESTRICT data, double value, size_t count);
void Fill(int32_t* HWY_RESTRICT data, int32_t value, size_t count);
void Fill(int64_t* HWY_RESTRICT data, int64_t value, size_t count);
void Fill(uint8_t* HWY_RESTRICT data, uint8_t value, size_t count);

// Copy array
void Copy(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, size_t count);
void Copy(double* HWY_RESTRICT dst, const double* HWY_RESTRICT src, size_t count);
void Copy(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count);
void Copy(int64_t* HWY_RESTRICT dst, const int64_t* HWY_RESTRICT src, size_t count);
void Copy(uint8_t* HWY_RESTRICT dst, const uint8_t* HWY_RESTRICT src, size_t count);

// =============================================================================
// P2: Additional Math Functions
// =============================================================================

// Combined sin and cos (more efficient than separate calls)
void SinCos(float* HWY_RESTRICT sin_out, float* HWY_RESTRICT cos_out, const float* HWY_RESTRICT a,
            size_t count);
void SinCos(double* HWY_RESTRICT sin_out, double* HWY_RESTRICT cos_out,
            const double* HWY_RESTRICT a, size_t count);

// Hypotenuse: out[i] = sqrt(a[i]^2 + b[i]^2)
void Hypot(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
           size_t count);
void Hypot(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
           size_t count);

// Inverse hyperbolic functions
void Asinh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Asinh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void Acosh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Acosh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void Atanh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Atanh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Power function: out[i] = base[i] ^ exp[i]
void Pow(float* HWY_RESTRICT out, const float* HWY_RESTRICT base, const float* HWY_RESTRICT exp,
         size_t count);
void Pow(double* HWY_RESTRICT out, const double* HWY_RESTRICT base, const double* HWY_RESTRICT exp,
         size_t count);

// Scalar power: out[i] = base[i] ^ scalar_exp
void PowScalar(float* HWY_RESTRICT out, const float* HWY_RESTRICT base, float scalar_exp,
               size_t count);
void PowScalar(double* HWY_RESTRICT out, const double* HWY_RESTRICT base, double scalar_exp,
               size_t count);

// =============================================================================
// P2: Additional Bitwise Operations
// =============================================================================

// AndNot: out[i] = (~a[i]) & b[i]
void AndNot(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
            size_t count);
void AndNot(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
            size_t count);
void AndNot(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
            const uint32_t* HWY_RESTRICT b, size_t count);
void AndNot(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
            const uint64_t* HWY_RESTRICT b, size_t count);

// Rotate bits left within each element
void RotateBitsLeft(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, int shift,
                    size_t count);
void RotateBitsLeft(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, int shift,
                    size_t count);

// Rotate bits right within each element
void RotateBitsRight(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, int shift,
                     size_t count);
void RotateBitsRight(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, int shift,
                     size_t count);

// Reverse bytes within each element
void ReverseBytes(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, size_t count);
void ReverseBytes(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void ReverseBytes(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P2: Matrix-Vector Operations
// =============================================================================

// Matrix-vector multiply: out = A * x (A is row-major, rows x cols matrix)
void MatVec(float* HWY_RESTRICT out, const float* HWY_RESTRICT A, const float* HWY_RESTRICT x,
            size_t rows, size_t cols);
void MatVec(double* HWY_RESTRICT out, const double* HWY_RESTRICT A, const double* HWY_RESTRICT x,
            size_t rows, size_t cols);

// Batched dot product: out[i] = dot(A[i], B[i]) where A,B are arrays of vectors
void BatchedDot(float* HWY_RESTRICT out, const float* HWY_RESTRICT A, const float* HWY_RESTRICT B,
                size_t num_vectors, size_t vector_len);
void BatchedDot(double* HWY_RESTRICT out, const double* HWY_RESTRICT A,
                const double* HWY_RESTRICT B, size_t num_vectors, size_t vector_len);

// =============================================================================
// P2: Additional Comparison/Selection
// =============================================================================

// Min ignoring NaN (returns non-NaN if one operand is NaN)
void MinNumber(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               size_t count);
void MinNumber(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               size_t count);

// Max ignoring NaN
void MaxNumber(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               size_t count);
void MaxNumber(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               size_t count);

// IfThenElseZero: out[i] = mask[i] ? yes[i] : 0
void IfThenElseZero(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                    const float* HWY_RESTRICT yes, size_t count);
void IfThenElseZero(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                    const int32_t* HWY_RESTRICT yes, size_t count);

// IfThenZeroElse: out[i] = mask[i] ? 0 : no[i]
void IfThenZeroElse(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                    const float* HWY_RESTRICT no, size_t count);
void IfThenZeroElse(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                    const int32_t* HWY_RESTRICT no, size_t count);

// =============================================================================
// P2: Half-Precision Conversions (Float16 / BFloat16)
// =============================================================================

// Convert Float32 array to Float16 (IEEE 754 half-precision)
// out is uint16_t* as float16_t requires hwy/base.h
void F32ToF16(uint16_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);

// Convert Float16 array to Float32
void F16ToF32(float* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT in, size_t count);

// Convert Float32 array to BFloat16 (Brain floating point)
void F32ToBF16(uint16_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);

// Convert BFloat16 array to Float32
void BF16ToF32(float* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT in, size_t count);

// Convert Float64 array to Float16
void F64ToF16(uint16_t* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count);

// Convert Float64 array to BFloat16
void F64ToBF16(uint16_t* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count);

// =============================================================================
// P0: Masked Arithmetic Operations
// =============================================================================
// Masked operations apply the operation only where mask is non-zero.
// Where mask is zero, the output is copied from the 'no' array.
// out[i] = mask[i] ? (a[i] op b[i]) : no[i]

// Masked addition: out[i] = mask[i] ? (a[i] + b[i]) : no[i]
void MaskedAdd(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count);
void MaskedAdd(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count);
void MaskedAdd(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count);
void MaskedAdd(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
               const int64_t* HWY_RESTRICT no, size_t count);

// Masked subtraction: out[i] = mask[i] ? (a[i] - b[i]) : no[i]
void MaskedSub(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count);
void MaskedSub(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count);
void MaskedSub(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count);
void MaskedSub(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
               const int64_t* HWY_RESTRICT no, size_t count);

// Masked multiplication: out[i] = mask[i] ? (a[i] * b[i]) : no[i]
void MaskedMul(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count);
void MaskedMul(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count);
void MaskedMul(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count);
void MaskedMul(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
               const int64_t* HWY_RESTRICT no, size_t count);

// Masked division: out[i] = mask[i] ? (a[i] / b[i]) : no[i]
void MaskedDiv(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count);
void MaskedDiv(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count);

// Masked min: out[i] = mask[i] ? min(a[i], b[i]) : no[i]
void MaskedMin(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count);
void MaskedMin(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count);
void MaskedMin(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count);

// Masked max: out[i] = mask[i] ? max(a[i], b[i]) : no[i]
void MaskedMax(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count);
void MaskedMax(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count);
void MaskedMax(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count);

// Masked absolute value: out[i] = mask[i] ? |a[i]| : no[i]
void MaskedAbs(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT no, size_t count);
void MaskedAbs(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT no, size_t count);
void MaskedAbs(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT no, size_t count);

// Masked negation: out[i] = mask[i] ? -a[i] : no[i]
void MaskedNeg(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT no, size_t count);
void MaskedNeg(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT no, size_t count);
void MaskedNeg(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT no, size_t count);

// =============================================================================
// P0: Widening Operations
// =============================================================================

// Widen multiply-accumulate: out[i] = a[i] * b[i] + c[i] (with widening)
// int16 * int16 -> int32
void WidenMulAccumulate(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                        const int16_t* HWY_RESTRICT b, const int32_t* HWY_RESTRICT c, size_t count);
// uint16 * uint16 -> uint32
void WidenMulAccumulate(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                        const uint16_t* HWY_RESTRICT b, const uint32_t* HWY_RESTRICT c,
                        size_t count);

// Sum of 2 adjacent elements with widening: out[i] = a[2i] + a[2i+1]
void SumsOf2(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count);
void SumsOf2(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, size_t count);

// Sum of 4 adjacent elements with widening: out[i] = sum(a[4i:4i+4])
void SumsOf4(int32_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count);
void SumsOf4(uint32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count);

// Multiply even lanes and widen: out[i] = a[2i] * b[2i]
void MulEven(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
             const int32_t* HWY_RESTRICT b, size_t count);
void MulEven(uint64_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
             const uint32_t* HWY_RESTRICT b, size_t count);

// Multiply odd lanes and widen: out[i] = a[2i+1] * b[2i+1]
void MulOdd(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
            size_t count);
void MulOdd(uint64_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
            const uint32_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P0: Additional Comparison Operations
// =============================================================================

// Check if value is negative: out[i] = (a[i] < 0) ? 0xFF : 0x00
void IsNegative(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void IsNegative(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void IsNegative(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// Check if either value is NaN: out[i] = (isnan(a[i]) || isnan(b[i])) ? 0xFF : 0x00
void IsEitherNaN(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                 const float* HWY_RESTRICT b, size_t count);
void IsEitherNaN(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                 const double* HWY_RESTRICT b, size_t count);

// =============================================================================
// P1: Extended FMA Operations
// =============================================================================

// Negative multiply-subtract: out[i] = -(a[i] * b[i]) - c[i]
void NegMulSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count);
void NegMulSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count);

// =============================================================================
// P1: CompressStore Operations
// =============================================================================

// Compress and store: write elements where mask is true directly to output
// Returns the number of elements written
size_t CompressStore(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                     const uint8_t* HWY_RESTRICT mask, size_t count);
size_t CompressStore(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                     const uint8_t* HWY_RESTRICT mask, size_t count);
size_t CompressStore(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                     const uint8_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// P2: Iota and FirstN Operations
// =============================================================================

// Fill with incrementing values: out[i] = start + i
void Iota(float* HWY_RESTRICT out, float start, size_t count);
void Iota(double* HWY_RESTRICT out, double start, size_t count);
void Iota(int32_t* HWY_RESTRICT out, int32_t start, size_t count);
void Iota(int64_t* HWY_RESTRICT out, int64_t start, size_t count);
void Iota(uint32_t* HWY_RESTRICT out, uint32_t start, size_t count);
void Iota(uint64_t* HWY_RESTRICT out, uint64_t start, size_t count);

// Create mask with first N elements set to true
// out[i] = (i < n) ? 0xFF : 0x00
void FirstN(uint8_t* HWY_RESTRICT out, size_t n, size_t count);

// =============================================================================
// P0: SumsOf8 Operation
// =============================================================================

// Sum of 8 adjacent elements with widening: out[i] = sum(a[8i:8i+8])
void SumsOf8(uint64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P0: TestBit Operation
// =============================================================================

// Test if specific bit is set: out[i] = (a[i] & (1 << bit)) ? 0xFF : 0x00
void TestBit(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t bit, size_t count);
void TestBit(uint8_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t bit, size_t count);
void TestBit(uint8_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t bit, size_t count);
void TestBit(uint8_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t bit, size_t count);

// =============================================================================
// P1: Extended FMA Operations
// =============================================================================

// Fused multiply sub/add alternating (opposite of MulAddSub):
// Even lanes: out[i] = a[i] * b[i] + c[i]
// Odd lanes: out[i] = a[i] * b[i] - c[i]
void MulSubAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count);
void MulSubAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count);

// =============================================================================
// P1: Reverse2, Reverse4, Reverse8 Operations
// =============================================================================

// Reverse adjacent pairs: out[2i], out[2i+1] = a[2i+1], a[2i]
void Reverse2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Reverse2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);
void Reverse2(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// Reverse in groups of 4
void Reverse4(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Reverse4(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// Reverse in groups of 8
void Reverse8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P1: DupEven, DupOdd Operations
// =============================================================================

// Duplicate even lanes: out[2i] = out[2i+1] = a[2i]
void DupEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void DupEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// Duplicate odd lanes: out[2i] = out[2i+1] = a[2i+1]
void DupOdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void DupOdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P1: InterleaveLower, InterleaveUpper Operations
// =============================================================================

// Interleave lower halves: out = a0,b0,a1,b1,...
void InterleaveLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                     const float* HWY_RESTRICT b, size_t count);
void InterleaveLower(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                     const int32_t* HWY_RESTRICT b, size_t count);

// Interleave upper halves
void InterleaveUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                     const float* HWY_RESTRICT b, size_t count);
void InterleaveUpper(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                     const int32_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P1: Mask Logical Operations
// =============================================================================

// Mask NOT: out[i] = ~mask[i]
void MaskNot(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

// Mask AND: out[i] = a[i] & b[i]
void MaskAnd(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
             const uint8_t* HWY_RESTRICT b, size_t count);

// Mask OR: out[i] = a[i] | b[i]
void MaskOr(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
            size_t count);

// Mask XOR: out[i] = a[i] ^ b[i]
void MaskXor(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
             const uint8_t* HWY_RESTRICT b, size_t count);

// Mask AND-NOT: out[i] = ~a[i] & b[i]
void MaskAndNot(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P2: AddSub Operation
// =============================================================================

// Alternating add/subtract (no FMA)
// Even lanes: out[i] = a[i] - b[i]
// Odd lanes: out[i] = a[i] + b[i]
void AddSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            size_t count);
void AddSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            size_t count);

// =============================================================================
// P2: MinMagnitude, MaxMagnitude Operations
// =============================================================================

// Select element with smaller absolute value
void MinMagnitude(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                  size_t count);
void MinMagnitude(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                  const double* HWY_RESTRICT b, size_t count);

// Select element with larger absolute value
void MaxMagnitude(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                  size_t count);
void MaxMagnitude(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                  const double* HWY_RESTRICT b, size_t count);

// =============================================================================
// P2: MaskedLoad, MaskedStore, BlendedStore Operations
// =============================================================================

// Load with mask: out[i] = mask[i] ? src[i] : fallback
void MaskedLoad(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, float fallback, size_t count);
void MaskedLoad(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, double fallback, size_t count);
void MaskedLoad(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, int32_t fallback, size_t count);

// Store with mask: dst[i] = mask[i] ? src[i] : dst[i]
void MaskedStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                 const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedStore(double* HWY_RESTRICT dst, const double* HWY_RESTRICT src,
                 const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                 const uint8_t* HWY_RESTRICT mask, size_t count);

// Blended store: dst[i] = mask[i] ? new_val[i] : dst[i]
void BlendedStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT new_val,
                  const uint8_t* HWY_RESTRICT mask, size_t count);
void BlendedStore(double* HWY_RESTRICT dst, const double* HWY_RESTRICT new_val,
                  const uint8_t* HWY_RESTRICT mask, size_t count);
void BlendedStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT new_val,
                  const uint8_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// P0: WidenMulPairwiseAdd Operation
// =============================================================================

// Multiply pairs and add: out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
// Output has count/2 elements
void WidenMulPairwiseAdd(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                         const int16_t* HWY_RESTRICT b, size_t count);
void WidenMulPairwiseAdd(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                         const uint16_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P1: BroadcastLane Operations
// =============================================================================

// Broadcast lane at index to all lanes
void BroadcastLane(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t lane, size_t count);
void BroadcastLane(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t lane,
                   size_t count);
void BroadcastLane(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t lane,
                   size_t count);

// =============================================================================
// P1: Slide Operations
// =============================================================================

// Slide1Up: out[0] = 0, out[i] = a[i-1] for i > 0
void Slide1Up(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Slide1Up(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// Slide1Down: out[i] = a[i+1] for i < count-1, out[count-1] = 0
void Slide1Down(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Slide1Down(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P1: Concat Operations
// =============================================================================

// ConcatLowerUpper: out = {a[0..half-1], b[half..count-1]}
void ConcatLowerUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                      const float* HWY_RESTRICT b, size_t count);
void ConcatLowerUpper(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count);

// ConcatUpperLower: out = {a[half..count-1], b[0..half-1]}
void ConcatUpperLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                      const float* HWY_RESTRICT b, size_t count);
void ConcatUpperLower(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count);

// ConcatEven: out = {a[0], a[2], ..., b[0], b[2], ...}
void ConcatEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                size_t count);
void ConcatEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const int32_t* HWY_RESTRICT b, size_t count);

// ConcatOdd: out = {a[1], a[3], ..., b[1], b[3], ...}
void ConcatOdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               size_t count);
void ConcatOdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P1: Mask Utility Operations
// =============================================================================

// Find index of first true element (assumes at least one true exists)
size_t FindKnownFirstTrue(const uint8_t* HWY_RESTRICT mask, size_t count);

// Find index of last true element (assumes at least one true exists)
size_t FindKnownLastTrue(const uint8_t* HWY_RESTRICT mask, size_t count);

// Store mask as packed bits (1 bit per lane), returns number of bytes written
size_t StoreMaskBits(uint8_t* HWY_RESTRICT bits_out, const uint8_t* HWY_RESTRICT mask,
                     size_t count);

// Load packed bits as mask (1 bit per lane)
void LoadMaskBits(uint8_t* HWY_RESTRICT mask_out, const uint8_t* HWY_RESTRICT bits, size_t count);

// =============================================================================
// P1: CompressBlendedStore and CompressNot Operations
// =============================================================================

// CompressBlendedStore: Like CompressStore but blends into existing dst
size_t CompressBlendedStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                            const uint8_t* HWY_RESTRICT mask, size_t count);
size_t CompressBlendedStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                            const uint8_t* HWY_RESTRICT mask, size_t count);

// CompressNot: Compress where mask is false (inverse of Compress)
size_t CompressNot(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                   const uint8_t* HWY_RESTRICT mask, size_t count);
size_t CompressNot(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                   const uint8_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// P1: InterleaveEven and InterleaveOdd Operations
// =============================================================================

// InterleaveEven: Take even elements and interleave
// out = {a[0], b[0], a[2], b[2], ...}
void InterleaveEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                    const float* HWY_RESTRICT b, size_t count);
void InterleaveEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                    const int32_t* HWY_RESTRICT b, size_t count);

// InterleaveOdd: Take odd elements and interleave
// out = {a[1], b[1], a[3], b[3], ...}
void InterleaveOdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                   const float* HWY_RESTRICT b, size_t count);
void InterleaveOdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                   const int32_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P1: Shuffle Operations
// =============================================================================

// Shuffle0123: Identity shuffle within 4-element blocks
// [0,1,2,3] -> [0,1,2,3]
void Shuffle0123(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void Shuffle0123(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count);

// Shuffle2301: Swap pairs within 4-element blocks
// [0,1,2,3] -> [2,3,0,1]
void Shuffle2301(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void Shuffle2301(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count);

// Shuffle1032: Swap adjacent within 4-element blocks
// [0,1,2,3] -> [1,0,3,2]
void Shuffle1032(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void Shuffle1032(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count);

// Shuffle01: Identity shuffle within 2-element blocks (64-bit lanes)
void Shuffle01(double* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count);
void Shuffle01(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT in, size_t count);

// Shuffle10: Swap within 2-element blocks (64-bit lanes)
// [0,1] -> [1,0]
void Shuffle10(double* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count);
void Shuffle10(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT in, size_t count);

// =============================================================================
// P1: TableLookup Operations
// =============================================================================

// TableLookupBytes: Byte-level table lookup
void TableLookupBytes(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT table,
                      const uint8_t* HWY_RESTRICT indices, size_t count, size_t table_size);

// TableLookupLanes: Lane-level table lookup
void TableLookupLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT table,
                      const int32_t* HWY_RESTRICT indices, size_t count, size_t table_size);
void TableLookupLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT table,
                      const int32_t* HWY_RESTRICT indices, size_t count, size_t table_size);

// =============================================================================
// P1: Mask Set Operations
// =============================================================================

// SetBeforeFirst: Set mask true for all lanes before the first true lane
void SetBeforeFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

// SetAtOrBeforeFirst: Set mask true for all lanes at or before the first true lane
void SetAtOrBeforeFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

// SetOnlyFirst: Set mask true only for the first true lane
void SetOnlyFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

// SetAtOrAfterFirst: Set mask true for all lanes at or after the first true lane
void SetAtOrAfterFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// P1: Masked Reduction Operations
// =============================================================================

// MaskedReduceSum: Sum of elements where mask is true
float MaskedReduceSum(const float* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                      size_t count);
double MaskedReduceSum(const double* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                       size_t count);
int32_t MaskedReduceSum(const int32_t* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                        size_t count);

// MaskedReduceMin: Minimum of elements where mask is true
float MaskedReduceMin(const float* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                      size_t count);
double MaskedReduceMin(const double* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                       size_t count);
int32_t MaskedReduceMin(const int32_t* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                        size_t count);

// MaskedReduceMax: Maximum of elements where mask is true
float MaskedReduceMax(const float* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                      size_t count);
double MaskedReduceMax(const double* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                       size_t count);
int32_t MaskedReduceMax(const int32_t* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                        size_t count);

// =============================================================================
// P1: Remaining Operations
// =============================================================================

// TwoTablesLookupLanes: Lookup from two tables based on indices
void TwoTablesLookupLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT table0,
                          const float* HWY_RESTRICT table1, const int32_t* HWY_RESTRICT indices,
                          size_t count, size_t table_size);
void TwoTablesLookupLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT table0,
                          const int32_t* HWY_RESTRICT table1, const int32_t* HWY_RESTRICT indices,
                          size_t count, size_t table_size);

// CompressBits: Compress using packed bit mask
size_t CompressBits(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                    const uint8_t* HWY_RESTRICT bits, size_t count);
size_t CompressBits(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                    const uint8_t* HWY_RESTRICT bits, size_t count);

// CompressBitsStore: Compress using packed bit mask and store
size_t CompressBitsStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                         const uint8_t* HWY_RESTRICT bits, size_t count);
size_t CompressBitsStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                         const uint8_t* HWY_RESTRICT bits, size_t count);

// LoadExpand: Expand packed bits to mask and load
void LoadExpand(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, size_t count);
void LoadExpand(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, size_t count);

// PairwiseSub: Pairwise subtraction
void PairwiseSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                 size_t count);
void PairwiseSub(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                 const int32_t* HWY_RESTRICT b, size_t count);

// SumsOfAdjQuadAbsDiff: Sum of absolute differences of adjacent quads
void SumsOfAdjQuadAbsDiff(uint16_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                          const uint8_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P2: Math Functions
// =============================================================================

// Cube root: out[i] = cbrt(a[i])
void Cbrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Cbrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Error function: out[i] = erf(a[i])
void Erf(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Erf(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// Complementary error function: out[i] = erfc(a[i])
void Erfc(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void Erfc(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// =============================================================================
// P2: Generation Operations
// =============================================================================

// IndicesFromVec: Create indices from mask (indices where mask is true)
void IndicesFromVec(int32_t* HWY_RESTRICT indices_out, const uint8_t* HWY_RESTRICT mask,
                    size_t count);

// IndicesFromNotVec: Create indices from inverted mask
void IndicesFromNotVec(int32_t* HWY_RESTRICT indices_out, const uint8_t* HWY_RESTRICT mask,
                       size_t count);

// =============================================================================
// P2: Type Conversions
// =============================================================================

// PromoteLowerTo: Promote lower half to wider type
void PromoteLowerTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void PromoteLowerTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count);
void PromoteLowerTo(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count);

// PromoteUpperTo: Promote upper half to wider type
void PromoteUpperTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count,
                    size_t half);
void PromoteUpperTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count,
                    size_t half);

// PromoteEvenTo: Promote even elements to wider type
void PromoteEvenTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void PromoteEvenTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count);

// PromoteOddTo: Promote odd elements to wider type
void PromoteOddTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void PromoteOddTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count);

// =============================================================================
// P2: Additional Arithmetic Operations
// =============================================================================

// Modulo: out[i] = a[i] % b[i]
void Mod(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count);
void Mod(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count);

// Saturated negation: out[i] = -a[i] (clamped to min/max)
void SaturatedNeg(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count);
void SaturatedNeg(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count);

// Saturated absolute: out[i] = |a[i]| (clamped)
void SaturatedAbs(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count);
void SaturatedAbs(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P2: Bitwise Operations
// =============================================================================

// Three-way XOR: out[i] = a[i] ^ b[i] ^ c[i]
void Xor3(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
          const int32_t* HWY_RESTRICT c, size_t count);
void Xor3(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
          const int64_t* HWY_RESTRICT c, size_t count);

// Three-way OR: out[i] = a[i] | b[i] | c[i]
void Or3(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         const int32_t* HWY_RESTRICT c, size_t count);
void Or3(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         const int64_t* HWY_RESTRICT c, size_t count);

// OrAnd: out[i] = a[i] | (b[i] & c[i])
void OrAnd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
           const int32_t* HWY_RESTRICT c, size_t count);
void OrAnd(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
           const int64_t* HWY_RESTRICT c, size_t count);

// ReverseBits: Reverse all bits in each element
void ReverseBits(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void ReverseBits(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// HighestSetBitIndex: Index of highest set bit (-1 if zero)
void HighestSetBitIndex(int32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count);
void HighestSetBitIndex(int32_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count);

// =============================================================================
// P2: Memory Operations
// =============================================================================

// LoadDup128: Load 128 bits and duplicate to fill vector
void LoadDup128(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void LoadDup128(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);

// GatherOffset: Gather from base + offsets
void GatherOffset(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                  const int32_t* HWY_RESTRICT offsets, size_t count);
void GatherOffset(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                  const int32_t* HWY_RESTRICT offsets, size_t count);

// ScatterOffset: Scatter to base + offsets
void ScatterOffset(float* HWY_RESTRICT base, const float* HWY_RESTRICT src,
                   const int32_t* HWY_RESTRICT offsets, size_t count);
void ScatterOffset(int32_t* HWY_RESTRICT base, const int32_t* HWY_RESTRICT src,
                   const int32_t* HWY_RESTRICT offsets, size_t count);

// MaskedGatherIndex: Gather with mask
void MaskedGatherIndex(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                       const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                       float fallback, size_t count);
void MaskedGatherIndex(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                       const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                       int32_t fallback, size_t count);

// MaskedScatterIndex: Scatter with mask
void MaskedScatterIndex(float* HWY_RESTRICT base, const float* HWY_RESTRICT src,
                        const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                        size_t count);
void MaskedScatterIndex(int32_t* HWY_RESTRICT base, const int32_t* HWY_RESTRICT src,
                        const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                        size_t count);

// SafeFillN: Fill with value up to count elements
void SafeFillN(float* HWY_RESTRICT dst, float value, size_t count);
void SafeFillN(int32_t* HWY_RESTRICT dst, int32_t value, size_t count);

// SafeCopyN: Copy up to count elements
void SafeCopyN(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, size_t count);
void SafeCopyN(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count);

// =============================================================================
// P2: Special Operations
// =============================================================================

// MulByPow2: Multiply by 2^exp
void MulByPow2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT exp, size_t count);
void MulByPow2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT exp, size_t count);

// GetExponent: Extract exponent
void GetExponent(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void GetExponent(int32_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// SignBit: Extract sign bit
void SignBit(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count);
void SignBit(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count);

// NaN: Fill with NaN
void NaN(float* HWY_RESTRICT out, size_t count);
void NaN(double* HWY_RESTRICT out, size_t count);

// Inf: Fill with positive infinity
void Inf(float* HWY_RESTRICT out, size_t count);
void Inf(double* HWY_RESTRICT out, size_t count);

// =============================================================================
// P3: Complex Number Operations
// =============================================================================

// ComplexConj: Complex conjugate (negate imaginary parts)
void ComplexConj(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);
void ComplexConj(double* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count);

// MulComplex: Complex multiplication
void MulComplex(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                size_t count);
void MulComplex(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                const double* HWY_RESTRICT b, size_t count);

// MulComplexAdd: Complex multiply-add
void MulComplexAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                   const float* HWY_RESTRICT b, const float* HWY_RESTRICT c, size_t count);

// =============================================================================
// P3: Saturation Operations
// =============================================================================

// SaturatedAdd: Saturating addition
void SaturatedAdd(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, size_t count);
void SaturatedAdd(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                  const uint16_t* HWY_RESTRICT b, size_t count);

// SaturatedSub: Saturating subtraction
void SaturatedSub(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, size_t count);
void SaturatedSub(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                  const uint16_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P3: Block Operations
// =============================================================================

// SlideUpBlocks: Slide up by blocks (128-bit granularity)
void SlideUpBlocks(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t blocks,
                   size_t count);

// SlideDownBlocks: Slide down by blocks (128-bit granularity)
void SlideDownBlocks(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t blocks,
                     size_t count);

// CombineShiftRightLanes: Combine hi/lo and shift right by lanes
void CombineShiftRightLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT hi,
                            const float* HWY_RESTRICT lo, size_t shift, size_t count);

// =============================================================================
// P3: Additional Masked Operations
// =============================================================================

// MaskedSqrt: Sqrt only where mask is true, fallback otherwise
void MaskedSqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, float fallback, size_t count);
void MaskedSqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, double fallback, size_t count);

// ZeroIfNegative: Set to zero if negative
void ZeroIfNegative(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void ZeroIfNegative(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);

// =============================================================================
// P2: Additional Type Conversion Operations
// =============================================================================

void ReorderDemote2To(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT hi,
                      const int32_t* HWY_RESTRICT lo, size_t count);
void ReorderDemote2To(uint16_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT hi,
                      const uint32_t* HWY_RESTRICT lo, size_t count);

void OrderedTruncate2To(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT hi,
                        const int32_t* HWY_RESTRICT lo, size_t count);
void OrderedTruncate2To(uint16_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT hi,
                        const uint32_t* HWY_RESTRICT lo, size_t count);

void ConvertInRangeTo(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void ConvertInRangeTo(int64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);

void ResizeBitCast(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void ResizeBitCast(float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src, size_t count);
void ResizeBitCast(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);
void ResizeBitCast(double* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src, size_t count);

// =============================================================================
// P2: Additional Special Operations
// =============================================================================

void MulByFloorPow2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                    const float* HWY_RESTRICT pow2, size_t count);
void MulByFloorPow2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                    const double* HWY_RESTRICT pow2, size_t count);

void GetBiasedExponent(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count);
void GetBiasedExponent(int32_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count);

void MulFixedPoint15(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                     const int16_t* HWY_RESTRICT b, size_t count);

void MulRound(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
              const int16_t* HWY_RESTRICT b, size_t count);
void MulRound(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, size_t count);

void RoundingShiftRight(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT src, int shift,
                        size_t count);
void RoundingShiftRight(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, int shift,
                        size_t count);

// =============================================================================
// P3: Additional Complex Number Operations
// =============================================================================

void MulComplexConj(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                    const float* HWY_RESTRICT b, size_t count);
void MulComplexConj(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                    const double* HWY_RESTRICT b, size_t count);

void MulComplexConjAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c, size_t count);
void MulComplexConjAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                       const double* HWY_RESTRICT b, const double* HWY_RESTRICT c, size_t count);

// =============================================================================
// P3: Per-Lane Block Shuffle
// =============================================================================

void Per4LaneBlockShuffle(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, uint8_t pattern,
                          size_t count);
void Per4LaneBlockShuffle(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                          uint8_t pattern, size_t count);

// =============================================================================
// P3: Additional Masked Operations
// =============================================================================

void MaskedReciprocal(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, float fallback, size_t count);
void MaskedReciprocal(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, double fallback, size_t count);

void MaskedShiftLeft(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                     const uint8_t* HWY_RESTRICT mask, int shift, int32_t fallback, size_t count);
void MaskedShiftLeft(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                     const uint8_t* HWY_RESTRICT mask, int shift, int64_t fallback, size_t count);

void MaskedShiftRight(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, int shift, int32_t fallback, size_t count);
void MaskedShiftRight(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, int shift, int64_t fallback, size_t count);

void MaskedSatAdd(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int8_t fallback,
                  size_t count);
void MaskedSatAdd(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int16_t fallback,
                  size_t count);

void MaskedSatSub(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int8_t fallback,
                  size_t count);
void MaskedSatSub(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int16_t fallback,
                  size_t count);

// =============================================================================
// P3: Masked Comparison Operations
// =============================================================================

void MaskedEq(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedEq(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count);

void MaskedNe(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedNe(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count);

void MaskedLt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedLt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count);

void MaskedLe(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedLe(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count);

void MaskedGt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedGt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count);

void MaskedGe(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count);
void MaskedGe(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count);

// =============================================================================
// P3.2: Cryptographic Operations
// =============================================================================

/// AES encryption round (SubBytes, ShiftRows, MixColumns, AddRoundKey)
void AESRound(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
              const uint8_t* HWY_RESTRICT round_key, size_t count);

/// AES final encryption round (SubBytes, ShiftRows, AddRoundKey - no MixColumns)
void AESLastRound(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                  const uint8_t* HWY_RESTRICT round_key, size_t count);

/// AES decryption round (InvShiftRows, InvSubBytes, AddRoundKey, InvMixColumns)
void AESRoundInv(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                 const uint8_t* HWY_RESTRICT round_key, size_t count);

/// AES final decryption round (InvShiftRows, InvSubBytes, AddRoundKey - no InvMixColumns)
void AESLastRoundInv(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                     const uint8_t* HWY_RESTRICT round_key, size_t count);

/// Apply InvMixColumns transformation for equivalent inverse cipher
void AESInvMixColumns(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state, size_t count);

/// AES key schedule assist
void AESKeyGenAssist(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT key, uint8_t rcon,
                     size_t count);

/// Carry-less multiplication of lower 64-bit halves (for GCM, CRC)
void CLMulLower(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                const uint64_t* HWY_RESTRICT b, size_t count);

/// Carry-less multiplication of upper 64-bit halves
void CLMulUpper(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                const uint64_t* HWY_RESTRICT b, size_t count);

// =============================================================================
// P3.3: Random Number Generation
// =============================================================================

/// Initialize random state from seed (Xoshiro256** algorithm)
void RandomStateInit(uint64_t* state, uint64_t seed);

/// Generate random 32-bit integers
void Random32(uint32_t* HWY_RESTRICT out, uint64_t* state, size_t count);

/// Generate random 64-bit integers
void Random64(uint64_t* HWY_RESTRICT out, uint64_t* state, size_t count);

/// Generate random floats in [0, 1)
void RandomFloat(float* HWY_RESTRICT out, uint64_t* state, size_t count);

// =============================================================================
// P3.4: Bit Packing
// =============================================================================

/// Pack MSBs of bytes into bits (8 bytes -> 1 byte)
void PackBits(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count);

/// Unpack bits to bytes (1 bit -> 0xFF or 0x00)
void UnpackBits(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count);

// =============================================================================
// P3.6: Algorithm Operations
// =============================================================================

/// Find first element > threshold, returns count if not found
size_t FindIfGreaterThan(const float* HWY_RESTRICT arr, float threshold, size_t count);
size_t FindIfGreaterThan(const int32_t* HWY_RESTRICT arr, int32_t threshold, size_t count);

/// Generate arithmetic sequence: out[i] = start + i * step
void Generate(float* HWY_RESTRICT out, float start, float step, size_t count);
void Generate(int32_t* HWY_RESTRICT out, int32_t start, int32_t step, size_t count);

/// Replace all occurrences of old_val with new_val
void Replace(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, float old_val, float new_val,
             size_t count);
void Replace(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, int32_t old_val,
             int32_t new_val, size_t count);

/// Replace values > threshold with new_val
void ReplaceIfGreaterThan(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, float threshold,
                          float new_val, size_t count);
void ReplaceIfGreaterThan(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                          int32_t threshold, int32_t new_val, size_t count);

// =============================================================================
// P3.5: Image Processing Operations
// =============================================================================

/// Single-channel floating-point image container
struct ImageF {
    float* data;    ///< Pointer to pixel data (row-major order)
    size_t width;   ///< Image width in pixels
    size_t height;  ///< Image height in pixels
    size_t stride;  ///< Row stride (may be larger than width for alignment)
};

/// Create an image with given dimensions (data is uninitialized)
ImageF ImageCreate(size_t width, size_t height);

/// Free image memory and reset all fields to zero/null
void ImageFree(ImageF* img);

/// Fill entire image with constant value
void ImageFill(ImageF* img, float value);

/// Copy image data (src and dst must have same dimensions)
void ImageCopy(const ImageF* src, ImageF* dst);

/// Element-wise image addition: out = a + b
void ImageAdd(const ImageF* a, const ImageF* b, ImageF* out);

/// Element-wise image subtraction: out = a - b
void ImageSub(const ImageF* a, const ImageF* b, ImageF* out);

/// Element-wise image multiplication: out = a * b
void ImageMul(const ImageF* a, const ImageF* b, ImageF* out);

/// Scale image by constant: out = src * scale
void ImageScale(const ImageF* src, float scale, ImageF* out);

/// Clamp image values to range [min_val, max_val]
void ImageClamp(const ImageF* src, float min_val, float max_val, ImageF* out);

/// Apply 3x3 convolution with custom kernel (9 elements, row-major)
void Convolve3x3(const ImageF* src, const float* kernel, ImageF* out);

/// 3x3 box blur (simple averaging filter)
void BoxBlur3x3(const ImageF* src, ImageF* out);

/// 3x3 Gaussian blur
void GaussianBlur3x3(const ImageF* src, ImageF* out);

/// Sobel edge detection (returns gradient magnitude)
void SobelEdge(const ImageF* src, ImageF* out);

/// Image sharpening using unsharp mask
void Sharpen(const ImageF* src, ImageF* out);

/// Binary thresholding: out = (src > threshold) ? 1.0 : 0.0
void Threshold(const ImageF* src, float threshold, ImageF* out);

/// RGB to grayscale using luminance formula (0.299*R + 0.587*G + 0.114*B)
void Grayscale(const ImageF* r, const ImageF* g, const ImageF* b, ImageF* out);

/// Downsample by factor of 2 (2x2 averaging)
/// out dimensions should be src dimensions / 2
void Downsample2x(const ImageF* src, ImageF* out);

/// Upsample by factor of 2 (bilinear interpolation)
/// out dimensions should be src dimensions * 2
void Upsample2x(const ImageF* src, ImageF* out);

// =============================================================================
// Gap Operations: Additional SIMD Operations
// =============================================================================

/// Masked FMA: out[i] = mask[i] ? (mul[i] * x[i] + add[i]) : no[i]
void MaskedMulAdd(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                  const float* HWY_RESTRICT mul, const float* HWY_RESTRICT x,
                  const float* HWY_RESTRICT add, const float* HWY_RESTRICT no, size_t count);

/// Masked negative FMA: out[i] = mask[i] ? (add[i] - mul[i] * x[i]) : no[i]
void MaskedNegMulAdd(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                     const float* HWY_RESTRICT mul, const float* HWY_RESTRICT x,
                     const float* HWY_RESTRICT add, const float* HWY_RESTRICT no, size_t count);

/// Interleave lower halves of a and b: a0,b0,a1,b1,...
void InterleaveWholeLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t count);

/// Interleave upper halves of a and b: a[half],b[half],a[half+1],b[half+1],...
void InterleaveWholeUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t count);

/// Conditional: out[i] = (v[i] < 0) ? yes[i] : 0
void IfNegativeThenElseZero(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                            const float* HWY_RESTRICT yes, size_t count);

/// Conditional: out[i] = (v[i] < 0) ? 0 : no[i]
void IfNegativeThenZeroElse(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                            const float* HWY_RESTRICT no, size_t count);

/// Bitwise conditional: out[i] = (mask[i] & yes[i]) | (~mask[i] & no[i])
void BitwiseIfThenElse(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT mask,
                       const int32_t* HWY_RESTRICT yes, const int32_t* HWY_RESTRICT no,
                       size_t count);

/// Set all mask bytes to 0 (false)
void MaskFalse(uint8_t* HWY_RESTRICT mask, size_t count);

/// Set all mask bytes to 0xFF (true) or 0 (false)
void SetMask(uint8_t* HWY_RESTRICT mask, bool value, size_t count);

/// Ceiling to int: out[i] = ceil(in[i]) as int32
void CeilInt(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);

/// Floor to int: out[i] = floor(in[i]) as int32
void FloorInt(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count);

/// Store int32 as int16 (simple truncation)
void TruncateStore(int16_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count);

/// Masked modulo: out[i] = mask[i] ? (a[i] % b[i]) : no[i]
void MaskedModOr(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                 const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                 const int32_t* HWY_RESTRICT no, size_t count);

// =============================================================================
// Final Gap Operations
// =============================================================================

/// Per 2-lane block shuffle
void Per2LaneBlockShuffle(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t count);

/// Masked IsNaN check: out[i] = mask[i] ? isnan(in[i]) : 0
void MaskedIsNaN(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                 const float* HWY_RESTRICT in, size_t count);

/// If negative then negate: out[i] = (v[i] < 0) ? -x[i] : x[i]
void IfNegativeThenNegOrUndefIfZero(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                                    const float* HWY_RESTRICT x, size_t count);

/// Masked set with fallback: out[i] = mask[i] ? value : no[i]
void MaskedSetOr(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, float value,
                 const float* HWY_RESTRICT no, size_t count);

/// Masked Q15 fixed-point multiply: out[i] = mask[i] ? (a[i]*b[i])>>15 : no[i]
void MaskedMulFixedPoint15(int16_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                           const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
                           const int16_t* HWY_RESTRICT no, size_t count);

/// Masked widening multiply-add: out[i] = mask[i] ? (a[2i]*b[2i] + a[2i+1]*b[2i+1]) : no[i]
void MaskedWidenMulPairwiseAdd(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
                               const int32_t* HWY_RESTRICT no, size_t count);

/// Masked absolute value: out[i] = mask[i] ? abs(a[i]) : no[i]
void MaskedAbsOr(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                 const float* HWY_RESTRICT a, const float* HWY_RESTRICT no, size_t count);

/// Insert scalar into upper half
void InsertIntoUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT vec, float scalar,
                     size_t count);

/// Masked gather with fallback: out[i] = mask[i] ? base[indices[i]] : no[i]
void MaskedGatherIndexOr(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                         const float* HWY_RESTRICT base, const int32_t* HWY_RESTRICT indices,
                         const float* HWY_RESTRICT no, size_t count);

/// Sum of absolute differences for groups of 8
void SumsOf8AbsDiff(uint64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                    const uint8_t* HWY_RESTRICT b, size_t count);

/// Combine two half-masks into one full mask
void CombineMasks(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT lo,
                  const uint8_t* HWY_RESTRICT hi, size_t half_count);

/// Extract lower half of mask
void LowerHalfOfMask(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

/// Extract upper half of mask
void UpperHalfOfMask(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count);

/// Promote uint8 mask to uint16 mask
void PromoteMaskTo(uint16_t* HWY_RESTRICT wide, const uint8_t* HWY_RESTRICT narrow, size_t count);

/// Demote uint16 mask to uint8 mask
void DemoteMaskTo(uint8_t* HWY_RESTRICT narrow, const uint16_t* HWY_RESTRICT wide, size_t count);

/// Zero-extend uint8 to uint32
void ZeroExtendResizeBitCast(uint32_t* HWY_RESTRICT dst, const uint8_t* HWY_RESTRICT src,
                             size_t count);

/// Convert double to float16 (stored as uint16)
void F64ToF16(uint16_t* HWY_RESTRICT dst, const double* HWY_RESTRICT src, size_t count);

/// Convert double to bfloat16 (stored as uint16)
void F64ToBF16(uint16_t* HWY_RESTRICT dst, const double* HWY_RESTRICT src, size_t count);

/// Matrix-vector multiplication: out = mat * vec
void MatVecMul(float* HWY_RESTRICT out, const float* HWY_RESTRICT mat,
               const float* HWY_RESTRICT vec, size_t rows, size_t cols);

/// Matrix-matrix multiplication: out = a * b (row-major)
void MatMul(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            size_t M, size_t K, size_t N);

}  // namespace simd
}  // namespace bud
