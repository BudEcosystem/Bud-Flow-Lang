# JIT Optimization Research: Closing the Gap with NumPy

## Executive Summary

This document presents extensive research findings on JIT optimization techniques to improve Bud Flow Lang's performance for operations where NumPy currently leads (ADD, DOT, etc.). The research identifies specific optimizations that can provide 2-10x speedups.

---

## Current Performance Gap Analysis

| Operation | Bud Flow Lang | NumPy | Gap | Root Cause |
|-----------|---------------|-------|-----|------------|
| **ADD** | 1.29x faster | baseline | NumPy leads on small arrays | Binding overhead, memory-bound |
| **DOT** | 0.8x (slower) | baseline | NumPy leads significantly | BLAS optimization, micro-kernels |

---

## Part 1: DOT Product Optimization

### Why NumPy/BLAS Wins

1. **Micro-Kernel Architecture**: BLAS implementations use hand-tuned micro-kernels that compute tiles of 6x16 elements using 12/16 AVX2 registers
2. **Register Blocking**: Maximizes data reuse within CPU registers before writing to cache
3. **Cache Blocking**: Tiles data to fit L1/L2/L3 cache at each level
4. **Multi-Accumulator Pattern**: Uses 4-8 independent accumulators to hide FMA latency

### Recommended Optimizations

#### 1.1 Implement BLAS-Style Micro-Kernels
```cpp
// Current: Simple accumulator (latency-bound)
float sum = 0;
for (int i = 0; i < n; i++)
    sum += a[i] * b[i];

// Optimized: 8 parallel accumulators with FMA
__m256 acc0 = _mm256_setzero_ps();
__m256 acc1 = _mm256_setzero_ps();
__m256 acc2 = _mm256_setzero_ps();
__m256 acc3 = _mm256_setzero_ps();

for (int i = 0; i < n; i += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
}
// Single horizontal reduction at end
float result = horizontal_sum(acc0 + acc1 + acc2 + acc3);
```

**Expected Speedup**: 2-4x

#### 1.2 AVX-512 Native Reduction
```cpp
// AVX-512 has native reduce_add - use it!
__m512 sum = _mm512_setzero_ps();
for (int i = 0; i < n; i += 16) {
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), sum);
}
return _mm512_reduce_add_ps(sum);  // Native horizontal sum
```

**Expected Speedup**: 1.5-2x over AVX2

#### 1.3 Eliminate Tail Loops with Masked Operations
```cpp
// SimSIMD approach: use AVX-512 masked loads
int remaining = n % 16;
__mmask16 mask = (1u << remaining) - 1;
__m512 a_tail = _mm512_maskz_loadu_ps(mask, a + n - remaining);
__m512 b_tail = _mm512_maskz_loadu_ps(mask, b + n - remaining);
```

**Expected Speedup**: 10-30% on small/odd-sized arrays

### Reference Implementation: SimSIMD

[SimSIMD](https://github.com/ashvardanian/simsimd) achieves **up to 200x faster** dot products than NumPy by:
- Eliminating Python binding overhead (90% of NumPy's time on small arrays!)
- Mixed-precision accumulation (int8 input → int32 accumulator)
- Masked loads for tail handling
- Direct SIMD intrinsics without abstraction layers

---

## Part 2: ADD (Elementwise) Optimization

### Why NumPy Wins on Small Arrays

1. **Memory-Bound Nature**: Elementwise ADD is purely memory-bound (1 op per element)
2. **JIT Overhead**: Compilation cost not amortized on small arrays
3. **NumPy's C Loop**: Zero per-call overhead, tight inner loop

### Recommended Optimizations

#### 2.1 Non-Temporal Stores for Large Arrays
```cpp
// For arrays > L3 cache: bypass cache hierarchy
if (n * sizeof(float) > l3_size) {
    for (int i = 0; i < n; i += 16) {
        __m512 result = _mm512_add_ps(
            _mm512_loadu_ps(a + i),
            _mm512_loadu_ps(b + i)
        );
        _mm512_stream_ps(out + i, result);  // Non-temporal store
    }
    _mm_sfence();  // Ensure completion
}
```

**Expected Speedup**: 1.5-2x for large arrays (>1MB)

**Caveat**: Non-temporal stores are slower for concurrency-limited workloads. Profile first!

#### 2.2 Software Prefetching with Stride Detection
```cpp
// Adaptive prefetch distance based on array size
size_t prefetch_distance = (n > 1000000) ? 64 : 16;

for (int i = 0; i < n; i += 8) {
    _mm_prefetch(a + i + prefetch_distance, _MM_HINT_T0);
    _mm_prefetch(b + i + prefetch_distance, _MM_HINT_T0);
    // Process current elements...
}
```

#### 2.3 Loop Fusion for Compound Operations
```cpp
// Before: 3 memory passes
temp1 = a + b;      // Read a,b; Write temp1
temp2 = temp1 * c;  // Read temp1,c; Write temp2
result = temp2 - d; // Read temp2,d; Write result

// After: 1 memory pass (fused)
for (int i = 0; i < n; i += 8) {
    result[i:i+8] = (a[i:i+8] + b[i:i+8]) * c[i:i+8] - d[i:i+8];
}
```

**Expected Speedup**: 2-3x for chained operations

---

## Part 3: Binding Layer Optimization

### The 90% Overhead Problem

From [NumPy vs BLAS research](https://ashvardanian.com/posts/numpy-vs-blas-costs/):
> "NumPy's binding overhead wastes up to 90% of BLAS throughput on dot products"

### Recommended Optimizations

#### 3.1 Direct Buffer Access (Zero-Copy)
```cpp
// Avoid: Creating intermediate Python objects
// Use: Direct pointer access with Python buffer protocol

// In nanobind:
nb::ndarray<float, nb::c_contig> get_buffer() {
    return nb::ndarray<float>(data_, {size_});
}
```

#### 3.2 Batch Small Operations
```cpp
// Instead of N calls to add(a, b) for small arrays,
// batch into single call: add_batch(arrays_a, arrays_b, count)
void add_batch(float** a, float** b, float** out,
               size_t* sizes, size_t count) {
    for (size_t i = 0; i < count; i++) {
        add_kernel(a[i], b[i], out[i], sizes[i]);
    }
}
```

#### 3.3 JIT Threshold Tuning
```cpp
// Current: Fixed thresholds
// Recommended: Size-adaptive thresholds

size_t get_jit_threshold(size_t array_size) {
    if (array_size < 256) return 50;      // High threshold, amortize cost
    if (array_size < 4096) return 10;     // Medium
    return 3;                              // Large arrays: JIT immediately
}
```

---

## Part 4: Auto-Tuning & Continuous Optimization

### 4.1 Online Autotuning

Based on research from [ACM TACO](https://dl.acm.org/doi/10.1145/3570641):

```cpp
class OnlineTuner {
    struct TuningState {
        size_t tile_size;
        size_t unroll_factor;
        float measured_throughput;
    };

    // Sample different configurations during warmup
    void tune(const Kernel& k, size_t array_size) {
        if (samples_ < kWarmupSamples) {
            // Try different tile sizes
            current_config_ = sample_config();
            measure_and_record();
        } else {
            // Use best configuration
            current_config_ = best_config_;
        }
    }
};
```

### 4.2 Cost Model-Driven Optimization

Implement a cost model based on [TVM Ansor](https://tvm.apache.org/2021/03/03/intro-auto-scheduler):

```cpp
float predict_cost(const Schedule& s) {
    float memory_cost = 0;
    float compute_cost = 0;

    // Memory: bytes / bandwidth
    memory_cost = s.bytes_read / memory_bandwidth_;
    memory_cost += s.bytes_written / memory_bandwidth_;

    // Compute: ops / (simd_width * frequency)
    compute_cost = s.flops / (simd_width_ * peak_flops_);

    // Roofline: max of memory-bound and compute-bound
    return std::max(memory_cost, compute_cost);
}
```

### 4.3 Profile-Guided Deoptimization

From [HotSpot JVM research](https://wiki.openjdk.org/display/HotSpot/PerformanceTechniques):

```cpp
class SpeculativeOptimizer {
    void specialize(Kernel& k, const ProfileData& profile) {
        // Specialize for common array size
        if (profile.size_histogram.mode() == 1024) {
            k.specialize_for_size(1024);
        }

        // Specialize for common dtype
        if (profile.dtype_counts[Float32] > 0.9 * profile.total) {
            k.specialize_for_dtype<float>();
        }
    }

    void deoptimize_if_needed(Kernel& k, const Input& input) {
        if (!k.assumptions_valid(input)) {
            k.fallback_to_generic();
        }
    }
};
```

---

## Part 5: Cache Optimization Strategies

### 5.1 Cache-Aware Tiling

From [CS:APP cache blocking research](https://csapp.cs.cmu.edu/2e/waside/waside-blocking.pdf):

```cpp
// Tile sizes for typical Intel CPU:
// L1: 32KB → block size ~32 for doubles
// L2: 256KB → block size ~128
// L3: 8MB → block size ~1024

constexpr size_t L1_BLOCK = 32;
constexpr size_t L2_BLOCK = 128;

void gemm_blocked(float* A, float* B, float* C, size_t n) {
    for (size_t ii = 0; ii < n; ii += L2_BLOCK) {
        for (size_t jj = 0; jj < n; jj += L2_BLOCK) {
            for (size_t kk = 0; kk < n; kk += L1_BLOCK) {
                // Micro-kernel operates on L1-sized blocks
                gemm_microkernel(A, B, C, ii, jj, kk);
            }
        }
    }
}
```

### 5.2 Data Layout Optimization

```cpp
// For matrix operations, use block-major format
// Eliminates TLB misses, maximizes cache line usage

struct BlockMajorMatrix {
    static constexpr size_t BLOCK_SIZE = 64;
    float* data_;
    size_t rows_, cols_;

    float& at(size_t i, size_t j) {
        size_t block_i = i / BLOCK_SIZE;
        size_t block_j = j / BLOCK_SIZE;
        size_t local_i = i % BLOCK_SIZE;
        size_t local_j = j % BLOCK_SIZE;

        size_t block_offset = (block_i * (cols_/BLOCK_SIZE) + block_j)
                            * BLOCK_SIZE * BLOCK_SIZE;
        return data_[block_offset + local_i * BLOCK_SIZE + local_j];
    }
};
```

---

## Part 6: Runtime CPU Dispatch

### 6.1 Feature Detection & Kernel Selection

Based on [ClickHouse's approach](https://clickhouse.com/blog/cpu-dispatch-in-clickhouse):

```cpp
enum class CPUFeature {
    SSE42 = 1 << 0,
    AVX2 = 1 << 1,
    AVX512F = 1 << 2,
    AVX512BW = 1 << 3,
};

CPUFeature detect_features() {
    uint32_t eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);

    CPUFeature features = CPUFeature::SSE42;
    if (ebx & (1 << 5)) features |= CPUFeature::AVX2;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1 << 16)) features |= CPUFeature::AVX512F;

    return features;
}

// Dispatch table
using DotFunc = float(*)(const float*, const float*, size_t);
DotFunc dot_dispatch[4] = {
    dot_sse42,
    dot_avx2,
    dot_avx512,
    dot_avx512_vnni
};
```

### 6.2 Highway's Approach (Already Used)

Highway already provides dynamic dispatch. Ensure we're using:
```cpp
HWY_DYNAMIC_DISPATCH(DotProduct)(a, b, n);  // Auto-selects best ISA
```

---

## Part 7: Expression Templates & Lazy Evaluation

### 7.1 Deferred Execution for Fusion

Based on [MLX's lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html):

```cpp
// Build expression tree, execute only when needed
template<typename Op, typename LHS, typename RHS>
class BinaryExpr {
    LHS lhs_;
    RHS rhs_;
public:
    float operator[](size_t i) const {
        return Op::apply(lhs_[i], rhs_[i]);
    }
};

// a + b + c builds tree, no temporaries until assignment
auto expr = a + b + c;  // BinaryExpr<Add, BinaryExpr<Add, A, B>, C>
result = expr;          // Single fused loop execution
```

### 7.2 Compile-Time Fusion with C++ Templates

```cpp
// Zero-overhead abstraction for fusion
template<typename... Ops>
class FusedKernel {
    std::tuple<Ops...> ops_;

    void execute(float* out, const float* in, size_t n) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 val = _mm256_loadu_ps(in + i);
            val = apply_all(val, ops_);  // Compile-time unrolled
            _mm256_storeu_ps(out + i, val);
        }
    }
};
```

---

## Part 8: Polyhedral Optimization

### 8.1 Integration with Polly (LLVM)

[Polly](https://polly.llvm.org/) can automatically optimize loop nests:

```bash
# Compile with Polly enabled
clang++ -O3 -mllvm -polly -mllvm -polly-vectorizer=stripmine kernel.cpp
```

### 8.2 Manual Polyhedral Transforms

For JIT, implement key transforms:

```cpp
// Loop tiling
void tile_loop(Loop* L, size_t tile_x, size_t tile_y) {
    // Original: for i in [0, N): for j in [0, M)
    // Tiled: for ii in [0, N, tile_x): for jj in [0, M, tile_y):
    //          for i in [ii, min(ii+tile_x, N)): for j in [jj, ...]
}

// Loop interchange (for better cache access)
void interchange(Loop* outer, Loop* inner) {
    // Swap loop ordering when legal
}
```

---

## Implementation Priority

| Priority | Optimization | Expected Gain | Complexity |
|----------|--------------|---------------|------------|
| **P0** | Multi-accumulator dot product | 2-4x for DOT | Low |
| **P0** | Eliminate binding overhead | 2-10x for small arrays | Medium |
| **P1** | Non-temporal stores for large arrays | 1.5-2x for ADD | Low |
| **P1** | Loop fusion in IR | 2-3x for chains | Medium |
| **P1** | AVX-512 masked tail handling | 10-30% overall | Low |
| **P2** | Online autotuning | 10-30% adaptive | High |
| **P2** | Expression templates | 20-50% for chains | Medium |
| **P3** | Polyhedral optimization | Variable | Very High |

---

## Validation Strategy

1. **Micro-benchmarks**: Test each optimization in isolation
2. **Differential Testing**: Compare against NumPy/BLAS for correctness
3. **Roofline Analysis**: Verify we're hitting memory/compute bounds
4. **Profile-guided**: Use perf/VTune to verify cache/memory improvements

```bash
# Roofline analysis
perf stat -e cache-misses,cache-references,instructions,cycles ./benchmark

# Memory bandwidth measurement
./likwid-perfctr -g MEM ./benchmark
```

---

## Sources

### BLAS & SIMD Optimization
- [SimSIMD - 200x Faster Dot Products](https://github.com/ashvardanian/simsimd)
- [NumPy vs BLAS: 90% Throughput Loss](https://ashvardanian.com/posts/numpy-vs-blas-costs/)
- [SIMD Dot Product Optimization Case Study](https://github.com/segfaultscribe/SIMD-Dot-product-Optimization)
- [AVX Vector Sum Reduction](https://www.aussieai.com/book/ch30-vectorized-sum-reduction)
- [OpenBLAS Wikipedia](https://en.wikipedia.org/wiki/OpenBLAS)
- [Intel MKL Wikipedia](https://en.wikipedia.org/wiki/Math_Kernel_Library)

### Memory & Cache Optimization
- [Memory Bandwidth - Algorithmica](https://en.algorithmica.org/hpc/cpu-cache/bandwidth/)
- [Non-Temporal Store Notes](https://sites.utexas.edu/jdm4372/2018/01/01/notes-on-non-temporal-aka-streaming-stores/)
- [Cache Blocking - CS:APP](https://csapp.cs.cmu.edu/2e/waside/waside-blocking.pdf)
- [Multi-Strided Prefetch Patterns](https://arxiv.org/html/2412.16001v1)

### JIT & Autotuning
- [TVM Ansor Auto-Scheduler](https://tvm.apache.org/2021/03/03/intro-auto-scheduler)
- [Halide Autoscheduler with ML](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)
- [Compiler Autotuning Survey](https://arxiv.org/pdf/1801.04405)
- [Profile-Guided Optimization - Wikipedia](https://en.wikipedia.org/wiki/Profile-guided_optimization)
- [HotSpot JVM Performance Techniques](https://wiki.openjdk.org/display/HotSpot/PerformanceTechniques)

### Loop Optimization
- [Loop Fusion - Wikipedia](https://en.wikipedia.org/wiki/Loop_fission_and_fusion)
- [Polyhedral Model - Wikipedia](https://en.wikipedia.org/wiki/Polytope_model)
- [Polly LLVM](https://polly.llvm.org/)
- [PLUTO Automatic Parallelizer](https://www.ece.lsu.edu/~pluto/index.html)

### Runtime Dispatch
- [ClickHouse CPU Dispatch](https://clickhouse.com/blog/cpu-dispatch-in-clickhouse)
- [Magnum Engine CPU Dispatch](https://blog.magnum.graphics/backstage/cpu-feature-detection-dispatch/)
- [Intel MKL ISA Dispatch](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-1/instruction-set-specific-dispatch-on-intel-archs.html)

### Expression Templates
- [Expression Templates - Wikipedia](https://en.wikipedia.org/wiki/Expression_templates)
- [Zero Cost Abstraction in C++](https://medium.com/@rahulchakraborty337/zero-cost-abstraction-in-c-fbc9be45772b)
- [MLX Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)

### Parallel Reductions
- [OpenMP SIMD Reductions](https://passlab.github.io/InteractiveOpenMPProgramming/SIMDandVectorArchitecture/6_SIMDReductionsAndScans.html)
- [Parallel Reduction Overview](https://www.sciencedirect.com/topics/computer-science/parallel-reduction)
