# Bud Flow Lang Performance Guide

This guide covers optimization techniques for getting maximum performance from Bud Flow Lang.

## Benchmark Results

Tested on Intel Core i7-10700KF @ 3.80GHz with AVX2 (256-bit SIMD):

| Operation | vs NumPy | vs JAX | Key Optimization |
|-----------|----------|--------|------------------|
| **SIN** | 2.93x faster | 4.2x faster | Highway SIMD transcendentals |
| **SUM** | 2.82x faster | 3.4x faster | Multi-accumulator reductions |
| **FMA** | 1.93x faster | 2.3x faster | Fused multiply-add instruction |
| **EXP** | 1.55x faster | 1.2x faster | Vectorized exponential |
| **MUL** | 1.36x faster | 1.5x faster | SIMD multiplication |
| **ADD** | 1.29x faster | 2.1x faster | SIMD addition |

### Detailed Reduction Performance

Multi-accumulator reductions provide significant speedups:

| Operation | 100K Elements | 1M Elements | Throughput |
|-----------|---------------|-------------|------------|
| sum | 0.012ms | 0.118ms | 8.4 Gelem/s |
| mean | 0.012ms | 0.117ms | 8.5 Gelem/s |
| min | 0.011ms | 0.121ms | 8.3 Gelem/s |
| max | 0.019ms | 0.125ms | 8.0 Gelem/s |

### FMA Fusion Benefits

| Method | 1M Elements | Speedup |
|--------|-------------|---------|
| NumPy (a*b+c) | 1.559ms | baseline |
| Bud (a*b+c) | 1.518ms | 1.03x |
| Bud FMA (fused) | 0.844ms | 1.85x |

## Architecture Overview

Bud Flow Lang uses a three-tier execution model:

| Tier | Description | Latency | Throughput |
|------|-------------|---------|------------|
| Tier 0 | Direct Highway dispatch | ~1 us | Baseline |
| Tier 1 | Copy-and-Patch JIT | <1 ms compile | 2-5x faster |
| Tier 2 | Fused kernels | Pre-compiled | 6-30x faster |

Hot paths are automatically promoted to higher tiers based on call frequency.

### Dynamic Tier Thresholds

JIT compilation thresholds adapt to array size:

| Array Size | Tier 1 Threshold | Tier 2 Threshold | Rationale |
|------------|------------------|------------------|-----------|
| < 256 | 50 calls | 200 calls | Amortize compile cost |
| 256-4K | 10 calls | 100 calls | Standard |
| > 4K | 3 calls | 30 calls | Immediate benefit |

## JIT Optimizations

Five key optimizations are implemented:

### 1. Multi-Accumulator Reductions

Uses 4 independent accumulators to hide instruction latency:

```cpp
// Standard reduction (latency-bound: 4-cycle dependency)
for (i = 0; i < n; i++)
    sum += a[i];

// Multi-accumulator (4x throughput)
for (i = 0; i < n; i += 4*lanes) {
    sum0 += a[i];
    sum1 += a[i + lanes];
    sum2 += a[i + 2*lanes];
    sum3 += a[i + 3*lanes];  // Independent, pipelined
}
result = sum0 + sum1 + sum2 + sum3;
```

**Speedup**: 2-4x for sum, mean, min, max, dot product.

### 2. Size-Specialized Kernels

| Array Size | Kernel Type | Optimization |
|------------|-------------|--------------|
| < 256 | Small | Fully unrolled, no loop overhead |
| 256-4K | Medium | 4x unrolled with SIMD |
| > 4K | Large | 8x unrolled + prefetching |

### 3. Kernel Fusion

Combines multiple operations into single memory passes:

```python
# Separate (3 memory passes):
temp = a * b
result = temp + c

# Fused (1 memory pass):
result = a.fma(b, c)  # 3x less memory traffic
```

Available fused kernels:
- `FMA`: a * b + c
- `AXPY`: a * x + y
- `AXPBY`: a * x + b * y
- `SumOfSquares`: reduce(a * a)
- `DotProduct`: reduce(a * b)

### 4. Software Prefetching

For large arrays, prefetch hints reduce cache misses:

```python
flow.set_prefetch_enabled(True)
# Now large operations prefetch 2-4 cache lines ahead
```

### 5. Dynamic Tier Thresholds

Array-size-aware JIT compilation:

```python
# Large arrays JIT immediately (3 calls)
# Small arrays wait longer (50 calls) to amortize compilation
```

## SIMD Acceleration

Bud Flow Lang uses [Highway](https://github.com/google/highway) for portable SIMD across:

- **x86-64**: SSE4.2, AVX2, AVX-512, AVX-512BF16
- **ARM64**: NEON, SVE, SVE2
- **RISC-V**: V extension
- **WebAssembly**: SIMD128

Highway automatically selects the best instruction set at runtime.

## Best Practices

### 1. Use Large Arrays

SIMD performs best with large, contiguous data:

```python
# Good: One operation on large array
result = a + b  # 1M elements

# Avoid: Many small operations
for i in range(1000):
    result[i] = a[i] + b[i]
```

**Minimum recommended size**: 256 elements for measurable speedup.

### 2. Use Fused Operations

```python
# Good: Fused multiply-add (single instruction)
result = a.fma(b, c)

# Less efficient: Separate ops (2 instructions, extra memory)
result = a * b + c
```

### 3. Enable Optimization Flags

```python
flow.set_tiling_enabled(True)      # Cache-aware execution
flow.set_prefetch_enabled(True)    # Software prefetching
```

### 4. Use In-Place Operations

In-place operations avoid allocation overhead:

```python
# Good: In-place
x += y
x *= 2.0

# Less efficient: Creates new array
x = x + y
x = x * 2.0
```

### 5. Avoid Small Intermediate Arrays

Break up chains into single expressions:

```python
# Good: Single expression
result = (a + b) * c - d

# Less efficient: Multiple intermediates
temp1 = a + b
temp2 = temp1 * c
result = temp2 - d
```

### 6. Batch Reduction Operations

Combine reductions when possible:

```python
# Good: Single dot product (fused reduction)
dot = a.dot(b)

# Less efficient: Separate operations
mul = a * b
dot = mul.sum()
```

## Profiling

### Measuring Performance

```python
import time

def benchmark(fn, *args, iterations=100, warmup=10):
    # Warmup (trigger JIT compilation)
    for _ in range(warmup):
        fn(*args)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - start)

    return sum(times) / iterations * 1000  # ms
```

### Expected Throughput

On modern x86-64 (AVX2):

| Operation | Throughput (Gelem/s) | Notes |
|-----------|---------------------|-------|
| Add | 15-25 | Memory-bound for large arrays |
| Mul | 15-25 | Memory-bound for large arrays |
| FMA | 25-40 | Compute-bound, excellent |
| Sum | 8-10 | Multi-accumulator optimized |
| Dot | 8-12 | Multi-accumulator optimized |
| Exp | 2-4 | Transcendental, compute-heavy |
| Sin | 2-5 | Transcendental, compute-heavy |
| Sqrt | 8-12 | Fast SIMD sqrt |

## Memory Bandwidth

For memory-bound operations, theoretical peak:

- **DDR4-3200**: ~50 GB/s
- **DDR5-4800**: ~75 GB/s

### Roofline Analysis

For an element-wise operation processing N float32s:
- **Memory traffic**: 3 * N * 4 bytes (2 reads + 1 write)
- **Ops**: N operations

Arithmetic intensity = N / (12N) = 0.083 ops/byte

At 50 GB/s bandwidth: ~4 billion ops/sec theoretical max for element-wise ops.

## Common Pitfalls

### 1. Python Loop Overhead

```python
# Bad: Python loop
for i in range(len(x)):
    x[i] += 1

# Good: Vectorized
x = x + 1.0
```

### 2. Frequent Array Creation

```python
# Bad: Creates temporary each iteration
for _ in range(1000):
    temp = flow.Bunch.zeros(n)
    temp = temp + x

# Good: Reuse arrays
temp = flow.Bunch.zeros(n)
for _ in range(1000):
    temp = x + 0  # Reuse
```

### 3. Mixing NumPy and Flow

```python
# Avoid: Frequent conversions
for _ in range(1000):
    np_arr = bunch.to_numpy()  # Slow!
    # ... numpy operations ...
    bunch = flow.Bunch.from_numpy(np_arr)  # Slow!

# Better: Stay in one domain
```

## Hardware-Specific Tips

### Intel (AVX2/AVX-512)

- Best with arrays aligned to 64 bytes (automatic with Bunch)
- Optimal for arrays > 4K elements
- AVX-512 provides 2x wider vectors but may reduce turbo

### AMD (Zen 3/4)

- Excellent AVX2 performance
- 256-bit execution often matches Intel AVX-512
- Good memory bandwidth with infinity fabric

### Apple Silicon (M1/M2/M3)

- NEON 128-bit SIMD
- Unified memory reduces copy overhead
- Excellent single-thread performance

### AWS Graviton (ARM64)

- NEON with optional SVE on Graviton3
- Wide vector units (256-bit+)
- Good for throughput workloads

## Optimization Checklist

1. [ ] Arrays are large enough (> 256 elements)
2. [ ] Using fused operations where possible (fma, dot)
3. [ ] Tiling and prefetch enabled for large arrays
4. [ ] Avoiding Python loops over array elements
5. [ ] Reusing arrays instead of frequent allocation
6. [ ] Staying within Bud Flow Lang (not mixing with NumPy)
