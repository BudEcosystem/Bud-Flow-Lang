# Bud Flow Lang Performance Guide

This guide covers optimization techniques for getting maximum performance from Bud Flow Lang.

## Architecture Overview

Bud Flow Lang uses a three-tier execution model:

| Tier | Description | Latency | Throughput |
|------|-------------|---------|------------|
| Tier 0 | Direct Highway dispatch (interpreter) | ~1 us | Baseline |
| Tier 1 | Copy-and-Patch JIT | <1 ms compile | 2-5x faster |
| Tier 2 | Fused kernels | 0 (pre-compiled) | 6-30x faster |

Hot paths are automatically promoted to higher tiers based on call frequency.

## SIMD Acceleration

Bud Flow Lang uses [Highway](https://github.com/google/highway) for portable SIMD across:

- **x86-64**: SSE4, AVX2, AVX-512
- **ARM64**: NEON, SVE
- **RISC-V**: V extension

Highway automatically selects the best instruction set at runtime.

## Best Practices

### 1. Use Large Arrays

SIMD performs best with large, contiguous data:

```python
# Good: One operation on large array
result = flow.add(big_a, big_b)  # 1M elements

# Avoid: Many small operations
for i in range(1000):
    result[i] = flow.add(a[i:i+1], b[i:i+1])
```

**Minimum recommended size**: 256 elements for measurable speedup.

### 2. Enable Operator Fusion

Fusion combines multiple operations into a single pass, reducing memory bandwidth:

```python
@flow.kernel(enable_fusion=True)  # Default
def fused(a, b, c):
    return a * b + c  # Fused Multiply-Add

# Without fusion: 3 memory passes
# With fusion: 1 memory pass (3x less bandwidth)
```

### 3. Use In-Place Operations

In-place operations avoid allocation overhead:

```python
# Good: In-place
x += y
x *= 2.0

# Less efficient: Creates new array
x = x + y
x = x * 2.0
```

### 4. Reuse Traced Kernels

Kernel tracing has compilation overhead. Reuse kernels:

```python
@flow.kernel
def my_kernel(a, b):
    return a + b * a

# First call: Trace and compile (~ms)
result = my_kernel(x, y)

# Subsequent calls: Use cached kernel (fast)
for _ in range(1000):
    result = my_kernel(x, y)
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

### 6. Use Aligned Memory

Bunch arrays are automatically 64-byte aligned for optimal SIMD. Converting from NumPy may lose alignment:

```python
# Optimal: Create directly
x = flow.arange(1000)

# May be unaligned: From NumPy
import numpy as np
x = flow.Bunch(np.random.randn(1000).tolist())
```

### 7. Batch Reduction Operations

Combine reductions when possible:

```python
# Good: Single dot product
dot = flow.dot(a, b)  # Fused reduction

# Less efficient: Separate operations
mul = flow.mul(a, b)
dot = flow.reduce_sum(mul)
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
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn(*args)
    elapsed = time.perf_counter() - start

    return elapsed / iterations
```

### Expected Throughput

On modern x86-64 (AVX2):

| Operation | Throughput (billion ops/sec) |
|-----------|------------------------------|
| Add | 15-25 |
| Mul | 15-25 |
| FMA | 25-40 |
| Exp | 2-4 |
| Sqrt | 8-12 |
| Dot product | 20-30 |

## Memory Bandwidth

For memory-bound operations, theoretical peak:

- **DDR4-3200**: ~50 GB/s
- **DDR5-4800**: ~75 GB/s

Element-wise operations on large arrays are typically memory-bound.

### Roofline Analysis

For an operation processing N floats:
- **Memory traffic**: 2 * N * 4 bytes (read input + write output)
- **Ops**: N operations

Arithmetic intensity = N / (8N) = 0.125 ops/byte

At 50 GB/s bandwidth: ~6.25 billion ops/sec theoretical max for element-wise ops.

## Optimization Levels

```python
@flow.kernel(opt_level=2)
def my_kernel(a, b):
    return a + b
```

| Level | Optimizations | Use Case |
|-------|---------------|----------|
| 0 | None | Debugging |
| 1 | Constant folding, DCE | Quick compile |
| 2 | + Fusion, strength reduction | Default |
| 3 | + Aggressive inlining | Hot loops |

## Common Pitfalls

### 1. Python Loop Overhead

```python
# Bad: Python loop
for i in range(len(x)):
    x[i] += 1

# Good: Vectorized
x += flow.ones(x.size())
```

### 2. Frequent Array Creation

```python
# Bad: Creates temporary each iteration
for _ in range(1000):
    temp = flow.zeros(n)
    temp += x

# Good: Reuse arrays
temp = flow.zeros(n)
for _ in range(1000):
    temp *= 0  # Reset
    temp += x
```

### 3. Mixing NumPy and Flow

```python
# Avoid: Frequent conversions
for _ in range(1000):
    np_arr = bunch.to_numpy()  # Slow!
    # ... numpy operations ...
    bunch = flow.Bunch(np_arr.tolist())  # Slow!

# Better: Stay in one domain
```

## Hardware-Specific Tips

### Intel (AVX-512)

- Best with arrays aligned to 64 bytes
- Optimal for large arrays (>4K elements)
- Turbo boost may reduce frequency under heavy SIMD load

### AMD (Zen 3/4)

- Excellent AVX2 performance
- 256-bit units often faster than 512-bit on power/efficiency
- Good memory bandwidth

### Apple Silicon (M1/M2/M3)

- NEON + Apple AMX for matrix ops
- Unified memory reduces copy overhead
- Excellent single-thread performance

### AWS Graviton (ARM64)

- NEON with SVE on Graviton3
- Wide vector units (256-bit+)
- Good for throughput-oriented workloads
