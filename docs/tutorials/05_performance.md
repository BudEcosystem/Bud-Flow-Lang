# Tutorial 5: Performance Optimization

Master the art of writing high-performance code with Bud Flow Lang.

---

## What You'll Learn

1. Understanding SIMD and why it matters
2. Benchmarking methodology
3. Operation fusion techniques
4. Memory bandwidth optimization
5. Common performance pitfalls
6. Profiling your code

---

## Prerequisites

- Completed all previous tutorials
- Basic understanding of CPU architecture

---

## Understanding SIMD

**SIMD** (Single Instruction, Multiple Data) processes multiple values simultaneously:

```
Scalar (1 at a time):
  [a₀] + [b₀] → [c₀]
  [a₁] + [b₁] → [c₁]
  [a₂] + [b₂] → [c₂]
  [a₃] + [b₃] → [c₃]
  ...

SIMD (8 at a time with AVX2):
  [a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇] + [b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇] → [c₀ c₁ c₂ c₃ c₄ c₅ c₆ c₇]
```

### Check Your SIMD Capabilities

```python
import bud_flow_lang_py as flow

flow.initialize()

# Detailed SIMD information
caps = flow.get_simd_capabilities()
print(caps)

# Programmatic access
info = flow.get_hardware_info()
print(f"\nSIMD width: {info['simd_width']} bytes = {info['simd_width']*8} bits")
print(f"Float32 per vector: {info['simd_width'] // 4}")

if info['has_avx512']:
    print("AVX-512: 16 float32s per instruction")
elif info['has_avx2']:
    print("AVX2: 8 float32s per instruction")
elif info['has_sse2']:
    print("SSE2: 4 float32s per instruction")
elif info['has_neon']:
    print("NEON: 4 float32s per instruction")
```

---

## Proper Benchmarking Methodology

### The Right Way to Benchmark

```python
import bud_flow_lang_py as flow
import time
import gc

def benchmark(func, iterations=100, warmup=20):
    """
    Properly benchmark a function.

    Key elements:
    1. Warmup runs (JIT, cache warmup)
    2. Garbage collection between runs
    3. Multiple iterations for statistics
    4. Use perf_counter for precision
    """
    # Warmup phase - critical for JIT and cache
    for _ in range(warmup):
        func()

    # Force garbage collection
    gc.collect()

    # Measurement phase
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)

    # Statistics
    import statistics
    return {
        'mean_ms': statistics.mean(times) * 1000,
        'std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'median_ms': statistics.median(times) * 1000,
    }

# Example usage
flow.initialize()

n = 1_000_000
a = flow.ones(n)
b = flow.ones(n)

stats = benchmark(lambda: a + b, iterations=100, warmup=30)

print("=== Benchmark Results ===")
print(f"Mean:   {stats['mean_ms']:.4f} ms")
print(f"Std:    {stats['std_ms']:.4f} ms")
print(f"Min:    {stats['min_ms']:.4f} ms")
print(f"Max:    {stats['max_ms']:.4f} ms")
print(f"Median: {stats['median_ms']:.4f} ms")

# Calculate throughput
elements_per_second = n / (stats['mean_ms'] / 1000)
print(f"\nThroughput: {elements_per_second / 1e9:.2f} Gelem/s")
```

### Common Benchmarking Mistakes

```python
# WRONG: No warmup
def bad_benchmark_1():
    start = time.time()  # Wrong: time.time() is imprecise
    result = func()
    return time.time() - start

# WRONG: Single measurement
def bad_benchmark_2():
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start  # Single sample has high variance

# WRONG: Including array creation in timing
def bad_benchmark_3():
    start = time.perf_counter()
    a = flow.ones(1000000)  # Included in timing!
    b = flow.ones(1000000)  # Included in timing!
    result = a + b
    return time.perf_counter() - start

# RIGHT: Arrays created before, proper methodology
def good_benchmark():
    a = flow.ones(1000000)
    b = flow.ones(1000000)
    return benchmark(lambda: a + b, iterations=100, warmup=30)
```

---

## Operation Fusion: The Key to Performance

### Why Fusion Matters

Every operation requires memory bandwidth. By fusing operations, we reduce memory traffic:

```python
import bud_flow_lang_py as flow
import time

flow.initialize()

n = 10_000_000
a = flow.ones(n)
b = flow.full(n, 2.0)
c = flow.full(n, 3.0)

# Benchmark separate operations
def separate_ops():
    temp = a * b  # Load a, b; Store temp
    return temp + c  # Load temp, c; Store result
    # Total: 5 memory operations

# Benchmark fused operation
def fused_op():
    return flow.fma(a, b, c)  # Load a, b, c; Store result
    # Total: 4 memory operations (but in single pass!)

sep_stats = benchmark(separate_ops, iterations=50, warmup=20)
fma_stats = benchmark(fused_op, iterations=50, warmup=20)

print(f"Separate ops: {sep_stats['mean_ms']:.3f} ms")
print(f"Fused FMA:    {fma_stats['mean_ms']:.3f} ms")
print(f"Speedup:      {sep_stats['mean_ms'] / fma_stats['mean_ms']:.2f}x")
```

### Fusible Patterns

| Pattern | Separate | Fused |
|---------|----------|-------|
| `a * b + c` | 2 ops | `flow.fma(a, b, c)` |
| `(a - b) / c` | 2 ops | Consider storing intermediate |
| `sqrt(a * a + b * b)` | 3 ops | FMA for inner part |

---

## Memory Bandwidth: The Ultimate Limit

### Calculating Theoretical Bandwidth

```python
import bud_flow_lang_py as flow
import time

flow.initialize()

def measure_bandwidth(operation_name, func, bytes_per_op, n_elements):
    """Measure effective memory bandwidth."""
    stats = benchmark(func, iterations=50, warmup=20)
    bandwidth_gbs = bytes_per_op / (stats['mean_ms'] / 1000) / 1e9
    return bandwidth_gbs, stats['mean_ms']

n = 10_000_000
a = flow.ones(n)
b = flow.ones(n)
c = flow.ones(n)

print("=== Memory Bandwidth Analysis ===")
print(f"Array size: {n:,} elements ({n * 4 / 1e6:.0f} MB each)")
print()

# Sum: Read 1 array
bw, t = measure_bandwidth("Sum", lambda: a.sum(), n * 4, n)
print(f"Sum:        {bw:5.1f} GB/s ({t:.3f} ms)")

# Add: Read 2 arrays, write 1
bw, t = measure_bandwidth("Add", lambda: a + b, n * 4 * 3, n)
print(f"Add:        {bw:5.1f} GB/s ({t:.3f} ms)")

# FMA: Read 3 arrays, write 1
bw, t = measure_bandwidth("FMA", lambda: flow.fma(a, b, c), n * 4 * 4, n)
print(f"FMA:        {bw:5.1f} GB/s ({t:.3f} ms)")

# Dot: Read 2 arrays (reduction)
bw, t = measure_bandwidth("Dot", lambda: flow.dot(a, b), n * 4 * 2, n)
print(f"Dot:        {bw:5.1f} GB/s ({t:.3f} ms)")

print()
print("Typical DDR4 bandwidth: 20-30 GB/s")
print("Typical DDR5 bandwidth: 40-60 GB/s")
```

### Arithmetic Intensity

**Arithmetic Intensity** = FLOPs / Bytes Transferred

| Operation | FLOPs | Bytes | Intensity |
|-----------|-------|-------|-----------|
| Add | 1 | 12 (2 read + 1 write) | 0.08 |
| FMA | 2 | 16 (3 read + 1 write) | 0.125 |
| Dot Product | 2N | 8N | 0.25 |

Low intensity = **memory-bound** (limited by bandwidth)
High intensity = **compute-bound** (limited by ALU)

---

## Optimal Array Sizes

### Size vs Performance

```python
import bud_flow_lang_py as flow

flow.initialize()

cache = flow.detect_cache_config()

print("=== Optimal Array Sizes ===")
print()
print("Size Category     Elements        Memory    Expected Performance")
print("-" * 70)
print(f"Tiny (L1)         < {cache['l1_size']//4:,}      < {cache['l1_size_kb']} KB   Highest (cache-resident)")
print(f"Small (L2)        < {cache['l2_size']//4:,}    < {cache['l2_size_kb']} KB  High (L2 cache)")
print(f"Medium (L3)       < {cache['l3_size']//4:,}  < {cache['l3_size_kb']//1024} MB   Good (L3 cache)")
print(f"Large (RAM)       > {cache['l3_size']//4:,}  > {cache['l3_size_kb']//1024} MB   Memory-bound")

# Benchmark across sizes
sizes = [1000, 10000, 100000, 1000000, 10000000]

print()
print("Benchmark: Dot Product Throughput")
print("-" * 50)

for n in sizes:
    a = flow.ones(n)
    b = flow.ones(n)

    stats = benchmark(lambda: flow.dot(a, b), iterations=50, warmup=20)
    throughput = n / (stats['mean_ms'] / 1000) / 1e9

    size_kb = n * 4 / 1024
    print(f"n={n:>10,} ({size_kb:>8.1f} KB): {throughput:6.2f} Gelem/s")

    del a, b
```

---

## Common Performance Pitfalls

### Pitfall 1: Creating Arrays in Hot Loops

```python
# BAD: Creates new arrays every iteration
def bad_loop():
    total = 0
    for i in range(1000):
        a = flow.ones(1000)  # Allocation in loop!
        total += a.sum()
    return total

# GOOD: Create once, reuse
def good_loop():
    a = flow.ones(1000)  # Create once
    total = 0
    for i in range(1000):
        total += a.sum()  # Reuse
    return total

# Benchmark
bad_stats = benchmark(bad_loop, iterations=10, warmup=2)
good_stats = benchmark(good_loop, iterations=10, warmup=2)

print(f"Bad pattern:  {bad_stats['mean_ms']:.3f} ms")
print(f"Good pattern: {good_stats['mean_ms']:.3f} ms")
print(f"Speedup: {bad_stats['mean_ms'] / good_stats['mean_ms']:.1f}x")
```

### Pitfall 2: Many Small Arrays Instead of One Large

```python
# BAD: 1000 small operations
def many_small():
    total = 0
    for i in range(1000):
        a = flow.ones(100)
        total += a.sum()
    return total

# GOOD: One large operation
def one_large():
    a = flow.ones(100000)
    return a.sum()

# Both compute the same thing (sum of 100,000 ones)
```

### Pitfall 3: Unnecessary Conversions

```python
import numpy as np

# BAD: Convert every time
def bad_conversion(np_arr):
    for _ in range(100):
        flow_arr = flow.flow(np_arr)  # Convert each iteration
        result = flow_arr.sum()
    return result

# GOOD: Convert once
def good_conversion(np_arr):
    flow_arr = flow.flow(np_arr)  # Convert once
    for _ in range(100):
        result = flow_arr.sum()  # Reuse
    return result
```

### Pitfall 4: Not Using FMA

```python
# SLOW: Separate multiply and add
result = a * b + c  # 2 operations, 2 memory passes

# FAST: Fused multiply-add
result = flow.fma(a, b, c)  # 1 operation, 1 memory pass
```

---

## Profiling Your Code

### Simple Timer

```python
import time

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed*1000:.3f} ms")

# Usage
n = 1_000_000
a = flow.ones(n)
b = flow.ones(n)

with Timer("Addition"):
    result = a + b

with Timer("Dot product"):
    result = flow.dot(a, b)

with Timer("FMA"):
    c = flow.ones(n)
    result = flow.fma(a, b, c)
```

### Throughput Calculator

```python
def print_throughput(name, n_elements, time_seconds, n_arrays_read, n_arrays_write):
    """Print throughput metrics for an operation."""
    bytes_transferred = (n_arrays_read + n_arrays_write) * n_elements * 4

    elem_per_sec = n_elements / time_seconds
    bandwidth = bytes_transferred / time_seconds

    print(f"{name}:")
    print(f"  Time: {time_seconds*1000:.3f} ms")
    print(f"  Throughput: {elem_per_sec/1e9:.2f} Gelem/s")
    print(f"  Bandwidth: {bandwidth/1e9:.2f} GB/s")
    print()

# Example
n = 10_000_000
a = flow.ones(n)
b = flow.ones(n)

start = time.perf_counter()
for _ in range(10):
    result = a + b
elapsed = (time.perf_counter() - start) / 10

print_throughput("Addition", n, elapsed, n_arrays_read=2, n_arrays_write=1)
```

---

## Performance Checklist

Use this checklist to optimize your code:

### Before Writing Code
- [ ] Understand the algorithm's arithmetic intensity
- [ ] Estimate if you'll be memory-bound or compute-bound
- [ ] Choose appropriate array sizes

### While Writing Code
- [ ] Use FMA for multiply-add patterns
- [ ] Minimize temporary array creation
- [ ] Keep frequently-used arrays alive (don't recreate)
- [ ] Use the largest arrays practical (batch operations)

### After Writing Code
- [ ] Benchmark with proper warmup
- [ ] Measure memory bandwidth achieved
- [ ] Compare against theoretical limits
- [ ] Profile to find bottlenecks

---

## Complete Example: Optimized Neural Network Layer

```python
import bud_flow_lang_py as flow
import time

flow.initialize()

def dense_layer_naive(x, weights, bias):
    """Naive dense layer: y = Wx + b"""
    # This creates temporaries
    wx = weights * x  # Temporary 1
    return wx + bias  # Uses temporary

def dense_layer_optimized(x, weights, bias):
    """Optimized using FMA: y = Wx + b"""
    return flow.fma(weights, x, bias)

def benchmark_layers():
    n = 1_000_000

    x = flow.ones(n)
    w = flow.full(n, 0.01)
    b = flow.full(n, 0.1)

    # Warmup
    for _ in range(20):
        dense_layer_naive(x, w, b)
        dense_layer_optimized(x, w, b)

    # Benchmark naive
    start = time.perf_counter()
    for _ in range(100):
        result = dense_layer_naive(x, w, b)
    naive_time = (time.perf_counter() - start) / 100 * 1000

    # Benchmark optimized
    start = time.perf_counter()
    for _ in range(100):
        result = dense_layer_optimized(x, w, b)
    opt_time = (time.perf_counter() - start) / 100 * 1000

    print("=== Dense Layer Performance ===")
    print(f"Input size: {n:,} elements")
    print(f"Naive:     {naive_time:.3f} ms")
    print(f"Optimized: {opt_time:.3f} ms")
    print(f"Speedup:   {naive_time/opt_time:.2f}x")

    # Verify correctness
    r1 = dense_layer_naive(x, w, b).to_numpy()
    r2 = dense_layer_optimized(x, w, b).to_numpy()
    print(f"Results match: {abs(r1[0] - r2[0]) < 1e-6}")

if __name__ == "__main__":
    benchmark_layers()
```

---

## Exercises

### Exercise 1: Benchmark Your System

Create a comprehensive benchmark that measures:
1. Peak addition throughput
2. Peak FMA throughput
3. Peak reduction throughput
4. Memory bandwidth achieved

Compare to theoretical limits for your hardware.

### Exercise 2: Optimize Euclidean Distance

Optimize this naive implementation:

```python
def euclidean_distance_naive(a, b):
    diff = a - b
    squared = diff * diff
    summed = squared.sum()
    return summed ** 0.5

# Can you reduce memory operations?
```

### Exercise 3: Operation Fusion Analysis

For this expression: `(a * b + c) * d + e`

1. Count memory operations for separate evaluation
2. Count memory operations for optimal fusion
3. Implement both and benchmark

---

## Next Steps

- [API Reference](../api/operations.md) - Complete operation list
- [NumPy Comparison Notebook](../notebooks/02_numpy_comparison.ipynb) - Detailed benchmarks
- [Examples](../../examples/) - Real-world code samples

---

## Summary

In this tutorial, you learned:

- How SIMD accelerates array operations
- Proper benchmarking methodology (warmup, statistics)
- Operation fusion reduces memory traffic
- Memory bandwidth is often the limiting factor
- Common pitfalls: array creation in loops, many small arrays
- How to profile and optimize your code

**Key Takeaways:**
1. Always use FMA for `a * b + c` patterns
2. Batch operations into large arrays
3. Create arrays once, reuse many times
4. Benchmark properly with warmup and multiple iterations
5. Know your hardware limits (bandwidth, cache sizes)
