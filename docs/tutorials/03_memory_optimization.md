# Tutorial 3: Memory Optimization

Learn how Bud Flow Lang optimizes memory access for maximum performance.

---

## What You'll Learn

1. Understanding cache hierarchy
2. How tiling improves performance
3. Prefetching strategies
4. Memory-bound vs compute-bound operations
5. Best practices for large arrays

---

## Prerequisites

- Completed [Tutorial 1](01_first_program.md) and [Tutorial 2](02_basic_operations.md)
- Basic understanding of CPU caches

---

## Understanding Cache Hierarchy

Modern CPUs have multiple levels of cache:

```
┌─────────────────────────────────────────────┐
│                   CPU Core                   │
├─────────────────────────────────────────────┤
│  L1 Cache (32KB)    ~4 cycles latency       │
├─────────────────────────────────────────────┤
│  L2 Cache (256KB)   ~12 cycles latency      │
├─────────────────────────────────────────────┤
│  L3 Cache (8MB+)    ~40 cycles latency      │
├─────────────────────────────────────────────┤
│  Main RAM (GBs)     ~100+ cycles latency    │
└─────────────────────────────────────────────┘
```

**Key insight**: Data that fits in cache is accessed 10-100x faster than main memory.

---

## Checking Your Cache Configuration

```python
import bud_flow_lang_py as flow

flow.initialize()

# Get cache information
cache = flow.detect_cache_config()

print("=== Your Cache Configuration ===")
print(f"L1 Cache: {cache['l1_size_kb']} KB")
print(f"L2 Cache: {cache['l2_size_kb']} KB")
print(f"L3 Cache: {cache['l3_size_kb']} KB")
print(f"Cache Line: {cache['line_size']} bytes")

# Calculate how many float32s fit in each level
f32_size = 4  # bytes
print(f"\nFloat32 capacity:")
print(f"  L1: {cache['l1_size'] // f32_size:,} elements")
print(f"  L2: {cache['l2_size'] // f32_size:,} elements")
print(f"  L3: {cache['l3_size'] // f32_size:,} elements")
```

**Sample Output:**
```
=== Your Cache Configuration ===
L1 Cache: 32 KB
L2 Cache: 256 KB
L3 Cache: 8192 KB
Cache Line: 64 bytes

Float32 capacity:
  L1: 8,192 elements
  L2: 65,536 elements
  L3: 2,097,152 elements
```

---

## What is Tiling?

Tiling (blocking) processes large arrays in smaller chunks that fit in cache.

### Without Tiling

```
Array: [████████████████████████████████████████]
         ↑ Stream through, causing cache misses
```

### With Tiling

```
Array: [████|████|████|████|████|████|████|████]
        tile  tile  tile  tile  tile  tile  tile  tile
         ↑ Process each tile while it's hot in cache
```

---

## How Tiling Works in Bud Flow Lang

```python
import bud_flow_lang_py as flow

flow.initialize()

# Check tiling status
status = flow.get_memory_optimization_status()

print("=== Memory Optimization Status ===")
print(f"Tiling enabled: {status['tiling_enabled']}")
print(f"Prefetch enabled: {status['prefetch_enabled']}")
print(f"Optimal tile size (float32): {status['optimal_tile_size_float32']:,} elements")

# Calculate optimal tile for your operation
element_size = 4   # float32 = 4 bytes
num_arrays = 3     # For operations like: c = a + b

optimal_tile = flow.optimal_tile_size(element_size, num_arrays)
print(f"\nOptimal tile for 3 arrays: {optimal_tile:,} elements")
print(f"  = {optimal_tile * element_size / 1024:.1f} KB per array")
```

---

## Controlling Memory Optimization

### Enable/Disable Tiling

```python
# Enable tiling (default)
flow.set_tiling_enabled(True)

# Check if enabled
if flow.is_tiling_enabled():
    print("Tiling is active")

# Disable for comparison
flow.set_tiling_enabled(False)
```

### Enable/Disable Prefetching

```python
# Enable prefetching (default)
flow.set_prefetch_enabled(True)

# Check if enabled
if flow.is_prefetch_enabled():
    print("Prefetching is active")

# Disable for comparison
flow.set_prefetch_enabled(False)
```

---

## Performance Impact of Tiling

Let's measure the impact of tiling on large arrays:

```python
import bud_flow_lang_py as flow
import time

def benchmark(name, func, iterations=50):
    """Benchmark a function."""
    # Warmup
    for _ in range(10):
        func()

    # Timing
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    return elapsed

flow.initialize()

# Large array that exceeds L3 cache
n = 10_000_000  # 40 MB per array
a = flow.ones(n)
b = flow.ones(n)
c = flow.ones(n)

print(f"Array size: {n:,} elements ({n * 4 / 1e6:.0f} MB each)")
print("=" * 50)

# Benchmark with tiling enabled
flow.set_tiling_enabled(True)
flow.set_prefetch_enabled(True)
time_optimized = benchmark("Optimized", lambda: flow.fma(a, b, c))

# Benchmark with tiling disabled
flow.set_tiling_enabled(False)
flow.set_prefetch_enabled(False)
time_basic = benchmark("Basic", lambda: flow.fma(a, b, c))

# Re-enable optimizations
flow.set_tiling_enabled(True)
flow.set_prefetch_enabled(True)

print(f"With optimization: {time_optimized:.3f} ms")
print(f"Without optimization: {time_basic:.3f} ms")
print(f"Speedup: {time_basic / time_optimized:.2f}x")
```

---

## Memory Bandwidth Analysis

Understanding whether your code is memory-bound or compute-bound:

```python
import bud_flow_lang_py as flow
import time

def measure_bandwidth(operation_name, func, bytes_accessed, iterations=50):
    """Measure effective memory bandwidth."""
    # Warmup
    for _ in range(10):
        func()

    # Timing
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = (time.perf_counter() - start) / iterations

    bandwidth_gbs = bytes_accessed / elapsed / 1e9
    return elapsed * 1000, bandwidth_gbs

flow.initialize()

# Get cache info for reference
cache = flow.detect_cache_config()

print("=== Memory Bandwidth Analysis ===\n")

# Test different sizes
sizes = [
    ("L1-sized", cache['l1_size'] // 4 // 2),        # Fit 2 arrays in L1
    ("L2-sized", cache['l2_size'] // 4 // 2),        # Fit 2 arrays in L2
    ("L3-sized", cache['l3_size'] // 4 // 2),        # Fit 2 arrays in L3
    ("Memory-sized", 10_000_000),                      # Exceeds caches
]

for name, n in sizes:
    a = flow.ones(n)
    b = flow.ones(n)

    # Dot product: reads 2*n*4 bytes
    bytes_read = 2 * n * 4

    time_ms, bw_gbs = measure_bandwidth(
        f"Dot ({name})",
        lambda: flow.dot(a, b),
        bytes_read
    )

    print(f"{name:15} ({n:>10,} elements): {time_ms:7.3f} ms, {bw_gbs:5.1f} GB/s")

    del a, b
```

**Sample Output:**
```
=== Memory Bandwidth Analysis ===

L1-sized        (     4,096 elements):   0.002 ms,  15.2 GB/s
L2-sized        (    32,768 elements):   0.012 ms,  22.1 GB/s
L3-sized        (   524,288 elements):   0.189 ms,  22.3 GB/s
Memory-sized    (10,000,000 elements):   4.521 ms,  17.7 GB/s
```

---

## Cache-Friendly Patterns

### Pattern 1: Process Data in Tiles

When you have multiple passes over data, process in tiles:

```python
def process_in_tiles(data, tile_size):
    """Process data in cache-friendly tiles."""
    n = data.size
    results = []

    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        # Process tile (data[start:end] stays in cache)
        # ... operations on tile ...

    return results

# Let Flow handle tiling automatically for built-in ops
a = flow.ones(1_000_000)
b = a.sum()  # Automatically uses optimal tiling
```

### Pattern 2: Fuse Operations

Reduce memory traffic by fusing operations:

```python
# Bad: 3 passes over memory
temp1 = a * b      # Pass 1: read a,b write temp1
temp2 = temp1 + c  # Pass 2: read temp1,c write temp2
result = temp2     # temp2 is result

# Good: 1 pass over memory
result = flow.fma(a, b, c)  # Pass 1: read a,b,c write result
```

### Pattern 3: Minimize Temporary Arrays

```python
# Bad: Creates multiple temporaries
result = a + b + c + d  # 3 temporary arrays

# Better: Chain additions to reuse memory
result = a + b
result = result + c
result = result + d

# Best for large arrays: Use in-place where possible
# (Bud Flow Lang handles this internally)
```

---

## When Tiling Helps Most

Tiling provides the biggest benefit when:

1. **Array size exceeds L3 cache**: Multi-MB arrays
2. **Multiple passes over data**: Chained operations
3. **Multiple input arrays**: Binary/ternary operations

```python
import bud_flow_lang_py as flow

flow.initialize()
cache = flow.detect_cache_config()

# Threshold where tiling starts helping
# Rule of thumb: when arrays exceed ~50% of L3 cache
threshold = cache['l3_size'] // 2

print(f"Tiling helps most for arrays > {threshold / 1e6:.1f} MB")
print(f"That's > {threshold // 4:,} float32 elements")
```

---

## Complete Example: Optimized Vector Operations

```python
import bud_flow_lang_py as flow
import time

def euclidean_distance_naive(a, b):
    """Naive implementation - multiple passes."""
    diff = a - b
    squared = diff * diff
    return squared.sum() ** 0.5

def euclidean_distance_optimized(a, b):
    """Optimized - fewer passes using dot product."""
    diff = a - b
    return flow.dot(diff, diff) ** 0.5

def benchmark_distances():
    flow.initialize()

    sizes = [100_000, 1_000_000, 10_000_000]

    print("=== Euclidean Distance Benchmark ===\n")

    for n in sizes:
        a = flow.linspace(0, 1, n)
        b = flow.linspace(1, 2, n)

        # Warmup
        for _ in range(5):
            euclidean_distance_naive(a, b)
            euclidean_distance_optimized(a, b)

        # Benchmark naive
        start = time.perf_counter()
        for _ in range(20):
            dist1 = euclidean_distance_naive(a, b)
        time_naive = (time.perf_counter() - start) / 20 * 1000

        # Benchmark optimized
        start = time.perf_counter()
        for _ in range(20):
            dist2 = euclidean_distance_optimized(a, b)
        time_opt = (time.perf_counter() - start) / 20 * 1000

        print(f"n = {n:>10,}:")
        print(f"  Naive:     {time_naive:.3f} ms (dist = {dist1:.6f})")
        print(f"  Optimized: {time_opt:.3f} ms (dist = {dist2:.6f})")
        print(f"  Speedup:   {time_naive / time_opt:.2f}x")
        print()

        del a, b

if __name__ == "__main__":
    benchmark_distances()
```

---

## Best Practices Summary

### Do:

1. **Use FMA for multiply-add patterns**: `flow.fma(a, b, c)` instead of `a * b + c`
2. **Let Flow handle tiling**: Built-in operations are already optimized
3. **Use large arrays**: SIMD + tiling work best with many elements
4. **Process arrays fully before moving to next**: Avoid interleaving
5. **Check bandwidth limits**: Know if you're memory-bound

### Don't:

1. **Create unnecessary temporaries**: Chain operations when possible
2. **Process tiny arrays repeatedly**: Batch into larger arrays
3. **Disable optimizations**: Unless benchmarking or debugging
4. **Assume more code = faster**: Simple fused ops beat complex logic

---

## Exercises

### Exercise 1: Cache Boundary Analysis

Write code to find where performance drops as array size increases:

```python
# Test array sizes from 1K to 100M elements
# Plot time vs size on a log-log scale
# Identify L1, L2, L3, and memory boundaries
```

### Exercise 2: FMA Fusion Benefit

Compare the performance of:
1. `a * b + c * d` (4 operations)
2. `fma(a, b, fma(c, d, zeros))` (2 fused operations)

For arrays of 10M elements.

### Exercise 3: Memory Bandwidth Calculation

Calculate and compare your measured bandwidth to your system's theoretical maximum:
- DDR4-3200: ~25 GB/s per channel
- DDR5-4800: ~38 GB/s per channel

---

## Next Steps

- [Tutorial 4: NumPy Interoperability](04_numpy_interop.md)
- [Tutorial 5: Performance Optimization](05_performance.md)
- [API Reference: Core](../api/core.md) - Memory configuration functions

---

## Summary

In this tutorial, you learned:

- How CPU cache hierarchy affects performance
- What tiling is and how it improves cache utilization
- How to check and control memory optimizations
- How to measure memory bandwidth
- Best practices for cache-friendly code
