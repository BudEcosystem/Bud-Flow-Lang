# Getting Started with Bud Flow Lang

This guide will get you up and running with Bud Flow Lang in 5 minutes.

---

## Prerequisites

Make sure you have Bud Flow Lang installed. See the [Installation Guide](installation.md) for details.

---

## Your First Program

```python
import bud_flow_lang_py as flow

# Step 1: Initialize the runtime
flow.initialize()

# Step 2: Create arrays
a = flow.ones(1000)       # Array of 1000 ones
b = flow.full(1000, 2.0)  # Array of 1000 twos

# Step 3: Perform operations
c = a + b                 # Element-wise addition
d = a * b                 # Element-wise multiplication
e = flow.dot(a, b)        # Dot product

# Step 4: Get results
print(f"Sum of (a + b): {c.sum()}")     # 3000.0
print(f"Sum of (a * b): {d.sum()}")     # 2000.0
print(f"Dot product: {e}")              # 2000.0

# Step 5: Convert to NumPy if needed
import numpy as np
np_array = c.to_numpy()
print(f"NumPy array: {np_array[:5]}")   # [3. 3. 3. 3. 3.]
```

---

## Understanding the Basics

### Initialization

Always initialize the runtime before using Flow:

```python
import bud_flow_lang_py as flow

# Initialize with default settings
flow.initialize()

# Check if initialized
if flow.is_initialized():
    print("Ready to go!")

# Get hardware info
info = flow.get_hardware_info()
print(f"Using {info['simd_width']*8}-bit SIMD")
```

### Creating Arrays

Flow provides several ways to create arrays:

```python
# Filled arrays
zeros = flow.zeros(100)           # All zeros
ones = flow.ones(100)             # All ones
filled = flow.full(100, 3.14)     # All 3.14

# Sequences
range_arr = flow.arange(10)             # [0, 1, 2, ..., 9]
range_arr = flow.arange(5, 0.0, 2.0)    # [0, 2, 4, 6, 8] - 5 elements, start 0, step 2
linear = flow.linspace(0.0, 1.0, 5)     # [0, 0.25, 0.5, 0.75, 1.0]

# From NumPy
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
flow_arr = flow.flow(np_arr)      # Create from NumPy

# From Python list
flow_arr = flow.flow([1.0, 2.0, 3.0])
```

### Array Properties

```python
a = flow.ones(1000)

print(f"Size: {a.size}")           # 1000
print(f"Shape: {a.shape}")         # (1000,)
print(f"Dtype: {a.dtype}")         # float32
print(f"Bytes: {a.nbytes}")        # 4000
```

### Arithmetic Operations

```python
a = flow.arange(5)     # [0, 1, 2, 3, 4]
b = flow.full(5, 2.0)  # [2, 2, 2, 2, 2]

# Element-wise operations
c = a + b    # [2, 3, 4, 5, 6]
c = a - b    # [-2, -1, 0, 1, 2]
c = a * b    # [0, 2, 4, 6, 8]
c = a / b    # [0, 0.5, 1, 1.5, 2]

# Scalar operations
c = a + 10   # Add scalar to each element
c = a * 2    # Multiply each element
```

### Reduction Operations

```python
a = flow.arange(10)  # [0, 1, 2, ..., 9]

total = a.sum()      # 45.0
minimum = a.min()    # 0.0
maximum = a.max()    # 9.0
average = a.mean()   # 4.5
```

### Dot Product

```python
a = flow.ones(1000)
b = flow.ones(1000)

# Method 1: Function
dot = flow.dot(a, b)  # 1000.0

# Method 2: Method
dot = a.dot(b)        # 1000.0
```

### Fused Multiply-Add (FMA)

FMA computes `a * b + c` in a single operation, which is faster and more accurate:

```python
a = flow.ones(1000)
b = flow.full(1000, 2.0)
c = flow.full(1000, 3.0)

# FMA: a * b + c = 1 * 2 + 3 = 5
result = flow.fma(a, b, c)
print(f"FMA result: {result.sum()}")  # 5000.0
```

---

## Working with NumPy

### Converting to NumPy

```python
import numpy as np

flow_arr = flow.ones(100)
np_arr = flow_arr.to_numpy()  # Zero-copy when possible

print(type(np_arr))  # <class 'numpy.ndarray'>
```

### Creating from NumPy

```python
import numpy as np

np_arr = np.random.randn(1000).astype(np.float32)
flow_arr = flow.flow(np_arr)

# Verify
print(f"NumPy sum: {np_arr.sum()}")
print(f"Flow sum: {flow_arr.sum()}")
```

### Mixed Operations

```python
import numpy as np

# Create arrays in both libraries
np_a = np.ones(1000, dtype=np.float32)
flow_b = flow.ones(1000)

# Convert and compute
flow_a = flow.flow(np_a)
result = flow_a + flow_b

# Back to NumPy for further processing
np_result = result.to_numpy()
```

---

## Memory Optimization

Flow automatically optimizes memory access for large arrays:

```python
# Check memory optimization status
status = flow.get_memory_optimization_status()
print(f"Tiling: {status['tiling_enabled']}")
print(f"Prefetch: {status['prefetch_enabled']}")

# Cache configuration
cache = flow.detect_cache_config()
print(f"L1: {cache['l1_size_kb']}KB")
print(f"L2: {cache['l2_size_kb']}KB")
print(f"L3: {cache['l3_size_kb']}KB")

# Optimal tile size for your CPU
tile_size = flow.optimal_tile_size(
    element_size=4,  # float32 = 4 bytes
    num_arrays=2     # Operating on 2 arrays
)
print(f"Optimal tile size: {tile_size} elements")

# Control optimization (usually leave enabled)
flow.set_tiling_enabled(True)
flow.set_prefetch_enabled(True)
```

---

## Best Practices

### 1. Initialize Once

```python
# Good: Initialize once at startup
flow.initialize()

# Bad: Don't initialize multiple times
# flow.initialize()  # Unnecessary
```

### 2. Use Large Arrays

Flow shines with large arrays where SIMD makes a difference:

```python
# Good: Large arrays benefit from SIMD
a = flow.ones(1_000_000)  # 1 million elements

# Less benefit: Small arrays have Python overhead
a = flow.ones(10)  # NumPy might be faster
```

### 3. Use Fused Operations

When possible, use fused operations like FMA:

```python
# Good: Single fused operation
result = flow.fma(a, b, c)  # a*b + c in one pass

# Less efficient: Separate operations
result = a * b + c  # Two passes over memory
```

### 4. Minimize Conversions

```python
# Good: Stay in Flow for multiple operations
a = flow.ones(1000)
b = a * 2
c = b + 1
d = c.sum()

# Less efficient: Converting back and forth
a = flow.ones(1000)
np_a = a.to_numpy()  # Conversion
np_b = np_a * 2
b = flow.flow(np_b)  # Conversion
```

### 5. Clean Up

```python
# The runtime cleans up automatically, but you can be explicit:
flow.shutdown()  # Called automatically at exit
```

---

## Example: Vector Normalization

```python
import bud_flow_lang_py as flow

flow.initialize()

# Create a random vector
import numpy as np
np_data = np.random.randn(1000000).astype(np.float32)
v = flow.flow(np_data)

# Compute L2 norm: sqrt(sum(v^2))
v_squared = v * v
sum_squared = v_squared.sum()
norm = sum_squared ** 0.5

# Normalize
v_normalized = v * (1.0 / norm)

print(f"Original norm: {norm}")
print(f"Normalized norm: {(v_normalized * v_normalized).sum() ** 0.5}")  # ~1.0
```

---

## Example: Dot Product Performance

```python
import bud_flow_lang_py as flow
import numpy as np
import time

flow.initialize()

# Create large arrays
n = 10_000_000  # 10 million elements
np_a = np.ones(n, dtype=np.float32)
np_b = np.ones(n, dtype=np.float32)
flow_a = flow.ones(n)
flow_b = flow.ones(n)

# Warmup
_ = flow.dot(flow_a, flow_b)
_ = np.dot(np_a, np_b)

# Benchmark Flow
start = time.perf_counter()
for _ in range(10):
    result = flow.dot(flow_a, flow_b)
flow_time = (time.perf_counter() - start) / 10

# Benchmark NumPy
start = time.perf_counter()
for _ in range(10):
    result = np.dot(np_a, np_b)
np_time = (time.perf_counter() - start) / 10

print(f"Flow: {flow_time*1000:.2f}ms")
print(f"NumPy: {np_time*1000:.2f}ms")

# Memory bandwidth
bytes_read = 2 * n * 4  # 2 arrays * n elements * 4 bytes
flow_bw = bytes_read / flow_time / 1e9
print(f"Flow bandwidth: {flow_bw:.1f} GB/s")
```

---

## Next Steps

- [Basic Operations Tutorial](tutorials/02_basic_operations.md) - Deep dive into the API
- [Memory Optimization](tutorials/03_memory_optimization.md) - Cache-aware computing
- [Performance Tuning](tutorials/05_performance.md) - Getting maximum speed
- [API Reference](api/core.md) - Complete API documentation
- [Jupyter Notebooks](notebooks/) - Interactive examples
