# Tutorial 4: NumPy Interoperability

Learn how to seamlessly work with NumPy arrays in Bud Flow Lang.

---

## What You'll Learn

1. Converting between NumPy and Flow arrays
2. Data type handling
3. Memory considerations
4. Common workflows
5. Performance tips for interoperability

---

## Prerequisites

- Completed previous tutorials
- NumPy installed (`pip install numpy`)

---

## Basic Conversion

### NumPy to Flow

```python
import bud_flow_lang_py as flow
import numpy as np

flow.initialize()

# Create a NumPy array
np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# Convert to Flow Bunch
flow_arr = flow.flow(np_arr)

print(f"NumPy array: {np_arr}")
print(f"Flow array:  {flow_arr.to_numpy()}")
print(f"Flow size:   {flow_arr.size}")
```

### Flow to NumPy

```python
# Create a Flow array
flow_arr = flow.arange(10)

# Convert to NumPy
np_arr = flow_arr.to_numpy()

print(f"Type: {type(np_arr)}")
print(f"NumPy array: {np_arr}")
print(f"NumPy dtype: {np_arr.dtype}")
```

---

## Data Type Handling

### Supported Types

| NumPy Type | Flow Type | Bytes |
|------------|-----------|-------|
| `np.float32` | `float32` | 4 |
| `np.float64` | `float64` | 8 |
| `np.int32` | `int32` | 4 |
| `np.int64` | `int64` | 8 |

### Explicit Type Conversion

```python
import numpy as np
import bud_flow_lang_py as flow

flow.initialize()

# Float64 NumPy array
np_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# Convert to Flow (preserves dtype)
flow_f64 = flow.flow(np_f64)
print(f"Flow dtype: {flow_f64.dtype}")  # float64

# Float32 (default for flow.ones, flow.zeros, etc.)
flow_f32 = flow.ones(100)
print(f"Flow default dtype: {flow_f32.dtype}")  # float32

# Create with explicit dtype
flow_f64_explicit = flow.ones(100, dtype='float64')
print(f"Explicit dtype: {flow_f64_explicit.dtype}")  # float64
```

### Converting NumPy Types

```python
# If your NumPy array is float64 but you want float32 in Flow:
np_f64 = np.random.randn(1000)  # Default is float64

# Option 1: Convert in NumPy first
np_f32 = np_f64.astype(np.float32)
flow_arr = flow.flow(np_f32)

# Option 2: NumPy will auto-convert if you create float32
np_f32 = np.array(np_f64, dtype=np.float32)
flow_arr = flow.flow(np_f32)
```

---

## Memory Semantics

### Copy Semantics (Default)

Data is **copied** when converting between NumPy and Flow:

```python
np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
flow_arr = flow.flow(np_arr)

# Modify NumPy - Flow is unaffected
np_arr[0] = 999.0
print(f"NumPy after modification: {np_arr}")
print(f"Flow (unchanged): {flow_arr.to_numpy()}")

# Modify Flow result - original Flow is unaffected
np_result = flow_arr.to_numpy()
np_result[0] = 888.0
print(f"NumPy result modified: {np_result}")
print(f"Flow (unchanged): {flow_arr.to_numpy()}")
```

**Output:**
```
NumPy after modification: [999.   2.   3.]
Flow (unchanged): [1. 2. 3.]
NumPy result modified: [888.   2.   3.]
Flow (unchanged): [1. 2. 3.]
```

### Why Copy?

1. **SIMD Alignment**: Flow arrays are aligned for optimal SIMD performance
2. **Memory Safety**: Prevents unexpected modifications
3. **Lifetime Management**: Each system manages its own memory

---

## Common Workflows

### Workflow 1: NumPy Data, Flow Computation

```python
import numpy as np
import bud_flow_lang_py as flow

flow.initialize()

# Generate data with NumPy (random numbers, loading files, etc.)
np_a = np.random.randn(100000).astype(np.float32)
np_b = np.random.randn(100000).astype(np.float32)

# Convert to Flow for fast computation
flow_a = flow.flow(np_a)
flow_b = flow.flow(np_b)

# Compute with Flow's SIMD acceleration
dot_product = flow.dot(flow_a, flow_b)
sum_ab = (flow_a + flow_b).sum()
fma_result = flow.fma(flow_a, flow_b, flow_a)

print(f"Dot product: {dot_product:.4f}")
print(f"Sum of a+b: {sum_ab:.4f}")
print(f"FMA mean: {fma_result.mean():.4f}")
```

### Workflow 2: Flow Computation, NumPy Analysis

```python
import numpy as np
import bud_flow_lang_py as flow

flow.initialize()

# Create and compute with Flow
x = flow.linspace(0, 10, 1001)
y = flow.sin(x)

# Convert to NumPy for analysis or plotting
x_np = x.to_numpy()
y_np = y.to_numpy()

# Use NumPy's rich analysis tools
print(f"Max value: {np.max(y_np):.4f}")
print(f"Argmax: {np.argmax(y_np)}")
print(f"Zero crossings: {np.sum(np.diff(np.sign(y_np)) != 0)}")

# Could also plot with matplotlib
# import matplotlib.pyplot as plt
# plt.plot(x_np, y_np)
# plt.show()
```

### Workflow 3: Mixed Computation Pipeline

```python
import numpy as np
import bud_flow_lang_py as flow

flow.initialize()

def normalize_l2(arr):
    """L2 normalize using Flow."""
    flow_arr = flow.flow(arr) if isinstance(arr, np.ndarray) else arr
    norm = flow.dot(flow_arr, flow_arr) ** 0.5
    return (flow_arr * (1.0 / norm)).to_numpy()

def cosine_similarity(a, b):
    """Compute cosine similarity."""
    a_norm = normalize_l2(a)
    b_norm = normalize_l2(b)

    # Convert for Flow computation
    flow_a = flow.flow(a_norm)
    flow_b = flow.flow(b_norm)

    return flow.dot(flow_a, flow_b)

# Example usage
vec1 = np.random.randn(1000).astype(np.float32)
vec2 = np.random.randn(1000).astype(np.float32)

similarity = cosine_similarity(vec1, vec2)
print(f"Cosine similarity: {similarity:.4f}")
```

---

## Working with Python Lists

```python
# From Python list (auto-converts to float32)
py_list = [1.0, 2.0, 3.0, 4.0, 5.0]
flow_arr = flow.flow(py_list)

print(f"From list: {flow_arr.to_numpy()}")
print(f"Dtype: {flow_arr.dtype}")

# Mixed integers and floats
mixed_list = [1, 2.5, 3, 4.5]
flow_arr = flow.flow(mixed_list)
print(f"From mixed list: {flow_arr.to_numpy()}")
```

---

## Array Properties

Both NumPy and Flow arrays have similar properties:

```python
flow_arr = flow.ones(1000)
np_arr = flow_arr.to_numpy()

print("=== Property Comparison ===")
print(f"{'Property':<15} {'Flow':<20} {'NumPy':<20}")
print("-" * 55)
print(f"{'Size':<15} {flow_arr.size:<20} {np_arr.size:<20}")
print(f"{'Shape':<15} {str(flow_arr.shape):<20} {str(np_arr.shape):<20}")
print(f"{'Dtype':<15} {str(flow_arr.dtype):<20} {str(np_arr.dtype):<20}")
print(f"{'Bytes':<15} {flow_arr.nbytes:<20} {np_arr.nbytes:<20}")
print(f"{'Itemsize':<15} {flow_arr.itemsize:<20} {np_arr.itemsize:<20}")
print(f"{'Ndim':<15} {flow_arr.ndim:<20} {np_arr.ndim:<20}")
```

---

## Performance Considerations

### When to Convert

**Convert to Flow when:**
- Performing many operations on the same data
- Using FMA (fused multiply-add)
- Data is large (>10K elements)

**Keep in NumPy when:**
- Need NumPy-specific operations (fancy indexing, broadcasting)
- Data is small (<1K elements)
- Single simple operation

### Minimizing Conversion Overhead

```python
import numpy as np
import bud_flow_lang_py as flow
import time

flow.initialize()

n = 100000

# BAD: Convert every iteration
def bad_pattern():
    np_arr = np.ones(n, dtype=np.float32)
    total = 0
    for _ in range(100):
        flow_arr = flow.flow(np_arr)  # Convert each time!
        total += flow_arr.sum()
    return total

# GOOD: Convert once, use many times
def good_pattern():
    np_arr = np.ones(n, dtype=np.float32)
    flow_arr = flow.flow(np_arr)  # Convert once
    total = 0
    for _ in range(100):
        total += flow_arr.sum()  # Reuse
    return total

# Benchmark
start = time.perf_counter()
bad_pattern()
bad_time = time.perf_counter() - start

start = time.perf_counter()
good_pattern()
good_time = time.perf_counter() - start

print(f"Bad pattern:  {bad_time:.3f}s")
print(f"Good pattern: {good_time:.3f}s")
print(f"Speedup: {bad_time/good_time:.1f}x")
```

---

## Complete Example: Data Processing Pipeline

```python
import numpy as np
import bud_flow_lang_py as flow

flow.initialize()

def process_sensor_data(raw_data):
    """
    Process sensor data using Flow for heavy computation.

    Args:
        raw_data: NumPy array of raw sensor readings

    Returns:
        dict with processed statistics
    """
    # Convert to Flow for fast processing
    data = flow.flow(raw_data.astype(np.float32))

    # Basic statistics (SIMD accelerated)
    mean_val = data.mean()
    min_val = data.min()
    max_val = data.max()

    # Normalize to [0, 1] range
    range_val = max_val - min_val
    if range_val > 0:
        normalized = (data - min_val) * (1.0 / range_val)
    else:
        normalized = data

    # Compute energy (sum of squares)
    energy = flow.dot(data, data)

    # RMS (root mean square)
    rms = (energy / data.size) ** 0.5

    return {
        'mean': mean_val,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'energy': energy,
        'rms': rms,
        'normalized': normalized.to_numpy()  # Convert back for further use
    }

# Example usage
np.random.seed(42)

# Simulate sensor data (e.g., accelerometer readings)
raw = np.random.randn(100000).astype(np.float32) * 10 + 5

results = process_sensor_data(raw)

print("=== Sensor Data Analysis ===")
print(f"Samples: {len(raw):,}")
print(f"Mean: {results['mean']:.4f}")
print(f"Min: {results['min']:.4f}")
print(f"Max: {results['max']:.4f}")
print(f"Range: {results['range']:.4f}")
print(f"RMS: {results['rms']:.4f}")
print(f"Total Energy: {results['energy']:.2f}")
print(f"Normalized range: [{results['normalized'].min():.4f}, {results['normalized'].max():.4f}]")
```

---

## Exercises

### Exercise 1: Matrix-Vector Product (Row-wise)

Using Flow's dot product, implement a function that multiplies a NumPy matrix by a vector:

```python
def matrix_vector_product(matrix, vector):
    """
    Compute matrix @ vector using Flow.

    Args:
        matrix: 2D NumPy array (m x n)
        vector: 1D NumPy array (n,)

    Returns:
        1D NumPy array (m,)
    """
    # Your code here
    pass

# Test
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
x = np.array([1, 2], dtype=np.float32)
result = matrix_vector_product(A, x)
# Expected: [5, 11, 17]
```

### Exercise 2: Batch Normalization

Implement batch normalization using Flow:

```python
def batch_normalize(data, epsilon=1e-5):
    """
    Normalize data to zero mean and unit variance.

    Args:
        data: NumPy array

    Returns:
        Normalized NumPy array
    """
    # Your code here
    pass
```

### Exercise 3: Weighted Average

Implement a weighted average function:

```python
def weighted_average(values, weights):
    """
    Compute weighted average: sum(values * weights) / sum(weights)
    """
    # Your code here - use Flow for computation
    pass
```

---

## Next Steps

- [Tutorial 5: Performance Optimization](05_performance.md)
- [API Reference](../api/core.md)
- [Notebooks](../notebooks/02_numpy_comparison.ipynb) - NumPy comparison benchmarks

---

## Summary

In this tutorial, you learned:

- Converting between NumPy and Flow arrays with `flow.flow()` and `.to_numpy()`
- Data type handling and conversion
- Copy semantics (data is always copied)
- Common workflows for mixed NumPy/Flow code
- Performance tips for minimizing conversion overhead
- Building data processing pipelines with both libraries
