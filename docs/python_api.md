# Bud Flow Lang - Python API Reference

A high-performance SIMD-accelerated array computation library with JIT compilation.

## Installation

```python
import bud_flow_lang_py as flow
```

## Core Types

### `Bunch`

The primary array type, representing a 1D array of scalar values with SIMD-aligned storage.

```python
# Create using factory functions (preferred)
x = flow.arange(10)       # [0.0, 1.0, 2.0, ..., 9.0]
x = flow.ones(10)         # [1.0, 1.0, 1.0, ..., 1.0]
x = flow.zeros(10)        # [0.0, 0.0, 0.0, ..., 0.0]
x = flow.full(10, 3.14)   # [3.14, 3.14, 3.14, ..., 3.14]

# Properties
x.size       # Number of elements (property, not method)
x.dtype      # Scalar type (e.g., "float32")

# Conversion
x.to_numpy() # Convert to numpy array
list(x.to_numpy())  # Convert to Python list
```

### Scalar Types

```python
flow.ScalarType.kFloat32   # 32-bit float (default)
flow.ScalarType.kFloat64   # 64-bit float (double)
flow.ScalarType.kInt32     # 32-bit signed integer
flow.ScalarType.kInt64     # 64-bit signed integer
flow.ScalarType.kUInt32    # 32-bit unsigned integer
flow.ScalarType.kUInt64    # 64-bit unsigned integer
```

## Array Creation Functions

### `zeros(n, dtype=kFloat32)`
Create an array filled with zeros.

```python
x = flow.zeros(100)                           # 100 float32 zeros
x = flow.zeros(100, flow.ScalarType.kFloat64) # 100 float64 zeros
```

### `ones(n, dtype=kFloat32)`
Create an array filled with ones.

```python
x = flow.ones(100)  # [1.0, 1.0, 1.0, ...]
```

### `full(n, value, dtype=kFloat32)`
Create an array filled with a constant value.

```python
x = flow.full(100, 3.14)  # [3.14, 3.14, 3.14, ...]
```

### `arange(n, dtype=kFloat32)`
Create an array with values [0, 1, 2, ..., n-1].

```python
x = flow.arange(10)  # [0.0, 1.0, 2.0, ..., 9.0]
```

### `linspace(start, stop, n, dtype=kFloat32)`
Create an array with n evenly spaced values from start to stop.

```python
x = flow.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
```

### `random_uniform(n, low=0.0, high=1.0, dtype=kFloat32)`
Create an array with random values uniformly distributed in [low, high).

```python
x = flow.random_uniform(100)           # Random values in [0, 1)
x = flow.random_uniform(100, -1.0, 1.0)  # Random values in [-1, 1)
```

## Arithmetic Operations

All operations are SIMD-accelerated using Highway for maximum performance.

### Binary Operations (Operator Overloading - Preferred)

```python
# Use operators for natural syntax
c = a + b    # Element-wise addition
c = a - b    # Element-wise subtraction
c = a * b    # Element-wise multiplication
c = a / b    # Element-wise division
c = -a       # Negation
```

### Unary Math Functions

```python
b = flow.abs(a)         # Absolute value: b = |a|
b = flow.sqrt(a)        # Square root: b = sqrt(a)
b = flow.exp(a)         # Exponential: b = e^a
b = flow.log(a)         # Natural logarithm: b = ln(a)
b = flow.sin(a)         # Sine: b = sin(a)
b = flow.cos(a)         # Cosine: b = cos(a)
b = flow.tanh(a)        # Hyperbolic tangent: b = tanh(a)
```

### Fused Operations

```python
d = flow.fma(a, b, c)   # Fused multiply-add: d = a * b + c
```

## Reduction Operations

```python
s = flow.sum(a)         # Sum of all elements
m = flow.max(a)         # Maximum element
m = flow.min(a)         # Minimum element
v = flow.mean(a)        # Mean of all elements
d = flow.dot(a, b)      # Dot product: sum(a * b)
```

## Operator Overloading

Bunch objects support Python operators for natural syntax:

```python
# Arithmetic operators
c = a + b    # Addition
c = a - b    # Subtraction
c = a * b    # Multiplication
c = a / b    # Division
c = -a       # Negation

# In-place operators
a += b       # In-place addition
a -= b       # In-place subtraction
a *= b       # In-place multiplication
a /= b       # In-place division

# Comparison (element-wise, returns Bunch)
c = a == b
c = a != b
c = a < b
c = a <= b
c = a > b
c = a >= b
```

## Slicing

```python
x = flow.arange(10)

# Slice access (returns new Bunch)
y = x[2:5]      # Elements 2, 3, 4
y = x[::2]      # Every other element
y = x[::-1]     # Reversed
y = x[3:]       # From index 3 to end
y = x[:-2]      # All except last 2

# Single element access
val = float(x[0])  # First element as Python float
```

## Kernel Tracing with `@flow.kernel`

The `@flow.kernel` decorator traces Python functions and compiles them to optimized SIMD code.

```python
@flow.kernel
def add_arrays(x, y):
    return x + y

# First call traces and compiles
result = add_arrays(x, y)

# Subsequent calls use compiled kernel
result = add_arrays(x, y)  # Fast!
```

**Note:** All inputs must be Bunch arrays. Constants and array creation functions
(like `flow.ones()`) cannot be called inside traced kernels - pass them as parameters instead.

```python
# Correct: Pass ones as a parameter
@flow.kernel
def add_one(x, ones):
    return x + ones

x = flow.arange(10)
ones = flow.ones(10)
result = add_one(x, ones)
```

### Options

```python
@flow.kernel(
    opt_level=2,        # Optimization level (0-3, default 2)
    enable_fusion=True  # Enable operator fusion (default True)
)
def my_kernel(a, b):
    return a * b + a
```

### Optimization Levels

- **Level 0**: No optimization (interpreter only)
- **Level 1**: Basic optimizations (constant folding, DCE)
- **Level 2**: Standard optimizations + fusion (default)
- **Level 3**: Aggressive optimizations

### Supported Operations in Kernels

- Arithmetic: `+`, `-`, `*`, `/`
- Unary: `-x`, `abs()`, `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `tanh()`
- Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Reductions: `sum()`, `max()`, `min()`

## IR Builder API

For advanced users, the IR builder provides low-level access to the compilation pipeline.

```python
# Create IR builder
builder = flow.IRBuilder()

# Define inputs
a = builder.input("a", 100, flow.ScalarType.kFloat32)
b = builder.input("b", 100, flow.ScalarType.kFloat32)

# Build computation graph
c = builder.add(a, b)
d = builder.mul(c, a)
e = builder.sqrt(d)

# Set output
builder.output(e)

# Compile and execute
module = builder.build()
result = module.execute(input_a, input_b)
```

## Performance Tips

1. **Use aligned arrays**: Bunch automatically uses SIMD-aligned memory.

2. **Batch operations**: Larger arrays = better SIMD utilization.
   ```python
   # Good: Single operation on large array
   result = flow.add(big_a, big_b)

   # Avoid: Many operations on small arrays
   for i in range(1000):
       result = flow.add(small_a[i], small_b[i])
   ```

3. **Use in-place operations**: When possible, use `+=`, `-=`, etc.

4. **Enable fusion**: Keep `enable_fusion=True` for automatic kernel fusion.

5. **Reuse kernels**: Traced kernels are cached - same code runs fast on repeat calls.

## Thread Safety

- Bunch objects are **not thread-safe** for mutation
- Kernels are thread-safe after compilation
- The JIT compiler uses internal locking

## Memory Management

- SIMD-aligned allocation (64-byte alignment for AVX-512)
- Arena allocator for IR nodes during compilation
- Automatic cleanup via RAII

## Error Handling

Errors are raised as Python exceptions:

```python
try:
    result = flow.div(a, b)  # May have division by zero
except RuntimeError as e:
    print(f"Operation failed: {e}")
```

## Version Information

```python
print(flow.__version__)  # Library version
```
