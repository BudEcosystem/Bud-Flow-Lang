# Tutorial 2: Basic Operations

This tutorial covers all fundamental operations in Bud Flow Lang.

---

## What You'll Learn

1. Element-wise arithmetic operations
2. Scalar operations
3. Unary mathematical functions
4. Reduction operations
5. Comparison operations

---

## Prerequisites

- Completed [Tutorial 1](01_first_program.md)
- Bud Flow Lang installed

---

## Element-wise Arithmetic

All arithmetic operations work element-by-element on arrays of the same size.

### Addition

```python
import bud_flow_lang_py as flow
flow.initialize()

a = flow.arange(5)      # [0, 1, 2, 3, 4]
b = flow.full(5, 10.0)  # [10, 10, 10, 10, 10]

c = a + b  # [10, 11, 12, 13, 14]
print(f"a + b = {c.to_numpy()}")
```

### Subtraction

```python
a = flow.arange(5)      # [0, 1, 2, 3, 4]
b = flow.full(5, 2.0)   # [2, 2, 2, 2, 2]

c = a - b  # [-2, -1, 0, 1, 2]
print(f"a - b = {c.to_numpy()}")
```

### Multiplication

```python
a = flow.arange(5)      # [0, 1, 2, 3, 4]
b = flow.full(5, 3.0)   # [3, 3, 3, 3, 3]

c = a * b  # [0, 3, 6, 9, 12]
print(f"a * b = {c.to_numpy()}")
```

### Division

```python
a = flow.arange(5, 1.0, 1.0)  # [1, 2, 3, 4, 5] - 5 elements starting at 1
b = flow.full(5, 2.0)          # [2, 2, 2, 2, 2]

c = a / b  # [0.5, 1.0, 1.5, 2.0, 2.5]
print(f"a / b = {c.to_numpy()}")
```

---

## Scalar Operations

You can combine arrays with scalar values:

```python
a = flow.arange(5)  # [0, 1, 2, 3, 4]

# Add scalar
b = a + 10      # [10, 11, 12, 13, 14]

# Subtract scalar
c = a - 1       # [-1, 0, 1, 2, 3]

# Multiply by scalar
d = a * 2       # [0, 2, 4, 6, 8]

# Divide by scalar
e = a / 2       # [0, 0.5, 1, 1.5, 2]

print(f"a + 10 = {b.to_numpy()}")
print(f"a * 2 = {d.to_numpy()}")
```

---

## Negation

```python
a = flow.flow([1.0, -2.0, 3.0, -4.0])

b = -a  # [-1.0, 2.0, -3.0, 4.0]
print(f"-a = {b.to_numpy()}")
```

---

## Unary Mathematical Functions

### Square Root

```python
a = flow.flow([1.0, 4.0, 9.0, 16.0, 25.0])

b = flow.sqrt(a)  # [1.0, 2.0, 3.0, 4.0, 5.0]
print(f"sqrt(a) = {b.to_numpy()}")
```

### Reciprocal Square Root

Computes `1/sqrt(x)` efficiently:

```python
a = flow.flow([1.0, 4.0, 9.0, 16.0])

b = flow.rsqrt(a)  # [1.0, 0.5, 0.333..., 0.25]
print(f"rsqrt(a) = {b.to_numpy()}")
```

### Absolute Value

```python
a = flow.flow([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

b = flow.abs(a)  # [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0]
# Also works with Python's abs():
c = abs(a)       # Same result

print(f"abs(a) = {b.to_numpy()}")
```

### Exponential

```python
a = flow.flow([0.0, 1.0, 2.0, 3.0])

b = flow.exp(a)  # [1.0, 2.718..., 7.389..., 20.085...]
print(f"exp(a) = {b.to_numpy()}")
```

### Natural Logarithm

```python
a = flow.flow([1.0, 2.718, 7.389, 20.085])

b = flow.log(a)  # [0.0, ~1.0, ~2.0, ~3.0]
print(f"log(a) = {b.to_numpy()}")
```

### Trigonometric Functions

```python
import math

# Create angles from 0 to 2*pi
x = flow.linspace(0, 2 * math.pi, 9)

# Sine
sin_x = flow.sin(x)
print(f"sin(x) = {sin_x.to_numpy()}")

# Cosine
cos_x = flow.cos(x)
print(f"cos(x) = {cos_x.to_numpy()}")
```

### Hyperbolic Tangent

```python
x = flow.linspace(-3, 3, 7)

tanh_x = flow.tanh(x)  # Values between -1 and 1
print(f"tanh(x) = {tanh_x.to_numpy()}")
```

---

## Reduction Operations

Reductions aggregate all elements of an array into a single value.

### Sum

```python
a = flow.arange(10)  # [0, 1, 2, ..., 9]

total = a.sum()  # 0+1+2+...+9 = 45
print(f"sum = {total}")

# Function form
total = flow.sum(a)  # Same result
```

### Minimum

```python
a = flow.flow([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

minimum = a.min()  # 1.0
print(f"min = {minimum}")
```

### Maximum

```python
a = flow.flow([5.0, 2.0, 8.0, 1.0, 9.0, 3.0])

maximum = a.max()  # 9.0
print(f"max = {maximum}")
```

### Mean (Average)

```python
a = flow.arange(10)  # [0, 1, 2, ..., 9]

average = a.mean()  # 4.5
print(f"mean = {average}")
```

---

## Dot Product

The dot product (inner product) multiplies corresponding elements and sums the results:

```python
a = flow.flow([1.0, 2.0, 3.0])
b = flow.flow([4.0, 5.0, 6.0])

# dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot = flow.dot(a, b)
print(f"dot product = {dot}")

# Method form
dot = a.dot(b)  # Same result
```

### Dot Product for Vector Length

```python
v = flow.flow([3.0, 4.0])

# Length = sqrt(v . v) = sqrt(9 + 16) = 5
length = flow.dot(v, v) ** 0.5
print(f"vector length = {length}")
```

---

## Fused Multiply-Add (FMA)

FMA computes `a * b + c` in a single operation:

```python
a = flow.ones(100)
b = flow.full(100, 2.0)
c = flow.full(100, 3.0)

# FMA: 1 * 2 + 3 = 5
result = flow.fma(a, b, c)
print(f"FMA result: {result.to_numpy()[:5]}")  # [5, 5, 5, 5, 5]
```

**Why use FMA?**

1. **Faster**: Single memory pass instead of two
2. **More accurate**: Only one rounding operation
3. **Hardware optimized**: Uses CPU FMA instructions

### FMA vs Separate Operations

```python
# Separate operations (2 passes over data)
result1 = a * b + c

# FMA (1 pass over data)
result2 = flow.fma(a, b, c)

# Both produce the same values, but FMA is faster
```

---

## Clamping Values

Restrict values to a range:

```python
a = flow.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Clamp to range [2, 7]
b = flow.clamp(a, 2.0, 7.0)
# Result: [2, 2, 2, 3, 4, 5, 6, 7, 7, 7]

print(f"clamped = {b.to_numpy()}")
```

---

## Linear Interpolation (Lerp)

Smoothly blend between two arrays:

```python
a = flow.zeros(5)     # [0, 0, 0, 0, 0]
b = flow.full(5, 10)  # [10, 10, 10, 10, 10]

# t=0: all a, t=1: all b, t=0.5: midpoint
result = flow.lerp(a, b, 0.5)  # [5, 5, 5, 5, 5]
print(f"lerp at 0.5 = {result.to_numpy()}")

# Different interpolation factors
print(f"lerp at 0.0 = {flow.lerp(a, b, 0.0).to_numpy()}")  # [0, 0, 0, 0, 0]
print(f"lerp at 0.25 = {flow.lerp(a, b, 0.25).to_numpy()}")  # [2.5, 2.5, ...]
print(f"lerp at 1.0 = {flow.lerp(a, b, 1.0).to_numpy()}")  # [10, 10, ...]
```

---

## Chained Operations

Operations can be chained for complex expressions:

```python
x = flow.linspace(-1, 1, 11)

# Polynomial: 2x^2 + 3x + 1
poly = x * x * 2 + x * 3 + 1

print("x values:", x.to_numpy())
print("2x^2 + 3x + 1:", poly.to_numpy())
```

### Complex Expression Example

```python
a = flow.arange(100)
b = flow.full(100, 2.0)
c = flow.full(100, 3.0)

# Complex calculation: sqrt(a^2 + b^2) * c
result = flow.sqrt(a * a + b * b) * c

print(f"First 5 results: {result.to_numpy()[:5]}")
```

---

## Complete Example: Statistics

```python
import bud_flow_lang_py as flow

def compute_statistics(arr):
    """Compute various statistics for an array."""
    return {
        'sum': arr.sum(),
        'min': arr.min(),
        'max': arr.max(),
        'mean': arr.mean(),
        'count': arr.size,
        'range': arr.max() - arr.min()
    }

def main():
    flow.initialize()

    # Create test data
    data = flow.linspace(0, 100, 1001)  # 0.0, 0.1, 0.2, ..., 100.0

    stats = compute_statistics(data)

    print("=== Array Statistics ===")
    print(f"Count: {stats['count']}")
    print(f"Sum:   {stats['sum']:.2f}")
    print(f"Mean:  {stats['mean']:.2f}")
    print(f"Min:   {stats['min']:.2f}")
    print(f"Max:   {stats['max']:.2f}")
    print(f"Range: {stats['range']:.2f}")

if __name__ == "__main__":
    main()
```

**Output:**
```
=== Array Statistics ===
Count: 1001
Sum:   50050.00
Mean:  50.00
Min:   0.00
Max:   100.00
Range: 100.00
```

---

## Complete Example: Signal Processing

```python
import bud_flow_lang_py as flow
import math

def main():
    flow.initialize()

    # Generate a noisy sine wave
    n = 1000
    t = flow.linspace(0, 4 * math.pi, n)

    # Pure sine wave
    signal = flow.sin(t)

    # Add "noise" (just another sine with different frequency)
    noise = flow.sin(t * 10) * 0.1

    # Noisy signal
    noisy = signal + noise

    # Simple smoothing (moving average approximation)
    # For actual moving average, you'd need more operations
    smoothed = noisy * 0.9  # Simple scaling as example

    # Statistics
    print("=== Signal Statistics ===")
    print(f"Signal mean: {signal.mean():.6f}")
    print(f"Noisy mean:  {noisy.mean():.6f}")
    print(f"Signal min/max: {signal.min():.4f} / {signal.max():.4f}")
    print(f"Noisy min/max:  {noisy.min():.4f} / {noisy.max():.4f}")

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Normalize a Vector

Write code to normalize a vector to unit length:

```python
def normalize(v):
    # Your code here
    # Hint: Use dot product to compute length, then divide
    pass

v = flow.flow([3.0, 4.0])
v_unit = normalize(v)
# Should be [0.6, 0.8] with length 1.0
```

<details>
<summary>Solution</summary>

```python
def normalize(v):
    length = flow.dot(v, v) ** 0.5
    return v * (1.0 / length)

v = flow.flow([3.0, 4.0])
v_unit = normalize(v)
print(f"Normalized: {v_unit.to_numpy()}")
print(f"Length: {flow.dot(v_unit, v_unit) ** 0.5}")
```

</details>

### Exercise 2: Softmax Function

Implement a basic softmax function:

```python
def softmax(x):
    # Your code here
    # softmax(x) = exp(x) / sum(exp(x))
    pass

x = flow.flow([1.0, 2.0, 3.0])
probs = softmax(x)
# Probabilities should sum to 1.0
```

<details>
<summary>Solution</summary>

```python
def softmax(x):
    exp_x = flow.exp(x)
    return exp_x * (1.0 / exp_x.sum())

x = flow.flow([1.0, 2.0, 3.0])
probs = softmax(x)
print(f"Softmax: {probs.to_numpy()}")
print(f"Sum: {probs.sum()}")  # Should be ~1.0
```

</details>

### Exercise 3: Polynomial Evaluation

Evaluate the polynomial `x^3 - 2x^2 + x - 1` for values -2 to 2:

```python
x = flow.linspace(-2, 2, 9)
# Your code here
```

<details>
<summary>Solution</summary>

```python
x = flow.linspace(-2, 2, 9)
result = x * x * x - x * x * 2 + x - 1
print("x:", x.to_numpy())
print("x^3 - 2x^2 + x - 1:", result.to_numpy())
```

</details>

---

## Next Steps

- [Tutorial 3: Memory Optimization](03_memory_optimization.md) - Cache-aware computing
- [Tutorial 4: NumPy Interoperability](04_numpy_interop.md) - Working with NumPy
- [API Reference: Operations](../api/operations.md) - Complete operations list

---

## Summary

In this tutorial, you learned:

- Element-wise arithmetic: `+`, `-`, `*`, `/`
- Scalar operations with arrays
- Unary functions: `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `abs`, `rsqrt`
- Reductions: `sum`, `min`, `max`, `mean`
- Dot product and FMA
- Utility functions: `clamp`, `lerp`
- Chaining operations for complex expressions
