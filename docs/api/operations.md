# Operations API Reference

This document covers arithmetic, mathematical, and reduction operations.

---

## Binary Operations

All binary operations are element-wise and SIMD-accelerated.

### Arithmetic Operators

```python
a = flow.arange(5)     # [0, 1, 2, 3, 4]
b = flow.full(5, 2.0)  # [2, 2, 2, 2, 2]

c = a + b    # Addition:       [2, 3, 4, 5, 6]
c = a - b    # Subtraction:    [-2, -1, 0, 1, 2]
c = a * b    # Multiplication: [0, 2, 4, 6, 8]
c = a / b    # Division:       [0, 0.5, 1, 1.5, 2]
```

### Scalar Operations

```python
a = flow.arange(5)

c = a + 10     # Add scalar:      [10, 11, 12, 13, 14]
c = a - 1      # Subtract scalar: [-1, 0, 1, 2, 3]
c = a * 2      # Multiply scalar: [0, 2, 4, 6, 8]
c = a / 2      # Divide scalar:   [0, 0.5, 1, 1.5, 2]
```

---

## Unary Operations

### `flow.sqrt(x)`

Element-wise square root.

```python
x = flow.flow([1.0, 4.0, 9.0, 16.0])
y = flow.sqrt(x)  # [1.0, 2.0, 3.0, 4.0]
```

---

### `flow.abs(x)` / `abs(x)`

Element-wise absolute value.

```python
x = flow.flow([-1.0, -2.0, 3.0])
y = flow.abs(x)  # [1.0, 2.0, 3.0]
y = abs(x)       # Same result
```

---

### `flow.exp(x)`

Element-wise exponential (e^x).

```python
x = flow.flow([0.0, 1.0, 2.0])
y = flow.exp(x)  # [1.0, 2.718..., 7.389...]
```

---

### `flow.log(x)`

Element-wise natural logarithm.

```python
x = flow.flow([1.0, 2.718, 10.0])
y = flow.log(x)  # [0.0, ~1.0, ~2.3]
```

---

### `flow.sin(x)`

Element-wise sine.

```python
x = flow.linspace(0, 3.14159, 5)
y = flow.sin(x)
```

---

### `flow.cos(x)`

Element-wise cosine.

```python
x = flow.linspace(0, 3.14159, 5)
y = flow.cos(x)
```

---

### `flow.tanh(x)`

Element-wise hyperbolic tangent.

```python
x = flow.linspace(-2, 2, 5)
y = flow.tanh(x)
```

---

### `flow.rsqrt(x)`

Element-wise reciprocal square root (1/sqrt(x)).

```python
x = flow.flow([1.0, 4.0, 9.0])
y = flow.rsqrt(x)  # [1.0, 0.5, 0.333...]
```

---

### Negation

```python
x = flow.flow([1.0, -2.0, 3.0])
y = -x  # [-1.0, 2.0, -3.0]
```

---

## Reduction Operations

Reductions aggregate array values to a single scalar.

### `bunch.sum()` / `flow.sum(x)`

Sum of all elements.

```python
a = flow.arange(10)
total = a.sum()       # 45.0
total = flow.sum(a)   # Same result
```

---

### `bunch.min()` / `flow.min(x)`

Minimum element.

```python
a = flow.flow([3.0, 1.0, 4.0, 1.0, 5.0])
minimum = a.min()     # 1.0
minimum = flow.min(a) # Same result
```

---

### `bunch.max()` / `flow.max(x)`

Maximum element.

```python
a = flow.flow([3.0, 1.0, 4.0, 1.0, 5.0])
maximum = a.max()     # 5.0
maximum = flow.max(a) # Same result
```

---

### `bunch.mean()` / `flow.mean(x)`

Arithmetic mean.

```python
a = flow.arange(10)
avg = a.mean()        # 4.5
avg = flow.mean(a)    # Same result
```

---

## Dot Product

### `flow.dot(a, b)` / `a.dot(b)`

Compute the dot product (inner product) of two vectors.

```python
a = flow.ones(1000)
b = flow.full(1000, 2.0)

# Function form
dot = flow.dot(a, b)  # 2000.0

# Method form
dot = a.dot(b)        # 2000.0
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | Bunch | First vector |
| `b` | Bunch | Second vector (same size as a) |

**Returns:** `float`

**Notes:**
- Uses SIMD-accelerated implementation
- For large arrays (>1024 elements), uses cache-aware tiling

---

## Fused Operations

Fused operations combine multiple operations into a single pass, improving performance.

### `flow.fma(a, b, c)`

Fused Multiply-Add: computes `a * b + c` in a single operation.

```python
a = flow.ones(1000)
b = flow.full(1000, 2.0)
c = flow.full(1000, 3.0)

result = flow.fma(a, b, c)  # a*b + c = 1*2 + 3 = 5
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | Bunch | First multiplicand |
| `b` | Bunch | Second multiplicand |
| `c` | Bunch | Addend |

**Returns:** `Bunch`

**Advantages:**
- **Faster**: Single pass over memory
- **More accurate**: Single rounding instead of two
- Uses hardware FMA instructions when available

---

### `flow.clamp(x, lo, hi)`

Clamp values to a range.

```python
x = flow.arange(10)
y = flow.clamp(x, 2.0, 7.0)  # [2, 2, 2, 3, 4, 5, 6, 7, 7, 7]
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `x` | Bunch | Input array |
| `lo` | float | Minimum value |
| `hi` | float | Maximum value |

**Returns:** `Bunch`

---

### `flow.lerp(a, b, t)`

Linear interpolation between two arrays.

```python
a = flow.zeros(5)
b = flow.ones(5)
t = 0.5

result = flow.lerp(a, b, t)  # [0.5, 0.5, 0.5, 0.5, 0.5]
# Equivalent to: a + t * (b - a)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | Bunch | Start values |
| `b` | Bunch | End values |
| `t` | float | Interpolation factor (0=a, 1=b) |

**Returns:** `Bunch`

---

## Chained Operations

Operations can be chained for complex computations:

```python
a = flow.arange(1000)
b = flow.full(1000, 2.0)
c = flow.full(1000, 3.0)

# Complex expression
result = (a + b) * (a - c)

# Polynomial: ax^2 + bx + c
x = flow.linspace(-1, 1, 100)
poly = a * x * x + b * x + c
```

---

## Performance Tips

### 1. Use FMA When Possible

```python
# Slower: Two separate operations
result = a * b + c

# Faster: Single fused operation
result = flow.fma(a, b, c)
```

### 2. Minimize Temporary Arrays

```python
# Creates 3 temporaries
result = a + b + c + d

# Better: Use accumulation
result = a + b
result = result + c
result = result + d
```

### 3. Batch Small Operations

```python
# Slower: Many small arrays
for i in range(1000):
    small = flow.ones(100)
    total += small.sum()

# Faster: One large array
large = flow.ones(100000)
total = large.sum()
```

---

## Examples

### Vector Normalization

```python
def normalize(v):
    """Normalize a vector to unit length."""
    norm = flow.dot(v, v) ** 0.5
    return v * (1.0 / norm)

v = flow.flow([3.0, 4.0])
v_unit = normalize(v)  # [0.6, 0.8]
```

### Polynomial Evaluation

```python
def evaluate_poly(x, coeffs):
    """Evaluate polynomial using Horner's method."""
    # coeffs = [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
    n = x.size
    result = flow.full(n, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = flow.fma(result, x, flow.full(n, c))
    return result

x = flow.linspace(-1, 1, 100)
# Evaluate: 1 + 2x + 3x^2
poly = evaluate_poly(x, [1.0, 2.0, 3.0])
```

### Distance Calculation

```python
def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    diff = a - b
    return flow.dot(diff, diff) ** 0.5

a = flow.flow([0.0, 0.0])
b = flow.flow([3.0, 4.0])
dist = euclidean_distance(a, b)  # 5.0
```

---

## See Also

- [Array Creation](creation.md) - Creating arrays
- [Core API](core.md) - Initialization and hardware info
- [Performance Guide](../tutorials/05_performance.md) - Optimization tips
