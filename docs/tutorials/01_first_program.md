# Tutorial 1: Your First Bud Flow Lang Program

Welcome! In this tutorial, you'll write your first program using Bud Flow Lang and learn the fundamentals of high-performance array computing.

---

## What You'll Learn

1. How to initialize the runtime
2. Creating arrays
3. Performing basic operations
4. Understanding SIMD acceleration

---

## Prerequisites

- Bud Flow Lang installed ([Installation Guide](../installation.md))
- Basic Python knowledge
- NumPy (optional, for comparison)

---

## Step 1: Import and Initialize

Every Bud Flow Lang program starts by importing the module and initializing the runtime:

```python
# Import the module
import bud_flow_lang_py as flow

# Initialize the SIMD runtime
flow.initialize()

# Verify initialization
print(f"Initialized: {flow.is_initialized()}")
```

**Why initialize?**

The `initialize()` function:
- Detects your CPU's SIMD capabilities
- Sets up the JIT compilation system
- Configures memory optimization (tiling, prefetching)

---

## Step 2: Check Your Hardware

Before computing, let's see what hardware acceleration is available:

```python
# Get hardware information
info = flow.get_hardware_info()

print("=== Your Hardware ===")
print(f"Architecture: {info['arch_family']}")
print(f"SIMD Width: {info['simd_width']} bytes ({info['simd_width']*8} bits)")
print(f"CPU Cores: {info['physical_cores']}")

# Check SIMD features
if info['has_avx2']:
    print("AVX2: Enabled (256-bit vectors)")
if info['has_avx512']:
    print("AVX-512: Enabled (512-bit vectors)")
if info['has_neon']:
    print("NEON: Enabled (ARM SIMD)")
```

**Sample Output:**
```
=== Your Hardware ===
Architecture: x86
SIMD Width: 32 bytes (256 bits)
CPU Cores: 8
AVX2: Enabled (256-bit vectors)
```

---

## Step 3: Create Your First Array

Now let's create some arrays:

```python
# Create an array of ones
a = flow.ones(10)
print(f"Array a: {a.to_numpy()}")

# Create an array of zeros
b = flow.zeros(10)
print(f"Array b: {b.to_numpy()}")

# Create a range
c = flow.arange(10)
print(f"Array c: {c.to_numpy()}")

# Create with a specific value
d = flow.full(10, 3.14)
print(f"Array d: {d.to_numpy()}")
```

**Output:**
```
Array a: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
Array b: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Array c: [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
Array d: [3.14 3.14 3.14 3.14 3.14 3.14 3.14 3.14 3.14 3.14]
```

---

## Step 4: Perform Operations

All operations are SIMD-accelerated:

```python
# Create test arrays
x = flow.arange(5)      # [0, 1, 2, 3, 4]
y = flow.full(5, 2.0)   # [2, 2, 2, 2, 2]

print(f"x = {x.to_numpy()}")
print(f"y = {y.to_numpy()}")
print()

# Arithmetic
print(f"x + y = {(x + y).to_numpy()}")
print(f"x - y = {(x - y).to_numpy()}")
print(f"x * y = {(x * y).to_numpy()}")
print(f"x / y = {(x / y).to_numpy()}")
```

**Output:**
```
x = [0. 1. 2. 3. 4.]
y = [2. 2. 2. 2. 2.]

x + y = [2. 3. 4. 5. 6.]
x - y = [-2. -1.  0.  1.  2.]
x * y = [0. 2. 4. 6. 8.]
x / y = [0.  0.5 1.  1.5 2. ]
```

---

## Step 5: Reductions

Compute aggregate values:

```python
a = flow.arange(10)  # [0, 1, 2, ..., 9]

print(f"Array: {a.to_numpy()}")
print(f"Sum:   {a.sum()}")    # 0+1+2+...+9 = 45
print(f"Min:   {a.min()}")    # 0
print(f"Max:   {a.max()}")    # 9
print(f"Mean:  {a.mean()}")   # 4.5
```

---

## Step 6: Dot Product

Compute the inner product of two vectors:

```python
a = flow.ones(100)
b = flow.full(100, 2.0)

# The dot product of [1,1,...] and [2,2,...] is 100 * 1 * 2 = 200
dot = flow.dot(a, b)
print(f"Dot product: {dot}")  # 200.0

# Alternative: method form
dot = a.dot(b)
print(f"Same result: {dot}")  # 200.0
```

---

## Step 7: Fused Multiply-Add (FMA)

FMA is a special operation that computes `a*b + c` in a single step:

```python
a = flow.ones(100)
b = flow.full(100, 2.0)
c = flow.full(100, 3.0)

# FMA: a * b + c = 1 * 2 + 3 = 5
result = flow.fma(a, b, c)
print(f"FMA result (first 5): {result.to_numpy()[:5]}")  # [5, 5, 5, 5, 5]
```

**Why use FMA?**

1. **Speed**: One memory pass instead of two
2. **Accuracy**: Single rounding instead of two
3. **Efficiency**: Uses hardware FMA instructions

---

## Step 8: Complete Example

Here's a complete program that puts it all together:

```python
#!/usr/bin/env python3
"""
My First Bud Flow Lang Program

This program demonstrates basic array operations with SIMD acceleration.
"""

import bud_flow_lang_py as flow

def main():
    # Initialize
    flow.initialize()

    # Print hardware info
    info = flow.get_hardware_info()
    print(f"Running on {info['arch_family']} with {info['simd_width']*8}-bit SIMD")
    print()

    # Create vectors
    n = 1000
    a = flow.ones(n)
    b = flow.full(n, 2.0)
    c = flow.full(n, 3.0)

    # Demonstrate operations
    print("=== Operations Demo ===")
    print(f"a = ones({n})")
    print(f"b = full({n}, 2.0)")
    print(f"c = full({n}, 3.0)")
    print()

    # Arithmetic
    print(f"sum(a + b) = {(a + b).sum()}")      # 3000
    print(f"sum(a * b) = {(a * b).sum()}")      # 2000

    # Dot product
    print(f"dot(a, b) = {flow.dot(a, b)}")      # 2000

    # FMA
    fma_result = flow.fma(a, b, c)
    print(f"sum(fma(a, b, c)) = {fma_result.sum()}")  # 5000

    print()
    print("Success! Your first Flow program is complete.")

if __name__ == "__main__":
    main()
```

Save this as `first_program.py` and run it:

```bash
python first_program.py
```

---

## Understanding the Output

When you run the program, you'll see:

```
Running on x86 with 256-bit SIMD

=== Operations Demo ===
a = ones(1000)
b = full(1000, 2.0)
c = full(1000, 3.0)

sum(a + b) = 3000.0
sum(a * b) = 2000.0
dot(a, b) = 2000.0
sum(fma(a, b, c)) = 5000.0

Success! Your first Flow program is complete.
```

---

## What's Happening Under the Hood?

When you run operations on Bud Flow Lang arrays:

1. **SIMD Vectorization**: Operations process multiple elements at once
   - AVX2: 8 float32s per instruction
   - AVX-512: 16 float32s per instruction

2. **Cache Optimization**: Large arrays are processed in tiles that fit in L1 cache

3. **Hardware FMA**: The `flow.fma()` function uses dedicated FMA instructions

---

## Exercises

Try these exercises to practice:

### Exercise 1: Vector Sum
Create two arrays of 10,000 elements and compute their element-wise sum:

```python
# Your code here
a = flow.arange(10000)
b = flow.ones(10000)
c = a + b
print(f"First 5 elements: {c.to_numpy()[:5]}")
```

### Exercise 2: Compute Mean
Calculate the mean of the numbers 1 to 1000:

```python
# Your code here
# Use linspace for start-to-stop range
a = flow.linspace(1, 1000, 1000)
print(f"Mean of 1..1000: {a.mean()}")  # Should be 500.5
```

### Exercise 3: Dot Product
Compute the dot product of [1, 2, 3, 4, 5] with itself:

```python
# Your code here
a = flow.flow([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Dot product: {flow.dot(a, a)}")  # 1+4+9+16+25 = 55
```

---

## Next Steps

- [Tutorial 2: Basic Operations](02_basic_operations.md) - Deep dive into operations
- [Tutorial 3: Memory Optimization](03_memory_optimization.md) - Cache-aware computing
- [API Reference](../api/core.md) - Complete function documentation

---

## Summary

In this tutorial, you learned:

- ✅ How to initialize Bud Flow Lang
- ✅ Creating arrays with `ones()`, `zeros()`, `arange()`, `full()`
- ✅ Basic arithmetic operations
- ✅ Reductions: `sum()`, `min()`, `max()`, `mean()`
- ✅ Dot product
- ✅ Fused Multiply-Add (FMA)

Congratulations on completing your first Bud Flow Lang program!
