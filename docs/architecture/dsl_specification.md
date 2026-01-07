# FlowSIMD: A First-Principles DSL for Accessible SIMD Programming
## Version 3.0 - Complete Redesign

---

# Table of Contents

1. [Philosophy and First Principles](#part-1-philosophy-and-first-principles)
2. [The Mental Model Bridge](#part-2-the-mental-model-bridge)
3. [Core Taxonomy](#part-3-core-taxonomy)
4. [The Five Fundamental Concepts](#part-4-the-five-fundamental-concepts)
5. [Three-Tier Architecture](#part-5-three-tier-architecture)
6. [Program Structure](#part-6-program-structure)
7. [Beginner Tier: Natural Parallelism](#part-7-beginner-tier-natural-parallelism)
8. [Developer Tier: Guided Performance](#part-8-developer-tier-guided-performance)
9. [Expert Tier: Full Highway Access](#part-9-expert-tier-full-highway-access)
10. [Cross-Tier Interoperability](#part-10-cross-tier-interoperability)
11. [JIT Architecture](#part-11-jit-architecture)
12. [Complete API Reference](#part-12-complete-api-reference)

---

# Part 1: Philosophy and First Principles

## 1.1 The Fundamental Problem

SIMD programming fails for most developers because it requires thinking like hardware:

```
Traditional SIMD Mental Model:
┌─────────────────────────────────────────────────────────────┐
│  "I have a 256-bit register with 8 lanes of 32-bit floats" │
│  "I need to create a mask where bits 0,2,4,6 are set"      │
│  "I'll use _mm256_shuffle_ps with control byte 0xB1"       │
└─────────────────────────────────────────────────────────────┘
```

Python developers think differently:

```
Python Developer Mental Model:
┌─────────────────────────────────────────────────────────────┐
│  "I have a collection of numbers"                           │
│  "I want to keep the ones greater than zero"                │
│  "I want to double each one"                                │
└─────────────────────────────────────────────────────────────┘
```

**First Principle #1**: The DSL must speak the developer's language, not the hardware's language.

## 1.2 What Python Developers Already Know

Python developers have powerful mental models we can leverage:

| Python Concept | Mental Model | SIMD Equivalent |
|----------------|--------------|-----------------|
| `[x*2 for x in data]` | "Transform each element" | Parallel map |
| `[x for x in data if x > 0]` | "Keep matching elements" | Mask + compress |
| `sum(data)` | "Combine into one" | Horizontal reduction |
| `zip(a, b)` | "Pair up elements" | Parallel load |
| `reversed(data)` | "Flip the order" | Reverse shuffle |
| `data[::2]` | "Every other one" | Strided access |

**First Principle #2**: Map SIMD operations to existing Python mental models.

## 1.3 The Three User Personas

### Persona 1: The Beginner
- **Background**: Python developer, understands lists/arrays, never heard of SIMD
- **Goal**: Make existing code faster without learning hardware
- **Needs**: Invisible complexity, automatic optimization, familiar syntax
- **Tolerance for complexity**: Near zero

### Persona 2: The Developer
- **Background**: Knows SIMD exists, understands vectorization conceptually
- **Goal**: Write portable, performant code with some control
- **Needs**: Abstractions with escape hatches, performance hints, portability
- **Tolerance for complexity**: Medium

### Persona 3: The Expert
- **Background**: Deep SIMD experience, understands ISAs, wants maximum control
- **Goal**: Access all hardware capabilities, mix with high-level code
- **Needs**: Full Highway API, platform-specific intrinsics, zero overhead
- **Tolerance for complexity**: High

**First Principle #3**: Progressive disclosure—each tier reveals more, but never requires it.

## 1.4 The Key Insight: Batches, Not Vectors

The word "vector" is overloaded and confusing. In math, it's a direction. In Python, it's sometimes a list. In SIMD, it's a register.

We introduce a new concept: **Bunch**

A **Bunch** is:
- A group of values processed together
- Size determined automatically by hardware
- The fundamental unit of parallel work

```python
# Traditional SIMD thinking (confusing):
vector = load_8_floats(data)  # Why 8? What if I have 16?

# FlowSIMD thinking (natural):
bunch = flow(data)  # "Here's my data, process it efficiently"
```

**First Principle #4**: Introduce new, unambiguous concepts rather than overloading existing terms.

## 1.5 Design Constraints from Highway

Our JIT backend is Google Highway. We must respect its constraints:

1. **Length-agnostic**: Never assume vector width at compile time
2. **Block-aware**: 128-bit blocks are the minimum unit on many platforms
3. **Tag-based dispatch**: Operations are selected by type descriptors
4. **Performance portability**: Operations must be efficient on ALL platforms
5. **Mask as first-class**: AVX-512/SVE use dedicated mask registers

These constraints inform our abstractions but must be invisible to beginners.

---

# Part 2: The Mental Model Bridge

## 2.1 Bridging Python to SIMD

We create explicit bridges between Python concepts and SIMD operations:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MENTAL MODEL BRIDGE                         │
├─────────────────────────────────────────────────────────────────┤
│  PYTHON CONCEPT          →      FLOWSIMD CONCEPT                │
├─────────────────────────────────────────────────────────────────┤
│  List/Array              →      Bunch (auto-sized parallel)     │
│  List comprehension      →      .each() / arithmetic ops        │
│  filter()                →      .where() / .keep()              │
│  map()                   →      .each() / .apply()              │
│  reduce()                →      .fold() / .sum() / .max()       │
│  enumerate()             →      .with_index()                   │
│  zip()                   →      .zip() / .pair()                │
│  reversed()              →      .reverse()                      │
│  sorted()                →      .sort()                         │
│  slice [::n]             →      .every(n)                       │
│  any() / all()           →      .any() / .all()                 │
│  if/else                 →      .where(cond, then, else)        │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 The "Invisible Parallelism" Principle

For beginners, parallelism should be invisible:

```python
# What the beginner writes:
@flow.kernel
def double_all(data):
    return flow(data) * 2

# What they think:
# "Multiply everything by 2"

# What actually happens (invisible to them):
# - Data is loaded in SIMD-width chunks
# - Parallel multiply instruction executed
# - Results stored back
# - Edge cases handled automatically
```

## 2.3 The "Familiar Syntax" Principle

Operators work naturally:

```python
@flow.kernel
def math_example(a, b):
    x = flow(a)
    y = flow(b)
    
    # All of these work as expected:
    result = x + y          # Element-wise add
    result = x * 2          # Scalar broadcast
    result = x ** 2         # Square each
    result = x > 0          # Creates a selection pattern
    result = x + y * 2      # Fused multiply-add (automatic!)
    
    return result
```

---

# Part 3: Core Taxonomy

## 3.1 Concept Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      FLOWSIMD TAXONOMY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LEVEL 0: Foundation                                            │
│  ├── Bunch          (parallel data container)                   │
│  ├── Pattern        (selection/mask abstraction)                │
│  └── Kernel         (compiled parallel function)                │
│                                                                 │
│  LEVEL 1: Transformations                                       │
│  ├── Each           (parallel element-wise operation)           │
│  ├── Fold           (parallel reduction)                        │
│  ├── Scan           (parallel prefix operation)                 │
│  └── Shape          (parallel data rearrangement)               │
│                                                                 │
│  LEVEL 2: Patterns & Selections                                 │
│  ├── Where          (conditional selection)                     │
│  ├── Keep/Drop      (filtering)                                 │
│  ├── Every          (strided selection)                         │
│  └── First/Last     (positional selection)                      │
│                                                                 │
│  LEVEL 3: Composition                                           │
│  ├── Pipe           (operation chaining)                        │
│  ├── Fork/Join      (parallel branches)                         │
│  └── Fuse           (operation fusion)                          │
│                                                                 │
│  LEVEL 4: Advanced (Developer/Expert)                           │
│  ├── Hint           (optimization guidance)                     │
│  ├── Vector         (explicit SIMD register)                    │
│  ├── Mask           (explicit predicate register)               │
│  └── Native         (platform-specific intrinsics)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 Type Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                       TYPE TAXONOMY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BUNCH TYPES (automatic width):                                 │
│  ├── Bunch[f32]     floating point 32-bit                       │
│  ├── Bunch[f64]     floating point 64-bit                       │
│  ├── Bunch[i8]      signed integer 8-bit                        │
│  ├── Bunch[i16]     signed integer 16-bit                       │
│  ├── Bunch[i32]     signed integer 32-bit                       │
│  ├── Bunch[i64]     signed integer 64-bit                       │
│  ├── Bunch[u8]      unsigned integer 8-bit                      │
│  ├── Bunch[u16]     unsigned integer 16-bit                     │
│  ├── Bunch[u32]     unsigned integer 32-bit                     │
│  └── Bunch[u64]     unsigned integer 64-bit                     │
│                                                                 │
│  PATTERN TYPES (boolean selections):                            │
│  ├── Pattern        (abstract selection)                        │
│  └── Pattern[T]     (typed selection for Bunch[T])              │
│                                                                 │
│  EXPLICIT TYPES (Developer/Expert only):                        │
│  ├── Vec[T, N]      N-wide vector (fixed size)                  │
│  ├── Mask[T, N]     N-wide mask (fixed size)                    │
│  └── Scalable[T]    Highway-style scalable vector               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3.3 Operation Categories

Following Highway's organization, but with intuitive names:

| FlowSIMD Category | Description | Highway Equivalent |
|-------------------|-------------|-------------------|
| **Arithmetic** | Math operations | Arithmetic ops |
| **Compare** | Comparisons → Patterns | Comparisons |
| **Logic** | Boolean on patterns | Logical ops |
| **Shape** | Rearrange elements | Swizzle ops |
| **Select** | Choose elements | Mask ops |
| **Fold** | Reduce to fewer | Reductions |
| **Memory** | Load/store data | Memory ops |
| **Convert** | Change types | Conversion ops |
| **Crypto** | Cryptographic ops | Crypto ops |

---

# Part 4: The Five Fundamental Concepts

## 4.1 Concept 1: Bunch (The Parallel Container)

A **Bunch** is the core abstraction replacing "vector", "SIMD register", and "array slice".

### Mental Model
Think of a Bunch as "a handful of values that travel together."

```python
import flow

# Creating Bunches
numbers = flow(data)              # From existing array
zeros = flow.zeros(n)             # N zeros
ones = flow.ones(n)               # N ones
sequence = flow.range(0, 100)     # 0 to 99
random = flow.random(n)           # Random values

# Bunch properties (runtime, NOT compile-time)
len(numbers)                      # Total element count
numbers.dtype                     # Element type (f32, i64, etc.)
```

### Key Insight: Size is Runtime, Not Compile-Time

```python
# WRONG mental model:
"I have a vector of 8 floats"

# RIGHT mental model:
"I have some floats that will be processed efficiently"
```

The JIT handles sizing:
- Loops over chunks of SIMD width
- Handles remainders automatically
- Chooses optimal width for hardware

### Highway Mapping

```cpp
// flow(data) maps to:
template<typename D>
void Process(D d, const float* data, size_t count) {
    const size_t N = Lanes(d);
    for (size_t i = 0; i < count; i += N) {
        const auto v = Load(d, data + i);
        // ... operations ...
    }
}
```

## 4.2 Concept 2: Pattern (The Selector)

A **Pattern** replaces "mask", "predicate", and "boolean vector."

### Mental Model
A Pattern is "which elements I care about right now."

```python
# Creating Patterns
positives = numbers > 0           # Elements greater than zero
evens = Pattern.every(2)          # Every other element (0, 2, 4, ...)
first_ten = Pattern.first(10)     # First 10 elements
last_five = Pattern.last(5)       # Last 5 elements

# Combining Patterns (like set operations)
both = pattern1 & pattern2        # Intersection (and)
either = pattern1 | pattern2      # Union (or)
inverse = ~pattern                # Complement (not)
exclusive = pattern1 ^ pattern2   # Exclusive or
```

### Using Patterns

```python
# Conditional operations
result = numbers.where(numbers > 0, else_=0)  # Keep positives, else zero

# Filtering
positives = numbers.keep(numbers > 0)         # Only positive values
cleaned = numbers.drop(numbers == 0)          # Remove zeros

# Selective operations
numbers.add_where(pattern, 10)                # Add 10 only where pattern is true
```

### Highway Mapping

```cpp
// numbers > 0 maps to:
Mask<D> pattern = Gt(d, numbers, Zero(d));

// numbers.where(cond, else_=0) maps to:
Vec<D> result = IfThenElse(pattern, numbers, Zero(d));

// Pattern.first(10) maps to:
Mask<D> pattern = FirstN(d, 10);
```

## 4.3 Concept 3: Each (The Transformer)

**Each** is the explicit parallel map operation, though most operations are implicitly "each."

### Mental Model
"Do this to every element, all at once."

```python
# Implicit each (preferred for simple ops):
doubled = numbers * 2             # Each element * 2
squared = numbers ** 2            # Each element squared

# Explicit each (for complex transforms):
result = numbers.each(lambda x: x * 2 + 1)

# Multi-input each:
result = flow.each(a, b, lambda x, y: x * y + 1)
```

### Built-in Each Operations

```python
# Arithmetic (all element-wise)
a + b, a - b, a * b, a / b       # Basic math
a // b                           # Integer division
a % b                            # Modulo
a ** n                           # Power
-a                               # Negate
abs(a)                           # Absolute value

# Math functions
a.sqrt()                         # Square root
a.exp()                          # e^x
a.log()                          # Natural log
a.sin(), a.cos(), a.tan()        # Trigonometry
a.floor(), a.ceil(), a.round()   # Rounding

# Fused operations (automatic when possible)
a * b + c                        # Fused multiply-add (FMA)
```

### Highway Mapping

```cpp
// numbers * 2 maps to:
Vec<D> result = Mul(numbers, Set(d, 2.0f));

// numbers.sqrt() maps to:
Vec<D> result = Sqrt(numbers);

// a * b + c maps to (automatically!):
Vec<D> result = MulAdd(a, b, c);  // FMA instruction
```

## 4.4 Concept 4: Shape (The Rearranger)

**Shape** operations rearrange elements without changing their values.

### Mental Model
"Change how the elements are arranged."

```python
# Reversing
backwards = numbers.reverse()              # [1,2,3,4] → [4,3,2,1]

# Rotating
rotated = numbers.rotate(2)                # [1,2,3,4] → [3,4,1,2]

# Repeating
repeated = numbers.repeat(3)               # [1,2] → [1,2,1,2,1,2]

# Stretching (duplicate each)
stretched = numbers.stretch(2)             # [1,2,3] → [1,1,2,2,3,3]

# Interleaving
mixed = a.interleave(b)                    # [a0,b0,a1,b1,...]
channels = rgb.deinterleave(3)             # Split R, G, B

# Grouping
groups = numbers.groups(4)                 # View as groups of 4
transposed = groups.transpose()            # Transpose within groups

# Neighbors (shifted views)
prev = numbers.prev()                      # [x,1,2,3] (shifted right)
next = numbers.next()                      # [2,3,4,x] (shifted left)
```

### The Neighbor Pattern (Critical for Stencils)

```python
# 1D stencil operations become trivial:
def blur_1d(data):
    s = flow(data)
    return (s.prev() + s + s.next()) / 3

def gradient(data):
    s = flow(data)
    return s.next() - s.prev()

def laplacian(data):
    s = flow(data)
    return s.prev() - 2 * s + s.next()
```

### Highway Mapping

```cpp
// numbers.reverse() maps to:
Vec<D> result = Reverse(d, numbers);

// numbers.rotate(2) maps to:
Vec<D> result = RotateRight<2>(numbers);

// a.interleave(b) maps to:
Vec<D> lo = InterleaveLower(d, a, b);
Vec<D> hi = InterleaveUpper(d, a, b);

// numbers.prev() maps to:
Vec<D> result = SlideDownLanes(d, numbers, 1);
```

## 4.5 Concept 5: Fold (The Reducer)

**Fold** combines multiple values into fewer values (or one value).

### Mental Model
"Combine all elements into a summary."

```python
# Common folds
total = numbers.sum()                      # Add all
product = numbers.product()                # Multiply all
smallest = numbers.min()                   # Find minimum
largest = numbers.max()                    # Find maximum
average = numbers.mean()                   # Average

# Location folds
min_idx = numbers.argmin()                 # Index of minimum
max_idx = numbers.argmax()                 # Index of maximum

# Boolean folds
has_positive = (numbers > 0).any()         # Any positive?
all_positive = (numbers > 0).all()         # All positive?
count_pos = (numbers > 0).count()          # How many positive?

# Custom fold
custom = numbers.fold(lambda acc, x: acc + x * x, init=0)
```

### Multi-Fold Optimization

```python
# BAD: Two separate passes
total = numbers.sum()    # Pass 1
maximum = numbers.max()  # Pass 2

# GOOD: Single pass with multiple accumulators
total, maximum = numbers.fold_many(sum, max)
```

### Prefix Operations (Scans)

```python
# Cumulative operations
cumsum = numbers.cumsum()                  # Running sum
cumprod = numbers.cumprod()                # Running product
cummax = numbers.cummax()                  # Running maximum

# Custom scan
result = numbers.scan(lambda acc, x: acc + x, init=0)
```

### Highway Mapping

```cpp
// numbers.sum() maps to:
TFromD<D> result = ReduceSum(d, numbers);

// numbers.max() maps to:
TFromD<D> result = MaxOfLanes(d, numbers);

// numbers.fold_many(sum, max) maps to:
// (Single loop with multiple accumulators)
Vec<D> sum_acc = Zero(d);
Vec<D> max_acc = Set(d, -inf);
for (...) {
    auto v = Load(d, ...);
    sum_acc = Add(sum_acc, v);
    max_acc = Max(max_acc, v);
}
result_sum = ReduceSum(d, sum_acc);
result_max = MaxOfLanes(d, max_acc);
```

---

# Part 5: Three-Tier Architecture

## 5.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE-TIER ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  TIER 1: BEGINNER                                         │  │
│  │  • Natural Python syntax                                  │  │
│  │  • Zero hardware knowledge required                       │  │
│  │  • Automatic everything                                   │  │
│  │  • 90% of SIMD use cases                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  TIER 2: DEVELOPER                                        │  │
│  │  • Optimization hints                                     │  │
│  │  • Explicit patterns and shapes                           │  │
│  │  • Memory layout control                                  │  │
│  │  • Platform-aware code                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  TIER 3: EXPERT                                           │  │
│  │  • Full Highway API access                                │  │
│  │  • Explicit vector types                                  │  │
│  │  • Platform intrinsics                                    │  │
│  │  • Manual loop control                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 Tier Characteristics

| Aspect | Beginner | Developer | Expert |
|--------|----------|-----------|--------|
| **Syntax** | Python-native | Python + hints | Python + Highway |
| **Loop control** | Automatic | Hinted | Manual |
| **Vector width** | Hidden | Queryable | Explicit |
| **Memory** | Automatic | Controllable | Manual |
| **Fusion** | Automatic | Guided | Manual |
| **Platform** | Portable | Portable + hints | Platform-specific OK |
| **Performance** | Good (90%) | Better (95%) | Best (100%) |
| **Learning curve** | Minimal | Moderate | Steep |

## 5.3 Tier Transitions

The key design principle: **Higher tiers extend, not replace**.

```python
# Beginner code:
@flow.kernel
def basic(x):
    return flow(x) * 2 + 1

# Same code with Developer hints:
@flow.kernel
def guided(x):
    with flow.hint.unroll(4):
        return flow(x) * 2 + 1

# Same code with Expert control:
@flow.kernel
def expert(x):
    from flow.highway import Vec, ScalableTag
    d = ScalableTag[float]()
    # ... explicit Highway code ...
```

---

# Part 6: Program Structure

## 6.1 Basic Structure

```python
import flow

# Define a kernel (compiled SIMD function)
@flow.kernel
def my_function(input_array):
    data = flow(input_array)      # Create Bunch from array
    result = data * 2 + 1         # Parallel operations
    return result                  # Automatic output handling

# Use the kernel
import numpy as np
data = np.random.randn(10000).astype(np.float32)
output = my_function(data)
```

## 6.2 Kernel Decorator

The `@flow.kernel` decorator:
1. Analyzes the function for SIMD compatibility
2. JIT-compiles to Highway C++ code
3. Caches the compiled kernel
4. Dispatches to best available instruction set at runtime

```python
@flow.kernel                       # Default: auto-target
@flow.kernel(target='avx2')       # Specific target
@flow.kernel(targets=['avx512', 'avx2', 'neon'])  # Multi-target dispatch
@flow.kernel(debug=True)          # Enable debug output
@flow.kernel(profile=True)        # Enable profiling
```

## 6.3 Input/Output Conventions

```python
@flow.kernel
def example(a, b, c):
    # Inputs: NumPy arrays, Python lists, or scalars
    x = flow(a)                   # Array → Bunch
    y = flow(b)                   # Array → Bunch
    scalar = c                    # Scalar used directly
    
    # Operations
    result = x * y + scalar
    
    # Output options:
    return result                 # Auto-allocate output array
    # OR
    result.into(output_array)     # Write to existing array
    # OR
    return result.value()         # Return scalar (for reductions)
```

## 6.4 Multiple Outputs

```python
@flow.kernel
def multi_output(data):
    x = flow(data)
    
    # Method 1: Tuple return
    return x * 2, x + 1
    
    # Method 2: Named return
    return {'doubled': x * 2, 'incremented': x + 1}
    
    # Method 3: Write to preallocated
    (x * 2).into(output1)
    (x + 1).into(output2)
```

## 6.5 In-Place Operations

```python
@flow.kernel
def inplace_example(data):
    x = flow(data)
    
    # Explicit in-place (modifies input)
    x.inplace_mul(2)              # data *= 2
    x.inplace_add(1)              # data += 1
    
    # Or return to same array
    return (x * 2 + 1).into(data)
```

---

# Part 7: Beginner Tier - Natural Parallelism

## 7.1 Design Goal

A Python developer should be productive within 5 minutes, with zero SIMD knowledge.

## 7.2 The Complete Beginner API

### Creating Data

```python
import flow

# From existing arrays
data = flow(numpy_array)          # From NumPy
data = flow([1, 2, 3, 4])         # From Python list

# Factory functions
zeros = flow.zeros(1000)          # 1000 zeros
ones = flow.ones(1000)            # 1000 ones
sequence = flow.range(100)        # 0 to 99
linspace = flow.linspace(0, 1, 100)  # 100 points from 0 to 1
random = flow.random(1000)        # Random [0, 1)
```

### Arithmetic (Element-wise)

```python
@flow.kernel
def arithmetic_demo(a, b):
    x, y = flow(a), flow(b)
    
    # Basic operations
    add = x + y                   # Add
    sub = x - y                   # Subtract
    mul = x * y                   # Multiply
    div = x / y                   # Divide
    
    # With scalars
    scaled = x * 2                # Broadcast scalar
    shifted = x + 10              # Broadcast scalar
    
    # Compound
    fma = x * y + z               # Fused multiply-add (automatic)
    
    # Unary
    neg = -x                      # Negate
    absolute = abs(x)             # Absolute value
    
    return add
```

### Math Functions

```python
@flow.kernel
def math_demo(data):
    x = flow(data)
    
    # Powers and roots
    squared = x ** 2              # Square
    cubed = x ** 3                # Cube
    root = x.sqrt()               # Square root
    
    # Exponential and logarithmic
    exp_x = x.exp()               # e^x
    log_x = x.log()               # ln(x)
    log2_x = x.log2()             # log₂(x)
    log10_x = x.log10()           # log₁₀(x)
    
    # Trigonometric
    sin_x = x.sin()               # sin(x)
    cos_x = x.cos()               # cos(x)
    tan_x = x.tan()               # tan(x)
    
    # Rounding
    floor_x = x.floor()           # Floor
    ceil_x = x.ceil()             # Ceiling
    round_x = x.round()           # Round to nearest
    
    return squared
```

### Comparisons and Selections

```python
@flow.kernel
def selection_demo(data):
    x = flow(data)
    
    # Comparisons create Patterns
    positive = x > 0              # Pattern: where x > 0
    negative = x < 0              # Pattern: where x < 0
    zero = x == 0                 # Pattern: where x == 0
    
    # Use patterns for conditional values
    abs_x = x.where(x > 0, else_=-x)  # Absolute value
    
    # Shorthand for common cases
    clamped = x.clamp(0, 100)     # Clamp to range
    relu = x.relu()               # max(0, x)
    
    return abs_x
```

### Reductions

```python
@flow.kernel
def reduction_demo(data):
    x = flow(data)
    
    # Sum and product
    total = x.sum()               # Add all elements
    product = x.product()         # Multiply all elements
    
    # Statistics
    mean = x.mean()               # Average
    variance = x.variance()       # Variance
    std = x.std()                 # Standard deviation
    
    # Min and max
    minimum = x.min()             # Smallest value
    maximum = x.max()             # Largest value
    
    # Locations
    min_idx = x.argmin()          # Index of smallest
    max_idx = x.argmax()          # Index of largest
    
    return total
```

### Filtering

```python
@flow.kernel
def filter_demo(data):
    x = flow(data)
    
    # Keep matching elements
    positives = x.keep(x > 0)     # Only positive values
    
    # Remove matching elements
    no_zeros = x.drop(x == 0)     # Remove zeros
    
    # Count
    count_pos = (x > 0).count()   # How many positive?
    
    return positives
```

## 7.3 Complete Beginner Examples

### Example 1: Normalize Audio

```python
import flow
import numpy as np

@flow.kernel
def normalize_audio(samples):
    """Normalize audio to [-1, 1] range."""
    s = flow(samples)
    peak = abs(s).max()
    return s / peak

# Usage
audio = np.random.randn(44100 * 10).astype(np.float32)  # 10 seconds
normalized = normalize_audio(audio)
```

### Example 2: ReLU Activation

```python
@flow.kernel
def relu(x):
    """Rectified Linear Unit: max(0, x)"""
    return flow(x).relu()

# Or equivalently:
@flow.kernel
def relu_explicit(x):
    data = flow(x)
    return data.where(data > 0, else_=0)
```

### Example 3: Softmax

```python
@flow.kernel
def softmax(x):
    """Numerically stable softmax."""
    data = flow(x)
    shifted = data - data.max()   # For numerical stability
    exp_x = shifted.exp()
    return exp_x / exp_x.sum()
```

### Example 4: Moving Average

```python
@flow.kernel
def moving_average_3(data):
    """3-point moving average."""
    s = flow(data)
    return (s.prev() + s + s.next()) / 3
```

### Example 5: Euclidean Distance

```python
@flow.kernel
def euclidean_distance(a, b):
    """Element-wise Euclidean distance."""
    diff = flow(a) - flow(b)
    return (diff ** 2).sum().sqrt()
```

## 7.4 Error Messages for Beginners

When beginners make mistakes, errors should teach:

```python
@flow.kernel
def bad_example(x):
    data = flow(x)
    for i in range(len(data)):    # ERROR: Scalar loop in kernel
        data[i] *= 2
    return data

# Error message:
"""
FlowSIMD Error: Scalar Loop Detected

You wrote:
    for i in range(len(data)):
        data[i] *= 2

This iterates one element at a time, defeating SIMD parallelism!

💡 Instead, write:
    data * 2

This processes ALL elements in parallel.

Learn more: https://flowsimd.dev/docs/parallel-thinking
"""
```

---

# Part 8: Developer Tier - Guided Performance

## 8.1 Design Goal

Give developers control without requiring hardware expertise.

## 8.2 Performance Hints

```python
from flow import hint

@flow.kernel
def hinted_kernel(x):
    with hint.unroll(4):              # Unroll loop 4x
        return flow(x) * 2 + 1

@flow.kernel
def more_hints(x):
    with hint.prefetch(distance=2):   # Prefetch 2 iterations ahead
        with hint.align(64):          # Assume 64-byte alignment
            return flow(x) * 2
```

### Available Hints

| Hint | Description |
|------|-------------|
| `hint.unroll(n)` | Unroll loop N times |
| `hint.prefetch(n)` | Prefetch N iterations ahead |
| `hint.align(n)` | Assume N-byte alignment |
| `hint.no_alias()` | Promise no pointer aliasing |
| `hint.contiguous()` | Data is contiguous |
| `hint.fast_math()` | Allow FP reassociation |
| `hint.target(isa)` | Target specific ISA |
| `hint.fuse()` | Force operation fusion |
| `hint.no_fuse()` | Prevent operation fusion |
| `hint.vectorize()` | Prefer vectorization |
| `hint.parallelize()` | Use multiple cores |

## 8.3 Explicit Patterns

```python
from flow import Pattern

@flow.kernel
def pattern_demo(data):
    x = flow(data)
    
    # Positional patterns
    first_10 = Pattern.first(10)           # First 10 elements
    last_10 = Pattern.last(10)             # Last 10 elements
    every_2 = Pattern.every(2)             # 0, 2, 4, 6, ...
    every_2_offset = Pattern.every(2, 1)   # 1, 3, 5, 7, ...
    range_pat = Pattern.range(10, 20)      # Elements 10-19
    
    # Combine patterns
    combined = first_10 & every_2          # First 10, but only evens
    
    # Apply patterns
    subset = x.at(first_10)                # Get first 10
    x.set_at(first_10, 0)                  # Zero out first 10
    
    return x
```

## 8.4 Explicit Shapes

```python
from flow import Shape

@flow.kernel
def shape_demo(data):
    x = flow(data)
    
    # Grouping
    grouped = x.groups(4)                  # Groups of 4
    reshaped = grouped.each(lambda g: g.reverse())  # Reverse each group
    
    # 2D interpretation
    matrix = x.as_2d(rows=100, cols=100)
    transposed = matrix.transpose()
    
    # Block operations
    blocks = x.blocks(16)                  # 16-element blocks
    block_sums = blocks.each(lambda b: b.sum())
    
    return reshaped.flatten()
```

## 8.5 Memory Control

```python
from flow import Memory

@flow.kernel
def memory_demo(data, output):
    # Explicit alignment
    x = flow.load_aligned(data, align=64)
    
    # Streaming stores (non-temporal)
    result = x * 2
    flow.store_streaming(output, result)
    
    # Prefetching
    flow.prefetch(data, offset=256)
    
    # Gather/Scatter
    indices = flow([0, 2, 4, 6])
    gathered = flow.gather(data, indices)
    flow.scatter(output, indices, gathered)
```

## 8.6 Platform Queries

```python
from flow import platform

@flow.kernel
def platform_aware(data):
    x = flow(data)
    
    # Query platform capabilities
    width = platform.simd_width(float)     # e.g., 8 for AVX2
    has_fma = platform.has_fma()           # FMA available?
    has_avx512 = platform.has_avx512()     # AVX-512 available?
    
    # Conditional based on platform
    if platform.has_avx512():
        # Use wider operations
        return x.process_wide()
    else:
        return x.process_normal()
```

## 8.7 Composable Pipelines

```python
from flow import Pipeline

# Define reusable pipeline
preprocess = Pipeline([
    lambda x: x - x.mean(),               # Center
    lambda x: x / x.std(),                # Normalize
    lambda x: x.clamp(-3, 3),             # Clip outliers
])

@flow.kernel
def use_pipeline(data):
    return flow(data).pipe(preprocess)

# Combine pipelines
full_pipeline = preprocess.then(postprocess)
```

## 8.8 Developer Examples

### Example 1: Optimized Dot Product

```python
from flow import hint

@flow.kernel
def dot_product(a, b):
    with hint.unroll(4), hint.fma():
        x, y = flow(a), flow(b)
        return (x * y).sum()
```

### Example 2: Strided Access

```python
@flow.kernel
def process_even_odd(data):
    x = flow(data)
    
    evens = x.at(Pattern.every(2, 0))     # 0, 2, 4, ...
    odds = x.at(Pattern.every(2, 1))      # 1, 3, 5, ...
    
    # Process separately
    evens_processed = evens * 2
    odds_processed = odds + 1
    
    # Interleave back
    return evens_processed.interleave(odds_processed)
```

### Example 3: Block Processing

```python
@flow.kernel
def block_normalize(data, block_size=64):
    """Normalize within each block."""
    x = flow(data)
    blocks = x.blocks(block_size)
    
    def normalize_block(block):
        return (block - block.mean()) / block.std()
    
    return blocks.each(normalize_block).flatten()
```

---

# Part 9: Expert Tier - Full Highway Access

## 9.1 Design Goal

Zero abstraction overhead. Full access to Highway's capabilities.

## 9.2 Direct Highway API

```python
from flow.highway import (
    Vec, Mask, ScalableTag, 
    Load, Store, Add, Mul, MulAdd,
    Lanes, FirstN, IfThenElse,
    # ... all Highway ops
)

@flow.kernel(mode='expert')
def highway_direct(data_ptr, output_ptr, count):
    d = ScalableTag[float]()
    N = Lanes(d)
    
    i = 0
    while i < count:
        v = Load(d, data_ptr + i)
        v = MulAdd(v, Set(d, 2.0), Set(d, 1.0))
        Store(v, d, output_ptr + i)
        i += N
```

## 9.3 Explicit Vector Types

```python
from flow.highway import Vec128, Vec256, Vec512

@flow.kernel(mode='expert', target='avx2')
def explicit_vectors(data):
    # Fixed-width vectors
    v = Vec256[float].load(data)
    result = v * 2.0 + 1.0
    return result.store()
```

## 9.4 Platform-Specific Code

```python
from flow.highway import HWY_TARGET, HWY_AVX2, HWY_AVX512, HWY_NEON

@flow.kernel(mode='expert')
def platform_specific(data):
    if HWY_TARGET == HWY_AVX512:
        # AVX-512 specific code
        from flow.native.x86 import _mm512_reduce_add_ps
        # ...
    elif HWY_TARGET == HWY_NEON:
        # NEON specific code
        from flow.native.arm import vaddvq_f32
        # ...
    else:
        # Portable fallback
        pass
```

## 9.5 Mixing Tiers

```python
@flow.kernel
def mixed_tiers(data):
    # Start with beginner-friendly code
    x = flow(data)
    preprocessed = x * 2 + 1
    
    # Drop to expert mode for critical section
    with flow.expert_mode():
        from flow.highway import ScalableTag, Lanes, Load
        d = ScalableTag[float]()
        # ... expert code ...
    
    # Back to beginner mode
    return preprocessed.sqrt()
```

## 9.6 Expert Examples

### Example 1: Manual Loop with Predication

```python
from flow.highway import *

@flow.kernel(mode='expert')
def predicated_loop(data, count):
    d = ScalableTag[float]()
    N = Lanes(d)
    acc = Zero(d)
    
    i = 0
    while i < count:
        # Create mask for valid lanes
        remaining = count - i
        mask = FirstN(d, min(remaining, N))
        
        # Masked load
        v = MaskedLoad(mask, d, data + i, Zero(d))
        
        # Accumulate
        acc = Add(acc, v)
        i += N
    
    return ReduceSum(d, acc)
```

### Example 2: Optimized Matrix Transpose

```python
from flow.highway import *

@flow.kernel(mode='expert', target='avx2')
def transpose_4x4(input, output):
    d = Fixed128[float]()  # 4 floats
    
    # Load 4 rows
    r0 = Load(d, input + 0)
    r1 = Load(d, input + 4)
    r2 = Load(d, input + 8)
    r3 = Load(d, input + 12)
    
    # Transpose using unpacks
    t0 = InterleaveLower(d, r0, r1)
    t1 = InterleaveUpper(d, r0, r1)
    t2 = InterleaveLower(d, r2, r3)
    t3 = InterleaveUpper(d, r2, r3)
    
    # Second stage
    out0 = InterleaveLower(d, t0, t2)
    out1 = InterleaveUpper(d, t0, t2)
    out2 = InterleaveLower(d, t1, t3)
    out3 = InterleaveUpper(d, t1, t3)
    
    # Store
    Store(out0, d, output + 0)
    Store(out1, d, output + 4)
    Store(out2, d, output + 8)
    Store(out3, d, output + 12)
```

### Example 3: AES Encryption

```python
from flow.highway import *
from flow.highway.crypto import AESRound, AESLastRound

@flow.kernel(mode='expert')
def aes_encrypt_block(state, round_keys, num_rounds):
    d = Fixed128[u8]()
    
    # Load state and XOR with first round key
    s = Load(d, state)
    s = Xor(s, Load(d, round_keys))
    
    # Main rounds
    for i in range(1, num_rounds):
        key = Load(d, round_keys + i * 16)
        s = AESRound(s, key)
    
    # Final round
    key = Load(d, round_keys + num_rounds * 16)
    s = AESLastRound(s, key)
    
    Store(s, d, state)
```

---

# Part 10: Cross-Tier Interoperability

## 10.1 Calling Between Tiers

```python
# Beginner function
@flow.kernel
def beginner_preprocess(data):
    return flow(data) * 2

# Developer function using beginner function
@flow.kernel
def developer_pipeline(data):
    preprocessed = beginner_preprocess.inline(data)
    with hint.unroll(4):
        return preprocessed + 1

# Expert function using both
@flow.kernel(mode='expert')
def expert_optimized(data, count):
    from flow.highway import ScalableTag, Lanes
    d = ScalableTag[float]()
    
    # Use beginner preprocessing
    preprocessed = beginner_preprocess(data)
    
    # Expert-level reduction
    # ...
```

## 10.2 Tier Escape Hatches

```python
@flow.kernel
def with_escape_hatches(data):
    x = flow(data)
    
    # Normal beginner code
    result = x * 2
    
    # Escape to developer tier for hints
    with flow.developer():
        result = result.with_hint(unroll=4)
    
    # Escape to expert tier for intrinsics
    with flow.expert():
        # Raw Highway access
        pass
    
    return result
```

## 10.3 Shared Memory Model

All tiers share the same memory model:
- Arrays are views into memory
- Bunches don't own memory
- Explicit materialization when needed

```python
@flow.kernel
def memory_shared(data):
    x = flow(data)              # View, no copy
    
    y = x * 2                   # Lazy, no memory
    z = y + 1                   # Still lazy
    
    # Materialization only at:
    result = z.collect()        # Explicit: creates array
    return z                    # Implicit: creates output array
```

---

# Part 11: JIT Architecture

## 11.1 Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    JIT COMPILATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PYTHON AST ANALYSIS                                         │
│     ├── Parse @flow.kernel function                             │
│     ├── Identify Bunch operations                               │
│     ├── Track data flow                                         │
│     └── Validate SIMD compatibility                             │
│                            ↓                                    │
│  2. FLOW IR GENERATION                                          │
│     ├── Convert to internal IR                                  │
│     ├── Type inference                                          │
│     ├── Operation canonicalization                              │
│     └── Pattern matching                                        │
│                            ↓                                    │
│  3. OPTIMIZATION PASSES                                         │
│     ├── Operation fusion                                        │
│     ├── Common subexpression elimination                        │
│     ├── Dead code elimination                                   │
│     ├── Loop optimization                                       │
│     └── Memory access optimization                              │
│                            ↓                                    │
│  4. HIGHWAY CODE GENERATION                                     │
│     ├── Map Flow ops to Highway ops                             │
│     ├── Generate loop structure                                 │
│     ├── Handle edge cases (alignment, remainder)                │
│     └── Multi-target code generation                            │
│                            ↓                                    │
│  5. C++ COMPILATION                                             │
│     ├── Compile Highway C++ code                                │
│     ├── Generate for each target ISA                            │
│     └── Create shared library                                   │
│                            ↓                                    │
│  6. RUNTIME DISPATCH                                            │
│     ├── Detect CPU capabilities                                 │
│     ├── Select best compiled version                            │
│     └── Cache selection                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 11.2 Operation Fusion

```python
# User writes:
@flow.kernel
def fusion_example(x):
    a = flow(x) * 2
    b = a + 1
    c = b.sqrt()
    return c

# JIT fuses into single loop:
"""
void fused(D d, const T* in, T* out, size_t n) {
    for (size_t i = 0; i < n; i += Lanes(d)) {
        auto v = Load(d, in + i);
        v = Sqrt(MulAdd(v, Set(d, 2), Set(d, 1)));
        Store(v, d, out + i);
    }
}
"""
```

## 11.3 Multi-Target Dispatch

```python
@flow.kernel(targets=['avx512', 'avx2', 'neon', 'scalar'])
def portable(x):
    return flow(x) * 2 + 1

# Generates 4 versions, dispatches at runtime based on CPU
```

## 11.4 Highway Code Templates

Each FlowSIMD operation maps to a Highway template:

```cpp
// flow(x) * 2 generates:
template<typename D>
void MulScalar(D d, const TFromD<D>* HWY_RESTRICT in,
               TFromD<D>* HWY_RESTRICT out,
               size_t count, TFromD<D> scalar) {
    const auto vscalar = Set(d, scalar);
    for (size_t i = 0; i < count; i += Lanes(d)) {
        const auto v = Load(d, in + i);
        Store(Mul(v, vscalar), d, out + i);
    }
}

// numbers.sum() generates:
template<typename D>
TFromD<D> Sum(D d, const TFromD<D>* HWY_RESTRICT in, size_t count) {
    auto acc = Zero(d);
    for (size_t i = 0; i < count; i += Lanes(d)) {
        acc = Add(acc, Load(d, in + i));
    }
    return ReduceSum(d, acc);
}

// x.where(cond, else_=0) generates:
template<typename D>
void Where(D d, const TFromD<D>* HWY_RESTRICT in,
           TFromD<D>* HWY_RESTRICT out, size_t count,
           TFromD<D> threshold, TFromD<D> else_val) {
    const auto vthresh = Set(d, threshold);
    const auto velse = Set(d, else_val);
    for (size_t i = 0; i < count; i += Lanes(d)) {
        const auto v = Load(d, in + i);
        const auto mask = Gt(v, vthresh);
        Store(IfThenElse(mask, v, velse), d, out + i);
    }
}
```

## 11.5 Edge Case Handling

```cpp
// Automatic handling of non-aligned data and remainders:
template<typename D>
void ProcessWithRemainder(D d, const T* in, T* out, size_t count) {
    const size_t N = Lanes(d);
    
    // Main vectorized loop
    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = Load(d, in + i);
        // ... process ...
        Store(result, d, out + i);
    }
    
    // Handle remainder with masked operations
    if (i < count) {
        const auto mask = FirstN(d, count - i);
        const auto v = MaskedLoad(mask, d, in + i, Zero(d));
        // ... process ...
        BlendedStore(result, mask, d, out + i);
    }
}
```

---

# Part 12: Complete API Reference

## 12.1 Bunch Creation

| Function | Description | Example |
|----------|-------------|---------|
| `flow(arr)` | From array | `flow([1,2,3])` |
| `flow.zeros(n)` | N zeros | `flow.zeros(100)` |
| `flow.ones(n)` | N ones | `flow.ones(100)` |
| `flow.range(a,b)` | Integer range | `flow.range(0, 100)` |
| `flow.linspace(a,b,n)` | Linear space | `flow.linspace(0, 1, 100)` |
| `flow.random(n)` | Random [0,1) | `flow.random(100)` |

## 12.2 Arithmetic Operations

| Operation | Description | Highway |
|-----------|-------------|---------|
| `a + b` | Add | `Add` |
| `a - b` | Subtract | `Sub` |
| `a * b` | Multiply | `Mul` |
| `a / b` | Divide | `Div` |
| `a // b` | Integer div | Custom |
| `a % b` | Modulo | Custom |
| `a ** n` | Power | Custom/`Exp` |
| `-a` | Negate | `Neg` |
| `abs(a)` | Absolute | `Abs` |
| `a.sqrt()` | Square root | `Sqrt` |
| `a.rsqrt()` | Reciprocal sqrt | `ApproximateReciprocal` |

## 12.3 Math Functions

| Method | Description | Highway |
|--------|-------------|---------|
| `.exp()` | e^x | `Exp` (contrib) |
| `.log()` | ln(x) | `Log` (contrib) |
| `.log2()` | log₂(x) | Custom |
| `.log10()` | log₁₀(x) | Custom |
| `.sin()` | sin(x) | `Sin` (contrib) |
| `.cos()` | cos(x) | `Cos` (contrib) |
| `.tan()` | tan(x) | Custom |
| `.floor()` | Floor | `Floor` |
| `.ceil()` | Ceiling | `Ceil` |
| `.round()` | Round | `Round` |
| `.trunc()` | Truncate | `Trunc` |

## 12.4 Comparison Operations

| Operation | Description | Highway |
|-----------|-------------|---------|
| `a > b` | Greater than | `Gt` |
| `a >= b` | Greater or equal | `Ge` |
| `a < b` | Less than | `Lt` |
| `a <= b` | Less or equal | `Le` |
| `a == b` | Equal | `Eq` |
| `a != b` | Not equal | `Ne` |

## 12.5 Pattern Operations

| Method | Description | Highway |
|--------|-------------|---------|
| `Pattern.first(n)` | First n true | `FirstN` |
| `Pattern.last(n)` | Last n true | Custom |
| `Pattern.every(n)` | Every nth | Custom |
| `Pattern.range(a,b)` | Range a to b | Custom |
| `p1 & p2` | And | `And` |
| `p1 \| p2` | Or | `Or` |
| `~p` | Not | `Not` |
| `p.any()` | Any true? | `FindFirstTrue >= 0` |
| `p.all()` | All true? | `AllTrue` |
| `p.count()` | Count true | `CountTrue` |
| `p.first_true()` | First index | `FindFirstTrue` |

## 12.6 Selection Operations

| Method | Description | Highway |
|--------|-------------|---------|
| `.where(c, else_)` | Conditional | `IfThenElse` |
| `.keep(p)` | Keep matching | `Compress` |
| `.drop(p)` | Drop matching | `Compress(~p)` |
| `.at(p)` | Get at pattern | `CompressStore` |
| `.set_at(p, v)` | Set at pattern | `BlendedStore` |
| `.clamp(lo, hi)` | Clamp range | `Clamp` |
| `.relu()` | max(0, x) | `Max(x, 0)` |

## 12.7 Shape Operations

| Method | Description | Highway |
|--------|-------------|---------|
| `.reverse()` | Reverse order | `Reverse` |
| `.rotate(n)` | Rotate by n | `RotateRight` |
| `.repeat(n)` | Repeat n times | Custom |
| `.stretch(n)` | Duplicate each | Custom |
| `.interleave(b)` | Interleave | `InterleaveLower/Upper` |
| `.deinterleave(n)` | Deinterleave | Custom |
| `.prev()` | Shift right | `SlideDownLanes` |
| `.next()` | Shift left | `SlideUpLanes` |
| `.broadcast(i)` | Broadcast lane | `Broadcast<i>` |
| `.groups(n)` | Group by n | Custom |
| `.transpose()` | Transpose | Custom |
| `.shuffle(idx)` | Permute | `TableLookupBytes` |

## 12.8 Fold Operations

| Method | Description | Highway |
|--------|-------------|---------|
| `.sum()` | Sum all | `ReduceSum` |
| `.product()` | Product all | Custom |
| `.min()` | Minimum | `MinOfLanes` |
| `.max()` | Maximum | `MaxOfLanes` |
| `.mean()` | Average | `ReduceSum/count` |
| `.argmin()` | Index of min | Custom |
| `.argmax()` | Index of max | Custom |
| `.fold(fn, init)` | Custom fold | Custom |
| `.fold_many(...)` | Multi-fold | Custom |

## 12.9 Scan Operations

| Method | Description | Highway |
|--------|-------------|---------|
| `.cumsum()` | Cumulative sum | `ScanSum` |
| `.cumprod()` | Cumulative product | Custom |
| `.cummax()` | Cumulative max | Custom |
| `.cummin()` | Cumulative min | Custom |
| `.scan(fn, init)` | Custom scan | Custom |

## 12.10 Memory Operations

| Method | Description | Highway |
|--------|-------------|---------|
| `.into(arr)` | Store to array | `Store` |
| `.collect()` | Materialize | Allocate + `Store` |
| `.value()` | Get scalar | `GetLane` |
| `flow.gather(a, idx)` | Gather | `GatherOffset` |
| `flow.scatter(a, idx, v)` | Scatter | `ScatterOffset` |

## 12.11 Composition

| Method | Description |
|--------|-------------|
| `.pipe(fn)` | Apply function |
| `.pipe(fn, arg)` | Apply with args |
| `.pipe_if(c, fn)` | Conditional pipe |
| `.tap(fn)` | Inspect (debug) |
| `.materialize()` | Force evaluation |
| `Pipeline([...])` | Reusable pipeline |
| `p1.then(p2)` | Chain pipelines |

## 12.12 Hints (Developer Tier)

| Hint | Description |
|------|-------------|
| `hint.unroll(n)` | Unroll n times |
| `hint.prefetch(n)` | Prefetch distance |
| `hint.align(n)` | Alignment assumption |
| `hint.no_alias()` | No aliasing |
| `hint.contiguous()` | Contiguous data |
| `hint.fast_math()` | Relaxed FP |
| `hint.fuse()` | Force fusion |
| `hint.no_fuse()` | Prevent fusion |
| `hint.target(isa)` | Target ISA |

---

# Appendix A: Highway Operation Mapping

Complete mapping from FlowSIMD to Highway operations:

```
┌────────────────────────────────────────────────────────────────┐
│ FlowSIMD Operation        │ Highway Function(s)               │
├────────────────────────────────────────────────────────────────┤
│ flow(arr)                 │ Load(d, ptr)                      │
│ x + y                     │ Add(x, y)                         │
│ x - y                     │ Sub(x, y)                         │
│ x * y                     │ Mul(x, y)                         │
│ x / y                     │ Div(x, y)                         │
│ x * y + z                 │ MulAdd(x, y, z)  [FMA]            │
│ -x                        │ Neg(x)                            │
│ abs(x)                    │ Abs(x)                            │
│ x.sqrt()                  │ Sqrt(x)                           │
│ x > y                     │ Gt(x, y) → Mask                   │
│ x.where(c, else_)         │ IfThenElse(mask, x, else_)        │
│ x.keep(p)                 │ Compress(x, mask)                 │
│ x.sum()                   │ ReduceSum(d, x)                   │
│ x.min()                   │ MinOfLanes(d, x)                  │
│ x.max()                   │ MaxOfLanes(d, x)                  │
│ x.reverse()               │ Reverse(d, x)                     │
│ x.prev()                  │ SlideDownLanes(d, x, 1)           │
│ x.next()                  │ SlideUpLanes(d, x, 1)             │
│ x.interleave(y)           │ InterleaveLower/Upper(d, x, y)    │
│ Pattern.first(n)          │ FirstN(d, n)                      │
│ p.any()                   │ FindFirstTrue(d, mask) >= 0       │
│ p.all()                   │ AllTrue(d, mask)                  │
│ p.count()                 │ CountTrue(d, mask)                │
└────────────────────────────────────────────────────────────────┘
```

---

# Appendix B: Migration from NumPy

```python
# NumPy                          # FlowSIMD
import numpy as np               import flow

a = np.array([1,2,3])           a = flow([1,2,3])
a * 2                            a * 2
np.sqrt(a)                       a.sqrt()
np.sum(a)                        a.sum()
np.max(a)                        a.max()
a[a > 0]                         a.keep(a > 0)
np.where(a > 0, a, 0)           a.where(a > 0, else_=0)
a[::-1]                          a.reverse()
np.cumsum(a)                     a.cumsum()
```

---

# Appendix C: Performance Comparison

| Operation | NumPy | FlowSIMD | Speedup |
|-----------|-------|----------|---------|
| Element-wise mul | 1.0x | 4-16x | 4-16x |
| Sum reduction | 1.0x | 3-8x | 3-8x |
| Conditional | 1.0x | 5-15x | 5-15x |
| Stencil (blur) | 1.0x | 8-20x | 8-20x |
| Softmax | 1.0x | 4-10x | 4-10x |

*Speedups vary by hardware and data size*

---

# Appendix D: Glossary

| Term | Definition |
|------|------------|
| **Bunch** | A collection of values processed in parallel (replaces "vector") |
| **Pattern** | A selection of elements (replaces "mask") |
| **Kernel** | A compiled SIMD function |
| **Each** | Element-wise parallel operation |
| **Fold** | Reduction to fewer/single value |
| **Shape** | Arrangement/ordering of elements |
| **Scan** | Prefix/cumulative operation |
| **Highway** | Google's portable SIMD C++ library (our backend) |
| **ISA** | Instruction Set Architecture (AVX2, NEON, etc.) |
| **Lane** | Single element position in a SIMD register |

---

*End of FlowSIMD DSL Specification v3.0*