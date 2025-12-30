# Array Creation API Reference

This document covers functions for creating Bunch (array) objects.

---

## Factory Functions

### `flow.zeros(count, dtype='float32')`

Create an array filled with zeros.

```python
a = flow.zeros(1000)                    # 1000 float32 zeros
b = flow.zeros(1000, dtype='float64')   # 1000 float64 zeros
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `count` | int | required | Number of elements |
| `dtype` | str | 'float32' | Data type |

**Returns:** `Bunch`

---

### `flow.ones(count, dtype='float32')`

Create an array filled with ones.

```python
a = flow.ones(1000)                     # 1000 float32 ones
b = flow.ones(1000, dtype='float64')    # 1000 float64 ones
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `count` | int | required | Number of elements |
| `dtype` | str | 'float32' | Data type |

**Returns:** `Bunch`

---

### `flow.full(count, value, dtype='float32')`

Create an array filled with a specified value.

```python
a = flow.full(1000, 3.14)               # 1000 elements of 3.14
b = flow.full(1000, 2.5, dtype='float64')
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `count` | int | required | Number of elements |
| `value` | float | required | Fill value |
| `dtype` | str | 'float32' | Data type |

**Returns:** `Bunch`

---

### `flow.arange(count, start=0.0, step=1.0)`

Create an array with evenly spaced values.

Values are: `[start, start+step, start+2*step, ...]`

```python
# One argument: count (start=0, step=1)
a = flow.arange(10)              # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Two arguments: count, start
b = flow.arange(5, 10.0)         # [10, 11, 12, 13, 14]

# Three arguments: count, start, step
c = flow.arange(5, 0.0, 2.0)     # [0, 2, 4, 6, 8]
d = flow.arange(10, 1.0, 0.5)    # [1.0, 1.5, 2.0, ..., 5.5]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `count` | int | required | Number of elements |
| `start` | float | 0.0 | Starting value |
| `step` | float | 1.0 | Step between values |

**Returns:** `Bunch`

**Note:** Unlike NumPy's `arange(start, stop, step)`, Flow's arange uses `arange(count, start, step)`. Use `linspace` for NumPy-like behavior with start/stop.

---

### `flow.linspace(start, stop, count, dtype='float32')`

Create an array with linearly spaced values.

```python
a = flow.linspace(0, 1, 5)       # [0.0, 0.25, 0.5, 0.75, 1.0]
b = flow.linspace(0, 10, 11)     # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `start` | float | required | Start value |
| `stop` | float | required | End value (inclusive) |
| `count` | int | required | Number of elements |
| `dtype` | str | 'float32' | Data type |

**Returns:** `Bunch`

---

### `flow.flow(data)`

Create a Bunch from existing data.

```python
import numpy as np

# From NumPy array
np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
a = flow.flow(np_arr)

# From Python list
b = flow.flow([1.0, 2.0, 3.0])

# From another Bunch (copy)
c = flow.flow(a)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `data` | array-like | NumPy array, list, or Bunch |

**Returns:** `Bunch`

**Notes:**
- For NumPy arrays, data is copied (not shared)
- Lists are converted to float32 by default

---

## The Bunch Class

### Properties

#### `bunch.size`

Number of elements in the array.

```python
a = flow.ones(1000)
print(a.size)  # 1000
```

**Returns:** `int`

---

#### `bunch.shape`

Shape of the array as a tuple.

```python
a = flow.ones(1000)
print(a.shape)  # (1000,)
```

**Returns:** `tuple`

---

#### `bunch.dtype`

Data type of the array.

```python
a = flow.ones(1000)
print(a.dtype)  # ScalarType.float32
```

**Returns:** `ScalarType`

---

#### `bunch.nbytes`

Total size in bytes.

```python
a = flow.ones(1000)  # float32
print(a.nbytes)  # 4000
```

**Returns:** `int`

---

#### `bunch.itemsize`

Size of each element in bytes.

```python
a = flow.ones(1000)  # float32
print(a.itemsize)  # 4
```

**Returns:** `int`

---

#### `bunch.ndim`

Number of dimensions (always 1 for now).

```python
a = flow.ones(1000)
print(a.ndim)  # 1
```

**Returns:** `int`

---

### Methods

#### `bunch.to_numpy()`

Convert to a NumPy array.

```python
a = flow.arange(5)
np_arr = a.to_numpy()
print(type(np_arr))  # <class 'numpy.ndarray'>
print(np_arr)        # [0. 1. 2. 3. 4.]
```

**Returns:** `numpy.ndarray`

---

#### `bunch.copy()`

Create a copy of the array.

```python
a = flow.ones(100)
b = a.copy()  # Independent copy
```

**Returns:** `Bunch`

---

## Examples

### Creating Various Arrays

```python
import bud_flow_lang_py as flow
import numpy as np

flow.initialize()

# Basic creation
zeros = flow.zeros(100)
ones = flow.ones(100)
pi_arr = flow.full(100, 3.14159)

# Sequences
indices = flow.arange(100)         # [0, 1, 2, ..., 99]
evens = flow.arange(50, 0.0, 2.0)  # [0, 2, 4, ..., 98]
normalized = flow.linspace(0, 1, 101)

# From external data
np_data = np.random.randn(100).astype(np.float32)
from_numpy = flow.flow(np_data)

py_list = [1.0, 2.0, 3.0, 4.0, 5.0]
from_list = flow.flow(py_list)
```

### Large Array Creation

```python
# Million-element arrays
a = flow.ones(1_000_000)
print(f"Size: {a.size:,} elements")
print(f"Memory: {a.nbytes / 1e6:.1f} MB")

# Ten million elements
b = flow.zeros(10_000_000)
print(f"Size: {b.size:,} elements")
print(f"Memory: {b.nbytes / 1e6:.1f} MB")
```

### Working with Different Types

```python
# Float32 (default, recommended)
f32 = flow.ones(100)
print(f"float32: {f32.itemsize} bytes per element")

# Float64 for higher precision
f64 = flow.ones(100, dtype='float64')
print(f"float64: {f64.itemsize} bytes per element")
```

---

## See Also

- [Operations](operations.md) - Performing calculations
- [Core API](core.md) - Initialization and configuration
