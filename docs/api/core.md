# Core API Reference

This document covers the core functions and classes in Bud Flow Lang.

---

## Initialization & Shutdown

### `flow.initialize()`

Initialize the Bud Flow Lang runtime. Must be called before using any other functions.

```python
import bud_flow_lang_py as flow

flow.initialize()
```

**Notes:**
- Safe to call multiple times (no-op if already initialized)
- Automatically registers cleanup handler for shutdown at exit

---

### `flow.shutdown()`

Manually shutdown the runtime and free resources.

```python
flow.shutdown()
```

**Notes:**
- Called automatically when Python exits
- Safe to call multiple times

---

### `flow.is_initialized()`

Check if the runtime is initialized.

```python
if flow.is_initialized():
    print("Ready to use!")
```

**Returns:** `bool`

---

## Hardware Information

### `flow.get_hardware_info()`

Get detailed hardware information.

```python
info = flow.get_hardware_info()
```

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `arch_family` | str | Architecture family ("x86", "ARM", etc.) |
| `is_64bit` | bool | True if 64-bit architecture |
| `simd_width` | int | SIMD register width in bytes |
| `physical_cores` | int | Number of physical CPU cores |
| `logical_cores` | int | Number of logical CPU cores |
| `has_sse2` | bool | SSE2 support (x86) |
| `has_avx` | bool | AVX support (x86) |
| `has_avx2` | bool | AVX2 support (x86) |
| `has_avx512` | bool | AVX-512 support (x86) |
| `has_neon` | bool | NEON support (ARM) |
| `has_sve` | bool | SVE support (ARM) |
| `supports_float16` | bool | Native float16 support |
| `supports_bfloat16` | bool | Native bfloat16 support |

**Example:**
```python
info = flow.get_hardware_info()
print(f"Using {info['simd_width']*8}-bit SIMD")
print(f"AVX2: {info['has_avx2']}")
```

---

### `flow.get_simd_width()`

Get the SIMD register width in bytes.

```python
width = flow.get_simd_width()  # e.g., 32 for AVX2
```

**Returns:** `int`

---

### `flow.get_simd_capabilities()`

Get a human-readable summary of SIMD capabilities.

```python
caps = flow.get_simd_capabilities()
print(caps)
# Output:
# Architecture: x86 (64-bit)
# SIMD Width: 256 bits
# Features: SSE2 SSSE3 SSE4 AVX AVX2 FMA
```

**Returns:** `str`

---

## Memory Configuration

### `flow.detect_cache_config()`

Detect CPU cache configuration.

```python
cache = flow.detect_cache_config()
```

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `l1_size` | int | L1 cache size in bytes |
| `l2_size` | int | L2 cache size in bytes |
| `l3_size` | int | L3 cache size in bytes |
| `line_size` | int | Cache line size in bytes |
| `l1_size_kb` | int | L1 cache size in KB |
| `l2_size_kb` | int | L2 cache size in KB |
| `l3_size_kb` | int | L3 cache size in KB |

**Example:**
```python
cache = flow.detect_cache_config()
print(f"L1: {cache['l1_size_kb']}KB, Line: {cache['line_size']} bytes")
```

---

### `flow.optimal_tile_size(element_size, num_arrays)`

Calculate the optimal tile size for cache-efficient operations.

```python
tile = flow.optimal_tile_size(4, 2)  # float32, 2 arrays
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `element_size` | int | Size of each element in bytes (4 for float32) |
| `num_arrays` | int | Number of arrays involved in the operation |

**Returns:** `int` - Optimal number of elements per tile

---

### `flow.get_memory_optimization_status()`

Get current memory optimization settings.

```python
status = flow.get_memory_optimization_status()
```

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `tiling_enabled` | bool | Whether tiling is enabled |
| `prefetch_enabled` | bool | Whether prefetching is enabled |
| `l1_size_kb` | int | L1 cache size |
| `l2_size_kb` | int | L2 cache size |
| `l3_size_kb` | int | L3 cache size |
| `line_size` | int | Cache line size |
| `optimal_tile_size_float32` | int | Optimal tile for float32 |

---

### `flow.set_tiling_enabled(enabled)`

Enable or disable cache-aware tiling.

```python
flow.set_tiling_enabled(True)   # Enable (default)
flow.set_tiling_enabled(False)  # Disable
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `enabled` | bool | Whether to enable tiling |

---

### `flow.is_tiling_enabled()`

Check if tiling is enabled.

```python
if flow.is_tiling_enabled():
    print("Tiling is active")
```

**Returns:** `bool`

---

### `flow.set_prefetch_enabled(enabled)`

Enable or disable software prefetching.

```python
flow.set_prefetch_enabled(True)   # Enable (default)
flow.set_prefetch_enabled(False)  # Disable
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `enabled` | bool | Whether to enable prefetching |

---

### `flow.is_prefetch_enabled()`

Check if prefetching is enabled.

```python
if flow.is_prefetch_enabled():
    print("Prefetching is active")
```

**Returns:** `bool`

---

## Type Information

### `flow.ScalarType`

Enumeration of supported scalar types.

```python
flow.ScalarType.float32   # 32-bit float (default)
flow.ScalarType.float64   # 64-bit float
flow.ScalarType.int32     # 32-bit integer
flow.ScalarType.int64     # 64-bit integer
```

---

### `flow.dtype_to_string(dtype)`

Convert a ScalarType to its string representation.

```python
s = flow.dtype_to_string(flow.ScalarType.float32)  # "float32"
```

---

### `flow.string_to_dtype(name)`

Convert a string to a ScalarType.

```python
dtype = flow.string_to_dtype("float32")  # ScalarType.float32
dtype = flow.string_to_dtype("f32")      # Also works
```

**Supported strings:**
- `"float32"`, `"f32"`, `"float"`
- `"float64"`, `"f64"`, `"double"`
- `"int32"`, `"i32"`, `"int"`
- `"int64"`, `"i64"`, `"long"`

---

## Version Information

### `flow.__version__`

Get the library version string.

```python
print(flow.__version__)  # e.g., "0.1.0"
```

---

## See Also

- [Array Creation](creation.md) - Creating arrays
- [Operations](operations.md) - Arithmetic and reduction operations
- [Memory Management](memory.md) - Advanced memory control
