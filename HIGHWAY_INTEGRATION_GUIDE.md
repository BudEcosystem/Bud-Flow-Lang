# Bud Flow Lang - Highway Integration Guide

**Version:** 1.0
**Date:** December 2024
**Purpose:** Low-level integration specification for Highway SIMD backend

---

## Table of Contents

1. [Highway Architecture Deep Dive](#1-highway-architecture-deep-dive)
2. [Integration Strategy](#2-integration-strategy)
3. [Core Integration Layer](#3-core-integration-layer)
4. [Code Generation Patterns](#4-code-generation-patterns)
5. [Multi-Target Dispatch Integration](#5-multi-target-dispatch-integration)
6. [Memory Management Integration](#6-memory-management-integration)
7. [Operation Mapping](#7-operation-mapping)
8. [Extension Requirements](#8-extension-requirements)
9. [Performance Considerations](#9-performance-considerations)
10. [Stability Guidelines](#10-stability-guidelines)
11. [Edge Cases and Mitigations](#11-edge-cases-and-mitigations)

---

## 1. Highway Architecture Deep Dive

### 1.1 Core Type System

Highway's type system is built on the `Simd<T, N, kPow2>` template:

```cpp
// highway/hwy/ops/shared-inl.h:213-336
template <typename Lane, size_t N, int kPow2>
struct Simd {
    using T = Lane;
    static constexpr size_t kPrivateLanes = ...;  // Computed max lanes
    static constexpr int kPrivatePow2 = kPow2;

    constexpr size_t MaxLanes() const { return kPrivateLanes; }
    constexpr size_t MaxBytes() const { return kPrivateLanes * sizeof(Lane); }

    // Type transformations
    template <typename NewT> using Rebind = ...;
    template <typename NewT> using Repartition = ...;
    using Half = Simd<T, N, kPow2 - 1>;
    using Twice = Simd<T, N, kPow2 + 1>;
};
```

**Key Type Aliases for Bud Flow Lang:**

| Alias | Purpose | Bud Flow Lang Usage |
|-------|---------|---------------------|
| `ScalableTag<T>` | Full vector, unknown size at compile time | Default for all operations |
| `CappedTag<T, N>` | Vector with max N lanes | When user specifies max size |
| `FixedTag<T, N>` | Exactly N lanes | For matrix operations with known dimensions |
| `Vec<D>` | Vector type for tag D | Result type for operations |
| `Mask<D>` | Mask type for tag D | Comparison results |

### 1.2 Lane Type Support

Highway supports the following lane types (base.h):

| Type | Sizes | Highway Name | Bud Flow Lang Mapping |
|------|-------|--------------|----------------------|
| `int8_t` | 1 byte | `int8_t` | `flow.i8` |
| `int16_t` | 2 bytes | `int16_t` | `flow.i16` |
| `int32_t` | 4 bytes | `int32_t` | `flow.i32` |
| `int64_t` | 8 bytes | `int64_t` | `flow.i64` |
| `uint8_t` | 1 byte | `uint8_t` | `flow.u8` |
| `uint16_t` | 2 bytes | `uint16_t` | `flow.u16` |
| `uint32_t` | 4 bytes | `uint32_t` | `flow.u32` |
| `uint64_t` | 8 bytes | `uint64_t` | `flow.u64` |
| `float` | 4 bytes | `float` | `flow.f32` |
| `double` | 8 bytes | `double` | `flow.f64` |
| `hwy::float16_t` | 2 bytes | `float16_t` | `flow.f16` |
| `hwy::bfloat16_t` | 2 bytes | `bfloat16_t` | `flow.bf16` |

### 1.3 Multi-Target Architecture

Highway compiles code for multiple targets via `foreach_target.h`:

```
Compilation Flow:
┌─────────────────┐
│  Source File    │
│  (user_code.cc) │
└────────┬────────┘
         │
         │ #include "hwy/foreach_target.h"
         ▼
┌─────────────────────────────────────────┐
│  HWY_TARGET = HWY_SSE2                  │
│  #include HWY_TARGET_INCLUDE            │ ──► Compile with SSE2 flags
├─────────────────────────────────────────┤
│  HWY_TARGET = HWY_AVX2                  │
│  #include HWY_TARGET_INCLUDE            │ ──► Compile with AVX2 flags
├─────────────────────────────────────────┤
│  HWY_TARGET = HWY_AVX3                  │
│  #include HWY_TARGET_INCLUDE            │ ──► Compile with AVX-512 flags
├─────────────────────────────────────────┤
│  HWY_TARGET = HWY_STATIC_TARGET         │
│  (final compilation)                    │ ──► Compile with best available
└─────────────────────────────────────────┘
```

**Target Constants (targets.h:88-163):**

| Target | Architecture | Vector Width |
|--------|--------------|--------------|
| `HWY_SCALAR` | Any | 1 lane |
| `HWY_EMU128` | Any | 128-bit emulated |
| `HWY_SSE2` | x86 | 128-bit |
| `HWY_SSE4` | x86 | 128-bit |
| `HWY_AVX2` | x86 | 256-bit |
| `HWY_AVX3` | x86 | 512-bit |
| `HWY_AVX3_DL` | x86 | 512-bit + VNNI |
| `HWY_AVX3_ZEN4` | x86 | 512-bit Zen4 |
| `HWY_AVX3_SPR` | x86 | 512-bit SPR |
| `HWY_AVX10_2` | x86 | AVX10.2 |
| `HWY_NEON` | ARM | 128-bit |
| `HWY_NEON_BF16` | ARM | 128-bit + bf16 |
| `HWY_SVE` | ARM | Scalable (128-2048) |
| `HWY_SVE2` | ARM | Scalable v2 |
| `HWY_RVV` | RISC-V | Scalable |
| `HWY_WASM` | WebAssembly | 128-bit |

### 1.4 Dynamic Dispatch Mechanism

Highway uses `HWY_EXPORT` and `HWY_DYNAMIC_DISPATCH` for runtime dispatch:

```cpp
// Define function in each target namespace
namespace hwy {
namespace HWY_NAMESPACE {

void MyFunction(const float* HWY_RESTRICT in, float* HWY_RESTRICT out, size_t n) {
    const ScalableTag<float> d;
    // ... implementation
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Export creates dispatch table (targets.h:219-226)
HWY_EXPORT(MyFunction);

// Call via dynamic dispatch
void CallMyFunction(const float* in, float* out, size_t n) {
    HWY_DYNAMIC_DISPATCH(MyFunction)(in, out, n);
}
```

**Dispatch Table Structure:**

```cpp
// Generated by HWY_EXPORT macro
struct MyFunction_FunctionTable {
    // Index 0: Uninitialized (calls FunctionCache to detect CPU)
    // Index 1-N: Target-specific implementations
    // Index N+1: Fallback (SCALAR or EMU128)
    void (*table[HWY_MAX_DYNAMIC_TARGETS + 2])(...);
};

// ChosenTarget (targets.h:341-384) manages runtime selection
struct ChosenTarget {
    std::atomic<int64_t> mask_{1};  // Bitmask of available targets

    void Update(int64_t targets);   // Called on first dispatch
    size_t GetIndex() const;        // Returns table index for current CPU
};
```

---

## 2. Integration Strategy

### 2.1 Architecture Overview

```
Bud Flow Lang Integration Architecture:
┌────────────────────────────────────────────────────────────────┐
│                    Python User Code                            │
│  @flow.kernel                                                  │
│  def compute(x): return (flow(x) * 2 + 1).sqrt()               │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              Bud Flow Lang Frontend                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ AST      │→ │ Type         │→ │ IR Builder               │  │
│  │ Analyzer │  │ Inferrer     │  │ (Lazy Evaluation Graph)  │  │
│  └──────────┘  └──────────────┘  └──────────────────────────┘  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              Code Generation Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Highway Code Generator                                    │  │
│  │ - Maps IR ops to Highway functions                        │  │
│  │ - Generates HWY_FOREACH_TARGET structure                  │  │
│  │ - Creates HWY_EXPORT dispatch tables                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              Highway Backend                                   │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Memory   │  │ Operations   │  │ Multi-Target Dispatch     │ │
│  │ Manager  │  │ (math, etc)  │  │ (runtime ISA selection)   │ │
│  └──────────┘  └──────────────┘  └───────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Integration Principles

1. **Zero-Copy Data Flow**: Use Highway's aligned allocators for all array storage
2. **Lazy Evaluation**: Build expression graph before generating Highway code
3. **Fused Operations**: Combine operations into single Highway loops where possible
4. **Runtime Dispatch**: Leverage `HWY_DYNAMIC_DISPATCH` for ISA selection
5. **Type Safety**: Map Python types to Highway lane types at JIT time

---

## 3. Core Integration Layer

### 3.1 C++ Integration Header

```cpp
// bud_flow_lang/integration/highway_bridge.h
#pragma once

#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/math/math-inl.h"

namespace bud_flow {

// Forward declaration of tag type
template <typename T>
using FlowTag = hwy::HWY_NAMESPACE::ScalableTag<T>;

// Memory management wrapper
template <typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t size)
        : data_(hwy::AllocateAligned<T>(size))
        , size_(size) {}

    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }
    size_t size() const { return size_; }

    // Check alignment for Highway operations
    bool is_aligned() const {
        return hwy::IsAligned(data_.get());
    }

private:
    hwy::AlignedFreeUniquePtr<T[]> data_;
    size_t size_;
};

// Kernel signature types
using KernelFunc = void(*)(const void* const* inputs,
                           void* const* outputs,
                           const size_t* sizes,
                           size_t num_elements);

// Dispatch table for generated kernels
struct KernelDispatcher {
    KernelFunc table[32];  // Up to 32 targets
    size_t num_targets;

    void call(const void* const* inputs,
              void* const* outputs,
              const size_t* sizes,
              size_t num_elements) const {
        const size_t idx = hwy::GetChosenTarget().GetIndex();
        table[idx](inputs, outputs, sizes, num_elements);
    }
};

}  // namespace bud_flow
```

### 3.2 Type Mapping System

```cpp
// bud_flow_lang/integration/type_mapping.h
#pragma once

#include "hwy/highway.h"

namespace bud_flow {

// Python dtype to Highway lane type mapping
enum class FlowDType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64
};

// Type traits for code generation
template <FlowDType DT> struct LaneTypeFor;
template <> struct LaneTypeFor<FlowDType::Float32>  { using type = float; };
template <> struct LaneTypeFor<FlowDType::Float64>  { using type = double; };
template <> struct LaneTypeFor<FlowDType::Float16>  { using type = hwy::float16_t; };
template <> struct LaneTypeFor<FlowDType::BFloat16> { using type = hwy::bfloat16_t; };
template <> struct LaneTypeFor<FlowDType::Int8>     { using type = int8_t; };
template <> struct LaneTypeFor<FlowDType::Int16>    { using type = int16_t; };
template <> struct LaneTypeFor<FlowDType::Int32>    { using type = int32_t; };
template <> struct LaneTypeFor<FlowDType::Int64>    { using type = int64_t; };
template <> struct LaneTypeFor<FlowDType::UInt8>    { using type = uint8_t; };
template <> struct LaneTypeFor<FlowDType::UInt16>   { using type = uint16_t; };
template <> struct LaneTypeFor<FlowDType::UInt32>   { using type = uint32_t; };
template <> struct LaneTypeFor<FlowDType::UInt64>   { using type = uint64_t; };

// Runtime type info
struct TypeInfo {
    FlowDType dtype;
    size_t element_size;
    const char* name;
    bool is_float;
    bool is_signed;
};

constexpr TypeInfo kTypeInfoTable[] = {
    {FlowDType::Float32,  4, "f32",  true,  true},
    {FlowDType::Float64,  8, "f64",  true,  true},
    {FlowDType::Float16,  2, "f16",  true,  true},
    {FlowDType::BFloat16, 2, "bf16", true,  true},
    {FlowDType::Int8,     1, "i8",   false, true},
    {FlowDType::Int16,    2, "i16",  false, true},
    {FlowDType::Int32,    4, "i32",  false, true},
    {FlowDType::Int64,    8, "i64",  false, true},
    {FlowDType::UInt8,    1, "u8",   false, false},
    {FlowDType::UInt16,   2, "u16",  false, false},
    {FlowDType::UInt32,   4, "u32",  false, false},
    {FlowDType::UInt64,   8, "u64",  false, false},
};

}  // namespace bud_flow
```

---

## 4. Code Generation Patterns

### 4.1 Basic Operation Template

For a simple element-wise operation like `y = x * 2 + 1`:

```cpp
// Generated code pattern
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "generated_kernel.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace bud_flow {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void kernel_mul2_add1(const float* HWY_RESTRICT in,
                      float* HWY_RESTRICT out,
                      size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    // Broadcast constants
    const auto k2 = hn::Set(d, 2.0f);
    const auto k1 = hn::Set(d, 1.0f);

    size_t i = 0;

    // Main vectorized loop
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, in + i);
        const auto result = hn::MulAdd(v, k2, k1);  // FMA: v * 2 + 1
        hn::StoreU(result, d, out + i);
    }

    // Remainder handling with LoadN/StoreN
    if (i < count) {
        const size_t remaining = count - i;
        const auto v = hn::LoadN(d, in + i, remaining);
        const auto result = hn::MulAdd(v, k2, k1);
        hn::StoreN(result, d, out + i, remaining);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace bud_flow
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace bud_flow {
HWY_EXPORT(kernel_mul2_add1);

void call_kernel_mul2_add1(const float* in, float* out, size_t count) {
    HWY_DYNAMIC_DISPATCH(kernel_mul2_add1)(in, out, count);
}
}  // namespace bud_flow
#endif
```

### 4.2 Fused Multi-Operation Pattern

For fused operations like `y = sqrt(x * 2 + 1)`:

```cpp
void kernel_fused_mul_add_sqrt(const float* HWY_RESTRICT in,
                               float* HWY_RESTRICT out,
                               size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    const auto k2 = hn::Set(d, 2.0f);
    const auto k1 = hn::Set(d, 1.0f);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, in + i);
        // Fused: sqrt(v * 2 + 1) - single pass through memory
        const auto fma_result = hn::MulAdd(v, k2, k1);
        const auto sqrt_result = hn::Sqrt(fma_result);
        hn::StoreU(sqrt_result, d, out + i);
    }

    // Remainder
    if (i < count) {
        const size_t remaining = count - i;
        const auto v = hn::LoadN(d, in + i, remaining);
        const auto fma_result = hn::MulAdd(v, k2, k1);
        const auto sqrt_result = hn::Sqrt(fma_result);
        hn::StoreN(sqrt_result, d, out + i, remaining);
    }
}
```

### 4.3 Reduction Pattern (Sum, Dot Product)

```cpp
float kernel_sum(const float* HWY_RESTRICT in, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto acc = hn::Zero(d);
    size_t i = 0;

    // Vectorized accumulation
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, in + i);
        acc = hn::Add(acc, v);
    }

    // Horizontal sum of accumulator
    float result = hn::ReduceSum(d, acc);

    // Scalar remainder
    for (; i < count; ++i) {
        result += in[i];
    }

    return result;
}

float kernel_dot_product(const float* HWY_RESTRICT a,
                         const float* HWY_RESTRICT b,
                         size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    auto acc = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= count; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        acc = hn::MulAdd(va, vb, acc);  // FMA accumulation
    }

    float result = hn::ReduceSum(d, acc);

    for (; i < count; ++i) {
        result += a[i] * b[i];
    }

    return result;
}
```

### 4.4 Transcendental Function Pattern

Using Highway's math library (contrib/math/math-inl.h):

```cpp
#include "hwy/contrib/math/math-inl.h"

void kernel_exp(const float* HWY_RESTRICT in,
                float* HWY_RESTRICT out,
                size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto v = hn::LoadU(d, in + i);
        const auto result = hn::Exp(d, v);  // From math-inl.h
        hn::StoreU(result, d, out + i);
    }

    if (i < count) {
        const size_t remaining = count - i;
        const auto v = hn::LoadN(d, in + i, remaining);
        const auto result = hn::Exp(d, v);
        hn::StoreN(result, d, out + i, remaining);
    }
}
```

### 4.5 Masked Operation Pattern

For conditional operations:

```cpp
void kernel_where(const float* HWY_RESTRICT cond,
                  const float* HWY_RESTRICT a,
                  const float* HWY_RESTRICT b,
                  float* HWY_RESTRICT out,
                  size_t count) {
    const hn::ScalableTag<float> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);

    size_t i = 0;
    for (; i + N <= count; i += N) {
        const auto vc = hn::LoadU(d, cond + i);
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);

        // Create mask from condition (non-zero = true)
        const auto mask = hn::Ne(vc, zero);
        const auto result = hn::IfThenElse(mask, va, vb);

        hn::StoreU(result, d, out + i);
    }

    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = (cond[i] != 0.0f) ? a[i] : b[i];
    }
}
```

---

## 5. Multi-Target Dispatch Integration

### 5.1 Target Selection Strategy

```cpp
// bud_flow_lang/integration/target_selector.h
#pragma once

#include "hwy/targets.h"

namespace bud_flow {

struct TargetCapabilities {
    int64_t target;
    const char* name;
    size_t vector_bytes;
    bool has_fma;
    bool has_avx512;
    bool has_sve;
    bool has_bf16;
};

// Query available targets at runtime
inline std::vector<TargetCapabilities> GetAvailableTargets() {
    std::vector<TargetCapabilities> result;

    for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
        TargetCapabilities caps;
        caps.target = target;
        caps.name = hwy::TargetName(target);

        // Determine capabilities
        switch (target) {
            case HWY_AVX3:
            case HWY_AVX3_DL:
            case HWY_AVX3_ZEN4:
            case HWY_AVX3_SPR:
            case HWY_AVX10_2:
                caps.vector_bytes = 64;
                caps.has_fma = true;
                caps.has_avx512 = true;
                caps.has_sve = false;
                caps.has_bf16 = (target >= HWY_AVX3_ZEN4);
                break;
            case HWY_AVX2:
                caps.vector_bytes = 32;
                caps.has_fma = true;
                caps.has_avx512 = false;
                caps.has_sve = false;
                caps.has_bf16 = false;
                break;
            case HWY_SVE:
            case HWY_SVE2:
                caps.vector_bytes = 0;  // Scalable
                caps.has_fma = true;
                caps.has_avx512 = false;
                caps.has_sve = true;
                caps.has_bf16 = (target == HWY_SVE2);
                break;
            // ... other targets
            default:
                caps.vector_bytes = 16;
                caps.has_fma = false;
                caps.has_avx512 = false;
                caps.has_sve = false;
                caps.has_bf16 = false;
        }

        result.push_back(caps);
    }

    return result;
}

// Get best target for current CPU
inline int64_t GetBestTarget() {
    auto targets = hwy::SupportedAndGeneratedTargets();
    return targets.empty() ? HWY_SCALAR : targets[0];
}

}  // namespace bud_flow
```

### 5.2 Kernel Dispatch Table Generation

```cpp
// Template for generating dispatch tables
#define BUD_FLOW_EXPORT_KERNEL(name) \
    HWY_EXPORT(name); \
    void call_##name(const void* const* inputs, \
                     void* const* outputs, \
                     const size_t* sizes, \
                     size_t num_elements) { \
        HWY_DYNAMIC_DISPATCH(name)(inputs, outputs, sizes, num_elements); \
    }

// Example generated kernel with full dispatch
namespace bud_flow {
namespace HWY_NAMESPACE {

void user_kernel_0(const void* const* inputs,
                   void* const* outputs,
                   const size_t* sizes,
                   size_t num_elements) {
    const float* in0 = static_cast<const float*>(inputs[0]);
    float* out0 = static_cast<float*>(outputs[0]);

    const hn::ScalableTag<float> d;
    // ... kernel body
}

}  // namespace HWY_NAMESPACE
}  // namespace bud_flow

#if HWY_ONCE
BUD_FLOW_EXPORT_KERNEL(user_kernel_0)
#endif
```

---

## 6. Memory Management Integration

### 6.1 Arena Allocator with Highway Alignment

```cpp
// bud_flow_lang/integration/arena_allocator.h
#pragma once

#include "hwy/aligned_allocator.h"
#include <vector>
#include <memory>

namespace bud_flow {

// Arena allocator optimized for SIMD operations
class SIMDArena {
public:
    // Alignment matches Highway's HWY_ALIGNMENT (128 bytes)
    static constexpr size_t kAlignment = HWY_ALIGNMENT;
    static constexpr size_t kBlockSize = 1 << 20;  // 1MB blocks

    struct Block {
        alignas(HWY_ALIGNMENT) char data[kBlockSize];
        size_t offset = 0;
    };

    SIMDArena() = default;

    // Non-copyable, movable
    SIMDArena(const SIMDArena&) = delete;
    SIMDArena& operator=(const SIMDArena&) = delete;
    SIMDArena(SIMDArena&&) = default;
    SIMDArena& operator=(SIMDArena&&) = default;

    // Allocate aligned memory from arena
    template <typename T>
    T* allocate(size_t count) {
        const size_t bytes = count * sizeof(T);
        // Round up to alignment
        const size_t aligned_bytes = (bytes + kAlignment - 1) & ~(kAlignment - 1);

        // Find or create block with space
        if (blocks_.empty() ||
            blocks_.back()->offset + aligned_bytes > kBlockSize) {
            blocks_.push_back(std::make_unique<Block>());
        }

        Block& block = *blocks_.back();
        char* ptr = block.data + block.offset;
        block.offset += aligned_bytes;

        // Verify alignment
        HWY_DASSERT(hwy::IsAligned(reinterpret_cast<T*>(ptr)));
        return reinterpret_cast<T*>(ptr);
    }

    // Reset arena for reuse (doesn't deallocate)
    void reset() {
        for (auto& block : blocks_) {
            block->offset = 0;
        }
    }

    // Total allocated bytes
    size_t total_allocated() const {
        size_t total = 0;
        for (const auto& block : blocks_) {
            total += block->offset;
        }
        return total;
    }

private:
    std::vector<std::unique_ptr<Block>> blocks_;
};

// Thread-local arena for kernel execution
inline SIMDArena& GetKernelArena() {
    thread_local SIMDArena arena;
    return arena;
}

}  // namespace bud_flow
```

### 6.2 NumPy Array Integration

```cpp
// bud_flow_lang/integration/numpy_bridge.h
#pragma once

#include <cstdint>

namespace bud_flow {

struct ArrayView {
    void* data;           // Pointer to data
    size_t* shape;        // Shape array
    size_t* strides;      // Strides array (in bytes)
    size_t ndim;          // Number of dimensions
    size_t itemsize;      // Size of each element
    bool contiguous;      // Whether memory is contiguous
    bool aligned;         // Whether data is aligned for SIMD

    // Check if array is suitable for direct Highway operations
    bool is_simd_ready() const {
        return contiguous && aligned;
    }

    // Get alignment status
    size_t alignment() const {
        return reinterpret_cast<uintptr_t>(data) % HWY_ALIGNMENT;
    }
};

// Convert numpy array to view (called from Python binding)
ArrayView from_numpy(void* data, size_t* shape, size_t* strides,
                     size_t ndim, size_t itemsize) {
    ArrayView view;
    view.data = data;
    view.shape = shape;
    view.strides = strides;
    view.ndim = ndim;
    view.itemsize = itemsize;

    // Check contiguity
    view.contiguous = true;
    size_t expected_stride = itemsize;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            view.contiguous = false;
            break;
        }
        expected_stride *= shape[i];
    }

    // Check alignment
    view.aligned = (reinterpret_cast<uintptr_t>(data) % HWY_ALIGNMENT) == 0;

    return view;
}

}  // namespace bud_flow
```

---

## 7. Operation Mapping

### 7.1 Arithmetic Operations

| Bud Flow Lang | Highway Function | Notes |
|--------------|------------------|-------|
| `x + y` | `Add(v1, v2)` | Element-wise |
| `x - y` | `Sub(v1, v2)` | Element-wise |
| `x * y` | `Mul(v1, v2)` | Element-wise |
| `x / y` | `Div(v1, v2)` | Element-wise |
| `x * y + z` | `MulAdd(v1, v2, v3)` | FMA - use when available |
| `x * y - z` | `MulSub(v1, v2, v3)` | FMS |
| `-x * y + z` | `NegMulAdd(v1, v2, v3)` | FNMA |
| `-x` | `Neg(v)` | Negate |
| `abs(x)` | `Abs(v)` | Absolute value |

### 7.2 Math Functions (from contrib/math/math-inl.h)

| Bud Flow Lang | Highway Function | Max Error (ULP) |
|--------------|------------------|-----------------|
| `sqrt(x)` | `Sqrt(v)` | 0.5 |
| `exp(x)` | `Exp(d, v)` | 1 |
| `exp2(x)` | `Exp2(d, v)` | 2 |
| `expm1(x)` | `Expm1(d, v)` | 4 |
| `log(x)` | `Log(d, v)` | 4 |
| `log2(x)` | `Log2(d, v)` | 2 |
| `log10(x)` | `Log10(d, v)` | 2 |
| `log1p(x)` | `Log1p(d, v)` | 2 |
| `sin(x)` | `Sin(d, v)` | 3 |
| `cos(x)` | `Cos(d, v)` | 3 |
| `sinh(x)` | `Sinh(d, v)` | 4 |
| `tanh(x)` | `Tanh(d, v)` | 4 |
| `asin(x)` | `Asin(d, v)` | 2 |
| `acos(x)` | `Acos(d, v)` | 2 |
| `atan(x)` | `Atan(d, v)` | 3 |
| `atan2(y, x)` | `Atan2(d, y, x)` | 3 |
| `hypot(x, y)` | `Hypot(d, x, y)` | 4 |

### 7.3 Comparison and Logical Operations

| Bud Flow Lang | Highway Function | Returns |
|--------------|------------------|---------|
| `x == y` | `Eq(v1, v2)` | Mask |
| `x != y` | `Ne(v1, v2)` | Mask |
| `x < y` | `Lt(v1, v2)` | Mask |
| `x <= y` | `Le(v1, v2)` | Mask |
| `x > y` | `Gt(v1, v2)` | Mask |
| `x >= y` | `Ge(v1, v2)` | Mask |
| `where(c, a, b)` | `IfThenElse(mask, va, vb)` | Vector |
| `x & y` | `And(v1, v2)` | Vector (bitwise) |
| `x \| y` | `Or(v1, v2)` | Vector (bitwise) |
| `x ^ y` | `Xor(v1, v2)` | Vector (bitwise) |
| `~x` | `Not(v)` | Vector (bitwise) |

### 7.4 Reduction Operations

| Bud Flow Lang | Highway Function | Notes |
|--------------|------------------|-------|
| `sum(x)` | `ReduceSum(d, v)` | Horizontal sum |
| `min(x)` | `ReduceMin(d, v)` | Horizontal min |
| `max(x)` | `ReduceMax(d, v)` | Horizontal max |
| `all(mask)` | `AllTrue(d, mask)` | All lanes true |
| `any(mask)` | `AllFalse(d, mask)` == false | Any lane true |
| `count(mask)` | `CountTrue(d, mask)` | Count true lanes |

### 7.5 Memory Operations

| Bud Flow Lang | Highway Function | Notes |
|--------------|------------------|-------|
| Load aligned | `Load(d, ptr)` | Requires alignment |
| Load unaligned | `LoadU(d, ptr)` | No alignment required |
| Load partial | `LoadN(d, ptr, n)` | Load n elements |
| Store aligned | `Store(v, d, ptr)` | Requires alignment |
| Store unaligned | `StoreU(v, d, ptr)` | No alignment required |
| Store partial | `StoreN(v, d, ptr, n)` | Store n elements |
| Masked load | `MaskedLoad(mask, d, ptr)` | Load where mask=true |
| Masked store | `BlendedStore(v, mask, d, ptr)` | Store where mask=true |

---

## 8. Extension Requirements

### 8.1 Custom Operations (Not in Highway)

The following operations need custom implementation:

```cpp
// bud_flow_lang/extensions/custom_ops.h

namespace bud_flow {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Power function (not in Highway math)
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V Pow(D d, V base, V exponent) {
    // x^y = exp(y * log(x))
    // Handle special cases for negative base
    const auto log_base = hn::Log(d, hn::Abs(base));
    const auto result = hn::Exp(d, hn::Mul(exponent, log_base));

    // Fix sign for negative base with integer exponent
    // (simplified - full implementation needs integer detection)
    return result;
}

// Sigmoid: 1 / (1 + exp(-x))
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V Sigmoid(D d, V x) {
    const auto one = hn::Set(d, 1.0f);
    const auto neg_x = hn::Neg(x);
    const auto exp_neg_x = hn::Exp(d, neg_x);
    return hn::Div(one, hn::Add(one, exp_neg_x));
}

// ReLU: max(0, x)
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V ReLU(D d, V x) {
    return hn::Max(hn::Zero(d), x);
}

// Leaky ReLU: x if x > 0, else alpha * x
template <class D, class V = hn::VFromD<D>, typename T = hn::TFromD<D>>
HWY_INLINE V LeakyReLU(D d, V x, T alpha = 0.01f) {
    const auto zero = hn::Zero(d);
    const auto alpha_v = hn::Set(d, alpha);
    const auto neg_part = hn::Mul(x, alpha_v);
    const auto mask = hn::Gt(x, zero);
    return hn::IfThenElse(mask, x, neg_part);
}

// Softmax normalization helper (for use in loops)
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V ExpNormalized(D d, V x, V max_val) {
    return hn::Exp(d, hn::Sub(x, max_val));
}

// Clamp: min(max(x, lo), hi)
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V Clamp(V x, V lo, V hi) {
    return hn::Min(hn::Max(x, lo), hi);
}

// Floor (truncate toward negative infinity)
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V Floor(D d, V x) {
    return hn::Floor(x);  // Available in Highway
}

// Ceil (truncate toward positive infinity)
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V Ceil(D d, V x) {
    return hn::Ceil(x);  // Available in Highway
}

// Round to nearest
template <class D, class V = hn::VFromD<D>>
HWY_INLINE V Round(D d, V x) {
    return hn::Round(x);  // Available in Highway
}

}  // namespace HWY_NAMESPACE
}  // namespace bud_flow
```

### 8.2 Broadcasting Support

```cpp
// bud_flow_lang/extensions/broadcast.h

namespace bud_flow {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Broadcast scalar to vector
template <class D, typename T = hn::TFromD<D>>
HWY_INLINE hn::VFromD<D> Broadcast(D d, T scalar) {
    return hn::Set(d, scalar);
}

// Broadcast along axis (for matrix operations)
// This generates code for row/column broadcasting
enum class BroadcastAxis { Row, Column };

template <class D, BroadcastAxis Axis>
struct BroadcastHelper;

// Row broadcast: same value for all columns in a row
template <class D>
struct BroadcastHelper<D, BroadcastAxis::Row> {
    static void apply(D d,
                      const hn::TFromD<D>* HWY_RESTRICT row_values,
                      hn::TFromD<D>* HWY_RESTRICT out,
                      size_t rows, size_t cols) {
        const size_t N = hn::Lanes(d);

        for (size_t r = 0; r < rows; ++r) {
            const auto val = hn::Set(d, row_values[r]);
            size_t c = 0;
            for (; c + N <= cols; c += N) {
                hn::StoreU(val, d, out + r * cols + c);
            }
            // Remainder
            for (; c < cols; ++c) {
                out[r * cols + c] = row_values[r];
            }
        }
    }
};

}  // namespace HWY_NAMESPACE
}  // namespace bud_flow
```

### 8.3 Complex Number Support

```cpp
// bud_flow_lang/extensions/complex.h

namespace bud_flow {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Complex represented as interleaved real/imag pairs
// [r0, i0, r1, i1, r2, i2, ...]

template <class D>
struct ComplexOps {
    using V = hn::VFromD<D>;
    using T = hn::TFromD<D>;

    // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    static HWY_INLINE void Mul(D d,
                               const T* HWY_RESTRICT a,
                               const T* HWY_RESTRICT b,
                               T* HWY_RESTRICT out,
                               size_t count) {
        const size_t N = hn::Lanes(d);

        for (size_t i = 0; i + N * 2 <= count * 2; i += N * 2) {
            // Load interleaved
            const auto a_real = hn::LoadU(d, a + i);      // Even indices
            const auto a_imag = hn::LoadU(d, a + i + N);  // Odd indices
            const auto b_real = hn::LoadU(d, b + i);
            const auto b_imag = hn::LoadU(d, b + i + N);

            // Real: ac - bd
            const auto ac = hn::Mul(a_real, b_real);
            const auto bd = hn::Mul(a_imag, b_imag);
            const auto out_real = hn::Sub(ac, bd);

            // Imag: ad + bc
            const auto ad = hn::Mul(a_real, b_imag);
            const auto bc = hn::Mul(a_imag, b_real);
            const auto out_imag = hn::Add(ad, bc);

            hn::StoreU(out_real, d, out + i);
            hn::StoreU(out_imag, d, out + i + N);
        }
        // Handle remainder...
    }

    // Complex magnitude: sqrt(real^2 + imag^2)
    static HWY_INLINE void Abs(D d,
                               const T* HWY_RESTRICT in,
                               T* HWY_RESTRICT out,
                               size_t count) {
        // Implementation using Hypot from Highway math
    }
};

}  // namespace HWY_NAMESPACE
}  // namespace bud_flow
```

---

## 9. Performance Considerations

### 9.1 Memory Access Patterns

```
CRITICAL: Memory access is the primary bottleneck for SIMD code

Optimal Pattern:
┌─────────────────────────────────────────────────────────────┐
│  Sequential, aligned access with prefetching               │
│  for (i = 0; i + N <= count; i += N) {                     │
│      // Prefetch ahead (if supported)                       │
│      __builtin_prefetch(ptr + i + 8 * N, 0, 3);            │
│      v = Load(d, ptr + i);  // Aligned load                │
│      // ... process                                         │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘

Suboptimal Patterns to Avoid:
┌─────────────────────────────────────────────────────────────┐
│  1. Strided access (e.g., column traversal of row-major)   │
│  2. Random/scatter access                                   │
│  3. Unaligned loads when alignment is possible             │
│  4. Mixing scalar and vector operations                    │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Loop Optimization Guidelines

```cpp
// GOOD: Single fused loop (1 memory pass)
for (i = 0; i + N <= count; i += N) {
    auto v = LoadU(d, in + i);
    v = MulAdd(v, k2, k1);  // x * 2 + 1
    v = Sqrt(v);             // sqrt(...)
    StoreU(v, d, out + i);
}

// BAD: Separate loops (multiple memory passes)
for (i = 0; i + N <= count; i += N) {
    auto v = LoadU(d, in + i);
    StoreU(MulAdd(v, k2, k1), d, temp + i);  // Extra memory traffic
}
for (i = 0; i + N <= count; i += N) {
    auto v = LoadU(d, temp + i);  // Reload from memory
    StoreU(Sqrt(v), d, out + i);
}
```

### 9.3 FMA Utilization

```cpp
// Always prefer FMA when available
// Instead of:
auto result = Add(Mul(a, b), c);

// Use:
auto result = MulAdd(a, b, c);  // Single instruction, higher precision

// FMA chains for polynomial evaluation (Estrin's scheme in Highway)
// See: contrib/math/math-inl.h:372-520
```

### 9.4 Reduction Optimization

```cpp
// For large reductions, use multiple accumulators
template <class D>
float sum_optimized(D d, const float* HWY_RESTRICT in, size_t count) {
    const size_t N = Lanes(d);

    // 4 accumulators reduce latency by allowing independent FP adds
    auto acc0 = Zero(d);
    auto acc1 = Zero(d);
    auto acc2 = Zero(d);
    auto acc3 = Zero(d);

    size_t i = 0;
    for (; i + 4 * N <= count; i += 4 * N) {
        acc0 = Add(acc0, LoadU(d, in + i + 0 * N));
        acc1 = Add(acc1, LoadU(d, in + i + 1 * N));
        acc2 = Add(acc2, LoadU(d, in + i + 2 * N));
        acc3 = Add(acc3, LoadU(d, in + i + 3 * N));
    }

    // Combine accumulators
    acc0 = Add(acc0, acc1);
    acc2 = Add(acc2, acc3);
    acc0 = Add(acc0, acc2);

    // Handle remainder and horizontal sum...
    return ReduceSum(d, acc0);
}
```

### 9.5 Branch Avoidance

```cpp
// AVOID: Branches in hot loops
for (i = 0; i < count; i += N) {
    auto v = LoadU(d, in + i);
    if (use_fast_path) {  // Branch on every iteration!
        // ...
    } else {
        // ...
    }
}

// PREFER: Hoist branches outside loop
if (use_fast_path) {
    for (i = 0; i < count; i += N) {
        auto v = LoadU(d, in + i);
        // fast path only
    }
} else {
    for (i = 0; i < count; i += N) {
        auto v = LoadU(d, in + i);
        // slow path only
    }
}

// For data-dependent branches, use masked operations
for (i = 0; i < count; i += N) {
    auto v = LoadU(d, in + i);
    auto mask = Gt(v, threshold);
    auto result = IfThenElse(mask, fast_result, slow_result);
    StoreU(result, d, out + i);
}
```

---

## 10. Stability Guidelines

### 10.1 Error Handling Strategy

```cpp
// bud_flow_lang/integration/error_handling.h

namespace bud_flow {

enum class ErrorCode {
    Success = 0,
    InvalidInput,
    AllocationFailed,
    AlignmentError,
    SizeOverflow,
    UnsupportedOperation,
    NumericalError
};

struct KernelResult {
    ErrorCode code;
    const char* message;

    bool ok() const { return code == ErrorCode::Success; }
};

// Pre-execution validation
inline KernelResult validate_inputs(const ArrayView* inputs,
                                    size_t num_inputs,
                                    const size_t* expected_sizes) {
    for (size_t i = 0; i < num_inputs; ++i) {
        // Check null
        if (!inputs[i].data) {
            return {ErrorCode::InvalidInput, "Null input pointer"};
        }

        // Check size
        size_t total_elements = 1;
        for (size_t d = 0; d < inputs[i].ndim; ++d) {
            // Overflow check
            if (total_elements > SIZE_MAX / inputs[i].shape[d]) {
                return {ErrorCode::SizeOverflow, "Size overflow"};
            }
            total_elements *= inputs[i].shape[d];
        }

        // Alignment warning (not error - we handle unaligned)
        if (!inputs[i].aligned) {
            // Log warning but continue - LoadU handles unaligned
        }
    }

    return {ErrorCode::Success, nullptr};
}

}  // namespace bud_flow
```

### 10.2 Numerical Stability

```cpp
// Guidelines for numerically stable SIMD code

// 1. Use Kahan summation for large reductions
template <class D>
float kahan_sum(D d, const float* HWY_RESTRICT in, size_t count) {
    auto sum = Zero(d);
    auto compensation = Zero(d);  // Running compensation for lost low-order bits

    const size_t N = Lanes(d);
    for (size_t i = 0; i + N <= count; i += N) {
        auto v = LoadU(d, in + i);
        auto y = Sub(v, compensation);
        auto t = Add(sum, y);
        compensation = Sub(Sub(t, sum), y);
        sum = t;
    }

    return ReduceSum(d, sum);
}

// 2. Handle denormals consistently
// Option: Flush denormals to zero for performance
#ifdef _MM_FLUSH_ZERO_ON
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

// 3. NaN propagation - Highway handles this, but verify
template <class D, class V>
HWY_INLINE bool has_nan(D d, V v) {
    // NaN != NaN is true
    auto nan_mask = Ne(v, v);
    return !AllFalse(d, nan_mask);
}

// 4. Overflow protection for exp()
template <class D, class V>
HWY_INLINE V safe_exp(D d, V x) {
    const auto max_val = Set(d, 88.0f);  // Approximate overflow threshold
    const auto min_val = Set(d, -88.0f);
    auto clamped = Clamp(x, min_val, max_val);
    return Exp(d, clamped);
}
```

### 10.3 Thread Safety

```cpp
// Highway dispatch is thread-safe after first initialization
// BUT: Arena allocator must be thread-local

// Safe pattern:
void parallel_kernel(const float* in, float* out, size_t n) {
    #pragma omp parallel
    {
        // Thread-local arena
        SIMDArena& arena = GetKernelArena();

        #pragma omp for
        for (size_t chunk = 0; chunk < n; chunk += CHUNK_SIZE) {
            size_t chunk_end = std::min(chunk + CHUNK_SIZE, n);
            // Use HWY_DYNAMIC_DISPATCH - it's thread-safe
            process_chunk(in + chunk, out + chunk, chunk_end - chunk);
        }

        arena.reset();  // Reset arena after each parallel region
    }
}

// Note: First call to HWY_DYNAMIC_DISPATCH initializes dispatch table
// This is thread-safe due to atomic operations in ChosenTarget
```

---

## 11. Edge Cases and Mitigations

### 11.1 Unaligned Memory

```cpp
// Detection
bool is_aligned(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % HWY_ALIGNMENT == 0;
}

// Handling: Always use LoadU/StoreU unless alignment is guaranteed
void process(const float* in, float* out, size_t n) {
    const ScalableTag<float> d;
    const size_t N = Lanes(d);

    // For user-provided arrays, always assume unaligned
    for (size_t i = 0; i + N <= n; i += N) {
        auto v = LoadU(d, in + i);   // Unaligned load
        // ... process
        StoreU(v, d, out + i);       // Unaligned store
    }
}

// Optimization for known-aligned internal buffers
void process_aligned(const float* HWY_RESTRICT in,
                     float* HWY_RESTRICT out,
                     size_t n) {
    HWY_DASSERT(is_aligned(in) && is_aligned(out));

    const ScalableTag<float> d;
    const size_t N = Lanes(d);

    for (size_t i = 0; i + N <= n; i += N) {
        auto v = Load(d, in + i);    // Aligned load (faster)
        // ... process
        Store(v, d, out + i);        // Aligned store (faster)
    }
}
```

### 11.2 Size Edge Cases

```cpp
// Handle all size cases correctly
void robust_kernel(const float* in, float* out, size_t n) {
    const ScalableTag<float> d;
    const size_t N = Lanes(d);

    // Case 1: n == 0
    if (n == 0) return;

    // Case 2: n < N (smaller than one vector)
    if (n < N) {
        // Use LoadN/StoreN for partial vector
        auto v = LoadN(d, in, n);
        // ... process
        StoreN(v, d, out, n);
        return;
    }

    // Case 3: n >= N (at least one full vector)
    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto v = LoadU(d, in + i);
        // ... process
        StoreU(v, d, out + i);
    }

    // Case 4: Remainder (n % N != 0)
    if (i < n) {
        const size_t remaining = n - i;
        auto v = LoadN(d, in + i, remaining);
        // ... process
        StoreN(v, d, out + i, remaining);
    }
}
```

### 11.3 Type Promotion/Demotion

```cpp
// Safe type conversions
template <class DFrom, class DTo>
void convert_type(DFrom d_from, DTo d_to,
                  const TFromD<DFrom>* in,
                  TFromD<DTo>* out,
                  size_t n) {
    const size_t N_from = Lanes(d_from);
    const size_t N_to = Lanes(d_to);

    // For promotion (e.g., float16 -> float32)
    if constexpr (sizeof(TFromD<DTo>) > sizeof(TFromD<DFrom>)) {
        for (size_t i = 0; i + N_from <= n; i += N_from) {
            auto v_from = LoadU(d_from, in + i);
            auto v_to = PromoteTo(d_to, v_from);
            StoreU(v_to, d_to, out + i);
        }
    }
    // For demotion (e.g., float32 -> float16)
    else {
        for (size_t i = 0; i + N_to <= n; i += N_to) {
            auto v_from = LoadU(d_from, in + i);
            auto v_to = DemoteTo(d_to, v_from);
            StoreU(v_to, d_to, out + i);
        }
    }
}
```

### 11.4 Scalable Vector Edge Cases (SVE/RVV)

```cpp
// For scalable vectors, N is not known at compile time
// Always use Lanes(d) at runtime, never hardcode vector sizes

void scalable_safe(const float* in, float* out, size_t n) {
    const ScalableTag<float> d;

    // WRONG: Assuming fixed size
    // constexpr size_t N = 8;  // BAD for SVE!

    // CORRECT: Query at runtime
    const size_t N = Lanes(d);

    for (size_t i = 0; i + N <= n; i += N) {
        // Use N from Lanes(d)
    }
}

// For compile-time decisions, use HWY_MAX_LANES_D
template <class D>
void preallocate_buffer(D d) {
    // Allocate enough for maximum possible vector size
    constexpr size_t kMaxLanes = HWY_MAX_LANES_D(D);
    alignas(HWY_ALIGNMENT) TFromD<D> buffer[kMaxLanes];
}
```

### 11.5 Mixed-Precision Operations

```cpp
// When mixing f32 and f64, explicit conversion is required
void mixed_precision(const float* f32_in,
                     const double* f64_in,
                     float* out,
                     size_t n) {
    const ScalableTag<float> df32;
    const ScalableTag<double> df64;
    const size_t N32 = Lanes(df32);
    const size_t N64 = Lanes(df64);

    // f64 vectors are half the size of f32 vectors
    HWY_DASSERT(N32 == 2 * N64);

    for (size_t i = 0; i + N32 <= n; i += N32) {
        // Load f32
        auto v32 = LoadU(df32, f32_in + i);

        // Load f64 and convert to f32
        auto v64_lo = LoadU(df64, f64_in + i);
        auto v64_hi = LoadU(df64, f64_in + i + N64);
        auto v64_as_f32 = Combine(df32,
                                   DemoteTo(Half<decltype(df32)>(), v64_hi),
                                   DemoteTo(Half<decltype(df32)>(), v64_lo));

        // Now both are f32 vectors
        auto result = Add(v32, v64_as_f32);
        StoreU(result, df32, out + i);
    }
}
```

---

## 12. Summary: Integration Checklist

### 12.1 Implementation Checklist

- [ ] Core integration header with Highway includes
- [ ] Type mapping system (Python dtype to Highway lane type)
- [ ] Arena allocator with HWY_ALIGNMENT
- [ ] NumPy array view bridge
- [ ] Code generation templates for each operation type
- [ ] Multi-target dispatch setup with HWY_EXPORT
- [ ] Custom operations (sigmoid, ReLU, etc.)
- [ ] Error handling and validation
- [ ] Thread-local arena management
- [ ] Comprehensive test suite

### 12.2 Performance Checklist

- [ ] All operations use fused loops (single memory pass)
- [ ] FMA used wherever applicable
- [ ] Multiple accumulators for large reductions
- [ ] Aligned allocations for internal buffers
- [ ] LoadU/StoreU for user-provided arrays
- [ ] No branches in hot loops
- [ ] Prefetching for large sequential access

### 12.3 Stability Checklist

- [ ] Input validation before kernel execution
- [ ] Overflow protection for math functions
- [ ] NaN/Inf handling consistent across targets
- [ ] Thread-safe dispatch initialization
- [ ] All size edge cases handled (n=0, n<N, n%N!=0)
- [ ] Scalable vector support (SVE/RVV)
- [ ] Mixed-precision operations explicit

---

**Document Version:** 1.0
**Last Updated:** December 2024
**Status:** Implementation Ready

