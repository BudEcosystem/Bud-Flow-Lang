# State-of-the-Art JIT Optimizations for Bud Flow Lang

This document presents comprehensive research on cutting-edge JIT compiler optimizations from academic papers, industry frameworks (JAX, XLA, TVM, LLVM), and modern compiler techniques. These optimizations can significantly improve kernel fusion, pipelining, prefetching, scheduling, runtime optimization, and caching in Bud Flow Lang.

---

## Table of Contents

1. [Kernel Fusion Strategies](#1-kernel-fusion-strategies)
2. [Tiered Compilation & Runtime Optimization](#2-tiered-compilation--runtime-optimization)
3. [Loop Optimization & Scheduling](#3-loop-optimization--scheduling)
4. [Memory & Cache Optimization](#4-memory--cache-optimization)
5. [SIMD/Vectorization Techniques](#5-simdvectorization-techniques)
6. [Prefetching Strategies](#6-prefetching-strategies)
7. [JIT Caching & Persistence](#7-jit-caching--persistence)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Kernel Fusion Strategies

### 1.1 XLA Fusion Architecture (Google)

XLA's fusion is its **single most important optimization**. It groups multiple operations into a single kernel to eliminate intermediate memory writes to HBM.

#### Fusion Pass Types:

| Pass | Description | Priority |
|------|-------------|----------|
| **InstructionFusion** | Vertical producer-consumer fusion | High |
| **FusionMerger** | Merges fusion instructions to reduce memory bandwidth | Medium |
| **MultiOutputFusion** | Sibling fusion (shared inputs) + producer-consumer | High |
| **HorizontalFusion** | Aggregates independent kernels | Medium |

#### Fusion Rules (`ShouldFuse` constraints):

```cpp
// XLA-style fusion constraints
bool ShouldFuse(Operation* producer, Operation* consumer) {
    // 1. Exclude expensive operations
    if (IsConvolution(producer) || IsSort(producer) || IsAllReduce(producer))
        return false;

    // 2. Check hardware limits
    if (EstimatedThreadsPerBlock(fused) > kMaxThreadsPerBlock)
        return false;

    // 3. Prevent code duplication
    if (producer->user_count() > 1 && CodeDuplicationTooHigh(producer))
        return false;

    // 4. Prevent nested loops
    if (WouldCreateNestedLoop(producer, consumer))
        return false;

    return true;
}
```

#### Cost Model-Based Fusion:

XLA has transitioned from heuristic-based to **cost-model-based fusion**:

```cpp
// Priority = modeled reduction in kernel runtime
float CalculateFusionPriority(Operation* producer, Operation* consumer) {
    float time_unfused = PredictTime(producer) + PredictTime(consumer);
    float time_fused = PredictTimeFused(producer, consumer);
    return time_unfused - time_fused;  // Higher = better
}
```

**Recommendation for Bud Flow Lang:**
- Implement a `FusionAnalyzer` class that categorizes operations
- Use cost model to prioritize fusion decisions
- Support multi-output fusion for operations sharing inputs

### 1.2 TVM Operator Fusion Categories

TVM uses `OpPatternKind` to categorize operators:

| Category | Value | Fusibility |
|----------|-------|------------|
| `kElemWise` | 0 | Fully fusible |
| `kBroadcast` | 1 | Fusible (preserves axis order) |
| `kInjective` | 2 | Fusible (output axis -> single input) |
| `kCommReduce` | 3 | Can be fused into consumer |
| `kOutEWiseFusable` | 4 | Complex ops fusible with element-wise |
| `kOpaque` | 8 | Cannot be fused |

**Fusion Rules:**
- `Injective -> Injective`: Chain of element-wise ops
- `Injective -> Reduce`: Element-wise followed by reduction
- `OutFusible -> Injective`: Complex op + element-wise

**Recommendation:**
Categorize Bud operations similarly:
```cpp
enum class OpPattern {
    kElementWise,    // +, -, *, /, sqrt, exp, sin, cos
    kReduction,      // sum, mean, min, max, dot
    kBroadcast,      // scalar broadcasting
    kComplex,        // matmul, conv
    kOpaque          // custom operations
};
```

### 1.3 Horizontal Fusion

Aggregates multiple independent kernel invocations into a single larger kernel launch:

**Benefits:**
- Minimizes kernel launch overhead (~5-10μs per launch)
- Increases thread-level parallelism
- Optimizes memory bandwidth

**Performance:** HFUSE achieves up to **60.8% speedup** over native kernels.

**Implementation Pattern:**
```cpp
// Before: 3 kernel launches
kernel_add(a, b, out1);
kernel_mul(c, d, out2);
kernel_sub(e, f, out3);

// After: 1 fused kernel launch
kernel_fused_horizontal(a, b, c, d, e, f, out1, out2, out3);
```

### 1.4 Vertical Fusion (Producer-Consumer)

Combines sequence of dependent data-parallel operations:

```cpp
// Before: 3 memory passes
temp1 = a * b;          // Write temp1 to memory
temp2 = temp1 + c;      // Read temp1, write temp2
result = sqrt(temp2);   // Read temp2, write result

// After: 1 memory pass (fused)
for (i = 0; i < n; i += SIMD_WIDTH) {
    auto va = Load(a + i);
    auto vb = Load(b + i);
    auto vc = Load(c + i);
    auto result = Sqrt(MulAdd(va, vb, vc));  // FMA + sqrt
    Store(out + i, result);
}
```

**Memory Traffic Reduction:** 3x less memory traffic.

---

## 2. Tiered Compilation & Runtime Optimization

### 2.1 V8-Style Tiered Compilation

V8 uses a **4-tier pipeline** for JavaScript:

| Tier | Name | Optimization Level | Compile Time |
|------|------|-------------------|--------------|
| 0 | Ignition | Interpreter | Immediate |
| 1 | Sparkplug | Baseline JIT | ~100μs |
| 2 | Maglev | Mid-tier SSA | ~1ms |
| 3 | TurboFan | Full optimization | ~10ms |

**Key Insight:** Each tier offers progressively better performance at the cost of compilation time.

**Recommendation for Bud Flow Lang:**
```cpp
enum class CompilationTier {
    kInterpreter,     // Immediate execution, full profiling
    kBaselineJIT,     // Fast compilation, basic optimizations
    kOptimizedJIT     // Full optimization, fusion, vectorization
};

struct TierThresholds {
    size_t baseline_threshold = 10;    // Calls before baseline JIT
    size_t optimized_threshold = 100;  // Calls before full optimization
};
```

### 2.2 On-Stack Replacement (OSR)

OSR allows switching from unoptimized to optimized code **during function execution**:

```cpp
// Example: Long-running loop becomes hot mid-execution
void compute(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {  // Loop may run for millions of iterations
        // After kOSRThreshold iterations, trigger OSR
        if (i % kOSRCheckInterval == 0 && ShouldOSR(loop_context)) {
            // Save state, recompile with optimizations, resume
            TransitionToOptimized(&loop_context, i);
        }
        data[i] = expensive_computation(data[i]);
    }
}
```

**Implementation Requirements:**
- Map execution state from interpreter to compiled code
- Place variable values in correct register/stack positions
- Resume from exact program point

### 2.3 Speculative Optimization with Deoptimization

Generate code under assumptions that may be invalidated at runtime:

**Techniques:**
1. **Type Specialization:** Assume array is always float32
2. **Shape Specialization:** Assume array shape doesn't change
3. **Constant Propagation:** Assume scalar values remain constant
4. **Dead Path Elimination:** Assume certain branches never taken

**Guard Insertion:**
```cpp
// Generated code with guards
void optimized_kernel(Bunch& a, Bunch& b) {
    // Guards verify assumptions
    DEOPT_IF(a.dtype() != DType::Float32, kTypeMismatch);
    DEOPT_IF(a.size() != cached_size_, kSizeMismatch);

    // Fast path (optimized for specific type/shape)
    fast_add_float32_simd(a.data(), b.data(), out.data(), cached_size_);
}
```

### 2.4 Inline Caching (IC)

Cache method lookup results directly at call sites:

| IC Type | Cache Size | Performance |
|---------|------------|-------------|
| Monomorphic | 1 type | ~8 cycles overhead |
| Polymorphic | 2-6 types | ~15-30 cycles |
| Megamorphic | Hash table | ~50+ cycles |

**Application to Bud Flow Lang:**
```cpp
// Cache dispatch decisions for operation types
class InlineCache {
    DType cached_dtype_ = DType::Unknown;
    KernelFn cached_kernel_ = nullptr;

public:
    void dispatch(const Bunch& a, const Bunch& b, Bunch& out) {
        if (a.dtype() == cached_dtype_) {
            cached_kernel_(a, b, out);  // Fast path
        } else {
            // Slow path: lookup and update cache
            cached_dtype_ = a.dtype();
            cached_kernel_ = LookupKernel(a.dtype());
            cached_kernel_(a, b, out);
        }
    }
};
```

### 2.5 Profile-Guided Optimization (PGO)

Use runtime profiling to guide optimization decisions:

**Key Optimizations:**
1. **Hot Path Identification:** Focus optimization on frequently executed code
2. **Branch Prediction Hints:** Annotate likely/unlikely branches
3. **Code Layout:** Reorder blocks for better cache locality
4. **Inlining Decisions:** Inline frequently-called functions

**Performance:** Go 1.22 achieved **2-14% improvement** with PGO.

**Implementation:**
```cpp
struct ProfileData {
    std::atomic<uint64_t> call_count{0};
    std::atomic<uint64_t> total_elements{0};
    std::atomic<uint64_t> branch_taken_count{0};
    std::atomic<uint64_t> branch_not_taken_count{0};
};

void profile_kernel_execution(KernelId id, size_t elements, bool branch_taken) {
    auto& profile = profiles_[id];
    profile.call_count.fetch_add(1, std::memory_order_relaxed);
    profile.total_elements.fetch_add(elements, std::memory_order_relaxed);
    // ... update branch stats
}
```

---

## 3. Loop Optimization & Scheduling

### 3.1 Halide-Style Algorithm/Schedule Separation

Halide's key innovation: **decouple algorithm from schedule**.

**Schedule Primitives:**
```python
# Algorithm (what to compute)
def blur(input):
    bx = (input[x-1] + input[x] + input[x+1]) / 3
    return (bx[y-1] + bx[y] + bx[y+1]) / 3

# Schedule (how to compute) - multiple options
blur.tile(x, y, xo, yo, xi, yi, 256, 32)  # 2D tiling
blur.vectorize(xi, 8)                       # SIMD
blur.parallel(yo)                           # Multi-thread
blur.compute_at(output, yi)                 # Fusion point
```

**Recommendation:** Implement schedule primitives:
```cpp
class Schedule {
public:
    Schedule& split(Var var, int factor, Var& outer, Var& inner);
    Schedule& tile(Var x, Var y, int tile_x, int tile_y);
    Schedule& vectorize(Var var, int width);
    Schedule& parallel(Var var);
    Schedule& unroll(Var var, int factor);
    Schedule& fuse(Var outer, Var inner, Var& fused);
    Schedule& reorder(std::vector<Var> vars);
    Schedule& compute_at(Func consumer, Var var);
};
```

### 3.2 TVM/Ansor Auto-Scheduling

Ansor (TVM's auto-scheduler) uses **template-free search**:

**Architecture:**
1. **ComputeDAG:** Analyze computation graph
2. **Search Space:** Hierarchical (sketch + annotation)
3. **Cost Model:** XGBoost predicts performance
4. **Search Algorithm:** Evolutionary with mutation/crossover

**Performance:** Up to **3.8x improvement** on Intel CPU over prior state-of-the-art.

**Recommendation:** For Bud Flow Lang, implement:
```cpp
class AutoScheduler {
    // Simplified auto-tuning for key parameters
    struct TuningSpace {
        std::vector<int> tile_sizes = {16, 32, 64, 128, 256};
        std::vector<int> unroll_factors = {1, 2, 4, 8};
        std::vector<int> vector_widths = {4, 8, 16};
    };

    Schedule findBestSchedule(const ComputeDAG& dag) {
        Schedule best;
        float best_time = INFINITY;

        for (auto config : generateConfigs(dag)) {
            Schedule s = applyConfig(dag, config);
            float time = measure(s);
            if (time < best_time) {
                best = s;
                best_time = time;
            }
        }
        return best;
    }
};
```

### 3.3 Software Pipelining (Modulo Scheduling)

Overlaps different iterations of loops to increase ILP:

**Key Concepts:**
- **Initiation Interval (II):** Time between successive iterations
- **Lower Bound:** Longest loop-carried dependency
- **Upper Bound:** Length of unscheduled loop body

**Example:**
```cpp
// Before: Sequential iterations
for (i = 0; i < n; i++) {
    load(a[i]);      // Cycle 1-4 (latency 4)
    compute(a[i]);   // Cycle 5-6 (latency 2)
    store(b[i]);     // Cycle 7-8 (latency 2)
}  // 8 cycles per iteration

// After: Software pipelined (II=2)
// Iteration i:   load  compute  store
// Iteration i+1:       load     compute  store
// Iteration i+2:                load     compute  store
// Throughput: 1 iteration every 2 cycles
```

**Performance:** LLVM's Swing Modulo Scheduling achieves **10-33% gains**.

### 3.4 Instruction Scheduling Algorithms

**List Scheduling (Most Common):**
```cpp
Schedule listSchedule(DAG& dag) {
    PriorityQueue ready;
    for (auto& node : dag.nodes_with_no_predecessors())
        ready.push(node);

    while (!ready.empty()) {
        Node* best = nullptr;
        float best_priority = -INFINITY;

        for (auto& node : ready) {
            float priority = calculatePriority(node);
            // Factors: critical path, resource availability, latency
            if (priority > best_priority) {
                best = node;
                best_priority = priority;
            }
        }

        schedule(best);
        for (auto& succ : best->successors())
            if (succ->all_predecessors_scheduled())
                ready.push(succ);
    }
}
```

**Trace Scheduling:**
- Optimize most frequently executed control flow path
- Use runtime profiles to identify hot traces
- Designed for VLIW architectures

### 3.5 Work Stealing Scheduler

For parallel execution of task graphs:

```cpp
class WorkStealingScheduler {
    std::vector<std::deque<Task*>> work_queues_;  // Per-thread queues

    void worker_loop(int thread_id) {
        while (true) {
            Task* task = nullptr;

            // Try own queue first (LIFO)
            if (!work_queues_[thread_id].empty()) {
                task = work_queues_[thread_id].back();
                work_queues_[thread_id].pop_back();
            } else {
                // Steal from random victim (FIFO)
                int victim = random_other_thread(thread_id);
                if (!work_queues_[victim].empty()) {
                    task = work_queues_[victim].front();
                    work_queues_[victim].pop_front();
                }
            }

            if (task) execute(task);
        }
    }
};
```

**Performance Bounds:**
- Expected time: T₁/P + O(T∞) where T₁ = serial time, P = processors
- Space: At most S₁ × P where S₁ = serial space

---

## 4. Memory & Cache Optimization

### 4.1 Optimal Tile Size Calculation

**Analytical Approach (1000x search space reduction):**

```cpp
struct TileSizeCalculator {
    size_t l1_size;
    size_t l2_size;
    size_t cache_line_size;

    // For matrix multiplication C = A × B
    // Tile sizes should fit working set in cache
    std::tuple<int, int, int> calculateTileSizes(int M, int N, int K) {
        // Working set: tile_m × tile_k (A) + tile_k × tile_n (B) + tile_m × tile_n (C)
        // Constraint: working_set <= cache_size

        // Use analytical bounds
        int max_tile = sqrt(l1_size / (3 * sizeof(float)));

        // Refine within bounds
        int tile_m = std::min(max_tile, roundToMultiple(M, SIMD_WIDTH));
        int tile_n = std::min(max_tile, roundToMultiple(N, SIMD_WIDTH));
        int tile_k = std::min(max_tile, K);

        return {tile_m, tile_n, tile_k};
    }
};
```

### 4.2 Multi-Level Cache Blocking

Block for each cache level:

```cpp
void matmul_blocked(float* A, float* B, float* C, int M, int N, int K) {
    // L2 blocking
    for (int i2 = 0; i2 < M; i2 += TILE_L2) {
        for (int j2 = 0; j2 < N; j2 += TILE_L2) {
            for (int k2 = 0; k2 < K; k2 += TILE_L2) {
                // L1 blocking
                for (int i1 = i2; i1 < min(i2+TILE_L2, M); i1 += TILE_L1) {
                    for (int j1 = j2; j1 < min(j2+TILE_L2, N); j1 += TILE_L1) {
                        for (int k1 = k2; k1 < min(k2+TILE_L2, K); k1 += TILE_L1) {
                            // Register blocking (innermost)
                            micro_kernel_simd(A, B, C, i1, j1, k1);
                        }
                    }
                }
            }
        }
    }
}
```

### 4.3 Cache-Oblivious Algorithms

Algorithms optimal without knowing cache parameters:

**Matrix Multiplication (recursive):**
```cpp
void matmul_recursive(float* A, float* B, float* C,
                      int M, int N, int K, int stride) {
    // Base case: small enough to fit in cache
    if (M * N * K <= BASE_SIZE) {
        matmul_base(A, B, C, M, N, K, stride);
        return;
    }

    // Recursive case: divide and conquer
    int M2 = M/2, N2 = N/2, K2 = K/2;

    // C00 = A00*B00 + A01*B10
    matmul_recursive(A, B, C, M2, N2, K2, stride);
    matmul_recursive(A + K2*stride, B + K2, C, M2, N2, K2, stride);

    // C01 = A00*B01 + A01*B11
    matmul_recursive(A, B + N2, C + N2, M2, N2, K2, stride);
    // ... continue for C10, C11
}
```

**Cache Complexity:** O(1 + mn/B + mnp/(B√M)) where M = cache size, B = line size.

### 4.4 XLA Buffer Assignment

XLA uses **static buffer assignment** with liveness analysis:

```cpp
class BufferAssignment {
    // Analyzes live ranges and assigns buffers
    void assignBuffers(HloModule* module) {
        // 1. Compute live ranges for each value
        auto live_ranges = computeLiveRanges(module);

        // 2. Build interference graph
        auto interference = buildInterferenceGraph(live_ranges);

        // 3. Color graph (assign buffers)
        auto coloring = colorGraph(interference);

        // 4. Assign physical buffers
        for (auto& [value, color] : coloring) {
            value->set_buffer(getOrCreateBuffer(color, value->shape()));
        }
    }
};
```

**Performance:** ~20% speedup through buffer reuse.

### 4.5 Non-Temporal Stores

Bypass cache for write-only streaming data:

```cpp
void stream_copy(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        // Load into registers
        __m256 v = _mm256_loadu_ps(src + i);

        // Non-temporal store (bypass cache)
        _mm256_stream_ps(dst + i, v);
    }
    _mm_sfence();  // Ensure stores complete
}
```

**Benefits:**
- Reduces memory bandwidth by ~50% for write streams
- Prevents cache pollution

**When to Use:**
- Large write-only data streams (> L3 cache size)
- Data that won't be reused soon

---

## 5. SIMD/Vectorization Techniques

### 5.1 Multi-Accumulator Reductions

Use multiple accumulators to hide instruction latency:

```cpp
float sum_multi_accum(const float* data, size_t n) {
    using D = hn::ScalableTag<float>;
    D d;
    const size_t N = hn::Lanes(d);

    // 4 independent accumulators
    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    size_t i = 0;
    for (; i + 4*N <= n; i += 4*N) {
        sum0 = hn::Add(sum0, hn::LoadU(d, data + i));
        sum1 = hn::Add(sum1, hn::LoadU(d, data + i + N));
        sum2 = hn::Add(sum2, hn::LoadU(d, data + i + 2*N));
        sum3 = hn::Add(sum3, hn::LoadU(d, data + i + 3*N));
    }

    // Combine accumulators
    auto total = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    return hn::ReduceSum(d, total);
}
```

**Speedup:** 2-4x for reductions (sum, mean, min, max, dot).

### 5.2 FMA (Fused Multiply-Add) Patterns

Single instruction for a*b + c:

```cpp
// Pattern 1: Basic FMA
result = a.fma(b, c);  // a*b + c

// Pattern 2: AXPY (a*x + y)
void axpy(float a, const float* x, float* y, size_t n) {
    auto va = hn::Set(d, a);
    for (size_t i = 0; i < n; i += N) {
        auto vx = hn::LoadU(d, x + i);
        auto vy = hn::LoadU(d, y + i);
        auto result = hn::MulAdd(va, vx, vy);  // a*x + y
        hn::StoreU(result, d, y + i);
    }
}

// Pattern 3: Dot product with FMA
float dot_fma(const float* a, const float* b, size_t n) {
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < n; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        sum = hn::MulAdd(va, vb, sum);  // sum += a*b
    }
    return hn::ReduceSum(d, sum);
}
```

### 5.3 Gather/Scatter Optimization

For non-contiguous memory access:

```cpp
// Gather: Load from non-contiguous addresses
void gather_example(const float* base, const int* indices, float* out, size_t n) {
    using D = hn::ScalableTag<float>;
    using DI = hn::RebindToSigned<D>;
    D d;
    DI di;

    for (size_t i = 0; i < n; i += hn::Lanes(d)) {
        auto idx = hn::LoadU(di, indices + i);
        auto values = hn::GatherIndex(d, base, idx);
        hn::StoreU(values, d, out + i);
    }
}

// Optimization: Convert gather to sequential loads when possible
// If indices are [0,1,2,3,4,5,6,7], use regular Load instead
bool isContiguous(const int* indices, size_t n) {
    for (size_t i = 1; i < n; ++i)
        if (indices[i] != indices[i-1] + 1) return false;
    return true;
}
```

### 5.4 Predicated/Masked Operations

Handle remainder elements without scalar loops:

```cpp
void add_masked(const float* a, const float* b, float* out, size_t n) {
    using D = hn::ScalableTag<float>;
    D d;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    // Main loop
    for (; i + N <= n; i += N) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, out + i);
    }

    // Remainder with mask
    if (i < n) {
        size_t remaining = n - i;
        auto mask = hn::FirstN(d, remaining);
        auto va = hn::MaskedLoad(mask, d, a + i);
        auto vb = hn::MaskedLoad(mask, d, b + i);
        hn::MaskedStore(mask, d, out + i, hn::Add(va, vb));
    }
}
```

### 5.5 SoA vs AoS Transformation

**Structure of Arrays (SoA)** is better for SIMD:

```cpp
// AoS (Array of Structures) - poor for SIMD
struct Point { float x, y, z; };
Point points[N];

// SoA (Structure of Arrays) - excellent for SIMD
struct PointsSoA {
    float x[N];
    float y[N];
    float z[N];
};

// SoA enables vectorized operations
void normalize_soa(PointsSoA& p, size_t n) {
    for (size_t i = 0; i < n; i += N) {
        auto vx = hn::LoadU(d, p.x + i);
        auto vy = hn::LoadU(d, p.y + i);
        auto vz = hn::LoadU(d, p.z + i);

        auto len = hn::Sqrt(hn::MulAdd(vx, vx, hn::MulAdd(vy, vy, hn::Mul(vz, vz))));

        hn::StoreU(hn::Div(vx, len), d, p.x + i);
        hn::StoreU(hn::Div(vy, len), d, p.y + i);
        hn::StoreU(hn::Div(vz, len), d, p.z + i);
    }
}
```

### 5.6 Pairwise Summation for Accuracy

Reduce floating-point error in reductions:

```cpp
float pairwise_sum(const float* data, size_t n) {
    if (n <= 16) {
        // Base case: simple sum
        float sum = 0;
        for (size_t i = 0; i < n; ++i) sum += data[i];
        return sum;
    }

    // Recursive case: divide and conquer
    size_t mid = n / 2;
    return pairwise_sum(data, mid) + pairwise_sum(data + mid, n - mid);
}
```

**Error Bound:** O(n × ε) instead of O(n² × ε) for sequential sum.

---

## 6. Prefetching Strategies

### 6.1 Adaptive Prefetch Distance

Runtime-adjustable prefetch distance based on observed latency:

```cpp
class AdaptivePrefetcher {
    size_t prefetch_distance_ = 8;  // Initial guess
    size_t last_miss_rate_ = 0;

    void adjust_distance(size_t cache_misses, size_t cache_hits) {
        size_t miss_rate = (cache_misses * 100) / (cache_misses + cache_hits);

        if (miss_rate > last_miss_rate_ + 5) {
            // More misses: prefetch further ahead
            prefetch_distance_ = std::min(prefetch_distance_ * 2, MAX_PREFETCH);
        } else if (miss_rate < last_miss_rate_ - 5) {
            // Fewer misses: reduce prefetch (less cache pollution)
            prefetch_distance_ = std::max(prefetch_distance_ / 2, MIN_PREFETCH);
        }

        last_miss_rate_ = miss_rate;
    }
};
```

**Performance:** Self-repairing prefetching achieves **23% improvement**.

### 6.2 Software Prefetch Insertion

```cpp
void process_array_prefetch(float* data, size_t n) {
    constexpr size_t PREFETCH_DISTANCE = 16;  // 16 cache lines ahead

    for (size_t i = 0; i < n; i += 8) {
        // Prefetch future data
        if (i + PREFETCH_DISTANCE * 64 / sizeof(float) < n) {
            __builtin_prefetch(data + i + PREFETCH_DISTANCE * 64 / sizeof(float), 0, 3);
        }

        // Process current data
        auto v = hn::LoadU(d, data + i);
        v = expensive_operation(v);
        hn::StoreU(v, d, data + i);
    }
}
```

### 6.3 Hardware Prefetcher Cooperation

Trigger hardware prefetchers with initial software prefetches:

```cpp
void streaming_process(float* data, size_t n) {
    // Issue initial software prefetches to "prime" hardware prefetcher
    for (size_t i = 0; i < 8; ++i) {
        __builtin_prefetch(data + i * 64 / sizeof(float), 0, 3);
    }

    // After initial prefetches, hardware prefetcher recognizes sequential pattern
    // and continues prefetching automatically
    for (size_t i = 0; i < n; i += 8) {
        auto v = hn::LoadU(d, data + i);
        hn::StoreU(process(v), d, data + i);
    }
}
```

### 6.4 Multi-Striding Optimization

Recent research (December 2024) shows multi-strided variants achieve:
- **12.55x speedup** over Polly
- **2.99x** over Intel MKL
- **2.18x higher memory throughput**

```cpp
// Multi-stride access pattern
void matmul_multistrided(float* A, float* B, float* C, int M, int N, int K) {
    // Stride through A with one pattern, B with another
    // This improves hardware prefetcher effectiveness

    for (int i = 0; i < M; i += STRIDE_A) {
        for (int j = 0; j < N; j += STRIDE_B) {
            for (int k = 0; k < K; k += STRIDE_K) {
                // Kernel with specific access pattern
                micro_kernel_strided(A, B, C, i, j, k);
            }
        }
    }
}
```

---

## 7. JIT Caching & Persistence

### 7.1 Hash-Based Kernel Cache

```cpp
class KernelCache {
    struct CacheKey {
        uint64_t ir_hash;        // Hash of IR bytecode
        DType dtype;
        size_t size_class;       // Small/Medium/Large
        uint32_t target_features;  // AVX2, AVX-512, etc.

        bool operator==(const CacheKey& other) const;
        size_t hash() const;
    };

    std::unordered_map<CacheKey, CompiledKernel> cache_;
    std::mutex mutex_;

public:
    CompiledKernel* lookup(const CacheKey& key) {
        std::lock_guard lock(mutex_);
        auto it = cache_.find(key);
        return it != cache_.end() ? &it->second : nullptr;
    }

    void insert(const CacheKey& key, CompiledKernel kernel) {
        std::lock_guard lock(mutex_);
        cache_.emplace(key, std::move(kernel));
    }
};
```

### 7.2 Persistent Cache with Invalidation

```cpp
class PersistentKernelCache {
    std::filesystem::path cache_dir_;

    // Cache key includes all factors that affect compilation
    std::string computeCacheKey(const IR& ir) {
        std::string key;
        key += hash_to_string(hash_ir(ir));
        key += "_" + get_target_triple();
        key += "_" + get_cpu_features();
        key += "_v" + std::to_string(COMPILER_VERSION);
        return key;
    }

    std::optional<CompiledKernel> loadFromDisk(const std::string& key) {
        auto path = cache_dir_ / (key + ".bin");
        if (!std::filesystem::exists(path)) return std::nullopt;

        // Validate cache entry
        auto metadata = readMetadata(path);
        if (metadata.version != COMPILER_VERSION) {
            std::filesystem::remove(path);  // Invalidate
            return std::nullopt;
        }

        return deserializeKernel(path);
    }

    void saveToDisk(const std::string& key, const CompiledKernel& kernel) {
        auto path = cache_dir_ / (key + ".bin");
        serializeKernel(path, kernel);
    }
};
```

### 7.3 LRU Eviction Policy

```cpp
template<typename K, typename V>
class LRUCache {
    size_t capacity_;
    std::list<std::pair<K, V>> items_;
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> index_;

public:
    V* get(const K& key) {
        auto it = index_.find(key);
        if (it == index_.end()) return nullptr;

        // Move to front (most recently used)
        items_.splice(items_.begin(), items_, it->second);
        return &it->second->second;
    }

    void put(const K& key, V value) {
        // Evict if at capacity
        if (items_.size() >= capacity_) {
            auto lru = items_.back();
            index_.erase(lru.first);
            items_.pop_back();
        }

        items_.emplace_front(key, std::move(value));
        index_[key] = items_.begin();
    }
};
```

### 7.4 Hot/Cold Code Separation

Separate frequently executed code from rarely executed code:

```cpp
class CodeLayout {
    void* hot_section_;    // Frequently executed code
    void* cold_section_;   // Rarely executed (error handling, slow paths)

    void layoutCode(const std::vector<Function>& functions,
                   const ProfileData& profile) {
        std::vector<Function*> hot, cold;

        for (auto& f : functions) {
            if (profile.is_hot(f.id())) {
                hot.push_back(&f);
            } else {
                cold.push_back(&f);
            }
        }

        // Sort hot functions by call frequency (most called first)
        std::sort(hot.begin(), hot.end(), [&](auto* a, auto* b) {
            return profile.call_count(a->id()) > profile.call_count(b->id());
        });

        // Layout hot functions contiguously for better I-cache
        emitToSection(hot_section_, hot);
        emitToSection(cold_section_, cold);
    }
};
```

**Performance:** LLVM hot/cold splitting achieves **>2% improvement** on clang bootstrap.

---

## 8. Implementation Roadmap

### Phase 1: Enhanced Kernel Fusion (High Priority)

| Feature | Description | Expected Speedup |
|---------|-------------|------------------|
| **Cost Model Fusion** | Replace heuristics with analytical cost model | 10-20% |
| **Multi-Output Fusion** | Fuse operations sharing inputs | 15-30% |
| **Horizontal Fusion** | Batch independent operations | 20-40% |
| **Operation Categories** | Categorize ops for fusion legality | Foundation |

### Phase 2: Advanced Runtime Optimization

| Feature | Description | Expected Speedup |
|---------|-------------|------------------|
| **Profile-Guided Optimization** | Use runtime stats for optimization | 5-15% |
| **Speculative Optimization** | Optimize for common cases | 10-25% |
| **Inline Caching** | Cache dispatch decisions | 5-10% |
| **On-Stack Replacement** | Optimize long-running loops | Variable |

### Phase 3: Memory & Cache Optimization

| Feature | Description | Expected Speedup |
|---------|-------------|------------------|
| **Adaptive Prefetching** | Runtime-adjustable prefetch distance | 10-30% |
| **Analytical Tiling** | Optimal tile size calculation | 15-25% |
| **Non-Temporal Stores** | Streaming store optimization | 20-50% for write-heavy |
| **Buffer Reuse** | Static buffer assignment | 15-20% |

### Phase 4: Scheduling & Parallelism

| Feature | Description | Expected Speedup |
|---------|-------------|------------------|
| **Auto-Scheduling** | TVM-style schedule search | 2-4x potential |
| **Work Stealing** | Parallel task execution | Scales with cores |
| **Software Pipelining** | Loop instruction overlap | 10-30% |

### Phase 5: Caching & Persistence

| Feature | Description | Expected Speedup |
|---------|-------------|------------------|
| **Hash-Based Cache** | In-memory kernel cache | Eliminate recompilation |
| **Persistent Cache** | Disk-based cache | Faster cold starts |
| **Hot/Cold Separation** | Improved I-cache usage | 2-5% |

---

## References

### Academic Papers
1. **Copy-and-Patch Compilation** - Haoran Xu, Fredrik Kjolstad (OOPSLA 2021)
2. **Ansor: Auto-Scheduling for TVM** - Zheng et al. (OSDI 2020)
3. **Halide: Decoupling Algorithm from Schedule** - Ragan-Kelley et al. (PLDI 2013)
4. **XLA Fusion Analysis** - arXiv:2301.13062 (2023)
5. **Cache-Oblivious Algorithms** - Frigo et al. (FOCS 1999)
6. **Multi-Striding for Memory-Bound Kernels** - arXiv:2412.16001 (2024)

### Framework Documentation
- [LLVM ORC JIT v2](https://llvm.org/docs/ORCv2.html)
- [LLVM Vectorizers](https://llvm.org/docs/Vectorizers.html)
- [XLA Architecture](https://openxla.org/xla/architecture)
- [JAX Documentation](https://docs.jax.dev/)
- [TVM Documentation](https://tvm.apache.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Triton Documentation](https://triton-lang.org/)
- [Google Highway](https://github.com/google/highway)

### GitHub Repositories
- [OpenXLA/XLA](https://github.com/openxla/xla)
- [Apache TVM](https://github.com/apache/tvm)
- [Halide](https://github.com/halide/Halide)
- [LLVM MLIR](https://github.com/llvm/llvm-project/tree/main/mlir)
- [Triton](https://github.com/triton-lang/triton)
- [PyTorch TorchInductor](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)
