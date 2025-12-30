/**
 * @file test_memory_optimization.cc
 * @brief TDD tests for memory access optimization features
 *
 * Tests for:
 * - Cache-aware loop tiling
 * - Software prefetching
 * - NUMA-aware allocation
 *
 * Following TDD: Write tests first, then implement to make them pass.
 */

#include "bud_flow_lang/common.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/numa_allocator.h"
#include "bud_flow_lang/memory/prefetch.h"
#include "bud_flow_lang/memory/tiled_executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace memory {
namespace {

// =============================================================================
// SECTION 1: Cache Configuration Tests
// =============================================================================

class CacheConfigTest : public ::testing::Test {
  protected:
    void SetUp() override { config_ = CacheConfig::detect(); }

    CacheConfig config_;
};

TEST_F(CacheConfigTest, DetectsCacheSizes) {
    // L1 cache should be detected (typically 32-64 KB per core)
    EXPECT_GT(config_.l1Size(), 0u);
    EXPECT_GE(config_.l1Size(), 16 * 1024);   // At least 16 KB
    EXPECT_LE(config_.l1Size(), 256 * 1024);  // At most 256 KB

    // L2 cache should be detected (typically 256 KB - 2 MB)
    EXPECT_GT(config_.l2Size(), 0u);
    EXPECT_GE(config_.l2Size(), 128 * 1024);        // At least 128 KB
    EXPECT_LE(config_.l2Size(), 16 * 1024 * 1024);  // At most 16 MB

    // L3 cache may or may not exist
    // If it exists, it should be larger than L2
    if (config_.l3Size() > 0) {
        EXPECT_GT(config_.l3Size(), config_.l2Size());
    }
}

TEST_F(CacheConfigTest, DetectsCacheLineSize) {
    // Cache line size should be 64 bytes on modern x86/ARM
    EXPECT_GE(config_.lineSize(), 32u);
    EXPECT_LE(config_.lineSize(), 128u);
    // Should be power of 2
    EXPECT_EQ(config_.lineSize() & (config_.lineSize() - 1), 0u);
}

TEST_F(CacheConfigTest, CalculatesOptimalTileSize) {
    // Tile size should be based on L1 cache
    size_t tile = config_.optimalTileSize(sizeof(float));
    EXPECT_GT(tile, 0u);

    // Tile should fit in L1 with some headroom
    size_t tile_bytes = tile * sizeof(float);
    EXPECT_LE(tile_bytes, config_.l1Size());

    // Should be aligned to SIMD width
    EXPECT_EQ(tile % (kSimdAlignment / sizeof(float)), 0u);
}

TEST_F(CacheConfigTest, CalculatesOptimalTileSizeForL2) {
    // Can also target L2 for larger working sets
    size_t tile = config_.optimalTileSizeForLevel(sizeof(float), CacheLevel::kL2);
    EXPECT_GT(tile, 0u);

    size_t tile_bytes = tile * sizeof(float);
    EXPECT_LE(tile_bytes, config_.l2Size());
}

TEST_F(CacheConfigTest, ManualConfiguration) {
    // Should be able to create manual configuration
    CacheConfig manual(32 * 1024, 256 * 1024, 8 * 1024 * 1024, 64);

    EXPECT_EQ(manual.l1Size(), 32u * 1024);
    EXPECT_EQ(manual.l2Size(), 256u * 1024);
    EXPECT_EQ(manual.l3Size(), 8u * 1024 * 1024);
    EXPECT_EQ(manual.lineSize(), 64u);
}

// =============================================================================
// SECTION 2: Tiled Executor Tests
// =============================================================================

class TiledExecutorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_ = CacheConfig::detect();
        executor_ = std::make_unique<TiledExecutor>(config_);
    }

    CacheConfig config_;
    std::unique_ptr<TiledExecutor> executor_;
};

TEST_F(TiledExecutorTest, ExecutesTiledBinaryOp) {
    const size_t n = 100000;  // Large enough to benefit from tiling
    std::vector<float> a(n), b(n), out(n);

    // Initialize with known values
    std::iota(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 2.0f);

    // Execute tiled addition
    executor_->binaryOp(out.data(), a.data(), b.data(), n, [](float x, float y) { return x + y; });

    // Verify results
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 2.0f);
    }
}

TEST_F(TiledExecutorTest, ExecutesTiledUnaryOp) {
    const size_t n = 100000;
    std::vector<float> a(n), out(n);

    std::iota(a.begin(), a.end(), 1.0f);  // Start from 1 to avoid sqrt(0) issues

    // Execute tiled sqrt
    executor_->unaryOp(out.data(), a.data(), n, [](float x) { return std::sqrt(x); });

    // Verify results
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], std::sqrt(static_cast<float>(i + 1)), 1e-5f);
    }
}

TEST_F(TiledExecutorTest, ExecutesTiledReduction) {
    const size_t n = 100000;
    std::vector<float> a(n);

    // Fill with 1.0 for easy verification
    std::fill(a.begin(), a.end(), 1.0f);

    // Execute tiled sum reduction
    float result = executor_->reduce(a.data(), n, 0.0f, [](float acc, float x) { return acc + x; });

    EXPECT_FLOAT_EQ(result, static_cast<float>(n));
}

TEST_F(TiledExecutorTest, ExecutesTiledDotProduct) {
    const size_t n = 100000;
    std::vector<float> a(n), b(n);

    // a = [1, 1, 1, ...], b = [1, 1, 1, ...]
    // dot = n
    std::fill(a.begin(), a.end(), 1.0f);
    std::fill(b.begin(), b.end(), 1.0f);

    float result = executor_->dotProduct(a.data(), b.data(), n);

    EXPECT_FLOAT_EQ(result, static_cast<float>(n));
}

TEST_F(TiledExecutorTest, HandlesSmallArraysWithoutTiling) {
    const size_t n = 10;  // Too small to tile
    std::vector<float> a(n), b(n), out(n);

    std::iota(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 1.0f);

    executor_->binaryOp(out.data(), a.data(), b.data(), n, [](float x, float y) { return x + y; });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 1.0f);
    }
}

TEST_F(TiledExecutorTest, HandlesMisalignedData) {
    const size_t n = 10003;  // Odd size, not aligned to tile boundaries
    std::vector<float> a(n), b(n), out(n);

    std::iota(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 3.0f);

    executor_->binaryOp(out.data(), a.data(), b.data(), n, [](float x, float y) { return x * y; });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) * 3.0f);
    }
}

TEST_F(TiledExecutorTest, TiledExecutionIsCorrectForFMA) {
    const size_t n = 100000;
    std::vector<float> a(n), b(n), c(n), out(n);

    // a = [0, 1, 2, ...], b = [2, 2, 2, ...], c = [1, 1, 1, ...]
    // FMA = a * b + c = [1, 3, 5, 7, ...]
    std::iota(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 2.0f);
    std::fill(c.begin(), c.end(), 1.0f);

    executor_->ternaryOp(out.data(), a.data(), b.data(), c.data(), n,
                         [](float x, float y, float z) { return x * y + z; });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) * 2.0f + 1.0f);
    }
}

// =============================================================================
// SECTION 3: Prefetch Tests
// =============================================================================

class PrefetchTest : public ::testing::Test {
  protected:
    void SetUp() override { prefetcher_ = std::make_unique<Prefetcher>(); }

    std::unique_ptr<Prefetcher> prefetcher_;
};

TEST_F(PrefetchTest, CalculatesOptimalPrefetchDistance) {
    // Prefetch distance should be positive
    size_t distance = prefetcher_->optimalDistance(sizeof(float));
    EXPECT_GT(distance, 0u);

    // Should be reasonable (typically 2-64 cache lines ahead in bytes)
    // At 8 lines of 64 bytes = 512 bytes, distance in floats = 128
    EXPECT_GE(distance, 32u);    // At least 2 cache lines in floats
    EXPECT_LE(distance, 4096u);  // At most 64 cache lines
}

TEST_F(PrefetchTest, PrefetchDoesNotCrash) {
    std::vector<float> data(1000);

    // Should not crash even with out-of-bounds prefetch
    EXPECT_NO_THROW(prefetcher_->prefetch(&data[0]));
    EXPECT_NO_THROW(prefetcher_->prefetch(&data[500]));
    EXPECT_NO_THROW(prefetcher_->prefetchRange(&data[0], 100));
}

TEST_F(PrefetchTest, PrefetchWithDifferentHints) {
    std::vector<float> data(1000);

    // Test different temporal locality hints
    EXPECT_NO_THROW(prefetcher_->prefetch(&data[0], PrefetchHint::kT0));   // All caches
    EXPECT_NO_THROW(prefetcher_->prefetch(&data[0], PrefetchHint::kT1));   // L2 and above
    EXPECT_NO_THROW(prefetcher_->prefetch(&data[0], PrefetchHint::kT2));   // L3 and above
    EXPECT_NO_THROW(prefetcher_->prefetch(&data[0], PrefetchHint::kNTA));  // Non-temporal
}

TEST_F(PrefetchTest, StreamingPrefetchPattern) {
    const size_t n = 100000;
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 0.0f);

    // Streaming prefetch with configurable distance
    size_t prefetch_distance = prefetcher_->optimalDistance(sizeof(float));
    float sum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        // Prefetch ahead
        if (i + prefetch_distance < n) {
            prefetcher_->prefetch(&data[i + prefetch_distance], PrefetchHint::kT0);
        }
        sum += data[i];
    }

    // Verify computation was correct
    // Use double for expected to avoid float precision loss in the formula
    double expected = static_cast<double>(n) * (static_cast<double>(n) - 1.0) / 2.0;
    // Allow 0.01% relative error due to float accumulation
    EXPECT_NEAR(static_cast<double>(sum), expected, expected * 0.0001);
}

// =============================================================================
// SECTION 4: NUMA Allocator Tests
// =============================================================================

class NumaAllocatorTest : public ::testing::Test {
  protected:
    void SetUp() override { allocator_ = std::make_unique<NumaAllocator>(); }

    std::unique_ptr<NumaAllocator> allocator_;
};

TEST_F(NumaAllocatorTest, DetectsNumaTopology) {
    NumaTopology topology = allocator_->topology();

    // Should have at least 1 NUMA node
    EXPECT_GE(topology.numNodes(), 1u);

    // Current node should be valid
    EXPECT_LT(topology.currentNode(), topology.numNodes());
}

TEST_F(NumaAllocatorTest, AllocatesOnCurrentNode) {
    const size_t size = 1024 * 1024;  // 1 MB

    void* ptr = allocator_->allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Memory should be usable
    std::memset(ptr, 0, size);

    allocator_->deallocate(ptr, size);
}

TEST_F(NumaAllocatorTest, AllocatesOnSpecificNode) {
    NumaTopology topology = allocator_->topology();

    if (topology.numNodes() < 2) {
        GTEST_SKIP() << "Only one NUMA node available, skipping multi-node test";
    }

    const size_t size = 1024 * 1024;

    // Allocate on node 0
    void* ptr0 = allocator_->allocateOnNode(size, 0);
    ASSERT_NE(ptr0, nullptr);
    std::memset(ptr0, 0, size);

    // Allocate on node 1
    void* ptr1 = allocator_->allocateOnNode(size, 1);
    ASSERT_NE(ptr1, nullptr);
    std::memset(ptr1, 1, size);

    allocator_->deallocate(ptr0, size);
    allocator_->deallocate(ptr1, size);
}

TEST_F(NumaAllocatorTest, AllocatesInterleaved) {
    NumaTopology topology = allocator_->topology();

    if (topology.numNodes() < 2) {
        GTEST_SKIP() << "Only one NUMA node available, skipping interleave test";
    }

    const size_t size = 16 * 1024 * 1024;  // 16 MB - large enough to span pages

    void* ptr = allocator_->allocateInterleaved(size);
    ASSERT_NE(ptr, nullptr);

    // Memory should be usable
    std::memset(ptr, 0, size);

    allocator_->deallocate(ptr, size);
}

TEST_F(NumaAllocatorTest, AllocatesAligned) {
    const size_t size = 4096;
    const size_t alignment = kSimdAlignment;

    void* ptr = allocator_->allocateAligned(size, alignment);
    ASSERT_NE(ptr, nullptr);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0u);

    allocator_->deallocate(ptr, size);
}

TEST_F(NumaAllocatorTest, ReportsMemoryPerNode) {
    NumaTopology topology = allocator_->topology();

    for (size_t node = 0; node < topology.numNodes(); ++node) {
        size_t total = topology.totalMemory(node);
        size_t free = topology.freeMemory(node);

        EXPECT_GT(total, 0u);
        EXPECT_LE(free, total);
    }
}

TEST_F(NumaAllocatorTest, HandlesAllocationFailureGracefully) {
    // Try to allocate more memory than available
    const size_t huge_size = SIZE_MAX / 2;

    void* ptr = allocator_->allocate(huge_size);
    EXPECT_EQ(ptr, nullptr);  // Should return nullptr, not crash
}

TEST_F(NumaAllocatorTest, ThreadLocalAllocationUsesLocalNode) {
    NumaTopology topology = allocator_->topology();

    if (topology.numNodes() < 2) {
        GTEST_SKIP() << "Only one NUMA node available";
    }

    // Allocate with local policy (default behavior)
    const size_t size = 1024 * 1024;
    void* ptr = allocator_->allocateLocal(size);
    ASSERT_NE(ptr, nullptr);

    // Touch the memory to ensure it's faulted in
    std::memset(ptr, 0, size);

    // Query which node the memory is on
    int node = allocator_->getNodeForAddress(ptr);
    EXPECT_GE(node, 0);
    EXPECT_LT(static_cast<size_t>(node), topology.numNodes());

    allocator_->deallocate(ptr, size);
}

// =============================================================================
// SECTION 5: Integration Tests
// =============================================================================

class MemoryOptimizationIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        cache_config_ = CacheConfig::detect();
        executor_ = std::make_unique<TiledExecutor>(cache_config_);
        numa_ = std::make_unique<NumaAllocator>();
    }

    CacheConfig cache_config_;
    std::unique_ptr<TiledExecutor> executor_;
    std::unique_ptr<NumaAllocator> numa_;
};

TEST_F(MemoryOptimizationIntegrationTest, TiledExecutionWithNumaMemory) {
    const size_t n = 1000000;
    const size_t bytes = n * sizeof(float);

    // Allocate arrays with NUMA-awareness
    float* a = static_cast<float*>(numa_->allocateAligned(bytes, kSimdAlignment));
    float* b = static_cast<float*>(numa_->allocateAligned(bytes, kSimdAlignment));
    float* out = static_cast<float*>(numa_->allocateAligned(bytes, kSimdAlignment));

    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(out, nullptr);

    // Initialize
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f;
    }

    // Execute with tiling
    executor_->binaryOp(out, a, b, n, [](float x, float y) { return x + y; });

    // Verify
    for (size_t i = 0; i < std::min(n, size_t(100)); ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 2.0f);
    }
    // Spot check middle and end
    EXPECT_FLOAT_EQ(out[n / 2], static_cast<float>(n / 2) + 2.0f);
    EXPECT_FLOAT_EQ(out[n - 1], static_cast<float>(n - 1) + 2.0f);

    numa_->deallocate(a, bytes);
    numa_->deallocate(b, bytes);
    numa_->deallocate(out, bytes);
}

TEST_F(MemoryOptimizationIntegrationTest, TiledExecutionWithPrefetching) {
    const size_t n = 1000000;
    std::vector<float> a(n), b(n), out(n);

    std::iota(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 1.0f);

    // Execute with prefetching enabled
    executor_->binaryOpWithPrefetch(out.data(), a.data(), b.data(), n,
                                    [](float x, float y) { return x + y; });

    // Verify results
    for (size_t i = 0; i < std::min(n, size_t(100)); ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 1.0f);
    }
}

// =============================================================================
// SECTION 6: Performance Regression Tests
// =============================================================================

class MemoryOptimizationBenchmarkTest : public ::testing::Test {
  protected:
    static constexpr size_t kBenchmarkSize = 10000000;  // 10M elements
    static constexpr int kWarmupIterations = 5;
    static constexpr int kMeasureIterations = 20;

    void SetUp() override {
        cache_config_ = CacheConfig::detect();
        executor_ = std::make_unique<TiledExecutor>(cache_config_);

        a_.resize(kBenchmarkSize);
        b_.resize(kBenchmarkSize);
        out_.resize(kBenchmarkSize);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < kBenchmarkSize; ++i) {
            a_[i] = dist(gen);
            b_[i] = dist(gen);
        }
    }

    double measureThroughput(std::function<void()> fn) {
        // Warmup
        for (int i = 0; i < kWarmupIterations; ++i) {
            fn();
        }

        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kMeasureIterations; ++i) {
            fn();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count() / kMeasureIterations;
        double elements_per_sec = kBenchmarkSize / seconds;
        return elements_per_sec / 1e9;  // Billion elements per second
    }

    CacheConfig cache_config_;
    std::unique_ptr<TiledExecutor> executor_;
    std::vector<float> a_, b_, out_;
};

TEST_F(MemoryOptimizationBenchmarkTest, TiledAdditionThroughput) {
    double throughput = measureThroughput([this]() {
        executor_->binaryOp(out_.data(), a_.data(), b_.data(), kBenchmarkSize,
                            [](float x, float y) { return x + y; });
    });

    // Should achieve reasonable throughput (at least 0.1 billion elements/sec)
    // Note: Threshold is very conservative to account for system load variability
    // and different hardware configurations
    EXPECT_GT(throughput, 0.1) << "Tiled addition throughput: " << throughput
                               << " billion elements/sec";

    // Log performance for visibility
    std::cout << "Tiled addition throughput: " << throughput << " billion elements/sec"
              << std::endl;
}

TEST_F(MemoryOptimizationBenchmarkTest, TiledReductionThroughput) {
    double throughput = measureThroughput([this]() {
        volatile float result = executor_->reduce(a_.data(), kBenchmarkSize, 0.0f,
                                                  [](float acc, float x) { return acc + x; });
        (void)result;
    });

    EXPECT_GT(throughput, 0.1) << "Tiled reduction throughput: " << throughput
                               << " billion elements/sec";

    std::cout << "Tiled reduction throughput: " << throughput << " billion elements/sec"
              << std::endl;
}

TEST_F(MemoryOptimizationBenchmarkTest, TiledDotProductThroughput) {
    double throughput = measureThroughput([this]() {
        volatile float result = executor_->dotProduct(a_.data(), b_.data(), kBenchmarkSize);
        (void)result;
    });

    EXPECT_GT(throughput, 0.1) << "Tiled dot product throughput: " << throughput
                               << " billion elements/sec";

    std::cout << "Tiled dot product throughput: " << throughput << " billion elements/sec"
              << std::endl;
}

}  // namespace
}  // namespace memory
}  // namespace bud
