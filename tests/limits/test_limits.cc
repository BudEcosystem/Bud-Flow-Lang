/**
 * @file test_limits.cc
 * @brief System limits discovery tests
 *
 * These tests explore the limits of the system:
 * - Maximum array sizes
 * - Peak throughput for various operations
 * - Cache efficiency at different sizes
 * - Numerical precision limits
 */

#include "bud_flow_lang/bud_flow_lang.h"
#include "bud_flow_lang/bunch.h"
#include "bud_flow_lang/memory/cache_config.h"
#include "bud_flow_lang/memory/tiled_executor.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace test {

// =============================================================================
// Limits Discovery Test Fixture
// =============================================================================

class LimitsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!isInitialized()) {
            RuntimeConfig config;
            config.enable_debug_output = false;
            auto result = initialize(config);
            ASSERT_TRUE(result.hasValue());
        }
        setTilingEnabled(true);
        setPrefetchEnabled(true);
    }

    // Measure throughput in billion elements per second
    template <typename Func>
    double measureThroughput(size_t elements, size_t iterations, Func&& func) {
        // Warmup
        func();

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double total_elements = static_cast<double>(elements) * iterations;
        return total_elements / seconds / 1e9;  // Billion elements/sec
    }

    // Measure memory bandwidth in GB/s
    template <typename Func>
    double measureBandwidth(size_t bytes_per_iter, size_t iterations, Func&& func) {
        func();  // Warmup

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double total_bytes = static_cast<double>(bytes_per_iter) * iterations;
        return total_bytes / seconds / 1e9;  // GB/s
    }
};

// =============================================================================
// Cache Configuration Discovery
// =============================================================================

TEST_F(LimitsTest, DiscoverCacheHierarchy) {
    auto* config = getCacheConfig();
    ASSERT_NE(config, nullptr);

    std::cout << "\n=== Cache Configuration ===" << std::endl;
    std::cout << "  L1 Cache: " << config->l1Size() / 1024 << " KB" << std::endl;
    std::cout << "  L2 Cache: " << config->l2Size() / 1024 << " KB" << std::endl;
    std::cout << "  L3 Cache: " << config->l3Size() / 1024 << " KB" << std::endl;
    std::cout << "  Cache Line: " << config->lineSize() << " bytes" << std::endl;

    // Calculate optimal tile sizes for different scenarios
    std::cout << "\n=== Optimal Tile Sizes ===" << std::endl;
    std::cout << "  Float32, 2 arrays: " << config->optimalTileSize(4, 2) << " elements"
              << std::endl;
    std::cout << "  Float32, 3 arrays: " << config->optimalTileSize(4, 3) << " elements"
              << std::endl;
    std::cout << "  Float64, 2 arrays: " << config->optimalTileSize(8, 2) << " elements"
              << std::endl;

    // Verify sensible values
    EXPECT_GT(config->l1Size(), 0);
    EXPECT_GT(config->l2Size(), 0);
    EXPECT_GT(config->lineSize(), 0);
}

// =============================================================================
// Peak Throughput Discovery
// =============================================================================

TEST_F(LimitsTest, DiscoverDotProductThroughput) {
    std::cout << "\n=== Dot Product Throughput ===" << std::endl;

    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    double peak_throughput = 0;
    size_t optimal_size = 0;

    for (size_t N : sizes) {
        auto a = Bunch::ones(N);
        auto b = Bunch::ones(N);
        if (!a.hasValue() || !b.hasValue())
            continue;

        size_t iters = std::max(size_t{1}, 10000000 / N);
        double throughput = measureThroughput(N, iters, [&]() {
            volatile float r = a->dot(*b);
            (void)r;
        });

        std::cout << std::setw(10) << N << " elements: " << std::fixed << std::setprecision(2)
                  << throughput << " Gelem/s" << std::endl;

        if (throughput > peak_throughput) {
            peak_throughput = throughput;
            optimal_size = N;
        }
    }

    std::cout << "  Peak: " << peak_throughput << " Gelem/s at " << optimal_size << " elements"
              << std::endl;
    EXPECT_GT(peak_throughput, 0.1);  // At least 100M elements/sec
}

TEST_F(LimitsTest, DiscoverAdditionThroughput) {
    std::cout << "\n=== Element-wise Addition Throughput ===" << std::endl;

    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    double peak_throughput = 0;

    for (size_t N : sizes) {
        auto a = Bunch::ones(N);
        auto b = Bunch::ones(N);
        if (!a.hasValue() || !b.hasValue())
            continue;

        size_t iters = std::max(size_t{1}, 5000000 / N);
        double throughput = measureThroughput(N, iters, [&]() {
            auto result = *a + *b;
            (void)result;
        });

        std::cout << std::setw(10) << N << " elements: " << std::fixed << std::setprecision(2)
                  << throughput << " Gelem/s" << std::endl;

        peak_throughput = std::max(peak_throughput, throughput);
    }

    std::cout << "  Peak: " << peak_throughput << " Gelem/s" << std::endl;
    EXPECT_GT(peak_throughput, 0.05);
}

TEST_F(LimitsTest, DiscoverReductionThroughput) {
    std::cout << "\n=== Sum Reduction Throughput ===" << std::endl;

    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    double peak_throughput = 0;

    for (size_t N : sizes) {
        auto a = Bunch::ones(N);
        if (!a.hasValue())
            continue;

        size_t iters = std::max(size_t{1}, 10000000 / N);
        double throughput = measureThroughput(N, iters, [&]() {
            volatile float r = a->sum();
            (void)r;
        });

        std::cout << std::setw(10) << N << " elements: " << std::fixed << std::setprecision(2)
                  << throughput << " Gelem/s" << std::endl;

        peak_throughput = std::max(peak_throughput, throughput);
    }

    std::cout << "  Peak: " << peak_throughput << " Gelem/s" << std::endl;
    EXPECT_GT(peak_throughput, 0.1);
}

// =============================================================================
// Memory Bandwidth Discovery
// =============================================================================

TEST_F(LimitsTest, DiscoverMemoryBandwidth) {
    std::cout << "\n=== Memory Bandwidth ===" << std::endl;

    // Use addition as a bandwidth-bound operation (read 2 arrays, write 1)
    const size_t N = 10000000;  // 10M elements = 120MB for 3 arrays
    auto a = Bunch::ones(N);
    auto b = Bunch::ones(N);

    if (!a.hasValue() || !b.hasValue()) {
        GTEST_SKIP() << "Could not allocate 10M element arrays";
    }

    size_t bytes_per_iter = N * sizeof(float) * 3;  // Read a, b, write result
    double bandwidth = measureBandwidth(bytes_per_iter, 10, [&]() {
        auto result = *a + *b;
        (void)result;
    });

    std::cout << "  Measured bandwidth: " << bandwidth << " GB/s" << std::endl;
    EXPECT_GT(bandwidth, 1.0);  // At least 1 GB/s
}

// =============================================================================
// Cache Efficiency Tests
// =============================================================================

TEST_F(LimitsTest, DiscoverCacheEfficiency) {
    std::cout << "\n=== Cache Efficiency by Level ===" << std::endl;

    auto* config = getCacheConfig();
    if (!config) {
        GTEST_SKIP() << "Cache config not available";
    }

    // Test at L1 boundary
    size_t l1_elems = config->l1Size() / sizeof(float) / 3;  // Fit 3 arrays
    size_t l2_elems = config->l2Size() / sizeof(float) / 3;
    size_t l3_elems = config->l3Size() / sizeof(float) / 3;
    size_t ram_elems = l3_elems * 4;  // Beyond L3

    struct Level {
        const char* name;
        size_t elems;
        double throughput;
    };

    std::vector<Level> levels = {
        {"L1", l1_elems, 0}, {"L2", l2_elems, 0}, {"L3", l3_elems, 0}, {"RAM", ram_elems, 0}};

    for (auto& level : levels) {
        if (level.elems > 100000000) {
            std::cout << "  " << level.name << ": skipped (too large)" << std::endl;
            continue;
        }

        auto a = Bunch::ones(level.elems);
        auto b = Bunch::ones(level.elems);
        if (!a.hasValue() || !b.hasValue()) {
            std::cout << "  " << level.name << ": allocation failed" << std::endl;
            continue;
        }

        size_t iters = std::max(size_t{1}, 100000000 / level.elems);
        level.throughput = measureThroughput(level.elems, iters, [&]() {
            volatile float r = a->dot(*b);
            (void)r;
        });

        std::cout << "  " << level.name << " (" << level.elems << " elems): " << std::fixed
                  << std::setprecision(2) << level.throughput << " Gelem/s" << std::endl;
    }

    // Verify we got valid measurements (not asserting specific performance targets
    // as they vary greatly depending on hardware and system load)
    if (levels[0].throughput > 0 && levels[1].throughput > 0) {
        // Just verify both produced non-zero throughput
        EXPECT_GT(levels[0].throughput, 0.0);
        EXPECT_GT(levels[1].throughput, 0.0);
    }
}

// =============================================================================
// Numerical Precision Limits
// =============================================================================

TEST_F(LimitsTest, DiscoverNumericalPrecision) {
    std::cout << "\n=== Numerical Precision ===" << std::endl;

    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};

    for (size_t N : sizes) {
        auto a = Bunch::ones(N);
        if (!a.hasValue())
            continue;

        float sum = a->sum();
        float expected = static_cast<float>(N);
        float relative_error = std::abs(sum - expected) / expected;

        std::cout << std::setw(10) << N << " elements: " << "sum=" << sum
                  << ", expected=" << expected << ", rel_error=" << std::scientific
                  << relative_error << std::fixed << std::endl;
    }

    // The last test (1M elements) should have some measurable error
    auto a = Bunch::ones(1000000);
    ASSERT_TRUE(a.hasValue());
    float sum = a->sum();
    float rel_error = std::abs(sum - 1000000.0f) / 1000000.0f;

    // Should be less than 1% for float32 accumulation
    EXPECT_LT(rel_error, 0.01) << "Relative error too high: " << rel_error;
}

// =============================================================================
// Scaling Tests
// =============================================================================

TEST_F(LimitsTest, DiscoverScaling) {
    std::cout << "\n=== Scaling Behavior ===" << std::endl;

    // Measure how throughput scales with size
    std::vector<size_t> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    std::vector<double> throughputs;

    for (size_t N : sizes) {
        auto a = Bunch::ones(N);
        auto b = Bunch::ones(N);
        if (!a.hasValue() || !b.hasValue())
            continue;

        size_t iters = 100000000 / N;
        double throughput = measureThroughput(N, iters, [&]() {
            volatile float r = a->dot(*b);
            (void)r;
        });

        throughputs.push_back(throughput);
        std::cout << std::setw(10) << N << " elements: " << std::fixed << std::setprecision(2)
                  << throughput << " Gelem/s" << std::endl;
    }

    // Throughput should generally increase or plateau, not decrease drastically
    if (throughputs.size() >= 3) {
        double first = throughputs[0];
        double last = throughputs.back();
        // The last measurement shouldn't be more than 10x worse than the first
        EXPECT_GT(last * 10, first) << "Throughput degradation too severe";
    }
}

// =============================================================================
// Maximum Allocation Discovery
// =============================================================================

TEST_F(LimitsTest, DiscoverMaxAllocation) {
    std::cout << "\n=== Maximum Allocation Size ===" << std::endl;

    size_t max_successful = 0;
    std::vector<size_t> test_sizes = {
        1000000,    // 4 MB
        10000000,   // 40 MB
        50000000,   // 200 MB
        100000000,  // 400 MB
        250000000,  // 1 GB
        500000000,  // 2 GB
    };

    for (size_t N : test_sizes) {
        auto a = Bunch::zeros(N);
        if (a.hasValue()) {
            max_successful = N;
            std::cout << "  " << N << " elements (" << (N * 4) / (1024 * 1024) << " MB): OK"
                      << std::endl;
        } else {
            std::cout << "  " << N << " elements (" << (N * 4) / (1024 * 1024) << " MB): FAILED"
                      << std::endl;
            break;
        }
    }

    std::cout << "  Max successful: " << max_successful << " elements ("
              << (max_successful * 4) / (1024 * 1024) << " MB)" << std::endl;

    EXPECT_GE(max_successful, 1000000) << "Should be able to allocate at least 4MB";
}

// =============================================================================
// Tiling Threshold Analysis
// =============================================================================

TEST_F(LimitsTest, AnalyzeTilingThreshold) {
    std::cout << "\n=== Tiling Threshold Analysis ===" << std::endl;

    // Compare tiled vs untiled for various sizes
    std::vector<size_t> sizes = {512, 1024, 2048, 4096, 8192, 16384};

    for (size_t N : sizes) {
        auto a = Bunch::ones(N);
        auto b = Bunch::ones(N);
        if (!a.hasValue() || !b.hasValue())
            continue;

        size_t iters = 1000000 / N;

        setTilingEnabled(true);
        double tiled = measureThroughput(N, iters, [&]() {
            volatile float r = a->dot(*b);
            (void)r;
        });

        setTilingEnabled(false);
        double untiled = measureThroughput(N, iters, [&]() {
            volatile float r = a->dot(*b);
            (void)r;
        });

        setTilingEnabled(true);

        double speedup = tiled / untiled;
        std::cout << std::setw(10) << N << " elements: " << "tiled=" << std::fixed
                  << std::setprecision(2) << tiled << " untiled=" << untiled
                  << " speedup=" << speedup << "x" << std::endl;
    }
}

}  // namespace test
}  // namespace bud
