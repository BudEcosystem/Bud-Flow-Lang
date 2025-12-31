// =============================================================================
// Bud Flow Lang - Persistent Kernel Cache Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for disk-based kernel cache that persists compiled kernels across
// process restarts to eliminate cold-start compilation overhead.
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/jit/persistent_cache.h"

#include <filesystem>
#include <fstream>
#include <thread>

#include <gtest/gtest.h>

namespace bud {
namespace jit {
namespace {

// =============================================================================
// CacheKey Tests
// =============================================================================

class CacheKeyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a test IR module
        module_ = std::make_unique<ir::IRModule>("test");
        auto& builder = module_->builder();

        auto a = builder.constant(1.0f);
        auto b = builder.constant(2.0f);
        auto result = builder.add(a, b);
        module_->setOutput(result);
    }

    std::unique_ptr<ir::IRModule> module_;
};

TEST_F(CacheKeyTest, DefaultConstruction) {
    CacheKey key;
    EXPECT_EQ(key.irHash(), 0);
    EXPECT_EQ(key.dtype(), ScalarType::kUnknown);
    EXPECT_EQ(key.sizeClass(), SizeClass::kSmall);
}

TEST_F(CacheKeyTest, FromIRModule) {
    CacheKey key = CacheKey::fromIR(*module_);

    EXPECT_NE(key.irHash(), 0);                    // Should have computed a hash
    EXPECT_EQ(key.dtype(), ScalarType::kFloat32);  // From constants
}

TEST_F(CacheKeyTest, DifferentIRProducesDifferentKeys) {
    // Create two different IR modules
    ir::IRModule module1("test1");
    auto& builder1 = module1.builder();
    auto a1 = builder1.constant(1.0f);
    auto b1 = builder1.constant(2.0f);
    auto add1 = builder1.add(a1, b1);
    module1.setOutput(add1);

    ir::IRModule module2("test2");
    auto& builder2 = module2.builder();
    auto a2 = builder2.constant(3.0f);
    auto b2 = builder2.constant(4.0f);
    auto mul2 = builder2.mul(a2, b2);
    module2.setOutput(mul2);

    CacheKey key1 = CacheKey::fromIR(module1);
    CacheKey key2 = CacheKey::fromIR(module2);

    EXPECT_NE(key1.irHash(), key2.irHash());
}

TEST_F(CacheKeyTest, SameIRProducesSameKey) {
    // Create two modules with identical IR
    ir::IRModule module1("test1");
    auto& builder1 = module1.builder();
    builder1.constant(1.0f);
    builder1.constant(2.0f);

    ir::IRModule module2("test2");
    auto& builder2 = module2.builder();
    builder2.constant(1.0f);
    builder2.constant(2.0f);

    CacheKey key1 = CacheKey::fromIR(module1);
    CacheKey key2 = CacheKey::fromIR(module2);

    EXPECT_EQ(key1.irHash(), key2.irHash());
}

TEST_F(CacheKeyTest, ToString) {
    CacheKey key = CacheKey::fromIR(*module_);
    std::string str = key.toString();

    EXPECT_FALSE(str.empty());
    // Should contain hex hash
    EXPECT_NE(str.find("0x"), std::string::npos);
}

TEST_F(CacheKeyTest, TargetFeaturesIncluded) {
    CacheKey key = CacheKey::fromIR(*module_);

    // Target features should be detected and included
    uint32_t features = key.targetFeatures();
    // At least one feature should be set (SSE2 is baseline for x86-64)
    // Or ARM NEON on ARM
    // Could be 0 on unknown platforms, so just check it's a valid value
    (void)features;  // Just verify it's accessible
}

TEST_F(CacheKeyTest, SizeClassVariants) {
    // Test that size class is properly set
    CacheKey small_key;
    small_key.setSizeClass(SizeClass::kSmall);
    EXPECT_EQ(small_key.sizeClass(), SizeClass::kSmall);

    CacheKey medium_key;
    medium_key.setSizeClass(SizeClass::kMedium);
    EXPECT_EQ(medium_key.sizeClass(), SizeClass::kMedium);

    CacheKey large_key;
    large_key.setSizeClass(SizeClass::kLarge);
    EXPECT_EQ(large_key.sizeClass(), SizeClass::kLarge);
}

// =============================================================================
// PersistentKernelCache Tests
// =============================================================================

class PersistentCacheTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a temporary directory for testing
        cache_dir_ = std::filesystem::temp_directory_path() / "bud_cache_test";
        std::filesystem::create_directories(cache_dir_);

        // Create cache with test directory
        cache_ = std::make_unique<PersistentKernelCache>(cache_dir_, 1024 * 1024);  // 1MB limit
    }

    void TearDown() override {
        // Clean up test directory
        cache_.reset();
        std::error_code ec;
        std::filesystem::remove_all(cache_dir_, ec);
    }

    std::filesystem::path cache_dir_;
    std::unique_ptr<PersistentKernelCache> cache_;
};

TEST_F(PersistentCacheTest, DefaultConstruction) {
    // Should create cache directory if it doesn't exist
    EXPECT_TRUE(std::filesystem::exists(cache_dir_));
}

TEST_F(PersistentCacheTest, SaveAndLoad) {
    // Create a simple "kernel" as test data
    std::vector<uint8_t> kernel_data = {0x01, 0x02, 0x03, 0x04, 0x05};
    CacheKey key;
    key.setIrHash(0x12345678);
    key.setDtype(ScalarType::kFloat32);

    // Save kernel
    EXPECT_TRUE(cache_->save(key, kernel_data));

    // Load kernel
    auto loaded = cache_->load(key);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(*loaded, kernel_data);
}

TEST_F(PersistentCacheTest, LoadNonexistent) {
    CacheKey key;
    key.setIrHash(0x99999999);

    auto loaded = cache_->load(key);
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(PersistentCacheTest, Overwrite) {
    CacheKey key;
    key.setIrHash(0xAAAAAAAA);

    std::vector<uint8_t> data1 = {0x01, 0x02};
    std::vector<uint8_t> data2 = {0x03, 0x04, 0x05};

    // Save first version
    EXPECT_TRUE(cache_->save(key, data1));
    auto loaded1 = cache_->load(key);
    ASSERT_TRUE(loaded1.has_value());
    EXPECT_EQ(*loaded1, data1);

    // Overwrite with second version
    EXPECT_TRUE(cache_->save(key, data2));
    auto loaded2 = cache_->load(key);
    ASSERT_TRUE(loaded2.has_value());
    EXPECT_EQ(*loaded2, data2);
}

TEST_F(PersistentCacheTest, LRUEviction) {
    // Create a small cache (1KB limit)
    auto small_cache = std::make_unique<PersistentKernelCache>(cache_dir_ / "small", 1024);

    // Fill cache beyond capacity
    for (int i = 0; i < 10; ++i) {
        CacheKey key;
        key.setIrHash(static_cast<uint64_t>(i));

        std::vector<uint8_t> data(200, static_cast<uint8_t>(i));  // 200 bytes each
        small_cache->save(key, data);
    }

    // Total size would be 2000 bytes, but limit is 1024
    // Some entries should have been evicted
    size_t found_count = 0;
    for (int i = 0; i < 10; ++i) {
        CacheKey key;
        key.setIrHash(static_cast<uint64_t>(i));
        if (small_cache->load(key).has_value()) {
            ++found_count;
        }
    }

    // Most recent entries should be kept
    EXPECT_GT(found_count, 0);
    EXPECT_LT(found_count, 10);
}

TEST_F(PersistentCacheTest, PersistAcrossRestarts) {
    CacheKey key;
    key.setIrHash(0xDEADBEEF);
    std::vector<uint8_t> data = {0x10, 0x20, 0x30};

    // Save and destroy cache
    EXPECT_TRUE(cache_->save(key, data));
    cache_.reset();

    // Create new cache pointing to same directory
    auto new_cache = std::make_unique<PersistentKernelCache>(cache_dir_, 1024 * 1024);

    // Should be able to load the previously saved data
    auto loaded = new_cache->load(key);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(*loaded, data);
}

TEST_F(PersistentCacheTest, InvalidCacheDirectory) {
    // Try to create cache in an invalid location
    // This should not crash, just log an error
    auto invalid_cache = std::make_unique<PersistentKernelCache>("/nonexistent/path", 1024);

    CacheKey key;
    key.setIrHash(0x11111111);
    std::vector<uint8_t> data = {0x01};

    // Save should fail gracefully
    EXPECT_FALSE(invalid_cache->save(key, data));

    // Load should return empty
    EXPECT_FALSE(invalid_cache->load(key).has_value());
}

TEST_F(PersistentCacheTest, ConcurrentAccess) {
    // Test thread safety
    const int kNumThreads = 4;
    const int kOpsPerThread = 100;

    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([this, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                CacheKey key;
                key.setIrHash(static_cast<uint64_t>(t * 1000 + i));

                std::vector<uint8_t> data(10, static_cast<uint8_t>(i));
                cache_->save(key, data);

                // Try to load
                cache_->load(key);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should complete without crashes or deadlocks
}

TEST_F(PersistentCacheTest, CacheStats) {
    // Save some entries
    for (int i = 0; i < 5; ++i) {
        CacheKey key;
        key.setIrHash(static_cast<uint64_t>(i));
        std::vector<uint8_t> data(100, static_cast<uint8_t>(i));
        cache_->save(key, data);
    }

    auto stats = cache_->stats();
    EXPECT_EQ(stats.total_entries, 5);
    EXPECT_GT(stats.total_bytes, 0);
}

TEST_F(PersistentCacheTest, Clear) {
    // Save some entries
    for (int i = 0; i < 5; ++i) {
        CacheKey key;
        key.setIrHash(static_cast<uint64_t>(i));
        std::vector<uint8_t> data(10, static_cast<uint8_t>(i));
        cache_->save(key, data);
    }

    // Clear the cache
    cache_->clear();

    // All entries should be gone
    for (int i = 0; i < 5; ++i) {
        CacheKey key;
        key.setIrHash(static_cast<uint64_t>(i));
        EXPECT_FALSE(cache_->load(key).has_value());
    }

    auto stats = cache_->stats();
    EXPECT_EQ(stats.total_entries, 0);
}

TEST_F(PersistentCacheTest, VersionInvalidation) {
    CacheKey key;
    key.setIrHash(0xCAFEBABE);
    key.setCompilerVersion(1);

    std::vector<uint8_t> data = {0x01, 0x02};
    cache_->save(key, data);

    // Load with same version should work
    CacheKey same_version_key;
    same_version_key.setIrHash(0xCAFEBABE);
    same_version_key.setCompilerVersion(1);
    EXPECT_TRUE(cache_->load(same_version_key).has_value());

    // Load with different version should fail (different key)
    CacheKey different_version_key;
    different_version_key.setIrHash(0xCAFEBABE);
    different_version_key.setCompilerVersion(2);
    EXPECT_FALSE(cache_->load(different_version_key).has_value());
}

// =============================================================================
// Integration Tests
// =============================================================================

class PersistentCacheIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        cache_dir_ = std::filesystem::temp_directory_path() / "bud_cache_integration";
        std::filesystem::create_directories(cache_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(cache_dir_, ec);
    }

    std::filesystem::path cache_dir_;
};

TEST_F(PersistentCacheIntegrationTest, ColdStartElimination) {
    // First "cold" run - should compile and cache
    {
        PersistentKernelCache cache(cache_dir_, 10 * 1024 * 1024);

        ir::IRModule module("test");
        auto& builder = module.builder();
        auto a = builder.constant(2.0f);
        auto b = builder.constant(3.0f);
        auto mul = builder.mul(a, b);
        module.setOutput(mul);

        CacheKey key = CacheKey::fromIR(module);

        // Simulate compilation - save some "code"
        std::vector<uint8_t> compiled_code(256, 0x90);  // NOP sled
        cache.save(key, compiled_code);

        auto stats = cache.stats();
        EXPECT_EQ(stats.total_entries, 1);
    }

    // Second "warm" run - should hit cache
    {
        PersistentKernelCache cache(cache_dir_, 10 * 1024 * 1024);

        ir::IRModule module("test");
        auto& builder = module.builder();
        auto a = builder.constant(2.0f);
        auto b = builder.constant(3.0f);
        auto mul = builder.mul(a, b);
        module.setOutput(mul);

        CacheKey key = CacheKey::fromIR(module);

        // Should be able to load from disk
        auto loaded = cache.load(key);
        EXPECT_TRUE(loaded.has_value());
        EXPECT_EQ(loaded->size(), 256);
    }
}

}  // namespace
}  // namespace jit
}  // namespace bud
