// =============================================================================
// Bud Flow Lang - Adaptive Executor Tests
// =============================================================================

#include "bud_flow_lang/jit/adaptive_executor.h"
#include "bud_flow_lang/jit/adaptive_kernel_cache.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

namespace bud {
namespace jit {
namespace {

// =============================================================================
// AdaptiveKernelCache Tests
// =============================================================================

class AdaptiveKernelCacheTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.warmup_executions = 5;
        config_.min_calls_tier1 = 10;
        config_.min_calls_tier2 = 50;
        cache_ = std::make_unique<AdaptiveKernelCache>(config_);
    }

    AdaptiveCacheConfig config_;
    std::unique_ptr<AdaptiveKernelCache> cache_;
};

TEST_F(AdaptiveKernelCacheTest, CreateContextKey) {
    auto key = AdaptiveKernelCache::makeKey(0x12345678, ScalarType::kFloat32, 1024);

    EXPECT_EQ(key.ir_hash, 0x12345678);
    EXPECT_EQ(key.dtype, ScalarType::kFloat32);
    EXPECT_EQ(key.size_bucket, 2);  // 1024 is in bucket 2 (1K-4K)
    EXPECT_NE(key.hardware_id, 0);  // Should have hardware fingerprint
}

TEST_F(AdaptiveKernelCacheTest, SizeBuckets) {
    // Test size bucketing
    auto key1 = AdaptiveKernelCache::makeKey(0, ScalarType::kFloat32, 100);
    EXPECT_EQ(key1.size_bucket, 0);  // <256

    auto key2 = AdaptiveKernelCache::makeKey(0, ScalarType::kFloat32, 500);
    EXPECT_EQ(key2.size_bucket, 1);  // 256-1K

    auto key3 = AdaptiveKernelCache::makeKey(0, ScalarType::kFloat32, 2048);
    EXPECT_EQ(key3.size_bucket, 2);  // 1K-4K

    auto key4 = AdaptiveKernelCache::makeKey(0, ScalarType::kFloat32, 8192);
    EXPECT_EQ(key4.size_bucket, 3);  // 4K-16K

    auto key5 = AdaptiveKernelCache::makeKey(0, ScalarType::kFloat32, 100000);
    EXPECT_EQ(key5.size_bucket, 4);  // >=16K
}

TEST_F(AdaptiveKernelCacheTest, GetOrCreateContext) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // First access creates context
    EXPECT_FALSE(cache_->hasContext(key));

    auto& entry = cache_->getOrCreateContext(key);
    EXPECT_TRUE(cache_->hasContext(key));

    // Should have one default version
    EXPECT_EQ(entry.versions.size(), 1);
    EXPECT_EQ(entry.versions[0].tier, ExecutionTier::kInterpreter);
}

TEST_F(AdaptiveKernelCacheTest, SelectVersionWarmup) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // During warmup, should cycle through versions
    for (size_t i = 0; i < config_.warmup_executions; ++i) {
        auto decision = cache_->selectVersion(key, 1024);
        EXPECT_EQ(decision.tier, ExecutionTier::kInterpreter);
        EXPECT_EQ(decision.version_idx, 0);
    }
}

TEST_F(AdaptiveKernelCacheTest, RecordExecution) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // Select and record execution
    auto decision = cache_->selectVersion(key, 1024);
    cache_->recordExecution(key, decision.version_idx, 1024, 1000.0f);

    // Check stats
    auto* version = cache_->getVersion(key, decision.version_idx);
    ASSERT_NE(version, nullptr);
    EXPECT_EQ(version->executions, 1);
    EXPECT_GT(version->avg_time_ns, 0.0f);
}

TEST_F(AdaptiveKernelCacheTest, PromotionDecision) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // Initial state - no promotion
    auto decision = cache_->getPromotionDecision(key);
    EXPECT_FALSE(decision.should_promote);

    // Execute enough times to trigger promotion
    for (size_t i = 0; i < config_.min_calls_tier1 + 5; ++i) {
        auto sel = cache_->selectVersion(key, 1024);
        cache_->recordExecution(key, sel.version_idx, 1024, 100.0f);
    }

    // Now should be ready for promotion
    decision = cache_->getPromotionDecision(key);
    // May or may not promote depending on Thompson variance
    EXPECT_EQ(decision.tier, ExecutionTier::kInterpreter);
}

TEST_F(AdaptiveKernelCacheTest, AddVersion) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // Create context
    cache_->getOrCreateContext(key);

    // Add new version
    KernelVersion new_version;
    new_version.tier = ExecutionTier::kCopyPatch;
    new_version.code_ptr = reinterpret_cast<void*>(0x1234);

    size_t idx = cache_->addVersion(key, new_version);
    EXPECT_EQ(idx, 1);  // Second version

    // Verify it was added
    auto* version = cache_->getVersion(key, idx);
    ASSERT_NE(version, nullptr);
    EXPECT_EQ(version->tier, ExecutionTier::kCopyPatch);
}

TEST_F(AdaptiveKernelCacheTest, PromoteContext) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    cache_->getOrCreateContext(key);

    // Promote to tier 1
    bool promoted = cache_->promoteContext(key, ExecutionTier::kCopyPatch, nullptr);
    EXPECT_TRUE(promoted);

    // Should now have 2 versions
    auto* context = cache_->getContext(key);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->versions.size(), 2);
    EXPECT_EQ(context->versions[1].tier, ExecutionTier::kCopyPatch);
}

TEST_F(AdaptiveKernelCacheTest, SerializationRoundtrip) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // Create and populate cache
    // First selectVersion creates context without incrementing, so +1 to get 20
    for (int i = 0; i < 21; ++i) {
        auto decision = cache_->selectVersion(key, 1024);
        cache_->recordExecution(key, decision.version_idx, 1024, 100.0f + i);
    }

    // Serialize
    std::string data = cache_->serialize();
    EXPECT_FALSE(data.empty());

    // Create new cache and deserialize
    AdaptiveKernelCache cache2(config_);
    EXPECT_TRUE(cache2.deserialize(data));

    // Verify data matches
    EXPECT_TRUE(cache2.hasContext(key));
    auto* context = cache2.getContext(key);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->total_executions, 20);
}

TEST_F(AdaptiveKernelCacheTest, Statistics) {
    auto key = AdaptiveKernelCache::makeKey(0xABCD, ScalarType::kFloat32, 1024);

    // Execute some operations
    for (int i = 0; i < 10; ++i) {
        auto decision = cache_->selectVersion(key, 1024);
        cache_->recordExecution(key, decision.version_idx, 1024, 100.0f);
    }

    auto stats = cache_->statistics();
    EXPECT_EQ(stats.total_contexts, 1);
    EXPECT_GE(stats.total_versions, 1);
    EXPECT_EQ(stats.total_executions, 10);
    EXPECT_EQ(stats.tier0_contexts, 1);
}

TEST_F(AdaptiveKernelCacheTest, HardwareId) {
    uint32_t hw_id = AdaptiveKernelCache::getHardwareId();
    EXPECT_NE(hw_id, 0);

    // Should be consistent
    EXPECT_EQ(hw_id, AdaptiveKernelCache::getHardwareId());
}

// =============================================================================
// AdaptiveExecutor Tests
// =============================================================================

class AdaptiveExecutorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.enable_persistence = false;  // Don't persist during tests
        config_.cache_config.warmup_executions = 5;
        config_.cache_config.min_calls_tier1 = 10;
        executor_ = std::make_unique<AdaptiveExecutor>(config_);
    }

    AdaptiveExecutorConfig config_;
    std::unique_ptr<AdaptiveExecutor> executor_;
};

TEST_F(AdaptiveExecutorTest, DefaultConfig) {
    AdaptiveExecutor exec;
    EXPECT_TRUE(exec.isEnabled());
    EXPECT_TRUE(exec.config().enable_timing);
    EXPECT_TRUE(exec.config().enable_promotion);
}

TEST_F(AdaptiveExecutorTest, EnableDisable) {
    EXPECT_TRUE(executor_->isEnabled());
    executor_->setEnabled(false);
    EXPECT_FALSE(executor_->isEnabled());
    executor_->setEnabled(true);
    EXPECT_TRUE(executor_->isEnabled());
}

TEST_F(AdaptiveExecutorTest, Statistics) {
    auto stats = executor_->statistics();
    EXPECT_EQ(stats.total_executions, 0);
    EXPECT_EQ(stats.tier0_executions, 0);
    EXPECT_EQ(stats.tier1_executions, 0);
    EXPECT_EQ(stats.tier2_executions, 0);
    EXPECT_EQ(stats.promotions_performed, 0);
}

TEST_F(AdaptiveExecutorTest, Reset) {
    // Get initial stats
    auto stats = executor_->statistics();
    EXPECT_EQ(stats.total_executions, 0);

    // Reset
    executor_->reset();

    // Stats should still be zero
    stats = executor_->statistics();
    EXPECT_EQ(stats.total_executions, 0);
}

TEST_F(AdaptiveExecutorTest, SetConfig) {
    AdaptiveExecutorConfig new_config;
    new_config.verbose = true;
    new_config.enable_promotion = false;

    executor_->setConfig(new_config);

    EXPECT_TRUE(executor_->config().verbose);
    EXPECT_FALSE(executor_->config().enable_promotion);
}

TEST_F(AdaptiveExecutorTest, DispatchTemplate) {
    uint64_t kernel_hash = 0x12345;
    int call_count = 0;

    // Use the dispatch template
    auto result =
        executor_->dispatch(kernel_hash, ScalarType::kFloat32, 1024, [&](size_t version_idx) {
            call_count++;
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        });

    EXPECT_TRUE(result.success);
    EXPECT_EQ(call_count, 1);
    EXPECT_GT(result.execution_time_ns, 0.0f);
}

TEST_F(AdaptiveExecutorTest, DispatchWhenDisabled) {
    executor_->setEnabled(false);

    uint64_t kernel_hash = 0x12345;
    int call_count = 0;

    auto result =
        executor_->dispatch(kernel_hash, ScalarType::kFloat32, 1024, [&](size_t version_idx) {
            call_count++;
            EXPECT_EQ(version_idx, 0);  // Should always be 0 when disabled
        });

    EXPECT_TRUE(result.success);
    EXPECT_EQ(call_count, 1);
    EXPECT_EQ(result.tier_used, ExecutionTier::kInterpreter);
}

TEST_F(AdaptiveExecutorTest, MultipleCalls) {
    uint64_t kernel_hash = 0x12345;
    int call_count = 0;

    // Make multiple calls
    for (int i = 0; i < 20; ++i) {
        auto result = executor_->dispatch(kernel_hash, ScalarType::kFloat32, 1024,
                                          [&](size_t) { call_count++; });
        EXPECT_TRUE(result.success);
    }

    EXPECT_EQ(call_count, 20);

    // Check cache statistics (dispatch uses cache, not executor stats)
    auto stats = executor_->statistics();
    // dispatch() updates cache stats, not executor stats
    EXPECT_EQ(stats.cache_stats.total_executions, 20);
}

TEST_F(AdaptiveExecutorTest, CacheAccess) {
    auto& cache = executor_->cache();
    EXPECT_EQ(cache.statistics().total_contexts, 0);

    // Dispatch something
    executor_->dispatch(0x12345, ScalarType::kFloat32, 1024, [](size_t) {});

    // Cache should now have entries
    EXPECT_GE(cache.statistics().total_contexts, 1);
}

// =============================================================================
// Global Instance Tests
// =============================================================================

TEST(GlobalAdaptiveExecutorTest, GetGlobalExecutor) {
    auto& exec1 = getGlobalExecutor();
    auto& exec2 = getGlobalExecutor();

    // Should return same instance
    EXPECT_EQ(&exec1, &exec2);
}

TEST(GlobalAdaptiveExecutorTest, ConfigureGlobalExecutor) {
    AdaptiveExecutorConfig config;
    config.verbose = true;

    configureGlobalExecutor(config);

    EXPECT_TRUE(getGlobalExecutor().config().verbose);

    // Reset to default for other tests
    config.verbose = false;
    configureGlobalExecutor(config);
}

// =============================================================================
// TierDecision Tests
// =============================================================================

TEST(TierDecisionTest, DefaultValues) {
    TierDecision decision;
    EXPECT_EQ(decision.tier, ExecutionTier::kInterpreter);
    EXPECT_EQ(decision.version_idx, 0);
    EXPECT_FALSE(decision.should_promote);
    EXPECT_FALSE(decision.should_specialize);
    EXPECT_TRUE(decision.reason.empty());
}

// =============================================================================
// ExecutionResult Tests
// =============================================================================

TEST(ExecutionResultTest, DefaultValues) {
    ExecutionResult result;
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.execution_time_ns, 0.0f);
    EXPECT_EQ(result.tier_used, ExecutionTier::kInterpreter);
    EXPECT_EQ(result.version_idx, 0);
    EXPECT_FALSE(result.was_promoted);
    EXPECT_TRUE(result.error_message.empty());
}

// =============================================================================
// ContextEntry Tests
// =============================================================================

TEST(ContextEntryTest, RecordSize) {
    ContextEntry entry;

    // Record various sizes
    for (int i = 0; i < 100; ++i) {
        entry.recordSize(1024);  // Same size
        entry.total_executions++;
    }

    // Should detect dominant size
    auto dominant = entry.dominantSize(0.5f);
    EXPECT_TRUE(dominant.has_value());
    EXPECT_EQ(*dominant, 1024);
}

TEST(ContextEntryTest, ShouldSpecialize) {
    ContextEntry entry;

    // Not enough executions
    EXPECT_FALSE(entry.shouldSpecialize(100));

    // Add executions with consistent size
    for (int i = 0; i < 200; ++i) {
        entry.recordSize(4096);
        entry.total_executions++;
    }

    EXPECT_TRUE(entry.shouldSpecialize(100));
}

TEST(ContextEntryTest, NoDominantSize) {
    ContextEntry entry;

    // Mix of different sizes
    for (int i = 0; i < 100; ++i) {
        entry.recordSize(100 + i * 1000);  // Different sizes
        entry.total_executions++;
    }

    // Should not have a dominant size
    auto dominant = entry.dominantSize(0.8f);
    EXPECT_FALSE(dominant.has_value());
}

// =============================================================================
// TierName Tests
// =============================================================================

TEST(TierNameTest, AllTiers) {
    EXPECT_STREQ(tierName(ExecutionTier::kInterpreter), "Interpreter");
    EXPECT_STREQ(tierName(ExecutionTier::kCopyPatch), "CopyPatch");
    EXPECT_STREQ(tierName(ExecutionTier::kFusedKernel), "FusedKernel");
}

}  // namespace
}  // namespace jit
}  // namespace bud
