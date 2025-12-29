// =============================================================================
// Bud Flow Lang - Arena Allocator Tests
// =============================================================================

#include "bud_flow_lang/arena.h"

#include <gtest/gtest.h>

namespace bud {
namespace {

TEST(ArenaTest, BasicAllocation) {
    Arena arena;

    void* ptr1 = arena.allocate(100);
    ASSERT_NE(ptr1, nullptr);

    void* ptr2 = arena.allocate(200);
    ASSERT_NE(ptr2, nullptr);

    // Pointers should be different
    EXPECT_NE(ptr1, ptr2);
}

TEST(ArenaTest, AlignmentGuarantee) {
    Arena arena;

    for (int i = 0; i < 100; ++i) {
        void* ptr = arena.allocate(i + 1);
        ASSERT_NE(ptr, nullptr);
        // Check SIMD alignment
        EXPECT_TRUE(isAligned(ptr, kSimdAlignment)) << "Allocation " << i << " not aligned";
    }
}

TEST(ArenaTest, CreateObject) {
    Arena arena;

    struct TestObj {
        int x;
        float y;
        TestObj(int a, float b) : x(a), y(b) {}
    };

    auto* obj = arena.create<TestObj>(42, 3.14f);
    ASSERT_NE(obj, nullptr);
    EXPECT_EQ(obj->x, 42);
    EXPECT_FLOAT_EQ(obj->y, 3.14f);
}

TEST(ArenaTest, AllocateArray) {
    Arena arena;

    float* arr = arena.allocateArray<float>(1000);
    ASSERT_NE(arr, nullptr);

    // Should be zero-initialized
    for (int i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(arr[i], 0.0f);
    }
}

TEST(ArenaTest, Reset) {
    Arena arena;

    arena.allocate(1000);
    size_t before = arena.totalAllocated();
    EXPECT_GT(before, 0u);

    arena.reset();
    EXPECT_EQ(arena.totalAllocated(), 0u);
}

TEST(ArenaTest, LargeAllocation) {
    Arena arena;

    // Allocate more than initial block size
    void* ptr = arena.allocate(1024 * 1024);  // 1 MB
    ASSERT_NE(ptr, nullptr);
}

TEST(ArenaTest, ManySmallAllocations) {
    Arena arena;

    // Many small allocations
    for (int i = 0; i < 10000; ++i) {
        void* ptr = arena.allocate(64);
        ASSERT_NE(ptr, nullptr);
    }

    EXPECT_GT(arena.blockCount(), 1u);
}

}  // namespace
}  // namespace bud
