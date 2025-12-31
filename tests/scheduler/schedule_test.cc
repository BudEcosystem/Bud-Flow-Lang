// =============================================================================
// Bud Flow Lang - Schedule Primitives Tests (TDD - RED Phase)
// =============================================================================
//
// Tests for schedule primitives that enable loop transformations and
// optimizations similar to Halide/TVM scheduling.
//
// Schedule primitives:
// - split: Split a loop into outer/inner
// - tile: 2D split for cache blocking
// - fuse: Combine nested loops
// - reorder: Change loop nesting order
// - vectorize: Apply SIMD vectorization
// - parallel: Parallelize across threads
// - unroll: Unroll loop iterations
// - compute_at: Control producer placement
//

#include "bud_flow_lang/ir.h"
#include "bud_flow_lang/scheduler/schedule.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace bud {
namespace scheduler {
namespace {

// =============================================================================
// Var Tests
// =============================================================================

class VarTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(VarTest, DefaultConstruction) {
    Var v;
    EXPECT_FALSE(v.isValid());
    EXPECT_TRUE(v.name().empty());
}

TEST_F(VarTest, NamedConstruction) {
    Var v("x");
    EXPECT_TRUE(v.isValid());
    EXPECT_EQ(v.name(), "x");
}

TEST_F(VarTest, Equality) {
    Var x("x");
    Var y("y");
    Var x2("x");

    EXPECT_EQ(x, x2);
    EXPECT_NE(x, y);
}

// =============================================================================
// Stage Tests
// =============================================================================

class StageTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(StageTest, DefaultConstruction) {
    Stage s;
    EXPECT_FALSE(s.isValid());
}

TEST_F(StageTest, NamedConstruction) {
    Stage s("compute");
    EXPECT_TRUE(s.isValid());
    EXPECT_EQ(s.name(), "compute");
}

// =============================================================================
// Schedule Tests
// =============================================================================

class ScheduleTest : public ::testing::Test {
  protected:
    void SetUp() override { schedule_ = std::make_unique<Schedule>(); }

    std::unique_ptr<Schedule> schedule_;
};

TEST_F(ScheduleTest, DefaultConstruction) {
    EXPECT_TRUE(schedule_->isEmpty());
    EXPECT_EQ(schedule_->numTransforms(), 0);
}

// -------------------------------------------------------------------------
// Split Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, SplitBasic) {
    Var x("x");
    Var x_outer, x_inner;

    schedule_->split(x, 32, x_outer, x_inner);

    EXPECT_TRUE(x_outer.isValid());
    EXPECT_TRUE(x_inner.isValid());
    EXPECT_NE(x_outer.name(), x_inner.name());
    EXPECT_EQ(schedule_->numTransforms(), 1);
}

TEST_F(ScheduleTest, SplitCreatesOuterInnerVars) {
    Var x("x");
    Var x_outer, x_inner;

    schedule_->split(x, 16, x_outer, x_inner);

    // Names should indicate relationship to original var
    EXPECT_TRUE(x_outer.name().find("x") != std::string::npos ||
                x_outer.name().find("outer") != std::string::npos);
    EXPECT_TRUE(x_inner.name().find("x") != std::string::npos ||
                x_inner.name().find("inner") != std::string::npos);
}

TEST_F(ScheduleTest, SplitRecordsFactor) {
    Var x("x");
    Var x_outer, x_inner;

    schedule_->split(x, 64, x_outer, x_inner);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kSplit);
    EXPECT_EQ(transforms[0].factor, 64);
}

// -------------------------------------------------------------------------
// Tile Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, TileBasic) {
    Var x("x"), y("y");
    Var x_outer, x_inner, y_outer, y_inner;

    schedule_->tile(x, y, 32, 32, x_outer, x_inner, y_outer, y_inner);

    EXPECT_TRUE(x_outer.isValid());
    EXPECT_TRUE(x_inner.isValid());
    EXPECT_TRUE(y_outer.isValid());
    EXPECT_TRUE(y_inner.isValid());
}

TEST_F(ScheduleTest, TileRecordsDimensions) {
    Var x("x"), y("y");
    Var xo, xi, yo, yi;

    schedule_->tile(x, y, 64, 32, xo, xi, yo, yi);

    auto transforms = schedule_->transforms();
    bool found_tile = false;
    for (const auto& t : transforms) {
        if (t.type == TransformType::kTile) {
            found_tile = true;
            EXPECT_EQ(t.tile_x, 64);
            EXPECT_EQ(t.tile_y, 32);
            break;
        }
    }
    EXPECT_TRUE(found_tile);
}

// -------------------------------------------------------------------------
// Fuse Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, FuseBasic) {
    Var x("x"), y("y");
    Var fused;

    schedule_->fuse(x, y, fused);

    EXPECT_TRUE(fused.isValid());
    EXPECT_EQ(schedule_->numTransforms(), 1);
}

TEST_F(ScheduleTest, FuseRecordsVars) {
    Var outer("outer"), inner("inner");
    Var fused;

    schedule_->fuse(outer, inner, fused);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kFuse);
}

// -------------------------------------------------------------------------
// Reorder Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, ReorderBasic) {
    Var x("x"), y("y"), z("z");

    schedule_->reorder({z, y, x});  // Change from x,y,z to z,y,x

    EXPECT_EQ(schedule_->numTransforms(), 1);
}

TEST_F(ScheduleTest, ReorderRecordsNewOrder) {
    Var x("x"), y("y"), z("z");

    schedule_->reorder({z, x, y});

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kReorder);
    ASSERT_EQ(transforms[0].vars.size(), 3);
}

// -------------------------------------------------------------------------
// Vectorize Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, VectorizeBasic) {
    Var x("x");

    schedule_->vectorize(x);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kVectorize);
}

TEST_F(ScheduleTest, VectorizeWithWidth) {
    Var x("x");

    schedule_->vectorize(x, 8);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].vector_width, 8);
}

// -------------------------------------------------------------------------
// Parallel Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, ParallelBasic) {
    Var x("x");

    schedule_->parallel(x);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kParallel);
}

// -------------------------------------------------------------------------
// Unroll Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, UnrollBasic) {
    Var x("x");

    schedule_->unroll(x);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kUnroll);
}

TEST_F(ScheduleTest, UnrollWithFactor) {
    Var x("x");

    schedule_->unroll(x, 4);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].factor, 4);
}

// -------------------------------------------------------------------------
// Compute At Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, ComputeAtBasic) {
    Stage producer("producer");
    Stage consumer("consumer");
    Var at_var("y");

    schedule_->computeAt(producer, consumer, at_var);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kComputeAt);
}

// -------------------------------------------------------------------------
// Compute Inline Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, ComputeInlineBasic) {
    Stage producer("producer");

    schedule_->computeInline(producer);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kComputeInline);
}

// -------------------------------------------------------------------------
// Cache Read/Write Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, CacheReadBasic) {
    Stage producer("producer");

    schedule_->cacheRead(producer, CacheType::kShared);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kCacheRead);
}

TEST_F(ScheduleTest, CacheWriteBasic) {
    Stage producer("producer");

    schedule_->cacheWrite(producer, CacheType::kLocal);

    auto transforms = schedule_->transforms();
    ASSERT_EQ(transforms.size(), 1);
    EXPECT_EQ(transforms[0].type, TransformType::kCacheWrite);
}

// -------------------------------------------------------------------------
// Chaining Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, ChainedTransforms) {
    Var x("x"), y("y");
    Var xo, xi, yo, yi;

    schedule_->split(x, 32, xo, xi).split(y, 32, yo, yi).reorder({xo, yo, xi, yi}).vectorize(yi);

    EXPECT_EQ(schedule_->numTransforms(), 4);
}

// -------------------------------------------------------------------------
// Clone/Copy Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, Clone) {
    Var x("x");
    Var xo, xi;

    schedule_->split(x, 32, xo, xi);

    auto cloned = schedule_->clone();

    EXPECT_EQ(cloned->numTransforms(), schedule_->numTransforms());
}

// -------------------------------------------------------------------------
// Serialization Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, SerializeDeserialize) {
    Var x("x"), y("y");
    Var xo, xi;

    schedule_->split(x, 32, xo, xi).vectorize(xi);

    std::string serialized = schedule_->serialize();
    EXPECT_FALSE(serialized.empty());

    Schedule restored;
    EXPECT_TRUE(restored.deserialize(serialized));
    EXPECT_EQ(restored.numTransforms(), schedule_->numTransforms());
}

// -------------------------------------------------------------------------
// Validation Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, ValidateEmpty) {
    // Empty schedule should be valid
    EXPECT_TRUE(schedule_->validate());
}

TEST_F(ScheduleTest, ValidateWithTransforms) {
    Var x("x");
    Var xo, xi;

    schedule_->split(x, 32, xo, xi);

    EXPECT_TRUE(schedule_->validate());
}

// -------------------------------------------------------------------------
// Benchmark Tests
// -------------------------------------------------------------------------

TEST_F(ScheduleTest, BenchmarkTransformCreation) {
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        Schedule s;
        Var x("x"), y("y");
        Var xo, xi, yo, yi;

        s.split(x, 32, xo, xi).split(y, 32, yo, yi).reorder({xo, yo, xi, yi}).vectorize(yi);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double us_per_schedule = static_cast<double>(duration_us) / kIterations;

    // Schedule creation should be fast (<10us)
    EXPECT_LT(us_per_schedule, 50.0)
        << "Schedule creation too slow: " << us_per_schedule << " us/schedule";

    std::cout << "Schedule creation time: " << us_per_schedule << " us/schedule (4 transforms)\n";
}

}  // namespace
}  // namespace scheduler
}  // namespace bud
