// =============================================================================
// Bud Flow Lang - Type System Tests
// =============================================================================

#include "bud_flow_lang/type_system.h"

#include <gtest/gtest.h>

namespace bud {
namespace {

TEST(ShapeTest, ScalarShape) {
    Shape s = Shape::scalar();
    EXPECT_EQ(s.rank(), 0u);
    EXPECT_EQ(s.totalElements(), 1u);
}

TEST(ShapeTest, VectorShape) {
    Shape s = Shape::vector(100);
    EXPECT_EQ(s.rank(), 1u);
    EXPECT_EQ(s[0], 100u);
    EXPECT_EQ(s.totalElements(), 100u);
}

TEST(ShapeTest, MatrixShape) {
    Shape s = Shape::matrix(32, 64);
    EXPECT_EQ(s.rank(), 2u);
    EXPECT_EQ(s[0], 32u);
    EXPECT_EQ(s[1], 64u);
    EXPECT_EQ(s.totalElements(), 32u * 64u);
}

TEST(ShapeTest, Equality) {
    EXPECT_EQ(Shape::vector(100), Shape::vector(100));
    EXPECT_NE(Shape::vector(100), Shape::vector(200));
    EXPECT_NE(Shape::vector(100), Shape::matrix(10, 10));
}

TEST(ShapeTest, BroadcastSameShape) {
    auto result = Shape::broadcast(Shape::vector(100), Shape::vector(100));
    ASSERT_TRUE(result.hasValue());
    EXPECT_EQ(*result, Shape::vector(100));
}

TEST(ShapeTest, BroadcastScalar) {
    auto result = Shape::broadcast(Shape::scalar(), Shape::vector(100));
    ASSERT_TRUE(result.hasValue());
    EXPECT_EQ(*result, Shape::vector(100));
}

TEST(ShapeTest, BroadcastMatrixVector) {
    auto result = Shape::broadcast(Shape::matrix(32, 64), Shape::vector(64));
    ASSERT_TRUE(result.hasValue());
    EXPECT_EQ(*result, Shape::matrix(32, 64));
}

TEST(ShapeTest, BroadcastIncompatible) {
    auto result = Shape::broadcast(Shape::vector(100), Shape::vector(200));
    EXPECT_TRUE(result.hasError());
}

TEST(TypeDescTest, Construction) {
    TypeDesc t(ScalarType::kFloat32, Shape::vector(100));
    EXPECT_EQ(t.scalarType(), ScalarType::kFloat32);
    EXPECT_EQ(t.shape().rank(), 1u);
    EXPECT_EQ(t.elementCount(), 100u);
    EXPECT_EQ(t.byteSize(), 400u);  // 100 * 4 bytes
}

TEST(TypeDescTest, FactoryMethods) {
    EXPECT_EQ(TypeDesc::f32().scalarType(), ScalarType::kFloat32);
    EXPECT_EQ(TypeDesc::f64().scalarType(), ScalarType::kFloat64);
    EXPECT_EQ(TypeDesc::i32().scalarType(), ScalarType::kInt32);
}

TEST(TypeInferrerTest, BinaryOpSameType) {
    TypeDesc a(ScalarType::kFloat32, Shape::vector(100));
    TypeDesc b(ScalarType::kFloat32, Shape::vector(100));

    auto result = TypeInferrer::inferBinaryOp(a, b);
    ASSERT_TRUE(result.hasValue());
    EXPECT_EQ(result->scalarType(), ScalarType::kFloat32);
    EXPECT_EQ(result->shape(), Shape::vector(100));
}

TEST(TypeInferrerTest, BinaryOpTypeMismatch) {
    TypeDesc a(ScalarType::kFloat32, Shape::vector(100));
    TypeDesc b(ScalarType::kFloat64, Shape::vector(100));

    auto result = TypeInferrer::inferBinaryOp(a, b);
    EXPECT_TRUE(result.hasError());
}

TEST(TypeInferrerTest, Reduction) {
    TypeDesc t(ScalarType::kFloat32, Shape::vector(100));

    auto result = TypeInferrer::inferReduction(t);
    ASSERT_TRUE(result.hasValue());
    EXPECT_TRUE(result->isScalar());
}

}  // namespace
}  // namespace bud
