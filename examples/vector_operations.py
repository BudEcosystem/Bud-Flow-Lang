#!/usr/bin/env python3
"""
Vector Operations with Bud Flow Lang

Common vector math operations implemented efficiently:
- Vector normalization
- Distance calculations
- Angle computations
- Projections
- Linear algebra basics

Run: python vector_operations.py
"""

import bud_flow_lang_py as flow
import math


def normalize(v):
    """
    Normalize a vector to unit length.

    Args:
        v: Flow Bunch representing a vector

    Returns:
        Unit vector in the same direction
    """
    length_squared = flow.dot(v, v)
    length = length_squared ** 0.5
    return v * (1.0 / length)


def magnitude(v):
    """
    Compute the magnitude (length) of a vector.

    Args:
        v: Flow Bunch representing a vector

    Returns:
        Scalar length of the vector
    """
    return flow.dot(v, v) ** 0.5


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.

    Args:
        a, b: Flow Bunch vectors of the same size

    Returns:
        Scalar distance
    """
    diff = a - b
    return flow.dot(diff, diff) ** 0.5


def squared_distance(a, b):
    """
    Compute squared Euclidean distance (avoids sqrt).

    Useful when comparing distances (if d1 < d2, then d1^2 < d2^2).

    Args:
        a, b: Flow Bunch vectors of the same size

    Returns:
        Squared distance
    """
    diff = a - b
    return flow.dot(diff, diff)


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.

    Returns:
        Value in [-1, 1], where 1 means identical direction
    """
    dot_product = flow.dot(a, b)
    norm_a = flow.dot(a, a) ** 0.5
    norm_b = flow.dot(b, b) ** 0.5
    return dot_product / (norm_a * norm_b)


def angle_between(a, b):
    """
    Compute angle between two vectors in radians.

    Args:
        a, b: Flow Bunch vectors

    Returns:
        Angle in radians [0, pi]
    """
    cos_angle = cosine_similarity(a, b)
    # Clamp to [-1, 1] to handle floating-point errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def project(a, b):
    """
    Project vector a onto vector b.

    Args:
        a: Vector to project
        b: Vector to project onto

    Returns:
        Projection of a onto b
    """
    # proj_b(a) = (a . b / b . b) * b
    scale = flow.dot(a, b) / flow.dot(b, b)
    return b * scale


def reject(a, b):
    """
    Compute rejection of vector a from vector b.

    This is the component of a perpendicular to b.

    Args:
        a: Vector to reject
        b: Vector to reject from

    Returns:
        Rejection of a from b
    """
    return a - project(a, b)


def lerp(a, b, t):
    """
    Linear interpolation between two vectors.

    Args:
        a: Start vector
        b: End vector
        t: Interpolation factor (0=a, 1=b)

    Returns:
        Interpolated vector
    """
    return flow.lerp(a, b, t)


def slerp(a, b, t):
    """
    Spherical linear interpolation between two unit vectors.

    Interpolates along the great circle connecting a and b.

    Args:
        a, b: Unit vectors
        t: Interpolation factor

    Returns:
        Interpolated unit vector
    """
    # Compute angle between vectors
    dot = flow.dot(a, b)
    dot = max(-1.0, min(1.0, dot))  # Clamp
    theta = math.acos(dot)

    if abs(theta) < 1e-6:
        # Vectors are parallel, use linear interpolation
        return lerp(a, b, t)

    sin_theta = math.sin(theta)
    sa = math.sin((1 - t) * theta) / sin_theta
    sb = math.sin(t * theta) / sin_theta

    return a * sa + b * sb


def centroid(vectors):
    """
    Compute centroid (average) of multiple vectors.

    Args:
        vectors: List of Flow Bunch vectors

    Returns:
        Centroid vector
    """
    n = len(vectors)
    result = vectors[0].copy()
    for v in vectors[1:]:
        result = result + v
    return result * (1.0 / n)


def demo_basic_operations():
    """Demonstrate basic vector operations."""
    print("\n=== Basic Vector Operations ===\n")

    # Create test vectors
    v1 = flow.flow([3.0, 4.0])
    v2 = flow.flow([1.0, 0.0])

    print(f"v1 = {v1.to_numpy()}")
    print(f"v2 = {v2.to_numpy()}")
    print()

    # Magnitude
    print(f"Magnitude of v1: {magnitude(v1):.4f}")
    print(f"  (Expected: 5.0 since |[3,4]| = sqrt(9+16) = 5)")
    print()

    # Normalization
    v1_norm = normalize(v1)
    print(f"Normalized v1: {v1_norm.to_numpy()}")
    print(f"  Length of normalized: {magnitude(v1_norm):.4f} (should be 1.0)")
    print()

    # Distance
    dist = euclidean_distance(v1, v2)
    print(f"Distance from v1 to v2: {dist:.4f}")
    diff = v1 - v2
    print(f"  Diff vector: {diff.to_numpy()}")
    print(f"  |[2, 4]| = sqrt(4+16) = {math.sqrt(20):.4f}")


def demo_similarity():
    """Demonstrate similarity and angle calculations."""
    print("\n=== Similarity and Angles ===\n")

    # Test vectors
    v1 = flow.flow([1.0, 0.0])   # Right
    v2 = flow.flow([0.0, 1.0])   # Up
    v3 = flow.flow([1.0, 1.0])   # Diagonal
    v4 = flow.flow([-1.0, 0.0])  # Left (opposite of v1)

    print("v1 = [1, 0] (right)")
    print("v2 = [0, 1] (up)")
    print("v3 = [1, 1] (diagonal)")
    print("v4 = [-1, 0] (left)")
    print()

    # Cosine similarity
    print("Cosine Similarities:")
    print(f"  v1 . v1: {cosine_similarity(v1, v1):.4f} (same vector)")
    print(f"  v1 . v2: {cosine_similarity(v1, v2):.4f} (perpendicular)")
    print(f"  v1 . v3: {cosine_similarity(v1, v3):.4f} (45 degrees)")
    print(f"  v1 . v4: {cosine_similarity(v1, v4):.4f} (opposite)")
    print()

    # Angles
    print("Angles (in degrees):")
    print(f"  v1 to v1: {math.degrees(angle_between(v1, v1)):.1f}")
    print(f"  v1 to v2: {math.degrees(angle_between(v1, v2)):.1f}")
    print(f"  v1 to v3: {math.degrees(angle_between(v1, v3)):.1f}")
    print(f"  v1 to v4: {math.degrees(angle_between(v1, v4)):.1f}")


def demo_projections():
    """Demonstrate vector projections."""
    print("\n=== Projections ===\n")

    # Project v1 onto v2
    v1 = flow.flow([3.0, 4.0])
    v2 = flow.flow([1.0, 0.0])

    print(f"v1 = {v1.to_numpy()}")
    print(f"v2 = {v2.to_numpy()}")
    print()

    proj = project(v1, v2)
    rej = reject(v1, v2)

    print(f"Projection of v1 onto v2: {proj.to_numpy()}")
    print(f"Rejection of v1 from v2: {rej.to_numpy()}")
    print()

    # Verify: proj + rej = v1
    reconstructed = proj + rej
    print(f"Projection + Rejection: {reconstructed.to_numpy()} (should equal v1)")
    print()

    # Verify orthogonality
    dot_proj_rej = flow.dot(proj, rej)
    print(f"Dot product of proj and rej: {dot_proj_rej:.6f} (should be ~0)")


def demo_interpolation():
    """Demonstrate interpolation methods."""
    print("\n=== Interpolation ===\n")

    # Linear interpolation
    start = flow.flow([0.0, 0.0])
    end = flow.flow([10.0, 10.0])

    print(f"Start: {start.to_numpy()}")
    print(f"End: {end.to_numpy()}")
    print()

    print("Linear interpolation:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = lerp(start, end, t)
        print(f"  t={t:.2f}: {result.to_numpy()}")
    print()

    # Spherical interpolation
    v1 = normalize(flow.flow([1.0, 0.0]))   # East
    v2 = normalize(flow.flow([0.0, 1.0]))   # North

    print("Spherical interpolation (unit vectors):")
    print(f"  v1 (East): {v1.to_numpy()}")
    print(f"  v2 (North): {v2.to_numpy()}")
    print()

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = slerp(v1, v2, t)
        angle = math.degrees(angle_between(v1, result))
        print(f"  t={t:.2f}: {result.to_numpy()} (angle from v1: {angle:.1f} deg)")


def demo_high_dimensional():
    """Demonstrate operations on high-dimensional vectors."""
    print("\n=== High-Dimensional Vectors ===\n")

    # Create high-dimensional random-ish vectors
    n = 1000
    v1 = flow.linspace(0, 1, n)
    v2 = flow.linspace(1, 0, n)

    print(f"Working with {n}-dimensional vectors")
    print()

    # Statistics
    print(f"v1 magnitude: {magnitude(v1):.4f}")
    print(f"v2 magnitude: {magnitude(v2):.4f}")
    print(f"Distance: {euclidean_distance(v1, v2):.4f}")
    print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")
    print(f"Angle: {math.degrees(angle_between(v1, v2)):.1f} degrees")


def demo_performance():
    """Demonstrate performance on large vectors."""
    import time

    print("\n=== Performance Demo ===\n")

    n = 1_000_000
    print(f"Vector size: {n:,} elements")

    v1 = flow.ones(n)
    v2 = flow.full(n, 2.0)

    # Warmup
    for _ in range(10):
        magnitude(v1)
        euclidean_distance(v1, v2)
        cosine_similarity(v1, v2)

    # Benchmark
    iterations = 50

    start = time.perf_counter()
    for _ in range(iterations):
        magnitude(v1)
    mag_time = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for _ in range(iterations):
        euclidean_distance(v1, v2)
    dist_time = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for _ in range(iterations):
        cosine_similarity(v1, v2)
    cos_time = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for _ in range(iterations):
        normalize(v1)
    norm_time = (time.perf_counter() - start) / iterations * 1000

    print(f"Magnitude:         {mag_time:.4f} ms")
    print(f"Euclidean distance: {dist_time:.4f} ms")
    print(f"Cosine similarity:  {cos_time:.4f} ms")
    print(f"Normalization:      {norm_time:.4f} ms")


def main():
    print("=" * 60)
    print("Vector Operations with Bud Flow Lang")
    print("=" * 60)

    flow.initialize()

    demo_basic_operations()
    demo_similarity()
    demo_projections()
    demo_interpolation()
    demo_high_dimensional()
    demo_performance()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
