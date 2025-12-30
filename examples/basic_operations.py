#!/usr/bin/env python3
"""Basic operations with Bud Flow Lang.

This example demonstrates fundamental array operations using the bud_flow_lang
library with SIMD acceleration.

Run from project root: python3 examples/basic_operations.py
"""

import sys
from pathlib import Path

# Add build directory to path if running from project directory
build_dir = Path(__file__).parent.parent / "build"
if build_dir.exists():
    sys.path.insert(0, str(build_dir))

import bud_flow_lang_py as flow


def main():
    print("=== Bud Flow Lang Basic Operations ===\n")

    # Array creation
    print("1. Array Creation")
    print("-" * 40)

    x = flow.arange(10)
    print(f"arange(10): {list(x.to_numpy())}")

    y = flow.ones(10)
    print(f"ones(10): {list(y.to_numpy())}")

    z = flow.full(10, 3.14)
    print(f"full(10, 3.14): {list(z.to_numpy()[:5])}...")

    w = flow.zeros(10)
    print(f"zeros(10): {list(w.to_numpy())}")

    # Basic arithmetic using operator overloading
    print("\n2. Basic Arithmetic (Operator Overloading)")
    print("-" * 40)

    a = flow.arange(5)
    b = flow.full(5, 2.0)

    print(f"a = {list(a.to_numpy())}")
    print(f"b = {list(b.to_numpy())}")
    print(f"a + b = {list((a + b).to_numpy())}")
    print(f"a - b = {list((a - b).to_numpy())}")
    print(f"a * b = {list((a * b).to_numpy())}")
    print(f"a / b = {list((a / b).to_numpy())}")

    # Unary operations
    print("\n3. Unary Operations")
    print("-" * 40)

    # Create perfect squares: [1, 4, 9, 16, 25] using (arange+1)^2
    indices = flow.arange(5) + flow.ones(5)  # [1, 2, 3, 4, 5]
    x = indices * indices  # [1, 4, 9, 16, 25]
    print(f"x = {list(x.to_numpy())}")
    print(f"sqrt(x) = {list(flow.sqrt(x).to_numpy())}")
    print(f"abs(-x) = {list(flow.abs(-x).to_numpy())}")
    print(f"exp(zeros) = {list(flow.exp(flow.zeros(5)).to_numpy())}")

    # Math functions
    print("\n4. Math Functions")
    print("-" * 40)

    x = flow.linspace(0.0, 3.14159, 5)
    print(f"x = {[round(v, 3) for v in x.to_numpy()]}")
    print(f"sin(x) = {[round(v, 3) for v in flow.sin(x).to_numpy()]}")
    print(f"cos(x) = {[round(v, 3) for v in flow.cos(x).to_numpy()]}")

    # Reductions
    print("\n5. Reduction Operations")
    print("-" * 40)

    x = flow.arange(10)
    print(f"x = {list(x.to_numpy())}")
    print(f"sum(x) = {flow.sum(x)}")
    print(f"mean(x) = {flow.mean(x)}")

    # Create [1, 2, 3] and [4, 5, 6] using arange
    a = flow.arange(3) + flow.ones(3)  # [1, 2, 3]
    b = flow.arange(3) + flow.full(3, 4.0)  # [4, 5, 6]
    print(f"a = {list(a.to_numpy())}")
    print(f"b = {list(b.to_numpy())}")
    print(f"dot(a, b) = {flow.dot(a, b)}")  # 1*4 + 2*5 + 3*6 = 32

    # Slicing
    print("\n6. Slicing")
    print("-" * 40)

    x = flow.arange(10)
    print(f"x = {list(x.to_numpy())}")
    print(f"x[2:5] = {list(x[2:5].to_numpy())}")
    print(f"x[::2] = {list(x[::2].to_numpy())}")

    # In-place operations
    print("\n7. In-place Operations")
    print("-" * 40)

    x = flow.arange(5)
    y = flow.ones(5)
    print(f"x before: {list(x.to_numpy())}")
    x += y
    print(f"x after x += y: {list(x.to_numpy())}")

    # FMA (Fused Multiply-Add)
    print("\n8. Fused Multiply-Add")
    print("-" * 40)

    # a = [1, 2, 3, 4], b = [2, 2, 2, 2], c = [1, 1, 1, 1]
    a = flow.arange(4) + flow.ones(4)  # [1, 2, 3, 4]
    b = flow.full(4, 2.0)              # [2, 2, 2, 2]
    c = flow.ones(4)                    # [1, 1, 1, 1]
    print(f"a = {list(a.to_numpy())}")
    print(f"b = {list(b.to_numpy())}")
    print(f"c = {list(c.to_numpy())}")
    print(f"fma(a, b, c) = a*b+c = {list(flow.fma(a, b, c).to_numpy())}")  # [3, 5, 7, 9]

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
