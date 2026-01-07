#!/usr/bin/env python3
"""Debug test for @flow.kernel decorator."""

import sys
sys.path.insert(0, '/home/bud/Desktop/bud_simd/bud_flow_lang/build')
import bud_flow_lang_py as flow
import numpy as np

# Initialize with debug logging
flow.initialize()

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

print("\n=== Test 1: Simple identity kernel ===")

@flow.kernel
def identity(x):
    print(f"  Inside kernel: x is {type(x)}")
    print(f"  x.size = {len(x)}")
    return x

x = flow.Bunch.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))
print(f"Input: {x.to_numpy()}")
print("Calling identity kernel...")
result = identity(x)
print(f"Result: {result.to_numpy()}")

print("\n=== Test 2: Simple addition kernel ===")

@flow.kernel
def add_one(x):
    print(f"  Inside add_one: x is {type(x)}")
    ones = flow.ones(len(x))
    print(f"  Created ones, is tracer: {hasattr(ones, 'is_tracing')}")
    result = x + ones
    print(f"  result is {type(result)}")
    return result

x = flow.Bunch.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))
print(f"Input: {x.to_numpy()}")
print("Calling add_one kernel...")
result = add_one(x)
print(f"Result: {result.to_numpy()}")
