#!/usr/bin/env python3
"""
Test the Stream abstraction for lazy, chunked processing.
"""

import sys
import os

# Add build directory to path
BUILD_DIR = os.path.join(os.path.dirname(__file__), '../../build')
sys.path.insert(0, BUILD_DIR)

import numpy as np

# Import the module
try:
    import bud_flow_lang_py as flow
except ImportError as e:
    print(f"Failed to import bud_flow_lang_py: {e}")
    sys.exit(1)

# Initialize if not already done
if not flow.is_initialized():
    flow.initialize()

# Test colors
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def test_passed(name):
    print(f"  {GREEN}[PASS]{RESET} {name}")

def test_failed(name, msg):
    print(f"  {RED}[FAIL]{RESET} {name}: {msg}")
    return False

all_passed = True

print(f"\n{BOLD}{CYAN}{'='*70}")
print("  BUD FLOW LANG - STREAM ABSTRACTION TEST SUITE")
print(f"{'='*70}{RESET}\n")

# ==============================================================================
# Test 1: Basic Stream creation
# ==============================================================================
print(f"{BOLD}{CYAN}{'='*70}")
print("  Basic Stream Creation")
print(f"{'='*70}{RESET}")

try:
    # Create stream from zeros
    stream = flow.Stream.zeros(100)
    assert stream.size == 100, f"Expected size 100, got {stream.size}"
    test_passed("Stream.zeros(100)")
except Exception as e:
    all_passed = test_failed("Stream.zeros(100)", str(e))

try:
    # Create stream from range
    stream = flow.Stream.range(0.0, 10.0, 1.0)
    assert stream.size == 10, f"Expected size 10, got {stream.size}"
    test_passed("Stream.range(0, 10, 1)")
except Exception as e:
    all_passed = test_failed("Stream.range(0, 10, 1)", str(e))

try:
    # Create stream from Bunch
    bunch = flow.Bunch.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
    stream = flow.Stream.from_bunch(bunch)
    assert stream.size == 5, f"Expected size 5, got {stream.size}"
    test_passed("Stream.from_bunch()")
except Exception as e:
    all_passed = test_failed("Stream.from_bunch()", str(e))

# ==============================================================================
# Test 2: Stream collect
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Stream Collect")
print(f"{'='*70}{RESET}")

try:
    # Collect range stream
    stream = flow.Stream.range(0.0, 5.0, 1.0)
    result = stream.collect()
    expected = [0.0, 1.0, 2.0, 3.0, 4.0]
    result_list = list(result)
    assert len(result_list) == len(expected), f"Length mismatch: {len(result_list)} vs {len(expected)}"
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("Stream.range().collect()")
except Exception as e:
    all_passed = test_failed("Stream.range().collect()", str(e))

try:
    # Collect from Bunch
    bunch = flow.Bunch.from_list([1.0, 2.0, 3.0])
    stream = flow.Stream.from_bunch(bunch)
    result = stream.collect()
    expected = [1.0, 2.0, 3.0]
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("Stream.from_bunch().collect()")
except Exception as e:
    all_passed = test_failed("Stream.from_bunch().collect()", str(e))

# ==============================================================================
# Test 3: Stream map operations
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Stream Map Operations")
print(f"{'='*70}{RESET}")

try:
    # Test map_op with multiply
    stream = flow.Stream.range(1.0, 5.0, 1.0)
    mapped = stream.map_op("multiply", 2.0)
    result = mapped.collect()
    expected = [2.0, 4.0, 6.0, 8.0]
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("map_op('multiply', 2.0)")
except Exception as e:
    all_passed = test_failed("map_op('multiply', 2.0)", str(e))

try:
    # Test map_op with add
    stream = flow.Stream.range(1.0, 4.0, 1.0)
    mapped = stream.map_op("add", 10.0)
    result = mapped.collect()
    expected = [11.0, 12.0, 13.0]
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("map_op('add', 10.0)")
except Exception as e:
    all_passed = test_failed("map_op('add', 10.0)", str(e))

try:
    # Test chained map operations
    stream = flow.Stream.range(1.0, 4.0, 1.0)
    chained = stream.map_op("multiply", 2.0).map_op("add", 1.0)
    result = chained.collect()
    expected = [3.0, 5.0, 7.0]  # (1*2+1, 2*2+1, 3*2+1)
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("Chained map_op('multiply').map_op('add')")
except Exception as e:
    all_passed = test_failed("Chained map_op operations", str(e))

# ==============================================================================
# Test 4: Stream reductions
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Stream Reductions")
print(f"{'='*70}{RESET}")

try:
    # Test sum
    stream = flow.Stream.range(1.0, 6.0, 1.0)
    total = stream.sum()
    expected = 15.0  # 1 + 2 + 3 + 4 + 5
    assert abs(total - expected) < 1e-6, f"Sum mismatch: {total} vs {expected}"
    test_passed("Stream.sum()")
except Exception as e:
    all_passed = test_failed("Stream.sum()", str(e))

try:
    # Test min
    bunch = flow.Bunch.from_list([3.0, 1.0, 4.0, 1.5, 9.0])
    stream = flow.Stream.from_bunch(bunch)
    min_val = stream.min()
    expected = 1.0
    assert abs(min_val - expected) < 1e-6, f"Min mismatch: {min_val} vs {expected}"
    test_passed("Stream.min()")
except Exception as e:
    all_passed = test_failed("Stream.min()", str(e))

try:
    # Test max
    bunch = flow.Bunch.from_list([3.0, 1.0, 4.0, 1.5, 9.0])
    stream = flow.Stream.from_bunch(bunch)
    max_val = stream.max()
    expected = 9.0
    assert abs(max_val - expected) < 1e-6, f"Max mismatch: {max_val} vs {expected}"
    test_passed("Stream.max()")
except Exception as e:
    all_passed = test_failed("Stream.max()", str(e))

# ==============================================================================
# Test 5: Stream take and skip
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Stream Take and Skip")
print(f"{'='*70}{RESET}")

try:
    # Test take
    stream = flow.Stream.range(0.0, 10.0, 1.0)
    taken = stream.take(3)
    assert taken.size == 3, f"Expected size 3, got {taken.size}"
    result = taken.collect()
    expected = [0.0, 1.0, 2.0]
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("Stream.take(3)")
except Exception as e:
    all_passed = test_failed("Stream.take(3)", str(e))

try:
    # Test skip
    stream = flow.Stream.range(0.0, 10.0, 1.0)
    skipped = stream.skip(7)
    assert skipped.size == 3, f"Expected size 3, got {skipped.size}"
    result = skipped.collect()
    expected = [7.0, 8.0, 9.0]
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("Stream.skip(7)")
except Exception as e:
    all_passed = test_failed("Stream.skip(7)", str(e))

# ==============================================================================
# Test 6: Stream with Pipeline
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Stream with Pipeline")
print(f"{'='*70}{RESET}")

try:
    # Create pipeline
    pipeline = flow.Pipeline()
    pipeline.multiply(2.0).add(1.0)

    # Apply to stream
    bunch = flow.Bunch.from_list([1.0, 2.0, 3.0])
    stream = flow.Stream.from_bunch(bunch)
    mapped = stream.map(pipeline)
    result = mapped.collect()
    expected = [3.0, 5.0, 7.0]  # (1*2+1, 2*2+1, 3*2+1)
    result_list = list(result)
    for i, (r, e) in enumerate(zip(result_list, expected)):
        assert abs(r - e) < 1e-6, f"Mismatch at index {i}: {r} vs {e}"
    test_passed("Stream.map(Pipeline)")
except Exception as e:
    all_passed = test_failed("Stream.map(Pipeline)", str(e))

# ==============================================================================
# Test 7: Chunk iteration
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Chunk Iteration")
print(f"{'='*70}{RESET}")

try:
    # Create stream with custom chunk size
    stream = flow.Stream.range(0.0, 100.0, 1.0).chunk_size(25)

    chunk_count = 0
    total_elements = 0
    for chunk in stream:
        chunk_count += 1
        total_elements += chunk.size

    assert chunk_count == 4, f"Expected 4 chunks, got {chunk_count}"
    assert total_elements == 100, f"Expected 100 total elements, got {total_elements}"
    test_passed("Chunk iteration (100 elements, chunk_size=25)")
except Exception as e:
    all_passed = test_failed("Chunk iteration", str(e))

# ==============================================================================
# Test 8: Large stream processing
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Large Stream Processing")
print(f"{'='*70}{RESET}")

try:
    # Create large stream (use smaller n to avoid float32 precision issues)
    n = 10000
    stream = flow.Stream.range(1.0, float(n + 1), 1.0)

    # Sum should be n*(n+1)/2
    total = stream.sum()
    expected = n * (n + 1) / 2
    # Use relative tolerance for large sums (float32 has ~7 digits of precision)
    rel_error = abs(total - expected) / expected
    assert rel_error < 1e-4, f"Sum relative error too high: {rel_error}"
    test_passed(f"Large stream sum (n={n})")
except Exception as e:
    all_passed = test_failed(f"Large stream sum", str(e))

try:
    # Test with transformation on large data
    n = 10000
    stream = flow.Stream.range(0.0, float(n), 1.0)
    mapped = stream.map_op("multiply", 2.0).map_op("add", 1.0)

    # Count should be n
    count = mapped.count()
    assert count == n, f"Count mismatch: {count} vs {n}"
    test_passed(f"Large stream with map operations (n={n})")
except Exception as e:
    all_passed = test_failed("Large stream with map", str(e))

# ==============================================================================
# Test 9: Predicates (any/all)
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Predicates")
print(f"{'='*70}{RESET}")

try:
    # Test any
    bunch = flow.Bunch.from_list([1.0, 2.0, 3.0, 10.0, 5.0])
    stream = flow.Stream.from_bunch(bunch)

    has_large = stream.any(lambda x: x > 5.0)
    assert has_large == True, f"Expected any(x > 5) to be True"

    has_negative = flow.Stream.from_bunch(bunch).any(lambda x: x < 0.0)
    assert has_negative == False, f"Expected any(x < 0) to be False"
    test_passed("Stream.any()")
except Exception as e:
    all_passed = test_failed("Stream.any()", str(e))

try:
    # Test all
    bunch = flow.Bunch.from_list([2.0, 4.0, 6.0, 8.0])
    stream = flow.Stream.from_bunch(bunch)

    all_positive = stream.all(lambda x: x > 0.0)
    assert all_positive == True, f"Expected all(x > 0) to be True"

    all_large = flow.Stream.from_bunch(bunch).all(lambda x: x > 5.0)
    assert all_large == False, f"Expected all(x > 5) to be False"
    test_passed("Stream.all()")
except Exception as e:
    all_passed = test_failed("Stream.all()", str(e))

# ==============================================================================
# Test 10: Stream stats
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  Stream Statistics")
print(f"{'='*70}{RESET}")

try:
    stream = flow.Stream.range(0.0, 1000.0, 1.0)
    result = stream.collect()

    stats = stream.stats
    assert stats.total_elements == 1000, f"Expected 1000 elements, got {stats.total_elements}"
    assert stats.chunks_processed > 0, f"Expected some chunks processed"
    test_passed("Stream.stats after collect()")
except Exception as e:
    all_passed = test_failed("Stream.stats", str(e))

# ==============================================================================
# Summary
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  TEST SUMMARY")
print(f"{'='*70}{RESET}")

if all_passed:
    print(f"\n{GREEN}{BOLD}*** ALL STREAM TESTS PASSED ***{RESET}\n")
else:
    print(f"\n{RED}{BOLD}*** SOME TESTS FAILED ***{RESET}\n")
    sys.exit(1)
