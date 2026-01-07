#!/usr/bin/env python3
"""
Test the ContinuousAutoTuner for runtime-adaptive kernel selection.
"""

import sys
import os
import time
import random

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add build directory to path
BUILD_DIR = os.path.join(os.path.dirname(__file__), '../../build')
sys.path.insert(0, BUILD_DIR)
print(f"Using build dir: {BUILD_DIR}", flush=True)

import numpy as np

# Import the module
try:
    import bud_flow_lang_py as flow
except ImportError as e:
    print(f"Failed to import bud_flow_lang_py: {e}", flush=True)
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
    print(f"  {GREEN}[PASS]{RESET} {name}", flush=True)

def test_failed(name, msg):
    print(f"  {RED}[FAIL]{RESET} {name}: {msg}", flush=True)
    return False

all_passed = True

print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  BUD FLOW LANG - CONTINUOUS AUTO-TUNER TEST SUITE", flush=True)
print(f"{'='*70}{RESET}\n", flush=True)

# ==============================================================================
# Test 1: Basic Configuration
# ==============================================================================
print(f"{BOLD}{CYAN}{'='*70}", flush=True)
print("  Basic Configuration", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    config = flow.ContinuousTunerConfig()
    assert config.prior_alpha == 1.0, f"Expected prior_alpha 1.0, got {config.prior_alpha}"
    assert config.prior_beta == 1.0, f"Expected prior_beta 1.0, got {config.prior_beta}"
    assert config.warmup_executions == 10, f"Expected warmup 10, got {config.warmup_executions}"
    test_passed("Default ContinuousTunerConfig")
except Exception as e:
    all_passed = test_failed("Default ContinuousTunerConfig", str(e))

try:
    config = flow.ContinuousTunerConfig()
    config.prior_alpha = 2.0
    config.prior_beta = 2.0
    config.exploration_bonus = 0.2
    config.warmup_executions = 5
    assert abs(config.prior_alpha - 2.0) < 1e-5
    assert abs(config.prior_beta - 2.0) < 1e-5
    assert abs(config.exploration_bonus - 0.2) < 1e-5
    assert config.warmup_executions == 5
    test_passed("Modified ContinuousTunerConfig")
except Exception as e:
    all_passed = test_failed("Modified ContinuousTunerConfig", str(e))

# ==============================================================================
# Test 2: Tuner Creation
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Tuner Creation", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()
    assert tuner.enabled == True, "Tuner should be enabled by default"
    test_passed("Create ContinuousAutoTuner()")
except Exception as e:
    all_passed = test_failed("Create ContinuousAutoTuner()", str(e))

try:
    config = flow.ContinuousTunerConfig()
    config.warmup_executions = 20
    tuner = flow.ContinuousAutoTuner(config)
    assert tuner.config.warmup_executions == 20
    test_passed("Create ContinuousAutoTuner(config)")
except Exception as e:
    all_passed = test_failed("Create ContinuousAutoTuner(config)", str(e))

# ==============================================================================
# Test 3: Kernel Registration
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Kernel Registration", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "default"
    variant.id = 0

    result = tuner.register_kernel("test_kernel", variant)
    assert result == True, "Registration should succeed"
    assert tuner.has_kernel("test_kernel") == True
    assert tuner.num_variants("test_kernel") == 1
    test_passed("Register single variant kernel")
except Exception as e:
    all_passed = test_failed("Register single variant kernel", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variants = []
    for i in range(3):
        v = flow.KernelVariant()
        v.name = f"variant_{i}"
        v.id = i
        variants.append(v)

    result = tuner.register_kernel_variants("multi_kernel", variants)
    assert result == True
    assert tuner.num_variants("multi_kernel") == 3
    test_passed("Register multi-variant kernel")
except Exception as e:
    all_passed = test_failed("Register multi-variant kernel", str(e))

try:
    tuner = flow.ContinuousAutoTuner()
    assert tuner.has_kernel("nonexistent") == False
    assert tuner.num_variants("nonexistent") == 0
    test_passed("Query nonexistent kernel")
except Exception as e:
    all_passed = test_failed("Query nonexistent kernel", str(e))

# ==============================================================================
# Test 4: Variant Selection (Thompson Sampling)
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Variant Selection (Thompson Sampling)", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    # Create 3 variants
    variants = []
    for i in range(3):
        v = flow.KernelVariant()
        v.name = f"variant_{i}"
        variants.append(v)

    tuner.register_kernel_variants("ts_kernel", variants)

    # During warmup, should round-robin
    selections = []
    for _ in range(9):  # 3 variants * 3 warmup passes
        idx = tuner.select_variant("ts_kernel")
        selections.append(idx)

    # Check that all variants are explored
    assert 0 in selections
    assert 1 in selections
    assert 2 in selections
    test_passed("Warmup round-robin selection")
except Exception as e:
    all_passed = test_failed("Warmup round-robin selection", str(e))

try:
    tuner = flow.ContinuousAutoTuner()
    config = flow.ContinuousTunerConfig()
    config.warmup_executions = 3
    tuner.set_config(config)

    variants = []
    for i in range(2):
        v = flow.KernelVariant()
        v.name = f"variant_{i}"
        variants.append(v)

    tuner.register_kernel_variants("fast_slow", variants)

    # Warmup phase - record times
    for i in range(10):
        idx = tuner.select_variant("fast_slow")
        # Variant 0 is "fast" (lower time), variant 1 is "slow"
        time_ns = 100.0 if idx == 0 else 500.0
        time_ns += random.uniform(-10, 10)  # Add noise
        tuner.record_execution("fast_slow", idx, time_ns)

    # After learning, variant 0 (fast) should be selected more often
    fast_count = 0
    for _ in range(100):
        idx = tuner.select_variant("fast_slow")
        if idx == 0:
            fast_count += 1
        # Record to continue learning
        time_ns = 100.0 if idx == 0 else 500.0
        tuner.record_execution("fast_slow", idx, time_ns)

    # Fast variant should be selected majority of time
    assert fast_count > 70, f"Expected fast variant to be selected >70%, got {fast_count}%"
    test_passed(f"Thompson Sampling learns faster variant ({fast_count}% selection)")
except Exception as e:
    all_passed = test_failed("Thompson Sampling learns faster variant", str(e))

# ==============================================================================
# Test 5: Execution Recording
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Execution Recording", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "measured"
    tuner.register_kernel("measured_kernel", variant)

    # Record several executions
    for _ in range(10):
        tuner.record_execution("measured_kernel", 0, 1000.0)

    # Get variant and check stats
    v = tuner.get_variant("measured_kernel", 0)
    assert v is not None
    assert v.total_executions == 10, f"Expected 10 executions, got {v.total_executions}"
    test_passed("Execution recording and stats")
except Exception as e:
    all_passed = test_failed("Execution recording and stats", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "size_tracked"
    tuner.register_kernel("size_kernel", variant)

    # Record with different sizes
    for size in [100, 1000, 10000]:
        for _ in range(5):
            tuner.record_execution_with_size("size_kernel", 0, size, float(size) / 10.0)

    # Get variant
    v = tuner.get_variant("size_kernel", 0)
    assert v.total_executions == 15
    test_passed("Execution recording with input sizes")
except Exception as e:
    all_passed = test_failed("Execution recording with input sizes", str(e))

# ==============================================================================
# Test 6: Statistics
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Statistics", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    variants = []
    for i in range(2):
        v = flow.KernelVariant()
        v.name = f"stat_variant_{i}"
        variants.append(v)

    tuner.register_kernel_variants("stats_kernel", variants)

    # Generate activity
    for _ in range(50):
        idx = tuner.select_variant("stats_kernel")
        tuner.record_execution("stats_kernel", idx, random.uniform(100, 200))

    stats = tuner.statistics()
    assert stats.total_selections >= 50, f"Expected >=50 selections, got {stats.total_selections}"
    assert stats.total_observations >= 50
    assert stats.variants_created >= 2
    test_passed(f"Statistics (selections={stats.total_selections}, observations={stats.total_observations})")
except Exception as e:
    all_passed = test_failed("Statistics", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "event_variant"
    tuner.register_kernel("event_kernel", variant)

    for _ in range(10):
        idx = tuner.select_variant("event_kernel")
        tuner.record_execution("event_kernel", idx, 100.0)

    events = tuner.recent_events(100)
    assert len(events) > 0, "Should have events"
    test_passed(f"Recent events (count={len(events)})")
except Exception as e:
    all_passed = test_failed("Recent events", str(e))

# ==============================================================================
# Test 7: Thompson State
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Thompson State", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    state = flow.ThompsonState()
    assert state.alpha == 1.0
    assert state.beta == 1.0
    assert state.num_samples == 0

    mean = state.mean()
    var = state.variance()
    assert 0.4 < mean < 0.6, f"Expected mean ~0.5, got {mean}"
    assert var > 0, f"Variance should be positive"
    test_passed(f"ThompsonState (mean={mean:.3f}, var={var:.3f})")
except Exception as e:
    all_passed = test_failed("ThompsonState", str(e))

# ==============================================================================
# Test 8: Enable/Disable
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Enable/Disable", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()
    assert tuner.enabled == True

    tuner.enabled = False
    assert tuner.enabled == False

    # When disabled, should return 0 always
    variant = flow.KernelVariant()
    variant.name = "disabled_test"
    tuner.register_kernel("disabled_kernel", variant)

    for _ in range(10):
        idx = tuner.select_variant("disabled_kernel")
        assert idx == 0

    tuner.enabled = True
    assert tuner.enabled == True
    test_passed("Enable/Disable toggle")
except Exception as e:
    all_passed = test_failed("Enable/Disable toggle", str(e))

# ==============================================================================
# Test 9: Persistence
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Persistence", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "persist_variant"
    tuner.register_kernel("persist_kernel", variant)

    # Generate some history
    for _ in range(20):
        idx = tuner.select_variant("persist_kernel")
        tuner.record_execution("persist_kernel", idx, 100.0)

    # Serialize
    data = tuner.serialize()
    assert len(data) > 0, "Serialized data should not be empty"
    assert "ContinuousAutoTuner" in data
    test_passed(f"Serialize (size={len(data)} bytes)")
except Exception as e:
    all_passed = test_failed("Serialize", str(e))

try:
    tuner1 = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "persist_variant"
    tuner1.register_kernel("persist_kernel", variant)

    for _ in range(20):
        idx = tuner1.select_variant("persist_kernel")
        tuner1.record_execution("persist_kernel", idx, 100.0)

    data = tuner1.serialize()

    # Create new tuner with same kernel structure
    tuner2 = flow.ContinuousAutoTuner()
    variant2 = flow.KernelVariant()
    variant2.name = "persist_variant"
    tuner2.register_kernel("persist_kernel", variant2)

    result = tuner2.deserialize(data)
    assert result == True, "Deserialize should succeed"
    test_passed("Deserialize")
except Exception as e:
    all_passed = test_failed("Deserialize", str(e))

# ==============================================================================
# Test 10: Warm Start (History Import)
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Warm Start (History Import)", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    variants = []
    for i in range(2):
        v = flow.KernelVariant()
        v.name = f"warmstart_{i}"
        variants.append(v)

    tuner.register_kernel_variants("warmstart_kernel", variants)

    # Import historical data - variant 0 was faster
    history = [(0, 100.0), (0, 105.0), (0, 95.0),
               (1, 500.0), (1, 510.0), (1, 490.0)]
    tuner.import_history("warmstart_kernel", history)

    # Export and verify
    exported = tuner.export_history("warmstart_kernel")
    assert len(exported) == 2, f"Expected 2 variants in history, got {len(exported)}"
    test_passed("History import/export")
except Exception as e:
    all_passed = test_failed("History import/export", str(e))

# ==============================================================================
# Test 11: Reset
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Reset", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "reset_variant"
    tuner.register_kernel("reset_kernel", variant)

    for _ in range(50):
        idx = tuner.select_variant("reset_kernel")
        tuner.record_execution("reset_kernel", idx, 100.0)

    stats_before = tuner.statistics()
    assert stats_before.total_selections > 0

    tuner.reset_statistics()

    stats_after = tuner.statistics()
    assert stats_after.total_selections == 0
    assert stats_after.total_observations == 0
    test_passed("Reset statistics")
except Exception as e:
    all_passed = test_failed("Reset statistics", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "reset_variant"
    tuner.register_kernel("reset_kernel", variant)

    for _ in range(50):
        idx = tuner.select_variant("reset_kernel")
        tuner.record_execution("reset_kernel", idx, 100.0)

    v_before = tuner.get_variant("reset_kernel", 0)
    assert v_before.total_executions > 0

    tuner.reset_kernel("reset_kernel")

    v_after = tuner.get_variant("reset_kernel", 0)
    assert v_after.total_executions == 0
    test_passed("Reset kernel")
except Exception as e:
    all_passed = test_failed("Reset kernel", str(e))

# ==============================================================================
# Test 12: Global Tuner
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Global Tuner", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    global_tuner = flow.get_global_tuner()
    assert global_tuner is not None
    assert global_tuner.enabled == True
    test_passed("Get global tuner")
except Exception as e:
    all_passed = test_failed("Get global tuner", str(e))

try:
    config = flow.ContinuousTunerConfig()
    config.warmup_executions = 15
    flow.configure_global_tuner(config)

    global_tuner = flow.get_global_tuner()
    assert global_tuner.config.warmup_executions == 15
    test_passed("Configure global tuner")
except Exception as e:
    all_passed = test_failed("Configure global tuner", str(e))

# ==============================================================================
# Test 13: Convergence Test (Learning Performance)
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Convergence Test (Learning Performance)", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()
    config = flow.ContinuousTunerConfig()
    config.warmup_executions = 5
    config.exploration_bonus = 0.05
    tuner.set_config(config)

    # Create 4 variants with different performance characteristics
    variants = []
    for i in range(4):
        v = flow.KernelVariant()
        v.name = f"perf_variant_{i}"
        variants.append(v)

    tuner.register_kernel_variants("convergence_kernel", variants)

    # Variant execution times: [best, good, medium, slow]
    variant_times = [50.0, 100.0, 200.0, 400.0]

    selections = {i: 0 for i in range(4)}

    # Run 100 iterations (reduced for faster testing)
    for _ in range(100):
        idx = tuner.select_variant("convergence_kernel")
        selections[idx] += 1

        # Add noise to times
        base_time = variant_times[idx]
        time_ns = base_time + random.gauss(0, base_time * 0.1)
        tuner.record_execution("convergence_kernel", idx, max(1.0, time_ns))

    # Best variant (index 0) should be selected most
    best_selections = selections[0]
    total_selections = sum(selections.values())
    best_ratio = best_selections / total_selections

    print(f"    Selection distribution: {selections}", flush=True)
    print(f"    Best variant ratio: {best_ratio:.2%}", flush=True)

    assert best_ratio > 0.5, f"Best variant should be selected >50%, got {best_ratio:.2%}"
    test_passed(f"Convergence to best variant ({best_ratio:.1%} selection rate)")
except Exception as e:
    all_passed = test_failed("Convergence test", str(e))

# ==============================================================================
# Test 14: Multi-Kernel Handling
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Multi-Kernel Handling", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    tuner = flow.ContinuousAutoTuner()

    # Register multiple kernels
    for k in range(5):
        variants = []
        for v in range(3):
            var = flow.KernelVariant()
            var.name = f"k{k}_v{v}"
            variants.append(var)
        tuner.register_kernel_variants(f"kernel_{k}", variants)

    # Run selections on all kernels
    for _ in range(100):
        for k in range(5):
            kernel_name = f"kernel_{k}"
            idx = tuner.select_variant(kernel_name)
            tuner.record_execution(kernel_name, idx, random.uniform(100, 200))

    # Verify all kernels work
    for k in range(5):
        assert tuner.has_kernel(f"kernel_{k}")
        assert tuner.num_variants(f"kernel_{k}") == 3

    stats = tuner.statistics()
    assert stats.total_selections >= 500
    test_passed(f"Multi-kernel handling (5 kernels, {stats.total_selections} selections)")
except Exception as e:
    all_passed = test_failed("Multi-kernel handling", str(e))

# ==============================================================================
# Test 15: Async Recording
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  Async Recording", flush=True)
print(f"{'='*70}{RESET}", flush=True)

try:
    # Test async config defaults
    config = flow.ContinuousTunerConfig()
    assert config.enable_async_recording == False, "Async should be disabled by default"
    assert config.async_queue_capacity == 4096
    assert config.async_batch_size == 32
    test_passed("Async config defaults")
except Exception as e:
    all_passed = test_failed("Async config defaults", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    # Should not be async by default
    assert tuner.is_async_recording_enabled() == False

    # Enable async
    tuner.enable_async_recording(True)
    assert tuner.is_async_recording_enabled() == True

    # Disable async
    tuner.enable_async_recording(False)
    assert tuner.is_async_recording_enabled() == False

    test_passed("Enable/disable async recording")
except Exception as e:
    all_passed = test_failed("Enable/disable async recording", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variants = []
    for i in range(2):
        v = flow.KernelVariant()
        v.name = f"async_variant_{i}"
        variants.append(v)

    tuner.register_kernel_variants("async_kernel", variants)

    # Enable async mode
    tuner.enable_async_recording(True)
    assert tuner.is_async_recording_enabled() == True

    # Record asynchronously
    for i in range(100):
        idx = i % 2
        time_ns = 100.0 if idx == 0 else 500.0
        tuner.record_execution_async("async_kernel", idx, time_ns)

    # Wait for processing
    tuner.flush_async_records()

    # Verify records were processed
    v0 = tuner.get_variant("async_kernel", 0)
    v1 = tuner.get_variant("async_kernel", 1)

    total_executions = v0.total_executions + v1.total_executions
    assert total_executions == 100, f"Expected 100 executions, got {total_executions}"

    # Check async stats
    stats = tuner.statistics()
    assert stats.async_records_queued >= 100, f"Expected >=100 queued, got {stats.async_records_queued}"
    assert stats.async_records_processed >= 100, f"Expected >=100 processed, got {stats.async_records_processed}"

    test_passed(f"Async recording (queued={stats.async_records_queued}, processed={stats.async_records_processed})")
except Exception as e:
    all_passed = test_failed("Async recording", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "async_size_variant"
    tuner.register_kernel("async_size_kernel", variant)

    tuner.enable_async_recording(True)

    # Record with sizes asynchronously
    for size in [100, 1000, 10000]:
        for _ in range(10):
            tuner.record_execution_async_with_size("async_size_kernel", 0, size, float(size) / 10.0)

    tuner.flush_async_records()

    v = tuner.get_variant("async_size_kernel", 0)
    assert v.total_executions == 30, f"Expected 30 executions, got {v.total_executions}"

    test_passed("Async recording with input sizes")
except Exception as e:
    all_passed = test_failed("Async recording with input sizes", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variants = []
    for i in range(2):
        v = flow.KernelVariant()
        v.name = f"async_learn_{i}"
        variants.append(v)

    tuner.register_kernel_variants("async_learn_kernel", variants)
    tuner.enable_async_recording(True)

    # Learning test with async - variant 0 is faster
    for _ in range(200):
        idx = tuner.select_variant("async_learn_kernel")
        time_ns = 100.0 if idx == 0 else 500.0
        time_ns += random.uniform(-10, 10)
        tuner.record_execution_async("async_learn_kernel", idx, time_ns)

    tuner.flush_async_records()

    # After learning, best variant should be 0
    best = tuner.get_best_variant("async_learn_kernel")
    assert best is not None

    # Count selections for the faster variant
    fast_count = 0
    for _ in range(100):
        idx = tuner.select_variant("async_learn_kernel")
        if idx == 0:
            fast_count += 1
        tuner.record_execution_async("async_learn_kernel", idx, 100.0 if idx == 0 else 500.0)

    tuner.flush_async_records()

    assert fast_count > 60, f"Async learning: expected fast variant >60%, got {fast_count}%"
    test_passed(f"Async recording learns correctly ({fast_count}% fast selections)")
except Exception as e:
    all_passed = test_failed("Async recording learns correctly", str(e))

try:
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "queue_test"
    tuner.register_kernel("queue_kernel", variant)

    tuner.enable_async_recording(True)

    # Flood the queue
    for _ in range(50):
        tuner.record_execution_async("queue_kernel", 0, 100.0)

    # Check queue size (may be partially processed already)
    queue_size = tuner.async_queue_size()

    tuner.flush_async_records()

    # After flush, queue should be empty
    assert tuner.async_queue_size() == 0

    test_passed(f"Async queue management (peak size ~{queue_size})")
except Exception as e:
    all_passed = test_failed("Async queue management", str(e))

try:
    # Test fallback when async is disabled
    tuner = flow.ContinuousAutoTuner()

    variant = flow.KernelVariant()
    variant.name = "fallback_variant"
    tuner.register_kernel("fallback_kernel", variant)

    # Async not enabled, but call async method
    for _ in range(10):
        tuner.record_execution_async("fallback_kernel", 0, 100.0)

    # Should still record (fallback to sync)
    v = tuner.get_variant("fallback_kernel", 0)
    assert v.total_executions == 10, f"Fallback: expected 10 executions, got {v.total_executions}"

    test_passed("Async fallback to sync when disabled")
except Exception as e:
    all_passed = test_failed("Async fallback to sync when disabled", str(e))

# ==============================================================================
# Summary
# ==============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}", flush=True)
print("  TEST SUMMARY", flush=True)
print(f"{'='*70}{RESET}", flush=True)

if all_passed:
    print(f"\n{GREEN}{BOLD}*** ALL CONTINUOUS AUTO-TUNER TESTS PASSED ***{RESET}\n", flush=True)
else:
    print(f"\n{RED}{BOLD}*** SOME TESTS FAILED ***{RESET}\n", flush=True)
    sys.exit(1)
