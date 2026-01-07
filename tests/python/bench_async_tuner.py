#!/usr/bin/env python3
"""
Benchmark: Sync vs Async Recording Performance
Measures the actual overhead reduction from async recording.
"""

import sys
import os
import time

sys.stdout.reconfigure(line_buffering=True)

BUILD_DIR = os.path.join(os.path.dirname(__file__), '../../build')
sys.path.insert(0, BUILD_DIR)

import bud_flow_lang_py as flow

if not flow.is_initialized():
    flow.initialize()

# Colors
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

print(f"\n{BOLD}{CYAN}{'='*70}")
print("  ASYNC TUNING PERFORMANCE BENCHMARK")
print(f"{'='*70}{RESET}\n")

# =============================================================================
# Benchmark Configuration
# =============================================================================
NUM_ITERATIONS = 10000
NUM_WARMUP = 1000

# =============================================================================
# Setup: Create two tuners (sync and async)
# =============================================================================
print(f"{YELLOW}Setting up tuners...{RESET}")

# Sync tuner
sync_tuner = flow.ContinuousAutoTuner()
variants = []
for i in range(3):
    v = flow.KernelVariant()
    v.name = f"sync_variant_{i}"
    variants.append(v)
sync_tuner.register_kernel_variants("sync_kernel", variants)

# Async tuner
async_tuner = flow.ContinuousAutoTuner()
variants = []
for i in range(3):
    v = flow.KernelVariant()
    v.name = f"async_variant_{i}"
    variants.append(v)
async_tuner.register_kernel_variants("async_kernel", variants)
async_tuner.enable_async_recording(True)

print(f"  Sync tuner ready")
print(f"  Async tuner ready (background thread started)")

# =============================================================================
# Warmup
# =============================================================================
print(f"\n{YELLOW}Warming up ({NUM_WARMUP} iterations each)...{RESET}")

for _ in range(NUM_WARMUP):
    idx = sync_tuner.select_variant("sync_kernel")
    sync_tuner.record_execution("sync_kernel", idx, 1000.0)

for _ in range(NUM_WARMUP):
    idx = async_tuner.select_variant("async_kernel")
    async_tuner.record_execution_async("async_kernel", idx, 1000.0)

async_tuner.flush_async_records()
print(f"  Warmup complete")

# =============================================================================
# Benchmark 1: Sync Recording Overhead
# =============================================================================
print(f"\n{BOLD}{CYAN}Benchmark 1: Synchronous Recording{RESET}")
print(f"  Iterations: {NUM_ITERATIONS}")

sync_times = []
for _ in range(NUM_ITERATIONS):
    idx = sync_tuner.select_variant("sync_kernel")

    start = time.perf_counter_ns()
    sync_tuner.record_execution("sync_kernel", idx, 1000.0)
    end = time.perf_counter_ns()

    sync_times.append(end - start)

sync_avg = sum(sync_times) / len(sync_times)
sync_min = min(sync_times)
sync_max = max(sync_times)
sync_p50 = sorted(sync_times)[len(sync_times) // 2]
sync_p99 = sorted(sync_times)[int(len(sync_times) * 0.99)]

print(f"  {GREEN}Average:{RESET} {sync_avg:.1f} ns")
print(f"  {GREEN}Min:{RESET}     {sync_min} ns")
print(f"  {GREEN}Max:{RESET}     {sync_max} ns")
print(f"  {GREEN}P50:{RESET}     {sync_p50} ns")
print(f"  {GREEN}P99:{RESET}     {sync_p99} ns")

# =============================================================================
# Benchmark 2: Async Recording Overhead (Non-Blocking)
# =============================================================================
print(f"\n{BOLD}{CYAN}Benchmark 2: Asynchronous Recording (Non-Blocking){RESET}")
print(f"  Iterations: {NUM_ITERATIONS}")

async_times = []
for _ in range(NUM_ITERATIONS):
    idx = async_tuner.select_variant("async_kernel")

    start = time.perf_counter_ns()
    async_tuner.record_execution_async("async_kernel", idx, 1000.0)
    end = time.perf_counter_ns()

    async_times.append(end - start)

# Wait for all async records to be processed
async_tuner.flush_async_records()

async_avg = sum(async_times) / len(async_times)
async_min = min(async_times)
async_max = max(async_times)
async_p50 = sorted(async_times)[len(async_times) // 2]
async_p99 = sorted(async_times)[int(len(async_times) * 0.99)]

print(f"  {GREEN}Average:{RESET} {async_avg:.1f} ns")
print(f"  {GREEN}Min:{RESET}     {async_min} ns")
print(f"  {GREEN}Max:{RESET}     {async_max} ns")
print(f"  {GREEN}P50:{RESET}     {async_p50} ns")
print(f"  {GREEN}P99:{RESET}     {async_p99} ns")

# =============================================================================
# Benchmark 3: Full Select + Record Cycle
# =============================================================================
print(f"\n{BOLD}{CYAN}Benchmark 3: Full Cycle (Select + Record){RESET}")
print(f"  Iterations: {NUM_ITERATIONS}")

# Sync full cycle
sync_full_times = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter_ns()
    idx = sync_tuner.select_variant("sync_kernel")
    sync_tuner.record_execution("sync_kernel", idx, 1000.0)
    end = time.perf_counter_ns()
    sync_full_times.append(end - start)

sync_full_avg = sum(sync_full_times) / len(sync_full_times)

# Async full cycle
async_full_times = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter_ns()
    idx = async_tuner.select_variant("async_kernel")
    async_tuner.record_execution_async("async_kernel", idx, 1000.0)
    end = time.perf_counter_ns()
    async_full_times.append(end - start)

async_tuner.flush_async_records()
async_full_avg = sum(async_full_times) / len(async_full_times)

print(f"  {GREEN}Sync Full Cycle:{RESET}  {sync_full_avg:.1f} ns")
print(f"  {GREEN}Async Full Cycle:{RESET} {async_full_avg:.1f} ns")

# =============================================================================
# Benchmark 4: Throughput Test (Burst Recording)
# =============================================================================
print(f"\n{BOLD}{CYAN}Benchmark 4: Throughput (Burst Recording){RESET}")
BURST_SIZE = 10000

# Sync throughput
start = time.perf_counter_ns()
for i in range(BURST_SIZE):
    sync_tuner.record_execution("sync_kernel", i % 3, 1000.0)
sync_burst_time = time.perf_counter_ns() - start

# Async throughput (non-blocking)
start = time.perf_counter_ns()
for i in range(BURST_SIZE):
    async_tuner.record_execution_async("async_kernel", i % 3, 1000.0)
async_burst_time = time.perf_counter_ns() - start

# Wait for async to finish processing
flush_start = time.perf_counter_ns()
async_tuner.flush_async_records()
flush_time = time.perf_counter_ns() - flush_start

print(f"  {GREEN}Sync burst ({BURST_SIZE} records):{RESET}")
print(f"    Total: {sync_burst_time / 1e6:.2f} ms")
print(f"    Per record: {sync_burst_time / BURST_SIZE:.1f} ns")
print(f"  {GREEN}Async burst ({BURST_SIZE} records):{RESET}")
print(f"    Enqueue time: {async_burst_time / 1e6:.2f} ms")
print(f"    Per enqueue: {async_burst_time / BURST_SIZE:.1f} ns")
print(f"    Flush wait: {flush_time / 1e6:.2f} ms")
print(f"    Total (enqueue + flush): {(async_burst_time + flush_time) / 1e6:.2f} ms")

# =============================================================================
# Async Stats
# =============================================================================
print(f"\n{BOLD}{CYAN}Async Tuner Statistics{RESET}")
stats = async_tuner.statistics()
print(f"  Records queued:    {stats.async_records_queued}")
print(f"  Records processed: {stats.async_records_processed}")
print(f"  Max queue size:    {stats.async_queue_max_size}")
print(f"  Avg latency:       {stats.async_avg_latency_ns / 1e3:.1f} µs")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  SUMMARY")
print(f"{'='*70}{RESET}")

speedup_record = sync_avg / async_avg if async_avg > 0 else 0
speedup_full = sync_full_avg / async_full_avg if async_full_avg > 0 else 0
speedup_burst = sync_burst_time / async_burst_time if async_burst_time > 0 else 0

print(f"\n  {BOLD}Record Overhead Reduction:{RESET}")
print(f"    Sync:  {sync_avg:.1f} ns")
print(f"    Async: {async_avg:.1f} ns")
print(f"    {GREEN}Speedup: {speedup_record:.1f}x{RESET}")

print(f"\n  {BOLD}Full Cycle Reduction:{RESET}")
print(f"    Sync:  {sync_full_avg:.1f} ns")
print(f"    Async: {async_full_avg:.1f} ns")
print(f"    {GREEN}Speedup: {speedup_full:.1f}x{RESET}")

print(f"\n  {BOLD}Burst Throughput:{RESET}")
print(f"    Sync:  {BURST_SIZE / (sync_burst_time / 1e9) / 1e6:.2f} M records/sec")
print(f"    Async: {BURST_SIZE / (async_burst_time / 1e9) / 1e6:.2f} M records/sec (enqueue only)")
print(f"    {GREEN}Speedup: {speedup_burst:.1f}x{RESET}")

# Determine if optimization is working
if speedup_record > 1.5:
    print(f"\n{GREEN}{BOLD}✓ Async optimization is working effectively!{RESET}")
    print(f"  The non-blocking record reduces hot-path overhead by {(1 - 1/speedup_record) * 100:.0f}%")
else:
    print(f"\n{YELLOW}{BOLD}⚠ Async overhead reduction is minimal{RESET}")
    print(f"  This may be due to low contention or fast mutex acquisition")
    print(f"  Async still helps with burst scenarios and reduces tail latency")

print()
