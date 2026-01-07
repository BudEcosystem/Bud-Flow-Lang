#!/usr/bin/env python3
"""
Benchmark: Concurrent Async Recording Performance
Tests where async actually helps - multi-threaded execution scenarios.
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

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
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

print(f"\n{BOLD}{CYAN}{'='*70}")
print("  CONCURRENT ASYNC TUNING BENCHMARK")
print(f"{'='*70}{RESET}\n")

# =============================================================================
# Configuration
# =============================================================================
NUM_THREADS = 8
ITERATIONS_PER_THREAD = 5000
SIMULATED_KERNEL_TIME_NS = 100  # Simulate 100ns kernel execution

print(f"{YELLOW}Configuration:{RESET}")
print(f"  Threads: {NUM_THREADS}")
print(f"  Iterations per thread: {ITERATIONS_PER_THREAD}")
print(f"  Total operations: {NUM_THREADS * ITERATIONS_PER_THREAD}")
print(f"  Simulated kernel time: {SIMULATED_KERNEL_TIME_NS} ns")

# =============================================================================
# Setup Tuners
# =============================================================================
print(f"\n{YELLOW}Setting up tuners...{RESET}")

def setup_tuner(name, async_mode=False):
    tuner = flow.ContinuousAutoTuner()
    variants = []
    for i in range(4):
        v = flow.KernelVariant()
        v.name = f"{name}_variant_{i}"
        variants.append(v)
    tuner.register_kernel_variants(f"{name}_kernel", variants)
    if async_mode:
        tuner.enable_async_recording(True)
    return tuner

sync_tuner = setup_tuner("sync", async_mode=False)
async_tuner = setup_tuner("async", async_mode=True)

# =============================================================================
# Worker Functions
# =============================================================================

def sync_worker(tuner, thread_id, iterations, results):
    """Sync recording - blocks on each record"""
    thread_times = []
    kernel_name = "sync_kernel"

    for _ in range(iterations):
        start = time.perf_counter_ns()

        # Select variant
        idx = tuner.select_variant(kernel_name)

        # Simulate kernel execution (busy wait)
        kernel_start = time.perf_counter_ns()
        while time.perf_counter_ns() - kernel_start < SIMULATED_KERNEL_TIME_NS:
            pass

        # Record execution (BLOCKING)
        tuner.record_execution(kernel_name, idx, float(SIMULATED_KERNEL_TIME_NS))

        end = time.perf_counter_ns()
        thread_times.append(end - start)

    results[thread_id] = thread_times

def async_worker(tuner, thread_id, iterations, results):
    """Async recording - returns immediately after enqueueing"""
    thread_times = []
    kernel_name = "async_kernel"

    for _ in range(iterations):
        start = time.perf_counter_ns()

        # Select variant
        idx = tuner.select_variant(kernel_name)

        # Simulate kernel execution (busy wait)
        kernel_start = time.perf_counter_ns()
        while time.perf_counter_ns() - kernel_start < SIMULATED_KERNEL_TIME_NS:
            pass

        # Record execution (NON-BLOCKING)
        tuner.record_execution_async(kernel_name, idx, float(SIMULATED_KERNEL_TIME_NS))

        end = time.perf_counter_ns()
        thread_times.append(end - start)

    results[thread_id] = thread_times

# =============================================================================
# Benchmark 1: Sync Multi-Threaded
# =============================================================================
print(f"\n{BOLD}{CYAN}Benchmark 1: Sync Recording ({NUM_THREADS} threads){RESET}")

sync_results = {}
threads = []

start_time = time.perf_counter_ns()

for i in range(NUM_THREADS):
    t = threading.Thread(target=sync_worker, args=(sync_tuner, i, ITERATIONS_PER_THREAD, sync_results))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

sync_total_time = time.perf_counter_ns() - start_time

# Calculate stats
all_sync_times = []
for times in sync_results.values():
    all_sync_times.extend(times)

sync_avg = sum(all_sync_times) / len(all_sync_times)
sync_p50 = sorted(all_sync_times)[len(all_sync_times) // 2]
sync_p99 = sorted(all_sync_times)[int(len(all_sync_times) * 0.99)]
sync_max = max(all_sync_times)

print(f"  Total time: {sync_total_time / 1e6:.2f} ms")
print(f"  {GREEN}Per-op Average:{RESET} {sync_avg:.1f} ns")
print(f"  {GREEN}P50:{RESET} {sync_p50} ns")
print(f"  {GREEN}P99:{RESET} {sync_p99} ns")
print(f"  {GREEN}Max:{RESET} {sync_max} ns")
print(f"  Throughput: {len(all_sync_times) / (sync_total_time / 1e9) / 1e6:.2f} M ops/sec")

# =============================================================================
# Benchmark 2: Async Multi-Threaded
# =============================================================================
print(f"\n{BOLD}{CYAN}Benchmark 2: Async Recording ({NUM_THREADS} threads){RESET}")

async_results = {}
threads = []

start_time = time.perf_counter_ns()

for i in range(NUM_THREADS):
    t = threading.Thread(target=async_worker, args=(async_tuner, i, ITERATIONS_PER_THREAD, async_results))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

async_enqueue_time = time.perf_counter_ns() - start_time

# Flush and measure
flush_start = time.perf_counter_ns()
async_tuner.flush_async_records()
flush_time = time.perf_counter_ns() - flush_start

async_total_time = async_enqueue_time + flush_time

# Calculate stats
all_async_times = []
for times in async_results.values():
    all_async_times.extend(times)

async_avg = sum(all_async_times) / len(all_async_times)
async_p50 = sorted(all_async_times)[len(all_async_times) // 2]
async_p99 = sorted(all_async_times)[int(len(all_async_times) * 0.99)]
async_max = max(all_async_times)

print(f"  Enqueue time: {async_enqueue_time / 1e6:.2f} ms")
print(f"  Flush time: {flush_time / 1e6:.2f} ms")
print(f"  Total time: {async_total_time / 1e6:.2f} ms")
print(f"  {GREEN}Per-op Average:{RESET} {async_avg:.1f} ns")
print(f"  {GREEN}P50:{RESET} {async_p50} ns")
print(f"  {GREEN}P99:{RESET} {async_p99} ns")
print(f"  {GREEN}Max:{RESET} {async_max} ns")
print(f"  Throughput (enqueue): {len(all_async_times) / (async_enqueue_time / 1e9) / 1e6:.2f} M ops/sec")

# =============================================================================
# Async Stats
# =============================================================================
print(f"\n{BOLD}{CYAN}Async Tuner Statistics{RESET}")
stats = async_tuner.statistics()
print(f"  Records queued:    {stats.async_records_queued}")
print(f"  Records processed: {stats.async_records_processed}")
print(f"  Max queue size:    {stats.async_queue_max_size}")
print(f"  Avg latency:       {stats.async_avg_latency_ns / 1e6:.2f} ms")

# =============================================================================
# Tail Latency Comparison
# =============================================================================
print(f"\n{BOLD}{CYAN}Tail Latency Comparison{RESET}")
print(f"  {'Metric':<15} {'Sync':>12} {'Async':>12} {'Improvement':>12}")
print(f"  {'-'*51}")
print(f"  {'P50':<15} {sync_p50:>12} {async_p50:>12} {sync_p50/async_p50:>11.2f}x")
print(f"  {'P99':<15} {sync_p99:>12} {async_p99:>12} {sync_p99/async_p99:>11.2f}x")
print(f"  {'Max':<15} {sync_max:>12} {async_max:>12} {sync_max/async_max:>11.2f}x")

# =============================================================================
# Summary
# =============================================================================
print(f"\n{BOLD}{CYAN}{'='*70}")
print("  SUMMARY")
print(f"{'='*70}{RESET}")

enqueue_speedup = sync_total_time / async_enqueue_time
total_speedup = sync_total_time / async_total_time
latency_reduction = (sync_avg - async_avg) / sync_avg * 100

print(f"\n  {BOLD}Wall-Clock Time:{RESET}")
print(f"    Sync total:     {sync_total_time / 1e6:.2f} ms")
print(f"    Async enqueue:  {async_enqueue_time / 1e6:.2f} ms ({GREEN}{enqueue_speedup:.2f}x faster{RESET})")
print(f"    Async total:    {async_total_time / 1e6:.2f} ms")

print(f"\n  {BOLD}Per-Operation Latency:{RESET}")
print(f"    Sync avg:  {sync_avg:.1f} ns")
print(f"    Async avg: {async_avg:.1f} ns")
if latency_reduction > 0:
    print(f"    {GREEN}Latency reduced by {latency_reduction:.1f}%{RESET}")
else:
    print(f"    {YELLOW}Latency increased by {-latency_reduction:.1f}%{RESET}")

print(f"\n  {BOLD}Contention Analysis:{RESET}")
print(f"    Sync P99/P50 ratio:  {sync_p99/sync_p50:.2f}x (higher = more contention)")
print(f"    Async P99/P50 ratio: {async_p99/async_p50:.2f}x")

if sync_p99/sync_p50 > async_p99/async_p50:
    print(f"    {GREEN}Async has lower tail latency variance{RESET}")

# Final verdict
print(f"\n  {BOLD}Verdict:{RESET}")
if async_enqueue_time < sync_total_time * 0.9:
    print(f"  {GREEN}✓ Async provides {enqueue_speedup:.1f}x faster completion for execution threads{RESET}")
    print(f"    Background processing handles updates without blocking producers")
else:
    print(f"  {YELLOW}⚠ Async overhead outweighs benefits in this scenario{RESET}")
    print(f"    Consider using sync mode for low-contention workloads")

print()
