"""
High-precision timing utilities for profiling inference bottlenecks.

Enabled via EXLLAMA_TIMING=1 environment variable.
Only accumulates timing during decoding stage (not prefill).
"""

import os
import time
from contextlib import contextmanager
from collections import defaultdict

# Global timing state
_enabled = os.environ.get("EXLLAMA_TIMING", "0") == "1"
_stats = defaultdict(lambda: {"total_ns": 0, "count": 0})
_in_prefill = True  # Start in prefill mode, ignore timing until first decode

# Print debug message when enabled
if _enabled:
    print("[TIMING] Timing enabled - summary will be printed after each generation")

def is_enabled():
    """Check if timing is enabled."""
    return _enabled

def set_prefill_mode(is_prefill: bool):
    """Set prefill mode. Timing is only accumulated during decode mode."""
    global _in_prefill
    if _in_prefill != is_prefill and _enabled:
        print(f"[TIMING] Prefill mode: {is_prefill}")
    _in_prefill = is_prefill

def record_timing(category: str, operation: str, duration_ns: int):
    """
    Record timing for a specific operation (only during decode stage).

    Args:
        category: High-level category (e.g., "attn", "mlp", "all_reduce")
        operation: Specific operation name (e.g., "flash_attn", "p2p_v2_kernel")
        duration_ns: Duration in nanoseconds
    """
    # Skip timing during prefill
    if not _enabled or _in_prefill:
        return

    key = f"{category}.{operation}"
    _stats[key]["total_ns"] += duration_ns
    _stats[key]["count"] += 1

def print_summary():
    """Print summary statistics sorted by total time (descending)."""
    if not _enabled:
        return

    print("\n" + "="*100)
    print("TIMING SUMMARY (Decoding Stage Only - Sorted by Total Time)")
    print("="*100)

    if not _stats:
        print("No timing data collected (still in prefill mode or no decode operations)")
        print("="*100 + "\n")
        return

    # Sort by total time (descending)
    sorted_stats = sorted(_stats.items(), key=lambda x: x[1]["total_ns"], reverse=True)

    total_ns = sum(s["total_ns"] for s in _stats.values())
    total_count = sum(s["count"] for s in _stats.values())

    print(f"{'Operation':<45} {'Count':>10} {'Total (ms)':>12} {'Avg (μs)':>12} {'Ops/sec':>12} {'%':>8}")
    print("-"*100)

    for key, stats in sorted_stats:
        total_ms = stats["total_ns"] / 1e6
        avg_us = (stats["total_ns"] / stats["count"]) / 1e3 if stats["count"] > 0 else 0
        ops_per_sec = 1e6 / avg_us if avg_us > 0 else 0
        pct = 100 * stats["total_ns"] / total_ns if total_ns > 0 else 0
        print(f"{key:<45} {stats['count']:>10} {total_ms:>12.3f} {avg_us:>12.3f} {ops_per_sec:>12.1f} {pct:>7.1f}%")

    print("-"*100)
    avg_total_us = (total_ns / total_count) / 1e3 if total_count > 0 else 0
    total_ops_per_sec = 1e6 / avg_total_us if avg_total_us > 0 else 0
    print(f"{'TOTAL':<45} {total_count:>10} {total_ns/1e6:>12.3f} {avg_total_us:>12.3f} {total_ops_per_sec:>12.1f}")
    print("="*100 + "\n")

def reset():
    """Reset all timing statistics."""
    global _stats, _in_prefill
    _stats = defaultdict(lambda: {"total_ns": 0, "count": 0})
    _in_prefill = True

@contextmanager
def timed_operation(category: str, operation: str, params: dict = None):
    """
    Context manager for timing an operation with nanosecond precision.

    Usage:
        with timed_operation("attn", "flash_attn", params):
            result = self.decode_flash_attn_nc(x, bsz, seqln, params)

    Timing is only accumulated during decode stage (not prefill).
    """
    # Check if we should skip timing (disabled or during prefill)
    # Use params if available, otherwise fall back to global state
    if params is not None:
        is_prefill = params.get("prefill") == True
    else:
        is_prefill = _in_prefill

    if not _enabled or is_prefill:
        yield
        return

    # High-precision timing (nanosecond resolution)
    start = time.time_ns()

    try:
        yield
    finally:
        # Record total time (nanoseconds)
        end = time.time_ns()
        duration_ns = end - start
        record_timing(category, operation, duration_ns)
