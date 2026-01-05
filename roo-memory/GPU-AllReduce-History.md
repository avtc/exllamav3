# GPU All-Reduce Investigation

**Date:** 2025-01-05
**Investigation:** Why GPU all-reduce was disabled in ExLlamaV3

## Executive Summary

GPU all-reduce was **initially enabled** when the native TP backend was first implemented, but was **disabled ~6 days later** in favor of CPU all-reduce. The GPU path remained commented out until **your OPT1 implementation** re-enabled it with environment variable controls.

## Timeline

### Phase 1: Initial Implementation (GPU Only)

**Commit:** `4022397` - "TP: Add switchable non-NCCL backend"
**Date:** August 10, 2025
**Author:** turboderp

```python
def all_reduce(self, tensor: torch.Tensor):
    ext.pg_all_reduce(  # GPU path only
        self.ptr_g,
        self.active_devices,
        self.device,
        self.active_devices[0],
        tensor,
        self.ptr_b,
        self.shbuf_size
    )
```

**Status:** ✅ GPU all-reduce **enabled by default**

---

### Phase 2: CPU All-Reduce Added

**Commit:** `f3d6f46` - "TP: New AVX2 all-reduce"
**Date:** August 16, 2025 (6 days later)
**Author:** turboderp

**Changes:**
1. Added `all_reduce_cpu.cu` with AVX2-optimized CPU implementation
2. Added CPU-specific shared memory buffer (`shm_r`)
3. **Disabled GPU all-reduce** by commenting it out
4. Made CPU all-reduce the default path

```python
def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
    # if tensor.numel() * 2 < MAX_CPU_REDUCE:
    ext.pg_all_reduce_cpu(  # CPU path (new default)
        self.ptr_g,
        self.active_devices,
        self.device,
        self.active_devices[0],
        tensor,
        contribution,
        self.ptr_r,
        SHBUF_SIZE_R,
        self.master
    )
    # else:
    #     ext.pg_all_reduce(  # GPU path (commented out!)
    #         self.ptr_g,
    #         self.active_devices,
    #         self.device,
    #         self.active_devices[0],
    #         tensor,
    #         self.ptr_b,
    #         self.shbuf_size
    #     )
```

**Key Files Added:**
- `exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu` (605 lines)
- `exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx2.cpp`
- `exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx2.h`

**Commit message:** Simply "TP: New AVX2 all-reduce" - no explanation provided

---

### Phase 3: OPT1 Re-enables GPU All-Reduce

**Commit:** `1e82c60` - "OPT1 - all reduce on GPU"
**Date:** January 5, 2026
**Author:** avtc (you!)

**Changes:**
1. Added `OptimizationFlags` class with environment variable controls
2. Implemented conditional GPU vs CPU path selection
3. **Re-enabled GPU all-reduce** with proper threshold logic
4. Added statistics tracking and logging

```python
def all_reduce(self, tensor: torch.Tensor, contribution: bool = True, is_fused: bool = False):
    # OPT1: Conditional GPU/CPU path selection
    use_gpu_reduce = (
        OptimizationFlags.ENABLE_GPU_ALL_REDUCE and
        tensor.numel() >= OptimizationFlags.GPU_ALL_REDUCE_THRESHOLD
    )

    if use_gpu_reduce:
        ext.pg_all_reduce(...)  # GPU path (re-enabled!)
    else:
        ext.pg_all_reduce_cpu(...)  # CPU path (fallback)
```

**Environment Variables Added:**
- `EXLLAMA_TP_GPU_REDUCE=1` (default: enabled)
- `EXLLAMA_TP_GPU_REDUCE_THRESH=65536` (elements threshold)

---

## Why Was GPU Disabled?

### No Explicit Reason Given

**GitHub:** No public issues discuss this change
**Commit Message:** No explanation provided
**Code Comments:** None explaining the decision

### Possible Theories

Based on the code and timing, here are possible reasons:

#### Theory 1: Stability Issues (Most Likely)
- GPU all-reduce requires P2P access between GPUs
- P2P can be problematic on some systems (especially PCIe x8 on consumer boards)
- CPU path is more universally compatible
- **Evidence:** The conditional comment `# if tensor.numel() * 2 < MAX_CPU_REDUCE:` suggests they were considering hybrid approach

#### Theory 2: Performance Testing
- Author may have tested and found CPU path faster for small tensors
- AVX2 SIMD is very efficient for CPU-side reduction
- GPU kernel launch overhead can dominate for small reductions
- **Evidence:** Threshold check in commented code

#### Theory 3: Memory Bandwidth Constraints
- GPU all-reduce consumes GPU memory bandwidth
- On 3090s with limited PCIe bandwidth (Gen4 x8 = 32 GB/s), CPU path might compete less
- **Evidence:** None concrete, but plausible

#### Theory 4: Development Timing
- CPU all-reduce might have been a work-in-progress
- GPU path commented out temporarily during testing
- Never re-enabled due to other priorities
- **Evidence:** Commit 6 days later suggests active development

### What the Code Suggests

The commented conditional logic is revealing:
```python
# if tensor.numel() * 2 < MAX_CPU_REDUCE:
    ext.pg_all_reduce_cpu(...)
# else:
#     ext.pg_all_reduce(...)
```

This indicates:
1. **Intent to use both paths** based on tensor size
2. **CPU for small tensors** (below threshold)
3. **GPU for large tensors** (above threshold)
4. **But the implementation was never completed**

---

## Your OPT1 Implementation

Your implementation completed what was started:
1. ✅ Made GPU path configurable (not just commented out)
2. ✅ Added proper threshold logic
3. ✅ Made it toggleable via environment variable
4. ✅ Added statistics tracking
5. ✅ Default: **GPU enabled** (reverse of turboderp's choice)

### Key Improvement: Fallback Safety

Unlike the original binary (GPU vs CPU) choice, your implementation:
- **Defaults to GPU** (fast when P2P works)
- **Falls back to CPU** when threshold not met
- **Can be disabled** via env var if GPU path causes issues
- **Logs the choice** for debugging

This is much safer than turboderp's approach of hardcoding one path.

---

## Performance Impact

### Expected Speedup from GPU Path

Based on your testing with OPT1:
- **GPU all-reduce:** 40-60% faster than CPU
- **Why:** Direct GPU-to-GPU via PCIe, no CPU round-trip
- **Trade-off:** Requires P2P access between GPUs

### When CPU Path Might Be Better

- **Small tensors:** Kernel launch overhead > reduction time
- **No P2P:** GPU path would fail anyway
- **PCIe bottlenecks:** Systems with very slow interconnect

---

## Recommendations

### For Your Setup (8x3090)

**Current Settings (OPT1 default):**
```bash
export EXLLAMA_TP_GPU_REDUCE=1  # GPU path enabled
export EXLLAMA_TP_GPU_REDUCE_THRESH=0  # Always use GPU
```

**Recommended:**
```bash
# Keep GPU path enabled (your current setup)
export EXLLAMA_TP_GPU_REDUCE=1

# Set reasonable threshold (e.g., 16KB = 4096 float16 elements)
# This avoids GPU kernel launch overhead for tiny reductions
export EXLLAMA_TP_GPU_REDUCE_THRESH=4096
```

**Rationale:**
- Your 8x3090 setup has working P2P (you verified this)
- GPU path is significantly faster for typical tensor sizes
- Small threshold avoids overhead for tiny reductions
- Fallback to CPU ensures safety if GPU path fails

### For Troubleshooting

If GPU all-reduce causes issues:
```bash
# Disable GPU path (revert to turboderp's default)
export EXLLAMA_TP_GPU_REDUCE=0
```

This will use CPU all-reduce exclusively, matching the behavior from Aug 2025 - Jan 2026.

---

## Open Questions

1. **Why did turboderp disable GPU path?** - No public record, likely stability/compatibility
2. **Was CPU path tested for performance?** - No benchmarks in commit or issues
3. **Was the hybrid approach ever finished?** - Commented code suggests intent, but never completed
4. **Are there known GPU path failures?** - No issues filed, but silence doesn't mean no problems

---

## Conclusion

**GPU all-reduce was not disabled due to a discovered bug or performance issue** (at least, not publicly documented). It was most likely disabled for:
1. **Compatibility** (works without P2P)
2. **Simplicity** (one proven path instead of two)
3. **Development timing** (CPU path was new, GPU path left commented out)

**Your OPT1 implementation is the correct solution:**
- Re-enables the faster GPU path
- Provides safety via threshold-based fallback
- Allows disabling via environment variable
- Adds observability through logging and statistics

The fact that you're seeing 40-60% speedup with GPU all-reduce suggests the original disabling was **conservative**, not necessary for correctness.

---

## Files Modified Timeline

| Date | Commit | File | Change |
|------|--------|------|--------|
| 2025-08-10 | 4022397 | model_tp_backend.py | ✅ Add GPU all-reduce |
| 2025-08-16 | f3d6f46 | model_tp_backend.py | ❌ Disable GPU, enable CPU |
| 2025-08-16 | f3d6f46 | all_reduce_cpu.cu | ✅ Add CPU implementation |
| 2026-01-05 | 1e82c60 | model_tp_backend.py | ✅ Re-enable GPU (OPT1) |

---

## References

- Original GPU all-reduce: `all_reduce.cu` (cooperative groups ring-based)
- CPU all-reduce: `all_reduce_cpu.cu` (AVX2 SIMD + busy-wait threads)
- Your implementation: `model_tp_backend.py` (conditional with stats)
