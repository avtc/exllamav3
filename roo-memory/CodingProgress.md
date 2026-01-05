# Coding Progress: ExLlamaV3 Multi-GPU Performance Optimization

**User Story:** Improve 8x3090 MoE inference from 25-35 t/s to match vLLM's 80-140 t/s

**Current Baseline:** 30-35 tokens/second
**Target:** 70-100 tokens/second (Phase 1+2)

---

## Phase 1: Quick Wins (GPU All-Reduce + Buffer Optimization)

### OPT1: Enable GPU All-Reduce ⚡

**Expected Speedup:** 40-60%
**Effort:** 30 minutes
**Status:** ✅ **IMPLEMENTED** - Ready for testing via TabbyAPI

#### Tasks:

- [x] **1.1** Research vLLM implementation and confirm GPU all-reduce approach
- [x] **1.2** Document findings in Research-vllm.md
- [x] **1.3** Create implementation plan in Plan.md
- [x] **1.4** Add OptimizationFlags class to model_tp_backend.py
- [x] **1.5** Implement environment variable support
- [x] **1.6** Modify all_reduce() method to use GPU path conditionally
- [x] **1.7** Add logging and statistics tracking
- [ ] **1.8** Test with GPU all-reduce enabled (via TabbyAPI)
- [ ] **1.9** Benchmark and measure speedup
- [ ] **1.10** Document results

---

## Phase 2: Medium Effort Optimizations

### OPT2: Fuse Attention + MoE All-Reduce ⚡

**Expected Speedup:** 30-50%
**Effort:** 2-3 hours
**Status:** ✅ **IMPLEMENTED** - Ready for testing via TabbyAPI

#### Tasks:

- [x] **2.1** Add ENABLE_FUSED_ALL_REDUCE flag (already in OptimizationFlags)
- [x] **2.2** Modify TransformerBlock to skip all-reduce in sub-modules
- [x] **2.3** Implement single all-reduce at end of block
- [x] **2.4** Update Attention module to respect _skip_tp_reduce flag
- [x] **2.5** Update MLP module to respect _skip_tp_reduce flag
- [x] **2.6** Update GatedMLP module to respect _skip_tp_reduce flag
- [x] **2.7** Update BlockSparseMLP module to respect _skip_tp_reduce flag
- [x] **2.8** Add fused all-reduce tracking to statistics
- [x] **2.9** Add is_fused parameter to all_reduce() method
- [ ] **2.10** Test with fusion enabled (via TabbyAPI)
- [ ] **2.11** Benchmark and measure speedup
- [ ] **2.12** Document results

---

### OPT3: Increase CPU All-Reduce Buffer (When GPU Path Can't Be Used)

**Expected Speedup:** 5-10% (when CPU path is used)
**Effort:** 1 hour
**Status:** ✅ **IMPLEMENTED** - Ready for testing via TabbyAPI

#### Tasks:

- [x] **3.1** Add CPU_REDUCE_BUFFER_MULTIPLIER flag (already in OptimizationFlags)
- [x] **3.2** Implement dynamic buffer sizing function
- [x] **3.3** Update TPBackendNative to use dynamic buffer size
- [x] **3.4** Update all_reduce_cpu to use instance variable
- [x] **3.5** Update run_cpu_reduce_jobs to use instance variable
- [x] **3.6** Add logging to show buffer size
- [ ] **3.7** Test with larger buffer
- [ ] **3.8** Document results

---

## Phase 3: Additional Optimizations (Future Work)

### OPT4: Eliminate Redundant Barrier

**Expected Speedup:** 5-10%
**Effort:** 2 hours
**Status:** ⏳ TODO

#### Tasks:

- [ ] **4.1** Analyze barrier synchronization points
- [ ] **4.2** Test removing explicit barrier
- [ ] **4.3** Verify correctness without barrier
- [ ] **4.4** Document results

---

### OPT5: Batched Sampling

**Expected Speedup:** 20-30%
**Effort:** 6-8 hours
**Status:** ✅ **IMPLEMENTED** - Ready for testing via TabbyAPI

#### Tasks:

- [x] **5.1** Add ENABLE_BATCHED_SAMPLING flag to OptimizationFlags
- [x] **5.2** Implement receive_logits_batched() method in job.py
- [x] **5.3** Modify generator loop to use batched sampling conditionally
- [x] **5.4** Add fallback to serial sampling when disabled
- [x] **5.5** Add environment variable support (EXLLAMA_BATCHED_SAMPLING)
- [ ] **5.6** Test with batched sampling enabled (via TabbyAPI)
- [ ] **5.7** Benchmark and measure speedup
- [ ] **5.8** Document results

---

## Testing & Benchmarking

### Benchmark Tasks:

- [ ] **B.1** Create benchmark script
- [ ] **B.2** Measure baseline performance (CPU all-reduce)
- [ ] **B.3** Measure GPU all-reduce performance
- [ ] **B.4** Measure fused all-reduce performance
- [ ] **B.5** Test with different models (GLM, DeepSeek, etc.)
- [ ] **B.6** Document all results

---

## Environment Variables

### Implemented Controls:

```bash
# Enable/disable GPU all-reduce (default: 1 = enabled)
export EXLLAMA_TP_GPU_REDUCE=1

# GPU all-reduce threshold in elements (default: 65536)
export EXLLAMA_TP_GPU_REDUCE_THRESH=0  # 0 = always use GPU

# Enable/disable fused all-reduce (default: 1 = enabled)
export EXLLAMA_TP_FUSED_REDUCE=1

# CPU buffer multiplier (default: 4)
export EXLLAMA_TP_CPU_BUFFER_MULT=4

# Enable/disable batched sampling (default: 1 = enabled)
export EXLLAMA_BATCHED_SAMPLING=1
```

---

## Results Log

### Baseline (Before Optimizations)
- **Date:** 2025-01-05
- **Setup:** 8x RTX 3090, PCIe Gen4 x8, P2P enabled, Single DDR5 DIMM
- **Model:** GLM-4.5-Air 4-bit
- **Performance:** 30-35 t/s

### OPT1 Results (GPU All-Reduce)
- **Date:** TBD
- **Configuration:** EXLLAMA_TP_GPU_REDUCE=1, THRESH=0
- **Performance:** TBD t/s
- **Speedup:** TBD%

### OPT2 Results (Fused All-Reduce)
- **Date:** TBD
- **Performance:** TBD t/s
- **Speedup:** TBD%

### OPT5 Results (Batched Sampling)
- **Date:** TBD
- **Configuration:** EXLLAMA_BATCHED_SAMPLING=1
- **Performance:** TBD t/s
- **Speedup:** TBD%

### Final Results (Phase 1+2+3)
- **Date:** TBD
- **Performance:** TBD t/s
- **Total Speedup:** TBD%

---

## Notes

- [x] All optimizations are toggleable via environment variables
- [ ] Each optimization tested independently
- [ ] Results compared against vLLM baseline (79-136 t/s)
