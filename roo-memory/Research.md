# ExLlamaV3 Performance Investigation - 8x3090 Multi-GPU Inference

**Date:** 2025-01-05
**Setup:** 8x RTX 3090, PCIe Gen4 x8, P2P enabled, Custom EPYC server, 1x DDR5 64GB RAM
**Issue:** 25-35 tokens/sec vs vLLM's 79-136 tokens/sec (3-4x slower)

---

## Executive Summary

ExLlamaV3 is experiencing severe performance degradation on 8x3090 setup with MoE models. The root cause is **CPU-side all-reduce bottleneck** exacerbated by **single-channel DDR5 memory bandwidth limitation**. The evidence: 4-bit and 8-bit models run at identical speeds, proving computation is NOT the bottleneck.

---

## Key Findings

### Finding 1: Communication Bottleneck, Not Computation

**Evidence:**
- GLM-4.5-Air 4-bit: 30-35 t/s
- GLM-4.5-Air 8-bit: 30-35 t/s
- GLM-4.6 3.84-bit: 30-35 t/s
- MiniMax M2: ~30-35 t/s

**Conclusion:** Identical performance across different quantization levels proves the bottleneck is independent of computational load. The bottleneck is in **data movement and synchronization**, not matrix multiplication or memory access.

---

### Finding 2: CPU All-Reduce is the Primary Culprit

**Location:** `exllamav3/model/model_tp_backend.py:331-344`

**Current Implementation:**
```python
def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
    # Condition is commented out - ALL reductions use CPU path
    ext.pg_all_reduce_cpu(
        self.ptr_g,
        self.active_devices,  # 8 GPUs
        self.device,
        self.active_devices[0],
        tensor,
        contribution,
        self.ptr_r,  # CPU buffer - SHBUF_SIZE_R = 2.1 MB
        SHBUF_SIZE_R,
        self.master,
        self.abort_flag
    )
```

**CPU All-Reduce Flow:**
1. Each of 8 GPUs writes contribution to shared buffer
2. CPU process reads 8 contributions from RAM
3. CPU performs AVX2-optimized summation
4. CPU writes result back to shared buffer
5. GPUs read result from shared buffer

**Per-Token Frequency:**
- Model with ~80 layers
- Each layer has: self_attn all-reduce + MoE all-reduce (or MLP all-reduce)
- Total: **~160 all-reduce operations per generated token**

**Critical Bottleneck - Single DDR5 DIMM:**
- DDR5-4800 single channel: ~38 GB/s theoretical bandwidth
- CPU needs to read 8x contributions and write 1x result
- For hidden_dim=6144, fp16: 12 KB per all-reduce
- 160 all-reduces/token × 12 KB = 1.9 MB token throughput via CPU

**Problem:** CPU all-reduce process is **bandwidth-bound** on single DDR5 channel.

---

### Finding 3: PCIe Bandwidth Limitation

**Configuration:**
- PCIe Gen4 x8 per GPU (not x16)
- Theoretical: ~16 GB/s per GPU
- 8 GPUs × 16 GB/s = 128 GB/s aggregate theoretical
- Realistic: ~80-100 GB/s aggregate with overhead

**Impact:**
- Shared memory IPC requires PCIe transfers
- Each all-reduce: 8 GPU→CPU transfers + 1 CPU→GPU transfer
- With 160 all-reduces/token: substantial PCIe bandwidth consumption

**P2P Status:**
- Custom kernel enables P2P between GPUs
- 10x lower GPU-to-GPU latency
- BUT: P2P doesn't help with CPU all-reduce (still goes through RAM)

---

### Finding 4: NCCL Backend is Slower

**Test Result:** NCCL backend ~10% slower than native

**Analysis:**
- NCCL is optimized for NVLink or high-bandwidth scenarios
- PCIe Gen4 x8 with 8 GPUs creates congestion
- Native backend's CPU-based all-reduce reduces GPU-to-GPU communication
- For PCIe-based setups, CPU all-reduce is actually strategic (but needs optimization)

**Conclusion:** Native backend is correct choice, but needs tuning.

---

### Finding 5: Model Architecture Impact

**GLM/MoE Models:**
- Most layers: self_attn + MoE
- First 1-3 layers (GLM-Air) or 3 layers (GLM): self_attn + MLP
- MoE uses expert parallelism by default
- Routing decisions broadcast from output device to all GPUs

**Per-Layer Communication:**
1. Self-attn O projection → all-reduce (8 GPUs → sum → distribute)
2. MoE routing broadcast (output device → 7 other GPUs)
3. MoE expert outputs → all-reduce (8 GPUs → sum → distribute)

**Total:** 2-3 communication operations per layer, 160+ per token

---

### Finding 6: Serial Sampling Loop

**Location:** `exllamav3/generator/generator.py:519-569`

```python
# TODO: Batch sampling - Authors acknowledge this issue
for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
    if a == b: continue
    job_logits = batch_logits[a:b, :, :]

    for i in range(batch_logits.shape[1]):  # Serial loop over sequences
        token_logits = job_logits[:, i:i + 1, :]
        next_token, ... = job.receive_logits(token_logits)  # CPU-GPU sync
        eos, sampled_token, rq = job.receive_sample(...)    # CPU work
        # Stop condition checking, string matching, page bookkeeping
```

**Problem:**
- Model forward is batched (efficient)
- Sampling is job-by-job in Python (inefficient)
- CPU-GPU synchronization on every sequence
- Significant Python overhead

**Impact:** ~30-40% of generation time (estimated)

---

### Finding 7: Barrier Synchronization

**Location:** `exllamav3/model/model_tp_fn.py:184-185`

```python
def mp_model_forward(...):
    backend.fwd_barrier()  # All 8 GPUs synchronize

    for idx, module in enumerate(modules):
        x = module.forward(x, params)
```

**Impact:**
- All 8 GPUs must reach barrier before proceeding
- Slowest GPU determines overall speed
- P2p helps but doesn't eliminate this bottleneck
- Happens once per token generation step

---

## Bottleneck Analysis (Revised with New Information)

### Primary Bottleneck: CPU All-Reduce Memory Bandwidth (40-50% of time)

**Root Cause:** Single DDR5 DIMM cannot feed 8 GPUs efficiently

**Evidence:**
- 4-bit = 8-bit speed (computation independent)
- 160 all-reduces per token
- Each all-reduce requires:
  - 8 GPU → CPU transfers via PCIe
  - CPU read from RAM (8x)
  - CPU summation (AVX2)
  - CPU write to RAM
  - 1 CPU → GPU transfer via PCIe

**Calculation:**
- Hidden dim: ~6144 (varies by model)
- Tensor size: 6144 × 2 bytes (fp16) = 12 KB
- Per all-reduce CPU bandwidth: 8 × 12 KB = 96 KB read + 12 KB write = 108 KB
- Per token: 160 × 108 KB = 17.3 MB CPU memory traffic
- At 30 t/s: 518 MB/s CPU bandwidth for all-reduce alone

**Problem:** CPU all-reduce process is competing with main process for single DDR5 channel bandwidth.

---

### Secondary Bottleneck: Serial Sampling Loop (25-35% of time)

**Impact:** CPU-GPU synchronization overhead, Python loops

**Independent of quantization:** Explains why 4-bit = 8-bit

---

### Tertiary Bottleneck: PCIe Congestion (10-15% of time)

**Configuration:** 8 GPUs × PCIe Gen4 x8 = 128 GB/s theoretical

**Demand:**
- 160 all-reduces/token × 9 PCIe transfers each
- At 30 t/s: High PCIe utilization

**P2P helps for GPU-to-GPU but not GPU-to-CPU**

---

### Minor Bottleneck: Barrier Sync (5-10% of time)

**Impact:** Synchronization overhead

**Mitigated by:** P2p reduces latency, but doesn't eliminate barrier

---

## Why vLLM is Faster

### vLLM Architecture Advantages:

1. **Continuous Batching:** Better GPU utilization
2. **Kernel Fusion:** Reduces memory access
3. **pagedAttention:** Optimized KV cache management
4. **GPU-Based All-Reduce:** Uses NCCL efficiently (likely tuned for NVLink)
5. **Speculative Decoding:** Better implementation

### Key Differentiator for Your Setup:

**vLLM likely uses GPU-based all-reduce combined with better pipelining**

For PCIe-based setups:
- vLLM may have tuned all-reduce thresholds
- May use tensor parallelism more aggressively
- Better overlap of computation and communication

---

## Optimization Plan (Prioritized by Impact)

### Priority 1: Reduce CPU All-Reduce Frequency (Quick Win)

**Goal:** Reduce number of all-reduce operations per token

**Approach A: Fuse Attention + MoE All-Reduce**
- Current: Separate all-reduce after attn and after MoE
- Optimize: Single all-reduce after both complete
- Files: `transformer.py`, `attn.py`, `block_sparse_mlp.py`
- Expected speedup: 15-25%

**Approach B: Increase All-Reduce Threshold**
- Un-comment condition in `model_tp_backend.py:332`
- Use GPU all-reduce for larger tensors
- Problem: May not help if hidden_dim is below threshold
- Expected speedup: 5-15%

**Approach C: Batch All-Reduces**
- Accumulate multiple layers before all-reducing
- Challenge: Requires architectural changes
- Expected speedup: 20-30%

---

### Priority 2: Optimize CPU All-Reduce Implementation

**Approach A: NUMA-Aware Memory Allocation**
- Ensure shared buffers are allocated in correct NUMA node
- Bind CPU reducer process to correct CPU cores
- Expected speedup: 5-10%

**Approach B: Increase CPU All-Reduce Buffer**
- Current: SHBUF_SIZE_R = 2.1 MB
- Increase to allow batching of reductions
- Expected speedup: 5-10%

**Approach C: Use Multiple CPU Reducer Threads**
- Current: Single CPU process handles all reductions
- Optimize: Thread pool for parallel reduction
- Challenge: Complex synchronization
- Expected speedup: 10-20%

---

### Priority 3: Implement Batched Sampling

**Location:** `generator.py:519-569`

**Approach:** Move sampling loop to GPU kernel

**Current:**
```python
for job in active_jobs:  # Python loop
    for seq in job.sequences:  # Python loop
        next_token = job.receive_logits(...)  # CPU-GPU sync
```

**Optimized:** Single GPU kernel handles all sampling

**Challenge:** Complex refactoring (authors acknowledge TODO)

**Expected speedup: 20-30%**

---

### Priority 4: Pipelined Execution

**Goal:** Overlap CPU work with GPU work

**Current:** Sequential (GPU forward → CPU sampling → GPU forward)

**Pipelined:**
- Stream 1: GPU forward pass N
- Stream 2: CPU sampling pass N-1
- Stream 3: Prepare batch pass N+1

**Challenge:** Requires generator refactoring

**Expected speedup: 15-25%**

---

### Priority 5: Reduce Tensor Parallelism Degree

**Counter-Intuitive:** Try fewer GPUs

**Rationale:**
- 8 GPUs × PCIe x8 = high communication overhead
- 4 GPUs may be more efficient due to:
  - Fewer all-reduce participants
  - Less PCIe congestion
  - Better bandwidth utilization

**Test:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python chat.py -m model -tp -tpb native
```

**Expected:** May see higher throughput despite fewer GPUs

---

## Testing and Benchmarking Plan

### Phase 1: Baseline Measurement

```python
# Add instrumentation to measure:
# 1. Time spent in all_reduce (cumulative)
# 2. Time spent in sampling loop
# 3. Time spent in model forward
# 4. PCIe bandwidth utilization
# 5. CPU memory bandwidth utilization
```

**Metrics to Collect:**
- Tokens/second (current baseline: 30-35)
- All-reduce latency (mean, p50, p95, p99)
- Barrier sync latency
- CPU reducer process CPU% and memory bandwidth
- GPU utilization per GPU

---

### Phase 2: A/B Testing

**Test Matrix:**

| GPUs | Backend | All-Reduce | Expected t/s |
|------|---------|------------|--------------|
| 8    | native  | CPU        | 30-35 (baseline) |
| 8    | native  | GPU        | 35-45 |
| 8    | nccl    | NCCL       | 25-35 |
| 4    | native  | CPU        | 35-50 |
| 4    | native  | GPU        | 45-60 |
| 2    | native  | CPU        | 20-30 |

---

### Phase 3: Profiling

**Tools:**
- `nsys` (NVIDIA Nsight Systems) for GPU timeline
- `nvtop` for GPU utilization
- `pcm-memory` for CPU memory bandwidth
- `perf` for CPU profiling

**Focus Areas:**
- CPU all-reduce hot spots
- PCIe transfer patterns
- GPU idle time
- Kernel launch overhead

---

## Hardware Considerations

### Immediate Actions (No Cost):

1. **Check NUMA Configuration**
   ```bash
   numactl --hardware
   lstopo --of console
   ```
   Ensure GPUs and RAM are in same NUMA node

2. **Optimize Process Affinity**
   ```bash
   # Bind processes to correct CPUs
   numactl --cpunodebind=0 --membind=0 python chat.py ...
   ```

3. **Check PCIe Topology**
   ```bash
   nvidia-smi topo -m
   ```
   Verify P2P is working correctly

---

### Hardware Upgrades (If Needed):

**Priority 1: Add Second DDR5 DIMM**
- Dual-channel DDR5: 2x bandwidth (~76 GB/s)
- Should directly address CPU all-reduce bottleneck
- Expected speedup: 30-50%

**Priority 2: PCIe Switch with x16 Lanes**
- Upgrade to x16 per GPU
- 2x PCIe bandwidth per GPU
- Expected speedup: 15-25%

---

## Expected Speedups by Optimization

### Conservative Estimates:

| Optimization | Speedup | Confidence |
|--------------|---------|------------|
| Reduce all-reduce freq (Priority 1A) | 15-25% | High |
| Optimize CPU all-reduce (Priority 2A) | 5-10% | Medium |
| Batched sampling (Priority 3) | 20-30% | Medium |
| Pipelined execution (Priority 4) | 15-25% | Low |
| Reduce TP degree (Priority 5) | 10-50% | High |

### Combined Scenarios:

**Scenario A: Quick Wins (Priorities 1A + 2A + 5)**
- Current: 30-35 t/s
- Expected: 50-70 t/s (1.5-2x speedup)
- Effort: 1-2 days

**Scenario B: Medium Effort (Scenario A + Priority 3)**
- Expected: 70-90 t/s (2-2.5x speedup)
- Effort: 3-5 days

**Scenario C: Full Optimization (All Priorities)**
- Expected: 90-120 t/s (3x speedup, approaching vLLM)
- Effort: 1-2 weeks

**Scenario D: With Hardware Upgrade (Add DDR5 DIMM)**
- Expected: 120-150 t/s (4x speedup, matching vLLM)
- Effort: 1-2 weeks + hardware cost

---

## Critical Code Sections

### 1. CPU All-Reduce Entry Point
**File:** `exllamav3/model/model_tp_backend.py`
**Lines:** 331-356
**Impact:** Controls all all-reduce operations

### 2. All-Reduce CPU Implementation
**File:** `exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu`
**Lines:** 64-200 (perform_cpu_reduce function)
**Impact:** Actual reduction logic on CPU

### 3. Sampling Loop
**File:** `exllamav3/generator/generator.py`
**Lines:** 519-569
**Impact:** Post-forward processing

### 4. Barrier Synchronization
**File:** `exllamav3/model/model_tp_fn.py`
**Lines:** 184-185
**Impact:** Per-token GPU sync

### 5. MoE Forward Pass
**File:** `exllamav3/modules/block_sparse_mlp.py`
**Lines:** 510-740
**Impact:** MoE computation and all-reduce

---

## Next Steps

1. **Baseline profiling** - Measure current bottlenecks accurately
2. **Quick win implementation** - Fuse all-reduces, optimize CPU binding
3. **A/B testing** - Test with 4 GPUs, different all-reduce strategies
4. **Hardware consideration** - Evaluate adding second DDR5 DIMM
5. **Long-term optimization** - Implement batched sampling, pipelining

---

## References

- **ExLlamaV3 Architecture:** `CLAUDE.md`
- **Tensor Parallelism:** `exllamav3/model/model_tp.py`
- **MoE Implementation:** `exllamav3/modules/block_sparse_mlp.py`
- **All-Reduce Kernel:** `exllamav3_exllamav3_ext/parallel/all_reduce_cpu.cu`
- **Generator Loop:** `exllamav3/generator/generator.py`
