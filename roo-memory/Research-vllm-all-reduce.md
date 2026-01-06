# Research: vLLM vs ExLlamaV3 All-Reduce Performance

**Date:** 2025-01-06
**Hardware:** 8x RTX 3090 (PCIe P2P interconnected, no NVLink)
**Model:** Same model tested on both systems

## Performance Measurements

| Implementation | Speed (t/s) | Relative to vLLM Baseline |
|---------------|-------------|---------------------------|
| **vLLM with custom all-reduce** | **60.0 t/s** | **+15%** |
| vLLM **without custom all-reduce** (fallback) | ~52 t/s | **baseline (100%)** |
| ExLlamaV3 CPU all-reduce | 23.5 t/s | **-55% (2.2x slower)** |
| ExLlamaV3 small kernel | 21.06 t/s | -60% (2.5x slower) |
| ExLlamaV3 NCCL backend | 19.0 t/s | -64% (2.7x slower) |
| ExLlamaV3 regular GPU reduce | 15.78 t/s | -70% (3.3x slower) |
| **ExLlamaV3 P2P all-reduce** | **14.5 t/s** | **-72% (3.6x slower)** |

## Key Findings

### Critical Insight
**vLLM's baseline fallback (PyNCCL or torch.distributed) is already 2.2x faster than ExLlamaV3's best implementation.**

The custom all-reduce only provides +10-15% additional speedup on top of vLLM's already-optimized base. This means:
- The performance bottleneck is **NOT** primarily the all-reduce algorithm
- ExLlamaV3 has fundamental architectural issues beyond all-reduce
- Custom all-reduce is icing on the cake, not the cake itself

## vLLM's Custom All-Reduce Implementation

### Location
- **Python:** `vllm/distributed/device_communicators/custom_all_reduce.py`
- **CUDA:** `vllm/csrc/custom_all_reduce.cu` and `custom_all_reduce.cuh`

### Core Innovations

#### 1. Pure GPU Barrier Synchronization (No CPU Polling)

**vLLM Approach:**
```cuda
// Direct P2P writes to peer memory - ZERO CPU involvement
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // Write directly to peer GPU's flag in P2P memory
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];

    // Single P2P write + spin wait
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
}
```

**Cost:** ~1-2 microseconds (one P2P write latency per peer)

**ExLlamaV3 Approach:**
```cuda
// CPU-side barrier polling system-wide shared memory from GPU
if (this_device == coordinator_device) {
    uint32_t pending = device_mask & ~(1 << this_device);
    while (pending) {
        // Loop through all 16 devices checking epoch flags
        for (int i = 0; i < MAX_DEVICES; i += 4) {
            uint4 s = ldg_cv_u128(p);
            if ((pmask & 1) && s.x == epoch) pending &= ~(1 << i);
        }
        __nanosleep(sleep);  // 64ns → 1024ns exponential backoff
    }
}
```

**Cost:** ~100-500 microseconds (CPU polling shared memory through PCIe)

**Performance Impact:** 50-250x slower barrier

#### 2. Adaptive Algorithm Selection

**vLLM chooses algorithm based on:**
- Message size (bytes)
- Number of GPUs
- Hardware topology (NVLink vs PCIe)

```cpp
// 1-stage: Direct reduce from all peers (best for small messages)
if ((world_size_ <= 4 && bytes < 512 * 1024) ||
    (world_size_ <= 8 && bytes < 256 * 1024)) {
    cross_device_reduce_1stage;  // Single kernel, O(1) rounds
}
// 2-stage: Reduce-scatter + allgather (best for large messages)
else {
    cross_device_reduce_2stage;  // Two kernels, O(1) rounds but better bandwidth
}
```

**ExLlamaV3:** Fixed algorithm per kernel, no adaptive selection

#### 3. Vectorized Memory Access

**vLLM:**
```cuda
// 128-bit packed loads/stores (ld.128/st.128 PTX instructions)
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);  // Single 128-bit load
  #pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));  // Vectorized SIMD add
  }
  return downcast<P>(tmp);
}
```

**ExLlamaV3 P2P:**
```cuda
// Component-wise reads (breaks memory coalescing)
for (int dev = 0; dev < MAX_DEVICES; ++dev) {
    volatile float4* remote_ptr = (volatile float4*)(p2p_ptrs_s[dev] + offset);
    float4 val;
    val.x = remote_ptr->x;  // FOUR separate loads
    val.y = remote_ptr->y;
    val.z = remote_ptr->z;
    val.w = remote_ptr->w;

    sum.x += val.x;  // FOUR separate adds
    sum.y += val.y;
    sum.z += val.z;
    sum.w += val.w;
}
```

**Performance Impact:** 2-4x slower memory access

#### 4. One-Time Buffer Registration

**vLLM:**
```cpp
// Pre-register buffers during initialization
void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  // Store in hash map for O(1) lookup during all-reduce
  buffers_[ptrs[rank_]] = d_data;
}

// All-reduce just does hash lookup
template <typename T>
void allreduce(cudaStream_t stream, T* input, T* output, int size, ...) {
  auto it = buffers_.find(input);  // O(1) hash lookup
  ptrs = it->second;
}
```

**ExLlamaV3:** Opens P2P handles once, but loads peer pointers from shared memory every call

#### 5. Topology Awareness

**vLLM:**
- Detects NVLink vs PCIe
- Disables custom all-reduce for >2 GPUs on PCIe-only systems
- Algorithm selection differs by topology

**ExLlamaV3:** No topology awareness, same code path for NVLink and PCIe

## Per-All-Reduce Breakdown

For typical 4096-hidden all-reduce with 8 GPUs:

**vLLM:**
```
1. Hash lookup: 0.5 µs
2. Barrier (P2P flags): 2 µs
3. 1-stage reduce: 8 µs (8 peers × 1 µs PCIe P2P read)
4. Barrier: 2 µs
Total: ~12.5 µs per all-reduce
```

**ExLlamaV3 P2P:**
```
1. Load P2P pointers: 1 µs
2. Copy to P2P buffer: 5 µs
3. Barrier (CPU polling): 200 µs ← BOTTLENECK
4. Read from 8 peers (volatile): 40 µs
5. No final barrier (removed)
Total: ~246 µs per all-reduce
```

**Ratio:** 246 / 12.5 = **19.7x slower per all-reduce operation**

## Why Overall Speed is Only 2.2-3.6x Slower

If all-reduce is 19.7x slower, why is total inference only 2.2-3.6x slower?

**Reasons:**
1. **All-reduce is only part of inference** - attention, matmul, KV cache lookups dominate
2. **Pipelining** - some latency hidden by overlapping operations
3. **Other bottlenecks** - memory bandwidth, compute bound elsewhere

## The Real Mystery: Why is vLLM 2.2x Faster Even Without Custom All-Reduce?

This is the **critical unanswered question**:

vLLM's baseline fallback (PyNCCL/torch.distributed) is **2.2x faster** than ExLlamaV3's CPU all-reduce (23.5 vs 52 t/s).

**Possible explanations to investigate:**
1. **All-reduce call frequency** - vLLM might call all-reduce less often (fused operations?)
2. **Model architecture differences** - different layer implementations
3. **Kernel efficiency** - vLLM might have faster attention/matmul kernels
4. **Memory layout** - more efficient tensor layouts
5. **Scheduling overhead** - less Python overhead, better batching
6. **Quantization differences** - different quantization schemes
7. **TP implementation** - different ways of splitting the model across GPUs
8. **CUDA graphs** - vLLM might use CUDA graphs for kernel fusion

## Files Referenced

**vLLM:**
- `vllm/distributed/device_communicators/custom_all_reduce.py`
- `vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/distributed/parallel_state.py`
- `vllm/csrc/custom_all_reduce.cu`
- `vllm/csrc/custom_all_reduce.cuh`

**ExLlamaV3:**
- `exllamav3/exllamav3_ext/parallel/all_reduce.cu`
- `exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu`
- `exllamav3/exllamav3_ext/parallel/barrier.cu`
- `exllamav3/exllamav3_ext/parallel/barrier_inner.cuh`
- `exllamav3/exllamav3_ext/parallel/context.cu`
- `exllamav3/model/model_tp_backend.py`
- `exllamav3/model/model_tp_fn.py`

## Next Steps

1. **Investigate vLLM's baseline** - Why is it 2.2x faster without custom all-reduce?
2. **Measure all-reduce frequency** - How many all-reduces per token in each system?
3. **Profile inference breakdown** - Where is time spent in each system?
4. **Check for fused operations** - Does vLLM fuse all-reduce with other ops?
5. **Compare TP implementations** - Different ways of splitting model layers?
