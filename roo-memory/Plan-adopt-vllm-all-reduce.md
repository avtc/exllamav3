# Implementation Plan: Adopt vLLM-Style All-Reduce for ExLlamaV3

**Goal:** Achieve 2-4x performance improvement by adopting vLLM's proven all-reduce techniques

**Expected Outcome:** Increase from 14.5-23.5 t/s to 40-60 t/s on 8x RTX 3090

**Reference:** `roo-memory/Research-vllm-all-reduce.md`

---

## Implementation Steps (In Priority Order)

### Step 1: Replace CPU Barriers with P2P Flag Barriers

**Priority:** HIGHEST
**Expected Speedup:** 2-3x
**Complexity:** Medium
**Files to Modify:**
- `exllamav3/exllamav3_ext/parallel/barrier_inner.cuh`
- `exllamav3/exllamav3_ext/parallel/all_reduce.cu`

**Current Implementation:**
- CPU polls shared memory from GPU
- Spin-wait loops checking epoch flags
- Exponential backoff (64ns → 1024ns)
- Latency: ~100-500 microseconds

**Target Implementation (vLLM-style):**
```cuda
// Add P2P barrier structure to context.cuh
struct P2PBarrier {
    alignas(128) uint32_t start[MAX_DEVICES][MAX_DEVICES];  // [writer][reader]
    alignas(128) uint32_t end[MAX_DEVICES][MAX_DEVICES];
    alignas(128) uint32_t flag[MAX_DEVICES];  // Incremental flags
};

// Pure GPU barrier - no CPU involvement
__device__ __forceinline__
void p2p_barrier_vllm_style(
    P2PBarrier* barrier,
    int rank,
    int world_size
) {
    uint32_t flag = barrier->flag[blockIdx.x] + 1;

    if (threadIdx.x < world_size) {
        // Write to all peers directly via P2P
        barrier->start[rank][threadIdx.x] = flag;

        // Wait for all peers to write to our slot
        while (barrier->start[threadIdx.x][rank] != flag);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        barrier->flag[blockIdx.x] = flag;
    }
}
```

**Advantages:**
- Direct GPU-to-GPU synchronization (no CPU polling)
- Latency: ~1-2 microseconds (50-250x faster)
- Works on PCIe P2P (not just NVLink)
- Simpler code path

**Testing:**
- Verify barrier correctness with unit tests
- Test with 2, 4, 8 GPUs
- Measure barrier latency with CUDA events

---

### Step 2: Vectorize Memory Access Patterns

**Priority:** HIGH
**Expected Speedup:** 1.5-2x
**Complexity:** Low
**Files to Modify:**
- `exllamav3/exllamav3_ext/parallel/all_reduce.cu` (P2P kernel)

**Current Implementation:**
```cuda
// Component-wise reads break memory coalescing
volatile float4* remote_ptr = (volatile float4*)(p2p_ptrs_s[dev] + offset);
float4 val;
val.x = remote_ptr->x;  // FOUR separate volatile loads
val.y = remote_ptr->y;
val.z = remote_ptr->z;
val.w = remote_ptr->w;
```

**Target Implementation:**
```cuda
// Use packed types for 128-bit aligned loads
template <typename T>
struct packed_t {
    using P = array_t<T, 16 / sizeof(T)>;  // e.g., float4
    using A = array_t<float, 16 / sizeof(T)>;  // Accumulator
};

// Single 128-bit load + vectorized reduction
template <typename P, int ngpus, typename A>
__device__ __forceinline__
P packed_reduce(const P* ptrs[], int idx) {
    A tmp = upcast(ptrs[0][idx]);  // Single 128-bit load
    #pragma unroll
    for (int i = 1; i < ngpus; i++) {
        packed_assign_add(tmp, upcast(ptrs[i][idx]));
    }
    return downcast<P>(tmp);
}

// Usage in kernel:
float4 sum = packed_reduce<float4, 8, float4_acc>((const float4**)p2p_ptrs, idx);
```

**Advantages:**
- Generates ld.128/st.128 PTX instructions
- Better memory coalescing
- 2-4x faster memory access
- Works on all GPU architectures

**Testing:**
- Verify bitwise identical results to current implementation
- Benchmark memory access patterns with Nsight Compute

---

### Step 3: Pre-Register IPC Buffers

**Priority:** MEDIUM
**Expected Speedup:** 1.2-1.5x
**Complexity:** Medium
**Files to Modify:**
- `exllamav3/exllamav3_ext/parallel/context.cu`
- `exllamav3/model/model_tp_backend.py`

**Current Implementation:**
- Opens P2P handles once at startup
- But loads peer pointers from shared memory every all-reduce call
- No caching of registered buffers

**Target Implementation:**
```cpp
// In context.cu, add buffer registry
std::unordered_map<void*, RankData*> registered_buffers;

void pg_register_buffer(uintptr_t ctx, void* buffer, int world_size) {
    // Get all peer IPC handles once
    RankData data;
    for (int i = 0; i < world_size; i++) {
        data.ptrs[i] = pg_get_p2p_ptr(i);  // Already opened
    }

    // Copy to device and store in map
    RankData* d_data;
    cudaMalloc(&d_data, sizeof(RankData));
    cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice);
    registered_buffers[buffer] = d_data;
}

// In all-reduce kernel, pass pre-registered buffer
RankData* get_registered_buffer(void* buffer) {
    auto it = registered_buffers.find(buffer);
    if (it != registered_buffers.end()) {
        return it->second;
    }
    return nullptr;  // Fallback to dynamic lookup
}
```

**Python Integration:**
```python
# In model_tp_backend.py
class TPBackendNative:
    def __init__(self):
        self.registered_tensors = set()

    def register_tensor(self, tensor):
        ptr = tensor.data_ptr()
        if ptr not in self.registered_tensors:
            ext.pg_register_buffer(self.ptr_g, ptr, self.world_size)
            self.registered_tensors.add(ptr)
```

**Advantages:**
- One-time registration overhead
- O(1) hash lookup vs repeated pointer loading
- Better for CUDA graphs (fixed pointers)

**Testing:**
- Verify peer pointers are correctly shared
- Test memory doesn't leak (unregister on cleanup)
- Benchmark hash lookup performance

---

### Step 4: Implement 1-Stage Algorithm for Small Messages

**Priority:** MEDIUM-HIGH
**Expected Speedup:** 1.5-2x for small messages
**Complexity:** Medium
**Files to Modify:**
- `exllamav3/exllamav3_ext/parallel/all_reduce.cu` (new kernel)
- `exllamav3/model/model_tp_backend.py` (dispatch logic)

**When to Use:**
- Message size < 256 KB for 8 GPUs
- Message size < 512 KB for 4 GPUs
- Always for 2 GPUs

**Algorithm:**
```cuda
template <typename T, int ngpus>
__global__ __launch_bounds__(512, 1)
void all_reduce_1stage(
    RankData* peer_ptrs,      // Pre-registered peer pointers
    T* __restrict__ output,
    int rank,
    int size
) {
    using P = typename packed_t<T>::P;  // e.g., float4
    using A = typename packed_t<T>::A;  // Accumulator

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // P2P barrier before reading
    p2p_barrier_vllm_style(barrier, rank, ngpus);

    // Direct read from all peers and accumulate
    const P* ptrs[8];
    #pragma unroll
    for (int i = 0; i < ngpus; i++) {
        ptrs[i] = (const P*)peer_ptrs->ptrs[i];
    }

    ((P*)output)[idx] = packed_reduce<P, ngpus, A>(ptrs, idx);

    // P2P barrier after writing
    p2p_barrier_vllm_style(barrier, rank, ngpus);
}
```

**Advantages:**
- Single kernel launch
- Single barrier (vs 2+ in multi-stage)
- Best for small messages
- Simpler code path

**Testing:**
- Compare against 2-stage for various sizes
- Find optimal threshold for 1-stage vs 2-stage
- Verify bitwise identical results

---

### Step 5: Add Adaptive Algorithm Selection

**Priority:** MEDIUM
**Expected Speedup:** 1.3-1.8x overall
**Complexity:** Low
**Files to Modify:**
- `exllamav3/model/model_tp_backend.py`

**Selection Logic:**
```python
def choose_all_reduce_algorithm(
    data_size_bytes: int,
    num_gpus: int,
    has_p2p: bool
) -> str:
    """
    Choose optimal all-reduce algorithm based on:
    - Message size
    - Number of GPUs
    - Hardware capabilities
    """

    # Always use 1-stage for 2 GPUs
    if num_gpus == 2:
        return "p2p_1stage"

    # Use 1-stage for small messages with P2P
    if has_p2p:
        if num_gpus <= 4 and data_size_bytes < 512 * 1024:
            return "p2p_1stage"
        if num_gpus <= 8 and data_size_bytes < 256 * 1024:
            return "p2p_1stage"

    # Use CPU all-reduce for large messages
    if data_size_bytes > 2 * 1024 * 1024:  # > 2MB
        return "cpu"

    # Default to 2-stage P2P
    return "p2p_2stage"
```

**Integration:**
```python
def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
    data_size = tensor.numel() * tensor.element_size()

    algorithm = choose_all_reduce_algorithm(
        data_size_bytes=data_size,
        num_gpus=self.world_size,
        has_p2p=self.has_p2p
    )

    if algorithm == "p2p_1stage":
        return ext.pg_all_reduce_p2p_1stage(
            self.ptr_g, tensor, ...
        )
    elif algorithm == "p2p_2stage":
        return ext.pg_all_reduce_p2p_2stage(
            self.ptr_g, tensor, ...
        )
    elif algorithm == "cpu":
        return ext.pg_all_reduce_cpu(
            self.ptr_g, tensor, ...
        )
```

**Advantages:**
- Automatically uses best algorithm for each call
- No manual tuning needed
- Adapts to different model sizes
- Future-proof for new hardware

**Testing:**
- Benchmark each algorithm at different sizes
- Verify thresholds are optimal
- Test on different GPU counts (2, 4, 8)
- Profile to ensure decision overhead is minimal

---

## Success Metrics

**Target Performance Improvements:**

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| P2P all-reduce | 14.5 t/s | 35-45 t/s | 2.4-3.1x |
| CPU all-reduce | 23.5 t/s | 30-35 t/s | 1.3-1.5x |
| Best overall | 23.5 t/s | 40-50 t/s | 1.7-2.1x |

**Per-All-Reduce Latency:**

| Implementation | Current | Target | Improvement |
|----------------|---------|--------|-------------|
| P2P barrier | ~200 µs | ~2 µs | 100x |
| Memory access | ~40 µs | ~10 µs | 4x |
| Total per call | ~246 µs | ~15 µs | 16.4x |

---

## Implementation Timeline

**Week 1:**
- Step 1: Replace CPU barriers with P2P barriers (2-3 days)
- Step 2: Vectorize memory access (1-2 days)

**Week 2:**
- Step 3: Pre-register buffers (2 days)
- Step 4: Implement 1-stage algorithm (2-3 days)

**Week 3:**
- Step 5: Adaptive selection (1-2 days)
- Testing and benchmarking (2-3 days)
- Documentation and cleanup (1 day)

**Total:** ~3 weeks for full implementation and testing

---

## Risks and Mitigations

**Risk 1: P2P barriers may not work on all hardware**
- Mitigation: Keep CPU barrier as fallback
- Test on diverse GPU configurations

**Risk 2: Vectorized access may have numerical differences**
- Mitigation: Extensive testing for bitwise identical results
- Add tolerance for floating-point differences

**Risk 3: Buffer registration increases memory usage**
- Mitigation: Limit cache size, LRU eviction
- Profile memory usage

**Risk 4: Algorithm selection adds overhead**
- Mitigation: Cache decisions, lazy evaluation
- Profile decision logic

**Risk 5: May not reach vLLM's 60 t/s**
- Mitigation: Focus on incremental improvements
- Investigate other bottlenecks (kernels, memory layout)

---

## Next Steps After Implementation

1. **Profile full inference pipeline** - Find other bottlenecks beyond all-reduce
2. **Compare with vLLM baseline** - Understand remaining 2.2x gap
3. **Investigate all-reduce frequency** - Are we calling it more often?
4. **Explore kernel fusion** - Can we fuse all-reduce with other operations?
5. **Consider CUDA graphs** - For better kernel launch overhead
