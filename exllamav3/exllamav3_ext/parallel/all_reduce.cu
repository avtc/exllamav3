#include <cuda_fp16.h>
#include "all_reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "ll.cuh"
#include "barrier_inner.cuh"
#include "p2p_barrier.cuh"
#include "vectorization.cuh"  // Vectorized memory access

#define MAX_NUM_THREADS 1024
#define BATCH_STAGE 2

__global__ __launch_bounds__(MAX_NUM_THREADS)
void pg_all_reduce_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ shbuf_ptr,
    const size_t data_size,
    const size_t shbuf_size,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();

    __shared__ bool r;
    int dir = blockIdx.x;

    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    uint8_t* data_end = data_ptr + data_size;
    const size_t reduce_stage_size = blockDim.x * sizeof(uint4);

    // Divide shared buffer among ranks
    size_t rank_shbuf_size = shbuf_size / num_ranks / reduce_stage_size * reduce_stage_size;

    // Divide each rank into segments divisible into stages, last segment may need padding
    size_t segment_size = CEIL_DIVIDE(data_size, num_ranks);
    segment_size = CEIL_DIVIDE(segment_size, reduce_stage_size) * reduce_stage_size;

    // Divide each workload and buffer into stages
    int num_stages = segment_size / reduce_stage_size;
    int num_buf_stages = rank_shbuf_size / reduce_stage_size;
    bool no_overflow = num_stages * 2 * (num_ranks - 1) < num_buf_stages - 2;

    // Indexing
    auto shbuf_stage_ptr = [&] (int rank, int stage_idx)
    {
        return shbuf_ptr +
               rank * rank_shbuf_size +
               (stage_idx % num_buf_stages) * reduce_stage_size;
    };

    auto data_stage_ptr = [&] (int segment_idx, int stage_idx)
    {
        return data_ptr +
               segment_idx * segment_size +
               (stage_idx % num_stages) * reduce_stage_size;
    };

    // Sync
    auto wait_min_stage = [&] (uint32_t* stage_ptr, int min_stage, uint64_t deadline)
    {
        if (t == 0)
        {
            if (ctx->busy_wait)
            {
                while ((int) ldg_acquire_sys_u32(stage_ptr) < min_stage)
                {
                     if (check_timeout(ctx, deadline, "all_reduce")) { *abort_flag = 1; break; }
                }
            }
            else
            {
                uint32_t sleep = SYNC_MIN_SLEEP;
                while ((int) ldg_acquire_sys_u32(stage_ptr) < min_stage)
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "all_reduce");
                    if (*abort_flag) break;
                }
            }
        }
        __syncthreads();
    };

    auto stage_ready = [&] (uint32_t* stage_ptr, int min_stage)
    {
        if (t == 0)
            r = ((int) ldg_acquire_sys_u32(stage_ptr) >= min_stage);
        __syncthreads();
        return r;
    };

    // Send to next rank, receive from previous rank
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));
    int dst_rank = (this_rank + 1) % num_ranks;
    int src_rank = (this_rank + num_ranks - 1) % num_ranks;

    // Loop around ring
    for (int iter = 0; iter < (num_ranks - 1) * 2; ++iter)
    {
        uint64_t deadline = sync_deadline();

        // Outgoing segment to (rank+1)%num_ranks is (rank+iter)%num_iters
        // Incoming segment from (rank-1)%num_ranks is (rank+iter-1)%num_iters
        int send_seg = (this_rank + num_ranks * 2 - iter) % num_ranks;
        int recv_seg = (this_rank + num_ranks * 2 - iter - 1) % num_ranks;

        int stage_beg = iter * num_stages;
        int stage_end = stage_beg + num_stages;
        int stage_send = stage_beg;
        int stage_recv = stage_beg;

        uint32_t sleep = SYNC_MIN_SLEEP;

        if (dir == 0)
        {
            while (stage_recv < stage_end)
            {
                __shared__ uint32_t sr;
                if (t == 0)
                    sr = (int) ldg_acquire_sys_u32(ctx->reduce_stage_produced + src_rank);
                __syncthreads();
                uint32_t stage_ready = sr;

                if (stage_recv < stage_ready)
                {
                    while (stage_recv < stage_ready)
                    {
                        sleep = SYNC_MIN_SLEEP;

                        // First num_ranks - 1 iterations: accumulate
                        if (iter < num_ranks - 1)
                        {
                            float4* src = (float4*) shbuf_stage_ptr(this_rank, stage_recv);
                            float4* dst = (float4*) data_stage_ptr(recv_seg, stage_recv);
                            if (dst + t < (float4*) data_end)
                            {
                                float4 a = dst[t];
                                float4 b = src[t];
                                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                                dst[t] = a;
                            }
                        }

                        // Last num_ranks - 1 iterations: copy
                        else
                        {
                            uint4* src = (uint4*) shbuf_stage_ptr(this_rank, stage_recv);
                            uint4* dst = (uint4*) data_stage_ptr(recv_seg, stage_recv);
                            if (dst + t < (uint4*) data_end) dst[t] = src[t];
                        }

                        // Advance
                        stage_recv++;
                    }
                    if (t == 0)
                    {
                        // __threadfence_system();
                        // __syncthreads();
                        stg_release_sys_u32(ctx->reduce_stage_consumed + this_rank, stage_recv);
                    }
                }
                else
                {
                    if (ctx->busy_wait)
                    {
                         if (check_timeout(ctx, deadline, "all_reduce (1)")) { *abort_flag = 1; break; }
                    }
                    else
                    {
                        __nanosleep(sleep);
                        if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                        else *abort_flag = check_timeout(ctx, deadline, "all_reduce (1)");
                        if (*abort_flag) break;
                    }
                }
            }
        }

        // Send
        if (dir == 1)
        {
            while (stage_send < stage_end)
            {
                bool ready_send = stage_send < stage_end &&
                                  (no_overflow || stage_ready(ctx->reduce_stage_consumed + dst_rank, stage_send - num_buf_stages + 1 + BATCH_STAGE));
                if (ready_send)
                {
                    for (int i = 0; i < BATCH_STAGE && stage_send < stage_end; ++i)
                    {
                        sleep = SYNC_MIN_SLEEP;
                        uint4* src = (uint4*) data_stage_ptr(send_seg, stage_send);
                        uint4* dst = (uint4*) shbuf_stage_ptr(dst_rank, stage_send);
                        if (src + t < (uint4*) data_end) dst[t] = src[t];

                        // Advance
                        stage_send++;
                    }

                    if (t == 0)
                    {
                        // __threadfence_system();
                        stg_release_sys_u32(ctx->reduce_stage_produced + this_rank, stage_send);
                    }
                }
                else
                {
                    if (ctx->busy_wait)
                    {
                         if (check_timeout(ctx, deadline, "all_reduce (2)")) { *abort_flag = 1; break; }
                    }
                    else
                    {
                        __nanosleep(sleep);
                        if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                        else *abort_flag = check_timeout(ctx, deadline, "all_reduce (2)");
                        if (*abort_flag) break;
                    }
                }
            }

            // Wait for destination to finish receiving
            wait_min_stage(ctx->reduce_stage_consumed + dst_rank, stage_end, deadline);
        }

        if (*abort_flag) break;
        grid.sync();
    }

    // Finished. Reset counters for next kernel
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);

    if (t == 0)
    {
        ctx->reduce_stage_consumed[dst_rank] = 0;
        ctx->reduce_stage_produced[this_rank] = 0;
        __threadfence_system();
    }
}

struct P2PPtrs { void* ptrs[MAX_DEVICES]; };
struct P2PBarrierPtrs { void* barriers[MAX_DEVICES]; };

__global__ __launch_bounds__(MAX_NUM_THREADS)
void pg_all_reduce_p2p_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    const size_t data_size,
    uint32_t* abort_flag,
    P2PPtrs p2p_ptrs
)
{
    int t = threadIdx.x;
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;

    // Load P2P pointers into shared memory for faster access
    extern __shared__ uint8_t smem[];
    uint8_t** p2p_ptrs_s = (uint8_t**)smem;
    
    if (t < MAX_DEVICES)
    {
        if ((device_mask >> t) & 1)
            p2p_ptrs_s[t] = (uint8_t*)p2p_ptrs.ptrs[t];
        else
            p2p_ptrs_s[t] = nullptr;
    }
    __syncthreads();

    // Validate our P2P pointer
    uint8_t* my_p2p_ptr = p2p_ptrs_s[this_device];
    if (!my_p2p_ptr) 
    {
        if (t == 0) {
            printf("ExLlamaV3: P2P ERROR - Device %d has no P2P pointer!\n", this_device);
            *abort_flag = 1;
        }
        return;
    }

    // Phase 1: Copy local data to P2P buffer
    for (size_t offset = t * 16; offset < data_size; offset += blockDim.x * 16)
    {
        *((uint4*)(my_p2p_ptr + offset)) = *((uint4*)(data_ptr + offset));
    }
    
    // Ensure all writes are visible system-wide
    __threadfence_system();
    __syncthreads();

    // Phase 2: Barrier - wait for all GPUs to finish writing
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
    if (*abort_flag) return;

    // Phase 3: Reduce - read from all P2P buffers and accumulate
    for (size_t offset = t * 16; offset < data_size; offset += blockDim.x * 16)
    {
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Accumulate from all active devices
        for (int dev = 0; dev < MAX_DEVICES; ++dev)
        {
            if (!p2p_ptrs_s[dev]) continue;
            
            // Use volatile pointer for system-wide visibility
            volatile float4* remote_ptr = (volatile float4*)(p2p_ptrs_s[dev] + offset);
            float4 val;
            val.x = remote_ptr->x;
            val.y = remote_ptr->y;
            val.z = remote_ptr->z;
            val.w = remote_ptr->w;
            
            sum.x += val.x;
            sum.y += val.y;
            sum.z += val.z;
            sum.w += val.w;
        }
        
        // Write result back to local data
        *((float4*)(data_ptr + offset)) = sum;
    }

    // Ensure all threads finished writing
    __threadfence_system();
    __syncthreads();

    // No final barrier needed - results written to local memory
    // Other devices will read their own copies in Phase 3
}

// vLLM-style P2P all-reduce kernel with GPU-only barriers
__global__ __launch_bounds__(MAX_NUM_THREADS)
void pg_all_reduce_p2p_kernel_v2
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    int this_device,
    uint8_t* __restrict__ data_ptr,
    const size_t data_size,
    P2PBarrier** barrier_ptr_array,    // Pre-built compact array
    float4** p2p_ptr_array,            // Pre-built compact array (float4* to avoid casts)
    int num_ranks,                     // Pre-calculated
    int this_rank                      // Pre-calculated
)
{
    int t = threadIdx.x;
    if (num_ranks <= 1) return;

    // Trace: Log kernel parameters (only thread 0 to avoid spam)
    if (t == 0) {
        printf("ExLlamaV3: [Device %d kernel] device_mask=0x%x, this_rank=%d/%d, data_size=%zu\n",
               this_device, device_mask, this_rank, num_ranks, data_size);
        printf("ExLlamaV3: [Device %d kernel] P2P pointer array received:\n", this_device);
        for (int i = 0; i < num_ranks; ++i) {
            printf("  p2p_ptr_array[%d] = %p\n", i, p2p_ptr_array[i]);
        }
    }

    // Validate our P2P pointer (direct from parameter, no shared memory needed)
    float4* my_p2p_ptr = p2p_ptr_array[this_rank];
    if (!my_p2p_ptr)
    {
        if (t == 0) {
            printf("ExLlamaV3: P2P ERROR - Device %d (rank %d) has no P2P pointer!\n", this_device, this_rank);
            printf("ExLlamaV3: P2P ERROR - Attempted to access p2p_ptr_array[%d] which is NULL\n", this_rank);
            printf("ExLlamaV3: P2P ERROR - This is a fatal error. Aborting kernel.\n");
        }
        // Explicitly fail - don't silently return
        assert(my_p2p_ptr != nullptr && "P2P pointer is null - check P2P handle initialization");
        return;  // Will never reach here due to assert
    }

    // Phase 1: Copy local data to P2P buffer
    for (size_t offset = t * 16; offset < data_size; offset += blockDim.x * 16)
    {
        *((uint4*)(my_p2p_ptr + offset)) = *((uint4*)(data_ptr + offset));
    }

    // Ensure all writes are visible system-wide
    __threadfence_system();
    __syncthreads();

    // Phase 2: vLLM-style P2P barrier - wait for all GPUs (GPU-only, no CPU polling)
    // Array already pre-built on host, no nested loops needed!
    p2p_barrier_vllm_style(barrier_ptr_array, this_rank, num_ranks, true);

    // Phase 3: Reduce - read from all P2P buffers and accumulate using vectorized loads
    // Single 128-bit load from each GPU (compiler generates ld.f32.v4)
    // Direct access from p2p_ptr_array parameter, no casts needed
    for (size_t offset = t * 16; offset < data_size; offset += blockDim.x * 16)
    {
        // Index in terms of float4 elements (16 bytes each)
        size_t vec_idx = offset / 16;

        // Load from first GPU (direct from parameter, no cast)
        float4 sum = p2p_ptr_array[0][vec_idx];

        // Accumulate from remaining GPUs (vectorized loads, no casts)
        for (int i = 1; i < num_ranks; ++i)
        {
            float4 val = p2p_ptr_array[i][vec_idx];
            sum.x += val.x;
            sum.y += val.y;
            sum.z += val.z;
            sum.w += val.w;
        }

        // Write result back to local data (128-bit store)
        *((float4*)(data_ptr + offset)) = sum;
    }

    // Ensure all threads finished writing
    __threadfence_system();
    __syncthreads();

    // No final barrier needed - results written to local memory
    // Other devices will read their own copies in Phase 3
}

__global__ __launch_bounds__(MAX_NUM_THREADS)
void pg_all_reduce_small_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ shbuf_ptr,
    const size_t data_size,
    const size_t shbuf_size,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));

    size_t rank_shbuf_size = shbuf_size / num_ranks;
    // Safety check handled by caller or assume it fits

    uint8_t* my_shbuf_ptr = shbuf_ptr + this_rank * rank_shbuf_size;
    uint8_t* data_end = data_ptr + data_size;

    // 1. Write to shared buffer
    uint8_t* src = data_ptr + t * 16;
    uint8_t* dst = my_shbuf_ptr + t * 16;
    
    // Copy in float4/uint4 chunks (16 bytes)
    while (src < data_end)
    {
        *((uint4*)dst) = *((uint4*)src);
        src += blockDim.x * 16;
        dst += blockDim.x * 16;
    }
    
    // 2. Sync
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
    if (*abort_flag) return;

    // 3. Read and Reduce
    for (size_t offset = t * 16; offset < data_size; offset += blockDim.x * 16)
    {
        float4 acc = {0.0f, 0.0f, 0.0f, 0.0f}; 
        bool first = true;
        
        for (int r = 0; r < num_ranks; ++r)
        {
             uint8_t* r_ptr = shbuf_ptr + r * rank_shbuf_size + offset;
             float4 val = *((float4*)r_ptr);
             
             if (first) { acc = val; first = false; }
             else 
             {
                 acc.x += val.x;
                 acc.y += val.y;
                 acc.z += val.z;
                 acc.w += val.w;
             }
        }
        
        *((float4*)(data_ptr + offset)) = acc;
    }

    // Finished. Sync/Barrier
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
}

void pg_all_reduce
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* shbuf_ptr = (uint8_t*) shbuf;
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 16 == 0, "data_size must be multiple of 16");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    long num_ranks = devices.size();
    
    // Check P2P availability with retry logic
    // After the second barrier in init_pg(), all devices should have opened handles
    static bool p2p_active = false;
    static P2PPtrs cached_p2p_ptrs;
    static int p2p_validation_attempts = 0;
    static const int MAX_P2P_RETRIES = 100;  // Retry for up to ~10 seconds total

    // Re-validate until success or max retries (in case some devices are slow to initialize)
    if (!p2p_active && p2p_validation_attempts < MAX_P2P_RETRIES)
    {
        void* my_ptr = pg_get_p2p_ptr(this_device);
        if (my_ptr)
        {
            // Populate cache - validate all pointers
            bool all_valid = true;
            int missing_count = 0;
            int missing_devices[MAX_DEVICES];

            for(int i=0; i<MAX_DEVICES; ++i) {
                cached_p2p_ptrs.ptrs[i] = pg_get_p2p_ptr(i);
                if ((device_mask >> i) & 1) {
                    if (!cached_p2p_ptrs.ptrs[i]) {
                        missing_devices[missing_count++] = i;
                        all_valid = false;
                    }
                }
            }

            p2p_validation_attempts++;

            if (all_valid) {
                p2p_active = true;
                printf("ExLlamaV3: [Device %d] P2P All-Reduce Active (validated all %ld devices on attempt %d)\n",
                       this_device, num_ranks, p2p_validation_attempts);
            } else {
                // Log which devices are missing
                if (p2p_validation_attempts == 1 || p2p_validation_attempts % 10 == 0) {
                    printf("ExLlamaV3: [Device %d] P2P validation attempt %d: Missing %d devices - ",
                           this_device, p2p_validation_attempts, missing_count);
                    for (int i = 0; i < missing_count && i < 8; ++i) {
                        printf("%d ", missing_devices[i]);
                    }
                    if (missing_count > 8) printf("...");
                    printf("\n");
                }

                // On final attempt, give up and fall back to host memory
                if (p2p_validation_attempts >= MAX_P2P_RETRIES) {
                    printf("ExLlamaV3: [Device %d] P2P validation FAILED after %d attempts - falling back to host memory\n",
                           this_device, MAX_P2P_RETRIES);
                    printf("ExLlamaV3: [Device %d] Missing devices: ", this_device);
                    for (int i = 0; i < missing_count; ++i) {
                        printf("%d ", missing_devices[i]);
                    }
                    printf("\n");
                }
            }
        }
        else
        {
            p2p_validation_attempts++;
            if (p2p_validation_attempts == 1) {
                printf("ExLlamaV3: [Device %d] P2P All-Reduce Not available (no local P2P pointer)\n", this_device);
            }
        }
    }

    // Get P2P buffer size from context.cu
    size_t p2p_buffer_size = 17 * 128 * 1024; // Match SHBUF_SIZE_R from model_tp_backend.py
    bool can_use_small_direct = (data_size <= 512 * 1024);
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();

    // Use P2P kernel if available and data fits
    if (p2p_active && data_size <= p2p_buffer_size)
    {
         int threads = MAX_NUM_THREADS;
         if (data_size < threads * 16) threads = CEIL_DIVIDE(data_size, 16);
         threads = ((threads + 31) / 32) * 32;

         void* kernelArgs[] =
         {
            (void*)& ctx,
            (void*)& device_mask,
            (void*)& this_device,
            (void*)& master_device,
            (void*)& data_ptr,
            (void*)& data_size,
            (void*)& abort_flag_ptr,
            (void*)& cached_p2p_ptrs
         };
         
         cudaLaunchCooperativeKernel
         (
            (void*)pg_all_reduce_p2p_kernel,
            dim3(1),
            dim3(threads),
            kernelArgs,
            sizeof(uint8_t*) * MAX_DEVICES, // Shared memory for pointer array
            stream
         );
    }
    else if (can_use_small_direct && (data_size <= shbuf_size / num_ranks))
    {
        // Host memory direct reduce (fallback)
        int threads = MAX_NUM_THREADS;
        if (data_size < threads * 16) threads = CEIL_DIVIDE(data_size, 16);
        threads = ((threads + 31) / 32) * 32;

        void* kernelArgs[] =
        {
            (void*)& ctx,
            (void*)& device_mask,
            (void*)& this_device,
            (void*)& master_device,
            (void*)& data_ptr,
            (void*)& shbuf_ptr,
            (void*)& data_size,
            (void*)& shbuf_size,
            (void*)& abort_flag_ptr
        };

        cudaLaunchCooperativeKernel
        (
            (void*)pg_all_reduce_small_kernel,
            dim3(1), 
            dim3(threads),
            kernelArgs,
            0,
            stream
        );
    }
    else
    {
        // Ring all-reduce for large tensors
        int threads = (int) CEIL_DIVIDE(CEIL_DIVIDE(data_size / 16ll, num_ranks), 32ll) * 32ll;
        threads = MIN(threads, MAX_NUM_THREADS);

        void* kernelArgs[] =
        {
            (void*)& ctx,
            (void*)& device_mask,
            (void*)& this_device,
            (void*)& master_device,
            (void*)& data_ptr,
            (void*)& shbuf_ptr,
            (void*)& data_size,
            (void*)& shbuf_size,
            (void*)& abort_flag_ptr
        };

        dim3 block_grid(2);
        dim3 block_dim(threads);

        cudaLaunchCooperativeKernel
        (
            (void*)pg_all_reduce_kernel,
            block_grid,
            block_dim,
            kernelArgs,
            0,
            stream
        );
    }

    cuda_check(cudaPeekAtLastError());
}

// vLLM-style P2P all-reduce with GPU-only barriers
void pg_all_reduce_p2p_v2
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 16 == 0, "data_size must be multiple of 16");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    long num_ranks = devices.size();

    if (num_ranks <= 1) return;

    // Pre-register P2P buffers using static caching (per-device)
    static bool p2p_v2_validated = false;
    static P2PPtrs cached_p2p_ptrs;
    static P2PBarrierPtrs cached_barrier_ptrs;
    static P2PBarrier* cached_barrier_array[MAX_DEVICES][MAX_DEVICES];  // [this_device][rank] - per-device cache
    static float4* cached_p2p_ptr_array[MAX_DEVICES][MAX_DEVICES];      // [this_device][rank] - per-device cache
    static int cached_num_ranks = 0;  // Same for all devices, no need for array
    static int cached_this_rank = 0;  // Each device knows its rank, no need for array

    // Validate once and cache (no retries - fail fast if P2P handles not opened)
    if (!p2p_v2_validated)
    {
        bool all_valid = true;

        // Cache P2P pointers
        for (int i = 0; i < MAX_DEVICES; ++i) {
            cached_p2p_ptrs.ptrs[i] = pg_get_p2p_ptr(i);
            if ((device_mask >> i) & 1 && !cached_p2p_ptrs.ptrs[i]) {
                all_valid = false;
                printf("ExLlamaV3: [Device %d] WARNING: P2P pointer for device %d is NULL\n", this_device, i);
            }
        }

        // Cache barrier pointers
        for (int i = 0; i < MAX_DEVICES; ++i) {
            cached_barrier_ptrs.barriers[i] = pg_get_p2p_barrier_ptr(i);
            if ((device_mask >> i) & 1 && !cached_barrier_ptrs.barriers[i]) {
                all_valid = false;
                printf("ExLlamaV3: [Device %d] WARNING: Barrier pointer for device %d is NULL\n", this_device, i);
            }
        }

        if (all_valid) {
            // Build compact arrays (only active devices) for kernel efficiency
            int idx = 0;
            for (int bit = 0; bit < MAX_DEVICES; ++bit) {
                if ((device_mask >> bit) & 1) {
                    cached_barrier_array[this_device][idx] = (P2PBarrier*)cached_barrier_ptrs.barriers[bit];  // Store in this device's row
                    cached_p2p_ptr_array[this_device][idx] = (float4*)cached_p2p_ptrs.ptrs[bit];              // Store in this device's row
                    idx++;
                }
            }
            cached_num_ranks = num_ranks;
            cached_this_rank = __builtin_popcount(device_mask & ((1 << this_device) - 1));

            p2p_v2_validated = true;
            printf("ExLlamaV3: [Device %d] P2P v2: Pre-registered buffers validated (rank=%d/%d)\n",
                   this_device, cached_this_rank, cached_num_ranks);

            // Trace: Log all pointers in this device's compact arrays
            printf("ExLlamaV3: [Device %d] P2P v2: Compact P2P pointer array:\n", this_device);
            for (int i = 0; i < cached_num_ranks; ++i) {
                printf("  [%d] = %p\n", i, cached_p2p_ptr_array[this_device][i]);
            }
        } else {
            printf("ExLlamaV3: [Device %d] P2P v2 ERROR: Not all pointers available. EXLLAMA_TP_P2P=1 requires all P2P handles to be opened.\n", this_device);
            TORCH_CHECK(false, "P2P validation failed - EXLLAMA_TP_P2P=1 requires all P2P handles to be opened");
        }
    }

    // Check data size fits in P2P buffer
    size_t p2p_buffer_size = 17 * 128 * 1024; // Match SHBUF_SIZE_R
    if (data_size > p2p_buffer_size)
    {
        printf("ExLlamaV3: [Device %d] P2P v2: Data too large (%zu > %zu), falling back\n",
               this_device, data_size, p2p_buffer_size);
        return;
    }

    // Launch v2 kernel with GPU-only barriers
    int threads = MAX_NUM_THREADS;
    if (data_size < threads * 16) threads = CEIL_DIVIDE(data_size, 16);
    threads = ((threads + 31) / 32) * 32;

    void* kernelArgs[] =
    {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& data_ptr,
        (void*)& data_size,
        (void*)cached_barrier_array[this_device],  // This device's barrier row (decays to P2PBarrier**)
        (void*)cached_p2p_ptr_array[this_device],  // This device's P2P row (decays to float4**)
        (void*)& cached_num_ranks,                // Pre-calculated
        (void*)& cached_this_rank                 // Pre-calculated
    };

    cudaLaunchCooperativeKernel
    (
        (void*)pg_all_reduce_p2p_kernel_v2,
        dim3(1),
        dim3(threads),
        kernelArgs,
        0,  // No shared memory needed (direct parameter access)
        stream
    );

    cuda_check(cudaPeekAtLastError());
}


__global__ void pg_verify_p2p_kernel
(
    P2PPtrs p2p_ptrs,
    uint32_t device_mask,
    int this_device,
    uint32_t* result
)
{
    if (threadIdx.x != 0) return;
    
    // Check if my pointer is valid
    uint8_t* my_ptr = (uint8_t*)p2p_ptrs.ptrs[this_device];
    if (!my_ptr) {
        printf("Device %d: My P2P pointer is NULL!\n", this_device);
        atomicOr(result, 1);
        return;
    }
    
    // Try to write and read from my buffer
    volatile uint32_t* test_ptr = (volatile uint32_t*)my_ptr;
    test_ptr[0] = 0x12345678 + this_device;
    __threadfence_system();
    
    uint32_t readback = test_ptr[0];
    if (readback != 0x12345678 + this_device) {
        printf("Device %d: Write-read test FAILED (wrote 0x%x, read 0x%x)\n", 
               this_device, 0x12345678 + this_device, readback);
        atomicOr(result, 2);
        return;
    }
    
    // Check peer pointers
    for (int dev = 0; dev < MAX_DEVICES; ++dev) {
        if (!((device_mask >> dev) & 1)) continue;
        if (dev == this_device) continue;
        
        uint8_t* peer_ptr = (uint8_t*)p2p_ptrs.ptrs[dev];
        if (!peer_ptr) {
            printf("Device %d: Peer %d pointer is NULL!\n", this_device, dev);
            atomicOr(result, 4);
        }
    }
}

