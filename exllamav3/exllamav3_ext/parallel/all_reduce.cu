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
    P2PPtrs p2p_ptrs // Passed by value
)
{
    int t = threadIdx.x;
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));

    // Load my P2P pointer
    uint8_t* my_p2p_ptr = (uint8_t*)p2p_ptrs.ptrs[this_device];
    if (!my_p2p_ptr) 
    {
        printf("ExLlamaV3: P2P FATAL: Rank %d has no P2P pointer!\n", this_device);
        *abort_flag = 1; 
        return; 
    }

    uint8_t* data_end = data_ptr + data_size;

    // 1. Copy data to my P2P buffer (Publish)
    uint8_t* src = data_ptr + t * 16;
    uint8_t* dst = my_p2p_ptr + t * 16;
    
    while (src < data_end)
    {
        *((uint4*)dst) = *((uint4*)src);
        src += blockDim.x * 16;
        dst += blockDim.x * 16;
    }
    
    // Ensure writes to P2P buffer are visible to peers
    __threadfence_system();
    __syncthreads(); // Ensure all threads finished writing

    // 2. Sync
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
    if (*abort_flag) return;

    // 3. Read and Reduce from Peers
    extern __shared__ uint8_t* p2p_ptrs_s[];
    
    if (t < MAX_DEVICES)
    {
         if ((device_mask >> t) & 1)
            p2p_ptrs_s[t] = (uint8_t*)p2p_ptrs.ptrs[t];
         else
            p2p_ptrs_s[t] = nullptr;
    }
    __syncthreads();

    for (size_t offset = t * 16; offset < data_size; offset += blockDim.x * 16)
    {
        float4 acc = {0.0f, 0.0f, 0.0f, 0.0f}; 
        bool first = true;
        
        for (int dev = 0; dev < MAX_DEVICES; ++dev)
        {
             if (!p2p_ptrs_s[dev]) 
             {
                 // If this device is supposed to be active (in mask) but has no pointer, it's an error.
                 // However, p2p_ptrs_s[dev] is set to nullptr if NOT in mask.
                 // We need to check if it SHOULD be there.
                 if ((device_mask >> dev) & 1) 
                 {
                     printf("ExLlamaV3: P2P FATAL: Rank %d missing peer pointer for Rank %d!\n", this_device, dev);
                     *abort_flag = 1;
                 }
                 continue;
             }
             
             uint8_t* r_ptr = p2p_ptrs_s[dev] + offset;
             volatile float4* v_ptr = (volatile float4*)r_ptr;
             float4 val;
             val.x = v_ptr->x;
             val.y = v_ptr->y;
             val.z = v_ptr->z;
             val.w = v_ptr->w;
             
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

    // Ensure output writes are visible
    __threadfence_system();
    __syncthreads();

    // Finished. Sync/Barrier
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
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
    
    // Check P2P availability locally (Optimized: Lazy static check)
    static bool p2p_checked = false;
    static bool p2p_active = false;
    static P2PPtrs cached_p2p_ptrs; // Cache pointers once
    
    if (!p2p_checked)
    {
        // Check if we have a pointer for OUR device. 
        // If we do, we assume P2P was set up correctly for this context.
        if (pg_get_p2p_ptr(this_device)) 
        {
             p2p_active = true;
             // Populate cache
             for(int i=0; i<MAX_DEVICES; ++i) cached_p2p_ptrs.ptrs[i] = pg_get_p2p_ptr(i);
             printf("ExLlamaV3: P2P All-Reduce Active\n");
        }
        else
        {
            // P2P not available or not configured for this rank
             printf("ExLlamaV3: P2P All-Reduce Not available\n");
        }
        p2p_checked = true;
    }

    // Heuristic for direct reduce (Host only)
    bool can_use_small_direct = (data_size <= 512 * 1024);
    
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();

    if (p2p_active && data_size <= shbuf_size)
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
            sizeof(uint8_t*) * MAX_DEVICES, // Shared memory size
            stream
         );
    }
    else if (can_use_small_direct && (data_size <= shbuf_size / num_ranks))
    {
        // Host Memory Direct Reduce (Fallback)
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