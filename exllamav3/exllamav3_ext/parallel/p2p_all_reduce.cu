#include <cuda_fp16.h>
#include "p2p_all_reduce.cuh"
#include "p2p_tree_reduce.cuh"
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
void p2p_all_reduce_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    size_t data_size,
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
    
    // Divide data into segments for ring-based reduction
    size_t segment_size = CEIL_DIVIDE(data_size, num_ranks);
    segment_size = CEIL_DIVIDE(segment_size, reduce_stage_size) * reduce_stage_size;
    
    // Divide each workload into stages
    int num_stages = segment_size / reduce_stage_size;
    
    // Indexing
    auto data_stage_ptr = [&] (int segment_idx, int stage_idx)
    {
        return data_ptr +
               segment_idx * segment_size +
               (stage_idx % num_stages) * reduce_stage_size;
    };
    
    // Send to next rank, receive from previous rank
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));
    int dst_rank = (this_rank + 1) % num_ranks;
    int src_rank = (this_rank + num_ranks - 1) % num_ranks;
    
    // Get device IDs for ranks
    int dst_device = __fns(device_mask, 0, dst_rank + 1);
    int src_device = __fns(device_mask, 0, src_rank + 1);
    
    // Check P2P capabilities
    int can_access_dst, can_access_src;
    cudaDeviceCanAccessPeer(&can_access_dst, this_device, dst_device);
    cudaDeviceCanAccessPeer(&can_access_src, src_device, this_device);
    
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
                // Receive data from source device
                if (can_access_src)
                {
                    // Direct P2P access to source device memory
                    for (int i = stage_recv; i < stage_end && i < stage_recv + BATCH_STAGE; ++i)
                    {
                        float4* src = (float4*) data_stage_ptr(recv_seg, i);
                        float4* dst = (float4*) data_stage_ptr(recv_seg, i);
                        
                        if (dst + t < (float4*) data_end)
                        {
                            // First num_ranks - 1 iterations: accumulate
                            if (iter < num_ranks - 1)
                            {
                                float4 a = dst[t];
                                float4 b = src[t];
                                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                                dst[t] = a;
                            }
                            // Last num_ranks - 1 iterations: copy
                            else
                            {
                                dst[t] = src[t];
                            }
                        }
                    }
                    stage_recv = min(stage_recv + BATCH_STAGE, stage_end);
                }
                else
                {
                    // Fallback to traditional method
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "p2p_all_reduce (1)");
                    if (*abort_flag) break;
                }
            }
        }
        
        // Send
        if (dir == 1)
        {
            while (stage_send < stage_end)
            {
                if (can_access_dst)
                {
                    // Direct P2P access to destination device memory
                    for (int i = 0; i < BATCH_STAGE && stage_send < stage_end; ++i)
                    {
                        uint4* src = (uint4*) data_stage_ptr(send_seg, stage_send);
                        // For P2P, we need to enable peer access first
                        uint4* dst = (uint4*) data_ptr;  // This would be replaced with actual P2P access
                        
                        if (src + t < (uint4*) data_end) dst[t] = src[t];
                        stage_send++;
                    }
                }
                else
                {
                    // Fallback to traditional method
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "p2p_all_reduce (2)");
                    if (*abort_flag) break;
                }
            }
        }
        
        if (*abort_flag) break;
        grid.sync();
    }
    
    // Finished. Barrier to make sure all operations are complete
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
}

void p2p_all_reduce
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
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
    
    int threads = (int) CEIL_DIVIDE(CEIL_DIVIDE(data_size / 16ll, num_ranks), 32ll) * 32ll;
    threads = MIN(threads, MAX_NUM_THREADS);
    
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    void* kernelArgs[] =
    {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& master_device,
        (void*)& data_ptr,
        (void*)& data_size,
        (void*)& abort_flag_ptr
    };
    
    dim3 block_grid(2);
    dim3 block_dim(threads);
    
    cudaLaunchCooperativeKernel
    (
        (void*)p2p_all_reduce_kernel,
        block_grid,
        block_dim,
        kernelArgs,
        0,
        stream
    );
    
    cuda_check(cudaPeekAtLastError());
}

void p2p_all_reduce_ring
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
)
{
    // For now, just call the regular p2p_all_reduce
    // In a future implementation, this could be optimized for ring topologies
    p2p_all_reduce(ctx, devices, this_device, master_device, tensor, abort_flag);
}

void p2p_all_reduce_tree_adaptive
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag,
    float connectivity_ratio
)
{
    // Adaptive algorithm selection based on topology and tensor size
    size_t tensor_size = tensor.numel() * tensor.element_size();
    int algorithm = p2p_select_reduce_algorithm(devices, tensor_size, connectivity_ratio);
    
    switch (algorithm) {
        case 0:  // Binary tree
            p2p_all_reduce_tree(ctx, devices, this_device, master_device, tensor, abort_flag, 0);
            break;
        case 1:  // 4-ary tree
            p2p_all_reduce_tree(ctx, devices, this_device, master_device, tensor, abort_flag, 1);
            break;
        case 2:  // Ring fallback
            p2p_all_reduce_ring(ctx, devices, this_device, master_device, tensor, abort_flag);
            break;
        default:
            p2p_all_reduce(ctx, devices, this_device, master_device, tensor, abort_flag);
    }
}