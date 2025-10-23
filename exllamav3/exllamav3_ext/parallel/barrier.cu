#include <cuda_fp16.h>
#include "barrier.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "timeout.cuh"

#include "barrier_inner.cuh"

__global__ void pg_barrier_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int coordinator_device,
    uint32_t* abort_flag
)
{
    pg_barrier_inner(ctx, device_mask, this_device, coordinator_device, abort_flag);
}

void pg_barrier
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);

    pg_barrier_kernel<<<1, 1, 0, stream>>>
    (
        (PGContext*) ctx,  // Shared, pinned
        device_mask,
        this_device,
        devices[0],
        (uint32_t*) abort_flag.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

// P2P-optimized barrier kernel for fully connected systems
#define NUM_THREADS_BARRIER 1024

__global__ __launch_bounds__(NUM_THREADS_BARRIER)
void pg_barrier_full_p2p_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    uint32_t* abort_flag,
    const int* peer_devices,
    int num_devices
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();

    __shared__ uint32_t local_epoch;
    __shared__ bool all_ready;
    
    // Initialize local epoch from global epoch
    if (t == 0)
    {
        local_epoch = ctx->barrier_epoch;
    }
    __syncthreads();
    
    // Each device increments its own epoch counter
    if (t == 0)
    {
        int this_rank = __popc(device_mask & ((1 << this_device) - 1));
        atomicAdd(&ctx->barrier_epoch_device[this_rank], 1);
    }
    __syncthreads();
    
    // Check if all devices have reached the same epoch
    uint32_t max_epoch = 0;
    for (int i = 0; i < num_devices; ++i)
    {
        int peer_rank = __popc(device_mask & ((1 << peer_devices[i]) - 1));
        uint32_t epoch = ctx->barrier_epoch_device[peer_rank];
        if (epoch > max_epoch)
            max_epoch = epoch;
    }
    
    // Synchronize all devices to the maximum epoch
    if (t == 0)
    {
        while (true)
        {
            uint32_t current_epoch = ctx->barrier_epoch;
            if (current_epoch >= max_epoch)
                break;
            
            // Wait loop with backoff
            uint32_t sleep = SYNC_MIN_SLEEP;
            __nanosleep(sleep);
            if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
            else *abort_flag = check_timeout(ctx, sync_deadline(), "barrier_p2p");
            if (*abort_flag) break;
        }
    }
    __syncthreads();
    
    // Reset local epoch counters for next barrier
    if (t == 0)
    {
        int this_rank = __popc(device_mask & ((1 << this_device) - 1));
        ctx->barrier_epoch_device[this_rank] = 0;
        __threadfence_system();
    }
    
    grid.sync();
}

void pg_barrier_full_p2p
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint32_t device_mask = 0;
    std::vector<int> peer_devices;
    for (int i : devices)
    {
        device_mask |= (1 << i);
        peer_devices.push_back(i);
    }

    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    int num_devices = peer_devices.size();
    void* kernelArgs[] =
    {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& abort_flag_ptr,
        (void*) peer_devices.data(),
        (void*)& num_devices
    };

    dim3 block_grid(1);
    dim3 block_dim(NUM_THREADS_BARRIER);

    cudaLaunchCooperativeKernel
    (
        (void*)pg_barrier_full_p2p_kernel,
        block_grid,
        block_dim,
        kernelArgs,
        0,
        stream
    );
    cuda_check(cudaPeekAtLastError());
}
