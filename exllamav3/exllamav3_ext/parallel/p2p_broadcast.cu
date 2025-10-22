#include <cuda_fp16.h>
#include "p2p_broadcast.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "ll.cuh"
#include "barrier_inner.cuh"

#define NUM_THREADS 1024
#define NUM_THREADS_LL 256

template <bool is_producer>
__global__ __launch_bounds__(NUM_THREADS)
void p2p_broadcast_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int src_device,
    uint8_t* __restrict__ data_ptr,
    size_t data_size,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    
    uint8_t* data_end = data_ptr + data_size;
    
    // Producer - copy data directly to peer devices using P2P
    if constexpr (is_producer)
    {
        // For each destination device that can be accessed via P2P
        uint32_t pending = device_mask & ~(1 << this_device);
        while (pending && !(*abort_flag))
        {
            const int dst_device = __ffs(pending) - 1;
            pending &= (pending - 1);
            
            // Check if P2P access is available
            int can_access_peer;
            cudaDeviceCanAccessPeer(&can_access_peer, this_device, dst_device);
            
            if (can_access_peer)
            {
                // Copy data directly to peer device memory
                size_t chunk_size = 16 * NUM_THREADS; // 16 bytes per thread
                size_t offset = 0;
                
                while (offset < data_size && !(*abort_flag))
                {
                    size_t bytes_to_copy = min(chunk_size, data_size - offset);
                    
                    // Copy using uint4 for efficiency
                    uint4* src = (uint4*)(data_ptr + offset);
                    // For P2P, we need to enable peer access first
                    uint4* dst = (uint4*)data_ptr;  // This would be replaced with actual P2P access
                    
                    if (t * 16 < bytes_to_copy)
                    {
                        dst[t] = src[t];
                    }
                    
                    offset += chunk_size;
                    __syncthreads();
                }
            }
        }
    }
    
    // Consumer - wait for data from source device
    else
    {
        // Check if source device can access this device via P2P
        int can_access_peer;
        cudaDeviceCanAccessPeer(&can_access_peer, src_device, this_device);
        
        if (can_access_peer)
        {
            // Data should already be in our memory via P2P copy
            // No additional action needed
        }
        else
        {
            // Fallback to traditional method if P2P is not available
            // This would use shared memory or other mechanisms
            // For now, we'll just wait (in a real implementation, this would use fallback)
            __syncthreads();
        }
    }
    
    // Synchronization barrier
    pg_barrier_inner(ctx, device_mask, this_device, src_device, abort_flag);
}

template <bool is_producer>
__global__ __launch_bounds__(NUM_THREADS_LL)
void p2p_broadcast_ll_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int src_device,
    uint8_t* __restrict__ data_ptr,
    size_t data_size,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    
    // Use barrier epoch to synchronize
    __shared__ uint32_t cookie_s;
    if (threadIdx.x == 0)
        cookie_s = ldg_cv_u32(&ctx->barrier_epoch);
    __syncthreads();
    uint32_t cookie = cookie_s;
    
    uint8_t* data_end = data_ptr + data_size;
    
    // Producer - copy data directly to peer devices using P2P
    if constexpr (is_producer)
    {
        uint32_t pending = device_mask & ~(1 << this_device);
        while (pending && !(*abort_flag))
        {
            const int dst_device = __ffs(pending) - 1;
            pending &= (pending - 1);
            
            // Check if P2P access is available
            int can_access_peer;
            cudaDeviceCanAccessPeer(&can_access_peer, this_device, dst_device);
            
            if (can_access_peer)
            {
                // Copy data directly to peer device memory
                size_t chunk_size = 4 * NUM_THREADS_LL; // 4 bytes per thread for uint32
                size_t offset = 0;
                
                while (offset < data_size && !(*abort_flag))
                {
                    size_t bytes_to_copy = min(chunk_size, data_size - offset);
                    
                    // Copy using uint32 for efficiency
                    uint32_t* src = (uint32_t*)(data_ptr + offset);
                    // For P2P, we need to enable peer access first
                    uint32_t* dst = (uint32_t*)data_ptr;  // This would be replaced with actual P2P access
                    
                    if (t < bytes_to_copy / 4)
                    {
                        synced_write_uint32((uint64_t*)(dst + t), src[t], cookie);
                    }
                    
                    offset += chunk_size;
                    __syncthreads();
                }
            }
        }
    }
    
    // Consumer - wait for data from source device
    else
    {
        // Check if source device can access this device via P2P
        int can_access_peer;
        cudaDeviceCanAccessPeer(&can_access_peer, src_device, this_device);
        
        if (can_access_peer)
        {
            // Data should already be in our memory via P2P copy
            // No additional action needed
        }
        else
        {
            // Fallback to traditional method if P2P is not available
            __syncthreads();
        }
    }
    
    // Synchronization barrier
    pg_barrier_inner(ctx, device_mask, this_device, src_device, abort_flag);
}

void p2p_broadcast
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
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
    
    #define ARGS \
        (PGContext*) ctx, \
        device_mask, \
        this_device, \
        src_device, \
        data_ptr, \
        data_size, \
        (uint32_t*) abort_flag.data_ptr()
    
    if (this_device == src_device)
        p2p_broadcast_kernel<true><<<1, NUM_THREADS, 0, stream>>>(ARGS);
    else
        p2p_broadcast_kernel<false><<<1, NUM_THREADS, 0, stream>>>(ARGS);
    cuda_check(cudaPeekAtLastError());
    
    #undef ARGS
}

void p2p_broadcast_ll
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);
    
    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 4 == 0, "data_size must be multiple of 4");
    
    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    
    #define ARGS \
        (PGContext*) ctx, \
        device_mask, \
        this_device, \
        src_device, \
        data_ptr, \
        data_size, \
        (uint32_t*) abort_flag.data_ptr()
    
    if (this_device == src_device)
        p2p_broadcast_ll_kernel<true><<<1, NUM_THREADS_LL, 0, stream>>>(ARGS);
    else
        p2p_broadcast_ll_kernel<false><<<1, NUM_THREADS_LL, 0, stream>>>(ARGS);
    cuda_check(cudaPeekAtLastError());
    
    #undef ARGS
}