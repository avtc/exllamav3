#include <cuda_fp16.h>
#include "p2p_gather.cuh"
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

#define NUM_THREADS 1024
#define STAGE_SIZE (NUM_THREADS * 16)

struct Offsets {
    int v[64 + 1];  // Using 64 as a reasonable max for devices
    __host__ __device__ int& operator[](int i)       { return v[i]; }
    __host__ __device__ int  operator[](int i) const { return v[i]; }
};

__global__ __launch_bounds__(NUM_THREADS)
void p2p_gather_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int out_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ out_data_ptr,
    Offsets all_offsets,
    int batch,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();
    
    // Reset synchronization flags for this device
    if (t == 0)
    {
        atomicExch(&ctx->gather_stage_produced[this_device], 0);
        atomicExch(&ctx->gather_stage_consumed[this_device], 0);
    }
    __syncthreads();
    
    uint8_t* data_end = data_ptr + (all_offsets[this_device + 1] - all_offsets[this_device]) * batch;
    
    // Divide shared buffer among ranks
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));
    
    const size_t stage_size = blockDim.x * sizeof(uint4);
    
    // Our slice
    int ldim = all_offsets[this_device + 1] - all_offsets[this_device];
    
    // Consumer - gather data from all devices
    bool is_consumer = this_device == out_device;
    if (is_consumer)
    {
        // For each source device
        uint32_t pending = device_mask;
        while (pending && !(*abort_flag))
        {
            const int src_device = __ffs(pending) - 1;
            pending &= (pending - 1);
            
            // Wait for producer to set data ready flag (except for our own device)
            if (src_device != this_device)
            {
                uint64_t deadline = sync_deadline();
                while (ldg_acquire_sys_u32(&ctx->gather_stage_produced[src_device]) == 0 && !(*abort_flag))
                {
                    if (check_timeout(ctx, deadline, "p2p_gather consumer")) break;
                    __nanosleep(SYNC_MIN_SLEEP);
                }
            }
            
            if (src_device == this_device)
            {
                // Copy our own data directly
                size_t bytes_to_copy = ldim * batch;
                size_t dst_offset = all_offsets[this_device];
                size_t stride = all_offsets[MAX_DEVICES];
                
                for (size_t offset = 0; offset < bytes_to_copy; offset += stage_size)
                {
                    size_t copy_t = offset + t * 16;
                    if (copy_t < bytes_to_copy)
                    {
                        size_t row = copy_t / ldim;
                        size_t col = copy_t % ldim;
                        size_t out_t = row * stride + col + dst_offset;
                        
                        uint4* src = (uint4*)(data_ptr + offset);
                        *((uint4*) (out_data_ptr + out_t)) = src[t];
                    }
                }
            }
            else
            {
                // Direct P2P access to source device memory
                // P2P access capability is pre-checked on host side
                size_t src_ldim = all_offsets[src_device + 1] - all_offsets[src_device];
                size_t bytes_to_recv = src_ldim * batch;
                size_t dst_offset = all_offsets[src_device];
                size_t stride = all_offsets[64];  // Using 64 as max devices
                
                // Check if P2P access is available for this source device
                bool p2p_available = (ctx->peer_device_ptrs[src_device] != nullptr);
                
                if (p2p_available)
                {
                    // P2P access is available - use direct pointer access
                    void* peer_base_ptr = ctx->peer_device_ptrs[src_device];
                    
                    for (size_t offset = 0; offset < bytes_to_recv; offset += stage_size)
                    {
                        size_t recv_t = offset + t * 16;
                        if (recv_t < bytes_to_recv)
                        {
                            size_t row = recv_t / src_ldim;
                            size_t col = recv_t % src_ldim;
                            size_t out_t = row * stride + col + dst_offset;
                            
                            // Access peer device memory directly
                            // Note: This requires peer access to be enabled on host side
                            uint4* src = (uint4*)((uint8_t*)peer_base_ptr + offset);
                            
                            // Ensure memory coherency with appropriate memory fence
                            __threadfence();
                            
                            *((uint4*) (out_data_ptr + out_t)) = src[t];
                        }
                    }
                }
                else
                {
                    // Fallback: P2P access not available, use cudaMemcpyPeerAsync
                    // This is slower but provides compatibility when P2P is not available
                    
                    // Use shared memory for temporary storage to improve performance
                    __shared__ uint4 temp_buffer[NUM_THREADS];
                    
                    for (size_t offset = 0; offset < bytes_to_recv; offset += stage_size)
                    {
                        size_t recv_t = offset + t * 16;
                        if (recv_t < bytes_to_recv)
                        {
                            size_t row = recv_t / src_ldim;
                            size_t col = recv_t % src_ldim;
                            size_t out_t = row * stride + col + dst_offset;
                            
                            // For threads that need to copy data
                            if (t == 0)
                            {
                                // Use cudaMemcpyPeerAsync for the chunk
                                size_t chunk_size = min(stage_size, bytes_to_recv - offset);
                                cudaError_t result = cudaMemcpyPeerAsync(
                                    out_data_ptr + out_t - col,  // Destination with offset
                                    out_device,                  // Destination device
                                    data_ptr + offset,           // Source with offset
                                    src_device,                  // Source device
                                    chunk_size,
                                    0  // Default stream
                                );
                                
                                if (result != cudaSuccess)
                                {
                                    // Set abort flag on failure
                                    *abort_flag = 1;
                                }
                            }
                            
                            // Wait for the async copy to complete before proceeding
                            __syncthreads();
                            
                            // Now copy from the temporary location to the final destination
                            if (recv_t < bytes_to_recv)
                            {
                                uint4* src = (uint4*)(out_data_ptr + out_t - col);
                                
                                // Ensure memory coherency
                                __threadfence();
                                
                                *((uint4*) (out_data_ptr + out_t)) = src[t];
                            }
                        }
                    }
                }
            }
            
            // Signal that data from this source has been consumed
            if (src_device != this_device)
            {
                __threadfence();
                stg_release_sys_u32(&ctx->gather_stage_consumed[src_device], 1);
            }
        }
    }
    
    // Producer - set data ready flag for this device
    if (!is_consumer)
    {
        // Ensure all memory writes are visible before setting the flag
        __threadfence();
        
        // Set the produced flag for this device using release semantics
        stg_release_sys_u32(&ctx->gather_stage_produced[this_device], 1);
        
        // Wait for consumer to acknowledge data consumption
        uint64_t deadline = sync_deadline();
        while (ldg_acquire_sys_u32(&ctx->gather_stage_consumed[this_device]) == 0 && !(*abort_flag))
        {
            if (check_timeout(ctx, deadline, "p2p_gather producer")) break;
            __nanosleep(SYNC_MIN_SLEEP);
        }
    }
    
    // Synchronization barrier
    pg_barrier_inner(ctx, device_mask, this_device, out_device, abort_flag);
}

void p2p_gather
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);
    
    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* out_data_ptr = (uint8_t*) OPTPTR(out_tensor);
    
    size_t esize = tensor.element_size();
    size_t send_ldim = tensor.size(-1) * esize;
    TORCH_CHECK(send_ldim % 128 == 0, "send_ldim must be multiple of 128");
    TORCH_CHECK(devices.size() == ldims.size(), "Must have one ldim per active device");
    int batch = out_data_ptr ? out_tensor.value().numel() / out_tensor.value().size(-1)
                             : tensor.numel() / tensor.size(-1);
    
    Offsets all_offsets = {};
    for (int i = 0; i < 64 + 1; ++i) all_offsets[i] = 0;
    for (int i = 0; i < devices.size(); ++i) all_offsets[devices[i]] = ldims[i] * esize;
    int p = 0;
    for (int i = 0; i < 64 + 1; ++i) { int q = p; p += all_offsets[i]; all_offsets[i] = q; }
    if (out_data_ptr)
        TORCH_CHECK(p == out_tensor.value().size(-1) * esize, "Gather: Output tensor last dimension mismatch");
    
    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    void* kernelArgs[] =
    {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& out_device,
        (void*)& data_ptr,
        (void*)& out_data_ptr,
        (void*)& all_offsets,
        (void*)& batch,
        (void*)& abort_flag_ptr
    };
    
    dim3 block_grid(1);
    dim3 block_dim(NUM_THREADS);
    
    cudaLaunchCooperativeKernel
    (
        (void*)p2p_gather_kernel,
        block_grid,
        block_dim,
        kernelArgs,
        0,
        stream
    );
    cuda_check(cudaPeekAtLastError());
}

void p2p_gather_direct
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    at::Tensor& abort_flag
)
{
    // For now, just call the regular p2p_gather
    // In a future implementation, this could be optimized for direct P2P access
    p2p_gather(ctx, devices, this_device, out_device, tensor, out_tensor, ldims, abort_flag);
}