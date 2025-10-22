#include <cuda_fp16.h>
#include "p2p_memory.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include <unordered_map>
#include <mutex>

// Global P2P context registry
static std::unordered_map<uintptr_t, P2PContext*> p2p_contexts;
static std::mutex p2p_mutex;

// P2P connectivity detection
bool detect_full_p2p_connectivity(const std::vector<int>& devices)
{
    for (size_t i = 0; i < devices.size(); ++i)
    {
        for (size_t j = 0; j < devices.size(); ++j)
        {
            if (i != j)
            {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]);
                if (!can_access)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

// P2P connection setup
void setup_p2p_connections(const std::vector<int>& devices)
{
    for (size_t i = 0; i < devices.size(); ++i)
    {
        for (size_t j = 0; j < devices.size(); ++j)
        {
            if (i != j)
            {
                const at::cuda::OptionalCUDAGuard device_guard(devices[i]);
                cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                {
                    cuda_check(err);
                }
            }
        }
    }
}

// P2P connection cleanup
void cleanup_p2p_connections(const std::vector<int>& devices)
{
    for (size_t i = 0; i < devices.size(); ++i)
    {
        for (size_t j = 0; j < devices.size(); ++j)
        {
            if (i != j)
            {
                const at::cuda::OptionalCUDAGuard device_guard(devices[i]);
                cudaError_t err = cudaDeviceDisablePeerAccess(devices[j]);
                if (err != cudaSuccess && err != cudaErrorPeerAccessNotEnabled)
                {
                    cuda_check(err);
                }
            }
        }
    }
}

// P2P buffer allocation
uintptr_t allocate_p2p_buffer(size_t size, int device)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    void* buffer = nullptr;
    cudaError_t err = cudaMalloc(&buffer, size);
    if (err != cudaSuccess)
    {
        cuda_check(err);
        return 0;
    }
    return reinterpret_cast<uintptr_t>(buffer);
}

// P2P buffer deallocation
void free_p2p_buffer(uintptr_t buffer, int device)
{
    if (buffer == 0) return;
    
    const at::cuda::OptionalCUDAGuard device_guard(device);
    void* ptr = reinterpret_cast<void*>(buffer);
    cuda_check(cudaFree(ptr));
}

// Get P2P buffer pointer for cross-device access
void* get_p2p_buffer_ptr(uintptr_t buffer, int src_device, int dst_device)
{
    // For P2P access, we return the same pointer since CUDA handles the mapping
    return reinterpret_cast<void*>(buffer);
}

// P2P memory copy operations
void p2p_memcpy_async(void* dst, const void* src, size_t count, int dst_device, int src_device, cudaStream_t stream)
{
    // Ensure we're on the correct device for the operation
    const at::cuda::OptionalCUDAGuard device_guard(dst_device);
    
    // Use cudaMemcpyPeerAsync for direct GPU-to-GPU copy
    cudaError_t err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
    if (err != cudaSuccess)
    {
        cuda_check(err);
    }
}

void p2p_memcpy(void* dst, const void* src, size_t count, int dst_device, int src_device)
{
    // Ensure we're on the correct device for the operation
    const at::cuda::OptionalCUDAGuard device_guard(dst_device);
    
    // Use cudaMemcpyPeer for synchronous direct GPU-to-GPU copy
    cudaError_t err = cudaMemcpyPeer(dst, dst_device, src, src_device, count);
    if (err != cudaSuccess)
    {
        cuda_check(err);
    }
}

// P2P memory synchronization
void p2p_sync(int device, cudaStream_t stream)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cuda_check(cudaStreamSynchronize(stream));
}

void p2p_barrier(const std::vector<int>& devices, int this_device, cudaStream_t stream)
{
    // Simple implementation using device events
    for (int device : devices)
    {
        if (device == this_device) continue;
        
        cudaEvent_t event;
        cuda_check(cudaEventCreate(&event));
        
        // Record event on this device
        const at::cuda::OptionalCUDAGuard guard(this_device);
        cuda_check(cudaEventRecord(event, stream));
        
        // Wait for event on other device
        const at::cuda::OptionalCUDAGuard other_guard(device);
        cuda_check(cudaStreamWaitEvent(stream, event, 0));
        
        cuda_check(cudaEventDestroy(event));
    }
}

// P2P context management
uintptr_t init_p2p_context(const std::vector<int>& devices, size_t buffer_size)
{
    std::lock_guard<std::mutex> lock(p2p_mutex);
    
    // Check if P2P connectivity is available
    if (!detect_full_p2p_connectivity(devices))
    {
        return 0;
    }
    
    // Setup P2P connections
    setup_p2p_connections(devices);
    
    // Create new P2P context
    P2PContext* ctx = new P2PContext(devices);
    
    // Allocate buffers for each device
    for (int device : devices)
    {
        uintptr_t buffer = allocate_p2p_buffer(buffer_size, device);
        if (buffer == 0)
        {
            // Cleanup on failure
            delete ctx;
            cleanup_p2p_connections(devices);
            return 0;
        }
        ctx->buffers.push_back(buffer);
    }
    
    // Mark P2P as enabled for all devices
    for (size_t i = 0; i < devices.size(); ++i)
    {
        ctx->p2p_enabled[i] = true;
    }
    
    uintptr_t ctx_ptr = reinterpret_cast<uintptr_t>(ctx);
    p2p_contexts[ctx_ptr] = ctx;
    
    return ctx_ptr;
}

void destroy_p2p_context(uintptr_t ctx_ptr)
{
    std::lock_guard<std::mutex> lock(p2p_mutex);
    
    auto it = p2p_contexts.find(ctx_ptr);
    if (it != p2p_contexts.end())
    {
        P2PContext* ctx = it->second;
        delete ctx;
        p2p_contexts.erase(it);
    }
}

// P2P validation and error checking
bool validate_p2p_connectivity(const std::vector<int>& devices)
{
    // Check if all devices support P2P
    for (int device : devices)
    {
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count));
        if (device < 0 || device >= device_count)
        {
            return false;
        }
    }
    
    // Check P2P connectivity
    return detect_full_p2p_connectivity(devices);
}

void check_p2p_error(int device, const char* operation)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "P2P error on device %d during %s: %s\n", 
                device, operation, cudaGetErrorString(err));
        cuda_check(err);
    }
}
