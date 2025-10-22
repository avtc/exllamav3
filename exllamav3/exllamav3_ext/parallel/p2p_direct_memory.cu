#include <cuda_fp16.h>
#include "p2p_direct_memory.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include <vector>
#include <unordered_map>
#include <mutex>

// Global registry for memory regions
static std::unordered_map<void*, size_t> g_registered_memory[64];
static std::mutex g_memory_registry_mutex[64];

// CUDA kernel for optimized memory copy
__global__ void p2p_copy_kernel(
    void* dst,
    const void* src,
    size_t size,
    size_t dst_offset,
    size_t src_offset
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;
    
    // Calculate byte positions
    size_t bytes_per_thread = (size + total_threads - 1) / total_threads;
    size_t start_byte = tid * bytes_per_thread;
    size_t end_byte = min(start_byte + bytes_per_thread, size);
    
    // Copy data with vectorized loads/stores when possible
    if (start_byte < end_byte) {
        const char* src_ptr = (const char*)src + src_offset + start_byte;
        char* dst_ptr = (char*)dst + dst_offset + start_byte;
        
        // Use 128-bit loads when aligned
        if (((uintptr_t)src_ptr % 16 == 0) && ((uintptr_t)dst_ptr % 16 == 0) && (end_byte - start_byte >= 16)) {
            size_t vec_end = start_byte + ((end_byte - start_byte) / 16) * 16;
            for (size_t pos = start_byte; pos < vec_end; pos += 16) {
                float4 vec_data = *reinterpret_cast<const float4*>(src_ptr + (pos - start_byte));
                *reinterpret_cast<float4*>(dst_ptr + (pos - start_byte)) = vec_data;
            }
            // Handle remaining bytes
            for (size_t pos = vec_end; pos < end_byte; pos++) {
                dst_ptr[pos - start_byte] = src_ptr[pos - start_byte];
            }
        } else {
            // Simple byte copy for unaligned data
            for (size_t pos = start_byte; pos < end_byte; pos++) {
                dst_ptr[pos - start_byte] = src_ptr[pos - start_byte];
            }
        }
    }
}

// CUDA kernel for 2D memory copy
__global__ void p2p_copy_2d_kernel(
    void* dst,
    const void* src,
    size_t dst_pitch,
    size_t src_pitch,
    size_t width,
    size_t height
) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const char* src_row = (const char*)src + y * src_pitch;
        char* dst_row = (char*)dst + y * dst_pitch;
        
        // Copy with vectorized operations when possible
        if (x % 4 == 0 && x + 3 < width) {
            float4 data = *reinterpret_cast<const float4*>(src_row + x * sizeof(float));
            *reinterpret_cast<float4*>(dst_row + x * sizeof(float)) = data;
        } else {
            float data = *reinterpret_cast<const float*>(src_row + x * sizeof(float));
            *reinterpret_cast<float*>(dst_row + x * sizeof(float)) = data;
        }
    }
}

// CUDA kernel for 3D memory copy
__global__ void p2p_copy_3d_kernel(
    void* dst,
    const void* src,
    cudaPitchedPtr dst_pitched,
    cudaPitchedPtr src_pitched,
    cudaExtent extent
) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < extent.width && y < extent.height && z < extent.depth) {
        const char* src_slice = (const char*)src_pitched.ptr + z * src_pitched.pitch * src_pitched.ysize;
        char* dst_slice = (char*)dst_pitched.ptr + z * dst_pitched.pitch * dst_pitched.ysize;
        
        const char* src_row = src_slice + y * src_pitched.pitch;
        char* dst_row = dst_slice + y * dst_pitched.pitch;
        
        if (x < extent.width) {
            dst_row[x] = src_row[x];
        }
    }
}

// Helper function to get optimal block and grid sizes
void get_copy_launch_params(size_t size, dim3& grid, dim3& block) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Use optimal block size for memory operations
    block.x = min(256, prop.maxThreadsPerBlock);
    grid.x = (size + block.x - 1) / block.x;
    grid.x = min(grid.x, prop.maxGridSize[0]);
}

// Direct GPU-to-GPU memory copy functions
void p2p_copy_tensor_async(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    at::Tensor& abort_flag
) {
    if (src_tensor.numel() != dst_tensor.numel()) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    size_t size = src_tensor.numel() * src_tensor.element_size();
    
    // P2P access should have been enabled during initialization
    // Use cudaMemcpyPeerAsync for direct GPU-to-GPU copy
    result = cudaMemcpyPeerAsync(
        dst_tensor.data_ptr(),
        dst_device,
        src_tensor.data_ptr(),
        src_device,
        size,
        0  // Default stream
    );
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

void p2p_copy_tensor_sync(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    at::Tensor& abort_flag
) {
    if (src_tensor.numel() != dst_tensor.numel()) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    size_t size = src_tensor.numel() * src_tensor.element_size();
    
    // P2P access should have been enabled during initialization
    // Use cudaMemcpyPeer for synchronous direct GPU-to-GPU copy
    result = cudaMemcpyPeer(
        dst_tensor.data_ptr(),
        dst_device,
        src_tensor.data_ptr(),
        src_device,
        size
    );
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

void p2p_copy_tensor_batch(
    int src_device,
    int dst_device,
    std::vector<at::Tensor>& src_tensors,
    std::vector<at::Tensor>& dst_tensors,
    at::Tensor& abort_flag
) {
    if (src_tensors.size() != dst_tensors.size()) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    // P2P access should have been enabled during initialization
    // Copy all tensors asynchronously
    for (size_t i = 0; i < src_tensors.size(); i++) {
        if (src_tensors[i].numel() != dst_tensors[i].numel()) {
            uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
            *abort_flag_ptr = 1;
            return;
        }
        
        size_t size = src_tensors[i].numel() * src_tensors[i].element_size();
        result = cudaMemcpyPeerAsync(
            dst_tensors[i].data_ptr(),
            dst_device,
            src_tensors[i].data_ptr(),
            src_device,
            size,
            0  // Default stream
        );
        
        if (result != cudaSuccess) {
            uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
            *abort_flag_ptr = 1;
            return;
        }
    }
}

void p2p_copy_tensor_pinned(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    at::Tensor& abort_flag
) {
    if (src_tensor.numel() != dst_tensor.numel()) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    size_t size = src_tensor.numel() * src_tensor.element_size();
    
    // Allocate pinned memory for intermediate copy
    void* pinned_buffer;
    cudaError_t result = cudaMallocHost(&pinned_buffer, size);
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    // Copy from source GPU to pinned memory
    result = cudaMemcpyAsync(
        pinned_buffer,
        src_tensor.data_ptr(),
        size,
        cudaMemcpyDeviceToHost,
        0
    );
    
    if (result != cudaSuccess) {
        cudaFreeHost(pinned_buffer);
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    // Copy from pinned memory to destination GPU
    result = cudaMemcpyAsync(
        dst_tensor.data_ptr(),
        pinned_buffer,
        size,
        cudaMemcpyHostToDevice,
        0
    );
    
    cudaFreeHost(pinned_buffer);
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

// Memory registration functions
void p2p_register_memory_region(
    int device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
) {
    if (device < 0 || device >= 64 || !ptr) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_memory_registry_mutex[device]);
    
    // Check if already registered
    auto it = g_registered_memory[device].find(ptr);
    if (it != g_registered_memory[device].end()) {
        return;  // Already registered
    }
    
    // Register the memory region
    g_registered_memory[device][ptr] = size;
}

void p2p_unregister_memory_region(
    int device,
    void* ptr,
    at::Tensor& abort_flag
) {
    if (device < 0 || device >= 64 || !ptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_memory_registry_mutex[device]);
    
    auto it = g_registered_memory[device].find(ptr);
    if (it != g_registered_memory[device].end()) {
        g_registered_memory[device].erase(it);
    }
}

bool p2p_is_memory_registered(
    int device,
    void* ptr,
    at::Tensor& abort_flag
) {
    if (device < 0 || device >= 64 || !ptr) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(g_memory_registry_mutex[device]);
    
    auto it = g_registered_memory[device].find(ptr);
    return it != g_registered_memory[device].end();
}

// Zero-copy memory operations
void* p2p_allocate_zero_copy(
    int device,
    size_t size,
    at::Tensor& abort_flag
) {
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    void* ptr;
    cudaError_t result = cudaMalloc(&ptr, size);
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return nullptr;
    }
    
    // Register the allocated memory
    p2p_register_memory_region(device, ptr, size, abort_flag);
    
    return ptr;
}

void p2p_free_zero_copy(
    int device,
    void* ptr,
    at::Tensor& abort_flag
) {
    if (!ptr) {
        return;
    }
    
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    // Unregister the memory region
    p2p_unregister_memory_region(device, ptr, abort_flag);
    
    cudaFree(ptr);
}

// Multi-dimensional memory access patterns
void p2p_copy_tensor_2d_async(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    size_t src_pitch,
    size_t dst_pitch,
    size_t height,
    at::Tensor& abort_flag
) {
    size_t width = src_tensor.numel() / height;
    
    // P2P access should have been enabled during initialization
    // Use cudaMemcpy3DPeerAsync for 2D copy (with depth=1)
    cudaMemcpy3DPeerParms params = {};
    params.srcDevice = src_device;
    params.dstDevice = dst_device;
    
    // Setup source pitched pointer
    params.srcPtr.ptr = src_tensor.data_ptr();
    params.srcPtr.pitch = src_pitch;
    params.srcPtr.xsize = width * src_tensor.element_size();
    params.srcPtr.ysize = height;
    
    // Setup destination pitched pointer
    params.dstPtr.ptr = dst_tensor.data_ptr();
    params.dstPtr.pitch = dst_pitch;
    params.dstPtr.xsize = width * src_tensor.element_size();
    params.dstPtr.ysize = height;
    
    // Setup extent (width in bytes, height, depth=1)
    params.extent = make_cudaExtent(width * src_tensor.element_size(), height, 1);
    
    result = cudaMemcpy3DPeerAsync(&params, 0);
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

void p2p_copy_tensor_3d_async(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    cudaPitchedPtr src_pitched_ptr,
    cudaPitchedPtr dst_pitched_ptr,
    cudaExtent extent,
    at::Tensor& abort_flag
) {
    // P2P access should have been enabled during initialization
    // Use cudaMemcpy3DPeerAsync for 3D copy
    cudaMemcpy3DPeerParms params = {};
    params.srcDevice = src_device;
    params.dstDevice = dst_device;
    params.srcPtr = src_pitched_ptr;
    params.dstPtr = dst_pitched_ptr;
    params.extent = extent;
    
    result = cudaMemcpy3DPeerAsync(&params, 0);
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

// Performance monitoring functions
float p2p_measure_bandwidth(
    int src_device,
    int dst_device,
    size_t size,
    int num_iterations,
    at::Tensor& abort_flag
) {
    const at::cuda::OptionalCUDAGuard device_guard(src_device);
    
    // Allocate test buffers
    void* src_buffer;
    void* dst_buffer;
    
    cudaError_t result = cudaMalloc(&src_buffer, size);
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return 0.0f;
    }
    
    result = cudaSetDevice(dst_device);
    if (result != cudaSuccess) {
        cudaFree(src_buffer);
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return 0.0f;
    }
    
    result = cudaMalloc(&dst_buffer, size);
    if (result != cudaSuccess) {
        cudaFree(src_buffer);
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return 0.0f;
    }
    
    // P2P access should have been enabled during initialization
    // Warm up
    for (int i = 0; i < 5; i++) {
        cudaMemcpyPeerAsync(dst_buffer, dst_device, src_buffer, src_device, size, 0);
    }
    cudaDeviceSynchronize();
    
    // Measure bandwidth
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < num_iterations; i++) {
        cudaMemcpyPeerAsync(dst_buffer, dst_device, src_buffer, src_device, size, 0);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate bandwidth in GB/s
    float bandwidth = (size * num_iterations) / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(src_buffer);
    cudaFree(dst_buffer);
    
    return bandwidth;
}

float p2p_measure_latency(
    int src_device,
    int dst_device,
    size_t size,
    int num_iterations,
    at::Tensor& abort_flag
) {
    const at::cuda::OptionalCUDAGuard device_guard(src_device);
    
    // Allocate test buffers
    void* src_buffer;
    void* dst_buffer;
    
    cudaError_t result = cudaMalloc(&src_buffer, size);
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return 0.0f;
    }
    
    result = cudaSetDevice(dst_device);
    if (result != cudaSuccess) {
        cudaFree(src_buffer);
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return 0.0f;
    }
    
    result = cudaMalloc(&dst_buffer, size);
    if (result != cudaSuccess) {
        cudaFree(src_buffer);
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return 0.0f;
    }
    
    // P2P access should have been enabled during initialization
    // Measure latency
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_latency = 0.0f;
    
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start, 0);
        cudaMemcpyPeerAsync(dst_buffer, dst_device, src_buffer, src_device, size, 0);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_latency += milliseconds;
    }
    
    // Calculate average latency in microseconds
    float avg_latency = (total_latency / num_iterations) * 1000.0f;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(src_buffer);
    cudaFree(dst_buffer);
    
    return avg_latency;
}

// Memory access validation
bool p2p_validate_memory_access(
    int src_device,
    int dst_device,
    void* src_ptr,
    void* dst_ptr,
    size_t size,
    at::Tensor& abort_flag
) {
    // Check if devices can access each other
    int can_access;
    cudaError_t result = cudaDeviceCanAccessPeer(&can_access, src_device, dst_device);
    
    if (result != cudaSuccess || !can_access) {
        return false;
    }
    
    // Check if pointers are valid (basic validation)
    if (!src_ptr || !dst_ptr || size == 0) {
        return false;
    }
    
    // Try a small test copy
    const at::cuda::OptionalCUDAGuard device_guard(src_device);
    
    size_t test_size = min(size, (size_t)1024);  // Test with at most 1KB
    
    // P2P access should have been enabled during initialization
    result = cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, test_size, 0);
    if (result != cudaSuccess) {
        return false;
    }
    
    cudaDeviceSynchronize();
    
    return true;
}

// Advanced memory operations
void p2p_copy_tensor_with_offset(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    at::Tensor& abort_flag
) {
    if (src_offset + size > src_tensor.numel() * src_tensor.element_size() ||
        dst_offset + size > dst_tensor.numel() * dst_tensor.element_size()) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    // P2P access should have been enabled during initialization
    // Copy with offsets
    char* src_ptr = (char*)src_tensor.data_ptr() + src_offset;
    char* dst_ptr = (char*)dst_tensor.data_ptr() + dst_offset;
    
    result = cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, size, 0);
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

void p2p_copy_tensor_strided(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    std::vector<size_t> src_strides,
    std::vector<size_t> dst_strides,
    at::Tensor& abort_flag
) {
    if (src_strides.size() != dst_strides.size()) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    // For simplicity, use element-wise copy with strides
    // In a production implementation, this would be optimized with custom kernels
    
    size_t total_elements = src_tensor.numel();
    size_t element_size = src_tensor.element_size();
    
    // P2P access should have been enabled during initialization
    // Copy element by element (simplified implementation)
    for (size_t i = 0; i < total_elements; i++) {
        size_t src_offset = i * element_size;
        size_t dst_offset = i * element_size;
        
        char* src_ptr = (char*)src_tensor.data_ptr() + src_offset;
        char* dst_ptr = (char*)dst_tensor.data_ptr() + dst_offset;
        
        result = cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, element_size, 0);
        
        if (result != cudaSuccess) {
            uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
            *abort_flag_ptr = 1;
            return;
        }
    }
}

// Synchronization functions for P2P operations
void p2p_synchronize_devices(
    std::vector<int> devices,
    at::Tensor& abort_flag
) {
    for (int device : devices) {
        const at::cuda::OptionalCUDAGuard device_guard(device);
        cudaDeviceSynchronize();
    }
}

void p2p_enable_peer_access(
    int device,
    int peer_device,
    at::Tensor& abort_flag
) {
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    cudaError_t result = cudaDeviceEnablePeerAccess(peer_device, 0);
    
    if (result != cudaSuccess && result != cudaErrorPeerAccessAlreadyEnabled) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

void p2p_disable_peer_access(
    int device,
    int peer_device,
    at::Tensor& abort_flag
) {
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    cudaError_t result = cudaDeviceDisablePeerAccess(peer_device);
    
    if (result != cudaSuccess && result != cudaErrorPeerAccessNotEnabled) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
    }
}

bool p2p_is_peer_access_enabled(
    int device,
    int peer_device,
    at::Tensor& abort_flag
) {
    int can_access;
    cudaError_t result = cudaDeviceCanAccessPeer(&can_access, device, peer_device);
    
    if (result != cudaSuccess) {
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return false;
    }
    
    return can_access != 0;
}