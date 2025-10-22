#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "all_reduce.cuh"
#include "broadcast.cuh"
#include "gather.cuh"
#include "barrier.cuh"
#include "p2p_memory.cuh"
#include "context.cuh"
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "timeout.cuh"

// Test function for P2P connectivity detection
bool test_p2p_connectivity(const std::vector<int>& devices) {
    return validate_p2p_connectivity(devices);
}

// Test function for P2P all_reduce
torch::Tensor test_p2p_all_reduce(torch::Tensor tensor, const std::vector<int>& devices) {
    // Check P2P connectivity first
    if (!test_p2p_connectivity(devices)) {
        throw std::runtime_error("Not all devices are P2P connected");
    }
    
    int this_device = tensor.device().index();
    auto abort_flag = torch::zeros({1}, torch::kInt32).to(tensor.device());
    
    // Create context
    size_t shbuf_size = 1024 * 1024; // 1MB buffer
    void* shbuf;
    cudaMalloc(&shbuf, shbuf_size);
    
    // Create PG context
    void* pg_ctx;
    cudaMallocHost(&pg_ctx, sizeof(PGContext));
    
    // Initialize context
    pg_init_context(reinterpret_cast<uintptr_t>(pg_ctx));
    
    std::vector<uintptr_t> device_ptrs(devices.begin(), devices.end());
    
    // Call P2P all_reduce
    pg_all_reduce_full_p2p(
        reinterpret_cast<uintptr_t>(pg_ctx),
        device_ptrs,
        this_device,
        devices[0],  // master device
        tensor,
        reinterpret_cast<uintptr_t>(shbuf),
        shbuf_size,
        abort_flag
    );
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(shbuf);
    cudaFreeHost(pg_ctx);
    
    return tensor;
}

// Test function for P2P broadcast
torch::Tensor test_p2p_broadcast(torch::Tensor tensor, const std::vector<int>& devices, int src_device) {
    // Check P2P connectivity first
    if (!test_p2p_connectivity(devices)) {
        throw std::runtime_error("Not all devices are P2P connected");
    }
    
    int this_device = tensor.device().index();
    auto abort_flag = torch::zeros({1}, torch::kInt32).to(tensor.device());
    
    // Create context
    size_t shbuf_size = 1024 * 1024; // 1MB buffer
    void* shbuf;
    cudaMalloc(&shbuf, shbuf_size);
    
    // Create PG context
    void* pg_ctx;
    cudaMallocHost(&pg_ctx, sizeof(PGContext));
    
    // Initialize context
    pg_init_context(reinterpret_cast<uintptr_t>(pg_ctx));
    
    std::vector<uintptr_t> device_ptrs(devices.begin(), devices.end());
    
    // Call P2P broadcast
    pg_broadcast_full_p2p(
        reinterpret_cast<uintptr_t>(pg_ctx),
        device_ptrs,
        this_device,
        src_device,
        tensor,
        reinterpret_cast<uintptr_t>(shbuf),
        shbuf_size,
        abort_flag
    );
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(shbuf);
    cudaFreeHost(pg_ctx);
    
    return tensor;
}

// Test function for P2P gather
torch::Tensor test_p2p_gather(torch::Tensor tensor, const std::vector<int>& devices, int out_device) {
    // Check P2P connectivity first
    if (!test_p2p_connectivity(devices)) {
        throw std::runtime_error("Not all devices are P2P connected");
    }
    
    int this_device = tensor.device().index();
    auto abort_flag = torch::zeros({1}, torch::kInt32).to(tensor.device());
    
    // Calculate output dimensions
    std::vector<size_t> ldims;
    for (int device : devices) {
        ldims.push_back(tensor.size(-1));
    }
    
    // Create output tensor
    std::vector<int64_t> out_sizes = tensor.sizes().vec();
    out_sizes.back() *= devices.size();
    auto out_tensor = torch::empty(out_sizes, tensor.options());
    
    // Create context
    size_t shbuf_size = 1024 * 1024; // 1MB buffer
    void* shbuf;
    cudaMalloc(&shbuf, shbuf_size);
    
    // Create PG context
    void* pg_ctx;
    cudaMallocHost(&pg_ctx, sizeof(PGContext));
    
    // Initialize context
    pg_init_context(reinterpret_cast<uintptr_t>(pg_ctx));
    
    std::vector<uintptr_t> device_ptrs(devices.begin(), devices.end());
    
    // Call P2P gather
    c10::optional<at::Tensor> out_tensor_optional(out_tensor);
    pg_gather_full_p2p(
        reinterpret_cast<uintptr_t>(pg_ctx),
        device_ptrs,
        this_device,
        out_device,
        tensor,
        out_tensor_optional,
        ldims,
        reinterpret_cast<uintptr_t>(shbuf),
        shbuf_size,
        abort_flag
    );
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(shbuf);
    cudaFreeHost(pg_ctx);
    
    return out_tensor;
}

// Test function for P2P barrier
void test_p2p_barrier(const std::vector<int>& devices) {
    // Check P2P connectivity first
    if (!test_p2p_connectivity(devices)) {
        throw std::runtime_error("Not all devices are P2P connected");
    }
    
    int this_device;
    cudaGetDevice(&this_device);
    auto abort_flag = torch::zeros({1}, torch::kInt32).to(torch::kCUDA);
    
    // Create PG context
    void* pg_ctx;
    cudaMallocHost(&pg_ctx, sizeof(PGContext));
    
    // Initialize context
    pg_init_context(reinterpret_cast<uintptr_t>(pg_ctx));
    
    std::vector<uintptr_t> device_ptrs(devices.begin(), devices.end());
    
    // Call P2P barrier
    pg_barrier_full_p2p(
        reinterpret_cast<uintptr_t>(pg_ctx),
        device_ptrs,
        this_device,
        abort_flag
    );
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFreeHost(pg_ctx);
}

// Bind test functions
PYBIND11_MODULE(p2p_test, m) {
    m.def("test_p2p_connectivity", &test_p2p_connectivity, "Test P2P connectivity");
    m.def("test_p2p_all_reduce", &test_p2p_all_reduce, "Test P2P all_reduce");
    m.def("test_p2p_broadcast", &test_p2p_broadcast, "Test P2P broadcast");
    m.def("test_p2p_gather", &test_p2p_gather, "Test P2P gather");
    m.def("test_p2p_barrier", &test_p2p_barrier, "Test P2P barrier");
}
