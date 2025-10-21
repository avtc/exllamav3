#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// Direct GPU-to-GPU memory copy functions
void p2p_copy_tensor_async(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    at::Tensor& abort_flag
);

void p2p_copy_tensor_sync(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    at::Tensor& abort_flag
);

void p2p_copy_tensor_batch(
    int src_device,
    int dst_device,
    std::vector<at::Tensor>& src_tensors,
    std::vector<at::Tensor>& dst_tensors,
    at::Tensor& abort_flag
);

void p2p_copy_tensor_pinned(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    at::Tensor& abort_flag
);

// Memory registration functions for P2P access
void p2p_register_memory_region(
    int device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
);

void p2p_unregister_memory_region(
    int device,
    void* ptr,
    at::Tensor& abort_flag
);

bool p2p_is_memory_registered(
    int device,
    void* ptr,
    at::Tensor& abort_flag
);

// Zero-copy memory operations
void* p2p_allocate_zero_copy(
    int device,
    size_t size,
    at::Tensor& abort_flag
);

void p2p_free_zero_copy(
    int device,
    void* ptr,
    at::Tensor& abort_flag
);

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
);

void p2p_copy_tensor_3d_async(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    cudaPitchedPtr src_pitched_ptr,
    cudaPitchedPtr dst_pitched_ptr,
    cudaExtent extent,
    at::Tensor& abort_flag
);

// Performance monitoring functions
float p2p_measure_bandwidth(
    int src_device,
    int dst_device,
    size_t size,
    int num_iterations,
    at::Tensor& abort_flag
);

float p2p_measure_latency(
    int src_device,
    int dst_device,
    size_t size,
    int num_iterations,
    at::Tensor& abort_flag
);

// Memory access validation
bool p2p_validate_memory_access(
    int src_device,
    int dst_device,
    void* src_ptr,
    void* dst_ptr,
    size_t size,
    at::Tensor& abort_flag
);

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
);

void p2p_copy_tensor_strided(
    int src_device,
    int dst_device,
    at::Tensor& src_tensor,
    at::Tensor& dst_tensor,
    std::vector<size_t> src_strides,
    std::vector<size_t> dst_strides,
    at::Tensor& abort_flag
);

// Synchronization functions for P2P operations
void p2p_synchronize_devices(
    std::vector<int> devices,
    at::Tensor& abort_flag
);

void p2p_enable_peer_access(
    int device,
    int peer_device,
    at::Tensor& abort_flag
);

void p2p_disable_peer_access(
    int device,
    int peer_device,
    at::Tensor& abort_flag
);

bool p2p_is_peer_access_enabled(
    int device,
    int peer_device,
    at::Tensor& abort_flag
);