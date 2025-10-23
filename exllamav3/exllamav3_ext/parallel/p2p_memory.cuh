#pragma once

#include <ATen/Tensor.h>
#include <vector>

// P2P memory pool management functions
void p2p_init_memory_pool(
    int device,
    size_t pool_size,
    at::Tensor& abort_flag
);

void p2p_cleanup_memory_pool(
    int device,
    at::Tensor& abort_flag
);

void* p2p_allocate_from_pool(
    int device,
    size_t size,
    at::Tensor& abort_flag
);

void p2p_free_to_pool(
    int device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
);

// P2P memory access functions
void* p2p_get_peer_device_ptr(
    int peer_device,
    at::Tensor& abort_flag
);


// P2P synchronization functions
void p2p_device_barrier(
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
);

// Enhanced direct memory pool functions
void p2p_init_direct_memory_pool(
    int device,
    size_t pool_size,
    std::vector<int> peer_devices,
    at::Tensor& abort_flag
);

void p2p_cleanup_direct_memory_pool(
    int device,
    at::Tensor& abort_flag
);

void* p2p_allocate_from_direct_pool(
    int device,
    size_t size,
    int peer_device,
    at::Tensor& abort_flag
);

void p2p_free_to_direct_pool(
    int device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
);

bool p2p_can_access_peer_direct(
    int device,
    int peer_device,
    at::Tensor& abort_flag
);

void p2p_register_peer_memory(
    int device,
    int peer_device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
);

void p2p_unregister_peer_memory(
    int device,
    int peer_device,
    void* ptr,
    at::Tensor& abort_flag
);

size_t p2p_get_direct_pool_usage(
    int device,
    at::Tensor& abort_flag
);

size_t p2p_get_direct_pool_size(
    int device,
    at::Tensor& abort_flag
);
