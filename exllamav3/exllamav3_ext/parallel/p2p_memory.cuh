#pragma once

#include <ATen/Tensor.h>
#include <vector>

// P2P memory management utilities for fully connected systems

// P2P connectivity detection and setup
bool detect_full_p2p_connectivity(const std::vector<int>& devices);
void setup_p2p_connections(const std::vector<int>& devices);
void cleanup_p2p_connections(const std::vector<int>& devices);

// P2P memory allocation and management
uintptr_t allocate_p2p_buffer(size_t size, int device);
void free_p2p_buffer(uintptr_t buffer, int device);
void* get_p2p_buffer_ptr(uintptr_t buffer, int src_device, int dst_device);

// P2P memory copy operations
void p2p_memcpy_async(void* dst, const void* src, size_t count, int dst_device, int src_device, cudaStream_t stream);
void p2p_memcpy(void* dst, const void* src, size_t count, int dst_device, int src_device);

// P2P memory synchronization
void p2p_sync(int device, cudaStream_t stream);
void p2p_barrier(const std::vector<int>& devices, int this_device, cudaStream_t stream);

// P2P context management
struct P2PContext {
    std::vector<int> devices;
    std::vector<uintptr_t> buffers;
    std::vector<bool> p2p_enabled;
    int num_devices;
    
    P2PContext(const std::vector<int>& devs) : devices(devs), num_devices(devs.size()) {
        p2p_enabled.resize(num_devices, false);
    }
    
    ~P2PContext() {
        cleanup();
    }
    
    void cleanup() {
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i] != 0) {
                free_p2p_buffer(buffers[i], devices[i]);
            }
        }
        cleanup_p2p_connections(devices);
    }
};

uintptr_t init_p2p_context(const std::vector<int>& devices, size_t buffer_size);
void destroy_p2p_context(uintptr_t ctx_ptr);

// P2P error checking and validation
bool validate_p2p_connectivity(const std::vector<int>& devices);
void check_p2p_error(int device, const char* operation);
