#include <cuda_fp16.h>
#include "p2p_memory.cuh"
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

// Global memory pool structure
struct P2PMemoryPool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    bool initialized;
    
    // Simple free list for memory management
    struct FreeBlock {
        size_t offset;
        size_t size;
        FreeBlock* next;
    };
    
    FreeBlock* free_list;
    // Mutex for thread safety would be needed in a real implementation
};

// Device-specific memory pools
static P2PMemoryPool g_memory_pools[64];  // MAX_DEVICES is not defined here, using 64 as a reasonable max

// Enhanced memory pool for direct P2P access
struct P2PDirectMemoryPool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    bool initialized;
    bool peer_access_enabled[64];  // Track peer access status for each device
    
    // Pre-registered memory regions for fast access
    struct RegisteredRegion {
        void* ptr;
        size_t size;
        int peer_device;
        bool active;
    };
    
    std::vector<RegisteredRegion> registered_regions;
    std::vector<int> target_peers;  // Track which peers we're targeting for deadlock detection
    std::mutex pool_mutex;
};

static P2PDirectMemoryPool g_direct_memory_pools[64];

// Initialize memory pool for a device
void p2p_init_memory_pool(
    int device,
    size_t pool_size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= MAX_DEVICES) {
        return;
    }
    
    P2PMemoryPool& pool = g_memory_pools[device];
    
    if (pool.initialized) {
        return; // Already initialized
    }
    
    // Allocate memory pool on the device
    void* base_ptr;
    cudaError_t result = cudaMalloc(&base_ptr, pool_size);
    if (result != cudaSuccess) {
        // Set abort flag on failure
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    pool.base_ptr = base_ptr;
    pool.total_size = pool_size;
    pool.used_size = 0;
    pool.initialized = true;
    
    // Initialize free list with entire pool
    pool.free_list = new P2PMemoryPool::FreeBlock;
    pool.free_list->offset = 0;
    pool.free_list->size = pool_size;
    pool.free_list->next = nullptr;
}

// Cleanup memory pool for a device
void p2p_cleanup_memory_pool(
    int device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= MAX_DEVICES) {
        return;
    }
    
    P2PMemoryPool& pool = g_memory_pools[device];
    
    if (!pool.initialized) {
        return;
    }
    
    // Free all memory blocks in the free list
    P2PMemoryPool::FreeBlock* current = pool.free_list;
    while (current) {
        P2PMemoryPool::FreeBlock* next = current->next;
        delete current;
        current = next;
    }
    
    // Free the base memory
    cudaFree(pool.base_ptr);
    
    // Reset pool state
    pool.base_ptr = nullptr;
    pool.total_size = 0;
    pool.used_size = 0;
    pool.initialized = false;
    pool.free_list = nullptr;
}

// Allocate memory from the pool
void* p2p_allocate_from_pool(
    int device,
    size_t size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= MAX_DEVICES) {
        return nullptr;
    }
    
    P2PMemoryPool& pool = g_memory_pools[device];
    
    if (!pool.initialized) {
        return nullptr;
    }
    
    // Align size to 16 bytes
    size = (size + 15) & ~15;
    
    // Find a free block that fits
    P2PMemoryPool::FreeBlock* prev = nullptr;
    P2PMemoryPool::FreeBlock* current = pool.free_list;
    
    while (current) {
        if (current->size >= size) {
            // Found a block
            void* ptr = (void*)((char*)pool.base_ptr + current->offset);
            
            // If the block is larger than needed, split it
            if (current->size > size) {
                P2PMemoryPool::FreeBlock* new_block = new P2PMemoryPool::FreeBlock;
                new_block->offset = current->offset + size;
                new_block->size = current->size - size;
                new_block->next = current->next;
                
                // Update current block
                current->size = size;
                current->next = new_block;
            }
            
            // Remove block from free list
            if (prev) {
                prev->next = current->next;
            } else {
                pool.free_list = current->next;
            }
            
            delete current;
            pool.used_size += size;
            return ptr;
        }
        
        prev = current;
        current = current->next;
    }
    
    // No suitable block found
    return nullptr;
}

// Free memory back to the pool
void p2p_free_to_pool(
    int device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= MAX_DEVICES || !ptr) {
        return;
    }
    
    P2PMemoryPool& pool = g_memory_pools[device];
    
    if (!pool.initialized) {
        return;
    }
    
    // Align size to 16 bytes
    size = (size + 15) & ~15;
    
    // Calculate offset
    size_t offset = (char*)ptr - (char*)pool.base_ptr;
    
    // Create new free block
    P2PMemoryPool::FreeBlock* new_block = new P2PMemoryPool::FreeBlock;
    new_block->offset = offset;
    new_block->size = size;
    
    // Insert into free list (simple insertion at beginning)
    new_block->next = pool.free_list;
    pool.free_list = new_block;
    
    pool.used_size -= size;
    
    // In a real implementation, we would coalesce adjacent free blocks
}

// Get pointer to peer device memory
void* p2p_get_peer_device_ptr(
    int peer_device,
    at::Tensor& abort_flag
)
{
    if (peer_device < 0 || peer_device >= MAX_DEVICES) {
        return nullptr;
    }
    
    P2PMemoryPool& pool = g_memory_pools[peer_device];
    
    if (!pool.initialized) {
        return nullptr;
    }
    
    return pool.base_ptr;
}

// Check if P2P access is possible between devices
bool p2p_can_access_peer(
    int device,
    int peer_device,
    at::Tensor& abort_flag
)
{
    if (device < 0 || device >= MAX_DEVICES || peer_device < 0 || peer_device >= MAX_DEVICES) {
        return false;
    }
    
    int can_access;
    cudaError_t result = cudaDeviceCanAccessPeer(&can_access, device, peer_device);
    
    if (result != cudaSuccess) {
        return false;
    }
    
    return can_access != 0;
}

// Device barrier for P2P synchronization
void p2p_device_barrier(
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    
    // Simple implementation using device synchronization
    // In a real implementation, this would use more sophisticated P2P synchronization
    cudaDeviceSynchronize();
    
    // Additional synchronization could be added here for multi-device scenarios
}

// Enhanced direct memory pool functions
void p2p_init_direct_memory_pool(
    int device,
    size_t pool_size,
    std::vector<int> peer_devices,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= 64) {
        printf("DEBUG: Invalid device %d for direct memory pool initialization\n", device);
        return;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (pool.initialized) {
        printf("DEBUG: Direct memory pool for device %d already initialized\n", device);
        return; // Already initialized
    }
    
    printf("DEBUG: Initializing direct memory pool for device %d, size: %zu bytes\n", device, pool_size);
    
    // Allocate memory pool on the device
    void* base_ptr;
    cudaError_t result = cudaMalloc(&base_ptr, pool_size);
    if (result != cudaSuccess) {
        printf("ERROR: Failed to allocate %zu bytes for device %d: %s\n",
               pool_size, device, cudaGetErrorString(result));
        uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
        *abort_flag_ptr = 1;
        return;
    }
    
    pool.base_ptr = base_ptr;
    pool.total_size = pool_size;
    pool.used_size = 0;
    pool.initialized = true;
    
    // Initialize peer access status
    for (int i = 0; i < 64; i++) {
        pool.peer_access_enabled[i] = false;
    }
    
    // Clear target peers vector
    pool.target_peers.clear();
    
    printf("DEBUG: Memory pool allocated successfully for device %d\n", device);
    
    // Peer access will be handled by Python side
    printf("DEBUG: Skipping peer access enable (handled by Python side) for device %d\n", device);
}

void p2p_cleanup_direct_memory_pool(
    int device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= 64) {
        return;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return;
    }
    
    // Disable peer access
    for (int i = 0; i < 64; i++) {
        if (pool.peer_access_enabled[i]) {
            cudaDeviceDisablePeerAccess(i);
            pool.peer_access_enabled[i] = false;
        }
    }
    
    // Free the base memory
    cudaFree(pool.base_ptr);
    
    // Reset pool state
    pool.base_ptr = nullptr;
    pool.total_size = 0;
    pool.used_size = 0;
    pool.initialized = false;
    pool.registered_regions.clear();
    pool.target_peers.clear();
}

void* p2p_allocate_from_direct_pool(
    int device,
    size_t size,
    int peer_device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= 64) {
        return nullptr;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return nullptr;
    }
    
    // Check if peer access is enabled
    if (peer_device >= 0 && peer_device < 64 && peer_device != device) {
        if (!pool.peer_access_enabled[peer_device]) {
            // Peer access should have been enabled during initialization
            // Return nullptr if not available
            return nullptr;
        }
    }
    
    // Align size to 16 bytes
    size = (size + 15) & ~15;
    
    // Check if we have enough space
    if (pool.used_size + size > pool.total_size) {
        return nullptr;
    }
    
    // Allocate from the end of the pool (simple strategy)
    void* ptr = (void*)((char*)pool.base_ptr + pool.used_size);
    pool.used_size += size;
    
    // Register this region for the peer device
    if (peer_device >= 0 && peer_device < 64 && peer_device != device) {
        P2PDirectMemoryPool::RegisteredRegion region;
        region.ptr = ptr;
        region.size = size;
        region.peer_device = peer_device;
        region.active = true;
        pool.registered_regions.push_back(region);
    }
    
    return ptr;
}

void p2p_free_to_direct_pool(
    int device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    
    if (device < 0 || device >= 64 || !ptr) {
        return;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return;
    }
    
    // Remove from registered regions
    for (auto it = pool.registered_regions.begin(); it != pool.registered_regions.end(); ++it) {
        if (it->ptr == ptr) {
            pool.registered_regions.erase(it);
            break;
        }
    }
    
    // Simple free strategy - just reduce used size
    // In a more sophisticated implementation, we would manage fragmentation
    size = (size + 15) & ~15;
    if (pool.used_size >= size) {
        pool.used_size -= size;
    }
}

bool p2p_can_access_peer_direct(
    int device,
    int peer_device,
    at::Tensor& abort_flag
)
{
    if (device < 0 || device >= 64 || peer_device < 0 || peer_device >= 64) {
        return false;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    if (!pool.initialized) {
        return false;
    }
    
    // Check if peer access is enabled in our pool
    if (pool.peer_access_enabled[peer_device]) {
        return true;
    }
    
    // Check actual CUDA P2P status
    int can_access;
    cudaError_t result = cudaDeviceCanAccessPeer(&can_access, device, peer_device);
    
    if (result == cudaSuccess && can_access) {
        // P2P is technically possible but not enabled in our pool
        printf("Warning: P2P possible from %d to %d but not enabled in pool\n", device, peer_device);
        return false;
    }
    
    return false;
}

void p2p_register_peer_memory(
    int device,
    int peer_device,
    void* ptr,
    size_t size,
    at::Tensor& abort_flag
)
{
    if (device < 0 || device >= 64 || peer_device < 0 || peer_device >= 64 || !ptr) {
        return;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return;
    }
    
    // Check if already registered
    for (const auto& region : pool.registered_regions) {
        if (region.ptr == ptr && region.peer_device == peer_device) {
            return; // Already registered
        }
    }
    
    // Register the region
    P2PDirectMemoryPool::RegisteredRegion region;
    region.ptr = ptr;
    region.size = size;
    region.peer_device = peer_device;
    region.active = true;
    pool.registered_regions.push_back(region);
}

void p2p_unregister_peer_memory(
    int device,
    int peer_device,
    void* ptr,
    at::Tensor& abort_flag
)
{
    if (device < 0 || device >= 64 || !ptr) {
        return;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return;
    }
    
    // Remove from registered regions
    for (auto it = pool.registered_regions.begin(); it != pool.registered_regions.end(); ++it) {
        if (it->ptr == ptr && it->peer_device == peer_device) {
            pool.registered_regions.erase(it);
            break;
        }
    }
}

size_t p2p_get_direct_pool_usage(
    int device,
    at::Tensor& abort_flag
)
{
    if (device < 0 || device >= 64) {
        return 0;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return 0;
    }
    
    return pool.used_size;
}

size_t p2p_get_direct_pool_size(
    int device,
    at::Tensor& abort_flag
)
{
    if (device < 0 || device >= 64) {
        return 0;
    }
    
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    if (!pool.initialized) {
        return 0;
    }
    
    return pool.total_size;
}

// Centralized P2P access management with deadlock prevention
void p2p_enable_all_peer_access(
    int device,
    std::vector<int> peer_devices,
    at::Tensor& abort_flag
)
{
    printf("DEBUG: ENTERING p2p_enable_all_peer_access for device %d\n", device);
    
    printf("DEBUG: About to create CUDAGuard for device %d\n", device);
    const at::cuda::OptionalCUDAGuard device_guard(device);
    printf("DEBUG: CUDAGuard created for device %d\n", device);
    
    if (device < 0 || device >= 64) {
        printf("DEBUG: Device %d out of range (max 64)\n", device);
        return;
    }
    
    printf("DEBUG: About to get pool for device %d\n", device);
    P2PDirectMemoryPool& pool = g_direct_memory_pools[device];
    printf("DEBUG: Got pool for device %d\n", device);
    
    // Check if pool is initialized
    if (!pool.initialized) {
        printf("DEBUG: Pool for device %d not initialized\n", device);
        return;
    }
    
    printf("DEBUG: Skipping mutex lock (already held by caller) for device %d\n", device);
    
    printf("DEBUG: Enabling P2P for device %d, peer_devices count: %zu\n", device, peer_devices.size());
    printf("DEBUG: About to iterate through peer devices\n");
    
    // Store target peers for deadlock detection
    if (pool.target_peers.empty()) {
        pool.target_peers.reserve(peer_devices.size());
    }
    
    // Enable P2P access for all specified peer devices with simplified approach
    // Only enable peer access from lower device ID to higher device ID to avoid circular dependencies
    for (int peer_device : peer_devices) {
        if (peer_device < 0 || peer_device >= 64 || peer_device == device) {
            printf("DEBUG: Skipping peer device %d (invalid or same as device)\n", peer_device);
            continue;
        }
        
        printf("DEBUG: Attempting to enable P2P from device %d to peer %d (current state: %s)\n",
               device, peer_device, pool.peer_access_enabled[peer_device] ? "enabled" : "disabled");
        
        if (!pool.peer_access_enabled[peer_device]) {
            // Simplified deadlock prevention: only enable from lower ID to higher ID
            if (device > peer_device) {
                printf("DEBUG: Skipping P2P enable from %d to %d (will be handled by device %d)\n",
                       device, peer_device, peer_device);
                pool.peer_access_enabled[peer_device] = false;
                continue;
            }
            
            printf("DEBUG: Enabling P2P access from device %d to peer %d\n", device, peer_device);
            
            // First check if peer access is already enabled
            int can_access;
            cudaError_t check_result = cudaDeviceCanAccessPeer(&can_access, device, peer_device);
            
            if (check_result == cudaSuccess && can_access) {
                // P2P access is already available, no need to enable it again
                pool.peer_access_enabled[peer_device] = true;
                printf("DEBUG: P2P access already available from device %d to %d\n", device, peer_device);
                continue;
            }
            
            // Try to enable peer access
            cudaError_t result = cudaDeviceEnablePeerAccess(peer_device, 0);
            printf("DEBUG: cudaDeviceEnablePeerAccess(%d) result: %s\n",
                   peer_device, cudaGetErrorString(result));
            
            if (result == cudaSuccess || result == cudaErrorPeerAccessAlreadyEnabled) {
                pool.peer_access_enabled[peer_device] = true;
                printf("SUCCESS: Enabled P2P access from device %d to %d\n", device, peer_device);
            } else {
                // Log error and abort on P2P access failure
                printf("ERROR: Failed to enable P2P access from device %d to %d: %s\n",
                       device, peer_device, cudaGetErrorString(result));
                
                // Set abort flag on P2P access failure
                if (abort_flag.defined()) {
                    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
                    *abort_flag_ptr = 1;
                    printf("CRITICAL: Aborting due to P2P access failure\n");
                }
                pool.peer_access_enabled[peer_device] = false;
            }
        } else {
            printf("DEBUG: P2P already enabled from device %d to %d\n", device, peer_device);
        }
    }
    
    printf("DEBUG: P2P enable completed for device %d\n", device);
}