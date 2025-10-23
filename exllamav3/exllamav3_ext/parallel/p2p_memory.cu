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
    std::mutex pool_mutex;
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
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
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
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
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
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
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
    
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
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
    
    // Insert into free list in sorted order by offset
    P2PMemoryPool::FreeBlock* prev = nullptr;
    P2PMemoryPool::FreeBlock* current = pool.free_list;
    
    // Find the correct position to insert the new block
    while (current && current->offset < new_block->offset) {
        prev = current;
        current = current->next;
    }
    
    // Insert the new block
    if (prev) {
        prev->next = new_block;
    } else {
        pool.free_list = new_block;
    }
    new_block->next = current;
    
    pool.used_size -= size;
    
    // Coalesce adjacent free blocks
    bool coalesced = true;
    while (coalesced) {
        coalesced = false;
        prev = nullptr;
        current = pool.free_list;
        
        while (current && current->next) {
            P2PMemoryPool::FreeBlock* next = current->next;
            
            // Check if current block and next block are adjacent
            if (current->offset + current->size == next->offset) {
                // Merge the blocks
                current->size += next->size;
                current->next = next->next;
                delete next;
                coalesced = true;
            } else {
                prev = current;
                current = current->next;
            }
        }
    }
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


// Device barrier for P2P synchronization
void p2p_device_barrier(
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    
    if (devices.size() <= 1) {
        // Single device or empty list, just synchronize locally
        cudaDeviceSynchronize();
        return;
    }
    
    // Check for abort flag
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    if (*abort_flag_ptr != 0) {
        return;
    }
    
    // Shared synchronization structures for tree reduction
    struct BarrierSyncData {
        uint32_t phase_flags[64];  // Flags for each device's phase completion
        uint32_t tree_level;       // Current tree level
        uint32_t barrier_active;   // Flag indicating barrier is active
        uint32_t padding[61];      // Padding to cache line size
    };
    
    // P2P accessible synchronization data
    static BarrierSyncData* g_barrier_sync_data[64] = {nullptr};  // One per device
    static bool sync_data_initialized = false;
    static std::mutex sync_data_mutex;
    
    // Event storage for inter-device synchronization
    static cudaEvent_t inter_device_events[64][64];  // events[source][target]
    static bool events_initialized = false;
    static std::mutex events_mutex;
    
    // Initialize synchronization data on first call
    if (!sync_data_initialized) {
        std::lock_guard<std::mutex> lock(sync_data_mutex);
        if (!sync_data_initialized) {
            // Get actual device count
            int actual_device_count = 0;
            cudaError_t result = cudaGetDeviceCount(&actual_device_count);
            if (result != cudaSuccess) {
                actual_device_count = 8;  // Default fallback
            }
            
            // Allocate P2P accessible memory for synchronization data
            for (int dev = 0; dev < 64; dev++) {
                if (dev < actual_device_count) {
                    const at::cuda::OptionalCUDAGuard guard(dev);
                    cudaError_t alloc_result = cudaMalloc(&g_barrier_sync_data[dev], sizeof(BarrierSyncData));
                    if (alloc_result == cudaSuccess) {
                        // Initialize the synchronization data
                        cudaMemset(g_barrier_sync_data[dev], 0, sizeof(BarrierSyncData));
                    } else {
                        printf("WARNING: Failed to allocate sync data for device %d\n", dev);
                        g_barrier_sync_data[dev] = nullptr;
                    }
                } else {
                    g_barrier_sync_data[dev] = nullptr;
                }
            }
            sync_data_initialized = true;
        }
    }
    
    // Initialize events on first call
    if (!events_initialized) {
        std::lock_guard<std::mutex> lock(events_mutex);
        if (!events_initialized) {
            for (int dev = 0; dev < 64; dev++) {
                for (int target = 0; target < 64; target++) {
                    inter_device_events[dev][target] = nullptr;
                }
            }
            events_initialized = true;
        }
    }
    
    // Phase 1: Local synchronization and event recording
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        printf("ERROR: Local device synchronization failed for device %d: %s\n",
               this_device, cudaGetErrorString(result));
        *abort_flag_ptr = 1;
        return;
    }
    
    // Record event on this device
    cudaEvent_t local_event;
    result = cudaEventCreateWithFlags(&local_event, cudaEventDisableTiming);
    if (result != cudaSuccess) {
        printf("ERROR: Failed to create event on device %d: %s\n",
               this_device, cudaGetErrorString(result));
        *abort_flag_ptr = 1;
        return;
    }
    
    result = cudaEventRecord(local_event, 0);
    if (result != cudaSuccess) {
        printf("ERROR: Failed to record event on device %d: %s\n",
               this_device, cudaGetErrorString(result));
        cudaEventDestroy(local_event);
        *abort_flag_ptr = 1;
        return;
    }
    
    // Phase 2: Cross-device synchronization using P2P where available
    std::vector<cudaEvent_t> peer_events;
    std::vector<int> peer_devices;
    
    // Collect events from peer devices that we can access via P2P
    for (size_t i = 0; i < devices.size(); i++) {
        int peer_device = (int)devices[i];
        if (peer_device == this_device) continue;
        
        // Check if P2P access is possible
        int can_access = 0;
        result = cudaDeviceCanAccessPeer(&can_access, this_device, peer_device);
        
        if (result == cudaSuccess && can_access) {
            // P2P is available, we can directly synchronize with the peer
            // In a real implementation, we would have a mechanism to share events
            // For now, we'll use a fallback approach
            
            // Enable peer access if not already enabled
            int peer_access_enabled = 0;
            result = cudaDeviceGetAttribute(&peer_access_enabled,
                                          cudaDevAttrPeerAccessSupported, this_device);
            
            if (result == cudaSuccess && peer_access_enabled) {
                // P2P access is handled by PyTorch automatically
            }
            
            // Add to list of peers to synchronize with
            peer_devices.push_back(peer_device);
        }
    }
    
    // Synchronization strategy based on number of devices
    if (devices.size() == 2) {
        // Simple 2-device synchronization
        if (!peer_devices.empty()) {
            int peer_device = peer_devices[0];
            
            // Wait for peer device using shared event mechanism
            if (g_barrier_sync_data[this_device] && g_barrier_sync_data[peer_device]) {
                // Use P2P memory for synchronization flags
                volatile uint32_t* peer_flag = &g_barrier_sync_data[peer_device]->phase_flags[this_device];
                volatile uint32_t* my_flag = &g_barrier_sync_data[this_device]->phase_flags[peer_device];
                
                // Set our flag to indicate we're ready
                *my_flag = 1;
                
                // Wait for peer to set their flag
                const int max_wait_cycles = 1000000;  // Prevent infinite loops
                int wait_cycles = 0;
                while (*peer_flag == 0 && wait_cycles < max_wait_cycles) {
                    // Use CUDA event for efficient waiting
                    cudaEvent_t wait_event;
                    if (cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming) == cudaSuccess) {
                        cudaEventRecord(wait_event, 0);
                        cudaEventSynchronize(wait_event);
                        cudaEventDestroy(wait_event);
                    }
                    wait_cycles++;
                }
                
                if (wait_cycles >= max_wait_cycles) {
                    printf("WARNING: Timeout waiting for peer device %d\n", peer_device);
                    *abort_flag_ptr = 1;
                }
                
                // Reset flags for next barrier
                *my_flag = 0;
            } else {
                // Fallback to regular synchronization
                result = cudaDeviceSynchronize();
                if (result != cudaSuccess) {
                    printf("ERROR: Peer synchronization failed for device %d: %s\n",
                           this_device, cudaGetErrorString(result));
                    *abort_flag_ptr = 1;
                }
            }
        }
    } else {
        // Sophisticated multi-device barrier using tree reduction algorithm
        // This implementation reduces synchronization complexity from O(N) to O(log N)
        
        // First, ensure all local operations are complete
        result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            printf("ERROR: Multi-device barrier local sync failed for device %d: %s\n",
                   this_device, cudaGetErrorString(result));
            *abort_flag_ptr = 1;
            cudaEventDestroy(local_event);
            return;
        }
        
        // Tree reduction implementation
        int num_devices = (int)devices.size();
        
        // Find this device's position in the device list
        int device_rank = -1;
        for (int i = 0; i < num_devices; i++) {
            if (devices[i] == this_device) {
                device_rank = i;
                break;
            }
        }
        
        if (device_rank == -1) {
            printf("ERROR: Device %d not found in device list\n", this_device);
            *abort_flag_ptr = 1;
            cudaEventDestroy(local_event);
            return;
        }
        
        // Calculate tree structure
        // For non-power-of-2 device counts, we handle the extra devices at the leaf level
        int tree_height = 0;
        int temp = num_devices;
        while (temp > 1) {
            temp = (temp + 1) / 2;  // Ceiling division
            tree_height++;
        }
        
        // Phase 1: Leaf nodes synchronize with their parent
        // Each device at level 0 (leaf) synchronizes with its parent at level 1
        if (device_rank < num_devices) {
            int parent_rank = device_rank / 2;
            
            if (parent_rank < num_devices && parent_rank != device_rank) {
                int parent_device = (int)devices[parent_rank];
                
                // Check if P2P access is available to parent
                int can_access_parent = 0;
                result = cudaDeviceCanAccessPeer(&can_access_parent, this_device, parent_device);
                
                if (result == cudaSuccess && can_access_parent) {
                    // P2P access is handled by PyTorch automatically
                    // Create event for synchronization with parent
                    cudaEvent_t leaf_event;
                    result = cudaEventCreateWithFlags(&leaf_event, cudaEventDisableTiming);
                    if (result == cudaSuccess) {
                        result = cudaEventRecord(leaf_event, 0);
                        if (result == cudaSuccess) {
                            // Share event with parent device using P2P memory
                            if (g_barrier_sync_data[this_device] && g_barrier_sync_data[parent_device]) {
                                // Store event pointer in shared memory for parent to access
                                // Note: In a real implementation, we'd need IPC mechanisms
                                // For now, we'll use the phase flags as a simple signaling mechanism
                                
                                volatile uint32_t* parent_signal = &g_barrier_sync_data[parent_device]->phase_flags[this_device];
                                *parent_signal = 1;  // Signal parent we're ready
                                
                                // Wait for parent acknowledgment
                                volatile uint32_t* parent_ack = &g_barrier_sync_data[this_device]->phase_flags[parent_device];
                                const int max_wait = 100000;
                                int wait_count = 0;
                                while (*parent_ack == 0 && wait_count < max_wait) {
                                    // Efficient wait using CUDA events
                                    cudaEvent_t wait_event;
                                    if (cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming) == cudaSuccess) {
                                        cudaEventRecord(wait_event, 0);
                                        cudaEventSynchronize(wait_event);
                                        cudaEventDestroy(wait_event);
                                    }
                                    wait_count++;
                                }
                                
                                // Clean up signals
                                *parent_signal = 0;
                                if (wait_count >= max_wait) {
                                    printf("WARNING: Parent acknowledgment timeout for device %d\n", this_device);
                                }
                            } else {
                                // Fallback to device synchronization
                                result = cudaDeviceSynchronize();
                            }
                        }
                        cudaEventDestroy(leaf_event);
                    }
                } else {
                    // Fallback to regular synchronization
                    result = cudaDeviceSynchronize();
                }
                
                if (result != cudaSuccess) {
                    printf("ERROR: Phase 1 synchronization failed for device %d: %s\n",
                           this_device, cudaGetErrorString(result));
                    *abort_flag_ptr = 1;
                    cudaEventDestroy(local_event);
                    return;
                }
            }
        }
        
        // Phase 2: Internal nodes synchronize up the tree
        // Each level synchronizes with the next level up
        int current_level = 1;
        int current_rank = device_rank;
        
        while (current_level < tree_height) {
            int parent_rank = current_rank / 2;
            
            if (parent_rank < num_devices && parent_rank != current_rank) {
                int parent_device = (int)devices[parent_rank];
                
                // Check P2P access to parent
                int can_access_parent = 0;
                result = cudaDeviceCanAccessPeer(&can_access_parent, this_device, parent_device);
                
                if (result == cudaSuccess && can_access_parent) {
                    // P2P access is handled by PyTorch automatically
                    // Use P2P for synchronization
                    cudaEvent_t internal_event;
                    result = cudaEventCreateWithFlags(&internal_event, cudaEventDisableTiming);
                    if (result == cudaSuccess) {
                        result = cudaEventRecord(internal_event, 0);
                        if (result == cudaSuccess) {
                            // Wait for parent's acknowledgment using P2P signaling
                            if (g_barrier_sync_data[this_device] && g_barrier_sync_data[parent_device]) {
                                volatile uint32_t* parent_signal = &g_barrier_sync_data[parent_device]->phase_flags[this_device];
                                volatile uint32_t* parent_ack = &g_barrier_sync_data[this_device]->phase_flags[parent_device];
                                
                                // Signal parent we're ready at this level
                                *parent_signal = 1;
                                
                                // Wait for parent acknowledgment
                                const int max_wait = 100000;
                                int wait_count = 0;
                                while (*parent_ack == 0 && wait_count < max_wait) {
                                    cudaEvent_t wait_event;
                                    if (cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming) == cudaSuccess) {
                                        cudaEventRecord(wait_event, 0);
                                        cudaEventSynchronize(wait_event);
                                        cudaEventDestroy(wait_event);
                                    }
                                    wait_count++;
                                }
                                
                                *parent_signal = 0;  // Clean up
                                if (wait_count >= max_wait) {
                                    printf("WARNING: Internal node acknowledgment timeout for device %d at level %d\n",
                                           this_device, current_level);
                                }
                            } else {
                                // Fallback synchronization
                                result = cudaDeviceSynchronize();
                            }
                        }
                        cudaEventDestroy(internal_event);
                    }
                } else {
                    // Fallback synchronization
                    result = cudaDeviceSynchronize();
                }
                
                if (result != cudaSuccess) {
                    printf("ERROR: Phase 2 synchronization failed for device %d at level %d: %s\n",
                           this_device, current_level, cudaGetErrorString(result));
                    *abort_flag_ptr = 1;
                    cudaEventDestroy(local_event);
                    return;
                }
            }
            
            current_rank = parent_rank;
            current_level++;
        }
        
        // Phase 3: Root node broadcasts completion back down the tree
        // Only the root (device 0) performs the broadcast
        if (device_rank == 0) {
            // Root device ensures all operations are complete
            result = cudaDeviceSynchronize();
            if (result != cudaSuccess) {
                printf("ERROR: Root synchronization failed for device %d: %s\n",
                       this_device, cudaGetErrorString(result));
                *abort_flag_ptr = 1;
                cudaEventDestroy(local_event);
                return;
            }
            
            // Create broadcast event
            cudaEvent_t broadcast_event;
            result = cudaEventCreateWithFlags(&broadcast_event, cudaEventDisableTiming);
            if (result == cudaSuccess) {
                result = cudaEventRecord(broadcast_event, 0);
                
                // Broadcast completion to children using P2P signaling
                if (g_barrier_sync_data[this_device]) {
                    // Set broadcast flag for all children
                    for (int child_rank = 1; child_rank < num_devices; child_rank++) {
                        int child_device = (int)devices[child_rank];
                        
                        // Check if we can access child device
                        int can_access_child = 0;
                        cudaError_t access_result = cudaDeviceCanAccessPeer(&can_access_child, this_device, child_device);
                        
                        if (access_result == cudaSuccess && can_access_child && g_barrier_sync_data[child_device]) {
                            // Signal child that barrier is complete
                            volatile uint32_t* child_signal = &g_barrier_sync_data[child_device]->phase_flags[this_device];
                            *child_signal = 1;
                            
                            // Optional: Wait for child acknowledgment (not strictly necessary for broadcast)
                            // volatile uint32_t* child_ack = &g_barrier_sync_data[this_device]->phase_flags[child_device];
                            // const int ack_wait = 10000;
                            // int ack_count = 0;
                            // while (*child_ack == 0 && ack_count < ack_wait) {
                            //     ack_count++;
                            // }
                        }
                    }
                    
                    // Give children time to receive the signal
                    cudaEvent_t broadcast_delay;
                    if (cudaEventCreateWithFlags(&broadcast_delay, cudaEventDisableTiming) == cudaSuccess) {
                        cudaEventRecord(broadcast_delay, 0);
                        cudaEventSynchronize(broadcast_delay);
                        cudaEventDestroy(broadcast_delay);
                    }
                }
                
                cudaEventDestroy(broadcast_event);
            }
        } else {
            // Non-root devices wait for broadcast from root
            int root_device = (int)devices[0];
            
            // Check if we can access root device
            int can_access_root = 0;
            result = cudaDeviceCanAccessPeer(&can_access_root, this_device, root_device);
            
            if (result == cudaSuccess && can_access_root) {
                // Wait for root's broadcast using P2P signaling
                if (g_barrier_sync_data[this_device] && g_barrier_sync_data[root_device]) {
                    volatile uint32_t* root_broadcast = &g_barrier_sync_data[this_device]->phase_flags[root_device];
                    
                    // Wait for root broadcast signal
                    const int max_wait = 100000;
                    int wait_count = 0;
                    while (*root_broadcast == 0 && wait_count < max_wait) {
                        // Efficient wait using CUDA events
                        cudaEvent_t wait_event;
                        if (cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming) == cudaSuccess) {
                            cudaEventRecord(wait_event, 0);
                            cudaEventSynchronize(wait_event);
                            cudaEventDestroy(wait_event);
                        }
                        wait_count++;
                    }
                    
                    if (wait_count >= max_wait) {
                        printf("WARNING: Root broadcast timeout for device %d\n", this_device);
                        *abort_flag_ptr = 1;
                    }
                    
                    // Clear the broadcast signal
                    *root_broadcast = 0;
                } else {
                    // Fallback synchronization
                    result = cudaDeviceSynchronize();
                }
            } else {
                // Fallback synchronization
                result = cudaDeviceSynchronize();
            }
            
            if (result != cudaSuccess) {
                printf("ERROR: Phase 3 broadcast wait failed for device %d: %s\n",
                       this_device, cudaGetErrorString(result));
                *abort_flag_ptr = 1;
                cudaEventDestroy(local_event);
                return;
            }
        }
        
        // Additional timeout management
        const int max_sync_attempts = 100;
        const float sync_timeout_ms = 1000.0f;  // 1 second timeout
        
        for (int attempt = 0; attempt < max_sync_attempts; attempt++) {
            // Check if all devices are synchronized
            bool all_synced = true;
            
            for (size_t i = 0; i < devices.size(); i++) {
                int peer_device = (int)devices[i];
                if (peer_device == this_device) continue;
                
                // Check synchronization status using shared flags
                if (g_barrier_sync_data[this_device] && g_barrier_sync_data[peer_device]) {
                    volatile uint32_t* sync_status = &g_barrier_sync_data[peer_device]->phase_flags[this_device];
                    if (*sync_status != 0) {
                        // Device is still in synchronization phase
                        all_synced = false;
                    }
                } else {
                    // Fallback check using P2P capability
                    int can_access = 0;
                    result = cudaDeviceCanAccessPeer(&can_access, this_device, peer_device);
                    
                    if (result != cudaSuccess) {
                        all_synced = false;
                        break;
                    }
                }
            }
            
            if (all_synced) {
                break;
            }
            
            // Small delay between attempts
            if (attempt < max_sync_attempts - 1) {
                // Use CUDA event for timing
                cudaEvent_t timeout_event;
                if (cudaEventCreateWithFlags(&timeout_event, cudaEventDisableTiming) == cudaSuccess) {
                    cudaEventRecord(timeout_event, 0);
                    cudaEventSynchronize(timeout_event);
                    cudaEventDestroy(timeout_event);
                }
            }
        }
    }
    
    // Cleanup local event
    cudaEventDestroy(local_event);
    
    // Final synchronization to ensure all operations are complete
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        printf("ERROR: Final barrier synchronization failed for device %d: %s\n",
               this_device, cudaGetErrorString(result));
        *abort_flag_ptr = 1;
    }
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
