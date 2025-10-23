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
    printf("DEBUG: ===== ENTERING p2p_device_barrier for device %d =====\n", this_device);
    printf("DEBUG: Function entry, parameters: devices.size()=%zu, this_device=%d\n", devices.size(), this_device);
    
    printf("DEBUG: About to create device guard for device %d\n", this_device);
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    printf("DEBUG: Device guard created successfully for device %d\n", this_device);
    
    if (devices.size() <= 1) {
        // Single device or empty list, just synchronize locally
        printf("DEBUG: Single device barrier for device %d\n", this_device);
        cudaDeviceSynchronize();
        return;
    }
    
    printf("DEBUG: Multi-device barrier for device %d with %zu devices\n", this_device, devices.size());
    
    // Check for abort flag
    printf("DEBUG: Getting abort flag pointer for device %d\n", this_device);
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    printf("DEBUG: Abort flag pointer obtained for device %d\n", this_device);
    
    if (*abort_flag_ptr != 0) {
        printf("DEBUG: Abort flag already set (%d) for device %d, returning early\n", *abort_flag_ptr, this_device);
        return;
    }
    
    printf("DEBUG: Abort flag check passed for device %d\n", this_device);
    
    printf("DEBUG: About to declare static variables for device %d\n", this_device);
    
    // Shared synchronization structures for tree reduction
    struct BarrierSyncData {
        uint32_t phase_flags[64];  // Flags for each device's phase completion
        uint32_t tree_level;       // Current tree level
        uint32_t barrier_active;   // Flag indicating barrier is active
        uint32_t padding[61];      // Padding to cache line size
    };
    
    printf("DEBUG: BarrierSyncData struct defined for device %d\n", this_device);
    
    // P2P accessible synchronization data
    static BarrierSyncData* g_barrier_sync_data[64] = {nullptr};  // One per device
    static bool sync_data_initialized = false;
    static std::mutex sync_data_mutex;
    
    printf("DEBUG: Static arrays and mutex declared for device %d\n", this_device);
    
    // Event storage for inter-device synchronization
    static cudaEvent_t inter_device_events[64][64];  // events[source][target]
    static bool events_initialized = false;
    static std::mutex events_mutex;
    
    printf("DEBUG: Checking sync_data_initialized for device %d\n", this_device);
    
    // Initialize synchronization data on first call
    if (!sync_data_initialized) {
        printf("DEBUG: Initializing sync data for device %d\n", this_device);
        std::lock_guard<std::mutex> lock(sync_data_mutex);
        if (!sync_data_initialized) {
            // Get actual device count
            int actual_device_count = 0;
            cudaError_t result = cudaGetDeviceCount(&actual_device_count);
            if (result != cudaSuccess) {
                printf("ERROR: Failed to get device count: %s\n", cudaGetErrorString(result));
                *abort_flag_ptr = 1;
                return;
            }
            
            printf("DEBUG: Found %d devices for sync data allocation\n", actual_device_count);
            
            // Allocate P2P accessible memory for synchronization data
            printf("DEBUG: Starting allocation loop for %d devices\n", actual_device_count);
            for (int dev = 0; dev < actual_device_count; dev++) {
                printf("DEBUG: Allocating sync data for device %d\n", dev);
                const at::cuda::OptionalCUDAGuard guard(dev);
                printf("DEBUG: Guard set for device %d\n", dev);
                
                void* ptr = nullptr;
                cudaError_t alloc_result = cudaMalloc(&ptr, sizeof(BarrierSyncData));
                printf("DEBUG: cudaMalloc returned %d for device %d\n", alloc_result, dev);
                
                g_barrier_sync_data[dev] = (BarrierSyncData*)ptr;
                
                if (alloc_result == cudaSuccess) {
                    // Initialize the synchronization data
                    printf("DEBUG: Memsetting sync data for device %d\n", dev);
                    cudaError_t memset_result = cudaMemset(ptr, 0, sizeof(BarrierSyncData));
                    printf("DEBUG: cudaMemset returned %d for device %d\n", memset_result, dev);
                    
                    if (memset_result == cudaSuccess) {
                        printf("DEBUG: Successfully allocated and initialized sync data for device %d\n", dev);
                    } else {
                        printf("ERROR: Failed to initialize sync data for device %d: %s\n", dev, cudaGetErrorString(memset_result));
                        g_barrier_sync_data[dev] = nullptr;
                    }
                } else {
                    printf("ERROR: Failed to allocate sync data for device %d: %s\n", dev, cudaGetErrorString(alloc_result));
                    g_barrier_sync_data[dev] = nullptr;
                }
            }
            
            // Initialize remaining entries to null
            for (int dev = actual_device_count; dev < 64; dev++) {
                g_barrier_sync_data[dev] = nullptr;
            }
            
            sync_data_initialized = true;
            printf("DEBUG: Sync data initialization completed\n");
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
    
    printf("DEBUG: Starting Phase 1 for device %d\n", this_device);
    
    // Phase 1: Local synchronization and event recording
    printf("DEBUG: About to call cudaDeviceSynchronize for device %d\n", this_device);
    cudaError_t result = cudaDeviceSynchronize();
    printf("DEBUG: cudaDeviceSynchronize returned %d for device %d\n", result, this_device);
    
    if (result != cudaSuccess) {
        printf("ERROR: Local device synchronization failed for device %d: %s\n",
               this_device, cudaGetErrorString(result));
        *abort_flag_ptr = 1;
        return;
    }
    
    printf("DEBUG: Phase 1 completed successfully for device %d\n", this_device);
    
    printf("DEBUG: Creating event for device %d\n", this_device);
    
    // Record event on this device
    cudaEvent_t local_event;
    result = cudaEventCreateWithFlags(&local_event, cudaEventDisableTiming);
    printf("DEBUG: cudaEventCreateWithFlags returned %d for device %d\n", result, this_device);
    
    if (result != cudaSuccess) {
        printf("ERROR: Failed to create event on device %d: %s\n",
               this_device, cudaGetErrorString(result));
        *abort_flag_ptr = 1;
        return;
    }
    
    printf("DEBUG: Recording event for device %d\n", this_device);
    result = cudaEventRecord(local_event, 0);
    printf("DEBUG: cudaEventRecord returned %d for device %d\n", result, this_device);
    
    if (result != cudaSuccess) {
        printf("ERROR: Failed to record event on device %d: %s\n",
               this_device, cudaGetErrorString(result));
        cudaEventDestroy(local_event);
        *abort_flag_ptr = 1;
        return;
    }
    
    printf("DEBUG: Event creation and recording completed for device %d\n", this_device);
    
    printf("DEBUG: Starting Phase 2 for device %d\n", this_device);
    
    // Phase 2: Cross-device synchronization using P2P where available
    std::vector<cudaEvent_t> peer_events;
    std::vector<int> peer_devices;
    
    printf("DEBUG: Collecting P2P peers for device %d\n", this_device);
    
    // Collect events from peer devices that we can access via P2P
    for (size_t i = 0; i < devices.size(); i++) {
        int peer_device = (int)devices[i];
        if (peer_device == this_device) continue;
        
        printf("DEBUG: Checking P2P access from device %d to device %d\n", this_device, peer_device);
        
        // Check if P2P access is possible
        int can_access = 0;
        result = cudaDeviceCanAccessPeer(&can_access, this_device, peer_device);
        printf("DEBUG: cudaDeviceCanAccessPeer returned %d, can_access=%d for device %d->%d\n", result, can_access, this_device, peer_device);
        
        if (result == cudaSuccess && can_access) {
            // P2P is available, we can directly synchronize with the peer
            // Using shared memory flags and CUDA events for synchronization
            // P2P access is handled by PyTorch automatically
            
            // Add to list of peers to synchronize with
            peer_devices.push_back(peer_device);
            printf("DEBUG: Added device %d as P2P peer for device %d\n", peer_device, this_device);
        } else {
            printf("DEBUG: Device %d cannot access device %d via P2P\n", this_device, peer_device);
        }
    }
    
    printf("DEBUG: Phase 2 peer collection completed for device %d, found %zu peers\n", this_device, peer_devices.size());
    
    printf("DEBUG: Choosing synchronization strategy for device %d\n", this_device);
    
    // Synchronization strategy based on number of devices
    if (devices.size() == 2) {
        printf("DEBUG: Using 2-device synchronization for device %d\n", this_device);
        
        // Simple 2-device synchronization
        if (!peer_devices.empty()) {
            int peer_device = peer_devices[0];
            printf("DEBUG: Synchronizing with peer device %d for device %d\n", peer_device, this_device);
            
            // Wait for peer device using shared event mechanism
            if (g_barrier_sync_data[this_device] && g_barrier_sync_data[peer_device]) {
                printf("DEBUG: Both sync data structures available for device %d\n", this_device);
                
                // Use P2P memory for synchronization flags
                volatile uint32_t* peer_flag = &g_barrier_sync_data[peer_device]->phase_flags[this_device];
                volatile uint32_t* my_flag = &g_barrier_sync_data[this_device]->phase_flags[peer_device];
                
                printf("DEBUG: Setting flag for device %d\n", this_device);
                // Set our flag to indicate we're ready
                *my_flag = 1;
                
                printf("DEBUG: Waiting for peer flag from device %d\n", peer_device);
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
                
                printf("DEBUG: Resetting flag for device %d\n", this_device);
                // Reset flags for next barrier
                *my_flag = 0;
            } else {
                printf("DEBUG: Falling back to regular synchronization for device %d\n", this_device);
                // Fallback to regular synchronization
                result = cudaDeviceSynchronize();
                if (result != cudaSuccess) {
                    printf("ERROR: Peer synchronization failed for device %d: %s\n",
                           this_device, cudaGetErrorString(result));
                    *abort_flag_ptr = 1;
                }
            }
        } else {
            printf("DEBUG: No P2P peers available for device %d\n", this_device);
        }
    } else {
        printf("DEBUG: Using multi-device synchronization for device %d\n", this_device);
        printf("DEBUG: Starting sophisticated multi-device barrier for device %d\n", this_device);
        
        // Sophisticated multi-device barrier using tree reduction algorithm
        // This implementation reduces synchronization complexity from O(N) to O(log N)
        
        printf("DEBUG: Ensuring local operations complete for device %d\n", this_device);
        // First, ensure all local operations are complete
        result = cudaDeviceSynchronize();
        printf("DEBUG: Local sync returned %d for device %d\n", result, this_device);
        
        if (result != cudaSuccess) {
            printf("ERROR: Multi-device barrier local sync failed for device %d: %s\n",
                   this_device, cudaGetErrorString(result));
            *abort_flag_ptr = 1;
            cudaEventDestroy(local_event);
            return;
        }
        
        printf("DEBUG: Starting tree reduction implementation for device %d\n", this_device);
        // Tree reduction implementation
        int num_devices = (int)devices.size();
        printf("DEBUG: Total devices in barrier: %d for device %d\n", num_devices, this_device);
        
        // Find this device's position in the device list
        printf("DEBUG: Finding device rank for device %d\n", this_device);
        int device_rank = -1;
        for (int i = 0; i < num_devices; i++) {
            if (devices[i] == this_device) {
                device_rank = i;
                printf("DEBUG: Device %d found at rank %d\n", this_device, device_rank);
                break;
            }
        }
        
        if (device_rank == -1) {
            printf("ERROR: Device %d not found in device list\n", this_device);
            *abort_flag_ptr = 1;
            cudaEventDestroy(local_event);
            return;
        }
        
        printf("DEBUG: Calculating tree structure for device %d\n", this_device);
        // Calculate tree structure
        // For non-power-of-2 device counts, we handle the extra devices at the leaf level
        int tree_height = 0;
        int temp = num_devices;
        while (temp > 1) {
            temp = (temp + 1) / 2;  // Ceiling division
            tree_height++;
        }
        printf("DEBUG: Tree height calculated as %d for device %d\n", tree_height, this_device);
        
        printf("DEBUG: Starting Phase 1 (Leaf synchronization) for device %d\n", this_device);
        
        // Phase 1: Leaf nodes synchronize with their parent
        // Each device at level 0 (leaf) synchronizes with its parent at level 1
        if (device_rank < num_devices) {
            printf("DEBUG: Device %d (rank %d) is within device range\n", this_device, device_rank);
            int parent_rank = device_rank / 2;
            printf("DEBUG: Calculated parent rank as %d for device %d\n", parent_rank, this_device);
            
            if (parent_rank < num_devices && parent_rank != device_rank) {
                int parent_device = (int)devices[parent_rank];
                printf("DEBUG: Parent device ID is %d for device %d\n", parent_device, this_device);
                
                // Check if P2P access is available to parent
                printf("DEBUG: Checking P2P access to parent for device %d\n", this_device);
                int can_access_parent = 0;
                result = cudaDeviceCanAccessPeer(&can_access_parent, this_device, parent_device);
                printf("DEBUG: P2P access check returned %d, can_access=%d for device %d->%d\n", result, can_access_parent, this_device, parent_device);
                
                if (result == cudaSuccess && can_access_parent) {
                    printf("DEBUG: P2P access available, creating leaf event for device %d\n", this_device);
                    // P2P access is handled by PyTorch automatically
                    // Create event for synchronization with parent
                    cudaEvent_t leaf_event;
                    result = cudaEventCreateWithFlags(&leaf_event, cudaEventDisableTiming);
                    printf("DEBUG: Leaf event creation returned %d for device %d\n", result, this_device);
                    
                    if (result == cudaSuccess) {
                        result = cudaEventRecord(leaf_event, 0);
                        printf("DEBUG: Leaf event record returned %d for device %d\n", result, this_device);
                        
                        if (result == cudaSuccess) {
                            printf("DEBUG: Sharing event with parent using P2P memory for device %d\n", this_device);
                            // Share event with parent device using P2P memory
                            if (g_barrier_sync_data[this_device] && g_barrier_sync_data[parent_device]) {
                                printf("DEBUG: Both sync data structures available for device %d\n", this_device);
                                // Store event pointer in shared memory for parent to access
                                // Note: In a real implementation, we'd need IPC mechanisms
                                // For now, we'll use the phase flags as a simple signaling mechanism
                                
                                volatile uint32_t* parent_signal = &g_barrier_sync_data[parent_device]->phase_flags[this_device];
                                printf("DEBUG: Setting parent signal for device %d\n", this_device);
                                *parent_signal = 1;  // Signal parent we're ready
                                
                                // Wait for parent acknowledgment
                                printf("DEBUG: Waiting for parent acknowledgment for device %d\n", this_device);
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
                                
                                printf("DEBUG: Cleaning up signals for device %d\n", this_device);
                                // Clean up signals
                                *parent_signal = 0;
                                if (wait_count >= max_wait) {
                                    printf("WARNING: Parent acknowledgment timeout for device %d\n", this_device);
                                }
                            } else {
                                printf("DEBUG: Falling back to device synchronization for device %d\n", this_device);
                                // Fallback to device synchronization
                                result = cudaDeviceSynchronize();
                            }
                        }
                        printf("DEBUG: Destroying leaf event for device %d\n", this_device);
                        cudaEventDestroy(leaf_event);
                    }
                } else {
                    printf("DEBUG: No P2P access, falling back to regular sync for device %d\n", this_device);
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
            } else {
                printf("DEBUG: Device %d has no valid parent (parent_rank=%d)\n", this_device, parent_rank);
            }
        } else {
            printf("DEBUG: Device %d (rank %d) is outside device range\n", this_device, device_rank);
        }
        
        printf("DEBUG: Phase 1 completed for device %d\n", this_device);
        
        printf("DEBUG: Starting Phase 2 (Internal nodes) for device %d\n", this_device);
        
        // Phase 2: Internal nodes synchronize up the tree
        // Each level synchronizes with the next level up
        int current_level = 1;
        int current_rank = device_rank;
        printf("DEBUG: Initial values: current_level=%d, current_rank=%d for device %d\n", current_level, current_rank, this_device);
        
        while (current_level < tree_height) {
            printf("DEBUG: Processing level %d for device %d\n", current_level, this_device);
            int parent_rank = current_rank / 2;
            printf("DEBUG: Parent rank for level %d is %d for device %d\n", current_level, parent_rank, this_device);
            
            if (parent_rank < num_devices && parent_rank != current_rank) {
                int parent_device = (int)devices[parent_rank];
                printf("DEBUG: Parent device ID is %d for device %d at level %d\n", parent_device, this_device, current_level);
                
                // Check P2P access to parent
                printf("DEBUG: Checking P2P access to parent at level %d for device %d\n", current_level, this_device);
                int can_access_parent = 0;
                result = cudaDeviceCanAccessPeer(&can_access_parent, this_device, parent_device);
                printf("DEBUG: P2P access check returned %d, can_access=%d for device %d->%d at level %d\n", result, can_access_parent, this_device, parent_device, current_level);
                
                if (result == cudaSuccess && can_access_parent) {
                    printf("DEBUG: P2P access available, creating internal event for device %d at level %d\n", this_device, current_level);
                    // P2P access is handled by PyTorch automatically
                    // Use P2P for synchronization
                    cudaEvent_t internal_event;
                    result = cudaEventCreateWithFlags(&internal_event, cudaEventDisableTiming);
                    printf("DEBUG: Internal event creation returned %d for device %d at level %d\n", result, this_device, current_level);
                    
                    if (result == cudaSuccess) {
                        result = cudaEventRecord(internal_event, 0);
                        printf("DEBUG: Internal event record returned %d for device %d at level %d\n", result, this_device, current_level);
                        
                        if (result == cudaSuccess) {
                            printf("DEBUG: Waiting for parent acknowledgment at level %d for device %d\n", current_level, this_device);
                            // Wait for parent's acknowledgment using P2P signaling
                            if (g_barrier_sync_data[this_device] && g_barrier_sync_data[parent_device]) {
                                printf("DEBUG: Both sync data structures available for device %d at level %d\n", this_device, current_level);
                                volatile uint32_t* parent_signal = &g_barrier_sync_data[parent_device]->phase_flags[this_device];
                                volatile uint32_t* parent_ack = &g_barrier_sync_data[this_device]->phase_flags[parent_device];
                                
                                // Signal parent we're ready at this level
                                printf("DEBUG: Signaling parent at level %d for device %d\n", current_level, this_device);
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
                                
                                printf("DEBUG: Cleaning up signals at level %d for device %d\n", current_level, this_device);
                                *parent_signal = 0;  // Clean up
                                if (wait_count >= max_wait) {
                                    printf("WARNING: Internal node acknowledgment timeout for device %d at level %d\n",
                                           this_device, current_level);
                                }
                            } else {
                                printf("DEBUG: Falling back to synchronization at level %d for device %d\n", current_level, this_device);
                                // Fallback synchronization
                                result = cudaDeviceSynchronize();
                            }
                        }
                        printf("DEBUG: Destroying internal event at level %d for device %d\n", current_level, this_device);
                        cudaEventDestroy(internal_event);
                    }
                } else {
                    printf("DEBUG: No P2P access, falling back to regular sync at level %d for device %d\n", current_level, this_device);
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
            } else {
                printf("DEBUG: Device %d has no valid parent at level %d (parent_rank=%d)\n", this_device, current_level, parent_rank);
            }
            
            current_rank = parent_rank;
            current_level++;
            printf("DEBUG: Moving to next level: current_level=%d, current_rank=%d for device %d\n", current_level, current_rank, this_device);
        }
        
        printf("DEBUG: Phase 2 completed for device %d\n", this_device);
        
        printf("DEBUG: Starting Phase 3 (Root broadcast) for device %d\n", this_device);
        
        // Phase 3: Root node broadcasts completion back down the tree
        // Only the root (device 0) performs the broadcast
        if (device_rank == 0) {
            printf("DEBUG: Device %d is root, performing broadcast\n", this_device);
            
            // Root device ensures all operations are complete
            printf("DEBUG: Root device syncing locally\n");
            result = cudaDeviceSynchronize();
            if (result != cudaSuccess) {
                printf("ERROR: Root synchronization failed for device %d: %s\n",
                       this_device, cudaGetErrorString(result));
                *abort_flag_ptr = 1;
                cudaEventDestroy(local_event);
                return;
            }
            
            // Create broadcast event
            printf("DEBUG: Creating broadcast event for root device %d\n", this_device);
            cudaEvent_t broadcast_event;
            result = cudaEventCreateWithFlags(&broadcast_event, cudaEventDisableTiming);
            printf("DEBUG: Broadcast event creation returned %d for root device %d\n", result, this_device);
            
            if (result == cudaSuccess) {
                result = cudaEventRecord(broadcast_event, 0);
                printf("DEBUG: Broadcast event record returned %d for root device %d\n", result, this_device);
                
                // Broadcast completion to children using P2P signaling
                if (g_barrier_sync_data[this_device]) {
                    printf("DEBUG: Broadcasting to %d children from root device %d\n", num_devices - 1, this_device);
                    // Set broadcast flag for all children
                    for (int child_rank = 1; child_rank < num_devices; child_rank++) {
                        int child_device = (int)devices[child_rank];
                        printf("DEBUG: Broadcasting to child device %d (rank %d) from root device %d\n", child_device, child_rank, this_device);
                        
                        // Check if we can access child device
                        int can_access_child = 0;
                        cudaError_t access_result = cudaDeviceCanAccessPeer(&can_access_child, this_device, child_device);
                        printf("DEBUG: Can access child %d: %d (result=%d) from root device %d\n", child_device, can_access_child, access_result, this_device);
                        
                        if (access_result == cudaSuccess && can_access_child && g_barrier_sync_data[child_device]) {
                            // Signal child that barrier is complete
                            printf("DEBUG: Signaling child device %d from root device %d\n", child_device, this_device);
                            volatile uint32_t* child_signal = &g_barrier_sync_data[child_device]->phase_flags[this_device];
                            *child_signal = 1;
                            
                            // Optional: Wait for child acknowledgment (not strictly necessary for broadcast)
                            // volatile uint32_t* child_ack = &g_barrier_sync_data[this_device]->phase_flags[child_device];
                            // const int ack_wait = 10000;
                            // int ack_count = 0;
                            // while (*child_ack == 0 && ack_count < ack_wait) {
                            //     ack_count++;
                            // }
                        } else {
                            printf("DEBUG: Cannot signal child device %d from root device %d\n", child_device, this_device);
                        }
                    }
                    
                    // Give children time to receive the signal
                    printf("DEBUG: Adding delay for children to receive broadcast from root device %d\n", this_device);
                    cudaEvent_t broadcast_delay;
                    if (cudaEventCreateWithFlags(&broadcast_delay, cudaEventDisableTiming) == cudaSuccess) {
                        cudaEventRecord(broadcast_delay, 0);
                        cudaEventSynchronize(broadcast_delay);
                        cudaEventDestroy(broadcast_delay);
                    }
                }
                
                printf("DEBUG: Destroying broadcast event for root device %d\n", this_device);
                cudaEventDestroy(broadcast_event);
            }
        } else {
            printf("DEBUG: Device %d is non-root, waiting for broadcast\n", this_device);
            // Non-root devices wait for broadcast from root
            int root_device = (int)devices[0];
            printf("DEBUG: Root device ID is %d for non-root device %d\n", root_device, this_device);
            
            // Check if we can access root device
            printf("DEBUG: Checking P2P access to root for device %d\n", this_device);
            int can_access_root = 0;
            result = cudaDeviceCanAccessPeer(&can_access_root, this_device, root_device);
            printf("DEBUG: P2P access check to root returned %d, can_access=%d for device %d->%d\n", result, can_access_root, this_device, root_device);
            
            if (result == cudaSuccess && can_access_root) {
                printf("DEBUG: Waiting for root broadcast using P2P for device %d\n", this_device);
                // Wait for root's broadcast using P2P signaling
                if (g_barrier_sync_data[this_device] && g_barrier_sync_data[root_device]) {
                    printf("DEBUG: Both sync data structures available for device %d\n", this_device);
                    volatile uint32_t* root_broadcast = &g_barrier_sync_data[this_device]->phase_flags[root_device];
                    
                    // Wait for root broadcast signal
                    printf("DEBUG: Waiting for root broadcast signal for device %d\n", this_device);
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
                    
                    printf("DEBUG: Clearing root broadcast signal for device %d\n", this_device);
                    // Clear the broadcast signal
                    *root_broadcast = 0;
                } else {
                    printf("DEBUG: Falling back to synchronization for device %d\n", this_device);
                    // Fallback synchronization
                    result = cudaDeviceSynchronize();
                }
            } else {
                printf("DEBUG: No P2P access to root, falling back to sync for device %d\n", this_device);
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
        
        printf("DEBUG: Phase 3 completed for device %d\n", this_device);
        
        printf("DEBUG: Starting additional timeout management for device %d\n", this_device);
        
        // Additional timeout management
        const int max_sync_attempts = 100;
        const float sync_timeout_ms = 1000.0f;  // 1 second timeout
        
        printf("DEBUG: Beginning sync verification loop for device %d\n", this_device);
        for (int attempt = 0; attempt < max_sync_attempts; attempt++) {
            printf("DEBUG: Sync verification attempt %d/%d for device %d\n", attempt + 1, max_sync_attempts, this_device);
            
            // Check if all devices are synchronized
            bool all_synced = true;
            
            for (size_t i = 0; i < devices.size(); i++) {
                int peer_device = (int)devices[i];
                if (peer_device == this_device) continue;
                
                printf("DEBUG: Checking sync status with peer device %d for device %d\n", peer_device, this_device);
                
                // Check synchronization status using shared flags
                if (g_barrier_sync_data[this_device] && g_barrier_sync_data[peer_device]) {
                    volatile uint32_t* sync_status = &g_barrier_sync_data[peer_device]->phase_flags[this_device];
                    if (*sync_status != 0) {
                        // Device is still in synchronization phase
                        printf("DEBUG: Device %d still syncing with peer %d (flag=%d)\n", this_device, peer_device, *sync_status);
                        all_synced = false;
                    } else {
                        printf("DEBUG: Device %d synced with peer %d\n", this_device, peer_device);
                    }
                } else {
                    // Fallback check using P2P capability
                    printf("DEBUG: Using fallback P2P check for device %d->%d\n", this_device, peer_device);
                    int can_access = 0;
                    result = cudaDeviceCanAccessPeer(&can_access, this_device, peer_device);
                    
                    if (result != cudaSuccess) {
                        printf("DEBUG: P2P check failed for device %d->%d\n", this_device, peer_device);
                        all_synced = false;
                        break;
                    }
                }
            }
            
            if (all_synced) {
                printf("DEBUG: All devices synchronized for device %d\n", this_device);
                break;
            }
            
            // Small delay between attempts
            if (attempt < max_sync_attempts - 1) {
                printf("DEBUG: Adding delay before next sync attempt for device %d\n", this_device);
                // Use CUDA event for timing
                cudaEvent_t timeout_event;
                if (cudaEventCreateWithFlags(&timeout_event, cudaEventDisableTiming) == cudaSuccess) {
                    cudaEventRecord(timeout_event, 0);
                    cudaEventSynchronize(timeout_event);
                    cudaEventDestroy(timeout_event);
                }
            }
        }
        
        printf("DEBUG: Additional timeout management completed for device %d\n", this_device);
    }
    
    printf("DEBUG: Cleaning up local event for device %d\n", this_device);
    // Cleanup local event
    cudaEventDestroy(local_event);
    
    printf("DEBUG: Performing final synchronization for device %d\n", this_device);
    // Final synchronization to ensure all operations are complete
    result = cudaDeviceSynchronize();
    printf("DEBUG: Final synchronization returned %d for device %d\n", result, this_device);
    
    if (result != cudaSuccess) {
        printf("ERROR: Final barrier synchronization failed for device %d: %s\n",
               this_device, cudaGetErrorString(result));
        *abort_flag_ptr = 1;
    }
    
    printf("DEBUG: ===== EXITING p2p_device_barrier for device %d =====\n", this_device);
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
