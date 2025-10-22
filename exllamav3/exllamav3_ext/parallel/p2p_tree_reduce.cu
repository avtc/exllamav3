#include <cuda_fp16.h>
#include "p2p_tree_reduce.cuh"
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

#define MAX_NUM_THREADS 1024
#define BATCH_STAGE 2
#define TREE_MAX_CHILDREN 4

// Tree structure for reduction
struct TreeInfo {
    int parent;
    int children[TREE_MAX_CHILDREN];
    int num_children;
    int level;
    int rank;
};

// Performance statistics
__device__ p2p_tree_reduce_stats d_tree_stats;

__global__ __launch_bounds__(MAX_NUM_THREADS)
void p2p_tree_reduce_kernel_up(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    size_t data_size,
    TreeInfo* __restrict__ tree_info,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();
    
    __shared__ bool r;
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    
    uint8_t* data_end = data_ptr + data_size;
    const size_t reduce_stage_size = blockDim.x * sizeof(uint4);
    
    // Get tree information for this device
    TreeInfo my_tree = tree_info[this_device];
    
    // Only proceed if this device has children (is not a leaf)
    if (my_tree.num_children == 0) {
        // Leaf nodes just wait for parent to collect data
        pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
        return;
    }
    
    // Reduce phase: collect data from children
    for (int child_idx = 0; child_idx < my_tree.num_children; ++child_idx) {
        int child_device = my_tree.children[child_idx];
        
        // Check P2P capability
        int can_access_child;
        cudaDeviceCanAccessPeer(&can_access_child, this_device, child_device);
        
        if (!can_access_child) {
            // Fallback for non-P2P connections
            continue;
        }
        
        // Process data in chunks
        size_t chunk_size = reduce_stage_size;
        size_t num_chunks = CEIL_DIVIDE(data_size, chunk_size);
        
        for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
            uint8_t* src_ptr = data_ptr + chunk * chunk_size;
            uint8_t* dst_ptr = data_ptr + chunk * chunk_size;
            size_t current_chunk_size = min(chunk_size, data_size - chunk * chunk_size);
            
            // Reduce data from child into this device's buffer
            for (size_t offset = t * sizeof(uint4); offset < current_chunk_size; offset += blockDim.x * sizeof(uint4)) {
                if (offset + sizeof(uint4) <= current_chunk_size) {
                    float4* src = (float4*)(src_ptr + offset);
                    float4* dst = (float4*)(dst_ptr + offset);
                    
                    // Atomic addition for reduction
                    float4 a = *dst;
                    float4 b = *src;
                    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                    *dst = a;
                }
            }
        }
    }
    
    // Synchronize after reduction
    grid.sync();
}

__global__ __launch_bounds__(MAX_NUM_THREADS)
void p2p_tree_reduce_kernel_down(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    size_t data_size,
    TreeInfo* __restrict__ tree_info,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();
    
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    
    uint8_t* data_end = data_ptr + data_size;
    const size_t reduce_stage_size = blockDim.x * sizeof(uint4);
    
    // Get tree information for this device
    TreeInfo my_tree = tree_info[this_device];
    
    // Broadcast phase: distribute reduced data to children
    if (my_tree.num_children > 0) {
        for (int child_idx = 0; child_idx < my_tree.num_children; ++child_idx) {
            int child_device = my_tree.children[child_idx];
            
            // Check P2P capability
            int can_access_child;
            cudaDeviceCanAccessPeer(&can_access_child, this_device, child_device);
            
            if (!can_access_child) {
                // Fallback for non-P2P connections
                continue;
            }
            
            // Process data in chunks
            size_t chunk_size = reduce_stage_size;
            size_t num_chunks = CEIL_DIVIDE(data_size, chunk_size);
            
            for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
                uint8_t* src_ptr = data_ptr + chunk * chunk_size;
                uint8_t* dst_ptr = data_ptr + chunk * chunk_size;
                size_t current_chunk_size = min(chunk_size, data_size - chunk * chunk_size);
                
                // Copy reduced data to child
                for (size_t offset = t * sizeof(uint4); offset < current_chunk_size; offset += blockDim.x * sizeof(uint4)) {
                    if (offset + sizeof(uint4) <= current_chunk_size) {
                        uint4* src = (uint4*)(src_ptr + offset);
                        uint4* dst = (uint4*)(dst_ptr + offset);
                        *dst = *src;
                    }
                }
            }
        }
    }
    
    // Synchronize after broadcast
    grid.sync();
}

// Build binary tree structure
void p2p_build_binary_tree(
    std::vector<uintptr_t> devices,
    int this_device,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children
)
{
    int num_devices = devices.size();
    parent.resize(num_devices, -1);
    children.resize(num_devices);
    
    // Build binary tree: parent = floor((i-1)/2)
    for (int i = 0; i < num_devices; ++i) {
        if (i > 0) {
            parent[i] = (i - 1) / 2;
            children[parent[i]].push_back(i);
        }
    }
}

// Build k-ary tree structure
void p2p_build_kary_tree(
    std::vector<uintptr_t> devices,
    int this_device,
    int k,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children
)
{
    int num_devices = devices.size();
    parent.resize(num_devices, -1);
    children.resize(num_devices);
    
    // Build k-ary tree
    for (int i = 0; i < num_devices; ++i) {
        if (i > 0) {
            parent[i] = (i - 1) / k;
            children[parent[i]].push_back(i);
        }
    }
}

// Build balanced tree structure
void p2p_build_balanced_tree(
    std::vector<uintptr_t> devices,
    int this_device,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children
)
{
    int num_devices = devices.size();
    parent.resize(num_devices, -1);
    children.resize(num_devices);
    
    // Build balanced tree by finding optimal root and branching
    // For simplicity, use binary tree for now
    p2p_build_binary_tree(devices, this_device, parent, children);
}

// Main tree reduction function
void p2p_all_reduce_tree(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag,
    int tree_type
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);
    
    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 16 == 0, "data_size must be multiple of 16");
    
    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    int num_ranks = devices.size();
    
    if (num_ranks <= 1) return;
    
    // Build tree structure
    std::vector<int> parent;
    std::vector<std::vector<int>> children;
    
    switch (tree_type) {
        case 0:  // Binary tree
            p2p_build_binary_tree(devices, this_device, parent, children);
            break;
        case 1:  // 4-ary tree
            p2p_build_kary_tree(devices, this_device, 4, parent, children);
            break;
        case 2:  // Balanced tree
            p2p_build_balanced_tree(devices, this_device, parent, children);
            break;
        default:
            p2p_build_binary_tree(devices, this_device, parent, children);
    }
    
    // Create tree info structure for device
    TreeInfo* d_tree_info;
    cudaMalloc(&d_tree_info, num_ranks * sizeof(TreeInfo));
    
    // Copy tree info to device
    std::vector<TreeInfo> h_tree_info(num_ranks);
    for (int i = 0; i < num_ranks; ++i) {
        h_tree_info[i].parent = parent[i];
        h_tree_info[i].num_children = min((int)children[i].size(), TREE_MAX_CHILDREN);
        for (int j = 0; j < h_tree_info[i].num_children; ++j) {
            h_tree_info[i].children[j] = children[i][j];
        }
        h_tree_info[i].level = (int)log2(i + 1);
        h_tree_info[i].rank = i;
    }
    
    cudaMemcpy(d_tree_info, h_tree_info.data(), num_ranks * sizeof(TreeInfo), cudaMemcpyHostToDevice);
    
    // Launch reduction kernel
    int threads = min((int)CEIL_DIVIDE(data_size / 16ll, 32ll) * 32ll, MAX_NUM_THREADS);
    
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    
    // Phase 1: Reduce up the tree
    void* kernelArgs_up[] = {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& master_device,
        (void*)& data_ptr,
        (void*)& data_size,
        (void*)& d_tree_info,
        (void*)& abort_flag_ptr
    };
    
    cudaLaunchCooperativeKernel(
        (void*)p2p_tree_reduce_kernel_up,
        dim3(1),
        dim3(threads),
        kernelArgs_up,
        0,
        stream
    );
    
    cuda_check(cudaPeekAtLastError());
    cudaStreamSynchronize(stream);
    
    // Phase 2: Broadcast down the tree
    void* kernelArgs_down[] = {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& master_device,
        (void*)& data_ptr,
        (void*)& data_size,
        (void*)& d_tree_info,
        (void*)& abort_flag_ptr
    };
    
    cudaLaunchCooperativeKernel(
        (void*)p2p_tree_reduce_kernel_down,
        dim3(1),
        dim3(threads),
        kernelArgs_down,
        0,
        stream
    );
    
    cuda_check(cudaPeekAtLastError());
    cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaFree(d_tree_info);
}

// Adaptive algorithm selection
int p2p_select_reduce_algorithm(
    std::vector<uintptr_t> devices,
    size_t tensor_size,
    float connectivity_ratio
)
{
    int num_devices = devices.size();
    
    // For small number of devices, ring might be better
    if (num_devices <= 4) {
        return 1;  // Ring
    }
    
    // For highly connected topologies, tree is better
    if (connectivity_ratio > 0.7) {
        return 0;  // Binary tree
    }
    
    // For medium connectivity, use k-ary tree
    if (connectivity_ratio > 0.4) {
        return 1;  // 4-ary tree
    }
    
    // For low connectivity, fallback to ring
    return 2;  // Ring
}

// Performance monitoring
void p2p_get_tree_reduce_stats(p2p_tree_reduce_stats* stats)
{
    cudaMemcpy(stats, &d_tree_stats, sizeof(p2p_tree_reduce_stats), cudaMemcpyDeviceToHost);
}