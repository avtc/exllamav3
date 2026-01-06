#include <cuda_fp16.h>
#include "context.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"

void pg_init_context(uintptr_t ctx)
{
    PGContext* ctx_ptr = (PGContext*) ctx;

    ctx_ptr->sync_timeout = 0;
    
    // Check environment variable for busy wait (default: false)
    const char* busy_wait_env = std::getenv("EXLLAMA_TP_BUSY_WAIT");
    ctx_ptr->busy_wait = (busy_wait_env && std::string(busy_wait_env) == "1") ? 1 : 0;
    
    ctx_ptr->barrier_epoch = 1;

    for (int i = 0; i < MAX_DEVICES; ++i)
    {
        ctx_ptr->barrier_epoch_device[i] = 0;
        ctx_ptr->broadcast_stage_device[i] = 0;
        ctx_ptr->reduce_stage_produced[i] = 0;
        ctx_ptr->reduce_stage_consumed[i] = 0;
        ctx_ptr->gather_stage_produced[i] = 0;
        ctx_ptr->gather_stage_consumed[i] = 0;
        ctx_ptr->cpusum_stage_device[i * REDUCE_STAGE_STRIDE] = 0;
        memset(ctx_ptr->p2p_handles[i], 0, 64);
    }

    ctx_ptr->reduce_jobs_head = 0;
    ctx_ptr->reduce_jobs_tail = 0;
    ctx_ptr->cpusum_stage_cpu = 0;
}

void pg_check_timeout(uintptr_t ctx)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    if (ctx_ptr->sync_timeout)
    {
        TORCH_CHECK(false, "Synchronization timeout");
    }
}

// Local cache of opened P2P pointers (process-local)
static void* g_p2p_ptrs[MAX_DEVICES] = {0};
static bool g_p2p_opened[MAX_DEVICES] = {0};

void pg_set_p2p_handle(uintptr_t ctx, int device, const char* handle_bytes)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    if (device >= 0 && device < MAX_DEVICES)
    {
        memcpy(ctx_ptr->p2p_handles[device], handle_bytes, 64);
    }
}

void pg_get_ipc_handle(uintptr_t ptr, char* handle_out)
{
    cudaIpcMemHandle_t handle;
    cuda_check(cudaIpcGetMemHandle(&handle, (void*)ptr));
    memcpy(handle_out, &handle, sizeof(handle));
}

void pg_open_p2p_handles(uintptr_t ctx, int my_device, uintptr_t my_ptr)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    
    // Iterate all potential peer devices
    for (int i = 0; i < MAX_DEVICES; ++i)
    {
        if (g_p2p_opened[i]) continue; // Already opened

        if (i == my_device)
        {
            // Use local pointer directly (cannot open own IPC handle)
            g_p2p_ptrs[i] = (void*)my_ptr;
            g_p2p_opened[i] = true;
            printf("ExLlamaV3: Device %d - using local P2P pointer: %p\n", i, g_p2p_ptrs[i]);
            continue;
        }
        
        // Check if handle is set (check if all zeros)
        bool is_zero = true;
        for(int j=0; j<64; ++j) {
            if (ctx_ptr->p2p_handles[i][j] != 0) {
                is_zero = false;
                break;
            }
        }
        
        if (!is_zero)
        {
            cudaIpcMemHandle_t handle;
            memcpy(&handle, ctx_ptr->p2p_handles[i], 64);
            void* ptr = nullptr;
            cudaError_t err = cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
            if (err == cudaSuccess)
            {
                g_p2p_ptrs[i] = ptr;
                g_p2p_opened[i] = true;
                printf("ExLlamaV3: Device %d - opened P2P handle for peer %d: %p\n", 
                       my_device, i, ptr);
            }
            else
            {
                printf("ExLlamaV3: WARNING: Device %d - cudaIpcOpenMemHandle failed for peer %d (error %d: %s)\n", 
                       my_device, i, (int)err, cudaGetErrorString(err));
            }
        }
    }
}

void* pg_get_p2p_ptr(int device)
{
    if (device >= 0 && device < MAX_DEVICES) return g_p2p_ptrs[device];
    return nullptr;
}

// Allocate P2P VRAM buffer
uintptr_t pg_mem_alloc(size_t size)
{
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        printf("ExLlamaV3: P2P cudaMalloc failed (size=%zu): %s\n", 
               size, cudaGetErrorString(err));
        return 0;
    }
    
    // Zero the memory to avoid stale data
    err = cudaMemset(ptr, 0, size);
    if (err != cudaSuccess) {
        printf("ExLlamaV3: P2P cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(ptr);
        return 0;
    }
    
    printf("ExLlamaV3: Allocated P2P buffer: %zu bytes at %p\n", size, ptr);
    return (uintptr_t)ptr;
}

void pg_mem_free(uintptr_t ptr)
{
    if (ptr) {
        cudaFree((void*)ptr);
    }
}

// Optional: P2P verification kernel (for debugging)
__global__ void pg_verify_p2p_kernel_impl
(
    void** p2p_ptrs,
    uint32_t device_mask,
    int this_device,
    uint32_t* result
)
{
    if (threadIdx.x != 0) return;
    
    // Check if my pointer is valid
    uint8_t* my_ptr = (uint8_t*)p2p_ptrs[this_device];
    if (!my_ptr) {
        printf("Device %d: My P2P pointer is NULL!\n", this_device);
        atomicOr(result, 1);
        return;
    }
    
    // Try to write and read from my buffer
    volatile uint32_t* test_ptr = (volatile uint32_t*)my_ptr;
    uint32_t test_value = 0x12345678 + this_device;
    test_ptr[0] = test_value;
    __threadfence_system();
    
    uint32_t readback = test_ptr[0];
    if (readback != test_value) {
        printf("Device %d: Write-read test FAILED (wrote 0x%x, read 0x%x)\n", 
               this_device, test_value, readback);
        atomicOr(result, 2);
        return;
    }
    
    // Check peer pointers
    for (int dev = 0; dev < MAX_DEVICES; ++dev) {
        if (!((device_mask >> dev) & 1)) continue;
        if (dev == this_device) continue;
        
        uint8_t* peer_ptr = (uint8_t*)p2p_ptrs[dev];
        if (!peer_ptr) {
            printf("Device %d: Peer %d pointer is NULL!\n", this_device, dev);
            atomicOr(result, 4);
        } else {
            // Try to read from peer (basic connectivity test)
            volatile uint32_t* peer_test = (volatile uint32_t*)peer_ptr;
            uint32_t peer_val = peer_test[0];
            // Just reading is enough to test connectivity
            (void)peer_val;
        }
    }
    
    printf("Device %d: P2P verification completed\n", this_device);
}

void pg_verify_p2p(uintptr_t ctx, std::vector<uintptr_t> devices, int this_device)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Collect all P2P pointers
    void* p2p_ptrs_host[MAX_DEVICES];
    for(int i=0; i<MAX_DEVICES; ++i) {
        p2p_ptrs_host[i] = pg_get_p2p_ptr(i);
    }
    
    // Copy to device
    void** p2p_ptrs_dev;
    cudaMalloc(&p2p_ptrs_dev, MAX_DEVICES * sizeof(void*));
    cudaMemcpy(p2p_ptrs_dev, p2p_ptrs_host, MAX_DEVICES * sizeof(void*), cudaMemcpyHostToDevice);
    
    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    
    // Create result tensor
    auto result = torch::zeros({1}, torch::dtype(torch::kInt32).device(this_device));
    uint32_t* result_ptr = result.data_ptr<uint32_t>();
    
    pg_verify_p2p_kernel_impl<<<1, 32, 0, stream>>>(
        p2p_ptrs_dev, device_mask, this_device, result_ptr
    );
    
    cudaStreamSynchronize(stream);
    
    uint32_t status = result.item<uint32_t>();
    if (status == 0) {
        printf("Device %d: P2P verification PASSED ✓\n", this_device);
    } else {
        printf("Device %d: P2P verification FAILED ✗ (status=0x%x)\n", this_device, status);
    }
    
    cudaFree(p2p_ptrs_dev);
}