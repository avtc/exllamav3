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

// vLLM-style P2P barrier implementation
static P2PBarrier* g_p2p_barrier = nullptr;

uintptr_t pg_p2p_barrier_create()
{
    if (g_p2p_barrier != nullptr) {
        printf("ExLlamaV3: P2P barrier already created\n");
        return (uintptr_t)g_p2p_barrier;
    }

    P2PBarrier* barrier = nullptr;
    cudaError_t err = cudaMalloc(&barrier, sizeof(P2PBarrier));
    if (err != cudaSuccess) {
        printf("ExLlamaV3: Failed to allocate P2P barrier: %s\n", cudaGetErrorString(err));
        return 0;  // Return 0 on error
    }

    err = cudaMemset(barrier, 0, sizeof(P2PBarrier));
    if (err != cudaSuccess) {
        printf("ExLlamaV3: Failed to initialize P2P barrier: %s\n", cudaGetErrorString(err));
        cudaFree(barrier);
        return 0;  // Return 0 on error
    }

    g_p2p_barrier = barrier;
    printf("ExLlamaV3: Created P2P barrier at %p\n", barrier);
    return (uintptr_t)barrier;  // Return pointer as integer
}

void pg_p2p_barrier_init(uintptr_t ctx, uintptr_t barrier_ptr)
{
    // Store barrier pointer in context for access from kernels
    PGContext* context = (PGContext*)ctx;
    // For now, we'll use the global pointer
    // In a multi-context setup, you'd store this per-context
}

