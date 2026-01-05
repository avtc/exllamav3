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
        ctx_ptr->gather_stage_consumed[i] = 0;
        ctx_ptr->cpusum_stage_device[i * REDUCE_STAGE_STRIDE] = 0;
        ctx_ptr->p2p_temp_buffers[i] = 0;
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

void pg_open_p2p_handles(uintptr_t ctx)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    
    // Iterate all potential peer devices
    // We don't know exactly which are valid active peers easily unless passed, 
    // but we can try to open all non-zero handles.
    // However, handles are opaque.
    // Rely on Python to set them correctly.
    // We can just iterate 0..MAX_DEVICES.
    
    for (int i = 0; i < MAX_DEVICES; ++i)
    {
        if (g_p2p_opened[i]) continue; // Already opened
        
        // Check if handle is set (check if all zeros? simplistic check)
        bool is_zero = true;
        for(int j=0; j<64; ++j) if (ctx_ptr->p2p_handles[i][j] != 0) { is_zero = false; break; }
        
        // If my own device, we can just use the pointer if we had it? 
        // No, we need to map via IPC if we want consistent access path or just use local ptr.
        // Actually for *my* device, I should use the local pointer I allocated.
        // But here we are in a consumer process. 
        // If I am device i, `tensor_p2p` is mine.
        // I can just store `tensor_p2p.data_ptr()` in `g_p2p_ptrs[i]`?
        // But `pg_open_p2p_handles` doesn't know my local pointer.
        
        // Wait, `cudaIpcOpenMemHandle` on my own handle -> works?
        // Usually yes, or fails.
        // But better to verify.
        
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
            }
            else
            {
                // Warn? 
                // Maybe it's my own handle and it failed? 
                // We will handle "my own" separately if needed, 
                // but IPC usually works locally too (loopback).
                // cudaGetLastError(); // Clear error
            }
        }
    }
}

void* pg_get_p2p_ptr(int device)
{
    if (device >= 0 && device < MAX_DEVICES) return g_p2p_ptrs[device];
    return nullptr;
}
