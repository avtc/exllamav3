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
             continue;
        }
        
        // Check if handle is set (check if all zeros? simplistic check)
        bool is_zero = true;
        for(int j=0; j<64; ++j) if (ctx_ptr->p2p_handles[i][j] != 0) { is_zero = false; break; }
        
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
                printf("ExLlamaV3: WARNING: cudaIpcOpenMemHandle failed for device %d (error %d)\n", i, (int)err);
            }
        }
    }
}

void* pg_get_p2p_ptr(int device)
{
    if (device >= 0 && device < MAX_DEVICES) return g_p2p_ptrs[device];
    return nullptr;
}
