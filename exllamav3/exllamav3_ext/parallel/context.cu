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
        
        // Initialize P2P fields
        ctx_ptr->peer_device_ptrs[i] = nullptr;
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

void pg_set_peer_device_ptr(uintptr_t ctx, int device, void* ptr)
{
    if (device < 0 || device >= MAX_DEVICES) {
        return;
    }
    
    PGContext* ctx_ptr = (PGContext*) ctx;
    ctx_ptr->peer_device_ptrs[device] = ptr;
}