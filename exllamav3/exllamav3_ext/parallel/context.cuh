#pragma once

#include <ATen/Tensor.h>

#define MAX_DEVICES 16
#define BROADCAST_STAGE_SIZE 16384
#define MAX_REDUCE_JOBS 2048
#define REDUCE_STAGE_STRIDE (64 / sizeof(uint32_t))

// Sync delay in nanoseconds
#define SYNC_MIN_SLEEP 64
#define SYNC_MAX_SLEEP 1024

// Timeout in seconds
#define SYNC_TIMEOUT 60ull

struct ReduceJob
{
    size_t data_size;
    uint32_t device_mask;
};

struct alignas(64) PGContext
{
    uint32_t sync_timeout;
    uint32_t busy_wait;
    uint32_t barrier_epoch;
    alignas(16) uint32_t barrier_epoch_device[MAX_DEVICES];
    alignas(16) uint32_t broadcast_stage_device[MAX_DEVICES];
    alignas(16) uint32_t reduce_stage_produced[MAX_DEVICES];
    alignas(16) uint32_t reduce_stage_consumed[MAX_DEVICES];
    alignas(16) uint32_t gather_stage_produced[MAX_DEVICES];
    alignas(16) uint32_t gather_stage_consumed[MAX_DEVICES];
    alignas(16) uint8_t p2p_handles[MAX_DEVICES][64]; // CUDA IPC handles (fixed 64 bytes)
    alignas(16) uint8_t p2p_barrier_handles[MAX_DEVICES][64]; // P2P barrier IPC handles

    // Maintain flags in separate 64-byte regions/cache lines
    alignas(64) uint32_t reduce_jobs_head; char _pad1[64 - sizeof(uint32_t)];
    alignas(64) uint32_t reduce_jobs_tail; char _pad2[64 - sizeof(uint32_t)];
    alignas(64) uint32_t cpusum_stage_device[MAX_DEVICES * REDUCE_STAGE_STRIDE];
    alignas(64) uint32_t cpusum_stage_cpu; char _pad4[64 - sizeof(uint32_t)];
    ReduceJob reduce_jobs[MAX_REDUCE_JOBS];
};

// Context management
void pg_init_context(uintptr_t ctx);
void pg_check_timeout(uintptr_t ctx);

// P2P memory management
void pg_set_p2p_handle(uintptr_t ctx, int device, const char* handle_bytes);
void pg_set_p2p_barrier_handle(uintptr_t ctx, int device, const char* handle_bytes);
void pg_get_ipc_handle(uintptr_t ptr, char* handle_out);
void pg_open_p2p_handles(uintptr_t ctx, int my_device, uintptr_t my_ptr);
void pg_open_p2p_barrier_handles(uintptr_t ctx, int my_device, uintptr_t my_barrier_ptr);
void* pg_get_p2p_ptr(int device);
void* pg_get_p2p_barrier_ptr(int device);

// P2P buffer allocation
uintptr_t pg_mem_alloc(size_t size);
void pg_mem_free(uintptr_t ptr);

// vLLM-style P2P barrier structure
struct P2PBarrier
{
    alignas(128) uint32_t start[MAX_DEVICES][MAX_DEVICES];  // [writer][reader]
    alignas(128) uint32_t end[MAX_DEVICES][MAX_DEVICES];
    alignas(128) uint32_t flag[MAX_DEVICES];  // Incremental flags per rank
};

// P2P barrier functions (pure GPU, no CPU polling)
void pg_p2p_barrier_init(uintptr_t ctx, uintptr_t barrier_ptr);
uintptr_t pg_p2p_barrier_create();