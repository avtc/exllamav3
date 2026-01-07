#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "context.cuh"

// Device-side vLLM-style P2P barrier
// Pure GPU-to-GPU synchronization, no CPU polling
// Optimized version with template-based world size for loop unrolling

// =============================================================================
// Memory access primitives (matching vLLM exactly)
// =============================================================================

// For START barrier: volatile only (fast, no memory ordering needed)
__device__ __forceinline__
void st_flag_volatile(uint32_t* addr, uint32_t flag) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(addr));
}

__device__ __forceinline__
uint32_t ld_flag_volatile(uint32_t* addr) {
    uint32_t flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
}

// For END barrier: release/acquire for memory ordering
__device__ __forceinline__
void st_flag_release(uint32_t* addr, uint32_t flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(addr));
#endif
}

__device__ __forceinline__
uint32_t ld_flag_acquire(uint32_t* addr) {
    uint32_t flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(addr));
#endif
    return flag;
}

// =============================================================================
// Template-optimized barrier for known world sizes
// =============================================================================

/**
 * vLLM-style P2P barrier - template version for compile-time unrolling
 *
 * @tparam ngpus Number of GPUs (must be known at compile time)
 * @param barrier_ptrs Array of P2PBarrier pointers (one per GPU)
 * @param rank This device's rank
 */
template <int ngpus>
__device__ __forceinline__
void p2p_barrier_start(P2PBarrier** barrier_ptrs, int rank)
{
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;

    P2PBarrier* local_barrier = barrier_ptrs[rank];
    uint32_t flag = local_barrier->flag[block_idx] + 1;

    if (tid < ngpus) {
        // Write our flag to ALL peer barriers (via P2P)
        #pragma unroll
        for (int peer = 0; peer < ngpus; peer++) {
            P2PBarrier* peer_barrier = barrier_ptrs[peer];
            st_flag_volatile(&peer_barrier->start[rank][tid], flag);
        }

        // Wait for all peers to write to our local barrier
        #pragma unroll
        for (int peer = 0; peer < ngpus; peer++) {
            while (ld_flag_volatile(&local_barrier->start[tid][peer]) != flag);
        }
    }

    __syncthreads();

    if (tid == 0) {
        local_barrier->flag[block_idx] = flag;
    }
}

/**
 * End barrier - uses release/acquire for memory ordering
 *
 * @tparam ngpus Number of GPUs
 * @tparam final_sync If true, uses volatile (no ordering needed after final sync)
 */
template <int ngpus, bool final_sync = false>
__device__ __forceinline__
void p2p_barrier_end(P2PBarrier** barrier_ptrs, int rank)
{
    __syncthreads();

    int tid = threadIdx.x;
    int block_idx = blockIdx.x;

    P2PBarrier* local_barrier = barrier_ptrs[rank];
    uint32_t flag = local_barrier->flag[block_idx] + 1;

    if (tid < ngpus) {
        #pragma unroll
        for (int peer = 0; peer < ngpus; peer++) {
            P2PBarrier* peer_barrier = barrier_ptrs[peer];
            if constexpr (!final_sync) {
                st_flag_release(&peer_barrier->end[rank][tid], flag);
            } else {
                st_flag_volatile(&peer_barrier->end[rank][tid], flag);
            }
        }

        #pragma unroll
        for (int peer = 0; peer < ngpus; peer++) {
            if constexpr (!final_sync) {
                while (ld_flag_acquire(&local_barrier->end[tid][peer]) != flag);
            } else {
                while (ld_flag_volatile(&local_barrier->end[tid][peer]) != flag);
            }
        }
    }

    if constexpr (!final_sync) {
        __syncthreads();
    }

    if (tid == 0) {
        local_barrier->flag[block_idx] = flag;
    }
}

// =============================================================================
// Legacy runtime-size barrier (for backward compatibility)
// =============================================================================

__device__ __forceinline__
void p2p_barrier_write_start(volatile uint32_t* addr, uint32_t flag)
{
    st_flag_volatile((uint32_t*)addr, flag);
}

__device__ __forceinline__
uint32_t p2p_barrier_read_start(volatile uint32_t* addr)
{
    return ld_flag_volatile((uint32_t*)addr);
}

__device__ __forceinline__
void p2p_barrier_write_end(volatile uint32_t* addr, uint32_t flag)
{
    st_flag_volatile((uint32_t*)addr, flag);
}

__device__ __forceinline__
uint32_t p2p_barrier_read_end(volatile uint32_t* addr)
{
    return ld_flag_volatile((uint32_t*)addr);
}

/**
 * Runtime version (cannot unroll loops) - kept for backward compatibility
 */
__device__ __forceinline__
void p2p_barrier_vllm_style(
    P2PBarrier** barrier_ptrs,
    int rank,
    int world_size,
    bool start_barrier = true
)
{
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;

    P2PBarrier* local_barrier = barrier_ptrs[rank];
    uint32_t flag = local_barrier->flag[block_idx] + 1;

    if (tid < world_size) {
        if (start_barrier) {
            for (int peer = 0; peer < world_size; peer++) {
                P2PBarrier* peer_barrier = barrier_ptrs[peer];
                st_flag_volatile(&peer_barrier->start[rank][tid], flag);
            }
            for (int peer = 0; peer < world_size; peer++) {
                while (ld_flag_volatile(&local_barrier->start[tid][peer]) != flag);
            }
        } else {
            for (int peer = 0; peer < world_size; peer++) {
                P2PBarrier* peer_barrier = barrier_ptrs[peer];
                st_flag_volatile(&peer_barrier->end[rank][tid], flag);
            }
            for (int peer = 0; peer < world_size; peer++) {
                while (ld_flag_volatile(&local_barrier->end[tid][peer]) != flag);
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        local_barrier->flag[block_idx] = flag;
    }
}
