#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "context.cuh"

// Device-side vLLM-style P2P barrier
// Pure GPU-to-GPU synchronization, no CPU polling

__device__ __forceinline__
void p2p_barrier_write_start(volatile uint32_t* addr, uint32_t flag)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // Use release semantics for Volta+
    asm volatile("st.release.sys.global.u32 [%1], %0;"
                 : : "r"(flag), "l"(addr));
#else
    // Fallback for older architectures
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;"
                 : : "r"(flag), "l"(addr));
#endif
}

__device__ __forceinline__
uint32_t p2p_barrier_read_start(volatile uint32_t* addr)
{
    uint32_t flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // Use acquire semantics for Volta+
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(addr));
#else
    // Fallback for older architectures
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
                 : "=r"(flag)
                 : "l"(addr));
#endif
    return flag;
}

__device__ __forceinline__
void p2p_barrier_write_end(volatile uint32_t* addr, uint32_t flag)
{
    // Use volatile writes for end barrier
    asm volatile("st.volatile.global.u32 [%1], %0;"
                 : : "r"(flag), "l"(addr));
}

__device__ __forceinline__
uint32_t p2p_barrier_read_end(volatile uint32_t* addr)
{
    uint32_t flag;
    // Use volatile reads for end barrier
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(addr));
    return flag;
}

/**
 * vLLM-style P2P barrier - pure GPU synchronization
 *
 * @param barrier Pointer to P2PBarrier structure in P2P-visible memory
 * @param rank This device's rank
 * @param world_size Total number of devices
 * @param start_barrier True if this is the start barrier, false if end barrier
 */
__device__ __forceinline__
void p2p_barrier_vllm_style(
    P2PBarrier* barrier,
    int rank,
    int world_size,
    bool start_barrier = true
)
{
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;

    // Get the next flag value for this block
    uint32_t flag = barrier->flag[block_idx] + 1;

    // Only first N threads participate (where N = world_size)
    if (tid < world_size) {
        if (start_barrier) {
            // START barrier: use start array
            // Write our flag to all peers
            p2p_barrier_write_start(&barrier->start[rank][tid], flag);

            // Wait for all peers to write to our slot
            for (int peer = 0; peer < world_size; peer++) {
                while (p2p_barrier_read_start(&barrier->start[tid][peer]) != flag) {
                    // Spin wait
                }
            }
        } else {
            // END barrier: use end array
            // Write our flag to all peers
            p2p_barrier_write_end(&barrier->end[rank][tid], flag);

            // Wait for all peers to write to our slot
            for (int peer = 0; peer < world_size; peer++) {
                while (p2p_barrier_read_end(&barrier->end[tid][peer]) != flag) {
                    // Spin wait
                }
            }
        }
    }

    // Synchronize within block
    __syncthreads();

    // Update flag for next barrier (only thread 0)
    if (tid == 0) {
        barrier->flag[block_idx] = flag;
    }
}

/**
 * Simplified version - single barrier (no start/end distinction)
 */
__device__ __forceinline__
void p2p_barrier_simple(P2PBarrier* barrier, int rank, int world_size)
{
    p2p_barrier_vllm_style(barrier, rank, world_size, true);
}
