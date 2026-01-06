#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Vectorized memory access utilities for custom all-reduce
// Adapted from vLLM's implementation

// Aligned array type to generate ld.128/st.128 PTX instructions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
    T data[sz];
    using type = T;
    static constexpr int size = sz;
};

// Packed type definitions for 128-bit vectorized loads
template <typename T>
struct packed_t {
    // The (P)acked type for load/store - 16-byte aligned
    using P = array_t<T, 16 / sizeof(T)>;
    // The (A)ccumulator type for reduction - always float for precision
    using A = array_t<float, 16 / sizeof(T)>;
};

// Type specializations
template <>
struct packed_t<half> {
    using P = array_t<half, 8>;      // 16 bytes / 2 bytes = 8 halves
    using A = array_t<float, 8>;     // Accumulate in float
};

#if defined(__CUDA_BF16__) && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 11
template <>
struct packed_t<__nv_bfloat16> {
    using P = array_t<__nv_bfloat16, 8>;  // 16 bytes / 2 bytes = 8 bf16
    using A = array_t<float, 8>;           // Accumulate in float
};
#endif

// Scalar upcast functions (T -> float)
__device__ __forceinline__ float upcast_s(float val) { return val; }
__device__ __forceinline__ float upcast_s(half val) { return __half2float(val); }
#if defined(__CUDA_BF16__)
__device__ __forceinline__ float upcast_s(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// Scalar downcast functions (float -> T)
__device__ __forceinline__ float downcast_s(float val) { return val; }
__device__ __forceinline__ half downcast_s(half) { return __float2half(val); }
#if defined(__CUDA_BF16__)
__device__ __forceinline__ __nv_bfloat16 downcast_s(__nv_bfloat16) { return __float2bfloat16(val); }
#endif

// Vector upcast: Convert packed type to float accumulator
template <typename T, int N>
__device__ __forceinline__
array_t<float, N> upcast(array_t<T, N> val) {
    array_t<float, N> out;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        out.data[i] = upcast_s(val.data[i]);
    }
    return out;
}

// Vector downcast: Convert float accumulator back to packed type
template <typename O>
__device__ __forceinline__
O downcast(array_t<float, O::size> val) {
    if constexpr (std::is_same<typename O::type, float>::value) {
        return val;
    } else {
        O out;
        #pragma unroll
        for (int i = 0; i < O::size; i++) {
            out.data[i] = downcast_s<typename O::type>(val.data[i]);
        }
        return out;
    }
}

// Vectorized assign-add: Add two packed vectors
template <typename T, int N>
__device__ __forceinline__
array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
        a.data[i] += b.data[i];
    }
    return a;
}

// Core vectorized reduction function
// Reads from all GPU pointers and accumulates in a single operation
template <typename P, int ngpus, typename A>
__device__ __forceinline__
P packed_reduce(const P* ptrs[], int idx) {
    // Load first GPU's data and upcast to float accumulator
    A tmp = upcast(ptrs[0][idx]);

    // Accumulate from remaining GPUs
    #pragma unroll
    for (int i = 1; i < ngpus; i++) {
        packed_assign_add(tmp, upcast(ptrs[i][idx]));
    }

    // Downcast back to packed type
    return downcast<P>(tmp);
}

// Version that works with raw pointers
template <typename P, int ngpus, typename A>
__device__ __forceinline__
P packed_reduce_ptrs(uint8_t* ptrs[], int idx) {
    // Cast first pointer to packed type
    const P* ptr0 = (const P*)ptrs[0];

    // Load and upcast to float accumulator
    A tmp = upcast(ptr0[idx]);

    // Accumulate from remaining GPUs
    #pragma unroll
    for (int i = 1; i < ngpus; i++) {
        const P* ptr = (const P*)ptrs[i];
        packed_assign_add(tmp, upcast(ptr[idx]));
    }

    // Downcast back to packed type
    return downcast<P>(tmp);
}
