#include <cuda_fp16.h>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stloader.h"
#include "hadamard.h"

#include "norm.cuh"
#include "hgemm.cuh"
#include "rope.cuh"
#include "activation.cuh"
#include "softcap.cuh"
#include "routing.cuh"
#include "gdn.cuh"
#include "causal_conv1d.cuh"
#include "add.cuh"

#include "quant/quantize.cuh"
#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/util.cuh"
#include "quant/exl3_devctx.cuh"

#include "generator/strings.h"
#include "generator/sampling_basic.cuh"
#include "generator/gumbel.cuh"
#include "generator/rep_pen.cuh"
#include "generator/cache.cuh"

#include "cache/q_cache.cuh"

#include "histogram.cuh"

#include "parallel/context.cuh"
#include "parallel/broadcast.cuh"
#include "parallel/barrier.cuh"
#include "parallel/gather.cuh"
#include "parallel/all_reduce.cuh"
#include "parallel/p2p_broadcast.cuh"
#include "parallel/p2p_all_reduce.cuh"
#include "parallel/p2p_gather.cuh"
#include "parallel/p2p_memory.cuh"
#include "parallel/p2p_tree_reduce.cuh"
#include "parallel/p2p_direct_memory.cuh"

#include "libtorch/gated_delta_net.h"
#include "libtorch/linear.h"
#include "libtorch/gated_rmsnorm.h"
#include "libtorch/mlp.h"
#include "libtorch/blocksparse_mlp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stloader_read", &stloader_read, "stloader_read");
    m.def("stloader_open_file", &stloader_open_file, "stloader_open_file");
    m.def("stloader_close_file", &stloader_close_file, "stloader_close_file");
    py::class_<TensorLoadJob>(m, "TensorLoadJob")
        .def(py::init<std::vector<uintptr_t>, size_t, size_t, uintptr_t, bool, bool, bool, int>());
    m.def("stloader_deferred_cpu", &stloader_deferred_cpu, py::arg("jobs"));
    m.def("stloader_deferred_cuda", &stloader_deferred_cuda, py::arg("jobs"), py::arg("max_chunk_size"));

    m.def("rms_norm", &rms_norm, "rms_norm");
    m.def("gated_rms_norm", &gated_rms_norm, "gated_rms_norm");
    m.def("softcap", &softcap, "softcap");

    m.def("routing_ds3_nogroup", &routing_ds3_nogroup, "routing_ds3_nogroup");
    m.def("routing_std", &routing_std, "routing_std");

    m.def("had_paley", &had_paley, "had_paley");
    m.def("had_paley2", &had_paley2, "had_paley2");

    m.def("pg_init_context", &pg_init_context, "pg_init_context");
    m.def("pg_broadcast", &pg_broadcast, "pg_broadcast");
    m.def("pg_broadcast_ll", &pg_broadcast_ll, "pg_broadcast_ll");
    m.def("pg_barrier", &pg_barrier, "pg_barrier");
    m.def("pg_gather", &pg_gather, "pg_gather");
    m.def("pg_all_reduce", &pg_all_reduce, "pg_all_reduce");
    m.def("pg_all_reduce_cpu", &pg_all_reduce_cpu, "pg_all_reduce_cpu");
    m.def("run_cpu_reduce_jobs", &run_cpu_reduce_jobs, "run_cpu_reduce_jobs");
    m.def("end_cpu_reduce_jobs", &end_cpu_reduce_jobs, "end_cpu_reduce_jobs");
    
    // P2P functions
    m.def("p2p_broadcast", &p2p_broadcast, "p2p_broadcast");
    m.def("p2p_broadcast_ll", &p2p_broadcast_ll, "p2p_broadcast_ll");
    m.def("p2p_all_reduce", &p2p_all_reduce, "p2p_all_reduce");
    m.def("p2p_all_reduce_ring", &p2p_all_reduce_ring, "p2p_all_reduce_ring");
    m.def("p2p_all_reduce_tree_adaptive", &p2p_all_reduce_tree_adaptive, "p2p_all_reduce_tree_adaptive");
    m.def("p2p_gather", &p2p_gather, "p2p_gather");
    m.def("p2p_gather_direct", &p2p_gather_direct, "p2p_gather_direct");
    m.def("p2p_init_memory_pool", &p2p_init_memory_pool, "p2p_init_memory_pool");
    m.def("p2p_cleanup_memory_pool", &p2p_cleanup_memory_pool, "p2p_cleanup_memory_pool");
    m.def("p2p_allocate_from_pool", &p2p_allocate_from_pool, "p2p_allocate_from_pool");
    m.def("p2p_free_to_pool", &p2p_free_to_pool, "p2p_free_to_pool");
    m.def("p2p_get_peer_device_ptr", &p2p_get_peer_device_ptr, "p2p_get_peer_device_ptr");
    m.def("p2p_can_access_peer", &p2p_can_access_peer, "p2p_can_access_peer");
    m.def("p2p_device_barrier", &p2p_device_barrier, "p2p_device_barrier");
    
    // Direct memory access functions
    m.def("p2p_copy_tensor_async", &p2p_copy_tensor_async, "p2p_copy_tensor_async");
    m.def("p2p_copy_tensor_sync", &p2p_copy_tensor_sync, "p2p_copy_tensor_sync");
    m.def("p2p_copy_tensor_batch", &p2p_copy_tensor_batch, "p2p_copy_tensor_batch");
    m.def("p2p_copy_tensor_pinned", &p2p_copy_tensor_pinned, "p2p_copy_tensor_pinned");
    m.def("p2p_copy_tensor_2d_async", &p2p_copy_tensor_2d_async, "p2p_copy_tensor_2d_async");
    m.def("p2p_copy_tensor_3d_async", &p2p_copy_tensor_3d_async, "p2p_copy_tensor_3d_async");
    m.def("p2p_copy_tensor_with_offset", &p2p_copy_tensor_with_offset, "p2p_copy_tensor_with_offset");
    m.def("p2p_copy_tensor_strided", &p2p_copy_tensor_strided, "p2p_copy_tensor_strided");
    
    // Memory registration functions
    m.def("p2p_register_memory_region", &p2p_register_memory_region, "p2p_register_memory_region");
    m.def("p2p_unregister_memory_region", &p2p_unregister_memory_region, "p2p_unregister_memory_region");
    m.def("p2p_is_memory_registered", &p2p_is_memory_registered, "p2p_is_memory_registered");
    
    // Zero-copy memory operations
    m.def("p2p_allocate_zero_copy", &p2p_allocate_zero_copy, "p2p_allocate_zero_copy");
    m.def("p2p_free_zero_copy", &p2p_free_zero_copy, "p2p_free_zero_copy");
    
    // Performance monitoring functions
    m.def("p2p_measure_bandwidth", &p2p_measure_bandwidth, "p2p_measure_bandwidth");
    m.def("p2p_measure_latency", &p2p_measure_latency, "p2p_measure_latency");
    
    // Memory access validation
    m.def("p2p_validate_memory_access", &p2p_validate_memory_access, "p2p_validate_memory_access");
    
    // Synchronization functions
    m.def("p2p_synchronize_devices", &p2p_synchronize_devices, "p2p_synchronize_devices");
    m.def("p2p_enable_peer_access", &p2p_enable_peer_access, "p2p_enable_peer_access");
    m.def("p2p_disable_peer_access", &p2p_disable_peer_access, "p2p_disable_peer_access");
    m.def("p2p_is_peer_access_enabled", &p2p_is_peer_access_enabled, "p2p_is_peer_access_enabled");
    
    // Enhanced direct memory pool functions
    m.def("p2p_init_direct_memory_pool", &p2p_init_direct_memory_pool, "p2p_init_direct_memory_pool");
    m.def("p2p_cleanup_direct_memory_pool", &p2p_cleanup_direct_memory_pool, "p2p_cleanup_direct_memory_pool");
    m.def("p2p_allocate_from_direct_pool", &p2p_allocate_from_direct_pool, "p2p_allocate_from_direct_pool");
    m.def("p2p_free_to_direct_pool", &p2p_free_to_direct_pool, "p2p_free_to_direct_pool");
    m.def("p2p_can_access_peer_direct", &p2p_can_access_peer_direct, "p2p_can_access_peer_direct");
    m.def("p2p_register_peer_memory", &p2p_register_peer_memory, "p2p_register_peer_memory");
    m.def("p2p_unregister_peer_memory", &p2p_unregister_peer_memory, "p2p_unregister_peer_memory");
    m.def("p2p_get_direct_pool_usage", &p2p_get_direct_pool_usage, "p2p_get_direct_pool_usage");
    m.def("p2p_get_direct_pool_size", &p2p_get_direct_pool_size, "p2p_get_direct_pool_size");

    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles");
    m.def("test_distribution", &test_distribution, "test_distribution");
    m.def("decode", &decode, "decode");
    m.def("pack_trellis", &pack_trellis, "pack_trellis");
    m.def("unpack_trellis", &unpack_trellis, "unpack_trellis");
    m.def("pack_signs", &pack_signs, "pack_signs");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("had_r_128", &had_r_128, "had_r_128");
    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");
    m.def("exl3_gemm_num_kernel_shapes", &exl3_gemm_num_kernel_shapes, "exl3_gemm_num_kernel_shapes");
    m.def("exl3_gemm_shape_compat", &exl3_gemm_shape_compat, "exl3_gemm_shape_compat");
    m.def("g_get_cc", &g_get_cc, "g_get_cc");
    m.def("g_get_num_sms", &g_get_num_sms, "g_get_num_sms");
    m.def("exl3_mgemm", &exl3_mgemm, "exl3_mgemm");
    m.def("hgemm", &hgemm, "hgemm");
    m.def("rope", &rope, "rope");
    m.def("silu_mul", &silu_mul, "silu_mul");
    m.def("gelu_mul", &gelu_mul, "gelu_mul");
    m.def("relu2_mul", &relu2_mul, "relu2_mul");
    m.def("xielu", &xielu, "xielu");
    m.def("add_sigmoid_gate", &add_sigmoid_gate, "add_sigmoid_gate");
    m.def("add", &add, "add");

    m.def("gated_delta_net_fused_op", &gated_delta_net_fused_op, "gated_delta_net_fused_op");
    m.def("cuda_recurrent_gated_delta_rule", &cuda_recurrent_gated_delta_rule, "cuda_recurrent_gated_delta_rule");

    m.def("argmax_sample", &argmax_sample, "argmax_sample");
    m.def("gumbel_sample", &gumbel_sample, "gumbel_sample");
    m.def("gumbel_noise_f16", &gumbel_noise_f16, "gumbel_noise_f16");
    m.def("gumbel_noise_f32", &gumbel_noise_f32, "gumbel_noise_f32");
    m.def("gumbel_noise_log", &gumbel_noise_log, "gumbel_noise_log");
    m.def("apply_rep_pens", &apply_rep_pens, "apply_rep_pens");
    m.def("apply_pres_freq_pens", &apply_pres_freq_pens, "apply_pres_freq_pens");

    m.def("cache_rotate", &cache_rotate, "cache_rotate");

    m.def("partial_strings_match", &partial_strings_match, "partial_strings_match");
    m.def("count_match_tensor", &count_match_tensor, "count_match_tensor");

    m.def("quant_cache_cont", &quant_cache_cont, "quant_cache_cont");
    m.def("dequant_cache_cont", &dequant_cache_cont, "dequant_cache_cont");
    m.def("quant_cache_paged", &quant_cache_paged, "quant_cache_paged");
    m.def("dequant_cache_paged", &dequant_cache_paged, "dequant_cache_paged");

    m.def("count_inf_nan", &count_inf_nan, "count_inf_nan");
    m.def("histogram", &histogram, "histogram");

    m.def("blocksparse_mlp_routing", &blocksparse_mlp_routing, "blocksparse_mlp_routing");

    #include "libtorch/linear_bc.h"
    #include "libtorch/gated_delta_net_bc.h"
    #include "libtorch/gated_rmsnorm_bc.h"
    #include "libtorch/mlp_bc.h"
    #include "libtorch/blocksparse_mlp_bc.h"
}