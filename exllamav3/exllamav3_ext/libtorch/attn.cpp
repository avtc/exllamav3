#include <Python.h>
#include "attn.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../activation.cuh"
#include "../norm.cuh"
#include "../rope.cuh"
#include "../quant/util.cuh"

void BC_Attention::run_proj_gr(const at::Tensor& x, int past_len, Graph* graph)
{
    // Ensure temp tensors
    if (!temp_q.defined()) {
        auto opts = torch::TensorOptions().dtype(torch::kHalf).device(x.device());
        temp_q = torch::empty({1, 1, num_heads * head_dim}, opts);
        temp_k = torch::empty({1, 1, num_kv_heads * head_dim}, opts);
        temp_v = torch::empty({1, 1, num_kv_heads * head_dim}, opts);
    }

    // 1. Projections
    // Note: Assuming x has correct shape [1, 1, hidden_size] or similar
    q_proj->run_gr(x, temp_q, graph);
    k_proj->run_gr(x, temp_k, graph);
    v_proj->run_gr(x, temp_v, graph);

    // 2. Norms
    if (q_norm_weight.defined()) {
        rms_norm_gr(temp_q, q_norm_weight, temp_q, norm_epsilon, graph);
    }
    if (k_norm_weight.defined()) {
        rms_norm_gr(temp_k, k_norm_weight, temp_k, norm_epsilon, graph);
    }

    // 3. RoPE
    rope_gr(
        temp_q, temp_k, 
        sin, cos, 
        past_len, 
        1, 
        num_heads, num_kv_heads, head_dim, 
        false, 
        graph
    );
}

std::vector<at::Tensor> BC_Attention::run_proj(const at::Tensor& x, int past_len)
{
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (!graph_proj.ready)
    {
        // Capture
        graph_proj.capture_begin();
        run_proj_gr(x, past_len, &graph_proj);
        graph_proj.capture_end();
    }

    // We execute RoPE eagerly because 'past_len' changes every step,
    // and updating scalar arguments in CUDA graphs is complex.
    
    if (!graph_proj.ready)
    {
        graph_proj.capture_begin();
        // Capture Projections + Norms
        // 1. Projections
        if (!temp_q.defined()) {
            auto opts = torch::TensorOptions().dtype(torch::kHalf).device(x.device());
            temp_q = torch::empty({1, 1, num_heads * head_dim}, opts);
            temp_k = torch::empty({1, 1, num_kv_heads * head_dim}, opts);
            temp_v = torch::empty({1, 1, num_kv_heads * head_dim}, opts);
        }
        q_proj->run_gr(x, temp_q, &graph_proj);
        k_proj->run_gr(x, temp_k, &graph_proj);
        v_proj->run_gr(x, temp_v, &graph_proj);
        
        // 2. Norms
        if (q_norm_weight.defined()) rms_norm_gr(temp_q, q_norm_weight, temp_q, norm_epsilon, &graph_proj);
        if (k_norm_weight.defined()) rms_norm_gr(temp_k, k_norm_weight, temp_k, norm_epsilon, &graph_proj);
        
        graph_proj.capture_end();
    }

    auto args = std::vector<PPTR>
    {
        PPTR(GP_gemm_A, (void*) x.data_ptr()), // Update 'A' pointer for all gemms
    };
    
    graph_proj.launch(args, stream);
    
    // RoPE (Eager)
    rope_(temp_q, temp_k, sin, cos, past_len, 1, num_heads, num_kv_heads, head_dim, false);
    
    return {temp_q, temp_k, temp_v};
}

void BC_Attention::run_out_gr(const at::Tensor& x, at::Tensor& out, Graph* graph)
{
    o_proj->run_gr(x, out, graph);
}

at::Tensor BC_Attention::run_out(const at::Tensor& x)
{
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (!temp_o.defined()) {
        temp_o = torch::empty({1, 1, hidden_size}, torch::TensorOptions().dtype(torch::kHalf).device(x.device()));
    }

    if (!graph_out.ready)
    {
        graph_out.capture_begin();
        run_out_gr(x, temp_o, &graph_out);
        graph_out.capture_end();
    }
    
    auto args = std::vector<PPTR>
    {
        PPTR(GP_gemm_A, (void*) x.data_ptr()),
        PPTR(GP_gemm_C, (void*) temp_o.data_ptr())
    };
    
    graph_out.launch(args, stream);
    return temp_o;
}
