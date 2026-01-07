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
#include "../quant/util.cuh"

std::vector<at::Tensor> BC_Attention::run_proj(const at::Tensor& x, int past_len)
{
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Validate input and ensure it's 2D [1, hidden_size] for GEMM
    at::Tensor x_2d;
    if (x.dim() == 3)
    {
        TORCH_CHECK(x.size(0) == 1 && x.size(1) == 1,
            "BC_Attention::run_proj expects 3D input with shape [1, 1, hidden_size]");
        x_2d = x.view({1, x.size(-1)});
    }
    else if (x.dim() == 2)
    {
        TORCH_CHECK(x.size(0) == 1,
            "BC_Attention::run_proj expects 2D input with shape [1, hidden_size]");
        x_2d = x;
    }
    else
    {
        TORCH_CHECK(false, "BC_Attention::run_proj: Input must be 2D or 3D tensor");
    }

    // Ensure temp tensors exist before graph capture
    if (!temp_q.defined()) {
        auto opts = torch::TensorOptions().dtype(torch::kHalf).device(x.device());
        temp_q = torch::empty({1, num_heads * head_dim}, opts);
        temp_k = torch::empty({1, num_kv_heads * head_dim}, opts);
        temp_v = torch::empty({1, num_kv_heads * head_dim}, opts);
    }

    // Capture GEMM projections in CUDA graph (if not already captured)
    if (!graph_proj.ready)
    {
        graph_proj.capture_begin();
        q_proj->run_gr(x_2d, temp_q, &graph_proj);
        k_proj->run_gr(x_2d, temp_k, &graph_proj);
        v_proj->run_gr(x_2d, temp_v, &graph_proj);
        graph_proj.capture_end();
    }

    // Launch GEMM graph with updated input pointer
    auto args = std::vector<PPTR>
    {
        PPTR(GP_gemm_A, (void*) x_2d.data_ptr()),
    };

    graph_proj.launch(args, stream);

    // RMS Norm (eager - not supported in CUDA graph)
    if (q_norm_weight.defined()) {
        rms_norm(temp_q, q_norm_weight, temp_q, norm_epsilon, 0.0, false);
    }
    if (k_norm_weight.defined()) {
        rms_norm(temp_k, k_norm_weight, temp_k, norm_epsilon, 0.0, false);
    }

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

    // Validate input and ensure it's 2D [1, num_q_heads * head_dim] for GEMM
    at::Tensor x_2d;
    if (x.dim() == 3)
    {
        TORCH_CHECK(x.size(0) == 1 && x.size(1) == 1,
            "BC_Attention::run_out expects 3D input with shape [1, 1, num_q_heads * head_dim]");
        x_2d = x.view({1, x.size(-1)});
    }
    else if (x.dim() == 2)
    {
        TORCH_CHECK(x.size(0) == 1,
            "BC_Attention::run_out expects 2D input with shape [1, num_q_heads * head_dim]");
        x_2d = x;
    }
    else
    {
        TORCH_CHECK(false, "BC_Attention::run_out: Input must be 2D or 3D tensor");
    }

    // Ensure temp output tensor exists
    if (!temp_o.defined()) {
        temp_o = torch::empty({1, hidden_size}, torch::TensorOptions().dtype(torch::kHalf).device(x.device()));
    }

    if (!graph_out.ready)
    {
        graph_out.capture_begin();
        run_out_gr(x_2d, temp_o, &graph_out);
        graph_out.capture_end();
    }

    auto args = std::vector<PPTR>
    {
        PPTR(GP_gemm_A, (void*) x_2d.data_ptr()),
        PPTR(GP_gemm_C, (void*) temp_o.data_ptr())
    };

    graph_out.launch(args, stream);
    return temp_o;
}
