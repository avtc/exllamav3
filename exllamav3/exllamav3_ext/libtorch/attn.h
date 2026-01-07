#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "linear.h"
#include "../graph.cuh"

struct BC_Attention
{
    int batch_size;
    int seq_len;
    int hidden_size;
    int num_heads;
    int head_dim;
    int num_kv_heads;

    std::shared_ptr<BC_LinearEXL3> q_proj;
    std::shared_ptr<BC_LinearEXL3> k_proj;
    std::shared_ptr<BC_LinearEXL3> v_proj;
    std::shared_ptr<BC_LinearEXL3> o_proj;

    at::Tensor q_norm_weight;
    at::Tensor k_norm_weight;
    float norm_epsilon;

    // Temp tensors
    at::Tensor temp_q;
    at::Tensor temp_k;
    at::Tensor temp_v;
    at::Tensor temp_o;

    Graph graph_proj;
    Graph graph_out;

    BC_Attention
    (
        std::shared_ptr<BC_LinearEXL3> _q_proj,
        std::shared_ptr<BC_LinearEXL3> _k_proj,
        std::shared_ptr<BC_LinearEXL3> _v_proj,
        std::shared_ptr<BC_LinearEXL3> _o_proj,
        at::Tensor _q_norm_weight,
        at::Tensor _k_norm_weight,
        float _norm_epsilon,
        int _hidden_size,
        int _num_heads,
        int _head_dim,
        int _num_kv_heads
    ) :
        q_proj              (_q_proj),
        k_proj              (_k_proj),
        v_proj              (_v_proj),
        o_proj              (_o_proj),
        q_norm_weight       (std::move(_q_norm_weight)),
        k_norm_weight       (std::move(_k_norm_weight)),
        norm_epsilon        (_norm_epsilon),
        hidden_size         (_hidden_size),
        num_heads           (_num_heads),
        head_dim            (_head_dim),
        num_kv_heads        (_num_kv_heads)
    {
        batch_size = 1;
        seq_len = 1;
    }

    // Part 1: Projections (CUDA graph captures GEMM, norms and RoPE run eagerly)
    std::vector<at::Tensor> run_proj(const at::Tensor& x, int past_len);

    // Part 2: Output Projection
    at::Tensor run_out(const at::Tensor& x);
    void run_out_gr(const at::Tensor& x, at::Tensor& out, Graph* graph);
};
