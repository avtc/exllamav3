#pragma once

#include <ATen/Tensor.h>

void p2p_all_reduce(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
);

void p2p_all_reduce_ring(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
);

void p2p_all_reduce_tree_adaptive(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag,
    float connectivity_ratio = 0.0f
);