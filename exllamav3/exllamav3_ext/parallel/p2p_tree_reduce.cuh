#pragma once

#include <ATen/Tensor.h>

// Tree-based all-reduce operations
void p2p_all_reduce_tree(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag,
    int tree_type = 0  // 0: binary, 1: k-ary (k=4), 2: balanced
);

// Tree building functions
void p2p_build_binary_tree(
    std::vector<uintptr_t> devices,
    int this_device,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children
);

void p2p_build_kary_tree(
    std::vector<uintptr_t> devices,
    int this_device,
    int k,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children
);

void p2p_build_balanced_tree(
    std::vector<uintptr_t> devices,
    int this_device,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children
);

// Tree-aware reduction kernels
void p2p_tree_reduce_up(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children,
    at::Tensor& abort_flag
);

void p2p_tree_reduce_down(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    std::vector<int>& parent,
    std::vector<std::vector<int>>& children,
    at::Tensor& abort_flag
);

// Performance monitoring
struct p2p_tree_reduce_stats {
    float reduction_time_ms;
    int communication_steps;
    size_t data_bytes;
    int tree_depth;
    bool is_fully_connected;
};

void p2p_get_tree_reduce_stats(
    p2p_tree_reduce_stats* stats
);

// Adaptive algorithm selection
int p2p_select_reduce_algorithm(
    std::vector<uintptr_t> devices,
    size_t tensor_size,
    float connectivity_ratio
);