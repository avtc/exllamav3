#pragma once

#include <ATen/Tensor.h>

void p2p_broadcast(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
);

void p2p_broadcast_ll(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    at::Tensor& abort_flag
);