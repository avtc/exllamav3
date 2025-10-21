#pragma once

#include <ATen/Tensor.h>

void p2p_gather(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    at::Tensor& abort_flag
);

void p2p_gather_direct(
    uintptr_t ctx,
    std::vector<uintuintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    at::Tensor& abort_flag
);