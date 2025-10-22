#pragma once

#include <ATen/Tensor.h>

void pg_gather
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
);

// P2P-optimized gather kernel for fully connected systems
void pg_gather_full_p2p
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
);
