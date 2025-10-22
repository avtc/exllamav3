#pragma once

#include <ATen/Tensor.h>
#include "context.cuh"

void pg_barrier
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
);

// P2P-optimized barrier kernel for fully connected systems
void pg_barrier_full_p2p
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
);