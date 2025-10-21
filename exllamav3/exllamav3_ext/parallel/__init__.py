"""
Parallel processing primitives for ExLlamaV3.

This module contains CUDA-accelerated parallel operations including:
- Broadcast operations
- Gather operations  
- All-reduce operations
- P2P memory operations
- Barriers and synchronization
"""

# Functions are implemented in C++/CUDA and exposed through PyBind11