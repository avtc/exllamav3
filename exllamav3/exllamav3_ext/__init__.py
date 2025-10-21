"""
ExLlamaV3 CUDA Extension Module

This module contains the CUDA-accelerated functions for ExLlamaV3,
including P2P memory operations, tensor parallelism primitives, and quantization utilities.
"""

# The actual functions are implemented in C++/CUDA and exposed through PyBind11
# They are automatically available when the extension is imported