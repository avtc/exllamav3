"""
Quantization utilities for ExLlamaV3.

This module contains CUDA-accelerated quantization operations including:
- EXL3 quantization and dequantization
- Packing and unpacking operations
- Hadamard transforms for quantization
- Kernel mapping and device contexts
"""

# Functions are implemented in C++/CUDA and exposed through PyBind11