"""
ExLlamaV3 CUDA Extension Module

This module contains the CUDA-accelerated functions for ExLlamaV3,
including P2P memory operations, tensor parallelism primitives, and quantization utilities.
"""

# Import the compiled extension from the parent directory
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

try:
    import exllamav3_ext
    # Copy all attributes from the compiled extension to this module
    for attr in dir(exllamav3_ext):
        if not attr.startswith('_'):
            globals()[attr] = getattr(exllamav3_ext, attr)
except ImportError:
    # Fallback for development when extension isn't compiled yet
    pass