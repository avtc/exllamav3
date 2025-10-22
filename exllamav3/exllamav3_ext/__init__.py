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
    import exllamav3_ext as compiled_ext
    print(f"Successfully imported compiled extension from: {compiled_ext.__file__}")
    # Copy all attributes from the compiled extension to this module
    for attr in dir(compiled_ext):
        if not attr.startswith('_'):
            globals()[attr] = getattr(compiled_ext, attr)
    print(f"Copied {len([a for a in dir(compiled_ext) if not a.startswith('_')])} attributes")
except ImportError as e:
    print(f"Failed to import compiled extension: {e}")
    # Fallback for development when extension isn't compiled yet
    pass
except Exception as e:
    print(f"Error during import: {e}")
    import traceback
    traceback.print_exc()