#!/usr/bin/env python3
"""
Script to check if the ExLlamaV3 CUDA extension is properly compiled.
"""

import sys
import os

def check_extension():
    """Check if the exllamav3_ext extension is available and compiled."""
    
    print("Checking ExLlamaV3 CUDA Extension...")
    
    try:
        import exllamav3_ext as ext
        print("✓ exllamav3_ext module imported successfully")
        
        # Check if key functions are available
        required_functions = [
            'pg_init_context',
            'pg_broadcast',
            'pg_barrier',
            'pg_all_reduce',
            'pg_gather',
            'p2p_broadcast',
            'p2p_all_reduce',
            'p2p_gather',
            'p2p_copy_tensor_async',
            'p2p_init_memory_pool',
            'p2p_cleanup_memory_pool'
        ]
        
        missing_functions = []
        for func_name in required_functions:
            if hasattr(ext, func_name):
                print(f"✓ {func_name} is available")
            else:
                print(f"✗ {func_name} is missing")
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"\nMissing functions: {missing_functions}")
            print("The extension may not be fully compiled.")
            return False
        else:
            print("\n✓ All required functions are available!")
            return True
            
    except ImportError as e:
        print(f"✗ Failed to import exllamav3_ext: {e}")
        print("The extension is not compiled or not installed properly.")
        return False

def main():
    """Main function."""
    if not check_extension():
        print("\n" + "="*50)
        print("TO COMPILE THE EXTENSION:")
        print("Run the following command in the exllamav3 directory:")
        print("pip install -e . --verbose")
        print("\nOr if you want to skip compilation:")
        print("export EXLLAMA_NOCOMPILE=1")
        print("pip install -e . --verbose")
        print("="*50)

if __name__ == "__main__":
    main()