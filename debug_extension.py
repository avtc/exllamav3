#!/usr/bin/env python3
"""
Debug script to check what's actually in the exllamav3_ext module.
"""

import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

print("Debugging exllamav3_ext module...")
try:
    import exllamav3_ext as ext
    print(f"Module: {ext}")
    print(f"Module file: {ext.__file__ if hasattr(ext, '__file__') else 'No file'}")
    print(f"Module type: {type(ext)}")
    
    print("\nAll attributes in the module:")
    for attr in dir(ext):
        if not attr.startswith('_'):
            try:
                attr_value = getattr(ext, attr)
                print(f"  {attr}: {type(attr_value)}")
            except Exception as e:
                print(f"  {attr}: <error getting value: {e}>")
    
    print("\nChecking for pg_init_context specifically:")
    if hasattr(ext, 'pg_init_context'):
        print("  pg_init_context exists!")
        print(f"  Type: {type(ext.pg_init_context)}")
    else:
        print("  pg_init_context does NOT exist!")
        
    print("\nChecking for context-related functions:")
    context_funcs = ['pg_init_context', 'pg_check_timeout', 'pg_broadcast', 'pg_barrier', 'pg_all_reduce', 'pg_gather']
    for func in context_funcs:
        if hasattr(ext, func):
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func}")
            
    print("\nChecking for P2P functions:")
    p2p_funcs = ['p2p_broadcast', 'p2p_all_reduce', 'p2p_gather', 'p2p_copy_tensor_async']
    for func in p2p_funcs:
        if hasattr(ext, func):
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func}")

except Exception as e:
    print(f"Error importing module: {e}")
    import traceback
    traceback.print_exc()