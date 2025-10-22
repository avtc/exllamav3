#!/usr/bin/env python3
"""
Test script to verify the import issue with pg_init_context.
"""

import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

print("Testing direct import...")
try:
    import exllamav3_ext as ext_direct
    print("✓ Direct import successful")
    print(f"pg_init_context available: {hasattr(ext_direct, 'pg_init_context')}")
except Exception as e:
    print(f"✗ Direct import failed: {e}")

print("\nTesting import through exllamav3.ext...")
try:
    from exllamav3.ext import exllamav3_ext as ext_indirect
    print("✓ Indirect import successful")
    print(f"pg_init_context available: {hasattr(ext_indirect, 'pg_init_context')}")
except Exception as e:
    print(f"✗ Indirect import failed: {e}")

print("\nTesting if they are the same object...")
try:
    import exllamav3_ext as ext_direct
    from exllamav3.ext import exllamav3_ext as ext_indirect
    print(f"Same object: {ext_direct is ext_indirect}")
    print(f"Direct module: {ext_direct}")
    print(f"Indirect module: {ext_indirect}")
except Exception as e:
    print(f"Comparison failed: {e}")