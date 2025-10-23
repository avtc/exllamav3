#!/usr/bin/env python3
"""
Debug script to identify the exact location of the segmentation fault
"""

import torch
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

try:
    import exllamav3_ext
except ImportError as e:
    print(f"Failed to import exllamav3_ext: {e}")
    print("Make sure the exllamav3 extension is built and installed")
    sys.exit(1)

def debug_barrier():
    """Debug the barrier function step by step"""
    
    # Check available CUDA devices
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA devices")
    
    if device_count < 2:
        print("Need at least 2 devices for P2P testing")
        return False
    
    # Test with just 2 devices first
    devices = [0, 1]
    abort_flag = torch.zeros((1,), dtype=torch.int32, device='cpu')
    
    print("Testing P2P access between devices...")
    for i in devices:
        for j in devices:
            if i != j:
                can_access = torch.cuda.device_can_access_peer(i, j)
                print(f"Device {i} can access device {j}: {can_access}")
    
    print("\nTrying individual steps...")
    
    # Step 1: Try setting device 0
    print("Step 1: Setting device 0")
    try:
        torch.cuda.set_device(0)
        print("✓ Successfully set device 0")
    except Exception as e:
        print(f"✗ Failed to set device 0: {e}")
        return False
    
    # Step 2: Try simple computation
    print("Step 2: Simple computation on device 0")
    try:
        x = torch.randn(100, 100, device='cuda:0')
        y = torch.matmul(x, x.T)
        print("✓ Computation successful on device 0")
    except Exception as e:
        print(f"✗ Computation failed on device 0: {e}")
        return False
    
    # Step 3: Try calling the barrier function
    print("Step 3: Calling p2p_device_barrier")
    try:
        exllamav3_ext.p2p_device_barrier(devices, 0, abort_flag)
        print("✓ Barrier call successful on device 0")
        
        # Check if abort flag was set
        if abort_flag.item() != 0:
            print(f"✗ Abort flag was set: {abort_flag.item()}")
            return False
        
    except Exception as e:
        print(f"✗ Barrier call failed on device 0: {e}")
        return False
    
    # Step 4: Try with device 1
    print("Step 4: Setting device 1")
    try:
        torch.cuda.set_device(1)
        print("✓ Successfully set device 1")
    except Exception as e:
        print(f"✗ Failed to set device 1: {e}")
        return False
    
    # Step 5: Try computation on device 1
    print("Step 5: Simple computation on device 1")
    try:
        x = torch.randn(100, 100, device='cuda:1')
        y = torch.matmul(x, x.T)
        print("✓ Computation successful on device 1")
    except Exception as e:
        print(f"✗ Computation failed on device 1: {e}")
        return False
    
    # Step 6: Try calling barrier on device 1
    print("Step 6: Calling p2p_device_barrier on device 1")
    try:
        exllamav3_ext.p2p_device_barrier(devices, 1, abort_flag)
        print("✓ Barrier call successful on device 1")
        
        # Check if abort flag was set
        if abort_flag.item() != 0:
            print(f"✗ Abort flag was set: {abort_flag.item()}")
            return False
            
    except Exception as e:
        print(f"✗ Barrier call failed on device 1: {e}")
        return False
    
    print("\nAll steps completed successfully!")
    return True

if __name__ == "__main__":
    success = debug_barrier()
    sys.exit(0 if success else 1)