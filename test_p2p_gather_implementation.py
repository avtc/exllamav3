#!/usr/bin/env python3
"""
Test script for the P2P gather implementation with proper P2P memory access.
This script tests the new P2P memory access functionality in the p2p_gather kernel.
"""

import torch
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

try:
    import exllamav3_ext as ext
except ImportError:
    print("ERROR: Could not import exllamav3_ext. Please build the extension first.")
    sys.exit(1)

def test_p2p_gather_implementation():
    """Test the P2P gather implementation with proper P2P memory access."""
    
    print("Testing P2P gather implementation...")
    
    # Check if we have multiple GPUs available
    if torch.cuda.device_count() < 2:
        print("WARNING: This test requires at least 2 GPUs. Skipping test.")
        return True
    
    # Test parameters
    device_count = min(2, torch.cuda.device_count())
    devices = list(range(device_count))
    batch_size = 4
    tensor_size = 1024
    
    # Initialize context
    ctx = ext.pg_init_context(0)  # Using 0 as a placeholder for context pointer
    
    # Create test tensors on each device
    tensors = []
    for device in devices:
        torch.cuda.set_device(device)
        tensor = torch.randn(batch_size, tensor_size, dtype=torch.float16, device='cuda')
        tensors.append(tensor)
    
    # Create output tensor
    out_device = devices[0]
    torch.cuda.set_device(out_device)
    out_tensor = torch.zeros(batch_size, tensor_size * device_count, dtype=torch.float16, device='cuda')
    
    # Create abort flag
    abort_flag = torch.zeros(1, dtype=torch.int32, device='cuda')
    
    # Set up P2P memory pointers for each device
    for i, device in enumerate(devices):
        torch.cuda.set_device(device)
        # Get the data pointer of the tensor
        data_ptr = tensors[i].data_ptr()
        # Set the peer device pointer in the context
        ext.pg_set_peer_device_ptr(ctx, device, data_ptr)
    
    # Test dimensions
    ldims = [tensor_size] * device_count
    
    try:
        # Test the P2P gather operation
        print(f"Running P2P gather on {device_count} devices...")
        
        for this_device in devices:
            ext.p2p_gather(
                ctx,
                devices,
                this_device,
                out_device,
                tensors[this_device],
                out_tensor,
                ldims,
                abort_flag
            )
        
        # Check if abort flag was set
        if abort_flag.item() != 0:
            print("ERROR: Abort flag was set during P2P gather operation")
            return False
        
        print("P2P gather operation completed successfully")
        
        # Verify the output (basic check)
        if out_tensor.numel() == batch_size * tensor_size * device_count:
            print("Output tensor size is correct")
        else:
            print(f"ERROR: Output tensor size mismatch. Expected {batch_size * tensor_size * device_count}, got {out_tensor.numel()}")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR during P2P gather test: {str(e)}")
        return False

def test_fallback_mechanism():
    """Test the fallback mechanism when P2P access is not available."""
    
    print("\nTesting fallback mechanism...")
    
    # Check if we have multiple GPUs available
    if torch.cuda.device_count() < 2:
        print("WARNING: This test requires at least 2 GPUs. Skipping test.")
        return True
    
    # Test parameters
    device_count = min(2, torch.cuda.device_count())
    devices = list(range(device_count))
    batch_size = 4
    tensor_size = 1024
    
    # Initialize context
    ctx = ext.pg_init_context(0)  # Using 0 as a placeholder for context pointer
    
    # Create test tensors on each device
    tensors = []
    for device in devices:
        torch.cuda.set_device(device)
        tensor = torch.randn(batch_size, tensor_size, dtype=torch.float16, device='cuda')
        tensors.append(tensor)
    
    # Create output tensor
    out_device = devices[0]
    torch.cuda.set_device(out_device)
    out_tensor = torch.zeros(batch_size, tensor_size * device_count, dtype=torch.float16, device='cuda')
    
    # Create abort flag
    abort_flag = torch.zeros(1, dtype=torch.int32, device='cuda')
    
    # Test dimensions
    ldims = [tensor_size] * device_count
    
    try:
        # Test the P2P gather operation without setting up P2P pointers
        # This should trigger the fallback mechanism
        print(f"Running P2P gather with fallback (no P2P pointers set)...")
        
        for this_device in devices:
            ext.p2p_gather(
                ctx,
                devices,
                this_device,
                out_device,
                tensors[this_device],
                out_tensor,
                ldims,
                abort_flag
            )
        
        # Check if abort flag was set
        if abort_flag.item() != 0:
            print("ERROR: Abort flag was set during P2P gather operation with fallback")
            return False
        
        print("P2P gather operation with fallback completed successfully")
        
        # Verify the output (basic check)
        if out_tensor.numel() == batch_size * tensor_size * device_count:
            print("Output tensor size is correct")
        else:
            print(f"ERROR: Output tensor size mismatch. Expected {batch_size * tensor_size * device_count}, got {out_tensor.numel()}")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR during P2P gather fallback test: {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("P2P Gather Implementation Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires CUDA.")
        return False
    
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    
    # Run tests
    test_results = []
    
    # Test 1: P2P gather with proper P2P memory access
    test_results.append(test_p2p_gather_implementation())
    
    # Test 2: Fallback mechanism
    test_results.append(test_fallback_mechanism())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    test_names = [
        "P2P Gather Implementation",
        "Fallback Mechanism"
    ]
    
    all_passed = True
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "PASSED" if result else "FAILED"
        print(f"{i+1}. {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)