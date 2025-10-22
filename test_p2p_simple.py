#!/usr/bin/env python3
"""
Simple P2P test to isolate the peer access issue.
"""

import torch
import sys
import os

# Add the exllamav3 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exllamav3'))

from exllamav3.model.model_tp_backend import TPBackendP2P
import exllamav3_ext as ext

def test_simple_p2p():
    """Test simple P2P functionality."""
    print("Simple P2P Test")
    
    # Check available devices
    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {device_count}")
    
    if device_count < 2:
        print("Error: At least 2 GPU devices are required for P2P test")
        return
    
    devices = [0, 1]
    
    # Initialize backend for each device
    backends = {}
    for device in devices:
        try:
            backend = TPBackendP2P(
                device=device,
                active_devices=devices,
                output_device=devices[0],
                init_method='tcp://localhost:12345',
                master=(device == devices[0]),
                uuid='p2p_test'
            )
            backends[device] = backend
            print(f"Initialized P2P backend for device {device}")
        except Exception as e:
            print(f"Error initializing backend for device {device}: {e}")
            return
    
    try:
        # Test creating tensors
        print("Creating test tensors...")
        src_device, dst_device = devices[0], devices[1]
        
        # Create tensors on CPU first
        size_elements = 1024 * 256  # 1MB
        src_tensor_cpu = torch.randn(size_elements, dtype=torch.float32)
        dst_tensor_cpu = torch.zeros(size_elements, dtype=torch.float32)
        
        print("Moving tensors to GPU...")
        # Move to GPU devices
        src_tensor = src_tensor_cpu.to(src_device)
        dst_tensor = dst_tensor_cpu.to(dst_device)
        
        print("Testing P2P copy...")
        # Test P2P copy
        ext.p2p_copy_tensor_async(src_device, dst_device, src_tensor, dst_tensor, backends[src_device].abort_flag)
        torch.cuda.synchronize()
        
        print("P2P test completed successfully!")
        
    finally:
        # Cleanup backends
        for device, backend in backends.items():
            try:
                backend.close()
                print(f"Closed P2P backend for device {device}")
            except Exception as e:
                print(f"Error closing backend for device {device}: {e}")

if __name__ == '__main__':
    test_simple_p2p()